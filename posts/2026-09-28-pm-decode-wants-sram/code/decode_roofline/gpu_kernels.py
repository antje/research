"""The torch and Triton side of gpu_decode.py: a decoder layer, a GEMV probe, a GEMM probe.

Imported only by gpu_decode.main(), so the rest of the package runs without torch.
"""

from __future__ import annotations

import statistics

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .model import ModelShape

BF16 = torch.bfloat16


def _median_time(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / 1e3)
    return statistics.median(ts)


def _graph(fn) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


# ------------------------------------------------------------------ one decoder layer

def make_layer(m: ModelShape) -> dict:
    h, hd, i = m.hidden, m.head_dim, m.intermediate
    sc = 0.02
    return {
        "wqkv": torch.randn((m.heads + 2 * m.kv_heads) * hd, h, dtype=BF16, device="cuda") * sc,
        "wo": torch.randn(h, h, dtype=BF16, device="cuda") * sc,
        "wgu": torch.randn(2 * i, h, dtype=BF16, device="cuda") * sc,
        "wd": torch.randn(h, i, dtype=BF16, device="cuda") * sc,
        "n1": torch.ones(h, dtype=BF16, device="cuda"),
        "n2": torch.ones(h, dtype=BF16, device="cuda"),
    }


def layer_weight_bytes(layer: dict) -> int:
    return sum(t.numel() * t.element_size() for k, t in layer.items() if k.startswith("w"))


def attention(m: ModelShape, q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor) -> torch.Tensor:
    """Decode attention: one query token per sequence over the whole cache, GQA."""
    b = q.shape[0]
    return F.scaled_dot_product_attention(q.view(b, m.heads, 1, m.head_dim), kc, vc, enable_gqa=True).reshape(b, m.hidden)


def layer_step(m: ModelShape, w: dict, kc: torch.Tensor, vc: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    b, h, hd = x.shape[0], m.hidden, m.head_dim
    y = F.rms_norm(x, (h,), w["n1"])
    q, k, v = F.linear(y, w["wqkv"]).split([m.heads * hd, m.kv_heads * hd, m.kv_heads * hd], dim=-1)
    kc[:, :, -1, :] = k.view(b, m.kv_heads, hd)       # the new token's K/V goes into the last slot
    vc[:, :, -1, :] = v.view(b, m.kv_heads, hd)
    x = x + F.linear(attention(m, q, kc, vc), w["wo"])
    y = F.rms_norm(x, (h,), w["n2"])
    g, u = F.linear(y, w["wgu"]).chunk(2, dim=-1)
    return x + F.linear(F.silu(g) * u, w["wd"])


def layer_times(m: ModelShape, batch: int, context: int, copies: int, warmup: int, iters: int) -> tuple[float, float]:
    """Median seconds per layer for the full step and for its attention alone."""
    layers = [make_layer(m) for _ in range(copies)]
    kcs = [torch.randn(batch, m.kv_heads, context, m.head_dim, dtype=BF16, device="cuda") for _ in range(copies)]
    vcs = [torch.randn_like(kc) for kc in kcs]
    x0 = torch.randn(batch, m.hidden, dtype=BF16, device="cuda")
    q0 = torch.randn(batch, m.hidden, dtype=BF16, device="cuda")
    out = torch.empty_like(x0)

    def full():
        x = x0
        for w, kc, vc in zip(layers, kcs, vcs):
            x = layer_step(m, w, kc, vc, x)
        out.copy_(x)

    def attn_only():
        for kc, vc in zip(kcs, vcs):
            out.copy_(attention(m, q0, kc, vc))

    g_full, g_attn = _graph(full), _graph(attn_only)
    t_full = _median_time(g_full.replay, warmup, iters) / copies
    t_attn = _median_time(g_attn.replay, warmup, iters) / copies
    del g_full, g_attn, layers, kcs, vcs
    torch.cuda.empty_cache()
    return t_full, t_attn


# ------------------------------------------------------------------ GEMV probe (Triton)

@triton.jit
def _gemv_rows(W, x, y, N, STEPS, K: tl.constexpr, ROWS: tl.constexpr):
    """y = W @ x, ROWS rows of W per loop step; STEPS = passes * N / ROWS, so the loop re-reads W."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, K)
    xv = tl.load(x + offs).to(tl.float32)
    for i in tl.range(pid, STEPS, nprog, num_stages=2):
        row = (i * ROWS) % N + tl.arange(0, ROWS)
        w = tl.load(W + row[:, None] * K + offs[None, :], cache_modifier=".cg").to(tl.float32)
        tl.store(y + row, tl.sum(w * xv[None, :], axis=1).to(tl.bfloat16))


GEMV_ROWS, GEMV_WARPS, GEMV_PROGRAMS_PER_SM = 2, 2, 4   # the fastest of 96 configs tried on a B200


def gemv(W: torch.Tensor, x: torch.Tensor, passes: int = 1) -> torch.Tensor:
    n, k = W.shape
    assert n % GEMV_ROWS == 0
    y = torch.empty(n, dtype=BF16, device=W.device)
    grid = (GEMV_PROGRAMS_PER_SM * torch.cuda.get_device_properties(W.device).multi_processor_count,)
    _gemv_rows[grid](W, x, y, n, passes * n // GEMV_ROWS, K=k, ROWS=GEMV_ROWS, num_warps=GEMV_WARPS)
    return y


def gemv_pass_time(rows: int, cols: int, passes: int, warmup: int, iters: int) -> float:
    """Median seconds for one pass over a rows x cols bf16 weight matrix."""
    W = torch.randn(rows, cols, dtype=BF16, device="cuda")
    x = torch.randn(cols, dtype=BF16, device="cuda")
    t = _median_time(lambda: gemv(W, x, passes), warmup, iters) / passes
    del W
    torch.cuda.empty_cache()
    return t


def gemm_tflops(n: int, warmup: int, iters: int) -> float:
    a = torch.randn(n, n, dtype=BF16, device="cuda")
    b = torch.randn(n, n, dtype=BF16, device="cuda")
    t = _median_time(lambda: a @ b, warmup, iters)
    return 2 * n**3 / t / 1e12
