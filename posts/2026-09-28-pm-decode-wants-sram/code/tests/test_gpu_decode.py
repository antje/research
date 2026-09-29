"""The GPU experiment's accounting runs on CPU; the kernels are checked only where CUDA exists."""

import math

import pytest

from decode_roofline import gpu_decode as g
from decode_roofline.model import QWEN2_5_72B
from decode_roofline.roofline import decode_step


def test_layer_params_match_the_shapes_the_gpu_allocates():
    m = QWEN2_5_72B
    qkv = (m.heads + 2 * m.kv_heads) * m.head_dim * m.hidden
    o = m.hidden * m.hidden
    mlp = 3 * m.hidden * m.intermediate
    assert g.layer_params(m) == qkv + o + mlp == 877_658_112


def test_layers_times_layer_step_plus_embeddings_is_the_package_step():
    m, b, ctx = QWEN2_5_72B, 64, 4096
    one, full = g.layer_step(m, b, ctx), decode_step(m, b, ctx)
    embed_bytes = m.vocab * m.hidden * 2 * 2
    assert math.isclose(one.bytes_kv * m.layers, full.bytes_kv)
    assert math.isclose(one.bytes_weights * m.layers + embed_bytes, full.bytes_weights)
    embed_flops = b * 2 * m.vocab * m.hidden * 2
    assert math.isclose(one.flops * m.layers + embed_flops, full.flops)


def test_layer_kv_is_16_mb_per_sequence_at_4k():
    c = g.layer_step(QWEN2_5_72B, 1, 4096)
    assert c.bytes_kv == 4096 * 2 * 8 * 128 * 2


def test_layer_copies_stay_in_budget():
    assert g.layer_copies(QWEN2_5_72B, 1, 4096) == g.MAX_COPIES
    for ctx, b in g.sweep():
        n = g.layer_copies(QWEN2_5_72B, b, ctx)
        assert 1 <= n <= g.MAX_COPIES
        assert n == 1 or n * g.layer_step(QWEN2_5_72B, b, ctx).bytes_total <= g.MEM_BUDGET
    assert g.layer_copies(QWEN2_5_72B, 4096, 4096) == 1


def test_sweep_covers_every_batch_at_both_contexts():
    s = g.sweep()
    assert len(s) == len(g.BATCHES) * len(g.CONTEXTS)
    assert [b for c, b in s if c == 4096] == g.BATCHES


def test_gemv_sizes():
    assert g.gemv_rows(4) * g.GEMV_COLS * 2 == 4 * 2**20
    assert g.gemv_passes(4096) >= 1
    assert g.gemv_passes(4) * 4 * 2**20 <= g.PASS_BYTES
    assert any(mib * 2**20 < 126e6 for mib in g.WORKING_SETS_MIB)
    assert g.WORKING_SETS_MIB[-1] * 2**20 > 10 * 132_644_864      # the last set is far past L2


def test_ceiling_is_the_posts_41_percent():
    assert round(g.ceiling() * 100) == 41


def test_layer_row_arithmetic():
    r = g.layer_row(4096, 1, t_layer=300e-6, t_attn=20e-6)
    assert math.isclose(r["attn_share"], 20 / 300)
    assert math.isclose(r["tok_s_80_layers"], 1 / (300e-6 * 80))
    assert 0 < r["speed_vs_bound"] < 1
    assert r["compute_used"] < r["compute_used_bound"]


def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    pytest.importorskip("triton")
    return torch


def test_gemv_kernel_matches_torch():
    torch = _cuda()
    from decode_roofline import gpu_kernels as k
    torch.manual_seed(0)
    W = torch.randn(1000, 8192, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(8192, dtype=torch.bfloat16, device="cuda")
    ref = W.float() @ x.float()
    for passes in (1, 3):
        y = k.gemv(W, x, passes).float()
        assert torch.allclose(y, ref, rtol=2e-2, atol=0.5)


def test_gqa_attention_matches_expanded_heads():
    torch = _cuda()
    from decode_roofline import gpu_kernels as k
    m = QWEN2_5_72B
    torch.manual_seed(0)
    q = torch.randn(2, m.hidden, dtype=torch.bfloat16, device="cuda")
    kc = torch.randn(2, m.kv_heads, 64, m.head_dim, dtype=torch.bfloat16, device="cuda")
    vc = torch.randn_like(kc)
    got = k.attention(m, q, kc, vc).float()
    rep = m.heads // m.kv_heads
    qh = q.float().view(2, m.heads, 1, m.head_dim)
    kh, vh = kc.float().repeat_interleave(rep, 1), vc.float().repeat_interleave(rep, 1)
    p = torch.softmax(qh @ kh.transpose(-1, -2) / m.head_dim**0.5, -1)
    ref = (p @ vh).reshape(2, m.hidden)
    assert torch.allclose(got, ref, atol=3e-2)


def test_layer_step_runs_and_writes_the_new_kv():
    torch = _cuda()
    from decode_roofline import gpu_kernels as k
    m = QWEN2_5_72B
    torch.manual_seed(0)
    w = k.make_layer(m)
    assert k.layer_weight_bytes(w) == g.layer_params(m) * 2
    kc = torch.zeros(2, m.kv_heads, 16, m.head_dim, dtype=torch.bfloat16, device="cuda")
    vc = torch.zeros_like(kc)
    x = torch.randn(2, m.hidden, dtype=torch.bfloat16, device="cuda")
    y = k.layer_step(m, w, kc, vc, x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert kc[:, :, -1].abs().sum() > 0 and kc[:, :, :-1].abs().sum() == 0
