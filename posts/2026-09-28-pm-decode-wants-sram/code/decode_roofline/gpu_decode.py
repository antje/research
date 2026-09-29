"""Measure what the roofline predicts, on a real CUDA GPU.

What: three measurements on one GPU, printed next to the package's spec-sheet bounds.
(1) What the chip delivers: dense BF16 GEMM throughput and HBM read bandwidth, and the
measured ridge they imply. (2) The SRAM cliff: a batch-1 decode GEMV (bf16, 8,192 columns)
swept over weight working sets from 4 MiB to 4 GiB, so small sets stay in L2 and large ones
stream from HBM. (3) One Qwen2.5-72B decoder layer (random bf16 weights, same shapes as
model.py), timed across batch sizes at 4,096 and 512 cached tokens, with the attention part
timed separately so the KV read shows up on its own.
Why: the post argues from spec sheets that decode is memory-bound and that tensor-core use
flattens below 41% at 4,096 tokens. This checks both on silicon.
Where: `python -m decode_roofline.gpu_decode` on a CUDA machine (needs torch, and triton,
which ships with torch on Linux). Writes results/b200-decode.{md,json}. The accounting
below is pure Python and tested on CPU; torch is imported only inside main().
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .chips import B200_BF16
from .model import QWEN2_5_72B, ModelShape
from .roofline import StepCost, bound, decode_step, intensity_limit

MODEL = QWEN2_5_72B
CHIP = B200_BF16
BATCHES = [1, 4, 16, 64, 256, 1024]          # the package's table 1 batches
CONTEXTS = [4096, 512]                       # the post's context, and the one that crosses the ridge
WORKING_SETS_MIB = [4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 512, 1024, 4096]
GEMV_COLS = 8192                             # hidden size: one row = one output of a decode GEMV
GEMM_N = 8192                                # square GEMM for the compute probe
SEED = 0
WARMUP, ITERS = 5, 30                        # layer graphs: replays before timing, timed replays
PROBE_WARMUP, PROBE_ITERS = 3, 15            # GEMM and GEMV probes
PASS_BYTES = 16e9                            # each GEMV launch reads about this many bytes
MEM_BUDGET = 100e9                           # weights + KV for the layer copies of one config
MAX_COPIES = 4                               # distinct layer copies per graph replay
RESULTS = Path(__file__).resolve().parents[1] / "results"
NAME = "b200-decode"
MiB = 2**20


def layer_params(m: ModelShape) -> int:
    """Weight parameters of one decoder layer: model.params() minus embeddings, per layer."""
    embed = m.vocab * m.hidden * (1 if m.tied_embeddings else 2)
    return (m.params() - embed) // m.layers


def layer_step(m: ModelShape, batch: int, context: int, w_bytes: int = 2, kv_bytes: int = 2) -> StepCost:
    """decode_step() for a single layer: that layer's weights once, its KV once per sequence."""
    return StepCost(
        batch=batch, context=context, w_bytes=w_bytes, kv_bytes=kv_bytes,
        bytes_weights=layer_params(m) * w_bytes,
        bytes_kv=batch * context * m.kv_bytes_per_token(kv_bytes) / m.layers,
        flops=batch * (2.0 * layer_params(m) + 4.0 * m.hidden * context),
    )


def layer_copies(m: ModelShape, batch: int, context: int, budget: float = MEM_BUDGET, cap: int = MAX_COPIES) -> int:
    """How many distinct layers (own weights, own KV) to chain in one timed graph."""
    c = layer_step(m, batch, context)
    return max(1, min(cap, int(budget // c.bytes_total)))


def sweep() -> list[tuple[int, int]]:
    """(context, batch) pairs for the layer measurement, in print order."""
    return [(ctx, b) for ctx in CONTEXTS for b in BATCHES]


def gemv_rows(mib: int, cols: int = GEMV_COLS) -> int:
    """Rows of a bf16 matrix with `cols` columns that fill `mib` MiB."""
    return mib * MiB // (cols * 2)


def gemv_passes(mib: int) -> int:
    """GEMV passes over the same weights per launch, so each launch reads about PASS_BYTES."""
    return max(1, int(PASS_BYTES // (mib * MiB)))


def ceiling() -> float:
    """The post's 41%: compute used can approach limit/ridge at 4,096 tokens, never more."""
    return intensity_limit(MODEL, 4096) / CHIP.ridge


def fmt(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


def layer_row(context: int, batch: int, t_layer: float, t_attn: float) -> dict:
    """Everything the tables print for one measured layer config, from the package's accounting."""
    c = layer_step(MODEL, batch, context)
    bd = bound(CHIP, c)
    full = bound(CHIP, decode_step(MODEL, batch, context))
    return {
        "context": context, "batch": batch, "t_layer_s": t_layer, "t_attn_s": t_attn,
        "bytes": c.bytes_total, "flops": c.flops, "t_bound_s": bd.step_time,
        "speed_vs_bound": bd.step_time / t_layer,
        "eff_tb_s": c.bytes_total / t_layer / 1e12,
        "tflops": c.flops / t_layer / 1e12,
        "compute_used": c.flops / t_layer / CHIP.peak_flops,
        "attn_share": t_attn / t_layer,
        "attn_us_per_seq": t_attn / batch * 1e6,
        "attn_kv_tb_s": c.bytes_kv / t_attn / 1e12,
        "rest_us_per_seq": (t_layer - t_attn) / batch * 1e6,
        "tok_s_80_layers": batch / (t_layer * MODEL.layers),
        "tok_s_bound_package": full.tok_s,
        "compute_used_bound": bd.compute_util,
    }


def layer_table(rows: list[dict], context: int) -> str:
    lines = [
        f"### One {MODEL.name} decoder layer, bf16, {context:,} cached tokens, measured vs the package's bound",
        "",
        "| batch | measured per layer | spec bound per layer | speed vs bound | package bytes / time | TFLOPS | compute used (of 2.25 PFLOPS) | bound on compute used | attention share | attention per sequence | rest per sequence | 80-layer tok/s, measured | tok/s bound (package, full model) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["context"] != context:
            continue
        lines.append(
            f"| {r['batch']:,} | {r['t_layer_s']*1e6:,.0f} us | {r['t_bound_s']*1e6:,.0f} us | {r['speed_vs_bound']*100:.0f}% "
            f"| {r['eff_tb_s']:.2f} TB/s | {r['tflops']:,.0f} | {r['compute_used']*100:.1f}% | {r['compute_used_bound']*100:.1f}% "
            f"| {r['attn_share']*100:.0f}% | {r['attn_us_per_seq']:.2f} us | {r['rest_us_per_seq']:.2f} us "
            f"| {r['tok_s_80_layers']:,.0f} | {r['tok_s_bound_package']:,.0f} |"
        )
    return "\n".join(lines)


def driver_version() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip().splitlines()[0] or "unknown"
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def main() -> None:
    import torch

    from . import gpu_kernels as k

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU; the spec-sheet tables run with `python -m decode_roofline`")
    torch.manual_seed(SEED)
    props = torch.cuda.get_device_properties(0)
    l2 = props.L2_cache_size

    # (1) what the chip delivers
    gemm_tflops = k.gemm_tflops(GEMM_N, PROBE_WARMUP, PROBE_ITERS)
    gemv = []
    for mib in WORKING_SETS_MIB:
        t = k.gemv_pass_time(gemv_rows(mib), GEMV_COLS, gemv_passes(mib), PROBE_WARMUP, PROBE_ITERS)
        gemv.append({"mib": mib, "passes": gemv_passes(mib), "t_pass_s": t, "tb_s": mib * MiB / t / 1e12,
                     "fits_l2": mib * MiB <= l2})
    hbm = gemv[-1]["tb_s"]
    sram = max(g["tb_s"] for g in gemv if g["fits_l2"])
    sram_mib = next(g["mib"] for g in gemv if g["tb_s"] == sram)
    ridge_measured = gemm_tflops * 1e12 / (hbm * 1e12)

    # (3) one decoder layer, batch sweep
    rows = []
    for ctx, b in sweep():
        n = layer_copies(MODEL, b, ctx)
        t_layer, t_attn = k.layer_times(MODEL, b, ctx, n, WARMUP, ITERS)
        rows.append(layer_row(ctx, b, t_layer, t_attn) | {"copies": n})

    head = [
        f"### What one {props.name} delivers (measured) next to its spec sheet",
        "",
        "| quantity | measured | spec sheet (package) |",
        "|---|---:|---:|",
        f"| dense BF16 GEMM, {GEMM_N:,} x {GEMM_N:,} x {GEMM_N:,} | {gemm_tflops:,.0f} TFLOPS ({gemm_tflops*1e12/CHIP.peak_flops*100:.0f}% of peak) | {CHIP.peak_flops/1e12:,.0f} TFLOPS |",
        f"| HBM read, batch-1 GEMV over {WORKING_SETS_MIB[-1]:,} MiB of bf16 weights | {hbm:.2f} TB/s ({hbm*1e12/CHIP.mem_bw*100:.0f}% of peak) | {CHIP.mem_bw/1e12:.0f} TB/s |",
        f"| ridge (FLOP per byte) | {ridge_measured:,.0f} | {CHIP.ridge:,.0f} |",
        "",
        f"L2 cache reported by torch: {l2:,} bytes ({l2/MiB:.1f} MiB). Intensity limit at 4,096 tokens: {intensity_limit(MODEL, 4096):.1f} FLOP/byte, below both ridges.",
    ]
    cliff = [
        f"### The SRAM cliff: a batch-1 decode GEMV (bf16, {GEMV_COLS:,} columns) by weight working set",
        "",
        "| weights | in L2? | time per pass | bandwidth |",
        "|---:|---|---:|---:|",
    ] + [f"| {g['mib']:,} MiB | {'yes' if g['fits_l2'] else 'no'} | {g['t_pass_s']*1e6:,.2f} us | {g['tb_s']:.2f} TB/s |" for g in gemv] + [
        "",
        f"From L2: up to {sram:.2f} TB/s (at {sram_mib} MiB). From HBM: {hbm:.2f} TB/s (at {WORKING_SETS_MIB[-1]:,} MiB). "
        f"Ratio {sram/hbm:.1f}x. One {MODEL.name} layer is {layer_params(MODEL)*2/1e9:.2f} GB of bf16 weights and the model is {MODEL.params()*2/1e9:.1f} GB, "
        f"so decode streams from HBM.",
    ]
    r4 = [r for r in rows if r["context"] == 4096]
    r5 = [r for r in rows if r["context"] == 512]
    used4 = ", ".join(f"{r['compute_used']*100:.1f}%" for r in r4)
    used5 = ", ".join(f"{r['compute_used']*100:.1f}%" for r in r5)
    batches = ", ".join(f"{b:,}" for b in BATCHES)
    summary = [
        f"At 4,096 tokens, compute used goes {used4} for batch {batches}. "
        f"The spec-sheet ceiling is {ceiling()*100:.0f}% (limit {intensity_limit(MODEL, 4096):.1f} / ridge {CHIP.ridge:.1f}).",
        f"At 512 tokens it goes {used5} for the same batches.",
        f"Attention per sequence at 4,096 tokens: {r4[0]['attn_us_per_seq']:.2f} us per layer at batch 1, then "
        f"{', '.join(format(r['attn_us_per_seq'], '.2f') for r in r4[3:])} us at batch {', '.join(format(r['batch'], ',') for r in r4[3:])}: "
        f"it floors at the time to read one sequence's {layer_step(MODEL, 1, 4096).bytes_kv/1e6:.1f} MB of KV "
        f"({r4[-1]['attn_kv_tb_s']:.2f} TB/s at batch {r4[-1]['batch']:,}). "
        f"The rest of the layer per sequence drops from {r4[0]['rest_us_per_seq']:,.2f} us to {r4[-1]['rest_us_per_seq']:.2f} us.",
        f"At batch {r4[-1]['batch']:,} and 4,096 tokens the attention alone takes {r4[-1]['t_attn_s']*1e6:,.0f} us per layer, about the whole layer's spec bound "
        f"({r4[-1]['t_bound_s']*1e6:,.0f} us); the matmuls run after it instead of hiding under it.",
        f"Batch 1 at 4,096 tokens runs at {r4[0]['speed_vs_bound']*100:.0f}% of the spec-sheet speed bound: {r4[0]['t_layer_s']*1e6:,.0f} us per layer against {r4[0]['t_bound_s']*1e6:,.0f} us, "
        f"or {r4[0]['tok_s_80_layers']:,.0f} tok/s for 80 layers against a full-model bound of {r4[0]['tok_s_bound_package']:,.0f}.",
    ]
    conditions = (
        f"Conditions: one {props.name} ({props.multi_processor_count} SMs), driver {driver_version()}, CUDA {torch.version.cuda}, torch {torch.__version__}, "
        f"clocks not locked (default boost), GPU otherwise idle. Seed {SEED}, random weights and KV; no real checkpoint is loaded. "
        f"Layer: RMSNorm, fused QKV projection, K/V of the new token written into the last of {', '.join(f'{c:,}' for c in CONTEXTS)} cache slots, "
        f"GQA attention (torch scaled_dot_product_attention, enable_gqa) over all slots, O projection, RMSNorm, SwiGLU MLP, residual adds; "
        f"no RoPE, no biases, no embedding or LM head. Each config chains {MAX_COPIES} distinct layers (own weights, own KV; fewer if they exceed {MEM_BUDGET/1e9:.0f} GB) "
        f"in one CUDA graph; {WARMUP} warmup replays, median of {ITERS} timed replays (CUDA events), divided by the layer count. "
        f"Attention share: the same graph with only the attention calls. 80-layer tok/s = batch / (80 x measured layer time). "
        f"Spec bound per layer: the package's bound() on that layer's bytes and FLOPs (B200 BF16 dense 2.25 PFLOPS, 8 TB/s). "
        f"GEMV probe: a Triton kernel, two weight rows per step, {GEMV_COLS:,} bf16 columns, fp32 accumulate, persistent grid, "
        f"repeated passes over the same weights inside one launch (about {PASS_BYTES/1e9:.0f} GB read per launch); GEMM probe: torch.matmul bf16. "
        f"Probes: {PROBE_WARMUP} warmup launches, median of {PROBE_ITERS}. Bandwidth = weight bytes / time. GB and TB are decimal, MiB binary."
    )
    md = "\n\n".join(["\n".join(head), "\n".join(cliff), layer_table(rows, 4096), layer_table(rows, 512),
                      "\n".join(summary), conditions]) + "\n"
    print(md)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{NAME}.md").write_text(md, encoding="utf-8")
    (RESULTS / f"{NAME}.json").write_text(json.dumps({
        "gpu": props.name, "sms": props.multi_processor_count, "l2_bytes": l2, "driver": driver_version(),
        "cuda": torch.version.cuda, "torch": torch.__version__, "seed": SEED,
        "gemm_tflops": gemm_tflops, "hbm_tb_s": hbm, "l2_tb_s": sram, "ridge_measured": ridge_measured,
        "ridge_spec": CHIP.ridge, "ceiling_4096": ceiling(),
        "gemv_sweep": gemv, "layer_sweep": rows, "conditions": conditions,
    }, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS / (NAME + '.md')} and {NAME}.json")


if __name__ == "__main__":
    main()
