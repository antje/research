"""Print the tables and write results/results.json and results/results.md.

What: three tables. (1) one decode step of Qwen2.5-72B at 4,096 context across batch sizes
on a B200; (2) ridge points per chip and the batch needed to reach them at three context
lengths; (3) what fits on chip. Every number is a spec-sheet bound, not a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

from .chips import B200_BF16, B200_FP8, CHIPS, WSE3
from .model import QWEN2_5_7B, QWEN2_5_72B
from .roofline import batch_at_ridge, bound, decode_step, fit, intensity_limit

CONTEXT = 4096
BATCHES = [1, 4, 16, 64, 256, 1024]
CONTEXTS = [512, 4096, 32768]
RESULTS = Path(__file__).resolve().parents[1] / "results"

GB = 1e9


def fmt(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


def table1() -> tuple[list[dict], str]:
    rows, lines = [], []
    lines.append(f"### One decode step, {QWEN2_5_72B.name}, bf16 weights and KV, {CONTEXT:,} cached tokens, on one B200 ({B200_BF16.precision}, {fmt(B200_BF16.mem_bw/1e12, 0)} TB/s)")
    lines.append("")
    lines.append("| batch | weights read | KV read | FLOPs | intensity (FLOP/byte) | limiter | tok/s bound | compute used |")
    lines.append("|---:|---:|---:|---:|---:|---|---:|---:|")
    for b in BATCHES:
        c = decode_step(QWEN2_5_72B, b, CONTEXT)
        bd = bound(B200_BF16, c)
        rows.append({"batch": b, "bytes_weights": c.bytes_weights, "bytes_kv": c.bytes_kv, "flops": c.flops,
                     "intensity": c.intensity, "limiter": bd.limiter, "tok_s_bound": bd.tok_s, "compute_util": bd.compute_util})
        lines.append(f"| {b:,} | {fmt(c.bytes_weights/GB)} GB | {fmt(c.bytes_kv/GB)} GB | {fmt(c.flops/1e12, 2)} TFLOP | {fmt(c.intensity)} | {bd.limiter} | {fmt(bd.tok_s, 0)} | {bd.compute_util*100:.2f}% |")
    lim = intensity_limit(QWEN2_5_72B, CONTEXT)
    lines.append("")
    lines.append(f"Intensity limit as batch grows (weights amortized, KV not): {fmt(lim)} FLOP/byte. B200 BF16 ridge: {fmt(B200_BF16.ridge, 0)} FLOP/byte.")
    lines.append("Intensity limits by context (bf16 KV): " + ", ".join(f"{c:,} tokens: {fmt(intensity_limit(QWEN2_5_72B, c))}" for c in CONTEXTS) + " FLOP/byte.")
    c1 = decode_step(QWEN2_5_72B, 1, CONTEXT)
    b1 = bound(B200_BF16, c1)
    lines.append(f"Batch 1 detail: {fmt(c1.bytes_total/GB)} GB moved, step time {b1.t_mem*1e3:.1f} ms on the B200, {fmt(c1.flops/1e12, 2)} TFLOP.")
    c6 = decode_step(QWEN2_5_72B, 6, CONTEXT)
    b6 = bound(B200_BF16, c6)
    lines.append(f"Batch 6 detail (the batch that saturates a WSE-3): on the B200, {fmt(c6.bytes_total/GB)} GB moved, t_mem {b6.t_mem*1e3:.1f} ms, t_comp {b6.t_comp*1e3:.2f} ms, compute used {b6.compute_util*100:.1f}%.")
    f = fit(QWEN2_5_72B, B200_BF16, 2)
    lines.append(f"Capacity on one B200 with bf16 weights: {fmt(f['free_bytes']/GB)} GB free, room for {f['kv_tokens_that_fit']:,} KV tokens, about {f['kv_tokens_that_fit'] // CONTEXT} sequences of {CONTEXT:,} tokens.")
    return rows, "\n".join(lines)


def table2() -> tuple[list[dict], str]:
    rows, lines = [], []
    lines.append(f"### Ridge points and the batch that reaches them, {QWEN2_5_72B.name}")
    lines.append("")
    lines.append("| chip | weights + KV precision | peak (dense) | bandwidth | ridge (FLOP/byte) | " + " | ".join(f"batch at ridge, {c:,} ctx" for c in CONTEXTS) + " |")
    lines.append("|---|---|---:|---:|---:|" + "---:|" * len(CONTEXTS))
    for label, chip, wb, kb in (("B200 BF16", B200_BF16, 2, 2), ("B200 FP8", B200_FP8, 1, 1), ("WSE-3", WSE3, 2, 2)):
        per_ctx = {c: batch_at_ridge(QWEN2_5_72B, chip, c, wb, kb) for c in CONTEXTS}
        limits = {c: intensity_limit(QWEN2_5_72B, c, kb) for c in CONTEXTS}
        rows.append({"chip": label, "peak_flops": chip.peak_flops, "mem_bw": chip.mem_bw, "ridge": chip.ridge,
                     "w_bytes": wb, "kv_bytes": kb, "batch_at_ridge": {str(c): per_ctx[c] for c in CONTEXTS},
                     "intensity_limit": {str(c): limits[c] for c in CONTEXTS}})
        cells = []
        for c in CONTEXTS:
            v = per_ctx[c]
            cells.append(f"{v:,}" if v is not None else f"never (limit {fmt(limits[c])})")
        peak = f"{chip.peak_flops/1e15:g} PFLOPS"
        bw = f"{chip.mem_bw/1e12:g} TB/s" if chip.mem_bw < 1e15 else f"{chip.mem_bw/1e15:g} PB/s"
        prec = "bf16 / bf16" if wb == 2 else "fp8 / fp8"
        lines.append(f"| {chip.name} ({chip.memory}) | {prec} | {peak} {chip.precision} | {bw} | {fmt(chip.ridge)} | " + " | ".join(cells) + " |")
    return rows, "\n".join(lines)


def table3() -> tuple[list[dict], str]:
    rows, lines = [], []
    lines.append("### What fits: weights vs the memory that feeds the compute")
    lines.append("")
    lines.append("| model | weights + KV precision | weights | B200 HBM3e 180 GB | WSE-3 SRAM 44 GB | KV/token |")
    lines.append("|---|---|---:|---|---|---:|")
    for m in (QWEN2_5_72B, QWEN2_5_7B):
        for prec, wb in (("bf16", 2), ("fp8", 1)):
            fb = fit(m, B200_BF16, wb, wb)
            fw = fit(m, WSE3, wb, wb)
            rows.append({"model": m.name, "precision": prec, "params": m.params(), "weights_bytes": fb["weights_bytes"],
                         "fits_b200": fb["fits"], "kv_tokens_b200": fb["kv_tokens_that_fit"],
                         "fits_wse3": fw["fits"], "kv_tokens_wse3": fw["kv_tokens_that_fit"],
                         "kv_bytes_per_token": m.kv_bytes_per_token(wb)})
            def cell(f):
                return (f"fits, room for {f['kv_tokens_that_fit']:,} KV tokens" if f["fits"] else "does not fit")
            lines.append(f"| {m.name} ({m.params()/1e9:.1f}B) | {prec} / {prec} | {fmt(fb['weights_bytes']/GB)} GB | {cell(fb)} | {cell(fw)} | {m.kv_bytes_per_token(wb)/1024:.0f} KiB |")
    return rows, "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    r1, t1 = table1()
    r2, t2 = table2()
    r3, t3 = table3()
    conditions = (
        "Conditions: spec-sheet bounds, not measurements. Bytes = weights once per step + KV once per sequence; "
        "FLOPs = 2 per weight + 4*layers*hidden*context for attention. Model shapes from the Hugging Face configs. "
        "B200 per-GPU numbers derived from NVIDIA's 8-GPU DGX/HGX B200 pages (dense = sparse/2). "
        "WSE-3 numbers as listed by Cerebras. GB and TB are decimal."
    )
    md = "\n\n".join([t1, t2, t3, conditions]) + "\n"
    print(md)
    (RESULTS / "results.md").write_text(md, encoding="utf-8")
    (RESULTS / "results.json").write_text(json.dumps({
        "model": QWEN2_5_72B.name, "params": QWEN2_5_72B.params(), "context": CONTEXT,
        "table1_b200_bf16_by_batch": r1, "table2_ridge": r2, "table3_fit": r3,
        "chips": {k: {"peak_flops": v.peak_flops, "mem_bw": v.mem_bw, "mem_bytes": v.mem_bytes, "ridge": v.ridge,
                      "precision": v.precision, "source": v.source, "derivation": v.derivation} for k, v in CHIPS.items()},
        "conditions": conditions,
    }, indent=2), encoding="utf-8")
    print(f"wrote {RESULTS / 'results.md'} and results.json")


if __name__ == "__main__":
    main()
