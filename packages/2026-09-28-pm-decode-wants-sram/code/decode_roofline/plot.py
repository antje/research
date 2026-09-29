"""Write the package diagram (../image.svg): intensity vs batch, with ridge lines.

What: a log-log chart of one decode step's arithmetic intensity against batch size for
three context lengths, with the B200 and WSE-3 ridge points as horizontal lines.
Why: the whole post is "the curve flattens under the ridge"; the picture should show it.
Where: run `python -m decode_roofline.plot`; the SVG follows ../../style/diagram.md and is
rendered to PNG by tools/render_svg.py.
"""

from __future__ import annotations

import math
from pathlib import Path

from .chips import B200_BF16, WSE3
from .model import QWEN2_5_72B
from .roofline import batch_at_ridge, decode_step, intensity_limit

INK, SURFACE, RULE, MUTED, PAPER, AMBER = "#0b0c0f", "#14161b", "#2a2d34", "#9a978f", "#f3efe6", "#f0b429"
W, H = 1600, 900
X0, X1, Y0, Y1 = 260, 1500, 215, 690          # plot area (pixels)
BMIN, BMAX = 1, 4096                          # batch axis (log2)
IMIN, IMAX = 1.0, 1000.0                      # intensity axis (log10)
CONTEXTS = [(512, PAPER, "512-token context"), (4096, AMBER, "4,096-token context"), (32768, MUTED, "32,768-token context")]
OUT = Path(__file__).resolve().parents[2] / "image.svg"


def x_of(batch: float) -> float:
    return X0 + (math.log2(batch) - math.log2(BMIN)) / (math.log2(BMAX) - math.log2(BMIN)) * (X1 - X0)


def y_of(intensity: float) -> float:
    i = min(max(intensity, IMIN), IMAX)
    return Y1 - (math.log10(i) - math.log10(IMIN)) / (math.log10(IMAX) - math.log10(IMIN)) * (Y1 - Y0)


def curve(context: int) -> str:
    pts = []
    n = 80
    for k in range(n + 1):
        b = BMIN * (BMAX / BMIN) ** (k / n)
        i = decode_step(QWEN2_5_72B, 1, context).flops * b / (QWEN2_5_72B.params() * 2 + b * context * QWEN2_5_72B.kv_bytes_per_token(2))
        pts.append(f"{x_of(b):.1f},{y_of(i):.1f}")
    return " ".join(pts)


def text(x: float, y: float, s: str, size: int = 22, fill: str = PAPER, weight: int = 400, mono: bool = False, anchor: str = "start") -> str:
    fam = 'font-family="Geist Mono, monospace"' if mono else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {fam}>{s}</text>'


def main() -> None:
    lim4k = intensity_limit(QWEN2_5_72B, 4096)
    b512 = batch_at_ridge(QWEN2_5_72B, B200_BF16, 512)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Geist, Inter, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="{INK}"/>',
        text(80, 110, "A 72B decode step never reaches the B200 ridge at 4k context", 44, PAPER, 500),
        text(80, 150, "One decode step of Qwen2.5-72B-Instruct, bf16 weights and KV, by batch size. Spec-sheet bounds.", 22, MUTED),
    ]
    # grid and axes
    for i in (1, 10, 100, 1000):
        y = y_of(i)
        parts.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="{RULE}" stroke-width="1" opacity="0.6"/>')
        parts.append(text(X0 - 16, y + 7, f"{i:,}", 20, MUTED, 400, True, "end"))
    for b in (1, 4, 16, 64, 256, 1024, 4096):
        x = x_of(b)
        parts.append(f'<line x1="{x:.1f}" y1="{Y0}" x2="{x:.1f}" y2="{Y1}" stroke="{RULE}" stroke-width="1" opacity="0.6"/>')
        parts.append(text(x, Y1 + 32, f"{b:,}", 20, MUTED, 400, True, "middle"))
    parts.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="{MUTED}" stroke-width="1"/>')
    parts.append(f'<line x1="{X0}" y1="{Y0}" x2="{X0}" y2="{Y1}" stroke="{MUTED}" stroke-width="1"/>')
    parts.append(text((X0 + X1) / 2, Y1 + 64, "batch size (sequences decoding together)", 22, MUTED, 400, False, "middle"))
    parts.append(f'<text x="{X0 - 200}" y="{(Y0 + Y1) / 2:.1f}" font-size="22" fill="{MUTED}" text-anchor="middle" transform="rotate(-90 {X0 - 200} {(Y0 + Y1) / 2:.1f})">FLOP per byte moved</text>')
    # ridge lines
    for chip, label, x, anchor in ((B200_BF16, f"B200 ridge: 2.25 PFLOPS dense BF16 / 8 TB/s HBM3e = {chip_ridge(B200_BF16)} FLOP/byte", X0 + 12, "start"),
                                   (WSE3, f"WSE-3 ridge: 125 PFLOPS / 21 PB/s on-chip SRAM = {chip_ridge(WSE3)} FLOP/byte", X1, "end")):
        y = y_of(chip.ridge)
        parts.append(f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="{PAPER}" stroke-width="2" stroke-dasharray="10 8"/>')
        parts.append(text(x, y - 10, label, 20, PAPER, 500, True, anchor))
    # curves
    for ctx, color, label in CONTEXTS:
        width = 4 if color == AMBER else 3
        parts.append(f'<polyline points="{curve(ctx)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')
    # curve labels at the right edge: 512 above its curve, the others below
    for ctx, color, label in CONTEXTS:
        i_end = decode_step(QWEN2_5_72B, BMAX, ctx).intensity
        dy = -14 if ctx == 512 else 30
        parts.append(text(X1 - 8, y_of(i_end) + dy, label, 20, PAPER if color == AMBER else color, 500, False, "end"))
    # the pointer: the 4k plateau, written under the amber curve
    parts.append(text(X0 + 20, y_of(lim4k) - 14, f"4,096 tokens: tops out at {lim4k:.0f} FLOP/byte, below the ridge at any batch", 22, AMBER, 600))
    # the crossing at 512 context
    if b512 is not None:
        x, y = x_of(b512), y_of(B200_BF16.ridge)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{INK}" stroke="{PAPER}" stroke-width="2"/>')
        parts.append(text(x + 16, y + 30, f"batch {b512} at 512 tokens", 20, PAPER, 400))
    # caption: three lines, conditions then sources
    parts.append(text(80, 800, "bytes = weights once per step + KV once per sequence; FLOPs = 2 per weight + attention over the context", 20, MUTED, 400, True))
    parts.append(text(80, 828, "B200 per GPU from NVIDIA DGX/HGX B200 pages (dense = sparse/2); WSE-3 as listed by Cerebras", 20, MUTED, 400, True))
    parts.append(text(80, 856, "Qwen2.5-72B-Instruct: 72.7B params, 80 layers, 8 KV heads, 320 KiB KV/token", 20, MUTED, 400, True))
    parts.append(text(1520, 856, "research · github.com/antje/research", 20, MUTED, 400, True, "end"))
    parts.append("</svg>")
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


def chip_ridge(chip) -> str:
    return f"{chip.ridge:.0f}"


if __name__ == "__main__":
    main()
