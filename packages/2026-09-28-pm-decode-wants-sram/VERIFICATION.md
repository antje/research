# Verification: 2026-09-28-pm-decode-wants-sram

| check | result | notes |
|---|---|---|
| 1. Source check (`tools/verify_claims.py`) | PASS | 70 numeric tokens across README.md, tweets.md, image.svg resolved to `sources.json` quotes, `code/results/results.md`, or `claims-allow.txt` (27 derivations: per-GPU B200 numbers from NVIDIA's 8-GPU pages, roundings, unit spellings). 0 unresolved. |
| 2. Fresh-eyes review (subagent, checklist in style/voice.md) | PASS after fixes | Reviewer re-derived every number from the configs and spec sheets (all matched within rounding) and checked the six live source pages. Two blocking findings, both fixed: tweet 3 said a B200 sits "under 1%" at the batch that saturates a WSE-3 (it is 2.2% at batch 6; now "about 2%", with the batch-6 detail added to results.md), and README said the WSE-3 is compute-bound in the regime where a B200 sits at 0.4% (batch 1; the WSE-3 is memory-bound there too; now "by batch 6"). Non-blocking findings applied: capacity caveat (bf16 weights leave 34.6 GB, about 25 sequences of 4k KV), NVIDIA sparse/dense footnotes quoted verbatim, Gimlet's token-generation sentence quoted instead of "reads as a decode story to me", WSE-3 precision caveat, assumptions sentence (single GPU, dense, one token per step, full KV read, peak bandwidth), FP8-KV-alone note in the rule, wording nits, image footer and amber usage, table 3 KV bytes matched to precision, numpy dependency removed. Remaining non-blocking: README body is over the 900-word target after the caveats; no $/GPU-hour figure (FOCUS theme 3 asks for tok/s/$ where possible). |
| 2b. `tools/lint_style.py` | PASS | 0 failures, 0 warnings; 4 tweets (all under 280 with the public link), 5-tweet thread, LinkedIn 2,298 chars (long form, link as the last line, 196-char first line), alt text 273 chars, no banned patterns, no em dashes |
| 3a. Execution | PASS | `cd code && uv run pytest -q` -> 8 passed; `uv run python -m decode_roofline` wrote `results/results.md` and `results/results.json`; `uv run python -m decode_roofline.plot` wrote `../image.svg` |
| 3b. Visual | PASS | `image.png` (1600x900) and `image-square.png` (1080x1080) rendered with the bundled Geist fonts and inspected: title, the three context curves, the two ridge lines (281 and 6 FLOP/byte), the batch-411 marker, the 116 FLOP/byte plateau label, and the caption all match README.md. One amber element (the 4,096-token curve and its label). No text under 20 px. |

## Claims cut or changed during verification
- "the GPU sits under 1% utilization" at batch 6 was wrong (2.2%); changed to "about 2% of its tensor cores".
- "The regime where a GPU sits at 0.4% compute utilization is the regime where this chip is compute-bound" overclaimed (batch 1 is memory-bound on the WSE-3 too); changed to the batch-6 statement.
- "flattens in the 30s" (tensor-core utilization) was replaced by "flattens below 41%", the actual limit (116.3 / 281.2), so the tell is a computed number rather than an eyeballed range.
- Thread tweets 3 and 5 and the LinkedIn variant were shortened to meet the length rules; no facts removed.
- No source-backed claim was removed.

## Honesty notes
- Gimlet Labs is cited through two of its public posts, like any other vendor; the package states no employer or job title for the author.
- Every number is a spec-sheet bound or a config-derived quantity; nothing was measured on hardware (the sandbox has no GPU).
- B200 per-GPU figures are derived from NVIDIA's 8-GPU DGX/HGX pages under the assumption that the listed tensor-core PFLOPS are with-sparsity figures (NVIDIA's "Sparse | Dense" footnote convention); the derivation is stated in `code/decode_roofline/chips.py`, `claims-allow.txt`, and the README sources list.
- WSE-3 peak (125 petaflops) has no stated precision in the Cerebras press release; the text says "as listed".

## Commands run
```
uv run python tools/fetch_sources.py --out packages/2026-09-28-pm-decode-wants-sram/raw
cd packages/2026-09-28-pm-decode-wants-sram/code && uv run pytest -q && uv run python -m decode_roofline && uv run python -m decode_roofline.plot
uv run python tools/render_svg.py packages/2026-09-28-pm-decode-wants-sram/image.svg
uv run python tools/verify_claims.py packages/2026-09-28-pm-decode-wants-sram
uv run python tools/lint_style.py packages/2026-09-28-pm-decode-wants-sram
```
