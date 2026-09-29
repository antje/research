### One decode step, Qwen2.5-72B-Instruct, bf16 weights and KV, 4,096 cached tokens, on one B200 (BF16 dense, 8 TB/s)

| batch | weights read | KV read | FLOPs | intensity (FLOP/byte) | limiter | tok/s bound | compute used |
|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 145.4 GB | 1.3 GB | 0.16 TFLOP | 1.1 | memory | 55 | 0.38% |
| 4 | 145.4 GB | 5.4 GB | 0.62 TFLOP | 4.1 | memory | 212 | 1.47% |
| 16 | 145.4 GB | 21.5 GB | 2.50 TFLOP | 15.0 | memory | 767 | 5.32% |
| 64 | 145.4 GB | 85.9 GB | 9.99 TFLOP | 43.2 | memory | 2,214 | 15.36% |
| 256 | 145.4 GB | 343.6 GB | 39.97 TFLOP | 81.7 | memory | 4,188 | 29.06% |
| 1,024 | 145.4 GB | 1,374.4 GB | 159.89 TFLOP | 105.2 | memory | 5,390 | 37.41% |

Intensity limit as batch grows (weights amortized, KV not): 116.3 FLOP/byte. B200 BF16 ridge: 281 FLOP/byte.
Intensity limits by context (bf16 KV): 512 tokens: 874.7, 4,096 tokens: 116.3, 32,768 tokens: 21.5 FLOP/byte.
Batch 1 detail: 146.8 GB moved, step time 18.3 ms on the B200, 0.16 TFLOP.
Batch 6 detail (the batch that saturates a WSE-3): on the B200, 153.5 GB moved, t_mem 19.2 ms, t_comp 0.42 ms, compute used 2.2%.
Capacity on one B200 with bf16 weights: 34.6 GB free, room for 105,566 KV tokens, about 25 sequences of 4,096 tokens.

### Ridge points and the batch that reaches them, Qwen2.5-72B-Instruct

| chip | weights + KV precision | peak (dense) | bandwidth | ridge (FLOP/byte) | batch at ridge, 512 ctx | batch at ridge, 4,096 ctx | batch at ridge, 32,768 ctx |
|---|---|---:|---:|---:|---:|---:|---:|
| NVIDIA B200 (HBM3e) | bf16 / bf16 | 2.25 PFLOPS BF16 dense | 8 TB/s | 281.2 | 411 | never (limit 116.3) | never (limit 21.5) |
| NVIDIA B200 (HBM3e) | fp8 / fp8 | 4.5 PFLOPS FP8 dense | 8 TB/s | 562.5 | 411 | never (limit 232.7) | never (limit 43.1) |
| Cerebras WSE-3 (on-chip SRAM) | bf16 / bf16 | 125 PFLOPS peak AI performance as listed | 21 PB/s | 6.0 | 6 | 6 | 6 |

### What fits: weights vs the memory that feeds the compute

| model | weights + KV precision | weights | B200 HBM3e 180 GB | WSE-3 SRAM 44 GB | KV/token |
|---|---|---:|---|---|---:|
| Qwen2.5-72B-Instruct (72.7B) | bf16 / bf16 | 145.4 GB | fits, room for 105,566 KV tokens | does not fit | 320 KiB |
| Qwen2.5-72B-Instruct (72.7B) | fp8 / fp8 | 72.7 GB | fits, room for 654,882 KV tokens | does not fit | 160 KiB |
| Qwen2.5-7B-Instruct (7.6B) | bf16 / bf16 | 15.2 GB | fits, room for 2,873,350 KV tokens | fits, room for 501,699 KV tokens | 56 KiB |
| Qwen2.5-7B-Instruct (7.6B) | fp8 / fp8 | 7.6 GB | fits, room for 6,012,301 KV tokens | fits, room for 1,268,998 KV tokens | 28 KiB |

Conditions: spec-sheet bounds, not measurements. Bytes = weights once per step + KV once per sequence; FLOPs = 2 per weight + 4*layers*hidden*context for attention. Model shapes from the Hugging Face configs. B200 per-GPU numbers derived from NVIDIA's 8-GPU DGX/HGX B200 pages (dense = sparse/2). WSE-3 numbers as listed by Cerebras. GB and TB are decimal.
