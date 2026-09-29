### What one NVIDIA B200 delivers (measured) next to its spec sheet

| quantity | measured | spec sheet (package) |
|---|---:|---:|
| dense BF16 GEMM, 8,192 x 8,192 x 8,192 | 1,593 TFLOPS (71% of peak) | 2,250 TFLOPS |
| HBM read, batch-1 GEMV over 4,096 MiB of bf16 weights | 7.25 TB/s (91% of peak) | 8 TB/s |
| ridge (FLOP per byte) | 220 | 281 |

L2 cache reported by torch: 132,644,864 bytes (126.5 MiB). Intensity limit at 4,096 tokens: 116.3 FLOP/byte, below both ridges.

### The SRAM cliff: a batch-1 decode GEMV (bf16, 8,192 columns) by weight working set

| weights | in L2? | time per pass | bandwidth |
|---:|---|---:|---:|
| 4 MiB | yes | 0.32 us | 13.19 TB/s |
| 8 MiB | yes | 0.63 us | 13.25 TB/s |
| 16 MiB | yes | 1.27 us | 13.18 TB/s |
| 32 MiB | yes | 2.54 us | 13.23 TB/s |
| 48 MiB | yes | 3.81 us | 13.22 TB/s |
| 64 MiB | yes | 5.22 us | 12.85 TB/s |
| 96 MiB | yes | 8.68 us | 11.60 TB/s |
| 128 MiB | no | 12.38 us | 10.85 TB/s |
| 192 MiB | no | 22.42 us | 8.98 TB/s |
| 256 MiB | no | 34.96 us | 7.68 TB/s |
| 512 MiB | no | 74.14 us | 7.24 TB/s |
| 1,024 MiB | no | 148.24 us | 7.24 TB/s |
| 4,096 MiB | no | 592.42 us | 7.25 TB/s |

From L2: up to 13.25 TB/s (at 8 MiB). From HBM: 7.25 TB/s (at 4,096 MiB). Ratio 1.8x. One Qwen2.5-72B-Instruct layer is 1.76 GB of bf16 weights and the model is 145.4 GB, so decode streams from HBM.

### One Qwen2.5-72B-Instruct decoder layer, bf16, 4,096 cached tokens, measured vs the package's bound

| batch | measured per layer | spec bound per layer | speed vs bound | package bytes / time | TFLOPS | compute used (of 2.25 PFLOPS) | bound on compute used | attention share | attention per sequence | rest per sequence | 80-layer tok/s, measured | tok/s bound (package, full model) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 338 us | 222 us | 66% | 5.25 TB/s | 6 | 0.2% | 0.4% | 3% | 10.94 us | 326.82 us | 37 | 55 |
| 4 | 354 us | 228 us | 64% | 5.15 TB/s | 21 | 0.9% | 1.5% | 5% | 4.80 us | 83.67 us | 141 | 212 |
| 16 | 380 us | 253 us | 67% | 5.33 TB/s | 80 | 3.5% | 5.3% | 12% | 2.74 us | 21.01 us | 526 | 767 |
| 64 | 520 us | 354 us | 68% | 5.44 TB/s | 233 | 10.3% | 15.2% | 33% | 2.64 us | 5.48 us | 1,539 | 2,214 |
| 256 | 1,071 us | 756 us | 71% | 5.65 TB/s | 452 | 20.1% | 28.4% | 56% | 2.33 us | 1.86 us | 2,987 | 4,188 |
| 1,024 | 3,922 us | 2,367 us | 60% | 4.83 TB/s | 493 | 21.9% | 36.3% | 59% | 2.28 us | 1.56 us | 3,263 | 5,390 |

### One Qwen2.5-72B-Instruct decoder layer, bf16, 512 cached tokens, measured vs the package's bound

| batch | measured per layer | spec bound per layer | speed vs bound | package bytes / time | TFLOPS | compute used (of 2.25 PFLOPS) | bound on compute used | attention share | attention per sequence | rest per sequence | 80-layer tok/s, measured | tok/s bound (package, full model) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 332 us | 220 us | 66% | 5.29 TB/s | 5 | 0.2% | 0.4% | 2% | 7.87 us | 324.22 us | 38 | 55 |
| 4 | 342 us | 220 us | 64% | 5.15 TB/s | 21 | 0.9% | 1.4% | 2% | 1.99 us | 83.57 us | 146 | 219 |
| 16 | 347 us | 224 us | 64% | 5.15 TB/s | 82 | 3.6% | 5.6% | 3% | 0.72 us | 20.99 us | 576 | 864 |
| 64 | 376 us | 236 us | 63% | 5.02 TB/s | 302 | 13.4% | 21.3% | 7% | 0.42 us | 5.46 us | 2,127 | 3,279 |
| 256 | 565 us | 287 us | 51% | 4.06 TB/s | 803 | 35.7% | 70.4% | 15% | 0.34 us | 1.87 us | 5,663 | 10,873 |
| 1,024 | 1,862 us | 806 us | 43% | 2.10 TB/s | 975 | 43.3% | 100.0% | 17% | 0.31 us | 1.51 us | 6,876 | 15,332 |

At 4,096 tokens, compute used goes 0.2%, 0.9%, 3.5%, 10.3%, 20.1%, 21.9% for batch 1, 4, 16, 64, 256, 1,024. The spec-sheet ceiling is 41% (limit 116.3 / ridge 281.2).
At 512 tokens it goes 0.2%, 0.9%, 3.6%, 13.4%, 35.7%, 43.3% for the same batches.
Attention per sequence at 4,096 tokens: 10.94 us per layer at batch 1, then 2.64, 2.33, 2.28 us at batch 64, 256, 1,024: it floors at the time to read one sequence's 16.8 MB of KV (7.37 TB/s at batch 1,024). The rest of the layer per sequence drops from 326.82 us to 1.56 us.
At batch 1,024 and 4,096 tokens the attention alone takes 2,330 us per layer, about the whole layer's spec bound (2,367 us); the matmuls run after it instead of hiding under it.
Batch 1 at 4,096 tokens runs at 66% of the spec-sheet speed bound: 338 us per layer against 222 us, or 37 tok/s for 80 layers against a full-model bound of 55.

Conditions: one NVIDIA B200 (148 SMs), driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, clocks not locked (default boost), GPU otherwise idle. Seed 0, random weights and KV; no real checkpoint is loaded. Layer: RMSNorm, fused QKV projection, K/V of the new token written into the last of 4,096, 512 cache slots, GQA attention (torch scaled_dot_product_attention, enable_gqa) over all slots, O projection, RMSNorm, SwiGLU MLP, residual adds; no RoPE, no biases, no embedding or LM head. Each config chains 4 distinct layers (own weights, own KV; fewer if they exceed 100 GB) in one CUDA graph; 5 warmup replays, median of 30 timed replays (CUDA events), divided by the layer count. Attention share: the same graph with only the attention calls. 80-layer tok/s = batch / (80 x measured layer time). Spec bound per layer: the package's bound() on that layer's bytes and FLOPs (B200 BF16 dense 2.25 PFLOPS, 8 TB/s). GEMV probe: a Triton kernel, two weight rows per step, 8,192 bf16 columns, fp32 accumulate, persistent grid, repeated passes over the same weights inside one launch (about 16 GB read per launch); GEMM probe: torch.matmul bf16. Probes: 3 warmup launches, median of 15. Bandwidth = weight bytes / time. GB and TB are decimal, MiB binary.
