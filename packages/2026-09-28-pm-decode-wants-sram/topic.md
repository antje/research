# Topic: Decode wants SRAM: a roofline you can run on your laptop

Theme: FOCUS.md § 3, item 3 (heterogeneous silicon economics with conditions attached). Backlog id b01. Hook: brief item 1 (Gimlet + Cerebras, Sep 28).

Claim: at 4,096 tokens of context, one decode step of a 72B dense model cannot reach a B200's ridge point at any batch size, because the KV read grows with the batch and caps arithmetic intensity at 116 FLOP/byte against a ridge of 281; an SRAM-centric chip moves the ridge to about 6 FLOP/byte, so small batches saturate compute there, at the price of a 44 GB capacity ceiling.

Mechanism: a decode step reads every weight once per step and every sequence's KV once; FLOPs are two per weight plus attention over the context. Intensity = FLOPs / bytes rises with batch as the weight read amortizes, but converges to FLOPs-per-token divided by KV-bytes-per-sequence. If that limit is below the chip's ridge (peak FLOPs / bandwidth), the step stays memory-bound at every batch. The ridge is a ratio, and on-chip SRAM bandwidth (21 PB/s on a WSE-3) lowers it by almost two orders of magnitude, but on-chip capacity (44 GB) decides what can run there.

What the code shows: bytes, FLOPs, intensity, and B200 bounds by batch (table 1); ridge points of B200 BF16, B200 FP8, and WSE-3 with the batch that reaches them at 512, 4,096, and 32,768 tokens (table 2); what fits in 180 GB of HBM3e versus 44 GB of SRAM (table 3); the diagram.

Numbers from sources: Qwen2.5-72B/7B config fields and parameter counts; B200 memory and tensor-core specs (NVIDIA DGX/HGX pages, per-GPU derivation); WSE-3 44 GB, 125 PFLOPS, 21 PB/s (Cerebras); SRAM sizes and the "distribute weights across chips" point (Gimlet SRAM post); the Cerebras partnership facts (Gimlet, Sep 28). Numbers from results/: everything computed (145.4 GB, 320 KiB, intensities, 116.3 limit, 281.2 and 6.0 ridges, batch 411, tok/s bounds, fit).

Transferable rule: before buying batch for decode, compute FLOPs per token divided by KV bytes per sequence. If it is under the ridge, batch raises throughput but never makes the step compute-bound; shrink KV bytes (FP8 KV, GQA, MLA), keep less context in cache, or move decode to memory with a lower ridge.
