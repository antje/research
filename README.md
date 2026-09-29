# research

Runnable notes on inference systems: heterogeneous inference, kernels, profilers, schedulers,
RL and training infrastructure, and the economics of running models on NVIDIA, AMD, Google TPU,
AWS Trainium, Cerebras, and d-Matrix silicon. Every post has a short explainer, a diagram, and
code you can run on a laptop (`uv run`). Numbers carry their conditions and sources.

| Date | Post | Summary |
|---|---|---|
| 2026-09-28 | [Decode wants SRAM](posts/2026-09-28-pm-decode-wants-sram/) | At 4,096 tokens of context, one decode step of a 72B dense model cannot reach a B200's ridge point at any batch size, because the KV read grows with the batch and caps arithmetic intensity at 116 FLOP/byte against a… |
