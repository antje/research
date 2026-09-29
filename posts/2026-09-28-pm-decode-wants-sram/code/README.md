# decode_roofline

What one LLM decode step costs in bytes and FLOPs, and which memory it wants. The tables below are spec-sheet bounds, not measurements. A separate GPU experiment (last section) measures the same decode step on one NVIDIA B200.

```bash
uv run pytest -q
uv run python -m decode_roofline        # prints three tables, writes results/results.{md,json}
uv run python -m decode_roofline.plot   # writes ../image.svg
```

Expected output: [`results/results.md`](results/results.md). The tables cover one decode step of Qwen2.5-72B-Instruct at 4,096 cached tokens on a B200 by batch size, the ridge points of a B200 (BF16 and FP8) and a Cerebras WSE-3 with the batch that reaches them at 512 / 4,096 / 32,768 tokens of context, and what fits in 180 GB of HBM3e versus 44 GB of on-chip SRAM.

Modules: [`model.py`](decode_roofline/model.py) (shapes from a Hugging Face config.json: parameters, KV bytes per token, FLOPs per token), [`chips.py`](decode_roofline/chips.py) (spec-sheet numbers with their derivation and source id; the sources are listed at the end of [`../README.md`](../README.md)), [`roofline.py`](decode_roofline/roofline.py) (bytes, FLOPs, intensity, ridge, intensity limit, batch at ridge, fit), [`plot.py`](decode_roofline/plot.py) (the diagram).

To try another model, add a `ModelShape` from its config.json fields (hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size, vocab_size, tie_word_embeddings).

## GPU experiment: the same decode step, measured

[`gpu_decode.py`](decode_roofline/gpu_decode.py) checks the roofline on a real GPU and prints the measurements next to the package's bounds. It runs three things:

1. What the chip delivers: a dense BF16 GEMM (8,192 cubed) and the HBM read bandwidth of a batch-1 GEMV, and the measured ridge they imply.
2. The SRAM cliff: the same batch-1 GEMV (bf16, 8,192 columns) over weight working sets from 4 MiB to 4 GiB. Sets that fit in L2 stream from on-chip SRAM; larger ones stream from HBM.
3. One Qwen2.5-72B decoder layer, with random bf16 weights in the shapes from [`model.py`](decode_roofline/model.py): fused QKV projection, GQA attention over the KV cache, O projection, SwiGLU MLP. It is timed at batch 1 to 1,024 and at 4,096 and 512 cached tokens, and the attention is timed on its own so the KV read shows up separately. Each result sits next to the package's `bound()` for that layer's bytes and FLOPs.

It needs a CUDA GPU and torch (triton ships with torch on Linux). The spec-sheet tables and the tests do not need either.

```bash
uv sync --extra gpu
uv run python -m decode_roofline.gpu_decode    # under 10 s on a B200; writes results/b200-decode.{md,json}
```

The committed [`results/b200-decode.md`](results/b200-decode.md) came from one NVIDIA B200 (driver 580.173.02, CUDA 13.0, torch 2.13.0), with clocks not locked. Its last line lists the conditions. [`gpu_kernels.py`](decode_roofline/gpu_kernels.py) holds the torch and Triton code, and [`gpu_decode.py`](decode_roofline/gpu_decode.py) holds the accounting, which is tested on CPU in [`tests/test_gpu_decode.py`](tests/test_gpu_decode.py). The GPU tests there skip when no CUDA device is present.
