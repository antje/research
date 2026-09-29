# decode_roofline

What one LLM decode step costs in bytes and FLOPs, and which memory it wants. Spec-sheet bounds, not measurements; the sandbox this was written in has no GPU, and no GPU path is included.

```bash
uv run pytest -q
uv run python -m decode_roofline        # prints three tables, writes results/results.{md,json}
uv run python -m decode_roofline.plot   # writes ../image.svg
```

Expected output: `results/results.md`. The tables cover one decode step of Qwen2.5-72B-Instruct at 4,096 cached tokens on a B200 by batch size, the ridge points of a B200 (BF16 and FP8) and a Cerebras WSE-3 with the batch that reaches them at 512 / 4,096 / 32,768 tokens of context, and what fits in 180 GB of HBM3e versus 44 GB of on-chip SRAM.

Modules: `model.py` (shapes from a Hugging Face config.json: parameters, KV bytes per token, FLOPs per token), `chips.py` (spec-sheet numbers with their derivation and source id; see `../sources.json`), `roofline.py` (bytes, FLOPs, intensity, ridge, intensity limit, batch at ridge, fit), `plot.py` (the diagram).

To try another model, add a `ModelShape` from its config.json fields (hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size, vocab_size, tie_word_embeddings).
