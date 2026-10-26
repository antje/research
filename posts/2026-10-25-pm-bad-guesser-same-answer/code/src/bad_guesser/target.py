"""Qwen3-8B's next-token distribution at a few fixed prefixes, two ways: through the model's own bf16 output head
(what vLLM softmaxes in float32, vllm-code-follows: `raw_target_logits.to(torch.float32)` then softmax) and through
the same final hidden state multiplied by the output head in float32. The gap between the two is the 'hardware
numerics' the papers and vLLM's documentation name (chen-numerics, vllm-docs-lossless), and it is the same gap plain
sampling without a drafter would see.

transformers and torch are imported inside the functions (the `gpu` extra). The weights come from the Hugging Face
cache, offline; nothing is downloaded.
"""

from __future__ import annotations

import glob
import os

MODEL = "Qwen/Qwen3-8B"
VOCAB = 151_936  # qwen3-8b-vocab
PREFIXES = [   # chosen for a spread of entropies, from one certain token to a wide-open continuation
    "import numpy as",
    "The capital of France is",
    "The number I am thinking of is",
    "Dear Sir or Madam,\n\nI am writing to",
    "My favorite color is",
    "Once upon a time, there was a",
]


def snapshot_dir() -> str:
    home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hits = glob.glob(os.path.join(home, "hub", "models--" + MODEL.replace("/", "--"), "snapshots", "*", "config.json"))
    if not hits:
        raise FileNotFoundError(f"{MODEL} is not in the Hugging Face cache under {home}")
    return os.path.dirname(hits[0])


def load(device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    snap = snapshot_dir()
    tok = AutoTokenizer.from_pretrained(snap)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).to(device).eval()
    assert model.config.vocab_size == VOCAB
    return tok, model


def next_token(tok, model, prefix: str) -> dict:
    """The logits for the token after `prefix`: the bf16 head's (as bf16), and the float32 head's on the same final
    hidden state (post-norm, the input to the head). Also the top-5 tokens for the report."""
    import torch

    ids = tok(prefix, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1]                        # final hidden state after the model's last norm
        logits_bf16 = out.logits[0, -1]                         # the model's own head, bf16 out
        logits_fp32 = h.float() @ model.lm_head.weight.float().T
        assert logits_bf16.dtype == torch.bfloat16
        # sanity: the two heads see the same hidden state (they differ by bf16 output rounding only, well under 1 logit)
        assert (logits_bf16.float() - logits_fp32).abs().max().item() < 0.5
    p32 = torch.softmax(logits_fp32, -1)
    top = torch.topk(p32, 5)
    return dict(prefix=prefix, tokens=ids.shape[1], logits_bf16=logits_bf16, logits_fp32=logits_fp32,
                entropy=-(p32 * torch.log(p32.clamp_min(1e-45))).sum().item(),
                top5=[(tok.decode([i]), v) for v, i in zip(top.values.tolist(), top.indices.tolist())])


def probabilities(logits_bf16, logits_fp32) -> dict:
    """Three versions of the target: the float32 head's softmax (the reference), vLLM's path (bf16 logits widened to
    float32, then softmax), and a softmax whose output is stored in bf16 (what a drafter emitting bf16 probabilities
    would hand over). All returned as float32 vectors."""
    import torch

    return {
        "float32 head, float32 softmax": torch.softmax(logits_fp32, -1),
        "bf16 head, float32 softmax (vLLM's path)": torch.softmax(logits_bf16.float(), -1),
        "bf16 head, probabilities stored in bf16": torch.softmax(logits_bf16.float(), -1).to(torch.bfloat16).float(),
    }
