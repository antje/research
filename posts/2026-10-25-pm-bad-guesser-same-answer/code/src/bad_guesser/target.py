"""Qwen3-8B's next-token distribution at a few fixed prefixes, two ways: through the model's own bf16 output head
(what vLLM's rejection sampler softmaxes after `raw_target_logits.to(torch.float32)`,
https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py) and through the same final
hidden state multiplied by the output head in float32. The gap between the two is the 'hardware numerics' that Chen
et al. (Accelerating Large Language Model Decoding with Speculative Sampling, https://arxiv.org/abs/2302.01318) and
vLLM's Speculative Decoding documentation (https://docs.vllm.ai/en/latest/features/speculative_decoding.html) name,
and it is the same gap plain sampling without a drafter would see.

transformers and torch are imported inside the functions (the `gpu` extra). The weights come from the Hugging Face
cache, offline; nothing is downloaded.
"""

from __future__ import annotations

import glob
import os

MODEL = "Qwen/Qwen3-8B"
VOCAB = 151_936  # Qwen/Qwen3-8B config.json, vocab_size: https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
PREFIXES = [   # chosen for a spread of entropies, from one certain token to a wide-open continuation
    "import numpy as",
    "The capital of France is",
    "The number I am thinking of is",
    "Dear Sir or Madam,\n\nI am writing to",
    "My favorite color is",
    "Once upon a time, there was a",
]
REFERENCE = "float32 head, float32 softmax"
BF16_STORED = "bf16 head, probabilities stored in bf16"


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


def stored_in_bf16(probs):
    """A probability vector rounded to bf16, as a drafter or target emitting bf16 probabilities would hand it over,
    then renormalized in float32. Rounding alone leaves a vector that no longer sums to 1 (0.5, 0.3, 0.2 rounds to a
    vector summing to 1.0009765625), and a rule fed that vector would compare one distribution and sample another.
    Returns the renormalized vector and the rounded vector's sum before renormalizing."""
    import torch

    r = probs.to(torch.bfloat16).float()
    s = r.double().sum().item()
    return r / r.sum(), s


def versions(logits_bf16, logits_fp32) -> dict:
    """Three versions of the target, each a float32 vector summing to 1: the float32 head's softmax (the reference),
    vLLM's path (bf16 logits widened to float32, then softmax), and that softmax stored in bf16 and renormalized.
    Each maps to (p, the sum of p before renormalizing)."""
    import torch

    vllm_path = torch.softmax(logits_bf16.float(), -1)
    ref = torch.softmax(logits_fp32, -1)
    return {
        REFERENCE: (ref / ref.sum(), ref.double().sum().item()),
        "bf16 head, float32 softmax (vLLM's path)": (vllm_path / vllm_path.sum(), vllm_path.double().sum().item()),
        BF16_STORED: stored_in_bf16(vllm_path),
    }
