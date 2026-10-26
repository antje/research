"""vLLM 0.31's own rejection sampler (vllm/v1/sample/rejection_sampler.py, the Triton kernels behind every
speculative-decoding step it serves) on the same target and drafts as the torch rule.

`rejection_sample` is called the way `RejectionSampler.forward` calls it: one request per row, one draft token per
request, the target's logits in float32 (it softmaxes them itself), the draft's probabilities as a [rows, vocab]
tensor, or None for the n-gram and greedy case where the kernel sets the draft probability to 1 (vllm-code-nodraft).
The first output column is the verified token; the second holds the bonus token when the draft was kept and the
placeholder -1 when it was not, which is how acceptance is counted here. Temperature is 1, so the kernel's in-place
division of the logits leaves them unchanged and the same logits tensor serves every batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rule import cdf64, draw


@dataclass
class KernelOutcome:
    hist: torch.Tensor
    n: int
    accepted: int
    bonus_consistent: bool   # the bonus column is present exactly when the first column equals the draft


def _metadata(rows: int, device):
    from vllm.v1.sample.logits_processor.state import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata

    ones, zeros = torch.ones(rows, device=device), torch.zeros(rows, device=device)
    return SamplingMetadata(temperature=ones, all_greedy=False, all_random=True, top_p=None, top_k=None, generators={},
                            max_num_logprobs=None, no_penalties=True, prompt_token_ids=None, frequency_penalties=zeros,
                            presence_penalties=zeros, repetition_penalties=ones, output_token_ids=[[] for _ in range(rows)],
                            allowed_token_ids_mask=None, bad_words_token_ids={}, logitsprocs=LogitsProcessors())


def speculate(logits: torch.Tensor, q: torch.Tensor | None, draft_from: torch.Tensor, n: int, gen: torch.Generator,
              rows: int = 8192) -> KernelOutcome:
    """Run vLLM's kernel n times in batches of `rows`. `logits` [vocab] float32 is the target; `q` [vocab] the draft
    probabilities handed to the kernel (None for the no-draft-probabilities path); `draft_from` the distribution the
    draft tokens are drawn from (equal to q when q is given; a one-hot for the n-gram case)."""
    from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID, rejection_sample

    device = logits.device
    V = logits.numel()
    meta = _metadata(rows, device)
    target = logits.repeat(rows, 1).contiguous()
    draft_probs = None if q is None else q.repeat(rows, 1).contiguous()
    cu = torch.arange(1, rows + 1, device=device, dtype=torch.int32)
    bonus = torch.zeros((rows, 1), device=device, dtype=torch.int64)
    cdf = cdf64(draft_from)
    hist = torch.zeros(V, dtype=torch.int64, device=device)
    accepted = 0
    consistent = True
    for _ in range(n // rows):
        ids = draw(cdf, rows, gen).to(torch.int32)
        out = rejection_sample(ids, [1] * rows, 1, cu, draft_probs, target, bonus, meta)
        kept = out[:, 1] != PLACEHOLDER_TOKEN_ID
        consistent &= bool(torch.equal(kept, out[:, 0] == ids))
        hist += torch.bincount(out[:, 0].long(), minlength=V)
        accepted += int(kept.sum().item())
    return KernelOutcome(hist, (n // rows) * rows, accepted, consistent)
