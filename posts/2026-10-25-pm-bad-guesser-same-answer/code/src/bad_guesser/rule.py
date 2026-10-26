"""The acceptance rule and four plausible wrong rules, run millions of times in torch, with the statistics that
decide whether a histogram is the target's.

The rule is Leviathan, Kalman and Matias, Fast Inference from Transformers via Speculative Decoding, § 2.3, with the
lenient variant of their Appendix A.5 (https://arxiv.org/abs/2211.17192). The wrong rules are the shapes a relaxed
checker takes: keep when the target finds the guess at least as likely, keep the target's top token, keep anything
above a probability threshold (the shape of Medusa's typical acceptance, Cai et al., Medusa, § 2.3.1,
https://arxiv.org/abs/2401.10774).

Sampling is by inverse CDF in float64 (a cumulative sum of the probabilities, a uniform draw, a binary search), so the
draws from the draft q and from the replacement distribution are exact to far below the sampling noise and the only
float32 arithmetic in the loop is the rule's own comparison p(x) / q(x) >= u, written as vLLM's rejection sampler
writes it, with u in float64 as vLLM draws it (vllm/v1/sample/rejection_sampler.py,
https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py). `arith` switches that comparison
and the replacement distribution to float64 so the two can be compared draw for draw.

p and q must each sum to 1 (`speculate` checks it): a vector that does not would mean the keep test reads one
distribution while the inverse-CDF sampler, which divides by the total, draws from another.

Two statistics per histogram: the total-variation distance to the target (half the sum of absolute differences, the
largest probability any event can gain) and a chi-square per degree of freedom over the tokens the run expects to see
at least `min_expected` times, the rest pooled into one bin. An exact rule gives chi-square per degree of freedom near
1 whatever the draw count; a rule that is off by a fixed amount grows linearly with the draw count.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

EXACT = "exact"             # keep with probability min(1, p/q), replace from norm(max(0, p - q))
LENIENT = "lenient"         # the same with q multiplied by l before the comparison (Leviathan et al., Appendix A.5)
RATIO_ONLY = "ratio-only"   # wrong: keep if p >= q, else sample the target afresh
ARGMAX = "argmax"           # wrong: keep if the guess is the target's most likely token, else sample the target afresh
THRESHOLD = "threshold"     # wrong: keep if p(guess) > t, else sample the target afresh (a hard typical-acceptance threshold)
WRONG = (RATIO_ONLY, ARGMAX, THRESHOLD)
MASS_TOLERANCE = 1e-4       # a float32 softmax sums to 1 within about 1e-6; a bf16-rounded vector misses by up to about 1e-3


@dataclass
class Outcome:
    hist: torch.Tensor      # int64 [vocab], how often each token came out
    n: int
    accepted: int           # how many guesses were kept


def check_mass(*vectors: torch.Tensor) -> None:
    for v in vectors:
        s = v.double().sum().item()
        assert abs(s - 1.0) < MASS_TOLERANCE, f"a probability vector sums to {s!r}, not 1: renormalize it first"


def cdf64(probs: torch.Tensor) -> torch.Tensor:
    c = torch.cumsum(probs.double(), 0)
    return c / c[-1]


def draw(cdf: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """n exact draws from the distribution whose cumulative sum is cdf. A token of zero mass is never returned."""
    u = torch.rand(n, dtype=torch.float64, device=cdf.device, generator=gen)
    return torch.searchsorted(cdf, u).clamp_(max=cdf.numel() - 1)


def overlap(p: torch.Tensor, q: torch.Tensor) -> float:
    """The predicted acceptance rate: the sum of min(p, q) (Leviathan et al., Theorem 3.5)."""
    return torch.minimum(p.double(), q.double()).sum().item()


def keep_test(rule: str, p: torch.Tensor, q: torch.Tensor, threshold: float) -> torch.Tensor:
    """For the wrong rules: the fixed per-token test that decides whether a guess is kept."""
    if rule == RATIO_ONLY:
        return p >= q
    if rule == ARGMAX:
        return torch.arange(p.numel(), device=p.device) == p.argmax()
    if rule == THRESHOLD:
        return p > threshold
    raise ValueError(rule)


def replacement(rule: str, p: torch.Tensor, q: torch.Tensor, lenience: float) -> torch.Tensor:
    """What a thrown-out guess is replaced from: the clipped difference for the exact and lenient rules, the target
    itself for the wrong ones."""
    if rule == EXACT:
        return torch.clamp(p - q, min=0)
    if rule == LENIENT:
        return torch.clamp(p - lenience * q, min=0)
    return p


def predicted(rule: str, p: torch.Tensor, q: torch.Tensor, lenience: float = 1.0, threshold: float = 0.05) -> torch.Tensor:
    """The output distribution each rule produces, in closed form (the exact rule's is p itself)."""
    p64, q64 = p.double(), q.double()
    if rule in (EXACT, LENIENT):
        l = lenience if rule == LENIENT else 1.0
        kept = torch.minimum(q64, p64 / l)
        r = replacement(rule, p64, q64, l)
        return kept + (1.0 - kept.sum()) * r / r.sum()
    kept = torch.where(keep_test(rule, p64, q64, threshold), q64, torch.zeros_like(q64))
    return kept + (1.0 - kept.sum()) * p64


def speculate(p: torch.Tensor, q: torch.Tensor, n: int, gen: torch.Generator, rule: str = EXACT, lenience: float = 1.0,
              threshold: float = 0.05, arith: torch.dtype = torch.float32, chunk: int = 1 << 22) -> Outcome:
    """Run the rule n times: guess from q, test, replace, count. p and q are float32 probability vectors on one device,
    each summing to 1."""
    check_mass(p, q)
    V = p.numel()
    pa, qa = p.to(arith), q.to(arith)
    cdf_q = cdf64(q)
    cdf_r = cdf64(replacement(rule, pa, qa, lenience).double())
    fixed = keep_test(rule, pa, qa, threshold) if rule in WRONG else None
    hist = torch.zeros(V, dtype=torch.int64, device=p.device)
    accepted = 0
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        x = draw(cdf_q, m, gen)
        u = torch.rand(m, dtype=torch.float64, device=p.device, generator=gen)   # vLLM draws u in float64
        if fixed is not None:
            keep = fixed[x]
        else:
            px, qx = pa[x], qa[x]
            if rule == LENIENT:
                qx = qx * lenience
            keep = (qx > 0) & (px / qx >= u)   # vLLM: draft_prob > 0 and target_prob / draft_prob >= uniform_prob
        y = draw(cdf_r, m, gen)
        out = torch.where(keep, x, y)
        hist += torch.bincount(out, minlength=V)
        accepted += int(keep.sum().item())
    return Outcome(hist, n, accepted)


def exact_draws(p: torch.Tensor, n: int, gen: torch.Generator, chunk: int = 1 << 22) -> torch.Tensor:
    """n draws from p itself: the histogram an exact rule should be indistinguishable from (the noise floor)."""
    cdf = cdf64(p)
    hist = torch.zeros(p.numel(), dtype=torch.int64, device=p.device)
    for start in range(0, n, chunk):
        hist += torch.bincount(draw(cdf, min(chunk, n - start), gen), minlength=p.numel())
    return hist


def arithmetic_disagreements(p: torch.Tensor, q: torch.Tensor, n: int, gen: torch.Generator, chunk: int = 1 << 22) -> dict:
    """The exact rule in float32 against float64 on the same guesses and the same uniforms: how many of n decisions
    (keep or not) and how many of n outputs differ. This is the rule's own share of the 'hardware numerics' that
    Chen et al. name as the one caveat (Accelerating Large Language Model Decoding with Speculative Sampling,
    https://arxiv.org/abs/2302.01318)."""
    check_mass(p, q)
    cdf_q = cdf64(q)
    r32, r64 = torch.clamp(p - q, min=0).double(), torch.clamp(p.double() - q.double(), min=0)
    c32, c64 = cdf64(r32), cdf64(r64)
    keep_diff = out_diff = 0
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        x = draw(cdf_q, m, gen)
        u = torch.rand(m, dtype=torch.float64, device=p.device, generator=gen)
        k32 = (q[x] > 0) & (p[x] / q[x] >= u)
        k64 = (q.double()[x] > 0) & (p.double()[x] / q.double()[x] >= u)
        v = torch.rand(m, dtype=torch.float64, device=p.device, generator=gen)
        y32 = torch.searchsorted(c32, v).clamp_(max=p.numel() - 1)
        y64 = torch.searchsorted(c64, v).clamp_(max=p.numel() - 1)
        keep_diff += int((k32 != k64).sum().item())
        out_diff += int((torch.where(k32, x, y32) != torch.where(k64, x, y64)).sum().item())
    return dict(n=n, keep_differ=keep_diff, output_differ=out_diff)


def total_variation(hist: torch.Tensor, target: torch.Tensor, n: int) -> float:
    return 0.5 * (hist.double() / n - target.double()).abs().sum().item()


def distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Total variation between two probability vectors, each checked to sum to 1."""
    check_mass(a, b)
    return 0.5 * (a.double() - b.double()).abs().sum().item()


def chi2_per_dof(hist: torch.Tensor, target: torch.Tensor, n: int, min_expected: float = 25.0) -> tuple[float, int]:
    """Pearson's chi-square over the tokens expected at least min_expected times, the rest pooled, per degree of
    freedom. Near 1 when the histogram is a sample of the target."""
    e = n * target.double()
    sel = e >= min_expected
    o = hist.double()
    chi = ((o[sel] - e[sel]) ** 2 / e[sel]).sum()
    e_rest, o_rest = n - e[sel].sum(), n - o[sel].sum()
    bins = int(sel.sum().item())
    if e_rest > min_expected:
        chi = chi + (o_rest - e_rest) ** 2 / e_rest
        bins += 1
    return (chi / max(bins - 1, 1)).item(), bins - 1


def max_gain(hist: torch.Tensor, target: torch.Tensor, n: int, min_expected: float = 100.0) -> float:
    """The largest ratio of measured to target probability over tokens the run expects often enough to read it;
    Leviathan's lenient rule promises it stays under 1 / lenience (Appendix A.5)."""
    e = n * target.double()
    sel = e >= min_expected
    return ((hist.double()[sel] / n) / target.double()[sel]).max().item()


def drafters(logits: torch.Tensor, p: torch.Tensor, gen: torch.Generator) -> dict[str, torch.Tensor]:
    """Draft distributions of every quality, built from the target's own logits (no second model): the target itself,
    flatter and sharper versions (temperature), the target with its top token removed, a one-hot guess on the top
    token (what an n-gram or greedy drafter hands vLLM: its kernel sets the draft probability to 1), a one-hot on a
    random token, uniform."""
    V = p.numel()
    top = int(p.argmax().item())
    rnd = int(torch.randint(V, (1,), device=p.device, generator=gen).item())
    one_hot = lambda i: torch.zeros(V, device=p.device).index_fill_(0, torch.tensor([i], device=p.device), 1.0)  # noqa: E731
    no_top = p.clone()
    no_top[top] = 0
    return {
        "the target itself": p,
        "flatter: temperature 2": torch.softmax(logits / 2, -1),
        "flatter: temperature 4": torch.softmax(logits / 4, -1),
        "sharper: temperature 0.5": torch.softmax(logits / 0.5, -1),
        "the target with its top token removed": no_top / no_top.sum(),
        "one-hot on the target's top token": one_hot(top),
        "one-hot on a random token": one_hot(rnd),
        "uniform over the vocabulary": torch.full((V,), 1.0 / V, device=p.device),
    }
