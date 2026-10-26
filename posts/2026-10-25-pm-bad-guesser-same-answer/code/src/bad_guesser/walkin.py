"""The post's A, B, C example in plain Python: a model with three tokens, a guesser who gets the order backwards.

For any target p and draft q, the checker keeps a guess x with probability min(1, p(x) / q(x)) and otherwise draws a
replacement from the normalized max(0, p - q). The chance of ending on token x is then
min(p(x), q(x)) + (p(x) - min(p(x), q(x))) = p(x), and the chance a guess survives is the overlap, the sum of
min(p, q). Leviathan, Kalman and Matias, Fast Inference from Transformers via Speculative Decoding, § 2.3 (the rule),
Appendix A.1 (the sum), Theorem 3.5 (the overlap): https://arxiv.org/abs/2211.17192. These closed forms are what the
GPU measurements are checked against.
"""

from __future__ import annotations

MODEL = [0.50, 0.30, 0.20]   # the model's chances of A, B, C
GUESSER = [0.20, 0.30, 0.50]  # the guesser's chances, the order reversed


def overlap(p: list[float], q: list[float]) -> float:
    """The chance a guess is kept: the sum over tokens of min(p, q)."""
    return sum(min(a, b) for a, b in zip(p, q))


def residual(p: list[float], q: list[float], lenience: float = 1.0) -> list[float]:
    """Where a thrown-out guess is replaced from: max(0, p - lenience * q), normalized."""
    r = [max(0.0, a - lenience * b) for a, b in zip(p, q)]
    s = sum(r)
    return [x / s for x in r] if s > 0 else list(p)


def output(p: list[float], q: list[float], lenience: float = 1.0) -> list[float]:
    """The distribution the exact rule (or Leviathan's lenient variant) produces, in closed form."""
    kept = [min(b, a / lenience) for a, b in zip(p, q)]   # q(x) * min(1, p(x) / (l q(x)))
    r = residual(p, q, lenience)
    rejected = 1.0 - sum(kept)
    return [k + rejected * x for k, x in zip(kept, r)]


def output_wrong(p: list[float], q: list[float], keep: list[bool]) -> list[float]:
    """A rule that keeps a guess by a fixed test (keep[x]) and otherwise samples the target afresh."""
    kept = [b if k else 0.0 for b, k in zip(q, keep)]
    rejected = 1.0 - sum(kept)
    return [k + rejected * a for k, a in zip(kept, p)]


def total_variation(a: list[float], b: list[float]) -> float:
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))
