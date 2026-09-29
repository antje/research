"""The roofline arithmetic for one decode step.

What: bytes and FLOPs of a step at a given batch and context; the chip's bound on it;
the intensity limit as batch grows; the batch that reaches the ridge; what fits on chip.
Why: these five functions are the whole mechanism the post explains.
Where: called by __main__.py (tables) and plot.py (diagram); tested in tests/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .chips import Chip
from .model import ModelShape


@dataclass(frozen=True)
class StepCost:
    batch: int
    context: int
    w_bytes: int
    kv_bytes: int
    bytes_weights: float
    bytes_kv: float
    flops: float

    @property
    def bytes_total(self) -> float:
        return self.bytes_weights + self.bytes_kv

    @property
    def intensity(self) -> float:
        """FLOPs per byte moved from the weight/KV memory."""
        return self.flops / self.bytes_total


def decode_step(model: ModelShape, batch: int, context: int, w_bytes: int = 2, kv_bytes: int = 2) -> StepCost:
    """Cost of generating one token for each of `batch` sequences with `context` cached tokens.

    Weights are read once per step regardless of batch; KV is read once per sequence.
    """
    return StepCost(
        batch=batch, context=context, w_bytes=w_bytes, kv_bytes=kv_bytes,
        bytes_weights=model.params() * w_bytes,
        bytes_kv=batch * context * model.kv_bytes_per_token(kv_bytes),
        flops=batch * model.decode_flops_per_token(context),
    )


@dataclass(frozen=True)
class Bound:
    chip: Chip
    cost: StepCost
    t_mem: float
    t_comp: float

    @property
    def limiter(self) -> str:
        return "memory" if self.t_mem >= self.t_comp else "compute"

    @property
    def step_time(self) -> float:
        return max(self.t_mem, self.t_comp)

    @property
    def tok_s(self) -> float:
        """Upper bound on tokens per second for the whole batch (spec sheet, not measured)."""
        return self.cost.batch / self.step_time

    @property
    def compute_util(self) -> float:
        """Fraction of peak compute the step can use at best."""
        return self.t_comp / self.step_time


def bound(chip: Chip, cost: StepCost) -> Bound:
    return Bound(chip=chip, cost=cost, t_mem=cost.bytes_total / chip.mem_bw, t_comp=cost.flops / chip.peak_flops)


def intensity_limit(model: ModelShape, context: int, kv_bytes: int = 2) -> float:
    """Intensity as batch grows without bound: weights amortize away, KV reads do not."""
    return model.decode_flops_per_token(context) / (context * model.kv_bytes_per_token(kv_bytes))


def batch_at_ridge(model: ModelShape, chip: Chip, context: int, w_bytes: int = 2, kv_bytes: int = 2) -> int | None:
    """Smallest batch whose intensity reaches the chip's ridge, or None if no batch does.

    intensity(B) = B*f / (W + B*K) >= R  <=>  B >= R*W / (f - R*K), when f > R*K.
    """
    f = model.decode_flops_per_token(context)
    w = model.params() * w_bytes
    k = context * model.kv_bytes_per_token(kv_bytes)
    r = chip.ridge
    if f <= r * k:
        return None
    return max(1, math.ceil(r * w / (f - r * k)))


def fit(model: ModelShape, chip: Chip, w_bytes: int = 2, kv_bytes: int = 2) -> dict:
    """Do the weights fit in the chip's weight memory, and how many KV tokens fit beside them?"""
    weights = model.params() * w_bytes
    free = chip.mem_bytes - weights
    return {
        "weights_bytes": weights,
        "fits": weights <= chip.mem_bytes,
        "free_bytes": max(0.0, free),
        "kv_tokens_that_fit": int(max(0.0, free) // model.kv_bytes_per_token(kv_bytes)),
    }
