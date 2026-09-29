"""Chip numbers from vendor spec sheets, with the derivation and the source id for each.

What: peak dense compute, bandwidth and capacity of the memory that holds the weights.
Why: the ridge point (peak FLOPs / bandwidth) is the whole argument of this package.
Where: consumed by roofline.py; quotes live in ../sources.json under the given ids.

GB and TB are decimal here (1e9, 1e12), matching how the vendors print them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chip:
    name: str
    peak_flops: float      # dense, at `precision`
    precision: str
    mem_bw: float          # bytes/s of the memory that holds the weights
    mem_bytes: float       # capacity of that memory
    memory: str
    source: str
    derivation: str

    @property
    def ridge(self) -> float:
        """FLOPs per byte at which the chip stops being memory-bound."""
        return self.peak_flops / self.mem_bw


# NVIDIA DGX B200 page: "1,440 GB total, 64 TB/s HBM3e bandwidth" for 8 GPUs -> 180 GB, 8 TB/s
# per GPU. HGX page: "FP16/BF16 Tensor Core: 36 PFLOPS" and "FP8/FP6 Tensor Core: 72 PFLOPS"
# for 8 GPUs, listed with sparsity (the page's "Sparse | Dense" footnote); dense is half,
# so per GPU: 2.25 PFLOPS BF16, 4.5 PFLOPS FP8.
B200_BF16 = Chip(
    name="NVIDIA B200", peak_flops=2.25e15, precision="BF16 dense", mem_bw=8.0e12,
    mem_bytes=180e9, memory="HBM3e", source="nvidia-dgx-b200, nvidia-hgx",
    derivation="64 TB/s / 8 GPUs; 36 PFLOPS sparse / 2 / 8 GPUs",
)
B200_FP8 = Chip(
    name="NVIDIA B200", peak_flops=4.5e15, precision="FP8 dense", mem_bw=8.0e12,
    mem_bytes=180e9, memory="HBM3e", source="nvidia-dgx-b200, nvidia-hgx",
    derivation="64 TB/s / 8 GPUs; 72 PFLOPS sparse / 2 / 8 GPUs",
)

# Cerebras WSE-3 press release: "44GB on-chip SRAM", "125 petaflops of peak AI performance"
# (precision not stated). Cerebras inference blog: "21 petabytes/s of aggregate memory
# bandwidth".
WSE3 = Chip(
    name="Cerebras WSE-3", peak_flops=125e15, precision="peak AI performance as listed",
    mem_bw=21e15, mem_bytes=44e9, memory="on-chip SRAM",
    source="cerebras-wse3-pr, cerebras-inference-blog", derivation="as listed",
)

CHIPS = {"B200 BF16": B200_BF16, "B200 FP8": B200_FP8, "WSE-3": WSE3}
