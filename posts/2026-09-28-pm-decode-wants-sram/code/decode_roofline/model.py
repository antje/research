"""Model shapes from a Hugging Face config.json, and what a decode step costs per token.

What: parameter count, KV-cache bytes per token, FLOPs per generated token.
Why: a roofline needs bytes moved and FLOPs done per step; both follow from the config.
Where: consumed by roofline.py. Config values carry their source id (see ../sources.json).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    intermediate: int
    vocab: int
    tied_embeddings: bool
    source: str

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def params(self) -> int:
        """Weight parameters implied by the config.

        Embedding and output projection (untied: both), per layer q/o projections
        (hidden x hidden each), k/v projections (hidden x kv_heads*head_dim each), and the
        gated MLP (three hidden x intermediate matrices). Norm weights and biases are left
        out; they are under 0.01% of the total.
        """
        embed = self.vocab * self.hidden * (1 if self.tied_embeddings else 2)
        attn = 2 * self.hidden * self.hidden + 2 * self.hidden * self.kv_heads * self.head_dim
        mlp = 3 * self.hidden * self.intermediate
        return embed + self.layers * (attn + mlp)

    def kv_bytes_per_token(self, kv_bytes: int = 2) -> int:
        """Bytes of KV cache one token occupies: K and V, every layer, every KV head."""
        return 2 * self.layers * self.kv_heads * self.head_dim * kv_bytes

    def decode_flops_per_token(self, context: int) -> float:
        """FLOPs to generate one token with `context` tokens already cached.

        Two FLOPs per weight for the matmuls, plus attention over the context: QK^T and PV
        each cost 2*context*hidden per layer (all query heads together span `hidden`).
        """
        return 2.0 * self.params() + 4.0 * self.layers * self.hidden * context


# Qwen2.5-72B-Instruct config.json (source id qwen72b-config); 72.7B on the model card.
QWEN2_5_72B = ModelShape(
    name="Qwen2.5-72B-Instruct", hidden=8192, layers=80, heads=64, kv_heads=8,
    intermediate=29568, vocab=152064, tied_embeddings=False, source="qwen72b-config",
)

# Qwen2.5-7B-Instruct config.json (source id qwen7b-config); 7.61B on the model card.
QWEN2_5_7B = ModelShape(
    name="Qwen2.5-7B-Instruct", hidden=3584, layers=28, heads=28, kv_heads=4,
    intermediate=18944, vocab=152064, tied_embeddings=False, source="qwen7b-config",
)

MODELS = {m.name: m for m in (QWEN2_5_72B, QWEN2_5_7B)}
