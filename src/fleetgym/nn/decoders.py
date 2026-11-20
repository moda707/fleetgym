"""Built-in decoders (design §20): the pointer decoder of the Attention Model.

Node actions get ``C * tanh(q . k_i / sqrt(D))`` (Kool et al. 2019, ``C = 10``) with the query from
the context; extra actions beyond the node slots (wait, no-op) get a logit from the context. Masked
actions get ``-inf``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from fleetgym.nn.adapters import GraphBatch
from fleetgym.nn.base import Decoder, Embeddings
from fleetgym.nn.registry import register

__all__ = ["PointerDecoder"]


@register("pointer")
class PointerDecoder(Decoder):
    """Pointer attention over node embeddings, plus logits for extra actions.

    Parameters
    ----------
    embed_dim
        Embedding size.
    n_extra
        Actions after the node slots (for example 1 for the wait action).
    clip
        Logit clipping constant ``C``.
    """

    def __init__(self, embed_dim: int, n_extra: int = 0, clip: float = 10.0) -> None:
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.extra = nn.Linear(embed_dim, n_extra) if n_extra else None
        self.clip = clip
        self.scale = 1.0 / math.sqrt(embed_dim)

    def logits(self, emb: Embeddings) -> torch.Tensor:
        """Unmasked logits ``(B, N + n_extra)``."""
        q = self.query(emb.context).unsqueeze(1)
        k = self.key(emb.node)
        node = self.clip * torch.tanh((q * k).sum(-1) * self.scale)
        if self.extra is None:
            return node
        return torch.cat([node, self.clip * torch.tanh(self.extra(emb.context))], dim=-1)

    def forward(self, emb: Embeddings, g: GraphBatch) -> torch.Tensor:
        """Masked logits."""
        logits = self.logits(emb)
        return logits.masked_fill(~g.action_mask, float("-inf"))
