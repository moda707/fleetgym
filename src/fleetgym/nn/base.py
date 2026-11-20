"""Abstract module hierarchy (design §20): encoders, decoders, embeddings.

Subclass `Encoder` (``GraphBatch -> Embeddings``) or `Decoder`
(``Embeddings, GraphBatch -> logits``), register it with `fleetgym.nn.register`, and check it with
`fleetgym.testing.check_module`.
Decoders must return ``-inf`` for masked actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from fleetgym.nn.adapters import GraphBatch

__all__ = ["Decoder", "Embeddings", "Encoder", "FleetGymModule", "masked_mean"]


@dataclass
class Embeddings:
    """Encoder output: ``node`` ``(B, N, D)`` and ``context`` ``(B, D)``."""

    node: torch.Tensor
    context: torch.Tensor


def masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` ``(B, N, D)`` over valid nodes (``(B, N)`` booleans)."""
    weights = valid.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class FleetGymModule(nn.Module, ABC):
    """Base of FleetGym modules.

    ``equivariant`` declares whether node outputs are permutation equivariant over customers, and
    ``fixed_size`` whether the module only accepts its configured number of node slots (then
    `check_module` skips the padding check).
    """

    equivariant: ClassVar[bool] = True
    fixed_size: ClassVar[bool] = False


class Encoder(FleetGymModule):
    """``GraphBatch -> Embeddings``.

    Parameters
    ----------
    node_features, vehicle_features, global_features
        Input sizes.
    embed_dim
        Embedding size ``D``.
    """

    def __init__(
        self, node_features: int, vehicle_features: int, global_features: int, embed_dim: int
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.node_in = nn.Linear(node_features, embed_dim)
        self.context_in = nn.Linear(embed_dim + vehicle_features + global_features, embed_dim)

    def embed_nodes(self, g: GraphBatch) -> torch.Tensor:
        """Initial node embeddings (zero at invalid nodes)."""
        return self.node_in(g.nodes) * g.node_valid.unsqueeze(-1)

    def pack(self, h: torch.Tensor, g: GraphBatch) -> Embeddings:
        """Embeddings with a context from the node mean, the vehicle and global features."""
        h = h * g.node_valid.unsqueeze(-1)
        pooled = masked_mean(h, g.node_valid)
        context = torch.tanh(self.context_in(torch.cat([pooled, g.vehicle, g.globals], dim=-1)))
        return Embeddings(node=h, context=context)

    @abstractmethod
    def forward(self, g: GraphBatch) -> Embeddings:
        """Encode the batch."""


class Decoder(FleetGymModule):
    """``Embeddings, GraphBatch -> logits (B, A)``, ``-inf`` at masked actions."""

    @abstractmethod
    def forward(self, emb: Embeddings, g: GraphBatch) -> torch.Tensor:
        """Action logits."""
