"""Built-in encoders (design §20): MLP, dense graph attention (GAT), Attention Model.

`GATEncoder` is a dense, mask-aware pure-torch implementation (plan Q31): every valid node attends
to every valid node, so no edge list (and no torch-geometric) is needed.
"""

from __future__ import annotations

import torch
from torch import nn

from fleetgym.nn.adapters import GraphBatch
from fleetgym.nn.base import Embeddings, Encoder
from fleetgym.nn.registry import register

__all__ = ["AttentionModelEncoder", "GATEncoder", "MLPEncoder"]


@register("mlp")
class MLPEncoder(Encoder):
    """Flat baseline: an MLP on the flattened nodes gives the context; nodes are embedded alone.

    Not permutation equivariant in its context, and tied to ``max_nodes`` slots.
    """

    equivariant = False
    fixed_size = True

    def __init__(
        self,
        node_features: int,
        vehicle_features: int,
        global_features: int,
        embed_dim: int = 64,
        *,
        max_nodes: int,
        hidden: int = 128,
    ) -> None:
        super().__init__(node_features, vehicle_features, global_features, embed_dim)
        self.flat = nn.Sequential(
            nn.Linear(max_nodes * node_features + vehicle_features + global_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        self.node = nn.Sequential(nn.Linear(node_features, embed_dim), nn.ReLU())

    def forward(self, g: GraphBatch) -> Embeddings:
        """Context from the flattened input; node embeddings from each node's features."""
        flat = (g.nodes * g.node_valid.unsqueeze(-1)).flatten(1)
        context = torch.tanh(self.flat(torch.cat([flat, g.vehicle, g.globals], dim=-1)))
        h = self.node(g.nodes) * g.node_valid.unsqueeze(-1)
        return Embeddings(node=h, context=context)


class _DenseGATLayer(nn.Module):
    """Multi-head graph attention over valid nodes (Veličković et al. 2018), dense form."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"embed_dim {dim} must be divisible by heads {heads}")
        self.heads, self.head_dim = heads, dim // heads
        self.w = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, n, d = h.shape
        z = self.w(h).view(b, n, self.heads, self.head_dim)
        src = (z * self.a_src).sum(-1)  # (B, N, H)
        dst = (z * self.a_dst).sum(-1)
        scores = nn.functional.leaky_relu(src.unsqueeze(2) + dst.unsqueeze(1), 0.2)  # (B,N,N,H)
        allowed = valid.unsqueeze(1).unsqueeze(-1)  # keys
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=2).nan_to_num(0.0)
        out = torch.einsum("bijh,bjhd->bihd", weights, z).reshape(b, n, d)
        return self.norm(h + torch.relu(out)) * valid.unsqueeze(-1)


@register("gat")
class GATEncoder(Encoder):
    """Stacked dense graph attention layers with residual connections and layer norm."""

    def __init__(
        self,
        node_features: int,
        vehicle_features: int,
        global_features: int,
        embed_dim: int = 64,
        *,
        n_layers: int = 2,
        heads: int = 4,
        **_: object,
    ) -> None:
        super().__init__(node_features, vehicle_features, global_features, embed_dim)
        self.layers = nn.ModuleList(_DenseGATLayer(embed_dim, heads) for _ in range(n_layers))

    def forward(self, g: GraphBatch) -> Embeddings:
        """Attention over the valid nodes, then pooling into a context."""
        h = self.embed_nodes(g)
        for layer in self.layers:
            h = layer(h, g.node_valid)
        return self.pack(h, g)


@register("attention")
class AttentionModelEncoder(Encoder):
    """Transformer encoder of the Attention Model (Kool et al. 2019), with padding masks."""

    def __init__(
        self,
        node_features: int,
        vehicle_features: int,
        global_features: int,
        embed_dim: int = 64,
        *,
        n_layers: int = 2,
        heads: int = 4,
        ff_dim: int = 128,
        **_: object,
    ) -> None:
        super().__init__(node_features, vehicle_features, global_features, embed_dim)
        layer = nn.TransformerEncoderLayer(
            embed_dim, heads, ff_dim, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, g: GraphBatch) -> Embeddings:
        """Self-attention among valid nodes."""
        h = self.embed_nodes(g)
        # A sample without valid nodes would make attention undefined; keep slot 0 visible then.
        padding = ~g.node_valid
        padding[:, 0] &= g.node_valid.any(dim=1)
        h = self.encoder(h, src_key_padding_mask=padding).nan_to_num(0.0)
        return self.pack(h, g)
