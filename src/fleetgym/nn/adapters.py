"""Observation adapter: padded observation dictionaries to a dense `GraphBatch`.

Node features are normalized per sample: each feature is divided by ``1 +`` its largest absolute
value over the sample's valid nodes, which keeps the scale of coordinates, times and demands similar
across instances, and is permutation equivariant and padding invariant.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

__all__ = ["GraphBatch", "to_graph_batch"]


@dataclass
class GraphBatch:
    """A batch of padded graphs.

    Attributes
    ----------
    nodes
        ``(B, N, F)`` node features.
    node_valid
        ``(B, N)`` booleans: real (and released) nodes.
    vehicle
        ``(B, Fv)`` features of the deciding vehicle.
    globals
        ``(B, G)`` global features.
    action_mask
        ``(B, A)`` booleans: valid actions.
    """

    nodes: torch.Tensor
    node_valid: torch.Tensor
    vehicle: torch.Tensor
    globals: torch.Tensor
    action_mask: torch.Tensor

    @property
    def n_nodes(self) -> int:
        """Node slots ``N``."""
        return int(self.nodes.shape[1])


def _normalize(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.unsqueeze(-1).to(x.dtype)
    scale = 1.0 + (x.abs() * weights).amax(dim=1, keepdim=True)
    return x / scale * weights


def to_graph_batch(obs: Mapping[str, torch.Tensor], *, normalize: bool = True) -> GraphBatch:
    """Build a `GraphBatch` from an observation dictionary of tensors (batched or not)."""
    nodes = obs["nodes"].float()
    if nodes.dim() == 2:
        obs = {k: v.unsqueeze(0) for k, v in obs.items()}
        nodes = obs["nodes"].float()
    valid = obs["node_valid"].reshape(nodes.shape[:2]) > 0.5
    vehicle = obs["vehicle"].float()
    globals_ = obs["globals"].float()
    if normalize:
        nodes = _normalize(nodes, valid)
        vehicle = vehicle / (1.0 + vehicle.abs())
        globals_ = globals_ / (1.0 + globals_.abs())
    return GraphBatch(
        nodes=nodes,
        node_valid=valid,
        vehicle=vehicle,
        globals=globals_,
        action_mask=obs["action_mask"].reshape(nodes.shape[0], -1) > 0.5,
    )
