"""SB3 policies assembled from FleetGym modules (design §19, §20; plan Q31).

`MaskableGraphPolicy` is a `sb3_contrib` ``MaskableActorCriticPolicy`` whose parts are FleetGym
modules placed in SB3's own slots, so SB3's rollout, masking, saving and loading code is unchanged:

* the features extractor (`GraphFeaturesExtractor`) runs an `Encoder` and returns the node
  embeddings and the context, flattened (``N * D + D`` features);
* the action network (`PointerHead`) applies a `PointerDecoder` to them (per-node logits keep the
  pointer structure; extra actions such as wait get logits from the context);
* the value network (`ValueHead`) reads the context.

Masking is applied by SB3 from the environment's ``action_masks()``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from fleetgym.nn.adapters import to_graph_batch
from fleetgym.nn.base import Embeddings
from fleetgym.nn.decoders import PointerDecoder
from fleetgym.nn.registry import get_module

__all__ = ["GraphFeaturesExtractor", "MaskableGraphPolicy", "PointerHead", "ValueHead"]


def _sizes(space: gym.spaces.Dict) -> tuple[int, int, int, int, int]:
    nodes = space["nodes"].shape
    assert nodes is not None
    vehicle = space["vehicle"].shape
    globals_ = space["globals"].shape
    mask = space["action_mask"].shape
    assert vehicle is not None
    assert globals_ is not None
    assert mask is not None
    return int(nodes[0]), int(nodes[1]), int(vehicle[0]), int(globals_[0]), int(mask[0])


class GraphFeaturesExtractor(BaseFeaturesExtractor):
    """Run a registered encoder; output the flattened node embeddings and the context.

    Parameters
    ----------
    observation_space
        A FleetGym observation space.
    encoder
        Registered encoder name (``"gat"``, ``"attention"``, ``"mlp"``) or class.
    embed_dim
        Embedding size.
    **encoder_kwargs
        Passed to the encoder.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        encoder: str | type = "gat",
        embed_dim: int = 64,
        **encoder_kwargs: Any,
    ) -> None:
        max_nodes, node_f, vehicle_f, global_f, _ = _sizes(observation_space)
        super().__init__(observation_space, features_dim=max_nodes * embed_dim + embed_dim)
        cls = get_module(encoder) if isinstance(encoder, str) else encoder
        self.max_nodes, self.embed_dim = max_nodes, embed_dim
        self.encoder = cls(
            node_f, vehicle_f, global_f, embed_dim, max_nodes=max_nodes, **encoder_kwargs
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flattened ``[node embeddings, context]``."""
        emb = self.encoder(to_graph_batch(observations))
        return torch.cat([emb.node.flatten(1), emb.context], dim=-1)


def _unpack(latent: torch.Tensor, max_nodes: int, embed_dim: int) -> Embeddings:
    node = latent[:, : max_nodes * embed_dim].reshape(-1, max_nodes, embed_dim)
    return Embeddings(node=node, context=latent[:, max_nodes * embed_dim :])


class PointerHead(nn.Module):
    """SB3 action network: a `PointerDecoder` on the unpacked features (unmasked logits)."""

    def __init__(self, max_nodes: int, embed_dim: int, n_actions: int) -> None:
        super().__init__()
        self.max_nodes, self.embed_dim = max_nodes, embed_dim
        self.decoder = PointerDecoder(embed_dim, n_extra=n_actions - max_nodes)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Logits ``(B, n_actions)``; SB3 applies the mask."""
        return self.decoder.logits(_unpack(latent, self.max_nodes, self.embed_dim))


class ValueHead(nn.Module):
    """SB3 value network on the context."""

    def __init__(self, max_nodes: int, embed_dim: int) -> None:
        super().__init__()
        self.max_nodes, self.embed_dim = max_nodes, embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Value ``(B, 1)``."""
        return self.net(latent[:, self.max_nodes * self.embed_dim :])


class MaskableGraphPolicy(MaskableActorCriticPolicy):
    """Masked actor-critic with a graph encoder and a pointer decoder.

    Use ``policy_kwargs=dict(features_extractor_kwargs=dict(encoder="gat", embed_dim=64))``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("features_extractor_class", GraphFeaturesExtractor)
        kwargs.setdefault("net_arch", [])
        kwargs.setdefault("ortho_init", False)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Any) -> None:
        super()._build(lr_schedule)
        extractor = self.features_extractor
        assert isinstance(extractor, GraphFeaturesExtractor)
        space = self.action_space
        assert isinstance(space, gym.spaces.Discrete)
        self.action_net = PointerHead(extractor.max_nodes, extractor.embed_dim, int(space.n))
        self.value_net = ValueHead(extractor.max_nodes, extractor.embed_dim)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),  # pyright: ignore[reportCallIssue]
            **self.optimizer_kwargs,
        )
