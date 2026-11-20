"""SB3 adapters (design §19).

Vectorized environments, default hyperparameters, policy settings, and an adapter that lets a
trained model run through the evaluation harness.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from fleetgym.baselines.protocol import PolicyBase
from fleetgym.envs.factory import make
from fleetgym.spec import ProblemSpec

__all__ = ["ALGORITHMS", "POLICIES", "SB3Policy", "make_sb3_vec", "policy_settings", "sb3_defaults"]

ALGORITHMS = ("maskable_ppo", "maskable_dqn")
POLICIES = ("gat", "attention", "mlp")


def make_sb3_vec(
    env: str | Path | ProblemSpec | Mapping[str, Any],
    n_envs: int = 1,
    *,
    seed: int = 0,
    subprocess: bool = False,
    **env_kwargs: Any,
) -> VecEnv:
    """``n_envs`` monitored copies of an environment in an SB3 vector env (copy ``i``: seed + i)."""
    if n_envs < 1:
        raise ValueError(f"n_envs must be positive, got {n_envs}")

    def factory(rank: int) -> Any:
        def build() -> Monitor:
            instance = make(env, **env_kwargs)
            instance.reset(seed=seed + rank)
            return Monitor(instance)

        return build

    factories = [factory(i) for i in range(n_envs)]
    vec = SubprocVecEnv(factories) if subprocess else DummyVecEnv(factories)
    vec.seed(seed)
    return vec


def sb3_defaults(env_id: str, algo: Literal["maskable_ppo", "maskable_dqn"]) -> dict[str, Any]:
    """Default hyperparameters of ``algo`` for ``env_id`` (the ``default`` entry if none)."""
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algo!r}; known: {', '.join(ALGORITHMS)}")
    path = resources.files("fleetgym.agents") / "hyperparams" / f"{algo}.yaml"
    table = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(table.get(env_id, table["default"]))


def policy_settings(
    algo: Literal["maskable_ppo", "maskable_dqn"], policy: str, *, embed_dim: int = 64
) -> tuple[Any, dict[str, Any]]:
    """The SB3 policy class and ``policy_kwargs`` for ``algo`` and a policy name."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")
    if algo == "maskable_ppo":
        from fleetgym.nn.policies import MaskableGraphPolicy

        extractor = {"encoder": policy, "embed_dim": embed_dim}
        return MaskableGraphPolicy, {"features_extractor_kwargs": extractor}
    from fleetgym.agents.maskable_dqn import MaskableDQNPolicy

    return MaskableDQNPolicy, {"net_arch": [128, 128]}


class SB3Policy(PolicyBase):
    """A trained SB3 model as a `Policy` (deterministic, masked actions)."""

    name = "sb3"

    def __init__(self, model: BaseAlgorithm, *, name: str | None = None) -> None:
        super().__init__()
        self.model = model
        if name:
            self.name = name

    def act(self, obs: dict[str, NDArray[Any]], info: dict[str, Any]) -> int:
        """The model's deterministic action under the current mask."""
        kwargs: dict[str, Any] = {"deterministic": True}
        if hasattr(self.model.policy, "_get_action_dist_from_latent"):
            kwargs["action_masks"] = np.asarray(info["action_mask"])
        action, _ = self.model.predict(obs, **kwargs)
        return int(action)
