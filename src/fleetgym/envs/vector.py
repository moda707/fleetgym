"""`make_vec` for vectorized environments."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym

from fleetgym.envs.factory import make
from fleetgym.spec import ProblemSpec

__all__ = ["make_vec"]


def make_vec(
    env: str | Path | ProblemSpec,
    n_envs: int,
    *,
    mode: Literal["sync", "async"] = "sync",
    overrides: Sequence[str] = (),
    **env_kwargs: Any,
) -> gym.vector.VectorEnv:
    """Create ``n_envs`` copies of an environment in a Gymnasium vector env.

    Seed the vector env with ``reset(seed=s)``: copy ``i`` then uses seed ``s + i``.
    """
    if n_envs < 1:
        raise ValueError(f"n_envs must be positive, got {n_envs}")
    factories = [partial(make, env, overrides=overrides, **env_kwargs) for _ in range(n_envs)]
    if mode == "async":
        return gym.vector.AsyncVectorEnv(factories)
    if mode == "sync":
        return gym.vector.SyncVectorEnv(factories)
    raise ValueError(f"mode must be 'sync' or 'async', got {mode!r}")
