"""Check that what the agent sees never depends on unrevealed data (design §22).

At every decision epoch of an episode, `check_non_anticipativity` forks the environment
(`FleetGymEnv.fork`), which keeps the state and every revealed value but redraws everything still
hidden. The observation, the action mask and the agent-facing ``info`` fields must be identical in
the original and in each fork; if they are not, some unrevealed value leaked into them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fleetgym.envs.env import FleetGymEnv
from fleetgym.errors import FleetGymError

__all__ = ["AnticipationError", "check_non_anticipativity"]

_INFO_KEYS = ("vehicle", "epoch_kind", "time", "action_mask")


class AnticipationError(FleetGymError):
    """An observation, mask or info field depends on information not yet revealed."""


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    x, y = np.asarray(a), np.asarray(b)
    if x.dtype.kind == "f" and y.dtype.kind == "f":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def check_non_anticipativity(
    env: FleetGymEnv, *, seed: int = 0, n_forks: int = 3, max_steps: int = 10_000
) -> int:
    """Play a random masked episode and compare it with forks at every epoch.

    Returns
    -------
    int
        The number of epochs checked.

    Raises
    ------
    AnticipationError
        At the first epoch where a fork sees something different.
    """
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    for step in range(max_steps):
        for k in range(n_forks):
            fork = env.fork(seed=1_000_003 * (step + 1) + k)
            fork_obs = fork._observation()  # pyright: ignore[reportPrivateUsage]
            fork_info = fork._info({})  # pyright: ignore[reportPrivateUsage]
            if not _same(obs, fork_obs):
                raise AnticipationError(f"the observation at epoch {step} depends on hidden data")
            for key in _INFO_KEYS:
                if not _same(info[key], fork_info[key]):
                    raise AnticipationError(f"info[{key!r}] at epoch {step} depends on hidden data")
        action = int(rng.choice(np.flatnonzero(env.action_masks())))
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return step + 1
    raise RuntimeError(f"episode did not end within {max_steps} steps")
