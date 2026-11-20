"""Random feasible policy: a uniform choice among the valid actions (a sanity check)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fleetgym.baselines.protocol import PolicyBase, valid_actions
from fleetgym.envs.env import FleetGymEnv

__all__ = ["RandomPolicy"]


class RandomPolicy(PolicyBase):
    """Uniform over the valid actions, from its own generator.

    Parameters
    ----------
    seed
        Seed of the policy's generator; `reset` restarts it, so episodes are reproducible.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, env: FleetGymEnv) -> None:
        """Bind to ``env`` and restart the generator."""
        super().reset(env)
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs: dict[str, NDArray[Any]], info: dict[str, Any]) -> int:
        """A uniformly random valid action."""
        return int(self.rng.choice(valid_actions(info)))
