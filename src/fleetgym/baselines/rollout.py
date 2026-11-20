"""Rollout policy over non-anticipative forks (design §18; Secomandi 2001, Goodson et al. 2013).

At each epoch, every valid action is evaluated by ``n_samples`` simulations: fork the environment
(`FleetGymEnv.fork`, which redraws everything not yet revealed), apply the action, and complete the
episode with the base policy. The action with the lowest mean cost-to-go is chosen (ties: lowest
action). On a deterministic config the forks equal the environment, so this is the exact one-step
improvement of the base policy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fleetgym.baselines.protocol import Policy, PolicyBase, valid_actions
from fleetgym.envs.env import FleetGymEnv

__all__ = ["Rollout", "cost_to_go"]


def cost_to_go(env: FleetGymEnv, first: int, base: Policy, *, max_steps: int = 1_000_000) -> float:
    """Cost of applying ``first`` in ``env``, then following ``base`` to the end (mutates env)."""
    obs, reward, terminated, truncated, info = env.step(first)
    cost = -float(reward)
    base.reset(env)
    for _ in range(max_steps):
        if terminated or truncated:
            return cost
        obs, reward, terminated, truncated, info = env.step(base.act(obs, info))
        cost -= float(reward)
    raise RuntimeError("the base policy did not finish the episode")


class Rollout(PolicyBase):
    """One-step lookahead with a base policy on forks.

    Parameters
    ----------
    base
        Factory of the base policy (a fresh one per simulation).
    n_samples
        Simulations per action.
    seed
        Seed of the forks (sample ``k`` of epoch ``t`` uses seed ``(seed, t, k)``).
    """

    name = "rollout"

    def __init__(self, base: Callable[[], Policy], *, n_samples: int = 4, seed: int = 0) -> None:
        super().__init__()
        self.base = base
        self.n_samples = n_samples
        self.seed = seed

    def act(self, obs: dict[str, NDArray[Any]], info: dict[str, Any]) -> int:
        """The valid action with the lowest mean simulated cost-to-go."""
        env = self.env
        actions = valid_actions(info)
        if actions.size == 1:
            return int(actions[0])
        epoch_index = env.simulator.state.epoch_count
        best_action, best_cost = int(actions[0]), np.inf
        for action in actions:
            costs = []
            for k in range(self.n_samples):
                fork_seed = int(
                    np.random.SeedSequence([self.seed, epoch_index, k]).generate_state(1)[0]
                )
                costs.append(cost_to_go(env.fork(seed=fork_seed), int(action), self.base()))
            mean = float(np.mean(costs))
            if mean < best_cost - 1e-12:
                best_action, best_cost = int(action), mean
        return best_action
