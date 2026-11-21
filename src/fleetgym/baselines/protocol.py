"""The `Policy` protocol shared by baselines and learned agents (design §18).

A policy is bound to an environment with `reset` (at the start of an episode, or at any state for
rollouts) and then chooses actions with ``act(obs, info)``. ``anticipative`` marks policies that
read the true future (only the perfect-information benchmark); their results are labeled and never
compared as if they were achievable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from fleetgym.envs.env import FleetGymEnv
from fleetgym.envs.ev_env import EVShortestPathEnv

__all__ = ["Episode", "Policy", "PolicyBase", "run_episode", "valid_actions"]


@runtime_checkable
class Policy(Protocol):
    """What the evaluation harness needs from a policy."""

    name: str
    anticipative: bool

    def reset(self, env: Any) -> None:
        """Bind to ``env`` (at the start of an episode or at any later state).

        ``env`` is a `FleetGymEnv`, or an `EVShortestPathEnv` for the EV policies of
        `fleetgym.baselines.ev`.
        """

    def act(self, obs: dict[str, NDArray[Any]], info: dict[str, Any]) -> int:
        """The action for the current epoch."""
        ...


class PolicyBase:
    """Convenience base class: stores the bound environment."""

    name = "policy"
    anticipative = False

    def __init__(self) -> None:
        self._env: FleetGymEnv | None = None

    @property
    def env(self) -> FleetGymEnv:
        """The bound environment."""
        if self._env is None:
            raise RuntimeError(f"{self.name}: call reset(env) first")
        return self._env

    def reset(self, env: FleetGymEnv) -> None:
        """Bind to ``env``."""
        self._env = env


def valid_actions(info: dict[str, Any]) -> NDArray[np.int64]:
    """Indices of the valid actions in ``info["action_mask"]``."""
    return np.flatnonzero(info["action_mask"]).astype(np.int64)


@dataclass
class Episode:
    """Outcome of one episode.

    Attributes
    ----------
    cost
        Total weighted cost (minus the episode return).
    costs
        Total cost by component.
    steps
        Number of decisions.
    decision_seconds
        Wall-clock time of each `Policy.act` call.
    """

    seed: int
    cost: float
    costs: dict[str, float]
    steps: int
    decision_seconds: list[float] = field(default_factory=list)


def run_episode(
    env: FleetGymEnv | EVShortestPathEnv,
    policy: Policy,
    *,
    seed: int,
    options: dict[str, Any] | None = None,
    max_steps: int = 1_000_000,
) -> Episode:
    """Play one episode of ``policy`` on ``env`` from ``env.reset(seed=seed, options=options)``.

    Raises
    ------
    RuntimeError
        If the episode does not end within ``max_steps``.
    """
    obs, info = env.reset(seed=seed, options=options)
    policy.reset(env)
    total = 0.0
    times: list[float] = []
    for step in range(max_steps):
        start = time.perf_counter()
        action = policy.act(obs, info)
        times.append(time.perf_counter() - start)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        if terminated or truncated:
            return Episode(
                seed=seed,
                cost=-total,
                costs=dict(env.simulator.state.total_costs),
                steps=step + 1,
                decision_seconds=times,
            )
    raise RuntimeError(f"{policy.name}: the episode did not end within {max_steps} steps")
