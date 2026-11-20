"""Perfect-information benchmark (design §18; plan Q29).

`PerfectInformation` reads the true future through `FleetGymEnv.clone` (anticipative), so its cost
is not achievable by a real policy: it measures the price of uncertainty.

* ``method="exact"``: exhaustive search over every action sequence on clones (depth-first, pruned by
  the best cost so far). The result is the hindsight optimum within the environment's action space,
  hence a **lower bound** on every non-anticipative policy's cost on that scenario. For tiny
  instances only (``max_nodes`` caps the work).
* ``method="rollout"``: one-step lookahead on clones with a base policy, a **perfect-information
  heuristic** (not a bound).

``bound`` tells which one a result is; report cards must say so (Q29).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from fleetgym.baselines.nearest import NearestNeighbor
from fleetgym.baselines.protocol import Policy, PolicyBase, valid_actions
from fleetgym.baselines.rollout import cost_to_go
from fleetgym.envs.env import FleetGymEnv

__all__ = ["PerfectInformation", "hindsight_optimum"]


def hindsight_optimum(env: FleetGymEnv, *, max_nodes: int = 9) -> tuple[float, list[int]]:
    """Minimal remaining cost from ``env``'s state and an action sequence achieving it.

    Searches a clone, so ``env`` is unchanged.

    Raises
    ------
    ValueError
        If the instance has more than ``max_nodes`` nodes (the search is exponential).
    """
    n = env.simulator.problem.instance.n_nodes
    if n > max_nodes:
        raise ValueError(f"exact hindsight search is limited to {max_nodes} nodes, got {n}")
    twin = env.clone()
    best: list[Any] = [np.inf, []]

    def search(cost: float, path: list[int]) -> None:
        if cost >= best[0] - 1e-12:
            return
        snapshot = twin.get_state()
        for action in np.flatnonzero(twin.action_masks()):
            twin.set_state(snapshot)
            _, reward, terminated, truncated, _ = twin.step(int(action))
            total = cost - float(reward)
            if terminated or truncated:
                if total < best[0] - 1e-12:
                    best[0], best[1] = total, [*path, int(action)]
            else:
                search(total, [*path, int(action)])
        twin.set_state(snapshot)

    search(0.0, [])
    return float(best[0]), list(best[1])


class PerfectInformation(PolicyBase):
    """Anticipative benchmark: knows the scenario.

    Parameters
    ----------
    method
        ``"exact"`` (a bound, tiny instances) or ``"rollout"`` (a heuristic).
    base
        Base policy factory for ``"rollout"`` (default: `NearestNeighbor`).
    """

    name = "perfect_information"
    anticipative = True

    def __init__(
        self,
        method: Literal["exact", "rollout"] = "exact",
        *,
        base: Callable[[], Policy] = NearestNeighbor,
        max_nodes: int = 9,
    ) -> None:
        super().__init__()
        self.method = method
        self.base = base
        self.max_nodes = max_nodes
        self._plan: list[int] = []

    @property
    def bound(self) -> bool:
        """True when the cost is a lower bound (exact search)."""
        return self.method == "exact"

    def reset(self, env: FleetGymEnv) -> None:
        """Bind to ``env``; the exact method solves the whole episode now."""
        super().reset(env)
        self._plan = []
        if self.method == "exact" and env.epoch is not None:
            _, self._plan = hindsight_optimum(env, max_nodes=self.max_nodes)

    def act(self, obs: dict[str, NDArray[Any]], info: dict[str, Any]) -> int:
        """Next action of the hindsight plan, or of the clone-based rollout."""
        actions = valid_actions(info)
        if self.method == "exact":
            if self._plan:
                return self._plan.pop(0)
            return int(actions[0])
        best_action, best_cost = int(actions[0]), np.inf
        for action in actions:
            cost = cost_to_go(self.env.clone(), int(action), self.base())
            if cost < best_cost - 1e-12:
                best_action, best_cost = int(action), cost
        return best_action
