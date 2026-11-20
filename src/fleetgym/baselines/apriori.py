"""A priori route policy: follow a fixed customer order and let the recourse rule handle failures.

This is the classical a priori optimization setting for stochastic demands (design §18, "a priori
route with recourse"): the route is fixed before any demand is revealed, and the spec's recourse
rule (for example ``return_to_depot``) repairs failures. The exact-expectation tests use it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fleetgym.envs.env import FleetGymEnv
from fleetgym.errors import InfeasibleActionError

__all__ = ["AprioriRoute"]


class AprioriRoute:
    """Visit customers in ``order``; return to the depot when no listed customer is allowed.

    Parameters
    ----------
    order
        Customer node indices in visiting order. Customers that are already handled or masked
        are skipped.

    Examples
    --------
    >>> policy = AprioriRoute([3, 1, 2])  # doctest: +SKIP
    >>> obs, info = env.reset(seed=0)  # doctest: +SKIP
    >>> obs, reward, terminated, truncated, info = env.step(policy.act(env))  # doctest: +SKIP
    """

    def __init__(self, order: Sequence[int]) -> None:
        self.order = [int(node) for node in order]

    def act(self, env: FleetGymEnv) -> int:
        """The next action for ``env``'s current epoch.

        Raises
        ------
        InfeasibleActionError
            At a recourse epoch: an a priori route needs a fixed recourse rule, not
            ``agent_decides``.
        """
        mask = env.action_masks()
        epoch = env._epoch  # pyright: ignore[reportPrivateUsage]
        if epoch is not None and epoch.kind == "recourse":
            raise InfeasibleActionError("AprioriRoute needs a fixed recourse rule")
        for node in self.order:
            if node < mask.size and mask[node]:
                return node
        if not mask[0]:
            return int(np.flatnonzero(mask)[0])
        return 0

    def run(self, env: FleetGymEnv, **reset_kwargs: object) -> float:
        """Play one episode from ``env.reset(**reset_kwargs)`` and return its total cost."""
        env.reset(**reset_kwargs)  # pyright: ignore[reportArgumentType]
        total = 0.0
        while True:
            _, reward, terminated, truncated, _ = env.step(self.act(env))
            total -= float(reward)
            if terminated or truncated:
                return total
