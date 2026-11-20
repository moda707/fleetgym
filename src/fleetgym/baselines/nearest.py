"""Nearest-neighbor policy (design §18, myopic rule).

* Vehicle epochs (``next_node``): the nearest valid customer (ties: lowest index); without one,
  wait if waiting is offered, else return to the depot (or the first valid action).
* Recourse epochs: return to the depot if allowed, else outsource.
* Request epochs: accept if feasible (``accept_reject``), or the vehicle with the cheapest insertion
  (``assign``); otherwise reject.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fleetgym.baselines.protocol import PolicyBase, valid_actions
from fleetgym.sim.info_state import DEPOT
from fleetgym.sim.planned import PlannedRoutingSimulator

__all__ = ["NearestNeighbor", "request_decision"]


def request_decision(policy: PolicyBase, info: dict[str, Any]) -> int:
    """Greedy request decision: accept or the cheapest vehicle if feasible, else reject."""
    env = policy.env
    mask = info["action_mask"]
    if env.problem_spec.decisions.interface == "accept_reject":
        return 1 if mask[1] else 0
    sim = env.simulator
    assert isinstance(sim, PlannedRoutingSimulator)
    epoch = env.epoch
    assert epoch is not None
    assert epoch.node is not None
    best = sim.insertion(epoch.node)
    reject = mask.size - 1
    return (
        best.vehicle
        if best is not None and best.vehicle < reject and mask[best.vehicle]
        else reject
    )


class NearestNeighbor(PolicyBase):
    """Go to the nearest valid customer."""

    name = "nearest"

    def act(self, obs: dict[str, NDArray[Any]], info: dict[str, Any]) -> int:
        """The nearest valid customer, else wait or the depot."""
        env = self.env
        epoch = env.epoch
        mask = info["action_mask"]
        if epoch is None:
            return int(valid_actions(info)[0])
        if epoch.kind == "request":
            return request_decision(self, info)
        if epoch.kind == "recourse":
            return DEPOT if mask[DEPOT] else int(valid_actions(info)[0])
        sim = env.simulator
        assert epoch.vehicle is not None
        here = sim.state.vehicles[epoch.vehicle].node
        n = sim.problem.instance.n_nodes
        customers = np.flatnonzero(mask[1:n]) + 1
        if customers.size:
            distances = sim.info.distances[here, customers]
            return int(customers[np.lexsort((customers, distances))[0]])
        wait = env.interface.max_nodes
        if env.interface.wait and mask[wait]:
            return wait
        return DEPOT if mask[DEPOT] else int(valid_actions(info)[0])
