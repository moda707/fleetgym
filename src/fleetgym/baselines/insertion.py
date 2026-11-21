"""Cheapest-insertion policy (design §18; wraps `fleetgym.sim.routing_rules`, plan Q27).

* Vehicle epochs (``next_node``): build the deciding vehicle's route from where it is by repeated
  cheapest insertion of the pending customers (the customer whose cheapest feasible insertion adds
  the least distance goes in first; ties by lower index), and drive to the first stop. Feasibility
  uses known information (forecasts for unrevealed demands). With an empty route: wait if offered,
  else return to the depot.
* Recourse and request epochs: as `NearestNeighbor` (return to the depot; accept if feasible).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fleetgym.baselines.nearest import NearestNeighbor, request_decision
from fleetgym.baselines.protocol import valid_actions
from fleetgym.sim.engine import Simulator
from fleetgym.sim.info_state import DEPOT
from fleetgym.sim.routing_rules import Anchor, RouteRules, best_insertion

__all__ = ["CheapestInsertion", "route_rules"]


def route_rules(sim: Simulator) -> RouteRules:
    """Feasibility rules of a plan in ``sim`` from known information."""
    spec = sim.problem.spec
    tw = spec.constraints.items.get("time_windows")
    info = sim.info
    return RouteRules(
        distances=info.distances,
        travel_times=info.travel_times,
        demands=info.demands,
        service_times=info.service_times,
        time_windows=sim.problem.instance.time_windows if tw is not None else None,
        hard_windows=tw is not None and tw.model_dump()["mode"] != "soft",
        capacity_on=sim.capacity_on,
        max_duration=(
            float(sim.problem.value("constraints.route_duration.max"))
            if "route_duration" in spec.constraints
            and spec.constraints["route_duration"].model_dump()["mode"] == "hard"
            else None
        ),
        shift_end=sim.shift_end,
    )


class CheapestInsertion(NearestNeighbor):
    """Route the deciding vehicle by cheapest insertion; follow the first stop."""

    name = "cheapest_insertion"

    def plan(self, vehicle: int, mask: NDArray[np.bool_]) -> list[int]:
        """The cheapest-insertion route of ``vehicle`` over the customers valid now."""
        sim = self.env.simulator
        v = sim.state.vehicles[vehicle]
        anchor = Anchor(v.node, max(v.time, sim.state.time), v.load, v.route_start, v.capacity)
        rules = route_rules(sim)
        n = sim.problem.instance.n_nodes
        left = [int(c) for c in np.flatnonzero(mask[1:n]) + 1]
        route: list[int] = []
        while left:
            options = [(best_insertion(rules, [anchor], [route], c), c) for c in left]
            feasible = [(ins.cost, c, ins.position) for ins, c in options if ins is not None]
            if not feasible:
                break
            _, customer, position = min(feasible)
            route.insert(position, customer)
            left.remove(customer)
        return route

    def act(self, obs: dict[str, NDArray[Any]], info: dict[str, Any]) -> int:
        """First stop of the cheapest-insertion route, else wait or the depot."""
        env = self.env
        epoch = env.epoch
        mask = info["action_mask"]
        if epoch is None or epoch.kind != "vehicle_idle":
            if epoch is not None and epoch.kind == "request":
                return request_decision(self, info)
            return super().act(obs, info)
        assert epoch.vehicle is not None
        route = self.plan(epoch.vehicle, mask)
        if route:
            return route[0]
        wait = env.interface.max_nodes
        if env.interface.wait and mask[wait]:
            return wait
        return DEPOT if mask[DEPOT] else int(valid_actions(info)[0])
