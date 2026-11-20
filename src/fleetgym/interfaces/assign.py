"""The `assign` action interface (design §12, plan Q26).

At each request epoch the agent picks the vehicle for the released request: action ``v`` inserts
it into vehicle ``v``'s plan (by the routing rule, masked when infeasible or when the vehicle does
not exist), and the last action, ``max_vehicles``, rejects it (always allowed).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from fleetgym.errors import InfeasibleActionError
from fleetgym.interfaces.accept_reject import planned
from fleetgym.interfaces.base import INTERFACE_CLASSES, ActionInterface
from fleetgym.sim.decisions import Assign, Decision, Reject
from fleetgym.sim.engine import Epoch, Simulator
from fleetgym.sim.state import VehicleStatus

__all__ = ["AssignVehicle"]


@INTERFACE_CLASSES.register("assign")
class AssignVehicle(ActionInterface):
    """``Discrete(max_vehicles + 1)``: a vehicle for the current request, or reject (last)."""

    name = "assign"

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """``Discrete(max_vehicles + 1)``."""
        return gym.spaces.Discrete(self.max_vehicles + 1)

    @property
    def n_actions(self) -> int:
        """``max_vehicles + 1``."""
        return self.max_vehicles + 1

    def mask(self, sim: Simulator, epoch: Epoch) -> NDArray[np.bool_]:
        """Vehicles with a feasible insertion, and reject."""
        planned_sim, node = planned(sim, epoch)
        mask = np.zeros(self.n_actions, dtype=bool)
        for v in planned_sim.state.vehicles[: self.max_vehicles]:
            if v.status != VehicleStatus.DONE:
                mask[v.index] = planned_sim.insertion(node, v.index) is not None
        mask[self.max_vehicles] = True
        return mask

    def decode(self, action: Any, sim: Simulator, epoch: Epoch) -> Decision:
        """``v < max_vehicles``: `Assign` to ``v``; ``max_vehicles``: `Reject`."""
        _, node = planned(sim, epoch)
        choice = int(action)
        if choice == self.max_vehicles:
            return Reject(node)
        if not 0 <= choice < self.max_vehicles:
            raise InfeasibleActionError(f"action {choice} is outside 0..{self.max_vehicles}")
        return Assign(node, choice)

    def repair(self, action: Any, mask: NDArray[np.bool_], sim: Simulator, epoch: Epoch) -> int:
        """An invalid assignment becomes the cheapest feasible vehicle, else a reject."""
        choice = int(action)
        if 0 <= choice < mask.size and mask[choice]:
            return choice
        planned_sim, node = planned(sim, epoch)
        best = planned_sim.insertion(node)
        if best is not None and best.vehicle < self.max_vehicles and mask[best.vehicle]:
            return best.vehicle
        return self.max_vehicles
