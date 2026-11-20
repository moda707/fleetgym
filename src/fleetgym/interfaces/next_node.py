"""The `next_node` action interface.

The agent picks the next node for the vehicle of the current epoch. Action ``i`` is node ``i``;
slots beyond the instance's nodes are padding and always masked. Choosing the depot returns to it
from a customer, ends the route of a vehicle waiting at the depot after a restock, or retires a
vehicle that has not left the depot (only allowed for fixed and limited fleets, or when nothing
else is feasible).

At a recourse epoch (``agent_decides``), the vehicle is at the failing customer ``i``: action 0
(the depot) means ``return_to_depot`` (restock and come back), and action ``i`` (stay) means
``outsource`` the rest. The mask allows only the rule's options.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from fleetgym.errors import InfeasibleActionError
from fleetgym.interfaces.base import INTERFACE_CLASSES, ActionInterface
from fleetgym.sim.decisions import Decision, Dispatch, EndRoute, RecourseChoice
from fleetgym.sim.engine import Epoch, Simulator
from fleetgym.sim.info_state import DEPOT

__all__ = ["NextNode"]


def _vehicle(epoch: Epoch) -> int:
    if epoch.vehicle is None:
        raise InfeasibleActionError("next_node needs a vehicle epoch, not a request epoch")
    return epoch.vehicle


@INTERFACE_CLASSES.register("next_node")
class NextNode(ActionInterface):
    """``Discrete(max_nodes)``: the next node of the vehicle that triggered the epoch."""

    name = "next_node"

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """``Discrete(max_nodes)``."""
        return gym.spaces.Discrete(self.max_nodes)

    @property
    def n_actions(self) -> int:
        """Equal to ``max_nodes``."""
        return self.max_nodes

    def mask(self, sim: Simulator, epoch: Epoch) -> NDArray[np.bool_]:
        """The simulator's node mask, padded to ``max_nodes``."""
        node_mask = sim.mask(_vehicle(epoch))
        out = np.zeros(self.max_nodes, dtype=bool)
        out[: node_mask.size] = node_mask
        return out

    def decode(self, action: Any, sim: Simulator, epoch: Epoch) -> Decision:
        """Node index to `Dispatch`, or `EndRoute` for the depot when the vehicle has not left."""
        node = int(action)
        if not 0 <= node < sim.problem.instance.n_nodes:
            raise InfeasibleActionError(f"action {node} is a padding slot, not a node")
        vehicle = _vehicle(epoch)
        v = sim.state.vehicles[vehicle]
        if epoch.kind == "recourse":
            if node == DEPOT:
                return RecourseChoice(vehicle, "return_to_depot")
            if node == epoch.node:
                return RecourseChoice(vehicle, "outsource")
            raise InfeasibleActionError(
                f"at a recourse epoch the action is 0 (return to depot) or {epoch.node} (outsource)"
            )
        if node == DEPOT and v.node == DEPOT:
            return EndRoute(vehicle)
        return Dispatch(vehicle, node)

    def repair(self, action: Any, mask: NDArray[np.bool_], sim: Simulator, epoch: Epoch) -> int:
        """The feasible node closest to the chosen node (or to the vehicle, for padding slots).

        Ties go to the lowest node index.
        """
        node = int(action)
        if 0 <= node < mask.size and mask[node]:
            return node
        feasible = np.flatnonzero(mask)
        distances = sim.info.distances
        origin = (
            node
            if 0 <= node < sim.problem.instance.n_nodes
            else sim.state.vehicles[_vehicle(epoch)].node
        )
        order = np.lexsort((feasible, distances[origin, feasible]))
        return int(feasible[order[0]])

    def explain(self, action: Any, mask: NDArray[np.bool_], sim: Simulator, epoch: Epoch) -> str:
        """Padding slot, recourse epoch, retiring an unlimited fleet, or the violated constraint."""
        node = int(action)
        n = sim.problem.instance.n_nodes
        if node >= n:
            return f"action {node} is a padding slot (the instance has {n} nodes)"
        if epoch.kind == "recourse":
            return (
                f"at a recourse epoch the valid actions are {np.flatnonzero(mask).tolist()}"
                " (0: return to the depot; the failing customer: outsource)"
            )
        vehicle = _vehicle(epoch)
        if node == DEPOT and sim.state.vehicles[vehicle].node == DEPOT:
            return "this vehicle cannot retire: the fleet is unlimited and customers are feasible"
        return sim._explain(vehicle, node)  # pyright: ignore[reportPrivateUsage]
