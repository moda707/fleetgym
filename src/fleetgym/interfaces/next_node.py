"""The `next_node` action interface.

The agent picks the next node for the vehicle of the current epoch. Action ``i`` is node ``i``;
slots beyond the instance's nodes are padding and always masked. Choosing the depot returns to it
from a customer, or retires a vehicle that has not left the depot (only allowed for fixed and
limited fleets, or when nothing else is feasible).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from fleetgym.errors import InfeasibleActionError
from fleetgym.interfaces.base import INTERFACE_CLASSES, ActionInterface
from fleetgym.sim.decisions import Decision, Dispatch, EndRoute
from fleetgym.sim.engine import Epoch, Simulator
from fleetgym.sim.info_state import DEPOT

__all__ = ["NextNode"]


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
        node_mask = sim.mask(epoch.vehicle)
        out = np.zeros(self.max_nodes, dtype=bool)
        out[: node_mask.size] = node_mask
        return out

    def decode(self, action: Any, sim: Simulator, epoch: Epoch) -> Decision:
        """Node index to `Dispatch`, or `EndRoute` for the depot when the vehicle has not left."""
        node = int(action)
        if not 0 <= node < sim.problem.instance.n_nodes:
            raise InfeasibleActionError(f"action {node} is a padding slot, not a node")
        v = sim.state.vehicles[epoch.vehicle]
        if node == DEPOT and v.node == DEPOT and not v.used:
            return EndRoute(epoch.vehicle)
        return Dispatch(epoch.vehicle, node)

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
            else sim.state.vehicles[epoch.vehicle].node
        )
        order = np.lexsort((feasible, distances[origin, feasible]))
        return int(feasible[order[0]])
