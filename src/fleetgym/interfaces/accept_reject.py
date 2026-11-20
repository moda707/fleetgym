"""The `accept_reject` action interface (design §12, plan Q26).

At each request epoch the agent decides on the released request: ``0`` rejects it, ``1`` accepts
it (the routing rule inserts it into the best vehicle's plan), ``2`` outsources it. Accept is masked
when no feasible insertion exists; outsource is masked when the spec has no ``outsourcing``
constraint. Reject is always allowed.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from fleetgym.errors import InfeasibleActionError
from fleetgym.interfaces.base import INTERFACE_CLASSES, ActionInterface
from fleetgym.sim.decisions import Accept, Decision, OutsourceRequest, Reject
from fleetgym.sim.engine import Epoch, Simulator
from fleetgym.sim.planned import PlannedRoutingSimulator

__all__ = ["AcceptReject", "planned"]

REJECT, ACCEPT, OUTSOURCE = 0, 1, 2


def planned(sim: Simulator, epoch: Epoch) -> tuple[PlannedRoutingSimulator, int]:
    """The planned-routing simulator and the request of a request epoch."""
    if not isinstance(sim, PlannedRoutingSimulator) or epoch.kind != "request":
        raise InfeasibleActionError("this interface decides on requests at request epochs")
    assert epoch.node is not None
    return sim, epoch.node


@INTERFACE_CLASSES.register("accept_reject")
class AcceptReject(ActionInterface):
    """``Discrete(3)``: reject, accept or outsource the current request."""

    name = "accept_reject"

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """``Discrete(3)``."""
        return gym.spaces.Discrete(3)

    @property
    def n_actions(self) -> int:
        """3."""
        return 3

    def mask(self, sim: Simulator, epoch: Epoch) -> NDArray[np.bool_]:
        """Reject always; accept if insertable; outsource if the spec allows it."""
        planned_sim, node = planned(sim, epoch)
        can_accept, can_outsource = planned_sim.accept_options(node)
        return np.array([True, can_accept, can_outsource], dtype=bool)

    def decode(self, action: Any, sim: Simulator, epoch: Epoch) -> Decision:
        """0: `Reject`, 1: `Accept`, 2: `OutsourceRequest`."""
        _, node = planned(sim, epoch)
        choice = int(action)
        if choice == REJECT:
            return Reject(node)
        if choice == ACCEPT:
            return Accept(node)
        if choice == OUTSOURCE:
            return OutsourceRequest(node)
        raise InfeasibleActionError(f"action {choice} is not 0, 1 or 2")

    def repair(self, action: Any, mask: NDArray[np.bool_], sim: Simulator, epoch: Epoch) -> int:
        """An invalid accept or outsource becomes a reject."""
        choice = int(action)
        return choice if 0 <= choice < 3 and mask[choice] else REJECT
