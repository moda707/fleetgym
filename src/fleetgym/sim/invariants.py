"""Runtime invariant monitors used in debug mode.

`InvariantMonitor.check` raises `InvariantViolation` (never a bare ``assert``, which ``python -O``
would strip) when the simulator state is inconsistent.
"""

from __future__ import annotations

import numpy as np

from fleetgym.errors import FleetGymError
from fleetgym.sim.engine import Simulator
from fleetgym.sim.state import RequestStatus, VehicleStatus

__all__ = ["InvariantMonitor", "InvariantViolation"]

_TOL = 1e-9


class InvariantViolation(FleetGymError):
    """The simulator reached an inconsistent state (a FleetGym bug)."""


class InvariantMonitor:
    """Check simulator invariants after every step.

    Checks: loads within capacity (when the capacity constraint is on), per-vehicle clocks never go
    back, route start before vehicle time, request statuses consistent with service records, and
    every assigned request has a vehicle heading to it.
    """

    def __init__(self) -> None:
        self._last_time: dict[int, float] = {}

    def reset(self) -> None:
        """Forget clocks from a previous episode."""
        self._last_time.clear()

    def check(self, sim: Simulator) -> None:
        """Raise `InvariantViolation` if any invariant fails."""
        state = sim.state
        capacity_on = "capacity" in sim.problem.spec.constraints
        for v in state.vehicles:
            if capacity_on and v.load > v.capacity + _TOL:
                raise InvariantViolation(
                    f"vehicle {v.index} carries {v.load} > capacity {v.capacity}"
                )
            if v.load < -_TOL:
                raise InvariantViolation(f"vehicle {v.index} has negative load {v.load}")
            last = self._last_time.get(v.index)
            if last is not None and v.time < last - _TOL:
                raise InvariantViolation(
                    f"vehicle {v.index} clock went back from {last} to {v.time}"
                )
            self._last_time[v.index] = v.time
            if v.route_start is not None and v.route_start > v.time + _TOL:
                raise InvariantViolation(f"vehicle {v.index} route starts after its clock")
        status = state.request_status
        served = status == RequestStatus.SERVED
        if np.any(np.isnan(state.service_start[served])) or np.any(state.served_by[served] < 0):
            raise InvariantViolation("a served request has no service record")
        unserved = (status == RequestStatus.PENDING) | (status == RequestStatus.UNSERVED)
        if np.any(state.served_by[unserved] >= 0):
            raise InvariantViolation("an unserved request has a serving vehicle")
        heading = {
            v.node
            for v in state.vehicles
            if v.status in (VehicleStatus.TRAVELING, VehicleStatus.SERVING)
        }
        for node in np.flatnonzero(status == RequestStatus.ASSIGNED):
            if int(node) not in heading:
                raise InvariantViolation(
                    f"request {node} is assigned but no vehicle is heading there"
                )
        if any(amount < -_TOL for amount in state.total_costs.values()):
            raise InvariantViolation(f"negative cost component in {state.total_costs}")
