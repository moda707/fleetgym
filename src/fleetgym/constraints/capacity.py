"""Vehicle capacity constraint."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fleetgym.constraints.base import CONSTRAINT_CLASSES, Constraint
from fleetgym.sim.info_state import InformationState

__all__ = ["Capacity"]


@CONSTRAINT_CLASSES.register("capacity")
class Capacity(Constraint):
    """A customer is feasible if its known demand fits in the vehicle's remaining capacity.

    A customer whose demand is not revealed yet is feasible: its demand may turn out to fit, and
    the spec's ``recourse.capacity_failure`` rule handles it if it does not (the simulator rejects
    specs where unrevealed demand meets the capacity constraint without a recourse rule).
    """

    name = "capacity"

    def mask(
        self, state: Any, info: InformationState, vehicle: int, candidates: NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        """Known demand of each candidate must not exceed ``capacity - load``."""
        v = info.vehicle(vehicle)
        fits = info.open_demands[candidates] <= v.capacity - v.load
        return fits | ~info.demand_known[candidates]
