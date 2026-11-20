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
    """A customer is feasible if its demand fits in the vehicle's remaining capacity."""

    name = "capacity"

    def mask(
        self, state: Any, info: InformationState, vehicle: int, candidates: NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        """Demand of each candidate must not exceed ``capacity - load``."""
        v = info.vehicle(vehicle)
        return info.demands[candidates] <= v.capacity - v.load
