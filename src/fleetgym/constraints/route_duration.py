"""Maximum route duration constraint."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fleetgym.constraints.base import CONSTRAINT_CLASSES, Constraint
from fleetgym.sim.info_state import DEPOT, InformationState

__all__ = ["RouteDuration"]


@CONSTRAINT_CLASSES.register("route_duration")
class RouteDuration(Constraint):
    """Route duration (return time minus departure time) must not exceed ``max``.

    A customer is masked if, after serving it, the vehicle could not be back at the depot within the
    limit. Duration includes travel, waiting and service times.
    """

    name = "route_duration"

    @property
    def limit(self) -> float:
        """The resolved maximum duration."""
        return float(self.problem.value("constraints.route_duration.max"))

    def mask(
        self, state: Any, info: InformationState, vehicle: int, candidates: NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        """Candidates from which the vehicle can still return in time."""
        v = info.vehicle(vehicle)
        start = v.route_start if v.route_start is not None else v.time
        _, _, end = info.projected(vehicle, candidates)
        back = end + info.travel_times[candidates, DEPOT]
        return back - start <= self.limit
