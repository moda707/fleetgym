"""Hard and soft time-window constraint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fleetgym.constraints.base import CONSTRAINT_CLASSES, Constraint
from fleetgym.sim.events import Event, EventKind
from fleetgym.sim.info_state import DEPOT, InformationState

__all__ = ["TimeWindows"]


@CONSTRAINT_CLASSES.register("time_windows")
class TimeWindows(Constraint):
    """Customer time windows ``[open, close]`` on service start.

    Hard windows (``soft: false``) mask a customer if service would start after it closes, or if the
    vehicle could not then return to the depot before the depot's window closes. With
    ``waiting: forbidden``, arriving before a window opens is also masked. Soft windows mask nothing
    and report ``lateness = max(0, service_start - close)`` at service start.
    """

    name = "time_windows"

    @property
    def soft(self) -> bool:
        """Whether lateness is allowed (and charged)."""
        return self.params.model_dump()["mode"] == "soft"

    def mask(
        self, state: Any, info: InformationState, vehicle: int, candidates: NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        """Feasible candidates under the window rules (see the class docstring)."""
        tw = info.time_windows
        assert tw is not None  # checked when binding
        arrival, start, end = info.projected(vehicle, candidates)
        ok = np.ones(candidates.shape, dtype=bool)
        if not info.timing.waiting_allowed:
            ok &= arrival >= tw[candidates, 0]
        if not self.soft:
            ok &= start <= tw[candidates, 1]
            back = end + info.travel_times[candidates, DEPOT]
            ok &= back <= tw[DEPOT, 1]
        return ok

    def penalty(self, state: Any, info: InformationState, event: Event) -> Mapping[str, float]:
        """Lateness at service start under soft windows."""
        if not self.soft or event.kind != EventKind.SERVICE_START or event.node == DEPOT:
            return {}
        tw = info.time_windows
        assert tw is not None
        return {"lateness": max(0.0, event.time - float(tw[event.node, 1]))}
