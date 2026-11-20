"""Read-only, non-anticipative view of the simulator state.

Constraints, observation builders and policies read the simulator only through this view. It
exposes what is known at the current time: in static problems that is the whole instance; with
uncertainty (M5) unrevealed values are hidden behind forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fleetgym.network.base import Network
from fleetgym.sim.state import SimState, VehicleState
from fleetgym.spec.binding import BoundProblem

__all__ = ["DEPOT", "InformationState", "TimingRules"]

DEPOT = 0


@dataclass(frozen=True)
class TimingRules:
    """How arrival times turn into service start times.

    Attributes
    ----------
    use_time_windows
        True when the spec has the ``time_windows`` constraint.
    waiting_allowed
        Whether a vehicle arriving early may wait for the window to open.
    """

    use_time_windows: bool
    waiting_allowed: bool


class InformationState:
    """What the decision maker knows now.

    Parameters
    ----------
    problem
        The bound problem.
    network
        Its network.
    state
        The simulator state (read through this view only; do not mutate).
    timing
        Service-start rules.
    """

    def __init__(
        self, problem: BoundProblem, network: Network, state: SimState, timing: TimingRules
    ) -> None:
        self.problem = problem
        self.network = network
        self._state = state
        self.timing = timing
        instance = problem.instance
        n = instance.n_nodes
        self.n_nodes = n
        self.demands: NDArray[np.float64] = instance.demands
        self.service_times: NDArray[np.float64] = (
            instance.service_times if instance.service_times is not None else np.zeros(n)
        )
        self.time_windows: NDArray[np.float64] | None = instance.time_windows
        self.travel_times = network.time_matrix()
        self.distances = network.distance_matrix()

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._state.time

    def vehicle(self, index: int) -> VehicleState:
        """State of vehicle ``index`` (read only)."""
        return self._state.vehicles[index]

    @property
    def n_vehicles(self) -> int:
        """Number of vehicles created so far."""
        return len(self._state.vehicles)

    def pending(self) -> NDArray[np.bool_]:
        """Mask of nodes whose request is pending (known, unassigned, unserved)."""
        return self._state.pending_mask()

    def request_status(self) -> NDArray[np.int8]:
        """Request status per node (see `RequestStatus`)."""
        return self._state.request_status

    def service_start(
        self, nodes: NDArray[np.int64], arrival: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Service start at ``nodes`` for the given arrival times."""
        if (
            self.timing.use_time_windows
            and self.time_windows is not None
            and self.timing.waiting_allowed
        ):
            start = np.maximum(arrival, self.time_windows[nodes, 0])
            return np.where(nodes == DEPOT, arrival, start)
        return arrival

    def projected(
        self, vehicle: int, nodes: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Arrival, service start and service end if ``vehicle`` goes to each node now."""
        v = self._state.vehicles[vehicle]
        arrival = v.time + self.travel_times[v.node, nodes]
        start = self.service_start(nodes, arrival)
        end = start + self.service_times[nodes]
        return arrival, start, end
