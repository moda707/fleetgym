"""Read-only, non-anticipative view of the simulator state.

Constraints, observation builders, recourse rules and policies read the simulator only through this
view. It exposes what is known at the current time: in static problems that is the whole instance;
for an uncertain source, revealed values are exact and unrevealed values are replaced by the
forecast (the distribution's mean with ``forecast: mean``, 0 with ``forecast: none``). The
``*_known`` masks tell them apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fleetgym.network.base import Network
from fleetgym.sim.state import RequestStatus, SimState, VehicleState
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
    forecasts
        Per uncertain source, what the agent assumes for unrevealed values.
    """

    def __init__(
        self,
        problem: BoundProblem,
        network: Network,
        state: SimState,
        timing: TimingRules,
        forecasts: Mapping[str, NDArray[np.float64]] | None = None,
    ) -> None:
        self.problem = problem
        self.network = network
        self._state = state
        self.timing = timing
        self._forecasts = dict(forecasts or {})
        instance = problem.instance
        n = instance.n_nodes
        self.n_nodes = n
        self._static_demands: NDArray[np.float64] = instance.demands
        self._static_service: NDArray[np.float64] = (
            instance.service_times if instance.service_times is not None else np.zeros(n)
        )
        self._all_known = np.ones(n, dtype=bool)
        self._all_known.setflags(write=False)
        self.time_windows: NDArray[np.float64] | None = instance.time_windows
        self.travel_times = network.time_matrix()
        self.distances = network.distance_matrix()

    def _known(self, source: str, static: NDArray[np.float64]) -> NDArray[np.float64]:
        if source not in self._forecasts:
            return static
        revealed = self._state.revealed[source]
        return np.where(revealed, self._state.known_values[source], self._forecasts[source])

    @property
    def demands(self) -> NDArray[np.float64]:
        """Demand per node: revealed values, forecasts elsewhere."""
        return self._known("demand", self._static_demands)

    @property
    def open_demands(self) -> NDArray[np.float64]:
        """Demand still to deliver where it is known, else as `demands`.

        Differs from `demands` only at pending customers that were partly served (``split``
        recourse): their demand is revealed, and the delivered part is known too.
        """
        demands = self.demands
        known = self.demand_known & (self._state.request_status == RequestStatus.PENDING)
        return np.where(known, np.minimum(demands, self._state.remaining_demand), demands)

    @property
    def service_times(self) -> NDArray[np.float64]:
        """Service time per node: revealed values, forecasts elsewhere."""
        return self._known("service_time", self._static_service)

    def known(self, source: str) -> NDArray[np.bool_]:
        """Mask of nodes whose ``source`` value is revealed (all nodes for certain sources)."""
        if source not in self._forecasts:
            return self._all_known
        return self._state.revealed[source]

    @property
    def demand_known(self) -> NDArray[np.bool_]:
        """Mask of nodes whose demand is revealed."""
        return self.known("demand")

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
