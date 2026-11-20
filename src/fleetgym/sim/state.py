"""Pure-data simulator state: vehicles, requests, clock and cost accumulators.

Everything here is plain data (dataclasses, lists, numpy arrays), so ``copy.deepcopy`` of a
`SimState` is an exact snapshot and restoring it reproduces the future bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from fleetgym.sim.events import EventQueue

__all__ = ["RequestStatus", "SimState", "VehicleState", "VehicleStatus"]


class VehicleStatus(IntEnum):
    """Lifecycle of a vehicle."""

    WAITING_TO_START = 0
    """Not yet active (sequential epochs: later vehicles wait for earlier routes to finish)."""
    IDLE = 1
    """At a node, free to receive a decision."""
    TRAVELING = 2
    """Driving to its next node."""
    SERVING = 3
    """Waiting for a time window to open or serving a customer."""
    DONE = 4
    """Route finished (back at the depot) or never used."""


class RequestStatus(IntEnum):
    """Lifecycle of a request (one per customer node in static problems)."""

    PENDING = 0
    """Known and not yet assigned to a vehicle (with ``accept_reject``/``assign``: awaiting the
    accept decision)."""
    ASSIGNED = 1
    """A vehicle is on its way (committed by a dispatch decision)."""
    SERVED = 2
    UNSERVED = 3
    """Still pending when the episode ended."""
    NOT_A_REQUEST = 4
    """Depot nodes."""
    NOT_RELEASED = 5
    """A dynamic request slot whose request has not arrived (or never will)."""
    REJECTED = 6
    CANCELLED = 7
    OUTSOURCED = 8


@dataclass
class VehicleState:
    """State of one vehicle.

    Attributes
    ----------
    node
        Current node; while traveling, the destination.
    time
        Time the vehicle became (or becomes) available at ``node``.
    load
        Demand delivered on the current route.
    route
        Nodes visited on the current route, starting with the depot.
    route_start
        Departure time from the depot, or ``None`` before the first departure.
    restocking
        Heading to the depot to restock (the route continues), not to end the route.
    recourse_target
        Customer to come back to after restocking.
    awaiting_recourse
        Customer whose capacity failure waits for the policy's recourse decision.
    plan
        Accepted customers still to visit after the current destination (planned routing).
    waiting
        Waiting for the next event (``next_node`` with dynamic requests).
    """

    index: int
    type_index: int
    capacity: float
    status: VehicleStatus
    node: int
    time: float
    load: float = 0.0
    route: list[int] = field(default_factory=list)
    route_start: float | None = None
    distance: float = 0.0
    restocking: bool = False
    recourse_target: int | None = None
    awaiting_recourse: int | None = None
    plan: list[int] = field(default_factory=list)
    waiting: bool = False

    @property
    def used(self) -> bool:
        """True once the vehicle has left the depot."""
        return self.route_start is not None


@dataclass
class SimState:
    """Complete simulator state.

    Attributes
    ----------
    time
        Simulation clock. With sequential epochs it restarts at the shift start for each vehicle.
    request_status
        `RequestStatus` per node.
    service_start
        Service start time per node (NaN until served).
    served_by
        Serving vehicle per node (-1 until served).
    active_vehicle
        Index of the vehicle whose decision is pending, if any.
    pending_costs
        Costs realized since the last decision epoch, by component.
    pending_vehicle_costs
        The part of ``pending_costs`` attributed to each vehicle (multi-agent rewards).
    total_costs
        Costs realized since the start of the episode, by component.
    remaining_demand
        True demand still to deliver per node (simulator-internal; the agent sees revealed values
        only, through `InformationState`).
    outsourced
        Quantity outsourced per node.
    revealed
        Per uncertainty source, which nodes' values the agent knows.
    known_values
        Per uncertainty source, the revealed values (NaN where unknown).
    """

    time: float
    vehicles: list[VehicleState]
    request_status: NDArray[np.int8]
    service_start: NDArray[np.float64]
    served_by: NDArray[np.int64]
    queue: EventQueue = field(default_factory=EventQueue)
    active_vehicle: int | None = None
    pending_costs: dict[str, float] = field(default_factory=dict)
    total_costs: dict[str, float] = field(default_factory=dict)
    epoch_count: int = 0
    terminated: bool = False
    fleet_exhausted: bool = False
    pending_vehicle_costs: dict[int, dict[str, float]] = field(default_factory=dict)
    remaining_demand: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    outsourced: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    revealed: dict[str, NDArray[np.bool_]] = field(default_factory=dict)
    known_values: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def add_cost(self, component: str, amount: float, vehicle: int | None = None) -> None:
        """Record a realized cost (zero amounts are ignored), attributed to ``vehicle`` if given."""
        if amount == 0:
            return
        self.pending_costs[component] = self.pending_costs.get(component, 0.0) + amount
        self.total_costs[component] = self.total_costs.get(component, 0.0) + amount
        if vehicle is not None:
            own = self.pending_vehicle_costs.setdefault(vehicle, {})
            own[component] = own.get(component, 0.0) + amount

    def take_pending_vehicle_costs(self) -> dict[int, dict[str, float]]:
        """Return and clear the per-vehicle part of the costs realized since the last epoch."""
        costs, self.pending_vehicle_costs = self.pending_vehicle_costs, {}
        return costs

    def take_pending_costs(self) -> dict[str, float]:
        """Return and clear the costs realized since the last epoch."""
        costs, self.pending_costs = self.pending_costs, {}
        return costs

    def pending_mask(self) -> NDArray[np.bool_]:
        """Boolean mask of nodes whose request is pending."""
        return self.request_status == RequestStatus.PENDING
