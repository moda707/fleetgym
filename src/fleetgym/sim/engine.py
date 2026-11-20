"""The `Simulator`: advances events and stops at decision epochs.

Semantics (ADR 0009)
--------------------
* A vehicle starts idle at the depot at the shift start (``fleet.shift[0]``, else 0).
* ``Dispatch(v, j)`` departs immediately: distance and travel time are charged at departure, the
  ``vehicles`` cost (1) on a vehicle's first departure, and request ``j`` becomes ``ASSIGNED``.
* On arrival at a customer, service starts at ``max(arrival, open)`` when time windows allow
  waiting (``waiting`` cost = the wait), otherwise at arrival. The request is ``SERVED`` at service
  start; the vehicle is idle again at service end.
* Arrival at the depot ends the route (single trip per vehicle in v0.1).
* Epochs: with ``sequential`` epochs one vehicle routes until it returns, then the next starts at
  the shift start; with ``on_event`` epochs vehicles act concurrently in time order (ties by index).
* A vehicle at the depot that has not left yet needs a decision only if some pending customer is
  feasible for it; otherwise it retires automatically. For an unlimited fleet this also stops the
  creation of new vehicles (they are identical), so the episode ends.
* At the end, every pending request becomes ``UNSERVED`` and costs 1 ``unserved``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from fleetgym.constraints.base import CONSTRAINT_CLASSES, Constraint
from fleetgym.errors import ConfigError, InfeasibleActionError, NotInScopeError
from fleetgym.eventlog import EventLog
from fleetgym.network.base import Network
from fleetgym.sim.decisions import Decision, EndRoute
from fleetgym.sim.events import Event, EventKind
from fleetgym.sim.info_state import DEPOT, InformationState, TimingRules
from fleetgym.sim.state import RequestStatus, SimState, VehicleState, VehicleStatus
from fleetgym.spec.binding import BoundProblem

__all__ = ["Epoch", "Simulator", "Transition"]


@dataclass(frozen=True)
class Epoch:
    """A decision epoch: ``vehicle`` must decide at ``time``."""

    time: float
    kind: Literal["vehicle_idle"]
    vehicle: int


@dataclass(frozen=True)
class Transition:
    """Result of applying a decision.

    Attributes
    ----------
    epoch
        The next decision epoch, or ``None`` if the episode ended.
    costs
        Costs realized since the previous epoch, by component.
    """

    epoch: Epoch | None
    costs: dict[str, float]

    @property
    def terminated(self) -> bool:
        """True if the episode ended."""
        return self.epoch is None


class Simulator:
    """Discrete-event simulator of one episode.

    Parameters
    ----------
    problem
        The bound problem.
    network
        Its network.
    log
        Event log to record into (a fresh one if omitted).

    Raises
    ------
    NotInScopeError
        For features of later milestones (uncertainty, dynamics, periodic epochs, fixed costs).
    ConfigError
        For unsupported combinations, such as an unlimited fleet with ``on_event`` epochs.
    """

    def __init__(
        self, problem: BoundProblem, network: Network, *, log: EventLog | None = None
    ) -> None:
        spec = problem.spec
        self.problem = problem
        self.network = network
        self.log = log if log is not None else EventLog()
        self._check_scope()
        self.mode: Literal["sequential", "on_event"] = (
            "sequential" if spec.decisions.epochs == "sequential" else "on_event"
        )
        self.unlimited = spec.fleet.sizing == "unlimited"
        tw = spec.constraints.items.get("time_windows")
        waiting = tw.model_dump()["waiting"] == "allowed" if tw is not None else True
        self.timing = TimingRules(use_time_windows=tw is not None, waiting_allowed=waiting)
        self.constraints: list[Constraint] = []
        for name in spec.constraints.names():
            if name not in CONSTRAINT_CLASSES:
                raise NotInScopeError(f"constraint {name!r} is not implemented in this version")
            self.constraints.append(CONSTRAINT_CLASSES.get(name)(problem, spec.constraints[name]))
        self.shift_start = float(spec.fleet.shift[0]) if spec.fleet.shift is not None else 0.0
        self.shift_end = float(spec.fleet.shift[1]) if spec.fleet.shift is not None else None
        self.state: SimState = self._initial_state()
        self.constraint_states: list[Any] = []
        self._info = InformationState(problem, network, self.state, self.timing)

    # ------------------------------------------------------------------ setup

    def _check_scope(self) -> None:
        spec = self.problem.spec
        if spec.dynamics is not None:
            raise NotInScopeError("dynamic requests are implemented in M6")
        if spec.uncertainty:
            raise NotInScopeError("uncertain parameters are implemented in M5")
        if spec.decisions.epochs not in ("sequential", "on_event"):
            raise NotInScopeError(f"{spec.decisions.epochs!r} epochs are implemented in M6")
        if spec.recourse.rules:
            raise NotInScopeError("recourse is implemented in M5")
        if spec.fleet.sizing == "unlimited":
            if len(spec.fleet.types) != 1:
                raise ConfigError(
                    "an unlimited fleet must have exactly one vehicle type",
                    source=spec.source,
                    key_path="fleet.types",
                )
            if spec.decisions.epochs != "sequential":
                raise ConfigError(
                    "an unlimited fleet needs `decisions.epochs: sequential` (with concurrent "
                    "epochs every vehicle would be idle at once); give `count` for a fixed fleet",
                    source=spec.source,
                    key_path="decisions.epochs",
                )
        if (
            spec.fleet.sizing != "unlimited"
            and "unserved" not in spec.objective.components()
            and "outsourcing" not in spec.constraints
        ):
            raise ConfigError(
                f"a {spec.fleet.sizing} fleet can leave requests unserved; give the objective an "
                "`unserved` weight (or use outsourcing)",
                source=spec.source,
                key_path="objective",
            )
        if any(vt.fixed_cost for vt in self.problem.vehicle_types):
            raise NotInScopeError(
                "vehicle fixed costs arrive with heterogeneous fleets (after v0.1)"
            )

    def _initial_state(self) -> SimState:
        instance = self.problem.instance
        n = instance.n_nodes
        status = np.full(n, RequestStatus.PENDING, dtype=np.int8)
        status[: instance.n_depots] = RequestStatus.NOT_A_REQUEST
        vehicles: list[VehicleState] = []
        if not self.unlimited:
            initial = (
                VehicleStatus.WAITING_TO_START if self.mode == "sequential" else VehicleStatus.IDLE
            )
            for type_index, vt in enumerate(self.problem.vehicle_types):
                assert vt.count is not None
                for _ in range(vt.count):
                    vehicles.append(self._new_vehicle(len(vehicles), type_index, initial))
        return SimState(
            time=self.shift_start,
            vehicles=vehicles,
            request_status=status,
            service_start=np.full(n, np.nan),
            served_by=np.full(n, -1, dtype=np.int64),
        )

    def _new_vehicle(self, index: int, type_index: int, status: VehicleStatus) -> VehicleState:
        capacity = self.problem.vehicle_types[type_index].capacity
        return VehicleState(
            index=index,
            type_index=type_index,
            capacity=float(capacity) if capacity is not None else float("inf"),
            status=status,
            node=DEPOT,
            time=self.shift_start,
        )

    # ------------------------------------------------------------------ public API

    @property
    def info(self) -> InformationState:
        """Read-only view of what is known now."""
        return self._info

    def start(self) -> Epoch | None:
        """Start the episode and run until the first decision epoch."""
        self.constraint_states = [c.init_state(self._info) for c in self.constraints]
        self.log.append("episode_start", time=self.state.time)
        if self.mode == "on_event":
            for v in self.state.vehicles:
                self._log_activation(v)
        return self._advance()

    def mask(self, vehicle: int) -> NDArray[np.bool_]:
        """Feasible next nodes for ``vehicle`` (length ``n_nodes``); depot = return or retire."""
        v = self.state.vehicles[vehicle]
        n = self.problem.instance.n_nodes
        mask = np.zeros(n, dtype=bool)
        candidates = np.flatnonzero(self.state.pending_mask()).astype(np.int64)
        if candidates.size:
            ok = self._customer_feasibility(vehicle, candidates)
            mask[candidates[ok]] = True
        if v.node != DEPOT:
            mask[DEPOT] = True
        else:
            mask[DEPOT] = (not self.unlimited) or not mask.any()
        return mask

    def apply(self, decision: Decision) -> Transition:
        """Apply a decision for the vehicle of the current epoch and run to the next epoch.

        Raises
        ------
        InfeasibleActionError
            If the decision is not for the current vehicle or violates a hard constraint.
        """
        state = self.state
        if state.terminated:
            raise InfeasibleActionError("the episode has ended")
        vehicle = decision.vehicle
        if vehicle != state.active_vehicle:
            raise InfeasibleActionError(
                f"vehicle {vehicle} has no pending decision (current: {state.active_vehicle})"
            )
        mask = self.mask(vehicle)
        v = state.vehicles[vehicle]
        if isinstance(decision, EndRoute):
            if v.node != DEPOT or v.used or not mask[DEPOT]:
                raise InfeasibleActionError(f"vehicle {vehicle} cannot retire here")
            self._retire(v)
        else:
            if not 0 <= decision.node < mask.size:
                raise InfeasibleActionError(f"node {decision.node} does not exist")
            if decision.node == DEPOT and v.node == DEPOT:
                raise InfeasibleActionError("use EndRoute for a vehicle that stays at the depot")
            if not mask[decision.node]:
                raise InfeasibleActionError(self._explain(vehicle, decision.node))
            self._dispatch(v, decision.node)
        state.active_vehicle = None
        epoch = self._advance()
        return Transition(epoch=epoch, costs=state.take_pending_costs())

    def snapshot(self) -> tuple[SimState, list[Any]]:
        """Deep copy of the full state (simulator and constraints)."""
        return copy.deepcopy(self.state), copy.deepcopy(self.constraint_states)

    def restore(self, snapshot: tuple[SimState, list[Any]]) -> None:
        """Restore a snapshot taken with `snapshot` (the snapshot itself is not shared)."""
        state, constraint_states = copy.deepcopy(snapshot)
        self.state = state
        self.constraint_states = constraint_states
        self._info = InformationState(self.problem, self.network, self.state, self.timing)

    # ------------------------------------------------------------------ feasibility

    def _customer_feasibility(
        self, vehicle: int, candidates: NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        ok = np.ones(candidates.shape, dtype=bool)
        for constraint, cstate in zip(self.constraints, self.constraint_states, strict=True):
            ok &= constraint.mask(cstate, self._info, vehicle, candidates)
        if self.shift_end is not None:
            _, _, end = self._info.projected(vehicle, candidates)
            ok &= end + self._info.travel_times[candidates, DEPOT] <= self.shift_end
        return ok

    def _explain(self, vehicle: int, node: int) -> str:
        status = RequestStatus(int(self.state.request_status[node]))
        if status != RequestStatus.PENDING:
            return f"node {node} cannot be visited: its request is {status.name.lower()}"
        nodes = np.array([node], dtype=np.int64)
        for constraint, cstate in zip(self.constraints, self.constraint_states, strict=True):
            if not constraint.mask(cstate, self._info, vehicle, nodes)[0]:
                return (
                    f"node {node} violates the {constraint.name} constraint for vehicle {vehicle}"
                )
        return f"node {node} is past the end of the shift for vehicle {vehicle}"

    # ------------------------------------------------------------------ transitions

    def _dispatch(self, v: VehicleState, node: int) -> None:
        state = self.state
        if v.route_start is None:
            v.route_start = v.time
            v.route = [DEPOT]
            state.add_cost("vehicles", 1.0)
            self.log.append("cost", time=v.time, vehicle=v.index, component="vehicles", amount=1.0)
        distance = float(self._info.distances[v.node, node])
        travel = float(self._info.travel_times[v.node, node])
        self.log.append(
            "depart", time=v.time, vehicle=v.index, from_node=v.node, node=node, value=distance
        )
        self._charge(v.time, "distance", distance, v.index)
        self._charge(v.time, "time", travel, v.index)
        v.distance += distance
        v.route.append(node)
        if node != DEPOT:
            state.request_status[node] = RequestStatus.ASSIGNED
        v.status = VehicleStatus.TRAVELING
        v.node = node
        v.time = v.time + travel
        state.queue.push(v.time, EventKind.VEHICLE_ARRIVAL, v.index, node)

    def _charge(
        self, time: float, component: str, amount: float, vehicle: int | None = None
    ) -> None:
        if amount == 0:
            return
        self.state.add_cost(component, amount)
        self.log.append("cost", time=time, vehicle=vehicle, component=component, amount=amount)

    def _retire(self, v: VehicleState) -> None:
        v.status = VehicleStatus.DONE
        self.log.append("vehicle_retire", time=v.time, vehicle=v.index)

    def _process(self, event: Event) -> None:
        state = self.state
        state.time = event.time
        v = state.vehicles[event.vehicle]
        if event.kind == EventKind.VEHICLE_ARRIVAL:
            self.log.append("arrive", time=event.time, vehicle=v.index, node=event.node)
            if event.node == DEPOT:
                assert v.route_start is not None
                duration = event.time - v.route_start
                self.log.append(
                    "route_end", time=event.time, vehicle=v.index, node=DEPOT, value=duration
                )
                v.status = VehicleStatus.DONE
                v.time = event.time
                return
            nodes = np.array([event.node], dtype=np.int64)
            start = float(self._info.service_start(nodes, np.array([event.time]))[0])
            self._charge(event.time, "waiting", start - event.time, v.index)
            v.status = VehicleStatus.SERVING
            state.queue.push(start, EventKind.SERVICE_START, v.index, event.node)
        elif event.kind == EventKind.SERVICE_START:
            node = event.node
            demand = float(self._info.demands[node])
            v.load += demand
            state.request_status[node] = RequestStatus.SERVED
            state.service_start[node] = event.time
            state.served_by[node] = v.index
            self.log.append(
                "service_start",
                time=event.time,
                vehicle=v.index,
                node=node,
                quantity=demand,
                load=v.load,
            )
            end = event.time + float(self._info.service_times[node])
            state.queue.push(end, EventKind.SERVICE_END, v.index, node)
        elif event.kind == EventKind.SERVICE_END:
            self.log.append("service_end", time=event.time, vehicle=v.index, node=event.node)
            v.status = VehicleStatus.IDLE
            v.time = event.time
        for i, constraint in enumerate(self.constraints):
            constraint.update(self.constraint_states[i], event)
            for component, amount in constraint.penalty(
                self.constraint_states[i], self._info, event
            ).items():
                self._charge(event.time, component, amount, v.index)

    def _advance(self) -> Epoch | None:
        state = self.state
        while True:
            vehicle = self._next_decision_vehicle()
            if vehicle is not None:
                v = state.vehicles[vehicle]
                state.active_vehicle = vehicle
                state.time = v.time
                state.epoch_count += 1
                return Epoch(time=v.time, kind="vehicle_idle", vehicle=vehicle)
            if state.queue:
                self._process(state.queue.pop())
                continue
            if self.mode == "sequential" and self._activate_next():
                continue
            self._terminate()
            return None

    def _next_decision_vehicle(self) -> int | None:
        idle = sorted(
            (v for v in self.state.vehicles if v.status == VehicleStatus.IDLE),
            key=lambda v: (v.time, v.index),
        )
        for v in idle:
            if v.node != DEPOT:
                return v.index
            if not self.state.pending_mask().any():
                self._retire(v)
                continue
            if not self.mask(v.index)[1:].any():
                self._retire(v)
                if self.unlimited:
                    self.state.fleet_exhausted = True
                continue
            return v.index
        return None

    def _activate_next(self) -> bool:
        state = self.state
        if not state.pending_mask().any() or state.fleet_exhausted:
            return False
        if self.unlimited:
            v = self._new_vehicle(len(state.vehicles), 0, VehicleStatus.IDLE)
            state.vehicles.append(v)
        else:
            waiting = [v for v in state.vehicles if v.status == VehicleStatus.WAITING_TO_START]
            if not waiting:
                return False
            v = waiting[0]
            v.status = VehicleStatus.IDLE
        v.time = self.shift_start
        state.time = self.shift_start
        self._log_activation(v)
        return True

    def _log_activation(self, v: VehicleState) -> None:
        self.log.append(
            "vehicle_activate", time=v.time, vehicle=v.index, node=DEPOT, value=v.capacity
        )

    def _terminate(self) -> None:
        state = self.state
        for node in np.flatnonzero(state.pending_mask()):
            state.request_status[node] = RequestStatus.UNSERVED
            self.log.append("unserved", node=int(node))
            self._charge(state.time, "unserved", 1.0)
        state.terminated = True
        state.active_vehicle = None
        self.log.append("episode_end", time=state.time)
