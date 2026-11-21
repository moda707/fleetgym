"""The `Simulator`: advances events and stops at decision epochs.

Semantics (ADRs 0009 and 0013)
------------------------------
* A vehicle starts idle at the depot at the shift start (``fleet.shift[0]``, else 0).
* ``Dispatch(v, j)`` departs immediately: distance and travel time are charged at departure, the
  ``vehicles`` cost (1) on a vehicle's first departure, and request ``j`` becomes ``ASSIGNED``.
* On arrival at a customer, service starts at ``max(arrival, open)`` when time windows allow
  waiting (``waiting`` cost = the wait), otherwise at arrival.
* At service start the vehicle delivers the customer's remaining (true) demand. If it does not fit
  in the free capacity, that is a **capacity failure**, handled by ``recourse.capacity_failure``
  (`fleetgym.sim.recourse`): serve what fits (or nothing), then restock at the depot and come back,
  outsource the rest, or ask the policy at a recourse epoch. Each service with a positive quantity
  takes the customer's service time. The request is ``SERVED`` once nothing is left to deliver.
* Arrival at the depot ends the route, except on a restock trip: the load is reset to 0 and the
  vehicle goes back to the customer it came from, or (preventive restock) waits for a decision at
  the depot; choosing the depot there ends the route.
* Epochs: with ``sequential`` epochs one vehicle routes until it returns, then the next starts at
  the shift start; with ``on_event`` epochs vehicles act concurrently in time order (ties by index).
* A vehicle at the depot needs a decision only if some pending customer is feasible for it;
  otherwise it retires (or, after a restock, ends its route) automatically. For an unlimited fleet
  this also stops the creation of new vehicles (they are identical), so the episode ends.
* Uncertain values come from the episode's `Scenario`; they are revealed to the agent per the
  spec's ``revealed`` setting (``reveal`` rows in the log).
* At the end, every pending request becomes ``UNSERVED`` and costs 1 ``unserved``.
* With the objective component ``served_demand``, each service records the quantity delivered
  (a revenue: its negative weight turns it into a reward when the demand is delivered).

Split recourse (ADR 0021)
-------------------------
* At a capacity failure the vehicle serves what fits; the customer is pending again with the demand
  left (``failure`` row with component ``split``), for any vehicle.
* A vehicle with no free capacity may only go to the depot. Every trip to the depot during the day
  restocks the vehicle and its route goes on; at the depot it chooses its next customer, or ends its
  day (with ``end_early: false`` only when no customer is reachable).
* The rule never forces travel, so it may be combined with route-duration and shift limits.

Dynamic requests with ``next_node`` (ADR 0015)
-----------------------------------------------
* Requests are released at their arrival time and become pending customers (accepted
  automatically); a cancellation removes a pending request; a vehicle driving to a cancelled request
  arrives, serves nothing, and decides again.
* ``Wait(v)``: the vehicle is asked again at the next event (a request arrives or cancels, a vehicle
  arrives or finishes a service) or at its wake-up time: the earlier of the end of the arrival
  window and its latest departure for the depot (depot closing time, shift end, route duration).
  Waiting is allowed before that time only, so it depends on known information only.
* Returning to the depot during the day reloads the vehicle (``restock``); at the depot it may leave
  again, wait, or end its route (``EndRoute``), which is allowed only while another vehicle is still
  working or when nothing is left to do.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from fleetgym.constraints.base import CONSTRAINT_CLASSES, Constraint
from fleetgym.errors import ConfigError, FleetGymError, InfeasibleActionError, NotInScopeError
from fleetgym.eventlog import EventLog
from fleetgym.network.base import Network
from fleetgym.sim.decisions import Decision, Dispatch, EndRoute, RecourseChoice, Wait
from fleetgym.sim.events import Event, EventKind
from fleetgym.sim.info_state import DEPOT, InformationState, TimingRules
from fleetgym.sim.recourse import RECOURSE_CLASSES, CapacityFailure, RecourseAction, RecourseRule
from fleetgym.sim.state import RequestStatus, SimState, VehicleState, VehicleStatus
from fleetgym.spec.binding import BoundProblem
from fleetgym.spec.models import Revelation
from fleetgym.uncertainty.requests import arrival_window
from fleetgym.uncertainty.revelation import AT_START, check_revelations
from fleetgym.uncertainty.sampling import deterministic_values, distributions, realize
from fleetgym.uncertainty.scenario import Scenario

__all__ = ["Epoch", "RecourseLoopError", "Simulator", "Transition"]

_TOL = 1e-9


class RecourseLoopError(FleetGymError):
    """A recourse rule would restock an empty vehicle, which repeats forever (a rule bug)."""


@dataclass(frozen=True)
class Epoch:
    """A decision epoch at ``time``.

    ``kind`` is ``"vehicle_idle"`` (``vehicle`` chooses its next move), ``"recourse"`` (``vehicle``
    resolves the capacity failure at ``node``) or ``"request"`` (decide on request ``node``; no
    vehicle).
    """

    time: float
    kind: Literal["vehicle_idle", "recourse", "request"]
    vehicle: int | None
    node: int | None = None


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
    vehicle_costs: dict[int, dict[str, float]] = field(default_factory=dict)

    @property
    def terminated(self) -> bool:
        """True if the episode ended."""
        return self.epoch is None


class Simulator:
    """Discrete-event simulator of one episode.

    ``option_constraints`` lists spec constraints that are decision options rather than feasibility
    rules (for example ``outsourcing`` with planned routing); they get no runtime class.

    Parameters
    ----------
    problem
        The bound problem.
    network
        Its network.
    log
        Event log to record into (a fresh one if omitted).
    scenario
        Realized values of the spec's uncertainty sources (required when the spec has any).

    Raises
    ------
    NotInScopeError
        For features of later milestones or on the recourse backlog.
    ConfigError
        For unsupported combinations, such as an unlimited fleet with ``on_event`` epochs.
    ScenarioMismatchError
        If ``scenario`` belongs to another spec or instance.
    """

    option_constraints: frozenset[str] = frozenset()

    def __init__(
        self,
        problem: BoundProblem,
        network: Network,
        *,
        log: EventLog | None = None,
        scenario: Scenario | None = None,
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
            if name in self.option_constraints:
                continue
            if name not in CONSTRAINT_CLASSES:
                raise NotInScopeError(f"constraint {name!r} is not implemented in this version")
            self.constraints.append(CONSTRAINT_CLASSES.get(name)(problem, spec.constraints[name]))
        self.capacity_on = "capacity" in spec.constraints
        self.shift_start = float(spec.fleet.shift[0]) if spec.fleet.shift is not None else 0.0
        self.shift_end = float(spec.fleet.shift[1]) if spec.fleet.shift is not None else None
        self.recourse: RecourseRule | None = None
        if "capacity_failure" in spec.recourse.rules:
            rule = spec.recourse.rules["capacity_failure"]
            self.recourse = RECOURSE_CLASSES.get(rule.name)(
                problem, rule.params, "recourse.capacity_failure"
            )
        self.reward_service = "served_demand" in spec.objective.components()
        self.scenario = scenario
        self._truth, self._forecasts = self._values(scenario)
        self.state: SimState = self._initial_state()
        self.constraint_states: list[Any] = []
        self._info = self._make_info()
        self.dynamic = spec.dynamics is not None
        # A trip to the depot during the day restocks and the route goes on (dynamic requests, and
        # recourse rules such as split); otherwise a trip to the depot ends the route.
        self._restock_trips = self.dynamic or (
            self.recourse is not None and self.recourse.restock_trips
        )
        self._init_requests(scenario)

    # ------------------------------------------------------------------ setup

    def _check_scope(self) -> None:
        spec = self.problem.spec
        if spec.dynamics is not None:
            if spec.uncertainty or spec.recourse.rules:
                raise NotInScopeError(
                    "uncertain parameters or recourse together with dynamic requests are after "
                    "v0.1 (ADR 0015)"
                )
            if spec.decisions.epochs != "on_event" or spec.fleet.sizing == "unlimited":
                raise ConfigError(
                    "dynamic requests need `decisions.epochs: on_event` and a fixed or limited "
                    "fleet",
                    source=spec.source,
                    key_path="decisions.epochs",
                )
        if spec.decisions.epochs not in ("sequential", "on_event"):
            raise NotInScopeError(f"{spec.decisions.epochs!r} epochs are implemented in M6")
        check_revelations(spec)
        tw = spec.constraints.items.get("time_windows")
        hard_time = (
            (tw is not None and not tw.model_dump()["soft"])
            or "route_duration" in spec.constraints
            or spec.fleet.shift is not None
        )
        late_demand = (
            "demand" in spec.uncertainty and spec.uncertainty["demand"].revealed not in AT_START
        )
        if late_demand and "capacity" in spec.constraints and not spec.recourse.rules:
            raise ConfigError(
                "demand revealed after dispatch can exceed the free capacity; give "
                "`recourse.capacity_failure` (for example return_to_depot)",
                source=spec.source,
                key_path="recourse",
            )
        rule_spec = spec.recourse.rules.get("capacity_failure")
        rule_cls = RECOURSE_CLASSES.get(rule_spec.name) if rule_spec is not None else None
        safe_with_limits = rule_cls is not None and rule_cls.no_forced_travel and tw is None
        if spec.recourse.rules and (tw is not None or hard_time) and not safe_with_limits:
            raise NotInScopeError(
                "recourse combined with time windows, route duration or shift limits is on the "
                "recourse backlog (docs/recourse.md); only rules that never force travel (split) "
                "work with route duration and shift limits"
            )
        if (
            "service_time" in spec.uncertainty
            and spec.uncertainty["service_time"].revealed not in AT_START
            and hard_time
        ):
            raise NotInScopeError(
                "stochastic service times with hard time windows, route duration or shift limits "
                "need time-window recourse (docs/recourse.md backlog)"
            )
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
            and not {"unserved", "served_demand"} & set(spec.objective.components())
            and "outsourcing" not in spec.constraints
        ):
            raise ConfigError(
                f"a {spec.fleet.sizing} fleet can leave requests unserved; give the objective an "
                "`unserved` weight, reward `served_demand`, or use outsourcing",
                source=spec.source,
                key_path="objective",
            )
        if any(vt.fixed_cost for vt in self.problem.vehicle_types):
            raise NotInScopeError(
                "vehicle fixed costs arrive with heterogeneous fleets (after v0.1)"
            )

    def _values(
        self, scenario: Scenario | None
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]]]:
        """True values (scenario or instance) and forecasts of the uncertain sources."""
        problem = self.problem
        spec = problem.spec
        truth = {s: deterministic_values(problem, s) for s in ("demand", "service_time")}
        forecasts: dict[str, NDArray[np.float64]] = {}
        if not spec.uncertainty:
            return truth, forecasts
        if scenario is None:
            raise ValueError("this spec has uncertain parameters: pass the episode's scenario")
        n = problem.instance.n_nodes
        scenario.check(
            spec_hash=problem.spec_hash,
            instance_hash=problem.instance_hash,
            sources=spec.uncertainty.keys(),
            n_nodes=n,
        )
        dists = distributions(problem)
        for name, uncertain in spec.uncertainty.items():
            truth[name] = np.asarray(scenario[name], dtype=float)
            if uncertain.forecast == "mean":
                forecasts[name] = realize(problem, name, dists[name].mean(n))
            else:
                forecasts[name] = np.zeros(n)
                forecasts[name][: problem.instance.n_depots] = deterministic_values(problem, name)[
                    : problem.instance.n_depots
                ]
        return truth, forecasts

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
        revealed: dict[str, NDArray[np.bool_]] = {}
        known: dict[str, NDArray[np.float64]] = {}
        for name, uncertain in self.problem.spec.uncertainty.items():
            at_start = uncertain.revealed in AT_START
            revealed[name] = np.full(n, at_start, dtype=bool)
            revealed[name][: instance.n_depots] = True
            known[name] = np.where(revealed[name], self._truth[name], np.nan)
        return SimState(
            time=self.shift_start,
            vehicles=vehicles,
            request_status=status,
            service_start=np.full(n, np.nan),
            served_by=np.full(n, -1, dtype=np.int64),
            remaining_demand=self._truth["demand"].copy(),
            outsourced=np.zeros(n),
            revealed=revealed,
            known_values=known,
        )

    def _make_info(self) -> InformationState:
        return InformationState(
            self.problem, self.network, self.state, self.timing, self._forecasts
        )

    def _init_requests(self, scenario: Scenario | None) -> None:
        """Release and cancellation times of requests (all at the start without dynamics)."""
        instance = self.problem.instance
        n = instance.n_nodes
        extra = instance.extra
        self.release = np.asarray(extra.get("release", np.full(n, self.shift_start)), dtype=float)
        self.cancel = np.asarray(extra.get("cancel", np.full(n, np.inf)), dtype=float)
        self.overflow = (
            np.asarray(scenario["request.overflow"], dtype=float)
            if scenario is not None and "request.overflow" in scenario.values
            else np.zeros(0)
        )
        self.window_end = (
            arrival_window(self.problem)[1] if self.problem.spec.dynamics is not None else -np.inf
        )
        if scenario is not None and "request.release" in scenario.values:
            base_hash = extra.get("base_instance_hash")
            if scenario.instance_hash != base_hash or scenario.spec_hash != self.problem.spec_hash:
                from fleetgym.errors import ScenarioMismatchError

                raise ScenarioMismatchError("the scenario belongs to another spec or instance")
        status = self.state.request_status
        late = ~(self.release <= self.shift_start + _TOL)
        status[late] = RequestStatus.NOT_RELEASED
        status[: instance.n_depots] = RequestStatus.NOT_A_REQUEST

    def _push_exogenous(self, now: float, *, strict: bool) -> None:
        """Schedule request arrivals, cancellations and overflow rejections after ``now``."""
        queue = self.state.queue
        status = self.state.request_status
        for node in range(self.problem.instance.n_depots, self.release.size):
            t = float(self.release[node])
            if status[node] == RequestStatus.NOT_RELEASED and np.isfinite(t) and t > now:
                queue.push(t, EventKind.REQUEST_ARRIVAL, -1, node)
            c = float(self.cancel[node])
            live = status[node] in (
                RequestStatus.NOT_RELEASED,
                RequestStatus.PENDING,
                RequestStatus.ASSIGNED,
            )
            if live and np.isfinite(c) and (c > now if strict else c >= now):
                queue.push(c, EventKind.CANCELLATION, -1, node)
        for k, t in enumerate(self.overflow):
            if t > now:
                queue.push(float(t), EventKind.OVERFLOW_ARRIVAL, -1, k)

    def reschedule_exogenous(self, now: float) -> None:
        """Replace future arrival, overflow and cancellation events with this scenario's.

        Used after restoring a snapshot into a simulator with a redrawn future (a fork). The demand
        still to deliver of requests not released yet is reset to the new scenario's.
        """
        import heapq

        exogenous = (EventKind.REQUEST_ARRIVAL, EventKind.OVERFLOW_ARRIVAL, EventKind.CANCELLATION)
        queue = self.state.queue
        queue.heap = [e for e in queue.heap if e.kind not in exogenous]
        heapq.heapify(queue.heap)
        future = self.state.request_status == RequestStatus.NOT_RELEASED
        self.state.remaining_demand[future] = self._truth["demand"][future]
        self._push_exogenous(now, strict=True)

    def _release(self, node: int, time: float) -> None:
        self.state.request_status[node] = RequestStatus.PENDING
        self.log.append("request", time=time, node=node)

    def latest_departure(self, vehicle: int) -> float:
        """Latest time ``vehicle`` can leave for the depot without breaking a hard limit."""
        v = self.state.vehicles[vehicle]
        travel = float(self._info.travel_times[v.node, DEPOT])
        limits = [np.inf]
        tw = self.problem.instance.time_windows
        tw_spec = self.problem.spec.constraints.items.get("time_windows")
        if tw is not None and tw_spec is not None and not tw_spec.model_dump()["soft"]:
            limits.append(float(tw[DEPOT, 1]) - travel)
        if self.shift_end is not None:
            limits.append(self.shift_end - travel)
        if "route_duration" in self.problem.spec.constraints and v.route_start is not None:
            limits.append(
                v.route_start + float(self.problem.value("constraints.route_duration.max")) - travel
            )
        return float(min(limits))

    def _wake_time(self, vehicle: int) -> float:
        return min(self.window_end, self.latest_departure(vehicle))

    def can_wait(self, vehicle: int) -> bool:
        """Whether ``vehicle`` may wait now (dynamic requests, before its wake-up time)."""
        if not self.dynamic:
            return False
        v = self.state.vehicles[vehicle]
        if v.awaiting_recourse is not None:
            return False
        return self.state.time < self._wake_time(vehicle) - _TOL

    def _work_left(self) -> bool:
        """Customers are pending, or requests can still arrive (known: the arrival window)."""
        return bool(self.state.pending_mask().any()) or self.state.time < self.window_end - _TOL

    def _can_end(self, v: VehicleState) -> bool:
        """With dynamic requests: another vehicle still works, or nothing is left to do."""
        if not self._work_left():
            return True
        return any(
            w.index != v.index and w.status != VehicleStatus.DONE for w in self.state.vehicles
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
        if self.dynamic:
            for node in range(self.problem.instance.n_depots, self.release.size):
                if self.release[node] <= self.shift_start + _TOL:
                    self._release(node, self.shift_start)
            self._push_exogenous(self.shift_start, strict=False)
        return self._advance()

    def mask(self, vehicle: int) -> NDArray[np.bool_]:
        """Feasible next nodes for ``vehicle`` (length ``n_nodes``); depot = return or end.

        At a recourse epoch, the mask has the depot (``return_to_depot``) and the failing customer
        (``outsource``) as allowed by the rule's options; see `recourse_options`.
        """
        v = self.state.vehicles[vehicle]
        n = self.problem.instance.n_nodes
        mask = np.zeros(n, dtype=bool)
        if v.awaiting_recourse is not None:
            options = self.recourse_options(vehicle)
            mask[DEPOT] = "return_to_depot" in options
            mask[v.awaiting_recourse] = "outsource" in options
            return mask
        candidates = np.flatnonzero(self.state.pending_mask()).astype(np.int64)
        rule = self.recourse
        if v.node != DEPOT and rule is not None and rule.must_restock(vehicle, self._info):
            candidates = candidates[:0]
        if candidates.size:
            ok = self._customer_feasibility(vehicle, candidates)
            mask[candidates[ok]] = True
        if v.node != DEPOT:
            mask[DEPOT] = True
        elif self.dynamic:
            mask[DEPOT] = self._can_end(v)
        elif v.used:
            mask[DEPOT] = True
        else:
            mask[DEPOT] = (not self.unlimited) or not mask.any()
        if v.node == DEPOT and mask[DEPOT] and self.recourse is not None:
            mask[DEPOT] = self.recourse.may_end(vehicle, self._info, bool(mask[1:].any()))
        return mask

    def recourse_options(self, vehicle: int) -> tuple[RecourseAction, ...]:
        """Recourse actions allowed for ``vehicle`` at its recourse epoch."""
        if self.state.vehicles[vehicle].awaiting_recourse is None or self.recourse is None:
            return ()
        return self.recourse.options()

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
        if not isinstance(decision, (Dispatch, EndRoute, RecourseChoice, Wait)):
            raise InfeasibleActionError(
                f"{type(decision).__name__} decides on a request; it needs the accept_reject or "
                "assign interface"
            )
        vehicle = decision.vehicle
        if vehicle != state.active_vehicle:
            raise InfeasibleActionError(
                f"vehicle {vehicle} has no pending decision (current: {state.active_vehicle})"
            )
        self._execute(decision)
        state.active_vehicle = None
        return self._transition()

    def apply_many(self, decisions: Sequence[Decision]) -> Transition:
        """Apply simultaneous decisions of the vehicles in `ready_epochs`, then advance.

        Decisions are executed in the given order, each against the state left by the previous ones
        (multi-agent Parallel API; conflicts are resolved by the caller).

        Raises
        ------
        InfeasibleActionError
            If a decision is for a vehicle without a decision now, or is infeasible.
        """
        state = self.state
        if state.terminated:
            raise InfeasibleActionError("the episode has ended")
        ready = {e.vehicle for e in self.ready_epochs()}
        for decision in decisions:
            if not isinstance(decision, (Dispatch, EndRoute, RecourseChoice, Wait)):
                raise InfeasibleActionError(f"{type(decision).__name__} is not a vehicle decision")
            if decision.vehicle not in ready:
                raise InfeasibleActionError(f"vehicle {decision.vehicle} has no decision now")
            self._execute(decision)
        state.active_vehicle = None
        return self._transition()

    def ready_epochs(self) -> list[Epoch]:
        """Every vehicle that needs a decision at the current time (in index order).

        `apply`'s epoch is one of them; the Parallel multi-agent API asks all of them at once.
        """
        state = self.state
        out = []
        for v in sorted(state.vehicles, key=lambda v: v.index):
            if v.status != VehicleStatus.IDLE or v.waiting or v.time > state.time + _TOL:
                continue
            if v.awaiting_recourse is not None:
                out.append(Epoch(state.time, "recourse", v.index, v.awaiting_recourse))
            elif v.node != DEPOT or self._depot_vehicle_decides(v):
                out.append(Epoch(state.time, "vehicle_idle", v.index))
        return out

    def _depot_vehicle_decides(self, v: VehicleState) -> bool:
        if self.dynamic:
            return self._work_left() and bool(self.mask(v.index).any() or self.can_wait(v.index))
        return bool(self.state.pending_mask().any()) and bool(self.mask(v.index)[1:].any())

    def _transition(self) -> Transition:
        epoch = self._advance()
        return Transition(
            epoch=epoch,
            costs=self.state.take_pending_costs(),
            vehicle_costs=self.state.take_pending_vehicle_costs(),
        )

    def _execute(self, decision: Decision) -> None:
        """Validate and carry out one vehicle decision (no advancing)."""
        assert isinstance(decision, (Dispatch, EndRoute, RecourseChoice, Wait))
        state = self.state
        vehicle = decision.vehicle
        v = state.vehicles[vehicle]
        if v.awaiting_recourse is not None:
            if not isinstance(decision, RecourseChoice):
                raise InfeasibleActionError(
                    f"vehicle {vehicle} must resolve its capacity failure at node "
                    f"{v.awaiting_recourse} (a RecourseChoice)"
                )
            if decision.action not in self.recourse_options(vehicle):
                raise InfeasibleActionError(
                    f"recourse {decision.action!r} is not allowed (options: "
                    f"{', '.join(self.recourse_options(vehicle))})"
                )
            node = v.awaiting_recourse
            v.awaiting_recourse = None
            self._resolve(v, node, decision.action)
        elif isinstance(decision, RecourseChoice):
            raise InfeasibleActionError(f"vehicle {vehicle} has no capacity failure to resolve")
        elif isinstance(decision, Wait):
            if not self.can_wait(vehicle):
                raise InfeasibleActionError(f"vehicle {vehicle} cannot wait now")
            v.time = max(v.time, state.time)
            v.waiting = True
            state.queue.push(self._wake_time(vehicle), EventKind.TIMER, v.index, v.node)
        else:
            v.time = max(v.time, state.time)
            mask = self.mask(vehicle)
            if isinstance(decision, EndRoute):
                if v.node != DEPOT or not mask[DEPOT]:
                    raise InfeasibleActionError(
                        f"vehicle {vehicle} cannot retire or end its route here"
                    )
                if v.used:
                    self._end_route(v, v.time)
                else:
                    self._retire(v)
            elif decision.node == DEPOT and v.node != DEPOT and self._restock_trips:
                v.restocking = True  # back to the depot to reload; the route goes on
                v.recourse_target = None
                self._travel(v, DEPOT)
            else:
                if not 0 <= decision.node < mask.size:
                    raise InfeasibleActionError(f"node {decision.node} does not exist")
                if decision.node == DEPOT and v.node == DEPOT:
                    raise InfeasibleActionError(
                        "use EndRoute for a vehicle that stays at the depot"
                    )
                if not mask[decision.node]:
                    raise InfeasibleActionError(self._explain(vehicle, decision.node))
                self._dispatch(v, decision.node)

    def snapshot(self) -> tuple[SimState, list[Any]]:
        """Deep copy of the full state (simulator and constraints)."""
        return copy.deepcopy(self.state), copy.deepcopy(self.constraint_states)

    def restore(self, snapshot: tuple[SimState, list[Any]]) -> None:
        """Restore a snapshot taken with `snapshot` (the snapshot itself is not shared)."""
        state, constraint_states = copy.deepcopy(snapshot)
        self.state = state
        self.constraint_states = constraint_states
        self._info = self._make_info()

    def revealed_masks(self) -> dict[str, NDArray[np.bool_]]:
        """Per uncertain source, a copy of the mask of revealed nodes."""
        return {name: mask.copy() for name, mask in self.state.revealed.items()}

    def replace_scenario(self, scenario: Scenario) -> None:
        """Continue the episode under another scenario that agrees on every revealed value.

        Used by `FleetGymEnv.fork`. Unrevealed demands are reset to the new scenario's values.
        Events already scheduled (a service in progress) keep their times.

        Raises
        ------
        ValueError
            If the new scenario contradicts a revealed value.
        """
        for name, revealed in self.state.revealed.items():
            if not np.array_equal(scenario[name][revealed], self._truth[name][revealed]):
                raise ValueError(f"the new scenario changes revealed values of {name!r}")
        self.scenario = scenario
        self._truth, self._forecasts = self._values(scenario)
        if "demand" in self.state.revealed:
            hidden = ~self.state.revealed["demand"]
            self.state.remaining_demand[hidden] = self._truth["demand"][hidden]
        self._info = self._make_info()

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

    # ------------------------------------------------------------------ revelation

    def _reveal(self, node: int, when: Revelation, vehicle: int | None, time: float) -> None:
        state = self.state
        for name, uncertain in self.problem.spec.uncertainty.items():
            if uncertain.revealed != when or state.revealed[name][node]:
                continue
            value = float(self._truth[name][node])
            state.revealed[name][node] = True
            state.known_values[name][node] = value
            self.log.append(
                "reveal", time=time, vehicle=vehicle, node=node, component=name, value=value
            )

    # ------------------------------------------------------------------ transitions

    def _dispatch(self, v: VehicleState, node: int) -> None:
        if v.route_start is None:
            v.route_start = v.time
            v.route = [DEPOT]
            self.state.add_cost("vehicles", 1.0, v.index)
            self.log.append("cost", time=v.time, vehicle=v.index, component="vehicles", amount=1.0)
        if node != DEPOT:
            self.state.request_status[node] = RequestStatus.ASSIGNED
            self._reveal(node, Revelation.ON_DEPARTURE, v.index, v.time)
        self._travel(v, node)

    def _travel(self, v: VehicleState, node: int) -> None:
        distance = float(self._info.distances[v.node, node])
        travel = float(self._info.travel_times[v.node, node])
        self.log.append(
            "depart", time=v.time, vehicle=v.index, from_node=v.node, node=node, value=distance
        )
        self._charge(v.time, "distance", distance, v.index)
        self._charge(v.time, "time", travel, v.index)
        v.distance += distance
        v.route.append(node)
        v.status = VehicleStatus.TRAVELING
        v.node = node
        v.time = v.time + travel
        self.state.queue.push(v.time, EventKind.VEHICLE_ARRIVAL, v.index, node)

    def _charge(
        self,
        time: float,
        component: str,
        amount: float,
        vehicle: int | None = None,
        node: int | None = None,
    ) -> None:
        if amount == 0:
            return
        self.state.add_cost(component, amount, vehicle)
        where = {"node": node} if node is not None else {}
        self.log.append(
            "cost", time=time, vehicle=vehicle, component=component, amount=amount, **where
        )

    def _retire(self, v: VehicleState) -> None:
        v.status = VehicleStatus.DONE
        self.log.append("vehicle_retire", time=v.time, vehicle=v.index)

    def _end_route(self, v: VehicleState, time: float) -> None:
        assert v.route_start is not None
        self.log.append(
            "route_end", time=time, vehicle=v.index, node=DEPOT, value=time - v.route_start
        )
        v.status = VehicleStatus.DONE
        v.time = time

    def _process(self, event: Event) -> None:
        state = self.state
        state.time = event.time
        if self.dynamic:
            self._process_dynamic(event)
            return
        self._process_vehicle(event)

    def _process_dynamic(self, event: Event) -> None:
        state = self.state
        kind = event.kind
        if kind == EventKind.TIMER:
            v = state.vehicles[event.vehicle]
            if v.waiting and abs(self._wake_time(v.index) - event.time) <= _TOL * max(
                1.0, event.time
            ):
                v.waiting = False
            return
        for w in state.vehicles:  # any other event wakes the waiting vehicles
            w.waiting = False
        if kind == EventKind.REQUEST_ARRIVAL:
            self._release(event.node, event.time)
        elif kind == EventKind.OVERFLOW_ARRIVAL:
            self.log.append("reject", time=event.time, component="overflow")
            self._charge(event.time, "rejected", 1.0)
        elif kind == EventKind.CANCELLATION:
            node = event.node
            status = RequestStatus(int(state.request_status[node]))
            serving = any(
                w.node == node and w.status == VehicleStatus.SERVING for w in state.vehicles
            )
            if status in (RequestStatus.PENDING, RequestStatus.ASSIGNED) and not serving:
                state.request_status[node] = RequestStatus.CANCELLED
                self.log.append("cancel", time=event.time, node=node)
        elif (
            kind == EventKind.VEHICLE_ARRIVAL
            and event.node != DEPOT
            and state.request_status[event.node] == RequestStatus.CANCELLED
        ):
            v = state.vehicles[event.vehicle]
            self.log.append("arrive", time=event.time, vehicle=v.index, node=event.node)
            v.time = event.time
            v.status = VehicleStatus.IDLE
        else:
            self._process_vehicle(event)

    def _process_vehicle(self, event: Event) -> None:
        state = self.state
        v = state.vehicles[event.vehicle]
        if event.kind == EventKind.VEHICLE_ARRIVAL:
            self.log.append("arrive", time=event.time, vehicle=v.index, node=event.node)
            v.time = event.time
            if event.node == DEPOT:
                if v.restocking:
                    self._restock(v)
                else:
                    self._end_route(v, event.time)
                return
            self._reveal(event.node, Revelation.ON_ARRIVAL, v.index, event.time)
            nodes = np.array([event.node], dtype=np.int64)
            start = float(self._info.service_start(nodes, np.array([event.time]))[0])
            self._charge(event.time, "waiting", start - event.time, v.index)
            v.status = VehicleStatus.SERVING
            state.queue.push(start, EventKind.SERVICE_START, v.index, event.node)
        elif event.kind == EventKind.SERVICE_START:
            self._service_start(v, event.node, event.time)
        elif event.kind == EventKind.SERVICE_END:
            self.log.append("service_end", time=event.time, vehicle=v.index, node=event.node)
            v.time = event.time
            self._reveal(event.node, Revelation.ON_COMPLETION, v.index, event.time)
            self._after_service(v, event.node, event.time)
        self._constraint_hooks(v, event)

    def _constraint_hooks(self, v: VehicleState, event: Event) -> None:
        for i, constraint in enumerate(self.constraints):
            constraint.update(self.constraint_states[i], event)
            for component, amount in constraint.penalty(
                self.constraint_states[i], self._info, event
            ).items():
                self._charge(event.time, component, amount, v.index)

    def _service_start(self, v: VehicleState, node: int, time: float) -> None:
        state = self.state
        remaining = float(state.remaining_demand[node])
        free = v.capacity - v.load if self.capacity_on else float("inf")
        quantity = remaining
        if remaining > free + _TOL:
            if self.recourse is None:  # pragma: no cover - excluded by _check_scope and masks
                raise InfeasibleActionError(f"capacity failure at node {node} without recourse")
            failure = CapacityFailure(v.index, node, time, remaining, free, v.capacity)
            empty = v.load <= _TOL
            quantity = free if empty or self.recourse.serve_first(failure, self._info) else 0.0
        if quantity <= 0 and remaining > 0:
            # Nothing is served now: resolve the failure at once.
            self._after_service(v, node, time)
            return
        v.load += quantity
        state.remaining_demand[node] = remaining - quantity
        if np.isnan(state.service_start[node]):
            state.service_start[node] = time
            state.served_by[node] = v.index
        self.log.append(
            "service_start", time=time, vehicle=v.index, node=node, quantity=quantity, load=v.load
        )
        if self.reward_service:
            self._charge(time, "served_demand", quantity, v.index, node)
        end = time + float(self._truth["service_time"][node])
        state.queue.push(end, EventKind.SERVICE_END, v.index, node)

    def _after_service(self, v: VehicleState, node: int, time: float) -> None:
        state = self.state
        v.time = time
        if state.remaining_demand[node] > _TOL:
            assert self.recourse is not None
            failure = CapacityFailure(
                v.index,
                node,
                time,
                float(state.remaining_demand[node]),
                v.capacity - v.load,
                v.capacity,
            )
            action = self.recourse.on_failure(failure, self._info)
            if action is None:
                v.status = VehicleStatus.IDLE
                v.awaiting_recourse = node
                return
            self._resolve(v, node, action)
            return
        state.remaining_demand[node] = 0.0
        self._complete(v, node)

    def _complete(self, v: VehicleState, node: int) -> None:
        """Customer ``node`` is fully handled; ``v`` is free there."""
        self.state.request_status[node] = RequestStatus.SERVED
        v.status = VehicleStatus.IDLE
        if self.recourse is not None and self.recourse.after_service(v.index, self._info):
            v.restocking = True
            v.recourse_target = None
            self._travel(v, DEPOT)

    def _resolve(self, v: VehicleState, node: int, action: RecourseAction) -> None:
        state = self.state
        remaining = float(state.remaining_demand[node])
        self.log.append(
            "failure",
            time=v.time,
            vehicle=v.index,
            node=node,
            quantity=remaining,
            load=v.load,
            component=action,
        )
        if action == "split":
            state.request_status[node] = RequestStatus.PENDING  # open again, for any vehicle
            v.status = VehicleStatus.IDLE
            return
        if action == "return_to_depot":
            if v.load <= _TOL:
                raise RecourseLoopError(
                    f"vehicle {v.index} would restock while empty at node {node}: the recourse "
                    "rule must serve what fits when the vehicle is empty"
                )
            v.restocking = True
            v.recourse_target = node
            self._travel(v, DEPOT)
            return
        assert self.recourse is not None
        cost = self.recourse.outsourcing_cost(node, remaining)
        self.log.append(
            "outsource", time=v.time, vehicle=v.index, node=node, quantity=remaining, amount=cost
        )
        self._charge(v.time, "outsourcing", cost, v.index, node)
        state.outsourced[node] += remaining
        state.remaining_demand[node] = 0.0
        if np.isnan(state.service_start[node]):
            state.service_start[node] = v.time
            state.served_by[node] = v.index
        self._complete(v, node)

    def _restock(self, v: VehicleState) -> None:
        v.restocking = False
        v.load = 0.0
        self.log.append("restock", time=v.time, vehicle=v.index, node=DEPOT)
        target = v.recourse_target
        v.recourse_target = None
        if target is not None:
            self._travel(v, target)
        else:
            v.status = VehicleStatus.IDLE

    def _advance(self) -> Epoch | None:
        state = self.state
        while True:
            epoch = self._next_epoch()
            if epoch is not None:
                state.active_vehicle = epoch.vehicle
                state.time = epoch.time
                state.epoch_count += 1
                return epoch
            if state.queue:
                self._process(state.queue.pop())
                continue
            if self.mode == "sequential" and self._activate_next():
                continue
            self._terminate()
            return None

    def _next_epoch(self) -> Epoch | None:
        idle = sorted(
            (v for v in self.state.vehicles if v.status == VehicleStatus.IDLE and not v.waiting),
            key=lambda v: (v.time, v.index),
        )
        for v in idle:
            if v.awaiting_recourse is not None:
                return Epoch(v.time, "recourse", v.index, v.awaiting_recourse)
            if self.dynamic:
                v.time = max(v.time, self.state.time)  # idle until now: masks use the present
            if v.node != DEPOT:
                return Epoch(v.time, "vehicle_idle", v.index)
            if self.dynamic:
                if self._work_left() and (self.mask(v.index).any() or self.can_wait(v.index)):
                    return Epoch(v.time, "vehicle_idle", v.index)
                if v.used:
                    self._end_route(v, v.time)
                else:
                    self._retire(v)
                continue
            if not self.state.pending_mask().any() or not self.mask(v.index)[1:].any():
                if self.unlimited and self.state.pending_mask().any():
                    self.state.fleet_exhausted = True
                if v.used:
                    self._end_route(v, v.time)
                else:
                    self._retire(v)
                continue
            return Epoch(v.time, "vehicle_idle", v.index)
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
