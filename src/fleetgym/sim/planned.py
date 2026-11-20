"""`PlannedRoutingSimulator`: requests arrive during the day; a routing rule drives the vehicles.

Used with the ``accept_reject`` and ``assign`` interfaces (ADR 0014). The agent decides on each
request when it is released (a ``request`` epoch); accepted requests are inserted into a vehicle's
plan by ``decisions.routing`` (`fleetgym.sim.routing_rules`), and vehicles follow their plans:

* drive first: an idle vehicle with a non-empty plan leaves at once for its next stop (its
  commitment); a vehicle never changes its committed stop;
* an idle vehicle with an empty plan waits where it is (plan Q26), until a request is inserted or
  the day is over (no request can arrive any more), when it returns to the depot (or retires if it
  never left). A vehicle waiting away from the depot leaves at the latest when returning later would
  break the depot's closing time, the shift end or the route duration; its route then ends;
* vehicles driving home are not offered new requests;
* requests known at the start are decided at the start, in node order, before any vehicle moves;
* a cancelled request is dropped from its vehicle's plan; a vehicle already driving to it arrives,
  serves nothing and continues;
* requests beyond ``max_requests`` with ``overflow: reject`` are rejected automatically when they
  arrive (``reject`` row with ``component: overflow``, one ``rejected`` cost each).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fleetgym.errors import ConfigError, InfeasibleActionError, NotInScopeError
from fleetgym.eventlog import EventLog
from fleetgym.network.base import Network
from fleetgym.sim.decisions import Accept, Assign, Decision, OutsourceRequest, Reject
from fleetgym.sim.engine import Epoch, Simulator, Transition
from fleetgym.sim.events import Event, EventKind
from fleetgym.sim.info_state import DEPOT
from fleetgym.sim.routing_rules import Anchor, Insertion, RouteRules, best_insertion
from fleetgym.sim.state import RequestStatus, SimState, VehicleState, VehicleStatus
from fleetgym.spec.binding import BoundProblem
from fleetgym.uncertainty.scenario import Scenario

__all__ = ["PlannedRoutingSimulator"]

_TOL = 1e-9


class PlannedRoutingSimulator(Simulator):
    """Simulator for request decisions with built-in routing.

    The ``outsourcing`` constraint is an option here (action ``outsource``), not a feasibility rule.

    Parameters
    ----------
    problem
        The problem bound to the episode instance (see
        `fleetgym.uncertainty.scenario.episode_instance`).
    network
        Its network.
    log
        Event log to record into.
    scenario
        The episode's scenario (release, cancellation and overflow times).
    """

    option_constraints = frozenset({"outsourcing"})

    def __init__(
        self,
        problem: BoundProblem,
        network: Network,
        *,
        log: EventLog | None = None,
        scenario: Scenario | None = None,
    ) -> None:
        super().__init__(problem, network, log=log, scenario=scenario)
        instance = problem.instance
        n = instance.n_nodes
        extra = instance.extra
        self.release = np.asarray(extra.get("release", np.full(n, self.shift_start)), dtype=float)
        if "release" not in extra:
            self.release[: instance.n_depots] = self.shift_start
        self.cancel = np.asarray(extra.get("cancel", np.full(n, np.inf)), dtype=float)
        self.overflow = (
            np.asarray(scenario["request.overflow"], dtype=float)
            if scenario is not None and "request.overflow" in scenario.values
            else np.zeros(0)
        )
        if scenario is not None and "request.release" in scenario.values:
            base_hash = extra.get("base_instance_hash")
            if scenario.instance_hash != base_hash or scenario.spec_hash != problem.spec_hash:
                from fleetgym.errors import ScenarioMismatchError

                raise ScenarioMismatchError("the scenario belongs to another spec or instance")
        spec = problem.spec
        tw = spec.constraints.items.get("time_windows")
        duration = (
            float(problem.value("constraints.route_duration.max"))
            if "route_duration" in spec.constraints
            else None
        )
        self.rules = RouteRules(
            distances=self._info.distances,
            travel_times=self._info.travel_times,
            demands=self._truth["demand"],
            service_times=self._truth["service_time"],
            time_windows=instance.time_windows if tw is not None else None,
            hard_windows=tw is not None and not tw.model_dump()["soft"],
            capacity_on=self.capacity_on,
            max_duration=duration,
            shift_end=self.shift_end,
        )
        self.outsourcing = spec.constraints.items.get("outsourcing")
        self.state.request_status[np.isinf(self.release)] = RequestStatus.NOT_RELEASED
        late = (self.release > self.shift_start + _TOL) & np.isfinite(self.release)
        self.state.request_status[late] = RequestStatus.NOT_RELEASED
        self.state.request_status[: instance.n_depots] = RequestStatus.NOT_A_REQUEST
        self._decision_queue: list[int] = []
        self._timers: dict[int, float] = {}

    # ------------------------------------------------------------------ scope

    def _check_scope(self) -> None:
        spec = self.problem.spec
        if spec.uncertainty:
            raise NotInScopeError(
                "uncertain parameters with accept_reject/assign are after v0.1 (ADR 0014)"
            )
        if spec.recourse.rules:
            raise NotInScopeError("recourse needs the next_node interface in this version")
        if spec.decisions.epochs != "on_event":
            raise ConfigError(
                f"interface {spec.decisions.interface!r} needs `decisions.epochs: on_event`",
                source=spec.source,
                key_path="decisions.epochs",
            )
        if spec.fleet.sizing == "unlimited":
            raise ConfigError(
                f"interface {spec.decisions.interface!r} needs a fixed or limited fleet",
                source=spec.source,
                key_path="fleet.sizing",
            )
        tw = spec.constraints.items.get("time_windows")
        if tw is not None and tw.model_dump()["waiting"] != "allowed":
            raise NotInScopeError("planned routing needs `time_windows.waiting: allowed`")
        if "rejected" not in spec.objective.components() and "outsourcing" not in spec.constraints:
            raise ConfigError(
                "requests can be rejected; give the objective a `rejected` weight",
                source=spec.source,
                key_path="objective",
            )
        if any(vt.fixed_cost for vt in self.problem.vehicle_types):
            raise NotInScopeError("vehicle fixed costs arrive with heterogeneous fleets")

    def _initial_state(self) -> SimState:
        state = super()._initial_state()
        for v in state.vehicles:
            v.status = VehicleStatus.IDLE
        return state

    # ------------------------------------------------------------------ public API

    def start(self) -> Epoch | None:
        """Start the episode, release the initial requests and run to the first epoch."""
        self.constraint_states = [c.init_state(self._info) for c in self.constraints]
        self.log.append("episode_start", time=self.state.time)
        for v in self.state.vehicles:
            self._log_activation(v)
        queue = self.state.queue
        depots = self.problem.instance.n_depots
        for node in range(depots, self.release.size):
            t = float(self.release[node])
            if np.isfinite(t) and t > self.shift_start + _TOL:
                queue.push(t, EventKind.REQUEST_ARRIVAL, -1, node)
            elif np.isfinite(t):
                self._release(node, t)
            if np.isfinite(self.cancel[node]):
                queue.push(float(self.cancel[node]), EventKind.CANCELLATION, -1, node)
        for k, t in enumerate(self.overflow):
            queue.push(float(t), EventKind.OVERFLOW_ARRIVAL, -1, k)
        return self._advance()

    def accept_options(self, node: int) -> tuple[bool, bool]:
        """Whether request ``node`` can be accepted (a feasible insertion exists) and outsourced."""
        return self.insertion(node) is not None, self.outsourcing is not None

    def insertion(self, node: int, vehicle: int | None = None) -> Insertion | None:
        """The routing rule's insertion of ``node`` (into ``vehicle`` only, if given)."""
        anchors = [self._anchor(v) for v in self.state.vehicles]
        plans = [v.plan for v in self.state.vehicles]
        active = [
            v.index
            for v in self.state.vehicles
            if v.status != VehicleStatus.DONE
            and not (v.status == VehicleStatus.TRAVELING and v.node == DEPOT)
        ]
        vehicles = active if vehicle is None else [vehicle] if vehicle in active else []
        return best_insertion(self.rules, anchors, plans, node, vehicles)

    def mask(self, vehicle: int) -> NDArray[np.bool_]:
        """Not used with planned routing (decisions are about requests)."""
        raise InfeasibleActionError("planned routing has no vehicle decisions")

    def apply(self, decision: Decision) -> Transition:
        """Apply a request decision and run to the next epoch.

        Raises
        ------
        InfeasibleActionError
            If the decision is not about the current request or is not allowed.
        """
        state = self.state
        if state.terminated:
            raise InfeasibleActionError("the episode has ended")
        if not isinstance(decision, (Accept, Assign, Reject, OutsourceRequest)):
            raise InfeasibleActionError(
                f"{type(decision).__name__} is a vehicle decision; this simulator decides on "
                "requests"
            )
        if not self._decision_queue or decision.node != self._decision_queue[0]:
            current = self._decision_queue[0] if self._decision_queue else None
            raise InfeasibleActionError(
                f"request {decision.node} has no pending decision (current: {current})"
            )
        node = self._decision_queue.pop(0)
        now = state.time
        if isinstance(decision, Reject):
            self._reject(node, now, "decision")
        elif isinstance(decision, OutsourceRequest):
            if self.outsourcing is None:
                raise InfeasibleActionError("this spec has no outsourcing")
            cost = self._outsourcing_cost(node)
            state.request_status[node] = RequestStatus.OUTSOURCED
            quantity = float(self._truth["demand"][node])
            self.log.append(
                "outsource",
                time=now,
                node=node,
                quantity=quantity,
                amount=cost,
                component="request",
            )
            self._charge(now, "outsourcing", cost, node=node)
        else:
            vehicle = decision.vehicle if isinstance(decision, Assign) else None
            if vehicle is not None and not 0 <= vehicle < len(state.vehicles):
                raise InfeasibleActionError(f"vehicle {vehicle} does not exist")
            insertion = self.insertion(node, vehicle)
            if insertion is None:
                where = "" if vehicle is None else f" into vehicle {vehicle}"
                raise InfeasibleActionError(f"request {node} has no feasible insertion{where}")
            v = state.vehicles[insertion.vehicle]
            v.plan.insert(insertion.position, node)
            state.request_status[node] = RequestStatus.ASSIGNED
            self.log.append(
                "accept",
                time=now,
                node=node,
                vehicle=insertion.vehicle,
                value=insertion.cost,
                action=insertion.position,
            )
        state.active_vehicle = None
        epoch = self._advance()
        return Transition(epoch=epoch, costs=state.take_pending_costs())

    # ------------------------------------------------------------------ internals

    def _outsourcing_cost(self, node: int) -> float:
        value = np.asarray(self.problem.value("constraints.outsourcing.value"), dtype=float)
        if value.ndim == 0:
            return float(value)
        if node >= value.size:
            raise ConfigError(
                "outsourcing costs from the instance cover only its own customers; give a number "
                "for dynamic requests",
                source=self.problem.spec.source,
                key_path="constraints.outsourcing.value",
            )
        return float(value[node])

    def _anchor(self, v: VehicleState) -> Anchor:
        now = self.state.time
        if v.status in (VehicleStatus.TRAVELING, VehicleStatus.SERVING):
            node = v.node
            arrival = v.time  # arrival time (traveling) or arrival at the node (serving)
            load = v.load
            served = self.state.request_status[node] == RequestStatus.SERVED
            if v.status == VehicleStatus.TRAVELING or not served:
                cancelled = self.state.request_status[node] == RequestStatus.CANCELLED
                if node != DEPOT and not cancelled:
                    load += float(self._truth["demand"][node])
            end = self._projected_end(v, node, arrival)
            return Anchor(node, end, load, v.route_start, v.capacity)
        return Anchor(v.node, max(v.time, now), v.load, v.route_start, v.capacity)

    def _projected_end(self, v: VehicleState, node: int, arrival: float) -> float:
        """When ``v`` is free at its committed ``node`` (arrival or service in progress)."""
        if node == DEPOT or self.state.request_status[node] == RequestStatus.CANCELLED:
            return arrival
        if v.status == VehicleStatus.SERVING:
            pending = [e for e in self.state.queue.heap if e.vehicle == v.index]
            if pending:
                last = max(pending, key=lambda e: (e.time, e.priority))
                if last.kind == EventKind.SERVICE_END:
                    return last.time
                if last.kind == EventKind.SERVICE_START:
                    return last.time + float(self._truth["service_time"][node])
        start = arrival
        tw = self.rules.time_windows
        if tw is not None:
            start = max(start, float(tw[node, 0]))
        return start + float(self._truth["service_time"][node])

    def _release(self, node: int, time: float) -> None:
        self.state.request_status[node] = RequestStatus.PENDING
        self.log.append("request", time=time, node=node)
        self._decision_queue.append(node)

    def _reject(self, node: int | None, time: float, why: str) -> None:
        where = {"node": node} if node is not None else {}
        self.log.append("reject", time=time, component=why, **where)
        if node is not None:
            self.state.request_status[node] = RequestStatus.REJECTED
        self._charge(time, "rejected", 1.0, **({"node": node} if node is not None else {}))

    def _process(self, event: Event) -> None:
        state = self.state
        if event.kind == EventKind.REQUEST_ARRIVAL:
            state.time = event.time
            self._release(event.node, event.time)
            return
        if event.kind == EventKind.TIMER:
            state.time = event.time
            if self._timers.get(event.vehicle) == event.time:
                del self._timers[event.vehicle]
            return
        if event.kind == EventKind.OVERFLOW_ARRIVAL:
            state.time = event.time
            self._reject(None, event.time, "overflow")
            return
        if event.kind == EventKind.CANCELLATION:
            state.time = event.time
            node = event.node
            status = RequestStatus(int(state.request_status[node]))
            if status in (RequestStatus.ASSIGNED, RequestStatus.PENDING):
                serving = any(
                    v.node == node and v.status == VehicleStatus.SERVING for v in state.vehicles
                )
                if serving:
                    return  # service already under way: too late to cancel
                state.request_status[node] = RequestStatus.CANCELLED
                self.log.append("cancel", time=event.time, node=node)
                for v in state.vehicles:
                    if node in v.plan:
                        v.plan.remove(node)
                if node in self._decision_queue:
                    self._decision_queue.remove(node)
            return
        if (
            event.kind == EventKind.VEHICLE_ARRIVAL
            and event.node != DEPOT
            and state.request_status[event.node] == RequestStatus.CANCELLED
        ):
            v = state.vehicles[event.vehicle]
            state.time = event.time
            self.log.append("arrive", time=event.time, vehicle=v.index, node=event.node)
            v.time = event.time
            v.status = VehicleStatus.IDLE
            return
        super()._process(event)

    def _complete(self, v: VehicleState, node: int) -> None:
        self.state.request_status[node] = RequestStatus.SERVED
        v.status = VehicleStatus.IDLE

    def _latest_departure(self, v: VehicleState) -> float:
        """Latest time ``v`` can leave its node for the depot without breaking a hard limit."""
        travel = float(self._info.travel_times[v.node, DEPOT])
        limits = [np.inf]
        tw = self.rules.time_windows
        if tw is not None and self.rules.hard_windows:
            limits.append(float(tw[DEPOT, 1]) - travel)
        if self.rules.shift_end is not None:
            limits.append(self.rules.shift_end - travel)
        if self.rules.max_duration is not None and v.route_start is not None:
            limits.append(v.route_start + self.rules.max_duration - travel)
        return float(min(limits))

    def _day_over(self) -> bool:
        """No request can arrive or wait for a decision any more."""
        if self._decision_queue:
            return False
        return not any(
            e.kind in (EventKind.REQUEST_ARRIVAL, EventKind.OVERFLOW_ARRIVAL)
            for e in self.state.queue.heap
        )

    def _move_idle_vehicles(self) -> bool:
        """Drive first: send idle vehicles to their next stop; send them home when the day is over.

        Returns True if any vehicle moved or ended its day.
        """
        state = self.state
        moved = False
        day_over = self._day_over()
        for v in sorted(state.vehicles, key=lambda v: v.index):
            if v.status != VehicleStatus.IDLE:
                continue
            if v.plan:
                target = v.plan.pop(0)
                v.time = max(v.time, state.time)
                if v.route_start is None:
                    v.route_start = v.time
                    v.route = [DEPOT]
                    state.add_cost("vehicles", 1.0)
                    self.log.append(
                        "cost", time=v.time, vehicle=v.index, component="vehicles", amount=1.0
                    )
                self._travel(v, target)
                moved = True
            elif day_over:
                if v.node == DEPOT and not v.used:
                    self._retire(v)
                elif v.node == DEPOT:
                    self._end_route(v, v.time)
                else:
                    v.time = max(v.time, state.time)
                    self._travel(v, DEPOT)
                moved = True
            elif v.node != DEPOT:
                latest = self._latest_departure(v)
                if state.time >= latest - _TOL:
                    v.time = max(v.time, state.time)
                    self._travel(v, DEPOT)
                    moved = True
                elif np.isfinite(latest) and self._timers.get(v.index) != latest:
                    state.queue.push(latest, EventKind.TIMER, v.index, v.node)
                    self._timers[v.index] = latest
        return moved

    def _advance(self) -> Epoch | None:
        state = self.state
        while True:
            if self._decision_queue:
                node = self._decision_queue[0]
                state.epoch_count += 1
                return Epoch(state.time, "request", None, node)
            if self._move_idle_vehicles():
                continue
            if state.queue:
                self._process(state.queue.pop())
                continue
            self._terminate_planned()
            return None

    def _terminate_planned(self) -> None:
        state = self.state
        status = state.request_status
        for node in np.flatnonzero(
            (status == RequestStatus.ASSIGNED) | (status == RequestStatus.PENDING)
        ):
            status[node] = RequestStatus.UNSERVED
            self.log.append("unserved", node=int(node))
            self._charge(state.time, "unserved", 1.0)
        state.terminated = True
        state.active_vehicle = None
        self.log.append("episode_end", time=state.time)

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
        status = self.state.request_status
        future = status == RequestStatus.NOT_RELEASED
        self.state.remaining_demand[future] = self._truth["demand"][future]
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
            if live and np.isfinite(c) and c > now:
                queue.push(c, EventKind.CANCELLATION, -1, node)
        for k, t in enumerate(self.overflow):
            if t > now:
                queue.push(float(t), EventKind.OVERFLOW_ARRIVAL, -1, k)

    def snapshot(self) -> tuple[SimState, list[Any]]:
        """Deep copy of the state (the pending decisions are stored with it)."""
        state, cstates = super().snapshot()
        return state, [*cstates, (list(self._decision_queue), dict(self._timers))]

    def restore(self, snapshot: tuple[SimState, list[Any]]) -> None:
        """Restore a snapshot taken with `snapshot`."""
        state, cstates = snapshot
        super().restore((state, cstates[:-1]))
        queue, timers = cstates[-1]
        self._decision_queue = list(queue)
        self._timers = dict(timers)
