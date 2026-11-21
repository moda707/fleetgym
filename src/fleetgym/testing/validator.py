"""Independent solution validator (architecture rule 11).

`validate` re-checks an episode from its event log, the spec, the instance and (with uncertainty)
the scenario only. It never imports the simulator, constraints, network or environment code
(enforced by the import-layer test): it recomputes distances, travel times, service starts, loads
and costs with its own short implementation of the rules in ADRs 0009 and 0013, so a bug in the
simulator cannot hide itself.

What is checked
---------------
* identity: the log was produced for this spec, instance and scenario (content hashes);
* vehicles start at the depot at the shift start with the right capacity, and the fleet size is
  respected;
* every departure leaves from where the vehicle is, when it is free, with the right arc distance;
* arrivals, service starts and service ends follow travel times, time windows and (realized)
  service times;
* hard constraints: capacity, time windows (including the depot), waiting rules, route duration,
  shift end;
* deliveries: each service delivers at most what is left and what fits; every customer's realized
  demand is delivered or outsourced in full, or the customer is reported unserved;
* recourse: failures report the right remaining demand and load, restocks happen at the depot,
  a ``return_to_depot`` resolution goes to the depot and back to the same customer, outsourcing is
  priced by the rule's parameters; for the built-in rules, the rule's behavior itself (which
  resolution, how much is served first, when preventive restocks are allowed); with ``split``, a
  failure leaves the vehicle full, a full vehicle goes to the depot, every trip to the depot during
  the day restocks, and a partly served customer may end the day unserved;
* the ``served_demand`` revenue equals the quantities delivered;
* revealed values equal the scenario's;
* every route returns to the depot;
* recomputed costs equal the logged costs, and the sum of logged rewards equals minus the weighted
  recomputed cost (plus any logged invalid-action penalties).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fleetgym.data.instance import Instance
from fleetgym.errors import ValidationFailure
from fleetgym.eventlog import EventLog
from fleetgym.spec.binding import bind
from fleetgym.spec.models import ProblemSpec
from fleetgym.uncertainty.scenario import Scenario, episode_instance

__all__ = ["ValidationReport", "Violation", "validate"]

DEPOT = 0
_BUILTIN_RULES = ("return_to_depot", "outsource", "agent_decides", "split")


@dataclass(frozen=True)
class Violation:
    """One problem found in a trajectory."""

    code: str
    message: str
    seq: int | None = None
    vehicle: int | None = None
    node: int | None = None

    def __str__(self) -> str:
        where = f"[row {self.seq}] " if self.seq is not None else ""
        return f"{where}{self.code}: {self.message}"


@dataclass
class ValidationReport:
    """Result of `validate`.

    Attributes
    ----------
    violations
        Every problem found; empty if the trajectory is valid.
    costs
        Costs recomputed by the validator, by component.
    logged_costs
        Costs recorded in the log, by component.
    episode_return
        Sum of logged rewards.
    expected_return
        Minus the weighted recomputed cost (and logged invalid-action penalties).
    """

    violations: list[Violation] = field(default_factory=list)
    costs: dict[str, float] = field(default_factory=dict)
    logged_costs: dict[str, float] = field(default_factory=dict)
    episode_return: float = 0.0
    expected_return: float = 0.0

    @property
    def ok(self) -> bool:
        """True if no violation was found."""
        return not self.violations

    def raise_if_invalid(self) -> None:
        """Raise `ValidationFailure` listing every violation."""
        if self.violations:
            lines = "\n".join(f"  - {v}" for v in self.violations[:50])
            more = (
                f"\n  ... and {len(self.violations) - 50} more" if len(self.violations) > 50 else ""
            )
            raise ValidationFailure(f"{len(self.violations)} violation(s):\n{lines}{more}")


@dataclass
class _Vehicle:
    capacity: float
    node: int = DEPOT
    free_at: float = 0.0
    load: float = 0.0
    route_start: float | None = None
    in_transit: tuple[int, float] | None = None
    arrival: float | None = None
    service: tuple[int, float] | None = None
    serving: bool = False
    """A ``service_start`` was logged for the current visit."""
    route_ended: bool = False
    retired: bool = False
    expected_target: int | None = None
    at_depot_open: bool = False
    """Arrived at the depot mid-route: a ``restock`` or ``route_end`` must follow."""
    must_go: int | None = None
    """Recourse obligation: the next departure must go to this node."""
    come_back_to: int | None = None
    """After the next restock, the vehicle must return to this customer."""
    last_completed_full: bool = False
    """The last customer it completed left it exactly full (preventive restock allowed)."""
    restock_trip: bool = False
    """On a trip to the depot that restocks (split recourse): a ``restock`` must follow."""


def _distances(spec: ProblemSpec, instance: Instance) -> NDArray[np.float64]:
    params = spec.network.params.model_dump()
    if spec.network.name == "matrix":
        assert instance.distance_matrix is not None
        return np.asarray(instance.distance_matrix, dtype=float)
    assert instance.coords is not None
    coords = np.asarray(instance.coords, dtype=float)
    n = coords.shape[0]
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            d = math.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
            out[i, j] = math.floor(d + 0.5) if params["rounding"] == "nearest_int" else d
    return out


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _at(value: Any, node: int) -> float:
    arr = np.asarray(value, dtype=float)
    return float(arr) if arr.ndim == 0 else float(arr[node])


def validate(
    log: EventLog | str | Path,
    spec: ProblemSpec,
    instance: Instance,
    *,
    scenario: Scenario | None = None,
    tol: float = 1e-9,
) -> ValidationReport:
    """Re-check a trajectory independently of the simulator.

    Parameters
    ----------
    log
        The event log, or the path of a Parquet log file.
    spec
        The spec the episode was run with.
    instance
        The instance the episode was run on.
    scenario
        The episode's scenario (required when the spec has uncertain parameters).
    tol
        Relative tolerance for floating-point comparisons.

    Raises
    ------
    ValueError
        If the spec has uncertain parameters and no scenario is given.
    """
    if not isinstance(log, EventLog):
        log = EventLog.read(log)
    if (spec.uncertainty or spec.dynamics is not None) and scenario is None:
        raise ValueError("this spec has uncertain parameters: pass the episode's scenario")
    report = ValidationReport()
    found = report.violations

    def fail(code: str, message: str, row: Mapping[str, Any] | None = None, **where: Any) -> None:
        seq = row.get("seq") if row is not None else None
        found.append(Violation(code, message, seq=seq, **where))

    # With dynamic requests the episode instance adds the scenario's request slots; accept either
    # the base instance or the episode instance.
    base_hash = str(instance.extra.get("base_instance_hash", "")) or instance.content_hash()
    if (
        scenario is not None
        and "request.release" in scenario.values
        and "base_instance_hash" not in instance.extra
    ):
        instance = episode_instance(instance, scenario)
    if log.metadata.get("spec_hash") != spec.content_hash():
        fail("identity", "the log was produced for a different spec")
    if log.metadata.get("instance_hash") != base_hash:
        fail("identity", "the log was produced for a different instance")
    if scenario is not None:
        if log.metadata.get("scenario_id") != scenario.id:
            fail("identity", "the log was produced for a different scenario")
        if scenario.spec_hash != spec.content_hash() or scenario.instance_hash != base_hash:
            fail("identity", "the scenario belongs to a different spec or instance")
    planned = spec.decisions.interface in ("accept_reject", "assign")
    dynamic = spec.dynamics is not None
    waits = planned or dynamic  # vehicles may stay idle before departing or ending
    release_times = np.asarray(
        instance.extra.get("release", np.zeros(instance.n_nodes)), dtype=float
    )
    cancel_times = np.asarray(
        instance.extra.get("cancel", np.full(instance.n_nodes, np.inf)), dtype=float
    )
    overflow_times = (
        [float(t) for t in scenario["request.overflow"]]
        if scenario is not None and "request.overflow" in scenario.values
        else []
    )
    released: set[int] = set()
    accepted: dict[int, int] = {}
    decided: set[int] = set()
    cancelled: set[int] = set()
    overflow_seen = 0

    problem = bind(spec, instance)
    dist = _distances(spec, instance)
    speed = float(spec.network.params.model_dump()["speed"])
    n = instance.n_nodes
    demands = np.asarray(instance.demands, dtype=float)
    service_times = (
        np.asarray(instance.service_times, dtype=float)
        if instance.service_times is not None
        else np.zeros(n)
    )
    truth: dict[str, NDArray[np.float64]] = {}
    if scenario is not None:
        for name in spec.uncertainty:
            truth[name] = np.asarray(scenario[name], dtype=float)
        demands = truth.get("demand", demands)
        service_times = truth.get("service_time", service_times)
    tw = (
        np.asarray(instance.time_windows, dtype=float)
        if instance.time_windows is not None
        else None
    )
    tw_params = spec.constraints.items.get("time_windows")
    tw_on = tw_params is not None and tw is not None
    tw_soft = bool(tw_params.model_dump()["soft"]) if tw_params is not None else False
    waiting_allowed = (
        tw_params.model_dump()["waiting"] == "allowed" if tw_params is not None else True
    )
    capacity_on = "capacity" in spec.constraints
    duration_max = (
        float(problem.value("constraints.route_duration.max"))
        if "route_duration" in spec.constraints
        else None
    )
    shift_start = float(spec.fleet.shift[0]) if spec.fleet.shift is not None else 0.0
    shift_end = float(spec.fleet.shift[1]) if spec.fleet.shift is not None else None
    type_of_vehicle: list[int] = []
    for t, vt in enumerate(problem.vehicle_types):
        type_of_vehicle += [t] * (vt.count or 0)
    fleet_limit = None if spec.fleet.sizing == "unlimited" else len(type_of_vehicle)

    rule_spec = spec.recourse.rules.get("capacity_failure")
    rule = rule_spec.name if rule_spec is not None else None
    rule_params = rule_spec.params.model_dump() if rule_spec is not None else {}
    builtin = rule in _BUILTIN_RULES
    allowed_actions: set[str] = (
        {"return_to_depot"}
        if rule == "return_to_depot"
        else {"outsource"}
        if rule == "outsource"
        else set(rule_params.get("options", ()))
        if rule == "agent_decides"
        else {"split"}
        if rule == "split"
        else {"return_to_depot", "outsource"}
        if rule is not None
        else set()
    )
    split = rule == "split"  # trips to the depot during the day restock (ADR 0021)
    reward_service = "served_demand" in spec.objective.components()
    partial_service = bool(rule_params.get("partial_service", True))
    preventive_ok = (
        rule == "return_to_depot" and rule_params.get("on_exact_depletion") == "restock"
    ) or (rule is not None and not builtin)

    def outsourcing_cost(node: int, quantity: float) -> float:
        per_customer = _at(problem.value("recourse.capacity_failure.per_customer"), node)
        per_unit = _at(problem.value("recourse.capacity_failure.per_unit"), node)
        return per_customer + per_unit * quantity

    vehicles: dict[int, _Vehicle] = {}
    remaining = demands.copy()
    started: set[int] = set()
    completed: dict[int, int] = {}
    unserved: set[int] = set()
    revealed: dict[str, set[int]] = {name: set() for name in truth}
    costs: dict[str, float] = {}
    logged: dict[str, float] = {}
    penalties = 0.0
    rewards = 0.0

    def charge(component: str, amount: float) -> None:
        if amount:
            costs[component] = costs.get(component, 0.0) + amount

    def vehicle_of(row: Mapping[str, Any]) -> _Vehicle | None:
        v = vehicles.get(row.get("vehicle", -1))
        if v is None:
            fail("unknown_vehicle", f"vehicle {row.get('vehicle')} was never activated", row)
        return v

    def complete(node: int, index: int, v: _Vehicle) -> None:
        completed[node] = index
        v.last_completed_full = capacity_on and _close(v.load, v.capacity, tol)

    for row in log.rows:
        kind = row["kind"]
        if kind == "vehicle_activate":
            index = row["vehicle"]
            if index in vehicles:
                fail("fleet", f"vehicle {index} activated twice", row, vehicle=index)
                continue
            if fleet_limit is not None and index >= fleet_limit:
                fail(
                    "fleet",
                    f"vehicle {index} exceeds the fleet of {fleet_limit}",
                    row,
                    vehicle=index,
                )
            t = type_of_vehicle[index] if index < len(type_of_vehicle) else 0
            capacity = problem.vehicle_types[t].capacity
            capacity = float(capacity) if capacity is not None else math.inf
            if not _close(row.get("value", math.nan), capacity, tol) and not (
                math.isinf(capacity) and math.isinf(row.get("value", 0.0))
            ):
                fail(
                    "capacity_value",
                    f"vehicle {index} has capacity {row.get('value')}, expected {capacity}",
                    row,
                )
            if row.get("node") != DEPOT or not _close(row["time"], shift_start, tol):
                fail(
                    "start",
                    f"vehicle {index} must start at the depot at {shift_start}",
                    row,
                    vehicle=index,
                )
            vehicles[index] = _Vehicle(capacity=capacity, free_at=shift_start)
        elif kind == "decision":
            v = vehicles.get(row.get("vehicle", -1))
            if v is not None and "node" in row:
                v.expected_target = row["node"]
        elif kind == "depart":
            v = vehicle_of(row)
            if v is None:
                continue
            index, origin, target, time = row["vehicle"], row["from_node"], row["node"], row["time"]
            if v.route_ended or v.retired or v.in_transit is not None or v.service is not None:
                fail("depart", f"vehicle {index} departs while not idle", row, vehicle=index)
            if v.at_depot_open:
                fail(
                    "depart",
                    f"vehicle {index} leaves the depot without restocking or ending its route",
                    row,
                    vehicle=index,
                )
            if origin != v.node:
                fail(
                    "depart",
                    f"vehicle {index} departs from {origin} but is at {v.node}",
                    row,
                    vehicle=index,
                )
            if waits and time < v.free_at - tol * max(1.0, abs(time)):
                fail(
                    "depart",
                    f"vehicle {index} departs at {time}, before it is free at {v.free_at}",
                    row,
                    vehicle=index,
                )
            elif not waits and not _close(time, v.free_at, tol):
                fail(
                    "depart",
                    f"vehicle {index} departs at {time}, but is free at {v.free_at}",
                    row,
                    vehicle=index,
                )
            if not 0 <= target < n:
                fail("depart", f"node {target} does not exist", row, vehicle=index)
                continue
            if v.must_go is not None:
                if target != v.must_go:
                    fail(
                        "recourse",
                        f"vehicle {index} had to go to {v.must_go} but left for {target}",
                        row,
                        vehicle=index,
                    )
                v.must_go = None
                v.expected_target = None
            elif v.expected_target is not None and v.expected_target != target:
                fail(
                    "decision",
                    f"vehicle {index} was sent to {v.expected_target} but left for {target}",
                    row,
                )
            elif (
                not planned
                and v.expected_target is None
                and target == DEPOT
                and v.route_start is not None
                and not (preventive_ok and v.last_completed_full)
            ):
                # A depot trip nobody decided must be a preventive restock by the recourse rule.
                fail(
                    "recourse",
                    f"vehicle {index} returns to the depot without a decision",
                    row,
                    vehicle=index,
                )
            v.expected_target = None
            if dynamic and not planned and target != DEPOT and target not in released:
                fail("request", f"vehicle {index} drives to {target} before it is released", row)
            if planned and target != DEPOT and accepted.get(target) != index:
                fail(
                    "assignment",
                    f"vehicle {index} drives to {target}, which was not accepted for it",
                    row,
                    vehicle=index,
                    node=target,
                )
            if origin == DEPOT and target == DEPOT:
                fail("depart", "depot to depot trip", row, vehicle=index)
            if split and target != DEPOT and capacity_on and v.load >= v.capacity * (1 - tol):
                fail(
                    "recourse",
                    f"vehicle {index} has no free capacity and must go to the depot, not {target}",
                    row,
                    vehicle=index,
                )
            if split and target == DEPOT and origin != DEPOT:
                v.restock_trip = True
            if target != DEPOT and (target in completed or target in unserved):
                fail(
                    "double_service",
                    f"node {target} is visited after it was completed",
                    row,
                    vehicle=index,
                    node=target,
                )
            arc = float(dist[origin, target])
            if not _close(row.get("value", math.nan), arc, tol):
                fail(
                    "distance",
                    f"arc {origin}->{target} logged {row.get('value')}, expected {arc}",
                    row,
                )
            if v.route_start is None:
                v.route_start = time
                charge("vehicles", 1.0)
            charge("distance", arc)
            charge("time", arc / speed)
            v.in_transit = (target, time + arc / speed)
        elif kind == "arrive":
            v = vehicle_of(row)
            if v is None:
                continue
            index = row["vehicle"]
            if v.in_transit is None or v.in_transit[0] != row["node"]:
                fail(
                    "arrive",
                    f"vehicle {index} arrives at {row['node']} without travelling there",
                    row,
                )
                continue
            target, expected = v.in_transit
            if not _close(row["time"], expected, tol):
                fail(
                    "travel_time",
                    f"vehicle {index} arrives at {row['time']}, expected {expected}",
                    row,
                )
            v.in_transit = None
            v.node = target
            v.arrival = row["time"]
            v.free_at = row["time"]
            if target == DEPOT:
                v.at_depot_open = True
                continue
            if target in cancelled:
                continue  # the request was cancelled on the way: nothing to serve
            start = row["time"]
            if tw_on:
                assert tw is not None
                if waiting_allowed:
                    start = max(start, float(tw[target, 0]))
                elif row["time"] < tw[target, 0] - tol:
                    fail(
                        "time_window",
                        f"early arrival at {target} with waiting forbidden",
                        row,
                        node=target,
                    )
            charge("waiting", start - row["time"])
            v.service = (target, start)
        elif kind == "service_start":
            v = vehicle_of(row)
            if v is None:
                continue
            node, time = row["node"], row["time"]
            if v.service is None or v.service[0] != node:
                fail("service", f"service at {node} without arriving there", row, node=node)
                continue
            if not _close(time, v.service[1], tol):
                fail(
                    "service_start",
                    f"service at {node} starts at {time}, expected {v.service[1]}",
                    row,
                )
            if node in completed:
                fail("double_service", f"node {node} served after it was completed", row, node=node)
            if planned and (node not in accepted or node in cancelled):
                fail("assignment", f"service at {node}, which is not an accepted request", row)
            quantity = float(row.get("quantity", math.nan))
            left = float(remaining[node])
            free = v.capacity - v.load if capacity_on else math.inf
            fits = left <= free * (1 + tol) + tol
            expected_q = left if fits else free
            if not fits and rule is None:
                fail("capacity", f"demand {left} at {node} exceeds free capacity {free}", row)
            elif not fits and builtin and not partial_service and v.load > tol:
                fail(
                    "recourse",
                    f"{rule} with partial_service false must not serve before restocking",
                    row,
                    node=node,
                )
            if builtin or rule is None:
                if not _close(quantity, expected_q, tol):
                    fail(
                        "quantity",
                        f"served {quantity} at {node}, expected {expected_q}",
                        row,
                        node=node,
                    )
            elif not (0 <= quantity <= min(left, free) * (1 + tol) + tol):
                fail("quantity", f"served {quantity} at {node}, at most {min(left, free)}", row)
            started.add(node)
            v.load += quantity
            remaining[node] = left - quantity
            if reward_service:
                charge("served_demand", quantity)
            if not _close(row.get("load", math.nan), v.load, tol):
                fail("load", f"logged load {row.get('load')}, recomputed {v.load}", row)
            if capacity_on and v.load > v.capacity * (1 + tol):
                fail("capacity", f"load {v.load} exceeds capacity {v.capacity}", row, node=node)
            if tw_on:
                assert tw is not None
                late = time - float(tw[node, 1])
                if tw_soft:
                    charge("lateness", max(0.0, late))
                elif late > tol * max(1.0, abs(time)):
                    fail(
                        "time_window",
                        f"service at {node} starts {late} after its window",
                        row,
                        node=node,
                    )
            v.service = (node, time + float(service_times[node]))
            v.serving = True
        elif kind == "service_end":
            v = vehicle_of(row)
            if v is None:
                continue
            node = row["node"]
            if v.service is None or v.service[0] != node or not v.serving:
                fail("service", f"service end at {node} without service", row)
                continue
            v.serving = False
            if not _close(row["time"], v.service[1], tol):
                fail(
                    "service_time",
                    f"service at {node} ends at {row['time']}, expected {v.service[1]}",
                    row,
                )
            v.free_at = row["time"]
            v.service = None
            if remaining[node] <= tol:
                remaining[node] = 0.0
                complete(node, row["vehicle"], v)
        elif kind == "failure":
            v = vehicle_of(row)
            if v is None:
                continue
            node, action = row["node"], row.get("component")
            if v.node != node or v.in_transit is not None:
                fail("recourse", f"failure at {node} where the vehicle is not", row, node=node)
                continue
            if v.service is not None and v.service[0] == node:
                v.service = None  # failure before any service: no service end follows
                v.free_at = row["time"]
            if rule is None:
                fail("recourse", "a capacity failure without a recourse rule", row, node=node)
            if not _close(row.get("quantity", math.nan), float(remaining[node]), tol):
                fail(
                    "recourse",
                    f"failure at {node} reports {row.get('quantity')} left, expected "
                    f"{remaining[node]}",
                    row,
                )
            if remaining[node] <= tol:
                fail("recourse", f"failure at {node} with nothing left to deliver", row)
            if not _close(row.get("load", math.nan), v.load, tol):
                fail("load", f"failure logs load {row.get('load')}, recomputed {v.load}", row)
            if action not in allowed_actions:
                fail("recourse", f"resolution {action!r} is not allowed for rule {rule!r}", row)
            if action == "return_to_depot":
                v.must_go = DEPOT
                v.come_back_to = node
            elif action == "split" and capacity_on and v.load < v.capacity * (1 - tol):
                fail(
                    "recourse",
                    f"split at {node} while the vehicle still has free capacity",
                    row,
                    node=node,
                )
        elif kind == "restock":
            v = vehicle_of(row)
            if v is None:
                continue
            if v.node != DEPOT or not v.at_depot_open or not _close(row["time"], v.free_at, tol):
                fail("restock", f"vehicle {row['vehicle']} restocks away from the depot", row)
            if split and not v.restock_trip:
                fail("restock", f"vehicle {row['vehicle']} restocks on a trip that ended", row)
            v.restock_trip = False
            v.at_depot_open = False
            v.load = 0.0
            v.free_at = row["time"]
            v.last_completed_full = False
            if v.come_back_to is not None:
                v.must_go, v.come_back_to = v.come_back_to, None
        elif kind == "outsource" and row.get("component") == "request":
            node = row["node"]
            if node not in released or node in decided:
                fail("request", f"outsourcing of {node}, which awaits no decision", row)
            decided.add(node)
            if "outsourcing" not in spec.constraints:
                fail("request", "outsourcing without an `outsourcing` constraint", row)
            else:
                cost = _at(problem.value("constraints.outsourcing.value"), node)
                if not _close(row.get("amount", math.nan), cost, tol):
                    fail("outsource", f"logged cost {row.get('amount')}, expected {cost}", row)
                charge("outsourcing", cost)
            completed[node] = -1
            started.add(node)
        elif kind == "request":
            node, time = row["node"], row["time"]
            if not 0 < node < n or node in released:
                fail("request", f"request {node} released twice or out of range", row)
            elif not _close(time, float(release_times[node]), tol):
                fail(
                    "request",
                    f"request {node} released at {time}, scenario says {release_times[node]}",
                    row,
                )
            released.add(node)
        elif kind == "accept":
            node, index = row["node"], row.get("vehicle", -1)
            if node not in released or node in decided or node in cancelled:
                fail("request", f"accept of {node}, which awaits no decision", row)
            if index not in vehicles:
                fail("unknown_vehicle", f"request {node} accepted for vehicle {index}", row)
            decided.add(node)
            accepted[node] = index
        elif kind == "reject":
            if row.get("component") == "overflow":
                if overflow_seen >= len(overflow_times) or not _close(
                    row["time"], overflow_times[overflow_seen], tol
                ):
                    fail("request", "an overflow rejection that the scenario does not have", row)
                overflow_seen += 1
            else:
                node = row.get("node")
                if node is None or node not in released or node in decided or node in cancelled:
                    fail("request", f"reject of {node}, which awaits no decision", row)
                else:
                    decided.add(node)
            charge("rejected", 1.0)
        elif kind == "cancel":
            node, time = row["node"], row["time"]
            if not _close(time, float(cancel_times[node]), tol):
                fail(
                    "cancel", f"{node} cancelled at {time}, scenario says {cancel_times[node]}", row
                )
            if node in started or node in completed:
                fail("cancel", f"{node} cancelled after its service started", row)
            cancelled.add(node)
        elif kind == "outsource":
            v = vehicle_of(row)
            if v is None:
                continue
            node = row["node"]
            quantity = float(row.get("quantity", math.nan))
            if not _close(quantity, float(remaining[node]), tol) or remaining[node] <= tol:
                fail(
                    "outsource",
                    f"outsourced {quantity} at {node}, {remaining[node]} was left",
                    row,
                    node=node,
                )
            if "outsource" not in allowed_actions:
                fail("recourse", f"rule {rule!r} cannot outsource", row, node=node)
            else:
                amount = outsourcing_cost(node, quantity)
                if not _close(row.get("amount", math.nan), amount, tol):
                    fail("outsource", f"logged cost {row.get('amount')}, expected {amount}", row)
                charge("outsourcing", amount)
            remaining[node] = 0.0
            started.add(node)
            complete(node, row["vehicle"], v)
        elif kind == "reveal":
            node, name = row["node"], row.get("component")
            if name not in truth:
                fail("reveal", f"reveal of {name!r}, which is not uncertain", row)
                continue
            if node in revealed[name]:
                fail("reveal", f"{name} of node {node} revealed twice", row)
            revealed[name].add(node)
            if not _close(row.get("value", math.nan), float(truth[name][node]), tol):
                fail(
                    "reveal",
                    f"{name} of node {node} revealed as {row.get('value')}, scenario has "
                    f"{truth[name][node]}",
                    row,
                )
        elif kind == "route_end":
            v = vehicle_of(row)
            if v is None:
                continue
            index, time = row["vehicle"], row["time"]
            early = time < v.free_at - tol * max(1.0, abs(time))
            if (
                v.node != DEPOT
                or v.arrival is None
                or early
                or (not waits and not _close(time, v.free_at, tol))
            ):
                fail("route_end", f"vehicle {index} ends its route away from the depot", row)
            if v.come_back_to is not None or v.must_go is not None:
                fail("recourse", f"vehicle {index} ends its route during a recourse trip", row)
            if split and v.restock_trip:
                fail("restock", f"vehicle {index} ends its route without restocking", row)
            v.at_depot_open = False
            assert v.route_start is not None or found
            duration = time - (v.route_start or 0.0)
            if not _close(row.get("value", math.nan), duration, tol):
                fail("duration", f"logged duration {row.get('value')}, recomputed {duration}", row)
            if duration_max is not None and duration > duration_max * (1 + tol):
                fail(
                    "route_duration",
                    f"route of vehicle {index} lasts {duration} > {duration_max}",
                    row,
                )
            if tw_on and not tw_soft:
                assert tw is not None
                if time > float(tw[DEPOT, 1]) * (1 + tol):
                    fail(
                        "time_window",
                        f"vehicle {index} returns at {time} after the depot closes",
                        row,
                    )
            if shift_end is not None and time > shift_end * (1 + tol):
                fail("shift", f"vehicle {index} returns at {time} after the shift ends", row)
            v.route_ended = True
        elif kind == "vehicle_retire":
            v = vehicle_of(row)
            if v is None:
                continue
            if v.route_start is not None or v.node != DEPOT:
                fail("retire", f"vehicle {row['vehicle']} retires after leaving the depot", row)
            v.retired = True
        elif kind == "unserved":
            node = row["node"]
            partly_served_ok = split and node not in completed and remaining[node] > tol
            if (node in started and not partly_served_ok) or node in unserved or not 0 < node < n:
                fail("unserved", f"node {node} cannot be reported unserved", row, node=node)
            unserved.add(node)
            charge("unserved", 1.0)
        elif kind == "cost":
            component, amount = row["component"], row["amount"]
            if component == "invalid_action":
                penalties += amount
            else:
                logged[component] = logged.get(component, 0.0) + amount
        elif kind == "epoch":
            rewards += row["reward"]

    for index, v in vehicles.items():
        if v.route_start is not None and not v.route_ended:
            found.append(
                Violation(
                    "incomplete", f"vehicle {index} never returned to the depot", vehicle=index
                )
            )
    customers = set(range(instance.n_depots, n))
    if dynamic and not planned:
        customers = released - cancelled
    if planned:
        customers = released - cancelled - (released - accepted.keys())
        undecided = released - decided - cancelled
        if undecided and any(r["kind"] == "episode_end" for r in log.rows):
            found.append(Violation("request", f"requests {sorted(undecided)} were never decided"))
    if dynamic:
        if overflow_seen != len(overflow_times) and any(
            r["kind"] == "episode_end" for r in log.rows
        ):
            found.append(
                Violation(
                    "request",
                    f"{len(overflow_times)} overflow requests, {overflow_seen} rejected",
                )
            )
        never = {
            int(i)
            for i in np.flatnonzero(np.isfinite(release_times))
            if int(i) >= instance.n_depots
            and float(release_times[int(i)]) <= float(log.rows[-1].get("time") or 0.0)
        } - released
        if never and any(r["kind"] == "episode_end" for r in log.rows):
            found.append(Violation("request", f"requests {sorted(never)} were never released"))
    ended = any(r["kind"] == "episode_end" for r in log.rows)
    missing = customers - completed.keys() - unserved
    if missing and ended:
        partial = sorted(missing & started)
        if partial:
            found.append(
                Violation("delivery", f"nodes {partial} were not delivered or outsourced in full")
            )
        untouched = sorted(missing - started)
        if untouched:
            found.append(
                Violation("coverage", f"nodes {untouched} are neither served nor unserved")
            )

    for component in sorted(costs.keys() | logged.keys()):
        a, b = costs.get(component, 0.0), logged.get(component, 0.0)
        if not _close(a, b, max(tol, 1e-9)):
            found.append(Violation("cost", f"{component}: logged {b}, recomputed {a}"))
    weights = spec.objective.scalar_weights()
    expected = -(sum(weights.get(c, 0.0) * a for c, a in costs.items()) + penalties)
    if not _close(rewards, expected, max(tol, 1e-9)):
        found.append(Violation("reward", f"sum of rewards {rewards}, expected {expected}"))
    report.costs = costs
    report.logged_costs = logged
    report.episode_return = rewards
    report.expected_return = expected
    return report
