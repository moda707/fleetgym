"""Independent solution validator (architecture rule 11).

`validate` re-checks an episode from its event log, the spec and the instance only. It never imports
the simulator, constraints, network or environment code (enforced by the import-layer test): it
recomputes distances, travel times, service starts, loads and costs with its own short
implementation of the rules in ADR 0009, so a bug in the simulator cannot hide itself.

What is checked
---------------
* identity: the log was produced for this spec and instance (content hashes);
* vehicles start at the depot at the shift start with the right capacity, and the fleet size is
  respected;
* every departure leaves from where the vehicle is, when it is free, with the right arc distance;
* arrivals, service starts and service ends follow travel times, time windows and service times;
* hard constraints: capacity, time windows (including the depot), waiting rules, route duration,
  shift end;
* every customer is served at most once, and served or reported unserved exactly once;
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

__all__ = ["ValidationReport", "Violation", "validate"]

DEPOT = 0


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
    route_ended: bool = False
    retired: bool = False
    expected_target: int | None = None


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


def validate(
    log: EventLog | str | Path,
    spec: ProblemSpec,
    instance: Instance,
    *,
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
    tol
        Relative tolerance for floating-point comparisons.
    """
    if not isinstance(log, EventLog):
        log = EventLog.read(log)
    report = ValidationReport()
    found = report.violations

    def fail(code: str, message: str, row: Mapping[str, Any] | None = None, **where: Any) -> None:
        seq = row.get("seq") if row is not None else None
        found.append(Violation(code, message, seq=seq, **where))

    if log.metadata.get("spec_hash") != spec.content_hash():
        fail("identity", "the log was produced for a different spec")
    if log.metadata.get("instance_hash") != instance.content_hash():
        fail("identity", "the log was produced for a different instance")

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

    vehicles: dict[int, _Vehicle] = {}
    served: dict[int, int] = {}
    unserved: set[int] = set()
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
            if origin != v.node:
                fail(
                    "depart",
                    f"vehicle {index} departs from {origin} but is at {v.node}",
                    row,
                    vehicle=index,
                )
            if not _close(time, v.free_at, tol):
                fail(
                    "depart",
                    f"vehicle {index} departs at {time}, but is free at {v.free_at}",
                    row,
                    vehicle=index,
                )
            if not 0 <= target < n:
                fail("depart", f"node {target} does not exist", row, vehicle=index)
                continue
            if v.expected_target is not None and v.expected_target != target:
                fail(
                    "decision",
                    f"vehicle {index} was sent to {v.expected_target} but left for {target}",
                    row,
                )
            v.expected_target = None
            if origin == DEPOT and target == DEPOT:
                fail("depart", "depot to depot trip", row, vehicle=index)
            if target != DEPOT and (target in served or target in unserved):
                fail(
                    "double_service",
                    f"node {target} is visited twice",
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
            if target != DEPOT:
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
            if node in served:
                fail("double_service", f"node {node} served twice", row, node=node)
            served[node] = row["vehicle"]
            v.load += float(demands[node])
            if not _close(row.get("quantity", math.nan), float(demands[node]), tol):
                fail(
                    "quantity",
                    f"served {row.get('quantity')} at {node}, demand is {demands[node]}",
                    row,
                )
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
        elif kind == "service_end":
            v = vehicle_of(row)
            if v is None:
                continue
            if v.service is None or v.service[0] != row["node"]:
                fail("service", f"service end at {row['node']} without service", row)
                continue
            if not _close(row["time"], v.service[1], tol):
                fail(
                    "service_time",
                    f"service at {row['node']} ends at {row['time']}, expected {v.service[1]}",
                    row,
                )
            v.free_at = row["time"]
            v.service = None
        elif kind == "route_end":
            v = vehicle_of(row)
            if v is None:
                continue
            index, time = row["vehicle"], row["time"]
            if v.node != DEPOT or v.arrival is None or not _close(time, v.arrival, tol):
                fail("route_end", f"vehicle {index} ends its route away from the depot", row)
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
            if node in served or node in unserved or not 0 < node < n:
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
    missing = customers - served.keys() - unserved
    if missing and any(r["kind"] == "episode_end" for r in log.rows):
        found.append(
            Violation("coverage", f"nodes {sorted(missing)} are neither served nor unserved")
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
