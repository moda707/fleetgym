"""Independent route oracle and brute-force solver for Gate 1 tests.

Written from the rules in ADR 0009 without using FleetGym's simulator, network or constraints.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from fleetgym.data import Instance
from fleetgym.spec import ProblemSpec


@dataclass(frozen=True)
class Rules:
    capacity: float | None
    time_windows: bool
    soft: bool
    waiting_allowed: bool
    max_duration: float | None
    shift_start: float
    shift_end: float | None
    rounding: str
    speed: float

    @classmethod
    def from_spec(cls, spec: ProblemSpec, instance: Instance) -> Rules:
        tw = spec.constraints.items.get("time_windows")
        tw_params = tw.model_dump() if tw is not None else {}
        duration = None
        if "route_duration" in spec.constraints:
            raw = spec.constraints["route_duration"].model_dump()["max"]
            duration = (
                float(instance.get("depot_due_time")) if raw == "depot_due_time" else float(raw)
            )
        cap = spec.fleet.types[0].capacity
        capacity = (
            cap if isinstance(cap, float) else (instance.capacity if cap is not None else None)
        )
        net = spec.network.params.model_dump()
        return cls(
            capacity=capacity if "capacity" in spec.constraints else None,
            time_windows=tw is not None,
            soft=tw_params.get("mode", "hard") == "soft",
            waiting_allowed=tw_params.get("waiting", "allowed") == "allowed",
            max_duration=duration,
            shift_start=float(spec.fleet.shift[0]) if spec.fleet.shift else 0.0,
            shift_end=float(spec.fleet.shift[1]) if spec.fleet.shift else None,
            rounding=net["rounding"],
            speed=float(net["speed"]),
        )


def distance(instance: Instance, rules: Rules, i: int, j: int) -> float:
    assert instance.coords is not None
    (x1, y1), (x2, y2) = instance.coords[i], instance.coords[j]
    d = math.hypot(x1 - x2, y1 - y2)
    return math.floor(d + 0.5) if rules.rounding == "nearest_int" else d


def route_feasible(instance: Instance, rules: Rules, route: Sequence[int]) -> bool:
    """Feasibility of a full route ``[0, c1, ..., ck, 0]`` started at the shift start."""
    t, load = rules.shift_start, 0.0
    tw = instance.time_windows
    service = instance.service_times
    for prev, node in itertools.pairwise(route):
        t += distance(instance, rules, prev, node) / rules.speed
        if node == 0:
            if rules.time_windows and not rules.soft and tw is not None and t > tw[0, 1] + 1e-9:
                return False
            if rules.max_duration is not None and t - rules.shift_start > rules.max_duration + 1e-9:
                return False
            if rules.shift_end is not None and t > rules.shift_end + 1e-9:
                return False
            continue
        start = t
        if rules.time_windows and tw is not None:
            if not rules.waiting_allowed and t < tw[node, 0] - 1e-9:
                return False
            if rules.waiting_allowed:
                start = max(t, float(tw[node, 0]))
            if not rules.soft and start > tw[node, 1] + 1e-9:
                return False
        load += float(instance.demands[node])
        if rules.capacity is not None and load > rules.capacity + 1e-9:
            return False
        t = start + (float(service[node]) if service is not None else 0.0)
    return True


def route_distance(instance: Instance, rules: Rules, route: Sequence[int]) -> float:
    return sum(distance(instance, rules, a, b) for a, b in itertools.pairwise(route))


def _partitions(items: list[int]) -> Iterator[list[list[int]]]:
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for partition in _partitions(rest):
        for i in range(len(partition)):
            yield [*partition[:i], [first, *partition[i]], *partition[i + 1 :]]
        yield [[first], *partition]


def best_route(instance: Instance, rules: Rules, block: Sequence[int]) -> float | None:
    best = None
    for order in itertools.permutations(block):
        route = [0, *order, 0]
        if route_feasible(instance, rules, route):
            d = route_distance(instance, rules, route)
            best = d if best is None else min(best, d)
    return best


def brute_force(instance: Instance, rules: Rules) -> list[tuple[int, float]]:
    """``(vehicles, distance)`` of the best routing of every feasible partition."""
    customers = list(range(1, instance.n_nodes))
    cache: dict[tuple[int, ...], float | None] = {}
    results = []
    for partition in _partitions(customers):
        total = 0.0
        for block in partition:
            key = tuple(sorted(block))
            if key not in cache:
                cache[key] = best_route(instance, rules, key)
            cost = cache[key]
            if cost is None:
                break
            total += cost
        else:
            results.append((len(partition), total))
    return results
