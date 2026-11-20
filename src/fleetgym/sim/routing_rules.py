"""Built-in routing rules for interfaces that do not route (``accept_reject``, ``assign``).

``cheapest_insertion`` (plan Q27: it lives here, and the `CheapestInsertion` baseline wraps it)
inserts a request into a vehicle's planned route at the position that adds the least distance,
among positions that keep the whole remaining route feasible: capacity, hard time windows (with
waiting), route duration, shift end and the depot's closing time. Ties go to the lowest vehicle
index, then the earliest position. Travel and service times are deterministic here, so a feasible
plan stays feasible.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["Anchor", "Insertion", "RouteRules", "best_insertion", "plan_feasible"]

_TOL = 1e-9


@dataclass(frozen=True)
class Anchor:
    """Where and when a vehicle can start its remaining plan.

    Attributes
    ----------
    node
        The node it is at, or will be at after its committed stop.
    time
        When it is free there.
    load
        Its load at that point.
    route_start
        When its route started (``None`` if it has not left the depot).
    capacity
        Its capacity.
    """

    node: int
    time: float
    load: float
    route_start: float | None
    capacity: float


@dataclass(frozen=True)
class RouteRules:
    """Feasibility rules of a plan (taken from the spec's constraints)."""

    distances: NDArray[np.float64]
    travel_times: NDArray[np.float64]
    demands: NDArray[np.float64]
    service_times: NDArray[np.float64]
    time_windows: NDArray[np.float64] | None
    hard_windows: bool
    capacity_on: bool
    max_duration: float | None
    shift_end: float | None


@dataclass(frozen=True)
class Insertion:
    """Where to insert: ``vehicle``'s plan at ``position``, adding ``cost`` distance."""

    vehicle: int
    position: int
    cost: float


def plan_feasible(rules: RouteRules, anchor: Anchor, plan: Sequence[int]) -> bool:
    """Whether visiting ``plan`` from ``anchor`` and returning to the depot is feasible."""
    load = anchor.load + float(sum(rules.demands[j] for j in plan))
    if rules.capacity_on and load > anchor.capacity + _TOL:
        return False
    t, here = anchor.time, anchor.node
    tw = rules.time_windows
    start = anchor.route_start
    for j in plan:
        if start is None:
            start = t
        t += float(rules.travel_times[here, j])
        if tw is not None:
            t = max(t, float(tw[j, 0]))
            if rules.hard_windows and t > float(tw[j, 1]) + _TOL:
                return False
        t += float(rules.service_times[j])
        here = j
    if here != 0:
        t += float(rules.travel_times[here, 0])
    if tw is not None and rules.hard_windows and t > float(tw[0, 1]) + _TOL:
        return False
    if rules.shift_end is not None and t > rules.shift_end + _TOL:
        return False
    return not (
        rules.max_duration is not None
        and start is not None
        and t - start > rules.max_duration + _TOL
    )


def best_insertion(
    rules: RouteRules,
    anchors: Sequence[Anchor],
    plans: Sequence[Sequence[int]],
    node: int,
    vehicles: Iterable[int] | None = None,
) -> Insertion | None:
    """The cheapest feasible insertion of ``node`` over ``vehicles`` (all by default)."""
    best: Insertion | None = None
    d = rules.distances
    for v in range(len(anchors)) if vehicles is None else vehicles:
        anchor, plan = anchors[v], list(plans[v])
        stops = [anchor.node, *plan, 0]
        for pos in range(len(plan) + 1):
            prev, nxt = stops[pos], stops[pos + 1]
            cost = float(d[prev, node] + d[node, nxt] - d[prev, nxt])
            if best is not None and cost >= best.cost - _TOL:
                continue
            if plan_feasible(rules, anchor, [*plan[:pos], node, *plan[pos:]]):
                best = Insertion(v, pos, cost)
    return best if best is None or math.isfinite(best.cost) else None
