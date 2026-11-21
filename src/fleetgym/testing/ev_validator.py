"""Independent validator of EV shortest-path episodes (M14, ADR 0026).

Re-checks a log of `fleetgym.envs.ev_env.EVShortestPathEnv` from the instance, the spec and the
scenario alone, with its own replay of the station queues (it imports neither the simulator nor
the network, architecture rule 11):

* geometry and energy: every departure and diversion starts where the EV is, with the battery it
  has; arrivals happen at ``distance / speed``; the battery never goes below 0 and the EV only
  drives to nodes it can reach;
* charging: at the station where the EV is, from its battery to a target above it and at most B,
  for ``Phi(target) - Phi(battery)``;
* queues: replayed from the scenario's exogenous EVs and the EV's own joins, charges and departures
  (first come, first served; at most ``kappa`` EVs; exogenous EVs leave a full station); the queue
  length seen on arrival, the time the EV reaches the head of the queue, every occupancy-indicator
  change and the charger being free when the EV starts charging are checked;
* costs: driving, waiting and charging times; their sum is the arrival time at the destination;
  logged rewards sum to minus the weighted cost.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np

from fleetgym.data import Instance
from fleetgym.eventlog import EventLog
from fleetgym.spec.models import ProblemSpec
from fleetgym.spec.models.ev import ev_shortest_path
from fleetgym.uncertainty.scenario import Scenario

__all__ = ["validate_ev"]


def _num(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value is not None else math.nan


def _phi(levels: np.ndarray, times: np.ndarray, b: float) -> float:
    xs = np.concatenate([[0.0], levels])
    ys = np.concatenate([[0.0], times])
    if b <= xs[-1]:
        return float(np.interp(b, xs, ys))
    return float(ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (b - xs[-1]))


class _Queues:
    """The stations, replayed independently from the scenario."""

    def __init__(self, instance: Instance, scenario: Scenario) -> None:
        extra = instance.extra
        role = np.asarray(extra["role"], dtype=object)
        self.stations = [i for i in range(instance.n_nodes) if role[i] == "station"]
        self.kappa = np.asarray(extra["queue_capacity"], dtype=float)
        self.serving: dict[int, tuple[str, float] | None] = dict.fromkeys(self.stations)
        self.line: dict[int, list[Any]] = {i: [] for i in self.stations}
        self.events: list[tuple[float, int, int, int, float]] = []
        self.seq = 0
        self.indicator = dict.fromkeys(self.stations, 0.0)
        self.changes: list[tuple[float, int, float]] = []
        self.head_times: list[tuple[float, int]] = []
        initial = iter(np.asarray(scenario["ev.initial_service"], dtype=float).tolist())
        counts = np.asarray(scenario["ev.initial"], dtype=float)
        for i in self.stations:
            durations = [next(initial) for _ in range(int(counts[i]))]
            if durations:
                self.serving[i] = ("x", durations[0])
                self._push(durations[0], 0, i, 0.0)
                self.line[i] = list(durations[1:])
            self.indicator[i] = float(bool(durations))
        for t, node, s in zip(
            np.asarray(scenario["ev.arrival"], float),
            np.asarray(scenario["ev.arrival_node"], float),
            np.asarray(scenario["ev.arrival_service"], float),
            strict=True,
        ):
            self._push(t, 1, int(node), float(s))
        self.now = 0.0

    def _push(self, t: float, kind: int, node: int, value: float) -> None:
        heapq.heappush(self.events, (t, kind, self.seq, node, value))
        self.seq += 1

    def count(self, node: int) -> int:
        return (self.serving[node] is not None) + len(self.line[node])

    def _mark(self, node: int) -> None:
        value = float(self.count(node) > 0)
        if value != self.indicator[node]:
            self.indicator[node] = value
            self.changes.append((self.now, node, value))

    def _next(self, node: int) -> None:
        if self.serving[node] is not None or not self.line[node]:
            return
        first = self.line[node].pop(0)
        if first == "ev":
            self.serving[node] = ("ev", self.now)
            self.head_times.append((self.now, node))
        else:
            self.serving[node] = ("x", self.now + float(first))
            self._push(self.now + float(first), 0, node, 0.0)

    def advance(self, until: float, tol: float) -> None:
        """Process every exogenous event up to ``until`` (ties included)."""
        while self.events and self.events[0][0] <= until + tol:
            t, kind, _, node, value = heapq.heappop(self.events)
            self.now = t
            if kind == 0:
                self.serving[node] = None
                self._next(node)
            elif self.count(node) < self.kappa[node]:
                self.line[node].append(value)
                self._next(node)
            self._mark(node)
        self.now = max(self.now, until)

    def join(self, node: int) -> None:
        self.line[node].append("ev")
        self._next(node)
        self._mark(node)

    def start(self, node: int) -> bool:
        """The EV starts charging; False if the charger is not free for it."""
        serving = self.serving[node]
        if serving is None and not self.line[node]:
            self.serving[node] = ("ev", self.now)
            self._mark(node)
            return True
        return serving is not None and serving[0] == "ev"

    def leave(self, node: int) -> None:
        if "ev" in self.line[node]:
            self.line[node].remove("ev")
        serving = self.serving[node]
        if serving is not None and serving[0] == "ev":
            self.serving[node] = None
            self._next(node)
        self._mark(node)


def validate_ev(
    log: EventLog,
    spec: ProblemSpec,
    instance: Instance,
    *,
    scenario: Scenario | None,
    tol: float = 1e-7,
) -> Any:
    """Re-check an EV shortest-path episode (see the module docstring); a `ValidationReport`."""
    from fleetgym.testing.validator import ValidationReport, Violation

    report = ValidationReport()
    if scenario is None:
        raise ValueError("the EV shortest path needs the episode's scenario")
    params = ev_shortest_path(spec.ev)
    assert params is not None

    def fail(code: str, message: str, row: dict[str, Any] | None = None) -> None:
        report.violations.append(
            Violation(
                code,
                message,
                seq=row.get("seq") if row else None,
                node=row.get("node") if row else None,
            )
        )

    if scenario.instance_hash != instance.content_hash():
        fail("identity", "the scenario belongs to another instance")
    extra = instance.extra
    coords = np.asarray(instance.coords, dtype=float)
    speed = float(spec.network.params.model_dump()["speed"])
    capacity = float(extra["battery_capacity"])
    consumption = float(extra["consumption"])
    levels = np.column_stack([np.asarray(extra[f"charge_level_{k}"], float) for k in (1, 2, 3)])
    times = np.column_stack([np.asarray(extra[f"charge_time_{k}"], float) for k in (1, 2, 3)])
    queues = _Queues(instance, scenario)

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

    battery = capacity
    node: int | None = 0
    position = (float(coords[0, 0]), float(coords[0, 1]))
    leg: dict[str, Any] | None = None  # the current drive
    charging: dict[str, Any] | None = None
    costs = {"drive": 0.0, "wait": 0.0, "charge": 0.0}
    finished_at: float | None = None
    rewards = 0.0

    def where_now(t: float) -> tuple[tuple[float, float], float]:
        if leg is None:
            return position, battery
        frac = 0.0 if leg["duration"] <= 0 else min(1.0, (t - leg["time"]) / leg["duration"])
        x0, y0 = leg["start"]
        x1, y1 = coords[leg["node"]]
        driven = frac * leg["distance"]
        return (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)), leg["battery"] - consumption * driven

    for row in log.rows:
        kind = row["kind"]
        t = _num(row, "time")
        if kind == "epoch":
            rewards += float(row.get("reward") or 0.0)
            continue
        if not math.isnan(t) and kind != "episode_start":
            queues.advance(t, tol)
        if kind in ("depart", "divert"):
            here, level = where_now(t)
            if leg is not None:
                costs["drive"] += t - leg["time"]
            start = (_num(row, "quantity"), _num(row, "amount"))
            if not (close(start[0], here[0]) and close(start[1], here[1])):
                fail("position", f"{kind} from {start}, the EV is at {here}", row)
            if not close(_num(row, "load"), level):
                fail("battery", f"{kind} with battery {row['load']}, the EV has {level}", row)
            target = int(row["node"])
            distance = float(math.hypot(coords[target, 0] - start[0], coords[target, 1] - start[1]))
            if not close(_num(row, "value"), distance):
                fail("distance", f"logged {row['value']}, recomputed {distance}", row)
            if consumption * distance > level + tol * max(1.0, level):
                fail(
                    "energy", f"node {target} needs {consumption * distance}, battery {level}", row
                )
            if node is not None and kind == "depart":
                queues.leave(node) if node in queues.serving else None
            leg = {
                "time": t,
                "start": start,
                "node": target,
                "distance": distance,
                "duration": distance / speed,
                "battery": level,
            }
            node = None
            battery = level
        elif kind == "arrive":
            if leg is None or int(row["node"]) != leg["node"]:
                fail("arrive", "arrival without a drive to this node", row)
                continue
            expected = leg["time"] + leg["duration"]
            if not close(t, expected):
                fail("travel_time", f"arrival at {t}, expected {expected}", row)
            battery = leg["battery"] - consumption * leg["distance"]
            if battery < -tol:
                fail("energy", f"battery {battery} below 0 on arrival", row)
            if not close(_num(row, "load"), battery):
                fail("battery", f"arrival battery {row['load']}, recomputed {battery}", row)
            costs["drive"] += t - leg["time"]
            node = leg["node"]
            position = (float(coords[node, 0]), float(coords[node, 1]))
            leg = None
        elif kind == "queue_seen":
            seen = queues.count(int(row["node"]))
            if int(row["value"]) != seen:
                fail("queue", f"queue of {row['value']} logged, the replay has {seen}", row)
        elif kind == "queue_join":
            n = int(row["node"])
            if node != n:
                fail("queue", "joining a queue away from the station", row)
            full = queues.count(n) >= queues.kappa[n]
            energy = consumption * np.hypot(coords[:, 0] - position[0], coords[:, 1] - position[1])
            others = np.ones(len(coords), dtype=bool)
            others[0] = others[n] = False  # never the origin, nor the station itself
            reachable = bool(np.any(energy[others] <= battery + tol))
            if full and params.queues.full == "leave" and reachable:
                fail("queue", f"joined a full station {n}", row)
            queues.join(n)
        elif kind == "queue_head":
            if (t, int(row["node"])) not in [(a, b) for a, b in queues.head_times if close(a, t)]:
                fail("queue", f"the EV is not first at station {row['node']} at {t}", row)
        elif kind == "station_leave":
            queues.leave(int(row["node"]))
        elif kind == "charge_start":
            n = int(row["node"])
            if node != n:
                fail("charge", "charging away from the station", row)
            if not queues.start(n):
                fail("charge", f"the charger of station {n} is busy at {t}", row)
            target = _num(row, "value")
            if not close(_num(row, "load"), battery):
                fail("battery", f"charging from {row['load']}, the EV has {battery}", row)
            if target > capacity + tol or target <= battery - tol:
                fail(
                    "charge", f"target {target} with battery {battery} and capacity {capacity}", row
                )
            charging = {
                "time": t,
                "node": n,
                "target": target,
                "duration": _phi(levels[n], times[n], target) - _phi(levels[n], times[n], battery),
            }
        elif kind == "charge_end":
            if charging is None:
                fail("charge", "end of a charge that did not start", row)
                continue
            expected = charging["time"] + charging["duration"]
            if not close(t, expected):
                fail("charge", f"charge ends at {t}, expected {expected}", row)
            costs["charge"] += t - charging["time"]
            battery = charging["target"]
            if not close(_num(row, "load"), battery):
                fail("battery", f"charged to {row['load']}, expected {battery}", row)
            charging = None
        elif kind == "episode_end":
            finished_at = t
            if node != 1:
                fail("destination", "the episode ended away from the destination", row)
    if finished_at is not None:
        costs["wait"] = finished_at - costs["drive"] - costs["charge"]
        logged = [
            (float(r["time"]), int(r["node"]), float(r["value"])) for r in log.of_kind("indicator")
        ]
        replay = [c for c in queues.changes if c[0] <= finished_at + tol]
        if len(logged) != len(replay) or any(
            a[1] != b[1] or a[2] != b[2] or not close(a[0], b[0])
            for a, b in zip(logged, replay, strict=False)
        ):
            fail(
                "indicator", f"{len(logged)} indicator changes logged, the replay has {len(replay)}"
            )
    report.costs = {k: v for k, v in costs.items() if v != 0.0}
    weights = spec.objective.scalar_weights()
    expected_return = -sum(weights.get(k, 0.0) * v for k, v in costs.items())
    report.episode_return = rewards
    report.expected_return = expected_return
    if finished_at is not None and not close(rewards, expected_return):
        fail("reward", f"sum of rewards {rewards}, expected {expected_return}")
    return report
