"""`EVShortestPathSimulator`: one EV, charging stations with exogenous queues (M14, ADR 0026).

The EV shortest path with charging-station occupancy of Dastpak, Errico, Jabali and Malucelli
(arXiv 2305.11773, Section 3; proposal 0003). Node 0 is the origin, node 1 the destination, the
other nodes are charging stations, each with one charger and an M/M/1/kappa queue of exogenous EVs
replayed from the scenario (`fleetgym.uncertainty.queues`).

Decision epochs (Section 3.2): the EV leaves the origin; an occupancy indicator changes while the
EV drives or waits (none while it charges); the EV arrives at a station; the EV reaches the head of
the queue; the target level is reached. A decision is a `EVMove` (drive to a node, from the current
position: diversion while driving is allowed) or an `EVStay` (charge at the current station up to
a target level, waiting in the queue first if the charger is busy).

Rules:

* energy: driving ``d`` costs ``consumption * d``; the battery never goes below 0;
* charging from ``b`` to ``y`` at station ``i`` takes ``Phi_i(y) - Phi_i(b)``, ``Phi_i`` piecewise
  linear through ``(0, 0)`` and the station's breakpoints (``charge_level_k``, ``charge_time_k``);
* queues are first come, first served; the EV counts toward ``kappa`` once it has joined; with
  ``queues.full: leave`` it cannot join a full station, unless no other node is reachable (then it
  joins anyway, so the episode can always go on);
* the EV holds the charger while it decides, at the head of the queue or after a charge; leaving
  frees it for the next EV;
* costs: ``drive``, ``wait`` (time at a station without charging) and ``charge``; their sum is the
  arrival time at the destination.
"""

from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from fleetgym.data import Instance
from fleetgym.errors import InfeasibleActionError, InstanceError
from fleetgym.eventlog import EventLog
from fleetgym.spec.models import ProblemSpec
from fleetgym.spec.models.ev import EVShortestPathParams, ev_shortest_path
from fleetgym.uncertainty.scenario import Scenario

__all__ = ["EVEpoch", "EVMove", "EVShortestPathSimulator", "EVState", "EVStay", "charging_time"]

ORIGIN, DESTINATION = 0, 1
_TOL = 1e-9
Status = Literal["origin", "driving", "arrived", "waiting", "head", "charging", "charged", "done"]

# event kinds, in processing order at equal times
_SERVICE_END, _EXO_ARRIVAL, _EV_ARRIVAL, _CHARGE_END = 0, 1, 2, 3


@dataclass(frozen=True)
class EVMove:
    """Drive to ``node`` (a station or the destination) from where the EV is."""

    node: int


@dataclass(frozen=True)
class EVStay:
    """Charge at the current station up to ``level`` (waiting first if the charger is busy)."""

    level: float


@dataclass(frozen=True)
class EVEpoch:
    """A decision epoch: why it happened (``kind``) and when."""

    time: float
    kind: str


@dataclass
class StationState:
    """One charger and its queue: ``in_service`` holds ``("x", end)``, ``("ev", end)`` or None."""

    in_service: tuple[str, float] | None = None
    queue: list[float | str] = field(default_factory=list)  # exogenous charging times, or "ev"

    def count(self) -> int:
        """EVs at the station (charging or waiting)."""
        return (self.in_service is not None) + len(self.queue)


@dataclass
class EVState:
    """Everything that changes during an episode (a deep copy is an exact snapshot)."""

    time: float
    battery: float
    status: Status
    node: int | None
    position: tuple[float, float]
    stations: dict[int, StationState]
    indicator: NDArray[np.float64]
    changed_at: NDArray[np.float64]
    heap: list[tuple[float, int, int, int, float]]
    seq: int = 0
    trip: int = 0
    target: int | None = None
    depart_time: float = 0.0
    depart_battery: float = 0.0
    depart_position: tuple[float, float] = (0.0, 0.0)
    arrive_time: float = 0.0
    charge_level: float = 0.0
    observed_queue: int = -1
    total_costs: dict[str, float] = field(default_factory=dict)
    pending_costs: dict[str, float] = field(default_factory=dict)
    epoch_count: int = 0
    terminated: bool = False

    def take_pending_costs(self) -> dict[str, float]:
        """Return and clear the costs realized since the last epoch."""
        out, self.pending_costs = self.pending_costs, {}
        return out


def charging_time(levels: NDArray[np.float64], times: NDArray[np.float64], b: float) -> float:
    """``Phi(b)``: time to charge from 0 to ``b`` (extrapolated past the last breakpoint)."""
    xs = np.concatenate([[0.0], levels])
    ys = np.concatenate([[0.0], times])
    if b <= xs[-1]:
        return float(np.interp(b, xs, ys))
    slope = (ys[-1] - ys[-2]) / max(xs[-1] - xs[-2], _TOL)
    return float(ys[-1] + slope * (b - xs[-1]))


class EVShortestPathSimulator:
    """Simulator of the EV shortest path with occupancy indicators.

    Parameters
    ----------
    spec
        A spec with ``ev.shortest_path`` and a Euclidean network.
    instance
        An instance in schema ``fleetgym/evspp/1``.
    scenario
        The episode's exogenous EVs (`fleetgym.uncertainty.queues.sample_stations`).
    log
        Event log to record into.
    """

    def __init__(
        self,
        spec: ProblemSpec,
        instance: Instance,
        scenario: Scenario,
        *,
        log: EventLog | None = None,
    ) -> None:
        params = ev_shortest_path(spec.ev)
        assert params is not None
        self.spec = spec
        self.params: EVShortestPathParams = params
        self.instance = instance
        self.scenario = scenario
        self.log = log if log is not None else EventLog()
        if spec.network.name != "euclidean" or instance.coords is None:
            raise InstanceError("the EV shortest path needs coordinates and a Euclidean network")
        extra = instance.extra
        role = np.asarray(extra["role"], dtype=object)
        if role[ORIGIN] != "origin" or role[DESTINATION] != "destination":
            raise InstanceError(
                f"{instance.name}: node 0 must be the origin and node 1 the destination"
            )
        self.n = instance.n_nodes
        self.stations = [i for i in range(self.n) if role[i] == "station"]
        self._station_set = set(self.stations)
        self.coords = np.asarray(instance.coords, dtype=float)
        self.speed = float(spec.network.params.model_dump()["speed"])
        self.capacity = float(extra["battery_capacity"])
        self.consumption = float(extra["consumption"])
        self.kappa = np.asarray(extra["queue_capacity"], dtype=float)
        self.levels = np.column_stack(
            [np.asarray(extra[f"charge_level_{k}"], float) for k in (1, 2, 3)]
        )
        self.times = np.column_stack(
            [np.asarray(extra[f"charge_time_{k}"], float) for k in (1, 2, 3)]
        )
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.distances = np.sqrt((diff**2).sum(-1))
        self.state = self._initial_state()

    # ------------------------------------------------------------------ setup

    def _initial_state(self) -> EVState:
        sc = self.scenario
        stations = {i: StationState() for i in self.stations}
        heap: list[tuple[float, int, int, int, float]] = []
        seq = 0
        services = iter(np.asarray(sc["ev.initial_service"], dtype=float).tolist())
        counts = np.asarray(sc["ev.initial"], dtype=float)
        for i in self.stations:
            present = int(counts[i])
            durations = [next(services) for _ in range(present)]
            if durations:
                stations[i].in_service = ("x", durations[0])
                heapq.heappush(heap, (durations[0], _SERVICE_END, seq, i, 0.0))
                seq += 1
                stations[i].queue = list(durations[1:])
        for t, node, s in zip(
            np.asarray(sc["ev.arrival"], float),
            np.asarray(sc["ev.arrival_node"], float),
            np.asarray(sc["ev.arrival_service"], float),
            strict=True,
        ):
            heapq.heappush(heap, (float(t), _EXO_ARRIVAL, seq, int(node), float(s)))
            seq += 1
        indicator = np.zeros(self.n)
        for i in self.stations:
            indicator[i] = float(stations[i].count() > 0)
        return EVState(
            time=0.0,
            battery=self.capacity,  # the EV leaves the origin fully charged (the user's choice)
            status="origin",
            node=ORIGIN,
            position=(float(self.coords[ORIGIN, 0]), float(self.coords[ORIGIN, 1])),
            stations=stations,
            indicator=indicator,
            changed_at=np.zeros(self.n),
            heap=heap,
            seq=seq,
        )

    def start(self) -> EVEpoch:
        """Start the episode: the first epoch is the departure from the origin."""
        self.log.append("episode_start", time=0.0)
        if not self._destination_reachable():
            raise InstanceError(
                f"{self.instance.name}: the destination cannot be reached, even charging to full "
                "at every station"
            )
        self.state.epoch_count += 1
        return EVEpoch(0.0, "origin")

    def _destination_reachable(self) -> bool:
        """Whether hops of at most a full battery lead from the origin to the destination."""
        hop = self.consumption * self.distances <= self.capacity + _TOL
        seen, frontier = {ORIGIN}, [ORIGIN]
        while frontier:
            node = frontier.pop()
            for nxt in np.flatnonzero(hop[node]):
                nxt = int(nxt)
                if nxt == DESTINATION:
                    return True
                if nxt not in seen and nxt in self._station_set:
                    seen.add(nxt)
                    frontier.append(nxt)
        return False

    # ------------------------------------------------------------------ queries

    def current_battery(self) -> float:
        """Battery now (decreasing linearly while driving)."""
        s = self.state
        if s.status != "driving":
            return s.battery
        driven = (s.time - s.depart_time) * self.speed
        return s.depart_battery - self.consumption * driven

    def current_position(self) -> tuple[float, float]:
        """Where the EV is now (on the straight line while driving)."""
        s = self.state
        if s.status != "driving" or s.target is None:
            return s.position
        span = s.arrive_time - s.depart_time
        frac = 0.0 if span <= 0 else min(1.0, (s.time - s.depart_time) / span)
        x0, y0 = s.depart_position
        x1, y1 = self.coords[s.target]
        return (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0))

    def distances_from_here(self) -> NDArray[np.float64]:
        """Distance from the EV to every node."""
        x, y = self.current_position()
        return np.hypot(self.coords[:, 0] - x, self.coords[:, 1] - y)

    def move_mask(self) -> NDArray[np.bool_]:
        """Nodes the EV may drive to now (reachable with its battery; never the origin)."""
        s = self.state
        mask = np.zeros(self.n, dtype=bool)
        if s.status in ("done", "charging"):
            return mask
        energy = self.consumption * self.distances_from_here()
        mask[:] = energy <= self.current_battery() + _TOL
        mask[ORIGIN] = False
        if s.node is not None and s.status != "driving":
            mask[s.node] = False  # staying is an EVStay
        return mask

    def may_stay(self) -> bool:
        """Whether the EV may charge (or wait to charge) at its current station now."""
        s = self.state
        if s.status in ("waiting", "head", "charged"):
            return True
        if s.status != "arrived" or s.node is None:
            return False
        station = s.stations[s.node]
        if station.count() < self.kappa[s.node] or self.params.queues.full == "join":
            return True
        return not np.any(self.move_mask())  # stranded at a full station: join anyway

    def phi(self, node: int, level: float) -> float:
        """``Phi_node(level)``."""
        return charging_time(self.levels[node], self.times[node], level)

    def station_levels(self, node: int) -> NDArray[np.float64]:
        """The charging breakpoints of a station."""
        return self.levels[node]

    # ------------------------------------------------------------------ decisions

    def apply(self, decision: EVMove | EVStay) -> tuple[EVEpoch | None, dict[str, float]]:
        """Apply a decision and run to the next epoch.

        Returns the next epoch (None when the EV reached the destination) and the costs realized.

        Raises
        ------
        InfeasibleActionError
            If the decision is not allowed now.
        """
        s = self.state
        if s.terminated:
            raise InfeasibleActionError("the episode has ended")
        if isinstance(decision, EVMove):
            self._move(decision.node)
        else:
            self._stay(decision.level)
        epoch = self._advance()
        return epoch, s.take_pending_costs()

    def _move(self, node: int) -> None:
        s = self.state
        if not (0 <= node < self.n) or not self.move_mask()[node]:
            raise InfeasibleActionError(f"node {node} cannot be reached now")
        battery = self.current_battery()
        position = self.current_position()
        if s.status in ("waiting", "head", "charged", "arrived") and s.node is not None:
            self._leave_station(s.node)
        distance = float(np.hypot(*(self.coords[node] - np.asarray(position))))
        if s.status == "driving" and s.target == node:
            return  # keep going
        kind = "divert" if s.status == "driving" else "depart"
        s.trip += 1
        s.status = "driving"
        s.node = None
        s.target = node
        s.depart_time = s.time
        s.depart_battery = battery
        s.depart_position = position
        s.arrive_time = s.time + distance / self.speed
        self.log.append(
            kind,
            time=s.time,
            node=node,
            value=distance,
            load=battery,
            quantity=position[0],
            amount=position[1],
        )
        self._push(s.arrive_time, _EV_ARRIVAL, node, float(s.trip))

    def _stay(self, level: float) -> None:
        s = self.state
        if not self.may_stay() or s.node is None:
            raise InfeasibleActionError("the EV cannot charge here now")
        level = min(float(level), self.capacity)
        if level <= s.battery + _TOL:
            raise InfeasibleActionError(
                f"target level {level} is not above the battery {s.battery}"
            )
        node = s.node
        station = s.stations[node]
        s.charge_level = level
        if s.status == "arrived":
            if station.count() == 0:
                self._start_charging(node)
            else:
                station.queue.append("ev")
                s.status = "waiting"
                self.log.append("queue_join", time=s.time, node=node, value=float(station.count()))
                self._update_indicator(node)
        elif s.status in ("head", "charged"):
            self._start_charging(node)
        # waiting: the new target applies when the charger is free

    def _start_charging(self, node: int) -> None:
        s = self.state
        station = s.stations[node]
        duration = self.phi(node, s.charge_level) - self.phi(node, s.battery)
        end = s.time + max(duration, 0.0)
        station.in_service = ("ev", end)
        s.status = "charging"
        self.log.append(
            "charge_start", time=s.time, node=node, load=s.battery, value=s.charge_level
        )
        self._push(end, _CHARGE_END, node, s.charge_level)
        self._update_indicator(node)

    def _leave_station(self, node: int) -> None:
        s = self.state
        station = s.stations[node]
        if "ev" in station.queue:
            station.queue.remove("ev")
        if station.in_service is not None and station.in_service[0] == "ev":
            station.in_service = None
        self.log.append("station_leave", time=s.time, node=node)
        if station.in_service is None:
            self._serve_next(node)
        self._update_indicator(node)

    # ------------------------------------------------------------------ events

    def _push(self, time: float, kind: int, node: int, value: float) -> None:
        s = self.state
        heapq.heappush(s.heap, (time, kind, s.seq, node, value))
        s.seq += 1

    def _serve_next(self, node: int) -> None:
        """Give the free charger to the next EV in line."""
        s = self.state
        station = s.stations[node]
        if station.in_service is not None or not station.queue:
            return
        head = station.queue[0]
        if head == "ev":
            station.queue.pop(0)
            station.in_service = ("ev", s.time)  # held while the EV decides
            s.status = "head"
            self.log.append("queue_head", time=s.time, node=node)
            return
        station.queue.pop(0)
        end = s.time + float(head)
        station.in_service = ("x", end)
        self._push(end, _SERVICE_END, node, 0.0)

    def _update_indicator(self, node: int) -> bool:
        s = self.state
        value = float(s.stations[node].count() > 0)
        if value == s.indicator[node]:
            return False
        s.indicator[node] = value
        s.changed_at[node] = s.time
        self.log.append("indicator", time=s.time, node=node, value=value)
        return True

    def _accrue(self, until: float) -> None:
        s = self.state
        dt = until - s.time
        if dt <= 0:
            return
        component = {"driving": "drive", "waiting": "wait", "charging": "charge"}.get(s.status)
        if component is not None:
            for costs in (s.total_costs, s.pending_costs):
                costs[component] = costs.get(component, 0.0) + dt
        s.time = until

    def _advance(self) -> EVEpoch | None:
        s = self.state
        while s.heap:
            time, kind, _, node, value = heapq.heappop(s.heap)
            if kind == _EV_ARRIVAL and int(value) != s.trip:
                continue  # an arrival planned before a diversion
            self._accrue(time)
            if kind == _SERVICE_END:
                station = s.stations[node]
                station.in_service = None
                self._serve_next(node)
                changed = self._update_indicator(node)
                if s.status == "head" and s.node == node:
                    return self._epoch("available")
                if changed and s.status in ("driving", "waiting"):
                    return self._epoch("indicator")
            elif kind == _EXO_ARRIVAL:
                station = s.stations[node]
                if station.count() >= self.kappa[node]:
                    continue  # a full station: the exogenous EV leaves
                station.queue.append(value)
                self._serve_next(node)
                if self._update_indicator(node) and s.status in ("driving", "waiting"):
                    return self._epoch("indicator")
            elif kind == _EV_ARRIVAL:
                return self._arrive(node)
            elif kind == _CHARGE_END:
                s.battery = value
                s.status = "charged"
                self.log.append("charge_end", time=s.time, node=node, load=s.battery)
                return self._epoch("charged")
        raise InfeasibleActionError("no event left: the EV is stuck")  # pragma: no cover

    def _arrive(self, node: int) -> EVEpoch | None:
        s = self.state
        s.battery = (
            s.depart_battery - self.consumption * (s.arrive_time - s.depart_time) * self.speed
        )
        s.battery = max(s.battery, 0.0) if s.battery > -_TOL else s.battery
        s.position = (float(self.coords[node, 0]), float(self.coords[node, 1]))
        s.node = node
        s.target = None
        self.log.append("arrive", time=s.time, node=node, load=s.battery)
        if node == DESTINATION:
            s.status = "done"
            s.terminated = True
            self.log.append("episode_end", time=s.time)
            return None
        s.status = "arrived"
        s.observed_queue = s.stations[node].count()
        self.log.append("queue_seen", time=s.time, node=node, value=float(s.observed_queue))
        return self._epoch("arrival")

    def _epoch(self, kind: str) -> EVEpoch:
        self.state.epoch_count += 1
        return EVEpoch(self.state.time, kind)

    # ------------------------------------------------------------------ snapshots

    def snapshot(self) -> EVState:
        """An exact copy of the state."""
        return copy.deepcopy(self.state)

    def restore(self, state: EVState) -> None:
        """Restore a snapshot."""
        self.state = copy.deepcopy(state)

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - debugging aid
        """A summary of the state."""
        s = self.state
        return {
            "time": s.time,
            "battery": self.current_battery(),
            "status": s.status,
            "node": s.node,
        }
