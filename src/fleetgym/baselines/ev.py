"""Baselines for the EV shortest path with occupancy indicators (M14, ADR 0026).

* `EVGreedy`: drive to the destination when it is reachable; otherwise to the reachable station
  closest to the destination, and charge to full there (waiting if needed).
* `EVSteadyState`: the benchmark of Dastpak et al. (Section 5.2, "RSS") in a simple form: it
  ignores the occupancy indicators and re-plans only on arrival at a station, at the head of a
  queue and after charging. A plan is a shortest path in time where each station costs its
  steady-state waiting time ``W_j = L_j / mu_j`` plus charging to full; at the station where the EV
  is, the observed queue ``xi`` costs ``xi / mu``. On the last leg it charges just enough (to the
  first grid level that reaches the destination). The paper's benchmark optimizes charging amounts
  with frvcpy; this one charges to full, so it is a baseline, not the paper's benchmark.
"""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fleetgym.envs.ev_env import EVShortestPathEnv
from fleetgym.sim.ev import DESTINATION
from fleetgym.uncertainty.queues import steady_state_wait

__all__ = ["EVGreedy", "EVSteadyState"]


class _EVPolicy:
    name = "ev_policy"
    anticipative = False

    def __init__(self) -> None:
        self._env: EVShortestPathEnv | None = None

    @property
    def env(self) -> EVShortestPathEnv:
        """The bound environment."""
        if self._env is None:
            raise RuntimeError(f"{self.name}: call reset(env) first")
        return self._env

    def reset(self, env: Any) -> None:
        """Bind to an `EVShortestPathEnv`."""
        if not isinstance(env, EVShortestPathEnv):
            raise TypeError(f"{self.name} needs an EVShortestPathEnv")
        self._env = env

    def _move(self, node: int) -> int:
        return node * self.env.slots

    def _charge(self, level: float) -> int:
        """The action that charges to the first grid level at least ``level`` (or to full)."""
        env = self.env
        sim = env.simulator
        node = sim.state.node
        assert node is not None
        levels = env.params.target_levels
        k = int(np.ceil(min(level, sim.capacity) / sim.capacity * levels - 1e-9))
        k = min(max(k, 1), levels)
        while k < levels and sim.capacity * k / levels <= sim.state.battery + 1e-9:
            k += 1
        return node * env.slots + k

    def _fallback(self, mask: NDArray[np.bool_]) -> int:
        return int(np.flatnonzero(mask)[0])


class EVGreedy(_EVPolicy):
    """Nearest-to-destination reachable station, charge to full (see the module docstring)."""

    name = "ev_greedy"

    def act(self, obs: dict[str, Any], info: dict[str, Any]) -> int:
        """The greedy action."""
        env = self.env
        sim = env.simulator
        mask = env.action_masks()
        s = sim.state
        moves = sim.move_mask()
        if moves[DESTINATION]:
            return self._move(DESTINATION)
        if s.status == "driving" and s.target is not None and mask[self._move(s.target)]:
            return self._move(s.target)
        at_station = s.status in ("arrived", "waiting", "head", "charged")
        if at_station and sim.may_stay() and sim.state.battery < sim.capacity - 1e-9:
            action = self._charge(sim.capacity)
            if mask[action]:
                return action
        to_destination = sim.distances[:, DESTINATION]
        stations = [i for i in sim.stations if moves[i]]
        if stations:
            return self._move(min(stations, key=lambda i: to_destination[i]))
        return self._fallback(mask)


class EVSteadyState(_EVPolicy):
    """Re-optimization with steady-state waiting times (see the module docstring)."""

    name = "ev_steady_state"

    def reset(self, env: Any) -> None:
        """Bind and precompute the steady-state waits."""
        super().reset(env)
        self._plan_for: str | None = None

    def _waits(self) -> NDArray[np.float64]:
        sim = self.env.simulator
        extra = sim.instance.extra
        lam = np.asarray(extra["arrival_rate"], float)
        mu = np.asarray(extra["service_rate"], float)
        waits = np.zeros(sim.n)
        for i in sim.stations:
            waits[i] = steady_state_wait(lam[i], mu[i], int(sim.kappa[i]))
        return waits

    def _shortest(
        self, start_battery: float, origin_dist: NDArray[np.float64], skip: int | None
    ) -> tuple[float, int]:
        """Least expected time to the destination from here, and the first node of the path."""
        sim = self.env.simulator
        waits = self._waits()
        cap = sim.capacity
        speed = sim.speed
        e = sim.consumption
        best = {}
        heap: list[tuple[float, int, int]] = []
        for j in [*sim.stations, DESTINATION]:
            if j == skip or e * origin_dist[j] > start_battery + 1e-9:
                continue
            cost = origin_dist[j] / speed
            if j != DESTINATION:
                arrive = start_battery - e * origin_dist[j]
                cost += waits[j] + sim.phi(j, cap) - sim.phi(j, max(arrive, 0.0))
            heapq.heappush(heap, (cost, j, j))
        while heap:
            cost, node, first = heapq.heappop(heap)
            if node in best:
                continue
            best[node] = (cost, first)
            if node == DESTINATION:
                return cost, first
            for j in [*sim.stations, DESTINATION]:
                if j in best or e * sim.distances[node, j] > cap + 1e-9:
                    continue
                step = sim.distances[node, j] / speed
                if j != DESTINATION:
                    arrive = cap - e * sim.distances[node, j]
                    step += waits[j] + sim.phi(j, cap) - sim.phi(j, arrive)
                heapq.heappush(heap, (cost + step, j, first))
        return float("inf"), -1

    def act(self, obs: dict[str, Any], info: dict[str, Any]) -> int:
        """Keep the plan between re-optimizations; re-plan at arrivals and after charging."""
        env = self.env
        sim = env.simulator
        s = sim.state
        mask = env.action_masks()
        moves = sim.move_mask()
        if moves[DESTINATION]:
            return self._move(DESTINATION)
        if s.status == "driving":
            assert s.target is not None
            return self._move(s.target)  # RSS does not react to indicators
        if s.status == "waiting":
            return self._charge(s.charge_level or sim.capacity)  # keep waiting
        here = sim.distances_from_here()
        leave_cost, leave_first = self._shortest(sim.current_battery(), here, skip=s.node)
        stay_action = None
        if s.status in ("arrived", "head", "charged") and sim.may_stay() and s.node is not None:
            node = s.node
            mu = float(np.asarray(sim.instance.extra["service_rate"], float)[node])
            queue = 0.0 if s.status in ("head", "charged") else s.observed_queue / mu
            need = sim.consumption * sim.distances[node, DESTINATION]
            if need <= sim.capacity + 1e-9:
                target = need
                rest = sim.distances[node, DESTINATION] / sim.speed
            else:
                target = sim.capacity
                rest, _ = self._shortest(sim.capacity, sim.distances[node], skip=node)
            stay_cost = queue + sim.phi(node, target) - sim.phi(node, s.battery) + rest
            action = self._charge(target)
            if mask[action] and stay_cost <= leave_cost:
                stay_action = action
        if stay_action is not None:
            return stay_action
        if leave_first >= 0 and mask[self._move(leave_first)]:
            return self._move(leave_first)
        return self._fallback(mask)
