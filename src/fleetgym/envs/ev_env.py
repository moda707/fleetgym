"""`EVShortestPathEnv`: the EV shortest path with charging-station occupancy (M14, ADR 0026).

A Gymnasium environment over `fleetgym.sim.ev.EVShortestPathSimulator` (proposal 0003). It has the
surface of `FleetGymEnv` that the evaluation tools use: ``action_masks``, ``event_log``,
``simulator``, ``scenario``, ``reward_model``, ``fork``, ``clone``, ``get_state``/``set_state``.

Actions: ``Discrete(max_nodes * slots)`` with ``slots = target_levels + 4``; action
``node * slots + slot``.

* ``slot 0``: drive to ``node`` (a station or the destination; from a station, or diverting while
  driving; choosing the current target while driving keeps going);
* ``slot 1..L``: at the current station, charge up to ``k * B / L`` (``L = target_levels``);
* ``slot L+1..L+3``: at the current station, charge up to its breakpoint 1, 2 or 3.

Charging at a busy station first waits in the queue. Observations: ``ev`` (battery / B, time,
position, status, queue length seen at the current station) and ``nodes`` (per node: position,
role, occupancy indicator and the time since it last changed, queue parameters, distance and
energy needed from the EV, padding flag). Exogenous arrivals and charging times are hidden.
"""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from fleetgym.data import Instance
from fleetgym.errors import ConfigError, InfeasibleActionError, NotInScopeError
from fleetgym.eventlog import EventLog
from fleetgym.interfaces.rewards import RewardModel
from fleetgym.sim.ev import DESTINATION, ORIGIN, EVEpoch, EVMove, EVShortestPathSimulator, EVStay
from fleetgym.spec.binding import instance_from_source
from fleetgym.spec.models import ProblemSpec
from fleetgym.spec.models.ev import ev_shortest_path
from fleetgym.uncertainty.queues import sample_stations, stationary_distribution
from fleetgym.uncertainty.scenario import Scenario
from fleetgym.uncertainty.streams import StreamFactory

__all__ = ["EVShortestPathEnv"]

_SEED_LIMIT = 2**63
_NODE_FEATURES = 13
_EV_FEATURES = 9
_BOUND = 1e9


def _derive_seed(root: int, index: int) -> int:
    ss = np.random.SeedSequence(entropy=root, spawn_key=(index,))
    return int(ss.generate_state(1, np.uint64)[0]) % _SEED_LIMIT


class EVShortestPathEnv(gym.Env[dict[str, NDArray[Any]], np.int64]):
    """Gymnasium environment of the EV shortest path with occupancy indicators.

    Parameters
    ----------
    spec
        A spec with ``ev.shortest_path`` (for example ``EVSPP-v1``).
    instance
        A fixed instance (schema ``fleetgym/evspp/1``); otherwise each episode draws one from the
        spec's instance source.
    max_nodes
        Node slots of observations and actions (default: the largest instance of the source).
    render_mode
        ``"ansi"``: `render` returns a one-line text summary.
    invalid_action
        ``"mask"``: an invalid action raises `InfeasibleActionError`. ``"repair"``: it is replaced
        by the first valid action.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}  # noqa: RUF012

    def __init__(
        self,
        spec: ProblemSpec,
        *,
        instance: Instance | None = None,
        max_nodes: int | None = None,
        render_mode: str | None = None,
        invalid_action: str = "mask",
    ) -> None:
        super().__init__()
        params = ev_shortest_path(spec.ev)
        if params is None:
            raise ConfigError("EVShortestPathEnv needs the `ev.shortest_path` section")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise NotInScopeError(f"render_mode {render_mode!r} is not available for the EV")
        if invalid_action not in ("mask", "repair"):
            raise ValueError(f"invalid_action must be mask or repair, got {invalid_action!r}")
        self.render_mode = render_mode
        self.invalid_action = invalid_action
        self.spec_ = spec
        self.params = params
        self._fixed_instance = instance
        if max_nodes is None:
            if instance is not None:
                max_nodes = instance.n_nodes
            else:
                from fleetgym.envs.env import _source_instances  # pyright: ignore[reportPrivateUsage]

                sizes = [i.n_nodes for i in _source_instances(spec)]
                if not sizes:
                    raise ConfigError("give `instance=` or `max_nodes=` (the source has no files)")
                max_nodes = max(sizes)
        self.max_nodes = int(max_nodes)
        self.slots = params.target_levels + 4
        self.action_space = gym.spaces.Discrete(self.max_nodes * self.slots)  # pyright: ignore[reportAttributeAccessIssue]
        self.observation_space = gym.spaces.Dict(  # pyright: ignore[reportAttributeAccessIssue]
            {
                "ev": gym.spaces.Box(-_BOUND, _BOUND, (_EV_FEATURES,), np.float32),
                "nodes": gym.spaces.Box(
                    -_BOUND, _BOUND, (self.max_nodes, _NODE_FEATURES), np.float32
                ),
                "action_mask": gym.spaces.Box(0, 1, (self.max_nodes * self.slots,), np.int8),
            }
        )
        self.reward_model = RewardModel(spec.objective)
        self._sim: EVShortestPathSimulator | None = None
        self._epoch: EVEpoch | None = None
        self._root_seed: int | None = None
        self._episode_index = 0
        self._episode_seed = 0

    # ------------------------------------------------------------------ properties

    @property
    def problem_spec(self) -> ProblemSpec:
        """The problem spec."""
        return self.spec_

    @property
    def simulator(self) -> EVShortestPathSimulator:
        """The simulator of the current episode."""
        if self._sim is None:
            raise gym.error.ResetNeeded("call reset() first")
        return self._sim

    @property
    def event_log(self) -> EventLog:
        """The event log of the current episode."""
        return self.simulator.log

    @property
    def scenario(self) -> Scenario:
        """The exogenous EVs of the current episode."""
        return self.simulator.scenario

    @property
    def base_instance(self) -> Instance:
        """The instance of the current episode."""
        return self.simulator.instance

    @property
    def problem(self) -> Any:
        """The spec and the instance of the current episode (``problem.spec``, ``.instance``)."""
        from types import SimpleNamespace

        return SimpleNamespace(spec=self.spec_, instance=self.base_instance)

    @property
    def epoch(self) -> EVEpoch | None:
        """The current decision epoch."""
        return self._epoch

    @property
    def episode_seed(self) -> int:
        """Seed of the current episode."""
        return self._episode_seed

    # ------------------------------------------------------------------ gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        """Start an episode (options: ``instance``, ``scenario``)."""
        if seed is not None and seed < 0:
            raise ValueError(f"seeds must be non-negative, got {seed}")
        super().reset(seed=seed)
        if seed is not None:
            self._root_seed, self._episode_index = int(seed), 0
        elif self._root_seed is None:
            self._root_seed = int(np.random.SeedSequence().entropy) % _SEED_LIMIT  # pyright: ignore[reportArgumentType]
            self._episode_index = 0
        root = self._root_seed
        self._episode_seed = (
            root if self._episode_index == 0 else _derive_seed(root, self._episode_index)
        )
        self._episode_index += 1
        options = options or {}
        unknown = set(options) - {"instance", "scenario"}
        if unknown:
            raise ValueError(f"unknown reset options {sorted(unknown)}")
        instance = options.get("instance") or self._fixed_instance
        if instance is None:
            streams = StreamFactory(self._episode_seed)
            instance = instance_from_source(self.spec_, streams.generator("instance"))
        if instance.n_nodes > self.max_nodes:
            raise ConfigError(
                f"instance {instance.name!r} has {instance.n_nodes} nodes but the environment has "
                f"{self.max_nodes} node slots; pass a larger max_nodes"
            )
        scenario = options.get("scenario")
        if scenario is None:
            values = sample_stations(
                instance, self._episode_seed, self.params.horizon, self.params.queues.initial
            )
            scenario = Scenario(
                self.spec_.content_hash(), instance.content_hash(), values, seed=self._episode_seed
            )
        elif scenario.instance_hash != instance.content_hash():
            from fleetgym.errors import ScenarioMismatchError

            raise ScenarioMismatchError("the scenario belongs to another instance")
        self._sim = EVShortestPathSimulator(self.spec_, instance, scenario)
        self._epoch = self._sim.start()
        return self._observation(), self._info({})

    def step(
        self, action: Any
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        """Apply an action (see the module docstring)."""
        sim = self.simulator
        if sim.state.terminated:
            raise gym.error.ResetNeeded("the episode has ended: call reset()")
        choice = int(action)
        mask = self.action_masks()
        if not (0 <= choice < self.max_nodes * self.slots) or not mask[choice]:
            if self.invalid_action != "repair":
                raise InfeasibleActionError(f"action {choice} is not allowed now")
            choice = int(np.flatnonzero(mask)[0])
        self._epoch, costs = sim.apply(self._decode(choice))
        reward = -self.reward_model.cost(costs)
        for component, amount in sorted(costs.items()):
            sim.log.append("cost", time=sim.state.time, component=component, amount=amount)
        sim.log.append("epoch", epoch=sim.state.epoch_count, reward=reward, time=sim.state.time)
        terminated = sim.state.terminated
        return self._observation(), reward, terminated, False, self._info(costs)

    def render(self) -> str | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """``ansi``: time, battery, status and where the EV is."""
        if self.render_mode != "ansi" or self._sim is None:
            return None
        sim = self._sim
        s = sim.state
        x, y = sim.current_position()
        return (
            f"t={s.time:.2f} battery={sim.current_battery():.2f}/{sim.capacity:g} "
            f"status={s.status} at=({x:.1f}, {y:.1f}) node={s.node}"
        )

    # ------------------------------------------------------------------ actions

    def target_level(self, slot: int) -> float:
        """The target battery level of a charging slot at the current station."""
        sim = self.simulator
        levels = self.params.target_levels
        if 1 <= slot <= levels:
            return sim.capacity * slot / levels
        node = sim.state.node
        assert node is not None
        return float(sim.station_levels(node)[slot - levels - 1])

    def _decode(self, action: int) -> EVMove | EVStay:
        node, slot = divmod(action, self.slots)
        if slot == 0:
            return EVMove(node)
        return EVStay(self.target_level(slot))

    def action_masks(self) -> NDArray[np.bool_]:
        """Valid actions now."""
        sim = self.simulator
        mask = np.zeros((self.max_nodes, self.slots), dtype=bool)
        if sim.state.terminated:
            mask[DESTINATION, 0] = True  # after the end: only "stay at the destination"
            return mask.ravel()
        mask[: sim.n, 0] = sim.move_mask()
        node = sim.state.node
        if node is not None and sim.state.status != "driving" and sim.may_stay():
            battery = sim.state.battery
            for slot in range(1, self.slots):
                mask[node, slot] = self.target_level(slot) > battery + 1e-9
        return mask.ravel()

    # ------------------------------------------------------------------ observations

    def _observation(self) -> dict[str, NDArray[Any]]:
        sim = self.simulator
        s = sim.state
        x, y = sim.current_position()
        status = s.status
        ev = np.array(
            [
                sim.current_battery() / sim.capacity,
                s.time,
                x,
                y,
                status == "driving",
                status == "waiting",
                status in ("arrived", "head", "charged"),
                s.observed_queue if status in ("arrived", "waiting", "head", "charged") else -1,
                float(s.node) if s.node is not None else -1.0,
            ],
            dtype=np.float32,
        )
        n = sim.n
        nodes = np.zeros((self.max_nodes, _NODE_FEATURES), dtype=np.float32)
        extra = sim.instance.extra
        dist = sim.distances_from_here()
        nodes[:n, 0:2] = sim.coords
        nodes[ORIGIN, 2] = 1.0
        nodes[DESTINATION, 3] = 1.0
        stations = np.asarray(sim.stations, dtype=np.int64)
        nodes[stations, 4] = 1.0
        nodes[:n, 5] = s.indicator
        nodes[stations, 6] = s.time - s.changed_at[stations]
        nodes[:n, 7] = np.asarray(extra["arrival_rate"], dtype=float)
        nodes[:n, 8] = np.asarray(extra["service_rate"], dtype=float)
        nodes[:n, 9] = np.asarray(extra["queue_capacity"], dtype=float)
        nodes[:n, 10] = dist
        nodes[:n, 11] = sim.consumption * dist / sim.capacity
        nodes[:n, 12] = 1.0
        return {"ev": ev, "nodes": nodes, "action_mask": self.action_masks().astype(np.int8)}

    def _info(self, costs: dict[str, float]) -> dict[str, Any]:
        sim = self.simulator
        return {
            "action_mask": self.action_masks(),
            "costs": dict(costs),
            "total_costs": dict(sim.state.total_costs),
            "epoch": self._epoch.kind if self._epoch is not None else None,
            "time": sim.state.time,
        }

    # ------------------------------------------------------------------ state

    def get_state(self) -> dict[str, Any]:
        """An exact snapshot of the episode."""
        sim = self.simulator
        return {
            "state": sim.snapshot(),
            "log": sim.log.copy(),
            "epoch": self._epoch,
            "instance": sim.instance,
            "scenario": sim.scenario,
        }

    def set_state(self, snapshot: dict[str, Any]) -> None:
        """Restore a snapshot taken with `get_state`."""
        sim = EVShortestPathSimulator(
            self.spec_, snapshot["instance"], snapshot["scenario"], log=snapshot["log"].copy()
        )
        sim.restore(snapshot["state"])
        self._sim = sim
        self._epoch = snapshot["epoch"]

    def clone(self) -> EVShortestPathEnv:
        """An exact copy that shares the true future (do not plan with it: use `fork`)."""
        twin = copy.copy(self)
        twin.set_state(self.get_state())
        return twin

    def fork(self, seed: int) -> EVShortestPathEnv:
        """A copy whose hidden future is redrawn from what the EV knows (non-anticipative).

        Every station's queue is redrawn from its occupancy indicator: empty when it is 0, else a
        number of EVs from the stationary distribution given at least one (with fresh exponential
        charging times); the station where the EV is keeps the queue it observed, with fresh
        charging times for the exogenous EVs. Future arrivals are redrawn after now. The EV's own
        place in a queue, and its charging, are kept. This ignores how long each indicator has kept
        its value, so it is a non-anticipative approximation of the posterior, not the posterior.
        """
        twin = self.clone()
        sim = twin.simulator
        s = sim.state
        rng = np.random.default_rng(seed)
        extra = sim.instance.extra
        lam = np.asarray(extra["arrival_rate"], dtype=float)
        mu = np.asarray(extra["service_rate"], dtype=float)
        s.heap = [e for e in s.heap if e[1] not in (0, 1)]  # keep the EV's own events
        for i in sim.stations:
            station = s.stations[i]
            ev_here = station.in_service is not None and station.in_service[0] == "ev"
            ev_queued = "ev" in station.queue
            known = s.node == i and s.status != "driving"
            if ev_here or ev_queued or known:
                # keep the order and the EV, redraw the exogenous charging times
                if station.in_service is not None and station.in_service[0] == "x":
                    end = s.time + float(rng.exponential(1.0 / mu[i]))
                    station.in_service = ("x", end)
                    sim._push(end, 0, i, 0.0)  # pyright: ignore[reportPrivateUsage]
                station.queue = [
                    "ev" if q == "ev" else float(rng.exponential(1.0 / mu[i]))
                    for q in station.queue
                ]
            else:
                k = int(sim.kappa[i])
                present = 0
                if s.indicator[i] > 0:
                    probs = stationary_distribution(lam[i], mu[i], k)[1:]
                    present = 1 + int(rng.choice(k, p=probs / probs.sum()))
                durations = rng.exponential(1.0 / mu[i], size=present).tolist()
                station.in_service = None
                station.queue = []
                if durations:
                    end = s.time + durations[0]
                    station.in_service = ("x", end)
                    sim._push(end, 0, i, 0.0)  # pyright: ignore[reportPrivateUsage]
                    station.queue = list(durations[1:])
            remaining = self.params.horizon - s.time
            if lam[i] > 0 and remaining > 0:
                m = int(rng.poisson(lam[i] * remaining))
                times = np.sort(rng.uniform(s.time, self.params.horizon, size=m))
                for t, d in zip(times, rng.exponential(1.0 / mu[i], size=m), strict=True):
                    sim._push(float(t), 1, i, float(d))  # pyright: ignore[reportPrivateUsage]
        return twin
