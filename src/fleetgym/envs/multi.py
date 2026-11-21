"""Multi-agent environments: one agent per vehicle (design §15, ADR 0016).

`FleetGymAECEnv` (PettingZoo AEC) follows the simulator: the agent whose vehicle has a decision
epoch acts. `FleetGymParallelEnv` (PettingZoo Parallel) asks every vehicle that must decide now at
once; busy vehicles send the no-op action (the last one). Both run on the single-agent
`FleetGymEnv` and its simulator, so dynamics, masks, costs and the validator are shared.

Rewards (``agents.reward``)
---------------------------
* ``global``: every agent receives the step's total reward.
* ``individual``: each agent receives the reward of the costs attributed to its vehicle, plus an
  equal share of the costs attributed to no vehicle (unserved and rejected requests), so individual
  rewards add up to the total.
* ``mixed``: ``reward_mix * global + (1 - reward_mix) * individual``.

Conflicts (Parallel API, ``agents.conflict``)
---------------------------------------------
Two vehicles choosing the same customer at once: ``priority`` (lowest vehicle index wins),
``distance`` (the nearest vehicle wins, ties by index), ``lottery`` (a random winner from the
``agents.conflict`` stream of the episode seed, keyed by the epoch and the customer), or ``reject``
(no decision of that step is applied; every agent receives ``-conflict_penalty``). A losing vehicle
keeps its decision and is asked again in the next step, at the same time. Conflicts are logged
(``conflict`` rows).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from pettingzoo import AECEnv, ParallelEnv

from fleetgym.envs.env import FleetGymEnv
from fleetgym.errors import ConfigError, InfeasibleActionError
from fleetgym.interfaces.observations import (
    GLOBAL_FEATURES,
    NODE_FEATURES,
    VEHICLE_FEATURES,
    PaddedGraphObservation,
)
from fleetgym.sim.decisions import Dispatch
from fleetgym.sim.engine import Epoch
from fleetgym.sim.info_state import DEPOT
from fleetgym.sim.state import VehicleStatus
from fleetgym.spec.models import ProblemSpec
from fleetgym.uncertainty.streams import StreamFactory

__all__ = ["FleetGymAECEnv", "FleetGymParallelEnv", "agent_name"]

Obs = dict[str, NDArray[Any]]


def agent_name(vehicle: int) -> str:
    """The agent of ``vehicle``: ``vehicle_<index>``."""
    return f"vehicle_{vehicle}"


def _known(options: dict[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (options or {}).items() if k in ("instance", "scenario")}


def _vehicle(agent: str) -> int:
    return int(agent.rsplit("_", 1)[1])


class _MultiBase:
    """Shared setup: the wrapped single-agent env, agents, rewards and the global state."""

    def __init__(self, spec: ProblemSpec, env_kwargs: Mapping[str, Any]) -> None:
        if spec.decisions.interface != "next_node":
            raise ConfigError(
                "multi-agent environments need `decisions.interface: next_node` (agents are "
                "vehicles)",
                source=spec.source,
                key_path="decisions.interface",
            )
        if spec.fleet.sizing == "unlimited" or spec.decisions.epochs != "on_event":
            raise ConfigError(
                "multi-agent environments need a fixed or limited fleet and on_event epochs",
                source=spec.source,
                key_path="fleet.sizing",
            )
        self.env = FleetGymEnv(spec, **dict(env_kwargs))
        from fleetgym.envs.env import _known_fleet_size  # pyright: ignore[reportPrivateUsage]

        n = _known_fleet_size(spec, env_kwargs.get("instance"))
        if n == 0:
            raise ConfigError(
                "the number of agents must be known: give the instances vehicle counts, "
                "`fleet.types[].count` as numbers, or pass an `instance=`",
                source=spec.source,
                key_path="fleet.types",
            )
        self.possible_agents = [agent_name(i) for i in range(n)]
        agents = spec.agents
        self.reward_mode = agents.reward
        self.reward_mix = agents.reward_mix if agents.reward_mix is not None else 1.0
        self.conflict = agents.conflict
        obs = self.env.observations
        size = (
            obs.max_nodes * (NODE_FEATURES + 1)
            + obs.max_vehicles * (VEHICLE_FEATURES + 1)
            + GLOBAL_FEATURES
        )
        limit = float(np.finfo(np.float32).max)
        self.state_space = gym.spaces.Box(-limit, limit, shape=(size,), dtype=np.float32)

    @property
    def problem_spec(self) -> ProblemSpec:
        """The problem spec."""
        return self.env.problem_spec

    def state(self) -> NDArray[np.float32]:
        """Global state for a centralized critic: all nodes, all vehicles and global features."""
        sim = self.env.simulator
        obs = self.env.observations.build(sim, None, np.zeros(self.env.interface.n_actions, bool))
        return np.concatenate(
            [
                obs["nodes"].ravel(),
                obs["node_valid"].astype(np.float32),
                obs["vehicles"].ravel(),
                obs["vehicle_valid"].astype(np.float32),
                obs["globals"],
            ]
        ).astype(np.float32)

    def _rewards(
        self, info: Mapping[str, Any], total: float, agents: list[str]
    ) -> dict[str, float]:
        if not agents:
            return {}
        model = self.env.reward_model
        if self.reward_mode == "global":
            return dict.fromkeys(agents, float(total))
        costs: dict[str, float] = dict(info.get("costs", {}))
        own: dict[int, dict[str, float]] = info.get("vehicle_costs", {})
        shared = dict(costs)
        for part in own.values():
            for component, amount in part.items():
                shared[component] = shared.get(component, 0.0) - amount
        share = float(model(shared)) / len(agents)
        individual = {a: float(model(own.get(_vehicle(a), {}))) + share for a in agents}
        if self.reward_mode == "individual":
            return individual
        mix = self.reward_mix
        return {a: mix * float(total) + (1 - mix) * individual[a] for a in agents}

    def _done(self, agent: str) -> bool:
        sim = self.env.simulator
        return sim.state.terminated or sim.state.vehicles[_vehicle(agent)].status == (
            VehicleStatus.DONE
        )


class FleetGymAECEnv(_MultiBase, AECEnv[str, Obs, int]):  # pyright: ignore[reportIncompatibleMethodOverride]
    """PettingZoo AEC environment: the vehicle with a decision epoch acts.

    Parameters
    ----------
    spec
        A spec with ``next_node``, ``on_event`` epochs and a fixed or limited fleet.
    **env_kwargs
        Passed to `FleetGymEnv` (``instance``, ``max_nodes``, ``invalid_action``, ...).
    """

    metadata = {"render_modes": ["ansi"], "name": "fleetgym_aec_v0", "is_parallelizable": True}  # noqa: RUF012

    def __init__(self, spec: ProblemSpec, **env_kwargs: Any) -> None:
        _MultiBase.__init__(self, spec, env_kwargs)
        AECEnv.__init__(self)
        self.render_mode = self.env.render_mode
        self.agents = []
        self._skip_agent_selection = None

    def observation_space(self, agent: str) -> gym.spaces.Space[Any]:
        """The single-agent observation space (the same for every vehicle)."""
        return self.env.observation_space

    def action_space(self, agent: str) -> gym.spaces.Space[Any]:
        """The single-agent action space."""
        return self.env.action_space

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        """Start an episode (``options``: ``instance`` and ``scenario`` as in `FleetGymEnv.reset`).

        Other option keys are ignored: PettingZoo's API test resets with arbitrary options.
        """
        self.env.reset(seed=seed, options=_known(options))
        self.agents = list(self.possible_agents)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {a: {} for a in self.agents}
        self._skip_agent_selection = None
        self.agent_selection = self._selected()
        self._mark_done()

    def _selected(self) -> str:
        epoch = self.env._epoch  # pyright: ignore[reportPrivateUsage]
        if epoch is not None and epoch.vehicle is not None:
            return agent_name(epoch.vehicle)
        live = [a for a in self.agents if not self.terminations.get(a, False)]
        return live[0] if live else (self.agents[0] if self.agents else self.possible_agents[0])

    def _mark_done(self) -> None:
        final = self.env._final_step_pending  # pyright: ignore[reportPrivateUsage]
        for agent in self.agents:
            if not final and self._done(agent):
                self.terminations[agent] = True
        self._deads_step_first()

    def observe(self, agent: str) -> Obs:
        """``agent``'s view; its action mask is all zeros unless it is the one to act."""
        if agent == self.agent_selection and not self.terminations.get(agent, False):
            mask = self.env.action_masks()
        else:
            mask = np.zeros(self.env.interface.n_actions, dtype=bool)
        return self.env.observe_vehicle(_vehicle(agent), mask)

    def step(self, action: int | None) -> None:
        """Apply the selected agent's action (``None`` for an agent that is done)."""
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)  # pyright: ignore[reportArgumentType]
            return
        assert action is not None
        self._cumulative_rewards[agent] = 0.0
        _, reward, terminated, _, info = self.env.step(int(action))
        live = list(self.agents)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self.rewards.update(self._rewards(info, float(reward), live))
        self.infos = {a: {} for a in self.agents}
        self.infos[agent] = info
        if terminated:
            for a in self.agents:
                self.terminations[a] = True
        self.agent_selection = self._selected()
        self._mark_done()
        self._accumulate_rewards()

    def render(self) -> str | None:
        """Text view (``render_mode="ansi"``)."""
        return self.env.render()

    def close(self) -> None:
        """Nothing to release."""


class FleetGymParallelEnv(_MultiBase, ParallelEnv[str, Obs, int]):
    """PettingZoo Parallel environment: every vehicle that must decide now acts at once.

    The action space is the single-agent one plus a last **no-op** action, the only valid action of
    a vehicle without a decision.

    Parameters
    ----------
    spec
        A spec with ``next_node``, ``on_event`` epochs and a fixed or limited fleet.
    conflict_penalty
        Penalty of a rejected joint action; required with ``agents.conflict: reject``.
    **env_kwargs
        Passed to `FleetGymEnv`.
    """

    metadata = {"render_modes": ["ansi"], "name": "fleetgym_parallel_v0"}  # noqa: RUF012

    def __init__(
        self, spec: ProblemSpec, *, conflict_penalty: float | None = None, **env_kwargs: Any
    ) -> None:
        if env_kwargs.get("invalid_action") == "penalize":
            raise ValueError("the Parallel API supports invalid_action 'mask' or 'repair'")
        _MultiBase.__init__(self, spec, env_kwargs)
        if (self.conflict == "reject") != (conflict_penalty is not None):
            raise ValueError(
                "conflict_penalty is required with `agents.conflict: reject`, and only then"
            )
        self.conflict_penalty = conflict_penalty
        self.render_mode = self.env.render_mode
        self.noop = self.env.interface.n_actions
        self._builder = PaddedGraphObservation(
            self.env.max_nodes, self.env.max_vehicles, self.noop + 1
        )
        self._observation_space = self._builder.space
        self._action_space = gym.spaces.Discrete(self.noop + 1)
        self.agents = []

    def observation_space(self, agent: str) -> gym.spaces.Space[Any]:
        """Observations with the no-op slot in the action mask (one shared space object)."""
        return self._observation_space

    def action_space(self, agent: str) -> gym.spaces.Space[Any]:
        """``Discrete(n_actions + 1)``: the single-agent actions, then the no-op."""
        return self._action_space

    def _mask(self, agent: str) -> NDArray[np.bool_]:
        mask = np.zeros(self.noop + 1, dtype=bool)
        vehicle = _vehicle(agent)
        env = self.env
        if env._final_step_pending:  # pyright: ignore[reportPrivateUsage]
            mask[self.noop] = True
            return mask
        if vehicle in env.ready_vehicles():
            mask[: self.noop] = env.vehicle_mask(vehicle)
        else:
            mask[self.noop] = True
        return mask

    def _observations(self) -> dict[str, Obs]:
        sim = self.env.simulator
        out = {}
        for agent in self.agents:
            mask = self._mask(agent)
            perspective = sim.state.vehicles[_vehicle(agent)]
            out[agent] = self._builder.build(
                sim, Epoch(sim.state.time, "vehicle_idle", perspective.index), mask
            )
        return out

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Obs], dict[str, dict[str, Any]]]:
        """Start an episode (``options`` as in `FleetGymAECEnv.reset`)."""
        self.env.reset(seed=seed, options=_known(options))
        self.agents = list(self.possible_agents)
        return self._observations(), {a: {} for a in self.agents}

    def _resolve(self, choices: dict[int, int]) -> tuple[dict[int, int], list[tuple[int, int]]]:
        """Keep one vehicle per contested customer; return kept actions and (loser, node) pairs."""
        sim = self.env.simulator
        by_node: dict[int, list[int]] = {}
        for vehicle, action in choices.items():
            epoch = next(e for e in sim.ready_epochs() if e.vehicle == vehicle)
            decision = self.env.interface.decode(action, sim, epoch)
            if isinstance(decision, Dispatch) and decision.node != DEPOT:
                by_node.setdefault(decision.node, []).append(vehicle)
        losers: list[tuple[int, int]] = []
        for node, vehicles in by_node.items():
            if len(vehicles) < 2:
                continue
            ranked = sorted(vehicles)
            if self.conflict == "distance":
                dist = sim.info.distances
                ranked = sorted(vehicles, key=lambda v: (dist[sim.state.vehicles[v].node, node], v))
            elif self.conflict == "lottery":
                rng = StreamFactory(self.env.episode_seed).generator(
                    "agents.conflict", sim.state.epoch_count, node
                )
                winner = int(rng.choice(sorted(vehicles)))
                ranked = [winner, *(v for v in sorted(vehicles) if v != winner)]
            losers += [(v, node) for v in ranked[1:]]
        kept = {v: a for v, a in choices.items() if v not in {lv for lv, _ in losers}}
        return kept, losers

    def step(
        self, actions: Mapping[str, int]
    ) -> tuple[
        dict[str, Obs],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Apply the joint action of the live agents."""
        env = self.env
        sim = env.simulator
        live = list(self.agents)
        missing = set(live) - set(actions)
        if missing:
            raise InfeasibleActionError(f"actions are missing for {sorted(missing)}")
        final = env._final_step_pending  # pyright: ignore[reportPrivateUsage]
        ready = set() if final else set(env.ready_vehicles())
        choices: dict[int, int] = {}
        for agent in live:
            vehicle, action = _vehicle(agent), int(actions[agent])
            if vehicle in ready:
                if action == self.noop:
                    raise InfeasibleActionError(f"{agent} must decide now; the no-op is masked")
                choices[vehicle] = action
            elif action != self.noop and env.invalid_action == "mask":
                raise InfeasibleActionError(f"{agent} has no decision now; send the no-op")
        kept, losers = self._resolve(choices)
        for vehicle, node in losers:
            sim.log.append(
                "conflict", time=sim.state.time, vehicle=vehicle, node=node, component=self.conflict
            )
        if losers and self.conflict == "reject":
            assert self.conflict_penalty is not None
            penalty = float(self.conflict_penalty)
            sim.log.append("cost", time=sim.state.time, component="invalid_action", amount=penalty)
            sim.log.append(
                "epoch", epoch=sim.state.epoch_count, reward=-penalty, time=sim.state.time
            )
            rewards = dict.fromkeys(live, -penalty)
            info: dict[str, Any] = {"conflict": True}
            terminated = False
        else:
            _, reward, terminated, _, info = env.step_joint(kept)
            rewards = self._rewards(info, float(reward), live)
        terminations = {a: bool(terminated) or self._done(a) for a in live}
        truncations = dict.fromkeys(live, False)
        infos = {a: dict(info) if _vehicle(a) in choices else {} for a in live}
        self.agents = [a for a in live if not terminations[a]]
        observations = self._observations()
        for agent in live:
            if agent not in observations:
                observations[agent] = self._builder.build(
                    sim,
                    Epoch(sim.state.time, "vehicle_idle", _vehicle(agent)),
                    np.eye(self.noop + 1, dtype=bool)[self.noop],
                )
        return observations, rewards, terminations, truncations, infos

    def render(self) -> str | None:
        """Text view (``render_mode="ansi"``)."""
        return self.env.render()

    def close(self) -> None:
        """Nothing to release."""
