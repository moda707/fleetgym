"""The single-agent `FleetGymEnv`."""

from __future__ import annotations

import copy
import importlib
import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.error import ResetNeeded
from numpy.typing import NDArray

from fleetgym._version import __version__
from fleetgym.data.instance import Instance
from fleetgym.data.instance_set import read_instance_set
from fleetgym.data.readers import read_instance
from fleetgym.errors import ConfigError, InfeasibleActionError, NotInScopeError
from fleetgym.eventlog import EventLog
from fleetgym.interfaces.base import INTERFACE_CLASSES, ActionInterface
from fleetgym.interfaces.observations import PaddedGraphObservation
from fleetgym.interfaces.rewards import RewardModel
from fleetgym.network.base import Network, build_network
from fleetgym.sim.decisions import RecourseChoice, Wait
from fleetgym.sim.engine import Epoch, Simulator, Transition
from fleetgym.sim.invariants import InvariantMonitor
from fleetgym.sim.planned import PlannedRoutingSimulator
from fleetgym.sim.state import RequestStatus, SimState
from fleetgym.spec.binding import (
    BoundProblem,
    bind,
    collection_instance,
    instance_from_source,
    source_files,
)
from fleetgym.spec.models import CollectionSource, FilesSource, ProblemSpec
from fleetgym.spec.models.sections import INTERFACES_NEEDING_ROUTING
from fleetgym.uncertainty.requests import resample_future_requests, sample_requests
from fleetgym.uncertainty.sampling import resample_unrevealed, sample_scenario
from fleetgym.uncertainty.scenario import Scenario, episode_instance
from fleetgym.uncertainty.streams import StreamFactory

__all__ = ["EnvState", "FleetGymEnv", "InvalidActionMode"]

InvalidActionMode = Literal["mask", "penalize", "repair"]

#: Figure render modes, bound by name to the visualization layer (which this layer never imports).
_FIGURE_RENDERERS = {
    "rgb_array": "fleetgym.viz.render:render_rgb_array",
    "human": "fleetgym.viz.render:render_human",
}
_RENDER_MODES = ["ansi"] + (
    list(_FIGURE_RENDERERS) if importlib.util.find_spec("matplotlib") is not None else []
)
_SEED_LIMIT = 2**63


@dataclass(frozen=True)
class EnvState:
    """Exact snapshot of an environment (see `FleetGymEnv.get_state`)."""

    sim: tuple[SimState, list[Any]]
    epoch: Epoch | None
    mask: NDArray[np.bool_]
    problem: BoundProblem
    network: Network
    log: EventLog
    episode_seed: int
    root_seed: int | None
    episode_index: int
    np_random_state: Mapping[str, Any]
    carry_costs: dict[str, float]
    final_step_pending: bool
    carry_vehicle_costs: dict[int, dict[str, float]] | None = None
    scenario: Scenario | None = None
    base_problem: BoundProblem | None = None


def _derive_seed(root: int, index: int) -> int:
    ss = np.random.SeedSequence(entropy=root, spawn_key=(index,))
    return int(ss.generate_state(1, np.uint64)[0]) % _SEED_LIMIT


def _infer_max_nodes(spec: ProblemSpec, instance: Instance | None) -> int:
    """Node slots: the instance's nodes, plus ``max_requests`` slots with dynamic requests."""
    dyn = spec.dynamics
    if dyn is None:
        return _instance_nodes(spec, instance)
    if dyn.initial_requests == "none":
        depots = instance.n_depots if instance is not None else 1
        return depots + dyn.max_requests
    return _instance_nodes(spec, instance) + dyn.max_requests


def _instance_nodes(spec: ProblemSpec, instance: Instance | None) -> int:
    if instance is not None:
        return instance.n_nodes
    source = spec.instance_source
    if isinstance(source, FilesSource):
        sizes = []
        for entry in source.paths:
            path = Path(entry)
            if not path.is_absolute() and spec.source and not spec.source.startswith("<"):
                path = Path(spec.source).parent / path
            if path.is_dir():
                sizes.append(read_instance_set(path).max_nodes)
            else:
                sizes.append(read_instance(path, fmt=source.format).n_nodes)
        return max(sizes)
    if isinstance(source, CollectionSource):
        return collection_instance(source).n_nodes
    raise ConfigError(
        "no instances: pass `instance=` (or `max_nodes=` and instances at reset), or give the spec "
        "an `instance_source` (files, an instance set, or a benchmark collection)",
        source=spec.source,
        key_path="instance_source",
    )


def _merge_costs(*parts: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for part in parts:
        for component, amount in part.items():
            merged[component] = merged.get(component, 0.0) + amount
    return merged


def _known_fleet_size(spec: ProblemSpec, instance: Instance | None) -> int:
    """Largest number of vehicles of a fixed or limited fleet over the possible instances.

    0 for an unlimited fleet, or when it cannot be known before reset.
    """
    if spec.fleet.sizing == "unlimited":
        return 0
    if instance is not None:
        return sum(vt.count or 0 for vt in bind(spec, instance).vehicle_types)
    if not spec.fleet.from_instance:
        counts = [vt.count for vt in spec.fleet.types]
        if all(isinstance(c, int) for c in counts):
            return sum(int(c) for c in counts)  # pyright: ignore[reportArgumentType]
    source = spec.instance_source
    if isinstance(source, FilesSource):
        sizes = []
        for entry in source.paths:
            path = Path(entry)
            if not path.is_absolute() and spec.source and not spec.source.startswith("<"):
                path = Path(spec.source).parent / path
            if path.is_dir():
                sizes.append(read_instance_set(path).max_vehicles)
            else:
                sizes.append(_fleet_count(read_instance(path, fmt=source.format)))
        return max(sizes, default=0)
    return max((_fleet_count(i) for i in _source_instances(spec)), default=0)


def _fleet_count(instance: Instance) -> int:
    return sum(int(vt.get("count", 0)) for vt in instance.vehicle_types)


def _source_instances(spec: ProblemSpec) -> list[Instance]:
    """Every instance a files or collection source can draw (for sizing before reset)."""
    source = spec.instance_source
    if isinstance(source, CollectionSource):
        return [collection_instance(source)]
    if not isinstance(source, FilesSource):
        return []
    out: list[Instance] = []
    for path in source_files(spec):
        out.append(read_instance(path, fmt=source.format))
    return out


def _thaw(value: Any) -> Any:
    from fleetgym.spec.models.base import deep_thaw

    return deep_thaw(value) if value is not None else {}


class FleetGymEnv(gym.Env[dict[str, NDArray[Any]], np.int64]):
    """Gymnasium environment for one FleetGym spec.

    Parameters
    ----------
    spec
        The problem spec (see `fleetgym.envs.make` to build from an ID or YAML file).
    instance
        Use this instance in every episode instead of drawing from ``spec.instance_source``.
    render_mode
        ``"ansi"`` or ``None``.
    invalid_action
        What happens when an action is masked: ``"mask"`` raises `InfeasibleActionError`;
        ``"repair"`` replaces it with the nearest feasible action; ``"penalize"`` leaves the state
        unchanged and returns ``-invalid_action_penalty``.
    invalid_action_penalty
        Required with ``invalid_action="penalize"``.
    max_nodes
        Node slots for observations and actions; inferred from the instance source if omitted.
    max_vehicles
        Vehicle slots in observations; defaults to ``max_nodes``.
    debug
        Check simulator invariants after every step.

    Notes
    -----
    ``info`` contains ``costs`` (components realized in the step), ``total_costs``,
    ``action_mask``, ``vehicle`` (the vehicle that must decide next, or ``None``), ``epoch_kind``
    (``"vehicle_idle"``, ``"recourse"`` or ``"request"``), ``time``, ``spec_hash`` and
    ``episode_seed``. The
    scenario is deliberately absent (its id would identify the future); see `scenario`.
    """

    metadata = {"render_modes": _RENDER_MODES, "render_fps": 4}  # noqa: RUF012 - Gymnasium convention

    def __init__(
        self,
        spec: ProblemSpec,
        *,
        instance: Instance | None = None,
        render_mode: str | None = None,
        invalid_action: InvalidActionMode = "mask",
        invalid_action_penalty: float | None = None,
        max_nodes: int | None = None,
        max_vehicles: int | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if spec.meta.status == "planned":
            raise NotInScopeError(
                f"{spec.id or 'this spec'} is planned and cannot be made into an environment yet",
                features=spec.reserved_features(),
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode!r}"
            )
        if invalid_action not in ("mask", "penalize", "repair"):
            raise ValueError(
                f"invalid_action must be mask, penalize or repair, got {invalid_action!r}"
            )
        if (invalid_action == "penalize") != (invalid_action_penalty is not None):
            raise ValueError(
                "invalid_action_penalty is required with invalid_action='penalize', and only then"
            )
        if spec.decisions.interface not in INTERFACE_CLASSES:
            raise NotInScopeError(f"interface {spec.decisions.interface!r} is implemented in M6")
        self.spec_ = spec
        self.render_mode = render_mode
        self.invalid_action: InvalidActionMode = invalid_action
        self.invalid_action_penalty = invalid_action_penalty
        self.debug = debug
        self._fixed_instance = instance
        self.max_nodes = max_nodes if max_nodes is not None else _infer_max_nodes(spec, instance)
        known_fleet = _known_fleet_size(spec, instance)
        if max_vehicles is not None:
            self.max_vehicles = max_vehicles
        elif spec.decisions.interface == "assign" and known_fleet:
            self.max_vehicles = known_fleet
        else:
            self.max_vehicles = max(self.max_nodes, known_fleet)
        if spec.dynamics is not None and spec.decisions.interface not in (
            *INTERFACES_NEEDING_ROUTING,
            "next_node",
        ):
            raise NotInScopeError(
                f"dynamic requests are not supported with {spec.decisions.interface!r}"
            )
        self.interface: ActionInterface = INTERFACE_CLASSES.get(spec.decisions.interface)(
            self.max_nodes,
            self.max_vehicles,
            wait=spec.dynamics is not None and spec.decisions.interface == "next_node",
        )
        self.observations = PaddedGraphObservation(
            self.max_nodes, self.max_vehicles, self.interface.n_actions
        )
        self.action_space = self.interface.action_space  # pyright: ignore[reportAttributeAccessIssue]
        self.observation_space = self.observations.space  # pyright: ignore[reportAttributeAccessIssue]
        self.reward_model = RewardModel(spec.objective)
        self._monitor = InvariantMonitor()
        self._sim: Simulator | None = None
        self._epoch: Epoch | None = None
        self._mask = np.zeros(self.interface.n_actions, dtype=bool)
        self._root_seed: int | None = None
        self._episode_index = 0
        self._episode_seed = 0
        self._network_cache: tuple[str, Network] | None = None
        self._carry_costs: dict[str, float] = {}
        self._carry_vehicle_costs: dict[int, dict[str, float]] = {}
        self._final_step_pending = False
        self._base_problem: BoundProblem | None = None

    # ------------------------------------------------------------------ properties

    @property
    def problem_spec(self) -> ProblemSpec:
        """The problem spec (``env.spec`` is Gymnasium's registration spec)."""
        return self.spec_

    @property
    def simulator(self) -> Simulator:
        """The simulator of the current episode."""
        if self._sim is None:
            raise ResetNeeded("call reset() first")
        return self._sim

    @property
    def event_log(self) -> EventLog:
        """The event log of the current episode."""
        return self.simulator.log

    @property
    def problem(self) -> BoundProblem:
        """The bound problem of the current episode."""
        return self.simulator.problem

    @property
    def epoch(self) -> Epoch | None:
        """The current decision epoch (``None`` once the episode has ended)."""
        return self._epoch

    @property
    def episode_seed(self) -> int:
        """Seed of the current episode."""
        return self._episode_seed

    @property
    def base_instance(self) -> Instance:
        """The instance the episode was drawn on (without dynamic request slots).

        `problem.instance` is the episode instance, which adds the slots of requests arriving
        during the day (`fleetgym.uncertainty.scenario.episode_instance`).
        """
        if self._base_problem is None:
            raise ResetNeeded("call reset() first")
        return self._base_problem.instance

    # ------------------------------------------------------------------ gymnasium API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        """Start an episode.

        Parameters
        ----------
        seed
            Episode seed. Later resets without a seed continue a deterministic sequence derived from
            the last seed given.
        options
            ``{"instance": Instance}`` uses that instance for this episode only.
            ``{"scenario": Scenario}`` replays that scenario instead of sampling one from the seed
            (it must belong to this spec and instance).
        """
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
        if self.spec_.dynamics is None and instance.n_nodes > self.max_nodes:
            raise ConfigError(
                f"instance {instance.name!r} has {instance.n_nodes} nodes but the environment has "
                f"{self.max_nodes} node slots; pass a larger max_nodes",
                source=self.spec_.source,
            )
        base_problem = bind(self.spec_, instance)
        n_vehicles = sum(vt.count or 0 for vt in base_problem.vehicle_types)
        if n_vehicles > self.max_vehicles:
            raise ConfigError(
                f"the fleet has {n_vehicles} vehicles but the environment has {self.max_vehicles} "
                "vehicle slots; pass a larger max_vehicles",
                source=self.spec_.source,
                key_path="fleet",
            )
        scenario: Scenario | None = options.get("scenario")
        stochastic = bool(self.spec_.uncertainty) or self.spec_.dynamics is not None
        if scenario is None and self.spec_.uncertainty:
            distances = (
                self._network_for(base_problem).distance_matrix()
                if "travel_time" in self.spec_.uncertainty
                else None
            )
            scenario = sample_scenario(base_problem, self._episode_seed, distances=distances)
        elif scenario is None and self.spec_.dynamics is not None:
            values = sample_requests(base_problem, self._episode_seed)
            scenario = Scenario(
                base_problem.spec_hash, base_problem.instance_hash, values, seed=self._episode_seed
            )
        elif scenario is not None and not stochastic:
            raise ValueError("this spec has no uncertain parameters; do not pass a scenario")
        episode = episode_instance(instance, scenario) if scenario is not None else instance
        if episode.n_nodes > self.max_nodes:
            raise ConfigError(
                f"the episode has {episode.n_nodes} node slots but the environment has "
                f"{self.max_nodes}; pass a larger max_nodes",
                source=self.spec_.source,
            )
        problem = base_problem if episode is instance else bind(self.spec_, episode)
        self._base_problem = base_problem
        log = EventLog(
            {
                "fleetgym_version": __version__,
                "env_id": self.spec_.id or "",
                "spec_hash": problem.spec_hash,
                "instance_hash": base_problem.instance_hash,
                "instance_name": instance.name,
                "episode_seed": str(self._episode_seed),
                "scenario_id": scenario.id if scenario is not None else "",
            }
        )
        self._sim = self._simulator_class()(
            problem, self._network_for(problem), log=log, scenario=scenario
        )
        self._monitor.reset()
        self._epoch = self._sim.start()
        # Costs realized before the first decision count in the first step's reward.
        self._carry_costs = self._sim.state.take_pending_costs()
        self._carry_vehicle_costs = self._sim.state.take_pending_vehicle_costs()
        # An episode has at least one step: if nothing can be decided, action 0 ends it.
        self._final_step_pending = self._epoch is None
        self._mask = self._current_mask()
        if self.debug:
            self._monitor.check(self._sim)
        return self._observation(), self._info({})

    def step(
        self, action: np.int64 | int
    ) -> tuple[dict[str, NDArray[Any]], SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply an action for the vehicle of the current epoch."""
        sim = self.simulator
        epoch = self._epoch
        if epoch is None and self._final_step_pending:
            return self._final_step(int(action))
        if epoch is None:
            raise ResetNeeded("the episode has ended; call reset()")
        if not self.action_space.contains(np.int64(action)):
            raise InfeasibleActionError(f"action {action!r} is outside the action space")
        chosen = int(action)
        if not self._mask[chosen]:
            if self.invalid_action == "mask":
                raise InfeasibleActionError(self._explain(chosen))
            if self.invalid_action == "penalize":
                assert self.invalid_action_penalty is not None
                penalty = float(self.invalid_action_penalty)
                sim.log.append(
                    "decision", vehicle=epoch.vehicle, action=chosen, epoch=sim.state.epoch_count
                )
                sim.log.append(
                    "cost",
                    time=epoch.time,
                    vehicle=epoch.vehicle,
                    component="invalid_action",
                    amount=penalty,
                )
                sim.log.append(
                    "epoch", epoch=sim.state.epoch_count, reward=-penalty, time=epoch.time
                )
                info = self._info({"invalid_action": penalty})
                info["invalid_action"] = True
                return self._observation(), -penalty, False, False, info
            chosen = int(self.interface.repair(chosen, self._mask, sim, epoch))
        decision = self.interface.decode(chosen, sim, epoch)
        target = (
            {}
            if isinstance(decision, (RecourseChoice, Wait))
            else {"node": getattr(decision, "node", 0)}
        )
        sim.log.append(
            "decision", vehicle=epoch.vehicle, action=chosen, epoch=sim.state.epoch_count, **target
        )
        transition = sim.apply(decision)
        return self._finish_step(transition, chosen != int(action), chosen)

    def step_joint(
        self, actions: Mapping[int, int]
    ) -> tuple[dict[str, NDArray[Any]], SupportsFloat, bool, bool, dict[str, Any]]:
        """Simultaneous decisions of vehicles that are ready now (multi-agent Parallel API).

        ``actions`` maps vehicles of `Simulator.ready_epochs` to actions; conflicts must
        be resolved by the caller. Invalid actions follow ``invalid_action`` (``penalize`` is not
        supported here).
        """
        sim = self.simulator
        if self._epoch is None and self._final_step_pending:
            return self._final_step(0)
        if self._epoch is None:
            raise ResetNeeded("the episode has ended; call reset()")
        epochs = {e.vehicle: e for e in sim.ready_epochs()}
        decisions = []
        for vehicle, action in sorted(actions.items()):
            if vehicle not in epochs:
                raise InfeasibleActionError(f"vehicle {vehicle} has no decision now")
            epoch = epochs[vehicle]
            mask = self.interface.mask(sim, epoch)
            chosen = int(action)
            if not (0 <= chosen < mask.size and mask[chosen]):
                if self.invalid_action != "repair":
                    raise InfeasibleActionError(self.interface.explain(chosen, mask, sim, epoch))
                chosen = int(self.interface.repair(chosen, mask, sim, epoch))
            decision = self.interface.decode(chosen, sim, epoch)
            target = (
                {}
                if isinstance(decision, (RecourseChoice, Wait))
                else {"node": getattr(decision, "node", 0)}
            )
            sim.log.append(
                "decision", vehicle=vehicle, action=chosen, epoch=sim.state.epoch_count, **target
            )
            decisions.append(decision)
        transition = sim.apply_many(decisions)
        return self._finish_step(transition, False, None)

    def ready_vehicles(self) -> list[int]:
        """Vehicles that must decide now (all of them act together in the Parallel API)."""
        if self._epoch is None:
            return []
        return [e.vehicle for e in self.simulator.ready_epochs() if e.vehicle is not None]

    def vehicle_mask(self, vehicle: int) -> NDArray[np.bool_]:
        """Valid actions of ``vehicle`` now (all False if it has no decision now)."""
        sim = self.simulator
        for epoch in sim.ready_epochs() if self._epoch is not None else []:
            if epoch.vehicle == vehicle:
                return self.interface.mask(sim, epoch)
        return np.zeros(self.interface.n_actions, dtype=bool)

    def observe_vehicle(self, vehicle: int, mask: NDArray[np.bool_]) -> dict[str, NDArray[Any]]:
        """The observation from ``vehicle``'s point of view, with ``mask`` as its action mask."""
        sim = self.simulator
        perspective = Epoch(sim.state.time, "vehicle_idle", vehicle)
        return self.observations.build(sim, perspective, mask)

    def _finish_step(
        self, transition: Transition, repaired: bool, chosen: int | None
    ) -> tuple[dict[str, NDArray[Any]], SupportsFloat, bool, bool, dict[str, Any]]:
        sim = self.simulator
        costs = _merge_costs(self._carry_costs, transition.costs)
        vehicle_costs = dict(self._carry_vehicle_costs)
        for vehicle, part in transition.vehicle_costs.items():
            vehicle_costs[vehicle] = _merge_costs(vehicle_costs.get(vehicle, {}), part)
        self._carry_costs = {}
        self._carry_vehicle_costs = {}
        reward = self.reward_model(costs)
        sim.log.append("epoch", epoch=sim.state.epoch_count, reward=reward, time=sim.state.time)
        self._epoch = transition.epoch
        self._mask = self._current_mask()
        if self.debug:
            self._monitor.check(sim)
        info = self._info(costs)
        info["vehicle_costs"] = vehicle_costs
        if repaired:
            info["repaired_action"] = chosen
        return self._observation(), reward, transition.terminated, False, info

    def _final_step(
        self, action: int
    ) -> tuple[dict[str, NDArray[Any]], SupportsFloat, bool, bool, dict[str, Any]]:
        """Step of an episode that ended during reset: only action 0 is valid."""
        sim = self.simulator
        if action != 0:
            if self.invalid_action == "mask":
                raise InfeasibleActionError(
                    "nothing can be decided in this episode; the only valid action is 0"
                )
            action = 0
        costs, self._carry_costs = self._carry_costs, {}
        vehicle_costs, self._carry_vehicle_costs = self._carry_vehicle_costs, {}
        self._final_step_pending = False
        reward = self.reward_model(costs)
        sim.log.append("decision", action=0, epoch=sim.state.epoch_count)
        sim.log.append("epoch", epoch=sim.state.epoch_count, reward=reward, time=sim.state.time)
        info = self._info(costs)
        info["vehicle_costs"] = vehicle_costs
        return self._observation(), reward, True, False, info

    def action_masks(self) -> NDArray[np.bool_]:
        """Valid actions now (the sb3-contrib ``MaskablePPO`` convention)."""
        return self._mask.copy()

    def render(self) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Render the episode so far.

        ``ansi``: a text summary of the routes. ``rgb_array``: the map and vehicle Gantt chart as an
        ``(H, W, 3)`` ``uint8`` array. ``human``: the same figure in a window. The figure modes need
        the ``viz`` extra (matplotlib); they are listed in ``metadata`` only when it is installed.
        """
        if self.render_mode in _FIGURE_RENDERERS:
            module, function = _FIGURE_RENDERERS[self.render_mode].split(":")
            return getattr(importlib.import_module(module), function)(self)
        if self.render_mode != "ansi":
            return None
        sim = self.simulator
        lines = [f"t={sim.state.time:.2f} costs={dict(sorted(sim.state.total_costs.items()))}"]
        for v in sim.state.vehicles:
            route = " ".join(map(str, v.route)) or "-"
            lines.append(f"vehicle {v.index} [{v.status.name.lower()}] load={v.load:g}: {route}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ snapshots

    def get_state(self) -> EnvState:
        """Exact snapshot: simulator, constraint states, event log, seeds and RNG state."""
        sim = self.simulator
        return EnvState(
            sim=sim.snapshot(),
            epoch=self._epoch,
            mask=self._mask.copy(),
            problem=sim.problem,
            network=sim.network,
            log=sim.log.copy(),
            episode_seed=self._episode_seed,
            root_seed=self._root_seed,
            episode_index=self._episode_index,
            np_random_state=copy.deepcopy(self.np_random.bit_generator.state),
            carry_costs=dict(self._carry_costs),
            final_step_pending=self._final_step_pending,
            carry_vehicle_costs={k: dict(v) for k, v in self._carry_vehicle_costs.items()},
            scenario=sim.scenario,
            base_problem=self._base_problem,
        )

    def set_state(self, state: EnvState) -> None:
        """Restore a snapshot from `get_state` (of this or an identically configured env)."""
        if state.problem.spec_hash != self.spec_.content_hash():
            raise ValueError("the snapshot belongs to a different spec")
        sim = self._simulator_class()(
            state.problem, state.network, log=state.log.copy(), scenario=state.scenario
        )
        sim.restore(state.sim)
        self._sim = sim
        self._epoch = state.epoch
        self._mask = state.mask.copy()
        self._episode_seed = state.episode_seed
        self._root_seed = state.root_seed
        self._episode_index = state.episode_index
        self.np_random.bit_generator.state = copy.deepcopy(state.np_random_state)
        self._carry_costs = dict(state.carry_costs)
        self._final_step_pending = state.final_step_pending
        self._carry_vehicle_costs = {
            k: dict(v) for k, v in (state.carry_vehicle_costs or {}).items()
        }
        self._base_problem = state.base_problem
        self._monitor.reset()

    def clone(self) -> FleetGymEnv:
        """An independent copy of this environment in its current state.

        The copy shares the future scenario with this environment (it is not non-anticipative).
        """
        twin = FleetGymEnv(
            self.spec_,
            instance=self._fixed_instance,
            render_mode=self.render_mode,
            invalid_action=self.invalid_action,
            invalid_action_penalty=self.invalid_action_penalty,
            max_nodes=self.max_nodes,
            max_vehicles=self.max_vehicles,
            debug=self.debug,
        )
        if self._sim is not None:
            twin.set_state(self.get_state())
        return twin

    def fork(self, seed: int) -> FleetGymEnv:
        """A non-anticipative copy: same state, with every unrevealed value redrawn from ``seed``.

        Planners (rollout, scenario sampling) must use forks, not `clone`, which shares the true
        future. Without uncertainty a fork equals a clone.
        """
        twin = self.clone()
        sim = self.simulator
        if sim.scenario is None:
            return twin
        if self.spec_.dynamics is not None:
            twin._replace_future(seed)  # pyright: ignore[reportPrivateUsage]
        else:
            fresh = resample_unrevealed(
                sim.problem,
                sim.scenario,
                sim.revealed_masks(),
                seed,
                distances=sim.network.distance_matrix(),
            )
            twin.simulator.replace_scenario(fresh)
        return twin

    def _replace_future(self, seed: int) -> None:
        """Redraw the requests that have not arrived and pending cancellations (dynamic fork)."""
        sim = self.simulator
        assert sim.scenario is not None
        assert self._base_problem is not None
        now = sim.state.time
        cancelled = sim.state.request_status == RequestStatus.CANCELLED
        values = resample_future_requests(
            self._base_problem, dict(sim.scenario.values), now, seed, cancelled
        )
        base = self._base_problem
        scenario = Scenario(base.spec_hash, base.instance_hash, values)
        episode = episode_instance(base.instance, scenario)
        problem = bind(self.spec_, episode)
        snapshot = sim.snapshot()
        twin = self._simulator_class()(
            problem, build_network(problem), log=sim.log.copy(), scenario=scenario
        )
        twin.restore(snapshot)
        twin.reschedule_exogenous(now)
        self._sim = twin

    @property
    def scenario(self) -> Scenario | None:
        """The scenario of the current episode (``None`` without uncertainty)."""
        return self.simulator.scenario

    # ------------------------------------------------------------------ helpers

    def _simulator_class(self) -> type[Simulator]:
        if self.spec_.decisions.interface in INTERFACES_NEEDING_ROUTING:
            return PlannedRoutingSimulator
        return Simulator

    def _network_for(self, problem: BoundProblem) -> Network:
        key = problem.instance_hash
        if self._network_cache is not None and self._network_cache[0] == key:
            return self._network_cache[1]
        network = build_network(problem)
        self._network_cache = (key, network)
        return network

    def _current_mask(self) -> NDArray[np.bool_]:
        if self._epoch is None:
            mask = np.zeros(self.interface.n_actions, dtype=bool)
            mask[0] = True  # keep masks non-empty at the end; any action raises ResetNeeded
            return mask
        return self.interface.mask(self.simulator, self._epoch)

    def _explain(self, action: int) -> str:
        assert self._epoch is not None
        return self.interface.explain(action, self._mask, self.simulator, self._epoch)

    def _observation(self) -> dict[str, NDArray[Any]]:
        return self.observations.build(self.simulator, self._epoch, self._mask)

    def _info(self, costs: dict[str, float]) -> dict[str, Any]:
        sim = self.simulator
        return {
            "costs": dict(costs),
            "total_costs": dict(sim.state.total_costs),
            "action_mask": self._mask.copy(),
            "vehicle": None if self._epoch is None else self._epoch.vehicle,
            "epoch_kind": None if self._epoch is None else self._epoch.kind,
            "time": sim.state.time,
            "spec_hash": sim.problem.spec_hash,
            "episode_seed": self._episode_seed,
        }
