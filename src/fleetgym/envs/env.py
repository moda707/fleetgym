"""The single-agent `FleetGymEnv`."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.error import ResetNeeded
from numpy.typing import NDArray

from fleetgym._version import __version__
from fleetgym.data.instance import Instance
from fleetgym.errors import ConfigError, InfeasibleActionError, NotInScopeError
from fleetgym.eventlog import EventLog
from fleetgym.interfaces.base import INTERFACE_CLASSES, ActionInterface
from fleetgym.interfaces.observations import PaddedGraphObservation
from fleetgym.interfaces.rewards import RewardModel
from fleetgym.network.base import Network, build_network
from fleetgym.sim.decisions import RecourseChoice
from fleetgym.sim.engine import Epoch, Simulator
from fleetgym.sim.invariants import InvariantMonitor
from fleetgym.sim.state import SimState
from fleetgym.spec.binding import BoundProblem, bind, instance_from_source
from fleetgym.spec.models import FilesSource, GeneratorSource, ProblemSpec
from fleetgym.uncertainty.sampling import resample_unrevealed, sample_scenario
from fleetgym.uncertainty.scenario import Scenario
from fleetgym.uncertainty.streams import StreamFactory

__all__ = ["EnvState", "FleetGymEnv", "InvalidActionMode"]

InvalidActionMode = Literal["mask", "penalize", "repair"]
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
    scenario: Scenario | None = None


def _derive_seed(root: int, index: int) -> int:
    ss = np.random.SeedSequence(entropy=root, spawn_key=(index,))
    return int(ss.generate_state(1, np.uint64)[0]) % _SEED_LIMIT


def _infer_max_nodes(spec: ProblemSpec, instance: Instance | None) -> int:
    if instance is not None:
        return instance.n_nodes
    source = spec.instance_source
    if isinstance(source, GeneratorSource):
        from fleetgym.data.generators import validate_generator_params

        params = validate_generator_params(source.name, _thaw(source.params))
        n_customers = getattr(params, "n_customers", None)
        if isinstance(n_customers, int):
            return n_customers + 1
    elif isinstance(source, FilesSource):
        rng = np.random.default_rng(0)
        sizes = []
        for path in source.paths:
            single = spec.model_copy(
                update={"instance_source": FilesSource(paths=(path,), format=source.format)}
            )
            sizes.append(instance_from_source(single.with_source(spec.source), rng).n_nodes)
        return max(sizes)
    raise ConfigError(
        "cannot infer the number of node slots; pass `max_nodes=` or an `instance=`",
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
    """Number of vehicles of a fixed or limited fleet when known before reset (else 0)."""
    if spec.fleet.sizing == "unlimited":
        return 0
    if instance is not None:
        return sum(vt.count or 0 for vt in bind(spec, instance).vehicle_types)
    counts = [vt.count for vt in spec.fleet.types]
    return sum(c for c in counts if isinstance(c, int))


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
    (``"vehicle_idle"`` or ``"recourse"``), ``time``, ``spec_hash`` and ``episode_seed``. The
    scenario is deliberately absent (its id would identify the future); see `scenario`.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}  # noqa: RUF012 - Gymnasium convention

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
        self.max_vehicles = (
            max_vehicles
            if max_vehicles is not None
            else max(self.max_nodes, _known_fleet_size(spec, instance))
        )
        self.interface: ActionInterface = INTERFACE_CLASSES.get(spec.decisions.interface)(
            self.max_nodes
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
        self._final_step_pending = False

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
    def episode_seed(self) -> int:
        """Seed of the current episode."""
        return self._episode_seed

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
        if instance.n_nodes > self.max_nodes:
            raise ConfigError(
                f"instance {instance.name!r} has {instance.n_nodes} nodes but the environment has "
                f"{self.max_nodes} node slots; pass a larger max_nodes",
                source=self.spec_.source,
            )
        problem = bind(self.spec_, instance)
        n_vehicles = sum(vt.count or 0 for vt in problem.vehicle_types)
        if n_vehicles > self.max_vehicles:
            raise ConfigError(
                f"the fleet has {n_vehicles} vehicles but the environment has {self.max_vehicles} "
                "vehicle slots; pass a larger max_vehicles",
                source=self.spec_.source,
                key_path="fleet",
            )
        scenario: Scenario | None = options.get("scenario")
        if scenario is None and self.spec_.uncertainty:
            scenario = sample_scenario(problem, self._episode_seed)
        elif scenario is not None and not self.spec_.uncertainty:
            raise ValueError("this spec has no uncertain parameters; do not pass a scenario")
        log = EventLog(
            {
                "fleetgym_version": __version__,
                "env_id": self.spec_.id or "",
                "spec_hash": problem.spec_hash,
                "instance_hash": problem.instance_hash,
                "instance_name": instance.name,
                "episode_seed": str(self._episode_seed),
                "scenario_id": scenario.id if scenario is not None else "",
            }
        )
        self._sim = Simulator(problem, self._network_for(problem), log=log, scenario=scenario)
        self._monitor.reset()
        self._epoch = self._sim.start()
        # Costs realized before the first decision count in the first step's reward.
        self._carry_costs = self._sim.state.take_pending_costs()
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
            {} if isinstance(decision, RecourseChoice) else {"node": getattr(decision, "node", 0)}
        )
        sim.log.append(
            "decision", vehicle=epoch.vehicle, action=chosen, epoch=sim.state.epoch_count, **target
        )
        transition = sim.apply(decision)
        costs = _merge_costs(self._carry_costs, transition.costs)
        self._carry_costs = {}
        reward = self.reward_model(costs)
        sim.log.append("epoch", epoch=sim.state.epoch_count, reward=reward, time=sim.state.time)
        self._epoch = transition.epoch
        self._mask = self._current_mask()
        if self.debug:
            self._monitor.check(sim)
        info = self._info(costs)
        if chosen != int(action):
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
        self._final_step_pending = False
        reward = self.reward_model(costs)
        sim.log.append("decision", action=0, epoch=sim.state.epoch_count)
        sim.log.append("epoch", epoch=sim.state.epoch_count, reward=reward, time=sim.state.time)
        return self._observation(), reward, True, False, self._info(costs)

    def action_masks(self) -> NDArray[np.bool_]:
        """Valid actions now (the sb3-contrib ``MaskablePPO`` convention)."""
        return self._mask.copy()

    def render(self) -> str | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Text summary of the routes so far (``render_mode="ansi"``)."""
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
            scenario=sim.scenario,
        )

    def set_state(self, state: EnvState) -> None:
        """Restore a snapshot from `get_state` (of this or an identically configured env)."""
        if state.problem.spec_hash != self.spec_.content_hash():
            raise ValueError("the snapshot belongs to a different spec")
        sim = Simulator(state.problem, state.network, log=state.log.copy(), scenario=state.scenario)
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
        if sim.scenario is not None:
            fresh = resample_unrevealed(sim.problem, sim.scenario, sim.revealed_masks(), seed)
            twin.simulator.replace_scenario(fresh)
        return twin

    @property
    def scenario(self) -> Scenario | None:
        """The scenario of the current episode (``None`` without uncertainty)."""
        return self.simulator.scenario

    # ------------------------------------------------------------------ helpers

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
        sim = self.simulator
        assert self._epoch is not None
        if action >= sim.problem.instance.n_nodes:
            n = sim.problem.instance.n_nodes
            return f"action {action} is a padding slot (the instance has {n} nodes)"
        v = sim.state.vehicles[self._epoch.vehicle]
        if self._epoch.kind == "recourse":
            return (
                f"at a recourse epoch the valid actions are {np.flatnonzero(self._mask).tolist()}"
                " (0: return to the depot; the failing customer: outsource)"
            )
        if action == 0 and v.node == 0:
            return "this vehicle cannot retire: the fleet is unlimited and customers are feasible"
        return sim._explain(self._epoch.vehicle, action)  # pyright: ignore[reportPrivateUsage]

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
