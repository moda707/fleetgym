# FleetGym implementation plan: Phase 0 and v0.1

Status: **draft for review** (2026-11-20). No library code is written until this plan is approved.

Sources: `docs/DESIGN.md` (architecture, naming, scope) and the build request (scope, process, milestones).
Where they disagree, the build request wins on scope and process, and the design wins on architecture.
Anything here that departs from the design is marked **[deviation]** and gets an ADR.
Questions that need your decision are collected in §8 and referenced as **Q#**.

---

## 1. Scope

**In scope (Phase 0 + v0.1, milestones M0 to M11).** Spec and config system, Euclidean and matrix
networks, VRPLIB/CVRPLIB and Solomon readers, uniform and clustered generators, discrete-event simulator,
capacity, time window, route duration and fleet size constraints, `next_node`, `accept_reject` and
`assign` interfaces, Gymnasium and PettingZoo envs, the independent validator, uncertainty and scenarios,
VRPSD with return-to-depot recourse, dynamic requests (Poisson, NHPP), core OR baselines, the SB3 stack
with neural network base classes, matplotlib map and Gantt views, and Gate 2 (`VRPSD-Outsource-v0`).

**Out of scope, extension points only.** EV module, time-dependent travel, GPU backend, C++ core,
OSM networks, web replay viewer, leaderboard. "Extension point" means a plugin registry entry type or
an abstract base class, plus a reserved spec field that raises `NotInScopeError` with the config path.
No stub implementations.

**Naming.** Distribution, import package, CLI and Gymnasium namespace are all `fleetgym`; the prose
name is FleetGym; the conventional import alias is `fg`. `routegym` is blocked on PyPI by `route-gym`
(PyPI rejects names differing only in separators), and `dynagym` collides with `SuReLI/dyna-gym` on
GitHub. `fleetgym` was free on PyPI and GitHub on 2026-11-20. See ADR 0001.

---

## 2. Architecture rules as implemented

### 2.1 Layers

Each layer imports only from layers with a lower number. Modules in the same layer may import each other
only where noted. This is enforced by `tests/architecture/test_layers.py`, which parses every module's
imports with `ast` and fails on an upward import (no new dependency needed).

| Layer | Packages | Notes |
| --- | --- | --- |
| L0 foundation | `fleetgym.errors`, `fleetgym.plugins`, `fleetgym._hashing`, `fleetgym._typing`, `fleetgym.eventlog` | No numpy-heavy logic. `eventlog` holds only the schema and Parquet I/O. **[deviation]** `eventlog` is a new top-level package so the validator can read logs without importing `sim`. |
| L1 data | `fleetgym.data` | `Instance`, readers, generators, downloads. Does not know about specs. |
| L2 spec | `fleetgym.spec` | Pydantic models, YAML loader, merge, overrides, `bind()`, config registry. |
| L3 models | `fleetgym.network`, `fleetgym.uncertainty` | Runtime objects built from spec models. Same-layer imports not allowed between the two. |
| L4 simulation | `fleetgym.sim`, `fleetgym.constraints`, `fleetgym.ev` (reserved) | `constraints` may import `sim.state` and `sim.events` only; `sim.engine` imports `constraints`. |
| L5 interfaces | `fleetgym.interfaces`, `fleetgym.envs` | `envs` imports `interfaces`. |
| L6 policies | `fleetgym.baselines`, `fleetgym.agents`, `fleetgym.nn` | `agents` may import `nn`. |
| L7 tooling | `fleetgym.eval`, `fleetgym.viz`, `fleetgym.testing`, `fleetgym.cli` | See the validator rule below. |

**Validator independence (rule 11).** `fleetgym.testing.validator` may import only L0 to L2 and
`fleetgym.uncertainty.scenario` (file I/O of realized values). It may not import `network`, `sim`,
`constraints`, `interfaces` or `envs`. It recomputes distances from instance coordinates with its own
short implementation. The duplication is deliberate: a bug in `network` must not hide itself. The layer
test enforces this rule separately.

### 2.2 Package layout

```text
src/fleetgym/
  __init__.py        public API: make, make_vec, load_spec, ProblemSpec, Uncertain, dist, ...
  errors.py          exception hierarchy (§3.10)
  plugins.py         generic Registry[T] and entry-point loading
  _hashing.py        canonical JSON and SHA-256
  eventlog/          schema.py, recorder-agnostic reader and writer (Parquet)
  data/              instance.py, readers/{vrplib.py, solomon.py}, generators.py, download.py, solution.py
  spec/              models/*.py, loader.py, merge.py, overrides.py, binding.py, registry.py, schema.py
  configs/           deterministic/, stochastic/, dynamic/, multi_agent/, electric/ (YAML only)
  network/           base.py, euclidean.py, matrix.py
  uncertainty/       distributions.py, revelation.py, streams.py, arrivals.py, requests.py,
                     scenario.py, sampling.py, dynamism.py
  constraints/       base.py, capacity.py, time_windows.py, route_duration.py, fleet_size.py
  sim/               events.py, state.py, decisions.py, engine.py, recourse.py, info_state.py,
                     routing_rules.py, recorder.py, invariants.py
  ev/                __init__.py only: docstring stating it is reserved for v0.2+
  interfaces/        base.py, next_node.py, accept_reject.py, assign.py, observations.py, rewards.py
  envs/              env.py, factory.py, vector.py, wrappers.py, multi_agent.py, gym_registration.py
  baselines/         protocol.py, random.py, nearest.py, insertion.py, pyvrp_reopt.py,
                     rollout.py, apriori.py, oracle.py
  agents/            sb3.py, maskable_dqn.py, callbacks.py, hyperparams/*.yaml
  nn/                base.py, adapters.py, encoders.py, decoders.py, policies.py, registry.py
  eval/              harness.py, metrics.py, stats.py, report_card.py, scenario_sets.py
  viz/               map.py, gantt.py, render.py
  testing/           validator.py, env_checker.py, non_anticipativity.py, strategies.py,
                     module_checker.py, golden.py
  cli/               __main__.py, show.py, validate.py, schema.py, replicate.py, train.py, eval.py, render.py
tests/               mirrors src/ plus architecture/, replication/, determinism/, golden/, fixtures/
docs/                DESIGN.md, IMPLEMENTATION_PLAN.md, adr/, (later) tutorials
```

**[deviation]** Configs live at `src/fleetgym/configs/` instead of a top-level `configs/`, so they ship in
the wheel and `fg.make("CVRP-v0")` works after `pip install`. ADR 0002.

---

## 3. Cross-cutting decisions (proposed; each becomes an ADR when implemented)

### 3.1 Immutable specs and the content hash (rule 2)
- All spec models subclass `FrozenModel` (`frozen=True`, `extra="forbid"`, `validate_default=True`).
  Sequences are typed as `tuple[...]`; maps use a read-only `FrozenMap` type with a pydantic core schema.
  A test mutates every container in every registered spec and expects an error.
- `ProblemSpec.content_hash()` = SHA-256 of canonical JSON of `model_dump(mode="json", exclude={"meta"})`
  with defaults included, sorted keys, no whitespace, floats in shortest round-trip `repr`, prefixed with
  a hash-scheme version (`fg1:`). Defaults are included, so changing a default changes the hash.
- `tests/golden/spec_hashes.json` stores the hash of every registered config. A change fails CI unless
  the env version was bumped or the snapshot was regenerated on purpose
  (`uv run python -m fleetgym.testing.golden --update`). This enforces the env-versioning rule in §5.
- An instance has its own `Instance.content_hash()`. A bound problem's identity is `(spec_hash, instance_hash)`.

### 3.2 Config loading pipeline (rule 3)
`load_spec(source, overrides)` runs these steps in order. Every error carries the file path and the
dotted key path (`configs/dynamic/DVRP-StochReq-v0.yaml: decisions.interface`).

1. Read YAML with a safe loader that rejects duplicate keys.
2. Resolve `extends` recursively by ID (registry) or relative path; detect cycles.
3. Merge parent then child (§3.3).
4. Apply CLI-style overrides `a.b.0.c=value` (values parsed as YAML scalars). **[deviation]** Own
   ~100-line parser instead of Hydra, which is not an allowed dependency. ADR.
5. Validate into `ProblemSpec` (pydantic). Plugin-provided parameter models (constraints,
   distributions, recourses) are resolved through the registries.
6. `from_instance` stays symbolic in the spec. It is resolved only by `bind(spec, instance)`.

### 3.3 Merge rules
- Maps merge deeply; scalars and lists in the child replace the parent's.
- `constraints` merges by name (it is a map keyed by constraint name, so this is the map rule).
- `null` in the child deletes the inherited key.
- If a child map changes a **discriminator** (`dist`, `type`, `process`, `model`, `function`), the
  subtree is replaced, not merged. Otherwise `gamma` → `poisson` would keep `shape` and fail validation.
- `meta` is never inherited. Otherwise a child would inherit its parent's replication targets
  (`meta.expected`). See Q7.

### 3.4 Randomness (rule 5)
- Episode root: `SeedSequence(entropy=episode_seed)`. Each uncertainty source gets
  `SeedSequence(entropy=episode_seed, spawn_key=(source_id,))`, where `source_id` is the first 32 bits of
  SHA-256 of the source name (`"demand"`, `"arrivals"`, `"request.demand"`, `"instance"`, ...).
  The stream depends on the source's **name**, not on the order of sources, so adding or removing a
  source never shifts the others. `SeedSequence.spawn()` is order-based and is therefore not used.
- Within a source, entities are drawn vectorized in index order, so draw `k` depends only on
  `(seed, source, k)` and on the distribution parameters of entities `≤ k`. Request attributes use one
  stream per attribute, so request `k`'s demand does not change when the arrival rate changes.
- Gymnasium's `reset(seed=s)` sets the episode seed to `s`. A reset without a seed takes the next
  episode seed from the env's own root `SeedSequence` (deterministic sequence).
- Tests: per-source isolation (change one source's distribution; every other source's draws are identical
  bit for bit), and prefix stability.

### 3.5 Scenarios (rule 6)
- A `Scenario` holds every exogenous realization of an episode and the identity
  `(spec_hash, instance_hash, episode_seed, scenario_id)`. Replaying a scenario against a different spec
  hash raises `ScenarioMismatchError`.
- Parquet format: **one long-format table** `(source: str, entity: int64, field: str, value: float64)`.
  The identity and a schema version go in the Parquet key-value metadata. One file per scenario keeps
  scenario sets easy to ship, and the format is source-agnostic, so new sources need no schema change.
  ADR.
- Decision-dependent randomness (for example the travel time of an arc only if it is driven) is the
  hard case. See Q21.

### 3.6 Simulator semantics (rule 7)
- Event queue ordered by `(time, kind_priority, seq)`. All events with timestamp `≤ t` are processed
  before an epoch at time `t` is presented, so the agent sees everything revealed by `t`. Ties at the
  same time follow a fixed kind priority (cancellation, request arrival, vehicle arrival, service end),
  documented in an ADR.
- Vehicle lifecycle: `IDLE` (decision needed) → `TRAVELING` → arrival → `WAITING` (early for a time
  window) → `SERVING` → `IDLE`. Events that need no decision never return control.
- The simulator understands interface-agnostic **primitive decisions**: `Dispatch(vehicle, node)`,
  `Wait(vehicle, until)`, `EndRoute(vehicle)`, `Accept(request)`, `Reject(request)`,
  `Outsource(request)`, `Assign(request, vehicle)`. Action interfaces translate Gymnasium actions into
  primitives, so the simulator never depends on an interface.
- Costs are charged when they are realized, not when they are committed. Distance is charged at
  departure; lateness is charged at service start. The reward returned by `step` is the negative sum of
  costs realized between the previous epoch and the new one, plus a terminal flush. A test asserts that
  `sum(rewards) == -validator_total_cost` to 1e-9 relative on every rollout.
- `terminated=True` when every request is in a terminal status and every vehicle has ended, or when the
  horizon is reached (time is part of the state, so this is termination, not truncation).
  `truncated` is reserved for wrappers.

### 3.7 State, snapshots and non-anticipativity (rule 10)
- `SimState`, constraint states and interface states are pure data (dataclasses of numpy arrays and
  scalars). They hold no callables or references to the scenario. `get_state()` returns a deep copy
  that includes every `Generator.bit_generator.state` the env owns. `set_state()` restores it. `clone()`
  is `set_state(get_state())` on a fresh env sharing the immutable scenario and network.
- Exactness test: run `k` steps, snapshot, run `m` steps; restore, replay the same actions. Observations,
  rewards, infos and the event log must be bitwise identical. The clone must also be independent of the
  original: mutating one never affects the other.
- **Design gap:** `clone()` copies the true future, so a rollout baseline that simulates on `clone()`
  is clairvoyant. Plan: add `env.fork(seed)`, which keeps everything revealed so far and resamples
  everything unrevealed from the spec's distributions. Rollout and scenario-based baselines must use
  `fork`. See Q22.
- Non-anticipativity test: two scenarios that agree on everything revealed up to epoch `k` must produce
  identical observations, masks and `info["state"]` up to `k`. This runs for every registered config
  and every non-anticipative baseline.

### 3.8 Event log (rule 11, needed in M3, not M10)
One row per event: `episode_id, seq, time, kind, vehicle, node, request, quantity, load_after, value,
cost_component, cost, reward`, nullable where not applicable, plus Parquet metadata (spec hash,
instance hash, scenario ID, package version, log schema version). Revealed values (`value`) are logged
at the moment they are revealed, so the validator can check them against the scenario.
**[deviation]** The event log is built in M3 because the M4 validator needs it; M10 adds only rendering
on top.

### 3.9 Plugin registries and extension points
`fleetgym.plugins.Registry[T]` with `register(name)`, `get(name)` and `names()`. Registering a name
twice raises an error (no silent override). Registries: constraints, distributions, arrival processes,
networks, travel-time models (reserved), recourses, action interfaces, observation builders, instance
readers, generators, baselines, nn modules, energy models (reserved). Entry-point groups:
`fleetgym.plugins` (modules to import) and `fleetgym.configs` (config folders). Config ID collisions
across folders raise an error that names both origins.

### 3.10 Errors
`FleetGymError` → `ConfigError(path, key_path)`, `InstanceError`, `ScenarioMismatchError`,
`InfeasibleActionError`, `NotInScopeError`, `ValidationFailure`. No silent fallbacks: no default
instance when one is missing, no automatic action repair unless `invalid_action="repair"` is set, and no
silent truncation of request streams (Q24).

### 3.11 Python versions and CI
The current numpy (2.5) and scipy (1.18) require Python ≥ 3.12; pyarrow, torch, pyvrp and hypothesis
require ≥ 3.11. Python 3.10 reached end of life in October 2026. Supporting 3.10 and 3.11 means CI there
runs against older numpy and scipy. See Q1. CI matrix (pending Q1): OS × Python, with jobs for ruff,
`pyright` (standard mode) and pytest. Extras (`sb3`, `solvers`, `viz`) are tested on one Python per OS
to keep the matrix small. A final job compares cross-platform determinism digests (§7).

---

## 4. Module breakdown and public interfaces

Signatures are the intended public surface. Bodies, private helpers and exact field lists are fixed during
each milestone and documented in docstrings. `NDArray` is `numpy.typing.NDArray`.

### 4.1 `fleetgym.data` (L1)

```python
@dataclass(frozen=True, eq=False)
class Instance:
    name: str
    source: str                                   # "cvrplib", "solomon", "generated:uniform", ...
    coords: NDArray[np.float64] | None            # (N, 2); node 0..D-1 are depots
    n_depots: int
    demands: NDArray[np.float64]                  # (N,), 0 at depots
    capacity: float | None
    n_vehicles: int | None
    time_windows: NDArray[np.float64] | None      # (N, 2)
    service_times: NDArray[np.float64] | None     # (N,)
    distance_matrix: NDArray[np.float64] | None   # explicit matrices (EXPLICIT edge weights)
    edge_weight_type: str | None                  # "EUC_2D", ...
    extra: Mapping[str, Any]                      # reader-specific fields, read-only
    def get(self, key: str) -> Any: ...           # raises InstanceError listing available keys
    def content_hash(self) -> str: ...

@dataclass(frozen=True)
class Solution:
    routes: tuple[tuple[int, ...], ...]           # customer indices, depots excluded
    cost: float | None                            # as published in the file

def read_instance(path: str | Path, *, fmt: Literal["vrplib", "solomon"] | None = None) -> Instance: ...
def read_solution(path: str | Path) -> Solution: ...

def generate_uniform(rng: np.random.Generator, n_customers: int, *, depot: Literal["center", "corner", "random"] = "center",
                     demand: tuple[int, int] = (1, 10), capacity: float | None = None,
                     time_windows: TimeWindowGen | None = None) -> Instance: ...
def generate_clustered(rng: np.random.Generator, n_customers: int, *, n_clusters: int, spread: float, ...) -> Instance: ...

def fetch(collection: str, name: str, *, cache_dir: Path | None = None) -> Path: ...  # explicit network call, SHA-256 checked
```

### 4.2 `fleetgym.spec` (L2)

```python
class FrozenModel(BaseModel): ...  # frozen, extra="forbid"


FromInstance = Literal[
    "from_instance"
]  # bare form; uses the field's documented default instance key


class FromInstanceRef(FrozenModel):
    from_instance: str  # explicit form: {from_instance: demands}


class ProblemSpec(FrozenModel):
    id: str | None
    meta: Meta  # not hashed, not inherited
    instance_source: InstanceSourceSpec  # operational instance source (Q9)
    network: NetworkSpec  # discriminated by `type`
    fleet: FleetSpec
    constraints: FrozenMap[str, ConstraintSpec]  # keyed by constraint name
    uncertainty: FrozenMap[str, UncertainSpec]  # keyed by source name
    dynamics: DynamicsSpec | None
    decisions: DecisionsSpec
    recourse: RecourseSpec  # location pending Q8
    objective: ObjectiveSpec
    agents: AgentsSpec
    ev: None  # reserved: a non-null value raises NotInScopeError

    def content_hash(self) -> str: ...
    def to_yaml(self) -> str: ...


def load_spec(
    source: str | Path | Mapping[str, Any], *, overrides: Sequence[str] = ()
) -> ProblemSpec: ...
def resolve_raw(
    source: str | Path, *, overrides: Sequence[str] = ()
) -> dict[str, Any]: ...  # for `fleetgym show`
def bind(spec: ProblemSpec, instance: Instance) -> BoundProblem: ...
def json_schema() -> dict[str, Any]: ...


@dataclass(frozen=True)
class BoundProblem:  # spec with every from_instance resolved to arrays
    spec: ProblemSpec
    instance: Instance
    spec_hash: str
    instance_hash: str
    # resolved views: capacities, fleet size, per-node parameters, ...


class ConfigRegistry:
    def ids(self, *, include_planned: bool = False) -> tuple[str, ...]: ...
    def path(self, env_id: str) -> Path: ...
    def status(self, env_id: str) -> Literal["stable", "experimental", "planned"]: ...


REGISTRY: ConfigRegistry  # lazy; scans package configs and entry points once
```

### 4.3 `fleetgym.network` (L3)

```python
Rounding = Literal["none", "nearest_int", "truncate_1dp"]  # final list pending Q18


class Network(ABC):
    n_nodes: int
    is_time_dependent: ClassVar[bool] = False  # extension point for TD models (out of scope)

    @abstractmethod
    def distance(self, i: int, j: int) -> float: ...
    @abstractmethod
    def travel_time(self, i: int, j: int, depart: float) -> float: ...
    def distance_matrix(self) -> NDArray[np.float64]: ...
    def time_matrix(self) -> NDArray[np.float64]: ...  # raises for time-dependent networks


class EuclideanNetwork(Network):
    def __init__(
        self, coords: NDArray[np.float64], *, rounding: Rounding, speed: float = 1.0
    ) -> None: ...


class MatrixNetwork(Network):
    def __init__(
        self, distance: NDArray[np.float64], time: NDArray[np.float64] | None = None
    ) -> None: ...


def build_network(problem: BoundProblem) -> Network: ...
```

### 4.4 `fleetgym.uncertainty` (L3)

```python
class Distribution(ABC):
    kind: ClassVar[Literal["continuous", "discrete"]]

    @abstractmethod
    def sample(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> NDArray[Any]: ...
    def mean(self) -> float: ...  # optional; NotImplementedError when undefined
    def quantile(self, q: float) -> float: ...  # optional
    def pmf(
        self, k: NDArray[np.int64]
    ) -> NDArray[np.float64]: ...  # discrete only; used by exact tests
    def support(self) -> tuple[float, float]: ...


# Fixed, Uniform, Normal, TruncatedNormal, LogNormal, Gamma, Weibull, Exponential, Poisson,
# NegativeBinomial, Bernoulli, Categorical, Empirical, KDE, Mixture, ScipyFrozen (adapter).
# Each docstring states its parameterization explicitly (numpy and scipy conventions differ).


class Revelation(StrEnum):
    NEVER = "never"
    A_PRIORI = "a_priori"
    ON_REQUEST = "on_request"
    ON_DEPARTURE = "on_departure"
    ON_ARRIVAL = "on_arrival"
    ON_COMPLETION = "on_completion"


@dataclass(frozen=True)
class Uncertain:
    dist: Distribution
    revealed: Revelation
    correlated_with: tuple[
        str, ...
    ] = ()  # latent factors: reserved, raises NotInScopeError in v0.1
    forecast: Literal["none", "mean"] = "mean"


class StreamFactory:
    def __init__(self, episode_seed: int) -> None: ...
    def generator(self, source: str) -> np.random.Generator: ...


class ArrivalProcess(ABC):
    def sample(self, rng: np.random.Generator, start: float, end: float) -> NDArray[np.float64]: ...
    def expected_count(self, start: float, end: float) -> float: ...


class PoissonProcess(ArrivalProcess): ...  # rate


class NHPP(ArrivalProcess): ...  # piecewise-constant rate, sampled by thinning


class RequestClass: ...  # joint attribute distributions; weight


@dataclass(frozen=True)
class Scenario:
    spec_hash: str
    instance_hash: str
    episode_seed: int
    scenario_id: str

    def value(self, source: str, entity: int, field: str = "value") -> float: ...
    def column(self, source: str, field: str = "value") -> NDArray[np.float64]: ...
    def to_parquet(self, path: str | Path) -> None: ...
    @classmethod
    def from_parquet(cls, path: str | Path) -> Scenario: ...


def sample_scenario(problem: BoundProblem, episode_seed: int) -> Scenario: ...
def resample_unrevealed(
    problem: BoundProblem, scenario: Scenario, revealed: RevealedSet, seed: int
) -> Scenario: ...
def degree_of_dynamism(release_times: NDArray[np.float64], start: float) -> float: ...
def effective_degree_of_dynamism(release_times: NDArray[np.float64], horizon: float) -> float: ...
```

### 4.5 `fleetgym.sim` and `fleetgym.constraints` (L4)

```python
class Constraint(ABC, Generic[S]):
    name: ClassVar[str]
    Params: ClassVar[type[FrozenModel]]
    hard: bool

    def init_state(self, problem: BoundProblem, info: InformationState) -> S: ...  # see Q16
    def mask(
        self, state: S, info: InformationState, vehicle: int, candidates: NDArray[np.int64]
    ) -> NDArray[np.bool_]: ...
    def check(self, state: S, info: InformationState, decision: Decision) -> Violation | None: ...
    def update(self, state: S, event: Event) -> None: ...
    def penalty(
        self, state: S, event: Event
    ) -> Mapping[str, float]: ...  # cost component -> amount
    def lookahead(
        self, state: S, info: InformationState, route: Sequence[int]
    ) -> bool: ...  # optional


# Capacity, TimeWindows (hard or soft, waiting allowed or not), RouteDuration, FleetSize

Decision = Dispatch | Wait | EndRoute | Accept | Reject | Outsource | Assign


@dataclass(frozen=True)
class Epoch:
    time: float
    kind: Literal["vehicle_idle", "request_arrival", "recourse", "periodic"]
    vehicles: tuple[int, ...]
    request: int | None


class Simulator:
    def __init__(
        self,
        problem: BoundProblem,
        network: Network,
        constraints: Sequence[Constraint[Any]],
        recourse: Recourse,
        scenario: Scenario,
        *,
        recorder: EventRecorder | None = None,
    ) -> None: ...
    def start(self) -> Epoch | None: ...
    def apply(
        self, decisions: Sequence[Decision]
    ) -> Transition: ...  # next epoch, realized costs, terminated
    def feasible(self, vehicle: int, candidates: NDArray[np.int64]) -> NDArray[np.bool_]: ...
    def information_state(self) -> InformationState: ...  # read-only, revealed info only
    def snapshot(self) -> SimState: ...
    def restore(self, state: SimState) -> None: ...


class Recourse(ABC): ...  # ReturnToDepot (M5), AgentDecides (M5), Outsource (M11, pending Q-M11)
```

`sim/routing_rules.py` holds the built-in routing used *inside* interfaces (cheapest insertion for
`accept_reject` and `assign`). The `baselines` layer reuses it, so no lower layer imports a baseline
(Q27).

### 4.6 `fleetgym.interfaces` and `fleetgym.envs` (L5)

```python
class ActionInterface(ABC):
    name: ClassVar[str]

    def action_space(self, problem: BoundProblem) -> gym.spaces.Space[Any]: ...
    def needs_decision(self, epoch: Epoch) -> bool: ...
    def mask(self, sim: Simulator, epoch: Epoch) -> NDArray[np.bool_]: ...
    def decode(self, action: Any, sim: Simulator, epoch: Epoch) -> list[Decision]: ...
    def repair(self, action: Any, mask: NDArray[np.bool_]) -> Any: ...


# NextNode: Discrete(max_nodes [+1 wait, Q14]); AcceptReject: Discrete(3); Assign: Discrete(V+1) (Q26)


class ObservationBuilder(ABC):
    def space(self, problem: BoundProblem) -> gym.spaces.Dict: ...
    def build(self, info: InformationState, epoch: Epoch) -> dict[str, NDArray[Any]]: ...


# PaddedGraphObs: nodes (max_nodes, F) float32, node_valid, vehicles (V, G), current_vehicle, globals,
# action_mask. Padded to max_nodes from the spec or bound problem.


class FleetGymEnv(gym.Env[dict[str, NDArray[Any]], Any]):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"]}

    def __init__(
        self,
        spec: ProblemSpec,
        *,
        render_mode: str | None = None,
        invalid_action: Literal["mask", "penalize", "repair"] = "mask",
        record_events: bool = False,
    ) -> None: ...
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        ...
        # options: {"instance": Instance, "scenario": Scenario}

    def step(self, action: Any):
        ...
        # info: costs (dict), action_mask, epoch, state (InformationState), spec_hash, scenario_id

    def action_masks(self) -> NDArray[np.bool_]: ...  # sb3-contrib convention
    def get_state(self) -> EnvState: ...
    def set_state(self, state: EnvState) -> None: ...
    def clone(self) -> FleetGymEnv: ...  # exact, includes the true future
    def fork(self, seed: int) -> FleetGymEnv: ...  # non-anticipative (Q22)
    @property
    def event_log(self) -> EventLog: ...


def make(
    env: str | Path | ProblemSpec, *, overrides: Sequence[str] = (), **env_kwargs: Any
) -> FleetGymEnv: ...
def make_vec(
    env: str | Path | ProblemSpec,
    n_envs: int,
    *,
    mode: Literal["sync", "async"] = "async",
    overrides: Sequence[str] = (),
    **env_kwargs: Any,
) -> gym.vector.VectorEnv: ...
def aec_env(env: str | Path | ProblemSpec, **kw: Any) -> FleetGymAECEnv: ...
def parallel_env(env: str | Path | ProblemSpec, **kw: Any) -> FleetGymParallelEnv: ...
```

**[deviation]** The design writes `make(id, **overrides)`. Dotted override paths cannot be keyword
names, so overrides go in `overrides=[...]` and `**env_kwargs` is for constructor arguments. ADR.
Every config is also registered with Gymnasium as `fleetgym/<ID>`, so `gymnasium.make("fleetgym/CVRP-v0")`
works.

### 4.7 `fleetgym.baselines` and `fleetgym.eval` (L6, L7)

```python
class Policy(Protocol):
    anticipative: ClassVar[bool]  # True only for the oracle; the harness labels results

    def reset(self, env: FleetGymEnv) -> None: ...
    def act(self, obs: dict[str, NDArray[Any]], info: dict[str, Any]) -> Any: ...


# RandomPolicy, NearestNeighbor, CheapestInsertion, AprioriRoute, PyVRPReoptimization, Rollout(base, n_samples),
# PerfectInformation (anticipative=True; receives the scenario explicitly)


def evaluate(
    policy: Policy,
    env: str | ProblemSpec,
    scenarios: ScenarioSet | Sequence[Scenario],
    *,
    decision_time_limit: float | None = None,
    record: bool = False,
) -> EvalResult: ...
def paired_comparison(
    a: EvalResult, b: EvalResult, *, test: Literal["wilcoxon", "t"] = "wilcoxon"
) -> Comparison: ...
def cvar(costs: NDArray[np.float64], alpha: float = 0.9) -> float: ...


class ScenarioSet: ...  # (env_id, split, seed range) -> deterministic scenarios
```

### 4.8 `fleetgym.agents` and `fleetgym.nn` (L6; extras `sb3`, `gnn`)

Importing either package without torch raises `ImportError` naming the extra to install.

```python
def sb3_defaults(env_id: str, algo: str) -> dict[str, Any]: ...      # agents/hyperparams/<algo>.yaml
class SB3ObsWrapper(gym.ObservationWrapper): ...                     # fixed-shape Dict of float32 arrays
class MaskableDQN(DQN): ...                                          # -inf on masked Q in selection and target
class ScenarioEval(BaseCallback): ...; class CostComponentLogger(BaseCallback): ...

# nn: FleetGymModule, ObservationAdapter, Encoder, ContextBuilder, Decoder, ValueHead,
#     MaskableGraphPolicy (MaskableActorCriticPolicy), MaskableQNetwork, register(name)
# built-ins: MLPEncoder, GATEncoder, AttentionModelEncoder, PointerDecoder, DynamicContext
```

### 4.9 `fleetgym.testing`, `fleetgym.viz`, `fleetgym.cli` (L7)

```python
def validate(
    log: EventLog | str | Path,
    spec: ProblemSpec,
    instance: Instance,
    scenario: Scenario | None = None,
) -> ValidationReport:
    ...
    # checks: event consistency (monotone clock, travel = distance/speed, service durations),
    # capacity, time windows, route duration, fleet size, request-status lifecycle,
    # realized values == scenario, recomputed costs == -sum(rewards)


def check_env(env: str | FleetGymEnv, *, n_rollouts: int = 20, seed: int = 0) -> None:
    ...
    # gymnasium check_env + mask/validator agreement + seeding + clone exactness + non-anticipativity


def check_non_anticipative(
    env: str, policy: Policy | None = None, *, n_pairs: int = 50
) -> None: ...
def check_module(module_cls: type[FleetGymModule], **kwargs: Any) -> None: ...  # M9


# strategies: problem_specs(), instances(max_nodes=8), masked_action_sequences(env)

# viz: plot_map(log, problem, t=None, ax=None), plot_gantt(log, problem, ax=None)
# cli: fleetgym {list, show, validate, schema, replicate, train, eval, render}  (argparse, stdlib only)
```

---

## 5. Milestones, dependency order and test plans

Order: M0 → M1 → M2 → M3 → **M4 (Gate 1, stop)** → M5 → M6 → {M7, M8} → M9 → M10 → **M11 (Gate 2, stop)**.
M7 and M8 are independent; M8 needs M5 (`fork`) and M6. Each milestone ends with: full test suite,
`STATUS.md`, `CHANGELOG.md`, ADRs for its decisions, a conventional commit, and a short summary to you.

### M0 Scaffolding
- `git init`, `pyproject.toml` (hatchling, src layout, extras per the build request), `uv.lock`,
  `.pre-commit-config.yaml` (ruff, ruff-format, pyright hook via `uv run`), `.github/workflows/ci.yml`,
  the package skeleton from §2.2 with module docstrings only, `LICENSE` (Q3), `STATUS.md`,
  `CHANGELOG.md`, `docs/adr/0001` and `0002`.
- Tests: import smoke test for every module; the layer test (§2.1) active from day one; CI green on the
  full matrix.

### M1 Spec and configs
- Pydantic models for every spec section, the loader pipeline (§3.2), merge rules (§3.3), overrides,
  `from_instance` and `FromInstanceRef`, the config registry and entry points, JSON Schema export
  (`fleetgym schema`), `fleetgym list | show | validate`, and the spec hash with its golden snapshot.
- Configs: `CVRP-v0` and `VRPTW-v0` as stable. `DVRP-StochReq-v0`, `VRPSD-Outsource-v0`,
  `EVRP-NL-v0` and `MA-DVRPTW-v0` are handled per Q6.
- Tests: unit tests per model (valid and invalid fixtures, with the error path asserted); merge-rule
  table tests (deep merge, by-name constraints, `null` deletion, discriminator replacement, `meta`
  not inherited, cycle detection); round trip `load → to_yaml → load` gives an equal hash; hash
  invariance under key order and YAML formatting; hash changes when any default changes; deep
  immutability; Hypothesis: random valid spec dicts always validate and round-trip; the JSON Schema
  validates every shipped config; CLI golden-output tests.

### M2 Network and instances
- `Instance`, VRPLIB/CVRPLIB and Solomon readers (through `vrplib`), `Solution` reader, uniform and
  clustered generators, `EuclideanNetwork` with the rounding modes from Q18, `MatrixNetwork`, `bind()`,
  and `fetch()` with checksums (never called by the default test run).
- Tests: readers against small fixture files (Q18 on bundling), distance matrices checked by hand on
  three-node examples for each rounding mode, symmetric and asymmetric matrices, generator determinism
  and statistics (counts, bounds, cluster spread), and `from_instance` errors naming the missing key.

### M3 Simulator core
- Event queue, state, primitive decisions, engine, the four constraints, information state, event
  recorder and Parquet log, `next_node` interface, padded observations, rewards, `FleetGymEnv`, `make`,
  Gymnasium registration, `get_state`, `set_state` and `clone`, and debug invariant monitors.
- Tests: unit tests per constraint (mask, check, update, penalty) on hand-built states; event ordering
  and tie-break tests; `check_env` on every non-planned config; determinism (same spec and seed give an
  identical trajectory digest, also across two subprocesses); snapshot exactness and clone
  independence; reward sum equals negative total cost; Hypothesis: random specs plus random masked
  action sequences never crash and never trip an invariant; a mask is never all false at a
  non-terminal epoch.

### M4 Validator and Gate 1
- `fleetgym.testing.validator` (independent; §2.1), `check_env`, and `fleetgym replicate` for model
  fidelity.
- Gate 1 tests:
  1. **Mask and validator agreement:** 10,000 random masked rollouts across all non-planned configs
     and generated instances; the validator accepts every one. Also the converse on small instances:
     for each epoch, every action the mask *rejects* is checked with `clone()` and must lead to a
     violation the validator reports, so masks are not overly strict either.
  2. **Model fidelity:** CVRPLIB and Solomon best-known solutions replayed as action sequences; each
     is feasible and the environment's cost equals the published cost within the tolerance agreed in
     Q18 and Q19.
  3. **Golden optima:** on instances with up to 7 customers, an exhaustive search over the environment
     (DFS with `clone()`) and an independent brute-force solver find the same optimum.
- Marked `@pytest.mark.slow` where needed. A fast subset runs on every push; the full gate runs in a
  scheduled job and before tags.
- **Stop for your approval.**

### M5 Uncertainty and scenarios
- Distribution API and built-ins, revelation model, stream factory, `Scenario` with Parquet I/O,
  scenario sampling, `fork()` with `resample_unrevealed`, `VRPSD-v0` with return-to-depot recourse,
  the `AprioriRoute` policy (needed for the exactness test), and the non-anticipativity checker.
- Tests: KS (continuous) and chi-square (discrete) tests for every sampler at fixed seeds with
  α = 1e-3 (deterministic, so not flaky); `mean()` and `quantile()` against scipy; per-source stream
  isolation; Parquet round trip is exact; scenario replay gives an identical trajectory; mismatched
  spec hash raises; **exact expectation**: for small instances with finite-support demands (for example
  5 customers × 3 demand values = 243 scenarios), the probability-weighted sum of environment costs over
  all scenarios equals an independent dynamic-programming computation of the a priori route's expected
  cost to 1e-9 (recourse convention per Q20); a Monte Carlo sanity check for Poisson demands within a
  99.9% confidence interval; non-anticipativity on `VRPSD-v0`.

### M6 Dynamic requests
- Poisson and NHPP (thinning) arrivals, request classes, cutoff, cancellations, degree of dynamism and
  effective degree of dynamism, `accept_reject` and `assign` interfaces, `routing_rules.cheapest_insertion`,
  periodic epochs if Q6 keeps `DVRPTW-v0`, and the `DVRP-StochReq-v0` config.
- Tests: arrival counts per period vs the rate integral (chi-square over many seeds); thinning
  correctness against a piecewise-constant rate; inter-arrival KS test for the homogeneous case;
  dynamism metrics on hand-computed examples; cutoff and cancellation behavior; accept masked exactly
  when no feasible insertion exists; overflow behavior per Q24; non-anticipativity, determinism and
  `check_env` on all dynamic configs.

### M7 Multi-agent
- `FleetGymAECEnv` and `FleetGymParallelEnv` on the same simulator (agent = vehicle), `state()` and
  `state_space` for a central critic, conflict resolution (priority, distance, lottery with its own RNG
  stream), global, individual or mixed reward, and per-agent termination.
- Tests: PettingZoo `api_test` and `parallel_api_test` for every multi-agent config; the AEC and
  Parallel versions give the same total cost on the same scenario and joint actions when no conflicts
  arise; conflict-rule unit tests; an AEC env with a single vehicle equals the single-agent env.

### M8 Baselines and evaluation harness
- `Policy` protocol, random, nearest neighbor, cheapest insertion, a priori route, PyVRP
  re-optimization (extra `solvers`), rollout over `fork()`, perfect information (anticipative), and a
  minimal `eval` harness (scenario sets, metrics, CIs, CVaR, paired tests, JSON report card).
- Tests: every baseline produces validator-accepted trajectories on every config it supports; the
  non-anticipativity check passes for every non-anticipative baseline; the perfect-information cost is
  ≤ every non-anticipative policy's cost on the same scenario for tiny instances solved exactly (Q29);
  rollout with a deterministic base policy on a deterministic config equals the base policy's
  improvement step computed by hand; baseline regression snapshots; statistics checked against scipy.

### M9 Training stack
- SB3 observation wrapper, `make_vec`, MaskablePPO defaults, `MaskableDQN`, callbacks, nn base
  classes, `MLPEncoder`, `GATEncoder`, `AttentionModelEncoder` + `PointerDecoder`, `MaskableGraphPolicy`,
  `check_module`, `fleetgym train` and `fleetgym eval`.
- Tests: `check_module` on every built-in module (shapes, -inf on masked logits, permutation
  equivariance over customers, batch and padding invariance); `MaskableDQN` never selects a masked
  action and its target ignores masked actions (unit test on a hand-built batch); agent sanity suite
  (tiny envs that MaskablePPO solves within a small budget, as a slow test); CLI smoke test for
  `train` with a 1,000-step budget, then `eval` on the checkpoint.

### M10 Visualization
- Matplotlib map and Gantt views from the event log, `rgb_array`, `human` and `ansi` render modes,
  `fleetgym render`.
- Tests: headless (`Agg`) rendering of every view for every config; `rgb_array` shape and dtype;
  rendering does not change the trajectory (digest equal with and without rendering); image smoke
  tests (no pixel-exact comparisons, which are brittle across matplotlib versions).

### M6b Dynamic requests with next_node (added at the user's request, 2026-11-20)
- Requests arriving during the day are accepted automatically (until the cutoff, for example the
  last 20% of the shift) and become customers the policy can choose; the set of nodes grows.
- `next_node` gains a last action, **wait**, under dynamics: the vehicle is asked again at the next
  event (a request arriving or cancelling, another vehicle arriving or finishing a service) or at its
  latest safe departure time. Waiting is allowed only on known information (the clock and the
  arrival window), never on whether more requests will come.
- At the depot during the day the vehicle is reloaded and may leave again, wait, or **terminate**
  (choosing the depot), terminating only if another vehicle is still working (as in the user's
  VRP-SDO paper).
- User decisions (2026-11-20): wait until the next event; depot visits reload, with leave, wait and
  terminate options. Cancellation timing stays an open item for the future (backlog).

### M10b Documentation site and examples (added at the user's request, 2026-11-20)
- A documentation site in the style of Grid2Op's (https://grid2op.readthedocs.io): getting started,
  concepts (specs and `extends`, scenarios and revelation, recourse, dynamic requests and planned
  routing, rewards, the validator), the environment catalog generated from the configs, the API
  reference generated from the NumPy docstrings, and the research conventions (replication,
  non-anticipativity, determinism).
- The central guide: **build your own VRP variant**, in steps: (1) a new variant from YAML only
  (`extends`, overrides, `fleetgym validate`); (2) a custom distribution; (3) a custom recourse rule;
  (4) a custom constraint; (5) a plugin package with entry points. Every code block is executed in
  the test suite so the guide cannot go stale.
- An `examples/` folder of runnable scripts and notebooks: first environment; stochastic demands with
  recourse; a dynamic day with accept/reject; comparing policies on fixed scenario sets; a custom
  variant end to end.
- Dependencies approved by the user (2026-11-20) for a `[docs]` extra: `sphinx`, a theme
  (`pydata-sphinx-theme` or `furo`), `myst-parser`, `sphinx-autodoc-typehints`, and `sphinx-gallery`
  if the examples gallery uses it. Hosting needs a Read the Docs project linked by the user.

### M11 Gate 2
- Ask you first for the exact VRP-SDO model, parameters, instances and published results. Then
  implement `VRPSD-Outsource-v0` and its recourse, plus replication tests against your published
  numbers.
- **Stop for your approval.**

---

## 6. Test and quality infrastructure

- Layout: `tests/` mirrors `src/fleetgym/`, plus `architecture/`, `determinism/`, `replication/`,
  `golden/` and `fixtures/` (small instance and solution files, generated scenarios).
- Markers: `slow` (Gate suites, training sanity), `network` (downloads; never in default CI),
  `extras_sb3`, `extras_solvers`, `extras_viz`. The default `pytest` run excludes `slow` and `network`.
- Hypothesis profiles: `dev` (fast) and `ci` (more examples, `derandomize=True` so CI is reproducible).
- Coverage: 90% line coverage on L0 to L5 (Q2 on tooling).
- Determinism across platforms: each OS job writes trajectory digests for a fixed set of
  (config, seed) pairs; a final job compares them. Discrete outcomes (actions, event kinds, request
  statuses) must match exactly; floats within 1e-9 relative, because libm `exp`/`log` may differ in the
  last bit across platforms (Q23).

---

## 7. Risks specific to this build

| Risk | Mitigation |
| --- | --- |
| BKS conventions (rounding, objective) differ between sources, so Gate 1 fails for reasons unrelated to bugs | Settle Q18 and Q19 before M4; record the source of every solution file in fixtures. |
| Rollout or policies accidentally see the future through `clone()` or `info` | `fork()`, the `anticipative` flag, and the non-anticipativity checker on every config and baseline. |
| Pure-Python engine too slow for 10,000-rollout tests | Small generated instances for property tests; profile in M3; numba only after profiling. |
| PyVRP API changes (0.14 now) | Pin a compatible range in the `solvers` extra; adapter isolated in one module. |
| SB3 classifiers stop at Python 3.13; pettingzoo requires < 3.15 | Test extras only on supported Pythons; record caps in the README. |
| Scope creep from the 42-variant catalog | Only the configs named in M1 to M11 are written; others are not stubbed. |

---

## 8. Open questions and design gaps

Each item has a recommendation. Items are grouped by the first milestone they block.

**Resolved (2026-11-20).** Q1: Python ≥ 3.12, CI on 3.12 to 3.14 (ADR 0005). Q2: `pytest-cov` and
`pytest-xdist` added to `[dev]`. Q3: MIT with `CITATION.cff` (ADR 0003). Q5: confirmed (Gate 1 = M4,
Gate 2 = M11). Q6: `meta.status`; start with a few configs and expand. Q7: `objective` is replaced, not
merged; `meta` is not inherited; a discriminator change replaces the subtree. Q20: standard plus custom
recourse, design to be discussed before M5 (see below).
**Adopted in M1 (recommendations, user unavailable):** Q8, Q9, Q10, Q11, Q12, Q24, Q28 (ADRs 0006,
0007). Adopted in M3: Q13 to Q17 (ADR 0009). M4: Q18/Q19 resolved for CVRPLIB X from the conventions CVRPLIB publishes (ADR 0010), and for Solomon from SINTEF TOP's published route files and conventions (ADR 0011). Adopted in M5: Q20 (implemented as agreed, ADR 0013), Q21 to Q23 (ADR 0012). Adopted in M6: Q24 to Q27 (ADR 0014). Adopted in M7: Q28 (ADR 0016). Adopted in M8: Q29, Q30 checked (ADR 0017). Not adopted yet: Q31.
**Gate 1 approved by the user (2026-11-20).** Q4: the repository is `github.com/moda707/fleetgym`
(personal account); the PyPI name is taken by the first release through trusted publishing
(`.github/workflows/release.yml`, set up by the user). Q20: the user asked to start M5 with the agreed
design (standard rules plus user-written custom rules) and to keep a list of later improvements and
extensions (`docs/recourse.md`).
**Policy for unanswered questions:** where the user has not answered, the recommendation is adopted
when its milestone starts and recorded in an ADR, so it can be overridden later. Exceptions that
require the user: Q18 and Q19 (Gate 1 data), Q20 (recourse design), Q-M11.

### Before M0
- **Q1. Python versions.** The request says ≥ 3.10 with CI on 3.10 to 3.13. Python 3.10 reached end of
  life in October 2026, and numpy 2.5 and scipy 1.18 require ≥ 3.12. *Recommendation:*
  `requires-python >= 3.12`, CI on 3.12, 3.13 and 3.14. The alternative keeps 3.10 and 3.11 against
  older numpy and scipy.
- **Q2. Coverage tooling.** The 90% target needs `pytest-cov` (or `coverage`) in `[dev]`. Optionally
  `pytest-xdist` to parallelize the Gate 1 suite. *Recommendation:* add both to `[dev]`.
- **Q3. License.** MIT or Apache-2.0 (design §25). *Recommendation:* Apache-2.0 if industrial
  contributors are likely, otherwise MIT.
- **Q4. GitHub home and PyPI name.** On 2026-11-20, `fleetgym` was unregistered on PyPI and the
  `fleetgym` GitHub organization name was free. Names are first come, first served, so
  *recommendation:* you create the `fleetgym` GitHub organization and reserve the PyPI name soon
  (I won't publish anything myself). Which owner/repo should go in project URLs? Until you decide,
  they stay empty.
- **Q5. Roadmap.** §26's roadmap and gate definitions are an embedded diagram missing from the text I
  received. I assume Phase 0 + v0.1 = M0 to M11, Gate 1 = M4 and Gate 2 = M11. Please confirm, or put
  the diagram's content into `docs/DESIGN.md`.

### Before M1
- **Q6. The six example configs.** `EVRP-NL-v0` needs the EV module (out of scope); `MA-DVRPTW-v0`
  extends `DVRPTW-v0`, which is not among the examples; `VRPSD-Outsource-v0` belongs to Gate 2 and needs
  your paper's details. *Recommendation:* add `meta.status: stable | experimental | planned`. Planned
  configs are written and must validate against the schema, are listed by `fleetgym list --all`, raise
  `NotInScopeError` on `make`, and are skipped by env checks. `EVRP-NL-v0` and `VRPSD-Outsource-v0`
  start as planned; for multi-agent, see Q28.
- **Q7. Merge pitfalls in the design's own examples.** (a) `DVRP-StochReq-v0` extends `VRPTW-v0`, whose
  objective is `{lexicographic: [...]}`, and adds `{distance, rejected}`. Under deep merge the result has
  both and is invalid. Should the example add `lexicographic: null`, or should `objective` always be
  replaced rather than merged? *Recommendation:* replace `objective` wholesale. (b) `meta` is not
  inherited (§3.3). (c) A discriminator change replaces the subtree (§3.3). Please confirm (b) and (c).
- **Q8. Inconsistencies between the Python and YAML forms.** `recourse` is top-level in the YAML but
  `decisions.recourse` in Python; `constraints` is a list in Python and a map in YAML; Python writes
  `mean="instance"`, YAML `from_instance`; YAML uses a flat `Uncertain` form (`{dist: poisson, mean: ...,
  revealed: ...}`). *Recommendation:* top-level `recourse` map (per failure type); constraints keyed by
  name everywhere (Python accepts a list and converts it); `from_instance` only; flat YAML form with
  `dist` as the discriminator.
- **Q9. Operational instance source vs `meta.instances`.** `meta` documents the variant and is excluded
  from the hash, but which instances an env samples changes its behavior. What should `fg.make("CVRP-v0")`
  do with no instance? *Recommendation:* a top-level, hashed `instance_source` (generator with
  parameters, or a file list) gives every config an offline default; `meta.instances` stays for
  replication only.
- **Q10. Instance references.** Bare `from_instance` needs a documented default mapping per field
  (`fleet.types[].capacity` → `capacity`, `count` → `n_vehicles`, `uncertainty.demand.mean` → `demands`).
  `route_duration.max: depot_due_time` is another symbolic value. *Recommendation:* support bare
  `from_instance` with a per-field default key, the explicit form `{from_instance: <key>}`, and treat
  `depot_due_time` as a named key in the instance (`time_windows[depot][1]`).
- **Q11. Overrides.** Hydra is not an allowed dependency. *Recommendation:* own dotted-path parser and
  `make(id, overrides=[...])` (§4.6).

### Before M3
- **Q12. Epoch order in static problems and unlimited fleets.** With an unlimited fleet, "every idle
  vehicle at the depot" is unbounded, so the time-ordered `on_event` rule has no natural meaning.
  RL4CO-style construction builds one route at a time. *Recommendation:* a `sequential` epoch rule (vehicle
  `k` completes its route before `k+1` starts; allowed only when there is no exogenous information) as
  the default for deterministic configs, plus `on_event` (time-ordered, concurrent) for everything else.
  Gate 1 checks that both modes give the same cost for the same routes on fixed-fleet instances.
- **Q13. Dead ends with a fixed fleet.** A mask cannot guarantee that all customers can still be served
  (that is a bin-packing question). When vehicles run out, the episode ends with unserved requests.
  *Recommendation:* a fixed or limited fleet requires an `objective.unserved` weight; validation fails
  without it. No hidden default penalty.
- **Q14. Waiting in `next_node`.** In dynamic settings, a vehicle may need to wait for future requests.
  `next_node` lists only nodes. *Recommendation:* an extra `WAIT` action (waits until the next event),
  masked out when the spec has no dynamics; at the depot, choosing the depot means `EndRoute`.
- **Q15. Lexicographic objective vs scalar reward.** `VRPTW-v0` uses `[vehicles, distance]`.
  *Recommendation:* the env reports both components in `info["costs"]` and the reward uses weights
  `vehicles: M, distance: 1`, with `M` a declared, documented constant (for example `M = 1e6`).
  The validator and replication compare components lexicographically. Alternative: vector reward
  only, which SB3 cannot use.
- **Q16. `Constraint.init_state(spec, scenario)` exposes the future.** A constraint with the scenario
  could build anticipative masks. *Recommendation:* constraints receive the `InformationState` only; the
  simulator feeds revealed values through events.
- **Q17. Event log timing.** The validator (M4) reads the event log, so the log has to exist in M3.
  The visualization in M10 then builds on it. Please confirm this reorder.

### Before M4
- **Q18. Rounding conventions and solution files.** CVRPLIB X uses `EUC_2D` with nearest-integer
  rounding. For Solomon, published values differ by source: double-precision distances,
  truncation to one decimal (some exact-method papers), hierarchical BKS (vehicles first) vs
  distance-only BKS. Which BKS set do you trust for each family, and do you have the solution files?
  Can I bundle a few small instances and solutions as test fixtures (licensing), or should tests
  download them under the `network` marker?
- **Q19. Tolerance.** Published costs are typically given to two decimals, so the design's
  `tolerance: 1e-6` cannot be met against the printed value. *Recommendation:* recompute the cost from
  the solution's routes and compare with 1e-6; separately check that it rounds to the published value.

### Before M5
- **Q20. VRPSD recourse conventions.** The exact expected cost depends on details I must not guess:
  (a) when demand equals the remaining load exactly, does the vehicle restock preemptively before the
  next customer, or only when a later failure occurs? (b) Can a single customer's demand exceed
  capacity (multiple returns), or is demand truncated at `Q`? (c) On a failure, is the customer served
  partially before the return trip (split service), with the trip depot → customer → resume? Please give
  the convention, ideally with the paper and equation, for the exactness test.
  **User direction (2026-11-20):** ship standard recourse rules (for example return to the depot and
  come back) *and* support user-written custom recourse. The design of the recourse interface is to be
  **discussed with the user before M5 starts**. Bring a concrete proposal: what a recourse rule receives
  (failure event, information state), what it may return (primitive decisions, costs), how it is
  registered and referenced from YAML, how "agent decides" fits (a recourse epoch for the policy),
  how the validator checks custom recourse independently, and how an exact-expectation test is written
  for a custom rule.
- **Q21. Decision-dependent randomness.** Travel and service times are realized only on arcs and visits
  that a policy chooses. Pre-sampling them is either an `n × n` matrix (one draw per arc per episode,
  repeated traversals share a draw) or keyed per traversal. *Recommendation:* v0.1 supports stochastic
  service times (one per customer visit) and defers stochastic travel times to `VRPSTT-v0` after v0.1.
  If you want them in v0.1, I suggest per-arc draws keyed by `(i, j, k)` (k = traversal count) from a
  counter-based stream, materialized lazily and recorded in the log.
- **Q22. `clone()` vs `fork()`.** As in §3.7. Confirm that rollout-type baselines must use `fork`, and
  that results from anticipative policies are always labeled and never mixed into leaderboards.
- **Q23. Cross-platform determinism.** Bitwise equality across OSes is not guaranteed for transcendental
  functions. *Recommendation:* exact on the same platform; across platforms, exact discrete outcomes and
  floats within 1e-9 relative.

### Before M6
- **Q24. Request overflow.** Observations are padded to `max_nodes`, but a Poisson stream can exceed any
  bound. *Recommendation:* `dynamics.max_requests` is required; overflow behavior is set by
  `overflow: error | reject` with default `error`. `reject` logs each automatic rejection and charges it
  as rejected.
- **Q25. Requests after the cutoff.** Are they not generated at all, or generated and automatically
  rejected (charged)? These give different costs and degree-of-dynamism values.
  *Recommendation:* not generated, and the cutoff truncates the arrival window.
- **Q26. Built-in routing under `accept_reject` and `assign`.** What do vehicles do with an empty plan
  (wait at their location, or return to the depot)? Is waiting drive-first? Is `outsource` always in the
  action space and masked when the spec has no outsourcing? What is the `+1` in `assign`'s
  `Discrete(V+1)`: reject or defer? *Recommendation:* drive-first, wait at the last customer; outsource
  masked when unavailable; `+1` = reject (defer later).
- **Q27. Layering of built-in heuristics.** The interfaces need cheapest insertion, which the design
  lists under baselines (a higher layer). *Recommendation:* the heuristic lives in
  `sim/routing_rules.py`; the `CheapestInsertion` baseline wraps it.

### Before M7
- **Q28. Multi-agent presets for v0.1.** `MA-DVRPTW-v0` depends on `DVRPTW-v0` (EURO-NeurIPS periodic
  epochs), which needs a `plan`-style interface to be faithful. Should every preset automatically get an
  `MA-` version, or only explicit files? *Recommendation:* explicit files only; ship `MA-CVRP-v0` and
  `MA-DVRP-v0` (next_node with WAIT) in v0.1; `MA-DVRPTW-v0` stays planned.

### Before M8
- **Q29. "Oracle bound".** The hindsight solution is a lower bound on non-anticipative policies only if
  it is solved to optimality. PyVRP is a heuristic, so its hindsight cost is not a bound.
  *Recommendation:* report it as "perfect-information heuristic"; call it a bound only on instances
  solved exactly (brute force or MIP) and say so in the report card.
- **Q30. PyVRP for re-optimization.** Re-optimizing mid-day needs vehicles starting en route with
  partial loads and time offsets. I will verify which of these PyVRP 0.14 models natively and report
  before implementing. No question yet; just flagging.

### Before M9
- **Q31. Graph encoders and SB3.** (a) `GATEncoder` without torch-geometric: *recommendation:* a
  dense, mask-aware pure-torch implementation in `[sb3]`, plus a PyG version in `[gnn]`. (b) The §19
  example uses `GATEncoder` as an SB3 `features_extractor_class`, which flattens node embeddings and
  loses the pointer structure. *Recommendation:* `MaskableGraphPolicy` keeps per-node embeddings and
  computes logits with a pointer decoder; the features-extractor path stays available for MLP-style
  heads. (c) The design names the policy both `MaskableGraphPolicy` (§19) and `RouteGymPolicy` (§20).
  *Recommendation:* `MaskableGraphPolicy`.

### M11
- **Q-M11.** Exact VRP-SDO model, parameters, instances and published results (requested in the build
  instructions; asked before M11 starts).
