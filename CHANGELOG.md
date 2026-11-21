# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package uses
[Semantic Versioning](https://semver.org/). Environment IDs are versioned separately (`-v0`, `-v1`).

## [Unreleased]

### Added
- M12 (ADR 0021): `VRP-VCSD-v0` and `VRPSD-Goodson-v0`, the problem settings of Dastpak, Errico and
  Jabali (arXiv 2109.10200); recourse rule `split` (serve what fits, the rest stays open for any
  vehicle, restock whenever the vehicle chooses), which works with route-duration and shift limits;
  revenue component `served_demand` (negative weight); distributions `discrete_uniform` and
  `scaled_categorical`; generator `zones`; instance source `collection`; examples
  `09_vrp_vcsd` and `10_vrpsd_goodson`.

### Changed
- Examples: one folder per example, each with its script and a notebook generated from it
  (`scripts/build_example_notebooks.py`); the tests run both and check their cells are in sync
  (outputs of a local run are ignored).
- Scope (user, 2026-11-20): the EV shortest path with charging-station occupancy (EVSPP-OI) is in
  scope as M14.

## [0.1.0a1] - 2026-11-20

First public pre-release: milestones M0 to M10b (v0.1 as built; Gate 2 review pending). Takes the
`fleetgym` name on PyPI.

### Changed
- VRP-SDO (`VRPSD-Outsource-v0`) is deferred after v0.1 by the user: its decision is a selection on
  top of routing. Gate 2 is the review of v0.1 (`docs/gates/gate2.md`).

### Added
- M10b documentation (ADR 0020): Sphinx site (`docs/site`, extra `docs`, Read the Docs config) with
  concepts, an environment catalog, guides (notably *Build your own VRP variant*) and API reference;
  `examples/` with eight runnable examples, all run by the tests; `fleetgym.register_constraint`.

### Added
- M10 visualization (ADR 0019): map and vehicle Gantt views from the event log (`fleetgym.viz`),
  `rgb_array` and `human` render modes (with the `viz` extra), and `fleetgym render`.

### Fixed
- CI default test jobs measure coverage of the core only; they failed the coverage gate after M9
  because modules that need extras were counted as uncovered.

### Added
- M9 training stack (ADR 0018): `fleetgym.nn` (graph batch adapter, `Encoder`/`Decoder` base classes,
  registry, `MLPEncoder`, dense `GATEncoder`, `AttentionModelEncoder`, `PointerDecoder`,
  `MaskableGraphPolicy`), `fleetgym.testing.check_module`, `fleetgym.agents` (`MaskableDQN`,
  `make_sb3_vec`, `sb3_defaults`, `SB3Policy`, `ScenarioEval`, `CostComponentLogger`), and the
  `fleetgym train` and `fleetgym eval` commands. CI type-checks and tests the optional extras.

### Added
- M8 baselines and evaluation (ADR 0017): `Policy` protocol and `run_episode`; random, nearest
  neighbor, cheapest insertion, a priori route, rollout over forks, PyVRP re-optimization (extra
  `solvers`), perfect-information benchmark (exact bound or heuristic); `fleetgym.eval` with fixed
  scenario seeds, validated evaluation, CI, CVaR, paired tests and JSON report cards;
  `check_policy_non_anticipative`; `FleetGymEnv.epoch`.

### Added
- M7 multi-agent (ADR 0016): `FleetGymAECEnv` and `FleetGymParallelEnv` (one agent per vehicle),
  conflict rules (priority, distance, lottery, reject), global, individual and mixed rewards,
  `state()` for a centralized critic, per-vehicle cost attribution, `Simulator.apply_many` and
  `ready_epochs`, `fleetgym.make_multi`, `REGISTRY.ids(agents=...)`; configs `MA-CVRP-v0` and
  `MA-DVRP-v0`.

### Added
- M6b dynamic requests with `next_node` (ADR 0015): requests accepted automatically as they arrive;
  a wait action (`Discrete(max_nodes + 1)`) that resumes at the next event or at the vehicle's
  wake-up time; mid-day depot reloads; terminating a route at the depot while another vehicle works.

### Added
- M6 dynamic requests:
  - arrival processes (`poisson`, `nhpp` by thinning), request classes (KDE or uniform locations,
    demand, time-window width, service time), cutoff, cancellations, overflow handling, and
    `initial_requests`; degree of dynamism and effective degree of dynamism;
  - planned routing: `PlannedRoutingSimulator`, `routing: cheapest_insertion`, the `accept_reject`
    and `assign` interfaces, outsourcing of requests, drive-first vehicles that wait and head home in
    time;
  - scenarios with request sources (`episode_instance`), forks that redraw the future of the day;
  - validator support for releases, decisions, cancellations, overflow and assignments;
  - `DVRP-StochReq-v0` is experimental; ADR 0014.
- The documentation site and an `examples/` folder are planned as milestone M10b.

### Changed
- Node features grow to 10 (under-decision flag); unreleased requests are hidden. `Epoch.vehicle`
  can be `None`; interfaces get `max_vehicles` and explain invalid actions.
- `fleetgym.make` accepts a spec as a dict. Scenario ids hash raw float64 bytes.

### Added
- M5 uncertainty and recourse:
  - samplers for every spec distribution (`fleetgym.uncertainty.distributions`) with mean, quantile
    and finite support; `register_distribution` for custom ones;
  - `Scenario` (per-source realized values, Parquet I/O, content id), sampling from per-source
    streams, replay through `reset(options={"scenario": ...})`, mismatch errors;
  - revelation of demand and service times (`a_priori`, `on_request`, `on_departure`,
    `on_arrival`, `on_completion`) with forecasts for unrevealed values;
  - recourse rules `return_to_depot`, `outsource` and `agent_decides` (recourse epochs), custom
    rules via `register_recourse`; restock trips; outsourcing costs;
  - `FleetGymEnv.fork(seed)` (non-anticipative copy) and `env.scenario`;
  - `VRPSD-v0` (experimental); `AprioriRoute` policy; `fleetgym.testing.exact` (scenario
    enumeration, exact expected cost); `fleetgym.testing.non_anticipativity`;
  - validator support for scenarios, partial service, restocks, outsourcing and rule behavior;
  - `Registry.unregister`.
- ADRs 0012 (uncertainty, scenarios, non-anticipativity) and 0013 (recourse semantics).

### Changed
- Observations: node features grow from 8 to 9 (demand-known flag). `info` gains `epoch_kind`.
- Event log schema version 2: `reveal`, `failure`, `restock` and `outsource` rows.
- `presence` is a reserved uncertainty source; the `split` recourse rule is reserved.
- Recourse rules have parameters; `VRPSD-Outsource-v0` (planned) states its provisional outsourcing
  cost, which changes its hash.
- The validator requires a `service_start` before a `service_end`.

## [0.1.0.dev0] - 2026-11-20

First pre-release (milestones M0 to M4, Gate 1 approved). Takes the `fleetgym` name on PyPI.

### Added
- Solomon replication: checksummed SINTEF TOP manifest for the 56 instances with 100 customers
  (`solomon/100`), archive-member downloads, `fleetgym replicate VRPTW-v0`, Gate 1 fidelity tests;
  ADR 0011. Collections carry a cost `tolerance`.
- Configs are validated against the shipped JSON Schema in tests (`jsonschema` in `[dev]`).
- Project metadata for `github.com/moda707/fleetgym`: project URLs, README, `CONTRIBUTING.md`,
  issue and pull-request templates, Dependabot for GitHub Actions, and a PyPI release workflow
  using trusted publishing.

### Changed
- `CVRP-v0` and `VRPTW-v0` are `stable` (both pass the replication suite). Hashes unchanged.
- Gate 1 approved.

### Added (M4)
- M4 validator and Gate 1: independent `fleetgym.testing.validator`; `fleetgym.testing.check_env`;
  replication (`replay_solution`, `fleetgym replicate`); checksummed CVRPLIB downloads
  (`fleetgym.data.download`, manifest for all 100 X instances); Gate 1 suite (mask exactness against an
  independent oracle, 10,000 validated rollouts, brute-force golden optima, CVRPLIB X fidelity).
- ADR 0010; Gate 1 report in `docs/gates/gate1.md`.

### Changed
- Lexicographic weights moved to `ObjectiveSpec.scalar_weights()` (spec layer).
- `EventLog.copy()` now copies rows, so copies share no mutable state.
- M3 simulator core: discrete-event engine (sequential and on-event epochs), pure-data state with exact
  snapshots, primitive decisions, capacity / time-window (hard, soft, waiting rules) / route-duration
  constraints with depot look-ahead, fleet sizing, invariant monitor; `InformationState` view;
  `next_node` interface with repair; padded observations; reward model with lexicographic
  scalarization; `FleetGymEnv` with three invalid-action modes, seeding, `get_state`/`set_state`/
  `clone`, `ansi` rendering; `fleetgym.make`, `make_vec`, Gymnasium registration (`fleetgym/<ID>`);
  event log with Parquet I/O; per-source random streams; Hypothesis strategies in `fleetgym.testing`.
- ADR 0009 (simulator semantics, rewards, environment conventions).
- M2 network and instances: immutable `Instance` with content hash; VRPLIB/CVRPLIB and Solomon readers
  (through `vrplib`); `Solution` reader; `uniform` and `clustered` generators with reachable time
  windows; `EuclideanNetwork` (`none`, `nearest_int` = `floor(d + 0.5)`) and `MatrixNetwork`;
  `bind(spec, instance)` resolving `from_instance` with documented default keys and compatibility
  checks; `instance_from_source`. `CVRP-v0` and `VRPTW-v0` gained default generator sources.
- ADR 0008 (instances, networks and binding).
- M1 spec and configs: frozen pydantic `ProblemSpec` with a stable content hash (`fg1:` + SHA-256,
  `meta` excluded); flat YAML forms for distributions, uncertain parameters, networks, arrival
  processes, recourse rules and objectives; `from_instance` references; reserved features and
  `meta.status` (stable, experimental, planned).
- Config loader: `extends` by ID or path, merge rules (deep maps, by-name constraints, `null` deletes,
  discriminator change replaces, `objective` replaced, `meta`/`id` not inherited), dotted overrides,
  duplicate-key detection, YAML 1.2 floats, one error report naming file and key path per problem.
- Config registry scanning `fleetgym/configs` and `fleetgym.configs` entry points; plugin registries
  with `fleetgym.plugins` entry points.
- Configs: `CVRP-v0`, `VRPTW-v0` (experimental); `DVRP-StochReq-v0`, `VRPSD-Outsource-v0`,
  `EVRP-NL-v0` (planned).
- JSON Schema export (`fleetgym schema`), shipped next to the configs for editor support.
- CLI: `fleetgym list`, `show`, `validate`, `schema`.
- ADRs 0006 (loading and merge rules) and 0007 (spec schema decisions).
- M0 scaffolding: `fleetgym` package skeleton (src layout, layers L0 to L7 with module docstrings),
  `pyproject.toml` (hatchling, extras `sb3`, `gnn`, `solvers`, `viz`, `all`, `dev`), `uv.lock`,
  pre-commit hooks, GitHub Actions CI (lint, type check, tests on Linux, macOS and Windows with
  Python 3.12 to 3.14, weekly slow suite), MIT license, `CITATION.cff`.
- `fleetgym.errors` exception hierarchy; `ConfigError` messages name the file and key path.
- `fleetgym --version` command.
- Static import-layer test enforcing architecture rules 1 and 11 (ADR 0004).
- Design, implementation plan and ADRs 0001 to 0005.
