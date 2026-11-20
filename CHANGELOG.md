# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package uses
[Semantic Versioning](https://semver.org/). Environment IDs are versioned separately (`-v0`, `-v1`).

## [Unreleased]

### Added
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
