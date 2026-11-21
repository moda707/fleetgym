# FleetGym: conventions for working in this repo

FleetGym (`fleetgym`, import alias `fg`) is a Gymnasium/PettingZoo library for RL on vehicle routing,
focused on stochastic and dynamic VRPs. The design doc calls it "RouteGym"; same library (ADR 0001).

## Sources of truth
- `docs/DESIGN.md`: architecture, naming, scope. Read the relevant sections before changing a module.
- `docs/IMPLEMENTATION_PLAN.md`: milestones M0 to M11, public interfaces, test plans, open questions (Q#).
- `docs/adr/`: decisions already made. Do not re-decide something an ADR settles; write a new ADR to change it.
- `STATUS.md`: where the build is. Read it first in every session.
- Build request rule: the build request wins on scope and process, `DESIGN.md` wins on architecture.

## Never trust remembered numbers
References in `DESIGN.md` were partly compiled from memory. Never take numbers, formulas, recourse
conventions or benchmark values from them or from your own memory. Use the actual instance and solution
files, or ask the user.

## Scope
v0.1 (M0 to M11) plus the paper-based extensions added by the user on 2026-11-20 (M12 to M14 in
the plan): VRP-VCSD and VRPSD with duration limits (M12), paper-based example notebooks (M13), and the
EV shortest-path problem with charging-station occupancy, EVSPP-OI (M14; the user brought this part of
the EV module into scope), and stochastic travel and service times (M15, added by the user on
2026-11-21: Taş et al. 2012 and Errico et al.). Not in scope: the rest of the EV module (EV routing
with several customers, `EVRP-NL-v1`), VRP-SDO (deferred by the user on 2026-11-20: its decision is selection on top of
routing), time-dependent travel, GPU backend, C++ core, OSM networks, web replay viewer, leaderboard.
For those, leave only a registry entry type, an abstract base class, or a reserved spec field that
raises `NotInScopeError`. No stub implementations.

## Architecture rules (non-negotiable)
1. Layers import only downward: L0 `errors, plugins, _hashing, _typing, eventlog` → L1 `data` → L2 `spec`
   → L3 `network, uncertainty` → L4 `sim, constraints` → L5 `interfaces, envs` → L6 `baselines, agents, nn`
   → L7 `eval, viz, testing, cli`. Enforced by `tests/architecture/test_layers.py`.
2. `ProblemSpec` is a frozen pydantic model, deeply immutable, with a stable content hash
   (`meta` excluded, defaults included). Registered config hashes are snapshotted in `tests/golden/`.
   Changing an env's dynamics or defaults means bumping its `-vN`.
3. One YAML file per environment, under `src/fleetgym/configs/<family>/<ID>.yaml`; the file stem equals `id`.
   `extends` merge rules: maps deep-merge, constraints merge by name, `null` deletes, a discriminator change
   replaces the subtree, `meta` is never inherited.
4. Every uncertain parameter is `Uncertain(dist, revealed, ...)`; `Fixed(value)` is the deterministic case.
5. RNG: one `SeedSequence` per episode; each uncertainty source gets a stream keyed by a hash of its
   **name** (never by order). Changing one source must not change other sources' draws.
6. Exogenous information is pre-sampled into a `Scenario` (Parquet) and replayed exactly.
7. Discrete-event simulator; the agent acts only at decision epochs. Interfaces translate actions into
   primitive decisions; the simulator never imports an interface.
8. Constraints implement `init_state, mask, check, update, penalty` and see revealed information only.
9. Reward per epoch = negative cost realized since the previous epoch. Components go in `info["costs"]`.
   `sum(rewards) == -validator_cost` is tested.
10. `get_state`, `set_state`, `clone` are exact, including RNG state. `clone` sees the true future; anything
    that plans (rollout, scenario methods) uses the non-anticipative `fork(seed)`.
11. `fleetgym.testing.validator` is independent: it may import only L0 to L2 and
    `uncertainty.scenario`. Never `network`, `sim`, `constraints`, `interfaces` or `envs`.

## Code style
- Python ≥ 3.12 (ADR 0005). Develop on 3.12 so newer syntax is not used by accident.
- Full type hints; `pyright` standard mode must pass; `ruff check` and `ruff format`.
- NumPy-style docstrings on every public object. Each distribution's docstring states its parameterization.
- American English in code, docs and messages.
- No silent fallbacks. Raise a `FleetGymError` subclass; config errors carry the file path and the dotted
  key path (`configs/dynamic/X-v0.yaml: decisions.interface`).
- Correctness first. Vectorize with NumPy where obvious; profile before reaching for numba.
- Dependencies: core = numpy, pydantic v2, pyyaml, gymnasium, pettingzoo, pyarrow, scipy, vrplib. Extras:
  `sb3` (stable-baselines3, sb3-contrib, torch), `gnn` (torch-geometric), `solvers` (pyvrp), `viz` (matplotlib),
  `dev` (pytest, pytest-cov, pytest-xdist, hypothesis, ruff, pyright, pre-commit).
  **Ask the user before adding any other dependency.**
- Ruff excludes Markdown (it would reformat code blocks; `docs/DESIGN.md` is a verbatim copy).
- Unanswered plan questions: adopt the plan's recommendation when its milestone starts and record an ADR.
  Exceptions that need the user: Q18/Q19 (Gate 1 data), Q20 (recourse design, discuss before M5), Gate 2.

## Commands
```bash
uv sync --all-extras              # environment
uv run pytest                     # default suite (excludes slow and network)
uv run pytest -m slow             # gate suites
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run fleetgym validate <config> # CLI (from M1)
```

## Testing discipline
- Write tests with each module, not after. Target 90% line coverage on L0 to L5.
- Required kinds: unit, determinism (same spec and seed give an identical trajectory), Hypothesis property
  tests (random specs and action sequences never crash; masks never allow what the validator rejects),
  golden tests with brute-force optima on tiny instances, KS or chi-square tests for samplers at fixed seeds.
- `check_env` (Gymnasium) and PettingZoo `api_test` must pass for every non-planned registered config.
- Markers: `slow`, `network` (never in default CI), `extras_sb3`, `extras_solvers`, `extras_viz`.

## Workflow
- One milestone at a time, in order. At the end of each: run the full suite, update `STATUS.md` and
  `CHANGELOG.md`, add ADRs, commit with a conventional commit message (`feat(spec): ...`), then give
  the user a short summary.
- **Stop and wait for user approval at Gate 1 (end of M4) and Gate 2 (M11).** Gate 2 is the review of v0.1
  (`docs/gates/gate2.md`); VRP-SDO is deferred until the user brings it back, with its model and data.
- Record each non-obvious design decision as `docs/adr/NNNN-kebab-title.md` (Context, Decision,
  Alternatives, Consequences).
- If the design is ambiguous or seems wrong, ask. Never invent behavior silently. Small naming
  improvements are fine if logged in an ADR.
- Commit only at milestone ends or when the user asks. Never publish to PyPI or push without being asked.
