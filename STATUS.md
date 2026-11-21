# Status

**Current milestone:** M14 and M15 done (EV shortest path, ADR 0026; stochastic travel and service
times, ADR 0025). Next: Gate 2 (`docs/gates/gate2.md`), awaiting the user's review. The multi-agent DVRP example (M13) is deferred by the user (2026-11-20; no suitable paper).
v0.1 (M0 to M10b) is built; Gate 2 (`docs/gates/gate2.md`) still awaits the user's approval.
VRP-SDO is deferred by the user (2026-11-20).
Repository: <https://github.com/moda707/fleetgym>. `fleetgym 0.1.0a1` is on PyPI (published by the
user on 2026-11-20).

| Milestone | State | Notes |
| --- | --- | --- |
| Plan | Done | Reviewed by the user on 2026-11-20; resolved questions listed in plan §8 |
| M0 Scaffolding | Done | Package skeleton, tooling, CI, layer test |
| M1 Spec and configs | Done | Models, loader, merge rules, overrides, registry, schema, CLI |
| M2 Network and instances | Done | Instance, readers, generators, networks, binding |
| M3 Simulator core | Done | Event engine, constraints, next_node, Gymnasium env, event log |
| M4 Validator and Gate 1 | Done, approved | CVRPLIB X (100) and Solomon (56) reproduce published results (ADRs 0010, 0011) |
| M5 Uncertainty and recourse | Done | Exact expectation equals an independent DP to 1e-9 (ADRs 0012, 0013) |
| M6 Dynamic requests | Done | Accept mask equals the independent route oracle; 2,000 random dynamic problems validate (ADR 0014) |
| M6b Dynamic next_node | Done | Wait, reload and terminate as decided with the user (ADR 0015) |
| M7 Multi-agent | Done | PettingZoo AEC and Parallel pass the API tests (ADR 0016) |
| M8 Baselines and evaluation | Done | Oracle bounds every non-anticipative baseline on tiny instances (ADR 0017) |
| M9 Training stack | Done | MaskablePPO learns a tiny instance optimally; MaskableDQN masks verified (ADR 0018) |
| M10 Visualization | Done | Map and Gantt views, rgb_array and human modes, `fleetgym render` (ADR 0019) |
| M10b Documentation and examples | Done | Sphinx site, build-your-own-variant guide, 8 tested examples (ADR 0020) |
| M11 Gate 2 | Awaiting approval | Review of v0.1; VRP-SDO deferred by the user |
| M12 VRP-VCSD | Done | `split` recourse, duration limits with recourse, `served_demand`; `VRP-VCSD-v0`, `VRPSD-Goodson-v0` (ADR 0021) |
| M13 Paper-based examples | Done | 7 paper examples and the build-your-own guide (ADR 0022); DVRP: Ulmer et al. (ADR 0024); MA-DVRP deferred by the user |
| M13b Spec/instance separation | Done | Schemas, JSON format and sets, model primitives, presets, families at -v1, generators removed (ADR 0023) |
| M15 Stochastic travel and service times | Done | Taş et al. (travel times, soft windows) and Errico et al. (service times, hard windows), ADR 0025 |
| M14 EVSPP-OI | Done | EV shortest path with charging-station occupancy (arXiv 2305.11773), ADR 0026 |

## Tests (M6)
- Default run: 723 passed (with the sb3, solvers and viz extras), coverage 96% (core-only 95%);
  ruff and pyright (standard) clean.
- M6: KS and chi-square tests of the arrival processes; accept mask equal to the independent route
  oracle; property tests on random dynamic problems (validator, determinism, non-anticipativity).
- M5: KS and chi-square tests for every sampler; exact expectation (1,024 scenarios, 6 recourse
  configurations) equal to an independent dynamic program to 1e-9; Poisson Monte Carlo within the
  99.9% interval; Hypothesis property tests on random stochastic specs (validator, invariants,
  determinism, non-anticipativity); recourse corruption tests for the validator.
- Gate suite (`-m "slow or network"`): 10,000 validated rollouts, all 100 CVRPLIB X best-known
  solutions reproduced exactly, all 56 Solomon best-known solutions reproduced to the published
  two decimals with the published vehicle counts.
- Gymnasium `check_env` passes for every registered config (with `invalid_action="repair"`).
- Determinism (also across processes), exact snapshots, independent clones.

## Registered configs
| ID | Status | Notes |
| --- | --- | --- |
| CVRP-v1 | stable | Reproduces CVRPLIB X exactly (rounding from the instance) |
| VRPTW-v1 | stable | Reproduces Solomon (SINTEF) best-known solutions |
| SVRP-v1 | experimental | Demand distribution from the instance, detour-to-depot recourse |
| DVRP-v1 | experimental | Arrivals and request distributions from the instance; accept/reject, at release or at vehicle arrivals |
| EVSPP-v1 | experimental | One EV, charging stations with M/M/1/kappa queues and occupancy indicators (own environment) |
| VRPSD-Outsource-v1 | planned | VRP-SDO, deferred (selection on top of routing) |
| EVRP-NL-v1 | planned | EV routing with several customers, out of scope (M14 covers the EV shortest path only) |

Presets: solomon1987-vrptw, florio2023-vrpsd, kool2019-cvrp, dastpak2022-vcsd, dastpak2022-goodson,
vera2019-hetfleet, ulmer2016-vrpssr, tas2012-vrpsttw, errico2018-vrptwst, dastpak2023-evsppoi.

## Known issues
- `docs/DESIGN.md` was transcribed from the attached design document. Its two embedded diagrams
  (architecture, §26 roadmap) are missing.
- GitHub repository description and topics must be set in the web UI (no `gh` CLI here).
