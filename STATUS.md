# Status

**Current milestone:** M8 Baselines and evaluation done (2026-11-20). Next: M9 Training stack.
Repository: <https://github.com/moda707/fleetgym>. Release `v0.1.0.dev0` tagged on `main` (the
GitHub release and PyPI upload are done by the user).

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
| M9 to M10 | Not started | |
| M10b Documentation and examples | Not started | Added at the user's request; needs dependency approval |
| M11 Gate 2 | Not started | Before M11, ask for the VRP-SDO model and data |

## Tests (M6)
- Default run: 694 passed, coverage 96%; ruff and pyright (standard) clean.
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
| CVRP-v0 | stable | Reproduces CVRPLIB X exactly |
| VRPTW-v0 | stable | Reproduces Solomon (SINTEF) best-known solutions; lexicographic objective (Q15) |
| DVRP-StochReq-v0 | experimental | NHPP requests, accept_reject with cheapest insertion; illustrative parameters |
| MA-CVRP-v0 | experimental | Multi-agent CVRP (Parallel, priority conflicts) |
| MA-DVRP-v0 | experimental | Multi-agent dynamic VRP with wait (Parallel) |
| VRPSD-v0 | experimental | Poisson demands revealed on arrival, return-to-depot recourse |
| VRPSD-Outsource-v0 | planned | until Gate 2 (M11); outsourcing cost provisional |
| EVRP-NL-v0 | planned | EV module out of scope |

## Known issues
- `docs/DESIGN.md` was transcribed from the attached design document. Its two embedded diagrams
  (architecture, §26 roadmap) are missing.
- PyPI: `fleetgym` is not registered yet. The name is taken by the first release (see
  `CONTRIBUTING.md`, "Releasing"); the user does the PyPI setup.
- GitHub repository description and topics must be set in the web UI (no `gh` CLI here).
