# Status

**Current milestone:** M5 Uncertainty and recourse done (2026-11-20). Next: M6 Dynamic requests.
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
| M6 to M11 | Not started | Stops at Gate 2 (M11); before M11, ask for the VRP-SDO model and data |

## Tests (M5)
- Default run: 602 passed, coverage 97%; ruff and pyright (standard) clean.
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
| DVRP-StochReq-v0 | planned | until M6 |
| VRPSD-v0 | experimental | Poisson demands revealed on arrival, return-to-depot recourse |
| VRPSD-Outsource-v0 | planned | until Gate 2 (M11); outsourcing cost provisional |
| EVRP-NL-v0 | planned | EV module out of scope |

## Known issues
- `docs/DESIGN.md` was transcribed from the attached design document. Its two embedded diagrams
  (architecture, §26 roadmap) are missing.
- PyPI: `fleetgym` is not registered yet. The name is taken by the first release (see
  `CONTRIBUTING.md`, "Releasing"); the user does the PyPI setup.
- GitHub repository description and topics must be set in the web UI (no `gh` CLI here).
