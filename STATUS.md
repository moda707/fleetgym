# Status

**Current milestone:** Gate 1 approved by the user (2026-11-20). Next: M5 (uncertainty, scenarios,
recourse: standard rules plus user-written custom rules; later improvements listed in
`docs/recourse.md`). Repository: <https://github.com/moda707/fleetgym>.

| Milestone | State | Notes |
| --- | --- | --- |
| Plan | Done | Reviewed by the user on 2026-11-20; resolved questions listed in plan §8 |
| M0 Scaffolding | Done | Package skeleton, tooling, CI, layer test |
| M1 Spec and configs | Done | Models, loader, merge rules, overrides, registry, schema, CLI |
| M2 Network and instances | Done | Instance, readers, generators, networks, binding |
| M3 Simulator core | Done | Event engine, constraints, next_node, Gymnasium env, event log |
| M4 Validator and Gate 1 | Done, approved | CVRPLIB X (100) and Solomon (56) reproduce published results (ADRs 0010, 0011) |
| M5 to M11 | Not started | Stops at Gate 2 (M11); before M11, ask for the VRP-SDO model and data |

## Tests (M4)
- Default run: see the latest numbers in the commit message of the last milestone; ruff and pyright
  (standard) clean; coverage above 90%.
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
| VRPSD-Outsource-v0 | planned | until Gate 2 (M11) |
| EVRP-NL-v0 | planned | EV module out of scope |

## Known issues
- `docs/DESIGN.md` was transcribed from the attached design document. Its two embedded diagrams
  (architecture, §26 roadmap) are missing.
- PyPI: `fleetgym` is not registered yet. The name is taken by the first release (see
  `CONTRIBUTING.md`, "Releasing"); the user does the PyPI setup.
- GitHub repository description and topics must be set in the web UI (no `gh` CLI here).
