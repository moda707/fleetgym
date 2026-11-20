# Status

**Current milestone:** M4 Validator and Gate 1 done. **Stopped at Gate 1 for user approval** (see
`docs/gates/gate1.md`). Next after approval: M5 (recourse interface design to discuss first).

| Milestone | State | Notes |
| --- | --- | --- |
| Plan | Done | Reviewed by the user on 2026-11-20; resolved questions listed in plan §8 |
| M0 Scaffolding | Done | Package skeleton, tooling, CI, layer test |
| M1 Spec and configs | Done | Models, loader, merge rules, overrides, registry, schema, CLI |
| M2 Network and instances | Done | Instance, readers, generators, networks, binding |
| M3 Simulator core | Done | Event engine, constraints, next_node, Gymnasium env, event log |
| M4 Validator and Gate 1 | Done, awaiting approval | Solomon fidelity pending user input (Q18/Q19) |
| M5 to M11 | Not started | M5 starts with the recourse design discussion; stops at Gate 2 (M11) |

## Tests (M4)
- 519 tests pass in the default run (1 skipped: Solomon fidelity), coverage 97%; ruff and pyright
  (standard) clean.
- Gate suite (`-m "slow or network"`): 104 tests pass, including 10,000 validated rollouts and all 100
  CVRPLIB X best-known solutions reproduced exactly.
- Gymnasium `check_env` passes for every registered config (with `invalid_action="repair"`).
- Determinism (also across processes), exact snapshots, independent clones.

## Registered configs
| ID | Status | Notes |
| --- | --- | --- |
| CVRP-v0 | experimental | Reproduces CVRPLIB X exactly; can become `stable` once Gate 1 is approved |
| VRPTW-v0 | experimental | lexicographic objective (scalarization decided in M3, Q15) |
| DVRP-StochReq-v0 | planned | until M6 |
| VRPSD-Outsource-v0 | planned | until Gate 2 (M11) |
| EVRP-NL-v0 | planned | EV module out of scope |

## Known issues
- `docs/DESIGN.md` was transcribed from the attached design document. Its two embedded diagrams
  (architecture, §26 roadmap) are missing.
- The JSON Schema is checked for consistency with the models but not run as a validator over the
  configs: that needs the `jsonschema` package, which is not an approved dependency (ask the user).
- Git commits use a repo-local identity (user's name and account email). Change it with
  `git config user.email ...` and `git commit --amend --reset-author` before pushing if needed.
- No remote repository yet; project URLs are empty until the GitHub home is chosen (plan Q4).
