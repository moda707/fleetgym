# Status

**Current milestone:** M3 Simulator core done. Next: M4 Validator and Gate 1 (stop for approval).

| Milestone | State | Notes |
| --- | --- | --- |
| Plan | Done | Reviewed by the user on 2026-11-20; resolved questions listed in plan §8 |
| M0 Scaffolding | Done | Package skeleton, tooling, CI, layer test |
| M1 Spec and configs | Done | Models, loader, merge rules, overrides, registry, schema, CLI |
| M2 Network and instances | Done | Instance, readers, generators, networks, binding |
| M3 Simulator core | Done | Event engine, constraints, next_node, Gymnasium env, event log |
| M4 Validator and Gate 1 | Next | Stops for approval; needs Q18/Q19 from the user |
| M5 to M11 | Not started | Stops at Gate 2 (M11) |

## Tests (M3)
- 479 tests passing (Python 3.12, macOS), coverage 97%, ruff and pyright (standard) clean.
- Gymnasium `check_env` passes for every registered config (with `invalid_action="repair"`).
- Hypothesis: random static problems (all epoch, fleet and constraint combinations) with random
  masked actions never crash, keep invariants, and satisfy return = -weighted total cost; a one-off
  stress run of 3,000 examples also checked exact clones.
- Determinism: identical event-log digests for the same seed, also across processes.
- Snapshot/restore replays bit for bit; clones are independent.

## Registered configs
| ID | Status | Notes |
| --- | --- | --- |
| CVRP-v0 | experimental | `epochs: sequential` (ADR 0007) |
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
