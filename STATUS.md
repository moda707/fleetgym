# Status

**Current milestone:** M1 Spec and configs done. Next: M2 Network and instances.

| Milestone | State | Notes |
| --- | --- | --- |
| Plan | Done | Reviewed by the user on 2026-11-20; resolved questions listed in plan §8 |
| M0 Scaffolding | Done | Package skeleton, tooling, CI, layer test |
| M1 Spec and configs | Done | Models, loader, merge rules, overrides, registry, schema, CLI |
| M2 Network and instances | Next | |
| M3 to M11 | Not started | Stops at Gate 1 (M4) and Gate 2 (M11) |

## Tests (M1)
- 342 tests passing (Python 3.12, macOS), including Hypothesis round-trip properties.
- Coverage 98% overall (spec package 95% to 100%). ruff and pyright (standard) clean.
- Golden spec hashes in `tests/golden/spec_hashes.json` for the 5 registered configs.

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
- Configs have no `instance_source` yet (added in M2 with the generators), so the golden hashes will
  change once in M2. No env has been released, so the IDs stay at `-v0`.
- The JSON Schema is checked for consistency with the models but not run as a validator over the
  configs: that needs the `jsonschema` package, which is not an approved dependency (ask the user).
- Git commits use a repo-local identity (user's name and account email). Change it with
  `git config user.email ...` and `git commit --amend --reset-author` before pushing if needed.
- No remote repository yet; project URLs are empty until the GitHub home is chosen (plan Q4).
