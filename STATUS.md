# Status

**Current milestone:** M2 Network and instances done. Next: M3 Simulator core.

| Milestone | State | Notes |
| --- | --- | --- |
| Plan | Done | Reviewed by the user on 2026-11-20; resolved questions listed in plan §8 |
| M0 Scaffolding | Done | Package skeleton, tooling, CI, layer test |
| M1 Spec and configs | Done | Models, loader, merge rules, overrides, registry, schema, CLI |
| M2 Network and instances | Done | Instance, readers, generators, networks, binding |
| M3 Simulator core | Next | |
| M4 to M11 | Not started | Stops at Gate 1 (M4) and Gate 2 (M11) |

## Tests (M2)
- 406 tests passing (Python 3.12, macOS), including Hypothesis properties for spec round trips and
  generator time-window reachability.
- Coverage 98% overall. ruff and pyright (standard) clean.
- Golden spec hashes updated once in M2 (configs gained `instance_source`); IDs stay `-v0`.

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
