# Status

**Current milestone:** M0 Scaffolding done. Next: M1 Spec and configs.

| Milestone | State | Notes |
| --- | --- | --- |
| Plan | Done | Reviewed by the user on 2026-11-20; resolved questions listed in plan §8 |
| M0 Scaffolding | Done | Package skeleton, tooling, CI, layer test |
| M1 Spec and configs | Next | |
| M2 to M11 | Not started | Stops at Gate 1 (M4) and Gate 2 (M11) |

## Tests (M0)
- 144 tests passing locally (Python 3.12, macOS): module imports and docstrings, error formatting, CLI
  `--version`, layer checker on the real package and 19 synthetic cases.
- ruff check, ruff format and pyright (standard): clean. Coverage 100% (44 statements).
- CI workflow written but not yet run (no remote).

## Known issues
- `docs/DESIGN.md` was transcribed from the attached design document. Its two embedded diagrams
  (architecture, §26 roadmap) are missing.
- Git commits use a repo-local identity (user's name and account email). Change it with
  `git config user.email ...` and `git commit --amend --reset-author` before pushing if needed.
- No remote repository yet; project URLs are empty until the GitHub home is chosen (plan Q4).
