# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package uses
[Semantic Versioning](https://semver.org/). Environment IDs are versioned separately (`-v0`, `-v1`).

## [Unreleased]

### Added
- M0 scaffolding: `fleetgym` package skeleton (src layout, layers L0 to L7 with module docstrings),
  `pyproject.toml` (hatchling, extras `sb3`, `gnn`, `solvers`, `viz`, `all`, `dev`), `uv.lock`,
  pre-commit hooks, GitHub Actions CI (lint, type check, tests on Linux, macOS and Windows with
  Python 3.12 to 3.14, weekly slow suite), MIT license, `CITATION.cff`.
- `fleetgym.errors` exception hierarchy; `ConfigError` messages name the file and key path.
- `fleetgym --version` command.
- Static import-layer test enforcing architecture rules 1 and 11 (ADR 0004).
- Design, implementation plan and ADRs 0001 to 0005.
