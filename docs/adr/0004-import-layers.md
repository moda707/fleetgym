# 0004: Import layers, enforced by a static test

Status: accepted (2026-11-20)

## Context
Architecture rule 1 (layers depend only downward) and rule 11 (independent validator) are easy to
break silently. An import-linter tool would be a new dependency.

## Decision
- Layers: L0 `errors, plugins, _hashing, _typing, _version, eventlog`; L1 `data`; L2 `spec, configs`;
  L3 `network, uncertainty`; L4 `sim, constraints, ev`; L5 `interfaces, envs`;
  L6 `baselines, agents, nn`; L7 `eval, viz, testing, cli`.
- Imports between different packages of the same layer are forbidden unless listed in
  `tests/architecture/layers.py:SAME_LAYER_ALLOWED` (for example `sim → constraints`,
  `constraints → sim.state`, `envs → interfaces`, `agents → nn`).
- The event log schema and I/O are a separate L0 package, `fleetgym.eventlog`, so the validator can read
  logs without importing the simulator.
- `fleetgym.testing.validator` may import only L0 to L2 and `fleetgym.uncertainty.scenario`.
- Layers L0 to L5 may not import torch, SB3, PyG, matplotlib or PyVRP, even lazily.
- The root `fleetgym/__init__.py` is the public facade; no library module imports it. The version lives
  in `fleetgym._version`.
- `tests/architecture/test_layers.py` checks all of this with `ast` (no imports executed), including
  `TYPE_CHECKING` blocks.

## Alternatives
- `import-linter`: declarative and mature, but a new dependency.
- Convention only: violations would be found late, if at all.

## Consequences
Adding a package means assigning it a layer in `layers.py` (a test fails otherwise). Relaxing a rule is
a one-line change that shows up clearly in review.
