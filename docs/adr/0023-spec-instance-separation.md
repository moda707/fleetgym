# 0023: Problem specs declare structure, instances carry numbers

Status: accepted (2026-11-20). Design: `docs/proposals/0001-spec-instance-separation.md`.

## Context
Configs mixed a model with one paper's numbers, paper grids lived in notebooks, behaviors were
bundled under paper-shaped names (`split`), and each paper needed its own instance generator.

## Decision (user, 2026-11-20)
- **Three layers**: generic model families (structure and behavior), instances (every number that
  differs between instances, including distribution parameters), and paper presets (a model, instance
  sets and the paper's grid).
- **No instance generators** in the library, and no hook producing instances: training and test
  data are the user's task. `GeneratorSource`, the generators and `register_generator` are removed.
- **Instance schema**: typed fields at the global, node, arc and vehicle-type levels; a core schema
  `fleetgym/core/1`; user schemas extend it in YAML. Specs, custom rules and the validator declare
  the fields they read, and binding checks them when a spec is loaded.
- **Formats**: one JSON file per instance; a set is a folder or glob of files with a manifest.
  CVRPLIB and Solomon readers convert to the format.
- **Model primitives**: orthogonal switches (`fleet.multi_trip`, `fleet.reload`, `fleet.end_route`,
  constraint `mode: hard|soft`, recourse `{serve, remainder}`) replace bundled rules.
- **IDs**: the `-v0` configs are removed; the new families start at `-v1`.

## Alternatives
A composable generator in the spec (rejected by the user: generation is the user's task); a Parquet
format for instance sets (not needed while sets are user-provided; can be added later).

## Consequences
Every config, the binding layer, the readers, the validator, the examples and the docs change
(milestone M13b, steps a to d). Spec hashes change; the golden snapshots are rebuilt.
