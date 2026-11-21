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

## Further decisions (user, 2026-11-20)
- **Whole distributions in the instance**: the model says what is uncertain and when it is
  revealed (`uncertainty.demand: {revealed: on_arrival}`); the instance's `distributions` block
  gives the family and parameters, which may name the instance's fields. A spec may still give a
  distribution for quick experiments.
- **Presets and data**: a paper's example notebook includes a function that generates instances in
  the format the problem config expects; it writes the paper's instance set once (seed recorded in
  the manifest). The preset states the schema and fields it needs and takes the instance folder.
  Benchmarks (Solomon, CVRPLIB) are downloaded with checksums. FleetGym ships no generated data.
- **Dynamic requests**: the arrival process and rates, the request attribute distributions and the
  cutoff move to the instance too (step d).
- **Reload time** (`fleet.reload_time`) is paid before the vehicle leaves the depot again; a vehicle
  that ends its route after returning pays nothing (found by property tests: paying on arrival
  made routes end past a hard duration limit).

## Alternatives
A composable generator in the spec (rejected by the user: generation is the user's task); a Parquet
format for instance sets (not needed while sets are user-provided; can be added later).

## Consequences
Every config, the binding layer, the readers, the validator, the examples and the docs change
(milestone M13b, steps a to d). Spec hashes change; the golden snapshots are rebuilt.
