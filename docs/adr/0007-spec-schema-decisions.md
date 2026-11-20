# 0007: Spec schema decisions for v0.1

Status: accepted (2026-11-20). Adopts the plan's recommendations for Q8, Q9, Q10, Q12, Q24 and Q28
while the user was unavailable; each can be revisited.

## Decisions
1. **YAML is the canonical form (Q8).** Constraints are a map keyed by name. Uncertain parameters are
   flat: `{dist: poisson, mean: from_instance, revealed: on_arrival}`. Recourse is a top-level map
   keyed by failure type (`recourse: {capacity_failure: return_to_depot}`), not
   `decisions.recourse`. `from_instance` is the only instance-reference keyword (the design's Python
   example writes `mean="instance"`).
2. **Plugin parameters live in spec-level registries** (`fleetgym.spec.registries`): distributions,
   constraints, networks, arrival processes, recourse rules, interfaces, routing rules, uncertainty
   sources, failure types and cost components. A config is validated without importing the
   simulator. Plugins register parameter models there.
3. **Reserved features.** Features declared in the design but not in v0.1 (EV section, `energy`
   constraint, time-dependent `network.time_model`, `travel_time` / `station_wait` /
   `energy_consumption` / `breakdown` sources, `correlated_with`, `time_window_failure` /
   `energy_failure`, interfaces other than `next_node`, `accept_reject`, `assign`, `hybrid` epochs,
   `diversion`, and EV/other cost components) validate as data, but only in configs with
   `meta.status: planned`. `make()` will raise `NotInScopeError` for them.
4. **`meta.status`** is `stable` (passes replication), `experimental` (implemented, not replicated) or
   `planned` (validates, not makeable). Planned configs are skipped by environment checks.
5. **Operational instance source (Q9).** A top-level, hashed `instance_source` (`generator` or `files`)
   says where instances come from when none is passed. `meta.instances` documents replication data
   only. Shipped configs get an `instance_source` in M2, when generators exist.
6. **Instance references (Q10).** `from_instance` (the field's default instance key, defined at binding
   in M2) or `{from_instance: <key>}`. `route_duration.max` also accepts `depot_due_time`.
7. **Epochs (Q12).** `decisions.epochs` adds `sequential` (one vehicle completes its route before the next
   starts; static problems only, so it is rejected with `dynamics`). `CVRP-v0` uses it.
8. **Dynamics (Q24).** `dynamics.max_requests` is required; `overflow: error | reject`, default `error`.
9. **Soft constraint penalties are priced by the objective.** The design's Python example has both
   `TimeWindows(late_penalty=2.0)` and `Objective(lateness=2.0)`. Constraints only report violation
   amounts as cost components (`lateness`); `objective` weights price them. One source of truth.
10. **Hash.** `fg1:` + SHA-256 of canonical JSON of the spec without `meta`, defaults included. Equality
    of specs compares fields only (not the file a spec came from).
11. **Multi-agent presets (Q28).** `MA-DVRPTW-v0` is not written in M1: it extends an unwritten
    `DVRPTW-v0` that needs a `plan` interface. Multi-agent presets are added in M7.
12. **Distributions without a settled parameterization are deferred**: `lognormal_multiplier`,
    `negative_binomial` and `mixture` are added when needed, each with an explicit parameterization.

## Consequences
The JSON Schema (`fleetgym/configs/fleetgym.schema.json`) lists every registered name, so editors
autocomplete them. It cannot require fields that a parent may provide, so only discriminators are
required in it; completeness is checked by `fleetgym validate`.
