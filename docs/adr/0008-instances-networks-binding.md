# 0008: Instances, networks and binding

Status: accepted (2026-11-20)

## Context
M2 connects specs to data. Several conventions affect costs directly and must be explicit.

## Decisions
- **Distances are computed by FleetGym, not taken from `vrplib`.** `vrplib` returns unrounded
  `EUC_2D` distances. `network.rounding: nearest_int` applies `floor(d + 0.5)` per arc (ties round
  up), which is not Python's `round` (round half to even). `none` keeps double precision. Further
  conventions (for example truncation used in some VRPTW literature) are added only once the user
  confirms which published solutions to match (plan Q18).
- **Explicit matrices** (`EDGE_WEIGHT_TYPE: EXPLICIT`) are kept as the instance's distance matrix and
  used by `network: {type: matrix}`.
- **Depots are the leading nodes.** Readers reject files whose depots are not nodes `1..D` (in file
  numbering); v0.1 binds only single-depot instances.
- **Scalar `SERVICE_TIME`** in a VRPLIB file applies to every customer and is 0 at the depot.
- **Default instance keys** for a bare `from_instance` (`fleetgym.spec.binding.DEFAULT_INSTANCE_KEYS`):
  vehicle `capacity` → `capacity`, `count` → `n_vehicles`, `uncertainty.demand.*` → `demands`,
  `uncertainty.service_time.*` → `service_times`, `route_duration.max` → `max_route_duration`,
  `outsourcing.value` → `outsourcing_costs`, request-class `location.data` → `customer_coords`.
  Any other field needs the explicit `{from_instance: <key>}` form; there is no guessing.
  `route_duration.max: depot_due_time` reads the closing time of the depot's window.
- **Binding checks compatibility**: a Euclidean network needs coordinates, a matrix network needs a
  matrix, the capacity constraint needs every vehicle type's capacity, time windows need instance
  windows. Planned specs cannot be bound (`NotInScopeError`).
- **Generators** (`uniform`, `clustered`): square `[0, size]^2`, depot at node 0, integer demands,
  and optional time windows that guarantee reachability at speed 1. Their parameters are validated
  when the config loads. The parameter values in the shipped configs are defaults for quick
  experiments, not benchmark settings.
- **Downloads** of benchmark collections move to M4, where the replication suite needs them.

## Consequences
Instance hashes ignore `name` and `source`, so the same data read from two files has one hash.
