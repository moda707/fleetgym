# Models, instances and presets

FleetGym separates three things (ADR 0023):

| Layer | Says | Example | Where |
| --- | --- | --- | --- |
| **Model** | which features exist and how they behave | "routes have a hard duration limit", "demand is revealed on arrival" | a spec: a registered family (`CVRP-v1`, `VRPTW-v1`, `SVRP-v1`, `DVRP-v1`) or your YAML |
| **Instance** | every number of one problem | the limit is 221.47; 3 vehicles of capacity 50; these customers; their demand distribution | a file in the FleetGym format, or a benchmark file |
| **Preset** | a published setting: a model, the paper's grid, where its instances come from | Dastpak et al.'s 15 VRP-VCSD instances | `fleetgym/presets/*.yaml` |

**FleetGym ships no instances and no instance generators.** Producing instances (and training data)
is your task; the examples show how, each with a function that writes the instances its model
expects. Benchmarks (CVRPLIB, Solomon) are downloaded and verified against checksums.

## Instance schemas

An instance follows a schema: typed fields at four levels.

| Level | Shape | Core fields (`fleetgym/core/1`) |
| --- | --- | --- |
| `global` | one value | `max_route_duration`, `shift_start`, `shift_end`, `reload_time`, `cutoff`, `edge_weight_type` |
| `node` | one value per node (depots first) | `kind`, `x`, `y`, `demand`, `service_time`, `tw_open`, `tw_close`, `deadline` |
| `arc` | one value per pair of nodes | `distance`, `travel_time` |
| `vehicle_type` | one row per type | `name`, `capacity`, `count`, `fixed_cost` |

A variant that needs other numbers extends the schema in YAML; every field has a type, bounds and a
unit, and is checked whenever an instance is built or read:

```{literalinclude} ../../../examples/05_custom_variant/rural-parcels.schema.yaml
:language: yaml
```

Load it with `fleetgym.data.load_schema(path)`, or let an instance file point to it
(`"schema_file"`). Custom constraints and recourse rules read the fields by name.

## The instance format

One JSON document per instance (`fleetgym.data.write_instance`, `read_instance`):

```json
{
  "format": "fleetgym-instance/1",
  "schema": "fleetgym/core/1",
  "name": "day-1",
  "global": {"max_route_duration": 221.47},
  "vehicle_types": [{"capacity": 50, "count": 3}],
  "nodes": {"kind": ["depot", "customer", "customer"], "x": [50, 13.2, 80.1], "y": [50, 7.9, 61.4],
            "demand": [0, 10, 5]},
  "distributions": {"demand": {"dist": "discrete_uniform", "center": "demand", "half_width": 5, "floor": 1}}
}
```

- `distributions` gives the distribution of each uncertain value (and, for dynamic requests, the
  arrival process `arrivals` and the request attributes `request.location`, `request.demand`, ...).
  A parameter that is a field name (`"center": "demand"`) refers to that field of the instance.
- `meta` holds provenance (free-form, not hashed).
- CVRPLIB and Solomon files are read directly and convert to the format with the same content hash.

## Instance sets

Many instances (a paper's test set, your training data) form a set: a folder of instance files and
a `manifest.json` with their SHA-256 checksums, the schema, the largest node count and fleet, and how
the files were produced. `write_instance_set` writes one; `read_instance_set` verifies every
checksum. A spec's `instance_source` may list files or set folders; each episode draws one instance.

```python
from fleetgym.data import write_instance_set
import fleetgym as fg

data = write_instance_set(my_instances, "instances/train", name="train", produced_by="seed 0")
env = fg.make({"extends": "CVRP-v1", "instance_source": {"type": "files", "paths": ["instances/train"]}})
```

## What the model says and what the instance gives

| Model (spec) | Instance |
| --- | --- |
| `fleet: {sizing: fixed}`, `fleet.types: from_instance` (default) | `vehicle_types` (capacities, counts) |
| `route_duration: {mode: hard}` or `{mode: soft}` | `max_route_duration` |
| `time_windows: {mode: hard}` | `tw_open`, `tw_close` |
| `uncertainty.demand: {revealed: on_arrival}` | `distributions.demand` |
| `dynamics: {max_requests: 250}` | `distributions.arrivals`, `distributions.request.*`, `cutoff` |
| `fleet.reload_time: from_instance` | `reload_time` |
| `network.rounding: from_instance` | `edge_weight_type` (CVRPLIB's `EUC_2D` rounds) |

Objective weights stay in the model. A spec may still give numbers directly for quick experiments.

## Presets

A preset reproduces a published problem setting. It cites the paper, lists its grid and says where
each grid point's instances come from:

```python
import fleetgym as fg

fg.preset_names()      # dastpak2022-vcsd, kool2019-cvrp, solomon1987-vrptw, ...
env = fg.make_preset("solomon1987-vrptw", instance="r101")                 # downloaded
env = fg.make_preset("dastpak2022-vcsd", instances="instances", density="high", capacity=25)
```

Presets whose instances are generated take the folder their example wrote (`instances=`); the
examples `09` to `14` write them. Every grid point resolves to one frozen spec and one instance set,
both hashed, so a reported result names exactly what was run.
