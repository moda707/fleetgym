# Proposal 0001: problem specs declare structure, instances carry numbers

Status: **accepted** with the user's answers below (2026-11-20); decisions in ADR 0023.

## Why

Paper-specific numbers ended up in the configs (`VRP-VCSD-v0` holds 15 zone indices, 221.47 and
Q = 50), and paper grids ended up in notebook code. Behaviors are bundled under paper-shaped names
(`split` means four things at once), and each paper so far has needed a bespoke generator. The user's
direction (2026-11-20):

1. **No instance generators in the library.** FleetGym publishes an instance schema; people provide
   instances in that format.
2. **The problem spec declares structure**: what exists (a duration limit), how it behaves (hard or
   soft, revealed when, recourse). **The instance carries the values** (the limit itself).
3. **Users can define the schema of their own instances**, as easily as they define a variant.
4. Orthogonal primitives in the spec (proposal A) and paper presets with their grids (proposal C).

## 1. Three layers

| Layer | Says | Example | Lives in |
| --- | --- | --- | --- |
| **Model** (problem spec) | which features exist and how they behave | "routes have a hard duration limit"; "demand is revealed on arrival"; "a depot visit reloads" | `configs/<family>/*.yaml`, few and generic |
| **Instance** | the numbers of one problem | the limit is 221.47; 3 vehicles of capacity 50; these 23 customers | instance files in the FleetGym format |
| **Preset** | a published setting: a model plus instance sets, and its grid | Dastpak et al.'s 15 VRP-VCSD instances | `configs/papers/*.yaml` |

The rule: **a model never holds a number that differs between instances of the same paper.**
Numbers that define the model itself may stay in the spec, such as objective weights, a penalty
per unit of lateness, or the shape of a distribution. Every numeric field may still be given
literally (`max: 120`) for quick experiments; the registered families do not do that.

## 2. The instance format

### 2.1 Schema

A schema lists typed fields at four levels:

| Level | Shape | Core fields (`fleetgym/core/1`) |
| --- | --- | --- |
| `global` | scalar | `name`; optional `max_route_duration`, `shift_start`, `shift_end`, `reload_time` |
| `node` | one value per node | `x`, `y` (or none with a matrix), `kind` (`depot`/`customer`), `demand`; optional `service_time`, `tw_open`, `tw_close`, `deadline` |
| `arc` | node x node matrix | optional `distance`, `travel_time` |
| `vehicle_type` | one row per type | `capacity`, `count`; optional `fixed_cost`, `max_route_duration` |

Each field declares `type` (`float`, `int`, `bool`, `str`), `required`, bounds (`min`, `max`),
`unit` (`time`, `distance`, `load`, `money`, none) and a description. Schemas have an ID and a
version, are content-hashed, and the instance hash covers the schema ID.

### 2.2 User-defined schemas

A user extends a schema in YAML and refers to it from their instances and specs:

```yaml
# my_lab/rural-parcels.schema.yaml
schema: my_lab/rural-parcels/1
extends: fleetgym/core/1
node:
  village: {type: int, min: 0, description: Village the customer belongs to}
  expected_demand: {type: float, min: 0, unit: load}
global:
  courier_price: {type: float, min: 0, unit: money, description: Price per unit outsourced}
```

Custom constraints, recourse rules and cost components read fields by name
(`info.instance["courier_price"]`, `info.node_field("village")`) and declare the fields they need,
so a spec is checked against the instance schema when it is loaded, before any episode runs: "the
constraint `deadlines` needs node field `deadline`, which schema `fleetgym/core/1` makes optional
and instance `x.json` does not have".

### 2.3 Files

* **One instance**: a JSON document (readable and diffable; fine up to a few thousand nodes).

  ```json
  {"format": "fleetgym-instance/1", "schema": "fleetgym/core/1", "name": "vcsd-moderate-0001",
   "global": {"max_route_duration": 221.47},
   "vehicle_types": [{"capacity": 50, "count": 3}],
   "nodes": {"kind": ["depot", "customer", "..."], "x": [50.0, 13.2, "..."], "y": [50.0, 7.9, "..."],
             "demand": [0, 10, "..."]}}
  ```
* **An instance set** (many instances, such as a paper's test set): a folder of instance files with
  a small manifest (schema, count, checksums, reference, how the files were produced).
* **Benchmarks** (CVRPLIB, Solomon): readers convert them to the format; manifests keep checksums.

### 2.4 Where randomness goes

Without generators there are two kinds of randomness, kept apart:

* **Which instance**: an episode draws an instance from an instance set (seeded, like today's
  `files` source). Changing customer bases (VRP-VCSD) and the training distributions of neural
  solvers (Kool et al.) become instance sets that the user provides (training data is the user's
  task). Examples may show a script that writes instance files; it is not library code.
* **Within the instance**: uncertain values revealed during the episode (demands, service times,
  requests arriving) stay in the spec's `uncertainty` and `dynamics` sections, pre-sampled into the
  scenario. Their parameters may come from the instance (`center: demand`).

## 3. Model primitives (proposal A)

Behaviors become independent switches, with values from the instance:

```yaml
fleet:
  multi_trip: true                 # a depot visit during the day reloads; the route goes on
  reload: {time: reload_time}      # instance field; none by default
  end_route: when_nothing_reachable   # or: anytime
constraints:
  capacity: {}
  route_duration: {mode: hard}     # value: instance max_route_duration (or vehicle type's)
  deadlines: {mode: soft, penalty: lateness}   # values: node deadline
recourse:
  capacity_failure: {serve: what_fits, remainder: open}
```

| Today | Becomes |
| --- | --- |
| `split` | `{serve: what_fits, remainder: open}` + `fleet.multi_trip` + `end_route: when_nothing_reachable` |
| `return_to_depot` | `{serve: what_fits, remainder: come_back}` (`partial_service: false` = `serve: none`) |
| `outsource` | `{serve: what_fits, remainder: outsource}` (price from the instance) |
| `agent_decides` | `{serve: what_fits, remainder: ask, options: [...]}` |
| Vera and Abad's refills | `fleet.multi_trip: true` |
| Marroquín-Cano et al.'s reload time | `fleet.reload.time` |
| `route_duration.max: 221.47` | `route_duration: {mode: hard}` + instance `max_route_duration` |

Custom rules stay possible: `remainder: my_lab.near_depot` names a registered rule.

## 4. Presets (proposal C)

```yaml
# configs/papers/dastpak2022-vcsd.yaml
preset: dastpak2022-vcsd
reference: {key: dastpak2022vcsd, arxiv: "2109.10200", sections: ["3.1", "6.1.1", "B.1.1"]}
model: SVRP-v1                     # the generic family
model_overrides:
  uncertainty.demand: {dist: discrete_uniform, center: demand, half_width: 5, floor: 1}
instances: {set: dastpak2022-vcsd/{density}-q{capacity}, split: test}
grid:
  density: [very_low, low, moderate, high, very_high]
  capacity: [25, 50, 75]
default: {density: moderate, capacity: 50}
```

`fg.make("dastpak2022-vcsd", density="high", capacity=25)`; `fleetgym presets` lists every preset
and grid point. Each grid point resolves to a frozen model spec plus an instance set, both hashed,
so reported results name exactly what was run. The instance sets of a preset are produced once,
with the script that made them and its seed recorded in their manifest.

## 5. Consequences

* Families shrink to a handful: `CVRP`, `VRPTW` (deterministic), `SVRP` (stochastic), `DVRP`
  (dynamic), with multi-agent as an option of each rather than separate IDs. Every config changes,
  so the families move to `-v1`.
* Generators (`uniform`, `clustered`, `zones`) and `register_generator` leave the library; their
  logic moves to example scripts that write instance sets. `GeneratorSource` is removed.
* `Instance` becomes schema-driven: core fields stay typed attributes, schema extensions replace the
  free-form `extra`.
* Binding checks the spec against the instance schema at load time.
* The validator reads the same fields, so it stays independent of the simulator.
* The 14 examples become presets plus notebooks; the build-your-own guide gains "define your
  instance schema" as step 1.

## 6. The user's answers (2026-11-20)

1. **Training data is the user's task.** FleetGym ships no generators and no hook that produces
   instances; users pass their own instances (files, or `instance=` / `reset(options=...)`).
2. **Distribution parameters come from the instances** (problems and methods use different
   distributions).
3. **The `-v0` IDs are removed.**
4. **Formats:** one JSON file per instance; a set is a folder (or glob) of files with a small
   manifest. No bulk format for now (section 2.3's Parquet sets are dropped).
5. **Order** as proposed: (a) instance schema, format and readers; (b) model primitives;
   (c) presets; (d) migration; then M14.
