# Problem specs and configs

A **spec** describes one VRP variant. It is written in YAML, validated into a frozen `ProblemSpec`,
and identified by a content hash (`spec.content_hash()`), which is stored with every log, scenario,
model and report so results can always be traced to the exact problem.

```yaml
id: VRPSD-v0
extends: CVRP-v0                 # inherit everything, then change what differs
meta: {name: VRP with stochastic demands, status: experimental}
uncertainty:
  demand: {dist: poisson, mean: from_instance, revealed: on_arrival}
recourse:
  capacity_failure: {rule: return_to_depot, partial_service: true}
```

## Sections

| Section | Describes |
| --- | --- |
| `instance_source` | where instances come from: a generator, or files (VRPLIB, Solomon) |
| `network` | distances and travel times (`euclidean` with rounding, or an explicit `matrix`) |
| `fleet` | vehicle types, capacities, counts (`fixed`, `limited`, `unlimited`) and shifts |
| `constraints` | capacity, time windows (hard or soft), route duration, outsourcing, your own |
| `uncertainty` | which parameters are random, their distribution, and when they are revealed |
| `dynamics` | requests arriving during the day: arrival process, request class, cutoff |
| `recourse` | what happens when revealed information makes planned service impossible |
| `decisions` | when the agent decides (`on_event`, `sequential`) and how (`next_node`, `accept_reject`, `assign`) |
| `objective` | weighted cost components, or a lexicographic order |
| `agents` | single or multi-agent, reward sharing, conflict rules |

## Rules worth knowing

- **Merging** (`extends`): maps merge deeply, constraints merge by name, `null` removes a key (the
  default applies), changing a discriminator (`dist`, `type`, `rule`, ...) replaces the whole
  subtree, and `meta` is never inherited.
- **Instance values**: `from_instance` reads a value from the instance (demands, capacity, the depot's
  closing time, ...), with a documented default key per field, or `{from_instance: key}`.
- **Overrides**: `fg.make("CVRP-v0", overrides=["fleet.types.0.capacity=80"])`.
- **Versions**: a change to an environment's dynamics or defaults changes its hash; published
  environments then get a new `-vN`. The hashes of registered configs are snapshotted in the tests.
- **Errors**: invalid configs raise `ConfigError` naming the file and the dotted key path, for
  example `configs/dynamic/X-v0.yaml: decisions.interface`.

Editors validate configs as you type with the JSON Schema (`fleetgym schema`): the shipped configs
start with `# yaml-language-server: $schema=...`.
