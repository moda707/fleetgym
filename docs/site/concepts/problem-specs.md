# Problem specs and configs

A **spec** (the model) describes what a VRP variant has and how it behaves; the numbers come from
the instances ({doc}`instances`). It is written in YAML, validated into a frozen `ProblemSpec`, and
identified by a content hash (`spec.content_hash()`), which is stored with every log, scenario,
model and report so results can always be traced to the exact problem.

```yaml
id: SVRP-v1
extends: CVRP-v1                 # inherit everything, then change what differs
meta: {name: VRP with stochastic demands, status: experimental}
uncertainty:
  demand: {revealed: on_arrival}             # the instance gives the distribution
recourse:
  capacity_failure: {serve: what_fits, remainder: come_back}
```

## Sections

| Section | Describes |
| --- | --- |
| `instance_source` | where instances come from: files or instance sets, or a benchmark collection |
| `network` | distances and travel times (`euclidean` with rounding, or an explicit `matrix`) |
| `fleet` | sizing (`fixed`, `limited`, `unlimited`), types (from the instance by default), shift, `multi_trip`, `reload_time`, `end_route` |
| `constraints` | capacity, time windows and route duration (`mode: hard\|soft`), outsourcing, your own |
| `uncertainty` | which parameters are random and when they are revealed (the distribution: the instance's, or given here) |
| `dynamics` | requests arriving during the day: arrival process, request class, cutoff |
| `recourse` | what happens when revealed information makes planned service impossible |
| `decisions` | when the agent decides (`on_event`, `sequential`) and how (`next_node`, `accept_reject`, `assign`) |
| `objective` | weighted cost components, or a lexicographic order |
| `agents` | single or multi-agent, reward sharing, conflict rules |

## Rules worth knowing

- **Merging** (`extends`): maps merge deeply, constraints merge by name, `null` removes a key (the
  default applies), changing a discriminator (`dist`, `type`, `rule`, ...) replaces the whole
  subtree, and `meta` is never inherited.
- **Instance values**: `from_instance` reads a value from the instance (a schema field, or derived
  values such as the depot's closing time), with a documented default key per field, or
  `{from_instance: key}`.
- **Overrides**: `fg.make("CVRP-v1", instance=inst, overrides=["objective.distance=2"])`.
- **Versions**: a change to an environment's dynamics or defaults changes its hash; published
  environments then get a new `-vN`. The hashes of registered configs are snapshotted in the tests.
- **Errors**: invalid configs raise `ConfigError` naming the file and the dotted key path, for
  example `configs/dynamic/X-v1.yaml: decisions.interface`.

Editors validate configs as you type with the JSON Schema (`fleetgym schema`): the shipped configs
start with `# yaml-language-server: $schema=...`.
