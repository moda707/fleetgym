# Build your own VRP variant

FleetGym is designed so that a new variant is mostly configuration and data. A variant has a
**model** (YAML: what exists and how it behaves) and **instances** (every number, in files that
follow an instance schema); see {doc}`../concepts/instances`. This guide builds one in steps: the
model from YAML alone, then your own instance schema and instances, then a custom distribution,
constraint and recourse rule, and finally a plugin package. Every file shown here is in
[`examples/05_custom_variant`](https://github.com/moda707/fleetgym/tree/main/examples/05_custom_variant)
and runs in the test suite; the notebook there (`custom_variant.ipynb`) walks through the same
steps interactively, with tests of each block, a custom policy and a fair comparison.

## Step 1: a model from YAML only

Most variants need no code. Start from the closest family and change what differs. The model says
*what* exists; the numbers stay in the instances.

```yaml
id: SVRP-Fleet-v1
extends: SVRP-v1                        # stochastic demands, detour-to-depot recourse
meta: {name: Stochastic demands with a fixed fleet, status: experimental}
fleet: {sizing: fixed, multi_trip: true}               # the instances give the vehicles
decisions: {interface: next_node, epochs: on_event}    # vehicles act concurrently
constraints:
  capacity: {}
  route_duration: {mode: soft}                         # limit from the instance; excess = overtime
recourse:
  capacity_failure: {serve: what_fits, remainder: outsource, per_unit: {from_instance: courier_price}}
objective: {distance: 1.0, outsourcing: 1.0, overtime: 2.0, unserved: 200.0}
```

```bash
fleetgym validate svrp-fleet.yaml     # errors name the file and the key path
fleetgym show svrp-fleet.yaml         # the resolved spec, with every default
```

Things you can change without code: the network (Euclidean with rounding, or a matrix), the fleet
(sizing, multi-trip routes, reload time, the end-of-day rule), constraints (capacity, hard or soft
time windows and duration limits, shifts), what is uncertain and when it is revealed, requests
arriving during the day, the recourse (`{serve, remainder}`), the decision interface and epochs,
the objective (weights or a lexicographic order), and multi-agent settings.

## Step 2: your instance schema and instances

If your variant needs numbers the core schema does not have, extend it:

```{literalinclude} ../../../examples/05_custom_variant/rural-parcels.schema.yaml
:language: yaml
```

Then write the code that makes your instances (FleetGym ships no generators). It builds `Instance`
objects in your schema, including the distribution of each uncertain value, and writes them as an
instance set (files plus a manifest with checksums and the seed):

```{literalinclude} ../../../examples/05_custom_variant/instances.py
:language: python
:start-at: "def village_instance"
```

## Step 3: a custom distribution

Register a parameter model (validated from the instance or YAML) and a sampler. The sampler draws
one value per node from the random generator it is given, so scenarios stay reproducible and
sources stay independent.

```{literalinclude} ../../../examples/05_custom_variant/plugins.py
:language: python
:start-at: "# 1. A distribution"
:end-before: "# 2. A constraint"
```

## Step 4: a custom constraint

A constraint masks infeasible customers for the vehicle that decides (it sees only known
information) and can report soft-violation costs. It reads its numbers from the instance
(`self.problem.value(...)`).

```{literalinclude} ../../../examples/05_custom_variant/plugins.py
:language: python
:start-at: "# 2. A constraint"
:end-before: "# 3. A recourse rule"
```

## Step 5: a custom recourse rule

A recourse rule decides what happens when a revealed demand does not fit: serve what fits first or
not (`serve_first`), then restock (`"return_to_depot"`), outsource (`"outsource"`), leave the rest
open (`"split"`), or return `None` to let the policy decide. It may also restock preventively
(`after_service`).

```{literalinclude} ../../../examples/05_custom_variant/plugins.py
:language: python
:start-at: "# 3. A recourse rule"
```

## Step 6: use them in YAML and run

```{literalinclude} ../../../examples/05_custom_variant/my_variant.yaml
:language: yaml
```

```{literalinclude} ../../../examples/05_custom_variant/custom_variant.py
:language: python
:start-at: "spec = fg.load_spec"
:end-before: "# ## 5. A policy of your own"
```

The independent validator checks the physics of every episode, custom rules included (loads,
restock trips, outsourcing costs, revealed values). Custom constraints are not known to the
validator: test them with your own checks.

## Step 7: share it as a plugin

Put the schema, the registrations and the instance code in a package and expose the registrations
through the `fleetgym.plugins` entry point; they load automatically when a config names them.

```toml
# pyproject.toml of your package
[project.entry-points."fleetgym.plugins"]
my_variants = "my_variants.plugins:register"
```

```python
# my_variants/plugins.py
def register() -> None:
    from my_variants import constraints, distributions, recourse  # noqa: F401
```

Ship your YAML files and schema with the package and load them by path, publish your instance sets
(with their manifests), or contribute a preset to FleetGym.

## Checklist for a new variant

- [ ] `fleetgym validate` passes, and the spec hash is recorded with your results.
- [ ] Instances follow your schema, and the instance set's manifest is published with your results.
- [ ] Episodes validate: `fleetgym.testing.validator.validate(log, spec, instance, scenario=...)`.
- [ ] Observations do not leak hidden values: `fleetgym.testing.check_non_anticipativity(env)`.
- [ ] Baselines run on it: `fleetgym eval my.yaml --instances my_set --baseline nearest`.
- [ ] If you change a published variant's dynamics or defaults, give it a new `-vN`.
- [ ] Custom neural modules pass `fleetgym.testing.check_module` ({doc}`training`).
