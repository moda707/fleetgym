# Build your own VRP variant

FleetGym is designed so that a new variant is mostly configuration. This guide builds one in steps:
first with YAML only, then with a custom distribution, constraint and recourse rule, and finally as a
plugin package others can install. Every file shown here is in
[`examples/05_custom_variant`](https://github.com/moda707/fleetgym/tree/main/examples/05_custom_variant)
and runs in the test suite; the notebook there (`custom_variant.ipynb`) walks through the same steps
interactively, with tests of each block, a custom policy and a fair comparison.

## Step 1: YAML only

Most variants need no code. Start from the closest registered environment and change what differs.

```yaml
id: VRPSD-Tight-v0
extends: VRPSD-v0                      # inherit everything else
meta: {name: Stochastic demands with small trucks, status: experimental}
fleet:
  sizing: fixed
  types: [{name: van, capacity: 15, count: 6}]
decisions: {interface: next_node, epochs: on_event}   # vehicles act concurrently
uncertainty:
  demand: {dist: truncated_normal, mean: from_instance, std: 2.0, low: 0, revealed: on_arrival}
recourse:
  capacity_failure: {rule: outsource, per_unit: 3.0}
objective: {distance: 1.0, outsourcing: 1.0, unserved: 200.0}
```

```bash
fleetgym validate vrpsd-tight.yaml     # errors name the file and the key path
fleetgym show vrpsd-tight.yaml         # the resolved spec, with every default
```

```python
import fleetgym as fg

env = fg.make("vrpsd-tight.yaml")                          # or a dict, or a ProblemSpec
env = fg.make("VRPSD-v0", overrides=["fleet.types.0.capacity=15"])  # quick changes
```

Things you can change without code: the network (Euclidean with rounding, or a matrix), the fleet,
constraints (capacity, hard or soft time windows, route duration, shifts), any parameter's
distribution and revelation time, requests arriving during the day, the recourse rule, the decision
interface and epochs, the objective (weights or a lexicographic order), and multi-agent settings.

## Step 2: a custom distribution

Register a parameter model (validated from YAML) and a sampler. The sampler draws one value per node
from the generator it is given, so scenarios stay reproducible and sources stay independent.

```{literalinclude} ../../../examples/05_custom_variant/plugins.py
:language: python
:start-at: "# 1. A distribution"
:end-before: "# 2. A constraint"
```

## Step 3: a custom constraint

A constraint masks infeasible customers for the vehicle that decides (it sees only known
information) and can report soft-violation costs. Its mask must look ahead far enough that a vehicle
can always return to the depot.

```{literalinclude} ../../../examples/05_custom_variant/plugins.py
:language: python
:start-at: "# 2. A constraint"
:end-before: "# 3. A recourse rule"
```

## Step 4: a custom recourse rule

A recourse rule decides what happens when a revealed demand does not fit: serve what fits first or
not (`serve_first`), then restock (`"return_to_depot"`) or outsource (`"outsource"`), or return `None`
to let the policy decide. It may also restock preventively (`after_service`).

```{literalinclude} ../../../examples/05_custom_variant/plugins.py
:language: python
:start-at: "# 3. A recourse rule"
:end-before: "# 4. An instance generator"
```

## Step 5: a custom instance generator

A generator turns a seeded random generator into an `Instance` (node 0 is the depot), so every
episode can have a new layout. Give its parameter model `n_customers` (or a `max_customers`
property) so environments can size their observations.

```{literalinclude} ../../../examples/05_custom_variant/plugins.py
:language: python
:start-at: "# 4. An instance generator"
```

## Step 6: use them in YAML and run

```{literalinclude} ../../../examples/05_custom_variant/my_variant.yaml
:language: yaml
```

```{literalinclude} ../../../examples/05_custom_variant/custom_variant.py
:language: python
:start-at: "spec = fg.load_spec"
```

The independent validator checks the physics of every episode, custom rules included (loads,
restock trips, outsourcing costs, revealed values). Custom constraints are not known to the
validator: test them with your own checks.

## Step 7: share it as a plugin

Put the registrations in a package and expose them through the `fleetgym.plugins` entry point; they
load automatically when a config names them.

```toml
# pyproject.toml of your package
[project.entry-points."fleetgym.plugins"]
my_variants = "my_variants.plugins:register"
```

```python
# my_variants/plugins.py
def register() -> None:
    from my_variants import constraints, distributions, generators, recourse  # noqa: F401
```

Ship your YAML files with the package and load them by path, or contribute them to FleetGym.

## Checklist for a new variant

- [ ] `fleetgym validate` passes, and the spec hash is recorded with your results.
- [ ] Episodes validate: `fleetgym.testing.validator.validate(log, spec, instance, scenario=...)`.
- [ ] Observations do not leak hidden values: `fleetgym.testing.check_non_anticipativity(env)`.
- [ ] Baselines run on it: `fleetgym eval my.yaml --baseline nearest`.
- [ ] If you change a published variant's dynamics or defaults, give it a new `-vN`.
- [ ] Custom neural modules pass `fleetgym.testing.check_module` ({doc}`training`).
