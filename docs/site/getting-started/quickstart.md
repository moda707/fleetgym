# Quickstart

## A first episode

Every environment is a Gymnasium environment. Observations are dictionaries of padded arrays, and
`info["action_mask"]` (or `env.action_masks()`) tells which actions are valid now. FleetGym ships
no instances, so the example writes a few first ({doc}`../concepts/instances`).

```{literalinclude} ../../../examples/01_first_environment/first_environment.py
:language: python
```

## A published setting

```python
import fleetgym as fg

env = fg.make_preset("solomon1987-vrptw", instance="r101")  # downloads R101, checks its checksum
obs, info = env.reset(seed=0)
```

## From the command line

```bash
fleetgym list                                  # registered model families
fleetgym show SVRP-v1                          # the resolved YAML of a model
fleetgym validate my-variant.yaml              # check a config: errors name the file and the key
fleetgym eval CVRP-v1 --instances data/cvrp --baseline nearest --episodes 32
fleetgym render VRPTW-v1 --instances data/vrptw --out day.png
```

## Where next

- {doc}`../concepts/problem-specs` and {doc}`../concepts/instances`: how a problem is described.
- {doc}`../guides/build-your-own-variant`: make your own variant.
- {doc}`../guides/training`: train an agent.
