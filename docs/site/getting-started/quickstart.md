# Quickstart

## A first episode

Every environment is a Gymnasium environment. Observations are dictionaries of padded arrays, and
`info["action_mask"]` (or `env.action_masks()`) tells which actions are valid now.

```{literalinclude} ../../../examples/01_first_environment/first_environment.py
:language: python
```

## From the command line

```bash
fleetgym list                       # registered environments
fleetgym show VRPSD-v0              # the resolved YAML of an environment
fleetgym validate my-variant.yaml   # check a config: errors name the file and the key
fleetgym eval CVRP-v0 --baseline nearest --episodes 32
fleetgym render DVRP-StochReq-v0 --out day.png
```

## Where next

- {doc}`../concepts/problem-specs`: how a problem is described.
- {doc}`../guides/build-your-own-variant`: make your own variant in a few minutes.
- {doc}`../guides/training`: train an agent.
