# FleetGym

**Reinforcement learning environments for vehicle routing, built for stochastic and dynamic problems.**

FleetGym turns vehicle routing problems into [Gymnasium](https://gymnasium.farama.org) and
[PettingZoo](https://pettingzoo.farama.org) environments. One YAML file describes a problem: the
fleet, the network, the constraints, what is uncertain and when it becomes known, how the agent
decides and what it pays. Changing a variant means editing that file, not writing a new simulator.

- **Problems as data**: CVRP, VRPTW, stochastic demands with recourse, requests arriving during
  the day, multi-agent fleets, and your own variants, all from YAML files that extend each other.
- **One simulator for everything**: a discrete-event engine with exact snapshots, an event log of
  everything that happened, and action masks that only use what is known.
- **Trustworthy results**: an independent validator re-checks every episode; benchmark solutions
  (CVRPLIB X, Solomon) replay exactly; expected costs match independent dynamic programs; every
  random value is pre-sampled into a scenario that can be replayed.
- **Easy to extend**: new distributions, constraints, recourse rules, encoders and decoders plug in
  by name. See {doc}`guides/build-your-own-variant`.
- **Batteries included**: OR baselines (nearest neighbor, insertion, rollout, PyVRP re-optimization,
  a perfect-information benchmark), an evaluation harness with confidence intervals and paired
  tests, SB3 training with graph policies, and figures of every episode.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/installation
getting-started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/problem-specs
concepts/simulation
concepts/uncertainty
concepts/recourse
concepts/dynamic-requests
concepts/multi-agent
concepts/validation
environments
```

```{toctree}
:maxdepth: 2
:caption: Guides

guides/build-your-own-variant
guides/stochastic-and-dynamic
guides/baselines-and-evaluation
guides/training
guides/visualization
```

```{toctree}
:maxdepth: 1
:caption: Reference

reference/api
reference/cli
reference/design-decisions
changelog
```
