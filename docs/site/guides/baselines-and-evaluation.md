# Baselines and evaluation

## Policies

Baselines and trained agents implement one protocol: `reset(env)`, then `act(obs, info)`.

| Baseline | Idea |
| --- | --- |
| `RandomPolicy` | uniform over valid actions (sanity check) |
| `NearestNeighbor` | nearest valid customer; wait or return when none |
| `CheapestInsertion` | route the deciding vehicle by cheapest insertion |
| `AprioriRoute` | a fixed customer order, failures handled by recourse |
| `Rollout` | one-step lookahead on forks with a base policy |
| `PyVRPReoptimization` | re-solve the known problem with PyVRP at every epoch (`fleetgym[solvers]`) |
| `PerfectInformation` | knows the scenario: an exact lower bound on tiny instances, otherwise a labeled heuristic |

## Evaluation protocol

Policies are compared on fixed test scenarios (`scenario_seeds(env_id, "test", n)`), every episode is
validated, and results carry confidence intervals, CVaR and paired tests.

```{literalinclude} ../../../examples/04_compare_policies/compare_policies.py
:language: python
```

From the command line:

```bash
fleetgym eval VRPSD-v0 --baseline insertion --episodes 64 --reference nearest --out report.json
```

A report card records the spec hash, seeds, costs, summary statistics, package versions and platform;
anticipative results must say whether they are a bound or a heuristic.
