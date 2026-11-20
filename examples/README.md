# FleetGym examples

Runnable scripts, from a first environment to a custom VRP variant. The documentation includes these
files verbatim, and the test suite runs every one of them (`tests/examples`), so they stay correct.

| Script | Shows | Needs |
| --- | --- | --- |
| `01_first_environment.py` | make, step with masks, costs, the independent validator | |
| `02_stochastic_demands.py` | scenarios, forecasts, recourse, replay, non-anticipative forks | |
| `03_dynamic_day.py` | requests during the day: accept/reject, and next_node with wait | |
| `04_compare_policies.py` | fixed test sets, confidence intervals, CVaR, paired tests, report card | |
| `05_custom_variant/` | a new variant: YAML, a custom distribution, constraint and recourse rule | |
| `06_train_agent.py` | MaskablePPO with a graph attention policy, evaluated against a baseline | `fleetgym[sb3]` |
| `07_multi_agent.py` | PettingZoo Parallel API, one agent per vehicle | |
| `08_render.py` | map and vehicle Gantt figure, `rgb_array` frames | `fleetgym[viz]` |

Run one with `python examples/01_first_environment.py`. Outputs go to `runs/`.
