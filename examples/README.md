# FleetGym examples

One folder per example, from a first environment to a custom VRP variant. Each folder has a script
and a notebook with the same code: run the script from a terminal, or open the notebook in Jupyter to
go through it cell by cell. The documentation includes the scripts verbatim, and the test suite runs
every script and every notebook (`tests/examples`), so they stay correct.

| Folder | Shows | Needs |
| --- | --- | --- |
| `01_first_environment/` | make, step with masks, costs, the independent validator | |
| `02_stochastic_demands/` | scenarios, forecasts, recourse, replay, non-anticipative forks | |
| `03_dynamic_day/` | requests during the day: accept/reject, and next_node with wait | |
| `04_compare_policies/` | fixed test sets, confidence intervals, CVaR, paired tests, report card | |
| `05_custom_variant/` | a new variant: YAML, a custom distribution, constraint and recourse rule | |
| `06_train_agent/` | MaskablePPO with a graph attention policy, evaluated against a baseline | `fleetgym[sb3]` |
| `07_multi_agent/` | PettingZoo Parallel API, one agent per vehicle | |
| `08_render/` | map and vehicle Gantt figure, `rgb_array` frames | `fleetgym[viz]` |

Run a script with `python examples/01_first_environment/first_environment.py`, or open
`first_environment.ipynb` with the notebook's folder as the working directory (Jupyter's default).
Outputs go to `runs/` in the working directory.

## Editing an example

The script is the source; the notebook is generated from it. Cells are marked with `# %%` (code) and
`# %% [markdown]` (Markdown, each line starting with `# `). After editing a script, regenerate the
notebooks:

```bash
uv run python scripts/build_example_notebooks.py
```

A test fails if a notebook is out of date.
