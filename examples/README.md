# FleetGym examples

One folder per example, from a first environment to a custom VRP variant. Each folder has a script
and a notebook with the same code: run the script from a terminal, or open the notebook in Jupyter to
go through it cell by cell. The documentation includes the scripts verbatim, and the test suite runs
every script and every notebook (`tests/examples`), so they stay correct.

FleetGym ships no instances: every example writes the instances it needs, in the FleetGym format, with
its own function (the paper examples keep it in `instances.py`). Examples 09 to 14 each reproduce
the problem setting of a published paper through a preset (cited in the notebook, with a table
mapping the paper's definitions to the model and the instances).

| Folder | Shows | Needs |
| --- | --- | --- |
| `01_first_environment/` | make, step with masks, costs, the independent validator | |
| `02_stochastic_demands/` | scenarios, forecasts, recourse, replay, non-anticipative forks | |
| `03_dynamic_day/` | requests during the day: accept/reject, and next_node with wait | |
| `04_compare_policies/` | fixed test sets, confidence intervals, CVaR, paired tests, report card | |
| `05_custom_variant/` | **build your own variant**, step by step: your instance schema and instances, a YAML model, a custom distribution, constraint, recourse rule and policy, verification and a fair comparison | |
| `06_train_agent/` | MaskablePPO with a graph attention policy, evaluated against a baseline | `fleetgym[sb3]` |
| `07_multi_agent/` | PettingZoo Parallel API, one agent per vehicle | |
| `08_render/` | map and vehicle Gantt figure, `rgb_array` frames | `fleetgym[viz]` |
| `09_vrp_vcsd/` | preset `dastpak2022-vcsd`: the VRP-VCSD of Dastpak, Errico and Jabali, the paper's 15 instances | |
| `10_vrpsd_goodson/` | preset `dastpak2022-goodson`: Goodson et al.'s VRPSD with duration limits on Solomon R101 (27 instances) | download |
| `11_cvrp_kool/` | preset `kool2019-cvrp`: the instance distribution of Kool et al. (2019), n = 20, 50, 100 | |
| `12_vrptw_solomon/` | preset `solomon1987-vrptw`: replaying R101's best-known solution reproduces its cost | download |
| `13_vrpsd_florio/` | preset `florio2023-vrpsd`: Poisson demands on CVRPLIB X instances (Florio et al. 2023) | download |
| `14_ma_cvrp_vera/` | preset `vera2019-hetfleet`: multi-agent CVRP with a heterogeneous fleet (Vera and Abad 2019) | |
| `15_dvrp_ulmer/` | preset `ulmer2016-vrpssr`: one vehicle, stochastic service requests decided at the vehicle's arrivals (Ulmer et al. 2016), 18 settings | |

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
