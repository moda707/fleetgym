# FleetGym

[![CI](https://github.com/moda707/fleetgym/actions/workflows/ci.yml/badge.svg)](https://github.com/moda707/fleetgym/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/moda707/fleetgym/blob/main/LICENSE)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)

Gymnasium and PettingZoo environments for reinforcement learning on vehicle routing problems,
with a focus on stochastic and dynamic variants where information arrives during the day.

> **Status: pre-alpha.** The v0.1 core is under construction (see [`STATUS.md`](https://github.com/moda707/fleetgym/blob/main/STATUS.md) and the
> [implementation plan](https://github.com/moda707/fleetgym/blob/main/docs/IMPLEMENTATION_PLAN.md)). APIs will change.

## What is there today
- **Problems as data.** Each environment is one YAML file: fleet, network, constraints, uncertainty,
  decisions and objective. Configs can `extend` each other and be overridden from the command line.
  Every spec has a stable content hash that is saved with logs and results.
- **A discrete-event simulator** with exact `get_state` / `set_state` / `clone` (including RNG state)
  and an event log of everything that happened.
- **An independent validator** that re-checks any episode from its event log, so a simulator bug
  cannot hide itself.
- **Uncertainty and recourse.** Uncertain demands and service times are pre-sampled into a scenario,
  revealed to the agent when the spec says, and replayed exactly. Capacity failures are handled by
  standard or user-written recourse rules; `fork(seed)` gives planners a copy that cannot see the
  future.
- **Dynamic requests.** Requests arrive during the day (Poisson or time-of-day rates), can cancel,
  and are accepted, rejected, outsourced or assigned by the agent while a built-in routing rule
  drives the vehicles.
- **Verified against benchmarks.** `CVRP-v0` reproduces all 100 CVRPLIB X best-known solutions
  exactly; `VRPTW-v0` reproduces all 56 Solomon (100-customer) best-known solutions from SINTEF.

| Environment | Status |
| --- | --- |
| `CVRP-v0` | stable |
| `VRPTW-v0` | stable |
| `DVRP-StochReq-v0` | experimental (requests arriving during the day, accept/reject) |
| `VRPSD-v0` | experimental (stochastic demands, return-to-depot recourse) |
| `VRPSD-Outsource-v0` | planned (stochastic demands with outsourcing) |
| `MA-CVRP-v0`, `MA-DVRP-v0` | experimental (multi-agent, PettingZoo: one agent per vehicle) |

## Quick start

```python
import fleetgym as fg

env = fg.make("CVRP-v0")
obs, info = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample(mask=env.action_masks().astype("int8"))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

Command line:

```bash
fleetgym list                       # registered environments
fleetgym show VRPTW-v0              # the resolved config
fleetgym validate my-config.yaml    # check a config
fleetgym replicate CVRP-v0          # replay published best-known solutions (downloads data)
```

## Train and evaluate

```bash
pip install "fleetgym[sb3]"
fleetgym train CVRP-v0 --algo maskable_ppo --policy gat --steps 100000 --out runs/cvrp
fleetgym eval CVRP-v0 --model runs/cvrp --episodes 32 --reference nearest --out runs/cvrp/report.json
```

Render an episode (map and vehicle Gantt chart) with `fleetgym render DVRP-StochReq-v0 --out day.png`
or `render_mode="rgb_array"` (needs `fleetgym[viz]`).

Baselines (random, nearest neighbor, cheapest insertion, rollout, PyVRP re-optimization and a
perfect-information benchmark) run through the same evaluation harness, on fixed test scenarios.

## Install (development)

```bash
git clone https://github.com/moda707/fleetgym.git
cd fleetgym
uv sync --extra dev
uv run pytest
```

Requires Python 3.12 or later. Optional extras: `sb3`, `gnn`, `solvers`, `viz`, `all`.
See [`CONTRIBUTING.md`](https://github.com/moda707/fleetgym/blob/main/CONTRIBUTING.md) for the development workflow.

## Benchmark data
Benchmark instances and best-known solutions are not stored in this repository. They are
downloaded on request from their publishers (CVRPLIB, SINTEF TOP), checked against SHA-256
checksums, and cached under `$FLEETGYM_DATA_DIR` (default `~/.cache/fleetgym`).

## Citing

If you use FleetGym in your research, please cite it. GitHub's "Cite this repository" button and
[`CITATION.cff`](https://github.com/moda707/fleetgym/blob/main/CITATION.cff) give the citation; a Zenodo DOI will be added with the first release.

## License

MIT. See [`LICENSE`](https://github.com/moda707/fleetgym/blob/main/LICENSE).
