# FleetGym

[![CI](https://github.com/moda707/fleetgym/actions/workflows/ci.yml/badge.svg)](https://github.com/moda707/fleetgym/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/fleetgym/badge/?version=latest)](https://fleetgym.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/moda707/fleetgym/blob/main/LICENSE)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)

Gymnasium and PettingZoo environments for reinforcement learning on vehicle routing problems,
with a focus on stochastic and dynamic variants where information arrives during the day.

> **Status: pre-alpha.** The v0.1 core is under construction (see [`STATUS.md`](https://github.com/moda707/fleetgym/blob/main/STATUS.md) and the
> [implementation plan](https://github.com/moda707/fleetgym/blob/main/docs/IMPLEMENTATION_PLAN.md)). APIs will change.

## What is there today
- **Models and instances.** A YAML model says what a problem has and how it behaves (fleet,
  constraints, what is uncertain and when it is revealed, decisions, objective); instances, in a
  published JSON format with schemas you can extend, carry every number. Models `extend` each
  other; every spec and instance has a stable content hash saved with logs and results.
- **Paper presets.** Published settings with their grids: Solomon, CVRPLIB (Florio et al.), Kool et
  al., Dastpak et al. (VRP-VCSD, Goodson's VRPSD), Vera and Abad, Ulmer et al. (dynamic requests),
  Taş et al. (stochastic travel times), Errico et al. (stochastic service times, hard windows),
  Dastpak et al. (an EV's shortest path over busy charging stations, `EVSPP-v1`). FleetGym ships no instances:
  benchmarks are downloaded with checksums, and each example writes the instances its preset needs.
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
- **Verified against benchmarks.** `CVRP-v1` reproduces all 100 CVRPLIB X best-known solutions
  exactly; `VRPTW-v1` reproduces all 56 Solomon (100-customer) best-known solutions from SINTEF.

| Model family | Status |
| --- | --- |
| `CVRP-v1` | stable |
| `VRPTW-v1` | stable |
| `SVRP-v1` | experimental (stochastic demands, detour-to-depot recourse) |
| `DVRP-v1` | experimental (requests arriving during the day, accept/reject) |
| `EVSPP-v1` | experimental (one EV, charging stations with queues and occupancy indicators) |
| `VRPSD-Outsource-v1` | planned (stochastic demands with outsourcing) |

Any family becomes multi-agent (PettingZoo, one agent per vehicle) with `agents: {mode: multi}`.

## Quick start

```python
import fleetgym as fg

env = fg.make_preset("solomon1987-vrptw", instance="r101")   # or fg.make("CVRP-v1", instance=...)
obs, info = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample(mask=env.action_masks().astype("int8"))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

Command line:

```bash
fleetgym list                       # registered model families
fleetgym show VRPTW-v1              # the resolved config
fleetgym validate my-config.yaml    # check a config
fleetgym replicate CVRP-v1          # replay published best-known solutions (downloads data)
```

## Train and evaluate

```bash
pip install "fleetgym[sb3]"
fleetgym train CVRP-v1 --instances data/train --algo maskable_ppo --policy gat --out runs/cvrp
fleetgym eval CVRP-v1 --instances data/test --model runs/cvrp --reference nearest --out report.json
```

Render an episode (map and vehicle Gantt chart) with `fleetgym render VRPTW-v1 --instances data --out day.png`
or `render_mode="rgb_array"` (needs `fleetgym[viz]`).

Baselines (random, nearest neighbor, cheapest insertion, rollout, PyVRP re-optimization and a
perfect-information benchmark) run through the same evaluation harness, on fixed test scenarios.

## Documentation and examples

The [documentation](https://fleetgym.readthedocs.io) (source in `docs/site`) covers
the concepts, every environment, and a step-by-step guide to building your own VRP variant. The
[`examples/`](https://github.com/moda707/fleetgym/tree/main/examples) folder has one folder per example, each with a
script and a matching notebook, from a first environment to a custom variant, training and
multi-agent routing.

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
