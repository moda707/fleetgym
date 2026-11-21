# %% [markdown]
# # VRP with a highly variable customer basis and stochastic demands (VRP-VCSD)
#
# **Paper.** M. Dastpak, F. Errico and O. Jabali, *Off-line approximate dynamic programming for the
# vehicle routing problem with a highly variable customer basis and stochastic demands*,
# arXiv:2109.10200. This example reproduces the paper's **problem setting** in FleetGym (not its
# method or results), with the preset `dastpak2022-vcsd`.
#
# **Problem.** Every morning a new set of customers requests a pickup (the paper's motivation is
# donor collection). Each customer has a known expected demand; the actual demand is revealed on
# the first visit. A homogeneous fleet of `m` vehicles with capacity `Q` starts at the depot.
# Deliveries can be split: a vehicle serves what fits and the rest stays open, for the same or
# another vehicle. A vehicle may go back to the depot to empty itself whenever it wants (and must,
# when it is full), and every vehicle must be back by the duration limit `L`. The objective is to
# maximize the expected served demand.
#
# | Paper (Section 3.1, Appendix B.1.1) | FleetGym |
# | --- | --- |
# | a new customer set each day | an instance set; each episode draws one instance |
# | m, Q, L, expected demands, demand distribution | the instance (`instances.py`) |
# | split service: the rest stays open | `recourse: {serve: what_fits, remainder: open}` |
# | reload any time, a full vehicle must | `fleet.multi_trip: true` |
# | end the day only when nothing is reachable | `fleet.end_route: when_nothing_reachable` |
# | duration limit L | `route_duration: {mode: hard}` (value from the instance) |
# | maximize served demand, rewarded on delivery | `objective: {served_demand: -1}` |
# | one vehicle decides at a time (decentralized) | `epochs: on_event` |

# %%
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))  # this folder's instances.py

import instances  # noqa: E402
import numpy as np  # noqa: E402

import fleetgym as fg  # noqa: E402
from fleetgym.baselines import NearestNeighbor, RandomPolicy, run_episode  # noqa: E402
from fleetgym.testing.validator import validate  # noqa: E402

preset = fg.load_preset("dastpak2022-vcsd")
print(preset.name, "| grid:", dict(preset.grid), "| notes:", *preset.notes, sep="\n  ")

# %% [markdown]
# ## The instances
#
# `instances.py` draws them as the paper's Appendix B.1.1 describes and writes one set per grid
# point (density x capacity), in the folders the preset expects. The seed is recorded in each set.

# %%
print(inspect.getsource(instances.vcsd_instance))
sets = instances.write_vcsd("instances", n_instances=8, seed=0)
print(len(sets), "instance sets:", ", ".join(sets))

# %% [markdown]
# ## One day, step by step
#
# The action is the next node of the vehicle that must decide; node 0 is the depot (a reload during
# the day). A full vehicle has only the depot in its mask. The reward is the demand delivered since
# the previous decision.

# %%
env = fg.make_preset("dastpak2022-vcsd", instances="instances")  # moderate density, Q = 50
obs, info = env.reset(seed=0)
instance = env.base_instance
print(
    instance.n_customers,
    "customers; m =",
    instance.vehicle_types[0]["count"],
    "; L =",
    instance.extra["max_route_duration"],
)
rng = np.random.default_rng(0)
served = 0.0
while True:
    action = int(rng.choice(np.flatnonzero(env.action_masks())))
    obs, reward, terminated, truncated, info = env.step(action)
    served += reward
    if terminated or truncated:
        break
log = env.event_log
print(
    f"served {served:.0f}; splits {len(log.of_kind('failure'))}; reloads {len(log.of_kind('restock'))}"
)
print("valid:", validate(log, env.problem_spec, env.base_instance, scenario=env.scenario).ok)

# %% [markdown]
# ## The paper's 15 instances, with two baselines
#
# Each grid point is the same model on another instance set.

# %%
for point in preset.points():
    env = fg.make_preset("dastpak2022-vcsd", instances="instances", **point)
    served = {
        policy.name: np.mean(
            [run_episode(env, policy, seed=s).costs.get("served_demand", 0.0) for s in range(5)]
        )
        for policy in (RandomPolicy(), NearestNeighbor())
    }
    print(
        f"{point['density']:>9} Q={point['capacity']:2d}: "
        + "  ".join(f"{k} {v:6.1f}" for k, v in served.items())
    )
