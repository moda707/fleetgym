# %% [markdown]
# # EV shortest path with charging-station occupancy (Dastpak et al.)
#
# **Paper.** M. Dastpak, F. Errico, O. Jabali and F. Malucelli, *Dynamic Routing for the Electric
# Vehicle Shortest Path Problem with Charging Station Occupancy Information*, arXiv:2305.11773
# (2023), Sections 3 and 5.1. This example reproduces the problem **setting** with the preset
# `dastpak2023-evsppoi` (family `EVSPP-v1`), not the paper's methods or results.
#
# **Problem.** An EV drives from an origin to a far destination and must recharge on the way at
# public charging stations. Each station has one charger shared with other EVs (an M/M/1/kappa
# queue). The EV sees, in real time, only whether each station is busy (a binary occupancy
# indicator); the number of EVs waiting is known only on arrival. At every indicator change,
# arrival, head of queue or end of charging, it decides where to go (it may change its destination
# while driving) and how much to charge. The objective is the arrival time: driving, waiting and
# charging.
#
# | Paper (Sections 3 and 5.1) | FleetGym |
# | --- | --- |
# | 350 x 700 km, origin (0, 0), destination (350, 700), 100 km/h | the instances (`instances.py`), `network.speed: 100` |
# | EVs of Table 1, charging breakpoints of Table 2 | instance fields `battery_capacity`, `consumption`, `charge_level_k`, `charge_time_k` |
# | stations: densities 2, 6, 10 per 100 x 100 km; technology mixes of Table 3 | the instances, nested as in Section 5.3 |
# | M/M/1/kappa queues, mu by technology, lambda of Table 4 | instance fields `arrival_rate`, `service_rate`, `queue_capacity`; `ev.shortest_path.queues` |
# | occupancy indicators in real time, queue length seen on arrival | observation `nodes` (indicator, time since it changed), `ev` (queue seen) |
# | decisions: next node and target battery level; diversion while driving | interface `ev_route`, `decisions.commitment: diversion` |
# | minimize the trip duration | `objective: {drive: 1, wait: 1, charge: 1}` |
#
# Choices the paper leaves open, made by the user: the EV leaves fully charged, queues start in
# their steady state, and a full station cannot be joined. Target levels are a grid of B / 20 plus
# the station's charging breakpoints.

# %%
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))  # this folder's instances.py

import instances  # noqa: E402
import numpy as np  # noqa: E402

import fleetgym as fg  # noqa: E402
from fleetgym.baselines import run_episode  # noqa: E402
from fleetgym.baselines.ev import EVGreedy, EVSteadyState  # noqa: E402
from fleetgym.testing.validator import validate  # noqa: E402

print(inspect.getsource(instances.evspp_instance))
sets = instances.write_evspp("instances")  # the instances of Table 6
print(
    len(sets),
    "distinct instances (Table 6 lists 18: its baseline instance repeats in every setting)",
)

# %% [markdown]
# ## One trip
#
# The baseline instance of the paper: BMW i3s, Moderate density (147 stations), all Normal
# chargers, utilization 65%, kappa = 1.

# %%
env = fg.make_preset("dastpak2023-evsppoi", instances="instances")
obs, info = env.reset(seed=0)
print(
    "stations:", len(env.simulator.stations), "| observation:", {k: v.shape for k, v in obs.items()}
)
episode = run_episode(env, EVSteadyState(), seed=0)
assert validate(env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario).ok
kinds = {}
for row in env.event_log.rows:
    kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
print(
    {k: round(v, 2) for k, v in episode.costs.items()}, "| hours in total:", round(episode.cost, 2)
)
print(
    "charges:",
    kinds.get("charge_start", 0),
    "| indicator changes seen:",
    kinds.get("indicator", 0),
    "| decisions:",
    episode.steps,
)

# %% [markdown]
# ## Baselines on the settings of Table 6
#
# Mean hours of waiting, charging and driving over 15 days (scenarios), for a greedy policy and a
# steady-state re-optimizer like the paper's benchmark (it ignores the indicators). For reference,
# the paper's benchmark RSS reports 1:48 waiting, 4:03 charging, 8:04 driving (13:55 in total) on
# the baseline instance (Table 7, PU2); its charging amounts are optimized, while this baseline
# charges to full.


# %%
def summary(point: tuple, policy, days: int = 15) -> dict[str, float]:
    """Mean waiting, charging and driving hours of ``policy`` at one grid point."""
    kappa, ev, density, mix, utilization = point
    env = fg.make_preset(
        "dastpak2023-evsppoi",
        instances="instances",
        kappa=kappa,
        ev=ev,
        density=density,
        mix=mix,
        utilization=utilization,
    )
    rows = []
    for day in range(days):
        rows.append(run_episode(env, policy, seed=day).costs)
    return {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in ("wait", "charge", "drive")}


for point in instances.TABLE_6[:6]:
    for policy in (EVGreedy(), EVSteadyState()):
        m = summary(point, policy)
        print(
            f"{instances.folder(*point):28s} {policy.name:16s} wait {m['wait']:5.2f} charge {m['charge']:5.2f} drive {m['drive']:5.2f} total {sum(m.values()):5.2f}"
        )

# %% [markdown]
# A policy that uses the occupancy indicators (the paper's ROI, or a learned policy on the
# observation above) can avoid busy stations while driving; the environment gives it the
# indicators and lets it divert at every change.
