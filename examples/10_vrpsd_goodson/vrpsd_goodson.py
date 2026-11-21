# %% [markdown]
# # VRP with stochastic demands and a duration limit (Goodson et al.)
#
# **Papers.** The setting of J. C. Goodson, B. W. Thomas and J. W. Ohlmann (2016), *Restocking-based
# rollout policies for the vehicle routing problem with stochastic demand and duration limits*
# (Transportation Science), as used in Section 6.2 and Appendix B.1.2 of M. Dastpak, F. Errico and
# O. Jabali, arXiv:2109.10200. This example reproduces the problem **setting** with the preset
# `dastpak2022-goodson`, not the methods or results.
#
# **Problem.** The customers are fixed: the first 75 customers of Solomon's R101, whose demands are
# the expected demands. Eleven vehicles of capacity `Q` serve them before the duration limit `L`.
# Actual demands are revealed on arrival: a multiple of the expected demand whose spread is the
# variability `U`. As in the VRP-VCSD, vehicles split deliveries, reload whenever they choose, and
# the objective is the expected served demand.
#
# | Paper (Appendix B.1.2) | FleetGym |
# | --- | --- |
# | R101, first 75 customers, 11 vehicles | the instance (`instances.py`, from the R101 download) |
# | `L` in {103.05, 171.75, 240.45}, `Q` in {25, 50, 75} | instance fields `max_route_duration`, `vehicle_types` |
# | `U`: factors of the expected demand | the instance's `distributions.demand` (`scaled_categorical`) |
# | decisions and objective of the VRP-VCSD | the preset's model |

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

print(inspect.getsource(instances.goodson_instance))
sets = instances.write_goodson("instances")
base = instances.r101_75()
print(len(sets), "instances; expected demand of the 75 customers:", base.demands.sum())

# %% [markdown]
# The paper reports a total expected demand of 1079 for these 75 customers; the trimmed instance
# matches it.

# %%
assert base.demands.sum() == 1079.0
env = fg.make_preset("dastpak2022-goodson", instances="instances")  # medium, Q = 50, moderate
for policy in (RandomPolicy(), NearestNeighbor()):
    served = []
    for seed in range(5):
        result = run_episode(env, policy, seed=seed)
        assert validate(
            env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario
        ).ok
        served.append(result.costs.get("served_demand", 0.0))
    print(f"{policy.name:>8}: served {np.mean(served):7.1f} of 1079 expected")

# %% [markdown]
# ## The 27 instances
#
# Each (L, Q, U) is the same model on another instance.

# %%
for limit in ("short", "medium", "long"):
    row = []
    for variability in ("low", "moderate", "high"):
        env = fg.make_preset(
            "dastpak2022-goodson",
            instances="instances",
            limit=limit,
            capacity=50,
            variability=variability,
        )
        result = run_episode(env, NearestNeighbor(), seed=0)
        row.append(f"U={variability}: {result.costs.get('served_demand', 0):6.1f}")
    print(f"L={limit:>6}, Q=50  " + "  ".join(row))
