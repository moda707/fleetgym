# %% [markdown]
# # VRP with stochastic demands and a duration limit (Goodson et al.)
#
# **Papers.** The setting of J. C. Goodson, B. W. Thomas and J. W. Ohlmann (2016), *Restocking-based
# rollout policies for the vehicle routing problem with stochastic demand and duration limits*
# (Transportation Science), as used in Section 6.2 and Appendix B.1.2 of M. Dastpak, F. Errico and
# O. Jabali, arXiv:2109.10200. This example reproduces the problem **setting**, not the methods or
# results.
#
# **Problem.** The customers are fixed: the first 75 customers of Solomon's R101, whose demands are
# the expected demands. Eleven vehicles of capacity `Q` serve them before the duration limit `L`.
# Actual demands are revealed on arrival: a multiple of the expected demand whose spread is the
# stochastic variability `U`. As in the VRP-VCSD, vehicles split deliveries, restock whenever they
# choose, and the objective is the expected served demand.
#
# | Paper (Appendix B.1.2) | FleetGym (`VRPSD-Goodson-v0`) |
# | --- | --- |
# | R101, first 75 customers, 11 vehicles | instance source `collection` (`first_customers: 75`) |
# | `L` in {103.05, 171.75, 240.45} | `route_duration.max` (default Medium) |
# | `Q` in {25, 50, 75} | vehicle capacity (default 50) |
# | `U` in {Low, Moderate, High}: factors of the expected demand | `scaled_categorical` (default Moderate) |
#
# Not stated in the paper's text, and assumed here: Euclidean distances without rounding, no service
# times and no time windows. The R101 file is downloaded from SINTEF on first use and checked
# against a SHA-256 checksum.

# %%
import numpy as np

import fleetgym as fg
from fleetgym.baselines import NearestNeighbor, RandomPolicy, run_episode
from fleetgym.testing.validator import validate

env = fg.make("VRPSD-Goodson-v0")
env.reset(seed=0)
instance = env.base_instance
print(instance.name, "customers:", instance.n_customers, "expected demand:", instance.demands.sum())

# %% [markdown]
# The paper reports a total expected demand of 1079 for these 75 customers; the trimmed instance
# matches it.

# %%
assert instance.demands.sum() == 1079.0

seeds = range(10)
for policy in (RandomPolicy(), NearestNeighbor()):
    served = []
    for seed in seeds:
        result = run_episode(env, policy, seed=seed)
        assert validate(env.event_log, env.problem_spec, instance, scenario=env.scenario).ok
        served.append(result.costs.get("served_demand", 0.0))
    print(f"{policy.name:>8}: served {np.mean(served):7.1f} of 1079 expected")

# %% [markdown]
# ## The 27 instances
#
# Each combination of `L`, `Q` and `U` is an override of the same config.

# %%
LIMITS = {"Short": 103.05, "Medium": 171.75, "Long": 240.45}
VARIABILITY = {  # factors of the expected demand and their probabilities
    "Low": ([0.5, 1.0, 1.5], [0.05, 0.9, 0.05]),
    "Moderate": ([0.0, 0.5, 1.0, 1.5, 2.0], [0.05, 0.15, 0.6, 0.15, 0.05]),
    "High": ([0.0, 0.5, 1.0, 1.5, 2.0], [0.2, 0.2, 0.2, 0.2, 0.2]),
}


def goodson(limit: str, capacity: int, variability: str) -> fg.ProblemSpec:
    """The paper's instance for a duration limit, a capacity and a variability."""
    raw = fg.load_spec("VRPSD-Goodson-v0").to_dict()
    factors, probs = VARIABILITY[variability]
    raw["constraints"]["route_duration"]["max"] = LIMITS[limit]
    raw["fleet"]["types"][0]["capacity"] = capacity
    raw["uncertainty"]["demand"].update(factors=factors, probs=probs)
    return fg.load_spec(raw)


for limit in LIMITS:
    row = []
    for variability in VARIABILITY:
        variant = fg.make(goodson(limit, 50, variability), instance=instance)
        result = run_episode(variant, NearestNeighbor(), seed=0)
        row.append(f"U={variability}: {result.costs.get('served_demand', 0):6.1f}")
    print(f"L={limit:>6}, Q=50  " + "  ".join(row))
