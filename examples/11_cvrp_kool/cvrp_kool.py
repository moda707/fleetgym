# %% [markdown]
# # CVRP as in learning-based routing (Kool et al. 2019)
#
# **Paper.** W. Kool, H. van Hoof and M. Welling, *Attention, Learn to Solve Routing Problems!*,
# ICLR 2019, arXiv:1803.08475 (Appendix C.1). Their instance distribution, taken from Nazari et
# al. (2018), is the standard benchmark of neural VRP solvers. This example reproduces the problem
# **setting** with `CVRP-v0`, not the model or its results.
#
# **Problem.** One depot and `n` customers, all uniform in the unit square; integer demands
# uniform on {1, ..., 9}; capacity 30, 40 or 50 for n = 20, 50 or 100. Routes start and end at the
# depot and carry at most the capacity; the objective is the total length. There is no fleet limit
# (one vehicle making as many routes as needed).
#
# | Paper | FleetGym |
# | --- | --- |
# | depot and customers uniform in the unit square | generator `uniform`, `size: 1`, `depot: random` |
# | demands uniform on {1..9}, capacity 30 / 40 / 50 | `demand: {low: 1, high: 9}`, `capacity` |
# | demands normalized by the capacity | not needed: the same instance in other units |
# | total route length, any number of routes | `objective: {distance: 1}`, `fleet.sizing: unlimited` |
# | Euclidean distance (implied; not stated in the paper) | `network: {type: euclidean, rounding: none}` |
# | the depot is never visited twice in a row | a route cannot be empty (same effect) |

# %%
import numpy as np

import fleetgym as fg
from fleetgym.baselines import NearestNeighbor, RandomPolicy, run_episode

CAPACITY = {20: 30, 50: 40, 100: 50}  # paper, Appendix C.1


def kool_spec(n: int) -> fg.ProblemSpec:
    """CVRP-v0 with the instance distribution of Kool et al. for ``n`` customers."""
    return fg.load_spec(
        {
            "extends": "CVRP-v0",
            "instance_source": {
                "type": "generator",
                "name": "uniform",
                "params": {
                    "n_customers": n,
                    "size": 1.0,
                    "depot": "random",
                    "demand": {"low": 1, "high": 9},
                    "capacity": CAPACITY[n],
                },
            },
            "network": {"type": "euclidean", "rounding": "none"},
        }
    )


env = fg.make(kool_spec(20))
obs, info = env.reset(seed=0)
instance = env.base_instance
print(instance.n_customers, "customers, capacity", instance.capacity)
print("demands:", instance.demands[1:].astype(int))

# %% [markdown]
# ## Baselines on fresh instances
#
# Each seed draws a new instance. Learned policies are trained on this distribution and evaluated
# on held-out instances (the paper uses 10,000 test instances).

# %%
for n in (20, 50, 100):
    env = fg.make(kool_spec(n))
    for policy in (RandomPolicy(), NearestNeighbor()):
        lengths = [run_episode(env, policy, seed=s).costs["distance"] for s in range(50)]
        routes = len(env.event_log.of_kind("route_end"))
        print(
            f"n={n:3d} {policy.name:>8}: mean length {np.mean(lengths):6.2f} ({routes} routes in the last)"
        )

# %% [markdown]
# The observation is a padded graph (`obs["nodes"]`: coordinates, demand, flags) with an action
# mask, so the attention models of `fleetgym.nn` (or your own) plug in directly; see the training
# example (`06_train_agent`).
