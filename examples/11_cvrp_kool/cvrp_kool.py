# %% [markdown]
# # CVRP as in learning-based routing (Kool et al. 2019)
#
# **Paper.** W. Kool, H. van Hoof and M. Welling, *Attention, Learn to Solve Routing Problems!*,
# ICLR 2019, arXiv:1803.08475 (Appendix C.1). Their instance distribution, taken from Nazari et
# al. (2018), is the standard benchmark of neural VRP solvers. This example reproduces the problem
# **setting** with the preset `kool2019-cvrp`, not the model or its results.
#
# **Problem.** One depot and `n` customers, all uniform in the unit square; integer demands uniform
# on {1, ..., 9}; capacity 30, 40 or 50 for n = 20, 50 or 100. Routes start and end at the depot and
# carry at most the capacity; the objective is the total length, with any number of routes.
#
# | Paper | FleetGym |
# | --- | --- |
# | depot and customers uniform in the unit square, demands 1..9, capacity by n | the instances (`instances.py`) |
# | total route length, any number of routes | `CVRP-v1`: distance objective, unlimited fleet |
# | Euclidean distance (implied; not stated in the paper) | no rounding (the instances carry no `edge_weight_type`) |
# | demands normalized by the capacity | not needed: the same instance in other units |
# | the depot is never visited twice in a row | a route cannot be empty (same effect) |

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

print(inspect.getsource(instances.kool_instance))
sets = instances.write_kool("instances", n_instances=50, seed=0)
print({n: len(s) for n, s in sets.items()})

# %% [markdown]
# ## Baselines on held-out instances
#
# Each episode draws an instance from the set. Learned policies are trained on this distribution
# and evaluated on other instances (the paper uses 10,000 test instances).

# %%
for n in (20, 50, 100):
    env = fg.make_preset("kool2019-cvrp", instances="instances", n=n)
    for policy in (RandomPolicy(), NearestNeighbor()):
        lengths = [run_episode(env, policy, seed=s).costs["distance"] for s in range(20)]
        print(f"n={n:3d} {policy.name:>8}: mean length {np.mean(lengths):6.2f}")

# %% [markdown]
# The observation is a padded graph (`obs["nodes"]`: coordinates, demand, flags) with an action
# mask, so the attention models of `fleetgym.nn` (or your own) plug in directly; see the training
# example (`06_train_agent`).
