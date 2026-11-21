# %% [markdown]
# # Dynamic routing with stochastic service requests (Ulmer et al.)
#
# **Paper.** M. W. Ulmer, J. C. Goodson, D. C. Mattfeld and M. Hennig, *Offline-Online Approximate
# Dynamic Programming for Dynamic Vehicle Routing with Stochastic Requests* (working paper, August
# 2016), Sections 2 and 6.1 and the Appendix. This example reproduces the problem **setting**, the
# VRPSSR, with the preset `ulmer2016-vrpssr` (family `DVRP-v1`), not the paper's methods or results.
#
# **Problem.** One uncapacitated vehicle starts at a central depot and must be back by T = 360
# minutes. Early-request customers are known at the start and must all be served. Late requests
# arrive during the day; at each arrival of the vehicle at a location, the requests made since the
# previous arrival are accepted or rejected for good. Accepted requests must be served. The
# objective is the expected number of late requests served.
#
# | Paper | FleetGym |
# | --- | --- |
# | 15 km (small) or 20 km (large) square area, central depot | the instances (`instances.py`) |
# | late requests: Poisson, lambda per T - 1 minutes in [1, T], lambda in {25, 50, 75} | instance `distributions.arrivals` (`nhpp` profile) |
# | early requests: Poisson(100 - lambda), known at time 0, all served | one realization per instance file; `dynamics.initial_requests: mandatory` |
# | early requests that do not fit by T: all served late, no late request confirmed (Ulmer, Mattfeld and Koester 2017) | `dynamics.initial_infeasible: serve_all` |
# | locations uniform, or normal clusters (2 or 3, std 1 km, scaled 0.75 for small) | instance `distributions.request.location` (`uniform`, `normal_mixture`) |
# | travel time: Euclidean km at 25 km/h, rounded up to whole minutes | `network: {speed: 25/60, time_rounding: ceil}` |
# | decide at the vehicle's arrival on the requests made since the last one | `decisions.request_epochs: on_vehicle_arrival` |
# | routing by cheapest insertion (the paper's reduced decision set, Appendix) | `decisions.routing: cheapest_insertion` |
# | return to the depot by T | the depot's time window [0, 360] |
# | maximize late requests served | `objective: {rejected: 1}` (every accepted request is served) |
#
# Differences, stated in the preset's notes: requests of one arrival are decided one by one in
# release order (the paper decides on the set jointly); a vehicle with nothing planned waits where
# it is and decides on requests as they arrive (in the paper's MDP waiting is a one-minute
# decision). For early customers that do not fit by T, this paper only assumes they do; the
# preset uses the rule the same authors state for the same problem in *Budgeting Time for Dynamic
# Vehicle Routing with Stochastic Customer Requests* (Ulmer, Mattfeld and Koester, Transportation
# Science 2017, Section 3.1).

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

print(inspect.getsource(instances.ulmer_instance))
sets = instances.write_ulmer("instances", n_instances=10, seed=0)

# %% [markdown]
# Each file is one day's early customers; the late requests are drawn per episode from the
# distributions the file carries.

# %%
preset = fg.load_preset("ulmer2016-vrpssr")
print(len(preset.points()), "settings; default", preset.point())
env = fg.make_preset("ulmer2016-vrpssr", instances="instances")  # large, high lambda, two clusters
env.reset(seed=0)
day = env.base_instance
print(day.name, "early customers:", day.n_customers)
profile = day.distributions["arrivals"]["profile"]
print("late requests: rate", profile["rates"][1], "per minute on", profile["breaks"][1:])
where = day.distributions["request.location"]
print("locations:", where["dist"], [list(c) for c in where["centers"]], "std", where["std"])
print("travel 0 -> 1:", env.simulator.network.time_matrix()[0, 1], "minutes (whole minutes)")

# %% [markdown]
# ## One day
#
# The myopic policy accepts every request that cheapest insertion can still fit (`NearestNeighbor`
# does this at request epochs). Requests wait for the vehicle's next arrival before they are
# decided.


# %%
def late_served(env) -> tuple[int, int]:
    """Late requests released and accepted (every accepted request is served)."""
    log = env.event_log
    late = {row["node"] for row in log.of_kind("request") if row["time"] > 0}
    accepted = {row["node"] for row in log.of_kind("accept")} & late
    return len(late), len(accepted)


result = run_episode(env, NearestNeighbor(), seed=0)
log = env.event_log
assert validate(log, env.problem_spec, env.base_instance, scenario=env.scenario).ok
released = {row["node"]: row["time"] for row in log.of_kind("request") if row["time"] > 0}
waited = [
    row["time"] - released[row["node"]] for row in log.of_kind("accept") if row["node"] in released
]
print("late requests released, served:", late_served(env))
print(f"a request waited {np.mean(waited):.1f} minutes on average for its decision")

# %% [markdown]
# ## The 18 settings
#
# Late requests served on average by the myopic and the random policy (myopic / random), over the
# instances and scenarios of each setting.

# %%
for area in ("small", "large"):
    for lam in ("low", "moderate", "high"):
        row = []
        for locations in ("uniform", "two_clusters", "three_clusters"):
            env = fg.make_preset(
                "ulmer2016-vrpssr", instances="instances", area=area, lam=lam, locations=locations
            )
            served = {}
            for policy in (NearestNeighbor(), RandomPolicy()):
                counts = []
                for seed in range(5):
                    run_episode(env, policy, seed=seed)
                    counts.append(late_served(env)[1])
                served[policy.name] = np.mean(counts)
            row.append(f"{locations}: {served['nearest']:5.1f} / {served['random']:5.1f}")
        print(f"{area:5} {lam:8}  " + "  ".join(row))

# %% [markdown]
# A rough check of the setting, not a replication: for uniform locations the myopic numbers are
# close to the paper's myopic policy (Table 2, row pi_m: small area 11.8, 25.3, 40.5; large area
# 0.2, 8.1, 27.7 for low, moderate and high lambda). With a large area and low lambda most days
# cannot even serve the early customers by T, so almost no late request is confirmed, as in the
# paper. Clustered locations: the numbers here are higher; the paper's policies work in a reduced decision set without waiting (Appendix), while
# here a vehicle with nothing planned waits for requests, which matters most when requests are
# concentrated and the vehicle runs out of work.
