# %% [markdown]
# # Stochastic travel times with soft time windows (Taş et al.)
#
# **Paper.** D. Taş, N. Dellaert, T. van Woensel and T. de Kok, *Vehicle Routing Problem with
# Stochastic Travel Times Including Soft Time Windows and Service Costs*, Beta working paper 364,
# Eindhoven, 2012, Sections 3 and 5. This example reproduces the problem **setting** with the
# preset `tas2012-vrpsttw`, not the tabu search.
#
# **Problem.** Vehicles serve Solomon's customers with soft time windows. Travel times are random:
# the time to drive one unit of distance is Gamma(alpha, lambda) with mean 1, so an arc of length d
# takes Gamma(alpha d, lambda), on average d. Service starts when the vehicle arrives (no waiting);
# serving early or late is penalized. The carrier pays for distance, vehicles and drivers'
# overtime beyond a shift of 480.
#
# | Paper | FleetGym |
# | --- | --- |
# | Solomon instances, capacity 200, shift w = 480 | the instances (`instances.py`) |
# | T per unit distance ~ Gamma(alpha, lambda); (alpha, lambda) = (16, 0.0625), (1, 1), (0.0625, 16): CV 0.25, 1, 4 | instance `distributions.travel_time` (`gamma_per_distance`), revealed on arrival |
# | soft windows, service on arrival, earliness and delay penalized | `time_windows: {mode: soft, early: serve}` |
# | overtime beyond w | `route_duration: {mode: soft}` with `max_route_duration = 480` |
# | costs (c_d, c_e, c_t, c_f, c_o) = (1, 0.1, 1, 400, 5/6) | `objective: {lateness: 1, earliness: 0.1, distance: 1, vehicles: 400, overtime: 5/6}` |
#
# The paper evaluates expected costs of a priori routes analytically; here each episode draws the
# travel times, so averages over episodes estimate those expectations for any policy.

# %%
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))  # this folder's instances.py

import instances  # noqa: E402
import numpy as np  # noqa: E402

import fleetgym as fg  # noqa: E402
from fleetgym.baselines import CheapestInsertion, NearestNeighbor, run_episode  # noqa: E402
from fleetgym.testing.validator import validate  # noqa: E402

print(inspect.getsource(instances.tas_instance))
NAMES = ["r101", "rc101", "c101", "r201"]
instances.write_tas("instances", NAMES)

# %% [markdown]
# ## Travel times
#
# The expected travel time of an arc is its distance; its realized time is known when the vehicle
# arrives. The spread grows with the coefficient of variation.

# %%
for cv in instances.CV:
    env = fg.make_preset("tas2012-vrpsttw", instances="instances", instance="r101", cv=cv)
    env.reset(seed=0)
    assert env.scenario is not None
    n = env.base_instance.n_nodes
    times = env.scenario["travel_time"].reshape(n, n)[0, 1:]
    expected = env.simulator.network.time_matrix()[0, 1:]
    ratio = times / expected
    print(
        f"CV {cv:>4}: realized / expected travel time from the depot: mean {ratio.mean():.2f}, std {ratio.std():.2f}"
    )

# %% [markdown]
# ## Cost components under variability
#
# The same policies at the three coefficients of variation: transport costs (distance, vehicles,
# overtime) and service costs (earliness, delay). Every episode is validated.


# %%
def costs(name: str, cv: str, policy, episodes: int = 10) -> dict[str, float]:
    """Mean cost components of ``policy`` over ``episodes`` episodes."""
    env = fg.make_preset("tas2012-vrpsttw", instances="instances", instance=name, cv=cv)
    rows = []
    for seed in range(episodes):
        result = run_episode(env, policy, seed=seed)
        assert validate(
            env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario
        ).ok
        rows.append(result.costs)
    keys = ("distance", "vehicles", "overtime", "earliness", "lateness")
    return {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in keys}


for name in NAMES:
    for cv in instances.CV:
        c = costs(name, cv, CheapestInsertion())
        transport = c["distance"] + 400 * c["vehicles"] + 5 / 6 * c["overtime"]
        service = c["lateness"] + 0.1 * c["earliness"]
        print(
            f"{name:6s} CV {cv:>4}: transport {transport:8.1f}  service {service:8.1f}  vehicles {c['vehicles']:4.1f}"
        )

# %% [markdown]
# Service costs grow with the variability of travel times while the routes stay the same; a policy
# that anticipates the variability (the paper's tabu search, or a learned policy) trades distance for
# reliability. Nearest neighbor for comparison on one instance:

# %%
print(costs("r101", "1", NearestNeighbor()))
