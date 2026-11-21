# %% [markdown]
# # VRPTW on Solomon's instances (Solomon 1987)
#
# **Paper.** M. M. Solomon, *Algorithms for the vehicle routing and scheduling problems with time
# window constraints*, Operations Research 35(2), 1987. Its 56 instances with 100 customers are the
# classical VRPTW benchmark; FleetGym uses the files and best-known solutions published by SINTEF
# (https://www.sintef.no/projectweb/top/vrptw/), downloaded and checked against SHA-256 checksums.
#
# **Problem.** Vehicles of capacity 200 (R101) leave the depot, serve every customer within its
# hard time window (waiting allowed when early, a service time at each customer), and return
# before the depot closes. The objective is hierarchical, as in the SINTEF tables: first the number
# of vehicles, then the total distance (Euclidean, not rounded, travel time = distance).
#
# | Benchmark | FleetGym (`VRPTW-v0`) |
# | --- | --- |
# | hard windows, waiting allowed, depot due time | `time_windows: {soft: false, waiting: allowed}`, `route_duration: depot_due_time` |
# | vehicles first, then distance | `objective: {lexicographic: [vehicles, distance]}` |
# | Euclidean, double precision | `network: {type: euclidean, rounding: none}` |
#
# Replaying a best-known solution through the environment must reproduce its published cost: the
# Gate 1 test suite does this for all 56 instances.

# %%
import fleetgym as fg
from fleetgym.baselines import NearestNeighbor, run_episode
from fleetgym.data import read_instance, read_solution
from fleetgym.data.download import fetch, load_collection
from fleetgym.testing.replication import replay_solution

NAME = "r101"
published = load_collection("solomon/100").info(NAME)
instance = read_instance(fetch("solomon/100", NAME), fmt="solomon")
solution = read_solution(fetch("solomon/100", NAME, kind="solution"))
print(instance.name, instance.n_customers, "customers, capacity", instance.capacity)
print("published:", published["vehicles"], "vehicles, distance", published["cost"])

# %% [markdown]
# ## Replay the best-known solution

# %%
spec = fg.load_spec("VRPTW-v0")
distance = replay_solution(spec, instance, solution)
vehicles = replay_solution(spec, instance, solution, component="vehicles")
assert distance.validation is not None
assert distance.validation.ok
print(f"replayed: {vehicles.cost:.0f} vehicles, distance {distance.cost:.2f}")
assert round(distance.cost or 0.0, 2) == published["cost"]

# %% [markdown]
# ## A baseline on the same instance

# %%
env = fg.make("VRPTW-v0", instance=instance)
result = run_episode(env, NearestNeighbor(), seed=0)
print(
    f"nearest neighbor: {result.costs['vehicles']:.0f} vehicles, distance {result.costs['distance']:.2f}"
)
