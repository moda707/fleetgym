# %% [markdown]
# # VRP with a highly variable customer basis and stochastic demands (VRP-VCSD)
#
# **Paper.** M. Dastpak, F. Errico and O. Jabali, *Off-line approximate dynamic programming for the
# vehicle routing problem with a highly variable customer basis and stochastic demands*,
# arXiv:2109.10200. This example reproduces the paper's **problem setting** in FleetGym (not its
# method or results).
#
# **Problem.** Every morning a new set of customers requests a pickup (the paper's motivation is
# donor collection). Each customer has a known expected demand; the actual demand is revealed on
# the first visit. A homogeneous fleet of `m` vehicles with capacity `Q` starts at the depot.
# Deliveries can be split: a vehicle serves what fits and the rest stays open, for the same or
# another vehicle. A vehicle may go back to the depot to empty itself whenever it wants (and must,
# when it is full), and every vehicle must be back by the duration limit `L`. The objective is to
# maximize the expected served demand.
#
# | Paper (Section 3.1, Appendix B.1.1) | FleetGym (`VRP-VCSD-v0`) |
# | --- | --- |
# | 100 x 100 area, depot at (50, 50), 5 x 5 zones, 15 active | generator `zones` |
# | customers per active zone (Moderate: 0-3 with 0.1, 0.4, 0.4, 0.1) | `customers_per_zone` |
# | expected demand uniform on {5, 10, 15} | generator `demand` |
# | actual demand uniform on expected +- 5 (+- 4 for 5), revealed on the first visit | `discrete_uniform`, `revealed: on_arrival` |
# | split service, restock any time, empty vehicles restock | recourse `split` |
# | duration limit `L`, `m` vehicles of capacity `Q` | `route_duration`, `fleet` |
# | maximize served demand, rewarded on delivery | objective `served_demand: -1` |
# | one vehicle decides at a time (decentralized) | `epochs: on_event` |
#
# The paper's Figure 4 zone layout is not in its text: the config uses a fixed random draw of 15
# zones (documented in the YAML).

# %%
import numpy as np

import fleetgym as fg
from fleetgym.baselines import NearestNeighbor, RandomPolicy, run_episode
from fleetgym.testing.validator import validate

spec = fg.load_spec("VRP-VCSD-v0")
print(spec.to_yaml())

# %% [markdown]
# ## A highly variable customer basis
#
# Each episode draws a new set of customers (the instance) and their actual demands (the scenario).

# %%
env = fg.make("VRP-VCSD-v0")
for seed in range(3):
    env.reset(seed=seed)
    instance = env.base_instance
    print(
        f"seed {seed}: {instance.n_customers} customers, expected total demand "
        f"{instance.demands.sum():.0f}, actual {env.scenario['demand'].sum():.0f}"
    )

# %% [markdown]
# ## One day, step by step
#
# The action is the next node of the vehicle that must decide; node 0 is the depot (a restock trip
# during the day). A full vehicle has only the depot in its mask. The reward is the demand
# delivered since the previous decision.

# %%
obs, info = env.reset(seed=0)
rng = np.random.default_rng(0)
served = 0.0
while True:
    mask = env.action_masks()
    action = int(rng.choice(np.flatnonzero(mask)))
    obs, reward, terminated, truncated, info = env.step(action)
    served += reward
    if terminated or truncated:
        break
log = env.event_log
print(
    f"served {served:.0f}; splits {len(log.of_kind('failure'))}; restocks {len(log.of_kind('restock'))}"
)
print("valid:", validate(log, env.problem_spec, env.base_instance, scenario=env.scenario).ok)

# %% [markdown]
# ## Baselines on common scenarios
#
# Policies follow one protocol (`reset(env)`, `act(obs, info)`); each runs on the same seeds.

# %%
seeds = range(20)
for policy in (RandomPolicy(), NearestNeighbor()):
    results = [run_episode(env, policy, seed=s) for s in seeds]
    served = np.array([r.costs.get("served_demand", 0.0) for r in results])
    print(f"{policy.name:>8}: served demand {served.mean():6.1f} +- {served.std(ddof=1):4.1f}")

# %% [markdown]
# ## The paper's other instances
#
# A density fixes the vehicle count and the duration limit; the capacity is 25, 50 or 75. Each
# combination is an override of the same config.

# %%
DENSITIES = {  # values of customers per zone, probabilities, m, L (paper Appendix B.1.1)
    "Very Low": ([0, 1, 2], [1 / 2, 1 / 3, 1 / 6], 2, 143.71),
    "Low": ([0, 1, 2], [1 / 3, 1 / 3, 1 / 3], 2, 201.38),
    "Moderate": ([0, 1, 2, 3], [0.1, 0.4, 0.4, 0.1], 3, 221.47),
    "High": ([2, 3, 4, 5], [0.1, 0.4, 0.4, 0.1], 7, 195.54),
    "Very High": ([4, 5, 6, 7], [0.1, 0.4, 0.4, 0.1], 11, 187.29),
}


def vcsd(density: str, capacity: int) -> fg.ProblemSpec:
    """The paper's instance for a density and a capacity."""
    values, probs, m, limit = DENSITIES[density]
    raw = spec.to_dict()
    raw["instance_source"]["params"]["customers_per_zone"] = {"values": values, "probs": probs}
    raw["fleet"]["types"][0].update(count=m, capacity=capacity)
    raw["constraints"]["route_duration"]["max"] = limit
    return fg.load_spec(raw)


for density in DENSITIES:
    row = []
    for capacity in (25, 50, 75):
        variant = fg.make(vcsd(density, capacity))
        result = run_episode(variant, NearestNeighbor(), seed=0)
        row.append(f"Q={capacity}: {result.costs.get('served_demand', 0):5.0f}")
    print(f"{density:>9}  " + "  ".join(row))
