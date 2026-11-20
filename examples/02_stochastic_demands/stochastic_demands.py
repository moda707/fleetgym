# %% [markdown]
# # Stochastic demands with recourse
#
# Scenarios, forecasts, recourse, exact replay and non-anticipative forks on `VRPSD-v0`
# (Poisson demands revealed on arrival, return-to-depot recourse).

# %%
import fleetgym as fg
from fleetgym.baselines import NearestNeighbor, Rollout, run_episode

# %% [markdown]
# Every random value of an episode is sampled once, at reset, into a scenario. Before a visit the
# agent sees the forecast (the mean), never the realized demand.

# %%
env = fg.make("VRPSD-v0")
obs, info = env.reset(seed=4)
scenario = env.scenario
print("scenario id:", scenario.id[:16], "sources:", scenario.sources)
print("known demand flags:", env.simulator.info.demand_known[:6])

# %% [markdown]
# Policies play through one protocol: `reset(env)`, then `act(obs, info)`. Rollout simulates each
# action on forks with a base policy.

# %%
nearest = run_episode(env, NearestNeighbor(), seed=4).cost
rollout = run_episode(env, Rollout(NearestNeighbor, n_samples=2), seed=4).cost
print(f"nearest neighbor: {nearest:.1f}   rollout over forks: {rollout:.1f}")
print("restock trips:", len(env.event_log.of_kind("restock")))

# %% [markdown]
# The same scenario can be replayed exactly: common random numbers for fair comparisons.

# %%
replay = fg.make("VRPSD-v0", instance=env.base_instance)
replay.reset(seed=123, options={"scenario": scenario})
print("replayed scenario:", replay.scenario == scenario)

# %% [markdown]
# Planners must use `fork(seed)`: it keeps what is known and redraws the hidden future.

# %%
obs, info = env.reset(seed=4)
fork = env.fork(seed=1)
print("fork shares the past, not the future:", fork.scenario != env.scenario)
