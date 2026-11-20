"""Stochastic demands with recourse: scenarios, forecasts, replay and non-anticipative forks.

Run: python examples/02_stochastic_demands.py
"""

import fleetgym as fg
from fleetgym.baselines import NearestNeighbor, Rollout, run_episode

env = fg.make("VRPSD-v0")  # Poisson demands, revealed on arrival, return-to-depot recourse
obs, info = env.reset(seed=4)
scenario = env.scenario  # every random value of the episode, sampled once
print("scenario id:", scenario.id[:16], "sources:", scenario.sources)

# Before a visit the agent sees the forecast (the mean), never the realized demand.
print("known demand flags:", env.simulator.info.demand_known[:6])

# Policies play through one protocol: reset(env), act(obs, info).
nearest = run_episode(env, NearestNeighbor(), seed=4).cost
rollout = run_episode(env, Rollout(NearestNeighbor, n_samples=2), seed=4).cost
print(f"nearest neighbor: {nearest:.1f}   rollout over forks: {rollout:.1f}")
print("restock trips:", len(env.event_log.of_kind("restock")))

# The same scenario can be replayed exactly (common random numbers for fair comparisons).
replay = fg.make("VRPSD-v0", instance=env.base_instance)
replay.reset(seed=123, options={"scenario": scenario})
print("replayed scenario:", replay.scenario == scenario)

# Planners must use fork(seed): it keeps what is known and redraws the hidden future.
obs, info = env.reset(seed=4)
fork = env.fork(seed=1)
print("fork shares the past, not the future:", fork.scenario != env.scenario)
