"""Requests arriving during the day: accept/reject with built-in routing, and next_node with wait.

Run: python examples/03_dynamic_day.py
"""

import numpy as np

import fleetgym as fg
from fleetgym.baselines import CheapestInsertion, NearestNeighbor, run_episode
from fleetgym.uncertainty.dynamism import degree_of_dynamism

# 1. The agent accepts, rejects or outsources each request; cheapest insertion routes the fleet.
env = fg.make("DVRP-StochReq-v0")
episode = run_episode(env, CheapestInsertion(), seed=2)
log = env.event_log
print(
    f"requests: {len(log.of_kind('request'))}, accepted: {len(log.of_kind('accept'))}, "
    f"rejected: {len(log.of_kind('reject'))}, cost: {episode.cost:.1f}"
)
release = env.scenario["request.release"]
print("degree of dynamism:", round(degree_of_dynamism(release, 0.0), 2))

# 2. The same day with next_node: requests are accepted automatically until the cutoff, and each
#    vehicle picks its next customer, waits for the next event, reloads at the depot or ends.
spec = {
    "extends": "DVRP-StochReq-v0",
    "dynamics": {"cutoff": 384},  # the last 20% of the 480-minute day takes no new requests
    "decisions": {"epochs": "on_event", "interface": "next_node"},
    "objective": {"distance": 1.0, "unserved": 50.0},
}
env = fg.make(spec)
obs, info = env.reset(seed=2)
print("actions:", env.action_space, "(the last one is wait)")
episode = run_episode(env, NearestNeighbor(), seed=2)
waits = sum(1 for row in env.event_log.of_kind("decision") if "node" not in row)
print(f"cost: {episode.cost:.1f}, waits: {waits}, unserved: {episode.costs.get('unserved', 0):.0f}")
assert np.isfinite(episode.cost)
