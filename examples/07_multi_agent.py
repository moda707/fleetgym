"""Multi-agent routing with PettingZoo: one agent per vehicle, all acting at once.

Run: python examples/07_multi_agent.py
"""

import numpy as np

import fleetgym as fg

env = fg.make_multi("MA-CVRP-v0", api="parallel")
observations, infos = env.reset(seed=0)
print("agents:", env.agents)
total = 0.0
while env.agents:
    actions = {}
    for agent in env.agents:
        mask = observations[agent]["action_mask"]
        actions[agent] = int(np.flatnonzero(mask)[-1])  # busy vehicles: only the no-op is valid
    observations, rewards, terminations, truncations, infos = env.step(actions)
    total += sum(rewards.values())
print("conflicts:", len(env.env.event_log.of_kind("conflict")))
print("global state for a centralized critic:", env.state().shape)
