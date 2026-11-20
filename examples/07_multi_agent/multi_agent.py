# %% [markdown]
# # Multi-agent routing
#
# PettingZoo Parallel API: one agent per vehicle, all deciding at once. Busy vehicles send the
# no-op, the last action.

# %%
import numpy as np

import fleetgym as fg

env = fg.make_multi("MA-CVRP-v0", api="parallel")
observations, infos = env.reset(seed=0)
print("agents:", env.agents)

# %% [markdown]
# Each agent takes its last valid action. Two vehicles choosing the same customer are resolved by
# the config's conflict rule (priority here).

# %%
total = 0.0
while env.agents:
    actions = {}
    for agent in env.agents:
        mask = observations[agent]["action_mask"]
        actions[agent] = int(np.flatnonzero(mask)[-1])
    observations, rewards, terminations, truncations, infos = env.step(actions)
    total += sum(rewards.values())
print("conflicts:", len(env.env.event_log.of_kind("conflict")))
print("global state for a centralized critic:", env.state().shape)
