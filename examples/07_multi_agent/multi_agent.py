# %% [markdown]
# # Multi-agent routing
#
# PettingZoo Parallel API: one agent per vehicle, all deciding at once. Any family becomes
# multi-agent with `agents.mode: multi`; the instances give the fleet (here 4 vehicles). Busy
# vehicles send the no-op, the last action.

# %%
from pathlib import Path

import numpy as np

import fleetgym as fg
from fleetgym.data import Instance, write_instance_set


# %%
def make_instances(n_instances: int, seed: int = 0) -> list[Instance]:
    """20 uniform customers, demands 1..9, 4 vehicles of capacity 50."""
    rng = np.random.default_rng(seed)
    return [
        Instance(
            name=f"fleet-{k}",
            source="examples/07",
            demands=np.concatenate([[0.0], rng.integers(1, 10, size=20)]),
            coords=np.vstack([[50.0, 50.0], rng.uniform(0, 100, size=(20, 2))]),
            vehicle_types=({"capacity": 50.0, "count": 4},),
        )
        for k in range(n_instances)
    ]


data = write_instance_set(
    make_instances(5), Path("instances") / "fleet", name="fleet", overwrite=True
)
spec = fg.load_spec(
    {
        "extends": "CVRP-v1",
        "instance_source": {"type": "files", "paths": [str(data.root)]},
        "fleet": {"sizing": "fixed"},
        "decisions": {"interface": "next_node", "epochs": "on_event"},
        "objective": {"distance": 1.0, "unserved": 100.0},
        "agents": {"mode": "multi", "api": "parallel", "reward": "global", "conflict": "priority"},
    }
)
env = fg.make_multi(spec, api="parallel")
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
