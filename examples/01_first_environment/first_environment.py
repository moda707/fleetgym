# %% [markdown]
# # Your first FleetGym environment
#
# A FleetGym problem has two parts: the **model** (a registered family such as `CVRP-v1`) says
# what exists and how it behaves; the **instance** carries every number (customers, demands, the
# vehicle capacity). FleetGym ships no instances, so this notebook writes a few first, in the
# FleetGym format, then makes an environment, steps it with valid actions, and re-checks the
# episode with the independent validator.

# %%
from pathlib import Path

import numpy as np

import fleetgym as fg
from fleetgym.data import Instance, read_instance, write_instance_set
from fleetgym.testing.validator import validate

# %% [markdown]
# ## 1. Instances
#
# An instance is data checked against an instance schema (`fleetgym/core/1` here): nodes (depot
# first) with coordinates and demands, and the vehicle types. This function draws random ones;
# your own data would be written the same way.


# %%
def make_instances(n_instances: int, n_customers: int = 20, seed: int = 0) -> list[Instance]:
    """Customers uniform in a 100 x 100 square, depot in the middle, demands 1..9, capacity 50."""
    rng = np.random.default_rng(seed)
    instances = []
    for k in range(n_instances):
        coords = np.vstack([[50.0, 50.0], rng.uniform(0, 100, size=(n_customers, 2))])
        demands = np.concatenate([[0.0], rng.integers(1, 10, size=n_customers)])
        instances.append(
            Instance(
                name=f"uniform-{k}",
                source="examples/01",
                demands=demands,
                coords=coords,
                vehicle_types=({"capacity": 50.0},),
            )
        )
    return instances


folder = Path("instances") / "cvrp"
data = write_instance_set(make_instances(5), folder, name="first instances", overwrite=True)
print(len(data), "instances in", data.root, "; schema", data.schema)
print(read_instance(data.paths[0]))

# %% [markdown]
# ## 2. An environment
#
# The model plus the instances. Each episode draws one instance from the set; the seed fixes
# which, and every random draw.

# %%
env = fg.make(
    {"extends": "CVRP-v1", "instance_source": {"type": "files", "paths": [str(data.root)]}}
)
obs, info = env.reset(seed=0)
print("observation keys:", sorted(obs))
print("valid actions now:", np.flatnonzero(info["action_mask"]))

# %% [markdown]
# Play a random valid next node until the episode ends. The reward is minus the cost of each step.

# %%
rng = np.random.default_rng(0)
total = 0.0
while True:
    action = int(rng.choice(np.flatnonzero(env.action_masks())))
    obs, reward, terminated, truncated, info = env.step(action)
    total += float(reward)
    if terminated or truncated:
        break
print("episode return:", total)
print("cost by component:", env.simulator.state.total_costs)

# %% [markdown]
# Every episode can be re-checked independently from its event log.

# %%
report = validate(env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario)
print("valid:", report.ok)
