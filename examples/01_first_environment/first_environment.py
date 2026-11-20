# %% [markdown]
# # Your first FleetGym environment
#
# Make an environment, step it with valid actions, read the costs, and re-check the episode with
# the independent validator. Run as a script (`python first_environment.py`) or as the notebook
# next to it.

# %%
import numpy as np

import fleetgym as fg
from fleetgym.testing.validator import validate

# %% [markdown]
# One YAML config is one environment. The seed fixes the instance and every random draw.

# %%
env = fg.make("CVRP-v0")
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
