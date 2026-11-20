"""Your first FleetGym environment: make, step with masks, read costs, validate.

Run: python examples/01_first_environment.py
"""

import numpy as np

import fleetgym as fg
from fleetgym.testing.validator import validate

env = fg.make("CVRP-v0")  # one YAML config = one environment
obs, info = env.reset(seed=0)  # the seed fixes the instance and every random draw
print("observation keys:", sorted(obs))
print("valid actions now:", np.flatnonzero(info["action_mask"]))

rng = np.random.default_rng(0)
total = 0.0
while True:
    action = int(rng.choice(np.flatnonzero(env.action_masks())))  # a random valid next node
    obs, reward, terminated, truncated, info = env.step(action)
    total += float(reward)
    if terminated or truncated:
        break

print("episode return:", total)
print("cost by component:", env.simulator.state.total_costs)

# Every episode can be re-checked independently from its event log.
report = validate(env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario)
print("valid:", report.ok)
