# %% [markdown]
# # Multi-agent CVRP with a heterogeneous fleet (Vera and Abad 2019)
#
# **Paper.** J. M. Vera and A. G. Abad, *Deep Reinforcement Learning for Routing a Heterogeneous
# Fleet of Vehicles*, arXiv:1912.03341 (Sections 3.1, 4 and Table 1). This example reproduces the
# problem **setting** with the multi-agent API (PettingZoo AEC), not the method or its results.
#
# **Problem.** A fixed fleet of vehicles with different capacities serves customers from one
# depot, each vehicle a separate agent. Depot and customers are uniform in the unit square. When a
# vehicle's load runs out it returns to the depot to refill and continues. The objective is the
# total route length of all vehicles, serving every customer. Instance VRP20 (Table 1): 20
# customers, 3 vehicles of capacities 20, 30 and 35.
#
# | Paper | FleetGym (preset `vera2019-hetfleet`) |
# | --- | --- |
# | 3 vehicles, capacities 20, 30, 35 | the instance: three vehicle types with `count: 1` |
# | depot and customers uniform in the unit square, demands 1-9 | the instances (`instances.py`) |
# | refill at the depot and continue | `fleet.multi_trip: true` |
# | total length, all customers served | `objective: {distance: 1, unserved: 100}` |
# | one vehicle acts at a time | PettingZoo AEC |
#
# Differences: the paper's text says demands are uniform on 11 to 99 while its Algorithm 1 uses 1 to
# 9; 1 to 9 is the one consistent with the capacities. Its vehicles act in a fixed alternating
# order at every step; here the vehicle that becomes free first acts (ties by index).

# %%
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))  # this folder's instances.py

import instances  # noqa: E402
import numpy as np  # noqa: E402

import fleetgym as fg  # noqa: E402
from fleetgym.testing.validator import validate  # noqa: E402

print(inspect.getsource(instances.vera_instance))
instances.write_vera("instances", n_instances=5, seed=0)
env = fg.make_preset("vera2019-hetfleet", instances="instances", size="vrp20")
env.reset(seed=0)
print("agents:", env.agents)

# %% [markdown]
# ## One episode with a nearest-customer rule per agent
#
# Each agent sees its own observation and action mask; a full vehicle can only go to the depot.

# %%


def nearest_action(observation) -> int:
    """The nearest valid customer of this agent, else the depot."""
    nodes, mask = observation["nodes"], observation["action_mask"]
    valid = np.flatnonzero(mask)
    customers = valid[(valid != 0) & (valid < nodes.shape[0])]
    if customers.size == 0:
        return int(valid[0])
    here = observation["vehicle"][:2]
    return int(customers[np.argmin(np.hypot(*(nodes[customers, :2] - here).T))])


total = 0.0
for _agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    total += reward
    env.step(None if terminated or truncated else nearest_action(observation))
single = env.env  # the single-agent environment behind the PettingZoo API
costs = single.simulator.state.total_costs
print(
    f"total length {costs['distance']:.3f}; unserved {costs.get('unserved', 0):.0f}; "
    f"refills {len(single.event_log.of_kind('restock'))}"
)
print("valid:", validate(single.event_log, single.problem_spec, single.problem.instance).ok)
