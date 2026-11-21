# %% [markdown]
# # VRP with stochastic demands on CVRPLIB instances (Florio et al. 2023)
#
# **Paper.** A. M. Florio, M. Gendreau, R. F. Hartl, S. Minner and T. Vidal, *Recent advances in
# vehicle routing with stochastic demands: Bayesian learning for correlated demands and elementary
# branch-price-and-cut*, European Journal of Operational Research 306 (2023), arXiv:2302.02538.
# Section 6.1 benchmarks the classical VRPSD on CVRPLIB instances (sets A, E, P and four X
# instances, among them X-n101-k25). This example reproduces that **setting** with the preset
# `florio2023-vrpsd` (family `SVRP-v1`), not the paper's methods or results.
#
# **Problem.** "Customer demands follow Poisson distributions with rates as given by the
# deterministic demands in the original instances" (Section 6.1); demands are revealed on arrival;
# no fleet limit and the original capacity (load factor 1). A vehicle that cannot fit a demand
# serves what fits, goes to the depot to restock and comes back (detour-to-depot recourse, which
# Section 3.1 notes all exact methods used before 2016; the paper itself optimizes the restocking
# decisions, which a FleetGym policy can do by returning to the depot early). The objective is the
# expected total distance.
#
# | Paper (Section 6.1) | FleetGym (preset `florio2023-vrpsd`) |
# | --- | --- |
# | Poisson demand, rate = the instance's demand, revealed on arrival | the preset's `uncertainty.demand: {dist: poisson, mean: from_instance}` (CVRPLIB files carry no distribution) |
# | restock at the depot, all demand served | `recourse: {capacity_failure: return_to_depot}` |
# | unlimited fleet, load factor 1 | `fleet.sizing: unlimited`, the instance's capacity |
# | CVRPLIB instance X-n101-k25 | the preset's grid: 4 X instances of Table 5, downloaded with checksums |

# %%
import numpy as np

import fleetgym as fg
from fleetgym.baselines import NearestNeighbor, run_episode
from fleetgym.baselines.protocol import PolicyBase, valid_actions
from fleetgym.data import read_instance, read_solution
from fleetgym.data.download import fetch
from fleetgym.testing.validator import validate

NAME = "X-n101-k25"
instance = read_instance(fetch("cvrplib/X", NAME))
solution = read_solution(fetch("cvrplib/X", NAME, kind="solution"))
print(instance.name, instance.n_customers, "customers, capacity", instance.capacity)
print("deterministic best-known cost:", solution.cost, "with", len(solution.routes), "routes")

# %% [markdown]
# ## A priori routes under stochastic demand
#
# Plan with the deterministic best-known routes, then drive them while demands are revealed. The
# policy below follows the planned routes; the recourse rule handles the failures. Its expected
# distance is above the deterministic cost: that gap is what VRPSD methods reduce.

# %%


class PlannedRoutes(PolicyBase):
    """Follow fixed routes; the recourse rule handles capacity failures."""

    name = "planned_routes"

    def __init__(self, routes):
        super().__init__()
        self.plan = [node for route in routes for node in (*route, 0)]

    def reset(self, env):
        """Start again from the first route."""
        super().reset(env)
        self.position = 0

    def act(self, obs, info):
        """The next planned node, skipping customers that are already handled."""
        mask = info["action_mask"]
        while self.position < len(self.plan):
            node = self.plan[self.position]
            self.position += 1
            if mask[node]:
                return node
        return int(valid_actions(info)[0])


env = fg.make_preset("florio2023-vrpsd", instance=NAME)
costs = []
for seed in range(100):
    result = run_episode(env, PlannedRoutes(solution.routes), seed=seed)
    costs.append(result.costs["distance"])
assert validate(env.event_log, env.problem_spec, instance, scenario=env.scenario).ok
print(
    f"planned routes: expected distance {np.mean(costs):.0f} +- {np.std(costs, ddof=1) / 10:.0f} "
    f"(deterministic {solution.cost:.0f}); restocks in the last episode: "
    f"{len(env.event_log.of_kind('restock'))}"
)

# %% [markdown]
# ## A dynamic policy for comparison
#
# Nearest neighbor decides after each revelation instead of following a plan.

# %%
nearest = [run_episode(env, NearestNeighbor(), seed=s).costs["distance"] for s in range(100)]
print(f"nearest neighbor: expected distance {np.mean(nearest):.0f}")
