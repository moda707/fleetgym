# %% [markdown]
# # Hard time windows with stochastic service times (Errico et al.)
#
# **Paper.** F. Errico, G. Desaulniers, M. Gendreau, W. Rei and L.-M. Rousseau, *The vehicle
# routing problem with hard time windows and stochastic service times*, EURO Journal on
# Transportation and Logistics 7 (2018) 223-251, Sections 3 and 5.1. This example reproduces the
# problem **setting** with the preset `errico2018-vrptwst`, not the branch-price-and-cut method.
#
# **Problem.** Technicians visit customers with hard time windows: service starts inside the
# window, a vehicle may wait if it arrives early, and it may never start after the window closes.
# Service times are random; the depot has no time window. A plan is a set of routes fixed in
# advance; it succeeds when every route meets all its windows once the service times are known.
# The paper minimizes the distance subject to a success probability of at least alpha.
#
# | Paper (Section 5.1) | FleetGym |
# | --- | --- |
# | Solomon instances, first 25 (all) or 50 (suffix 1) customers, no demands or capacity | the instances (`instances.py`), `capacity: null` |
# | discrete triangular service times on a 0.1 grid (four families) | instance `distributions.service_time` (`discrete_triangular`) |
# | hard windows, waiting allowed, no depot window | `time_windows: {mode: hard, depot: ignore}` |
# | route success: no customer reached after its window (eq. 10) | a customer that can no longer be reached in time is skipped and ends `unserved` |
# | minimize distance s.t. P(success) >= alpha | `objective: {distance: 1, unserved: 1000}`; success rate reported per plan |
#
# In FleetGym a vehicle picks its next stop after the current service ends, so it knows when it
# would arrive: a customer it can no longer serve in time is skipped. For a priori routes, a route
# fails exactly when one of its customers is skipped, so the share of episodes without skipped
# customers estimates the paper's success probability.

# %%
import dataclasses
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))  # this folder's instances.py

import instances  # noqa: E402
import numpy as np  # noqa: E402

import fleetgym as fg  # noqa: E402
from fleetgym.baselines import CheapestInsertion, run_episode  # noqa: E402
from fleetgym.baselines.protocol import PolicyBase, valid_actions  # noqa: E402
from fleetgym.testing.validator import validate  # noqa: E402

print(inspect.getsource(instances.errico_instance))
POINTS = ["r101-25", "rc101-25", "c101-25", "r201-25"]
instances.write_errico("instances", POINTS, families=["basic", "large_support"])

# %% [markdown]
# ## A priori plans from deterministic models
#
# The paper compares its stochastic plans with deterministic VRPTW plans that use the median or
# the worst-case service time (Tables 8 to 11). We build both kinds of plan with cheapest
# insertion on the deterministic instance, then run each plan as fixed routes under the random
# service times.


# %%
def deterministic_plan(instance: fg.data.Instance, service: str) -> list[list[int]]:
    """Routes of cheapest insertion with every service time at its median or worst case."""
    params = instance.distributions["service_time"]
    value = float(params["mode"] if service == "median" else params["high"])
    times = np.where(np.arange(instance.n_nodes) == 0, 0.0, value)
    fixed = dataclasses.replace(
        instance, service_times=times, distributions={}, vehicle_types=({"name": "technician"},)
    )
    model = {
        k: v
        for k, v in fg.load_preset("errico2018-vrptwst").spec_data.items()
        if k != "uncertainty"
    }
    # routes are built one after the other, with as many vehicles as needed
    model |= {
        "fleet": {"sizing": "unlimited"},
        "decisions": {"interface": "next_node", "epochs": "sequential"},
    }
    env = fg.make(fg.load_spec(model), instance=fixed)
    run_episode(env, CheapestInsertion(), seed=0)
    routes: dict[int, list[int]] = {}
    for row in env.event_log.of_kind("depart"):
        if row["node"] != 0:
            routes.setdefault(row["vehicle"], []).append(row["node"])
    return list(routes.values())


class FixedRoutes(PolicyBase):
    """Vehicle k follows route k, skipping customers it can no longer serve in time."""

    name = "fixed_routes"

    def __init__(self, routes: list[list[int]]) -> None:
        super().__init__()
        self.routes = routes

    def reset(self, env):
        """Start every route from its first customer."""
        super().reset(env)
        self.todo = [list(r) for r in self.routes]

    def act(self, obs, info):
        """The next planned customer that is still allowed, else the depot."""
        epoch = self.env.epoch
        valid = set(valid_actions(info).tolist())
        todo = self.todo[epoch.vehicle] if epoch.vehicle < len(self.todo) else []
        while todo:
            node = todo.pop(0)
            if node in valid:
                return node
        return 0


def evaluate_plan(env, routes, episodes=200):
    """Success rate (no skipped customer) and mean distance over random service times."""
    successes, distances = 0, []
    for seed in range(episodes):
        run_episode(env, FixedRoutes(routes), seed=seed)
        costs = env.simulator.state.total_costs
        successes += costs.get("unserved", 0.0) == 0
        distances.append(costs["distance"])
    assert validate(env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario).ok
    return successes / episodes, float(np.mean(distances))


# %%
for family in ("basic", "large_support"):
    print(f"{family} (alpha = {instances.ALPHA[family]})")
    for point in POINTS:
        env = fg.make_preset(
            "errico2018-vrptwst", instances="instances", family=family, instance=point
        )
        env.reset(seed=0)
        rows = []
        for service in ("median", "worst"):
            plan = deterministic_plan(env.base_instance, service)
            success, distance = evaluate_plan(env, plan)
            rows.append(
                f"{service}: {len(plan)} routes, distance {distance:6.1f}, success {success:5.1%}"
            )
        print(f"  {point:9s} " + " | ".join(rows))

# %% [markdown]
# As in the paper (Tables 8 to 11), plans built on median service times are shorter but fail
# often, more so with the large support, while worst-case plans succeed at a higher distance; the
# paper's stochastic plans sit between the two, at the success probability alpha. The paper finds
# wide windows (suffix 2) barely affected; the cheapest-insertion plan for r201 above is not the
# paper's optimal plan and fails more often, which shows how much the success rate depends on the
# plan. A learned or rollout policy may also re-plan after each service (the environment allows
# it); the a priori plans above do not.
