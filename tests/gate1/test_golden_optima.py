"""Gate 1: exhaustive search through the environment finds the brute-force optimum."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings

from fleetgym.envs import FleetGymEnv
from fleetgym.spec import load_spec
from fleetgym.testing.strategies import instances
from tests.gate1.oracle import Rules, brute_force, route_feasible

CVRP = {
    "fleet": {"types": [{"name": "v", "capacity": "from_instance"}]},
    "network": {"type": "euclidean", "rounding": "nearest_int"},
    "constraints": {"capacity": {}},
    "decisions": {"epochs": "sequential"},
}
VRPTW = {
    "fleet": {"types": [{"name": "v", "capacity": "from_instance"}]},
    "network": {"type": "euclidean", "rounding": "none"},
    "constraints": {
        "capacity": {},
        "time_windows": {},
        "route_duration": {"max": "depot_due_time"},
    },
    "decisions": {"epochs": "sequential"},
}


def env_search(env: FleetGymEnv) -> list[dict[str, float]]:
    """Total costs of every terminal state reachable with valid actions (depth-first)."""
    leaves: list[dict[str, float]] = []
    env.reset(seed=0)

    def explore() -> None:
        state = env.get_state()
        for action in np.flatnonzero(env.action_masks()):
            env.set_state(state)
            _, _, terminated, _, info = env.step(int(action))
            if terminated:
                leaves.append(info["total_costs"])
            else:
                explore()

    explore()
    return leaves


def individually_feasible(instance, rules) -> bool:
    return all(route_feasible(instance, rules, [0, c, 0]) for c in range(1, instance.n_nodes))


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(instance=instances(max_customers=5, time_windows=False))
@pytest.mark.parametrize("objective", [{"distance": 1.0}, {"distance": 1.0, "vehicles": 30.0}])
def test_cvrp_optimum(objective, instance):
    spec = load_spec({**CVRP, "objective": objective})
    rules = Rules.from_spec(spec, instance)
    assume(individually_feasible(instance, rules))
    weight = objective.get("vehicles", 0.0)
    expected = min(d + weight * k for k, d in brute_force(instance, rules))
    leaves = env_search(FleetGymEnv(spec, instance=instance))
    assert all(leaf.get("unserved", 0.0) == 0 for leaf in leaves)
    found = min(leaf.get("distance", 0.0) + weight * leaf.get("vehicles", 0.0) for leaf in leaves)
    assert found == pytest.approx(expected, abs=1e-9)


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(instance=instances(max_customers=5, time_windows=True))
def test_vrptw_lexicographic_optimum(instance):
    spec = load_spec({**VRPTW, "objective": {"lexicographic": ["vehicles", "distance"]}})
    rules = Rules.from_spec(spec, instance)
    assume(individually_feasible(instance, rules))
    expected = min(brute_force(instance, rules))
    leaves = env_search(FleetGymEnv(spec, instance=instance))
    assert all(leaf.get("unserved", 0.0) == 0 for leaf in leaves)
    found = min((leaf.get("vehicles", 0.0), leaf.get("distance", 0.0)) for leaf in leaves)
    assert found[0] == expected[0]
    assert found[1] == pytest.approx(expected[1], abs=1e-9)
