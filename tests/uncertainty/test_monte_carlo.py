"""Monte Carlo sanity check: Poisson demands, sampled scenarios vs the exact dynamic program."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from fleetgym.baselines.apriori import AprioriRoute
from fleetgym.data import Instance
from fleetgym.envs.env import FleetGymEnv
from fleetgym.spec import load_spec
from tests.uncertainty.vrpsd_dp import apriori_expected_cost

COORDS = np.array([[0, 0], [4, 3], [8, 0], [5, -6], [-3, -4], [-7, 1], [-2, 6]], dtype=float)
MEANS = np.array([0.0, 1.5, 2.5, 1.0, 3.0, 2.0, 0.5])
Q = 6.0
ORDER = (1, 2, 3, 4, 5, 6)
N = 4000
Z = stats.norm.ppf(1 - 0.001 / 2)  # 99.9% two-sided


def test_poisson_mean_cost_is_within_the_confidence_interval():
    instance = Instance(
        name="mc",
        source="test",
        demands=MEANS,
        coords=COORDS,
        capacity=Q,
        distributions={"demand": {"dist": "poisson", "mean": "demand"}},
    )
    spec = load_spec({"extends": "SVRP-v1", "network": {"type": "euclidean", "rounding": "none"}})
    env = FleetGymEnv(spec, instance=instance)
    policy = AprioriRoute(ORDER)
    costs = np.array([policy.run(env, seed=seed) for seed in range(N)])

    def support(node: int):
        top = int(stats.poisson.ppf(1 - 1e-15, MEANS[node])) + 1
        values = np.arange(top + 1, dtype=float)
        return values, stats.poisson.pmf(values, MEANS[node])

    def distance(i: int, j: int) -> float:
        return math.hypot(*(COORDS[i] - COORDS[j]))

    exact = apriori_expected_cost(distance, ORDER, Q, support)
    half_width = Z * costs.std(ddof=1) / math.sqrt(N)
    assert abs(costs.mean() - exact) <= half_width, (costs.mean(), exact, half_width)
