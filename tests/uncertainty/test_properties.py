"""Property tests: random stochastic problems never crash, keep invariants and validate."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fleetgym.envs.env import FleetGymEnv
from fleetgym.testing.non_anticipativity import check_non_anticipativity
from fleetgym.testing.strategies import random_masked_rollout, stochastic_problems
from fleetgym.testing.validator import validate


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(problem=stochastic_problems(), seed=st.integers(0, 2**31))
def test_random_stochastic_rollouts_validate(problem, seed):
    spec, instance = problem
    env = FleetGymEnv(spec, instance=instance, debug=True)
    random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
    report = validate(env.event_log, spec, instance, scenario=env.scenario)
    assert report.ok, report.violations[:5]
    total = -sum(r["reward"] for r in env.event_log.of_kind("epoch"))
    weights = spec.objective.scalar_weights()
    recomputed = sum(weights.get(c, 0.0) * a for c, a in env.simulator.state.total_costs.items())
    assert abs(total - recomputed) <= 1e-6 * max(1.0, abs(total))


@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(problem=stochastic_problems(max_customers=4), seed=st.integers(0, 2**31))
def test_random_stochastic_problems_are_non_anticipative(problem, seed):
    spec, instance = problem
    check_non_anticipativity(FleetGymEnv(spec, instance=instance), seed=seed, n_forks=2)


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(problem=stochastic_problems(), seed=st.integers(0, 2**31))
def test_same_seed_same_stochastic_episode(problem, seed):
    spec, instance = problem
    digests = set()
    for _ in range(2):
        env = FleetGymEnv(spec, instance=instance)
        random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
        digests.add(env.event_log.digest())
    assert len(digests) == 1
