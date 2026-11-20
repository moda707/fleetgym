"""Property tests: random dynamic problems never crash, keep invariants and validate."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fleetgym.envs.env import FleetGymEnv
from fleetgym.testing.non_anticipativity import check_non_anticipativity
from fleetgym.testing.strategies import dynamic_problems, random_masked_rollout
from fleetgym.testing.validator import validate


@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(problem=dynamic_problems(), seed=st.integers(0, 2**31))
def test_random_dynamic_rollouts_validate_and_repeat(problem, seed):
    spec, instance = problem
    env = FleetGymEnv(spec, instance=instance, debug=True)
    random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
    report = validate(env.event_log, spec, instance, scenario=env.scenario)
    assert report.ok, report.violations[:5]
    digest = env.event_log.digest()
    random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
    assert env.event_log.digest() == digest


@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(problem=dynamic_problems(), seed=st.integers(0, 2**31))
def test_random_dynamic_problems_are_non_anticipative(problem, seed):
    spec, instance = problem
    check_non_anticipativity(FleetGymEnv(spec, instance=instance), seed=seed, n_forks=2)
