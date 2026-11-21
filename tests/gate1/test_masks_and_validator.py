"""Gate 1: masks agree with an independent oracle and the validator accepts every rollout."""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import fleetgym as fg
from fleetgym.envs import FleetGymEnv
from fleetgym.testing.strategies import problems, random_masked_rollout
from fleetgym.testing.validator import validate
from tests.families import make_env
from tests.gate1.oracle import Rules, route_feasible
from tests.registered import registered


def check_masks_during_rollout(env: FleetGymEnv, seed: int) -> None:
    """Every epoch: customer c is unmasked iff 'go to c now, then return' is feasible (oracle)."""
    _, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    instance = env.problem.instance
    rules = Rules.from_spec(env.problem_spec, instance)
    terminated = False
    while not terminated and info["vehicle"] is not None:
        sim = env.simulator
        v = sim.state.vehicles[info["vehicle"]]
        route = v.route if v.route else [0]
        mask = env.action_masks()
        for c in range(1, instance.n_nodes):
            pending = sim.state.request_status[c] == 0
            expected = pending and route_feasible(instance, rules, [*route, c, 0])
            assert bool(mask[c]) == expected, (c, route, mask)
        _, _, terminated, _, info = env.step(int(rng.choice(np.flatnonzero(mask))))
    if not terminated:
        # nothing could be decided at reset: the single final step (action 0) ends the episode
        assert env.action_masks().tolist()[:1] == [True]
        assert not env.action_masks()[1:].any()
        _, _, terminated, _, _ = env.step(0)
    assert terminated


@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(problem=problems(), seed=st.integers(0, 2**31))
def test_masks_equal_one_step_feasibility(problem, seed):
    spec, instance = problem
    check_masks_during_rollout(FleetGymEnv(spec, instance=instance), seed)


@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(problem=problems(), seed=st.integers(0, 2**31))
def test_validator_accepts_random_problem_rollouts(problem, seed):
    spec, instance = problem
    env = FleetGymEnv(spec, instance=instance)
    random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
    validate(env.event_log, spec, instance).raise_if_invalid()


@pytest.mark.parametrize("env_id", registered("single"))
def test_validator_accepts_registered_config_rollouts(env_id: str):
    env = make_env(env_id)
    for seed in range(10):
        random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
        validate(
            env.event_log, env.problem_spec, env.problem.instance, scenario=env.scenario
        ).raise_if_invalid()


@pytest.mark.slow
def test_gate1_ten_thousand_rollouts():
    """Gate 1: 10,000 random masked rollouts across configs and random problems, all valid."""
    count = 0
    for env_id in fg.REGISTRY.ids(agents="single"):
        env = make_env(env_id)
        for seed in range(2_000):
            random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
            validate(
                env.event_log, env.problem_spec, env.problem.instance, scenario=env.scenario
            ).raise_if_invalid()
            count += 1

    @settings(
        max_examples=6_000,
        deadline=None,
        database=None,
        suppress_health_check=tuple(HealthCheck),  # pyright: ignore[reportArgumentType]
    )
    @given(problem=problems(max_customers=8), seed=st.integers(0, 2**31))
    def random_problems(problem, seed):
        nonlocal count
        spec, instance = problem
        env = FleetGymEnv(spec, instance=instance)
        check_masks_during_rollout(env, seed)
        validate(env.event_log, spec, instance).raise_if_invalid()
        count += 1

    random_problems()
    assert count >= 10_000
