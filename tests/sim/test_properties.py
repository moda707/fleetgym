"""Hypothesis: random static problems and random masked action sequences."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fleetgym.envs import FleetGymEnv
from fleetgym.sim.state import RequestStatus, VehicleStatus
from fleetgym.testing.strategies import problems, random_masked_rollout


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(problem=problems(), seed=st.integers(0, 2**31))
def test_random_rollouts_end_cleanly(problem, seed):
    spec, instance = problem
    env = FleetGymEnv(spec, instance=instance, debug=True)
    total = random_masked_rollout(env, np.random.default_rng(seed), seed=seed, max_steps=500)
    state = env.simulator.state
    assert state.terminated
    assert set(state.request_status[1:].tolist()) <= {RequestStatus.SERVED, RequestStatus.UNSERVED}
    assert all(
        v.status in (VehicleStatus.DONE, VehicleStatus.WAITING_TO_START) for v in state.vehicles
    )
    assert all(v.node == 0 for v in state.vehicles)
    assert total == -env.reward_model.cost(state.total_costs) or np.isclose(
        total, -env.reward_model.cost(state.total_costs), rtol=1e-12
    )
    unserved = int(np.sum(state.request_status == RequestStatus.UNSERVED))
    assert state.total_costs.get("unserved", 0.0) == unserved
    if "capacity" in spec.constraints:
        assert all(v.load <= v.capacity + 1e-9 for v in state.vehicles)
