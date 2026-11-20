"""Planned routing (accept_reject, assign): masks, vehicle behavior, validator, forks."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import fleetgym as fg
from fleetgym.data import Instance
from fleetgym.envs.env import FleetGymEnv
from fleetgym.errors import ConfigError, InfeasibleActionError, NotInScopeError
from fleetgym.sim.decisions import Accept, Dispatch
from fleetgym.spec import load_spec
from fleetgym.testing.non_anticipativity import check_non_anticipativity
from fleetgym.testing.strategies import instances, random_masked_rollout
from fleetgym.testing.validator import validate
from fleetgym.uncertainty.scenario import Scenario
from tests.gate1.oracle import Rules, route_feasible


def static_spec(interface="accept_reject", count=2, **extra):
    return load_spec(
        {
            "fleet": {
                "sizing": "fixed",
                "types": [{"name": "v", "capacity": "from_instance", "count": count}],
            },
            "network": {"type": "euclidean", "rounding": "none"},
            "constraints": {"capacity": {}, "time_windows": {}, "route_duration": {"max": 200.0}},
            "decisions": {
                "epochs": "on_event",
                "interface": interface,
                "routing": "cheapest_insertion",
            },
            "objective": {"distance": 1.0, "rejected": 100.0},
            **extra,
        }
    )


@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(
    instance=instances(max_customers=6, time_windows=True),
    seed=st.integers(0, 2**31),
    count=st.integers(1, 3),
)
def test_accept_mask_equals_the_oracle_at_the_start(instance, seed, count):
    spec = static_spec(count=count)
    env = FleetGymEnv(spec, instance=instance, debug=True)
    env.reset(seed=seed)
    rules = Rules.from_spec(spec, instance)
    rng = np.random.default_rng(seed)
    while env._epoch is not None:  # pyright: ignore[reportPrivateUsage]
        epoch = env._epoch  # pyright: ignore[reportPrivateUsage]
        assert epoch.kind == "request" and epoch.time == 0.0
        node = epoch.node
        assert node is not None
        plans = [v.plan for v in env.simulator.state.vehicles]
        feasible = any(
            route_feasible(instance, rules, [0, *p[:pos], node, *p[pos:], 0])
            for p in plans
            for pos in range(len(p) + 1)
        )
        mask = env.action_masks()
        assert mask[1] == feasible and mask[0] and not mask[2]
        action = int(rng.choice(np.flatnonzero(mask)))
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    report = validate(env.event_log, spec, instance)
    assert report.ok, report.violations[:5]


LINE = Instance(
    name="line",
    source="test",
    demands=np.array([0.0, 1.0, 1.0, 1.0]),
    coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]]),
    capacity=2.0,
    time_windows=np.array([[0.0, 100.0], [0.0, 100.0], [0.0, 100.0], [0.0, 100.0]]),
)


def test_drive_first_insertion_and_capacity_rejection():
    spec = static_spec(count=1)
    env = FleetGymEnv(spec, instance=LINE, debug=True)
    _, info = env.reset(seed=0)
    assert info["epoch_kind"] == "request" and info["vehicle"] is None
    env.step(1)  # accept 1
    env.step(1)  # accept 2 (inserted after 1: cheapest)
    _, _, _, _, info = env.step(0) if not env.action_masks()[1] else env.step(1)
    sim = env.simulator
    assert [r["node"] for r in env.event_log.of_kind("accept")] == [1, 2]
    assert env.event_log.of_kind("reject")[0]["node"] == 3  # capacity 2 is full
    assert sim.state.terminated
    route = [r["node"] for r in env.event_log.of_kind("depart")]
    assert route == [2, 1, 0]  # both positions add 2; ties go to the earliest position
    assert validate(env.event_log, spec, LINE).ok
    with pytest.raises(InfeasibleActionError):
        sim.apply(Accept(1))


def test_assign_masks_vehicles_and_rejects_with_the_last_action():
    spec = static_spec("assign", count=2)
    env = FleetGymEnv(spec, instance=LINE)
    env.reset(seed=0)
    assert env.action_space.n == 3  # pyright: ignore[reportAttributeAccessIssue]
    assert env.action_masks().tolist() == [True, True, True]
    env.step(1)  # request 1 -> vehicle 1
    env.step(1)  # request 2 -> vehicle 1 (full now)
    assert env.action_masks().tolist() == [True, False, True]
    _, _, terminated, _, info = env.step(2)  # reject request 3
    assert terminated
    assert {r["node"]: r["vehicle"] for r in env.event_log.of_kind("accept")} == {1: 1, 2: 1}
    assert validate(env.event_log, spec, LINE).ok
    repaired = FleetGymEnv(spec, instance=LINE, invalid_action="repair")
    repaired.reset(seed=0)
    repaired.step(1)
    repaired.step(1)
    _, _, _, _, info = repaired.step(1)  # vehicle 1 is full: cheapest feasible vehicle instead
    assert info["repaired_action"] == 0


def test_planned_routing_scope_checks():
    with pytest.raises(ConfigError, match="on_event"):
        FleetGymEnv(
            static_spec(
                decisions={
                    "epochs": "sequential",
                    "interface": "accept_reject",
                    "routing": "cheapest_insertion",
                }
            ),
            instance=LINE,
        ).reset(seed=0)
    unlimited = load_spec(
        {
            **static_spec().to_dict(),
            "fleet": {"sizing": "unlimited", "types": [{"name": "v", "capacity": 2}]},
        }
    )
    with pytest.raises(ConfigError, match="fixed or limited"):
        FleetGymEnv(unlimited, instance=LINE).reset(seed=0)
    no_reject = load_spec({**static_spec().to_dict(), "objective": {"distance": 1.0}})
    with pytest.raises(ConfigError, match="rejected"):
        FleetGymEnv(no_reject, instance=LINE).reset(seed=0)
    dyn_next = {
        "extends": "DVRP-StochReq-v0",
        "decisions": {"epochs": "on_event", "interface": "next_node"},
    }
    with pytest.raises(NotInScopeError, match="accept_reject or assign"):
        fg.make(dyn_next)
    env = FleetGymEnv(static_spec(), instance=LINE)
    env.reset(seed=0)
    with pytest.raises(InfeasibleActionError, match="vehicle decision"):
        env.simulator.apply(Dispatch(0, 1))
    with pytest.raises(InfeasibleActionError, match="no vehicle decisions"):
        env.simulator.mask(0)


def test_outsourcing_option():
    spec = load_spec(
        {
            **static_spec().to_dict(),
            "constraints": {"capacity": {}, "outsourcing": {"cost": "per_request", "value": 7.5}},
            "objective": {"distance": 1.0, "outsourcing": 1.0},
        }
    )
    env = FleetGymEnv(spec, instance=LINE)
    env.reset(seed=0)
    assert env.action_masks().tolist() == [True, True, True]
    _, _, _, _, info = env.step(2)
    assert info["costs"]["outsourcing"] == 7.5
    while not env.simulator.state.terminated:
        env.step(1 if env.action_masks()[1] else 2)
    assert validate(env.event_log, spec, LINE).ok


DYN = {
    "extends": "DVRP-StochReq-v0",
    "dynamics": {
        "arrivals": {"process": "poisson", "rate": 0.15},
        "cutoff": 300,
        "max_requests": 40,
        "overflow": "reject",
        "request_class": {
            "location": {"dist": "kde", "data": "from_instance"},
            "demand": {"dist": "categorical", "values": [5, 10, 20], "probs": [0.5, 0.3, 0.2]},
            "time_window_width": {"dist": "uniform", "low": 60, "high": 180},
        },
        "cancellations": {"dist": "bernoulli", "p": 0.4},
    },
}


@pytest.mark.parametrize("interface", ["accept_reject", "assign"])
def test_dynamic_rollouts_validate_and_replay(interface):
    spec = {
        **DYN,
        "decisions": {
            "epochs": "on_event",
            "interface": interface,
            "routing": "cheapest_insertion",
        },
    }
    env = fg.make(spec, debug=True)
    kinds: dict[str, int] = {}
    for seed in range(12):
        random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
        report = validate(env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario)
        assert report.ok, report.violations[:5]
        assert validate(
            env.event_log, env.problem_spec, env.problem.instance, scenario=env.scenario
        ).ok
        for row in env.event_log.rows:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    assert kinds["cancel"] > 0 and kinds["accept"] > 0 and kinds["reject"] > 0
    scenario, rows = env.scenario, env.event_log.rows
    assert scenario is not None
    replay = fg.make(spec, instance=env.base_instance)
    replay.reset(seed=123, options={"scenario": scenario})
    rng = np.random.default_rng(11)
    while True:
        _, _, terminated, _, _ = replay.step(int(rng.choice(np.flatnonzero(replay.action_masks()))))
        if terminated:
            break
    assert replay.event_log.rows == rows


def test_unreleased_requests_are_hidden_and_forks_redraw_the_future():
    env = fg.make(DYN)
    obs, _ = env.reset(seed=4)
    status = env.simulator.state.request_status
    hidden = np.flatnonzero(status == 5)
    assert hidden.size > 0
    assert np.all(obs["node_valid"][hidden] == 0) and np.all(obs["nodes"][hidden] == 0)
    for _ in range(25):
        env.step(int(np.flatnonzero(env.action_masks())[-1]))
    now = env.simulator.state.time
    truth = env.scenario
    assert truth is not None
    forks = [env.fork(seed=k) for k in range(5)]
    released = truth["request.release"] <= now
    for f in forks:
        assert f.scenario is not None
        np.testing.assert_array_equal(
            f.scenario["request.release"][released], truth["request.release"][released]
        )
        np.testing.assert_array_equal(
            f.scenario["request.x"][released], truth["request.x"][released]
        )
        future = f.scenario["request.release"][~released]
        assert np.all(future[np.isfinite(future)] > now)
    assert any(
        not np.array_equal(f.scenario["request.release"], truth["request.release"])
        for f in forks
        if f.scenario
    )
    assert env.clone().scenario == truth
    for f in forks:
        random_masked_rollout_from_here(f)


def random_masked_rollout_from_here(env):
    rng = np.random.default_rng(0)
    while True:
        _, _, terminated, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        if terminated:
            return


@pytest.mark.parametrize("seed", [0, 1])
def test_dynamic_configs_are_non_anticipative(seed):
    assert check_non_anticipativity(fg.make(DYN), seed=seed, n_forks=2) > 0
    assert check_non_anticipativity(fg.make("DVRP-StochReq-v0"), seed=seed, n_forks=1) > 0


def test_validator_catches_dynamic_corruptions():
    spec = {
        **DYN,
        "decisions": {
            "epochs": "on_event",
            "interface": "accept_reject",
            "routing": "cheapest_insertion",
        },
    }
    env = fg.make(spec)
    random_masked_rollout(env, np.random.default_rng(2), seed=2)
    base, scenario, s = env.base_instance, env.scenario, env.problem_spec

    def codes(mutate):
        log = env.event_log.copy()
        mutate(log.rows)
        return {v.code for v in validate(log, s, base, scenario=scenario).violations}

    def first(rows, kind, **match):
        return next(
            r for r in rows if r["kind"] == kind and all(r.get(k) == v for k, v in match.items())
        )

    dynamic_request = next(
        r for r in env.event_log.rows if r["kind"] == "request" and r["time"] > 0
    )
    assert "request" in codes(
        lambda rows: first(rows, "request", node=dynamic_request["node"]).update(time=-1.0)
    )
    accepted = first(env.event_log.rows, "accept")
    assert "assignment" in codes(
        lambda rows: first(rows, "accept", node=accepted["node"]).update(
            vehicle=99 if False else (accepted["vehicle"] + 1) % 4
        )
    )
    assert "request" in codes(
        lambda rows: rows.insert(rows.index(first(rows, "accept")) + 1, dict(first(rows, "accept")))
    )
    assert "request" in codes(
        lambda rows: rows.remove(first(rows, "request", node=dynamic_request["node"]))
    )
    if env.event_log.of_kind("cancel"):
        assert "cancel" in codes(lambda rows: first(rows, "cancel").update(time=0.5))
    other = (
        Scenario(
            scenario.spec_hash,
            scenario.instance_hash,
            {**scenario.values, "request.overflow": np.array([1.0])},
        )
        if scenario
        else None
    )
    assert "identity" in {
        v.code for v in validate(env.event_log, s, base, scenario=other).violations
    }
    with pytest.raises(ValueError, match="scenario"):
        validate(env.event_log, s, base)
