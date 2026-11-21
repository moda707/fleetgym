"""Split recourse, restock trips, duration limits and the served-demand revenue (ADR 0021).

Hand-made line instance: depot at x = 0, customers at x = 1, 2, 3; capacity 4; speed 1, so travel
times equal distances. Scenarios are fixed, so every number below is computed by hand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import fleetgym as fg
from fleetgym.data import Instance
from fleetgym.envs.env import FleetGymEnv
from fleetgym.errors import NotInScopeError
from fleetgym.eventlog import EventLog
from fleetgym.spec import load_spec
from fleetgym.testing.non_anticipativity import check_non_anticipativity
from fleetgym.testing.validator import validate
from fleetgym.uncertainty.scenario import Scenario
from tests.families import spec_for

LINE = Instance(
    name="line",
    source="test",
    demands=np.array([0.0, 2.0, 2.0, 2.0]),
    coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
    capacity=4.0,
)


def spec(*, vehicles: int = 2, duration: float | None = None, end_early: bool = False, **extra):
    constraints: dict[str, Any] = {"capacity": {}}
    if duration is not None:
        constraints["route_duration"] = {"max": duration}
    raw = {
        "fleet": {
            "sizing": "fixed",
            "types": [{"name": "v", "capacity": "from_instance", "count": vehicles}],
            "multi_trip": True,
            "end_route": "anytime" if end_early else "when_nothing_reachable",
        },
        "network": {"type": "euclidean", "rounding": "none"},
        "constraints": constraints,
        "uncertainty": {
            "demand": {"dist": "poisson", "mean": "from_instance", "revealed": "on_arrival"}
        },
        "recourse": {"capacity_failure": "split"},
        "decisions": {"interface": "next_node", "epochs": "on_event"},
        "objective": {"served_demand": -1.0},
    }
    raw.update(extra)
    return load_spec(raw)


def start(s, demands):
    env = FleetGymEnv(s, instance=LINE, debug=True)
    scenario = Scenario(s.content_hash(), LINE.content_hash(), {"demand": np.array(demands, float)})
    obs, info = env.reset(seed=0, options={"scenario": scenario})
    return env, obs, info


def valid(env) -> None:
    report = validate(env.event_log, env.problem_spec, LINE, scenario=env.scenario)
    assert report.ok, report.violations


def test_split_leaves_the_rest_open_for_another_vehicle():
    s = spec()
    env, obs, info = start(s, [0, 6, 1, 1])
    assert info["vehicle"] == 0
    obs, *_, info = env.step(1)  # vehicle 0 to customer 1
    assert info["vehicle"] == 1
    obs, *_, info = env.step(2)  # vehicle 1 to customer 2
    # t = 1: vehicle 0 serves 4 of 6 at customer 1; 2 stay open for any vehicle.
    assert info["vehicle"] == 0 and env.simulator.state.time == pytest.approx(1.0)
    failure = env.event_log.of_kind("failure")[0]
    assert (failure["node"], failure["quantity"], failure["component"]) == (1, 2.0, "split")
    assert np.flatnonzero(env.action_masks()).tolist() == [0]  # full: only the depot
    assert obs["nodes"][1, 2] == pytest.approx(2.0)  # the open demand, not the revealed 6
    obs, *_, info = env.step(0)  # restock trip
    # t = 2: vehicle 0 is back and restocked (decides first: ties go by index); it may take the
    # rest of customer 1 or customer 3. It takes 3, leaving customer 1 to vehicle 1.
    assert info["vehicle"] == 0 and env.event_log.of_kind("restock")[0]["time"] == 2.0
    assert env.action_masks()[1] and env.action_masks()[3] and not env.action_masks()[0]
    obs, *_, info = env.step(3)
    # Vehicle 1 at customer 2 (served 1, load 1) takes the open 2 of customer 1.
    assert info["vehicle"] == 1
    assert np.flatnonzero(env.action_masks()).tolist() == [0, 1]
    obs, reward, *_, info = env.step(1)
    assert reward == pytest.approx(2.0)  # the open 2 delivered at t = 3, before the next epoch
    served = [
        (r["vehicle"], r["node"], r["quantity"]) for r in env.event_log.of_kind("service_start")
    ]
    assert served[:2] == [(0, 1, 4.0), (1, 2, 1.0)]
    total = 0.0
    while True:
        valid_actions = np.flatnonzero(env.action_masks())
        action = int(valid_actions[1]) if valid_actions.size > 1 else int(valid_actions[0])
        _, reward, terminated, _, _ = env.step(action)
        total += float(reward)
        if terminated:
            break
    valid(env)
    costs = env.simulator.state.total_costs
    assert costs["served_demand"] == pytest.approx(8.0)  # everything delivered
    assert "unserved" not in costs
    rewards = sum(r["reward"] for r in env.event_log.of_kind("epoch"))
    assert rewards == pytest.approx(8.0)
    restocks = env.event_log.of_kind("restock")
    assert restocks and restocks[0]["vehicle"] == 0


def test_any_depot_trip_restocks_and_end_early_controls_ending():
    for end_early in (False, True):
        env, *_ = start(spec(vehicles=1, end_early=end_early), [0, 1, 1, 1])
        assert env.action_masks()[0] == end_early  # customers are reachable
        env.step(1)
        env.step(0)  # back to the depot with load 1: a restock, not the end of the route
        assert [r["kind"] for r in env.event_log.rows if r["kind"] in ("restock", "route_end")] == [
            "restock"
        ]
        assert env.simulator.state.vehicles[0].load == 0.0
        assert env.action_masks()[0] == end_early
        assert env.action_masks()[2] and env.action_masks()[3]
        if end_early:
            *_, terminated, _, _ = env.step(0)
            assert terminated
            assert env.simulator.state.total_costs["unserved"] == 2.0
        valid(env) if end_early else None


def test_duration_limit_masks_customers_that_cannot_return_in_time():
    # Limit 5: from the depot, customer 3 needs 3 + 3 = 6 > 5.
    env, *_ = start(spec(vehicles=1, duration=5.0), [0, 1, 1, 1])
    assert np.flatnonzero(env.action_masks()).tolist() == [1, 2]
    env.step(2)  # t = 2 at customer 2: customer 1 (back at 2 + 1 + 1 = 4) fits, 3 does not
    assert np.flatnonzero(env.action_masks()).tolist() == [0, 1]
    env.step(1)
    *_, terminated, _, _ = env.step(0)  # at the depot at t = 4: nothing reachable, the day ends
    assert terminated
    valid(env)
    (end,) = env.event_log.of_kind("route_end")
    assert end["value"] == pytest.approx(4.0)
    assert env.simulator.state.total_costs["unserved"] == 1.0


def test_a_partly_served_customer_can_end_unserved():
    # One vehicle, limit 3.5: it serves 4 of customer 1's 6 and restocks at t = 2; going back to
    # customer 1 would return at t = 4 > 3.5, so the open 2 stay unserved.
    env, *_ = start(spec(vehicles=1, duration=3.5), [0, 6, 0, 0])
    env.step(1)
    env.step(0)
    while not env.simulator.state.terminated:  # customers 2 and 3 (demand 0) still fit in time
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    valid(env)
    assert 1 in [r["node"] for r in env.event_log.of_kind("unserved")]
    assert env.simulator.state.request_status[1] == 3  # UNSERVED, after a partial service
    assert env.simulator.state.total_costs["served_demand"] == pytest.approx(4.0)


def test_rules_that_force_travel_still_reject_duration_limits():
    raw = spec().to_dict()
    raw["recourse"] = {"capacity_failure": "return_to_depot"}
    raw["constraints"] = {"capacity": {}, "route_duration": {"max": 9.0}}
    with pytest.raises(NotInScopeError, match="only rules that never force travel"):
        FleetGymEnv(load_spec(raw), instance=LINE).reset(seed=0)


def test_split_rollouts_are_non_anticipative_and_deterministic():
    s = spec(duration=9.0)
    check_non_anticipativity(FleetGymEnv(s, instance=LINE), seed=3, n_forks=3)
    digests = set()
    for _ in range(2):
        env = FleetGymEnv(s, instance=LINE)
        env.reset(seed=11)
        while not env.simulator.state.terminated:
            env.step(int(np.flatnonzero(env.action_masks())[-1]))
        digests.add(env.event_log.digest())
    assert len(digests) == 1


# ------------------------------------------------------------------ the validator


def run_reference():
    env, *_ = start(spec(), [0, 6, 1, 1])
    for action in (1, 2, 0, 1):
        env.step(action)
    while not env.simulator.state.terminated:
        env.step(int(np.flatnonzero(env.action_masks())[-1]))
    valid(env)
    return env


def codes_after(mutate) -> set[str]:
    env = run_reference()
    rows = [dict(r) for r in env.event_log.rows]
    mutate(rows)
    log = EventLog(dict(env.event_log.metadata))
    log.rows = rows
    return {v.code for v in validate(log, env.problem_spec, LINE, scenario=env.scenario).violations}


def first(rows, kind, **match):
    return next(
        r for r in rows if r["kind"] == kind and all(r.get(k) == v for k, v in match.items())
    )


def test_validator_catches_split_violations():
    def full_vehicle_to_customer(rows):
        depart = first(rows, "depart", vehicle=0, from_node=1)
        depart.update(node=3, value=2.0)

    assert "recourse" in codes_after(full_vehicle_to_customer)

    def no_restock(rows):
        rows[:] = [r for r in rows if r["kind"] != "restock"]

    assert "depart" in codes_after(no_restock) or "restock" in codes_after(no_restock)

    def wrong_revenue(rows):
        first(rows, "cost", component="served_demand").update(amount=5.0)

    assert "cost" in codes_after(wrong_revenue)

    def split_with_room(rows):
        first(rows, "failure").update(load=1.0)

    assert "load" in codes_after(split_with_room)


# ------------------------------------------------------------------ reload time (fleet)


def test_reload_time_comes_from_the_instance_and_delays_departure():
    timed = Instance(
        name="line-reload",
        source="test",
        demands=LINE.demands,
        coords=LINE.coords,
        capacity=4.0,
        extra={"reload_time": 2.5},
    )
    raw = spec(vehicles=1).to_dict()
    raw["fleet"]["reload_time"] = "from_instance"
    s = load_spec(raw)
    env = FleetGymEnv(s, instance=timed, debug=True)
    scenario = Scenario(
        s.content_hash(), timed.content_hash(), {"demand": np.array([0, 1, 1, 1.0])}
    )
    env.reset(seed=0, options={"scenario": scenario})
    env.step(1)
    env.step(0)  # back at t = 2; the 2.5 reload is paid only if it leaves again
    (restock,) = env.event_log.of_kind("restock")
    assert (restock["time"], restock["value"]) == (2.0, 2.5)
    assert env.simulator.state.time == pytest.approx(2.0)
    env.step(2)
    departures = [r["time"] for r in env.event_log.of_kind("depart")]
    assert departures[-1] == pytest.approx(4.5)
    while not env.simulator.state.terminated:
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    report = validate(env.event_log, s, timed, scenario=scenario)
    assert report.ok, report.violations
    rows = [dict(r) for r in env.event_log.rows]
    next(r for r in rows if r["kind"] == "restock")["value"] = 0.0
    log = EventLog(dict(env.event_log.metadata))
    log.rows = rows
    assert "restock" in {v.code for v in validate(log, s, timed, scenario=scenario).violations}


def test_planned_routing_rejects_fleet_behaviors():
    raw = spec_for("DVRP-v1").to_dict()
    raw["fleet"]["multi_trip"] = True
    with pytest.raises(NotInScopeError, match=r"fleet\.multi_trip"):
        fg.make(load_spec(raw)).reset(seed=0)


def test_soft_duration_limit_charges_overtime_instead_of_masking():
    raw = spec(vehicles=1).to_dict()
    raw["constraints"]["route_duration"] = {"mode": "soft", "max": 3.0}
    raw["objective"] = {"served_demand": -1.0, "overtime": 2.0}
    s = load_spec(raw)
    env, *_ = start(s, [0, 1, 1, 1])
    assert np.flatnonzero(env.action_masks()).tolist() == [1, 2, 3]  # nothing masked
    for node in (3, 2, 1, 0):
        env.step(node)
    valid(env)
    costs = env.simulator.state.total_costs
    assert costs["overtime"] == pytest.approx(6.0 - 3.0)  # route 0-3-2-1-0 lasts 6
    rewards = sum(r["reward"] for r in env.event_log.of_kind("epoch"))
    assert rewards == pytest.approx(3.0 - 2.0 * 3.0)
