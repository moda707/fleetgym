"""next_node with requests arriving during the day (ADR 0015): wait, reload, terminate."""

from __future__ import annotations

import numpy as np
import pytest

from fleetgym.data import Instance
from fleetgym.envs.env import FleetGymEnv
from fleetgym.errors import InfeasibleActionError
from fleetgym.sim.decisions import Wait
from fleetgym.spec import load_spec
from fleetgym.spec.models import ProblemSpec
from fleetgym.testing.validator import validate
from fleetgym.uncertainty.scenario import Scenario

# Depot at 0; one known customer at x = 1; one dynamic slot.
BASE = Instance(
    name="line",
    source="test",
    demands=np.array([0.0, 1.0]),
    coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
    capacity=2.0,
)


def spec(count: int = 2, **constraints: dict) -> ProblemSpec:
    return load_spec(
        {
            "fleet": {
                "sizing": "fixed",
                "types": [{"name": "v", "capacity": "from_instance", "count": count}],
            },
            "network": {"type": "euclidean", "rounding": "none"},
            "constraints": {"capacity": {}, **constraints},
            "dynamics": {
                "arrivals": {"process": "poisson", "rate": 0.1},
                "cutoff": 50.0,
                "max_requests": 1,
                "request_class": {
                    "location": {"dist": "uniform", "low": 0, "high": 1},
                    "demand": {"dist": "fixed", "value": 1},
                },
            },
            "decisions": {"epochs": "on_event", "interface": "next_node"},
            "objective": {"distance": 1.0, "unserved": 100.0},
        }
    )


def scenario_for(s: ProblemSpec, release: float, x: float, cancel: float = np.inf) -> Scenario:
    values = {
        "request.release": np.array([0.0, 0.0, release]),
        "request.x": np.array([0.0, 1.0, x]),
        "request.y": np.zeros(3),
        "request.demand": np.array([0.0, 1.0, 1.0]),
        "request.tw_open": np.zeros(3),
        "request.tw_close": np.zeros(3),
        "request.service_time": np.zeros(3),
        "request.cancel": np.array([np.inf, np.inf, cancel]),
        "request.overflow": np.zeros(0),
    }
    return Scenario(s.content_hash(), BASE.content_hash(), values)


def start(s: ProblemSpec, scenario: Scenario) -> FleetGymEnv:
    env = FleetGymEnv(s, instance=BASE, debug=True)
    env.reset(seed=0, options={"scenario": scenario})
    return env


WAIT = 3  # max_nodes = 2 static + 1 slot; the wait action comes last


def test_wait_until_a_request_arrives_then_serve_it():
    s = spec(count=1)
    env = start(s, scenario_for(s, release=10.0, x=4.0))
    assert env.action_space.n == 4  # pyright: ignore[reportAttributeAccessIssue]
    mask = env.action_masks()
    assert mask.tolist() == [False, True, False, True]  # customer 1 or wait; the slot is hidden
    env.step(1)  # serve the known customer
    _, _, _, _, info = env.step(WAIT)  # wait at customer 1 until the next event
    assert info["time"] == pytest.approx(10.0)  # the request arrived
    assert env.action_masks()[2]
    env.step(2)
    env.step(0)  # back to the depot to reload
    # The only vehicle may not end its route while requests can still arrive: it waits.
    assert env.action_masks().tolist() == [False, False, False, True]
    _, _, _, _, info = env.step(WAIT)
    assert info["time"] == pytest.approx(50.0)  # woken at the end of the arrival window
    assert env.simulator.state.terminated  # nothing left: the route ended automatically
    assert validate(env.event_log, s, BASE, scenario=env.scenario).ok
    assert env.simulator.state.total_costs["distance"] == pytest.approx(1 + 3 + 4)


def test_wait_wakes_at_the_end_of_the_arrival_window_and_nothing_leaks():
    s = spec(count=1)
    env = start(s, scenario_for(s, release=np.inf, x=0.0))  # no request will come
    env.step(1)
    assert env.action_masks()[WAIT]  # waiting is allowed: the window is still open
    _, _, _, _, info = env.step(WAIT)
    assert info["time"] == pytest.approx(50.0)  # woken at the cutoff, not earlier
    assert not env.action_masks()[WAIT]
    _, _, terminated, _, _ = env.step(0)
    assert terminated
    assert validate(env.event_log, s, BASE, scenario=env.scenario).ok


def test_route_duration_bounds_the_wait():
    s = spec(count=1, route_duration={"max": 20.0})
    env = start(s, scenario_for(s, release=np.inf, x=0.0))
    env.step(1)  # leaves at 0, at customer 1 at time 1
    _, _, _, _, info = env.step(WAIT)
    assert info["time"] == pytest.approx(19.0)  # latest departure: 0 + 20 - 1
    env.step(0)
    report = validate(env.event_log, s, BASE, scenario=env.scenario)
    assert report.ok, report.violations


def test_reload_at_the_depot_and_leave_again():
    s = spec(count=1)
    env = start(s, scenario_for(s, release=5.0, x=0.5))
    env.step(1)
    env.step(0)  # back to the depot to reload: the route goes on
    sim = env.simulator
    assert sim.state.vehicles[0].used and sim.state.vehicles[0].load == 0
    assert env.event_log.of_kind("restock")
    assert not env.action_masks()[0]  # the only vehicle cannot end while the window is open
    env.step(WAIT)  # wait at the depot for the request at time 5
    env.step(2)
    env.step(0)
    while not env.simulator.state.terminated:
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert validate(env.event_log, s, BASE, scenario=env.scenario).ok
    assert len(env.event_log.of_kind("route_end")) == 1


def test_terminate_needs_another_working_vehicle():
    s = spec(count=2)
    env = start(s, scenario_for(s, release=20.0, x=0.5))
    _, info = env.reset(seed=0, options={"scenario": scenario_for(s, release=20.0, x=0.5)})
    assert info["vehicle"] == 0 and env.action_masks()[0]  # vehicle 1 still works
    env.step(0)  # vehicle 0 retires
    assert info["vehicle"] == 0
    assert env.simulator.state.vehicles[0].status.name == "DONE"
    assert not env.action_masks()[0]  # vehicle 1 is the last one and work is left
    with pytest.raises(InfeasibleActionError):
        env.step(0)
    env.step(1)
    env.step(WAIT)
    env.step(2)
    env.step(0)
    while not env.simulator.state.terminated:
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert validate(env.event_log, s, BASE, scenario=env.scenario).ok


def test_cancelled_request_on_the_way():
    s = spec(count=1)
    env = start(s, scenario_for(s, release=2.0, x=0.0, cancel=3.0))
    env.step(WAIT)  # at the depot, wait for the request at time 2
    env.step(2)  # drive to it (distance 0 + ...)
    while not env.simulator.state.terminated:
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert validate(env.event_log, s, BASE, scenario=env.scenario).ok


def test_wait_is_rejected_when_not_allowed():
    s = spec(count=1)
    env = start(s, scenario_for(s, release=np.inf, x=0.0))
    env.step(1)
    env.step(WAIT)
    with pytest.raises(InfeasibleActionError, match="cannot wait"):
        env.simulator.apply(Wait(0))
