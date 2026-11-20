import numpy as np
import pytest

from fleetgym.errors import ConfigError, InfeasibleActionError, NotInScopeError
from fleetgym.sim.decisions import Dispatch, EndRoute
from fleetgym.sim.events import EventKind, EventQueue
from fleetgym.sim.state import RequestStatus, SimState, VehicleStatus
from tests.sim.helpers import LINE_XS, env_for, line_instance, play

# depot at 0, customers at x = 2, 5, 9 with demands 4, 3, 5 (capacity 10)
LINE = line_instance(LINE_XS, [0, 4, 3, 5])


def test_event_queue_orders_by_time_then_priority_then_insertion():
    q = EventQueue()
    q.push(5.0, EventKind.SERVICE_END, 0, 1)
    q.push(5.0, EventKind.VEHICLE_ARRIVAL, 1, 2)
    q.push(5.0, EventKind.VEHICLE_ARRIVAL, 2, 3)
    q.push(1.0, EventKind.SERVICE_END, 3, 4)
    order = [(e.time, e.kind, e.vehicle) for e in (q.pop() for _ in range(len(q)))]
    assert order == [
        (1.0, EventKind.SERVICE_END, 3),
        (5.0, EventKind.VEHICLE_ARRIVAL, 1),
        (5.0, EventKind.VEHICLE_ARRIVAL, 2),
        (5.0, EventKind.SERVICE_END, 0),
    ]


def test_sim_state_cost_bookkeeping():
    state = SimState(
        time=0.0,
        vehicles=[],
        request_status=np.zeros(2, dtype=np.int8),
        service_start=np.zeros(2),
        served_by=np.zeros(2, dtype=np.int64),
    )
    state.add_cost("distance", 2.0)
    state.add_cost("distance", 0.0)
    state.add_cost("time", 1.0)
    assert state.take_pending_costs() == {"distance": 2.0, "time": 1.0}
    state.add_cost("distance", 1.0)
    assert state.take_pending_costs() == {"distance": 1.0}
    assert state.total_costs == {"distance": 3.0, "time": 1.0}


def test_sequential_cvrp_costs_by_hand():
    env = env_for(LINE, objective={"distance": 1.0, "vehicles": 100.0})
    # route 1: 0 -> 1 -> 2 -> 0 (load 7, distance 2 + 3 + 5 = 10); route 2: 0 -> 3 -> 0 (18)
    total, info, terminated = play(env, [1, 2, 0, 3, 0])
    assert terminated
    assert info["total_costs"] == {"vehicles": 2.0, "distance": 28.0, "time": 28.0}
    assert total == pytest.approx(-(28 + 200))
    sim = env.simulator
    assert [v.route for v in sim.state.vehicles] == [[0, 1, 2, 0], [0, 3, 0]]
    assert all(v.status == VehicleStatus.DONE for v in sim.state.vehicles)
    assert np.all(sim.state.request_status[1:] == RequestStatus.SERVED)
    np.testing.assert_array_equal(sim.state.served_by, [-1, 0, 0, 1])


def test_rewards_follow_steps():
    env = env_for(LINE)
    env.reset(seed=0)
    rewards = [env.step(a)[1] for a in (1, 2, 0, 3, 0)]
    # distances charged at departure: 2, 3, 5 (return), 9, 9
    assert rewards == [-2.0, -3.0, -5.0, -9.0, -9.0]


def test_capacity_mask_and_unlimited_depot_rule():
    env = env_for(LINE)
    _, info = env.reset(seed=0)
    # fresh vehicle of an unlimited fleet: the depot is masked while customers are feasible
    assert info["action_mask"].tolist() == [False, True, True, True]
    env.step(3)  # load 5
    assert env.action_masks().tolist() == [True, True, True, False]
    env.step(1)  # load 9
    assert env.action_masks().tolist() == [True, False, False, False]


def test_demand_above_capacity_is_unserved_and_fleet_stops():
    inst = line_instance([0, 2, 5], [0, 4, 30])
    env = env_for(inst, objective={"distance": 1.0, "unserved": 50.0})
    total, info, terminated = play(env, [1, 0])
    assert terminated
    assert info["total_costs"]["unserved"] == 1.0
    assert total == pytest.approx(-(4 + 50))
    states = env.simulator.state
    assert states.request_status[2] == RequestStatus.UNSERVED
    assert [r["node"] for r in env.event_log.of_kind("unserved")] == [2]
    # a second vehicle was created, found nothing feasible and retired; no third one
    assert len(states.vehicles) == 2
    assert states.vehicles[1].route == []


def test_fixed_fleet_sequential_can_retire_and_ends_with_unserved():
    env = env_for(
        LINE,
        fleet={
            "sizing": "fixed",
            "types": [{"name": "v", "capacity": "from_instance", "count": 2}],
        },
        objective={"distance": 1.0, "unserved": 100.0},
    )
    _, info = env.reset(seed=0)
    assert info["action_mask"][0]  # retiring is allowed with a fixed fleet
    total, info, terminated = play(env, [0, 0])
    assert terminated
    assert info["total_costs"] == {"unserved": 3.0}
    assert total == -300.0
    assert len(env.event_log.of_kind("vehicle_retire")) == 2


def test_on_event_vehicles_interleave_in_time_and_assignment_is_exclusive():
    env = env_for(
        LINE,
        fleet={
            "sizing": "fixed",
            "types": [{"name": "v", "capacity": "from_instance", "count": 2}],
        },
        decisions={"epochs": "on_event"},
        objective={"distance": 1.0, "unserved": 100.0},
    )
    _, info = env.reset(seed=0)
    assert info["vehicle"] == 0 and info["time"] == 0
    _, _, _, _, info = env.step(3)  # vehicle 0 -> x=9
    assert info["vehicle"] == 1 and info["time"] == 0  # vehicle 1 still idle at t=0
    assert not info["action_mask"][3]  # customer 3 is assigned to vehicle 0
    _, _, _, _, info = env.step(1)  # vehicle 1 -> x=2 arrives at t=2
    assert info["vehicle"] == 1 and info["time"] == 2.0
    _, _, _, _, info = env.step(2)  # vehicle 1 -> x=5 at t=5
    assert info["vehicle"] == 1 and info["time"] == 5.0
    _, _, _, _, info = env.step(0)  # vehicle 1 returns (t=10); vehicle 0 is at x=9 since t=9
    assert info["vehicle"] == 0 and info["time"] == 9.0
    _, _, terminated, _, info = env.step(0)
    assert terminated
    assert info["total_costs"]["distance"] == 9 + 9 + 2 + 3 + 5


def test_time_windows_waiting_and_hard_masks():
    # customer 1 at x=2 opens at 10; customer 2 at x=5 closes at 4; depot closes at 30
    inst = line_instance(
        [0, 2, 5], [0, 1, 1], time_windows=[(0, 30), (10, 20), (0, 4)], service=[0, 3, 0]
    )
    env = env_for(
        inst, constraints={"time_windows": {}}, objective={"distance": 1.0, "waiting": 1.0}
    )
    _, info = env.reset(seed=0)
    # customer 2 reachable at t=5 > close 4: masked
    assert info["action_mask"].tolist() == [False, True, False]
    _, reward, _, _, info = env.step(1)
    # arrive t=2, wait to 10 (waiting 8), serve 3 -> idle at t=13
    assert info["time"] == 13.0
    assert info["costs"] == {"vehicles": 1.0, "distance": 2.0, "time": 2.0, "waiting": 8.0}
    assert reward == -10.0
    service = env.event_log.of_kind("service_start")[0]
    assert (service["node"], service["time"]) == (1, 10.0)


def test_hard_windows_look_ahead_to_the_depot():
    # reaching customer 1 is on time, but returning would miss the depot close (8 + 8 > 10)
    inst = line_instance([0, 8], [0, 1], time_windows=[(0, 10), (0, 100)])
    env = env_for(
        inst, constraints={"time_windows": {}}, objective={"distance": 1.0, "unserved": 10.0}
    )
    _, info = env.reset(seed=0)
    assert env.simulator.state.terminated  # nothing feasible: the only vehicle retired at once
    assert info["total_costs"] == {"unserved": 1.0}


def test_waiting_forbidden_masks_early_arrivals():
    inst = line_instance([0, 2, 5], [0, 1, 1], time_windows=[(0, 100), (10, 20), (0, 50)])
    env = env_for(inst, constraints={"time_windows": {"waiting": "forbidden"}})
    _, info = env.reset(seed=0)
    assert info["action_mask"].tolist() == [False, False, True]


def test_soft_windows_charge_lateness():
    inst = line_instance([0, 5], [0, 1], time_windows=[(0, 100), (0, 3)])
    env = env_for(
        inst,
        constraints={"time_windows": {"soft": True}},
        objective={"distance": 1.0, "lateness": 10.0},
    )
    _, info = env.reset(seed=0)
    assert info["action_mask"][1]
    _, reward, _, _, info = env.step(1)
    assert info["costs"]["lateness"] == 2.0
    assert reward == -(5 + 20)


def test_route_duration_and_shift_masks():
    env = env_for(LINE, constraints={"capacity": {}, "route_duration": {"max": 12}})
    _, info = env.reset(seed=0)
    assert info["action_mask"].tolist() == [False, True, True, False]  # 9 + 9 > 12
    env.step(2)  # at x=5, t=5
    assert env.action_masks().tolist() == [True, True, False, False]  # via 2: 5+3+2=10 <= 12
    shift_env = env_for(
        LINE,
        fleet={"types": [{"name": "v", "capacity": "from_instance"}], "shift": [0, 10]},
    )
    _, info = shift_env.reset(seed=0)
    assert info["action_mask"].tolist() == [False, True, True, False]


def test_simulator_rejects_invalid_decisions():
    env = env_for(LINE)
    env.reset(seed=0)
    sim = env.simulator
    with pytest.raises(InfeasibleActionError, match="no pending decision"):
        sim.apply(Dispatch(5, 1))
    with pytest.raises(InfeasibleActionError, match="does not exist"):
        sim.apply(Dispatch(0, 9))
    with pytest.raises(InfeasibleActionError, match="use EndRoute"):
        sim.apply(Dispatch(0, 0))
    with pytest.raises(InfeasibleActionError, match="cannot retire"):
        sim.apply(EndRoute(0))
    sim.apply(Dispatch(0, 3))
    sim.apply(Dispatch(0, 1))  # load 5 + 4 = 9
    with pytest.raises(InfeasibleActionError, match="capacity"):
        sim.apply(Dispatch(0, 2))  # demand 3 does not fit
    with pytest.raises(InfeasibleActionError, match="served"):
        sim.apply(Dispatch(0, 1))
    sim.apply(Dispatch(0, 0))
    sim.apply(Dispatch(1, 2))
    sim.apply(Dispatch(1, 0))
    with pytest.raises(InfeasibleActionError, match="ended"):
        sim.apply(Dispatch(1, 0))


@pytest.mark.parametrize(
    ("changes", "error", "match"),
    [
        ({"decisions": {"epochs": "on_event"}}, ConfigError, "sequential"),
        (
            {"fleet": {"types": [{"name": "a", "capacity": 1}, {"name": "b", "capacity": 2}]}},
            ConfigError,
            "exactly one vehicle type",
        ),
        (
            {"uncertainty": {"demand": {"dist": "poisson", "mean": 1, "revealed": "on_arrival"}}},
            ConfigError,
            "recourse.capacity_failure",
        ),
        (
            {
                "uncertainty": {"demand": {"dist": "poisson", "mean": 1, "revealed": "on_arrival"}},
                "recourse": {"capacity_failure": "return_to_depot"},
                "constraints": {"capacity": {}, "route_duration": {"max": 100}},
            },
            NotInScopeError,
            "recourse backlog",
        ),
        (
            {
                "uncertainty": {
                    "service_time": {"dist": "exponential", "mean": 1, "revealed": "on_arrival"}
                },
                "constraints": {"capacity": {}, "route_duration": {"max": 100}},
            },
            NotInScopeError,
            "time-window recourse",
        ),
        (
            {
                "uncertainty": {
                    "demand": {"dist": "poisson", "mean": 1, "revealed": "on_completion"}
                }
            },
            ConfigError,
            "cannot be revealed",
        ),
        (
            {
                "dynamics": {"arrivals": {"process": "poisson", "rate": 1}, "max_requests": 3},
                "decisions": {"epochs": "on_event"},
            },
            NotInScopeError,
            "M6",
        ),
        ({"decisions": {"epochs": "periodic", "period": 5}}, NotInScopeError, "M6"),
        (
            {"constraints": {"outsourcing": {"cost": "per_customer", "value": 1}}},
            NotInScopeError,
            "outsourcing",
        ),
        (
            {"fleet": {"types": [{"name": "v", "capacity": 10, "fixed_cost": 5}]}},
            NotInScopeError,
            "fixed costs",
        ),
    ],
)
def test_out_of_scope_and_invalid_combinations(changes, error, match):
    with pytest.raises(error, match=match):
        env_for(LINE, **changes).reset(seed=0)


def test_fixed_fleet_needs_an_unserved_weight():
    fleet = {"sizing": "fixed", "types": [{"name": "v", "capacity": "from_instance", "count": 1}]}
    with pytest.raises(ConfigError, match="`unserved` weight"):
        env_for(LINE, fleet=fleet).reset(seed=0)
    lexicographic = {"lexicographic": ["unserved", "distance"]}
    env_for(LINE, fleet=fleet, objective=lexicographic).reset(seed=0)


def test_default_constraint_check_uses_the_mask():
    env = env_for(LINE)
    env.reset(seed=0)
    sim = env.simulator
    capacity = sim.constraints[0]
    sim.apply(Dispatch(0, 3))
    sim.apply(Dispatch(0, 1))
    violation = capacity.check(None, sim.info, Dispatch(0, 2))
    assert violation is not None and violation.constraint == "capacity"
    assert capacity.check(None, sim.info, Dispatch(0, 0)) is None
    assert capacity.check(None, sim.info, EndRoute(0)) is None
