from collections.abc import Callable

import numpy as np
import pytest

from fleetgym.sim.engine import Simulator
from fleetgym.sim.invariants import InvariantMonitor, InvariantViolation
from fleetgym.sim.state import RequestStatus, VehicleStatus
from tests.sim.helpers import LINE_XS, env_for, line_instance


def started() -> Simulator:
    env = env_for(line_instance(LINE_XS, [0, 4, 3, 5]))
    env.reset(seed=0)
    env.step(1)
    return env.simulator


def overload(sim: Simulator) -> None:
    sim.state.vehicles[0].load = 99.0


def negative_load(sim: Simulator) -> None:
    sim.state.vehicles[0].load = -1.0


def late_route_start(sim: Simulator) -> None:
    sim.state.vehicles[0].route_start = 1e9


def served_without_record(sim: Simulator) -> None:
    sim.state.request_status[2] = RequestStatus.SERVED


def unserved_with_vehicle(sim: Simulator) -> None:
    sim.state.served_by[3] = 0


def orphan_assignment(sim: Simulator) -> None:
    sim.state.request_status[3] = RequestStatus.ASSIGNED


def negative_cost(sim: Simulator) -> None:
    sim.state.total_costs["distance"] = -1.0


@pytest.mark.parametrize(
    ("corrupt", "match"),
    [
        (overload, "capacity"),
        (negative_load, "negative load"),
        (late_route_start, "route starts after"),
        (served_without_record, "no service record"),
        (unserved_with_vehicle, "serving vehicle"),
        (orphan_assignment, "no vehicle is heading"),
        (negative_cost, "negative cost"),
    ],
)
def test_monitor_detects_corruption(corrupt: Callable[[Simulator], None], match: str):
    sim = started()
    monitor = InvariantMonitor()
    monitor.check(sim)
    corrupt(sim)
    with pytest.raises(InvariantViolation, match=match):
        monitor.check(sim)


def test_monitor_detects_clock_going_back():
    sim = started()
    monitor = InvariantMonitor()
    monitor.check(sim)
    sim.state.vehicles[0].time -= 1.0
    with pytest.raises(InvariantViolation, match="clock went back"):
        monitor.check(sim)
    monitor.reset()
    monitor.check(sim)


def test_information_state_view():
    sim = started()
    info = sim.info
    assert info.time == sim.state.time == 2.0
    assert info.n_vehicles == 1
    assert info.vehicle(0).status == VehicleStatus.IDLE
    assert info.pending().tolist() == [False, False, True, True]
    assert info.request_status()[1] == RequestStatus.SERVED
    arrival, start, end = info.projected(0, np.array([2, 3]))
    np.testing.assert_array_equal(arrival, [5.0, 9.0])
    np.testing.assert_array_equal(start, arrival)
    np.testing.assert_array_equal(end, arrival)
