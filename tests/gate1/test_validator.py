"""Unit tests: the validator detects each kind of corruption."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from fleetgym.errors import ValidationFailure
from fleetgym.eventlog import EventLog
from fleetgym.spec import load_spec
from fleetgym.testing.validator import validate
from tests.sim.helpers import env_for, line_instance

TW_INSTANCE = line_instance(
    [0, 2, 5, 9],
    [0, 4, 3, 5],
    time_windows=[(0, 100), (0, 50), (10, 60), (0, 90)],
    service=[0, 1, 2, 3],
)
CHANGES = {
    "constraints": {"capacity": {}, "time_windows": {}, "route_duration": {"max": 60}},
    "objective": {"distance": 1.0, "waiting": 0.5},
}


def valid_episode():
    env = env_for(TW_INSTANCE, **CHANGES)
    env.reset(seed=0)
    for action in (1, 2, 0, 3, 0):
        env.step(action)
    return env


def first(log: EventLog, kind: str, **match: Any) -> dict[str, Any]:
    return next(
        r for r in log.rows if r["kind"] == kind and all(r.get(k) == v for k, v in match.items())
    )


def test_valid_episode_passes_and_recomputes_costs(tmp_path):
    env = valid_episode()
    report = validate(env.event_log, env.problem_spec, TW_INSTANCE)
    assert report.ok, report.violations
    assert report.costs == pytest.approx(env.simulator.state.total_costs)
    assert report.episode_return == pytest.approx(report.expected_return)
    path = tmp_path / "log.parquet"
    env.event_log.write(path)
    assert validate(path, env.problem_spec, TW_INSTANCE).ok
    report.raise_if_invalid()


def corrupt_arrival(log: EventLog) -> None:
    first(log, "arrive", node=2)["time"] += 1


def corrupt_distance(log: EventLog) -> None:
    first(log, "depart", node=1)["value"] += 1


def corrupt_origin(log: EventLog) -> None:
    first(log, "depart", node=2)["from_node"] = 3


def corrupt_service_start(log: EventLog) -> None:
    first(log, "service_start", node=2)["time"] = 6.0


def corrupt_quantity(log: EventLog) -> None:
    first(log, "service_start", node=1)["quantity"] = 99.0


def corrupt_load(log: EventLog) -> None:
    first(log, "service_start", node=2)["load"] = 1.0


def corrupt_service_end(log: EventLog) -> None:
    first(log, "service_end", node=1)["time"] += 5


def corrupt_cost(log: EventLog) -> None:
    first(log, "cost", component="distance")["amount"] += 1


def corrupt_reward(log: EventLog) -> None:
    first(log, "epoch")["reward"] -= 1


def drop_route_end(log: EventLog) -> None:
    log.rows[:] = [
        r
        for r in log.rows
        if not (r["kind"] in ("arrive", "route_end") and r.get("node") == 0 and r["vehicle"] == 1)
    ]


def double_service(log: EventLog) -> None:
    log.rows.append({"seq": len(log.rows), "kind": "unserved", "node": 1})


def wrong_start(log: EventLog) -> None:
    first(log, "vehicle_activate")["time"] = 5.0


def wrong_decision(log: EventLog) -> None:
    first(log, "decision")["node"] = 3


@pytest.mark.parametrize(
    ("corrupt", "code"),
    [
        (corrupt_arrival, "travel_time"),
        (corrupt_distance, "distance"),
        (corrupt_origin, "depart"),
        (corrupt_service_start, "service_start"),
        (corrupt_quantity, "quantity"),
        (corrupt_load, "load"),
        (corrupt_service_end, "service_time"),
        (corrupt_cost, "cost"),
        (corrupt_reward, "reward"),
        (drop_route_end, "incomplete"),
        (double_service, "unserved"),
        (wrong_start, "start"),
        (wrong_decision, "decision"),
    ],
)
def test_corruptions_are_detected(corrupt: Callable[[EventLog], None], code: str):
    env = valid_episode()
    log = env.event_log.copy()
    corrupt(log)
    report = validate(log, env.problem_spec, TW_INSTANCE)
    assert code in {v.code for v in report.violations}, report.violations
    with pytest.raises(ValidationFailure, match=code):
        report.raise_if_invalid()
    assert validate(env.event_log, env.problem_spec, TW_INSTANCE).ok  # the original is untouched


def test_constraint_violations_against_a_stricter_spec():
    env = valid_episode()
    strict = load_spec(
        env.problem_spec.to_dict()
        | {
            "constraints": {"capacity": {}, "time_windows": {}, "route_duration": {"max": 5}},
            "fleet": {"types": [{"name": "v", "capacity": 6}]},
        }
    )
    codes = {v.code for v in validate(env.event_log, strict, TW_INSTANCE).violations}
    assert {"identity", "capacity", "capacity_value", "route_duration"} <= codes


def test_time_window_violations_are_detected():
    late = line_instance(
        [0, 2, 5, 9],
        [0, 4, 3, 5],
        time_windows=[(0, 100), (0, 50), (0, 5), (0, 90)],
        service=[0, 1, 2, 3],
    )
    env = valid_episode()
    codes = {v.code for v in validate(env.event_log, env.problem_spec, late).violations}
    assert "time_window" in codes


def test_fleet_and_unknown_vehicle_violations():
    env = valid_episode()
    fixed = load_spec(
        env.problem_spec.to_dict()
        | {
            "fleet": {
                "sizing": "fixed",
                "types": [{"name": "v", "capacity": "from_instance", "count": 1}],
            },
            "decisions": {"epochs": "sequential"},
            "objective": {"distance": 1.0, "unserved": 1.0},
        }
    )
    codes = {v.code for v in validate(env.event_log, fixed, TW_INSTANCE).violations}
    assert "fleet" in codes
    log = env.event_log.copy()
    log.rows.insert(1, {"seq": -1, "kind": "arrive", "vehicle": 7, "node": 1, "time": 0.0})
    assert "unknown_vehicle" in {
        v.code for v in validate(log, env.problem_spec, TW_INSTANCE).violations
    }


def test_many_violations_are_truncated_in_the_message():
    env = valid_episode()
    log = env.event_log.copy()
    for row in log.rows:
        if row["kind"] == "epoch":
            row["reward"] = 0.0
    for _ in range(60):
        log.rows.append({"seq": 0, "kind": "unserved", "node": 1})
    with pytest.raises(ValidationFailure, match=r"and .* more"):
        validate(log, env.problem_spec, TW_INSTANCE).raise_if_invalid()


def test_matrix_networks_are_supported():
    from fleetgym.data import Instance

    inst = Instance(
        name="m",
        source="t",
        demands=np.array([0.0, 1.0]),
        distance_matrix=np.array([[0.0, 3.0], [4.0, 0.0]]),
        capacity=5,
    )
    env = env_for(inst, network={"type": "matrix"})
    env.reset(seed=0)
    env.step(1)
    env.step(0)
    report = validate(env.event_log, env.problem_spec, inst)
    assert report.ok and report.costs["distance"] == 7.0
