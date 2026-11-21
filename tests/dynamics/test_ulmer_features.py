"""Model features added for the setting of Ulmer, Goodson, Mattfeld and Hennig (ADR 0024).

Clustered request locations (``normal_mixture``), travel times rounded up to whole units
(``network.time_rounding: ceil``), requests known at the start that must be served
(``initial_requests: mandatory``) and request decisions at vehicle arrivals
(``decisions.request_epochs: on_vehicle_arrival``).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import fleetgym as fg
from fleetgym.data import Instance
from fleetgym.errors import ConfigError, InstanceError
from fleetgym.network.euclidean import EuclideanNetwork
from fleetgym.spec import load_spec
from fleetgym.testing.validator import validate
from fleetgym.uncertainty.distributions import build_distribution, normal_mixture_points

CENTERS = [[5.0, 5.0], [5.0, 15.0], [15.0, 10.0]]


def test_normal_mixture_draws_clusters_by_weight_with_the_given_std():
    params = {"centers": CENTERS, "std": 1.0, "weights": [1, 2, 1]}
    points = normal_mixture_points(params, np.random.default_rng(0), 40_000)
    centers = np.asarray(CENTERS)
    nearest = np.argmin(((points[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    observed = np.bincount(nearest, minlength=3)
    assert stats.chisquare(observed, np.array([0.25, 0.5, 0.25]) * len(points)).pvalue > 1e-3
    residuals = (points - centers[nearest]).ravel()
    assert stats.kstest(residuals, "norm").pvalue > 1e-3


def test_normal_mixture_parameters_are_checked():
    with pytest.raises(ConfigError, match="same, positive dimension"):
        load_spec(
            {
                "uncertainty": {
                    "demand": {"dist": "normal_mixture", "centers": [[1], [1, 2]], "std": 1}
                }
            }
        )
    with pytest.raises(ConfigError, match="one per center"):
        load_spec(
            {
                "uncertainty": {
                    "demand": {
                        "dist": "normal_mixture",
                        "centers": [[1], [2]],
                        "std": 1,
                        "weights": [1],
                    }
                }
            }
        )
    scalar = build_distribution(
        "normal_mixture", {"centers": [[2.0], [8.0]], "std": 0.5, "weights": [3, 1]}
    )
    assert scalar.mean(2) == pytest.approx([3.5, 3.5])
    with pytest.raises(ValueError, match="one-dimensional"):
        build_distribution("normal_mixture", {"centers": CENTERS, "std": 1.0}).sample(
            np.random.default_rng(0), 3
        )


def test_travel_times_round_up_to_whole_units_and_distances_do_not():
    coords = np.array([[0.0, 0.0], [5.0, 0.0], [1.0, 0.0], [3.0, 4.0]])
    net = EuclideanNetwork(coords, speed=25 / 60, time_rounding="ceil")  # km at 25 km/h, minutes
    times, distances = net.time_matrix(), net.distance_matrix()
    assert times[0, 1] == 12.0  # exactly 12 minutes stays 12
    assert times[0, 2] == 3.0  # 2.4 minutes rounds up
    assert times[0, 3] == 12.0
    assert distances[0, 2] == 1.0
    assert np.array_equal(EuclideanNetwork(coords, speed=25 / 60).time_matrix()[0, 2], 2.4)


def day(n_early: int, close: float = 360.0, seed: int = 0) -> Instance:
    rng = np.random.default_rng(seed)
    n = n_early + 1
    return Instance(
        name=f"day-{n_early}",
        source="test",
        demands=np.zeros(n),
        coords=np.vstack([[10.0, 10.0], rng.uniform(0, 20, (n_early, 2))]),
        time_windows=np.tile([0.0, close], (n, 1)),
        vehicle_types=({"count": 1},),
        distributions={
            "arrivals": {
                "process": "nhpp",
                "profile": {"breaks": [0, 1, close], "rates": [0, 50 / 359]},
            },
            "request.location": {"dist": "normal_mixture", "centers": CENTERS, "std": 1.0},
        },
    )


MODEL = {
    "fleet": {"sizing": "fixed"},
    "network": {"type": "euclidean", "speed": 25 / 60, "time_rounding": "ceil"},
    "constraints": {"time_windows": {}},
    "dynamics": {"max_requests": 150, "initial_requests": "mandatory"},
    "decisions": {
        "interface": "accept_reject",
        "routing": "cheapest_insertion",
        "request_epochs": "on_vehicle_arrival",
    },
    "objective": {"rejected": 1.0},
}


def run(spec, instance, seed, choose):
    env = fg.make(spec, instance=instance)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    while not env.simulator.state.terminated:
        env.step(choose(np.flatnonzero(env.action_masks()), rng))
    return env


def test_mandatory_initial_requests_are_accepted_at_the_start():
    spec = load_spec(MODEL)
    inst = day(25)
    env = run(spec, inst, 0, lambda valid, rng: int(rng.choice(valid)))
    log = env.event_log
    accepts_at_start = [r["node"] for r in log.of_kind("accept") if r["time"] == 0.0]
    assert accepts_at_start == list(range(1, 26))
    rejected = {r.get("node") for r in log.of_kind("reject")}
    assert not rejected & set(range(1, 26))
    served = {r["node"] for r in log.of_kind("arrive") if r["node"] != 0}
    assert set(range(1, 26)) <= served
    assert validate(log, spec, inst, scenario=env.scenario).ok


def test_mandatory_initial_requests_that_do_not_fit_raise():
    with pytest.raises(InstanceError, match="must be served"):
        fg.make(load_spec(MODEL), instance=day(25, close=40.0)).reset(seed=0)


def test_the_validator_rejects_a_rejected_mandatory_request():
    free = load_spec({**MODEL, "dynamics": {"max_requests": 150, "initial_requests": "instance"}})
    inst = day(5)
    env = run(free, inst, 1, lambda valid, rng: 0)  # reject everything, the early ones too
    assert validate(env.event_log, free, inst, scenario=env.scenario).ok
    strict = load_spec(MODEL)
    report = validate(env.event_log, strict, inst, scenario=env.scenario)
    assert not report.ok
    assert any("mandatory" in v.message for v in report.violations)


@pytest.mark.parametrize("seed", range(6))
def test_requests_are_decided_at_vehicle_arrivals_in_release_order(seed):
    spec = load_spec(MODEL)
    inst = day(10, seed=seed)
    env = run(spec, inst, seed, lambda valid, rng: int(valid[-1]))  # accept when possible
    log = env.event_log
    assert validate(log, spec, inst, scenario=env.scenario).ok
    release = {r["node"]: r["time"] for r in log.of_kind("request") if r["time"] > 0}
    arrivals = {r["time"] for r in log.of_kind("arrive")}
    decisions = [
        r for r in log.rows if r["kind"] in ("accept", "reject") and r.get("node") in release
    ]
    assert len(decisions) == len(release)
    held = 0
    for row in decisions:
        t, node = row["time"], row["node"]
        assert t >= release[node]
        if t > release[node]:  # held: decided when the vehicle arrived somewhere
            assert t in arrivals
            held += 1
    order = [row["node"] for row in decisions]
    assert order == sorted(order, key=lambda node: (release[node], node))
    assert held > 0


def test_on_release_is_the_default_and_decides_at_once():
    spec = load_spec({**MODEL, "decisions": {**MODEL["decisions"], "request_epochs": "on_release"}})
    assert (
        load_spec(
            {k: v for k, v in MODEL.items() if k != "decisions"}
            | {"decisions": {"interface": "accept_reject", "routing": "cheapest_insertion"}}
        ).decisions.request_epochs
        == "on_release"
    )
    env = run(spec, day(10), 0, lambda valid, rng: int(valid[-1]))
    release = {r["node"]: r["time"] for r in env.event_log.of_kind("request") if r["time"] > 0}
    for row in env.event_log.rows:
        if row["kind"] in ("accept", "reject") and row.get("node") in release:
            assert row["time"] == release[row["node"]]


def test_request_epochs_need_a_request_interface():
    with pytest.raises(ConfigError, match="request interfaces"):
        load_spec({"decisions": {"interface": "next_node", "request_epochs": "on_vehicle_arrival"}})


def test_literal_location_parameters_come_from_the_instance():
    inst = day(5)
    uniform = {"dist": "uniform", "low": 0.0, "high": 15.0}
    inst = Instance(
        **{**inst.__dict__, "distributions": {**inst.distributions, "request.location": uniform}}
    )
    env = fg.make(load_spec(MODEL), instance=inst)
    env.reset(seed=0)
    assert env.scenario is not None
    late = np.isfinite(env.scenario["request.release"]) & (env.scenario["request.release"] > 0)
    coords = env.simulator.problem.instance.coords
    assert coords is not None
    points = coords[late]
    assert points.size and np.all((points >= 0) & (points < 15))


def test_serve_all_plans_infeasible_early_requests_late_and_accepts_nothing_else():
    spec = load_spec(
        {**MODEL, "dynamics": {**MODEL["dynamics"], "initial_infeasible": "serve_all"}}
    )
    inst = day(25, close=40.0)
    env = run(spec, inst, 0, lambda valid, rng: int(valid[-1]))  # would accept if it could
    log = env.event_log
    assert len(log.of_kind("initial_overtime")) == 1
    early = set(range(1, 26))
    assert early <= {r["node"] for r in log.of_kind("accept")}
    assert not ({r["node"] for r in log.of_kind("accept")} - early)  # no late request confirmed
    assert early <= {r["node"] for r in log.of_kind("arrive")}
    assert log.of_kind("route_end")[-1]["time"] > 40.0  # back late
    assert validate(log, spec, inst, scenario=env.scenario).ok
    # the same log under `error` is not valid
    strict = load_spec(MODEL)
    report = validate(log, strict, inst, scenario=env.scenario)
    assert any("initial_infeasible: serve_all" in v.message for v in report.violations)


def test_serve_all_changes_nothing_when_the_early_requests_fit():
    lenient = load_spec(
        {**MODEL, "dynamics": {**MODEL["dynamics"], "initial_infeasible": "serve_all"}}
    )
    inst = day(10)
    a = run(lenient, inst, 3, lambda valid, rng: int(valid[-1])).event_log
    assert not a.of_kind("initial_overtime")
    b = run(load_spec(MODEL), inst, 3, lambda valid, rng: int(valid[-1])).event_log
    assert [(r["kind"], r.get("time"), r.get("node")) for r in a.rows] == [
        (r["kind"], r.get("time"), r.get("node")) for r in b.rows
    ]


def test_initial_infeasible_needs_mandatory_requests():
    with pytest.raises(ConfigError, match="initial_requests: mandatory"):
        load_spec({**MODEL, "dynamics": {"max_requests": 5, "initial_infeasible": "serve_all"}})
