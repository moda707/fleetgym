"""Sampling the requests of a day."""

from pathlib import Path

import numpy as np
import pytest

from fleetgym.errors import ConfigError
from fleetgym.spec import load_spec
from fleetgym.spec.binding import bind, instance_from_source
from fleetgym.uncertainty.requests import RequestOverflowError, arrival_window, sample_requests
from fleetgym.uncertainty.scenario import Scenario, episode_instance
from tests.families import instance_source


def dynamic(**changes):
    base = {
        "extends": "VRPTW-v1",
        "instance_source": instance_source("vrptw"),
        "fleet": {
            "sizing": "fixed",
            "types": [{"name": "t", "capacity": "from_instance", "count": 2}],
        },
        "dynamics": {
            "arrivals": {"process": "poisson", "rate": 0.1},
            "cutoff": 200,
            "max_requests": 60,
            "request_class": {
                "location": {"dist": "kde", "data": "from_instance"},
                "demand": {"dist": "categorical", "values": [1, 2], "probs": [0.5, 0.5]},
                "time_window_width": {"dist": "uniform", "low": 30, "high": 60},
                "service_time": {"dist": "exponential", "mean": 5},
            },
            "cancellations": {"dist": "bernoulli", "p": 0.5},
        },
        "decisions": {
            "epochs": "on_event",
            "interface": "accept_reject",
            "routing": "cheapest_insertion",
        },
        "objective": {"distance": 1.0, "rejected": 10.0},
    }
    for key, value in changes.items():
        if key == "dynamics":
            base["dynamics"] = {**base["dynamics"], **value}
        else:
            base[key] = value
    spec = load_spec(base)
    return bind(spec, instance_from_source(spec, np.random.default_rng(0)))


def test_requests_respect_the_window_and_attributes():
    problem = dynamic()
    values = sample_requests(problem, 3)
    static = problem.instance.n_nodes
    release = values["request.release"]
    assert release.size == static + 60
    assert np.all(release[1:static] == 0.0)
    dynamic_times = release[static:][np.isfinite(release[static:])]
    assert dynamic_times.size > 0 and np.all(np.diff(dynamic_times) > 0)
    assert np.all(dynamic_times < 200.0)  # cutoff: none generated after it
    k = dynamic_times.size
    s = slice(static, static + k)
    np.testing.assert_array_equal(values["request.tw_open"][s], dynamic_times)
    width = values["request.tw_close"][s] - dynamic_times
    assert problem.instance.time_windows is not None
    depot_close = problem.instance.time_windows[0, 1]
    assert np.all(
        (width >= 30 - 1e-9) & (width <= 60 + 1e-9)
        | np.isclose(values["request.tw_close"][s], depot_close)
    )
    assert set(values["request.demand"][s]) <= {1.0, 2.0}
    cancel = values["request.cancel"][s]
    some = np.isfinite(cancel)
    assert 0 < some.sum() < k
    assert np.all(cancel[some] >= dynamic_times[some]) and np.all(
        cancel[some] <= values["request.tw_close"][s][some]
    )
    assert np.all(np.isinf(values["request.cancel"][:static]))
    assert np.all(np.isinf(release[static + k :]))
    assert values["request.overflow"].size == 0
    np.testing.assert_array_equal(sample_requests(problem, 3)["request.x"], values["request.x"])


def test_attribute_streams_are_independent():
    a = sample_requests(dynamic(), 5)
    other_demand = {
        "location": {"dist": "kde", "data": "from_instance"},
        "demand": {"dist": "categorical", "values": [7, 9], "probs": [0.5, 0.5]},
        "time_window_width": {"dist": "uniform", "low": 30, "high": 60},
        "service_time": {"dist": "exponential", "mean": 5},
    }
    b = sample_requests(dynamic(dynamics={"request_class": other_demand}), 5)
    for name in (
        "request.release",
        "request.x",
        "request.y",
        "request.service_time",
        "request.cancel",
    ):
        np.testing.assert_array_equal(a[name], b[name])
    assert not np.array_equal(a["request.demand"], b["request.demand"])


def test_overflow_raises_or_is_recorded():
    with pytest.raises(RequestOverflowError, match="max_requests"):
        sample_requests(dynamic(dynamics={"max_requests": 2}), 0)
    values = sample_requests(dynamic(dynamics={"max_requests": 2, "overflow": "reject"}), 0)
    static = values["request.release"].size - 2
    assert np.all(np.isfinite(values["request.release"][static:]))
    assert values["request.overflow"].size > 0
    assert np.all(values["request.overflow"] > values["request.release"][-1])


def test_initial_requests_none_and_uniform_locations():
    problem = dynamic(
        dynamics={
            "initial_requests": "none",
            "request_class": {"location": {"dist": "uniform", "low": 0, "high": 10}},
        }
    )
    values = sample_requests(problem, 1)
    assert values["request.release"].size == 1 + 60
    k = int(np.isfinite(values["request.release"][1:]).sum())
    assert np.all((values["request.x"][1 : 1 + k] >= 0) & (values["request.x"][1 : 1 + k] < 10))
    assert np.all(values["request.demand"][1:] == 0)  # no demand distribution: 0


def test_arrival_window_and_location_errors():
    assert arrival_window(dynamic()) == (0.0, 200.0)
    nhpp = {"process": "nhpp", "profile": {"breaks": [0, 50, 100], "rates": [0.1, 0.2]}}
    assert arrival_window(dynamic(dynamics={"arrivals": nhpp, "cutoff": None})) == (0.0, 100.0)
    bad = dynamic(dynamics={"request_class": {"location": {"dist": "normal", "mean": 0, "std": 1}}})
    with pytest.raises(ConfigError, match="kde"):
        sample_requests(bad, 0)
    missing = dynamic(dynamics={"request_class": {}})
    with pytest.raises(ConfigError, match="location"):
        sample_requests(missing, 0)


def test_episode_instance_and_parquet_round_trip(tmp_path: Path):
    problem = dynamic(dynamics={"max_requests": 2, "overflow": "reject"})
    values = sample_requests(problem, 0)
    scenario = Scenario(problem.spec_hash, problem.instance_hash, values, seed=0)
    again = Scenario.read(scenario.write(tmp_path / "s.parquet"))
    assert again == scenario
    np.testing.assert_array_equal(again["request.overflow"], values["request.overflow"])
    episode = episode_instance(problem.instance, scenario)
    assert episode.n_nodes == problem.instance.n_nodes + 2
    assert episode.extra["_base_instance_hash"] == problem.instance_hash
    assert episode.extra["_n_static"] == problem.instance.n_nodes
    assert episode.coords is not None
    np.testing.assert_array_equal(
        episode.coords[: problem.instance.n_nodes], problem.instance.coords
    )
    assert episode_instance(problem.instance, Scenario("a", "b", {})) is problem.instance
