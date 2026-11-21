"""The paper examples' instances follow their papers, and the presets run on them (ADR 0023).

Numbers are from the papers: Dastpak, Errico and Jabali (arXiv 2109.10200, Appendices B.1.1 and
B.1.2), Kool et al. (arXiv 1803.08475, Appendix C.1), Vera and Abad (arXiv 1912.03341, Table 1).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from scipy import stats

import fleetgym as fg
from fleetgym.baselines import NearestNeighbor, run_episode
from fleetgym.data import read_instance
from fleetgym.testing.validator import validate

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "instances"


def example_module(folder: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"instances_{folder}", EXAMPLES / folder / "instances.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VCSD = example_module("09_vrp_vcsd")


def zone_of(points: np.ndarray) -> np.ndarray:
    return (np.floor(points[:, 1] / 20) * 5 + np.floor(points[:, 0] / 20)).astype(int)


def test_vcsd_instances_follow_appendix_b11():
    rng = np.random.default_rng(0)
    counts, demands = [], []
    for k in range(800):
        instance = VCSD.vcsd_instance(rng, "moderate", 50, f"t{k}")
        assert instance.coords is not None
        np.testing.assert_array_equal(instance.coords[0], [50.0, 50.0])
        zones = zone_of(instance.coords[1:])
        assert set(zones) <= set(VCSD.ACTIVE_ZONES)
        counts += [int(np.sum(zones == z)) for z in VCSD.ACTIVE_ZONES]
        demands += instance.demands[1:].tolist()
    observed = np.bincount(counts, minlength=4)
    assert stats.chisquare(observed, np.array([0.1, 0.4, 0.4, 0.1]) * len(counts)).pvalue > 1e-3
    assert np.mean(counts) * 15 == pytest.approx(22.5, rel=0.03)  # 1.5 customers per zone
    assert set(demands) == {5.0, 10.0, 15.0}
    assert np.mean(demands) == pytest.approx(10.0, rel=0.03)


@pytest.mark.parametrize(
    ("density", "vehicles", "limit"),
    [
        ("very_low", 2, 143.71),
        ("low", 2, 201.38),
        ("moderate", 3, 221.47),
        ("high", 7, 195.54),
        ("very_high", 11, 187.29),
    ],
)
def test_vcsd_density_fixes_fleet_and_limit(density, vehicles, limit):
    instance = VCSD.vcsd_instance(np.random.default_rng(1), density, 25, "t")
    assert instance.vehicle_types[0]["count"] == vehicles
    assert instance.vehicle_types[0]["capacity"] == 25.0
    assert instance.extra["max_route_duration"] == limit
    assert dict(instance.distributions["demand"]) == VCSD.DEMAND


def test_vcsd_preset_runs_on_the_examples_instances(tmp_path):
    VCSD.write_vcsd(tmp_path, n_instances=3, seed=0)
    preset = fg.load_preset("dastpak2022-vcsd")
    for point in preset.points():
        env = fg.make_preset("dastpak2022-vcsd", instances=tmp_path, **point)
        assert isinstance(env, fg.FleetGymEnv)
        episode = run_episode(env, NearestNeighbor(), seed=1)
        assert validate(
            env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario
        ).ok
        assert episode.costs.get("served_demand", 0.0) > 0
        assert env.scenario is not None
        expected = env.base_instance.demands[1:]
        width = np.where(expected == 5, 4, 5)  # +- 4 when the expected demand is 5
        assert np.all(np.abs(env.scenario["demand"][1:] - expected) <= width)


def test_goodson_instances_run_offline_on_a_local_base(tmp_path, monkeypatch):
    goodson = example_module("10_vrpsd_goodson")
    base = read_instance(FIXTURES / "tiny_solomon.txt")
    import dataclasses

    base = dataclasses.replace(base, time_windows=None, service_times=None)
    monkeypatch.setattr(goodson, "r101_75", lambda: base)
    goodson.write_goodson(tmp_path)
    env = fg.make_preset(
        "dastpak2022-goodson", instances=tmp_path, limit="long", capacity=25, variability="high"
    )
    assert isinstance(env, fg.FleetGymEnv)
    run_episode(env, NearestNeighbor(), seed=0)
    assert validate(env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario).ok
    assert env.base_instance.extra["max_route_duration"] == 240.45
    assert env.scenario is not None
    factors = env.scenario["demand"][1:] / base.demands[1:]
    assert set(np.round(factors, 9)) <= {0.0, 0.5, 1.0, 1.5, 2.0}


@pytest.mark.network
def test_goodson_base_is_the_papers_r101_subset():
    base = example_module("10_vrpsd_goodson").r101_75()
    assert base.n_customers == 75 and base.demands.sum() == 1079.0  # the paper's total


def test_kool_and_vera_instances_follow_their_papers():
    kool = example_module("11_cvrp_kool")
    for n, capacity in ((20, 30), (50, 40), (100, 50)):
        instance = kool.kool_instance(np.random.default_rng(n), n, "t")
        assert instance.n_customers == n and instance.vehicle_types[0]["capacity"] == capacity
        assert instance.coords is not None and np.all(
            (instance.coords >= 0) & (instance.coords <= 1)
        )
        assert set(instance.demands[1:]) <= set(range(1, 10))
    vera = example_module("14_ma_cvrp_vera")
    instance = vera.vera_instance(np.random.default_rng(0), "vrp20", "t")
    assert [vt["capacity"] for vt in instance.vehicle_types] == [20.0, 30.0, 35.0]
    assert instance.n_customers == 20


ULMER = example_module("15_dvrp_ulmer")


def test_ulmer_instances_follow_section_6_1():
    rng = np.random.default_rng(3)
    early = [
        ULMER.ulmer_instance(rng, "small", "moderate", "uniform", "t").n_customers
        for _ in range(400)
    ]
    assert np.mean(early) == pytest.approx(50, rel=0.03)  # Poisson(100 - lambda)
    day = ULMER.ulmer_instance(rng, "large", "high", "three_clusters", "t")
    assert day.coords is not None
    np.testing.assert_array_equal(day.coords[0], [10.0, 10.0])  # central depot
    np.testing.assert_array_equal(day.time_windows[0], [0.0, 360.0])
    profile = day.distributions["arrivals"]["profile"]
    assert tuple(profile["breaks"]) == (0.0, 1.0, 360.0)
    assert profile["rates"][1] == pytest.approx(75 / 359)
    where = day.distributions["request.location"]
    assert [list(c) for c in where["centers"]] == [[5, 5], [5, 15], [15, 10]]
    assert tuple(where["weights"]) == (1.0, 2.0, 1.0) and where["std"] == 1.0
    small = ULMER.location_distribution("small", "two_clusters")
    assert small["centers"] == [[3.75, 3.75], [3.75, 11.25]] and small["std"] == 0.75
    assert ULMER.location_distribution("small", "uniform") == {
        "dist": "uniform",
        "low": 0.0,
        "high": 15.0,
    }


def test_ulmer_preset_runs_on_the_examples_instances(tmp_path):
    ULMER.write_ulmer(tmp_path, n_instances=1, seed=0)
    points: list[dict[str, Any]] = [{"area": "small", "lam": "low", "locations": "uniform"}, {}]
    for point in points:
        env = fg.make_preset("ulmer2016-vrpssr", instances=tmp_path, **point)
        assert isinstance(env, fg.FleetGymEnv)
        run_episode(env, NearestNeighbor(), seed=2)
        assert validate(
            env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario
        ).ok
        early = set(range(1, env.base_instance.n_customers + 1))
        assert early <= {row["node"] for row in env.event_log.of_kind("accept")}
        times = env.simulator.network.time_matrix()
        assert np.array_equal(times, np.round(times))  # whole minutes
    # large area, low lambda, uniform: about 75 early customers rarely fit by T (2017 rule)
    days = [
        ULMER.ulmer_instance(np.random.default_rng(k), "large", "low", "uniform", "t")
        for k in range(5)
    ]
    spec = fg.load_preset("ulmer2016-vrpssr").spec(
        tmp_path, area="large", lam="low", locations="uniform"
    )
    overtime = 0
    for k, day in enumerate(days):
        env = fg.make(spec, instance=day)
        run_episode(env, NearestNeighbor(), seed=k)
        assert validate(env.event_log, env.problem_spec, day, scenario=env.scenario).ok
        overtime += bool(env.event_log.of_kind("initial_overtime"))
    assert overtime >= 3


def local_solomon(customers: int | None = None):
    import dataclasses

    base = read_instance(FIXTURES / "tiny_solomon.txt")
    if customers is None:
        return base
    keep = customers + 1
    assert (
        base.time_windows is not None and base.service_times is not None and base.coords is not None
    )
    return dataclasses.replace(
        base,
        coords=base.coords[:keep],
        demands=base.demands[:keep],
        time_windows=base.time_windows[:keep],
        service_times=base.service_times[:keep],
    )


def test_tas_instances_follow_section_5(tmp_path, monkeypatch):
    tas = example_module("16_vrptw_tas")
    base = local_solomon()
    monkeypatch.setattr(tas, "solomon", lambda name: base)
    for cv, (alpha, lam) in {"0.25": (16, 0.0625), "1": (1, 1), "4": (0.0625, 16)}.items():
        inst = tas.tas_instance(base, cv)
        assert inst.vehicle_types[0]["capacity"] == 200.0
        assert inst.extra["max_route_duration"] == 480.0
        params = inst.distributions["travel_time"]
        assert (params["shape"], params["scale"]) == (alpha, lam)
        assert params["shape"] * params["scale"] == 1.0  # expected travel time = distance
        assert 1 / np.sqrt(params["shape"]) == pytest.approx(float(cv))
    tas.write_tas(tmp_path, ["r101"])
    env = fg.make_preset("tas2012-vrpsttw", instances=tmp_path, instance="r101", cv="4")
    assert isinstance(env, fg.FleetGymEnv)
    run_episode(env, NearestNeighbor(), seed=0)
    assert validate(env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario).ok
    assert env.event_log.of_kind("reveal")  # travel times revealed on arrival


def test_errico_instances_follow_section_5_1(tmp_path, monkeypatch):
    errico = example_module("17_vrptw_errico")
    assert len(errico.paper_instances()) == 85
    base = local_solomon(4)
    monkeypatch.setattr(errico, "solomon", lambda name, customers: base)
    expected = {
        "basic": (8.0, 10.0, 12.0),
        "low_probability": (8.0, 10.0, 12.0),
        "large_support": (5.0, 10.0, 15.0),
        "positive_skewed": (5.0, 7.0, 15.0),
    }
    for family, (low, mode, high) in expected.items():
        inst = errico.errico_instance(base, family)
        params = inst.distributions["service_time"]
        assert (params["low"], params["mode"], params["high"], params["step"]) == (
            low,
            mode,
            high,
            0.1,
        )
        assert np.all(inst.demands == 0)
    assert errico.FAMILIES["large_support"]["c"] == (45.0, 90.0, 135.0)
    assert errico.ALPHA == {
        "basic": 0.95,
        "low_probability": 0.85,
        "large_support": 0.95,
        "positive_skewed": 0.95,
    }
    errico.write_errico(tmp_path, ["r101-25"], families=["large_support"])
    env = fg.make_preset(
        "errico2018-vrptwst", instances=tmp_path, family="large_support", instance="r101-25"
    )
    assert isinstance(env, fg.FleetGymEnv)
    run_episode(env, NearestNeighbor(), seed=0)
    assert validate(env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario).ok
