"""The paper examples' instances follow their papers, and the presets run on them (ADR 0023).

Numbers are from the papers: Dastpak, Errico and Jabali (arXiv 2109.10200, Appendices B.1.1 and
B.1.2), Kool et al. (arXiv 1803.08475, Appendix C.1), Vera and Abad (arXiv 1912.03341, Table 1).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

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
