"""VRP-VCSD-v0 and VRPSD-Goodson-v0 follow Dastpak, Errico and Jabali (arXiv 2109.10200).

Every number below is from the paper: Section 6.1.1 and Appendix B.1.1 (VRP-VCSD), Section 6.2.1 and
Appendix B.1.2 (VRPSD).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import fleetgym as fg
import fleetgym.spec.binding as binding
from fleetgym.baselines import NearestNeighbor, run_episode
from fleetgym.errors import InstanceError
from fleetgym.spec import load_spec
from fleetgym.spec.binding import collection_instance
from fleetgym.spec.models import CollectionSource
from fleetgym.testing.validator import validate

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "instances"


def test_vcsd_config_matches_the_paper_moderate_instance():
    spec = load_spec("VRP-VCSD-v0")
    (vehicle,) = spec.fleet.types
    assert (vehicle.capacity, vehicle.count) == (50, 3)  # Moderate: m = 3; Q in {25, 50, 75}
    assert spec.constraints["route_duration"].model_dump()["max"] == 221.47
    params = spec.to_dict()["instance_source"]["params"]
    assert len(params["active_zones"]) == 15 and params["grid"] == 5 and params["size"] == 100
    assert params["customers_per_zone"] == {"values": [0, 1, 2, 3], "probs": [0.1, 0.4, 0.4, 0.1]}
    assert params["demand"]["values"] == [5, 10, 15]
    assert spec.objective.scalar_weights() == {"served_demand": -1.0}


def test_vcsd_episodes_sample_the_paper_distributions():
    env = fg.make("VRP-VCSD-v0")
    n_customers, total_expected = [], []
    for seed in range(300):
        env.reset(seed=seed)
        instance = env.base_instance
        expected = instance.demands[1:]
        assert env.scenario is not None
        actual = env.scenario["demand"][1:]
        width = np.where(expected == 5, 4, 5)  # +- 4 when the expected demand is 5
        assert np.all(np.abs(actual - expected) <= width)
        assert np.all(actual >= 1) and np.all(actual == np.round(actual))
        n_customers.append(instance.n_customers)
        total_expected.append(expected.sum())
    # Paper: 1.5 expected customers per zone, 15 zones, average expected demand 10.
    assert np.mean(n_customers) == pytest.approx(22.5, rel=0.05)
    assert np.mean(total_expected) == pytest.approx(225.0, rel=0.05)


def test_vcsd_density_variants_load_as_overrides():
    very_high = load_spec(
        "VRP-VCSD-v0",
        overrides=[
            "instance_source.params.customers_per_zone.values=[4,5,6,7]",
            "fleet.types.0.count=11",
            "fleet.types.0.capacity=25",
            "constraints.route_duration.max=187.29",
        ],
    )
    env = fg.make(very_high)
    assert env.max_nodes == 15 * 7 + 1
    episode = run_episode(env, NearestNeighbor(), seed=1)
    assert validate(env.event_log, env.problem_spec, env.base_instance, scenario=env.scenario).ok
    assert episode.costs["served_demand"] > 0


def _fake_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(binding, "fetch", lambda collection, name: FIXTURES / "tiny_solomon.txt")


def test_collection_source_keeps_the_first_customers_and_drops_fields(monkeypatch):
    _fake_fetch(monkeypatch)
    source = CollectionSource(
        collection="solomon/100",
        name="r101",
        first_customers=2,
        drop=("time_windows", "service_times"),
    )
    instance = collection_instance(source)
    assert instance.n_customers == 2 and instance.name.endswith("-n2")
    assert instance.time_windows is None and instance.service_times is None
    np.testing.assert_array_equal(instance.coords, [[0, 0], [3, 4], [0, 2]])
    np.testing.assert_array_equal(instance.demands, [0, 5, 3])
    kept = collection_instance(CollectionSource(collection="solomon/100", name="r101"))
    assert kept.time_windows is not None and kept.n_customers >= 3
    with pytest.raises(InstanceError, match="fewer than first_customers"):
        collection_instance(
            CollectionSource(collection="solomon/100", name="r101", first_customers=99)
        )
    with pytest.raises(ValueError, match="no instance"):
        CollectionSource(collection="solomon/100", name="r999")


def test_goodson_config_runs_offline_on_a_local_instance(monkeypatch):
    _fake_fetch(monkeypatch)
    spec = load_spec("VRPSD-Goodson-v0")
    instance = collection_instance(
        CollectionSource(
            collection="solomon/100", name="r101", drop=("time_windows", "service_times")
        )
    )
    env = fg.make(spec, instance=instance)
    for seed in range(5):
        run_episode(env, NearestNeighbor(), seed=seed)
        assert validate(env.event_log, spec, instance, scenario=env.scenario).ok
        assert env.scenario is not None
        factors = env.scenario["demand"][1:] / instance.demands[1:]
        assert set(np.round(factors, 9)) <= {0.0, 0.5, 1.0, 1.5, 2.0}


@pytest.mark.network
def test_goodson_instance_is_the_papers_r101_subset():
    env = fg.make("VRPSD-Goodson-v0")
    env.reset(seed=0)
    instance = env.base_instance
    assert instance.n_customers == 75
    assert instance.demands.sum() == 1079.0  # the paper's total expected demand
    assert instance.time_windows is None and instance.service_times is None
    (vehicle,) = env.problem_spec.fleet.types
    assert (vehicle.capacity, vehicle.count) == (50, 11)
    assert env.problem_spec.constraints["route_duration"].model_dump()["max"] == 171.75
