"""Distributions given by instances (ADR 0023): the model says what is uncertain, not how."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import fleetgym as fg
from fleetgym.data import Instance, read_instance, write_instance
from fleetgym.errors import ConfigError
from fleetgym.spec import load_spec
from fleetgym.testing.validator import validate

MODEL = {
    "fleet": {"sizing": "unlimited", "types": [{"name": "v", "capacity": "from_instance"}]},
    "network": {"type": "euclidean", "rounding": "none"},
    "constraints": {"capacity": {}},
    "uncertainty": {"demand": {"revealed": "on_arrival"}},
    "recourse": {"capacity_failure": "return_to_depot"},
    "decisions": {"interface": "next_node", "epochs": "sequential"},
    "objective": {"distance": 1.0},
}
GOODSON_MODERATE = {
    "dist": "scaled_categorical",
    "scale": "demand",
    "factors": [0.0, 0.5, 1.0, 1.5, 2.0],
    "probs": [0.05, 0.15, 0.6, 0.15, 0.05],
}


def instance(**distributions) -> Instance:
    return Instance(
        name="line",
        source="test",
        demands=np.array([0.0, 4.0, 6.0, 10.0]),
        coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        capacity=12.0,
        distributions=distributions,
    )


def test_the_instance_gives_the_distribution_and_may_name_its_fields(tmp_path):
    spec = load_spec(MODEL)
    assert spec.uncertainty["demand"].dist is None
    inst = instance(demand=GOODSON_MODERATE)
    env = fg.make(spec, instance=inst)
    draws = []
    for seed in range(300):
        env.reset(seed=seed)
        assert env.scenario is not None
        draws.append(env.scenario["demand"][1:] / inst.demands[1:])
    assert set(np.unique(np.round(draws, 9))) == {0.0, 0.5, 1.0, 1.5, 2.0}
    assert np.mean(draws) == pytest.approx(1.0, abs=0.05)
    np.testing.assert_allclose(env.simulator.info.demands[1:], inst.demands[1:])  # forecast = mean
    back = read_instance(write_instance(inst, tmp_path / "i.json"))
    assert list(back.distributions["demand"]["factors"]) == GOODSON_MODERATE["factors"]
    assert back.content_hash() == inst.content_hash()


def test_two_instances_of_one_model_can_have_different_distributions():
    spec = load_spec(MODEL)
    low = instance(
        demand={**GOODSON_MODERATE, "factors": [0.5, 1, 1.5], "probs": [0.05, 0.9, 0.05]}
    )
    poisson = instance(demand={"dist": "poisson", "mean": "demand"})
    assert low.content_hash() != poisson.content_hash()
    for inst in (low, poisson):
        env = fg.make(spec, instance=inst)
        env.reset(seed=1)
        while not env.simulator.state.terminated:
            env.step(int(np.flatnonzero(env.action_masks())[0]))
        assert validate(env.event_log, spec, inst, scenario=env.scenario).ok


def test_missing_or_invalid_instance_distributions_name_the_source():
    spec = load_spec(MODEL)
    with pytest.raises(ConfigError, match=r"uncertainty\.demand: .*no distributions\.demand"):
        fg.make(spec, instance=instance()).reset(seed=0)
    bad = instance(demand={"dist": "poisson", "average": "demand"})
    with pytest.raises(ConfigError, match=r"distributions\.demand is invalid"):
        fg.make(spec, instance=bad).reset(seed=0)
    unknown = instance(demand={"dist": "lognormalish", "mu": 1})
    with pytest.raises(ConfigError, match="lognormalish"):
        fg.make(spec, instance=unknown).reset(seed=0)


def test_a_spec_distribution_still_wins_over_the_instance():
    spec = load_spec(
        {
            **MODEL,
            "uncertainty": {
                "demand": {"dist": "fixed", "value": "from_instance", "revealed": "on_arrival"}
            },
        }
    )
    env = fg.make(spec, instance=dataclasses.replace(instance(demand=GOODSON_MODERATE)))
    env.reset(seed=3)
    assert env.scenario is not None
    np.testing.assert_array_equal(env.scenario["demand"], [0.0, 4.0, 6.0, 10.0])
