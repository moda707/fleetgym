"""Vehicle types, capacities and counts given by instances (ADR 0023)."""

from __future__ import annotations

import numpy as np
import pytest

import fleetgym as fg
from fleetgym.data import Instance, write_instance_set
from fleetgym.errors import ConfigError
from fleetgym.spec import load_spec
from fleetgym.spec.binding import bind

MODEL = {
    "fleet": {"sizing": "fixed"},
    "network": {"type": "euclidean", "rounding": "none"},
    "constraints": {"capacity": {}},
    "decisions": {"interface": "next_node", "epochs": "on_event"},
    "objective": {"distance": 1.0, "unserved": 100.0},
}


def instance(types, n: int = 6, seed: int = 0) -> Instance:
    rng = np.random.default_rng(seed)
    return Instance(
        name=f"heterogeneous-{seed}",
        source="test",
        demands=np.concatenate([[0.0], rng.integers(1, 10, n).astype(float)]),
        coords=rng.uniform(0, 1, (n + 1, 2)),
        vehicle_types=tuple(types),
    )


VERA_VRP20 = [
    {"name": "small", "capacity": 20, "count": 1},
    {"name": "medium", "capacity": 30, "count": 1},
    {"name": "large", "capacity": 35, "count": 1},
]


def test_the_model_says_fixed_the_instance_says_which_vehicles():
    spec = load_spec(MODEL)
    assert spec.fleet.from_instance
    assert spec.to_dict()["fleet"]["types"] == "from_instance"
    problem = bind(spec, instance(VERA_VRP20))
    assert [(v.name, v.capacity, v.count) for v in problem.vehicle_types] == [
        ("small", 20.0, 1),
        ("medium", 30.0, 1),
        ("large", 35.0, 1),
    ]
    env = fg.make(spec, instance=instance(VERA_VRP20))
    env.reset(seed=0)
    assert [v.capacity for v in env.simulator.state.vehicles] == [20.0, 30.0, 35.0]


def test_sizing_rules_are_checked_against_the_instance():
    with pytest.raises(ConfigError, match="has no count, which a fixed fleet needs"):
        bind(load_spec(MODEL), instance([{"capacity": 20}]))
    unlimited = load_spec(
        {
            **MODEL,
            "fleet": {"sizing": "unlimited"},
            "objective": {"distance": 1.0},
            "decisions": {"interface": "next_node", "epochs": "sequential"},
        }
    )
    with pytest.raises(ConfigError, match="needs exactly one"):
        bind(unlimited, instance(VERA_VRP20))
    assert bind(unlimited, instance([{"capacity": 20, "count": 4}])).vehicle_types[0].count is None
    with pytest.raises(ConfigError, match="has no vehicle types"):
        bind(
            load_spec(MODEL),
            Instance(
                name="bare", source="t", demands=np.array([0.0, 1.0]), coords=np.zeros((2, 2))
            ),
        )


def test_multi_agent_size_comes_from_an_instance_set(tmp_path):
    write_instance_set(
        [instance(VERA_VRP20, seed=1), instance(VERA_VRP20[:2], seed=2)],
        tmp_path / "set",
        name="vera",
    )
    spec = load_spec(
        {
            **MODEL,
            "instance_source": {"type": "files", "paths": [str(tmp_path / "set")]},
            "agents": {"mode": "multi", "api": "parallel"},
        }
    )
    env = fg.make_multi(spec, api="parallel")
    assert env.possible_agents == ["vehicle_0", "vehicle_1", "vehicle_2"]


def test_the_working_day_can_come_from_the_instance():
    spec = load_spec({**MODEL, "fleet": {"sizing": "fixed", "shift": "from_instance"}})
    day = instance(VERA_VRP20)
    with pytest.raises(ConfigError, match="no global field shift_end"):
        bind(spec, day)
    timed = Instance(**{**day.__dict__, "extra": {"shift_start": 60.0, "shift_end": 540.0}})
    assert bind(spec, timed).shift == (60.0, 540.0)
    env = fg.make(spec, instance=timed)
    env.reset(seed=0)
    assert env.simulator.state.time == 60.0
