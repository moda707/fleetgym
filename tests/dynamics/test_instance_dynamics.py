"""Dynamic-request numbers given by instances (ADR 0023)."""

from __future__ import annotations

import numpy as np
import pytest

import fleetgym as fg
from fleetgym.data import Instance
from fleetgym.errors import ConfigError
from fleetgym.spec import load_spec
from fleetgym.testing.validator import validate

MODEL = {
    "fleet": {"sizing": "fixed"},
    "network": {"type": "euclidean", "rounding": "none"},
    "constraints": {"capacity": {}},
    "dynamics": {"max_requests": 30, "overflow": "reject", "initial_requests": "none"},
    "decisions": {"interface": "next_node", "epochs": "on_event"},
    "objective": {"distance": 1.0, "unserved": 50.0, "rejected": 50.0},
}


def city(rate: float, cutoff: float) -> Instance:
    rng = np.random.default_rng(0)
    return Instance(
        name=f"city-{rate}",
        source="test",
        demands=np.concatenate([[0.0], np.ones(8)]),
        coords=rng.uniform(0, 50, (9, 2)),
        vehicle_types=({"capacity": 20, "count": 2},),
        extra={"cutoff": cutoff, "shift_end": 200.0},
        distributions={
            "arrivals": {"process": "poisson", "rate": rate},
            "request.location": {"dist": "kde", "data": "customer_coords"},
            "request.demand": {
                "dist": "categorical",
                "values": [1, 2, 4],
                "probs": [0.5, 0.3, 0.2],
            },
            "cancellation": {"dist": "bernoulli", "p": 0.1},
        },
    )


def test_the_instance_gives_arrivals_request_attributes_and_cutoff():
    spec = load_spec({**MODEL, "fleet": {"sizing": "fixed", "shift": "from_instance"}})
    assert spec.dynamics is not None and spec.dynamics.arrivals is None
    busy, quiet = city(rate=0.2, cutoff=120.0), city(rate=0.02, cutoff=60.0)
    counts = {}
    for inst in (busy, quiet):
        env = fg.make(spec, instance=inst, max_nodes=31)
        released = []
        for seed in range(20):
            env.reset(seed=seed)
            assert env.scenario is not None
            release = env.scenario["request.release"]
            arrived = release[np.isfinite(release)][1:]
            assert np.all(arrived < float(inst.extra["cutoff"]))
            assert set(env.scenario["request.demand"][1:][np.isfinite(release[1:])]) <= {
                1.0,
                2.0,
                4.0,
            }
            released.append(arrived.size)
            while not env.simulator.state.terminated:
                env.step(int(np.flatnonzero(env.action_masks())[0]))
            assert validate(env.event_log, spec, inst, scenario=env.scenario).ok
        counts[inst.name] = np.mean(released)
    assert counts["city-0.2"] > 5 * counts["city-0.02"]


def test_missing_arrivals_name_the_instance():
    inst = city(0.1, 100.0)
    bare = Instance(**{**inst.__dict__, "distributions": {}})
    with pytest.raises(ConfigError, match=r"dynamics\.arrivals: .*no distributions\.arrivals"):
        fg.make(load_spec(MODEL), instance=bare, max_nodes=31).reset(seed=0)
