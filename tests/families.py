"""Registered families with the test suite's instance sets (``tests/fixtures/sets``).

FleetGym ships no instance generators and no data (ADR 0023): a family needs instances. The tests
use small committed sets built by ``tests/fixtures/sets/build.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import fleetgym as fg
from fleetgym.envs.ev_env import EVShortestPathEnv
from fleetgym.spec import ProblemSpec, load_spec

SETS = Path(__file__).resolve().parent / "fixtures" / "sets"
#: The set each family runs on by default.
FAMILY_SETS = {
    "CVRP-v1": "cvrp",
    "VRPTW-v1": "vrptw",
    "SVRP-v1": "svrp",
    "DVRP-v1": "dvrp",
    "EVSPP-v1": "evspp",
}


def instance_source(name: str) -> dict[str, Any]:
    """A files source over the test set ``name``."""
    return {"type": "files", "paths": [str(SETS / name)]}


def spec_for(
    env_id: str,
    *,
    instances: str | None = None,
    overrides: Sequence[str] = (),
    **changes: Mapping[str, Any] | Any,
) -> ProblemSpec:
    """``env_id`` extended with a test instance set (its default set unless ``instances``)."""
    data: dict[str, Any] = {
        "extends": env_id,
        "instance_source": instance_source(instances or FAMILY_SETS[env_id]),
        **changes,
    }
    return load_spec(data, overrides=list(overrides))


def make_env(env_id: str, *, instances: str | None = None, **kwargs: Any) -> fg.FleetGymEnv:
    """`fleetgym.make_any` of ``env_id`` on a test instance set.

    Typed as `FleetGymEnv`: for ``EVSPP-v1`` it is an `EVShortestPathEnv`, which has the surface the
    generic tests over registered configs use (narrow with `ev` for the rest).
    """
    return cast("fg.FleetGymEnv", fg.make_any(spec_for(env_id, instances=instances), **kwargs))


def ev(env: object) -> EVShortestPathEnv:
    """``env`` as an `EVShortestPathEnv`."""
    assert isinstance(env, EVShortestPathEnv)
    return env


#: Multi-agent CVRP on the fixed-fleet test set (formerly the MA-CVRP-v0 config).
MA_CVRP = {
    "fleet": {"sizing": "fixed"},
    "decisions": {"interface": "next_node", "epochs": "on_event"},
    "objective": {"distance": 1.0, "unserved": 100.0},
    "agents": {"mode": "multi", "api": "parallel", "reward": "global", "conflict": "priority"},
}
#: Multi-agent dynamic VRP with next_node and wait (formerly the MA-DVRP-v0 config).
MA_DVRP = {
    "dynamics": {"max_requests": 250, "cutoff": 384},
    "decisions": {"interface": "next_node", "epochs": "on_event"},
    "objective": {"distance": 1.0, "unserved": 50.0},
    "agents": {"mode": "multi", "api": "parallel", "reward": "global", "conflict": "priority"},
}


def ma_cvrp(**changes: Any) -> ProblemSpec:
    """Multi-agent CVRP on the fixed-fleet test set."""
    return spec_for("CVRP-v1", instances="fleet", **{**MA_CVRP, **changes})


def ma_dvrp(**changes: Any) -> ProblemSpec:
    """Multi-agent dynamic VRP on the dynamic test set."""
    return spec_for("DVRP-v1", **{**MA_DVRP, **changes})


def first_instance(env_id: str, instances: str | None = None):
    """The first instance of a family's test set (for APIs that take one instance)."""
    from fleetgym.data import read_instance_set

    return read_instance_set(SETS / (instances or FAMILY_SETS[env_id])).load(0)
