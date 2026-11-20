"""Builders for simulator tests: tiny hand-checkable instances and specs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from fleetgym.data import Instance
from fleetgym.envs import FleetGymEnv
from fleetgym.spec import load_spec

LINE_XS = [0, 2, 5, 9]


def line_instance(
    xs: Sequence[float],
    demands: Sequence[float],
    *,
    capacity: float = 10,
    time_windows: list[tuple[float, float]] | None = None,
    service: list[float] | None = None,
    n_vehicles: int | None = None,
) -> Instance:
    """Nodes on the x axis at ``xs`` (node 0 is the depot), so distances are |x_i - x_j|."""
    return Instance(
        name="line",
        source="test",
        demands=np.asarray(demands, dtype=float),
        coords=np.column_stack([np.asarray(xs, dtype=float), np.zeros(len(xs))]),
        capacity=capacity,
        n_vehicles=n_vehicles,
        time_windows=None if time_windows is None else np.asarray(time_windows, dtype=float),
        service_times=None if service is None else np.asarray(service, dtype=float),
    )


def spec(**changes: Any):
    data: dict[str, Any] = {
        "fleet": {"types": [{"name": "v", "capacity": "from_instance"}]},
        "constraints": {"capacity": {}},
        "objective": {"distance": 1.0},
        "decisions": {"epochs": "sequential"},
    }
    data.update(changes)
    return load_spec(data)


def env_for(instance: Instance, **spec_changes: Any) -> FleetGymEnv:
    return FleetGymEnv(spec(**spec_changes), instance=instance, debug=True)


def play(
    env: FleetGymEnv, actions: list[int], *, seed: int = 0
) -> tuple[float, dict[str, Any], bool]:
    """Reset, play the actions and return (episode return, last info, terminated)."""
    env.reset(seed=seed)
    total, info, terminated = 0.0, {}, False
    for action in actions:
        _, reward, terminated, _, info = env.step(action)
        total += float(reward)
    return total, info, terminated
