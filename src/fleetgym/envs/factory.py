"""`make` and related factories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fleetgym.envs.env import FleetGymEnv
from fleetgym.spec import ProblemSpec, load_spec

__all__ = ["make"]


def make(
    env: str | Path | ProblemSpec | Mapping[str, Any],
    *,
    overrides: Sequence[str] = (),
    **env_kwargs: Any,
) -> FleetGymEnv:
    """Create an environment from an ID, a YAML file or a spec.

    Parameters
    ----------
    env
        Environment ID (``"CVRP-v0"``), path to a YAML config, or a `ProblemSpec`.
    overrides
        Dotted-path overrides, for example ``["fleet.types.0.capacity=80"]``.
    **env_kwargs
        Passed to `FleetGymEnv` (``instance``, ``invalid_action``, ``max_nodes``, ...).

    Raises
    ------
    fleetgym.errors.ConfigError
        If the config is invalid.
    fleetgym.errors.NotInScopeError
        If the config is planned or uses features of a later version.
    """
    return FleetGymEnv(load_spec(env, overrides=overrides), **env_kwargs)
