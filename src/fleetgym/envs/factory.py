"""`make` and related factories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, overload

from fleetgym.envs.env import FleetGymEnv
from fleetgym.envs.multi import FleetGymAECEnv, FleetGymParallelEnv
from fleetgym.errors import ConfigError
from fleetgym.spec import ProblemSpec, load_spec

__all__ = ["make", "make_multi", "make_preset"]


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
        Environment ID (``"CVRP-v1"``), path to a YAML config, or a `ProblemSpec`.
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
    spec = load_spec(env, overrides=overrides)
    if spec.agents.mode == "multi":
        raise ConfigError(
            f"{spec.id or 'this spec'} is multi-agent: use fleetgym.make_multi",
            source=spec.source,
            key_path="agents.mode",
        )
    return FleetGymEnv(spec, **env_kwargs)


@overload
def make_multi(
    env: str | Path | ProblemSpec | Mapping[str, Any],
    *,
    api: Literal["aec"],
    overrides: Sequence[str] = (),
    **env_kwargs: Any,
) -> FleetGymAECEnv: ...
@overload
def make_multi(
    env: str | Path | ProblemSpec | Mapping[str, Any],
    *,
    api: Literal["parallel"],
    overrides: Sequence[str] = (),
    **env_kwargs: Any,
) -> FleetGymParallelEnv: ...
@overload
def make_multi(
    env: str | Path | ProblemSpec | Mapping[str, Any],
    *,
    api: None = None,
    overrides: Sequence[str] = (),
    **env_kwargs: Any,
) -> FleetGymAECEnv | FleetGymParallelEnv: ...
def make_multi(
    env: str | Path | ProblemSpec | Mapping[str, Any],
    *,
    api: Literal["aec", "parallel"] | None = None,
    overrides: Sequence[str] = (),
    **env_kwargs: Any,
) -> FleetGymAECEnv | FleetGymParallelEnv:
    """Create a PettingZoo environment (one agent per vehicle).

    Parameters
    ----------
    env
        Environment ID, YAML path, spec or spec dict.
    api
        ``"aec"`` or ``"parallel"``; defaults to the spec's ``agents.api``.
    overrides
        Dotted-path overrides.
    **env_kwargs
        Passed to the environment (``instance``, ``conflict_penalty`` for Parallel, ...).
    """
    spec = load_spec(env, overrides=overrides)
    chosen = api or spec.agents.api
    if chosen == "parallel":
        return FleetGymParallelEnv(spec, **env_kwargs)
    return FleetGymAECEnv(spec, **env_kwargs)


def make_preset(
    name: str,
    *,
    instances: str | Path | None = None,
    env_kwargs: Mapping[str, Any] | None = None,
    **choice: Any,
) -> FleetGymEnv | FleetGymAECEnv | FleetGymParallelEnv:
    """Make the environment of a paper preset at one point of its grid.

    Parameters
    ----------
    name
        The preset, for example ``"dastpak2022-vcsd"`` (see `fleetgym.preset_names`).
    instances
        The folder the paper's example wrote the instances to (presets whose instances are
        generated); benchmark presets download theirs.
    env_kwargs
        Keyword arguments for the environment (``render_mode``, ``max_nodes``, ...).
    **choice
        Grid values, for example ``density="high", capacity=25``; the rest are the defaults.

    Returns
    -------
    FleetGymEnv, or a PettingZoo environment when the preset is multi-agent.
    """
    from fleetgym.spec.presets import load_preset

    spec = load_preset(name).spec(instances, **choice)
    kwargs = dict(env_kwargs or {})
    if spec.agents.mode == "multi":
        if spec.agents.api == "parallel":
            return make_multi(spec, api="parallel", **kwargs)
        return make_multi(spec, api="aec", **kwargs)
    return make(spec, **kwargs)
