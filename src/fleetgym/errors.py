"""Exception hierarchy for FleetGym (layer L0).

Every error raised on purpose by FleetGym is a `FleetGymError`. Configuration errors carry the
file that caused them and the dotted key path inside it, so a message points straight at the
offending line of a YAML file.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "ConfigError",
    "FleetGymError",
    "InfeasibleActionError",
    "InstanceError",
    "NotInScopeError",
    "ScenarioMismatchError",
    "ValidationFailure",
]


class FleetGymError(Exception):
    """Base class of all errors raised on purpose by FleetGym."""


class ConfigError(FleetGymError):
    """A config or spec is invalid.

    Parameters
    ----------
    message
        What is wrong, in plain words.
    source
        The config file (or a description such as ``"<dict>"``) that caused the error.
    key_path
        Dotted path of the offending key inside the config, for example
        ``"fleet.types.0.capacity"``.
    """

    def __init__(
        self,
        message: str,
        *,
        source: str | Path | None = None,
        key_path: str | None = None,
    ) -> None:
        self.message = message
        self.source = str(source) if source is not None else None
        self.key_path = key_path
        super().__init__(self._format())

    def _format(self) -> str:
        location = ""
        if self.source is not None and self.key_path:
            location = f"{self.source}: {self.key_path}: "
        elif self.source is not None:
            location = f"{self.source}: "
        elif self.key_path:
            location = f"{self.key_path}: "
        return f"{location}{self.message}"


class InstanceError(FleetGymError):
    """An instance file is malformed or lacks a field that a spec requires."""


class ScenarioMismatchError(FleetGymError):
    """A scenario was replayed against a spec or instance it was not sampled from."""


class InfeasibleActionError(FleetGymError):
    """An action violates a hard constraint and the invalid-action mode is ``"mask"``."""


class NotInScopeError(FleetGymError):
    """A feature is declared in the design but not implemented in this version."""


class ValidationFailure(FleetGymError):
    """The independent validator rejected a trajectory."""
