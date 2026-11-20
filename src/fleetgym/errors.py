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
    details
        Further problems found in the same pass, as ``(source, key_path, message)`` triples.
        They are listed one per line after the main message.
    """

    def __init__(
        self,
        message: str,
        *,
        source: str | Path | None = None,
        key_path: str | None = None,
        details: tuple[tuple[str | None, str | None, str], ...] = (),
    ) -> None:
        self.message = message
        self.source = str(source) if source is not None else None
        self.key_path = key_path
        self.details = details
        super().__init__(self._format())

    @staticmethod
    def _location(source: str | None, key_path: str | None) -> str:
        if source is not None and key_path:
            return f"{source}: {key_path}: "
        if source is not None:
            return f"{source}: "
        if key_path:
            return f"{key_path}: "
        return ""

    def _format(self) -> str:
        lines = [f"{self._location(self.source, self.key_path)}{self.message}"]
        lines += [f"  - {self._location(s, k)}{m}" for s, k, m in self.details]
        return "\n".join(lines)


class InstanceError(FleetGymError):
    """An instance file is malformed or lacks a field that a spec requires."""


class ScenarioMismatchError(FleetGymError):
    """A scenario was replayed against a spec or instance it was not sampled from."""


class InfeasibleActionError(FleetGymError):
    """An action violates a hard constraint and the invalid-action mode is ``"mask"``."""


class NotInScopeError(FleetGymError):
    """A feature is declared in the design but not implemented in this version.

    Parameters
    ----------
    message
        What is not available.
    features
        Dotted paths of the reserved features that triggered the error.
    """

    def __init__(self, message: str, *, features: tuple[str, ...] = ()) -> None:
        self.features = features
        detail = f" (reserved features: {', '.join(features)})" if features else ""
        super().__init__(f"{message}{detail}")


class ValidationFailure(FleetGymError):
    """The independent validator rejected a trajectory."""
