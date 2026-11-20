"""Registry of environment configs, built by scanning config folders (ADR 0002).

Folders scanned:

* ``fleetgym/configs`` inside the installed package;
* every package or directory registered in the ``fleetgym.configs`` entry-point group. An entry
  point may name a package (its folder is scanned) or a callable returning a directory.

Each ``*.yaml`` file is one environment. Its file stem must equal its ``id``. Two files with the
same ID are an error that names both files.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import resources
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import Literal, cast

from fleetgym.errors import ConfigError
from fleetgym.spec.yaml_io import load_yaml_file

__all__ = ["CONFIG_ENTRY_POINT_GROUP", "REGISTRY", "ConfigEntry", "ConfigRegistry"]

CONFIG_ENTRY_POINT_GROUP = "fleetgym.configs"

Status = Literal["stable", "experimental", "planned"]
_STATUSES: tuple[Status, ...] = ("stable", "experimental", "planned")


@dataclass(frozen=True)
class ConfigEntry:
    """One registered config file."""

    id: str
    path: Path
    status: Status
    origin: str

    @property
    def family(self) -> str:
        """Name of the folder the config lives in, for example ``deterministic``."""
        return self.path.parent.name


def _package_config_dir() -> Path:
    return Path(str(resources.files("fleetgym.configs")))


def _entry_point_dirs() -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for ep in entry_points(group=CONFIG_ENTRY_POINT_GROUP):
        try:
            target = ep.load()
        except Exception as exc:
            raise ConfigError(
                f"config entry point {ep.name!r} ({ep.value}) failed to load: {exc}"
            ) from exc
        if isinstance(target, ModuleType):
            directory = Path(str(resources.files(target)))
        elif callable(target):
            directory = Path(cast(Callable[[], str | Path], target)())
        else:
            raise ConfigError(
                f"config entry point {ep.name!r} must name a package or a callable returning a path"
            )
        found.append((directory, f"entry point {ep.name}"))
    return found


class ConfigRegistry:
    """Index of config files by environment ID.

    Parameters
    ----------
    roots
        ``(directory, origin)`` pairs to scan. ``None`` scans the package configs and every
        ``fleetgym.configs`` entry point. Scanning is lazy and happens once.
    """

    def __init__(self, roots: Iterable[tuple[Path, str]] | None = None) -> None:
        self._roots = list(roots) if roots is not None else None
        self._entries: dict[str, ConfigEntry] | None = None

    def _scan(self) -> dict[str, ConfigEntry]:
        if self._entries is not None:
            return self._entries
        roots = self._roots
        if roots is None:
            roots = [(_package_config_dir(), "fleetgym"), *_entry_point_dirs()]
        entries: dict[str, ConfigEntry] = {}
        for directory, origin in roots:
            for path in sorted(directory.rglob("*.yaml")):
                entry = self._read_entry(path, origin)
                if entry.id in entries:
                    other = entries[entry.id]
                    raise ConfigError(
                        f"environment ID {entry.id!r} is defined twice: {other.path} "
                        f"({other.origin}) and {path} ({origin})",
                        source=path,
                    )
                entries[entry.id] = entry
        self._entries = entries
        return entries

    @staticmethod
    def _read_entry(path: Path, origin: str) -> ConfigEntry:
        data = load_yaml_file(path)
        env_id = data.get("id")
        if not isinstance(env_id, str):
            raise ConfigError("registered configs need an `id`", source=path, key_path="id")
        if env_id != path.stem:
            raise ConfigError(
                f"the file name must match the ID: expected {env_id}.yaml",
                source=path,
                key_path="id",
            )
        meta = data.get("meta") or {}
        status = meta.get("status", "experimental") if isinstance(meta, dict) else "experimental"
        if status not in _STATUSES:
            raise ConfigError(
                f"unknown status {status!r}; use one of {', '.join(_STATUSES)}",
                source=path,
                key_path="meta.status",
            )
        return ConfigEntry(id=env_id, path=path, status=status, origin=origin)

    def ids(self, *, include_planned: bool = False) -> tuple[str, ...]:
        """Return registered IDs, sorted; planned configs only if ``include_planned``."""
        return tuple(
            sorted(
                env_id
                for env_id, entry in self._scan().items()
                if include_planned or entry.status != "planned"
            )
        )

    def entries(self, *, include_planned: bool = True) -> tuple[ConfigEntry, ...]:
        """Return registered entries, sorted by ID."""
        return tuple(self._scan()[i] for i in self.ids(include_planned=include_planned))

    def entry(self, env_id: str) -> ConfigEntry:
        """Return the entry of ``env_id``.

        Raises
        ------
        ConfigError
            If the ID is not registered; the message lists close matches.
        """
        entries = self._scan()
        if env_id not in entries:
            import difflib

            close = difflib.get_close_matches(env_id, entries, n=3)
            hint = f"; did you mean {', '.join(close)}?" if close else ""
            raise ConfigError(f"unknown environment ID {env_id!r}{hint}")
        return entries[env_id]

    def path(self, env_id: str) -> Path:
        """Return the config file of ``env_id``."""
        return self.entry(env_id).path

    def status(self, env_id: str) -> Status:
        """Return the status of ``env_id``."""
        return self.entry(env_id).status

    def __contains__(self, env_id: object) -> bool:
        return env_id in self._scan()

    def refresh(self) -> None:
        """Forget the scan so the next query rescans the folders."""
        self._entries = None


REGISTRY = ConfigRegistry()
"""The default registry: package configs plus ``fleetgym.configs`` entry points."""
