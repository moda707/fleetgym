"""Generic plugin registries and entry-point loading (layer L0).

FleetGym is extended through named registries: constraints, distributions, recourse rules,
action interfaces and so on. A registry maps a name used in YAML configs to an object (usually
a class). Registering the same name twice is an error; there is no silent override.

Third-party packages hook in through two entry-point groups:

``fleetgym.plugins``
    Each entry point names a module. Importing it registers the package's plugins.
``fleetgym.configs``
    Each entry point names a package (or a callable returning a directory) that contains
    YAML configs. See `fleetgym.spec.registry`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib import import_module
from importlib.metadata import entry_points
from typing import overload

from fleetgym.errors import FleetGymError

__all__ = [
    "PLUGIN_ENTRY_POINT_GROUP",
    "PluginError",
    "Registry",
    "load_plugins",
]

PLUGIN_ENTRY_POINT_GROUP = "fleetgym.plugins"

_plugins_loaded = False


class PluginError(FleetGymError):
    """A plugin name is unknown, or registered twice."""


class Registry[T]:
    """A named collection of plugins of one kind.

    Parameters
    ----------
    kind
        Human-readable kind used in error messages, for example ``"constraint"``.

    Examples
    --------
    >>> SHAPES: Registry[type] = Registry("shape")
    >>> @SHAPES.register("square")
    ... class Square: ...
    >>> SHAPES.get("square") is Square
    True
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, T] = {}

    @overload
    def register(self, name: str) -> Callable[[T], T]: ...
    @overload
    def register(self, name: str, obj: T) -> T: ...
    def register(self, name: str, obj: T | None = None) -> T | Callable[[T], T]:
        """Register ``obj`` under ``name``, directly or as a decorator.

        Parameters
        ----------
        name
            Name used in configs. Must be unique within this registry.
        obj
            The object to register. If omitted, a decorator is returned.

        Raises
        ------
        PluginError
            If ``name`` is already registered.
        """
        if obj is None:

            def decorator(target: T) -> T:
                self._add(name, target)
                return target

            return decorator
        self._add(name, obj)
        return obj

    def _add(self, name: str, obj: T) -> None:
        if not name or not isinstance(name, str):
            raise PluginError(f"{self.kind} names must be non-empty strings, got {name!r}")
        if name in self._entries:
            raise PluginError(
                f"{self.kind} {name!r} is already registered "
                f"(existing: {self._entries[name]!r}, new: {obj!r})"
            )
        self._entries[name] = obj

    def get(self, name: str) -> T:
        """Return the plugin registered under ``name``.

        Raises
        ------
        PluginError
            If no plugin has that name; the message lists the known names.
        """
        if name not in self._entries:
            load_plugins()
        try:
            return self._entries[name]
        except KeyError:
            known = ", ".join(sorted(self._entries)) or "none"
            raise PluginError(f"unknown {self.kind} {name!r}; known: {known}") from None

    def names(self) -> tuple[str, ...]:
        """Return all registered names, sorted."""
        load_plugins()
        return tuple(sorted(self._entries))

    def items(self) -> Iterator[tuple[str, T]]:
        """Iterate over ``(name, plugin)`` pairs, sorted by name."""
        for name in self.names():
            yield name, self._entries[name]

    def __contains__(self, name: object) -> bool:
        if name not in self._entries:
            load_plugins()
        return name in self._entries

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {len(self._entries)} entries)"


def load_plugins() -> None:
    """Import every module registered in the ``fleetgym.plugins`` entry-point group.

    Runs once per process; later calls do nothing. A plugin that fails to import raises
    `PluginError` naming the entry point, rather than being skipped.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for ep in entry_points(group=PLUGIN_ENTRY_POINT_GROUP):
        try:
            import_module(ep.value.split(":", 1)[0])
        except Exception as exc:
            raise PluginError(
                f"plugin entry point {ep.name!r} ({ep.value}) failed to import: {exc}"
            ) from exc
