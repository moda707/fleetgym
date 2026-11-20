"""Dotted-path overrides such as ``fleet.types.0.count=8`` (ADR 0006).

The design proposes Hydra for command-line overrides; Hydra is not a FleetGym dependency, so this
small parser covers the same syntax:

* the path is split on dots; a segment that indexes a list must be an integer that exists;
* the value is parsed as YAML (``8`` is an int, ``[1, 2]`` a list, ``{a: 1}`` a map);
* missing intermediate maps are created;
* a ``null`` value deletes the key (the default applies), as in ``extends``;
* an override replaces the value at its path; it does not merge maps.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from fleetgym.errors import ConfigError
from fleetgym.spec.yaml_io import load_yaml

__all__ = ["apply_overrides", "parse_override"]

_SOURCE = "<override>"


def parse_override(text: str) -> tuple[tuple[str | int, ...], Any]:
    """Split ``"a.b.0=value"`` into ``(("a", "b", 0), value)``.

    Raises
    ------
    ConfigError
        If the text has no ``=``, an empty path segment, or an unparsable value.
    """
    if "=" not in text:
        raise ConfigError(f"override {text!r} must look like `key.path=value`", source=_SOURCE)
    raw_path, raw_value = text.split("=", 1)
    segments = raw_path.strip().split(".")
    if not raw_path.strip() or any(not s for s in segments):
        raise ConfigError(f"override {text!r} has an empty key path segment", source=_SOURCE)
    path = tuple(int(s) if s.isdigit() else s for s in segments)
    if raw_value.strip() == "":
        raise ConfigError(f"override {text!r} has no value (use `null` to delete)", source=_SOURCE)
    value = load_yaml(raw_value, source=f"{_SOURCE} {text}")
    return path, value


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return a copy of ``data`` with every override applied in order.

    Raises
    ------
    ConfigError
        If a path indexes a list out of range, uses a non-integer list index, or descends into a
        scalar.
    """
    out = copy.deepcopy(data)
    for text in overrides:
        path, value = parse_override(text)
        _set(out, path, value, text)
    return out


def _set(root: dict[str, Any], path: tuple[str | int, ...], value: Any, text: str) -> None:
    node: Any = root
    for i, segment in enumerate(path[:-1]):
        where = ".".join(map(str, path[: i + 1]))
        if isinstance(node, list):
            node = node[_list_index(node, segment, where, text)]
        elif isinstance(node, dict):
            key = str(segment)
            if key not in node or node[key] is None:
                node[key] = {}
            node = node[key]
        else:
            raise ConfigError(
                f"override {text!r}: cannot descend into a scalar at `{where}`", source=_SOURCE
            )
        if not isinstance(node, dict | list):
            raise ConfigError(
                f"override {text!r}: `{where}` is a scalar, not a map or list", source=_SOURCE
            )
    last = path[-1]
    where = ".".join(map(str, path))
    if isinstance(node, list):
        index = _list_index(node, last, where, text)
        if value is None:
            raise ConfigError(f"override {text!r}: cannot delete a list item", source=_SOURCE)
        node[index] = value
    elif value is None:
        node.pop(str(last), None)
    else:
        node[str(last)] = value


def _list_index(node: list[Any], segment: str | int, where: str, text: str) -> int:
    if not isinstance(segment, int):
        raise ConfigError(
            f"override {text!r}: `{where}` indexes a list and must be an integer", source=_SOURCE
        )
    if segment >= len(node):
        raise ConfigError(
            f"override {text!r}: index {segment} out of range at `{where}` (length {len(node)})",
            source=_SOURCE,
        )
    return segment
