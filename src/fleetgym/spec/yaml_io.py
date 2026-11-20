"""YAML reading and writing for configs.

The loader is PyYAML's safe loader with two changes:

* duplicate keys in a mapping are an error (PyYAML silently keeps the last one);
* numbers in exponent notation such as ``1e-6`` or ``1.5e2`` are floats (YAML 1.2 behavior;
  PyYAML follows YAML 1.1, which needs a dot and a signed exponent, and reads them as strings).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from fleetgym.errors import ConfigError

__all__ = ["dump_yaml", "load_yaml", "load_yaml_file"]


class _ConfigLoader(yaml.SafeLoader):
    """Safe loader that rejects duplicate keys and reads ``1e-6`` as a float."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)  # pyright: ignore[reportArgumentType]
            if key in seen:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


_ConfigLoader.yaml_implicit_resolvers = {
    key: list(value) for key, value in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def load_yaml(text: str, *, source: str = "<string>") -> Any:
    """Parse YAML text with the config loader.

    Raises
    ------
    ConfigError
        On a syntax error or a duplicate key, with the line number.
    """
    try:
        return yaml.load(text, Loader=_ConfigLoader)  # safe loader subclass
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark
        where = f"line {mark.line + 1}, column {mark.column + 1}: " if mark is not None else ""
        raise ConfigError(f"invalid YAML: {where}{exc.problem}", source=source) from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}", source=source) from None


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Read a config file whose top level is a mapping.

    Raises
    ------
    ConfigError
        If the file is missing, is not valid YAML, or its top level is not a mapping.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError("config file not found", source=path) from None
    data = load_yaml(text, source=str(path))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError("the top level of a config must be a mapping", source=path)
    return data


def dump_yaml(data: Any) -> str:
    """Write data as block-style YAML, keeping key order."""
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
