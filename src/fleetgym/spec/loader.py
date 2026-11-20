"""Loading configs into `ProblemSpec` objects (implementation plan §3.2).

Pipeline: read YAML → resolve ``extends`` recursively → merge (`fleetgym.spec.merge`) → apply
overrides (`fleetgym.spec.overrides`) → validate with pydantic. Validation errors become one
`ConfigError` that names, for every problem, the file that set the value and the dotted key path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fleetgym.errors import ConfigError
from fleetgym.spec.merge import drop_nulls, merge_configs
from fleetgym.spec.models import ProblemSpec
from fleetgym.spec.overrides import apply_overrides, parse_override
from fleetgym.spec.registry import REGISTRY, ConfigRegistry
from fleetgym.spec.yaml_io import load_yaml_file

__all__ = ["ConfigSource", "Resolved", "load_spec", "resolve", "resolve_raw"]

ConfigSource = str | Path | Mapping[str, Any]
"""An environment ID, a path to a YAML file, or the config itself as a mapping."""

_PATH_SUFFIXES = (".yaml", ".yml")


@dataclass
class Resolved:
    """A config after ``extends`` and overrides, before validation.

    Attributes
    ----------
    data
        The merged config.
    layers
        ``(source, raw data)`` from the root ancestor to the most derived file, then overrides.
        Used to report which file set a value.
    """

    data: dict[str, Any]
    layers: list[tuple[str, Any]] = field(default_factory=list)

    @property
    def source(self) -> str:
        """The most derived config file (or ``<dict>``)."""
        files = [s for s, _ in self.layers if not s.startswith("<override")]
        return files[-1] if files else "<dict>"

    def locate(self, path: Sequence[str | int]) -> str:
        """Return the source that last set ``path`` (or its closest enclosing key)."""
        for depth in range(len(path), 0, -1):
            prefix = tuple(path[:depth])
            for source, raw in reversed(self.layers):
                if _contains(raw, prefix):
                    return source
        return self.source


def _contains(data: Any, path: tuple[str | int, ...]) -> bool:
    node = data
    for segment in path:
        if isinstance(node, Mapping) and segment in node:
            node = node[segment]
        elif isinstance(node, list) and isinstance(segment, int) and segment < len(node):
            node = node[segment]
        else:
            return False
    return True


def _looks_like_path(text: str) -> bool:
    return text.endswith(_PATH_SUFFIXES) or "/" in text or "\\" in text


def _locate_config(ref: str, base_dir: Path | None, registry: ConfigRegistry) -> Path:
    if _looks_like_path(ref):
        path = Path(ref)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        return path.resolve()
    return registry.path(ref)


def _resolve_file(path: Path, registry: ConfigRegistry, stack: tuple[Path, ...]) -> Resolved:
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ConfigError(f"`extends` cycle: {chain}", source=path, key_path="extends")
    data = load_yaml_file(path)
    return _resolve_data(data, str(path), path.parent, registry, (*stack, path))


def _resolve_data(
    data: Mapping[str, Any],
    source: str,
    base_dir: Path | None,
    registry: ConfigRegistry,
    stack: tuple[Path, ...],
) -> Resolved:
    own = dict(data)
    parent_ref = own.pop("extends", None)
    if parent_ref is None:
        return Resolved(data=drop_nulls(own), layers=[(source, own)])
    if not isinstance(parent_ref, str):
        raise ConfigError("`extends` must be an ID or a path", source=source, key_path="extends")
    try:
        parent_path = _locate_config(parent_ref, base_dir, registry)
    except ConfigError as exc:
        raise ConfigError(exc.message, source=source, key_path="extends") from None
    parent = _resolve_file(parent_path, registry, stack)
    return Resolved(data=merge_configs(parent.data, own), layers=[*parent.layers, (source, own)])


def resolve(
    source: ConfigSource,
    *,
    overrides: Sequence[str] = (),
    registry: ConfigRegistry | None = None,
) -> Resolved:
    """Resolve ``extends`` and overrides without validating.

    Parameters
    ----------
    source
        An environment ID, a path to a YAML file, or a config mapping. A mapping's relative
        ``extends`` paths resolve against the current directory.
    overrides
        Dotted-path overrides such as ``"fleet.types.0.count=8"``.
    registry
        Registry used to resolve environment IDs; defaults to the global one.
    """
    registry = registry or REGISTRY
    if isinstance(source, Mapping):
        resolved = _resolve_data(source, "<dict>", Path.cwd(), registry, ())
    else:
        text = str(source)
        if isinstance(source, Path) or _looks_like_path(text):
            path = Path(source).resolve()
        else:
            path = registry.path(text)
        resolved = _resolve_file(path, registry, ())
    resolved.data = _section_order(resolved.data)
    if overrides:
        resolved.data = apply_overrides(resolved.data, overrides)
        for text in overrides:
            override_path, value = parse_override(text)
            layer: Any = value
            for segment in reversed(override_path):
                layer = {segment: layer}
            resolved.layers.append((f"<override {text}>", layer))
    return resolved


def _section_order(data: dict[str, Any]) -> dict[str, Any]:
    """Order top-level keys like `ProblemSpec` fields; unknown keys keep their order at the end."""
    order = list(ProblemSpec.model_fields)
    known = [k for k in order if k in data]
    return {k: data[k] for k in known + [k for k in data if k not in order]}


def resolve_raw(
    source: ConfigSource,
    *,
    overrides: Sequence[str] = (),
    registry: ConfigRegistry | None = None,
) -> dict[str, Any]:
    """Return the merged config (what ``fleetgym show`` prints), without validation."""
    return resolve(source, overrides=overrides, registry=registry).data


def load_spec(
    source: ConfigSource | ProblemSpec,
    *,
    overrides: Sequence[str] = (),
    registry: ConfigRegistry | None = None,
) -> ProblemSpec:
    """Load and validate a config.

    Parameters
    ----------
    source
        An environment ID (``"CVRP-v0"``), a path to a YAML file, a config mapping, or an existing
        `ProblemSpec` (returned as is unless overrides are given).
    overrides
        Dotted-path overrides applied after ``extends``, for example ``["fleet.types.0.count=8"]``.
    registry
        Registry used to resolve environment IDs; defaults to the global one.

    Raises
    ------
    ConfigError
        For any problem, naming the file and key path that caused it.
    """
    if isinstance(source, ProblemSpec):
        if not overrides:
            return source
        source = source.to_dict()
    resolved = resolve(source, overrides=overrides, registry=registry)
    try:
        spec = ProblemSpec.model_validate(resolved.data)
    except ValidationError as exc:
        raise _to_config_error(exc, resolved) from None
    return spec.with_source(resolved.source)


def _key_path(data: Any, loc: Sequence[str | int]) -> tuple[str | int, ...]:
    """Map a pydantic error location onto keys that exist in the raw config.

    Pydantic inserts names that do not exist in the YAML: union members (``float``,
    ``InstanceRef``) and internal wrapper fields. Inside a map such segments are skipped; at a
    scalar the walk stops. A missing final key (a required field) is kept.
    """
    node = data
    path: list[str | int] = []
    for i, segment in enumerate(loc):
        if isinstance(node, Mapping) and segment in node:
            node = node[segment]
            path.append(segment)
        elif isinstance(node, list) and isinstance(segment, int) and segment < len(node):
            node = node[segment]
            path.append(segment)
        elif isinstance(node, Mapping):
            if i == len(loc) - 1:
                path.append(segment)
            continue
        else:
            break
    return tuple(path)


def _clean_message(message: str) -> str:
    for prefix in ("Value error, ", "Assertion failed, "):
        if message.startswith(prefix):
            return message[len(prefix) :]
    return message


def _to_config_error(error: ValidationError, resolved: Resolved) -> ConfigError:
    errors = error.errors()
    located = [(_key_path(resolved.data, err["loc"]), err) for err in errors]
    specific = {path for path, err in located if err["type"] != "too_short"}
    grouped: dict[tuple[str | int, ...], list[str]] = {}
    for path, err in located:
        # A list whose only item failed also reports "too short"; the item error says more.
        if err["type"] == "too_short" and any(
            other[: len(path)] == path and len(other) > len(path) for other in specific
        ):
            continue
        message = _clean_message(err["msg"])
        messages = grouped.setdefault(path, [])
        if message not in messages:
            messages.append(message)
    items = [
        (resolved.locate(path), ".".join(map(str, path)) or None, "; or ".join(msgs))
        for path, msgs in grouped.items()
    ]
    first_source, first_key, first_message = items[0]
    return ConfigError(
        first_message, source=first_source, key_path=first_key, details=tuple(items[1:])
    )
