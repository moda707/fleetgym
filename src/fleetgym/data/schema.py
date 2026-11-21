"""Instance schemas: the typed fields an instance may carry (ADR 0023).

A schema lists fields at four levels:

* ``global``: one scalar per instance (a duration limit, a reload time);
* ``node``: one value per node (coordinates, demands, deadlines);
* ``arc``: one value per ordered pair of nodes (an explicit distance matrix);
* ``vehicle_type``: one value per vehicle type (capacity, count).

Problem specs say which features exist and how they behave; instances carry every number, in the
fields of their schema. FleetGym's core schema is ``fleetgym/core/1``. A user schema extends it in
YAML::

    schema: my_lab/rural-parcels/1
    extends: fleetgym/core/1
    node:
      village: {type: int, min: 0, description: Village of the customer}
    global:
      courier_price: {type: float, min: 0, unit: money}

and is loaded with `load_schema` (or registered with `register_schema`). Field names starting
with ``_`` are reserved for FleetGym's internal use.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import yaml

from fleetgym._hashing import content_hash
from fleetgym.errors import InstanceError

__all__ = [
    "CORE_SCHEMA",
    "LEVELS",
    "FieldSpec",
    "InstanceSchema",
    "build_schema",
    "get_schema",
    "load_schema",
    "register_schema",
    "schema_ids",
]

Level = Literal["global", "node", "arc", "vehicle_type"]
LEVELS: tuple[Level, ...] = ("global", "node", "arc", "vehicle_type")
FieldType = Literal["float", "int", "bool", "str"]
Unit = Literal["time", "distance", "load", "money", "count", "none"]


@dataclass(frozen=True)
class FieldSpec:
    """One field of an instance schema.

    Attributes
    ----------
    type
        ``float``, ``int``, ``bool`` or ``str``.
    required
        Instances of the schema must have the field.
    min, max
        Inclusive bounds for numbers (``None``: unbounded).
    choices
        Allowed values for strings.
    unit
        What the number measures, for documentation and consistency checks.
    description
        One line for people reading the schema.
    """

    type: FieldType = "float"
    required: bool = False
    min: float | None = None
    max: float | None = None
    choices: tuple[str, ...] | None = None
    unit: Unit = "none"
    description: str = ""

    def check(self, values: Any, where: str) -> list[str]:
        """Problems with ``values`` (a scalar or an array) for this field, as messages."""
        array = np.asarray(values, dtype=object if self.type == "str" else None)
        flat = array.ravel()
        if self.type == "str":
            bad = [v for v in flat if not isinstance(v, str)]
            if bad:
                return [f"{where}: expected text, got {bad[0]!r}"]
            if self.choices is not None:
                wrong = sorted({str(v) for v in flat} - set(self.choices))
                if wrong:
                    return [f"{where}: {wrong[0]!r} is not one of {', '.join(self.choices)}"]
            return []
        if self.type == "bool":
            if not all(isinstance(v, bool | np.bool_) for v in flat):
                return [f"{where}: expected true or false"]
            return []
        try:
            numbers = np.asarray(flat, dtype=float)
        except (TypeError, ValueError):
            return [f"{where}: expected numbers"]
        problems = []
        if not np.all(np.isfinite(numbers)):
            problems.append(f"{where}: values must be finite")
        elif self.type == "int" and not np.all(numbers == np.round(numbers)):
            problems.append(f"{where}: values must be integers")
        if self.min is not None and numbers.size and np.nanmin(numbers) < self.min:
            problems.append(f"{where}: values must be at least {self.min}")
        if self.max is not None and numbers.size and np.nanmax(numbers) > self.max:
            problems.append(f"{where}: values must be at most {self.max}")
        return problems

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form (for hashing and YAML)."""
        out: dict[str, Any] = {"type": self.type, "required": self.required, "unit": self.unit}
        for key in ("min", "max", "choices", "description"):
            value = getattr(self, key)
            if value not in (None, ""):
                out[key] = list(value) if isinstance(value, tuple) else value
        return out


@dataclass(frozen=True)
class InstanceSchema:
    """A named, versioned set of instance fields at the four levels.

    Attributes
    ----------
    id
        ``<namespace>/<name>/<version>``, for example ``fleetgym/core/1``.
    fields
        Field specs by level and name, inherited fields included.
    extends
        The schema this one extends, if any.
    """

    id: str
    fields: Mapping[Level, Mapping[str, FieldSpec]]
    extends: str | None = None
    source: str | None = field(default=None, compare=False)

    def field(self, level: Level, name: str) -> FieldSpec | None:
        """The spec of ``name`` at ``level``, or ``None`` if the schema does not declare it."""
        return self.fields.get(level, {}).get(name)

    def names(self, level: Level) -> tuple[str, ...]:
        """Field names at ``level``, in declaration order."""
        return tuple(self.fields.get(level, {}))

    def level_of(self, name: str) -> Level | None:
        """The level that declares ``name``, if any."""
        return next((lv for lv in LEVELS if name in self.fields.get(lv, {})), None)

    def content_hash(self) -> str:
        """Hash of the id and every field spec."""
        data = {
            "id": self.id,
            "fields": {
                lv: {n: f.to_dict() for n, f in fs.items()} for lv, fs in self.fields.items()
            },
        }
        return content_hash(data)

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form with every inherited field (for documentation)."""
        out: dict[str, Any] = {"schema": self.id}
        for level in LEVELS:
            fields = self.fields.get(level, {})
            if fields:
                out[level] = {name: spec.to_dict() for name, spec in fields.items()}
        return out


def _fields(raw: Mapping[str, Any], where: str) -> dict[str, FieldSpec]:
    out = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not name or name.startswith("_"):
            raise InstanceError(
                f"{where}: invalid field name {name!r} (names starting with _ are reserved)"
            )
        if not isinstance(spec, Mapping):
            raise InstanceError(f"{where}.{name}: expected a mapping of field properties")
        unknown = set(spec) - {"type", "required", "min", "max", "choices", "unit", "description"}
        if unknown:
            raise InstanceError(f"{where}.{name}: unknown properties {sorted(unknown)}")
        kind = spec.get("type", "float")
        if kind not in ("float", "int", "bool", "str"):
            raise InstanceError(f"{where}.{name}.type: {kind!r} is not float, int, bool or str")
        unit = spec.get("unit", "none")
        if unit not in ("time", "distance", "load", "money", "count", "none"):
            raise InstanceError(f"{where}.{name}.unit: unknown unit {unit!r}")
        bounds = {k: spec.get(k) for k in ("min", "max")}
        for key, value in bounds.items():
            if value is not None and not (isinstance(value, int | float) and math.isfinite(value)):
                raise InstanceError(f"{where}.{name}.{key}: expected a finite number")
        choices = spec.get("choices")
        out[name] = FieldSpec(
            type=kind,
            required=bool(spec.get("required", False)),
            min=bounds["min"],
            max=bounds["max"],
            choices=tuple(choices) if choices is not None else None,
            unit=unit,
            description=str(spec.get("description", "")),
        )
    return out


_SCHEMAS: dict[str, InstanceSchema] = {}


def build_schema(raw: Mapping[str, Any], *, source: str | None = None) -> InstanceSchema:
    """Build a schema from its plain-data form (see the module docstring).

    Raises
    ------
    InstanceError
        If the schema is malformed, extends an unknown schema, or redeclares an inherited field.
    """
    where = source or "<schema>"
    if "schema" not in raw or not isinstance(raw["schema"], str):
        raise InstanceError(f"{where}: needs `schema: <namespace>/<name>/<version>`")
    schema_id = raw["schema"]
    if schema_id.count("/") != 2:
        raise InstanceError(
            f"{where}: schema id {schema_id!r} must be <namespace>/<name>/<version>"
        )
    unknown = set(raw) - {"schema", "extends", "description", *LEVELS}
    if unknown:
        raise InstanceError(
            f"{where}: unknown keys {sorted(unknown)} (levels: {', '.join(LEVELS)})"
        )
    parent_id = raw.get("extends")
    fields: dict[Level, dict[str, FieldSpec]] = {lv: {} for lv in LEVELS}
    if parent_id is not None:
        parent = get_schema(parent_id)
        for level in LEVELS:
            fields[level].update(parent.fields.get(level, {}))
    for level in LEVELS:
        new = _fields(raw.get(level) or {}, f"{where}: {level}")
        taken = [name for name in new if any(name in fields[lv] for lv in LEVELS)]
        if taken:
            raise InstanceError(f"{where}: {level} field(s) {taken} are already declared")
        fields[level].update(new)
    frozen: dict[Level, Mapping[str, FieldSpec]] = {
        lv: MappingProxyType(dict(fs)) for lv, fs in fields.items()
    }
    return InstanceSchema(schema_id, MappingProxyType(frozen), parent_id, source)


def register_schema(schema: InstanceSchema | Mapping[str, Any]) -> InstanceSchema:
    """Register a schema so instances can name it; returns the registered schema.

    Raises
    ------
    InstanceError
        If another schema with the same id is registered.
    """
    built = schema if isinstance(schema, InstanceSchema) else build_schema(schema)
    existing = _SCHEMAS.get(built.id)
    if existing is not None:
        if existing.content_hash() == built.content_hash():
            return existing
        raise InstanceError(f"schema {built.id!r} is already registered with other fields")
    _SCHEMAS[built.id] = built
    return built


def load_schema(path: str | Path) -> InstanceSchema:
    """Read a schema YAML file and register it."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InstanceError(f"{path}: cannot read the schema: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise InstanceError(f"{path}: a schema is a mapping")
    return register_schema(build_schema(raw, source=str(path)))


def get_schema(schema_id: str) -> InstanceSchema:
    """A registered schema by id.

    Raises
    ------
    InstanceError
        If no schema has that id.
    """
    if schema_id not in _SCHEMAS:
        raise InstanceError(
            f"unknown instance schema {schema_id!r}; registered: {', '.join(sorted(_SCHEMAS))}. "
            "Load a user schema with fleetgym.data.load_schema(path)"
        )
    return _SCHEMAS[schema_id]


def schema_ids() -> tuple[str, ...]:
    """Ids of the registered schemas, sorted."""
    return tuple(sorted(_SCHEMAS))


CORE_SCHEMA = register_schema(
    {
        "schema": "fleetgym/core/1",
        "description": "FleetGym's core instance fields",
        "global": {
            "max_route_duration": {
                "min": 0,
                "unit": "time",
                "description": "Longest route allowed",
            },
            "shift_start": {"unit": "time", "description": "Start of the working day"},
            "shift_end": {"unit": "time", "description": "End of the working day"},
            "reload_time": {"min": 0, "unit": "time", "description": "Time to reload at the depot"},
            "edge_weight_type": {"type": "str", "description": "VRPLIB distance convention"},
        },
        "node": {
            "kind": {
                "type": "str",
                "required": True,
                "choices": ["depot", "customer"],
                "description": "Depots come first",
            },
            "x": {"unit": "distance", "description": "First coordinate"},
            "y": {"unit": "distance", "description": "Second coordinate"},
            "demand": {"required": True, "min": 0, "unit": "load", "description": "Demand"},
            "service_time": {"min": 0, "unit": "time", "description": "Service duration"},
            "tw_open": {"unit": "time", "description": "Earliest service start"},
            "tw_close": {"unit": "time", "description": "Latest service start"},
            "deadline": {"unit": "time", "description": "Latest service start before lateness"},
        },
        "arc": {
            "distance": {"min": 0, "unit": "distance", "description": "Explicit distance"},
            "travel_time": {"min": 0, "unit": "time", "description": "Explicit travel time"},
        },
        "vehicle_type": {
            "capacity": {"min": 0, "unit": "load", "description": "Capacity of each vehicle"},
            "count": {"type": "int", "min": 1, "unit": "count", "description": "Vehicles"},
            "fixed_cost": {"min": 0, "unit": "money", "description": "Cost of using a vehicle"},
            "name": {"type": "str", "description": "Vehicle type name"},
        },
    }
)
"""The core schema ``fleetgym/core/1``: the fields FleetGym's built-in features read."""
