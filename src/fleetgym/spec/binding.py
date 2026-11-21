"""Resolve `from_instance` references against an instance into a `BoundProblem`.

Binding happens when an environment is reset with an instance. It

1. resolves every ``from_instance`` value in the spec to the instance's data, using the explicit
   key (``{from_instance: demands}``) or the field's default key (`DEFAULT_INSTANCE_KEYS`);
2. resolves symbolic values (``route_duration.max: depot_due_time``);
3. checks that the instance has what the spec needs (coordinates for a Euclidean network,
   time windows for the time-window constraint, a capacity for every vehicle type, ...);
4. refuses specs that cannot run in this version (``meta.status: planned``).

Errors name the config file and key path.
"""

from __future__ import annotations

import dataclasses
import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from fleetgym.data.download import fetch, load_collection
from fleetgym.data.generators import generate
from fleetgym.data.instance import Instance
from fleetgym.data.instance_set import read_instance_set
from fleetgym.data.readers import read_instance
from fleetgym.errors import ConfigError, InstanceError, NotInScopeError
from fleetgym.spec.models import (
    CollectionSource,
    DistSpec,
    FilesSource,
    GeneratorSource,
    ProblemSpec,
)
from fleetgym.spec.models.base import FROM_INSTANCE

__all__ = [
    "DEFAULT_INSTANCE_KEYS",
    "BoundProblem",
    "BoundVehicleType",
    "bind",
    "collection_instance",
    "instance_from_source",
    "source_files",
]

#: Default instance key for a bare ``from_instance`` at a YAML path (``*`` matches one segment).
DEFAULT_INSTANCE_KEYS: tuple[tuple[str, str], ...] = (
    ("fleet.types.*.capacity", "capacity"),
    ("fleet.types.*.count", "n_vehicles"),
    ("uncertainty.demand.*", "demands"),
    ("uncertainty.service_time.*", "service_times"),
    ("constraints.route_duration.max", "max_route_duration"),
    ("fleet.reload_time", "reload_time"),
    ("constraints.outsourcing.value", "outsourcing_costs"),
    ("dynamics.request_class.location.data", "customer_coords"),
)

#: Symbolic values accepted at specific paths, and the instance key they stand for.
SYMBOLIC_VALUES: tuple[tuple[str, str, str], ...] = (
    ("constraints.route_duration.max", "depot_due_time", "depot_due_time"),
)

_SKIPPED_SECTIONS = ("meta", "ev")
#: Paths whose ``from_instance`` is bound by dedicated code (vehicle types from the instance).
_BOUND_ELSEWHERE = frozenset({"fleet.types", "fleet.shift"})


@dataclass(frozen=True)
class BoundVehicleType:
    """A vehicle type with every instance reference resolved."""

    name: str
    capacity: float | None
    count: int | None
    fixed_cost: float


@dataclass(frozen=True, eq=False)
class BoundProblem:
    """A spec bound to one instance.

    Attributes
    ----------
    spec
        The spec.
    instance
        The instance.
    refs
        Resolved values by dotted YAML path, for every ``from_instance`` and symbolic value.
    vehicle_types
        Vehicle types with capacities and counts resolved.
    """

    spec: ProblemSpec
    instance: Instance
    refs: Mapping[str, Any]
    vehicle_types: tuple[BoundVehicleType, ...]
    distributions: Mapping[str, DistSpec] = field(default_factory=dict)
    """The distribution of each uncertainty source: the spec's, or else the instance's."""
    shift: tuple[float, float] | None = None
    """The working day (start, end): the spec's, or the instance's with ``from_instance``."""

    @property
    def spec_hash(self) -> str:
        """Content hash of the spec."""
        return self.spec.content_hash()

    @property
    def instance_hash(self) -> str:
        """Content hash of the instance."""
        return self.instance.content_hash()

    def value(self, path: str) -> Any:
        """Return the value at a dotted YAML path, with instance references resolved.

        Raises
        ------
        KeyError
            If the spec has no value at ``path``.
        """
        if path in self.refs:
            return self.refs[path]
        node: Any = self.spec.to_dict()
        for segment in path.split("."):
            if isinstance(node, list):
                node = node[int(segment)]
            elif isinstance(node, Mapping) and segment in node:
                node = node[segment]
            else:
                raise KeyError(path)
        return node


def _walk(node: Any, path: tuple[str, ...]) -> list[tuple[str, Any]]:
    """Every ``(dotted path, value)`` whose value is an instance reference or a string."""
    found: list[tuple[str, Any]] = []
    if isinstance(node, Mapping):
        if set(node) == {FROM_INSTANCE}:
            return [(".".join(path), node)]
        for key, value in node.items():
            found += _walk(value, (*path, str(key)))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += _walk(value, (*path, str(i)))
    elif isinstance(node, str):
        found.append((".".join(path), node))
    return found


def _default_key(path: str) -> str | None:
    for pattern, key in DEFAULT_INSTANCE_KEYS:
        if fnmatch.fnmatchcase(path, pattern):
            return key
    return None


def _config_error(spec: ProblemSpec, path: str, message: str) -> ConfigError:
    return ConfigError(message, source=spec.source, key_path=path)


def bind(spec: ProblemSpec, instance: Instance) -> BoundProblem:
    """Bind ``spec`` to ``instance``.

    Raises
    ------
    NotInScopeError
        If the spec is planned or uses reserved features.
    ConfigError
        If a reference cannot be resolved or the instance lacks something the spec needs; the
        message names the config path and the instance.
    """
    if spec.meta.status == "planned":
        raise NotInScopeError(
            f"{spec.id or spec.source or 'spec'} is planned and cannot be run in this version",
            features=spec.reserved_features(),
        )
    data = spec.to_dict()
    refs: dict[str, Any] = {}
    for section in data:
        if section in _SKIPPED_SECTIONS:
            continue
        for path, value in _walk(data[section], (section,)):
            if path in _BOUND_ELSEWHERE:
                continue
            key: str | None
            if isinstance(value, Mapping):
                key = str(value[FROM_INSTANCE])
            elif value == FROM_INSTANCE:
                key = _default_key(path)
                if key is None:
                    raise _config_error(
                        spec,
                        path,
                        "no default instance key for this field; write {from_instance: <key>}",
                    )
            else:
                key = next((k for p, sym, k in SYMBOLIC_VALUES if p == path and sym == value), None)
            if key is None:
                continue
            try:
                refs[path] = instance.get(key)
            except InstanceError as exc:
                raise _config_error(spec, path, str(exc)) from None
    vehicle_types = (
        _instance_vehicle_types(spec, instance)
        if spec.fleet.from_instance
        else tuple(_bind_vehicle_type(spec, i, refs) for i in range(len(spec.fleet.types)))
    )
    _check_compatibility(spec, instance, vehicle_types)
    dists = {
        name: _bind_distribution(spec, instance, name, uncertain.dist, refs)
        for name, uncertain in spec.uncertainty.items()
    }
    return BoundProblem(
        spec=spec,
        instance=instance,
        refs=MappingProxyType(refs),
        vehicle_types=vehicle_types,
        distributions=MappingProxyType(dists),
        shift=_bind_shift(spec, instance),
    )


def _bind_shift(spec: ProblemSpec, instance: Instance) -> tuple[float, float] | None:
    shift = spec.fleet.shift
    if shift is None:
        return None
    if shift != "from_instance":
        return (float(shift[0]), float(shift[1]))
    if "shift_end" not in instance.extra:
        raise _config_error(
            spec, "fleet.shift", f"instance {instance.name!r} has no global field shift_end"
        )
    start, end = float(instance.extra.get("shift_start", 0.0)), float(instance.extra["shift_end"])
    if not start < end:
        raise _config_error(
            spec, "fleet.shift", f"instance {instance.name!r}: shift_start must be before shift_end"
        )
    return (start, end)


def _bind_distribution(
    spec: ProblemSpec,
    instance: Instance,
    name: str,
    dist: DistSpec | None,
    refs: dict[str, Any],
) -> DistSpec:
    """The spec's distribution of ``name``, or the instance's, with field references resolved."""
    path = f"uncertainty.{name}"
    if dist is not None:
        return dist
    raw = instance.distributions.get(name)
    if raw is None:
        raise _config_error(
            spec,
            path,
            f"the spec gives no distribution for {name!r} and instance {instance.name!r} has no "
            f"distributions.{name}",
        )
    fields = set(instance.keys()) | set(instance.extra)
    data: dict[str, Any] = {}
    for key, value in raw.items():
        if (
            key != "dist"
            and isinstance(value, str)
            and (value in _FIELD_ALIASES or value in fields)
        ):
            field_name = _FIELD_ALIASES.get(value, value)
            refs[f"{path}.{key}"] = instance.get(field_name)
            data[key] = {FROM_INSTANCE: field_name}
        else:
            data[key] = list(value) if isinstance(value, tuple) else value
    try:
        return DistSpec.model_validate(data)
    except ValueError as exc:
        raise _config_error(
            spec, path, f"instance {instance.name!r} distributions.{name} is invalid: {exc}"
        ) from None


#: Schema field names that the instance stores under another key.
_FIELD_ALIASES: Mapping[str, str] = MappingProxyType(
    {"demand": "demands", "service_time": "service_times"}
)


def _scalar(spec: ProblemSpec, path: str, value: Any, *, integer: bool) -> float | int:
    array = np.asarray(value)
    if array.ndim != 0 or not np.isfinite(array) or array <= 0:
        raise _config_error(
            spec, path, f"expected a positive number from the instance, got {value!r}"
        )
    if integer:
        if float(array) != int(array):
            raise _config_error(spec, path, f"expected an integer from the instance, got {value!r}")
        return int(array)
    return float(array)


def _instance_vehicle_types(spec: ProblemSpec, instance: Instance) -> tuple[BoundVehicleType, ...]:
    """The instance's vehicle types, checked against the spec's sizing."""
    if not instance.vehicle_types:
        raise _config_error(
            spec, "fleet.types", f"instance {instance.name!r} has no vehicle types (vehicle_types)"
        )
    if spec.fleet.sizing == "unlimited" and len(instance.vehicle_types) != 1:
        raise _config_error(
            spec,
            "fleet.sizing",
            f"instance {instance.name!r} has {len(instance.vehicle_types)} vehicle types; an "
            "unlimited fleet needs exactly one",
        )
    out = []
    for i, vt in enumerate(instance.vehicle_types):
        count = vt.get("count")
        if spec.fleet.sizing != "unlimited" and count is None:
            raise _config_error(
                spec,
                "fleet.sizing",
                f"instance {instance.name!r}: vehicle type {i} has no count, which a "
                f"{spec.fleet.sizing} fleet needs",
            )
        capacity = vt.get("capacity")
        out.append(
            BoundVehicleType(
                name=str(vt.get("name", f"type{i}")),
                capacity=None if capacity is None else float(capacity),
                count=None if spec.fleet.sizing == "unlimited" or count is None else int(count),
                fixed_cost=float(vt.get("fixed_cost", 0.0)),
            )
        )
    return tuple(out)


def _bind_vehicle_type(spec: ProblemSpec, index: int, refs: Mapping[str, Any]) -> BoundVehicleType:
    vt = spec.fleet.explicit_types()[index]
    base = f"fleet.types.{index}"
    capacity: float | None = None
    if f"{base}.capacity" in refs:
        capacity = float(_scalar(spec, f"{base}.capacity", refs[f"{base}.capacity"], integer=False))
    elif vt.capacity is not None:
        capacity = float(vt.capacity)  # pyright: ignore[reportArgumentType]
    count: int | None = None
    if f"{base}.count" in refs:
        count = int(_scalar(spec, f"{base}.count", refs[f"{base}.count"], integer=True))
    elif vt.count is not None:
        count = int(vt.count)  # pyright: ignore[reportArgumentType]
    return BoundVehicleType(name=vt.name, capacity=capacity, count=count, fixed_cost=vt.fixed_cost)


def _check_compatibility(
    spec: ProblemSpec, instance: Instance, vehicle_types: tuple[BoundVehicleType, ...]
) -> None:
    def fail(path: str, message: str) -> ConfigError:
        return _config_error(spec, path, f"instance {instance.name!r}: {message}")

    if spec.network.name == "euclidean" and instance.coords is None:
        raise fail("network", "a euclidean network needs node coordinates")
    if spec.network.name == "matrix" and instance.distance_matrix is None:
        raise fail("network", "a matrix network needs an explicit distance matrix")
    if "capacity" in spec.constraints:
        for i, vt in enumerate(vehicle_types):
            if vt.capacity is None:
                raise fail(f"fleet.types.{i}.capacity", "the capacity constraint needs a capacity")
    if "time_windows" in spec.constraints and instance.time_windows is None:
        raise fail("constraints.time_windows", "the instance has no time windows")
    if instance.n_depots != 1:
        raise fail("fleet", f"{instance.n_depots} depots; this version supports one depot")


def instance_from_source(spec: ProblemSpec, rng: np.random.Generator) -> Instance:
    """Draw an instance from ``spec.instance_source``.

    A generator source generates a new instance from ``rng``. A files source picks one of its files
    uniformly at random with ``rng``; relative paths resolve against the config file's folder.

    Raises
    ------
    ConfigError
        If the spec has no ``instance_source``, or the source cannot produce an instance.
    """
    source = spec.instance_source
    if source is None:
        raise ConfigError(
            "no `instance_source`: pass an instance explicitly, or declare a generator or files",
            source=spec.source,
            key_path="instance_source",
        )
    try:
        if isinstance(source, GeneratorSource):
            params = source.params if source.params is not None else {}
            return generate(source.name, _thaw(params), rng)
        if isinstance(source, CollectionSource):
            return collection_instance(source)
        assert isinstance(source, FilesSource)
        files = source_files(spec)
        return read_instance(files[int(rng.integers(len(files)))], fmt=source.format)
    except InstanceError as exc:
        raise ConfigError(str(exc), source=spec.source, key_path="instance_source") from None


def source_files(spec: ProblemSpec) -> list[Path]:
    """The instance files of a files source: listed files, and every file of listed sets.

    Relative paths resolve against the config file's folder.

    Raises
    ------
    InstanceError
        If an instance set's manifest is invalid.
    """
    source = spec.instance_source
    assert isinstance(source, FilesSource)
    files: list[Path] = []
    for entry in source.paths:
        path = Path(entry)
        if not path.is_absolute() and spec.source is not None and not spec.source.startswith("<"):
            path = Path(spec.source).parent / path
        files += list(read_instance_set(path).paths) if path.is_dir() else [path]
    return files


def collection_instance(source: CollectionSource) -> Instance:
    """Fetch (download and verify, or read from the cache) a collection instance and trim it.

    ``first_customers`` keeps the depot and the first ``n`` customers in file order; ``drop``
    removes time windows and/or service times.

    Raises
    ------
    InstanceError
        If the download fails, or the instance has fewer customers than ``first_customers``.
    """
    path = fetch(source.collection, source.name)
    instance = read_instance(path, fmt=load_collection(source.collection).format)  # pyright: ignore[reportArgumentType]
    keep = instance.n_nodes
    if source.first_customers is not None:
        if source.first_customers > instance.n_customers:
            raise InstanceError(
                f"{source.collection}/{source.name} has {instance.n_customers} customers, fewer "
                f"than first_customers={source.first_customers}"
            )
        keep = instance.n_depots + source.first_customers
    if keep == instance.n_nodes and not source.drop:
        return instance

    def cut(array: Any) -> Any:
        return None if array is None else np.asarray(array)[:keep]

    matrix = instance.distance_matrix
    suffix = f"-n{source.first_customers}" if source.first_customers is not None else ""
    return dataclasses.replace(
        instance,
        name=instance.name + suffix,
        demands=cut(instance.demands),
        coords=cut(instance.coords),
        distance_matrix=None if matrix is None else np.asarray(matrix)[:keep, :keep],
        time_windows=None if "time_windows" in source.drop else cut(instance.time_windows),
        service_times=None if "service_times" in source.drop else cut(instance.service_times),
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value
