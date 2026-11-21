"""The immutable `Instance` container and its content hash.

An instance holds the static data of one problem instance: node coordinates, demands, capacity,
time windows and so on. Depots are the first ``n_depots`` nodes. Arrays are read-only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fleetgym._hashing import content_hash
from fleetgym.errors import InstanceError

__all__ = ["Instance", "freeze_array"]

FloatArray = NDArray[np.float64]

#: Instance fields available through `Instance.get`.
FIELD_KEYS = (
    "capacity",
    "coords",
    "demands",
    "distance_matrix",
    "n_vehicles",
    "service_times",
    "time_windows",
)


def freeze_array(value: Any, *, dtype: Any = np.float64) -> NDArray[Any]:
    """Return a read-only copy of ``value`` as an array of ``dtype``."""
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _freeze_extra(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = value.copy()
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_extra(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_extra(v) for v in value)
    return value


def _array_digest(array: NDArray[Any]) -> list[Any]:
    little = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    return [str(little.dtype), list(little.shape), hashlib.sha256(little.tobytes()).hexdigest()]


def _hashable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _array_digest(value)
    if isinstance(value, Mapping):
        return {str(k): _hashable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_hashable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True, eq=False)
class Instance:
    """Static data of one problem instance.

    Attributes
    ----------
    name
        Instance name, for example ``"X-n101-k25"``.
    source
        Where it came from: ``"vrplib"``, ``"solomon"``, ``"generated:uniform"``, ...
    demands
        Demand per node, shape ``(N,)``; zero at depots.
    coords
        Node coordinates, shape ``(N, 2)``, or ``None`` for matrix-only instances.
    capacity
        Vehicle capacity given by the instance, if any.
    n_vehicles
        Number of vehicles given by the instance, if any.
    time_windows
        ``[earliest, latest]`` service start per node, shape ``(N, 2)``.
    service_times
        Service duration per node, shape ``(N,)``.
    distance_matrix
        Explicit distances, shape ``(N, N)``, for instances that give them.
    edge_weight_type
        The VRPLIB ``EDGE_WEIGHT_TYPE``, if read from a VRPLIB file.
    n_depots
        Number of depots; they are nodes ``0 .. n_depots - 1``.
    schema
        Id of the instance schema (`fleetgym.data.schema`), ``fleetgym/core/1`` by default.
    vehicle_types
        One mapping per vehicle type (``capacity``, ``count``, ``fixed_cost``, ``name`` and schema
        extensions). Given ``capacity``/``n_vehicles`` instead, it is one type built from them.
    extra
        Fields of the schema that are not attributes: global scalars (for example
        ``max_route_duration``), node arrays of shape ``(N,)`` (for example ``deadline``) and arc
        matrices of shape ``(N, N)``, checked against the schema. Names starting with ``_`` are
        FleetGym internals and are not checked.
    meta
        Provenance (file header lines, references): free-form, not checked and not hashed.
    distributions
        Distributions of the uncertain values, by source (``{"demand": {"dist": "poisson",
        "mean": "demand"}}``). Parameters may name fields of the instance. They are checked when
        a problem spec is bound to the instance (`fleetgym.spec.binding.bind`).
    """

    name: str
    source: str
    demands: FloatArray
    coords: FloatArray | None = None
    capacity: float | None = None
    n_vehicles: int | None = None
    time_windows: FloatArray | None = None
    service_times: FloatArray | None = None
    distance_matrix: FloatArray | None = None
    edge_weight_type: str | None = None
    n_depots: int = 1
    extra: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "fleetgym/core/1"
    vehicle_types: tuple[Mapping[str, Any], ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)
    distributions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "demands", freeze_array(self.demands))
        n = self.demands.shape[0]
        if self.demands.ndim != 1 or n == 0:
            raise InstanceError(f"{self.name}: demands must be a non-empty 1-D array")
        if not 1 <= self.n_depots < n:
            raise InstanceError(f"{self.name}: needs 1 to {n - 1} depots, got {self.n_depots}")
        shapes = {
            "coords": (n, 2),
            "time_windows": (n, 2),
            "service_times": (n,),
            "distance_matrix": (n, n),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value is None:
                continue
            array = freeze_array(value)
            if array.shape != shape:
                raise InstanceError(
                    f"{self.name}: {name} has shape {array.shape}, expected {shape}"
                )
            if not np.all(np.isfinite(array)):
                raise InstanceError(f"{self.name}: {name} contains NaN or infinite values")
            set_(self, name, array)
        if not np.all(np.isfinite(self.demands)) or np.any(self.demands < 0):
            raise InstanceError(f"{self.name}: demands must be finite and non-negative")
        if self.capacity is not None:
            set_(self, "capacity", float(self.capacity))
            if not self.capacity > 0:
                raise InstanceError(f"{self.name}: capacity must be positive, got {self.capacity}")
        if self.n_vehicles is not None:
            if int(self.n_vehicles) != self.n_vehicles or self.n_vehicles < 1:
                raise InstanceError(f"{self.name}: n_vehicles must be a positive integer")
            set_(self, "n_vehicles", int(self.n_vehicles))
        if self.time_windows is not None and np.any(
            self.time_windows[:, 0] > self.time_windows[:, 1]
        ):
            raise InstanceError(f"{self.name}: a time window opens after it closes")
        set_(self, "extra", MappingProxyType({k: _freeze_extra(v) for k, v in self.extra.items()}))
        set_(self, "meta", MappingProxyType({k: _freeze_extra(v) for k, v in self.meta.items()}))
        for source, dist in self.distributions.items():
            if not isinstance(dist, Mapping) or "dist" not in dist:
                raise InstanceError(
                    f"{self.name}: distributions.{source} must be a mapping with a `dist` name"
                )
        set_(
            self,
            "distributions",
            MappingProxyType({k: _freeze_extra(dict(v)) for k, v in self.distributions.items()}),
        )
        self._canonical_extra()
        self._init_vehicle_types()
        problems = self.schema_problems()
        if problems:
            raise InstanceError(f"{self.name}: " + "; ".join(problems))

    def _canonical_extra(self) -> None:
        """Store schema fields with their schema type, so equal data hashes equally."""
        from fleetgym.data.schema import get_schema

        try:
            schema = get_schema(self.schema)
        except InstanceError:
            return  # reported by schema_problems
        dtypes = {"float": np.float64, "int": np.int64, "bool": np.bool_}
        out = {}
        for name, value in self.extra.items():
            level = None if name.startswith("_") else schema.level_of(name)
            spec = schema.field(level, name) if level is not None else None
            if spec is not None and spec.type in dtypes and level != "vehicle_type":
                try:
                    array = np.asarray(value, dtype=float)
                    if spec.type == "int" and np.all(array == np.round(array)):
                        array = array.astype(np.int64)
                    elif spec.type == "bool":
                        array = np.asarray(value, dtype=np.bool_)
                    value = (
                        array.item() if array.ndim == 0 else freeze_array(array, dtype=array.dtype)
                    )
                except (TypeError, ValueError):
                    pass  # reported by schema_problems
            out[name] = value
        object.__setattr__(self, "extra", MappingProxyType(out))

    def _init_vehicle_types(self) -> None:
        set_ = object.__setattr__
        types = [dict(t) for t in self.vehicle_types]
        if not types and self.capacity is not None:
            single: dict[str, Any] = {"capacity": self.capacity}
            if self.n_vehicles is not None:
                single["count"] = self.n_vehicles
            types = [single]
        elif types and self.capacity is None and len(types) == 1:
            if "capacity" in types[0]:
                set_(self, "capacity", float(types[0]["capacity"]))
            if "count" in types[0]:
                set_(self, "n_vehicles", int(types[0]["count"]))
        set_(self, "vehicle_types", tuple(MappingProxyType(t) for t in types))

    def schema_problems(self) -> list[str]:
        """Every way this instance breaks its schema, as messages (empty if it conforms)."""
        from fleetgym.data.schema import get_schema

        try:
            schema = get_schema(self.schema)
        except InstanceError as exc:
            return [str(exc)]
        n = self.n_nodes
        problems: list[str] = []
        for name, value in self.extra.items():
            if name.startswith("_"):
                continue
            level = schema.level_of(name)
            spec = schema.field(level, name) if level is not None else None
            if level is None or level == "vehicle_type" or spec is None:
                problems.append(f"field {name!r} is not declared by schema {self.schema!r}")
                continue
            shape = {"global": (), "node": (n,), "arc": (n, n)}[level]
            if np.shape(value) != shape:
                problems.append(
                    f"{level} field {name!r} has shape {np.shape(value)}, expected {shape}"
                )
                continue
            problems += spec.check(value, f"{level} field {name!r}")
        for name, value in (("demand", self.demands), ("service_time", self.service_times)):
            spec = schema.field("node", name)
            if spec is not None and value is not None:
                problems += spec.check(value, f"node field {name!r}")
        for index, vt in enumerate(self.vehicle_types):
            for name, value in vt.items():
                spec = schema.field("vehicle_type", name)
                if spec is None:
                    problems.append(
                        f"vehicle type field {name!r} is not declared by {self.schema!r}"
                    )
                else:
                    problems += spec.check(value, f"vehicle type {index} field {name!r}")
        present = set(self.extra) | {"kind", "demand"}
        if self.coords is not None:
            present |= {"x", "y"}
        if self.service_times is not None:
            present.add("service_time")
        if self.time_windows is not None:
            present |= {"tw_open", "tw_close"}
        if self.distance_matrix is not None:
            present.add("distance")
        for level in ("global", "node", "arc"):
            for name in schema.names(level):
                spec = schema.field(level, name)
                if spec is not None and spec.required and name not in present:
                    problems.append(f"required {level} field {name!r} is missing")
        for name in schema.names("vehicle_type"):
            spec = schema.field("vehicle_type", name)
            if spec is not None and spec.required:
                missing = [i for i, vt in enumerate(self.vehicle_types) if name not in vt]
                if missing or not self.vehicle_types:
                    problems.append(f"required vehicle type field {name!r} is missing")
        return problems

    # ------------------------------------------------------------------- derived values

    @property
    def n_nodes(self) -> int:
        """Number of nodes, depots included."""
        return int(self.demands.shape[0])

    @property
    def n_customers(self) -> int:
        """Number of non-depot nodes."""
        return self.n_nodes - self.n_depots

    def keys(self) -> tuple[str, ...]:
        """Keys accepted by `get` for this instance: set fields, derived values and extras."""
        present = [k for k in FIELD_KEYS if getattr(self, k) is not None]
        derived = ["n_customers", "n_nodes"]
        if self.coords is not None:
            derived.append("customer_coords")
        if self.time_windows is not None:
            derived.append("depot_due_time")
        return tuple(sorted({*present, *derived, *self.extra}))

    def get(self, key: str) -> Any:
        """Return the value of ``key``: a field, a derived value or an extra field.

        Derived keys: ``customer_coords`` (coordinates of non-depot nodes), ``depot_due_time``
        (closing time of depot 0's time window), ``n_customers``, ``n_nodes``.

        Raises
        ------
        InstanceError
            If the instance has no value for ``key``; the message lists the available keys.
        """
        value: Any = None
        if key in FIELD_KEYS:
            value = getattr(self, key)
        elif key == "customer_coords" and self.coords is not None:
            value = self.coords[self.n_depots :]
        elif key == "depot_due_time" and self.time_windows is not None:
            value = float(self.time_windows[0, 1])
        elif key == "n_customers":
            value = self.n_customers
        elif key == "n_nodes":
            value = self.n_nodes
        elif key in self.extra:
            value = self.extra[key]
        if value is None:
            raise InstanceError(
                f"instance {self.name!r} has no value for {key!r}; "
                f"available: {', '.join(self.keys())}"
            )
        return value

    def content_hash(self) -> str:
        """Versioned SHA-256 of every field except ``name``, ``source`` and ``meta``."""
        data = {
            "demands": self.demands,
            "coords": self.coords,
            "capacity": self.capacity,
            "n_vehicles": self.n_vehicles,
            "time_windows": self.time_windows,
            "service_times": self.service_times,
            "distance_matrix": self.distance_matrix,
            "edge_weight_type": self.edge_weight_type,
            "n_depots": self.n_depots,
            "extra": self.extra,
            "schema": self.schema,
            "vehicle_types": [dict(t) for t in self.vehicle_types],
            "distributions": self.distributions,
        }
        return content_hash(_hashable(data))

    def __repr__(self) -> str:
        return (
            f"Instance(name={self.name!r}, source={self.source!r}, n_nodes={self.n_nodes}, "
            f"capacity={self.capacity}, n_vehicles={self.n_vehicles})"
        )
