"""The FleetGym instance format: one JSON document per instance (ADR 0023).

Layout (``fleetgym-instance/1``)::

    {
      "format": "fleetgym-instance/1",
      "schema": "fleetgym/core/1",
      "schema_file": "rural-parcels.schema.yaml",     # optional: a user schema, relative path
      "name": "...", "source": "...", "meta": {...},  # meta is free-form and not hashed
      "global": {"max_route_duration": 221.47},
      "vehicle_types": [{"capacity": 50, "count": 3}],
      "nodes": {"kind": ["depot", "customer", ...], "x": [...], "y": [...], "demand": [...]},
      "arcs": {"distance": [[...], ...]},
      "distributions": {"demand": {"dist": "poisson", "mean": "demand"}}
    }

Node columns all have one value per node, depots first. Every field must be declared by the schema;
the document is checked against it when read, and errors name the file and the field.
``distributions`` gives the distribution of each uncertain value; a parameter that is a field name
(``"mean": "demand"``) refers to that field of the instance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from fleetgym.data.instance import Instance
from fleetgym.data.schema import get_schema, load_schema, schema_ids
from fleetgym.errors import InstanceError

__all__ = ["FORMAT", "instance_to_dict", "read_fleetgym_json", "write_instance"]

FORMAT = "fleetgym-instance/1"
_TOP_LEVEL = {
    "format",
    "schema",
    "schema_file",
    "name",
    "source",
    "meta",
    "global",
    "vehicle_types",
    "nodes",
    "arcs",
    "distributions",
}
# Fields stored as Instance attributes rather than in `extra`.
_NODE_ATTRIBUTES = {"kind", "x", "y", "demand", "service_time", "tw_open", "tw_close"}


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def instance_to_dict(instance: Instance) -> dict[str, Any]:
    """The JSON-ready form of ``instance`` (internal ``_`` fields are not written)."""
    n = instance.n_nodes
    nodes: dict[str, Any] = {
        "kind": ["depot"] * instance.n_depots + ["customer"] * instance.n_customers,
    }
    if instance.coords is not None:
        nodes["x"] = instance.coords[:, 0]
        nodes["y"] = instance.coords[:, 1]
    nodes["demand"] = instance.demands
    if instance.service_times is not None:
        nodes["service_time"] = instance.service_times
    if instance.time_windows is not None:
        nodes["tw_open"] = instance.time_windows[:, 0]
        nodes["tw_close"] = instance.time_windows[:, 1]
    globals_: dict[str, Any] = {}
    arcs: dict[str, Any] = {}
    if instance.edge_weight_type is not None:
        globals_["edge_weight_type"] = instance.edge_weight_type
    if instance.distance_matrix is not None:
        arcs["distance"] = instance.distance_matrix
    for name, value in instance.extra.items():
        if name.startswith("_"):
            continue
        shape = np.shape(value)
        if shape == ():
            globals_[name] = value
        elif shape == (n,):
            nodes[name] = value
        else:
            arcs[name] = value
    out: dict[str, Any] = {
        "format": FORMAT,
        "schema": instance.schema,
        "name": instance.name,
        "source": instance.source,
    }
    if instance.meta:
        out["meta"] = instance.meta
    if globals_:
        out["global"] = globals_
    if instance.vehicle_types:
        out["vehicle_types"] = [dict(t) for t in instance.vehicle_types]
    out["nodes"] = nodes
    if arcs:
        out["arcs"] = arcs
    if instance.distributions:
        out["distributions"] = {k: dict(v) for k, v in instance.distributions.items()}
    return _plain(out)


def write_instance(instance: Instance, path: str | Path) -> Path:
    """Write ``instance`` as a FleetGym JSON document; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instance_to_dict(instance), indent=1) + "\n", encoding="utf-8")
    return path


def _column(nodes: Mapping[str, Any], name: str, n: int, where: str) -> np.ndarray:
    values = nodes[name]
    if not isinstance(values, list) or len(values) != n:
        raise InstanceError(f"{where}: nodes.{name} must be a list of {n} values (one per node)")
    if name == "kind":
        return np.asarray(values, dtype=object)
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise InstanceError(f"{where}: nodes.{name} must hold numbers") from None


def read_fleetgym_json(path: str | Path) -> Instance:
    """Read a FleetGym JSON instance and check it against its schema.

    Raises
    ------
    InstanceError
        If the file is not a valid ``fleetgym-instance/1`` document or breaks its schema; the
        message names the file and the field.
    """
    path = Path(path)
    where = str(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstanceError(f"{where}: cannot read the instance: {exc}") from exc
    if not isinstance(raw, Mapping) or raw.get("format") != FORMAT:
        raise InstanceError(f'{where}: not a FleetGym instance (expected "format": "{FORMAT}")')
    unknown = set(raw) - _TOP_LEVEL
    if unknown:
        raise InstanceError(f"{where}: unknown top-level keys {sorted(unknown)}")
    schema_id = raw.get("schema", "fleetgym/core/1")
    try:
        if "schema_file" in raw and schema_id not in schema_ids():
            load_schema(path.parent / str(raw["schema_file"]))
        schema = get_schema(schema_id)
    except InstanceError as exc:
        raise InstanceError(f"{where}: {exc}") from None
    nodes = raw.get("nodes")
    if not isinstance(nodes, Mapping) or "kind" not in nodes or "demand" not in nodes:
        raise InstanceError(f"{where}: `nodes` needs at least `kind` and `demand` columns")
    n = len(nodes["kind"]) if isinstance(nodes["kind"], list) else -1
    if n < 2:
        raise InstanceError(f"{where}: nodes.kind must list at least a depot and a customer")
    kinds = _column(nodes, "kind", n, where)
    depots = int(np.sum(kinds == "depot"))
    if not np.all(kinds[:depots] == "depot") or not np.all(kinds[depots:] == "customer"):
        raise InstanceError(f"{where}: nodes.kind must list the depots first, then customers")
    for pair in (("x", "y"), ("tw_open", "tw_close")):
        if (pair[0] in nodes) != (pair[1] in nodes):
            raise InstanceError(f"{where}: nodes.{pair[0]} and nodes.{pair[1]} go together")
    columns = {name: _column(nodes, name, n, where) for name in nodes if name != "kind"}
    extra: dict[str, Any] = {}
    for name, values in columns.items():
        if name not in _NODE_ATTRIBUTES:
            if schema.field("node", name) is None:
                raise InstanceError(f"{where}: nodes.{name} is not a node field of {schema_id!r}")
            extra[name] = values
    globals_ = dict(raw.get("global") or {})
    edge_weight_type = globals_.pop("edge_weight_type", None)
    for name, value in globals_.items():
        if schema.field("global", name) is None:
            raise InstanceError(f"{where}: global.{name} is not a global field of {schema_id!r}")
        extra[name] = value
    distance = None
    for name, value in dict(raw.get("arcs") or {}).items():
        if schema.field("arc", name) is None:
            raise InstanceError(f"{where}: arcs.{name} is not an arc field of {schema_id!r}")
        matrix = np.asarray(value, dtype=float)
        if name == "distance":
            distance = matrix
        else:
            extra[name] = matrix
    coords = np.column_stack([columns["x"], columns["y"]]) if "x" in columns else None
    windows = (
        np.column_stack([columns["tw_open"], columns["tw_close"]]) if "tw_open" in columns else None
    )
    try:
        return Instance(
            name=str(raw.get("name", path.stem)),
            source=str(raw.get("source", f"file:{path.name}")),
            demands=columns["demand"],
            coords=coords,
            time_windows=windows,
            service_times=columns.get("service_time"),
            distance_matrix=distance,
            edge_weight_type=None if edge_weight_type is None else str(edge_weight_type),
            n_depots=depots,
            extra=extra,
            schema=schema_id,
            vehicle_types=tuple(raw.get("vehicle_types") or ()),
            meta=dict(raw.get("meta") or {}),
            distributions=dict(raw.get("distributions") or {}),
        )
    except InstanceError as exc:
        raise InstanceError(f"{where}: {exc}") from None
