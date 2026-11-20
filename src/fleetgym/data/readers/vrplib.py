"""VRPLIB and CVRPLIB reader built on the `vrplib` package.

Distances are not taken from ``vrplib``: for coordinate-based edge weights the network computes them
with the rounding the spec declares (``vrplib`` returns unrounded ``EUC_2D`` distances). Explicit
edge weights (``EDGE_WEIGHT_TYPE: EXPLICIT``) are kept as the instance's distance matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import vrplib

from fleetgym.data.instance import Instance
from fleetgym.errors import InstanceError

__all__ = ["read_vrplib"]

_MAPPED = {
    "name",
    "node_coord",
    "demand",
    "capacity",
    "depot",
    "vehicles",
    "time_window",
    "service_time",
    "edge_weight",
    "edge_weight_type",
}


def _leading_depots(depot: Any, path: Path) -> int:
    depots = np.atleast_1d(np.asarray(depot, dtype=int))
    if depots.size == 0:
        raise InstanceError(f"{path}: DEPOT_SECTION is empty")
    if not np.array_equal(np.sort(depots), np.arange(depots.size)):
        raise InstanceError(
            f"{path}: depots must be the first nodes (got depot indices {depots.tolist()}); "
            "other layouts are not supported in this version"
        )
    return int(depots.size)


def read_vrplib(path: str | Path) -> Instance:
    """Read a VRPLIB-format instance (CVRPLIB, VRPLIB, TSPLIB-style ``.vrp`` files).

    A scalar ``SERVICE_TIME`` specification is applied to every customer (zero at depots).

    Raises
    ------
    InstanceError
        If the file is missing, malformed, or lays out depots in an unsupported way.
    """
    path = Path(path)
    try:
        raw = vrplib.read_instance(path, instance_format="vrplib", compute_edge_weights=False)
    except FileNotFoundError:
        raise InstanceError(f"{path}: instance file not found") from None
    except (ValueError, RuntimeError, IndexError) as exc:
        raise InstanceError(f"{path}: not a valid VRPLIB instance ({exc})") from exc

    n_depots = _leading_depots(raw.get("depot", [0]), path)
    dimension = int(raw.get("dimension", 0))
    coords = raw.get("node_coord")
    demands = raw.get("demand")
    if demands is None:
        demands = np.zeros(dimension or (len(coords) if coords is not None else 0))
    n = len(demands)

    service = raw.get("service_time")
    if service is not None and np.ndim(service) == 0:
        service = np.full(n, float(service))
        service[:n_depots] = 0.0

    edge_weight_type = raw.get("edge_weight_type")
    matrix = raw.get("edge_weight") if edge_weight_type == "EXPLICIT" else None
    vehicles = raw.get("vehicles")

    return Instance(
        name=str(raw.get("name", path.stem)),
        source="vrplib",
        demands=np.asarray(demands, dtype=float),
        coords=None if coords is None else np.asarray(coords, dtype=float),
        capacity=None if raw.get("capacity") is None else float(raw["capacity"]),
        n_vehicles=None if vehicles is None else int(vehicles),
        time_windows=None
        if raw.get("time_window") is None
        else np.asarray(raw["time_window"], dtype=float),
        service_times=None if service is None else np.asarray(service, dtype=float),
        distance_matrix=None if matrix is None else np.asarray(matrix, dtype=float),
        edge_weight_type=None if edge_weight_type is None else str(edge_weight_type),
        n_depots=n_depots,
        extra={k: v for k, v in raw.items() if k not in _MAPPED},
    )
