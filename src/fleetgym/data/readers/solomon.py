"""Solomon-format VRPTW reader built on the `vrplib` package.

Covers Solomon (1987) and Gehring-Homberger files: node 0 is the depot, the header gives the number
of vehicles and their capacity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vrplib

from fleetgym.data.instance import Instance
from fleetgym.errors import InstanceError

__all__ = ["read_solomon"]


def read_solomon(path: str | Path) -> Instance:
    """Read a Solomon-format instance.

    Raises
    ------
    InstanceError
        If the file is missing or not in Solomon format.
    """
    path = Path(path)
    try:
        raw = vrplib.read_instance(path, instance_format="solomon", compute_edge_weights=False)
    except FileNotFoundError:
        raise InstanceError(f"{path}: instance file not found") from None
    except (ValueError, RuntimeError, IndexError) as exc:
        raise InstanceError(f"{path}: not a valid Solomon instance ({exc})") from exc
    return Instance(
        name=str(raw["name"]),
        source="solomon",
        demands=np.asarray(raw["demand"], dtype=float),
        coords=np.asarray(raw["node_coord"], dtype=float),
        capacity=float(raw["capacity"]),
        n_vehicles=int(raw["vehicles"]),
        time_windows=np.asarray(raw["time_window"], dtype=float),
        service_times=np.asarray(raw["service_time"], dtype=float),
        n_depots=1,
    )
