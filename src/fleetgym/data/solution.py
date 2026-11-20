"""The `Solution` container and solution-file reader.

Route entries are kept exactly as written in the file. How they map to instance node indices
differs between benchmark collections and is decided where solutions are replayed
(`fleetgym.testing.replication`), against the published cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import vrplib

from fleetgym.errors import InstanceError

__all__ = ["Solution", "read_solution"]


@dataclass(frozen=True)
class Solution:
    """Routes and published cost of a solution file.

    Attributes
    ----------
    routes
        One tuple per route, entries as written in the file (depots excluded).
    cost
        The cost stated in the file, if any.
    extra
        Other key-value lines of the file.
    """

    routes: tuple[tuple[int, ...], ...]
    cost: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))


def read_solution(path: str | Path) -> Solution:
    """Read a VRPLIB-style solution file (``Route #k: ...`` lines and a ``Cost`` line).

    Raises
    ------
    InstanceError
        If the file is missing or contains no routes.
    """
    path = Path(path)
    try:
        raw = vrplib.read_solution(path)
    except FileNotFoundError:
        raise InstanceError(f"{path}: solution file not found") from None
    routes = tuple(tuple(int(i) for i in route) for route in raw.pop("routes", []))
    if not routes:
        raise InstanceError(f"{path}: no `Route` lines found")
    cost = raw.pop("cost", None)
    return Solution(routes=routes, cost=None if cost is None else float(cost), extra=raw)
