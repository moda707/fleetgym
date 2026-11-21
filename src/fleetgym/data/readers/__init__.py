"""Instance file readers, selected by format name or file extension."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from fleetgym.data.instance import Instance
from fleetgym.data.readers.fleetgym_json import read_fleetgym_json, write_instance
from fleetgym.data.readers.solomon import read_solomon
from fleetgym.data.readers.vrplib import read_vrplib
from fleetgym.errors import InstanceError
from fleetgym.plugins import Registry

__all__ = [
    "READERS",
    "InstanceFormat",
    "read_fleetgym_json",
    "read_instance",
    "read_solomon",
    "read_vrplib",
    "write_instance",
]

InstanceFormat = Literal["fleetgym", "vrplib", "solomon"]

READERS: Registry[Callable[[Path], Instance]] = Registry("instance format")
READERS.register("fleetgym", read_fleetgym_json)
READERS.register("vrplib", read_vrplib)
READERS.register("solomon", read_solomon)

_EXTENSIONS = {".json": "fleetgym", ".vrp": "vrplib", ".txt": "solomon"}


def read_instance(path: str | Path, *, fmt: str | None = None) -> Instance:
    """Read an instance file.

    Parameters
    ----------
    path
        The instance file.
    fmt
        Format name (``"fleetgym"``, ``"vrplib"``, ``"solomon"`` or a registered plugin). Inferred
        from the extension when omitted: ``.json`` is the FleetGym format, ``.vrp`` is VRPLIB,
        ``.txt`` is Solomon.

    Raises
    ------
    InstanceError
        If the format cannot be inferred or the file cannot be read.
    """
    path = Path(path)
    if fmt is None:
        fmt = _EXTENSIONS.get(path.suffix.lower())
        if fmt is None:
            raise InstanceError(
                f"{path}: cannot infer the instance format from {path.suffix!r}; pass fmt="
            )
    return READERS.get(fmt)(path)
