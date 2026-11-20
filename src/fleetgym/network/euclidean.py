"""Euclidean network with explicit rounding conventions.

Rounding modes
--------------
``none``
    Double-precision Euclidean distance.
``nearest_int``
    TSPLIB ``EUC_2D`` convention: ``floor(d + 0.5)`` per arc. Ties round up, unlike Python's
    `round`, which rounds half to even.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fleetgym.errors import InstanceError
from fleetgym.network.base import NETWORK_BUILDERS, Network
from fleetgym.spec.binding import BoundProblem

__all__ = ["EuclideanNetwork", "Rounding", "euclidean_distances"]

Rounding = Literal["none", "nearest_int"]


def euclidean_distances(coords: NDArray[np.float64], rounding: Rounding) -> NDArray[np.float64]:
    """Pairwise Euclidean distances with the given rounding convention."""
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    if rounding == "nearest_int":
        distances = np.floor(distances + 0.5)
    elif rounding != "none":
        raise ValueError(f"unknown rounding {rounding!r}")
    return distances


class EuclideanNetwork(Network):
    """Euclidean distances between coordinates; travel time is distance divided by ``speed``.

    Parameters
    ----------
    coords
        Node coordinates, shape ``(n_nodes, 2)``.
    rounding
        Rounding convention applied to every arc (see the module docstring).
    speed
        Positive travel speed.
    """

    def __init__(
        self, coords: NDArray[np.float64], *, rounding: Rounding = "none", speed: float = 1.0
    ) -> None:
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must have shape (n, 2), got {coords.shape}")
        if not speed > 0:
            raise ValueError(f"speed must be positive, got {speed}")
        self.rounding: Rounding = rounding
        self.speed = float(speed)
        self._distances = euclidean_distances(coords, rounding)
        self._times = self._distances / self.speed
        self._distances.setflags(write=False)
        self._times.setflags(write=False)

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return int(self._distances.shape[0])

    def distance_matrix(self) -> NDArray[np.float64]:
        """Read-only distance matrix."""
        return self._distances

    def time_matrix(self) -> NDArray[np.float64]:
        """Read-only travel-time matrix."""
        return self._times


def _build(problem: BoundProblem) -> Network:
    params = problem.spec.network.params
    coords = problem.instance.coords
    if coords is None:
        raise InstanceError(
            f"instance {problem.instance.name!r} has no coordinates; a euclidean network needs them"
        )
    return EuclideanNetwork(
        coords,
        rounding=params.rounding,  # pyright: ignore[reportAttributeAccessIssue]
        speed=params.speed,  # pyright: ignore[reportAttributeAccessIssue]
    )


NETWORK_BUILDERS.register("euclidean", _build)
