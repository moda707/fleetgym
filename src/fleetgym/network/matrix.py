"""Network defined by explicit distance and time matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fleetgym.errors import InstanceError
from fleetgym.network.base import NETWORK_BUILDERS, Network
from fleetgym.spec.binding import BoundProblem

__all__ = ["MatrixNetwork"]


class MatrixNetwork(Network):
    """Distances (and travel times) given as matrices; asymmetric matrices are allowed.

    Parameters
    ----------
    distance
        Square, finite, non-negative distance matrix with a zero diagonal.
    time
        Travel-time matrix of the same shape; defaults to ``distance / speed``.
    speed
        Used only when ``time`` is omitted.
    """

    def __init__(
        self,
        distance: NDArray[np.float64],
        time: NDArray[np.float64] | None = None,
        *,
        speed: float = 1.0,
    ) -> None:
        self._distances = self._check(distance, "distance")
        if time is None:
            if not speed > 0:
                raise ValueError(f"speed must be positive, got {speed}")
            self._times = self._distances / float(speed)
        else:
            self._times = self._check(time, "time")
            if self._times.shape != self._distances.shape:
                raise ValueError("time and distance matrices must have the same shape")
        self._distances.setflags(write=False)
        self._times.setflags(write=False)

    @staticmethod
    def _check(matrix: NDArray[np.float64], name: str) -> NDArray[np.float64]:
        array = np.array(matrix, dtype=np.float64, copy=True)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"{name} matrix must be square, got shape {array.shape}")
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValueError(f"{name} matrix must be finite and non-negative")
        if np.any(np.diag(array) != 0):
            raise ValueError(f"{name} matrix must have a zero diagonal")
        return array

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
    matrix = problem.instance.distance_matrix
    if matrix is None:
        raise InstanceError(
            f"instance {problem.instance.name!r} has no distance matrix; a matrix network needs one"
        )
    speed = problem.spec.network.params.speed  # pyright: ignore[reportAttributeAccessIssue]
    return MatrixNetwork(matrix, speed=speed)


NETWORK_BUILDERS.register("matrix", _build)
