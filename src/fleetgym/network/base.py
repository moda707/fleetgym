"""Abstract `Network` interface and the network registry.

A network gives distances and travel times between instance nodes. Travel time may depend on the
departure time (FIFO time-dependent models are an extension point, out of scope for v0.1); the
networks in this version are time-independent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from fleetgym.plugins import Registry
from fleetgym.spec.binding import BoundProblem

__all__ = ["NETWORK_BUILDERS", "Network", "build_network"]


class Network(ABC):
    """Distances and travel times between nodes ``0 .. n_nodes - 1``."""

    is_time_dependent: ClassVar[bool] = False

    @property
    @abstractmethod
    def n_nodes(self) -> int:
        """Number of nodes."""

    @abstractmethod
    def distance_matrix(self) -> NDArray[np.float64]:
        """Read-only ``(n_nodes, n_nodes)`` distance matrix."""

    @abstractmethod
    def time_matrix(self) -> NDArray[np.float64]:
        """Read-only ``(n_nodes, n_nodes)`` travel-time matrix (time-independent networks)."""

    def distance(self, i: int, j: int) -> float:
        """Distance from node ``i`` to node ``j``."""
        return float(self.distance_matrix()[i, j])

    def travel_time(self, i: int, j: int, depart: float) -> float:
        """Travel time from ``i`` to ``j`` leaving at ``depart`` (ignored if time-independent)."""
        return float(self.time_matrix()[i, j])


NETWORK_BUILDERS: Registry[Callable[[BoundProblem], Network]] = Registry("network builder")
"""Runtime builders keyed by the spec's ``network.type``."""


def build_network(problem: BoundProblem) -> Network:
    """Build the network declared by ``problem.spec.network`` for ``problem.instance``."""
    from fleetgym.network import euclidean, matrix  # noqa: F401  (register built-ins)

    return NETWORK_BUILDERS.get(problem.spec.network.name)(problem)
