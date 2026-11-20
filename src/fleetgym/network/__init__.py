"""Travel networks: Euclidean and matrix distance and time models (layer L3).

Time-dependent and road-network models are extension points (out of scope for v0.1).
"""

from fleetgym.network.base import NETWORK_BUILDERS, Network, build_network
from fleetgym.network.euclidean import EuclideanNetwork, euclidean_distances
from fleetgym.network.matrix import MatrixNetwork

__all__ = [
    "NETWORK_BUILDERS",
    "EuclideanNetwork",
    "MatrixNetwork",
    "Network",
    "build_network",
    "euclidean_distances",
]
