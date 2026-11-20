"""Observation builders: fixed-shape dictionaries of padded arrays.

Layout of `PaddedGraphObservation` (all ``float32`` except masks, which are ``int8``):

``nodes`` ``(max_nodes, 10)``
    x, y, demand, window open, window close, service time, pending flag, depot flag, demand-known
    flag, under-decision flag (the request of a request epoch). Coordinates are 0 for matrix
    instances; window columns are 0 without time windows. Demand and service time are the revealed
    values where known and forecasts elsewhere (column 8 says which, for demand).
``node_valid`` ``(max_nodes,)``
    1 for real nodes, 0 for padding and for requests not released yet (whose rows are all 0).
``vehicle`` ``(6,)``
    The deciding vehicle: x, y, remaining capacity, time, elapsed route time, at-depot flag.
``vehicles`` ``(max_vehicles, 6)``
    Every vehicle: status, x, y, remaining capacity, time, used flag.
``vehicle_valid`` ``(max_vehicles,)``
    1 for vehicles that exist.
``globals`` ``(4,)``
    Time, pending requests, vehicles used, fraction of requests served.
``action_mask`` ``(n_actions,)``
    1 for valid actions.

Values are raw (not normalized); wrappers normalize. Infinite capacities are reported as 0.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from fleetgym.sim.engine import Epoch, Simulator
from fleetgym.sim.info_state import DEPOT
from fleetgym.sim.state import RequestStatus

__all__ = ["GLOBAL_FEATURES", "NODE_FEATURES", "VEHICLE_FEATURES", "PaddedGraphObservation"]

NODE_FEATURES = 10
VEHICLE_FEATURES = 6
GLOBAL_FEATURES = 4


_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _box(shape: tuple[int, ...]) -> gym.spaces.Box:
    # Finite bounds: Gymnasium warns about infinite Box bounds.
    return gym.spaces.Box(-_FLOAT32_MAX, _FLOAT32_MAX, shape=shape, dtype=np.float32)


def _flag(shape: tuple[int, ...]) -> gym.spaces.Box:
    return gym.spaces.Box(0, 1, shape=shape, dtype=np.int8)


class PaddedGraphObservation:
    """Build padded dictionary observations.

    Parameters
    ----------
    max_nodes
        Node slots.
    max_vehicles
        Vehicle slots.
    n_actions
        Length of the action mask.
    """

    def __init__(self, max_nodes: int, max_vehicles: int, n_actions: int) -> None:
        self.max_nodes = max_nodes
        self.max_vehicles = max_vehicles
        self.n_actions = n_actions

    @property
    def space(self) -> gym.spaces.Dict:
        """The observation space."""
        return gym.spaces.Dict(
            {
                "nodes": _box((self.max_nodes, NODE_FEATURES)),
                "node_valid": _flag((self.max_nodes,)),
                "vehicle": _box((VEHICLE_FEATURES,)),
                "vehicles": _box((self.max_vehicles, VEHICLE_FEATURES)),
                "vehicle_valid": _flag((self.max_vehicles,)),
                "globals": _box((GLOBAL_FEATURES,)),
                "action_mask": _flag((self.n_actions,)),
            }
        )

    def build(
        self, sim: Simulator, epoch: Epoch | None, action_mask: NDArray[np.bool_]
    ) -> dict[str, NDArray[Any]]:
        """Observation at ``epoch`` (``None`` at the end of the episode)."""
        info = sim.info
        state = sim.state
        instance = sim.problem.instance
        n = instance.n_nodes
        coords = instance.coords if instance.coords is not None else np.zeros((n, 2))

        nodes = np.zeros((self.max_nodes, NODE_FEATURES), dtype=np.float32)
        nodes[:n, 0:2] = coords
        nodes[:n, 2] = info.demands
        if info.time_windows is not None:
            nodes[:n, 3:5] = info.time_windows
        nodes[:n, 5] = info.service_times
        nodes[:n, 6] = state.request_status == RequestStatus.PENDING
        nodes[:n, 7] = np.arange(n) < instance.n_depots
        nodes[:n, 8] = info.demand_known
        if epoch is not None and epoch.kind == "request" and epoch.node is not None:
            nodes[epoch.node, 9] = 1.0
        node_valid = np.zeros(self.max_nodes, dtype=np.int8)
        node_valid[:n] = 1
        hidden = np.flatnonzero(state.request_status == RequestStatus.NOT_RELEASED)
        nodes[hidden] = 0.0
        node_valid[hidden] = 0

        vehicles = np.zeros((self.max_vehicles, VEHICLE_FEATURES), dtype=np.float32)
        vehicle_valid = np.zeros(self.max_vehicles, dtype=np.int8)
        for v in state.vehicles[: self.max_vehicles]:
            remaining = v.capacity - v.load if np.isfinite(v.capacity) else 0.0
            vehicles[v.index] = (int(v.status), *coords[v.node], remaining, v.time, float(v.used))
            vehicle_valid[v.index] = 1

        current = np.zeros(VEHICLE_FEATURES, dtype=np.float32)
        if epoch is not None and epoch.vehicle is not None:
            v = state.vehicles[epoch.vehicle]
            remaining = v.capacity - v.load if np.isfinite(v.capacity) else 0.0
            elapsed = v.time - v.route_start if v.route_start is not None else 0.0
            current[:] = (*coords[v.node], remaining, v.time, elapsed, float(v.node == DEPOT))

        requests = state.request_status != RequestStatus.NOT_A_REQUEST
        served = state.request_status == RequestStatus.SERVED
        globals_ = np.array(
            [
                state.time,
                float(np.sum(state.request_status == RequestStatus.PENDING)),
                float(sum(v.used for v in state.vehicles)),
                float(served.sum() / max(int(requests.sum()), 1)),
            ],
            dtype=np.float32,
        )
        return {
            "nodes": nodes,
            "node_valid": node_valid,
            "vehicle": current,
            "vehicles": vehicles,
            "vehicle_valid": vehicle_valid,
            "globals": globals_,
            "action_mask": action_mask.astype(np.int8),
        }
