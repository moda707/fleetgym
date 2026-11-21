"""Myopic re-optimization with PyVRP (design §18; extra ``solvers``; plan Q30, ADR 0017).

At every vehicle epoch (``next_node``), the problem known now is solved with PyVRP's hybrid genetic
search, and the deciding vehicle drives to the first stop of its route:

* clients: the pending customers, with their known demand (forecasts for unrevealed demands),
  service time and, with hard time windows, their windows; optional (``required=False``) with a
  prize equal to the objective's ``unserved`` weight relative to ``distance`` when it has one;
* vehicles: each vehicle that is not done starts where it will next be free (its current node, or
  its committed destination after the service there) at that time, with its remaining capacity and
  route duration (PyVRP models these natively: per-type start depot, ``tw_early``, capacity,
  ``shift_duration``); vehicles that have not left share one type at the depot, together with the
  vehicles still to start (sequential epochs). PyVRP charges nothing for an unused vehicle, but a
  vehicle on the road must still drive home: each gets a required dummy client at its position that
  only it can reach (its own routing profile), so it is always used and pays its return;
* PyVRP needs integers: distances and times are scaled by ``scale`` (1 for integer distances,
  otherwise 1000) and rounded; the environment always computes the true cost.

Soft time windows, recourse and waiting rules are not modeled (a myopic heuristic). When the
solution suggests an action that is not valid (rounding), the policy falls back to
`NearestNeighbor` and counts it in ``fallbacks``. With dynamic requests and an empty route, the
vehicle waits when it may.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fleetgym.baselines.nearest import NearestNeighbor
from fleetgym.envs.env import FleetGymEnv
from fleetgym.sim.engine import Simulator
from fleetgym.sim.info_state import DEPOT
from fleetgym.sim.state import RequestStatus, VehicleStatus

__all__ = ["PyVRPReoptimization"]

_BIG = 2**40
_HUGE = 10**9  # an arc no vehicle should use


def _require_pyvrp() -> Any:
    try:
        import pyvrp
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError("PyVRPReoptimization needs PyVRP: pip install fleetgym[solvers]") from exc
    return pyvrp


class PyVRPReoptimization(NearestNeighbor):
    """Re-solve the known problem with PyVRP at every vehicle epoch.

    Parameters
    ----------
    iterations
        PyVRP iterations per solve (``MaxIterations``; deterministic, unlike a time limit).
    seed
        PyVRP seed.
    """

    name = "pyvrp_reoptimization"

    def __init__(self, *, iterations: int = 300, seed: int = 0) -> None:
        super().__init__()
        self.iterations = iterations
        self.seed = seed
        self.fallbacks = 0
        _require_pyvrp()

    def reset(self, env: FleetGymEnv) -> None:
        """Bind to ``env``."""
        super().reset(env)
        self.fallbacks = 0

    def act(self, obs: dict[str, NDArray[Any]], info: dict[str, Any]) -> int:
        """First stop of the deciding vehicle's re-optimized route."""
        env = self.env
        epoch = env.epoch
        mask = info["action_mask"]
        if epoch is None or epoch.kind != "vehicle_idle" or epoch.vehicle is None:
            return super().act(obs, info)
        route = self.route(epoch.vehicle)
        if route:
            if mask[route[0]]:
                return route[0]
            self.fallbacks += 1
            return super().act(obs, info)
        wait = env.interface.max_nodes
        if env.interface.wait and mask[wait]:
            return wait
        if mask[DEPOT]:
            return DEPOT
        self.fallbacks += 1
        return super().act(obs, info)

    def route(self, vehicle: int) -> list[int]:
        """The re-optimized route (customer nodes) of ``vehicle`` from now."""
        sim = self.env.simulator
        pending = [int(n) for n in np.flatnonzero(sim.state.pending_mask())]
        if not pending:
            return []
        pyvrp = _require_pyvrp()
        from pyvrp.stop import MaxIterations

        data, clients, owner = self._data(sim, pending, vehicle, pyvrp)
        result = pyvrp.solve(
            data, stop=MaxIterations(self.iterations), seed=self.seed, display=False
        )
        for route in result.best.routes():
            if route.vehicle_type() == owner:
                return [clients[a.idx] for a in route if a.is_client() and a.idx < len(clients)]
        return []

    # ------------------------------------------------------------------ problem data

    def _data(
        self, sim: Simulator, pending: list[int], vehicle: int, pyvrp: Any
    ) -> tuple[Any, list[int], int]:
        info = sim.info
        problem = sim.problem
        spec = problem.spec
        instance = problem.instance
        dist = info.distances
        scale = 1.0 if np.allclose(dist, np.round(dist)) else 1000.0
        demands = info.demands
        load_scale = 1.0 if np.allclose(demands, np.round(demands)) else 1000.0
        tw_spec = spec.constraints.items.get("time_windows")
        hard_tw = tw_spec is not None and tw_spec.model_dump()["mode"] != "soft"
        windows = instance.time_windows if hard_tw else None
        late = _BIG
        if windows is not None:
            late = round(float(windows[DEPOT, 1]) * scale)
        if sim.shift_end is not None:
            late = min(late, round(sim.shift_end * scale))
        max_duration = (
            float(problem.value("constraints.route_duration.max"))
            if "route_duration" in spec.constraints
            and spec.constraints["route_duration"].model_dump()["mode"] == "hard"
            else None
        )

        starts: list[tuple[list[int], int, float, float, float | None]] = []
        # (vehicles, start node, start time, used capacity, route start) per vehicle type
        depot_group: list[int] = []
        now = sim.state.time
        for v in sim.state.vehicles:
            if v.status == VehicleStatus.DONE:
                continue
            if v.status == VehicleStatus.WAITING_TO_START or (
                v.node == DEPOT and not v.used and v.status == VehicleStatus.IDLE
            ):
                depot_group.append(v.index)
                continue
            node, time, load = v.node, max(v.time, now), v.load
            if v.status in (VehicleStatus.TRAVELING, VehicleStatus.SERVING) and node != DEPOT:
                status = sim.state.request_status[node]
                if status != RequestStatus.SERVED:
                    load += float(demands[node])
                    start = time if v.status == VehicleStatus.TRAVELING else v.time
                    if windows is not None:
                        start = max(start, float(windows[node, 0]))
                    time = start + float(info.service_times[node])
            starts.append(([v.index], node, time, load, v.route_start))
        future = len(pending) if sim.unlimited else 0
        if depot_group or future:
            starts.insert(0, (depot_group, DEPOT, now, 0.0, None))

        start_nodes = [DEPOT]
        for _, node, *_rest in starts:
            if node not in start_nodes:
                start_nodes.append(node)
        on_road = [t for t, (_, node, *_rest) in enumerate(starts) if node != DEPOT]
        # Locations: start depots, pending customers, then one dummy per vehicle on the road (a
        # required client only that vehicle can reach, so it counts as used and pays its way home).
        nodes = start_nodes + [c for c in pending if c not in start_nodes]
        dummy_nodes = [starts[t][1] for t in on_road]
        ids = np.array(nodes + dummy_nodes)
        coords = instance.coords if instance.coords is not None else np.zeros((instance.n_nodes, 2))
        locations = [pyvrp.Location(float(coords[n, 0]), float(coords[n, 1])) for n in ids]
        index = {n: i for i, n in enumerate(nodes)}
        depots = [pyvrp.Depot(location=index[n]) for n in start_nodes]
        depot_of = {n: i for i, n in enumerate(start_nodes)}
        base_d = np.round(dist[np.ix_(ids, ids)] * scale).astype(np.int64)
        base_t = np.round(info.travel_times[np.ix_(ids, ids)] * scale).astype(np.int64)
        first_dummy = len(nodes)
        unreachable = base_d.copy()
        unreachable[first_dummy:, :] = _HUGE
        unreachable[:, first_dummy:] = _HUGE
        np.fill_diagonal(unreachable, 0)
        base_unreachable_t = np.where(unreachable == _HUGE, _HUGE, base_t)
        distance_matrices, duration_matrices = [unreachable], [base_unreachable_t]
        profile_of: dict[int, int] = {}
        for k, t in enumerate(on_road):
            loc = first_dummy + k
            own_d, own_t = unreachable.copy(), duration_matrices[0].copy()
            own_d[loc, :first_dummy] = base_d[loc, :first_dummy]
            own_t[loc, :first_dummy] = base_t[loc, :first_dummy]
            start_loc = index[starts[t][1]]
            own_d[start_loc, loc] = own_t[start_loc, loc] = own_d[loc, loc] = own_t[loc, loc] = 0
            distance_matrices.append(own_d)
            duration_matrices.append(own_t)
            profile_of[t] = len(distance_matrices) - 1

        weights = spec.objective.scalar_weights()
        unserved = weights.get("unserved")
        per_distance = weights.get("distance", 1.0) or 1.0
        clients = []
        for c in pending:
            kwargs: dict[str, Any] = {
                "location": index[c],
                "delivery": [round(float(demands[c]) * load_scale)],
                "service_duration": round(float(info.service_times[c]) * scale),
            }
            if windows is not None:
                kwargs["tw_early"] = round(float(windows[c, 0]) * scale)
                kwargs["tw_late"] = round(float(windows[c, 1]) * scale)
            if unserved is not None:
                kwargs["required"] = False
                kwargs["prize"] = round(unserved / per_distance * scale)
            clients.append(pyvrp.Client(**kwargs))
        for k in range(len(on_road)):
            clients.append(pyvrp.Client(location=first_dummy + k, delivery=[0]))

        fixed = round(weights.get("vehicles", 0.0) / per_distance * scale)
        capacity_of = {v.index: v.capacity for v in sim.state.vehicles}
        vehicle_types = []
        owner = -1
        for t, (members, node, time, load, route_start) in enumerate(starts):
            count = len(members) + (future if node == DEPOT and t == 0 else 0)
            reference = members[0] if members else None
            cap = (
                capacity_of.get(reference, float("inf"))
                if reference is not None
                else (float(problem.vehicle_types[0].capacity or np.inf))
            )
            free = cap - load
            capacity = _BIG if not np.isfinite(free) else round(free * load_scale)
            shift = _BIG
            if max_duration is not None:
                used = time - route_start if route_start is not None else 0.0
                shift = max(0, round((max_duration - used) * scale))
            vehicle_types.append(
                pyvrp.VehicleType(
                    num_available=max(count, 1),
                    capacity=[max(capacity, 0)],
                    start_depot=depot_of[node],
                    end_depot=0,
                    fixed_cost=fixed if node == DEPOT else 0,
                    tw_early=round(time * scale),
                    tw_late=max(late, round(time * scale)),
                    shift_duration=shift,
                    profile=profile_of.get(t, 0),
                )
            )
            if vehicle in members:
                owner = t
        data = pyvrp.ProblemData(
            locations=locations,
            clients=clients,
            depots=depots,
            vehicle_types=vehicle_types,
            distance_matrices=distance_matrices,
            duration_matrices=duration_matrices,
        )
        return data, pending, owner
