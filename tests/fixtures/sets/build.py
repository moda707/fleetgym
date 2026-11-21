"""Build the small instance sets the test suite uses (test data, not library code).

FleetGym ships no instance generators (ADR 0023); the tests need instances for every family, so
this script writes a few small, deterministic sets next to it. Run it after changing it; the sets
are committed and verified by their manifests::

    python tests/fixtures/sets/build.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from fleetgym.data import Instance, write_instance_set

HERE = Path(__file__).resolve().parent
N_INSTANCES = 6


def uniform(
    rng: np.random.Generator,
    n: int,
    *,
    capacity: float,
    count: int | None = None,
    demand: tuple[int, int] = (1, 9),
    time_windows: bool = False,
    name: str = "uniform",
    size: float = 100.0,
) -> Instance:
    """Depot at the center of a ``size`` x ``size`` square, customers uniform, integer demands."""
    coords = np.vstack([[size / 2, size / 2], rng.uniform(0.0, size, size=(n, 2))])
    demands = np.concatenate([[0.0], rng.integers(demand[0], demand[1] + 1, size=n)]).astype(float)
    windows = service = None
    if time_windows:  # every customer reachable and servable within a 480-minute day
        horizon, service_time = 480.0, 10.0
        from_depot = np.hypot(*(coords[1:] - coords[0]).T)
        latest = horizon - service_time - from_depot
        width = rng.uniform(60.0, 180.0, size=n)
        start = from_depot + rng.uniform(0.0, 1.0, size=n) * np.maximum(
            latest - width - from_depot, 0
        )
        windows = np.vstack(
            [[0.0, horizon], np.column_stack([start, np.minimum(start + width, latest)])]
        )
        service = np.concatenate([[0.0], np.full(n, service_time)])
    vehicle: dict[str, float | int] = {"capacity": capacity}
    if count is not None:
        vehicle["count"] = count
    return Instance(
        name=name,
        source="tests",
        demands=demands,
        coords=coords,
        time_windows=windows,
        service_times=service,
        vehicle_types=(vehicle,),
    )


def build() -> None:
    sets: dict[str, list[Instance]] = {}
    rng = np.random.default_rng(2026)
    sets["cvrp"] = [uniform(rng, 20, capacity=50, name=f"cvrp-{i}") for i in range(N_INSTANCES)]
    rng = np.random.default_rng(2027)
    sets["vrptw"] = [
        uniform(rng, 20, capacity=100, demand=(1, 20), time_windows=True, name=f"vrptw-{i}")
        for i in range(N_INSTANCES)
    ]
    poisson = {"demand": {"dist": "poisson", "mean": "demand"}}
    sets["svrp"] = [
        Instance(**{**i.__dict__, "name": i.name.replace("cvrp", "svrp"), "distributions": poisson})
        for i in sets["cvrp"]
    ]
    rng = np.random.default_rng(2028)
    sets["fleet"] = [
        uniform(rng, 20, capacity=50, count=4, name=f"fleet-{i}") for i in range(N_INSTANCES)
    ]
    dynamic = {
        "arrivals": {
            "process": "nhpp",
            "profile": {"breaks": [0, 120, 300, 480], "rates": [0.2, 0.5, 0.1]},
        },
        "request.location": {"dist": "kde", "data": "customer_coords"},
        "request.demand": {"dist": "categorical", "values": [5, 10, 20], "probs": [0.5, 0.3, 0.2]},
        "request.time_window_width": {"dist": "uniform", "low": 60, "high": 180},
        "cancellation": {"dist": "bernoulli", "p": 0.02},
    }
    rng = np.random.default_rng(2029)
    sets["dvrp"] = []
    for i in range(N_INSTANCES):
        base = uniform(
            rng, 20, capacity=100, count=4, demand=(1, 20), time_windows=True, name=f"dvrp-{i}"
        )
        sets["dvrp"].append(
            Instance(**{**base.__dict__, "extra": {"cutoff": 360.0}, "distributions": dynamic})
        )
    rng = np.random.default_rng(2030)
    sets["small"] = [
        uniform(rng, 5, capacity=10, demand=(1, 5), name=f"small-{i}") for i in range(N_INSTANCES)
    ]
    sets["small_sd"] = [
        Instance(
            **{**i.__dict__, "name": i.name.replace("small", "small_sd"), "distributions": poisson}
        )
        for i in sets["small"]
    ]
    rng = np.random.default_rng(2031)
    sets["tiny"] = [
        uniform(rng, 4, capacity=10, demand=(1, 3), name=f"tiny-{i}", size=20.0)
        for i in range(N_INSTANCES)
    ]
    for name, instances in sets.items():
        folder = HERE / name
        if folder.exists():
            shutil.rmtree(folder)
        write_instance_set(
            instances,
            folder,
            name=f"tests/{name}",
            reference="FleetGym test suite",
            produced_by="tests/fixtures/sets/build.py",
        )
        print(f"wrote {folder} ({len(instances)} instances)")


if __name__ == "__main__":
    build()
