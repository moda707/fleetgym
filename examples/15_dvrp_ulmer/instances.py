"""Instances of the VRPSSR of Ulmer, Goodson, Mattfeld and Hennig (2016).

Section 6.1: a square service area of 15 km (small) or 20 km (large) with a central depot; a day of
T = 360 minutes; late requests arrive in [1, T] as a Poisson process with lambda requests per
T - 1 minutes, lambda in {25, 50, 75} (low, moderate, high); the number of early-request customers
is Poisson with mean 100 - lambda. Locations are uniform over the area, or normal around two
cluster centers (5, 5), (5, 15) with equal probability, or three centers (5, 5), (5, 15), (15, 10)
with the second twice as likely as the others, all with a standard deviation of 1 km (the centers
and the deviation are scaled by 0.75 for the small area). Requests outside the area are kept.

Each instance is one realization of the early-request customers and carries the distributions
of the late requests; the paper uses 250 realizations per setting. The paper assumes the early
customers fit in one route by T. When they do not (most days with a large area, low lambda and
uniform locations), the preset applies the rule of Ulmer, Mattfeld and Koester (2017, Section
3.1): all early customers are served, late, and no late request is confirmed that day.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fleetgym.data import Instance, InstanceSet, write_instance_set

T = 360.0
SIDE = {"small": 15.0, "large": 20.0}
LAMBDA = {"low": 25, "moderate": 50, "high": 75}
CLUSTERS = {  # large area; scaled by 0.75 for the small one
    "two_clusters": ([[5.0, 5.0], [5.0, 15.0]], [1.0, 1.0]),
    "three_clusters": ([[5.0, 5.0], [5.0, 15.0], [15.0, 10.0]], [1.0, 2.0, 1.0]),
}


def location_distribution(area: str, locations: str) -> dict[str, Any]:
    """The request-location distribution of a setting, in the instance format."""
    if locations == "uniform":
        return {"dist": "uniform", "low": 0.0, "high": SIDE[area]}
    centers, weights = CLUSTERS[locations]
    scale = 0.75 if area == "small" else 1.0
    return {
        "dist": "normal_mixture",
        "centers": [[scale * x, scale * y] for x, y in centers],
        "std": scale * 1.0,
        "weights": weights,
    }


def _points(dist: dict[str, Any], rng: np.random.Generator, size: int) -> np.ndarray:
    if dist["dist"] == "uniform":
        return rng.uniform(dist["low"], dist["high"], size=(size, 2))
    centers, w = np.asarray(dist["centers"]), np.asarray(dist["weights"], dtype=float)
    which = rng.choice(len(centers), size=size, p=w / w.sum())
    return centers[which] + rng.normal(0.0, dist["std"], size=(size, 2))


def ulmer_instance(
    rng: np.random.Generator, area: str, lam: str, locations: str, name: str
) -> Instance:
    """One realization of the early-request customers of a setting."""
    rate = LAMBDA[lam]
    where = location_distribution(area, locations)
    n_early = int(rng.poisson(100 - rate))
    depot = [SIDE[area] / 2, SIDE[area] / 2]
    n = n_early + 1
    return Instance(
        name=name,
        source="examples/15 (Ulmer et al. 2016, Section 6.1)",
        coords=np.vstack([depot, _points(where, rng, n_early)]),
        demands=np.zeros(n),  # uncapacitated
        time_windows=np.tile([0.0, T], (n, 1)),  # back at the depot by T
        vehicle_types=({"count": 1},),
        distributions={
            # lambda requests per T - 1 minutes, in [1, T]
            "arrivals": {
                "process": "nhpp",
                "profile": {"breaks": [0.0, 1.0, T], "rates": [0.0, rate / (T - 1)]},
            },
            "request.location": where,
        },
    )


def write_ulmer(root: str | Path, n_instances: int, seed: int = 0) -> dict[str, InstanceSet]:
    """Write ``n_instances`` realizations per setting, as sets ``<root>/<area>-<lam>-<locations>``."""
    rng = np.random.default_rng(seed)
    sets = {}
    for area in SIDE:
        for lam in LAMBDA:
            for locations in ("uniform", *CLUSTERS):
                point = f"{area}-{lam}-{locations}"
                days = [
                    ulmer_instance(rng, area, lam, locations, f"{point}-{k}")
                    for k in range(n_instances)
                ]
                sets[point] = write_instance_set(
                    days,
                    Path(root) / point,
                    name=f"VRPSSR {point}",
                    reference="Ulmer, Goodson, Mattfeld and Hennig (2016), Section 6.1",
                    produced_by=f"examples/15_dvrp_ulmer/instances.py, seed {seed}",
                    overwrite=True,
                )
    print(f"wrote {len(sets)} sets of {n_instances} days")
    return sets
