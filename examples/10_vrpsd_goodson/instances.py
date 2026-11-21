"""Instances of the VRPSD with a duration limit (Goodson, Thomas and Ohlmann 2016).

As described by Dastpak, Errico and Jabali (arXiv 2109.10200, Section 6.2.1 and Appendix B.1.2).

The first 75 customers of Solomon's R101 (their demands are the expected demands), 11 vehicles,
a duration limit L in {103.05, 171.75, 240.45} (Short, Medium, Long), a capacity Q in {25, 50, 75},
and a demand variability U: the actual demand is a multiple of the expected demand,
{1/2, 1, 3/2} with probabilities (0.05, 0.9, 0.05) for Low, {0, 1/2, 1, 3/2, 2} with
(0.05, 0.15, 0.6, 0.15, 0.05) for Moderate and uniform for High. R101 is downloaded from SINTEF
and checked against its checksum. Not stated in the paper's text and assumed here: no service
times or time windows.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from fleetgym.data import Instance, InstanceSet, write_instance_set
from fleetgym.spec.binding import collection_instance
from fleetgym.spec.models import CollectionSource

LIMITS = {"short": 103.05, "medium": 171.75, "long": 240.45}
CAPACITIES = (25, 50, 75)
VARIABILITY = {
    "low": ((0.5, 1.0, 1.5), (0.05, 0.9, 0.05)),
    "moderate": ((0.0, 0.5, 1.0, 1.5, 2.0), (0.05, 0.15, 0.6, 0.15, 0.05)),
    "high": ((0.0, 0.5, 1.0, 1.5, 2.0), (0.2, 0.2, 0.2, 0.2, 0.2)),
}
VEHICLES = 11


def r101_75() -> Instance:
    """The first 75 customers of R101, without time windows or service times."""
    source = CollectionSource(
        collection="solomon/100",
        name="r101",
        first_customers=75,
        drop=("time_windows", "service_times"),
    )
    return collection_instance(source)


def goodson_instance(base: Instance, limit: str, capacity: int, variability: str) -> Instance:
    """R101's first 75 customers with one (L, Q, U) of the paper."""
    factors, probs = VARIABILITY[variability]
    return dataclasses.replace(
        base,
        name=f"r101-75-{limit}-q{capacity}-{variability}",
        source="examples/10 (Goodson et al. 2016, via Dastpak et al. Appendix B.1.2)",
        capacity=None,
        n_vehicles=None,
        vehicle_types=({"capacity": float(capacity), "count": VEHICLES},),
        extra={"max_route_duration": LIMITS[limit]},
        distributions={
            "demand": {
                "dist": "scaled_categorical",
                "scale": "demand",
                "factors": list(factors),
                "probs": list(probs),
            }
        },
    )


def write_goodson(root: str | Path) -> dict[str, InstanceSet]:
    """Write the 27 instances, each as a set ``<root>/<limit>-q<capacity>-<variability>``."""
    base = r101_75()
    sets = {}
    for limit in LIMITS:
        for capacity in CAPACITIES:
            for variability in VARIABILITY:
                point = f"{limit}-q{capacity}-{variability}"
                sets[point] = write_instance_set(
                    [goodson_instance(base, limit, capacity, variability)],
                    Path(root) / point,
                    name=f"VRPSD Goodson {point}",
                    reference="Goodson, Thomas and Ohlmann (2016); Dastpak et al., arXiv 2109.10200",
                    produced_by="examples/10_vrpsd_goodson/instances.py",
                    overwrite=True,
                )
    return sets
