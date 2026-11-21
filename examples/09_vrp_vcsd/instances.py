"""Instances of the VRP-VCSD (Dastpak, Errico and Jabali, arXiv 2109.10200, Appendix B.1.1).

A 100 x 100 service area cut into 5 x 5 zones, 15 of them active; the number of customers in each
active zone follows the density's distribution, customers are uniform in their zone, expected
demands are uniform on {5, 10, 15}, and the actual demand (revealed on the first visit) is uniform
on the expected demand +- 5 (+- 4 when it is 5). The density fixes the fleet size m and the
duration limit L; the capacity Q is 25, 50 or 75. The depot is at (50, 50).

The paper's Figure 4 layout is not in its text: the 15 active zones are a fixed random draw
(``numpy.random.default_rng(2109).choice(25, 15, replace=False)``, sorted).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fleetgym.data import Instance, InstanceSet, write_instance_set

ACTIVE_ZONES = (0, 2, 3, 5, 6, 10, 12, 13, 14, 17, 18, 19, 20, 21, 23)
#: density: (customers per zone, their probabilities, vehicles m, duration limit L)
DENSITIES = {
    "very_low": ((0, 1, 2), (1 / 2, 1 / 3, 1 / 6), 2, 143.71),
    "low": ((0, 1, 2), (1 / 3, 1 / 3, 1 / 3), 2, 201.38),
    "moderate": ((0, 1, 2, 3), (0.1, 0.4, 0.4, 0.1), 3, 221.47),
    "high": ((2, 3, 4, 5), (0.1, 0.4, 0.4, 0.1), 7, 195.54),
    "very_high": ((4, 5, 6, 7), (0.1, 0.4, 0.4, 0.1), 11, 187.29),
}
CAPACITIES = (25, 50, 75)
EXPECTED_DEMANDS = (5, 10, 15)
DEMAND = {"dist": "discrete_uniform", "center": "demand", "half_width": 5, "floor": 1}


def vcsd_instance(rng: np.random.Generator, density: str, capacity: int, name: str) -> Instance:
    """One day's customers for a density and a capacity."""
    values, probs, vehicles, limit = DENSITIES[density]
    while True:  # an instance needs a customer: redraw if every zone is empty
        counts = rng.choice(values, size=len(ACTIVE_ZONES), p=probs)
        if counts.sum() > 0:
            break
    points = []
    for zone, count in zip(ACTIVE_ZONES, counts, strict=True):
        corner = np.array([zone % 5, zone // 5], dtype=float) * 20.0
        points.append(corner + rng.uniform(0.0, 20.0, size=(count, 2)))
    customers = np.vstack(points)
    expected = rng.choice(EXPECTED_DEMANDS, size=len(customers))
    return Instance(
        name=name,
        source="examples/09 (Dastpak et al., Appendix B.1.1)",
        demands=np.concatenate([[0.0], expected]),
        coords=np.vstack([[50.0, 50.0], customers]),
        vehicle_types=({"capacity": float(capacity), "count": vehicles},),
        extra={"max_route_duration": limit},
        distributions={"demand": DEMAND},
    )


def write_vcsd(root: str | Path, n_instances: int, seed: int = 0) -> dict[str, InstanceSet]:
    """Write one instance set per density and capacity, ``<root>/<density>-q<capacity>``."""
    rng = np.random.default_rng(seed)
    sets = {}
    for density in DENSITIES:
        for capacity in CAPACITIES:
            point = f"{density}-q{capacity}"
            day = [
                vcsd_instance(rng, density, capacity, f"{point}-{k}") for k in range(n_instances)
            ]
            sets[point] = write_instance_set(
                day,
                Path(root) / point,
                name=f"VRP-VCSD {point}",
                reference="Dastpak, Errico and Jabali, arXiv 2109.10200, Appendix B.1.1",
                produced_by=f"examples/09_vrp_vcsd/instances.py, seed {seed}",
                overwrite=True,
            )
    return sets
