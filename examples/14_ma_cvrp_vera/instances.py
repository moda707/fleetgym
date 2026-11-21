"""Instances of Vera and Abad (2019).

Section 4 and Table 1: the depot and customers uniform in the unit square, integer demands on 1..9 (Algorithm 1; the text says 11..99, which does not fit the
capacities), and a fixed fleet of three vehicles with different capacities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fleetgym.data import Instance, InstanceSet, write_instance_set

#: Table 1: customers and the three vehicles' capacities
SIZES = {
    "vrp10": (10, (10, 15, 20)),
    "vrp20": (20, (20, 30, 35)),
    "vrp50": (50, (60, 70, 80)),
    "vrp80": (80, (80, 100, 120)),
}


def vera_instance(rng: np.random.Generator, size: str, name: str) -> Instance:
    """One instance of a Table 1 size."""
    n, capacities = SIZES[size]
    return Instance(
        name=name,
        source="examples/14 (Vera and Abad 2019, Table 1)",
        demands=np.concatenate([[0.0], rng.integers(1, 10, size=n)]),
        coords=rng.uniform(0.0, 1.0, size=(n + 1, 2)),
        vehicle_types=tuple(
            {"name": label, "capacity": float(c), "count": 1}
            for label, c in zip(("small", "medium", "large"), capacities, strict=True)
        ),
    )


def write_vera(root: str | Path, n_instances: int, seed: int = 0) -> dict[str, InstanceSet]:
    """Write one set per size, ``<root>/<size>``."""
    rng = np.random.default_rng(seed)
    return {
        size: write_instance_set(
            [vera_instance(rng, size, f"{size}-{k}") for k in range(n_instances)],
            Path(root) / size,
            name=f"Vera and Abad {size}",
            reference="Vera and Abad (2019), arXiv 1912.03341, Table 1",
            produced_by=f"examples/14_ma_cvrp_vera/instances.py, seed {seed}",
            overwrite=True,
        )
        for size in SIZES
    }
