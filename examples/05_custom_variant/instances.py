"""Instances of the rural parcel collection variant, in the schema ``rural-parcels.schema.yaml``.

FleetGym ships no instances: a variant comes with the code that makes its data. This one draws
customers around a few villages and writes them as an instance set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fleetgym.data import Instance, InstanceSet, load_schema, write_instance_set

HERE = Path(__file__).resolve().parent
SCHEMA = load_schema(HERE / "rural-parcels.schema.yaml")


def village_instance(rng: np.random.Generator, name: str, n_customers: int = 24) -> Instance:
    """Customers normal around 4 villages in a 100 x 100 area; the depot in the middle."""
    centers = rng.uniform(15.0, 85.0, size=(4, 2))
    village = rng.integers(0, 4, size=n_customers)
    customers = np.clip(centers[village] + rng.normal(0.0, 6.0, size=(n_customers, 2)), 0, 100)
    return Instance(
        name=name,
        source="examples/05",
        schema=SCHEMA.id,
        demands=np.concatenate(
            [[0.0], rng.integers(1, 9, size=n_customers)]
        ),  # the triangular mode
        coords=np.vstack([[50.0, 50.0], customers]),
        vehicle_types=({"capacity": 15.0, "count": 6},),
        extra={
            "village": np.concatenate([[0], village + 1]),
            "max_stops": 5,
            "restock_radius": 40.0,
            "courier_price": 3.0,
        },
        distributions={
            "demand": {"dist": "triangular_demand", "low": 0, "mode": "demand", "high": 12}
        },
    )


def write_villages(folder: str | Path, n_instances: int = 10, seed: int = 0) -> InstanceSet:
    """Write ``n_instances`` village instances as an instance set (the seed is recorded)."""
    rng = np.random.default_rng(seed)
    instances = [village_instance(rng, f"villages-{k}") for k in range(n_instances)]
    return write_instance_set(
        instances,
        folder,
        name="rural parcels",
        produced_by=f"examples/05_custom_variant/instances.py, seed {seed}",
        overwrite=True,
    )
