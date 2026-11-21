"""Instances of Taş, Dellaert, van Woensel and de Kok (2012).

Section 5: Solomon's instances with 100 customers, vehicle capacity Q = 200 for every class, a
driver shift of w = 480, and travel times whose expected value is the Euclidean distance: the
travel time per unit distance is Gamma(alpha, lambda) with (alpha, lambda) = (16, 0.0625),
(1, 1) or (0.0625, 16), a coefficient of variation of 0.25, 1 or 4 (Sections 3.2 and 5.2). The
Solomon files are downloaded from SINTEF and checked against their checksums.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from pathlib import Path

from fleetgym.data import Instance, InstanceSet, write_instance_set
from fleetgym.spec.binding import collection_instance
from fleetgym.spec.models import CollectionSource

#: Coefficient of variation per unit distance -> (alpha, lambda) of Section 5.2
CV = {"0.25": (16.0, 0.0625), "1": (1.0, 1.0), "4": (0.0625, 16.0)}
CAPACITY = 200.0
SHIFT = 480.0


def solomon(name: str) -> Instance:
    """Solomon's instance ``name`` (for example ``r101``), downloaded and verified."""
    return collection_instance(CollectionSource(collection="solomon/100", name=name))


def tas_instance(base: Instance, cv: str) -> Instance:
    """``base`` with the capacity, shift and travel-time distribution of one coefficient of variation."""
    alpha, lam = CV[cv]
    return dataclasses.replace(
        base,
        name=f"{base.name}-cv{cv}",
        source="examples/16 (Taş et al. 2012, Section 5)",
        capacity=None,
        n_vehicles=None,
        vehicle_types=({"capacity": CAPACITY},),
        extra={**base.extra, "max_route_duration": SHIFT},
        distributions={"travel_time": {"dist": "gamma_per_distance", "shape": alpha, "scale": lam}},
    )


def write_tas(
    root: str | Path, names: Iterable[str], cvs: Iterable[str] = tuple(CV)
) -> dict[str, InstanceSet]:
    """Write one set per instance and coefficient of variation, ``<root>/<name>-cv<cv>``."""
    sets = {}
    for name in names:
        base = solomon(name)
        for cv in cvs:
            point = f"{name}-cv{cv}"
            sets[point] = write_instance_set(
                [tas_instance(base, cv)],
                Path(root) / point,
                name=f"Taş et al. {point}",
                reference="Taş, Dellaert, van Woensel and de Kok (2012), Section 5",
                produced_by="examples/16_vrptw_tas/instances.py",
                overwrite=True,
            )
    return sets
