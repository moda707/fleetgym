"""Instances of Errico, Desaulniers, Gendreau, Rei and Rousseau (2018).

Section 5.1: Solomon's instances without demands or capacity, reduced to their first 25 customers
(all classes) or 50 customers (classes with suffix 1). Service times are discrete triangular on a
grid of 0.1, by family:

* basic: median 10 on {8.0, ..., 12.0} (R, RC) or 90 on {70.0, ..., 110.0} (C), alpha = 0.95;
* low_probability: the same distributions, alpha = 0.85;
* large_support: median 10 on {5.0, ..., 15.0} or 90 on {45.0, ..., 135.0}, alpha = 0.95;
* positive_skewed: the large support with 7.0 (R, RC) or 63.0 (C), alpha = 0.95. The paper calls
  them medians, but no triangular distribution on these supports has them as median (the smallest
  possible median is 7.93, or 71.4 for C), so they are used as modes.

How the paper discretizes the triangle is not stated: each grid point gets the probability of its
cell under the continuous density (``discrete_triangular``). The depot has no time window (the
preset ignores it). The Solomon files are downloaded from SINTEF and checked against checksums.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from fleetgym.data import Instance, InstanceSet, write_instance_set
from fleetgym.spec.binding import collection_instance
from fleetgym.spec.models import CollectionSource

#: (low, mode, high) of the service time per family and class group (R and RC, or C)
FAMILIES = {
    "basic": {"r": (8.0, 10.0, 12.0), "c": (70.0, 90.0, 110.0)},
    "low_probability": {"r": (8.0, 10.0, 12.0), "c": (70.0, 90.0, 110.0)},
    "large_support": {"r": (5.0, 10.0, 15.0), "c": (45.0, 90.0, 135.0)},
    "positive_skewed": {"r": (5.0, 7.0, 15.0), "c": (45.0, 63.0, 135.0)},
}
ALPHA = {"basic": 0.95, "low_probability": 0.85, "large_support": 0.95, "positive_skewed": 0.95}
STEP = 0.1


def solomon(name: str, customers: int) -> Instance:
    """The first ``customers`` customers of Solomon's instance ``name``, downloaded and verified."""
    return collection_instance(
        CollectionSource(collection="solomon/100", name=name, first_customers=customers)
    )


def errico_instance(base: Instance, family: str) -> Instance:
    """``base`` without demands, with the service-time distribution of ``family``."""
    group = "c" if base.name.lower().startswith("c") else "r"
    low, mode, high = FAMILIES[family][group]
    n = base.n_customers
    return dataclasses.replace(
        base,
        name=f"{base.name}-{n}-{family}",
        source="examples/17 (Errico et al. 2018, Section 5.1)",
        demands=np.zeros(base.n_nodes),
        capacity=None,
        n_vehicles=None,
        vehicle_types=({"count": n},),  # never binding: at most one route per customer
        distributions={
            "service_time": {
                "dist": "discrete_triangular",
                "low": low,
                "mode": mode,
                "high": high,
                "step": STEP,
            }
        },
    )


def paper_instances() -> list[str]:
    """The 85 instances of Section 5.1: ``<name>-25`` for all 56, ``<name>-50`` for suffix 1."""
    import fleetgym as fg

    names = list(fg.load_preset("solomon1987-vrptw").grid["instance"])
    return [f"{n}-25" for n in names] + [f"{n}-50" for n in names if n[-3] == "1"]


def write_errico(
    root: str | Path, instances: Iterable[str], families: Iterable[str] = tuple(FAMILIES)
) -> dict[str, InstanceSet]:
    """Write one set per family and instance (``r101-25``), ``<root>/<family>-<instance>``."""
    sets = {}
    for point in instances:
        name, size = point.rsplit("-", 1)
        base = solomon(name, int(size))
        for family in families:
            folder = f"{family}-{point}"
            sets[folder] = write_instance_set(
                [errico_instance(base, family)],
                Path(root) / folder,
                name=f"Errico et al. {folder}",
                reference="Errico, Desaulniers, Gendreau, Rei and Rousseau (2018), Section 5.1",
                produced_by="examples/17_vrptw_errico/instances.py",
                overwrite=True,
            )
    return sets
