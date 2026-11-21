"""Instances in the distribution of Kool, van Hoof and Welling (2019).

Appendix C.1 (after Nazari et al. 2018): the depot and n customers uniform in the unit square, integer demands uniform on
{1, ..., 9}, capacity 30, 40 or 50 for n = 20, 50 or 100.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fleetgym.data import Instance, InstanceSet, write_instance_set

CAPACITY = {20: 30, 50: 40, 100: 50}


def kool_instance(rng: np.random.Generator, n: int, name: str) -> Instance:
    """One instance with ``n`` customers."""
    return Instance(
        name=name,
        source="examples/11 (Kool et al. 2019, Appendix C.1)",
        demands=np.concatenate([[0.0], rng.integers(1, 10, size=n)]),
        coords=rng.uniform(0.0, 1.0, size=(n + 1, 2)),  # the depot is uniform too
        vehicle_types=({"capacity": float(CAPACITY[n])},),
    )


def write_kool(root: str | Path, n_instances: int, seed: int = 0) -> dict[int, InstanceSet]:
    """Write one set per size, ``<root>/n<n>``."""
    rng = np.random.default_rng(seed)
    return {
        n: write_instance_set(
            [kool_instance(rng, n, f"kool-n{n}-{k}") for k in range(n_instances)],
            Path(root) / f"n{n}",
            name=f"Kool et al. n={n}",
            reference="Kool, van Hoof and Welling (2019), arXiv 1803.08475, Appendix C.1",
            produced_by=f"examples/11_cvrp_kool/instances.py, seed {seed}",
            overwrite=True,
        )
        for n in CAPACITY
    }
