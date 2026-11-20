"""Degree of dynamism of a set of requests (design §7, dynamism knobs).

Definitions used here (check against the papers before comparing published values):

* degree of dynamism (Lund et al. 1996): the share of requests released after the start,
  ``n_dynamic / n_total``;
* effective degree of dynamism (Larsen et al. 2002): the mean of ``(t_i - start) / (end - start)``
  over all requests, where ``t_i`` is request ``i``'s release time (``start`` for requests known at
  the start) and ``[start, end)`` is the planning horizon.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["degree_of_dynamism", "effective_degree_of_dynamism"]


def _released(release: ArrayLike) -> np.ndarray:
    times = np.asarray(release, dtype=float)
    return times[np.isfinite(times)]


def degree_of_dynamism(release: ArrayLike, start: float) -> float:
    """Share of requests released after ``start`` (0 without requests)."""
    times = _released(release)
    return float(np.mean(times > start)) if times.size else 0.0


def effective_degree_of_dynamism(release: ArrayLike, start: float, end: float) -> float:
    """Mean relative release time over the horizon ``[start, end)`` (0 without requests).

    Raises
    ------
    ValueError
        If ``end <= start``.
    """
    if end <= start:
        raise ValueError(f"the horizon must have end > start, got [{start}, {end})")
    times = _released(release)
    if times.size == 0:
        return 0.0
    return float(np.mean((times - start) / (end - start)))
