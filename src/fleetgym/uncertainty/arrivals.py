"""Request arrival processes (design §7).

* ``poisson``: homogeneous Poisson process with ``rate`` arrivals per time unit, sampled from
  exponential inter-arrival times.
* ``nhpp``: nonhomogeneous Poisson process with a piecewise-constant rate profile, sampled by
  thinning (Lewis and Shedler): candidates from a homogeneous process at the maximum rate, each kept
  with probability ``rate(t) / max_rate``.

Arrivals are generated on ``[start, end)``; the cutoff truncates the window (plan Q25: requests
after the cutoff are not generated).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["expected_count", "piecewise_rate", "poisson_arrivals", "thinned_arrivals"]


def poisson_arrivals(
    rng: np.random.Generator, rate: float, start: float, end: float
) -> NDArray[np.float64]:
    """Arrival times of a homogeneous Poisson process with ``rate`` on ``[start, end)``."""
    times: list[float] = []
    if rate <= 0 or end <= start:
        return np.zeros(0)
    t = start
    while True:
        t += float(rng.exponential(1.0 / rate))
        if t >= end:
            return np.asarray(times)
        times.append(t)


def piecewise_rate(
    breaks: Sequence[float], rates: Sequence[float], t: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Rate at times ``t``: ``rates[i]`` on ``[breaks[i], breaks[i+1])``, 0 outside the profile."""
    edges = np.asarray(breaks, dtype=float)
    values = np.asarray(rates, dtype=float)
    index = np.searchsorted(edges, t, side="right") - 1
    inside = (index >= 0) & (index < values.size)
    return np.where(inside, values[np.clip(index, 0, values.size - 1)], 0.0)


def thinned_arrivals(
    rng: np.random.Generator,
    breaks: Sequence[float],
    rates: Sequence[float],
    start: float,
    end: float,
) -> NDArray[np.float64]:
    """Arrival times of a piecewise-constant NHPP on ``[start, end)`` by thinning."""
    top = float(max(rates)) if len(rates) else 0.0
    candidates = poisson_arrivals(rng, top, start, end)
    if candidates.size == 0:
        return candidates
    keep = rng.random(candidates.size) * top < piecewise_rate(breaks, rates, candidates)
    return candidates[keep]


def expected_count(
    breaks: Sequence[float], rates: Sequence[float], start: float, end: float
) -> float:
    """Integral of the piecewise-constant rate over ``[start, end)``."""
    total = 0.0
    for lo, hi, rate in zip(breaks, breaks[1:], rates, strict=False):
        overlap = min(hi, end) - max(lo, start)
        if overlap > 0:
            total += rate * overlap
    return total
