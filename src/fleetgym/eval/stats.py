"""Paired comparisons of two policies on the same scenarios (design §23)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from fleetgym.eval.metrics import mean_ci

__all__ = ["PairedComparison", "compare"]


@dataclass(frozen=True)
class PairedComparison:
    """``a - b`` on the same scenarios: mean difference, its CI, t-test and Wilcoxon p-values."""

    n: int
    mean_difference: float
    ci_low: float
    ci_high: float
    t_pvalue: float
    wilcoxon_pvalue: float


def compare(a: Sequence[float], b: Sequence[float]) -> PairedComparison:
    """Paired comparison of costs ``a`` and ``b`` (same scenarios, same order).

    Raises
    ------
    ValueError
        If the lengths differ or fewer than two pairs are given.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("compare needs two equally long sequences with at least two pairs")
    diff = x - y
    mean, low, high = mean_ci(diff.tolist())
    if np.allclose(diff, 0.0):
        t_p = w_p = 1.0
    else:
        t_p = float(stats.ttest_rel(x, y).pvalue)  # pyright: ignore[reportAttributeAccessIssue]
        w_p = float(stats.wilcoxon(x, y).pvalue)  # pyright: ignore[reportAttributeAccessIssue]
    return PairedComparison(x.size, mean, low, high, t_p, w_p)
