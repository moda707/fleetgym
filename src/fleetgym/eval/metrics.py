"""Episode and summary metrics (design §23): mean with a 95% CI, CVaR, service rates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from fleetgym.baselines.protocol import Episode

__all__ = ["Summary", "cvar", "mean_ci", "summarize"]


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and the Student-t confidence interval ``(mean, low, high)``.

    With one value the interval is the value itself.
    """
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    if x.size < 2:
        return mean, mean, mean
    half = float(stats.t.ppf((1 + confidence) / 2, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size))
    return mean, mean - half, mean + half


def cvar(values: Sequence[float], alpha: float = 0.9) -> float:
    """Conditional value at risk of costs: the mean of the worst ``1 - alpha`` share.

    The tail holds ``ceil((1 - alpha) * n)`` episodes (at least one).
    """
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    k = max(1, math.ceil((1 - alpha) * x.size - 1e-12))
    return float(x[:k].mean())


@dataclass(frozen=True)
class Summary:
    """Summary of a set of episodes."""

    n: int
    mean_cost: float
    ci_low: float
    ci_high: float
    cvar90: float
    mean_components: dict[str, float]
    mean_decision_ms: float
    max_decision_ms: float


def summarize(episodes: Sequence[Episode]) -> Summary:
    """Summary metrics of ``episodes``."""
    costs = [e.cost for e in episodes]
    mean, low, high = mean_ci(costs)
    components = sorted({c for e in episodes for c in e.costs})
    times = [t for e in episodes for t in e.decision_seconds] or [0.0]
    return Summary(
        n=len(episodes),
        mean_cost=mean,
        ci_low=low,
        ci_high=high,
        cvar90=cvar(costs, 0.9),
        mean_components={
            c: float(np.mean([e.costs.get(c, 0.0) for e in episodes])) for c in components
        },
        mean_decision_ms=1000 * float(np.mean(times)),
        max_decision_ms=1000 * float(np.max(times)),
    )
