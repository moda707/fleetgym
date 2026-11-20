"""Reward models: weighted aggregation of cost components.

The reward of a step is the negative weighted sum of the costs realized during the step, so the
episode return is the negative total cost. A lexicographic objective ``[c_1, ..., c_L]`` is
scalarized with weights ``LEXICOGRAPHIC_BASE ** (L - k)`` for ``c_k`` (ADR 0009); every component is
still reported separately in ``info["costs"]``, and benchmarks compare components lexicographically.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from fleetgym.spec.models import ObjectiveSpec

__all__ = ["LEXICOGRAPHIC_BASE", "RewardModel"]

LEXICOGRAPHIC_BASE = 1e6
"""Weight ratio between consecutive levels of a lexicographic objective."""


class RewardModel:
    """Turn realized costs into a scalar reward.

    Parameters
    ----------
    objective
        The spec's objective.
    """

    def __init__(self, objective: ObjectiveSpec) -> None:
        if objective.lexicographic is not None:
            levels = objective.lexicographic
            weights = {c: LEXICOGRAPHIC_BASE ** (len(levels) - 1 - k) for k, c in enumerate(levels)}
        else:
            weights = dict(objective.weights)
        self.weights: Mapping[str, float] = MappingProxyType(weights)

    def cost(self, costs: Mapping[str, float]) -> float:
        """Weighted cost; components without a weight do not count."""
        return float(sum(self.weights.get(c, 0.0) * amount for c, amount in costs.items()))

    def __call__(self, costs: Mapping[str, float]) -> float:
        """The reward: minus the weighted cost."""
        return -self.cost(costs)
