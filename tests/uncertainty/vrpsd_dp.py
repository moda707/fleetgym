"""Independent dynamic program for the expected cost of an a priori route (test helper).

Written from docs/recourse.md; shares no code with the simulator.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any


def apriori_expected_cost(
    distance: Callable[[int, int], float],
    order: Sequence[int],
    capacity: float,
    support: Callable[[int], tuple[Any, Any]],
    *,
    rule: str = "return_to_depot",
    partial: bool = True,
    depletion: str = "continue",
    per_customer: float = 0.0,
    per_unit: float = 0.0,
) -> float:
    """Expected cost of visiting ``order`` with one vehicle under the given recourse.

    ``support(node)`` gives the demand values and probabilities of a customer.
    """
    q = capacity
    k_last = len(order)

    @cache
    def f(loc: int, k: int, free: float) -> float:
        if k == k_last:
            return distance(loc, 0)
        j = order[k]
        total = distance(loc, j)
        values, probs = support(j)
        for xi, p in zip(values, probs, strict=True):
            cost, after = 0.0, free - xi
            if xi > free:
                serve_now = partial or free == q
                left = xi - free if serve_now else xi
                if rule == "return_to_depot":
                    trips = math.ceil(left / q)
                    cost += trips * 2 * distance(j, 0)
                    after = trips * q - left
                else:
                    cost += per_customer + per_unit * left
                    after = 0.0 if serve_now else free
            if (
                rule == "return_to_depot"
                and depletion == "restock"
                and after == 0
                and k + 1 < k_last
            ):
                cont = distance(j, 0) + f(0, k + 1, q)
            else:
                cont = f(j, k + 1, after)
            total += p * (cost + cont)
        return total

    return f(0, 0, q)
