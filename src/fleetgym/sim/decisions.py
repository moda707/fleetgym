"""Interface-agnostic primitive decisions.

Action interfaces translate Gymnasium actions into these primitives; the simulator understands only
primitives, so it never depends on an interface. More primitives (``Wait``, ``Accept``,
``Reject``, ``Outsource``, ``Assign``) arrive with dynamic requests in M6.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Decision", "Dispatch", "EndRoute"]


@dataclass(frozen=True)
class Dispatch:
    """Send ``vehicle`` to ``node`` (the depot ends its route on arrival)."""

    vehicle: int
    node: int


@dataclass(frozen=True)
class EndRoute:
    """Retire ``vehicle``, which is at the depot and has not left it (it stays unused)."""

    vehicle: int


Decision = Dispatch | EndRoute
