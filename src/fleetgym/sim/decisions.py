"""Interface-agnostic primitive decisions.

Action interfaces translate Gymnasium actions into these primitives; the simulator understands only
primitives, so it never depends on an interface. More primitives (``Wait``, ``Accept``,
``Reject``, ``Assign``) arrive with dynamic requests in M6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["Decision", "Dispatch", "EndRoute", "RecourseChoice"]


@dataclass(frozen=True)
class Dispatch:
    """Send ``vehicle`` to ``node`` (the depot ends its route on arrival)."""

    vehicle: int
    node: int


@dataclass(frozen=True)
class EndRoute:
    """End ``vehicle``'s day at the depot.

    A vehicle that never left is retired (it stays unused); a vehicle at the depot after a
    preventive restock ends its route there.
    """

    vehicle: int


@dataclass(frozen=True)
class RecourseChoice:
    """Resolve the capacity failure of ``vehicle`` at a recourse epoch."""

    vehicle: int
    action: Literal["return_to_depot", "outsource"]


Decision = Dispatch | EndRoute | RecourseChoice
