"""Interface-agnostic primitive decisions.

Action interfaces translate Gymnasium actions into these primitives; the simulator understands only
primitives, so it never depends on an interface. ``Accept``, ``Reject``, ``OutsourceRequest`` and
``Assign`` decide on a request at a request epoch (planned routing, ADR 0014).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Accept",
    "Assign",
    "Decision",
    "Dispatch",
    "EndRoute",
    "OutsourceRequest",
    "RecourseChoice",
    "Reject",
    "Wait",
]


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


@dataclass(frozen=True)
class Accept:
    """Accept request ``node``; the routing rule inserts it into the best vehicle's plan."""

    node: int


@dataclass(frozen=True)
class Assign:
    """Accept request ``node`` and insert it into ``vehicle``'s plan."""

    node: int
    vehicle: int


@dataclass(frozen=True)
class Reject:
    """Reject request ``node`` (cost component ``rejected``)."""

    node: int


@dataclass(frozen=True)
class OutsourceRequest:
    """Hand request ``node`` to a third party (cost component ``outsourcing``)."""

    node: int


@dataclass(frozen=True)
class Wait:
    """``vehicle`` waits where it is until the next event (or its latest safe departure)."""

    vehicle: int


Decision = Dispatch | EndRoute | RecourseChoice | Wait | Accept | Assign | Reject | OutsourceRequest
