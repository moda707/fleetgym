"""The `Constraint` interface and the constraint registry.

A constraint answers, for the vehicle about to decide, which candidate customers keep the plan
feasible (`mask`), and reports soft-violation costs (`penalty`). It sees only the
`InformationState` (ADR 0009): never the scenario, so masks cannot use future information.

Masks are evaluated for customer candidates only; returning to the depot is never masked by a
constraint, which is why every hard constraint must look ahead far enough that a vehicle can always
return feasibly from wherever a mask lets it go.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from fleetgym.plugins import Registry
from fleetgym.sim.decisions import Decision, Dispatch
from fleetgym.sim.events import Event
from fleetgym.sim.info_state import InformationState
from fleetgym.spec.binding import BoundProblem
from fleetgym.spec.models import FrozenModel

__all__ = ["CONSTRAINT_CLASSES", "Constraint", "Violation"]


@dataclass(frozen=True)
class Violation:
    """Why a decision violates a constraint."""

    constraint: str
    message: str


class Constraint(ABC):
    """Base class of constraints.

    Parameters
    ----------
    problem
        The bound problem.
    params
        The constraint's validated parameters from the spec.
    """

    name: ClassVar[str]

    def __init__(self, problem: BoundProblem, params: FrozenModel) -> None:
        self.problem = problem
        self.params = params

    def init_state(self, info: InformationState) -> Any:
        """Initial constraint state (pure data). Most constraints need none."""
        return None

    @abstractmethod
    def mask(
        self, state: Any, info: InformationState, vehicle: int, candidates: NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        """Which candidate customers ``vehicle`` may visit next without violating the constraint."""

    def check(self, state: Any, info: InformationState, decision: Decision) -> Violation | None:
        """Hard feasibility of one decision (default: the mask of a dispatch to a customer)."""
        if isinstance(decision, Dispatch) and decision.node != 0:
            nodes = np.array([decision.node])
            if not self.mask(state, info, decision.vehicle, nodes)[0]:
                return Violation(self.name, f"node {decision.node} is infeasible for {self.name}")
        return None

    def update(self, state: Any, event: Event) -> None:  # noqa: B027 - optional hook
        """Update the constraint state after an event (default: nothing)."""

    def penalty(self, state: Any, info: InformationState, event: Event) -> Mapping[str, float]:
        """Soft-violation costs caused by ``event``, by cost component (default: none)."""
        return {}


CONSTRAINT_CLASSES: Registry[type[Constraint]] = Registry("constraint implementation")
"""Runtime constraint classes keyed by the spec's constraint name."""
