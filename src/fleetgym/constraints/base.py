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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from fleetgym.plugins import PluginError, Registry
from fleetgym.sim.decisions import Decision, Dispatch
from fleetgym.sim.events import Event
from fleetgym.sim.info_state import InformationState
from fleetgym.spec.binding import BoundProblem
from fleetgym.spec.models import FrozenModel
from fleetgym.spec.registries import CONSTRAINTS, PluginEntry

__all__ = ["CONSTRAINT_CLASSES", "Constraint", "Violation", "register_constraint"]


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


def register_constraint(
    name: str, *, params: type[FrozenModel], description: str = ""
) -> Callable[[type[Constraint]], type[Constraint]]:
    """Register a custom constraint: its spec parameter model and its runtime class.

    The constraint is then usable in YAML under ``constraints.<name>``. Its ``mask`` must look ahead
    far enough that a vehicle can always return to the depot (see the module docstring). The
    independent validator does not know custom constraints; test them with your own checks.

    Examples
    --------
    >>> class MaxStopsParams(FrozenModel):  # doctest: +SKIP
    ...     max: int
    >>> @register_constraint("max_stops", params=MaxStopsParams)  # doctest: +SKIP
    ... class MaxStops(Constraint): ...
    """

    def decorator(cls: type[Constraint]) -> type[Constraint]:
        if name in CONSTRAINTS:
            raise PluginError(f"constraint {name!r} is already registered")
        CONSTRAINTS.register(name, PluginEntry(params, description=description))
        CONSTRAINT_CLASSES.register(name, cls)
        cls.name = name
        return cls

    return decorator
