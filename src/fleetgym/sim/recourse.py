"""Recourse rules: what happens when revealed information makes planned service impossible.

Design: ``docs/recourse.md`` and ADR 0013. In this version the only failure type is
``capacity_failure``: at service start, the demand still to deliver at a customer exceeds the
vehicle's free capacity. The simulator then

1. asks the rule whether to serve what fits first (`RecourseRule.serve_first`);
2. after that (possibly empty) service, asks how to resolve the rest (`RecourseRule.on_failure`):
   ``"return_to_depot"`` (restock at the depot, come back, and check again), ``"outsource"``
   (the rest goes to a third party at a cost) or ``"split"`` (the customer is pending again with the
   demand left, for any vehicle). Returning ``None`` hands the choice to the policy at a recourse
   decision epoch;
3. after each customer is complete, asks whether to restock preventively
   (`RecourseRule.after_service`).

Rules see only the `InformationState`, never the scenario. Custom rules subclass `RecourseRule` and
register with `register_recourse`; the YAML then names them like a built-in rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from fleetgym.errors import ConfigError
from fleetgym.plugins import Registry
from fleetgym.sim.info_state import InformationState
from fleetgym.spec.binding import BoundProblem
from fleetgym.spec.models import FrozenModel
from fleetgym.spec.registries import RECOURSE_RULES, PluginEntry

__all__ = [
    "RECOURSE_CLASSES",
    "AgentDecides",
    "CapacityFailure",
    "Outsource",
    "RecourseAction",
    "RecourseRule",
    "ReturnToDepot",
    "Split",
    "register_recourse",
]

RecourseAction = Literal["return_to_depot", "outsource", "split"]


@dataclass(frozen=True)
class CapacityFailure:
    """A capacity failure at a customer.

    Attributes
    ----------
    vehicle
        The vehicle serving the customer.
    node
        The customer.
    time
        When the failure is detected (service start) or resolved (after a partial service).
    remaining
        Demand still to deliver at the customer.
    free_capacity
        The vehicle's free capacity (capacity minus load).
    capacity
        The vehicle's capacity.
    """

    vehicle: int
    node: int
    time: float
    remaining: float
    free_capacity: float
    capacity: float


class RecourseRule(ABC):
    """Base class of recourse rules for ``capacity_failure``.

    Parameters
    ----------
    problem
        The bound problem (for parameters that reference the instance).
    params
        The rule's validated parameters from the spec.
    path
        Dotted spec path of the rule, for example ``recourse.capacity_failure``.
    """

    name: ClassVar[str]
    #: Every trip to the depot during the day restocks the vehicle and its route goes on (the
    #: policy may send a vehicle to the depot at any time); otherwise a trip to the depot ends it.
    restock_trips: ClassVar[bool] = False
    #: Resolutions never make the vehicle travel: safe with route-duration and shift limits.
    no_forced_travel: ClassVar[bool] = False

    def __init__(self, problem: BoundProblem, params: FrozenModel, path: str) -> None:
        self.problem = problem
        self.params = params
        self.path = path

    def serve_first(self, failure: CapacityFailure, info: InformationState) -> bool:
        """Serve what fits before resolving the failure.

        Default: the ``partial_service`` param. The simulator always serves what fits when the
        vehicle's whole capacity is free, so demand above capacity cannot loop on restocks.
        """
        return bool(getattr(self.params, "partial_service", True))

    @abstractmethod
    def on_failure(self, failure: CapacityFailure, info: InformationState) -> RecourseAction | None:
        """How to resolve the demand left after `serve_first`; ``None`` asks the policy."""

    def options(self) -> tuple[RecourseAction, ...]:
        """Actions the policy may choose at a recourse epoch (rules that return ``None``)."""
        return ()

    def after_service(self, vehicle: int, info: InformationState) -> bool:
        """Restock at the depot now, after completing a customer (default: never)."""
        return False

    def must_restock(self, vehicle: int, info: InformationState) -> bool:
        """The vehicle may only go to the depot now (default: never)."""
        return False

    def may_end(self, vehicle: int, info: InformationState, customers_reachable: bool) -> bool:
        """Whether a vehicle at the depot may end its day now (default: always)."""
        return True

    def outsourcing_cost(self, node: int, quantity: float) -> float:
        """Cost of outsourcing ``quantity`` at ``node``: ``per_customer + per_unit * quantity``.

        Raises
        ------
        ConfigError
            If the rule has no outsourcing costs.
        """
        if not hasattr(self.params, "per_unit"):
            raise ConfigError(
                f"recourse rule {self.name!r} cannot outsource: it has no outsourcing cost",
                source=self.problem.spec.source,
                key_path=self.path,
            )
        per_customer = _at(self.problem.value(f"{self.path}.per_customer"), node)
        per_unit = _at(self.problem.value(f"{self.path}.per_unit"), node)
        return per_customer + per_unit * quantity


def _at(value: Any, node: int) -> float:
    arr = np.asarray(value, dtype=float)
    return float(arr) if arr.ndim == 0 else float(arr[node])


RECOURSE_CLASSES: Registry[type[RecourseRule]] = Registry("recourse rule implementation")
"""Runtime recourse classes keyed by the spec's rule name."""


@RECOURSE_CLASSES.register("return_to_depot")
class ReturnToDepot(RecourseRule):
    """Restock at the depot and come back (``partial_service``, ``on_exact_depletion``)."""

    name = "return_to_depot"

    def on_failure(self, failure: CapacityFailure, info: InformationState) -> RecourseAction:
        """Always ``return_to_depot``."""
        return "return_to_depot"

    def after_service(self, vehicle: int, info: InformationState) -> bool:
        """With ``on_exact_depletion: restock``: restock when emptied and customers are pending."""
        if getattr(self.params, "on_exact_depletion", "continue") != "restock":
            return False
        v = info.vehicle(vehicle)
        return v.load >= v.capacity and bool(info.pending().any())


@RECOURSE_CLASSES.register("outsource")
class Outsource(RecourseRule):
    """Outsource the demand that does not fit."""

    name = "outsource"

    def on_failure(self, failure: CapacityFailure, info: InformationState) -> RecourseAction:
        """Always ``outsource``."""
        return "outsource"


@RECOURSE_CLASSES.register("agent_decides")
class AgentDecides(RecourseRule):
    """The policy chooses among ``options`` at a recourse decision epoch."""

    name = "agent_decides"

    def on_failure(self, failure: CapacityFailure, info: InformationState) -> None:
        """Always ``None``: ask the policy."""
        return None

    def options(self) -> tuple[RecourseAction, ...]:
        """The ``options`` param."""
        return tuple(getattr(self.params, "options", ()))


@RECOURSE_CLASSES.register("split")
class Split(RecourseRule):
    """Serve what fits; the rest stays open for any vehicle (Dastpak, Errico and Jabali).

    An empty vehicle (no free capacity) must go to the depot; every trip to the depot during the day
    restocks the vehicle and its route goes on. With ``end_early: false`` a vehicle at the depot may
    end its day only when no customer is reachable.
    """

    name = "split"
    restock_trips = True
    no_forced_travel = True

    def serve_first(self, failure: CapacityFailure, info: InformationState) -> bool:
        """Always: serve what fits."""
        return True

    def on_failure(self, failure: CapacityFailure, info: InformationState) -> RecourseAction:
        """Always ``split``: the customer is pending again with the demand left."""
        return "split"

    def must_restock(self, vehicle: int, info: InformationState) -> bool:
        """When the vehicle has no free capacity."""
        v = info.vehicle(vehicle)
        return v.load >= v.capacity

    def may_end(self, vehicle: int, info: InformationState, customers_reachable: bool) -> bool:
        """``end_early``, or when no customer is reachable."""
        return bool(getattr(self.params, "end_early", False)) or not customers_reachable


def register_recourse(
    name: str, *, params: type[FrozenModel], description: str = ""
) -> Callable[[type[RecourseRule]], type[RecourseRule]]:
    """Register a custom recourse rule under ``name`` (spec parameters and runtime class).

    The parameter model is validated from YAML like a built-in rule's; give it ``per_customer``
    and ``per_unit`` fields if the rule outsources.

    Examples
    --------
    >>> from fleetgym.spec.models import FrozenModel
    >>> class NearDepotParams(FrozenModel):
    ...     radius: float
    ...     per_unit: float = 1.0
    ...     per_customer: float = 0.0
    >>> @register_recourse("docs.near_depot_restock", params=NearDepotParams)  # doctest: +SKIP
    ... class NearDepot(RecourseRule):
    ...     def on_failure(self, failure, info):
    ...         near = info.distances[failure.node, 0] <= self.params.radius
    ...         return "return_to_depot" if near else "outsource"
    """

    def decorator(cls: type[RecourseRule]) -> type[RecourseRule]:
        if name in RECOURSE_RULES:
            raise ConfigError(f"recourse rule {name!r} is already registered")
        RECOURSE_RULES.register(name, PluginEntry(params, description=description))
        RECOURSE_CLASSES.register(name, cls)
        cls.name = name
        return cls

    return decorator
