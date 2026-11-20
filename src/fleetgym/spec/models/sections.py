"""Models of the sections of a `ProblemSpec` other than uncertainty.

Sections: ``meta``, ``instance_source``, ``network``, ``fleet``, ``constraints``, ``dynamics``,
``decisions``, ``recourse``, ``objective`` and ``agents``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Annotated, Any, ClassVar, Literal

from pydantic import (
    Field,
    GetJsonSchemaHandler,
    ValidationError,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from fleetgym.data.generators import validate_generator_params
from fleetgym.plugins import PluginError
from fleetgym.spec.models.base import (
    FlatPluginModel,
    FrozenJSON,
    FrozenMap,
    FrozenModel,
    InstanceRef,
    NonNegativeOrRef,
    PositiveOrRef,
    ReservedParams,
    StatusLiteral,
    deep_thaw,
    reraise_nested,
    validate_plugin_params,
)
from fleetgym.spec.models.distributions import DistSpec
from fleetgym.spec.registries import (
    ARRIVAL_PROCESSES,
    CONSTRAINTS,
    COST_COMPONENTS,
    FAILURE_TYPES,
    INTERFACES,
    NETWORKS,
    RECOURSE_RULES,
    ROUTING_RULES,
    PluginEntry,
)

__all__ = [
    "LEXICOGRAPHIC_BASE",
    "AgentsSpec",
    "ArrivalSpec",
    "DecisionsSpec",
    "DynamicsSpec",
    "FilesSource",
    "FleetSpec",
    "GeneratorSource",
    "InstanceSourceSpec",
    "Meta",
    "NetworkSpec",
    "ObjectiveSpec",
    "RecourseRuleSpec",
    "RequestClassSpec",
    "VehicleTypeSpec",
]

PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class _NoParams(FrozenModel):
    """Plugin without parameters."""


# --------------------------------------------------------------------------- meta


class Meta(FrozenModel):
    """Documentation and replication data. Not hashed and never inherited through ``extends``.

    Attributes
    ----------
    status
        ``stable``: passes the replication suite. ``experimental``: implemented, not yet
        replicated. ``planned``: validates against the schema but cannot be made into an
        environment in this version.
    references
        Citation keys of the defining papers.
    instances
        Benchmark instances used for replication (free-form).
    expected
        Replication targets (free-form until the replication suite defines its schema).
    """

    name: str | None = None
    description: str | None = None
    status: StatusLiteral = "experimental"
    references: tuple[str, ...] = ()
    instances: FrozenJSON = None
    expected: FrozenJSON = None


# --------------------------------------------------------------------------- instance source


class GeneratorSource(FrozenModel):
    """Generate a fresh instance per episode with a registered generator.

    ``params`` are checked against the generator's parameter model when the config is loaded.
    """

    type: Literal["generator"] = "generator"
    name: str
    params: FrozenJSON = None

    @model_validator(mode="after")
    def _known_generator(self) -> GeneratorSource:
        try:
            validate_generator_params(self.name, deep_thaw(self.params) if self.params else {})
        except PluginError as exc:
            raise ValidationError.from_exception_data(
                type(self).__name__,
                [
                    {
                        "type": "value_error",
                        "loc": ("name",),
                        "input": self.name,
                        "ctx": {"error": ValueError(str(exc))},
                    }
                ],
            ) from None
        except ValidationError as exc:
            raise reraise_nested(exc, type(self).__name__, ("params",)) from None
        return self


class FilesSource(FrozenModel):
    """Load instances from files (relative paths resolve against the config file)."""

    type: Literal["files"] = "files"
    paths: tuple[str, ...] = Field(min_length=1)
    format: Literal["vrplib", "solomon"] | None = None


InstanceSourceSpec = Annotated[GeneratorSource | FilesSource, Field(discriminator="type")]


# --------------------------------------------------------------------------- network


class EuclideanParams(FrozenModel):
    """Euclidean distances between instance coordinates.

    Attributes
    ----------
    rounding
        ``none`` keeps double precision. ``nearest_int`` rounds each arc to the nearest integer,
        as in the TSPLIB ``EUC_2D`` convention.
    speed
        Travel time equals distance divided by ``speed``.
    time_model
        Reserved for time-dependent travel (not in this version).
    """

    rounding: Literal["none", "nearest_int"] = "none"
    speed: PositiveFloat = 1.0
    time_model: FrozenJSON = None


class MatrixParams(FrozenModel):
    """Explicit distance (and optional time) matrices taken from the instance."""

    speed: PositiveFloat = 1.0


NETWORKS.register("euclidean", PluginEntry(EuclideanParams, description="Euclidean coordinates"))
NETWORKS.register("matrix", PluginEntry(MatrixParams, description="Explicit matrices"))


class NetworkSpec(FlatPluginModel):
    """The travel network, written flat in YAML: ``{type: euclidean, rounding: none}``."""

    discriminator: ClassVar[str] = "type"
    registry = NETWORKS


# --------------------------------------------------------------------------- fleet


class VehicleTypeSpec(FrozenModel):
    """One vehicle type.

    Attributes
    ----------
    capacity
        Vehicle capacity; ``from_instance`` reads the instance's capacity.
    count
        Number of vehicles; required for ``fixed`` and ``limited`` fleets, forbidden for
        ``unlimited`` ones. ``from_instance`` reads the instance's vehicle count.
    fixed_cost
        Cost charged once per vehicle used.
    """

    name: str = Field(min_length=1)
    capacity: PositiveOrRef | None = None
    count: PositiveInt | InstanceRef | None = None
    fixed_cost: NonNegativeFloat = 0.0


class FleetSpec(FrozenModel):
    """The fleet: vehicle types, sizing mode and shift.

    Attributes
    ----------
    sizing
        ``fixed``: exactly ``count`` vehicles per type are available. ``limited``: at most
        ``count`` may be used. ``unlimited``: as many as needed.
    shift
        Start and end of the working day, or ``None`` for no shift limits.
    """

    sizing: Literal["fixed", "limited", "unlimited"] = "unlimited"
    types: tuple[VehicleTypeSpec, ...] = Field(min_length=1)
    shift: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> FleetSpec:
        names = [t.name for t in self.types]
        if len(set(names)) != len(names):
            raise ValueError(f"vehicle type names must be unique, got {names}")
        for t in self.types:
            if self.sizing == "unlimited" and t.count is not None:
                raise ValueError(
                    f"vehicle type {t.name!r}: `count` is not allowed with unlimited sizing"
                )
            if self.sizing != "unlimited" and t.count is None:
                raise ValueError(
                    f"vehicle type {t.name!r}: `count` is required with {self.sizing} sizing"
                )
        if self.shift is not None and not self.shift[0] < self.shift[1]:
            raise ValueError(f"`shift` must be (start, end) with start < end, got {self.shift}")
        return self


# --------------------------------------------------------------------------- constraints


class CapacityParams(FrozenModel):
    """Vehicle capacity (single dimension)."""


class TimeWindowsParams(FrozenModel):
    """Customer time windows.

    Attributes
    ----------
    soft
        Hard windows are masked; soft windows allow lateness, which is reported as the
        ``lateness`` cost component and priced by ``objective.lateness``.
    waiting
        Whether a vehicle arriving early may wait for the window to open.
    """

    soft: bool = False
    waiting: Literal["allowed", "forbidden"] = "allowed"


class RouteDurationParams(FrozenModel):
    """Maximum route duration.

    ``max`` is a number, ``from_instance``, or ``depot_due_time`` (the closing time of the depot's
    time window in the instance).
    """

    max: PositiveOrRef | Literal["depot_due_time"]


class OutsourcingParams(FrozenModel):
    """Outsourcing of requests to a third party at a cost (semantics finalized at Gate 2)."""

    cost: Literal["per_customer", "per_request"]
    value: NonNegativeOrRef


for _name, _model, _description in (
    ("capacity", CapacityParams, "Vehicle capacity"),
    ("time_windows", TimeWindowsParams, "Hard or soft time windows"),
    ("route_duration", RouteDurationParams, "Maximum route duration"),
    ("outsourcing", OutsourcingParams, "Outsourcing at a cost"),
):
    CONSTRAINTS.register(_name, PluginEntry(_model, description=_description))
CONSTRAINTS.register(
    "energy",
    PluginEntry(ReservedParams, reserved=True, description="Battery and energy (EV module)"),
)


def _validate_keyed_plugins(registry: Any, data: Any, title: str) -> dict[str, FrozenModel]:
    """Validate a map ``{plugin_name: params}`` against ``registry``, keeping error locations."""
    if not isinstance(data, Mapping):
        raise ValueError(f"expected a mapping of {registry.kind} names to parameters")
    out: dict[str, FrozenModel] = {}
    errors: list[ValidationError] = []
    for key, params in data.items():
        if isinstance(params, FrozenModel):
            out[key] = params
            continue
        try:
            out[key] = validate_plugin_params(
                registry, key, params if params is not None else {}, title=title, loc=(key,)
            )
        except ValidationError as exc:
            errors.append(exc)
    if errors:
        details = [d for e in errors for d in reraise_nested(e, title).errors()]
        raise ValidationError.from_exception_data(title, details)  # pyright: ignore[reportArgumentType]
    return out


class ConstraintsSpec(FrozenModel):
    """Constraints keyed by name: ``{capacity: {}, time_windows: {soft: false}}``."""

    items: FrozenMap[str, FrozenModel] = Field(default_factory=dict)

    @model_validator(mode="wrap")
    @classmethod
    def _keyed(cls, data: Any, handler: Callable[[Any], Any]) -> Any:
        if isinstance(data, ConstraintsSpec):
            return data
        if data is None:
            data = {}
        return handler({"items": _validate_keyed_plugins(CONSTRAINTS, data, cls.__name__)})

    @model_serializer(mode="plain")
    def _out(self) -> dict[str, Any]:
        return {k: v.model_dump(mode="json") for k, v in self.items.items()}

    def __contains__(self, name: object) -> bool:
        return name in self.items

    def __getitem__(self, name: str) -> FrozenModel:
        return self.items[name]

    def names(self) -> tuple[str, ...]:
        """Names of the configured constraints, in config order."""
        return tuple(self.items)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        properties: dict[str, JsonSchemaValue] = {}
        for name, entry in CONSTRAINTS.items():
            schema = handler.resolve_ref_schema(handler(entry.params.__pydantic_core_schema__))
            properties[name] = {**schema, "description": entry.description}
        return {"type": "object", "properties": properties, "additionalProperties": False}


# --------------------------------------------------------------------------- dynamics


class PoissonArrivalParams(FrozenModel):
    """Homogeneous Poisson arrivals with ``rate`` requests per time unit."""

    rate: PositiveFloat


class RateProfile(FrozenModel):
    """Piecewise-constant rate: ``rates[i]`` applies on ``[breaks[i], breaks[i+1])``."""

    breaks: tuple[float, ...] = Field(min_length=2)
    rates: tuple[NonNegativeFloat, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _shape(self) -> RateProfile:
        if len(self.breaks) != len(self.rates) + 1:
            raise ValueError(
                "`breaks` needs one more entry than `rates` "
                f"({len(self.breaks)} vs {len(self.rates)})"
            )
        if any(b >= c for b, c in zip(self.breaks, self.breaks[1:], strict=False)):
            raise ValueError("`breaks` must be strictly increasing")
        return self


class NHPPArrivalParams(FrozenModel):
    """Nonhomogeneous Poisson arrivals with a piecewise-constant rate profile."""

    profile: RateProfile


ARRIVAL_PROCESSES.register(
    "poisson", PluginEntry(PoissonArrivalParams, description="Homogeneous Poisson process")
)
ARRIVAL_PROCESSES.register(
    "nhpp", PluginEntry(NHPPArrivalParams, description="Nonhomogeneous Poisson process")
)


class ArrivalSpec(FlatPluginModel):
    """Request arrival process, flat in YAML: ``{process: poisson, rate: 0.3}``."""

    discriminator: ClassVar[str] = "process"
    registry = ARRIVAL_PROCESSES


class RequestClassSpec(FrozenModel):
    """Distributions of the attributes of a dynamically arriving request."""

    location: DistSpec | None = None
    demand: DistSpec | None = None
    time_window_width: DistSpec | None = None
    service_time: DistSpec | None = None


class DynamicsSpec(FrozenModel):
    """Requests arriving during the day.

    Attributes
    ----------
    cutoff
        Time after which no requests arrive.
    max_requests
        Upper bound on the number of requests per episode; observations are padded to it.
    overflow
        What happens if a scenario has more requests than ``max_requests``: ``error`` raises,
        ``reject`` rejects the extra requests automatically (logged and charged as rejected).
    initial_requests
        ``instance``: the instance's customers are requests known at the start. ``none``: only
        requests arriving during the day (the instance's customers serve as data, for example
        for the location distribution).
    """

    arrivals: ArrivalSpec
    request_class: RequestClassSpec = Field(default_factory=RequestClassSpec)
    cutoff: NonNegativeFloat | None = None
    cancellations: DistSpec | None = None
    max_requests: PositiveInt
    overflow: Literal["error", "reject"] = "error"
    initial_requests: Literal["instance", "none"] = "instance"


# --------------------------------------------------------------------------- decisions


class InterfaceParams(FrozenModel):
    """Marker for action interfaces (parameters are given in ``decisions``)."""


for _name, _description, _reserved in (
    ("next_node", "Next node for the vehicle that triggered the epoch", False),
    ("accept_reject", "Accept, reject or outsource each new request", False),
    ("assign", "Vehicle for each new request", False),
    ("vehicle_node", "Vehicle and next node jointly", True),
    ("plan", "Full set of routes", True),
    ("operator", "Local-search operator selection", True),
    ("wait_reposition", "Wait or reposition idle vehicles", True),
    ("charge", "Charging decisions (EV module)", True),
):
    INTERFACES.register(
        _name, PluginEntry(InterfaceParams, reserved=_reserved, description=_description)
    )
ROUTING_RULES.register(
    "cheapest_insertion",
    PluginEntry(_NoParams, description="Insert at the cheapest feasible position"),
)

#: Interfaces whose routing is done by a built-in rule (``decisions.routing`` is required).
INTERFACES_NEEDING_ROUTING = frozenset({"accept_reject", "assign"})


class DecisionsSpec(FrozenModel):
    """When the agent decides and what it decides.

    Attributes
    ----------
    epochs
        ``on_event``: when a vehicle becomes idle or a request arrives (time-ordered).
        ``sequential``: one vehicle builds its whole route before the next starts (static
        problems only). ``periodic``: every ``period`` time units. ``hybrid``: reserved.
    interface
        Name of the action interface.
    routing
        Built-in routing rule for interfaces that do not route themselves.
    commitment
        ``next_stop``: a vehicle commits only to its next destination. ``diversion``: reserved.
    """

    epochs: Literal["on_event", "sequential", "periodic", "hybrid"] = "on_event"
    period: PositiveFloat | None = None
    interface: str = "next_node"
    routing: str | None = None
    commitment: Literal["next_stop", "diversion"] = "next_stop"

    @model_validator(mode="after")
    def _consistent(self) -> DecisionsSpec:
        if self.interface not in INTERFACES:
            raise ValueError(
                f"unknown interface {self.interface!r}; known: {', '.join(INTERFACES.names())}"
            )
        if self.routing is not None and self.routing not in ROUTING_RULES:
            raise ValueError(
                f"unknown routing rule {self.routing!r}; known: {', '.join(ROUTING_RULES.names())}"
            )
        if self.interface in INTERFACES_NEEDING_ROUTING and self.routing is None:
            raise ValueError(f"interface {self.interface!r} requires `routing`")
        if (self.epochs == "periodic") != (self.period is not None):
            raise ValueError("`period` is required with periodic epochs and only then")
        return self


# --------------------------------------------------------------------------- recourse


for _name, _description in (
    ("capacity_failure", "Demand revealed at a customer exceeds the remaining load"),
):
    FAILURE_TYPES.register(_name, PluginEntry(_NoParams, description=_description))
for _name, _description in (
    ("time_window_failure", "Arrival after a hard time window closes"),
    ("energy_failure", "Battery shortfall (EV module)"),
):
    FAILURE_TYPES.register(_name, PluginEntry(_NoParams, reserved=True, description=_description))


class ReturnToDepotParams(FrozenModel):
    """Classical detour-to-depot recourse (docs/recourse.md).

    Attributes
    ----------
    partial_service
        ``True``: serve what fits, restock at the depot, come back and finish. ``False``: restock
        first, then come back and serve (demand above capacity still needs several trips).
    on_exact_depletion
        ``continue``: a vehicle emptied exactly by a customer continues, and fails at the next
        customer with positive demand. ``restock``: it restocks at the depot right away (when
        customers are still pending) and decides its next customer from there.
    """

    partial_service: bool = True
    on_exact_depletion: Literal["continue", "restock"] = "continue"


class _OutsourceCosts(FrozenModel):
    per_customer: NonNegativeOrRef = 0.0
    per_unit: NonNegativeOrRef = 0.0

    def _require_costs(self) -> None:
        if not {"per_customer", "per_unit"} & self.model_fields_set:
            raise ValueError(
                "outsourcing needs its cost: give `per_customer` and/or `per_unit` (provisional "
                "until the VRP-SDO model is fixed at Gate 2)"
            )


class OutsourceParams(_OutsourceCosts):
    """Outsource the demand that does not fit; the vehicle continues its route.

    The cost (component ``outsourcing``) is ``per_customer + per_unit * quantity`` for each
    outsourced customer.

    Attributes
    ----------
    partial_service
        ``True``: serve what fits and outsource the rest. ``False``: outsource the whole remaining
        demand of the failing customer.
    """

    partial_service: bool = True

    @model_validator(mode="after")
    def _costs(self) -> OutsourceParams:
        self._require_costs()
        return self


class AgentDecidesParams(_OutsourceCosts):
    """Hand each failure to the policy at a recourse decision epoch.

    Attributes
    ----------
    options
        What the policy may choose: ``return_to_depot`` (restock and come back) and/or
        ``outsource`` (priced like the ``outsource`` rule).
    partial_service
        Serve what fits before the recourse epoch.
    """

    options: tuple[Literal["return_to_depot", "outsource"], ...] = Field(
        default=("return_to_depot", "outsource"), min_length=1
    )
    partial_service: bool = True

    @model_validator(mode="after")
    def _costs(self) -> AgentDecidesParams:
        if len(set(self.options)) != len(self.options):
            raise ValueError("`options` lists an option twice")
        if "outsource" in self.options:
            self._require_costs()
        return self


for _name, _model, _description in (
    ("return_to_depot", ReturnToDepotParams, "Restock at the depot and come back"),
    ("outsource", OutsourceParams, "Outsource the demand that does not fit"),
    ("agent_decides", AgentDecidesParams, "The policy chooses at a recourse decision epoch"),
):
    RECOURSE_RULES.register(_name, PluginEntry(_model, description=_description))
RECOURSE_RULES.register(
    "split",
    PluginEntry(
        _NoParams,
        reserved=True,
        description="Leave the rest as a new request (needs dynamic requests)",
    ),
)


class RecourseRuleSpec(FlatPluginModel):
    """A recourse rule: a name, or ``{rule: <name>, <params>}``."""

    discriminator: ClassVar[str] = "rule"
    registry = RECOURSE_RULES

    @classmethod
    def _preprocess(cls, data: Any) -> Any:
        return {"rule": data} if isinstance(data, str) else data

    @model_serializer(mode="wrap")
    def _flatten_out(self, handler: Callable[[Any], Any]) -> Any:
        params = self.params.model_dump(mode="json")
        return self.name if not params else {"rule": self.name, **params}


class RecourseSpec(FrozenModel):
    """Recourse rules keyed by failure type: ``{capacity_failure: return_to_depot}``."""

    rules: FrozenMap[str, RecourseRuleSpec] = Field(default_factory=dict)

    @model_validator(mode="wrap")
    @classmethod
    def _keyed(cls, data: Any, handler: Callable[[Any], Any]) -> Any:
        if isinstance(data, RecourseSpec):
            return data
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ValueError("expected a mapping of failure types to recourse rules")
        unknown = [k for k in data if k not in FAILURE_TYPES]
        if unknown:
            raise ValueError(
                f"unknown failure type(s) {unknown}; known: {', '.join(FAILURE_TYPES.names())}"
            )
        return handler({"rules": data})

    @model_serializer(mode="plain")
    def _out(self) -> dict[str, Any]:
        return {k: v.model_dump(mode="json") for k, v in self.rules.items()}

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        rule = {"anyOf": [{"enum": list(RECOURSE_RULES.names())}, {"type": "object"}]}
        return {
            "type": "object",
            "properties": dict.fromkeys(FAILURE_TYPES.names(), rule),
            "additionalProperties": False,
        }


# --------------------------------------------------------------------------- objective


for _name, _description in (
    ("distance", "Total travel distance"),
    ("time", "Total travel time"),
    ("vehicles", "Number of vehicles used"),
    ("lateness", "Total lateness under soft time windows"),
    ("waiting", "Total waiting time"),
    ("unserved", "Requests left unserved"),
    ("rejected", "Requests rejected"),
    ("outsourcing", "Outsourcing cost"),
):
    COST_COMPONENTS.register(_name, PluginEntry(_NoParams, description=_description))
for _name, _description in (
    ("energy", "Energy cost (EV module)"),
    ("charging_time", "Time spent charging (EV module)"),
    ("emissions", "Emissions"),
    ("overtime", "Work beyond the shift"),
    ("imbalance", "Workload imbalance"),
):
    COST_COMPONENTS.register(_name, PluginEntry(_NoParams, reserved=True, description=_description))


LEXICOGRAPHIC_BASE = 1e6
"""Weight ratio between consecutive levels of a lexicographic objective (ADR 0009)."""


class ObjectiveSpec(FrozenModel):
    """The objective: a weighted sum of cost components, or a lexicographic order.

    YAML forms::

        objective: {distance: 1.0, rejected: 50.0}
        objective: {lexicographic: [vehicles, distance]}

    Weights must be non-negative and finite. Exactly one form is used.
    """

    weights: FrozenMap[str, Annotated[float, Field(ge=0.0)]] = Field(default_factory=dict)
    lexicographic: tuple[str, ...] | None = None

    @model_validator(mode="before")
    @classmethod
    def _flat_in(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "weights" not in data:
            weights = {k: v for k, v in data.items() if k != "lexicographic"}
            out: dict[str, Any] = {"weights": weights}
            if "lexicographic" in data:
                out["lexicographic"] = data["lexicographic"]
            return out
        return data

    @model_validator(mode="after")
    def _consistent(self) -> ObjectiveSpec:
        if self.lexicographic is not None and self.weights:
            raise ValueError("use either weights or `lexicographic`, not both")
        if self.lexicographic is None and not self.weights:
            raise ValueError("the objective needs at least one weighted cost component")
        components = list(self.weights) + list(self.lexicographic or ())
        lex = self.lexicographic
        if lex is not None and len(set(lex)) != len(lex):
            raise ValueError("`lexicographic` components must be unique")
        unknown = [c for c in components if c not in COST_COMPONENTS]
        if unknown:
            raise ValueError(
                f"unknown cost component(s) {unknown}; known: {', '.join(COST_COMPONENTS.names())}"
            )
        if any(not math.isfinite(w) for w in self.weights.values()):
            raise ValueError("weights must be finite")
        return self

    @model_serializer(mode="plain")
    def _flat_out(self) -> dict[str, Any]:
        if self.lexicographic is not None:
            return {"lexicographic": list(self.lexicographic)}
        return dict(self.weights)

    def scalar_weights(self) -> dict[str, float]:
        """Weights of the scalar objective; lexicographic levels use powers of the base."""
        if self.lexicographic is not None:
            levels = self.lexicographic
            return {c: LEXICOGRAPHIC_BASE ** (len(levels) - 1 - k) for k, c in enumerate(levels)}
        return dict(self.weights)

    def components(self) -> tuple[str, ...]:
        """Cost components used by the objective, in config order."""
        return tuple(self.lexicographic) if self.lexicographic is not None else tuple(self.weights)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        names = list(COST_COMPONENTS.names())
        return {
            "type": "object",
            "properties": {
                **{n: {"type": "number", "minimum": 0} for n in names},
                "lexicographic": {"type": "array", "items": {"enum": names}, "uniqueItems": True},
            },
            "additionalProperties": False,
        }


# --------------------------------------------------------------------------- agents


class AgentsSpec(FrozenModel):
    """Single- or multi-agent setup.

    Attributes
    ----------
    reward
        ``global``: every agent receives the total reward. ``individual``: each agent receives its
        own costs. ``mixed``: ``reward_mix * global + (1 - reward_mix) * individual``.
    conflict
        How to resolve two agents choosing the same request.
    """

    mode: Literal["single", "multi"] = "single"
    api: Literal["aec", "parallel"] = "aec"
    reward: Literal["global", "individual", "mixed"] = "global"
    reward_mix: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    conflict: Literal["priority", "distance", "lottery", "reject"] = "priority"

    @model_validator(mode="after")
    def _mix(self) -> AgentsSpec:
        if (self.reward == "mixed") != (self.reward_mix is not None):
            raise ValueError("`reward_mix` is required with mixed rewards and only then")
        return self
