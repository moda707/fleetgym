"""The `ProblemSpec`: one immutable description of a routing problem."""

from __future__ import annotations

import re
from typing import Any

import yaml
from pydantic import Field, PrivateAttr, field_validator, model_validator

from fleetgym._hashing import content_hash, short_hash
from fleetgym.spec.models.base import FrozenJSON, FrozenMap, FrozenModel
from fleetgym.spec.models.distributions import UncertainSpec
from fleetgym.spec.models.sections import (
    AgentsSpec,
    ConstraintsSpec,
    DecisionsSpec,
    DynamicsSpec,
    FleetSpec,
    InstanceSourceSpec,
    Meta,
    NetworkSpec,
    ObjectiveSpec,
    RecourseSpec,
)
from fleetgym.spec.registries import (
    CONSTRAINTS,
    COST_COMPONENTS,
    FAILURE_TYPES,
    INTERFACES,
    RECOURSE_RULES,
    UNCERTAINTY_SOURCES,
)

__all__ = ["ENV_ID_PATTERN", "ProblemSpec"]

#: Environment IDs: a name followed by ``-v<version>``, for example ``DVRP-v1``.
ENV_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*-v\d+$")


def _default_network() -> NetworkSpec:
    return NetworkSpec.model_validate({"type": "euclidean"})


class ProblemSpec(FrozenModel):
    """Immutable specification of a routing problem.

    Usually loaded from YAML with `fleetgym.spec.load_spec`. Every field is deeply immutable.
    The content hash covers every field except ``meta``, with defaults included.

    Attributes
    ----------
    id
        Environment ID, for example ``CVRP-v1``. Optional for ad-hoc specs.
    meta
        Documentation and replication data (not hashed).
    instance_source
        Where instances come from when none is passed to the environment.
    ev
        Reserved for the EV module (not in this version).
    """

    id: str | None = None
    meta: Meta = Field(default_factory=Meta)
    instance_source: InstanceSourceSpec | None = None
    network: NetworkSpec = Field(default_factory=_default_network)
    fleet: FleetSpec
    constraints: ConstraintsSpec = Field(default_factory=ConstraintsSpec)
    uncertainty: FrozenMap[str, UncertainSpec] = Field(default_factory=dict)
    dynamics: DynamicsSpec | None = None
    decisions: DecisionsSpec = Field(default_factory=DecisionsSpec)
    recourse: RecourseSpec = Field(default_factory=RecourseSpec)
    objective: ObjectiveSpec
    agents: AgentsSpec = Field(default_factory=AgentsSpec)
    ev: FrozenJSON = None

    _source: str | None = PrivateAttr(default=None)

    @field_validator("id")
    @classmethod
    def _id_format(cls, value: str | None) -> str | None:
        if value is not None and not ENV_ID_PATTERN.match(value):
            raise ValueError(
                f"environment IDs look like `Name-v0` (letters, digits, `_.-`), got {value!r}"
            )
        return value

    @field_validator("uncertainty")
    @classmethod
    def _known_sources(cls, value: Any) -> Any:
        unknown = [k for k in value if k not in UNCERTAINTY_SOURCES]
        if unknown:
            raise ValueError(
                f"unknown uncertainty source(s) {unknown}; "
                f"known: {', '.join(UNCERTAINTY_SOURCES.names())}"
            )
        return value

    @property
    def is_ev_shortest_path(self) -> bool:
        """True for the EV shortest path with occupancy indicators (``ev.shortest_path``)."""
        from collections.abc import Mapping

        return isinstance(self.ev, Mapping) and set(self.ev) == {"shortest_path"}

    @model_validator(mode="after")
    def _ev_section(self) -> ProblemSpec:
        if self.is_ev_shortest_path:
            from fleetgym.spec.models.ev import ev_shortest_path

            try:
                ev_shortest_path(self.ev)
            except ValueError as error:
                raise ValueError(f"ev.shortest_path: {error}") from None
            if self.decisions.interface != "ev_route":
                raise ValueError("`ev.shortest_path` needs `decisions.interface: ev_route`")
        elif self.decisions.interface == "ev_route":
            raise ValueError("`decisions.interface: ev_route` needs the `ev.shortest_path` section")
        return self

    @model_validator(mode="after")
    def _cross_section_rules(self) -> ProblemSpec:
        if self.decisions.epochs == "sequential" and self.dynamics is not None:
            raise ValueError("sequential epochs are only allowed without `dynamics`")
        reserved = self.reserved_features()
        if reserved and self.meta.status != "planned":
            raise ValueError(
                "uses features not implemented in this version: "
                f"{', '.join(reserved)}; set `meta.status: planned`"
            )
        return self

    # ----------------------------------------------------------------------- queries

    def reserved_features(self) -> tuple[str, ...]:
        """Dotted paths of every reserved (not yet implemented) feature this spec uses."""
        found: list[str] = []
        if self.ev is not None and not self.is_ev_shortest_path:
            found.append("ev")
        if self.network.reserved:
            found.append("network")
        if getattr(self.network.params, "time_model", None) is not None:
            found.append("network.time_model")
        found += [
            f"constraints.{n}" for n in self.constraints.names() if CONSTRAINTS.get(n).reserved
        ]
        found += [
            f"uncertainty.{n}" for n in self.uncertainty if UNCERTAINTY_SOURCES.get(n).reserved
        ]
        found += [
            f"uncertainty.{n}.correlated_with"
            for n, u in self.uncertainty.items()
            if u.correlated_with
        ]
        found += [
            f"recourse.{n}"
            for n, rule in self.recourse.rules.items()
            if FAILURE_TYPES.get(n).reserved or RECOURSE_RULES.get(rule.name).reserved
        ]
        if INTERFACES.get(self.decisions.interface).reserved:
            found.append("decisions.interface")
        if self.decisions.epochs == "hybrid":
            found.append("decisions.epochs")
        if self.decisions.commitment == "diversion" and not self.is_ev_shortest_path:
            found.append("decisions.commitment")
        found += [
            f"objective.{c}" for c in self.objective.components() if COST_COMPONENTS.get(c).reserved
        ]
        return tuple(found)

    def __eq__(self, other: object) -> bool:
        """Compare fields only; the config file a spec came from does not matter."""
        if not isinstance(other, ProblemSpec):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash(self.content_hash())

    @property
    def source(self) -> str | None:
        """The config file this spec was loaded from, if any (not part of the hash)."""
        return self._source

    def hashable_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible data covered by the content hash (all but ``meta``)."""
        return self.model_dump(mode="json", exclude={"meta"})

    def content_hash(self) -> str:
        """Versioned SHA-256 hash of the spec without ``meta``, defaults included."""
        return content_hash(self.hashable_dict())

    def short_hash(self) -> str:
        """First 12 hex digits of `content_hash`, for display."""
        return short_hash(self.content_hash())

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-compatible data, defaults included, in YAML form."""
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        """Fully resolved YAML, defaults included. Loading it gives a spec with the same hash."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def with_source(self, source: str | None) -> ProblemSpec:
        """Return this spec annotated with the config file it came from."""
        self._source = source
        return self
