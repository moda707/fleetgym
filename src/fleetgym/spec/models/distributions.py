"""Distribution and uncertainty specs.

A distribution is written flat in YAML, ``{dist: <name>, <params>}``. An uncertain parameter adds
when its value is revealed: ``{dist: poisson, mean: from_instance, revealed: on_arrival}``.

Each parameter model states its parameterization explicitly, because numpy and scipy conventions
differ. The samplers implementing them live in `fleetgym.uncertainty` (M5).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from enum import StrEnum
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

from fleetgym.spec.models.base import (
    FlatPluginModel,
    FrozenModel,
    InstanceRef,
    NonNegativeOrRef,
    NumberOrRef,
    PositiveOrRef,
    plugin_variants_schema,
    reraise_nested,
)
from fleetgym.spec.registries import DISTRIBUTIONS, UNCERTAINTY_SOURCES, PluginEntry

__all__ = [
    "DistSpec",
    "Revelation",
    "UncertainSpec",
]

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]

_PROB_TOL = 1e-9


class Revelation(StrEnum):
    """When the realization of an uncertain parameter becomes known to the agent."""

    NEVER = "never"
    A_PRIORI = "a_priori"
    ON_REQUEST = "on_request"
    ON_DEPARTURE = "on_departure"
    ON_ARRIVAL = "on_arrival"
    ON_COMPLETION = "on_completion"


# --------------------------------------------------------------------------- parameter models


class FixedParams(FrozenModel):
    """Degenerate distribution: always ``value``."""

    value: NumberOrRef


class UniformParams(FrozenModel):
    """Continuous uniform on ``[low, high)``."""

    low: NumberOrRef
    high: NumberOrRef

    @model_validator(mode="after")
    def _ordered(self) -> UniformParams:
        if isinstance(self.low, float) and isinstance(self.high, float) and self.low >= self.high:
            raise ValueError(f"`low` ({self.low}) must be smaller than `high` ({self.high})")
        return self


class NormalParams(FrozenModel):
    """Normal with mean ``mean`` and standard deviation ``std``."""

    mean: NumberOrRef
    std: PositiveOrRef


class TruncatedNormalParams(FrozenModel):
    """Normal(``mean``, ``std``) conditioned on ``low <= X <= high``.

    ``mean`` and ``std`` are the parameters of the untruncated normal, not the moments of the
    truncated one. At least one bound is required.
    """

    mean: NumberOrRef
    std: PositiveOrRef
    low: float | None = None
    high: float | None = None

    @model_validator(mode="after")
    def _bounds(self) -> TruncatedNormalParams:
        if self.low is None and self.high is None:
            raise ValueError("at least one of `low` and `high` is required")
        if self.low is not None and self.high is not None and self.low >= self.high:
            raise ValueError(f"`low` ({self.low}) must be smaller than `high` ({self.high})")
        return self


class LogNormalParams(FrozenModel):
    """Log-normal: ``X = exp(mu + sigma * Z)`` with ``Z`` standard normal.

    ``mu`` and ``sigma`` are the parameters of the underlying normal, not the mean and standard
    deviation of ``X``.
    """

    mu: float
    sigma: PositiveFloat


class GammaParams(FrozenModel):
    """Gamma with shape ``shape`` and mean ``mean`` (scale = mean / shape)."""

    shape: PositiveFloat
    mean: PositiveOrRef


class WeibullParams(FrozenModel):
    """Weibull with shape ``shape`` and scale ``scale``: ``P(X > x) = exp(-(x / scale)^shape)``."""

    shape: PositiveFloat
    scale: PositiveOrRef


class ExponentialParams(FrozenModel):
    """Exponential with mean ``mean`` (rate = 1 / mean)."""

    mean: PositiveOrRef


class PoissonParams(FrozenModel):
    """Poisson with mean ``mean``."""

    mean: NonNegativeOrRef


class BernoulliParams(FrozenModel):
    """Bernoulli: 1 with probability ``p``, else 0."""

    p: Probability


def _check_probabilities(probs: tuple[float, ...], n_values: int) -> None:
    if len(probs) != n_values:
        raise ValueError(f"`probs` has {len(probs)} entries but `values` has {n_values}")
    if any(p < 0 for p in probs):
        raise ValueError("`probs` must be non-negative")
    total = math.fsum(probs)
    if abs(total - 1.0) > _PROB_TOL:
        raise ValueError(f"`probs` must sum to 1 (got {total!r})")


class CategoricalParams(FrozenModel):
    """Discrete distribution on ``values`` with probabilities ``probs`` (summing to 1)."""

    values: tuple[float, ...] = Field(min_length=1)
    probs: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _probs(self) -> CategoricalParams:
        _check_probabilities(self.probs, len(self.values))
        return self


class DiscreteUniformParams(FrozenModel):
    """Discrete uniform on the integers ``center - w, ..., center + w``.

    The half width is ``w = min(half_width, center - floor)`` per element, so no value falls below
    ``floor`` (no bound when ``floor`` is omitted); a center at or below ``floor`` (such as a
    depot's zero demand) is always the value itself. Every value has probability ``1 / (2 w + 1)``
    and the mean is ``center``. ``center`` must be an integer (per element, possibly
    ``from_instance``). Example (Gendreau, Laporte and Seguin 1995, as used by Dastpak et al.):
    ``{center: from_instance, half_width: 5, floor: 1}`` gives ``5 +- 4``, ``10 +- 5``, ``15 +- 5``.
    """

    center: NumberOrRef
    half_width: int = Field(ge=0)
    floor: float | None = None


class ScaledCategoricalParams(FrozenModel):
    """``scale * factors[i]`` with probability ``probs[i]``.

    ``scale`` is a number or a per-element value (for example ``from_instance``: each customer's
    expected demand); the mean is ``scale * sum(factors * probs)``. Example (Goodson, Thomas and
    Ohlmann 2016, low variability): ``factors: [0.5, 1, 1.5]``, ``probs: [0.05, 0.9, 0.05]``.
    """

    scale: NumberOrRef
    factors: tuple[float, ...] = Field(min_length=1)
    probs: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _probs(self) -> ScaledCategoricalParams:
        _check_probabilities(self.probs, len(self.factors))
        return self


class EmpiricalParams(FrozenModel):
    """Resample observed ``values`` with replacement, uniformly or with ``weights``."""

    values: tuple[float, ...] | InstanceRef
    weights: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _weights(self) -> EmpiricalParams:
        if self.weights is not None:
            if isinstance(self.values, InstanceRef):
                raise ValueError("`weights` cannot be combined with `values: from_instance`")
            if len(self.weights) != len(self.values) or any(w < 0 for w in self.weights):
                raise ValueError("`weights` must be non-negative, one per value")
            if not any(w > 0 for w in self.weights):
                raise ValueError("at least one weight must be positive")
        return self


class KDEParams(FrozenModel):
    """Gaussian kernel density estimate fitted to ``data`` (points of any dimension).

    ``bandwidth`` is ``"scott"``, ``"silverman"`` or a positive factor, with the meaning of
    ``scipy.stats.gaussian_kde``'s ``bw_method``.
    """

    data: InstanceRef | tuple[tuple[float, ...], ...]
    bandwidth: Literal["scott", "silverman"] | PositiveFloat = "scott"


class NormalMixtureParams(FrozenModel):
    """Isotropic normal clusters.

    A center is drawn with probability ``weights`` (equal by default), then each coordinate is
    ``Normal(center, std)``.

    Centers are points of any one dimension (``[[5, 5], [5, 15]]`` for request locations, or
    ``[[3], [8]]`` for a scalar). Example (Ulmer et al., three clusters, large area):
    ``centers: [[5, 5], [5, 15], [15, 10]]``, ``std: 1``, ``weights: [1, 2, 1]``.
    """

    centers: tuple[tuple[float, ...], ...] = Field(min_length=1)
    std: PositiveFloat
    weights: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _shape(self) -> NormalMixtureParams:
        if len({len(c) for c in self.centers}) != 1 or not self.centers[0]:
            raise ValueError("all `centers` must have the same, positive dimension")
        if self.weights is not None:
            if len(self.weights) != len(self.centers) or any(w < 0 for w in self.weights):
                raise ValueError("`weights` must be non-negative, one per center")
            if not any(w > 0 for w in self.weights):
                raise ValueError("at least one weight must be positive")
        return self


_BUILTINS: tuple[tuple[str, type[FrozenModel], str], ...] = (
    ("fixed", FixedParams, "Always the same value (deterministic case)"),
    ("uniform", UniformParams, "Continuous uniform on [low, high)"),
    ("normal", NormalParams, "Normal(mean, std)"),
    ("truncated_normal", TruncatedNormalParams, "Normal(mean, std) truncated to [low, high]"),
    ("lognormal", LogNormalParams, "exp(mu + sigma Z)"),
    ("gamma", GammaParams, "Gamma with shape and mean"),
    ("weibull", WeibullParams, "Weibull with shape and scale"),
    ("exponential", ExponentialParams, "Exponential with mean"),
    ("poisson", PoissonParams, "Poisson with mean"),
    ("bernoulli", BernoulliParams, "1 with probability p, else 0"),
    ("categorical", CategoricalParams, "Discrete values with probabilities"),
    ("discrete_uniform", DiscreteUniformParams, "Integers center +- half width, equally likely"),
    ("scaled_categorical", ScaledCategoricalParams, "Multiples of a scale with probabilities"),
    ("empirical", EmpiricalParams, "Resampling of observed values"),
    ("kde", KDEParams, "Gaussian kernel density estimate"),
    ("normal_mixture", NormalMixtureParams, "Isotropic normal clusters with weights"),
)
for _name, _model, _description in _BUILTINS:
    DISTRIBUTIONS.register(_name, PluginEntry(_model, description=_description))


class _NoParams(FrozenModel):
    """Marker for registries whose entries carry no parameters."""


for _name, _description in (
    ("demand", "Customer demand"),
    ("service_time", "Service duration at a customer"),
):
    UNCERTAINTY_SOURCES.register(_name, PluginEntry(_NoParams, description=_description))
for _name, _description in (
    ("presence", "Whether a known customer requests service (stochastic customers, after v0.1)"),
    ("travel_time", "Stochastic travel times (deferred until after v0.1)"),
    ("station_wait", "Waiting time at a charging station (EV module)"),
    ("energy_consumption", "Noise on energy consumption (EV module)"),
    ("breakdown", "Vehicle breakdowns"),
):
    UNCERTAINTY_SOURCES.register(
        _name, PluginEntry(_NoParams, reserved=True, description=_description)
    )


# --------------------------------------------------------------------------- spec models


class DistSpec(FlatPluginModel):
    """A distribution written flat in YAML: ``{dist: <name>, <params>}``."""

    discriminator: ClassVar[str] = "dist"
    registry = DISTRIBUTIONS


_UNCERTAIN_KEYS = ("revealed", "correlated_with", "forecast")


class UncertainSpec(FrozenModel):
    """An uncertain parameter: a distribution plus when its realization is revealed.

    YAML forms (flat)::

        demand: {revealed: on_arrival}           # the instance gives the distribution
        demand: {dist: poisson, mean: from_instance, revealed: on_arrival}

    Attributes
    ----------
    dist
        The distribution of the parameter, or ``None``: each instance gives it in its
        ``distributions`` block (ADR 0023).
    revealed
        When the realization becomes known.
    correlated_with
        Names of latent factors (reserved; must be empty in this version).
    forecast
        What the agent sees before the realization: nothing, or the distribution's mean.
    """

    dist: DistSpec | None = None
    revealed: Revelation
    correlated_with: tuple[str, ...] = ()
    forecast: Literal["none", "mean"] = "mean"

    @model_validator(mode="wrap")
    @classmethod
    def _flatten_in(cls, data: Any, handler: Callable[[Any], Any]) -> Any:
        if isinstance(data, Mapping) and isinstance(data.get("dist"), str):
            dist_part = {k: v for k, v in data.items() if k not in _UNCERTAIN_KEYS}
            try:
                dist = DistSpec.model_validate(dist_part)
            except ValidationError as exc:
                raise reraise_nested(exc, cls.__name__) from None
            rest = {k: data[k] for k in _UNCERTAIN_KEYS if k in data}
            return handler({"dist": dist, **rest})
        return handler(data)

    @model_serializer(mode="wrap")
    def _flatten_out(self, handler: Callable[[Any], Any]) -> dict[str, Any]:
        out = dict(self.dist.model_dump(mode="json")) if self.dist is not None else {}
        out["revealed"] = str(self.revealed)
        out["correlated_with"] = list(self.correlated_with)
        out["forecast"] = self.forecast
        return out

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        extra = {
            "revealed": {"enum": [r.value for r in Revelation]},
            "correlated_with": {"type": "array", "items": {"type": "string"}},
            "forecast": {"enum": ["none", "mean"]},
        }
        schema = plugin_variants_schema(DISTRIBUTIONS, "dist", handler, extra_properties=extra)
        for variant in schema["oneOf"]:
            variant["required"] = [*variant["required"], "revealed"]
        schema["oneOf"].append(
            {
                "type": "object",
                "description": "The distribution comes from each instance",
                "properties": extra,
                "required": ["revealed"],
                "additionalProperties": False,
            }
        )
        return schema
