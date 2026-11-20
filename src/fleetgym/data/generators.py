"""Uniform and clustered instance generators.

Generators are registered by name; a config refers to them through
``instance_source: {type: generator, name: uniform, params: {...}}``. Every generator takes a
``numpy.random.Generator`` and returns an `Instance`; the same generator state gives the same
instance on every platform.

Layout conventions (all generators):

* coordinates lie in the square ``[0, size]^2``; node 0 is the depot;
* demands are integers drawn uniformly from ``[demand.low, demand.high]``; the depot's is 0;
* with ``time_windows``, travel time equals Euclidean distance (speed 1), the depot's window is
  ``[0, horizon]``, and each customer's window is placed so that a vehicle leaving the depot at time
  0 can reach it, serve it and return by ``horizon``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from fleetgym.data.instance import Instance
from fleetgym.errors import InstanceError
from fleetgym.plugins import Registry

__all__ = [
    "GENERATORS",
    "ClusteredParams",
    "DemandParams",
    "GeneratorEntry",
    "TimeWindowParams",
    "UniformParams",
    "generate",
    "generate_clustered",
    "generate_uniform",
    "validate_generator_params",
]


class _Params(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DemandParams(_Params):
    """Integer demands drawn uniformly from ``[low, high]``."""

    low: int = Field(default=1, ge=0)
    high: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> DemandParams:
        if self.low > self.high:
            raise ValueError(f"demand.low ({self.low}) exceeds demand.high ({self.high})")
        return self


class TimeWindowParams(_Params):
    """Time windows of width drawn uniformly from ``[width_low, width_high]``."""

    horizon: float = Field(gt=0)
    width_low: float = Field(gt=0)
    width_high: float = Field(gt=0)
    service_time: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> TimeWindowParams:
        if self.width_low > self.width_high:
            raise ValueError("width_low exceeds width_high")
        return self


class UniformParams(_Params):
    """Customers uniform in the square ``[0, size]^2``.

    Attributes
    ----------
    depot
        ``center``, ``corner`` (the origin) or ``random``.
    capacity
        Vehicle capacity stored in the instance (``None`` leaves it to the spec).
    n_vehicles
        Vehicle count stored in the instance (``None`` leaves it to the spec).
    """

    n_customers: int = Field(ge=1)
    size: float = Field(default=100.0, gt=0)
    depot: Literal["center", "corner", "random"] = "center"
    demand: DemandParams = Field(default_factory=DemandParams)
    capacity: float | None = Field(default=None, gt=0)
    n_vehicles: int | None = Field(default=None, ge=1)
    time_windows: TimeWindowParams | None = None


class ClusteredParams(UniformParams):
    """Customers in Gaussian clusters around centers drawn uniformly in the square.

    Attributes
    ----------
    n_clusters
        Number of cluster centers.
    spread
        Standard deviation of customers around their center; points are clipped to the square.
    """

    n_clusters: int = Field(ge=1)
    spread: float = Field(gt=0)


@dataclass(frozen=True)
class GeneratorEntry:
    """A registered generator and its parameter model."""

    function: Callable[[np.random.Generator, Any], Instance]
    params: type[_Params]


GENERATORS: Registry[GeneratorEntry] = Registry("instance generator")


def _depot(rng: np.random.Generator, params: UniformParams) -> NDArray[np.float64]:
    if params.depot == "center":
        return np.array([params.size / 2, params.size / 2])
    if params.depot == "corner":
        return np.zeros(2)
    return rng.uniform(0.0, params.size, size=2)


def _assemble(
    rng: np.random.Generator,
    params: UniformParams,
    depot: NDArray[np.float64],
    customers: NDArray[np.float64],
    name: str,
    source: str,
) -> Instance:
    coords = np.vstack([depot, customers])
    n = coords.shape[0]
    demands = np.zeros(n)
    demands[1:] = rng.integers(params.demand.low, params.demand.high + 1, size=n - 1)
    time_windows = service = None
    if params.time_windows is not None:
        time_windows, service = _time_windows(rng, coords, params.time_windows, name)
    return Instance(
        name=name,
        source=source,
        demands=demands,
        coords=coords,
        capacity=params.capacity,
        n_vehicles=params.n_vehicles,
        time_windows=time_windows,
        service_times=service,
    )


def _time_windows(
    rng: np.random.Generator, coords: NDArray[np.float64], tw: TimeWindowParams, name: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = coords.shape[0]
    from_depot = np.hypot(*(coords[1:] - coords[0]).T)
    earliest = from_depot
    latest = tw.horizon - tw.service_time - from_depot
    if np.any(earliest > latest):
        worst = int(np.argmax(earliest - latest)) + 1
        raise InstanceError(
            f"{name}: horizon {tw.horizon} is too short to reach, serve and leave customer {worst}"
        )
    width = rng.uniform(tw.width_low, tw.width_high, size=n - 1)
    start = earliest + rng.uniform(0.0, 1.0, size=n - 1) * np.maximum(
        latest - width - earliest, 0.0
    )
    end = np.minimum(start + width, latest)
    windows = np.vstack([[0.0, tw.horizon], np.column_stack([start, end])])
    service = np.concatenate([[0.0], np.full(n - 1, tw.service_time)])
    return windows, service


def generate_uniform(rng: np.random.Generator, params: UniformParams) -> Instance:
    """Generate an instance with customers uniform in the square."""
    depot = _depot(rng, params)
    customers = rng.uniform(0.0, params.size, size=(params.n_customers, 2))
    name = f"uniform-n{params.n_customers}"
    return _assemble(rng, params, depot, customers, name, "generated:uniform")


def generate_clustered(rng: np.random.Generator, params: ClusteredParams) -> Instance:
    """Generate an instance with customers in Gaussian clusters."""
    depot = _depot(rng, params)
    centers = rng.uniform(0.0, params.size, size=(params.n_clusters, 2))
    assignment = rng.integers(0, params.n_clusters, size=params.n_customers)
    offsets = rng.normal(0.0, params.spread, size=(params.n_customers, 2))
    customers = np.clip(centers[assignment] + offsets, 0.0, params.size)
    name = f"clustered-n{params.n_customers}-c{params.n_clusters}"
    return _assemble(rng, params, depot, customers, name, "generated:clustered")


GENERATORS.register("uniform", GeneratorEntry(generate_uniform, UniformParams))  # pyright: ignore[reportArgumentType]
GENERATORS.register("clustered", GeneratorEntry(generate_clustered, ClusteredParams))  # pyright: ignore[reportArgumentType]


def validate_generator_params(name: str, params: Mapping[str, Any] | None) -> _Params:
    """Validate ``params`` for the generator ``name``.

    Raises
    ------
    fleetgym.plugins.PluginError
        If no generator has that name.
    pydantic.ValidationError
        If the parameters are invalid.
    """
    return GENERATORS.get(name).params.model_validate(dict(params or {}))


def generate(name: str, params: Mapping[str, Any] | None, rng: np.random.Generator) -> Instance:
    """Generate an instance with the registered generator ``name``.

    Raises
    ------
    InstanceError
        If the parameters are invalid or the generator cannot satisfy them.
    """
    entry = GENERATORS.get(name)
    try:
        validated = entry.params.model_validate(dict(params or {}))
    except ValidationError as exc:
        raise InstanceError(f"invalid parameters for generator {name!r}: {exc}") from exc
    return entry.function(rng, validated)
