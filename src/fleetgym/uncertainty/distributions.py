"""Samplers for the distributions declared in specs (design §6).

Each class implements the parameterization stated in its spec model
(`fleetgym.spec.models.distributions`). Parameters may be scalars or per-element arrays (for example
one Poisson mean per node, from ``mean: from_instance``); they broadcast against ``size``.

Every distribution offers:

* ``sample(rng, size)``: ``size`` independent draws as a float array (element ``i`` uses
  element ``i`` of array parameters);
* ``mean(size)`` and ``quantile(q, size)``: per element;
* ``support(size)``: per element, the finite support as ``(values, probabilities)``, or ``None``
  when the support is infinite or continuous. Exact-expectation tests enumerate scenarios from it.

Custom distributions register a spec parameter model and a class with `register_distribution`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from fleetgym.errors import ConfigError
from fleetgym.plugins import Registry
from fleetgym.spec.models import FrozenModel
from fleetgym.spec.registries import DISTRIBUTIONS, PluginEntry

__all__ = [
    "DISTRIBUTION_CLASSES",
    "Distribution",
    "Support",
    "build_distribution",
    "register_distribution",
]

FloatArray = NDArray[np.float64]
Support = tuple[FloatArray, FloatArray]
"""Finite support of one element: values and their probabilities (summing to 1)."""


def _param(value: ArrayLike, size: int, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(size, float(arr))
    if arr.shape != (size,):
        raise ValueError(
            f"parameter `{name}` has shape {arr.shape}; expected a scalar or ({size},)"
        )
    return arr


class Distribution(ABC):
    """Base class of samplers. Subclasses take the spec model's fields as keyword arguments."""

    name: ClassVar[str]
    #: Names of parameters that may be per-element arrays.
    vector_params: ClassVar[tuple[str, ...]] = ()

    def __init__(self, **params: Any) -> None:
        self.params = params

    def p(self, name: str, size: int) -> FloatArray:
        """Parameter ``name`` broadcast to ``size`` elements."""
        return _param(self.params[name], size, name)

    @abstractmethod
    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``size`` independent draws."""

    @abstractmethod
    def mean(self, size: int) -> FloatArray:
        """Mean of each element."""

    def quantile(self, q: float, size: int) -> FloatArray:
        """``q``-quantile of each element (the smallest ``x`` with ``P(X <= x) >= q``)."""
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be in [0, 1], got {q}")
        return self._quantile(q, size)

    @abstractmethod
    def _quantile(self, q: float, size: int) -> FloatArray: ...

    def support(self, size: int) -> list[Support] | None:
        """Finite support per element, or ``None`` if not finite."""
        return None


def _discrete_quantile(values: FloatArray, probs: FloatArray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(probs[order])
    index = int(np.searchsorted(cdf, q - 1e-12, side="left"))
    return float(values[order][min(index, len(values) - 1)])


DISTRIBUTION_CLASSES: Registry[type[Distribution]] = Registry("distribution implementation")
"""Sampler classes keyed by the spec's ``dist`` name."""


@DISTRIBUTION_CLASSES.register("fixed")
class Fixed(Distribution):
    """Always ``value`` (the deterministic case)."""

    name = "fixed"
    vector_params = ("value",)

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """The value, without consuming randomness."""
        return self.p("value", size).copy()

    def mean(self, size: int) -> FloatArray:
        """The value."""
        return self.p("value", size)

    def _quantile(self, q: float, size: int) -> FloatArray:
        return self.p("value", size)

    def support(self, size: int) -> list[Support]:
        """One point per element."""
        return [(np.array([v]), np.array([1.0])) for v in self.p("value", size)]


@DISTRIBUTION_CLASSES.register("uniform")
class Uniform(Distribution):
    """Continuous uniform on ``[low, high)``."""

    name = "uniform"
    vector_params = ("low", "high")

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``rng.uniform(low, high)``."""
        return rng.uniform(self.p("low", size), self.p("high", size))

    def mean(self, size: int) -> FloatArray:
        """``(low + high) / 2``."""
        return (self.p("low", size) + self.p("high", size)) / 2

    def _quantile(self, q: float, size: int) -> FloatArray:
        low, high = self.p("low", size), self.p("high", size)
        return low + q * (high - low)


@DISTRIBUTION_CLASSES.register("normal")
class Normal(Distribution):
    """Normal with mean ``mean`` and standard deviation ``std``."""

    name = "normal"
    vector_params = ("mean", "std")

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``rng.normal(mean, std)``."""
        return rng.normal(self.p("mean", size), self.p("std", size))

    def mean(self, size: int) -> FloatArray:
        """``mean``."""
        return self.p("mean", size)

    def _quantile(self, q: float, size: int) -> FloatArray:
        return stats.norm.ppf(q, loc=self.p("mean", size), scale=self.p("std", size))


@DISTRIBUTION_CLASSES.register("truncated_normal")
class TruncatedNormal(Distribution):
    """Normal(``mean``, ``std``) conditioned on ``low <= X <= high`` (untruncated parameters)."""

    name = "truncated_normal"
    vector_params = ("mean", "std")

    def _frozen(self, size: int) -> Any:
        mean, std = self.p("mean", size), self.p("std", size)
        low = self.params.get("low")
        high = self.params.get("high")
        a = (low - mean) / std if low is not None else np.full(size, -np.inf)
        b = (high - mean) / std if high is not None else np.full(size, np.inf)
        return stats.truncnorm(a, b, loc=mean, scale=std)

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``scipy.stats.truncnorm`` draws with ``rng``."""
        return np.asarray(self._frozen(size).rvs(size=size, random_state=rng), dtype=float)

    def mean(self, size: int) -> FloatArray:
        """Mean of the truncated distribution."""
        return np.asarray(self._frozen(size).mean(), dtype=float)

    def _quantile(self, q: float, size: int) -> FloatArray:
        return np.asarray(self._frozen(size).ppf(q), dtype=float)


@DISTRIBUTION_CLASSES.register("lognormal")
class LogNormal(Distribution):
    """``exp(mu + sigma * Z)`` with ``Z`` standard normal."""

    name = "lognormal"

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``rng.lognormal(mu, sigma)``."""
        return rng.lognormal(self.p("mu", size), self.p("sigma", size))

    def mean(self, size: int) -> FloatArray:
        """``exp(mu + sigma^2 / 2)``."""
        return np.exp(self.p("mu", size) + self.p("sigma", size) ** 2 / 2)

    def _quantile(self, q: float, size: int) -> FloatArray:
        return stats.lognorm.ppf(q, s=self.p("sigma", size), scale=np.exp(self.p("mu", size)))


@DISTRIBUTION_CLASSES.register("gamma")
class Gamma(Distribution):
    """Gamma with shape ``shape`` and mean ``mean`` (scale = mean / shape)."""

    name = "gamma"
    vector_params = ("mean",)

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``rng.gamma(shape, mean / shape)``."""
        shape = self.p("shape", size)
        return rng.gamma(shape, self.p("mean", size) / shape)

    def mean(self, size: int) -> FloatArray:
        """``mean``."""
        return self.p("mean", size)

    def _quantile(self, q: float, size: int) -> FloatArray:
        shape = self.p("shape", size)
        return stats.gamma.ppf(q, a=shape, scale=self.p("mean", size) / shape)


@DISTRIBUTION_CLASSES.register("weibull")
class Weibull(Distribution):
    """Weibull: ``P(X > x) = exp(-(x / scale)^shape)``."""

    name = "weibull"
    vector_params = ("scale",)

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``scale * rng.weibull(shape)``."""
        return self.p("scale", size) * rng.weibull(self.p("shape", size))

    def mean(self, size: int) -> FloatArray:
        """``scale * Gamma(1 + 1 / shape)``."""
        shape = self.p("shape", size)
        return self.p("scale", size) * np.array([math.gamma(1 + 1 / k) for k in shape])

    def _quantile(self, q: float, size: int) -> FloatArray:
        return stats.weibull_min.ppf(q, c=self.p("shape", size), scale=self.p("scale", size))


@DISTRIBUTION_CLASSES.register("exponential")
class Exponential(Distribution):
    """Exponential with mean ``mean``."""

    name = "exponential"
    vector_params = ("mean",)

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``rng.exponential(mean)``."""
        return rng.exponential(self.p("mean", size))

    def mean(self, size: int) -> FloatArray:
        """``mean``."""
        return self.p("mean", size)

    def _quantile(self, q: float, size: int) -> FloatArray:
        return stats.expon.ppf(q, scale=self.p("mean", size))


@DISTRIBUTION_CLASSES.register("poisson")
class Poisson(Distribution):
    """Poisson with mean ``mean``."""

    name = "poisson"
    vector_params = ("mean",)

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``rng.poisson(mean)`` as floats."""
        return rng.poisson(self.p("mean", size)).astype(float)

    def mean(self, size: int) -> FloatArray:
        """``mean``."""
        return self.p("mean", size)

    def _quantile(self, q: float, size: int) -> FloatArray:
        return np.asarray(stats.poisson.ppf(q, self.p("mean", size)), dtype=float)

    def support(self, size: int) -> list[Support] | None:
        """Finite only when every mean is 0."""
        if np.all(self.p("mean", size) == 0):
            return [(np.array([0.0]), np.array([1.0]))] * size
        return None


@DISTRIBUTION_CLASSES.register("bernoulli")
class Bernoulli(Distribution):
    """1 with probability ``p``, else 0."""

    name = "bernoulli"

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``rng.random() < p`` as floats."""
        return (rng.random(size) < self.p("p", size)).astype(float)

    def mean(self, size: int) -> FloatArray:
        """``p``."""
        return self.p("p", size)

    def _quantile(self, q: float, size: int) -> FloatArray:
        return (q > 1 - self.p("p", size)).astype(float)

    def support(self, size: int) -> list[Support]:
        """``{0: 1 - p, 1: p}``."""
        return [(np.array([0.0, 1.0]), np.array([1 - p, p])) for p in self.p("p", size)]


class _Discrete(Distribution):
    """Shared code of distributions on a finite list of values."""

    def _values_probs(self) -> tuple[FloatArray, FloatArray]:
        raise NotImplementedError

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``rng.choice(values, p=probs)``."""
        values, probs = self._values_probs()
        return rng.choice(values, size=size, p=probs).astype(float)

    def mean(self, size: int) -> FloatArray:
        """Probability-weighted mean of the values."""
        values, probs = self._values_probs()
        return np.full(size, float(np.dot(values, probs)))

    def _quantile(self, q: float, size: int) -> FloatArray:
        values, probs = self._values_probs()
        return np.full(size, _discrete_quantile(values, probs, q))

    def support(self, size: int) -> list[Support]:
        """The same values and probabilities for every element."""
        values, probs = self._values_probs()
        return [(values.copy(), probs.copy()) for _ in range(size)]


@DISTRIBUTION_CLASSES.register("categorical")
class Categorical(_Discrete):
    """``values`` with probabilities ``probs``."""

    name = "categorical"

    def _values_probs(self) -> tuple[FloatArray, FloatArray]:
        probs = np.asarray(self.params["probs"], dtype=float)
        return np.asarray(self.params["values"], dtype=float), probs / probs.sum()


@DISTRIBUTION_CLASSES.register("discrete_uniform")
class DiscreteUniform(Distribution):
    """Integers ``center - w, ..., center + w``, equally likely.

    ``w = min(half_width, center - floor)`` per element, and 0 where ``center <= floor`` (such as
    a depot's zero demand); the mean is ``center``. ``center`` may differ per element and must be
    integer.
    """

    name = "discrete_uniform"
    vector_params = ("center",)

    def _bounds(self, size: int) -> tuple[FloatArray, FloatArray]:
        center = self.p("center", size)
        if not np.all(center == np.round(center)):
            raise ValueError("discrete_uniform: `center` must be integer")
        width = np.full(size, float(self.params["half_width"]))
        floor = self.params.get("floor")
        if floor is not None:  # a center at or below the floor (a depot's 0) is the value itself
            width = np.clip(center - float(floor), 0.0, width)
        return center - width, center + width

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``rng.integers(low, high + 1)`` per element, as floats."""
        low, high = self._bounds(size)
        return rng.integers(low.astype(np.int64), high.astype(np.int64) + 1).astype(float)

    def mean(self, size: int) -> FloatArray:
        """``center``."""
        low, high = self._bounds(size)
        return (low + high) / 2

    def _quantile(self, q: float, size: int) -> FloatArray:
        return np.array([_discrete_quantile(v, p, q) for v, p in self.support(size)])

    def support(self, size: int) -> list[Support]:
        """Every integer between the bounds, with equal probabilities."""
        out = []
        for lo, hi in zip(*self._bounds(size), strict=True):
            values = np.arange(lo, hi + 1.0)
            out.append((values, np.full(values.size, 1.0 / values.size)))
        return out


@DISTRIBUTION_CLASSES.register("scaled_categorical")
class ScaledCategorical(Distribution):
    """``scale * factors[i]`` with probability ``probs[i]``; ``scale`` may differ per element."""

    name = "scaled_categorical"
    vector_params = ("scale",)

    def _factors(self) -> tuple[FloatArray, FloatArray]:
        probs = np.asarray(self.params["probs"], dtype=float)
        return np.asarray(self.params["factors"], dtype=float), probs / probs.sum()

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``scale * rng.choice(factors, p=probs)``."""
        factors, probs = self._factors()
        return self.p("scale", size) * rng.choice(factors, size=size, p=probs)

    def mean(self, size: int) -> FloatArray:
        """``scale * sum(factors * probs)``."""
        factors, probs = self._factors()
        return self.p("scale", size) * float(np.dot(factors, probs))

    def _quantile(self, q: float, size: int) -> FloatArray:
        return np.array([_discrete_quantile(v, p, q) for v, p in self.support(size)])

    def support(self, size: int) -> list[Support]:
        """The scaled factors per element (equal values are kept separate)."""
        factors, probs = self._factors()
        return [(scale * factors, probs.copy()) for scale in self.p("scale", size)]


@DISTRIBUTION_CLASSES.register("empirical")
class Empirical(_Discrete):
    """Resampling of observed ``values``, uniformly or with ``weights``."""

    name = "empirical"

    def _values_probs(self) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(self.params["values"], dtype=float).ravel()
        weights = self.params.get("weights")
        w = np.ones(values.size) if weights is None else np.asarray(weights, dtype=float)
        return values, w / w.sum()


@DISTRIBUTION_CLASSES.register("kde")
class KDE(Distribution):
    """Gaussian kernel density estimate (``scipy.stats.gaussian_kde``) of one-dimensional ``data``.

    Multi-dimensional data (for example locations) is used by dynamic request classes (M6).
    """

    name = "kde"

    def _kde(self) -> Any:
        data = np.asarray(self.params["data"], dtype=float)
        if data.ndim != 1:
            raise ValueError("a scalar uncertain parameter needs one-dimensional `data`")
        return stats.gaussian_kde(data, bw_method=self.params.get("bandwidth", "scott"))

    def sample(self, rng: np.random.Generator, size: int) -> FloatArray:
        """``gaussian_kde.resample`` with ``rng``."""
        return np.asarray(self._kde().resample(size, seed=rng)[0], dtype=float)

    def mean(self, size: int) -> FloatArray:
        """Mean of the data (the kernel is symmetric)."""
        return np.full(size, float(np.mean(np.asarray(self.params["data"], dtype=float))))

    def _quantile(self, q: float, size: int) -> FloatArray:
        raise NotImplementedError("the KDE sampler has no quantile function")


def build_distribution(name: str, params: Mapping[str, Any]) -> Distribution:
    """Instantiate the sampler ``name`` with resolved parameters.

    Raises
    ------
    KeyError
        If no sampler is registered under ``name``.
    """
    return DISTRIBUTION_CLASSES.get(name)(**dict(params))


def register_distribution(
    name: str, *, params: type[FrozenModel], description: str = ""
) -> Callable[[type[Distribution]], type[Distribution]]:
    """Register a custom distribution: its spec parameter model and its sampler class.

    Examples
    --------
    >>> class TriangularParams(FrozenModel):
    ...     low: float
    ...     mode: float
    ...     high: float
    >>> @register_distribution("my_triangular", params=TriangularParams)  # doctest: +SKIP
    ... class Triangular(Distribution): ...
    """

    def decorator(cls: type[Distribution]) -> type[Distribution]:
        if name in DISTRIBUTIONS:
            raise ConfigError(f"distribution {name!r} is already registered")
        DISTRIBUTIONS.register(name, PluginEntry(params, description=description))
        DISTRIBUTION_CLASSES.register(name, cls)
        cls.name = name
        return cls

    return decorator
