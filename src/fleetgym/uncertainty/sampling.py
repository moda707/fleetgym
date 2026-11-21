"""Sampling scenarios from a bound problem (architecture rules 5 and 6).

Each source draws from its own stream, ``StreamFactory(seed).generator("uncertainty.<source>")``,
keyed by the source name, so adding or changing one source never changes another source's draws.
A source draws one value per node; depot values are then set to their deterministic values
(demand 0, the instance's service time), and customer values are checked for validity. The
``travel_time`` source draws one value per arc from the network's distances (``distances``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from fleetgym.errors import ConfigError
from fleetgym.spec.binding import BoundProblem
from fleetgym.spec.models import DistSpec
from fleetgym.uncertainty.distributions import Distribution, build_distribution, gamma_arc_times
from fleetgym.uncertainty.scenario import Scenario
from fleetgym.uncertainty.streams import StreamFactory

__all__ = [
    "deterministic_values",
    "distribution_at",
    "distributions",
    "realize",
    "resample_unrevealed",
    "sample_scenario",
]


def distribution_at(problem: BoundProblem, path: str, dist: DistSpec) -> Distribution:
    """The sampler of the distribution at spec path ``path``, with instance references resolved.

    Parameters that refer to the instance are resolved by binding (``problem.refs``); the others
    are the distribution's own values, from the spec or from the instance's ``distributions``.
    """
    fields = dist.params.model_dump()
    params = {key: problem.refs.get(f"{path}.{key}", value) for key, value in fields.items()}
    params = {k: (np.asarray(v, dtype=float) if _numeric(v) else v) for k, v in params.items()}
    return build_distribution(dist.name, params)


def distributions(problem: BoundProblem) -> dict[str, Distribution]:
    """The sampler of every uncertainty source, with instance references resolved."""
    return {
        name: distribution_at(problem, f"uncertainty.{name}", problem.distributions[name])
        for name in problem.spec.uncertainty
    }


def _numeric(value: object) -> bool:
    if isinstance(value, (int, float, np.ndarray)) and not isinstance(value, bool):
        return True
    return isinstance(value, (list, tuple)) and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in value
    )


def deterministic_values(problem: BoundProblem, source: str) -> NDArray[np.float64]:
    """The instance's value of ``source`` per node (used at depots and when a source is fixed)."""
    instance = problem.instance
    n = instance.n_nodes
    if source == "demand":
        return np.asarray(instance.demands, dtype=float)
    if source == "service_time":
        return (
            np.asarray(instance.service_times, dtype=float)
            if instance.service_times is not None
            else np.zeros(n)
        )
    raise KeyError(source)


def realize(problem: BoundProblem, source: str, draws: NDArray[np.float64]) -> NDArray[np.float64]:
    """Turn raw draws (one per node) into realized values.

    Depot values are replaced by their deterministic values.

    Raises
    ------
    ConfigError
        If a customer value is negative or not finite (the distribution's support is wrong for
        this source).
    """
    values = np.array(draws, dtype=float)
    depots = problem.instance.n_depots
    values[:depots] = deterministic_values(problem, source)[:depots]
    bad = ~np.isfinite(values) | (values < 0)
    if np.any(bad):
        node = int(np.flatnonzero(bad)[0])
        raise ConfigError(
            f"{source} drew {values[node]} at node {node}; {source} values must be finite and "
            "non-negative, so use a distribution with that support (for example truncated_normal "
            "with low: 0)",
            source=problem.spec.source,
            key_path=f"uncertainty.{source}",
        )
    return values


def _draw(
    problem: BoundProblem,
    dists: Mapping[str, Distribution],
    streams: StreamFactory,
    distances: NDArray[np.float64] | None,
) -> dict[str, NDArray[np.float64]]:
    n = problem.instance.n_nodes
    out = {}
    for name, dist in dists.items():
        rng = streams.generator(f"uncertainty.{name}")
        if name == "travel_time":
            if dist.name != "gamma_per_distance":
                raise ConfigError(
                    f"travel_time needs the gamma_per_distance distribution, not {dist.name!r}",
                    source=problem.spec.source,
                    key_path="uncertainty.travel_time",
                )
            if distances is None:
                raise ValueError("travel_time needs the network's distances")
            out[name] = gamma_arc_times(dist.params, distances, rng).ravel()
        else:
            out[name] = realize(problem, name, dist.sample(rng, n))
    return out


def sample_scenario(
    problem: BoundProblem, seed: int, *, distances: NDArray[np.float64] | None = None
) -> Scenario:
    """Sample the scenario of an episode from its seed.

    ``distances`` (the network's distance matrix) is needed when ``travel_time`` is uncertain.
    """
    values = _draw(problem, distributions(problem), StreamFactory(seed), distances)
    return Scenario(problem.spec_hash, problem.instance_hash, values, seed=seed)


def resample_unrevealed(
    problem: BoundProblem,
    scenario: Scenario,
    revealed: Mapping[str, NDArray[np.bool_]],
    seed: int,
    *,
    distances: NDArray[np.float64] | None = None,
) -> Scenario:
    """A scenario that keeps revealed values and redraws the others from ``seed``.

    This is what a non-anticipative planner may assume about the future (`FleetGymEnv.fork`).
    """
    fresh = _draw(problem, distributions(problem), StreamFactory(seed), distances)
    values = {
        name: np.where(revealed[name], scenario[name], fresh[name]) for name in scenario.sources
    }
    return Scenario(problem.spec_hash, problem.instance_hash, values, seed=None)
