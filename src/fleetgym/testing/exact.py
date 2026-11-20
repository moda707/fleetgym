"""Exact expectations by enumerating every scenario of finite-support uncertainty.

`expected_cost` runs a policy on every scenario of a spec whose uncertain distributions have finite
support (``categorical``, ``bernoulli``, ``empirical``, ``fixed``) and returns the probability-
weighted total cost. Tests compare it with independent closed forms or dynamic programs, for the
built-in recourse rules and for custom ones.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator

import numpy as np

from fleetgym.data.instance import Instance
from fleetgym.envs.env import FleetGymEnv
from fleetgym.spec.binding import bind
from fleetgym.spec.models import ProblemSpec
from fleetgym.uncertainty.sampling import distributions, realize
from fleetgym.uncertainty.scenario import Scenario

__all__ = ["enumerate_scenarios", "expected_cost"]

MAX_SCENARIOS = 1_000_000


def enumerate_scenarios(
    spec: ProblemSpec, instance: Instance, *, max_scenarios: int = MAX_SCENARIOS
) -> Iterator[tuple[float, Scenario]]:
    """Every scenario with its probability (zero-probability outcomes are skipped).

    Raises
    ------
    ValueError
        If a distribution's support is not finite, or there are more than ``max_scenarios``.
    """
    problem = bind(spec, instance)
    n, depots = instance.n_nodes, instance.n_depots
    axes: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for name, dist in distributions(problem).items():
        support = dist.support(n)
        if support is None:
            raise ValueError(f"uncertainty.{name} has no finite support; it cannot be enumerated")
        for node in range(depots, n):
            values, probs = support[node]
            keep = probs > 0
            axes.append((name, node, values[keep], probs[keep]))
    count = math.prod(len(a[2]) for a in axes)
    if count > max_scenarios:
        raise ValueError(f"{count} scenarios exceed max_scenarios={max_scenarios}")
    base = {name: np.zeros(n) for name in spec.uncertainty}
    for combo in itertools.product(*(range(len(a[2])) for a in axes)):
        values = {name: arr.copy() for name, arr in base.items()}
        probability = 1.0
        for (name, node, vals, probs), k in zip(axes, combo, strict=True):
            values[name][node] = vals[k]
            probability *= float(probs[k])
        realized = {name: realize(problem, name, arr) for name, arr in values.items()}
        yield probability, Scenario(problem.spec_hash, problem.instance_hash, realized)


def expected_cost(
    spec: ProblemSpec,
    instance: Instance,
    run_episode: Callable[[FleetGymEnv, Scenario], float],
    *,
    max_scenarios: int = MAX_SCENARIOS,
) -> float:
    """Probability-weighted cost of ``run_episode`` over every scenario.

    Parameters
    ----------
    run_episode
        Plays one episode on the given env and scenario (``env.reset(options={"scenario": s})``)
        and returns its total cost.
    """
    env = FleetGymEnv(spec, instance=instance)
    total = math.fsum(
        p * run_episode(env, scenario)
        for p, scenario in enumerate_scenarios(spec, instance, max_scenarios=max_scenarios)
    )
    return total
