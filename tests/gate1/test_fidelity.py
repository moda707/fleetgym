"""Gate 1: model fidelity. Published best-known solutions replay through the environment.

Downloads CVRPLIB files (marker ``network``); the full X set is also ``slow``.
CVRPLIB states the X-set conventions: Euclidean distances rounded to the nearest integer (TSPLIB),
cost = total distance, free number of routes, which is exactly ``CVRP-v0``.
"""

import pytest

from fleetgym.data import read_instance, read_solution
from fleetgym.data.download import fetch, load_collection
from fleetgym.spec import load_spec
from fleetgym.testing.replication import replay_solution

X = load_collection("cvrplib/X")


def replay(name: str) -> None:
    instance = read_instance(fetch("cvrplib/X", name))
    solution = read_solution(fetch("cvrplib/X", name, kind="solution"))
    assert solution.cost == X.published_cost(name)
    result = replay_solution(load_spec("CVRP-v0"), instance, solution)
    assert result.feasible, result.errors
    assert result.validation is not None
    assert result.validation.ok, result.validation.violations
    assert result.cost == result.published  # integer costs: exact equality
    assert result.matches(0.0)


@pytest.mark.network
@pytest.mark.parametrize("name", ["X-n101-k25", "X-n106-k14", "X-n110-k13"])
def test_cvrplib_x_quick(name: str):
    replay(name)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("name", X.names())
def test_cvrplib_x_full_set(name: str):
    replay(name)


@pytest.mark.skip(
    reason="awaiting the user's choice of Solomon BKS files and rounding (plan Q18/Q19)"
)
def test_solomon_fidelity():
    raise NotImplementedError
