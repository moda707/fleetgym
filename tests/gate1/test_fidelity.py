"""Gate 1: model fidelity. Published best-known solutions replay through the environment.

Downloads benchmark files (marker ``network``); the full sets are also ``slow``.

- CVRPLIB states the X-set conventions: Euclidean distances rounded to the nearest integer (TSPLIB),
  cost = total distance, free number of routes, which is exactly ``CVRP-v0``.
- SINTEF TOP states the Solomon conventions: hierarchical objective (vehicles, then distance),
  travel time equals Euclidean distance, double precision, totals published rounded to two decimals.
  That is ``VRPTW-v0``. The replayed distance must round to the published value and the vehicle
  count must match (plan Q19, ADR 0011).
"""

import pytest

from fleetgym.data import read_instance, read_solution
from fleetgym.data.download import fetch, load_collection
from fleetgym.testing.replication import replay_solution
from tests.families import spec_for

X = load_collection("cvrplib/X")
SOLOMON = load_collection("solomon/100")


def replay(name: str) -> None:
    instance = read_instance(fetch("cvrplib/X", name))
    solution = read_solution(fetch("cvrplib/X", name, kind="solution"))
    assert solution.cost == X.published_cost(name)
    result = replay_solution(spec_for("CVRP-v1"), instance, solution)
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


def replay_solomon(name: str) -> None:
    published = SOLOMON.info(name)
    instance = read_instance(fetch("solomon/100", name), fmt="solomon")
    solution = read_solution(fetch("solomon/100", name, kind="solution"))
    spec = spec_for("VRPTW-v1")
    result = replay_solution(spec, instance, solution)
    assert result.feasible, result.errors
    assert result.validation is not None
    assert result.validation.ok, result.validation.violations
    assert result.cost is not None
    assert round(result.cost, 2) == published["cost"]
    assert abs(result.cost - published["cost"]) <= 0.005 + 1e-9
    vehicles = replay_solution(spec, instance, solution, component="vehicles")
    assert vehicles.cost == len(solution.routes) == published["vehicles"]


@pytest.mark.network
@pytest.mark.parametrize("name", ["c101", "r101", "rc101", "r208"])
def test_solomon_quick(name: str):
    replay_solomon(name)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("name", SOLOMON.names())
def test_solomon_full_set(name: str):
    replay_solomon(name)


def test_solomon_manifest_is_complete():
    assert len(SOLOMON.names()) == 56
    assert SOLOMON.info("r111")["cost"] == 1096.73
    assert SOLOMON.info("c101")["vehicles"] == 10
