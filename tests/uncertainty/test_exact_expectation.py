"""Exact expectation (plan M5): environment vs an independent dynamic program.

A single vehicle follows an a priori route through 5 customers whose demands are i.i.d. categorical
on {0, 2, 4, 5} (4^5 = 1024 scenarios), capacity 4, revealed on arrival. The probability-weighted
cost of the environment over every scenario must equal the dynamic program below to 1e-9, for each
recourse rule and parameter combination. The program (`tests.uncertainty.vrpsd_dp`) is
written from the rules in docs/recourse.md and shares no code with the simulator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fleetgym.baselines.apriori import AprioriRoute
from fleetgym.data import Instance
from fleetgym.envs.env import FleetGymEnv
from fleetgym.spec import load_spec
from fleetgym.spec.models import ProblemSpec
from fleetgym.testing.exact import enumerate_scenarios, expected_cost
from fleetgym.testing.validator import validate
from fleetgym.uncertainty.scenario import Scenario
from tests.uncertainty.vrpsd_dp import apriori_expected_cost

COORDS = np.array([[0, 0], [3, 4], [6, 1], [2, -5], [-4, -3], [-6, 2]], dtype=float)
VALUES = (0.0, 2.0, 4.0, 5.0)
PROBS = (0.1, 0.4, 0.3, 0.2)
Q = 4.0
ORDER = (2, 1, 5, 4, 3)
PER_CUSTOMER, PER_UNIT = 1.5, 0.7

INSTANCE = Instance(
    name="exact-5",
    source="test",
    demands=np.array([0.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
    coords=COORDS,
    capacity=Q,
)


def d(i: int, j: int) -> float:
    return math.hypot(*(COORDS[i] - COORDS[j]))


def spec_for(rule: dict) -> ProblemSpec:
    return load_spec(
        {
            "fleet": {"sizing": "unlimited", "types": [{"name": "v", "capacity": "from_instance"}]},
            "network": {"type": "euclidean", "rounding": "none"},
            "constraints": {"capacity": {}},
            "objective": {"distance": 1.0, "outsourcing": 1.0},
            "decisions": {"interface": "next_node", "epochs": "sequential"},
            "uncertainty": {
                "demand": {
                    "dist": "categorical",
                    "values": list(VALUES),
                    "probs": list(PROBS),
                    "revealed": "on_arrival",
                }
            },
            "recourse": {"capacity_failure": rule},
        }
    )


def dynamic_program(rule: str, partial: bool, depletion: str) -> float:
    """Expected cost of the a priori route ``ORDER`` under the given recourse."""
    return apriori_expected_cost(
        d,
        ORDER,
        Q,
        lambda node: (VALUES, PROBS),
        rule=rule,
        partial=partial,
        depletion=depletion,
        per_customer=PER_CUSTOMER,
        per_unit=PER_UNIT,
    )


RULES = [
    ("return_to_depot", True, "continue"),
    ("return_to_depot", True, "restock"),
    ("return_to_depot", False, "continue"),
    ("return_to_depot", False, "restock"),
    ("outsource", True, "continue"),
    ("outsource", False, "continue"),
]


def rule_dict(rule: str, partial: bool, depletion: str) -> dict:
    if rule == "return_to_depot":
        return {"rule": rule, "partial_service": partial, "on_exact_depletion": depletion}
    return {
        "rule": rule,
        "partial_service": partial,
        "per_customer": PER_CUSTOMER,
        "per_unit": PER_UNIT,
    }


@pytest.mark.parametrize(("rule", "partial", "depletion"), RULES)
def test_expected_cost_equals_dynamic_program(rule, partial, depletion):
    spec = spec_for(rule_dict(rule, partial, depletion))
    policy = AprioriRoute(ORDER)

    def run(env: FleetGymEnv, scenario: Scenario) -> float:
        cost = policy.run(env, seed=0, options={"scenario": scenario})
        report = validate(env.event_log, spec, INSTANCE, scenario=scenario)
        assert report.ok, report.violations[:3]
        assert len(env.simulator.state.vehicles) == 1  # one a priori route
        return cost

    expected = dynamic_program(rule, partial, depletion)
    assert expected_cost(spec, INSTANCE, run) == pytest.approx(expected, abs=1e-9, rel=0)


def test_enumeration_covers_every_scenario_once():
    spec = spec_for(rule_dict(*RULES[0]))
    scenarios = list(enumerate_scenarios(spec, INSTANCE))
    assert len(scenarios) == 4**5
    assert math.fsum(p for p, _ in scenarios) == pytest.approx(1.0, abs=1e-12)
    assert len({s.id for _, s in scenarios}) == 4**5
    assert all(s["demand"][0] == 0 for _, s in scenarios)
    with pytest.raises(ValueError, match="exceed"):
        next(enumerate_scenarios(spec, INSTANCE, max_scenarios=10))


def test_the_parameters_change_the_expected_cost():
    costs = {r: dynamic_program(*r) for r in RULES}
    assert len({round(c, 9) for c in costs.values()}) == len(RULES)
