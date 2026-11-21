from pathlib import Path

import numpy as np
import pytest

import fleetgym as fg
from fleetgym.errors import ConfigError, ScenarioMismatchError
from fleetgym.spec import load_spec
from fleetgym.spec.binding import bind, instance_from_source
from fleetgym.spec.models import ProblemSpec
from fleetgym.testing.strategies import random_masked_rollout
from fleetgym.testing.validator import validate
from fleetgym.uncertainty.sampling import resample_unrevealed, sample_scenario
from fleetgym.uncertainty.scenario import Scenario
from tests.families import instance_source, make_env

DEMAND = {"dist": "poisson", "mean": "from_instance", "revealed": "on_arrival"}
SERVICE = {"dist": "gamma", "shape": 2.0, "mean": 3.0, "revealed": "on_completion"}


def stochastic(**uncertainty) -> ProblemSpec:
    return load_spec(
        {
            "extends": "CVRP-v1",
            "instance_source": instance_source("cvrp"),
            "uncertainty": uncertainty or {"demand": DEMAND},
            "recourse": {"capacity_failure": "return_to_depot"},
        }
    )


def problem_for(spec):
    return bind(spec, instance_from_source(spec, np.random.default_rng(0)))


def test_same_seed_same_scenario_and_depots_are_deterministic():
    problem = problem_for(stochastic(demand=DEMAND, service_time=SERVICE))
    a, b = sample_scenario(problem, 4), sample_scenario(problem, 4)
    assert a == b and a.id == b.id and a.seed == 4
    assert sample_scenario(problem, 5) != a
    assert a["demand"][0] == 0 and a["service_time"][0] == 0
    assert a.sources == ("demand", "service_time")
    assert np.all(a["demand"] == np.round(a["demand"]))  # Poisson draws are integers


def test_sources_have_independent_streams():
    with_service = problem_for(stochastic(demand=DEMAND, service_time=SERVICE))
    demand_only = problem_for(stochastic(demand=DEMAND))
    np.testing.assert_array_equal(
        sample_scenario(with_service, 9)["demand"], sample_scenario(demand_only, 9)["demand"]
    )


def test_parquet_round_trip_is_exact(tmp_path: Path):
    problem = problem_for(stochastic(demand=DEMAND, service_time=SERVICE))
    scenario = sample_scenario(problem, 11)
    path = scenario.write(tmp_path / "s" / "scenario.parquet")
    again = Scenario.read(path)
    assert again == scenario and again.seed == 11
    for name in scenario.sources:
        np.testing.assert_array_equal(again[name], scenario[name])
    with pytest.raises(ValueError, match="read-only"):
        again["demand"][1] = 3.0
    empty = Scenario("a", "b", {})
    assert Scenario.read(empty.write(tmp_path / "e.parquet")) == empty


def test_read_rejects_other_parquet_files(tmp_path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"a": [1]}), path)
    with pytest.raises(ScenarioMismatchError, match="not a FleetGym scenario"):
        Scenario.read(path)


def test_scenario_shape_checks():
    with pytest.raises(ValueError, match="one-dimensional"):
        Scenario("a", "b", {"demand": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="different lengths"):
        Scenario("a", "b", {"demand": np.zeros(2), "service_time": np.zeros(3)})


def test_replay_gives_the_identical_trajectory_and_mismatches_raise():
    spec = stochastic(demand=DEMAND, service_time=SERVICE)
    env = fg.make(spec)
    random_masked_rollout(env, np.random.default_rng(3), seed=3)
    scenario, instance = env.scenario, env.problem.instance
    assert scenario is not None
    replay = fg.make(spec, instance=instance)
    replay.reset(seed=999, options={"scenario": scenario})
    assert replay.scenario == scenario
    rng = np.random.default_rng(3)
    for _ in range(10_000):
        _, _, terminated, _, _ = replay.step(int(rng.choice(np.flatnonzero(replay.action_masks()))))
        if terminated:
            break
    assert replay.event_log.rows == env.event_log.rows
    assert replay.event_log.metadata["scenario_id"] == scenario.id
    assert validate(replay.event_log, spec, instance, scenario=scenario).ok
    other = stochastic(demand={**DEMAND, "revealed": "on_departure"})
    with pytest.raises(ScenarioMismatchError, match="spec"):
        fg.make(other, instance=instance).reset(seed=0, options={"scenario": scenario})
    with pytest.raises(ScenarioMismatchError, match="sources"):
        fg.make(spec, instance=instance).reset(
            seed=0,
            options={
                "scenario": Scenario(
                    scenario.spec_hash, scenario.instance_hash, {"demand": scenario["demand"]}
                )
            },
        )
    with pytest.raises(ValueError, match="no uncertain parameters"):
        make_env("CVRP-v1").reset(seed=0, options={"scenario": scenario})
    with pytest.raises(ValueError, match="scenario"):
        validate(env.event_log, spec, instance)


def test_negative_draws_are_rejected_with_the_config_path():
    spec = stochastic(demand={"dist": "normal", "mean": 0.0, "std": 5.0, "revealed": "on_arrival"})
    with pytest.raises(ConfigError, match="non-negative") as exc:
        sample_scenario(problem_for(spec), 0)
    assert exc.value.key_path == "uncertainty.demand"


def test_resample_unrevealed_keeps_revealed_values():
    problem = problem_for(stochastic())
    scenario = sample_scenario(problem, 1)
    revealed = {"demand": np.arange(problem.instance.n_nodes) % 2 == 0}
    fresh = resample_unrevealed(problem, scenario, revealed, seed=77)
    np.testing.assert_array_equal(
        fresh["demand"][revealed["demand"]], scenario["demand"][revealed["demand"]]
    )
    assert not np.array_equal(fresh["demand"], scenario["demand"])
    assert fresh.seed is None
