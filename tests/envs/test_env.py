import subprocess
import sys
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env

import fleetgym as fg
import fleetgym.envs
from fleetgym.data import read_instance
from fleetgym.errors import ConfigError, InfeasibleActionError, NotInScopeError
from fleetgym.eventlog import EventLog
from fleetgym.testing.strategies import random_masked_rollout
from tests.sim.helpers import LINE_XS, env_for, line_instance, play

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "instances"
REGISTERED = fg.REGISTRY.ids()


def rollout_digest(env_id: str, seed: int) -> tuple[str, float]:
    env = fg.make(env_id)
    total = random_masked_rollout(env, np.random.default_rng(seed), seed=seed)
    return env.event_log.digest(), total


# ------------------------------------------------------------------ gymnasium conformance


@pytest.mark.parametrize("env_id", REGISTERED)
def test_check_env_passes_for_every_registered_config(env_id: str):
    # check_env steps random, unmasked actions, so it runs with invalid_action="repair"
    env = gym.make(f"fleetgym/{env_id}", invalid_action="repair", render_mode="ansi")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env.unwrapped)


def test_registration_and_planned_configs():
    assert {f"fleetgym/{i}" for i in REGISTERED} <= set(gym.registry)
    assert "fleetgym/EVRP-NL-v0" not in gym.registry
    with pytest.raises(NotInScopeError, match="planned"):
        fg.make("EVRP-NL-v0")
    assert fleetgym.envs.register_envs() == []  # idempotent


@pytest.mark.parametrize("env_id", REGISTERED)
def test_observations_are_in_the_space(env_id: str):
    env = fg.make(env_id)
    obs, info = env.reset(seed=1)
    rng = np.random.default_rng(1)
    terminated = False
    while not terminated:
        assert env.observation_space.contains(obs)
        assert obs["action_mask"].any()
        np.testing.assert_array_equal(obs["action_mask"].astype(bool), info["action_mask"])
        obs, _, terminated, _, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    assert env.observation_space.contains(obs)
    assert obs["action_mask"].any()


# ------------------------------------------------------------------ invalid actions


def test_invalid_action_modes():
    env = env_for(line_instance(LINE_XS, [0, 4, 3, 5]))
    env.reset(seed=0)
    with pytest.raises(InfeasibleActionError, match="unlimited"):
        env.step(0)
    with pytest.raises(InfeasibleActionError, match="outside the action space"):
        env.step(17)

    repair = fg.FleetGymEnv(
        env.problem_spec, instance=env.problem.instance, invalid_action="repair"
    )
    repair.reset(seed=0)
    _, _, _, _, info = repair.step(0)  # depot masked: repaired to the nearest feasible node (x=2)
    assert info["repaired_action"] == 1

    penalize = fg.FleetGymEnv(
        env.problem_spec,
        instance=env.problem.instance,
        invalid_action="penalize",
        invalid_action_penalty=7.0,
    )
    obs0, _ = penalize.reset(seed=0)
    obs1, reward, terminated, _, info = penalize.step(0)
    assert reward == -7.0 and not terminated and info["invalid_action"]
    np.testing.assert_array_equal(obs0["nodes"], obs1["nodes"])
    assert penalize.event_log.of_kind("cost")[-1]["component"] == "invalid_action"


def test_padding_slots_and_repair_from_padding():
    inst = line_instance([0, 2], [0, 1])
    env = fg.FleetGymEnv(
        env_for(inst).problem_spec, instance=inst, max_nodes=4, invalid_action="repair"
    )
    _, info = env.reset(seed=0)
    assert info["action_mask"].tolist() == [False, True, False, False]
    assert env.step(3)[4]["repaired_action"] == 1
    strict = fg.FleetGymEnv(env.problem_spec, instance=inst, max_nodes=4)
    strict.reset(seed=0)
    with pytest.raises(InfeasibleActionError, match="padding"):
        strict.step(2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"invalid_action": "ignore"}, "invalid_action"),
        ({"invalid_action": "penalize"}, "invalid_action_penalty"),
        ({"invalid_action_penalty": 1.0}, "invalid_action_penalty"),
        ({"render_mode": "human"}, "render_mode"),
    ],
)
def test_constructor_argument_checks(kwargs, match):
    with pytest.raises(ValueError, match=match):
        fg.make("CVRP-v0", **kwargs)


def test_step_before_reset_and_after_end():
    env = env_for(line_instance([0, 2], [0, 1]))
    with pytest.raises(ResetNeeded):
        env.step(1)
    _, _, terminated = play(env, [1, 0])
    assert terminated
    assert env.action_masks().tolist() == [True, False]
    with pytest.raises(ResetNeeded):
        env.step(0)


# ------------------------------------------------------------------ seeding and determinism


def test_same_seed_same_episode_and_seed_sequence():
    a = fg.make("VRPTW-v0")
    b = fg.make("VRPTW-v0")
    for env in (a, b):
        env.reset(seed=5)
    assert a.problem.instance_hash == b.problem.instance_hash
    assert a.episode_seed == 5
    a.reset()
    b.reset()
    assert a.episode_seed == b.episode_seed != 5
    assert a.problem.instance_hash == b.problem.instance_hash
    c = fg.make("VRPTW-v0")
    c.reset(seed=6)
    assert c.problem.instance_hash != a.problem.instance_hash
    unseeded = fg.make("CVRP-v0")
    unseeded.reset()
    assert unseeded.episode_seed >= 0
    with pytest.raises(ValueError, match="non-negative"):
        unseeded.reset(seed=-1)
    with pytest.raises(ValueError, match="unknown reset options"):
        unseeded.reset(options={"scenario": None})


@pytest.mark.parametrize("env_id", REGISTERED)
def test_same_seed_same_trajectory(env_id: str):
    assert rollout_digest(env_id, 3) == rollout_digest(env_id, 3)
    assert rollout_digest(env_id, 3)[0] != rollout_digest(env_id, 4)[0]


def test_trajectories_match_across_processes():
    code = (
        "import numpy as np, fleetgym as fg;"
        "from fleetgym.testing.strategies import random_masked_rollout;"
        "env = fg.make('VRPTW-v0');"
        "t = random_masked_rollout(env, np.random.default_rng(11), seed=11);"
        "print(env.event_log.digest(), repr(t))"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(2)
    }
    digest, total = rollout_digest("VRPTW-v0", 11)
    assert outputs == {f"{digest} {total!r}\n"}


# ------------------------------------------------------------------ snapshots


@pytest.mark.parametrize("env_id", REGISTERED)
def test_snapshot_restore_is_exact(env_id: str):
    env = fg.make(env_id)
    env.reset(seed=2)
    rng = np.random.default_rng(2)
    for _ in range(3):
        env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
    snapshot = env.get_state()
    actions, first = [], []
    terminated = False
    while not terminated:
        action = int(rng.choice(np.flatnonzero(env.action_masks())))
        obs, reward, terminated, _, info = env.step(action)
        actions.append(action)
        first.append((obs, reward, info["total_costs"]))
    digest = env.event_log.digest()

    env.set_state(snapshot)
    for action, (obs, reward, totals) in zip(actions, first, strict=True):
        obs2, reward2, _, _, info2 = env.step(action)
        assert reward2 == reward and info2["total_costs"] == totals
        for key in obs:
            np.testing.assert_array_equal(obs2[key], obs[key])
    assert env.event_log.digest() == digest


def test_clone_is_independent():
    env = fg.make("CVRP-v0")
    env.reset(seed=4)
    env.step(int(np.flatnonzero(env.action_masks())[0]))
    twin = env.clone()
    assert twin.event_log.digest() == env.event_log.digest()
    twin.step(int(np.flatnonzero(twin.action_masks())[-1]))
    assert len(twin.event_log) > len(env.event_log)
    assert env.simulator.state.epoch_count == 2
    assert twin.simulator.state.epoch_count == 3
    fresh = fg.make("CVRP-v0").clone()
    with pytest.raises(ResetNeeded):
        fresh.step(0)
    with pytest.raises(ValueError, match="different spec"):
        fg.make("VRPTW-v0").set_state(env.get_state())


# ------------------------------------------------------------------ rewards, costs and logs


@pytest.mark.parametrize("env_id", REGISTERED)
def test_return_equals_minus_weighted_total_cost(env_id: str):
    env = fg.make(env_id)
    total = random_masked_rollout(env, np.random.default_rng(9), seed=9)
    costs = env.simulator.state.total_costs
    assert total == pytest.approx(-env.reward_model.cost(costs), rel=1e-12)
    logged = sum(r["reward"] for r in env.event_log.of_kind("epoch"))
    assert logged == pytest.approx(total, rel=1e-12)
    by_component: dict[str, float] = {}
    for row in env.event_log.of_kind("cost"):
        by_component[row["component"]] = by_component.get(row["component"], 0.0) + row["amount"]
    assert by_component == pytest.approx(costs)


def test_lexicographic_reward_scalarization():
    env = fg.make("VRPTW-v0")
    total = random_masked_rollout(env, np.random.default_rng(0), seed=0)
    costs = env.simulator.state.total_costs
    assert total == pytest.approx(-(1e6 * costs["vehicles"] + costs["distance"]))


def test_event_log_round_trips_through_parquet(tmp_path: Path):
    env = fg.make("VRPTW-v0")
    random_masked_rollout(env, np.random.default_rng(1), seed=1)
    path = tmp_path / "log.parquet"
    env.event_log.write(path)
    again = EventLog.read(path)
    assert again.rows == env.event_log.rows
    assert again.metadata["spec_hash"] == env.problem.spec_hash
    assert again.metadata["instance_hash"] == env.problem.instance_hash
    assert again.metadata["episode_seed"] == "1"
    assert again.metadata["env_id"] == "VRPTW-v0"


def test_render_ansi():
    env = fg.make("CVRP-v0", render_mode="ansi")
    random_masked_rollout(env, np.random.default_rng(0), seed=0)
    text = env.render()
    assert text is not None and text.startswith("t=") and "vehicle 0 [done]" in text
    assert fg.make("CVRP-v0").render() is None


# ------------------------------------------------------------------ instances and slots


def test_fixed_instance_and_reset_option():
    inst = read_instance(FIXTURES / "tiny.vrp")
    env = fg.make("CVRP-v0", instance=inst)
    assert env.max_nodes == 5
    env.reset(seed=0)
    assert env.problem.instance is inst
    other = read_instance(FIXTURES / "tiny_solomon.txt")
    env.reset(seed=0, options={"instance": other})
    assert env.problem.instance is other
    assert env.simulator.network.distance(0, 1) == 5.0  # (0, 0) to (3, 4)
    big = fg.make("CVRP-v0")  # 21 node slots
    with pytest.raises(ConfigError, match="node slots"):
        fg.make("CVRP-v0", max_nodes=3).reset(seed=0)
    assert big.max_nodes == 21


def test_max_nodes_inference(tmp_path: Path):
    files = fg.load_spec(
        {
            "fleet": {"types": [{"name": "v", "capacity": "from_instance"}]},
            "constraints": {"capacity": {}},
            "objective": {"distance": 1},
            "decisions": {"epochs": "sequential"},
            "instance_source": {
                "type": "files",
                "paths": [str(FIXTURES / "tiny.vrp"), str(FIXTURES / "tiny_solomon.txt")],
            },
        }
    )
    assert fg.FleetGymEnv(files).max_nodes == 5
    no_source = files.model_copy(update={"instance_source": None})
    with pytest.raises(ConfigError, match="max_nodes"):
        fg.FleetGymEnv(no_source)


def test_vehicle_slots_must_fit_the_fleet():
    spec = env_for(line_instance([0, 1], [0, 1])).problem_spec
    fixed = fg.load_spec(
        spec.to_dict()
        | {
            "fleet": {"sizing": "fixed", "types": [{"name": "v", "capacity": 5, "count": 3}]},
            "objective": {"distance": 1, "unserved": 1},
        }
    )
    env = fg.FleetGymEnv(fixed, instance=line_instance([0, 1], [0, 1]), max_vehicles=2)
    with pytest.raises(ConfigError, match="vehicle slots"):
        env.reset(seed=0)


def test_make_vec():
    vec = fg.make_vec("CVRP-v0", 3, invalid_action="repair")
    obs, _ = vec.reset(seed=0)
    assert obs["nodes"].shape == (3, 21, 8)
    obs, rewards, *_ = vec.step(np.array([1, 2, 3]))
    assert rewards.shape == (3,)
    vec.close()
    with pytest.raises(ValueError, match="n_envs"):
        fg.make_vec("CVRP-v0", 0)
    with pytest.raises(ValueError, match="mode"):
        fg.make_vec("CVRP-v0", 1, mode="threads")  # pyright: ignore[reportArgumentType]


def test_episode_without_any_decision_has_one_final_step():
    # route duration 5 cannot reach the only customer (distance 8): nothing can be decided
    inst = line_instance([0, 8], [0, 1])
    changes = {
        "constraints": {"route_duration": {"max": 5}},
        "objective": {"distance": 1.0, "unserved": 10.0},
    }
    env = env_for(inst, **changes)
    obs, info = env.reset(seed=0)
    assert info["vehicle"] is None
    assert obs["action_mask"].tolist() == [1, 0]
    with pytest.raises(InfeasibleActionError, match="only valid action is 0"):
        env.step(1)
    _, reward, terminated, _, info = env.step(0)
    assert terminated and reward == -10.0
    assert info["costs"] == {"unserved": 1.0}
    with pytest.raises(ResetNeeded):
        env.step(0)
    assert sum(r["reward"] for r in env.event_log.of_kind("epoch")) == -10.0

    repair = fg.FleetGymEnv(env.problem_spec, instance=inst, invalid_action="repair")
    repair.reset(seed=0)
    assert repair.step(1)[2] is True
