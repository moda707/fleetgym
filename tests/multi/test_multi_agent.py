"""Multi-agent environments (PettingZoo AEC and Parallel)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from pettingzoo.test import api_test, parallel_api_test

import fleetgym as fg
from fleetgym.data import Instance
from fleetgym.envs.multi import FleetGymAECEnv, FleetGymParallelEnv, agent_name
from fleetgym.errors import ConfigError, InfeasibleActionError
from fleetgym.spec import load_spec
from fleetgym.spec.models import ProblemSpec
from fleetgym.testing.validator import validate

# Depot at 0; customers 1 and 2 left, 3 and 4 right of the depot.
LINE = Instance(
    name="line4",
    source="test",
    demands=np.array([0.0, 1.0, 1.0, 1.0, 1.0]),
    coords=np.array([[0.0, 0.0], [-1.0, 0.0], [-2.0, 0.0], [1.0, 0.0], [3.0, 0.0]]),
    capacity=10.0,
)


def spec(count: int = 2, **agents: Any) -> ProblemSpec:
    return load_spec(
        {
            "fleet": {
                "sizing": "fixed",
                "types": [{"name": "v", "capacity": "from_instance", "count": count}],
            },
            "network": {"type": "euclidean", "rounding": "none"},
            "constraints": {"capacity": {}},
            "decisions": {"interface": "next_node", "epochs": "on_event"},
            "objective": {"distance": 1.0, "unserved": 100.0},
            "agents": {"mode": "multi", **agents},
        }
    )


@pytest.mark.parametrize("env_id", ["MA-CVRP-v0", "MA-DVRP-v0"])
def test_pettingzoo_api(env_id):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # dict observations, inactive agents' masks
        api_test(fg.make_multi(env_id, api="aec"), num_cycles=300)
        parallel_api_test(fg.make_multi(env_id, api="parallel"), num_cycles=300)


def test_factories_and_registration():
    import gymnasium as gym

    with pytest.raises(ConfigError, match="make_multi"):
        fg.make("MA-CVRP-v0")
    assert isinstance(fg.make_multi("MA-CVRP-v0"), FleetGymParallelEnv)
    assert isinstance(fg.make_multi("MA-CVRP-v0", api="aec"), FleetGymAECEnv)
    assert "fleetgym/MA-CVRP-v0" not in gym.registry
    assert set(fg.REGISTRY.ids(agents="multi")) == {"MA-CVRP-v0", "MA-DVRP-v0"}
    with pytest.raises(ConfigError, match="next_node"):
        FleetGymAECEnv(load_spec("DVRP-StochReq-v0"))
    with pytest.raises(ConfigError, match="fixed or limited"):
        FleetGymAECEnv(load_spec("CVRP-v0"))
    with pytest.raises(ConfigError, match="number of agents"):
        FleetGymAECEnv(
            load_spec(
                {
                    **spec().to_dict(),
                    "fleet": {
                        "sizing": "fixed",
                        "types": [{"name": "v", "capacity": 5, "count": "from_instance"}],
                    },
                }
            ),
            max_nodes=5,
        )


def play_aec(env: FleetGymAECEnv, policy) -> float:
    total = 0.0
    for agent in env.agent_iter():
        _, reward, terminated, truncated, _ = env.last()
        total += reward
        if terminated or truncated:
            env.step(None)
        else:
            env.step(policy(env, agent))
    return total


def test_single_vehicle_aec_equals_the_single_agent_env():
    s1 = load_spec({**spec(count=1).to_dict(), "agents": {"mode": "single"}})
    single = fg.make(s1, instance=LINE)
    single.reset(seed=3)
    rng = np.random.default_rng(0)
    actions, total = [], 0.0
    while True:
        a = int(rng.choice(np.flatnonzero(single.action_masks())))
        actions.append(a)
        _, r, done, _, _ = single.step(a)
        total += float(r)
        if done:
            break
    aec = FleetGymAECEnv(spec(count=1), instance=LINE)
    aec.reset(seed=3)
    queue = iter(actions)
    assert play_aec(aec, lambda env, agent: next(queue)) == pytest.approx(total)


def own_customers(vehicle: int) -> list[int]:
    return [1, 2] if vehicle == 0 else [3, 4]


def disjoint_policy(env_mask, vehicle: int, noop: int | None = None) -> int:
    for node in own_customers(vehicle):
        if env_mask[node]:
            return node
    return 0


def test_aec_and_parallel_agree_without_conflicts():
    aec = FleetGymAECEnv(spec(), instance=LINE)
    aec.reset(seed=0)
    total_aec = play_aec(
        aec, lambda env, agent: disjoint_policy(env.observe(agent)["action_mask"], int(agent[-1]))
    )
    par = FleetGymParallelEnv(spec(), instance=LINE)
    obs, _ = par.reset(seed=0)
    total_par = 0.0
    while par.agents:
        actions = {}
        for agent in par.agents:
            mask = obs[agent]["action_mask"]
            actions[agent] = par.noop if mask[par.noop] else disjoint_policy(mask, int(agent[-1]))
        obs, rewards, _, _, _ = par.step(actions)
        total_par += sum(rewards.values())
    distance = 2 + 2 + 3 + 3
    assert aec.env.simulator.state.total_costs["distance"] == pytest.approx(distance)
    assert par.env.simulator.state.total_costs["distance"] == pytest.approx(distance)
    # Global rewards: every live agent receives the total, so the sums match too.
    assert total_aec == pytest.approx(total_par)
    assert validate(par.env.event_log, par.problem_spec, LINE).ok


def first_step_both_choose(env: FleetGymParallelEnv, node: int):
    obs, _ = env.reset(seed=0)
    assert all(not obs[a]["action_mask"][env.noop] for a in env.agents)
    return env.step(dict.fromkeys(env.agents, node))


@pytest.mark.parametrize(("rule", "winner"), [("priority", 0), ("distance", 0)])
def test_conflict_rules(rule, winner):
    env = FleetGymParallelEnv(spec(conflict=rule), instance=LINE)
    obs, _, _, _, _ = first_step_both_choose(env, 3)
    rows = env.env.event_log.of_kind("conflict")
    assert [(r["vehicle"], r["node"], r["component"]) for r in rows] == [(1 - winner, 3, rule)]
    loser = agent_name(1 - winner)
    assert not obs[loser]["action_mask"][env.noop]  # the loser decides again, at the same time
    assert not obs[loser]["action_mask"][3]


def test_lottery_is_reproducible():
    winners = set()
    for seed in range(8):
        env = FleetGymParallelEnv(spec(conflict="lottery"), instance=LINE)
        env.reset(seed=seed)
        env.step(dict.fromkeys(env.agents, 3))
        first = env.env.event_log.of_kind("conflict")[0]["vehicle"]
        again = FleetGymParallelEnv(spec(conflict="lottery"), instance=LINE)
        again.reset(seed=seed)
        again.step(dict.fromkeys(again.agents, 3))
        assert again.env.event_log.of_kind("conflict")[0]["vehicle"] == first
        winners.add(first)
    assert winners == {0, 1}


def test_reject_conflicts_penalize_and_change_nothing():
    with pytest.raises(ValueError, match="conflict_penalty"):
        FleetGymParallelEnv(spec(conflict="reject"), instance=LINE)
    env = FleetGymParallelEnv(spec(conflict="reject"), instance=LINE, conflict_penalty=5.0)
    _, rewards, _, _, infos = first_step_both_choose(env, 3)
    assert rewards == {"vehicle_0": -5.0, "vehicle_1": -5.0}
    assert all(v.used is False for v in env.env.simulator.state.vehicles)
    assert infos["vehicle_0"]["conflict"] is True


def test_parallel_action_checks():
    env = FleetGymParallelEnv(spec(), instance=LINE)
    env.reset(seed=0)
    with pytest.raises(InfeasibleActionError, match="missing"):
        env.step({"vehicle_0": 1})
    with pytest.raises(InfeasibleActionError, match="no-op is masked"):
        env.step({"vehicle_0": env.noop, "vehicle_1": 3})
    env.step({"vehicle_0": 1, "vehicle_1": 3})
    obs = env._observations()  # pyright: ignore[reportPrivateUsage]
    busy = [a for a in env.agents if obs[a]["action_mask"][env.noop]]
    if busy:
        ready = [a for a in env.agents if a not in busy]
        actions = dict.fromkeys(busy, 0) | {
            a: int(np.flatnonzero(obs[a]["action_mask"])[0]) for a in ready
        }
        with pytest.raises(InfeasibleActionError, match="send the no-op"):
            env.step(actions)


@pytest.mark.parametrize("mode", ["individual", "mixed"])
def test_individual_rewards_add_up(mode):
    extra: dict[str, Any] = {"reward": mode} | ({"reward_mix": 0.25} if mode == "mixed" else {})
    env = FleetGymParallelEnv(spec(**extra), instance=LINE)
    obs, _ = env.reset(seed=0)
    reward_sum, split = 0.0, False
    while env.agents:
        actions = {
            a: env.noop
            if obs[a]["action_mask"][env.noop]
            else disjoint_policy(obs[a]["action_mask"], int(a[-1]))
            for a in env.agents
        }
        obs, rewards, _, _, _ = env.step(actions)
        split |= len(set(np.round(list(rewards.values()), 9))) > 1
        reward_sum += sum(rewards.values())
    assert split  # the vehicles' own costs differ
    if mode == "individual":
        # Individual rewards add up to the total reward (only distance costs here).
        assert reward_sum == pytest.approx(-env.env.simulator.state.total_costs["distance"])


def test_aec_rewards_and_state():
    env = FleetGymAECEnv(spec(reward="individual"), instance=LINE)
    env.reset(seed=0)
    state = env.state()
    assert state.shape == env.state_space.shape and env.state_space.contains(state)
    total = play_aec(
        env, lambda e, agent: disjoint_policy(e.observe(agent)["action_mask"], int(agent[-1]))
    )
    assert total == pytest.approx(-env.env.simulator.state.total_costs["distance"])
    assert env.render() is None


def test_dynamic_multi_agent_episodes_validate():
    env = fg.make_multi("MA-DVRP-v0", api="parallel")
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        while env.agents:
            actions = {
                a: int(rng.choice(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents
            }
            obs, _, _, _, _ = env.step(actions)
        inner = env.env
        report = validate(
            inner.event_log, inner.problem_spec, inner.base_instance, scenario=inner.scenario
        )
        assert report.ok, report.violations[:5]
