"""MaskableDQN, SB3 helpers, callbacks and the train/eval CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import fleetgym as fg  # noqa: E402
from fleetgym.agents import (  # noqa: E402
    CostComponentLogger,
    MaskableDQN,
    MaskableDQNPolicy,
    SB3Policy,
    ScenarioEval,
    make_sb3_vec,
    policy_settings,
    sb3_defaults,
)
from fleetgym.cli.__main__ import main  # noqa: E402

pytestmark = pytest.mark.extras_sb3

TINY = {
    "extends": "CVRP-v0",
    "instance_source": {
        "type": "generator",
        "name": "uniform",
        "params": {"n_customers": 4, "size": 20, "capacity": 10, "demand": {"low": 1, "high": 3}},
    },
}


def dqn(steps: int = 0) -> MaskableDQN:
    vec = make_sb3_vec(TINY, 2, seed=0)
    model = MaskableDQN(
        MaskableDQNPolicy, vec, seed=0, learning_starts=20, buffer_size=500, batch_size=16
    )
    if steps:
        model.learn(steps)
    return model


def test_masked_q_values_targets_and_exploration():
    model = dqn()
    env = fg.make(TINY)
    obs, _ = env.reset(seed=1)
    tensor_obs, _ = model.policy.obs_to_tensor(obs)
    q = model.q_net(tensor_obs)
    mask = torch.as_tensor(obs["action_mask"]).bool().reshape(q.shape)
    assert torch.isneginf(q[~mask]).all() and torch.isfinite(q[mask]).all()
    target_max = model.q_net_target(tensor_obs).max(dim=1).values
    assert torch.isfinite(target_max).all()  # the max ignores masked actions
    model.exploration_rate = 1.0
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=False)
        assert obs["action_mask"][int(action)]
    action, _ = model.predict(obs, deterministic=True)
    assert obs["action_mask"][int(action)]


def test_dqn_trains_without_masked_actions_and_fully_masked_targets_are_zero():
    model = dqn(steps=300)  # warm-up and training steps would raise on a masked action
    assert model._n_updates > 0  # pyright: ignore[reportPrivateUsage]
    batch = model.replay_buffer.sample(8)  # type: ignore[union-attr]
    next_obs = {k: v.clone() for k, v in batch.next_observations.items()}  # type: ignore[union-attr]
    next_obs["action_mask"][:] = 0
    next_q = model.q_net_target(next_obs).max(dim=1).values
    assert torch.isneginf(next_q).all()  # train() replaces these by 0 before bootstrapping


def test_helpers_and_callbacks(tmp_path):
    assert sb3_defaults("anything", "maskable_ppo")["gamma"] == 1.0
    with pytest.raises(ValueError, match="unknown algorithm"):
        sb3_defaults("x", "sac")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="unknown policy"):
        policy_settings("maskable_ppo", "transformer-xl")
    with pytest.raises(ValueError, match="positive"):
        make_sb3_vec(TINY, 0)
    from sb3_contrib import MaskablePPO

    policy, kwargs = policy_settings("maskable_ppo", "attention", embed_dim=16)
    model = MaskablePPO(
        policy, make_sb3_vec(TINY, 2), n_steps=16, batch_size=16, policy_kwargs=kwargs
    )
    evaluator = ScenarioEval(fg.make(TINY), seeds=[1, 2], every=16)
    model.learn(64, callback=[evaluator, CostComponentLogger()])
    assert evaluator.history and evaluator.history[0][0] == 16
    assert all(np.isfinite(cost) for _, cost in evaluator.history)
    assert SB3Policy(model, name="ppo").name == "ppo"


@pytest.mark.parametrize("algo", ["maskable_ppo", "maskable_dqn"])
def test_cli_train_then_eval(tmp_path: Path, algo: str, capsys):
    out = tmp_path / algo
    args = ["train", "CVRP-v0", "--algo", algo, "--policy", "mlp", "--steps", "1000"]
    assert main([*args, "--n-envs", "2", "--embed-dim", "16", "--out", str(out)]) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["algo"] == algo and run["steps"] == 1000
    report = tmp_path / "report.json"
    args = ["eval", "CVRP-v0", "--model", str(out), "--episodes", "3", "--reference", "nearest"]
    assert main([*args, "--out", str(report)]) == 0
    card = json.loads(report.read_text())
    assert card["training"]["algo"] == algo and card["reference"]["policy"] == "nearest"
    assert main(["eval", "CVRP-v0", "--baseline", "insertion", "--episodes", "2"]) == 0
    assert main(["eval", "CVRP-v0", "--model", str(tmp_path), "--episodes", "2"]) == 1
    assert main(["eval", "VRPTW-v0", "--model", str(out), "--episodes", "2"]) == 1


@pytest.mark.slow
def test_maskable_ppo_learns_a_tiny_instance():
    """Agent sanity: on one fixed 4-customer instance, PPO reaches the optimal route."""
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from fleetgym.baselines import PerfectInformation, run_episode
    from fleetgym.data import generate

    instance = generate(
        "uniform",
        {"n_customers": 4, "size": 20, "capacity": 10, "demand": {"low": 1, "high": 3}},
        np.random.default_rng(3),
    )
    env = fg.make(TINY, instance=instance)
    optimum = run_episode(env, PerfectInformation(), seed=0).cost
    policy, kwargs = policy_settings("maskable_ppo", "attention", embed_dim=32)
    model = MaskablePPO(
        policy,
        DummyVecEnv([lambda: fg.make(TINY, instance=instance) for _ in range(4)]),
        n_steps=64,
        batch_size=64,
        learning_rate=1e-3,
        gamma=1.0,
        seed=0,
        policy_kwargs=kwargs,
    )
    model.learn(20_000)
    assert run_episode(env, SB3Policy(model), seed=0).cost == pytest.approx(optimum)
