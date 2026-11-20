"""Train a MaskablePPO agent with a graph attention policy, then evaluate it (extra `sb3`).

Run: python examples/06_train_agent.py
"""

from sb3_contrib import MaskablePPO

from fleetgym.agents import SB3Policy, make_sb3_vec, policy_settings
from fleetgym.baselines import NearestNeighbor
from fleetgym.eval import evaluate, report_card, scenario_seeds

SMALL = {
    "extends": "CVRP-v0",
    "instance_source": {
        "type": "generator",
        "name": "uniform",
        "params": {"n_customers": 8, "size": 100, "capacity": 20, "demand": {"low": 1, "high": 9}},
    },
}

vec = make_sb3_vec(SMALL, n_envs=4, seed=0)
policy, policy_kwargs = policy_settings("maskable_ppo", "gat", embed_dim=32)
model = MaskablePPO(
    policy, vec, n_steps=64, batch_size=64, gamma=1.0, seed=0, policy_kwargs=policy_kwargs
)
model.learn(2_000)  # a real run uses far more steps

seeds = scenario_seeds("small-cvrp", "test", 8)
agent = evaluate(SMALL, SB3Policy(model, name="ppo-gat"), seeds)
baseline = evaluate(SMALL, NearestNeighbor, seeds)
card = report_card(agent, reference=baseline, training={"steps": 2_000})
gap = card["reference"]["gap"]
print(
    f"agent {agent.summary.mean_cost:.1f} vs nearest {baseline.summary.mean_cost:.1f}: {gap:+.1%}"
)
