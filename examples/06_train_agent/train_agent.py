# %% [markdown]
# # Training an agent
#
# MaskablePPO with a graph attention policy on a small CVRP, evaluated against nearest neighbor on
# fixed test scenarios. Needs `fleetgym[sb3]`.

# %%
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from fleetgym.agents import SB3Policy, make_sb3_vec, policy_settings
from fleetgym.baselines import NearestNeighbor
from fleetgym.data import Instance, write_instance_set
from fleetgym.eval import evaluate, report_card, scenario_seeds

# %% [markdown]
# Training data is yours to provide: this function draws small random CVRP instances (8 customers,
# capacity 20). Training and test instances are kept apart.


# %%
def make_instances(n_instances: int, seed: int) -> list[Instance]:
    """8 customers uniform in a 100 x 100 square, demands 1..9, capacity 20."""
    rng = np.random.default_rng(seed)
    return [
        Instance(
            name=f"small-{seed}-{k}",
            source="examples/06",
            demands=np.concatenate([[0.0], rng.integers(1, 10, size=8)]),
            coords=np.vstack([[50.0, 50.0], rng.uniform(0, 100, size=(8, 2))]),
            vehicle_types=({"capacity": 20.0},),
        )
        for k in range(n_instances)
    ]


train = write_instance_set(
    make_instances(200, seed=0), Path("instances/train"), name="train", overwrite=True
)
test = write_instance_set(
    make_instances(20, seed=1), Path("instances/test"), name="test", overwrite=True
)
SMALL = {"extends": "CVRP-v1", "instance_source": {"type": "files", "paths": [str(train.root)]}}
TEST = {"extends": "CVRP-v1", "instance_source": {"type": "files", "paths": [str(test.root)]}}

# %% [markdown]
# Train. A real run uses far more steps than this demonstration.

# %%
vec = make_sb3_vec(SMALL, n_envs=4, seed=0)
policy, policy_kwargs = policy_settings("maskable_ppo", "gat", embed_dim=32)
model = MaskablePPO(
    policy, vec, n_steps=64, batch_size=64, gamma=1.0, seed=0, policy_kwargs=policy_kwargs
)
model.learn(2_000)

# %% [markdown]
# Evaluate the trained policy and the baseline on the same test scenarios, on test instances.

# %%
seeds = scenario_seeds("small-cvrp", "test", 8)
agent = evaluate(TEST, SB3Policy(model, name="ppo-gat"), seeds)
baseline = evaluate(TEST, NearestNeighbor, seeds)
card = report_card(agent, reference=baseline, training={"steps": 2_000})
gap = card["reference"]["gap"]
print(
    f"agent {agent.summary.mean_cost:.1f} vs nearest {baseline.summary.mean_cost:.1f}: {gap:+.1%}"
)
