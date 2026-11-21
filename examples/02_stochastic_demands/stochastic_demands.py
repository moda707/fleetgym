# %% [markdown]
# # Stochastic demands with recourse
#
# `SVRP-v1`: each customer's demand is revealed when the vehicle arrives; the instance gives its
# distribution. A vehicle that cannot fit a demand serves what fits, restocks at the depot and
# comes back. This notebook writes instances with Poisson demands, then looks at scenarios,
# forecasts, recourse, exact replay and non-anticipative forks.

# %%
from pathlib import Path

import numpy as np

import fleetgym as fg
from fleetgym.baselines import NearestNeighbor, Rollout, run_episode
from fleetgym.data import Instance, write_instance_set


# %%
def make_instances(n_instances: int, n_customers: int = 20, seed: int = 0) -> list[Instance]:
    """Uniform customers; expected demands 1..9; the actual demand is Poisson around them."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_instances):
        coords = np.vstack([[50.0, 50.0], rng.uniform(0, 100, size=(n_customers, 2))])
        expected = np.concatenate([[0.0], rng.integers(1, 10, size=n_customers)])
        out.append(
            Instance(
                name=f"poisson-{k}",
                source="examples/02",
                demands=expected,
                coords=coords,
                vehicle_types=({"capacity": 50.0},),
                # the distribution belongs to the instance; "demand" names the instance's field
                distributions={"demand": {"dist": "poisson", "mean": "demand"}},
            )
        )
    return out


data = write_instance_set(
    make_instances(5), Path("instances") / "svrp", name="svrp", overwrite=True
)
spec = fg.load_spec(
    {"extends": "SVRP-v1", "instance_source": {"type": "files", "paths": [str(data.root)]}}
)

# %% [markdown]
# Every random value of an episode is sampled once, at reset, into a scenario. Before a visit the
# agent sees the forecast (the mean), never the realized demand.

# %%
env = fg.make(spec)
obs, info = env.reset(seed=4)
scenario = env.scenario
print("scenario id:", scenario.id[:16], "sources:", scenario.sources)
print("known demand flags:", env.simulator.info.demand_known[:6])

# %% [markdown]
# Policies play through one protocol: `reset(env)`, then `act(obs, info)`. Rollout simulates each
# action on forks with a base policy.

# %%
nearest = run_episode(env, NearestNeighbor(), seed=4).cost
rollout = run_episode(env, Rollout(NearestNeighbor, n_samples=2), seed=4).cost
print(f"nearest neighbor: {nearest:.1f}   rollout over forks: {rollout:.1f}")
print("restock trips:", len(env.event_log.of_kind("restock")))

# %% [markdown]
# The same scenario can be replayed exactly: common random numbers for fair comparisons.

# %%
replay = fg.make(spec, instance=env.base_instance)
replay.reset(seed=123, options={"scenario": scenario})
print("replayed scenario:", replay.scenario == scenario)

# %% [markdown]
# Planners must use `fork(seed)`: it keeps what is known and redraws the hidden future.

# %%
obs, info = env.reset(seed=4)
fork = env.fork(seed=1)
print("fork shares the past, not the future:", fork.scenario != env.scenario)
