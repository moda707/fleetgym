# %% [markdown]
# # Comparing policies
#
# Fixed test scenarios, validated episodes, confidence intervals, CVaR, a paired test and a JSON
# report card, on `SVRP-v1` instances written by this notebook.

# %%
from pathlib import Path

import numpy as np

import fleetgym as fg
from fleetgym.baselines import CheapestInsertion, NearestNeighbor
from fleetgym.data import Instance, write_instance_set
from fleetgym.eval import compare, evaluate, report_card, scenario_seeds, write_report_card


# %%
def make_instances(n_instances: int, seed: int = 0) -> list[Instance]:
    """20 uniform customers, Poisson demands around 1..9, capacity 50."""
    rng = np.random.default_rng(seed)
    return [
        Instance(
            name=f"svrp-{k}",
            source="examples/04",
            demands=np.concatenate([[0.0], rng.integers(1, 10, size=20)]),
            coords=np.vstack([[50.0, 50.0], rng.uniform(0, 100, size=(20, 2))]),
            vehicle_types=({"capacity": 50.0},),
            distributions={"demand": {"dist": "poisson", "mean": "demand"}},
        )
        for k in range(n_instances)
    ]


data = write_instance_set(
    make_instances(8), Path("instances") / "svrp", name="svrp", overwrite=True
)
spec = fg.load_spec(
    {"extends": "SVRP-v1", "instance_source": {"type": "files", "paths": [str(data.root)]}}
)

# %% [markdown]
# Test seeds are derived from a name: every policy faces the same scenarios, and they are never
# used for training. `evaluate` validates every episode.

# %%
seeds = scenario_seeds("svrp-example", "test", 16)
nearest = evaluate(spec, NearestNeighbor, seeds)
insertion = evaluate(spec, CheapestInsertion, seeds)
for result in (nearest, insertion):
    s = result.summary
    interval = f"[{s.ci_low:.1f}, {s.ci_high:.1f}]"
    print(f"{result.policy:>20}: {s.mean_cost:8.1f}  {interval}  CVaR90 {s.cvar90:.1f}")

# %% [markdown]
# Paired comparison on the same scenarios, and a report card for the record.

# %%
paired = compare(insertion.costs, nearest.costs)
print(f"insertion - nearest: {paired.mean_difference:.1f} (t-test p = {paired.t_pvalue:.3f})")
card = report_card(insertion, reference=nearest)
path = write_report_card(card, Path("runs") / "example_report.json")
print("report card:", path)
