# %% [markdown]
# # Build your own VRP variant
#
# This notebook builds a new variant step by step, from a YAML change to new code, and checks it the
# way FleetGym checks its own environments. The running example is **rural parcel collection**:
#
# * customers cluster around a few villages, and the layout changes every day;
# * the parcel volume of a customer is uncertain until the van arrives;
# * a van stops at most 5 times per route (a driver rule);
# * if a parcel does not fit, the van goes back to restock when the depot is near, otherwise a
#   courier takes the rest at a price;
# * six vans of capacity 15; the objective is distance plus courier costs (and a large penalty for
#   customers left unserved).
#
# Steps: (1) variants from YAML alone; (2) the four building blocks in `plugins.py`; (3) test each
# block on its own; (4) assemble them in `my_variant.yaml`; (5) run and verify; (6) a policy of
# your own; (7) a fair comparison; (8) share it as a plugin.

# %%
import inspect
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))

import fleetgym as fg  # noqa: E402
from fleetgym.baselines import NearestNeighbor, run_episode  # noqa: E402
from fleetgym.baselines.protocol import PolicyBase, valid_actions  # noqa: E402
from fleetgym.errors import ConfigError  # noqa: E402
from fleetgym.eval import compare, evaluate, scenario_seeds  # noqa: E402
from fleetgym.testing.non_anticipativity import check_non_anticipativity  # noqa: E402
from fleetgym.testing.validator import validate  # noqa: E402

# %% [markdown]
# ## 1. Variants from YAML alone
#
# Most variants need no code: start from the closest registered environment and change what
# differs. Every field has a default you can see with `spec.to_yaml()` (or `fleetgym show`).

# %%
small_vans = fg.load_spec(
    "VRPSD-v0",
    overrides=[
        "fleet.sizing=fixed",
        "fleet.types.0.capacity=15",
        "fleet.types.0.count=6",
        "decisions.epochs=on_event",
        "objective.unserved=200",
    ],
)
print("hash:", small_vans.content_hash()[:20])
print(run_episode(fg.make(small_vans), NearestNeighbor(), seed=0).costs)

# %% [markdown]
# A dict works too, with `extends` and only the changes. Mistakes are caught when the spec is
# loaded, and the message names the key path:

# %%
try:
    fg.load_spec({"extends": "VRPSD-v0", "recourse": {"capacity_failure": "teleport"}})
except (ConfigError, ValueError) as error:
    print(str(error).splitlines()[0])
    print("...", next(line for line in str(error).splitlines() if "teleport" in line).strip())

# %% [markdown]
# ## 2. The building blocks
#
# Four things the YAML cannot express, each written once in `plugins.py` and registered under a
# name. Importing the module registers them.

# %%
import plugins  # noqa: E402

for block in (
    plugins.villages,
    plugins.TriangularDemand,
    plugins.MaxStops,
    plugins.NearDepotRestock,
):
    print(inspect.getsource(block))

# %% [markdown]
# * **Generator** (`fg.register_generator`): a function of a seeded random generator returning an
#   `Instance`. Draw every random number from the generator you are given, so the same seed gives
#   the same instance.
# * **Distribution** (`fg.register_distribution`): `sample`, `mean` (the forecast the agent sees
#   before the value is revealed) and optionally `quantile` and `support`.
# * **Constraint** (`fg.register_constraint`): a mask over candidate customers, computed from known
#   information only.
# * **Recourse rule** (`fg.register_recourse`): what to do when a revealed demand does not fit.
#
# ## 3. Test each block on its own

# %%
from fleetgym.data import generate  # noqa: E402
from fleetgym.uncertainty.distributions import build_distribution  # noqa: E402

village_map = generate("villages", {"n_customers": 24}, np.random.default_rng(0))
again = generate("villages", {"n_customers": 24}, np.random.default_rng(0))
assert village_map.content_hash() == again.content_hash()  # same seed, same instance
print(village_map.n_customers, "customers; demands", village_map.demands[1:].astype(int))

demand = build_distribution("triangular_demand", {"low": 0, "mode": 4.0, "high": 12})
draws = demand.sample(np.random.default_rng(1), 20_000)
print(f"mean of draws {draws.mean():.2f}, forecast {demand.mean(1)[0]:.2f}")
assert abs(draws.mean() - demand.mean(1)[0]) < 0.1

# %% [markdown]
# ## 4. Assemble the variant in YAML

# %%
print((HERE / "my_variant.yaml").read_text())

# %%
spec = fg.load_spec(HERE / "my_variant.yaml")
print("spec hash:", spec.content_hash()[:20])

# %% [markdown]
# ## 5. Run it and verify it
#
# Run a baseline. The custom rule restocks near the depot and outsources far from it.

# %%
env = fg.make(spec)
episode = run_episode(env, NearestNeighbor(), seed=0)
print(f"cost {episode.cost:.1f}: {episode.costs}")
log = env.event_log
print("failures resolved:", [row["component"] for row in log.of_kind("failure")])

# %% [markdown]
# Three checks every variant should pass:
#
# * the **independent validator** re-checks the physics of the episode from its log (loads,
#   travel, restock trips, outsourcing costs, revealed values), custom rule included;
# * **non-anticipativity**: no observation, mask or policy decision may depend on a value that is
#   not revealed yet;
# * the **custom constraint** holds on every route (the validator does not know custom
#   constraints, so test them yourself).

# %%
report = validate(log, env.problem_spec, env.base_instance, scenario=env.scenario)
print("valid:", report.ok)
print("non-anticipative epochs checked:", check_non_anticipativity(fg.make(spec), seed=1))
customers: dict[int, set[int]] = {}  # a return after a restock is not a new stop
for row in log.of_kind("depart"):
    if row["node"] != 0:
        customers.setdefault(row["vehicle"], set()).add(row["node"])
print("customers per van:", {van: len(nodes) for van, nodes in customers.items()})
assert max(len(nodes) for nodes in customers.values()) <= 5

# %% [markdown]
# ## 6. A policy of your own
#
# A policy implements `reset(env)` and `act(obs, info)`. This one prefers the nearest customer
# whose *expected* demand still fits in the van, so it causes fewer failures than plain nearest
# neighbor. It reads only known information (the forecast, not the hidden demand).

# %%


class NearestThatFits(PolicyBase):
    """The nearest valid customer whose expected demand fits; else the nearest valid action."""

    name = "nearest_that_fits"

    def act(self, obs, info):
        """Choose the next node for the vehicle that decides."""
        sim = self.env.simulator
        valid = valid_actions(info)
        epoch = self.env.epoch
        if epoch is None or epoch.kind != "vehicle_idle":
            return int(valid[0])
        vehicle = sim.state.vehicles[epoch.vehicle]
        here = vehicle.node
        known = sim.info
        customers = valid[valid != 0]
        if customers.size == 0:
            return int(valid[0])
        free = vehicle.capacity - vehicle.load
        fits = customers[known.open_demands[customers] <= free]
        pool = fits if fits.size else customers
        return int(pool[np.argmin(known.distances[here, pool])])


print(run_episode(env, NearestThatFits(), seed=0).costs)

# %% [markdown]
# ## 7. A fair comparison
#
# Compare policies on the same scenarios (common random numbers), never on the seeds you tuned
# on. `evaluate` validates every episode; `compare` runs paired tests.

# %%
seeds = scenario_seeds("MyVariant-v0", "test", 20)
nearest = evaluate(spec, NearestNeighbor, seeds)
mine = evaluate(spec, NearestThatFits, seeds)
for result in (nearest, mine):
    s = result.summary
    print(f"{result.policy:>18}: {s.mean_cost:7.1f}  [{s.ci_low:.1f}, {s.ci_high:.1f}]")
paired = compare(mine.costs, nearest.costs)
print(f"difference {paired.mean_difference:.1f}, paired t-test p = {paired.t_pvalue:.3f}")

# %% [markdown]
# Read the p-value before the means: if it is large, the policies cannot be told apart on these
# scenarios (here the forecast-aware rule rarely changes a decision), and a claim of improvement
# would need a better policy or more scenarios.

# %% [markdown]
# ## 8. Share it
#
# Put `plugins.py` in a package and expose a function that imports it under the `fleetgym.plugins`
# entry point; the blocks then load automatically when a config names them:
#
# ```toml
# [project.entry-points."fleetgym.plugins"]
# rural_parcels = "rural_parcels.plugins:register"
# ```
#
# Checklist: the spec validates and its hash is reported with your results; episodes pass the
# validator; non-anticipativity holds; custom constraints have their own tests; and changing a
# published variant's dynamics or defaults gets a new `-vN`.
