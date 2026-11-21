# %% [markdown]
# # Build your own VRP variant
#
# This notebook builds a new variant step by step and checks it the way FleetGym checks its own
# environments. The running example is **rural parcel collection**:
#
# * customers cluster around a few villages, and the layout changes every day;
# * the parcel volume of a customer is uncertain until the van arrives;
# * a van makes at most a given number of stops per route (a driver rule);
# * if a parcel does not fit, the van goes back to restock when the depot is near, otherwise a
#   courier takes the rest at a price;
# * the objective is distance plus courier costs (and a large penalty for unserved customers).
#
# FleetGym separates the **model** (what exists and how it behaves, in YAML) from the
# **instances** (every number, in files that follow an instance schema). Steps: (1) your instance
# schema and instances; (2) the building blocks in `plugins.py`; (3) the model in YAML; (4) run and
# verify; (5) a policy of your own; (6) a fair comparison; (7) share it.

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
from fleetgym.data import read_instance  # noqa: E402
from fleetgym.errors import ConfigError  # noqa: E402
from fleetgym.eval import compare, evaluate, scenario_seeds  # noqa: E402
from fleetgym.testing.non_anticipativity import check_non_anticipativity  # noqa: E402
from fleetgym.testing.validator import validate  # noqa: E402

# %% [markdown]
# ## 1. Your instance schema and instances
#
# The schema extends FleetGym's core fields (coordinates, demands, vehicle types, ...) with the
# numbers this variant needs. Field types, bounds and units are checked whenever an instance is
# built or read.

# %%
print((HERE / "rural-parcels.schema.yaml").read_text())

# %% [markdown]
# `instances.py` draws instances in that schema, including the distribution of the uncertain
# demand (it names the instance's own `demand` field as its mode), and writes them as an instance
# set: files plus a manifest with checksums and the seed that produced them.

# %%
import instances  # noqa: E402

print(inspect.getsource(instances.village_instance))
data = instances.write_villages(HERE / "instances" / "rural", n_instances=10, seed=0)
print(len(data), "instances; schema", data.schema)
first = read_instance(data.paths[0])
print("globals:", {k: first.extra[k] for k in ("max_stops", "restock_radius", "courier_price")})

# %% [markdown]
# ## 2. The building blocks
#
# Three things the YAML cannot express, each written once in `plugins.py` and registered under a
# name. Importing the module registers them. They read their numbers from the instance
# (`self.problem.value(...)`), never from code.

# %%
import plugins  # noqa: E402

for block in (plugins.TriangularDemand, plugins.MaxStops, plugins.NearDepotRestock):
    print(inspect.getsource(block))

# %% [markdown]
# * **Distribution** (`fg.register_distribution`): `sample`, `mean` (the forecast the agent sees
#   before the value is revealed) and optionally `quantile` and `support`.
# * **Constraint** (`fg.register_constraint`): a mask over candidate customers, computed from known
#   information only.
# * **Recourse rule** (`fg.register_recourse`): what to do when a revealed demand does not fit.
#
# Test each block on its own before using it:

# %%
from fleetgym.uncertainty.distributions import build_distribution  # noqa: E402

demand = build_distribution("triangular_demand", {"low": 0, "mode": 4.0, "high": 12})
draws = demand.sample(np.random.default_rng(1), 20_000)
print(f"mean of draws {draws.mean():.2f}, forecast {demand.mean(1)[0]:.2f}")
assert abs(draws.mean() - demand.mean(1)[0]) < 0.1

# %% [markdown]
# ## 3. The model in YAML
#
# The model extends the closest family (`SVRP-v1`, stochastic demands) and says what exists. Every
# number refers to an instance field (`{from_instance: max_stops}`).

# %%
print((HERE / "my_variant.yaml").read_text())

# %%
spec = fg.load_spec(HERE / "my_variant.yaml")
print("spec hash:", spec.content_hash()[:20])

# %% [markdown]
# Mistakes are caught when the spec is loaded, and the message names the key path:

# %%
try:
    fg.load_spec(
        {"extends": "SVRP-v1", "recourse": {"capacity_failure": {"remainder": "teleport"}}}
    )
except (ConfigError, ValueError) as error:
    print(str(error).splitlines()[0])

# %% [markdown]
# ## 4. Run it and verify it
#
# Run a baseline. The custom rule restocks near the depot and calls the courier far from it.

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
limit = int(env.base_instance.extra["max_stops"])
customers: dict[int, set[int]] = {}  # a return after a restock is not a new stop
for row in log.of_kind("depart"):
    if row["node"] != 0:
        customers.setdefault(row["vehicle"], set()).add(row["node"])
print("customers per van:", {van: len(nodes) for van, nodes in customers.items()}, "limit", limit)
assert max(len(nodes) for nodes in customers.values()) <= limit

# %% [markdown]
# ## 5. A policy of your own
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
        known = sim.info
        candidates = valid[valid != 0]
        if candidates.size == 0:
            return int(valid[0])
        free = vehicle.capacity - vehicle.load
        fits = candidates[known.open_demands[candidates] <= free]
        pool = fits if fits.size else candidates
        return int(pool[np.argmin(known.distances[vehicle.node, pool])])


print(run_episode(env, NearestThatFits(), seed=0).costs)

# %% [markdown]
# ## 6. A fair comparison
#
# Compare policies on the same scenarios (common random numbers), never on the seeds you tuned
# on. `evaluate` validates every episode; `compare` runs paired tests.

# %%
seeds = scenario_seeds("MyVariant-v1", "test", 20)
nearest = evaluate(spec, NearestNeighbor, seeds)
mine = evaluate(spec, NearestThatFits, seeds)
for result in (nearest, mine):
    s = result.summary
    print(f"{result.policy:>18}: {s.mean_cost:7.1f}  [{s.ci_low:.1f}, {s.ci_high:.1f}]")
paired = compare(mine.costs, nearest.costs)
print(f"difference {paired.mean_difference:.1f}, paired t-test p = {paired.t_pvalue:.3f}")

# %% [markdown]
# Read the p-value before the means: if it is large, the policies cannot be told apart on these
# scenarios, and a claim of improvement would need a better policy or more scenarios.
#
# ## 7. Share it
#
# Put the schema, `plugins.py` and `instances.py` in a package and expose a function that imports
# the plugins under the `fleetgym.plugins` entry point; the blocks then load automatically when a
# config names them:
#
# ```toml
# [project.entry-points."fleetgym.plugins"]
# rural_parcels = "rural_parcels.plugins:register"
# ```
#
# Checklist: the spec validates and its hash is reported with your results; the instance set's
# manifest (with its checksums) is published with them; episodes pass the validator;
# non-anticipativity holds; custom constraints have their own tests; and changing a published
# variant's dynamics or defaults gets a new `-vN`.
