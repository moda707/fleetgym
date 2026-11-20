# %% [markdown]
# # Build your own VRP variant
#
# `my_variant.yaml` describes a new variant: stochastic demands from a custom distribution, a limit
# on stops per route (a custom constraint) and a custom recourse rule. `plugins.py` defines the
# three building blocks; importing it registers them by name.

# %%
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))

import plugins  # noqa: E402,F401  (registers the custom distribution, constraint and rule)

import fleetgym as fg  # noqa: E402
from fleetgym.baselines import NearestNeighbor, run_episode  # noqa: E402
from fleetgym.testing.non_anticipativity import check_non_anticipativity  # noqa: E402
from fleetgym.testing.validator import validate  # noqa: E402

# %% [markdown]
# Load the variant. Config errors name the file and the key path.

# %%
spec = fg.load_spec(HERE / "my_variant.yaml")
print("spec hash:", spec.content_hash()[:20])

# %% [markdown]
# Run a baseline. The custom rule restocks near the depot and outsources far from it.

# %%
env = fg.make(spec)
episode = run_episode(env, NearestNeighbor(), seed=0)
print(f"cost {episode.cost:.1f}: {episode.costs}")
log = env.event_log
print("failures resolved:", [row["component"] for row in log.of_kind("failure")])

# %% [markdown]
# The independent validator re-checks the physics of the episode, custom rule included, and no
# observation may reveal a hidden demand.

# %%
report = validate(log, env.problem_spec, env.base_instance, scenario=env.scenario)
print("valid:", report.ok)
print("non-anticipative epochs checked:", check_non_anticipativity(fg.make(spec), seed=1))
