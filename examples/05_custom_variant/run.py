"""Build and run a custom VRP variant: register plugins, load the YAML, run, validate.

Run: python examples/05_custom_variant/run.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import plugins  # noqa: E402,F401  (registers the custom distribution, constraint and rule)

import fleetgym as fg  # noqa: E402
from fleetgym.baselines import NearestNeighbor, run_episode  # noqa: E402
from fleetgym.testing.non_anticipativity import check_non_anticipativity  # noqa: E402
from fleetgym.testing.validator import validate  # noqa: E402

spec = fg.load_spec(HERE / "my_variant.yaml")  # errors name the file and the key path
print("spec hash:", spec.content_hash()[:20])

env = fg.make(spec)
episode = run_episode(env, NearestNeighbor(), seed=0)
print(f"cost {episode.cost:.1f}: {episode.costs}")
log = env.event_log
print("failures resolved:", [row["component"] for row in log.of_kind("failure")])

# The independent validator re-checks the physics of any episode, custom rule included.
report = validate(log, env.problem_spec, env.base_instance, scenario=env.scenario)
print("valid:", report.ok)
# And no observation may reveal a hidden demand.
print("non-anticipative epochs checked:", check_non_anticipativity(fg.make(spec), seed=1))
