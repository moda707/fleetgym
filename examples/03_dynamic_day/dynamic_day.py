# %% [markdown]
# # Requests arriving during the day
#
# `DVRP-v1`: requests arrive during the day and are accepted or rejected; accepted requests are
# inserted where they are cheapest. The instance gives the arrival process, the distributions of a
# request's location, demand and time window, and the cutoff. The second part routes the vehicles
# with `next_node` instead, including a wait action.

# %%
from pathlib import Path

import numpy as np

import fleetgym as fg
from fleetgym.baselines import CheapestInsertion, NearestNeighbor, run_episode
from fleetgym.data import Instance, write_instance_set
from fleetgym.uncertainty.dynamism import degree_of_dynamism


# %%
def make_instances(n_instances: int, seed: int = 0) -> list[Instance]:
    """A 480-minute day, 4 vehicles of capacity 100; requests arrive faster in the late morning."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_instances):
        coords = np.vstack([[50.0, 50.0], rng.uniform(0, 100, size=(20, 2))])
        out.append(
            Instance(
                name=f"day-{k}",
                source="examples/03",
                demands=np.zeros(21),  # no customers known in advance: only the area's sites
                coords=coords,
                time_windows=np.tile([0.0, 480.0], (21, 1)),
                service_times=np.concatenate([[0.0], np.full(20, 10.0)]),
                vehicle_types=({"capacity": 100.0, "count": 4},),
                extra={"cutoff": 360.0},
                distributions={
                    "arrivals": {
                        "process": "nhpp",
                        "profile": {"breaks": [0, 120, 300, 480], "rates": [0.2, 0.5, 0.1]},
                    },
                    "request.location": {"dist": "kde", "data": "customer_coords"},
                    "request.demand": {
                        "dist": "categorical",
                        "values": [5, 10, 20],
                        "probs": [0.5, 0.3, 0.2],
                    },
                    "request.time_window_width": {"dist": "uniform", "low": 60, "high": 180},
                    "cancellation": {"dist": "bernoulli", "p": 0.02},
                },
            )
        )
    return out


data = write_instance_set(
    make_instances(3), Path("instances") / "dvrp", name="dvrp", overwrite=True
)
source = {"type": "files", "paths": [str(data.root)]}

# %% [markdown]
# ## Accept or reject, with cheapest-insertion routing
#
# The instance's sites only feed the location distribution (`initial_requests: none`).

# %%
spec = fg.load_spec(
    {
        "extends": "DVRP-v1",
        "instance_source": source,
        "dynamics": {"max_requests": 250, "initial_requests": "none"},
    }
)
env = fg.make(spec)
episode = run_episode(env, CheapestInsertion(), seed=2)
log = env.event_log
print(
    f"requests: {len(log.of_kind('request'))}, accepted: {len(log.of_kind('accept'))}, "
    f"rejected: {len(log.of_kind('reject'))}, cost: {episode.cost:.1f}"
)
release = env.scenario["request.release"]
print("degree of dynamism:", round(degree_of_dynamism(release, 0.0), 2))

# %% [markdown]
# ## Routing with `next_node`
#
# Requests are accepted automatically until the cutoff (here the last 20% of the day takes no new
# requests). Each vehicle picks its next customer, waits for the next event, reloads at the depot,
# or ends its route.

# %%
spec = fg.load_spec(
    {
        "extends": "DVRP-v1",
        "instance_source": source,
        "dynamics": {"max_requests": 250, "initial_requests": "none", "cutoff": 384},
        "decisions": {"epochs": "on_event", "interface": "next_node"},
        "objective": {"distance": 1.0, "unserved": 50.0},
    }
)
env = fg.make(spec)
obs, info = env.reset(seed=2)
print("actions:", env.action_space, "(the last one is wait)")
episode = run_episode(env, NearestNeighbor(), seed=2)
waits = sum(1 for row in env.event_log.of_kind("decision") if "node" not in row)
print(f"cost: {episode.cost:.1f}, waits: {waits}, unserved: {episode.costs.get('unserved', 0):.0f}")
assert np.isfinite(episode.cost)
