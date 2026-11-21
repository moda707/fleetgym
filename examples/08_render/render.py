# %% [markdown]
# # Rendering an episode
#
# The map and the vehicle Gantt chart, drawn from the event log. Needs `fleetgym[viz]`.

# %%
from pathlib import Path

import numpy as np

import fleetgym as fg
from fleetgym.baselines import NearestNeighbor, run_episode
from fleetgym.data import Instance, write_instance
from fleetgym.viz import episode_figure

# %% [markdown]
# One instance with time windows: 30 customers, 3 vehicles of capacity 60, a 480-minute day.


# %%
def make_instance(seed: int = 0) -> Instance:
    """30 uniform customers with 2-hour windows; service 10 minutes; 3 vehicles of capacity 60."""
    rng = np.random.default_rng(seed)
    coords = np.vstack([[50.0, 50.0], rng.uniform(0, 100, size=(30, 2))])
    opens = rng.uniform(0, 300, size=30)
    windows = np.vstack([[0.0, 480.0], np.column_stack([opens, opens + 120.0])])
    return Instance(
        name="render-day",
        source="examples/08",
        demands=np.concatenate([[0.0], rng.integers(1, 10, size=30)]),
        coords=coords,
        time_windows=windows,
        service_times=np.concatenate([[0.0], np.full(30, 10.0)]),
        vehicle_types=({"capacity": 60.0, "count": 3},),
    )


path = write_instance(make_instance(), Path("instances") / "day.json")

# %% [markdown]
# Play an episode, take a frame (for videos), and save the figure.

# %%
spec = fg.load_spec(
    {
        "extends": "VRPTW-v1",
        "instance_source": {"type": "files", "paths": [str(path)]},
        "fleet": {"sizing": "fixed"},
        "constraints": {"time_windows": {"mode": "soft"}},
        "decisions": {"interface": "next_node", "epochs": "on_event"},
        "objective": {"distance": 1.0, "lateness": 1.0, "unserved": 100.0},
    }
)
env = fg.make(spec, render_mode="rgb_array")
run_episode(env, NearestNeighbor(), seed=3)
frame = env.render()
print("frame:", frame.shape)
out = Path("runs") / "day.png"
out.parent.mkdir(exist_ok=True)
figure = episode_figure(env)
figure.savefig(out)
print("saved", out)
figure  # noqa: B018  (a notebook shows the figure here)
