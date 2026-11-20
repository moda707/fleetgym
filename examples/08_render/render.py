# %% [markdown]
# # Rendering an episode
#
# The map and the vehicle Gantt chart, drawn from the event log. Needs `fleetgym[viz]`.

# %%
from pathlib import Path

import fleetgym as fg
from fleetgym.baselines import NearestNeighbor, run_episode
from fleetgym.viz import episode_figure

# %% [markdown]
# Play an episode, take a frame (for videos), and save the figure.

# %%
env = fg.make("DVRP-StochReq-v0", render_mode="rgb_array")
run_episode(env, NearestNeighbor(), seed=3)
frame = env.render()
print("frame:", frame.shape)
out = Path("runs") / "dvrp_day.png"
out.parent.mkdir(exist_ok=True)
figure = episode_figure(env)
figure.savefig(out)
print("saved", out)
figure  # noqa: B018  (a notebook shows the figure here)
