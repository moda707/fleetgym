"""Render an episode: the map and the vehicle Gantt chart (extra `viz`).

Run: python examples/08_render.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import fleetgym as fg
from fleetgym.baselines import NearestNeighbor, run_episode
from fleetgym.viz import episode_figure

env = fg.make("DVRP-StochReq-v0", render_mode="rgb_array")
run_episode(env, NearestNeighbor(), seed=3)
frame = env.render()  # an (H, W, 3) array, for videos
print("frame:", frame.shape)
out = Path("runs") / "dvrp_day.png"
out.parent.mkdir(exist_ok=True)
episode_figure(env).savefig(out)
print("saved", out)
