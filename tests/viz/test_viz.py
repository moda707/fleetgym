"""Visualization: headless rendering of every view, render modes, CLI (no pixel comparisons)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import fleetgym as fg  # noqa: E402
from fleetgym.baselines import NearestNeighbor, run_episode  # noqa: E402
from fleetgym.cli.__main__ import main  # noqa: E402
from fleetgym.data import read_instance  # noqa: E402
from fleetgym.errors import ConfigError  # noqa: E402
from fleetgym.viz import (  # noqa: E402
    customer_status,
    draw_gantt,
    draw_map,
    episode_figure,
    vehicle_segments,
)
from tests.registered import registered  # noqa: E402

pytestmark = pytest.mark.extras_viz
SINGLE = registered("single")
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "instances"


@pytest.mark.parametrize("env_id", SINGLE)
def test_every_view_renders_for_every_config(env_id):
    env = fg.make(env_id, render_mode="rgb_array")
    run_episode(env, NearestNeighbor(), seed=2)
    image = env.render()
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
    assert image.std() > 0  # not a blank canvas
    figure = episode_figure(env)
    assert len(figure.axes) == 2
    sim = env.simulator
    statuses = customer_status(
        sim.log, sim.problem.instance.n_nodes, sim.problem.instance.n_depots, sim.state.time
    )
    assert statuses and set(statuses.values()) <= {
        "served",
        "unserved",
        "rejected",
        "outsourced",
        "cancelled",
        "pending",
    }
    assert vehicle_segments(sim.log)


def test_rendering_does_not_change_the_trajectory():
    digests = []
    for mode in (None, "rgb_array", "ansi"):
        env = fg.make("VRPSD-v0", render_mode=mode)
        obs, info = env.reset(seed=5)
        policy = NearestNeighbor()
        policy.reset(env)
        while True:
            if mode is not None:
                env.render()
            obs, _, terminated, _, info = env.step(policy.act(obs, info))
            if terminated:
                break
        digests.append(env.event_log.digest())
    assert len(set(digests)) == 1


def test_human_mode_and_partial_episodes():
    env = fg.make("DVRP-StochReq-v0", render_mode="human")
    _, info = env.reset(seed=1)
    assert env.render() is None
    env.step(int(np.flatnonzero(info["action_mask"])[0]))
    assert env.render() is None
    assert "human" in fg.FleetGymEnv.metadata["render_modes"]


def test_views_on_axes_and_errors():
    import matplotlib.pyplot as plt

    env = fg.make("CVRP-v0")
    run_episode(env, NearestNeighbor(), seed=0)
    fig, (a, b) = plt.subplots(1, 2)
    draw_map(a, env.event_log, env.problem.instance, t=0.0)
    draw_gantt(b, env.event_log)
    plt.close(fig)
    matrix = read_instance(FIXTURES / "tiny_explicit.vrp")
    fig, ax = plt.subplots()
    with pytest.raises(ConfigError, match="coordinates"):
        draw_map(ax, env.event_log, matrix)
    plt.close(fig)


def test_cli_render(tmp_path: Path, capsys):
    out = tmp_path / "figs" / "episode.png"
    assert (
        main(["render", "VRPTW-v0", "--policy", "insertion", "--seed", "1", "--out", str(out)]) == 0
    )
    assert out.exists() and out.stat().st_size > 1000
    assert "saved" in capsys.readouterr().out
