"""Render modes ``rgb_array`` and ``human`` for `FleetGymEnv` (extra ``viz``).

The environment binds these functions by name when a render mode needs them (it never imports this
layer). The figure shows the map and the vehicle Gantt chart of the episode so far.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from fleetgym.viz.gantt import draw_gantt
from fleetgym.viz.map import draw_map

__all__ = ["episode_figure", "render_human", "render_rgb_array"]


def episode_figure(env: Any, *, width: float = 9.0, height: float = 4.0) -> Figure:
    """Map and Gantt chart of ``env``'s current episode."""
    figure = Figure(figsize=(width, height), dpi=100)
    left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.3]})
    sim = env.simulator
    name = env.problem_spec.id or "episode"
    draw_map(
        left,
        sim.log,
        sim.problem.instance,
        t=sim.state.time,
        title=f"{name}, t = {sim.state.time:.1f}",
    )
    draw_gantt(right, sim.log)
    figure.tight_layout()
    return figure


def render_rgb_array(env: Any) -> np.ndarray:
    """The current figure as an ``(H, W, 3)`` ``uint8`` array."""
    figure = episode_figure(env)
    canvas = FigureCanvasAgg(figure)  # off-screen, whatever the active backend
    canvas.draw()
    return np.array(np.asarray(canvas.buffer_rgba())[:, :, :3], dtype=np.uint8)


_window: dict[str, Any] = {}


def render_human(env: Any) -> None:
    """Show the current figure in a window (nothing is shown with a non-interactive backend)."""
    import matplotlib.pyplot as plt

    if "figure" not in _window or not plt.fignum_exists(_window["figure"].number):
        _window["figure"] = plt.figure(figsize=(9.0, 4.0))
    fig = _window["figure"]
    fig.clf()
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.3]})
    sim = env.simulator
    draw_map(left, sim.log, sim.problem.instance, t=sim.state.time)
    draw_gantt(right, sim.log)
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo", "template"):
        plt.pause(0.001)
