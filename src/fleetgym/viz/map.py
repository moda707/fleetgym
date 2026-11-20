"""Map view of an episode, drawn from its event log (design §21).

Customers are colored by their status at time ``t`` (pending, served, unserved, rejected or
outsourced, cancelled; requests not released yet are not drawn), routes are drawn per vehicle from
the ``depart``/``arrive`` rows up to ``t``, and vehicles are marked where they are at ``t``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes

from fleetgym.data.instance import Instance
from fleetgym.errors import ConfigError
from fleetgym.eventlog import EventLog

__all__ = ["STATUS_COLORS", "customer_status", "draw_map"]

STATUS_COLORS = {
    "pending": "#8c8c8c",
    "served": "#2a9d55",
    "unserved": "#d1495b",
    "rejected": "#e09f3e",
    "outsourced": "#7b61c4",
    "cancelled": "#bdbdbd",
}


def customer_status(log: EventLog, n_nodes: int, n_depots: int, t: float) -> dict[int, str]:
    """Status of every released customer at time ``t``."""
    status: dict[int, str] = {}
    known_at_start = not log.of_kind("request")
    if known_at_start:
        status = dict.fromkeys(range(n_depots, n_nodes), "pending")
    for row in log.rows:
        time = row.get("time")
        if time is not None and time > t:
            break
        kind, node = row["kind"], row.get("node")
        if node is None or node < n_depots:
            continue
        if kind == "request":
            status[node] = "pending"
        elif kind == "service_start":
            status[node] = "served"
        elif kind == "reject":
            status[node] = "rejected"
        elif kind == "outsource":
            status[node] = "outsourced"
        elif kind == "cancel":
            status[node] = "cancelled"
        elif kind == "unserved":
            status[node] = "unserved"
    return status


def draw_map(
    ax: Axes, log: EventLog, instance: Instance, *, t: float | None = None, title: str = ""
) -> Axes:
    """Draw the map of ``log`` on ``instance`` (the episode instance) at time ``t`` (default: end).

    Raises
    ------
    ConfigError
        If the instance has no coordinates (matrix-only instances have no map).
    """
    if instance.coords is None:
        raise ConfigError(f"{instance.name}: the map view needs node coordinates")
    coords = np.asarray(instance.coords)
    end = t if t is not None else max((r.get("time") or 0.0) for r in log.rows)
    status = customer_status(log, instance.n_nodes, instance.n_depots, end)
    colors: list[str] = []
    nodes: list[int] = []
    for node, state in sorted(status.items()):
        nodes.append(node)
        colors.append(STATUS_COLORS[state])
    if nodes:
        ax.scatter(coords[nodes, 0], coords[nodes, 1], c=colors, s=28, zorder=3, linewidths=0)
    depots = coords[: instance.n_depots]
    ax.scatter(depots[:, 0], depots[:, 1], marker="s", s=70, c="#1f1f1f", zorder=4)
    cmap = _vehicle_colors()
    position: dict[int, Any] = {}
    for row in log.rows:
        time = row.get("time")
        if time is not None and time > end:
            break
        if row["kind"] == "depart":
            vehicle = int(row["vehicle"])
            a, b = coords[row["from_node"]], coords[row["node"]]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=cmap(vehicle), lw=1.4, alpha=0.85, zorder=2)
            position[vehicle] = b
        elif row["kind"] == "vehicle_activate":
            position[int(row["vehicle"])] = coords[int(row["node"])]
    for vehicle, (x, y) in position.items():
        ax.scatter([x], [y], marker="^", s=55, color=cmap(vehicle), edgecolors="black", zorder=5)
    from matplotlib.lines import Line2D

    shown = sorted(set(status.values()))
    handles = [
        Line2D([], [], marker="o", ls="", color=STATUS_COLORS[name], label=name) for name in shown
    ]
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or f"t = {end:.1f}", fontsize=10)
    return ax


def _vehicle_colors() -> Any:
    import matplotlib

    palette = matplotlib.colormaps["tab10"]
    return lambda i: palette(i % 10)
