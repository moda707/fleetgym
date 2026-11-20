"""Vehicle Gantt view from the event log (design §21): travel, waiting, service and reloads."""

from __future__ import annotations

from matplotlib.axes import Axes

from fleetgym.eventlog import EventLog

__all__ = ["GANTT_COLORS", "draw_gantt", "vehicle_segments"]

GANTT_COLORS = {
    "travel": "#4c78a8",
    "waiting": "#f2cf5b",
    "service": "#2a9d55",
    "restock": "#9d755d",
}


def vehicle_segments(log: EventLog) -> dict[int, list[tuple[str, float, float]]]:
    """Per vehicle, ``(kind, start, end)`` segments (travel, waiting, service; restocks: points)."""
    segments: dict[int, list[tuple[str, float, float]]] = {}
    open_travel: dict[int, float] = {}
    arrived: dict[int, float] = {}
    serving: dict[int, float] = {}
    for row in log.rows:
        kind = row["kind"]
        vehicle = row.get("vehicle")
        if vehicle is None or kind not in (
            "depart",
            "arrive",
            "service_start",
            "service_end",
            "restock",
        ):
            continue
        vehicle = int(vehicle)
        out = segments.setdefault(vehicle, [])
        time = float(row["time"])
        if kind == "depart":
            open_travel[vehicle] = time
        elif kind == "arrive":
            out.append(("travel", open_travel.pop(vehicle, time), time))
            arrived[vehicle] = time
        elif kind == "service_start":
            start = arrived.pop(vehicle, time)
            if time > start:
                out.append(("waiting", start, time))
            serving[vehicle] = time
        elif kind == "service_end":
            out.append(("service", serving.pop(vehicle, time), time))
        elif kind == "restock":
            out.append(("restock", time, time))
    return segments


def draw_gantt(ax: Axes, log: EventLog, *, title: str = "Vehicles") -> Axes:
    """Draw one row per vehicle."""
    segments = vehicle_segments(log)
    for row, vehicle in enumerate(sorted(segments)):
        for kind, start, end in segments[vehicle]:
            if kind == "restock":
                ax.plot([start], [row], marker="|", ms=12, color=GANTT_COLORS[kind])
            else:
                ax.barh(
                    row, max(end - start, 1e-9), left=start, height=0.6, color=GANTT_COLORS[kind]
                )
    from matplotlib.patches import Patch

    kinds = sorted({kind for parts in segments.values() for kind, _, _ in parts} - {"restock"})
    if kinds:
        ax.legend(
            handles=[Patch(color=GANTT_COLORS[k], label=k) for k in kinds],
            fontsize=7,
            loc="lower right",
            framealpha=0.8,
        )
    ax.set_yticks(range(len(segments)), [f"v{v}" for v in sorted(segments)])
    ax.set_xlabel("time")
    ax.set_title(title, fontsize=10)
    return ax
