"""The ``ev.shortest_path`` section: the EV shortest path with charging-station occupancy (M14).

Dastpak, Errico, Jabali and Malucelli (arXiv 2305.11773): one EV drives from an origin to a
destination over public charging stations, each an M/M/1/kappa queue of exogenous EVs whose binary
occupancy indicators the EV sees in real time (proposal 0003, ADR 0026). Other ``ev`` content (EV
routing with several customers) stays reserved.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, PositiveFloat, PositiveInt

from fleetgym.spec.models.base import FrozenModel

__all__ = ["EVQueueParams", "EVShortestPathParams", "ev_shortest_path"]


class EVQueueParams(FrozenModel):
    """Exogenous EVs at the stations.

    Attributes
    ----------
    model
        ``mm1k``: one charger, Poisson arrivals (rate ``lambda``), exponential charging (rate
        ``mu``), at most ``kappa`` EVs at the station; exogenous EVs leave a full station.
    initial
        ``steady_state``: when the EV leaves the origin, each station holds a number of exogenous
        EVs drawn from the M/M/1/kappa stationary distribution (the user's choice). ``empty``:
        every station is empty.
    full
        ``leave``: the EV cannot join a full station (the user's choice). ``join``: it may.
    """

    model: Literal["mm1k"] = "mm1k"
    initial: Literal["steady_state", "empty"] = "steady_state"
    full: Literal["leave", "join"] = "leave"


class EVShortestPathParams(FrozenModel):
    """The EV shortest path with occupancy indicators.

    Attributes
    ----------
    target_levels
        Size of the action grid of target battery levels: ``k * B / target_levels`` for
        ``k = 1..target_levels``, plus each station's charging breakpoints.
    horizon
        Hours of exogenous activity sampled per episode; after it, no exogenous EV arrives.
    queues
        The exogenous EVs (`EVQueueParams`).
    """

    target_levels: PositiveInt = 20
    horizon: PositiveFloat = 72.0
    queues: EVQueueParams = Field(default_factory=EVQueueParams)


def ev_shortest_path(ev: Any) -> EVShortestPathParams | None:
    """The parsed ``ev.shortest_path`` section, or None when ``ev`` has none."""
    if isinstance(ev, Mapping) and set(ev) == {"shortest_path"}:
        from fleetgym.spec.models.base import deep_thaw

        return EVShortestPathParams.model_validate(deep_thaw(ev["shortest_path"]) or {})
    return None
