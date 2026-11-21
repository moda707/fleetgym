"""Exogenous EVs at charging stations: M/M/1/kappa queues (proposal 0003, ADR 0026).

A station ``i`` has one charger; exogenous EVs arrive as a Poisson process of rate ``lambda_i``,
charge for an exponential time of rate ``mu_i``, and do not join when ``kappa_i`` EVs are already
there. An episode's scenario holds, per station, the number of exogenous EVs present when the EV
leaves the origin (stationary distribution, or 0), their charging times, and the arrivals of the
horizon with the charging time each would need. Each station draws from its own stream
(``ev.station.<node>``), so changing one station's parameters never changes another's draws.

Scenario sources (all row-major by station node):

* ``ev.initial``: one value per node, the exogenous EVs present at time 0 (0 off stations);
* ``ev.initial_service``: their charging times, station after station, the first is charging;
* ``ev.arrival``, ``ev.arrival_node``, ``ev.arrival_service``: every arrival of the horizon.

The first charging time at time 0 is a full exponential time: the remaining time of a charge in
progress has the same distribution (memorylessness).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fleetgym.data import Instance
from fleetgym.uncertainty.streams import StreamFactory

__all__ = ["STATION_SOURCES", "sample_stations", "stationary_distribution", "steady_state_wait"]

STATION_SOURCES = (
    "ev.initial",
    "ev.initial_service",
    "ev.arrival",
    "ev.arrival_node",
    "ev.arrival_service",
)


def stationary_distribution(lam: float, mu: float, kappa: int) -> NDArray[np.float64]:
    """P(n EVs at the station), n = 0..kappa, of an M/M/1/kappa queue: proportional to rho^n."""
    rho = lam / mu
    weights = rho ** np.arange(kappa + 1, dtype=float)
    return weights / weights.sum()


def steady_state_wait(lam: float, mu: float, kappa: int) -> float:
    """Expected time until the charger is free for an EV arriving at a random time (L / mu).

    ``L`` is the mean number of EVs at the station in steady state; each needs ``1 / mu`` on
    average (Dastpak et al., eq. 11, with the residual charging time of the one being charged
    also ``1 / mu`` by memorylessness).
    """
    probs = stationary_distribution(lam, mu, kappa)
    return float(np.dot(np.arange(kappa + 1), probs) / mu)


def sample_stations(
    instance: Instance, seed: int, horizon: float, initial: str
) -> dict[str, NDArray[np.float64]]:
    """The ``ev.*`` scenario sources of one episode (see the module docstring)."""
    n = instance.n_nodes
    role = np.asarray(instance.extra["role"], dtype=object)
    lam = np.asarray(instance.extra["arrival_rate"], dtype=float)
    mu = np.asarray(instance.extra["service_rate"], dtype=float)
    kappa = np.asarray(instance.extra["queue_capacity"], dtype=float)
    streams = StreamFactory(seed)
    counts = np.zeros(n)
    init_service: list[float] = []
    arrivals: list[tuple[float, int, float]] = []
    for node in range(n):
        if role[node] != "station":
            continue
        rng = streams.generator(f"ev.station.{node}")
        k = int(kappa[node])
        present = 0
        if initial == "steady_state" and lam[node] > 0:
            present = int(rng.choice(k + 1, p=stationary_distribution(lam[node], mu[node], k)))
        counts[node] = present
        init_service.extend(rng.exponential(1.0 / mu[node], size=present).tolist())
        if lam[node] > 0:
            m = int(rng.poisson(lam[node] * horizon))
            times = np.sort(rng.uniform(0.0, horizon, size=m))
            services = rng.exponential(1.0 / mu[node], size=m)
            arrivals.extend(zip(times.tolist(), [node] * m, services.tolist(), strict=True))
    arrivals.sort()
    return {
        "ev.initial": counts,
        "ev.initial_service": np.asarray(init_service, dtype=float),
        "ev.arrival": np.asarray([a[0] for a in arrivals], dtype=float),
        "ev.arrival_node": np.asarray([a[1] for a in arrivals], dtype=float),
        "ev.arrival_service": np.asarray([a[2] for a in arrivals], dtype=float),
    }
