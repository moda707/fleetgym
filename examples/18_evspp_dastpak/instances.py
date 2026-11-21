"""Instances of the EV shortest path with charging-station occupancy (Dastpak et al. 2023).

Dastpak, Errico, Jabali and Malucelli, arXiv 2305.11773, Section 5.1 and 5.3:

* a 350 x 700 km region, origin (0, 0), destination (350, 700), Euclidean distances, 100 km/h;
* three EVs (Table 1): Peugeot I-on 16 kWh at 0.125 kWh/km, BMW i3s 42.2 kWh at 0.165, Renault Zoe
  52 kWh at 0.165, with the charging breakpoints of Table 2 (85, 95 and 100% state of charge);
* station densities of 2, 6 and 10 per 100 x 100 km (49, 147 and 245 stations), uniform locations;
* charging technologies drawn per station from PU1, PU2, PU3 (all Slow, Normal or Fast) or the mixes
  PM1, PM2, PM3 (Table 3), with service rates mu = 0.28, 0.56, 1.12 per hour;
* a utilization of 40, 65 or 90% at every station, the arrival rates of Table 4, and a queue
  capacity kappa in {1, 2, 3}.

As in Section 5.3, one set of locations is drawn for the Moderate density; High adds stations to it
and Low removes some; one technology assignment is drawn for PM2 and changed into PM1 (3% of the
stations from Normal and 7% from Fast to Slow) and PM3 (35% of the stations from Slow: 20% to Normal,
15% to Fast). The EV leaves the origin fully charged (the user's choice).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from fleetgym.data import Instance, InstanceSet, write_instance_set

ORIGIN, DESTINATION = (0.0, 0.0), (350.0, 700.0)
#: Table 1: battery (kWh), consumption (kWh/km)
EVS = {"peugeot": (16.0, 0.125), "bmw": (42.2, 0.165), "renault": (52.0, 0.165)}
#: Table 2: breakpoint levels (kWh) and charging times (h) by technology
BREAKPOINTS = {
    "peugeot": (
        (13.6, 15.2, 16.0),
        {"slow": (1.26, 1.54, 2.04), "normal": (0.62, 0.77, 1.01), "fast": (0.31, 0.39, 0.51)},
    ),
    "bmw": (
        (35.9, 40.1, 42.2),
        {"slow": (3.32, 4.06, 5.38), "normal": (1.64, 2.03, 2.66), "fast": (0.82, 1.03, 1.35)},
    ),
    "renault": (
        (44.2, 49.4, 52.0),
        {"slow": (4.09, 5.00, 6.63), "normal": (2.02, 2.50, 3.28), "fast": (1.01, 1.27, 1.66)},
    ),
}
TECHNOLOGIES = ("slow", "normal", "fast")
MU = {"slow": 0.28, "normal": 0.56, "fast": 1.12}
#: Table 4: arrival rates (EVs/h) by utilization and technology
LAMBDA = {
    "low": {"slow": 0.11, "normal": 0.22, "fast": 0.45},
    "medium": {"slow": 0.18, "normal": 0.36, "fast": 0.73},
    "high": {"slow": 0.25, "normal": 0.50, "fast": 1.01},
}
DENSITY = {"low": 49, "moderate": 147, "high": 245}
#: Table 3: shares of Slow, Normal, Fast
MIXES = {
    "PU1": (1.0, 0.0, 0.0),
    "PU2": (0.0, 1.0, 0.0),
    "PU3": (0.0, 0.0, 1.0),
    "PM1": (0.55, 0.42, 0.03),
    "PM2": (0.45, 0.45, 0.10),
    "PM3": (0.10, 0.65, 0.25),
}


def station_locations(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Nested locations: Moderate, High = Moderate plus more, Low = a subset of Moderate."""
    moderate = rng.uniform((0.0, 0.0), (350.0, 700.0), size=(DENSITY["moderate"], 2))
    extra = rng.uniform((0.0, 0.0), (350.0, 700.0), size=(DENSITY["high"] - DENSITY["moderate"], 2))
    keep = np.sort(rng.choice(DENSITY["moderate"], size=DENSITY["low"], replace=False))
    return {"low": moderate[keep], "moderate": moderate, "high": np.vstack([moderate, extra])}


def technologies(rng: np.random.Generator, n: int) -> dict[str, list[str]]:
    """One assignment per mix; PM1 and PM3 are changed from PM2 (Section 5.3)."""
    out = {f"PU{k + 1}": [t] * n for k, t in enumerate(TECHNOLOGIES)}
    shares = np.asarray(MIXES["PM2"])
    pm2 = list(rng.choice(TECHNOLOGIES, size=n, p=shares / shares.sum()))
    out["PM2"] = pm2

    def convert(source: list[str], frm: str, to: str, count: int) -> list[str]:
        result = list(source)
        idx = [i for i, t in enumerate(result) if t == frm]
        for i in rng.choice(idx, size=min(count, len(idx)), replace=False):
            result[int(i)] = to
        return result

    pm1 = convert(pm2, "normal", "slow", round(0.03 * n))
    out["PM1"] = convert(pm1, "fast", "slow", round(0.07 * n))
    to_normal = convert(pm2, "slow", "normal", round(0.20 * n))
    out["PM3"] = convert(to_normal, "slow", "fast", round(0.15 * n))
    return out


def evspp_instance(
    locations: np.ndarray, techs: list[str], ev: str, utilization: str, kappa: int, name: str
) -> Instance:
    """One instance: origin, destination, then the stations."""
    n = len(locations) + 2
    capacity, consumption = EVS[ev]
    levels, times = BREAKPOINTS[ev]
    tech = ["", "", *techs]
    station = np.arange(n) >= 2
    extra = {
        "battery_capacity": capacity,
        "consumption": consumption,
        "role": ["origin", "destination"] + ["station"] * len(locations),
        "technology": tech,
        "arrival_rate": np.array(
            [LAMBDA[utilization][t] if s else 0.0 for t, s in zip(tech, station, strict=True)]
        ),
        "service_rate": np.array([MU[t] if s else 0.0 for t, s in zip(tech, station, strict=True)]),
        "queue_capacity": np.where(station, kappa, 1),
    }
    for k in range(3):
        extra[f"charge_level_{k + 1}"] = np.where(station, levels[k], 0.0)
        extra[f"charge_time_{k + 1}"] = np.array(
            [times[t][k] if s else 0.0 for t, s in zip(tech, station, strict=True)]
        )
    return Instance(
        name=name,
        source="examples/18 (Dastpak et al. 2023, Section 5.1)",
        demands=np.zeros(n),
        coords=np.vstack([ORIGIN, DESTINATION, locations]),
        vehicle_types=({"count": 1},),
        schema="fleetgym/evspp/1",
        extra=extra,
    )


#: Table 6: the 18 instances of the experiments, as (kappa, ev, density, mix, utilization)
TABLE_6 = [
    *[(1, "bmw", "moderate", p, "medium") for p in ("PU1", "PU2", "PU3", "PM1", "PM2", "PM3")],
    *[(k, "bmw", "moderate", "PU2", "medium") for k in (2, 3)],
    *[(1, v, "moderate", "PU2", "medium") for v in ("peugeot", "renault")],
    *[(1, "bmw", d, "PU2", "medium") for d in ("low", "high")],
    *[(1, "bmw", "moderate", "PU2", u) for u in ("low", "high")],
]


def folder(kappa: int, ev: str, density: str, mix: str, utilization: str) -> str:
    """The preset's folder of one grid point."""
    return f"{ev}-{density}-{mix}-{utilization}-k{kappa}"


def write_evspp(
    root: str | Path, points: Iterable[tuple[int, str, str, str, str]] = TABLE_6, seed: int = 0
) -> dict[str, InstanceSet]:
    """Write one set per grid point (Table 6 by default), sharing the nested samples."""
    rng = np.random.default_rng(seed)
    locations = station_locations(rng)
    techs = {d: technologies(rng, len(locations[d])) for d in DENSITY}
    sets = {}
    for kappa, ev, density, mix, utilization in points:
        name = folder(kappa, ev, density, mix, utilization)
        inst = evspp_instance(locations[density], techs[density][mix], ev, utilization, kappa, name)
        sets[name] = write_instance_set(
            [inst],
            Path(root) / name,
            name=f"EVSPP-OI {name}",
            reference="Dastpak, Errico, Jabali and Malucelli (2023), arXiv 2305.11773, Section 5.1",
            produced_by=f"examples/18_evspp_dastpak/instances.py, seed {seed}",
            overwrite=True,
        )
    return sets
