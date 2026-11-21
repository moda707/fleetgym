"""Sampling the requests of a day (design §7, ADR 0014).

Node slots of an episode: the depots, then the requests known at the start (the instance's
customers when ``dynamics.initial_requests`` is ``instance``), then ``max_requests`` slots for
requests arriving during the day, in arrival order. Unused slots have release time ``inf``.

Each attribute draws from its own named stream (``dynamics.arrivals``, ``dynamics.location``,
``dynamics.demand``, ``dynamics.time_window_width``, ``dynamics.service_time``,
``dynamics.cancellations``), so changing one attribute's distribution does not change the others.

Request attributes
------------------
* location: a point from ``kde`` (Gaussian KDE of two-dimensional ``data``, by default the
  instance's customer coordinates), ``uniform`` (on the square ``[low, high)^2``) or
  ``normal_mixture`` (normal clusters around two-dimensional ``centers``; points outside any
  service area are kept);
* demand and service time: scalar distributions (0 when not given);
* time window (when the spec has the ``time_windows`` constraint): opens at the release time and
  closes ``time_window_width`` later, capped at the depot's closing time; without a width it closes
  with the depot;
* cancellation: ``cancellations`` (a 0/1 distribution, for example ``bernoulli``) says whether the
  request cancels; a cancelling request cancels at a time drawn uniformly between its release and
  its deadline (window close, or the end of the arrival window without time windows).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from fleetgym.errors import ConfigError, FleetGymError
from fleetgym.spec.binding import BoundProblem
from fleetgym.spec.models import DistSpec, ProblemSpec
from fleetgym.uncertainty.arrivals import poisson_arrivals, thinned_arrivals
from fleetgym.uncertainty.distributions import normal_mixture_points
from fleetgym.uncertainty.sampling import distribution_at
from fleetgym.uncertainty.streams import StreamFactory

__all__ = [
    "RequestOverflowError",
    "arrival_window",
    "resample_future_requests",
    "sample_requests",
]


class RequestOverflowError(FleetGymError):
    """More requests arrived than ``dynamics.max_requests`` with ``overflow: error``."""


def _error(spec: ProblemSpec, path: str, message: str) -> ConfigError:
    return ConfigError(message, source=spec.source, key_path=path)


def arrival_window(problem: BoundProblem) -> tuple[float, float]:
    """The arrival window ``[start, end)``.

    It runs from the shift start to the earliest of the cutoff, the end of the rate profile, the
    depot's closing time and the shift end.

    Raises
    ------
    ConfigError
        If the window has no end (homogeneous Poisson without a cutoff or depot time window).
    """
    spec = problem.spec
    assert problem.dynamics is not None
    dyn = problem.dynamics
    assert dyn.arrivals is not None  # binding resolves it (spec or instance)
    start = problem.shift[0] if problem.shift is not None else 0.0
    ends: list[float] = []
    if dyn.cutoff is not None:
        ends.append(float(dyn.cutoff))
    if dyn.arrivals.name == "nhpp":
        ends.append(float(dyn.arrivals.params.model_dump()["profile"]["breaks"][-1]))
    instance = problem.instance
    # No request arrives after the depot closes or the shift ends.
    if instance.time_windows is not None:
        ends.append(float(instance.time_windows[0, 1]))
    if problem.shift is not None:
        ends.append(problem.shift[1])
    if not ends:
        raise _error(
            spec, "dynamics.cutoff", "arrivals need an end: give `cutoff` (or a depot time window)"
        )
    return start, min(ends)


def _points(
    problem: BoundProblem, dist: DistSpec, rng: np.random.Generator, size: int
) -> NDArray[np.float64]:
    spec = problem.spec
    path = "dynamics.request_class.location"
    params = dist.params.model_dump()

    def param(name: str) -> object:
        # A field reference is resolved by binding; a literal is the distribution's own value
        # (the distribution may come from the instance, where the spec has no such path).
        key = f"{path}.{name}"
        return problem.refs[key] if key in problem.refs else params[name]

    if dist.name == "kde":
        data = np.asarray(param("data"), dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise _error(spec, f"{path}.data", "location data must be (x, y) points")
        if size == 0:
            return np.zeros((0, 2))
        kde = stats.gaussian_kde(data.T, bw_method=params.get("bandwidth", "scott"))
        return np.asarray(kde.resample(size, seed=rng).T, dtype=float)
    if dist.name == "uniform":
        low = float(param("low"))  # pyright: ignore[reportArgumentType]
        high = float(param("high"))  # pyright: ignore[reportArgumentType]
        return rng.uniform(low, high, size=(size, 2))
    if dist.name == "normal_mixture":
        if len(params["centers"][0]) != 2:
            raise _error(spec, f"{path}.centers", "location centers must be (x, y) points")
        return normal_mixture_points(params, rng, size)
    raise _error(
        spec,
        path,
        f"locations need a `kde`, `uniform` or `normal_mixture` distribution, not {dist.name!r}",
    )


def _scalar(
    problem: BoundProblem, attr: str, rng: np.random.Generator, size: int
) -> NDArray[np.float64]:
    spec = problem.spec
    assert problem.dynamics is not None
    dist = getattr(problem.dynamics.request_class, attr)
    if dist is None:
        return np.zeros(size)
    path = f"dynamics.request_class.{attr}"
    values = distribution_at(problem, path, dist).sample(rng, size)
    if np.any(~np.isfinite(values) | (values < 0)):
        raise _error(spec, path, f"{attr} must be finite and non-negative; check the distribution")
    return values


def sample_requests(problem: BoundProblem, seed: int) -> dict[str, NDArray[np.float64]]:
    """The ``request.*`` scenario sources of one episode (see `fleetgym.uncertainty.scenario`).

    Raises
    ------
    RequestOverflowError
        If more than ``max_requests`` requests arrive and ``overflow`` is ``error``.
    """
    spec = problem.spec
    assert problem.dynamics is not None
    dyn = problem.dynamics
    assert dyn.arrivals is not None  # binding resolves it (spec or instance)
    instance = problem.instance
    streams = StreamFactory(seed)
    start, end = arrival_window(problem)
    arrivals = dyn.arrivals.params.model_dump()
    rng = streams.generator("dynamics.arrivals")
    if dyn.arrivals.name == "poisson":
        times = poisson_arrivals(rng, float(arrivals["rate"]), start, end)
    else:
        profile = arrivals["profile"]
        times = thinned_arrivals(rng, profile["breaks"], profile["rates"], start, end)
    overflow = times[dyn.max_requests :]
    times = times[: dyn.max_requests]
    if overflow.size and dyn.overflow == "error":
        raise RequestOverflowError(
            f"{dyn.max_requests + overflow.size} requests arrived but `dynamics.max_requests` is "
            f"{dyn.max_requests}; raise it or set `overflow: reject`"
        )
    k = times.size
    depots = instance.n_depots
    static = instance.n_nodes if dyn.initial_requests != "none" else depots
    n = static + dyn.max_requests

    release = np.full(n, np.inf)
    release[:static] = start
    release[static : static + k] = times
    coords = np.zeros((n, 2))
    if instance.coords is not None:
        coords[:static] = instance.coords[:static]
        coords[static:] = instance.coords[0]
    location = dyn.request_class.location
    if location is None:
        raise _error(spec, "dynamics.request_class.location", "dynamic requests need a location")
    coords[static : static + k] = _points(
        problem, location, streams.generator("dynamics.location"), k
    )
    demand = np.zeros(n)
    demand[:static] = instance.demands[:static]
    demand[static : static + k] = _scalar(
        problem, "demand", streams.generator("dynamics.demand"), k
    )
    service = np.zeros(n)
    if instance.service_times is not None:
        service[:static] = instance.service_times[:static]
    service[static : static + k] = _scalar(
        problem, "service_time", streams.generator("dynamics.service_time"), k
    )
    tw_open = np.zeros(n)
    tw_close = np.zeros(n)
    deadline = np.full(n, end)
    if instance.time_windows is not None:
        depot_close = float(instance.time_windows[0, 1])
        tw_open[:static] = instance.time_windows[:static, 0]
        tw_close[:static] = instance.time_windows[:static, 1]
        tw_open[static:] = instance.time_windows[0, 0]
        tw_close[static:] = depot_close
        width = (
            _scalar(
                problem, "time_window_width", streams.generator("dynamics.time_window_width"), k
            )
            if dyn.request_class.time_window_width is not None
            else np.full(k, np.inf)
        )
        tw_open[static : static + k] = times
        tw_close[static : static + k] = np.minimum(times + width, depot_close)
        deadline = tw_close.copy()
    cancel = np.full(n, np.inf)
    if dyn.cancellations is not None:
        rng_c = streams.generator("dynamics.cancellations")
        flags = distribution_at(problem, "dynamics.cancellations", dyn.cancellations).sample(
            rng_c, k
        )
        if not np.all(np.isin(flags, (0.0, 1.0))):
            raise _error(spec, "dynamics.cancellations", "cancellations must be 0/1 draws")
        when = rng_c.uniform(times, np.maximum(deadline[static : static + k], times))
        cancel[static : static + k] = np.where(flags == 1.0, when, np.inf)
    return {
        "request.release": release,
        "request.x": coords[:, 0],
        "request.y": coords[:, 1],
        "request.demand": demand,
        "request.tw_open": tw_open,
        "request.tw_close": tw_close,
        "request.service_time": service,
        "request.cancel": cancel,
        "request.overflow": overflow,
    }


def resample_future_requests(
    problem: BoundProblem,
    current: dict[str, NDArray[np.float64]] | Mapping[str, NDArray[np.float64]],
    now: float,
    seed: int,
    cancelled: NDArray[np.bool_],
) -> dict[str, NDArray[np.float64]]:
    """Request sources that keep every request released by ``now`` and redraw the future.

    Used by `FleetGymEnv.fork`. Requests released at or before ``now`` keep their slots and
    attributes. Their cancellation is redrawn conditionally on not having cancelled by ``now``
    (``cancelled`` marks those that already did, which keep their time): with cancellation
    probability ``p`` and time uniform on ``[release, deadline]``, the conditional probability of a
    later cancellation is ``p (D - now) / (D - r) / (1 - p (now - r) / (D - r))``, uniform on
    ``[now, D]``. Arrivals after ``now`` are drawn afresh on ``[now, end)`` into the remaining
    slots.
    """
    assert problem.dynamics is not None
    dyn = problem.dynamics
    assert dyn.arrivals is not None  # binding resolves it (spec or instance)
    fresh_problem_seed = int(seed)
    out = {name: np.array(values, dtype=float) for name, values in current.items()}
    release = out["request.release"]
    known = release <= now + 1e-12
    future_slots = np.flatnonzero(~known)
    _, end = arrival_window(problem)
    streams = StreamFactory(fresh_problem_seed)
    rng = streams.generator("dynamics.arrivals")
    arrivals = dyn.arrivals.params.model_dump()
    if dyn.arrivals.name == "poisson":
        times = poisson_arrivals(rng, float(arrivals["rate"]), now, end)
    else:
        profile = arrivals["profile"]
        times = thinned_arrivals(rng, profile["breaks"], profile["rates"], now, end)
    times = times[times > now]
    keep = min(times.size, future_slots.size)
    overflow = times[keep:]
    times = times[:keep]
    slots = future_slots[:keep]
    instance = problem.instance
    base = instance.coords[0] if instance.coords is not None else np.zeros(2)
    out["request.release"][future_slots] = np.inf
    out["request.x"][future_slots] = base[0]
    out["request.y"][future_slots] = base[1]
    for name in ("request.demand", "request.service_time", "request.cancel"):
        out[name][future_slots] = 0.0 if name != "request.cancel" else np.inf
    if instance.time_windows is not None:
        out["request.tw_open"][future_slots] = instance.time_windows[0, 0]
        out["request.tw_close"][future_slots] = instance.time_windows[0, 1]
    location = dyn.request_class.location
    assert location is not None
    k = times.size
    out["request.release"][slots] = times
    points = _points(problem, location, streams.generator("dynamics.location"), k)
    out["request.x"][slots] = points[:, 0]
    out["request.y"][slots] = points[:, 1]
    out["request.demand"][slots] = _scalar(
        problem, "demand", streams.generator("dynamics.demand"), k
    )
    out["request.service_time"][slots] = _scalar(
        problem, "service_time", streams.generator("dynamics.service_time"), k
    )
    deadline = np.full(out["request.release"].size, end)
    if instance.time_windows is not None:
        depot_close = float(instance.time_windows[0, 1])
        width = (
            _scalar(
                problem, "time_window_width", streams.generator("dynamics.time_window_width"), k
            )
            if dyn.request_class.time_window_width is not None
            else np.full(k, np.inf)
        )
        out["request.tw_open"][slots] = times
        out["request.tw_close"][slots] = np.minimum(times + width, depot_close)
        deadline = out["request.tw_close"].copy()
    if dyn.cancellations is not None:
        rng_c = streams.generator("dynamics.cancellations")
        dist = distribution_at(problem, "dynamics.cancellations", dyn.cancellations)
        p = float(np.mean(dist.mean(1)))
        flags = dist.sample(rng_c, k)
        when = rng_c.uniform(times, np.maximum(deadline[slots], times))
        out["request.cancel"][slots] = np.where(flags == 1.0, when, np.inf)
        static_end = np.isfinite(release) & known & ~cancelled
        for node in np.flatnonzero(static_end):
            r, d = float(release[node]), float(deadline[node])
            if node < problem.instance.n_depots or r <= float(release[0]) + 1e-12 or d <= now:
                continue
            span = d - r
            gone = p * (now - r) / span
            later = p * (d - now) / span / (1.0 - gone) if gone < 1.0 else 0.0
            out["request.cancel"][node] = (
                float(rng_c.uniform(now, d)) if rng_c.random() < later else np.inf
            )
    out["request.overflow"] = (
        np.concatenate(
            [
                np.asarray(current.get("request.overflow", np.zeros(0)))[
                    np.asarray(current.get("request.overflow", np.zeros(0))) <= now
                ],
                overflow,
            ]
        )
        if dyn.overflow == "reject"
        else np.zeros(0)
    )
    if overflow.size and dyn.overflow == "error":
        raise RequestOverflowError(
            "a fork drew more requests than `dynamics.max_requests`; raise it or set "
            "`overflow: reject`"
        )
    return out
