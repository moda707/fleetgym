"""Building blocks of a new VRP variant: a distribution, a constraint, a recourse rule, a generator.

Importing this module registers them; the YAML file then uses them by name. To ship them as a
package, expose a function that imports this module under the ``fleetgym.plugins`` entry point.
"""

import numpy as np
from pydantic import BaseModel, Field

import fleetgym as fg
from fleetgym.data import Instance
from fleetgym.spec.models import FrozenModel
from fleetgym.spec.models.base import NumberOrRef


# 1. A distribution: triangular demand on [low, high] with a per-customer mode.
class TriangularParams(FrozenModel):
    """Triangular on ``[low, high]`` with mode ``mode`` (from the instance's demand by default)."""

    low: float
    mode: NumberOrRef
    high: float


@fg.register_distribution("triangular_demand", params=TriangularParams)
class TriangularDemand(fg.Distribution):
    """Triangular, rounded to whole units."""

    def sample(self, rng, size):
        """``size`` draws, one per node (``mode`` may differ per node)."""
        low, high = self.p("low", size), self.p("high", size)
        mode = np.clip(self.p("mode", size), low, high)
        return np.round(rng.triangular(low, mode, high))

    def mean(self, size):
        """``(low + mode + high) / 3`` (the forecast before a demand is revealed)."""
        low, high = self.p("low", size), self.p("high", size)
        return (low + np.clip(self.p("mode", size), low, high) + high) / 3

    def _quantile(self, q, size):
        raise NotImplementedError("not needed here")


# 2. A constraint: at most `max` customers per route.
class MaxStopsParams(FrozenModel):
    """Maximum number of customers on one route."""

    max: int


@fg.register_constraint("max_stops", params=MaxStopsParams)
class MaxStops(fg.Constraint):
    """A customer is feasible while the route has fewer than ``max`` customers."""

    def mask(self, state, info, vehicle, candidates):
        """Customers allowed while the vehicle's route has fewer than ``max`` customers."""
        stops = len(
            {node for node in info.vehicle(vehicle).route if node != 0}
        )  # revisits count once
        return np.full(candidates.shape, stops < self.params.max)


# 3. A recourse rule: restock if the customer is near the depot, else outsource the rest.
class NearDepotParams(FrozenModel):
    """``radius`` around the depot; outsourcing costs ``per_unit`` per unit."""

    radius: float
    per_unit: float
    per_customer: float = 0.0


@fg.register_recourse("near_depot_restock", params=NearDepotParams)
class NearDepotRestock(fg.RecourseRule):
    """Return to the depot when it is close, else outsource."""

    def on_failure(self, failure, info):
        """Restock near the depot, outsource far from it."""
        near = info.distances[failure.node, 0] <= self.params.radius
        return "return_to_depot" if near else "outsource"


# 4. An instance generator: customers around a few villages, a new layout every episode.
class VillagesParams(BaseModel):
    """``n_customers`` around ``n_villages`` village centers in ``[0, size]^2``; depot at the center."""

    n_customers: int = Field(ge=1)
    n_villages: int = Field(default=4, ge=1)
    size: float = 100.0
    spread: float = 6.0
    demand_low: int = 1
    demand_high: int = 8


@fg.register_generator("villages", params=VillagesParams)
def villages(rng: np.random.Generator, params: VillagesParams) -> Instance:
    """Village centers uniform in the square, customers normal around them (clipped to it)."""
    centers = rng.uniform(0.15 * params.size, 0.85 * params.size, size=(params.n_villages, 2))
    village = rng.integers(0, params.n_villages, size=params.n_customers)
    offsets = rng.normal(0.0, params.spread, size=(params.n_customers, 2))
    customers = np.clip(centers[village] + offsets, 0.0, params.size)
    demands = rng.integers(params.demand_low, params.demand_high + 1, size=params.n_customers)
    return Instance(
        name=f"villages-n{params.n_customers}",
        source="generated:villages",
        demands=np.concatenate([[0.0], demands]),
        coords=np.vstack([[params.size / 2, params.size / 2], customers]),
    )
