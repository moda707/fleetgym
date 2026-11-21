"""Building blocks of a new VRP variant: a distribution, a constraint and a recourse rule.

Importing this module registers them; the YAML file then uses them by name. To ship them as a
package, expose a function that imports this module under the ``fleetgym.plugins`` entry point.
"""

import numpy as np

import fleetgym as fg
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


# 2. A constraint: at most `max` customers per route (the number comes from the instance).
class MaxStopsParams(FrozenModel):
    """Maximum number of customers on one route: a number, or a field of the instance."""

    max: NumberOrRef


@fg.register_constraint("max_stops", params=MaxStopsParams)
class MaxStops(fg.Constraint):
    """A customer is feasible while the route has fewer than ``max`` customers."""

    def mask(self, state, info, vehicle, candidates):
        """Customers allowed while the vehicle's route has fewer than ``max`` customers."""
        limit = int(self.problem.value("constraints.max_stops.max"))  # the instance's number
        route = info.vehicle(vehicle).route
        stops = len({node for node in route if node != 0})  # a revisit is not a new stop
        return np.full(candidates.shape, stops < limit)


# 3. A recourse rule: restock if the customer is near the depot, else outsource the rest.
class NearDepotParams(FrozenModel):
    """``radius`` around the depot; outsourcing costs ``per_unit`` per unit (numbers or fields)."""

    radius: NumberOrRef
    per_unit: NumberOrRef
    per_customer: NumberOrRef = 0.0


@fg.register_recourse("near_depot_restock", params=NearDepotParams)
class NearDepotRestock(fg.RecourseRule):
    """Return to the depot when it is close, else outsource."""

    def on_failure(self, failure, info):
        """Restock near the depot, outsource far from it."""
        radius = float(self.problem.value(f"{self.path}.radius"))
        near = info.distances[failure.node, 0] <= radius
        return "return_to_depot" if near else "outsource"
