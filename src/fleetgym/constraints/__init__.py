"""Constraint plugins: capacity, time windows, route duration (layer L4).

Fleet size is enforced structurally by the simulator (see `fleetgym.constraints.fleet_size`).
"""

from fleetgym.constraints.base import CONSTRAINT_CLASSES, Constraint, Violation
from fleetgym.constraints.capacity import Capacity
from fleetgym.constraints.route_duration import RouteDuration
from fleetgym.constraints.time_windows import TimeWindows

__all__ = [
    "CONSTRAINT_CLASSES",
    "Capacity",
    "Constraint",
    "RouteDuration",
    "TimeWindows",
    "Violation",
]
