"""Instances: the `Instance` container, file readers, generators and solutions (layer L1)."""

from fleetgym.data.generators import GENERATORS, generate, register_generator
from fleetgym.data.instance import Instance
from fleetgym.data.readers import READERS, read_instance
from fleetgym.data.solution import Solution, read_solution

__all__ = [
    "GENERATORS",
    "READERS",
    "Instance",
    "Solution",
    "generate",
    "read_instance",
    "read_solution",
    "register_generator",
]
