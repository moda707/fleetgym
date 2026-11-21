"""Instances: the `Instance` container, schemas, file formats, instance sets and solutions (L1)."""

from fleetgym.data.generators import GENERATORS, generate, register_generator
from fleetgym.data.instance import Instance
from fleetgym.data.instance_set import InstanceSet, read_instance_set, write_instance_set
from fleetgym.data.readers import READERS, read_instance, write_instance
from fleetgym.data.schema import (
    CORE_SCHEMA,
    FieldSpec,
    InstanceSchema,
    get_schema,
    load_schema,
    register_schema,
)
from fleetgym.data.solution import Solution, read_solution

__all__ = [
    "CORE_SCHEMA",
    "GENERATORS",
    "READERS",
    "FieldSpec",
    "Instance",
    "InstanceSchema",
    "InstanceSet",
    "Solution",
    "generate",
    "get_schema",
    "load_schema",
    "read_instance",
    "read_instance_set",
    "read_solution",
    "register_generator",
    "register_schema",
    "write_instance",
    "write_instance_set",
]
