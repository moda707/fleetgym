"""Registries of names that configs may use, with the parameter model of each.

These registries hold *spec-level* information only (names, parameter models, reserved flags), so
that a config can be validated without importing the simulator. Runtime implementations register
in their own layers under the same names.

Third-party plugins register here, for example::

    from fleetgym.spec.registries import CONSTRAINTS
    from fleetgym.spec.models.base import FrozenModel
    from fleetgym.spec.registries import PluginEntry

    class MaxStopsParams(FrozenModel):
        limit: int

    CONSTRAINTS.register("max_stops", PluginEntry(MaxStopsParams, description="Stop limit"))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetgym.plugins import Registry

if TYPE_CHECKING:
    from fleetgym.spec.models.base import FrozenModel

__all__ = [
    "ARRIVAL_PROCESSES",
    "CONSTRAINTS",
    "COST_COMPONENTS",
    "DISTRIBUTIONS",
    "FAILURE_TYPES",
    "INTERFACES",
    "NETWORKS",
    "RECOURSE_RULES",
    "ROUTING_RULES",
    "UNCERTAINTY_SOURCES",
    "PluginEntry",
]


@dataclass(frozen=True)
class PluginEntry:
    """A named plugin known to the spec layer.

    Parameters
    ----------
    params
        Pydantic model of the plugin's parameters in YAML.
    reserved
        True for features declared in the design but not implemented in this version.
    description
        One-line description shown in the JSON Schema.
    """

    params: type[FrozenModel]
    reserved: bool = False
    description: str = ""


DISTRIBUTIONS: Registry[PluginEntry] = Registry("distribution")
CONSTRAINTS: Registry[PluginEntry] = Registry("constraint")
NETWORKS: Registry[PluginEntry] = Registry("network type")
ARRIVAL_PROCESSES: Registry[PluginEntry] = Registry("arrival process")
RECOURSE_RULES: Registry[PluginEntry] = Registry("recourse rule")
INTERFACES: Registry[PluginEntry] = Registry("action interface")
ROUTING_RULES: Registry[PluginEntry] = Registry("routing rule")

#: Uncertainty source names (``uncertainty.<source>``). The params model is unused (empty).
UNCERTAINTY_SOURCES: Registry[PluginEntry] = Registry("uncertainty source")
#: Failure types that a recourse rule can handle (``recourse.<failure_type>``).
FAILURE_TYPES: Registry[PluginEntry] = Registry("failure type")
#: Cost components usable in ``objective`` and reported in ``info["costs"]``.
COST_COMPONENTS: Registry[PluginEntry] = Registry("cost component")
