"""FleetGym: Gymnasium and PettingZoo environments for vehicle routing.

FleetGym models stochastic and dynamic vehicle routing problems as sequential decision
problems. Every environment is defined by one YAML config, and every source of randomness
is pre-sampled into a scenario so that policies can be compared on common random numbers.

Examples
--------
>>> import fleetgym as fg
>>> from fleetgym.data import read_instance
>>> env = fg.make("CVRP-v1", instance=read_instance("my_instance.json"))  # doctest: +SKIP
>>> obs, info = env.reset(seed=0)  # doctest: +SKIP

FleetGym ships no instances: models declare the problem, instances (in the FleetGym format, or
benchmark files) carry the numbers. Paper presets reproduce published settings:
``fg.make_preset("solomon1987-vrptw", instance="r101")``.
"""

from fleetgym._version import __version__
from fleetgym.constraints.base import Constraint, register_constraint
from fleetgym.envs import (
    EVShortestPathEnv,
    FleetGymAECEnv,
    FleetGymEnv,
    FleetGymParallelEnv,
    make,
    make_any,
    make_ev,
    make_multi,
    make_preset,
    make_vec,
)
from fleetgym.sim.recourse import RecourseRule, register_recourse
from fleetgym.spec import REGISTRY, ProblemSpec, load_spec
from fleetgym.spec.presets import Preset, load_preset, preset_names
from fleetgym.uncertainty.distributions import Distribution, register_distribution
from fleetgym.uncertainty.scenario import Scenario

__all__ = [
    "REGISTRY",
    "Constraint",
    "Distribution",
    "EVShortestPathEnv",
    "FleetGymAECEnv",
    "FleetGymEnv",
    "FleetGymParallelEnv",
    "Preset",
    "ProblemSpec",
    "RecourseRule",
    "Scenario",
    "__version__",
    "load_preset",
    "load_spec",
    "make",
    "make_any",
    "make_ev",
    "make_multi",
    "make_preset",
    "make_vec",
    "preset_names",
    "register_constraint",
    "register_distribution",
    "register_recourse",
]
