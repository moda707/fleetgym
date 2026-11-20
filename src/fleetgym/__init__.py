"""FleetGym: Gymnasium and PettingZoo environments for vehicle routing.

FleetGym models stochastic and dynamic vehicle routing problems as sequential decision
problems. Every environment is defined by one YAML config, and every source of randomness
is pre-sampled into a scenario so that policies can be compared on common random numbers.

Examples
--------
>>> import fleetgym as fg
>>> env = fg.make("CVRP-v0")
>>> obs, info = env.reset(seed=0)
"""

from fleetgym._version import __version__
from fleetgym.constraints.base import Constraint, register_constraint
from fleetgym.envs import (
    FleetGymAECEnv,
    FleetGymEnv,
    FleetGymParallelEnv,
    make,
    make_multi,
    make_vec,
)
from fleetgym.sim.recourse import RecourseRule, register_recourse
from fleetgym.spec import REGISTRY, ProblemSpec, load_spec
from fleetgym.uncertainty.distributions import Distribution, register_distribution
from fleetgym.uncertainty.scenario import Scenario

__all__ = [
    "REGISTRY",
    "Constraint",
    "Distribution",
    "FleetGymAECEnv",
    "FleetGymEnv",
    "FleetGymParallelEnv",
    "ProblemSpec",
    "RecourseRule",
    "Scenario",
    "__version__",
    "load_spec",
    "make",
    "make_multi",
    "make_vec",
    "register_constraint",
    "register_distribution",
    "register_recourse",
]
