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
from fleetgym.envs import FleetGymEnv, make, make_vec
from fleetgym.sim.recourse import RecourseRule, register_recourse
from fleetgym.spec import REGISTRY, ProblemSpec, load_spec
from fleetgym.uncertainty.distributions import Distribution, register_distribution
from fleetgym.uncertainty.scenario import Scenario

__all__ = [
    "REGISTRY",
    "Distribution",
    "FleetGymEnv",
    "ProblemSpec",
    "RecourseRule",
    "Scenario",
    "__version__",
    "load_spec",
    "make",
    "make_vec",
    "register_distribution",
    "register_recourse",
]
