"""FleetGym: Gymnasium and PettingZoo environments for vehicle routing.

FleetGym models stochastic and dynamic vehicle routing problems as sequential decision
problems. Every environment is defined by one YAML config, and every source of randomness
is pre-sampled into a scenario so that policies can be compared on common random numbers.
"""

from fleetgym._version import __version__

__all__ = ["__version__"]
