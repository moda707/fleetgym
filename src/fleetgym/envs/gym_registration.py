"""Register every config with Gymnasium under the ``fleetgym/`` namespace.

After ``import fleetgym``, ``gymnasium.make("fleetgym/CVRP-v0")`` works. Without importing first,
use Gymnasium's module prefix: ``gymnasium.make("fleetgym:fleetgym/CVRP-v0")``. Planned configs
and multi-agent configs (PettingZoo, see `fleetgym.make_multi`) are not registered.
"""

from __future__ import annotations

import gymnasium as gym

from fleetgym.spec import REGISTRY, ConfigRegistry

__all__ = ["NAMESPACE", "register_envs"]

NAMESPACE = "fleetgym"


def register_envs(registry: ConfigRegistry | None = None) -> list[str]:
    """Register every non-planned config; return the Gymnasium IDs registered by this call."""
    registry = registry or REGISTRY
    added = []
    for env_id in registry.ids(agents="single"):
        gym_id = f"{NAMESPACE}/{env_id}"
        if gym_id in gym.registry:
            continue
        gym.register(id=gym_id, entry_point="fleetgym.envs.factory:make", kwargs={"env": env_id})
        added.append(gym_id)
    return added
