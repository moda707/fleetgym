"""The `ActionInterface` base class and the interface registry.

An action interface defines the Gymnasium action space, the action mask, and how an action becomes
primitive simulator decisions. The simulator never imports an interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from fleetgym.plugins import Registry
from fleetgym.sim.decisions import Decision
from fleetgym.sim.engine import Epoch, Simulator

__all__ = ["INTERFACE_CLASSES", "ActionInterface"]


class ActionInterface(ABC):
    """Translate between Gymnasium actions and simulator decisions.

    Parameters
    ----------
    max_nodes
        Number of node slots (observations and actions are padded to it).
    max_vehicles
        Number of vehicle slots.
    wait
        Offer a wait action (``next_node`` with dynamic requests).
    """

    name: ClassVar[str]

    def __init__(self, max_nodes: int, max_vehicles: int = 0, *, wait: bool = False) -> None:
        self.max_nodes = max_nodes
        self.max_vehicles = max_vehicles
        self.wait = wait

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Space[Any]:
        """The Gymnasium action space."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Length of the action mask."""

    @abstractmethod
    def mask(self, sim: Simulator, epoch: Epoch) -> NDArray[np.bool_]:
        """Valid actions at ``epoch`` (length `n_actions`)."""

    @abstractmethod
    def decode(self, action: Any, sim: Simulator, epoch: Epoch) -> Decision:
        """The decision an action stands for."""

    @abstractmethod
    def repair(self, action: Any, mask: NDArray[np.bool_], sim: Simulator, epoch: Epoch) -> Any:
        """A valid action close to ``action`` (used with ``invalid_action="repair"``)."""

    def explain(self, action: Any, mask: NDArray[np.bool_], sim: Simulator, epoch: Epoch) -> str:
        """Why ``action`` is not valid now (message of `InfeasibleActionError`)."""
        return f"action {action} is not valid now; valid actions: {np.flatnonzero(mask).tolist()}"


INTERFACE_CLASSES: Registry[type[ActionInterface]] = Registry("action interface implementation")
"""Runtime interface classes keyed by the spec's ``decisions.interface``."""
