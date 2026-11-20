"""Registry of neural network modules: ``@register("name")`` and `get_module`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fleetgym.plugins import Registry

__all__ = ["MODULES", "get_module", "register"]

MODULES: Registry[type[Any]] = Registry("neural network module")


def register(name: str) -> Callable[[type[Any]], type[Any]]:
    """Register a module class under ``name`` (usable by name in configs and the CLI)."""
    return MODULES.register(name)


def get_module(name: str) -> type[Any]:
    """The module class registered under ``name``."""
    return MODULES.get(name)
