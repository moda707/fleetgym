import importlib
import pkgutil

import pytest

import fleetgym

MODULES = sorted(
    m.name for m in pkgutil.walk_packages(fleetgym.__path__, prefix="fleetgym.") if not m.ispkg
) + sorted(m.name for m in pkgutil.walk_packages(fleetgym.__path__, prefix="fleetgym.") if m.ispkg)


def test_version_is_set():
    assert isinstance(fleetgym.__version__, str)
    assert fleetgym.__version__


@pytest.mark.parametrize("name", MODULES)
def test_module_imports_and_has_docstring(name: str):
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} has no module docstring"
