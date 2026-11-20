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


#: Modules that need an optional extra, and the package that extra provides.
OPTIONAL = {
    "fleetgym.nn": "torch",
    "fleetgym.nn.policies": "sb3_contrib",
    "fleetgym.agents": "sb3_contrib",
    "fleetgym.testing.module_checker": "torch",
    "fleetgym.viz": "matplotlib",
}


@pytest.mark.parametrize("name", MODULES)
def test_module_imports_and_has_docstring(name: str):
    for prefix, dependency in OPTIONAL.items():
        if name == prefix or name.startswith(prefix + "."):
            pytest.importorskip(dependency)
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} has no module docstring"
