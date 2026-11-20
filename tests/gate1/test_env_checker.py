import pytest

import fleetgym as fg
import fleetgym.testing.env_checker as checker
from fleetgym.errors import ValidationFailure
from fleetgym.spec import load_spec
from fleetgym.testing.validator import ValidationReport, Violation


@pytest.mark.parametrize("env_id", fg.REGISTRY.ids(agents="single"))
def test_registered_configs_pass(env_id: str):
    checker.check_env(env_id, n_rollouts=3)


def test_unregistered_spec_passes():
    checker.check_env(load_spec("CVRP-v0", overrides=["fleet.types.0.capacity=40"]), n_rollouts=2)


def test_invalid_rollouts_are_reported(monkeypatch: pytest.MonkeyPatch):
    bad = ValidationReport(violations=[Violation("cost", "made up")])
    monkeypatch.setattr(checker, "validate", lambda *a, **k: bad)
    with pytest.raises(ValidationFailure, match="seed 0 is invalid"):
        checker.check_env("CVRP-v0", n_rollouts=1)
