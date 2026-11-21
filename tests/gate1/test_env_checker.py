import pytest

import fleetgym.testing.env_checker as checker
from fleetgym.errors import ValidationFailure
from fleetgym.testing.validator import ValidationReport, Violation
from tests.families import spec_for
from tests.registered import registered


@pytest.mark.parametrize("env_id", registered("single"))
def test_registered_configs_pass(env_id: str):
    checker.check_env(spec_for(env_id), n_rollouts=3)


def test_unregistered_spec_passes():
    checker.check_env(spec_for("CVRP-v1", overrides=["objective.distance=2"]), n_rollouts=2)


def test_invalid_rollouts_are_reported(monkeypatch: pytest.MonkeyPatch):
    bad = ValidationReport(violations=[Violation("cost", "made up")])
    monkeypatch.setattr(checker, "validate", lambda *a, **k: bad)
    with pytest.raises(ValidationFailure, match="seed 0 is invalid"):
        checker.check_env(spec_for("CVRP-v1"), n_rollouts=1)
