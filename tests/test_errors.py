from pathlib import Path

import pytest

from fleetgym.errors import ConfigError, FleetGymError, NotInScopeError


@pytest.mark.parametrize(
    ("source", "key_path", "expected"),
    [
        (Path("configs/a.yaml"), "fleet.types.0", "configs/a.yaml: fleet.types.0: bad"),
        ("configs/a.yaml", None, "configs/a.yaml: bad"),
        (None, "objective", "objective: bad"),
        (None, None, "bad"),
    ],
)
def test_config_error_message_names_location(source, key_path, expected):
    err = ConfigError("bad", source=source, key_path=key_path)
    assert str(err) == expected
    assert err.message == "bad"
    assert err.key_path == key_path


def test_all_errors_share_a_base_class():
    assert issubclass(ConfigError, FleetGymError)
    assert issubclass(NotInScopeError, FleetGymError)
