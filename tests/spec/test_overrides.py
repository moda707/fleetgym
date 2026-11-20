import pytest

from fleetgym.errors import ConfigError
from fleetgym.spec.overrides import apply_overrides, parse_override

DATA = {"fleet": {"types": [{"name": "truck", "count": 3}]}, "objective": {"distance": 1.0}}


@pytest.mark.parametrize(
    ("text", "path", "value"),
    [
        ("fleet.types.0.count=8", ("fleet", "types", 0, "count"), 8),
        ("a=1.5", ("a",), 1.5),
        ("a=1e-6", ("a",), 1e-6),
        ("a=[1, 2]", ("a",), [1, 2]),
        ("a={b: 1}", ("a",), {"b": 1}),
        ("a=null", ("a",), None),
        ("a=x=y", ("a",), "x=y"),
        ("a=true", ("a",), True),
    ],
)
def test_parse_override(text, path, value):
    assert parse_override(text) == (path, value)


@pytest.mark.parametrize("text", ["novalue", "a.=1", "=1", "a=", "a..b=1"])
def test_malformed_overrides(text):
    with pytest.raises(ConfigError):
        parse_override(text)


def test_apply_sets_creates_and_deletes_without_mutating():
    out = apply_overrides(
        DATA,
        [
            "fleet.types.0.count=8",
            "objective.rejected=50",
            "dynamics.cutoff=10",
            "objective.distance=null",
        ],
    )
    assert out["fleet"]["types"][0]["count"] == 8
    assert out["objective"] == {"rejected": 50}
    assert out["dynamics"] == {"cutoff": 10}
    assert DATA["fleet"]["types"][0]["count"] == 3


def test_override_replaces_maps():
    assert apply_overrides(DATA, ["objective={time: 1}"])["objective"] == {"time": 1}


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("fleet.types.5.count=1", "out of range"),
        ("fleet.types.x.count=1", "must be an integer"),
        ("fleet.types.0.name.x=1", "scalar"),
        ("objective.distance.x=1", "scalar"),
        ("fleet.types.0=null", "cannot delete a list item"),
        ("fleet.types.9=1", "out of range"),
    ],
)
def test_invalid_paths(text, match):
    with pytest.raises(ConfigError, match=match):
        apply_overrides(DATA, [text])
