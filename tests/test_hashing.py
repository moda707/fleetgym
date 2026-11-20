import math

import pytest

from fleetgym._hashing import canonical_json, content_hash, short_hash


def test_canonical_json_ignores_key_order():
    assert canonical_json({"b": 1, "a": [1.5, {"d": 2, "c": 3}]}) == canonical_json(
        {"a": [1.5, {"c": 3, "d": 2}], "b": 1}
    )


def test_canonical_json_is_compact_and_unicode():
    assert canonical_json({"a": "é", "b": [1, 2]}) == '{"a":"é","b":[1,2]}'


def test_floats_round_trip_shortest_repr():
    assert canonical_json(0.1) == "0.1"
    assert canonical_json(1e-6) == "1e-06"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_rejected(bad: float):
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json({"x": bad})


def test_content_hash_is_versioned_and_stable():
    h = content_hash({"a": 1})
    assert h.startswith("fg1:")
    assert len(h) == 4 + 64
    assert h == content_hash({"a": 1})
    assert h != content_hash({"a": 1.0})
    assert short_hash(h) == h[4:16]
