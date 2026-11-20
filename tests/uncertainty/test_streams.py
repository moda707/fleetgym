import numpy as np
import pytest

from fleetgym.uncertainty.streams import StreamFactory, source_id


def test_streams_are_reproducible():
    a = StreamFactory(42).generator("demand").random(5)
    b = StreamFactory(42).generator("demand").random(5)
    np.testing.assert_array_equal(a, b)


def test_streams_depend_on_name_not_on_order():
    first = StreamFactory(1)
    second = StreamFactory(1)
    _ = first.generator("arrivals").random(100)  # drawing another source first changes nothing
    np.testing.assert_array_equal(
        first.generator("demand").random(5), second.generator("demand").random(5)
    )


def test_sources_and_seeds_are_independent():
    streams = StreamFactory(1)
    assert not np.array_equal(
        streams.generator("demand").random(5), streams.generator("service_time").random(5)
    )
    assert not np.array_equal(
        StreamFactory(1).generator("demand").random(5),
        StreamFactory(2).generator("demand").random(5),
    )
    assert not np.array_equal(
        streams.generator("demand", 0).random(5), streams.generator("demand", 1).random(5)
    )


def test_source_id_is_stable():
    # pinned: changing the hashing would silently change every scenario
    assert source_id("demand") == 2132278495
    assert 0 <= source_id("anything") < 2**32


def test_negative_seed_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        StreamFactory(-1)
