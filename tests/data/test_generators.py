import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fleetgym.data import GENERATORS, generate
from fleetgym.data.generators import validate_generator_params
from fleetgym.errors import InstanceError
from fleetgym.plugins import PluginError

TW = {"horizon": 480, "width_low": 60, "width_high": 180, "service_time": 10}


def test_generators_are_registered():
    assert GENERATORS.names() == ("clustered", "uniform")


def test_same_seed_same_instance_and_different_seed_differs():
    params = {"n_customers": 30, "capacity": 50, "time_windows": TW}
    a = generate("uniform", params, np.random.default_rng(7))
    b = generate("uniform", params, np.random.default_rng(7))
    c = generate("uniform", params, np.random.default_rng(8))
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()


@pytest.mark.parametrize(("depot", "expected"), [("center", [50, 50]), ("corner", [0, 0])])
def test_uniform_layout(depot, expected):
    inst = generate(
        "uniform",
        {"n_customers": 500, "depot": depot, "demand": {"low": 2, "high": 4}, "n_vehicles": 3},
        np.random.default_rng(0),
    )
    assert inst.coords is not None
    np.testing.assert_array_equal(inst.coords[0], expected)
    assert inst.n_customers == 500
    assert inst.n_vehicles == 3
    assert inst.capacity is None
    customers = inst.coords[1:]
    assert customers.min() >= 0 and customers.max() <= 100
    assert inst.demands[0] == 0
    assert set(np.unique(inst.demands[1:])) == {2.0, 3.0, 4.0}
    # mean of U[0, 100] is 50; 500 points give a standard error under 1.3
    assert abs(customers.mean() - 50) < 5


def test_random_depot_is_inside_the_square():
    inst = generate(
        "uniform", {"n_customers": 3, "depot": "random", "size": 10}, np.random.default_rng(1)
    )
    assert inst.coords is not None
    assert np.all((inst.coords[0] >= 0) & (inst.coords[0] <= 10))


def test_clustered_points_concentrate_near_centers():
    inst = generate(
        "clustered",
        {"n_customers": 400, "n_clusters": 1, "spread": 2.0, "depot": "corner"},
        np.random.default_rng(3),
    )
    assert inst.coords is not None
    customers = inst.coords[1:]
    # one cluster with spread 2: nearly all points within 4 spreads of their mean
    distance = np.linalg.norm(customers - customers.mean(axis=0), axis=1)
    assert np.quantile(distance, 0.99) < 8 * 2.0
    assert inst.name == "clustered-n400-c1"
    assert inst.source == "generated:clustered"


@given(
    seed=st.integers(0, 2**32 - 1),
    n=st.integers(1, 40),
    width=st.tuples(st.floats(1, 100), st.floats(0, 100)),
    service=st.floats(0, 30),
)
def test_time_windows_are_reachable(seed, n, width, service):
    tw = {
        "horizon": 400,
        "width_low": width[0],
        "width_high": width[0] + width[1],
        "service_time": service,
    }
    inst = generate("uniform", {"n_customers": n, "time_windows": tw}, np.random.default_rng(seed))
    assert (
        inst.coords is not None and inst.time_windows is not None and inst.service_times is not None
    )
    from_depot = np.hypot(*(inst.coords[1:] - inst.coords[0]).T)
    start, end = inst.time_windows[1:, 0], inst.time_windows[1:, 1]
    assert np.all(start >= from_depot - 1e-9)
    assert np.all(start <= end)
    assert np.all(end + service + from_depot <= 400 + 1e-9)
    assert inst.time_windows[0].tolist() == [0, 400]
    assert inst.service_times[0] == 0 and np.all(inst.service_times[1:] == service)


def test_impossible_horizon_is_an_error():
    tw = {"horizon": 10, "width_low": 1, "width_high": 2}
    with pytest.raises(InstanceError, match="too short"):
        generate("uniform", {"n_customers": 5, "time_windows": tw}, np.random.default_rng(0))


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"n_customers": 0},
        {"n_customers": 5, "demand": {"low": 5, "high": 1}},
        {"n_customers": 5, "time_windows": {"horizon": 10, "width_low": 5, "width_high": 1}},
        {"n_customers": 5, "unknown": 1},
    ],
)
def test_invalid_params(params):
    with pytest.raises(InstanceError, match="invalid parameters"):
        generate("uniform", params, np.random.default_rng(0))


def test_validate_params_and_unknown_generator():
    assert (
        validate_generator_params(
            "clustered", {"n_customers": 3, "n_clusters": 2, "spread": 1}
        ).model_dump()["n_clusters"]
        == 2
    )
    with pytest.raises(PluginError, match="unknown instance generator"):
        generate("blobs", {}, np.random.default_rng(0))
