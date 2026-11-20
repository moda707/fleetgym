import numpy as np
import pytest

from fleetgym.data import Instance
from fleetgym.errors import InstanceError


def make(**changes) -> Instance:
    fields = {
        "name": "t",
        "source": "test",
        "demands": [0, 1, 2],
        "coords": [[0, 0], [1, 0], [0, 1]],
        "capacity": 5,
    }
    fields.update(changes)
    return Instance(**fields)


def test_arrays_are_read_only_copies():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    inst = make(coords=coords, extra={"arr": np.arange(3), "nested": {"l": [1, 2]}})
    coords[0, 0] = 99
    assert inst.coords is not None and inst.coords[0, 0] == 0
    for array in (inst.demands, inst.coords, inst.extra["arr"]):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 1
    with pytest.raises(TypeError):
        inst.extra["x"] = 1  # pyright: ignore[reportIndexIssue]
    assert inst.extra["nested"]["l"] == (1, 2)
    assert inst.demands.dtype == np.float64


def test_derived_values_and_get():
    inst = make(time_windows=[[0, 100], [5, 10], [0, 50]], n_vehicles=2)
    assert inst.n_nodes == 3
    assert inst.n_customers == 2
    assert inst.get("capacity") == 5.0
    assert inst.get("n_vehicles") == 2
    assert inst.get("depot_due_time") == 100.0
    np.testing.assert_array_equal(inst.get("customer_coords"), [[1, 0], [0, 1]])
    assert inst.get("n_customers") == 2
    assert inst.get("n_nodes") == 3
    assert "depot_due_time" in inst.keys()  # noqa: SIM118 - Instance.keys is not a dict view
    assert "service_times" not in inst.keys()  # noqa: SIM118
    with pytest.raises(InstanceError, match=r"no value for 'service_times'; available: .*capacity"):
        inst.get("service_times")
    with pytest.raises(InstanceError, match="no value for 'name'"):
        inst.get("name")
    assert make(extra={"distance": 7}).get("distance") == 7


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"demands": []}, "non-empty"),
        ({"demands": [[0, 1]]}, "1-D"),
        ({"coords": [[0, 0]]}, "shape"),
        ({"demands": [0, -1, 2]}, "non-negative"),
        ({"demands": [0, np.nan, 2]}, "finite"),
        ({"coords": [[0, 0], [np.inf, 0], [0, 1]]}, "NaN or infinite"),
        ({"capacity": 0}, "positive"),
        ({"n_vehicles": 0}, "positive integer"),
        ({"n_vehicles": 1.5}, "positive integer"),
        ({"time_windows": [[0, 10], [5, 4], [0, 1]]}, "opens after"),
        ({"n_depots": 3}, "depots"),
        ({"n_depots": 0}, "depots"),
        ({"distance_matrix": np.zeros((2, 2))}, "shape"),
    ],
)
def test_validation(changes, match):
    with pytest.raises(InstanceError, match=match):
        make(**changes)


def test_content_hash_covers_data_not_name():
    a = make()
    assert a.content_hash() == make(name="other", source="x").content_hash()
    assert a.content_hash() != make(capacity=6).content_hash()
    assert a.content_hash() != make(demands=[0, 1, 3]).content_hash()
    assert a.content_hash() != make(extra={"k": np.arange(2)}).content_hash()
    assert (
        make(extra={"k": np.arange(2)}).content_hash()
        == make(extra={"k": np.arange(2)}).content_hash()
    )
    assert a.content_hash().startswith("fg1:")


def test_repr_is_short():
    assert (
        repr(make())
        == "Instance(name='t', source='test', n_nodes=3, capacity=5.0, n_vehicles=None)"
    )
