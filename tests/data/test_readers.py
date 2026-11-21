from pathlib import Path

import numpy as np
import pytest

from fleetgym.data import read_instance, read_solution
from fleetgym.errors import InstanceError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "instances"


def test_read_vrplib_coordinates():
    inst = read_instance(FIXTURES / "tiny.vrp")
    assert inst.name == "tiny-n5"
    assert inst.source == "vrplib"
    assert inst.n_nodes == 5 and inst.n_depots == 1
    assert inst.capacity == 10.0
    assert inst.n_vehicles is None
    np.testing.assert_array_equal(inst.demands, [0, 3, 4, 5, 2])
    assert inst.coords is not None
    np.testing.assert_array_equal(inst.coords[4], [2.5, 0])
    assert inst.edge_weight_type == "EUC_2D"
    assert inst.distance_matrix is None
    assert inst.meta["type"] == "CVRP"
    assert inst.meta["dimension"] == 5


def test_read_vrplib_explicit_matrix_and_scalar_service_time():
    inst = read_instance(FIXTURES / "tiny_explicit.vrp")
    assert inst.coords is None
    assert inst.distance_matrix is not None
    np.testing.assert_array_equal(inst.distance_matrix, [[0, 4, 6], [5, 0, 2], [7, 3, 0]])
    assert inst.n_vehicles == 2
    assert inst.service_times is not None
    np.testing.assert_array_equal(inst.service_times, [0, 7, 7])


def test_depots_must_lead():
    with pytest.raises(InstanceError, match="depots must be the first nodes"):
        read_instance(FIXTURES / "tiny_depot2.vrp")


def test_read_solomon():
    inst = read_instance(FIXTURES / "tiny_solomon.txt")
    assert inst.name == "TINY01"
    assert inst.source == "solomon"
    assert inst.n_vehicles == 3 and inst.capacity == 20.0
    assert inst.time_windows is not None and inst.service_times is not None
    np.testing.assert_array_equal(inst.time_windows[2], [10, 60])
    np.testing.assert_array_equal(inst.service_times, [0, 10, 10, 5])
    assert inst.get("depot_due_time") == 100.0


def test_format_inference_and_errors(tmp_path: Path):
    with pytest.raises(InstanceError, match="cannot infer"):
        read_instance(tmp_path / "x.dat")
    with pytest.raises(InstanceError, match="not found"):
        read_instance(tmp_path / "missing.vrp")
    with pytest.raises(InstanceError, match="not found"):
        read_instance(tmp_path / "missing.txt")
    bad = tmp_path / "bad.txt"
    bad.write_text("not a solomon file\n")
    with pytest.raises(InstanceError, match="not a valid Solomon"):
        read_instance(bad)
    bad_vrp = tmp_path / "bad.vrp"
    bad_vrp.write_text("garbage line without colon\n")
    with pytest.raises(InstanceError, match="not a valid VRPLIB"):
        read_instance(bad_vrp)
    inst = read_instance(FIXTURES / "tiny_solomon.txt", fmt="solomon")
    assert inst.name == "TINY01"


def test_read_solution(tmp_path: Path):
    sol = read_solution(FIXTURES / "tiny.sol")
    assert sol.routes == ((1, 3), (2, 4))
    assert sol.cost == 17.0
    empty = tmp_path / "e.sol"
    empty.write_text("Cost 3\n")
    with pytest.raises(InstanceError, match="no `Route`"):
        read_solution(empty)
    with pytest.raises(InstanceError, match="not found"):
        read_solution(tmp_path / "missing.sol")
