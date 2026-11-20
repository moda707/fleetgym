import numpy as np
import pytest

from fleetgym.network import EuclideanNetwork, MatrixNetwork, euclidean_distances


def test_euclidean_without_rounding():
    net = EuclideanNetwork(np.array([[0, 0], [3, 4], [2.5, 0]]), speed=2.0)
    assert net.n_nodes == 3
    assert net.distance(0, 1) == 5.0
    assert net.distance(0, 2) == 2.5
    assert net.travel_time(0, 1, depart=123.0) == 2.5
    np.testing.assert_allclose(net.distance_matrix(), net.distance_matrix().T)
    assert not net.is_time_dependent


def test_nearest_int_rounds_ties_up_unlike_python_round():
    coords = np.array([[0, 0], [2.5, 0], [3.5, 0], [1, 1], [0.5, 0]])
    d = euclidean_distances(coords, "nearest_int")
    assert d[0, 1] == 3.0  # Python round(2.5) == 2
    assert d[0, 2] == 4.0  # Python round(3.5) == 4
    assert d[0, 3] == 1.0  # sqrt(2) = 1.414...
    assert d[0, 4] == 1.0  # Python round(0.5) == 0
    assert round(2.5) == 2  # documents why floor(d + 0.5) is used


def test_matrices_are_read_only():
    net = EuclideanNetwork(np.array([[0.0, 0.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="read-only"):
        net.distance_matrix()[0, 1] = 3
    with pytest.raises(ValueError, match="read-only"):
        net.time_matrix()[0, 1] = 3


def test_euclidean_argument_checks():
    with pytest.raises(ValueError, match="shape"):
        EuclideanNetwork(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="speed"):
        EuclideanNetwork(np.zeros((2, 2)), speed=0)
    with pytest.raises(ValueError, match="unknown rounding"):
        euclidean_distances(np.zeros((2, 2)), "ceil")  # pyright: ignore[reportArgumentType]


def test_matrix_network_asymmetric_and_time():
    dist = np.array([[0, 4, 6], [5, 0, 2], [7, 3, 0]], dtype=float)
    net = MatrixNetwork(dist, speed=2)
    assert net.distance(0, 1) == 4 and net.distance(1, 0) == 5
    assert net.travel_time(1, 0, 0.0) == 2.5
    times = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    assert MatrixNetwork(dist, times).travel_time(0, 2, 0.0) == 1
    dist[0, 1] = 100
    assert net.distance(0, 1) == 4  # copied


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        (np.zeros((2, 3)), "square"),
        (np.array([[0, -1], [1, 0]]), "non-negative"),
        (np.array([[0, np.nan], [1, 0]]), "finite"),
        (np.array([[1, 1], [1, 0]]), "zero diagonal"),
    ],
)
def test_matrix_network_rejects_bad_matrices(matrix, match):
    with pytest.raises(ValueError, match=match):
        MatrixNetwork(matrix)


def test_matrix_network_other_checks():
    with pytest.raises(ValueError, match="same shape"):
        MatrixNetwork(np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="speed"):
        MatrixNetwork(np.zeros((2, 2)), speed=-1)
