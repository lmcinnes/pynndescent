import numpy as np
import pytest

from pynndescent import NNDescent, distances


@pytest.mark.parametrize("angle", [0.0, np.pi / 6, np.pi / 4, np.pi / 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("corrected", [False, True])
def test_true_angular_returns_distance(angle, dtype, corrected):
    x = np.array([1.0, 0.0], dtype=dtype)
    y = np.array([np.cos(angle), np.sin(angle)], dtype=dtype)
    if corrected:
        actual = distances.true_angular_from_alt_cosine(
            distances.alternative_cosine(x, y)
        )
    else:
        actual = distances.true_angular(x, y)
    np.testing.assert_allclose(actual, angle / np.pi, atol=1e-7)


@pytest.mark.parametrize("query", [False, True])
def test_true_angular_index_distances(query):
    angles = np.deg2rad(np.arange(0, 61, 10))
    data = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    init_graph = np.tile(np.arange(len(data), dtype=np.int32), (len(data), 1))
    index = NNDescent(
        data,
        metric="true_angular",
        n_neighbors=len(data),
        n_trees=0,
        n_iters=2,
        init_graph=init_graph,
        n_jobs=1,
        random_state=0,
    )
    if query:
        query_angles = np.deg2rad([5.0, 25.0, 45.0])
        queries = np.column_stack((np.cos(query_angles), np.sin(query_angles)))
        indices, actual = index.query(queries, k=3)
        expected = np.abs(query_angles[:, None] - angles[indices]) / np.pi
    else:
        neighbor_graph = index.neighbor_graph
        assert neighbor_graph is not None
        indices, actual = neighbor_graph
        expected = np.abs(angles[:, None] - angles[indices]) / np.pi
    np.testing.assert_allclose(actual, expected, atol=2e-4)
    assert np.all(np.diff(actual, axis=1) >= -2e-4)
