import numpy as np
import pytest

from pynndescent.distances import sinkhorn


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale_x,scale_y", [(1.0, 1.0), (3.0, 7.0)])
@pytest.mark.parametrize("zero_bins", [False, True])
def test_sinkhorn_normalized_marginals(dtype, scale_x, scale_y, zero_bins):
    # Equal marginals and symmetric costs admit an analytic transport plan.
    regularization = 0.5
    expected = 2.0 / (1.0 + np.exp(2.0 / regularization))
    if zero_bins:
        x = np.array([0.0, 0.5, 0.0, 0.5], dtype=dtype)
        y = np.array([0.5, 0.0, 0.5, 0.0], dtype=dtype)
        cost = np.full((4, 4), 9.0, dtype=dtype)
        cost[np.ix_([1, 3], [0, 2])] = [[0.0, 2.0], [2.0, 0.0]]
    else:
        x = np.array([0.5, 0.5], dtype=dtype)
        y = x.copy()
        cost = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=dtype)
    x *= scale_x
    y *= scale_y
    originals = [a.copy() for a in (x, y, cost)]
    assert sinkhorn(x, y, cost, regularization) == pytest.approx(expected, rel=1e-6)
    for actual, original in zip((x, y, cost), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sinkhorn_single_nonzero_bin_uses_masked_cost(dtype):
    x = np.array([0.0, 0.0, 4.0], dtype=dtype)
    y = np.array([0.0, 7.0, 0.0, 0.0], dtype=dtype)
    cost = np.arange(12, dtype=dtype).reshape(3, 4)
    assert sinkhorn(x, y, cost, 2.0) == pytest.approx(9.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sinkhorn_index_with_zero_bins(dtype):
    from pynndescent import NNDescent

    # Point masses have a unique transport plan regardless of regularization.
    data = np.eye(4, dtype=dtype) * np.arange(1, 5, dtype=dtype)[:, None]
    cost = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]).astype(dtype)
    index = NNDescent(
        data,
        metric="sinkhorn",
        metric_kwds={"cost": cost},
        n_neighbors=4,
        random_state=42,
        n_jobs=1,
    )
    indices, distances = index.neighbor_graph
    for row in range(len(data)):
        np.testing.assert_array_equal(np.sort(indices[row]), np.arange(4))
        np.testing.assert_allclose(distances[row], cost[row, indices[row]], atol=1e-6)
