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


def test_sinkhorn_from_workqueue_parallel_region(tmp_path):
    # A subprocess isolates Numba's once-per-process threading-layer selection
    # and turns a nested-workqueue abort into a test failure, not a suite abort.
    import os
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import numba
        import numpy as np
        from pynndescent.distances import sinkhorn

        @numba.njit(parallel=True)
        def distances(x, y, cost):
            result = np.empty(4)
            for i in numba.prange(4):
                result[i] = sinkhorn(x[i], y, cost, 0.5)
            return result

        x = np.array([0.0, 0.5, 0.0, 0.5])[None, :] * np.arange(1, 5)[:, None]
        y = np.array([0.5, 0.0, 0.5, 0.0])
        cost = np.full((4, 4), 9.0)
        cost[np.ix_([1, 3], [0, 2])] = [[0.0, 2.0], [2.0, 0.0]]
        expected = 2.0 / (1.0 + np.exp(4.0))
        np.testing.assert_allclose(distances(x, y, cost), expected, rtol=1e-6)
        assert numba.threading_layer() == "workqueue"
        """)
    env = dict(
        os.environ,
        NUMBA_THREADING_LAYER="workqueue",
        NUMBA_NUM_THREADS="2",
        NUMBA_CACHE_DIR=str(tmp_path),
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
