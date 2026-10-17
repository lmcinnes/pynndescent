import numba
import numpy as np
import pytest
from scipy.sparse import csr_array

from pynndescent.search_graph import build_search_graph, permute_graph, permute_rows


def _reference(rows, dists, max_degree, order):
    """The previous scipy pipeline (forward CSR, maximum with the transpose,
    no diagonal, degree pruning, binarise, relabel)."""
    n = rows.shape[0]
    mask = rows >= 0
    forward = csr_array(
        (dists[mask], (np.repeat(np.arange(n), rows.shape[1])[mask.ravel()], rows[mask])),
        shape=(n, n),
    )
    graph = forward.maximum(forward.T.tocsr()).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    for i in range(n):
        row = graph.data[graph.indptr[i] : graph.indptr[i + 1]]
        if row.shape[0] > max_degree:
            cut = np.sort(row)[max_degree]
            row[row > cut] = 0.0
    graph.eliminate_zeros()
    graph = (graph != 0).astype(np.uint8)
    graph = graph[order][:, order].tocsr()
    graph.sort_indices()
    return graph.indptr, graph.indices, forward.data.min()


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("max_degree", [3, 8, 100])
def test_search_graph_matches_scipy_pipeline(seed, max_degree):
    rng = np.random.default_rng(seed)
    n, k = 600, 12
    rows = np.full((n, k), -1, dtype=np.int32)
    dists = np.full((n, k), np.inf, dtype=np.float32)
    for i in range(n):
        m = rng.integers(0, k + 1)
        rows[i, :m] = rng.choice(n, size=m, replace=False)  # may include i itself
        # rounded, partly negative and asymmetric distances: ties and zero results
        dists[i, :m] = np.round(rng.normal(1.0, 1.0, size=m), 1)
    order = rng.permutation(n)

    expected_indptr, expected_indices, expected_min = _reference(
        rows, dists, max_degree, order
    )
    original_threads = numba.get_num_threads()
    try:
        for threads in sorted({1, 3, original_threads}):
            numba.set_num_threads(threads)
            indptr, indices, min_distance = build_search_graph(rows, dists, max_degree)
            indptr, indices = permute_graph(indptr, indices, order)
            np.testing.assert_array_equal(indptr, expected_indptr)
            np.testing.assert_array_equal(indices, expected_indices)
            assert min_distance == expected_min
    finally:
        numba.set_num_threads(original_threads)


def test_permute_rows():
    data = np.random.default_rng(0).normal(size=(100, 7)).astype(np.float32)
    order = np.random.default_rng(1).permutation(100)
    np.testing.assert_array_equal(permute_rows(data, order), data[order])


def _reference_diversify(rows, dists, distance):
    """Plain prune-with-probability-1 relative neighbourhood pruning."""
    eps = np.finfo(np.float32).eps
    out_rows = np.full_like(rows, -1)
    out_dists = np.full_like(dists, np.inf)
    for i in range(rows.shape[0]):
        kept = [(rows[i, 0], dists[i, 0])]
        for j in range(1, rows.shape[1]):
            u = rows[i, j]
            if u < 0:
                break
            if not any(dc > eps and distance(u, c) < dists[i, j] for c, dc in kept):
                kept.append((u, dists[i, j]))
        for m, (c, dc) in enumerate(kept):
            out_rows[i, m] = c
            out_dists[i, m] = dc
    return out_rows, out_dists


def _knn_rows(data_dense, k):
    from sklearn.neighbors import NearestNeighbors

    dists, rows = NearestNeighbors(n_neighbors=k).fit(data_dense).kneighbors(data_dense)
    rows = rows.astype(np.int32)
    dists = (dists**2).astype(np.float32)  # squared euclidean, as used internally
    rows[::7, -3:] = -1  # some short rows
    dists[::7, -3:] = np.inf
    return rows, dists


def test_diversify_matches_reference():
    from pynndescent.distances import squared_euclidean
    from pynndescent.pynndescent_ import diversify

    data = np.random.default_rng(0).normal(size=(400, 6)).astype(np.float32)
    rows, dists = _knn_rows(data, 15)
    expected = _reference_diversify(
        rows, dists, lambda a, b: squared_euclidean(data[a], data[b])
    )
    rng_state = np.array([1, 2, 3], dtype=np.int64)
    result = diversify(rows.copy(), dists.copy(), data, squared_euclidean, rng_state, 1.0)
    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])


def test_sparse_diversify_matches_reference():
    from scipy.sparse import csr_matrix

    from pynndescent.sparse import diversify, sparse_squared_euclidean

    rng = np.random.default_rng(1)
    dense = rng.normal(size=(400, 20)).astype(np.float32)
    dense[rng.random(dense.shape) < 0.6] = 0.0
    data = csr_matrix(dense)
    data.sort_indices()
    rows, dists = _knn_rows(dense, 15)

    def row(a):
        return (
            data.indices[data.indptr[a] : data.indptr[a + 1]],
            data.data[data.indptr[a] : data.indptr[a + 1]],
        )

    expected = _reference_diversify(
        rows, dists, lambda a, b: sparse_squared_euclidean(*row(a), *row(b))
    )
    rng_state = np.array([1, 2, 3], dtype=np.int64)
    result = diversify(
        rows.copy(), dists.copy(), data.indices, data.indptr, data.data,
        sparse_squared_euclidean, rng_state, 1.0,
    )
    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])


def test_partial_diversify_is_deterministic():
    from pynndescent.distances import squared_euclidean
    from pynndescent.pynndescent_ import diversify

    data = np.random.default_rng(2).normal(size=(2000, 6)).astype(np.float32)
    rows, dists = _knn_rows(data, 20)
    rng_state = np.array([5, 6, 7], dtype=np.int64)

    def run(p):
        return diversify(rows.copy(), dists.copy(), data, squared_euclidean, rng_state, p)

    original_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        single = run(0.5)
        numba.set_num_threads(original_threads)
        multi = run(0.5)
    finally:
        numba.set_num_threads(original_threads)
    np.testing.assert_array_equal(single[0], multi[0])

    n_full = (run(1.0)[0] >= 0).sum()
    n_half = (single[0] >= 0).sum()
    n_none = (run(0.0)[0] >= 0).sum()
    assert n_full < n_half < n_none == (rows >= 0).sum()
