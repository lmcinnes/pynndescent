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
