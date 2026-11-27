import numba
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from pynndescent import NNDescent
from pynndescent.utils import query_rng_state, tau_rand_int


def test_query_rng_states_are_valid_and_independent():
    seed = np.array([-2147483648, 0, 2147483647], dtype=np.int64)
    states = np.array([query_rng_state(seed, i) for i in range(512)])
    assert np.all(states > np.array([1, 7, 15]))
    assert len(np.unique(states, axis=0)) == len(states)
    np.testing.assert_array_equal(seed, [-2147483648, 0, 2147483647])
    np.testing.assert_array_equal(states[0], query_rng_state(seed, 0))


@pytest.mark.parametrize(
    "path,tree_init,leaf_budget",
    [
        ("cached", True, 1),
        ("cached", True, 4),
        ("closure", True, 1),
        ("closure", False, 1),
        ("sparse", True, 1),
        ("sparse", False, 1),
    ],
)
def test_parallel_query_rng_is_reproducible(path, tree_init, leaf_budget):
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("Requires at least two Numba threads")
    rng = np.random.RandomState(27)
    data = rng.normal(size=(300, 8)).astype(np.float32)
    queries = rng.normal(size=(80, 8)).astype(np.float32)
    queries[40:] = queries[:40]
    sparse = path == "sparse"
    index = NNDescent(
        csr_matrix(data) if sparse else data,
        n_neighbors=15,
        random_state=17,
        search_tree_leaf_size=1,
        search_tree_leaf_budget=leaf_budget,
        tree_init=tree_init,
        parallel_batch_queries=True,
    )
    index.prepare(compile_search=path == "closure")
    if tree_init:
        leaves = index._search_forest[0].children
        leaves = leaves[leaves[:, 0] <= 0]
        assert leaf_budget * np.max(leaves[:, 0] - leaves[:, 1]) < 10
    else:
        assert not index._search_forest
    if not sparse:
        assert index._uses_cached_search == (path == "cached")
    query_data = csr_matrix(queries) if sparse else queries

    original_threads = numba.get_num_threads()
    try:
        outputs = []
        for threads in (1, 2, 2, 1):
            numba.set_num_threads(threads)
            outputs.append(index.query(query_data, k=10, epsilon=0.1))
    finally:
        numba.set_num_threads(original_threads)

    for indices, distances in outputs[1:]:
        np.testing.assert_array_equal(indices, outputs[0][0])
        np.testing.assert_array_equal(distances, outputs[0][1])

    true_indices = NearestNeighbors(n_neighbors=10).fit(data).kneighbors(
        queries, return_distance=False
    )
    recall = np.mean(
        [
            len(set(actual) & set(expected)) / 10
            for actual, expected in zip(outputs[0][0], true_indices)
        ]
    )
    assert recall >= 0.90

    if path == "cached":
        index._init_cached_search_function = lambda: False
        index._init_search_function()
        closure = index.query(query_data, k=10, epsilon=0.1)
        np.testing.assert_array_equal(closure[0], outputs[0][0])
        np.testing.assert_allclose(closure[1], outputs[0][1], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("sparse", [False, True])
def test_serial_query_preserves_sequential_rng(sparse):
    rng = np.random.RandomState(9)
    data = rng.normal(size=(64, 4)).astype(np.float32)
    queries = rng.normal(size=(12, 4)).astype(np.float32)
    index = NNDescent(
        csr_matrix(data) if sparse else data,
        n_neighbors=5,
        random_state=11,
        tree_init=False,
        parallel_batch_queries=False,
    )
    index.prepare()
    # With no tree or graph edges, the result consists solely of random seeds.
    index._search_graph = csr_matrix((64, 64), dtype=np.uint8)
    if sparse:
        index._init_sparse_search_function()
    else:
        index._init_search_function()
    actual_indices, actual_distances = index.query(
        csr_matrix(queries) if sparse else queries, k=5
    )

    state = index.search_rng_state.copy()
    for row, query in enumerate(queries):
        candidates = set()
        for _ in range(5):
            candidates.add(abs(int(tau_rand_int(state))) % len(data))
        expected = sorted(candidates, key=lambda i: np.linalg.norm(data[i] - query))
        np.testing.assert_array_equal(actual_indices[row, : len(expected)], expected)
        np.testing.assert_allclose(
            actual_distances[row, : len(expected)],
            np.linalg.norm(data[expected] - query, axis=1),
            rtol=1e-6,
        )
        assert np.all(np.isinf(actual_distances[row, len(expected) :]))
