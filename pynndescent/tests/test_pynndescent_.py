import os
import io
import re
import pathlib
import pytest
from contextlib import redirect_stdout

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
import pickle
import joblib
import scipy

from pynndescent import NNDescent, PyNNDescentTransformer
from pynndescent.distances import (
    alternative_dot,
    quantized_int8_alternative_cosine,
    quantized_int8_alternative_dot,
    quantized_int8_alternative_cosine_with_norms,
    quantized_int8_alternative_dot_with_norms,
    quantized_int8_direct_cosine,
    quantized_int8_direct_cosine_with_norms,
)
from pynndescent.pynndescent_ import _int8_code_inv_norms, _quantize_int8_rows
from pynndescent.kernels import select_kernel


def test_nn_descent_neighbor_accuracy(nn_data, seed):
    knn_indices, _ = NNDescent(
        nn_data, "euclidean", {}, 10, random_state=np.random.RandomState(seed)
    ).neighbor_graph

    tree = KDTree(nn_data)
    true_indices = tree.query(nn_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (nn_data.shape[0] * 10)
    assert (
        percent_correct >= 0.98
    ), "NN-descent did not get 99% accuracy on nearest neighbors"


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan"])
@pytest.mark.parametrize("use_init_graph", [False, True])
def test_neighbor_graph_has_single_self_loop(nn_data, seed, metric, use_init_graph):
    data = nn_data[:-2]  # drop the duplicated all-zero rows
    init_graph = None
    if use_init_graph:
        init_graph = NNDescent(
            data, metric=metric, n_neighbors=10, random_state=seed
        ).neighbor_graph[0]
    knn_indices, _ = NNDescent(
        data,
        metric=metric,
        n_neighbors=10,
        init_graph=init_graph,
        random_state=np.random.RandomState(seed),
    ).neighbor_graph

    assert np.all(knn_indices[:, 0] == np.arange(data.shape[0]))
    for row in knn_indices:
        assert np.unique(row).shape[0] == row.shape[0]


def test_angular_nn_descent_neighbor_accuracy(nn_data, seed):
    knn_indices, _ = NNDescent(
        nn_data, "cosine", {}, 10, random_state=np.random.RandomState(seed)
    ).neighbor_graph

    angular_data = normalize(nn_data, norm="l2")
    tree = KDTree(angular_data)
    true_indices = tree.query(angular_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (nn_data.shape[0] * 10)
    assert (
        percent_correct >= 0.98
    ), "NN-descent did not get 99% accuracy on nearest neighbors"


def test_bitpacked_nn_descent_neighbor_accuracy(nn_data, seed):
    bitpacked_data = (nn_data * 256).astype(np.uint8)
    unpacked_data = np.zeros(
        (bitpacked_data.shape[0], bitpacked_data.shape[1] * 8), dtype=np.float32
    )
    for i in range(unpacked_data.shape[0]):
        for j in range(unpacked_data.shape[1]):
            unpacked_data[i, j] = (bitpacked_data[i, j // 8] & (1 << (j % 8))) > 0

    knn_indices, _ = NNDescent(
        bitpacked_data, "bit_jaccard", {}, 10, random_state=np.random.RandomState(seed)
    ).neighbor_graph

    nn_finder = NearestNeighbors(n_neighbors=10, metric="jaccard").fit(unpacked_data)
    true_indices = nn_finder.kneighbors(unpacked_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (nn_data.shape[0] * 10)
    assert (
        percent_correct >= 0.60
    ), "NN-descent did not get 60% accuracy on nearest neighbors"


@pytest.mark.skipif(
    list(
        map(
            int,
            re.findall(r"[0-9]+\.[0-9]+\.?[0-9]*", scipy.version.version)[0].split("."),
        )
    )
    < [1, 3, 0],
    reason="requires scipy >= 1.3.0",
)
def test_sparse_nn_descent_neighbor_accuracy(sparse_nn_data, seed):
    knn_indices, _ = NNDescent(
        sparse_nn_data, "euclidean", n_neighbors=20, random_state=None
    ).neighbor_graph

    tree = KDTree(sparse_nn_data.toarray())
    true_indices = tree.query(sparse_nn_data.toarray(), 10, return_distance=False)

    num_correct = 0.0
    for i in range(sparse_nn_data.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (sparse_nn_data.shape[0] * 10)
    assert (
        percent_correct >= 0.85
    ), "Sparse NN-descent did not get 95% accuracy on nearest neighbors"


@pytest.mark.skipif(
    list(map(int, scipy.version.version.split("."))) < [1, 3, 0],
    reason="requires scipy >= 1.3.0",
)
def test_sparse_angular_nn_descent_neighbor_accuracy(sparse_nn_data):
    knn_indices, _ = NNDescent(
        sparse_nn_data, "cosine", {}, 20, random_state=None
    ).neighbor_graph

    angular_data = normalize(sparse_nn_data, norm="l2").toarray()
    tree = KDTree(angular_data)
    true_indices = tree.query(angular_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(sparse_nn_data.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (sparse_nn_data.shape[0] * 10)
    assert (
        percent_correct >= 0.85
    ), "Sparse angular NN-descent did not get 98% accuracy on nearest neighbors"


def test_nn_descent_query_accuracy(nn_data):
    nnd = NNDescent(nn_data[200:], "euclidean", n_neighbors=10, random_state=None)
    knn_indices, _ = nnd.query(nn_data[:200], k=10, epsilon=0.2)

    tree = KDTree(nn_data[200:])
    true_indices = tree.query(nn_data[:200], 10, return_distance=False)

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert (
        percent_correct >= 0.95
    ), "NN-descent query did not get 95% accuracy on nearest neighbors"


def test_nn_descent_query_accuracy_angular(nn_data):
    nnd = NNDescent(nn_data[200:], "cosine", n_neighbors=30, random_state=None)
    knn_indices, _ = nnd.query(nn_data[:200], k=10, epsilon=0.32)

    nn = NearestNeighbors(metric="cosine").fit(nn_data[200:])
    true_indices = nn.kneighbors(nn_data[:200], n_neighbors=10, return_distance=False)

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert (
        percent_correct >= 0.95
    ), "NN-descent query did not get 95% accuracy on nearest neighbors"


def test_dense_query_workspace_reuses_buffers_without_mutating_results(nn_data):
    index = NNDescent(nn_data[200:], "euclidean", n_neighbors=20, random_state=42)
    first_indices, first_distances = index.query(nn_data[:10], k=10, epsilon=0.2)
    expected_indices = first_indices.copy()
    expected_distances = first_distances.copy()
    workspace = index._query_workspace

    index.query(nn_data[10:20], k=10, epsilon=0.2)

    assert workspace is index._query_workspace
    np.testing.assert_array_equal(first_indices, expected_indices)
    np.testing.assert_allclose(first_distances, expected_distances)


def test_spill_tree_leaf_budget_dense_query(nn_data):
    index = NNDescent(
        nn_data[200:],
        "cosine",
        n_neighbors=20,
        search_tree_leaf_budget=2,
        random_state=42,
    )
    indices, distances = index.query(nn_data[:20], k=10, epsilon=0.2)

    assert indices.shape == (20, 10)
    assert distances.shape == (20, 10)
    assert np.all(indices >= 0)


def test_direct_cosine_and_indices_only_query(nn_data):
    index = NNDescent(nn_data[200:], "direct_cosine", n_neighbors=20, random_state=42)
    indices, distances = index.query(nn_data[:20], k=10, epsilon=0.2)
    indices_only = index.query_indices(nn_data[:20], k=10, epsilon=0.2)

    np.testing.assert_array_equal(indices_only, indices)
    assert np.all((distances >= 0.0) & (distances <= 1.0))

    normalized_queries = normalize(nn_data[:20], norm="l2")
    fast_indices, fast_distances = index.query(
        normalized_queries, k=10, epsilon=0.2, pre_normalized=True
    )
    np.testing.assert_array_equal(fast_indices, indices)
    np.testing.assert_allclose(fast_distances, distances, atol=3.0e-7)

    with pytest.raises(ValueError, match="pre_normalized"):
        NNDescent(nn_data[200:], "euclidean", n_neighbors=20, random_state=42).query(
            normalized_queries, k=10, pre_normalized=True
        )


def test_cosine_uses_normalized_dot_product(nn_data):
    train = nn_data[200:].astype(np.float32)
    original_train = train.copy()
    queries = nn_data[:20]
    index = NNDescent(train, "cosine", n_neighbors=20, random_state=42)

    assert index._distance_func is alternative_dot
    assert select_kernel(index._distance_func, index._raw_data) != 0
    np.testing.assert_array_equal(train, original_train)

    indices, distances = index.query(queries, k=10, epsilon=0.2)
    expected = cosine_distances(queries, train)
    np.testing.assert_allclose(
        distances, np.take_along_axis(expected, indices, 1), atol=1.0e-5
    )

    fast_indices, fast_distances = index.query(
        normalize(queries, norm="l2"), k=10, epsilon=0.2, pre_normalized=True
    )
    np.testing.assert_array_equal(fast_indices, indices)
    np.testing.assert_allclose(fast_distances, distances, atol=1.0e-6)

    graph_indices, graph_distances = index.neighbor_graph
    expected_graph = cosine_distances(train)
    nonzero_rows = np.linalg.norm(train, axis=1) > 0
    np.testing.assert_allclose(
        graph_distances[nonzero_rows],
        np.take_along_axis(expected_graph, graph_indices, 1)[nonzero_rows],
        atol=1.0e-5,
    )


def test_n_search_trees_is_deprecated(nn_data):
    with pytest.warns(FutureWarning, match="n_search_trees"):
        NNDescent(nn_data[:100], n_neighbors=5, n_search_trees=2)


@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_zero_angular_query_is_not_searched(nn_data, metric):
    index = NNDescent(nn_data[200:], metric, n_neighbors=20, random_state=42)
    index.query(nn_data[:10], k=10, epsilon=0.2)

    queries = nn_data[10:20].copy()
    queries[0] = 0.0
    indices, distances = index.query(queries, k=10, epsilon=0.2)

    # Row 0 must be empty (corrected infinite distance), not a stale result
    # left over in the reused workspace from the previous query.
    np.testing.assert_array_equal(distances[0], 1.0)
    assert np.all(distances[1:] < 1.0)


def test_int8_quantized_distance_self_similarity():
    codes = _quantize_int8_rows(
        np.array([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    )

    assert codes.dtype == np.int8
    assert quantized_int8_alternative_cosine(codes[0], codes[0]) == 0.0
    assert quantized_int8_alternative_dot(codes[0], codes[0]) == 0.0
    assert np.all(codes[1] == 0)


def test_int8_quantized_distances_use_precomputed_norms():
    codes = _quantize_int8_rows(
        np.array([[1.0, -2.0, 3.0], [4.0, 1.5, -2.0]], dtype=np.float32)
    )
    inv_norms = _int8_code_inv_norms(codes)

    np.testing.assert_allclose(
        quantized_int8_alternative_cosine(codes[0], codes[1]),
        quantized_int8_alternative_cosine_with_norms(
            codes[0], codes[1], inv_norms[0], inv_norms[1]
        ),
    )
    np.testing.assert_allclose(
        quantized_int8_alternative_dot(codes[0], codes[1]),
        quantized_int8_alternative_dot_with_norms(
            codes[0], codes[1], inv_norms[0], inv_norms[1]
        ),
    )
    similarity = np.dot(codes[0].astype(np.float32), codes[1].astype(np.float32))
    similarity *= inv_norms[0] * inv_norms[1]
    expected_direct_cosine = 1.0 if similarity <= 0.0 else 1.0 - similarity
    np.testing.assert_allclose(
        quantized_int8_direct_cosine_with_norms(
            codes[0], codes[1], inv_norms[0], inv_norms[1]
        ),
        expected_direct_cosine,
    )
    np.testing.assert_allclose(
        quantized_int8_direct_cosine(codes[0], codes[1]),
        expected_direct_cosine,
    )

    max_codes = np.full(784, 127, dtype=np.int8)
    max_inv_norm = np.float32(1.0 / np.sqrt(784 * 127 * 127))
    assert (
        abs(
            quantized_int8_alternative_dot_with_norms(
                max_codes, max_codes, max_inv_norm, max_inv_norm
            )
        )
        < 1.0e-5
    )


def test_int8_quantized_query_and_width_controls(nn_data):
    train = nn_data[200:]
    queries = nn_data[:50]
    index = NNDescent(
        train,
        "cosine",
        n_neighbors=30,
        random_state=42,
        quantization="int8",
    )

    indices, distances = index.query(
        queries, k=10, epsilon=0.2, candidate_width=40, rerank_width=20
    )
    proxy_indices, proxy_distances = index.query(
        queries, k=10, epsilon=0.2, candidate_width=40, rerank_width=0
    )

    assert indices.shape == (queries.shape[0], 10)
    assert distances.shape == (queries.shape[0], 10)
    assert proxy_indices.shape == (queries.shape[0], 10)
    assert proxy_distances.shape == (queries.shape[0], 10)
    assert np.all(indices >= 0)
    assert np.all(proxy_indices >= 0)

    with pytest.raises(ValueError, match="candidate_width"):
        index.query(queries, k=10, candidate_width=9)
    with pytest.raises(ValueError, match="rerank_width"):
        index.query(queries, k=10, candidate_width=20, rerank_width=5)

    index.query(queries, k=10)
    assert index._query_workspace[0].shape[1] == int(1.66 * 10)
    index.query(queries, k=10, proxy_beam_size=4)
    assert index._query_workspace[0].shape[1] == 40


def test_proxy_only_results_survive_later_queries(nn_data):
    index = NNDescent(
        nn_data[200:],
        "direct_cosine",
        n_neighbors=30,
        random_state=42,
        quantization="int8",
    )
    indices, distances = index.query(
        nn_data[:50], k=10, epsilon=0.2, candidate_width=20, rerank_width=0
    )
    expected_indices = indices.copy()
    expected_distances = distances.copy()

    index.query(nn_data[50:100], k=10, epsilon=0.2, candidate_width=20, rerank_width=0)

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(distances, expected_distances)


def test_direct_cosine_int8_quantized_query(nn_data):
    train = nn_data[200:]
    queries = nn_data[:50]
    normalized_train = normalize(train.copy(), norm="l2")
    index = NNDescent(
        train,
        "direct_cosine",
        n_neighbors=30,
        random_state=42,
        quantization="int8",
    )

    indices, distances = index.query(
        queries, k=10, epsilon=0.2, candidate_width=40, rerank_width=20
    )
    proxy_indices, proxy_distances = index.query(
        queries, k=10, epsilon=0.2, candidate_width=40, rerank_width=0
    )

    assert indices.shape == (queries.shape[0], 10)
    assert proxy_indices.shape == (queries.shape[0], 10)
    assert np.all((distances >= 0.0) & (distances <= 1.0))
    assert np.all((proxy_distances >= 0.0) & (proxy_distances <= 1.0))
    normalized_queries = normalize(queries, norm="l2")
    expected_distances = 1.0 - np.einsum(
        "ij,ikj->ik", normalized_queries, normalized_train[indices]
    )
    np.testing.assert_allclose(distances, expected_distances, rtol=1.0e-5, atol=3.0e-7)


def test_sparse_nn_descent_query_accuracy(sparse_nn_data):
    nnd = NNDescent(
        sparse_nn_data[200:], "euclidean", n_neighbors=15, random_state=None
    )
    knn_indices, _ = nnd.query(sparse_nn_data[:200], k=10, epsilon=0.24)

    tree = KDTree(sparse_nn_data[200:].toarray())
    true_indices = tree.query(sparse_nn_data[:200].toarray(), 10, return_distance=False)

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert (
        percent_correct >= 0.95
    ), "Sparse NN-descent query did not get 95% accuracy on nearest neighbors"


def test_sparse_nn_descent_query_accuracy_angular(sparse_nn_data):
    nnd = NNDescent(sparse_nn_data[200:], "cosine", n_neighbors=50, random_state=None)
    knn_indices, _ = nnd.query(sparse_nn_data[:200], k=10, epsilon=0.36)

    nn = NearestNeighbors(metric="cosine").fit(sparse_nn_data[200:].toarray())
    true_indices = nn.kneighbors(
        sparse_nn_data[:200].toarray(), n_neighbors=10, return_distance=False
    )

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert (
        percent_correct >= 0.95
    ), "Sparse NN-descent query did not get 95% accuracy on nearest neighbors"


def test_bitpacked_nn_descent_query_accuracy(nn_data):
    bitpacked_data = (nn_data * 256).astype(np.uint8)
    unpacked_data = np.zeros(
        (bitpacked_data.shape[0], bitpacked_data.shape[1] * 8), dtype=np.float32
    )
    for i in range(unpacked_data.shape[0]):
        for j in range(unpacked_data.shape[1]):
            unpacked_data[i, j] = (bitpacked_data[i, j // 8] & (1 << (j % 8))) > 0

    nnd = NNDescent(
        bitpacked_data[200:], "bit_jaccard", n_neighbors=50, random_state=None
    )
    knn_indices, _ = nnd.query(bitpacked_data[:200], k=10, epsilon=0.36)

    nn = NearestNeighbors(metric="jaccard").fit(unpacked_data[200:])
    true_indices = nn.kneighbors(
        unpacked_data[:200], n_neighbors=10, return_distance=False
    )

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert (
        percent_correct >= 0.80
    ), "Sparse NN-descent query did not get 95% accuracy on nearest neighbors"


def test_transformer_equivalence(nn_data):
    N_NEIGHBORS = 15
    EPSILON = 0.15
    train = nn_data[:400]
    test = nn_data[:200]

    # Note we shift N_NEIGHBORS to conform to sklearn's KNeighborTransformer defn
    nnd = NNDescent(
        data=train, n_neighbors=N_NEIGHBORS + 1, random_state=42, compressed=False
    )
    indices, dists = nnd.query(test, k=N_NEIGHBORS, epsilon=EPSILON)
    sort_idx = np.argsort(indices, axis=1)
    indices_sorted = np.vstack(
        [indices[i, sort_idx[i]] for i in range(sort_idx.shape[0])]
    )
    dists_sorted = np.vstack([dists[i, sort_idx[i]] for i in range(sort_idx.shape[0])])

    # Note we shift N_NEIGHBORS to conform to sklearn' KNeighborTransformer defn
    transformer = PyNNDescentTransformer(
        n_neighbors=N_NEIGHBORS, search_epsilon=EPSILON, random_state=42
    ).fit(train, compress_index=False)
    Xt = transformer.transform(test).sorted_indices()

    assert np.all(Xt.indices == indices_sorted.flatten())
    assert np.allclose(Xt.data, dists_sorted.flat)


def test_random_state_none(nn_data, spatial_data):
    knn_indices, _ = NNDescent(
        nn_data, "euclidean", {}, 10, random_state=None
    ).neighbor_graph

    tree = KDTree(nn_data)
    true_indices = tree.query(nn_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert (
        percent_correct >= 0.99
    ), "NN-descent did not get 99% accuracy on nearest neighbors"


def test_deterministic():
    seed = np.random.RandomState(42)

    x1 = seed.normal(0, 100, (1000, 50))
    x2 = seed.normal(0, 100, (1000, 50))

    index1 = NNDescent(x1, random_state=np.random.RandomState(42))
    neighbors1, distances1 = index1.query(x2)

    index2 = NNDescent(x1, random_state=np.random.RandomState(42))
    neighbors2, distances2 = index2.query(x2)

    np.testing.assert_equal(neighbors1, neighbors2)
    np.testing.assert_equal(distances1, distances2)


# This tests a recursion error on cosine metric reported at:
# https://github.com/lmcinnes/umap/issues/99
# graph_data used is a cut-down version of that provided by @scharron
# It contains lots of all-zero vectors and some other duplicates
def test_rp_trees_should_not_stack_overflow_with_duplicate_data(seed, cosine_hang_data):

    n_neighbors = 10
    knn_indices, _ = NNDescent(
        cosine_hang_data,
        "cosine",
        {},
        n_neighbors,
        random_state=np.random.RandomState(seed),
        n_trees=20,
    ).neighbor_graph

    for i in range(cosine_hang_data.shape[0]):
        assert len(knn_indices[i]) == len(
            np.unique(knn_indices[i])
        ), "Duplicate graph_indices in knn graph"


def test_deduplicated_data_behaves_normally(seed, cosine_hang_data):

    data = np.unique(cosine_hang_data, axis=0)
    data = data[~np.all(data == 0, axis=1)]
    data = data[:1000]

    n_neighbors = 10
    knn_indices, _ = NNDescent(
        data,
        "cosine",
        {},
        n_neighbors,
        random_state=np.random.RandomState(seed),
        n_trees=20,
    ).neighbor_graph

    for i in range(data.shape[0]):
        assert len(knn_indices[i]) == len(
            np.unique(knn_indices[i])
        ), "Duplicate graph_indices in knn graph"

    angular_data = normalize(data, norm="l2")
    tree = KDTree(angular_data)
    true_indices = tree.query(angular_data, n_neighbors, return_distance=False)

    num_correct = 0
    for i in range(data.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    proportion_correct = num_correct / (data.shape[0] * n_neighbors)
    assert (
        proportion_correct >= 0.95
    ), "NN-descent did not get 95% accuracy on nearest neighbors"


def test_rp_trees_should_not_stack_overflow_with_near_duplicate_data(
    seed, cosine_near_duplicates_data
):

    n_neighbors = 10
    knn_indices, _ = NNDescent(
        cosine_near_duplicates_data,
        "cosine",
        {},
        n_neighbors,
        random_state=np.random.RandomState(seed),
        n_trees=20,
    ).neighbor_graph

    for i in range(cosine_near_duplicates_data.shape[0]):
        assert len(knn_indices[i]) == len(
            np.unique(knn_indices[i])
        ), "Duplicate graph_indices in knn graph"


def test_output_when_verbose_is_true(spatial_data, seed):
    out = io.StringIO()
    with redirect_stdout(out):
        _ = NNDescent(
            data=spatial_data,
            metric="euclidean",
            metric_kwds={},
            n_neighbors=4,
            random_state=np.random.RandomState(seed),
            n_trees=5,
            n_iters=2,
            verbose=True,
        )
    output = out.getvalue()
    assert re.match("^.*5 trees", output, re.DOTALL)
    assert re.match("^.*2 iterations", output, re.DOTALL)


def test_no_output_when_verbose_is_false(spatial_data, seed):
    out = io.StringIO()
    with redirect_stdout(out):
        _ = NNDescent(
            data=spatial_data,
            metric="euclidean",
            metric_kwds={},
            n_neighbors=4,
            random_state=np.random.RandomState(seed),
            n_trees=5,
            n_iters=2,
            verbose=False,
        )
    output = out.getvalue().strip()
    assert len(output) == 0


# same as the previous two test, but this time using the PyNNDescentTransformer
# interface
def test_transformer_output_when_verbose_is_true(spatial_data, seed):
    out = io.StringIO()
    with redirect_stdout(out):
        _ = PyNNDescentTransformer(
            n_neighbors=4,
            metric="euclidean",
            metric_kwds={},
            random_state=np.random.RandomState(seed),
            n_trees=5,
            n_iters=2,
            verbose=True,
        ).fit_transform(spatial_data)
    output = out.getvalue()
    assert re.match("^.*5 trees", output, re.DOTALL)
    assert re.match("^.*2 iterations", output, re.DOTALL)


def test_transformer_output_when_verbose_is_false(spatial_data, seed):
    out = io.StringIO()
    with redirect_stdout(out):
        _ = PyNNDescentTransformer(
            n_neighbors=4,
            metric="standardised_euclidean",
            metric_kwds={"sigma": np.ones(spatial_data.shape[1])},
            random_state=np.random.RandomState(seed),
            n_trees=5,
            n_iters=2,
            verbose=False,
        ).fit_transform(spatial_data)
    output = out.getvalue().strip()
    assert len(output) == 0


def test_pickle_unpickle():
    seed = np.random.RandomState(42)

    x1 = seed.normal(0, 100, (1000, 50))
    x2 = seed.normal(0, 100, (1000, 50))

    index1 = NNDescent(x1, "euclidean", {}, 10, random_state=None)
    neighbors1, distances1 = index1.query(x2)

    mem_temp = io.BytesIO()
    pickle.dump(index1, mem_temp)
    mem_temp.seek(0)
    index2 = pickle.load(mem_temp)

    neighbors2, distances2 = index2.query(x2)

    np.testing.assert_equal(neighbors1, neighbors2)
    np.testing.assert_equal(distances1, distances2)


@pytest.mark.parametrize("quantization", [None, "int8"])
def test_cosine_pickle_unpickle(quantization):
    seed = np.random.RandomState(42)
    x1 = seed.normal(0, 1, (1000, 32))
    x2 = seed.normal(0, 1, (100, 32))

    index1 = NNDescent(
        x1, "cosine", n_neighbors=15, quantization=quantization, random_state=1
    )
    neighbors1, distances1 = index1.query(x2)
    index2 = pickle.loads(pickle.dumps(index1))
    neighbors2, distances2 = index2.query(x2)

    np.testing.assert_equal(neighbors1, neighbors2)
    np.testing.assert_allclose(distances1, distances2, atol=1.0e-6)


def test_compressed_pickle_unpickle():
    seed = np.random.RandomState(42)

    x1 = seed.normal(0, 100, (1000, 50))
    x2 = seed.normal(0, 100, (1000, 50))

    index1 = NNDescent(x1, "euclidean", {}, 10, random_state=None, compressed=True)
    neighbors1, distances1 = index1.query(x2)

    mem_temp = io.BytesIO()
    pickle.dump(index1, mem_temp)
    mem_temp.seek(0)
    index2 = pickle.load(mem_temp)

    neighbors2, distances2 = index2.query(x2)

    np.testing.assert_equal(neighbors1, neighbors2)
    np.testing.assert_equal(distances1, distances2)


def test_transformer_pickle_unpickle():
    seed = np.random.RandomState(42)

    x1 = seed.normal(0, 100, (1000, 50))
    x2 = seed.normal(0, 100, (1000, 50))

    index1 = PyNNDescentTransformer(n_neighbors=10).fit(x1)
    result1 = index1.transform(x2)

    mem_temp = io.BytesIO()
    pickle.dump(index1, mem_temp)
    mem_temp.seek(0)
    index2 = pickle.load(mem_temp)

    result2 = index2.transform(x2)

    np.testing.assert_equal(result1.indices, result2.indices)
    np.testing.assert_equal(result1.data, result2.data)


def test_joblib_dump():
    seed = np.random.RandomState(42)

    x1 = seed.normal(0, 100, (1000, 50))
    x2 = seed.normal(0, 100, (1000, 50))

    index1 = NNDescent(x1, "euclidean", {}, 10, random_state=None)
    neighbors1, distances1 = index1.query(x2)

    mem_temp = io.BytesIO()
    joblib.dump(index1, mem_temp)
    mem_temp.seek(0)
    index2 = joblib.load(mem_temp)

    neighbors2, distances2 = index2.query(x2)

    np.testing.assert_equal(neighbors1, neighbors2)
    np.testing.assert_equal(distances1, distances2)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_update_no_prepare_query_accuracy(nn_data, metric):
    nnd = NNDescent(nn_data[200:800], metric=metric, n_neighbors=10, random_state=None)
    nnd.update(xs_fresh=nn_data[800:])

    knn_indices, _ = nnd.query(nn_data[:200], k=10, epsilon=0.2)

    true_nnd = NearestNeighbors(metric=metric).fit(nn_data[200:])
    true_indices = true_nnd.kneighbors(nn_data[:200], 10, return_distance=False)

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert percent_correct >= 0.95, (
        "NN-descent query did not get 95% " "accuracy on nearest neighbors"
    )


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_update_w_prepare_query_accuracy(nn_data, metric):
    nnd = NNDescent(
        nn_data[200:800],
        metric=metric,
        n_neighbors=10,
        random_state=None,
        compressed=False,
    )
    nnd.prepare()

    nnd.update(xs_fresh=nn_data[800:])
    nnd.prepare()

    knn_indices, _ = nnd.query(nn_data[:200], k=10, epsilon=0.2)

    true_nnd = NearestNeighbors(metric=metric).fit(nn_data[200:])
    true_indices = true_nnd.kneighbors(nn_data[:200], 10, return_distance=False)

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert percent_correct >= 0.95, (
        "NN-descent query did not get 95% " "accuracy on nearest neighbors"
    )


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_update_w_prepare_query_accuracy(nn_data, metric):
    nnd = NNDescent(
        nn_data[200:800],
        metric=metric,
        n_neighbors=10,
        random_state=None,
        compressed=False,
    )
    nnd.prepare()

    nnd.update(xs_fresh=nn_data[800:])
    nnd.prepare()

    knn_indices, _ = nnd.query(nn_data[:200], k=10, epsilon=0.2)

    true_nnd = NearestNeighbors(metric=metric).fit(nn_data[200:])
    true_indices = true_nnd.kneighbors(nn_data[:200], 10, return_distance=False)

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert percent_correct >= 0.95, (
        "NN-descent query did not get 95% " "accuracy on nearest neighbors"
    )


def evaluate_predictions(neighbors_true, neigbhors_computed, n_neighbors):
    n_correct = 0
    n_all = neighbors_true.shape[0] * n_neighbors
    for i in range(neighbors_true.shape[0]):
        n_correct += np.sum(np.isin(neighbors_true[i], neigbhors_computed[i]))
    return n_correct / n_all


@pytest.mark.parametrize("metric", ["manhattan", "euclidean", "cosine"])
@pytest.mark.parametrize("case", list(range(8)))  # the number of cases in update_data
def test_update_with_changed_data(update_data, case, metric):
    def evaluate(nn_descent, xs_to_fit, xs_to_query):
        true_nn = NearestNeighbors(metric=metric, n_neighbors=k).fit(xs_to_fit)
        neighbors, _ = nn_descent.query(xs_to_query, k=k)
        neighbors_expected = true_nn.kneighbors(xs_to_query, k, return_distance=False)
        p_correct = evaluate_predictions(neighbors_expected, neighbors, k)
        assert p_correct >= 0.95, (
            "NN-descent query did not get 95% " "accuracy on nearest neighbors"
        )

    k = 10
    xs_orig, xs_fresh, xs_updated, indices_updated = update_data[case]
    queries1 = xs_orig

    # original
    index = NNDescent(xs_orig, metric=metric, n_neighbors=40, random_state=1234)
    index.prepare()
    evaluate(index, xs_orig, queries1)
    # updated
    index.update(
        xs_fresh=xs_fresh, xs_updated=xs_updated, updated_indices=indices_updated
    )
    if xs_fresh is not None:
        xs = np.vstack((xs_orig, xs_fresh))
        queries2 = np.vstack((queries1, xs_fresh))
    else:
        xs = xs_orig
        queries2 = queries1
    if indices_updated is not None:
        xs[indices_updated] = xs_updated
    evaluate(index, xs, queries2)
    if indices_updated is not None:
        evaluate(index, xs, xs_updated)


def test_init_forest_respects_n_jobs_and_is_used_by_update(nn_data, monkeypatch):
    import numba
    import pynndescent.pynndescent_ as nnd_module

    thread_counts = []
    original = nnd_module.make_hybrid_leaf_array

    def spy(*args, **kwargs):
        thread_counts.append(numba.get_num_threads())
        return original(*args, **kwargs)

    monkeypatch.setattr(nnd_module, "make_hybrid_leaf_array", spy)
    threads_before = numba.get_num_threads()

    index = NNDescent(nn_data[:600], n_neighbors=10, n_jobs=1, random_state=42)
    assert thread_counts == [1]
    assert numba.get_num_threads() == threads_before

    index.update(xs_fresh=nn_data[600:])
    assert len(thread_counts) == 2


@pytest.mark.parametrize("n_trees", [1, 2, 3, 10])
def test_tree_numbers_after_multiple_updates(n_trees):
    trees_after_update = max(2, int(np.round(n_trees / 3)))

    nnd = NNDescent(np.array([[1.0]]), n_neighbors=1, n_trees=n_trees)

    assert nnd.n_trees == n_trees, "NN-descent update changed the number of trees"
    assert (
        nnd.n_trees_after_update == trees_after_update
    ), "The value of the n_trees_after_update in NN-descent after update(s) is wrong"
    for i in range(5):
        nnd.update(xs_fresh=np.array([[i]], dtype=np.float64))
        assert (
            nnd.n_trees == trees_after_update
        ), "The value of the n_trees in NN-descent after update(s) is wrong"
        assert (
            nnd.n_trees_after_update == trees_after_update
        ), "The value of the n_trees_after_update in NN-descent after update(s) is wrong"


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_tree_init_false(nn_data, metric):
    nnd = NNDescent(
        nn_data[200:], metric=metric, n_neighbors=10, random_state=None, tree_init=False
    )
    nnd.prepare()

    knn_indices, _ = nnd.query(nn_data[:200], k=10, epsilon=0.2)

    true_nnd = NearestNeighbors(metric=metric).fit(nn_data[200:])
    true_indices = true_nnd.kneighbors(nn_data[:200], 10, return_distance=False)

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert percent_correct >= 0.95, (
        "NN-descent query did not get 95% " "accuracy on nearest neighbors"
    )


@pytest.mark.parametrize(
    "metric", ["euclidean", "manhattan"]
)  # cosine makes no sense for 1D
def test_one_dimensional_data(nn_data, metric):
    nnd = NNDescent(
        nn_data[200:, :1],
        metric=metric,
        n_neighbors=20,
        random_state=None,
        tree_init=False,
    )
    nnd.prepare()

    knn_indices, _ = nnd.query(nn_data[:200, :1], k=10, epsilon=0.2)

    true_nnd = NearestNeighbors(metric=metric).fit(nn_data[200:, :1])
    true_indices = true_nnd.kneighbors(nn_data[:200, :1], 10, return_distance=False)

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert percent_correct >= 0.95, (
        "NN-descent query did not get 95% " "accuracy on nearest neighbors"
    )


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_tree_no_split(small_data, sparse_small_data, metric):
    k = 10
    for data, data_type in zip([small_data, sparse_small_data], ["dense", "sparse"]):
        n_instances = data.shape[0]
        leaf_size = n_instances + 1  # just to be safe
        data_train = data[n_instances // 2 :]
        data_test = data[: n_instances // 2]

        nnd = NNDescent(
            data_train,
            metric=metric,
            n_neighbors=data_train.shape[0] - 1,
            random_state=None,
            tree_init=True,
            leaf_size=leaf_size,
        )
        nnd.prepare()
        knn_indices, _ = nnd.query(data_test, k=k, epsilon=0.2)

        true_nnd = NearestNeighbors(metric=metric).fit(data_train)
        true_indices = true_nnd.kneighbors(data_test, k, return_distance=False)

        num_correct = 0.0
        for i in range(true_indices.shape[0]):
            num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

        percent_correct = num_correct / (true_indices.shape[0] * k)
        assert (
            percent_correct >= 0.95
        ), "NN-descent query did not get 95% for accuracy on nearest neighbors on {} data".format(
            data_type
        )


@pytest.mark.skipif(
    "NUMBA_DISABLE_JIT" in os.environ, reason="Too expensive for disabled Numba"
)
def test_bad_data():
    test_data_dir = pathlib.Path(__file__).parent / "test_data"
    data = np.sqrt(np.load(test_data_dir / "pynndescent_bug_np.npz")["arr_0"])
    index = NNDescent(data, metric="cosine")
