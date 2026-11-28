import os
import subprocess
import sys
import warnings
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


@pytest.mark.parametrize(
    "sparse_type", [scipy.sparse.csr_matrix, scipy.sparse.csr_array]
)
@pytest.mark.parametrize(
    "representation,metric",
    [
        ("duplicates", "euclidean"),
        ("stored_zeros", "jaccard"),
        ("unsorted", "euclidean"),
    ],
)
@pytest.mark.parametrize("entry_point", ["training", "fast_query", "fallback_query"])
def test_sparse_storage_preserves_distances(
    sparse_type, representation, metric, entry_point
):
    data = np.vstack((np.eye(6), 1 - np.eye(6))).astype(np.float32)
    canonical = sparse_type(data)
    if representation == "duplicates":
        supplied = sparse_type(
            (
                np.repeat(canonical.data / 2, 2),
                np.repeat(canonical.indices, 2),
                canonical.indptr * 2,
            ),
            shape=data.shape,
        )
    elif representation == "unsorted":
        order = np.concatenate(
            [
                np.arange(stop - 1, start - 1, -1)
                for start, stop in zip(canonical.indptr[:-1], canonical.indptr[1:])
            ]
        )
        supplied = sparse_type(
            (canonical.data[order], canonical.indices[order], canonical.indptr.copy()),
            shape=data.shape,
        )
    else:
        supplied = sparse_type(
            (
                data.ravel().copy(),
                np.tile(np.arange(6), len(data)),
                np.arange(0, data.size + 1, 6),
            ),
            shape=data.shape,
        )
    query_rows = np.arange(6, 9)
    if entry_point != "training":
        supplied = supplied[query_rows]
    saved_storage = tuple(
        array.copy() for array in (supplied.data, supplied.indices, supplied.indptr)
    )
    index = NNDescent(
        supplied if entry_point == "training" else canonical,
        metric=metric,
        n_neighbors=6,
        n_trees=0,
        n_iters=1,
        init_graph=np.tile(np.arange(len(data), dtype=np.int32), (len(data), 1)),
        random_state=42,
        n_jobs=1,
    )
    expected = scipy.spatial.distance.cdist(data, data, metric=metric)
    if entry_point == "training":
        indices, distances = index.neighbor_graph
    else:
        options = {"candidate_width": 4} if entry_point == "fallback_query" else {}
        indices, distances = index.query(supplied, k=4, **options)
    assert np.all((indices >= 0) & (indices < len(data)))
    np.testing.assert_allclose(
        distances,
        expected[
            (np.arange(len(data)) if entry_point == "training" else query_rows)[
                :, None
            ],
            indices,
        ],
        atol=1e-6,
    )
    for current, saved in zip(
        (supplied.data, supplied.indices, supplied.indptr), saved_storage
    ):
        np.testing.assert_array_equal(current, saved)


def test_haversine_antipodal_neighbor_graph():
    data = np.deg2rad(
        np.array(
            [[-89, 15], [-85, 15], [-70, 15], [89, -165], [85, -165], [70, -165]],
            dtype=np.float32,
        )
    )
    init_graph = np.tile(np.arange(len(data), dtype=np.int32), (len(data), 1))
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        index = NNDescent(
            data,
            metric="haversine",
            n_neighbors=len(data),
            n_trees=0,
            n_iters=2,
            init_graph=init_graph,
            n_jobs=1,
            random_state=0,
        )
    assert not any(
        "Failed to correctly find n_neighbors" in str(warning.message)
        for warning in caught_warnings
    )

    indices, distances = index.neighbor_graph
    assert np.isfinite(distances).all()
    np.testing.assert_array_equal(np.sort(indices, axis=1), init_graph)
    np.testing.assert_allclose(distances[:, -1], np.pi, rtol=0, atol=1e-6)



def test_transformer_no_warning_when_n_neighbors_equals_n_samples():
    data = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        dtype=np.float32,
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        graph = PyNNDescentTransformer(
            n_neighbors=len(data) - 1,
            random_state=0,
            n_jobs=1,
        ).fit_transform(data)

    assert not any(
        "Failed to correctly find n_neighbors" in str(warning.message)
        for warning in caught_warnings
    )
    assert graph.shape == (len(data), len(data))
    assert graph.nnz == len(data) * len(data)


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


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_sparse_search_tree_hyperplanes_are_ragged(sparse_nn_data, metric):
    index = NNDescent(sparse_nn_data, metric, n_neighbors=20, random_state=42)
    index.prepare()
    tree = index._search_forest[0]
    ptr, inds, vals = tree.hyperplanes
    sizes = np.diff(ptr)
    is_leaf = tree.children[:, 0] <= 0

    # Hyperplanes are stored without padding: leaves have none, internal nodes
    # exactly their non-zeros (padding used to make sparse trees GiBs large)
    assert ptr.shape[0] == tree.children.shape[0] + 1
    assert np.all(sizes[is_leaf] == 0)
    assert np.all(sizes[~is_leaf] > 0)
    assert inds.shape == vals.shape == (ptr[-1],)
    assert np.all((inds >= 0) & (inds < sparse_nn_data.shape[1]))
    for node in np.flatnonzero(~is_leaf):
        assert np.all(np.diff(inds[ptr[node] : ptr[node + 1]]) > 0)


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


@pytest.mark.parametrize("parallel_batch_queries", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1e-3])
def test_proxy_inner_product_search_ranks_by_inner_product(
    scale, parallel_batch_queries
):
    # Varied norms, as in attention keys: best angles are not best inner products
    rng = np.random.default_rng(0)
    data = rng.normal(size=(300, 8)) * rng.uniform(0.2, 3.0, size=(300, 1))
    data = (data * scale).astype(np.float32)
    queries = rng.normal(size=(20, 8)).astype(np.float32)
    index = NNDescent(
        data,
        "proxy_inner_product",
        n_neighbors=15,
        random_state=42,
        parallel_batch_queries=parallel_batch_queries,
    )

    indices, _ = index.query(queries, k=10, epsilon=0.5, rerank_width=0)

    expected = np.argsort(-(queries @ data.T), axis=1)[:, :10]
    np.testing.assert_array_equal(indices, expected)


def _mips_data():
    # Varied norms, so that the largest inner products are not the best angles
    rng = np.random.default_rng(0)
    data = rng.normal(size=(300, 8)) * rng.uniform(0.2, 3.0, size=(300, 1))
    queries = rng.normal(size=(20, 8))
    return data.astype(np.float32), queries.astype(np.float32)


@pytest.mark.parametrize("quantization", [None, "sq8", "sq4"])
@pytest.mark.parametrize("parallel_batch_queries", [False, True])
def test_mips_query_ranks_by_inner_product(quantization, parallel_batch_queries):
    data, queries = _mips_data()
    index = NNDescent(
        data,
        "mips",
        n_neighbors=15,
        random_state=42,
        quantization=quantization,
        parallel_batch_queries=parallel_batch_queries,
    )

    indices, distances = index.query(queries, k=10, epsilon=0.5, candidate_width=60)

    inner_products = queries @ data.T
    expected = np.argsort(-inner_products, axis=1)[:, :10]
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(
        distances, -np.take_along_axis(inner_products, indices, 1), atol=1e-4
    )
    np.testing.assert_array_equal(
        index.query_indices(queries, k=10, epsilon=0.5, candidate_width=60), indices
    )
    restored = pickle.loads(pickle.dumps(index))
    np.testing.assert_array_equal(
        restored.query(queries, k=10, epsilon=0.5, candidate_width=60)[0], indices
    )


def _mips_recall(indices, queries, data):
    expected = np.argsort(-(queries @ data.T), axis=1)[:, : indices.shape[1]]
    return np.mean(
        [len(set(a) & set(b)) / indices.shape[1] for a, b in zip(indices, expected)]
    )


@pytest.mark.parametrize("quantization", ["sq8", "sq4"])
@pytest.mark.parametrize("integer_query", [True, False])
@pytest.mark.parametrize("rerank_precision", ["float32", "float16"])
def test_mips_quantized_search(quantization, integer_query, rerank_precision):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2000, 32)) * rng.uniform(0.2, 3.0, size=(2000, 1))
    data = data.astype(np.float32)
    queries = rng.normal(size=(30, 32)).astype(np.float32)
    index = NNDescent(
        data,
        "mips",
        n_neighbors=15,
        random_state=42,
        quantization=quantization,
        rerank_precision=rerank_precision,
    )
    index._sq_integer_query = integer_query

    indices, distances = index.query(queries, k=10, epsilon=0.1, candidate_width=100)

    # Codes cover only the original features, not the appended one
    assert index._sq_offsets.shape[0] == data.shape[1]
    assert _mips_recall(indices, queries, data) >= 0.95
    atol = 0.05 if rerank_precision == "float16" else 1e-3
    np.testing.assert_allclose(
        distances, -np.take_along_axis(queries @ data.T, indices, 1), atol=atol
    )


def test_mips_quantized_search_with_projection_and_query_sample():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2000, 32)) * rng.uniform(0.2, 3.0, size=(2000, 1))
    data = data.astype(np.float32)
    queries = rng.normal(size=(30, 32)).astype(np.float32)
    sample = rng.normal(size=(300, 32)).astype(np.float32)
    index = NNDescent(
        data,
        "mips",
        n_neighbors=15,
        random_state=42,
        quantization="sq8",
        quantization_dims=24,
    )
    index.prepare(query_sample=sample)

    indices = index.query_indices(queries, k=10, epsilon=0.1, candidate_width=150)

    assert index._sq_codes.shape[0] == data.shape[0]
    assert _mips_recall(indices, queries, data) >= 0.9


def test_mips_has_no_neighbor_graph():
    # Its k-NN graph is built on the augmented data, so does not rank by inner product
    data, _ = _mips_data()
    index = NNDescent(data, "mips", n_neighbors=15, random_state=42)
    with pytest.raises(NotImplementedError):
        index.neighbor_graph


def test_mips_query_sample_and_update():
    data, queries = _mips_data()
    expected = np.argsort(-(queries @ data.T), axis=1)[:, :10]
    index = NNDescent(data[:250], "mips", n_neighbors=15, random_state=42)
    with pytest.raises(ValueError, match="larger norm"):
        index.update(xs_fresh=data[250:] * 100)

    # Only points that do not exceed the indexed norms can be added
    fresh = data[250:] * 0.1
    index.update(xs_fresh=fresh)
    combined = np.vstack([data[:250], fresh])
    combined_expected = np.argsort(-(queries @ combined.T), axis=1)[:, :10]
    np.testing.assert_array_equal(
        index.query(queries, k=10, epsilon=0.5)[0], combined_expected
    )

    rng = np.random.default_rng(1)
    sample = rng.normal(size=(100, 8)).astype(np.float32)
    adapted = NNDescent(data, "mips", n_neighbors=15, random_state=42)
    adapted.prepare(query_sample=sample)
    np.testing.assert_array_equal(adapted.query(queries, k=10, epsilon=0.5)[0], expected)


def test_mips_rejects_sparse_data(sparse_nn_data):
    with pytest.raises(ValueError, match="dense"):
        NNDescent(sparse_nn_data, "mips", n_neighbors=10)


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


def test_cosine_distances_are_not_negative():
    # Duplicate rows give a cosine a hair above 1 in float32, which came out as
    # a distance of about -2e-7 and made sklearn reject the graph (#214).
    rng = np.random.default_rng(0)
    data = np.repeat(rng.normal(size=(300, 50)).astype(np.float32), 10, axis=0)
    index = NNDescent(data, "cosine", n_neighbors=10, random_state=42)

    assert index.neighbor_graph[1].min() >= 0.0
    assert index.query(data[:500], k=10)[1].min() >= 0.0
    graph = PyNNDescentTransformer(metric="cosine", random_state=42).fit_transform(data)
    assert graph.data.min() >= 0.0


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


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "correlation"])
def test_sparse_pickle_unpickle(sparse_nn_data, metric):
    index1 = NNDescent(sparse_nn_data, metric, n_neighbors=10, random_state=42)
    neighbors1, distances1 = index1.query(sparse_nn_data[:20], k=5)
    index2 = pickle.loads(pickle.dumps(index1))
    neighbors2, distances2 = index2.query(sparse_nn_data[:20], k=5)

    np.testing.assert_equal(neighbors1, neighbors2)
    np.testing.assert_allclose(distances1, distances2, atol=1.0e-6)


def test_padded_sparse_search_tree_from_older_pickles(sparse_nn_data):
    from pynndescent.rp_trees import FlatTree

    index = NNDescent(sparse_nn_data, "cosine", n_neighbors=10, random_state=42)
    neighbors1, distances1 = index.query(sparse_nn_data[:20], k=5)

    # Older indexes stored sparse hyperplanes as (n_nodes, 2, width) arrays of
    # (indices, values) padded with -1
    ptr, inds, vals = index._search_forest[0].hyperplanes
    width = max(1, int(np.diff(ptr).max()))
    padded = np.zeros((ptr.shape[0] - 1, 2, width), dtype=np.float32)
    padded[:, 0, :] = -1
    for node in range(ptr.shape[0] - 1):
        size = ptr[node + 1] - ptr[node]
        padded[node, 0, :size] = inds[ptr[node] : ptr[node + 1]]
        padded[node, 1, :size] = vals[ptr[node] : ptr[node + 1]]
    state = index.__getstate__()
    old_tree = state["_search_forest"][0]
    state["_search_forest"] = (FlatTree(padded, *old_tree[1:]),)

    restored = NNDescent.__new__(NNDescent)
    restored.__setstate__(state)
    neighbors2, distances2 = restored.query(sparse_nn_data[:20], k=5)
    assert isinstance(restored._search_forest[0].hyperplanes, tuple)
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


@pytest.mark.parametrize(
    "sparse_type", [scipy.sparse.csr_matrix, scipy.sparse.csr_array]
)
@pytest.mark.parametrize("prepared", [False, True])
def test_sparse_update_failure_preserves_index(sparse_type, prepared):
    random_state = np.random.RandomState(42)
    data = np.arange(48, dtype=np.float32).reshape(12, 4) + 1
    index = NNDescent(
        sparse_type(data),
        n_neighbors=4,
        n_trees=0,
        n_iters=1,
        random_state=random_state,
        n_jobs=1,
    )
    if prepared:
        index.prepare()
    saved_random_state = random_state.get_state()
    raw_data = index._raw_data
    saved_data = raw_data.copy()
    saved_graph = tuple(array.copy() for array in index._neighbor_graph)
    for _ in range(2):
        with pytest.raises(NotImplementedError, match="Sparse update not complete yet"):
            index.update(xs_fresh=sparse_type([[2, 3, 4, 5]], dtype=np.float32))
        for current, saved in zip(random_state.get_state(), saved_random_state):
            np.testing.assert_equal(current, saved)
        assert index._raw_data is raw_data
        np.testing.assert_array_equal(index._raw_data.toarray(), saved_data.toarray())
        for current, saved in zip(index._neighbor_graph, saved_graph):
            np.testing.assert_array_equal(current, saved)
    indices, distances = index.query(sparse_type(data[:1]), k=1)
    np.testing.assert_array_equal(indices, [[0]])
    np.testing.assert_allclose(distances, [[0.0]], atol=1e-6)


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


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("mode", ["append", "replace", "both"])
def test_dot_update_distances(mode, prepared):
    rng = np.random.RandomState(42)
    data = rng.lognormal(size=(80, 8)).astype(np.float32)
    data[0] = 0
    original = data.copy()
    index = NNDescent(data[:48], metric="dot", n_neighbors=10, random_state=42)
    if prepared:
        index.prepare()

    fresh = data[48:64] if mode != "replace" else None
    updated = data[64:] if mode != "append" else None
    positions = np.arange(0, 48, 3) if updated is not None else None
    index.update(xs_fresh=fresh, xs_updated=updated, updated_indices=positions)
    np.testing.assert_array_equal(data, original)

    expected_data = data[:48].copy()
    if updated is not None:
        expected_data[positions] = updated
    if fresh is not None:
        expected_data = np.vstack((expected_data, fresh))

    normalized = normalize(expected_data.astype(np.float64))
    expected = 1 - np.maximum(normalized @ normalized.T, 0)
    # neighbor_graph lists each point as its own first neighbor at distance 0,
    # including the all-zero row (whose dot distance to everything else is 1)
    np.fill_diagonal(expected, 0.0)
    indices, distances = index.neighbor_graph
    assert np.all(indices >= 0)
    np.testing.assert_allclose(
        distances, expected[np.arange(len(indices))[:, None], indices], atol=1e-6
    )

    indices, distances = index.query(data[16:32], k=5)
    expected = 1 - np.maximum(
        normalize(data[16:32].astype(np.float64)) @ normalized.T, 0
    )
    np.testing.assert_allclose(
        distances, expected[np.arange(len(indices))[:, None], indices], atol=1e-6
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


def test_graph_update_ids_survive_float32_buffers():
    import numba
    from pynndescent.utils import (
        int32_bits_as_float32,
        float32_bits_as_int32,
        sort_updates_by_block,
    )

    # Ids above 2**24 are not exactly representable as float32 values.
    ids = np.array(
        [0, 1, 2**23 - 1, 2**24, 2**24 + 1, 2**25 + 7, 2**31 - 2], dtype=np.int32
    )

    @numba.njit()
    def encode(ids, buf):
        for j in range(ids.shape[0]):
            buf[0, j, 0] = int32_bits_as_float32(ids[j])
            buf[0, j, 1] = int32_bits_as_float32(ids[(j + 1) % ids.shape[0]])
            buf[0, j, 2] = np.float32(j)

    @numba.njit()
    def decode(buf, n, out):
        for j in range(n):
            out[j, 0] = float32_bits_as_int32(buf[0, j, 0])
            out[j, 1] = float32_bits_as_int32(buf[0, j, 1])

    n = ids.shape[0]
    expected = set(zip(ids, np.roll(ids, -1)))
    updates = np.zeros((1, n, 3), dtype=np.float32)
    encode(ids, updates)

    decoded = np.zeros((n, 2), dtype=np.int32)
    decode(updates, n, decoded)
    assert set(map(tuple, decoded)) == expected

    sorted_updates = np.zeros((1, 2 * n, 3), dtype=np.float32)
    block_offsets = np.zeros((1, 2), dtype=np.int64)
    sort_updates_by_block(
        updates, np.array([n], dtype=np.int32), sorted_updates, block_offsets, 2**31 - 1, 1
    )
    decode(sorted_updates, n, decoded)
    assert set(map(tuple, decoded)) == expected


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


@pytest.mark.parametrize(
    "metric,kwargs",
    [
        ("euclidean", {}),
        ("cosine", {}),
        ("euclidean", {"search_tree_leaf_budget": 4}),
        ("euclidean", {"parallel_batch_queries": True}),
        ("proxy_inner_product", {}),
    ],
)
def test_cached_search_matches_closure(nn_data, metric, kwargs):
    # The disk-cached dense search function must return exactly what the
    # per-index search closure returns.
    index = NNDescent(
        nn_data[:600], metric=metric, n_neighbors=10, random_state=42, **kwargs
    )
    index.prepare(compile_search=False)
    assert index._uses_cached_search  # cached path in use
    cached_indices, cached_dists = index.query(nn_data[600:700], k=8, epsilon=0.1)

    # Build the closure for the same index and compare
    index._init_cached_search_function = lambda: False
    index._init_search_function()
    assert not index._uses_cached_search
    closure_indices, closure_dists = index.query(nn_data[600:700], k=8, epsilon=0.1)

    np.testing.assert_array_equal(cached_indices, closure_indices)
    np.testing.assert_allclose(cached_dists, closure_dists, rtol=1e-6)


def test_proxy_cached_search_matches_closure_without_rerank(nn_data):
    index = NNDescent(
        nn_data[:300], metric="proxy_inner_product", n_neighbors=10, random_state=42
    )
    index.prepare(compile_search=False)
    assert index._uses_cached_search
    queries = nn_data[300:350]
    cached = index.query(queries, k=8, epsilon=0.1, rerank_width=0)

    index._init_cached_search_function = lambda: False
    index._init_search_function()
    assert not index._uses_cached_search
    closure = index.query(queries, k=8, epsilon=0.1, rerank_width=0)

    np.testing.assert_array_equal(cached[0], closure[0])
    np.testing.assert_array_equal(cached[1], closure[1])


def test_query_rejects_wrong_shape(nn_data):
    index = NNDescent(nn_data[:300], n_neighbors=10, random_state=42)
    index.prepare()
    with pytest.raises(ValueError):
        index.query(nn_data[0], k=5)  # 1-D
    with pytest.raises(ValueError):
        index.query(nn_data[:5, :-1], k=5)  # wrong number of features


def test_prepare_compile_search_option(nn_data):
    index = NNDescent(nn_data[:600], n_neighbors=10, random_state=42)
    index.prepare(compile_search=True)
    assert not index._uses_cached_search  # compiled, specialised on request
    compiled_indices, compiled_dists = index.query(nn_data[600:650], k=8, epsilon=0.1)

    # The shared, cached search function is the default
    index.prepare()
    assert index._uses_cached_search
    cached_indices, cached_dists = index.query(nn_data[600:650], k=8, epsilon=0.1)
    np.testing.assert_array_equal(cached_indices, compiled_indices)
    np.testing.assert_allclose(cached_dists, compiled_dists, rtol=1e-6)

    # The choice survives pickling
    restored = pickle.loads(pickle.dumps(index))
    assert restored._uses_cached_search
    restored_indices, _ = restored.query(nn_data[600:650], k=8, epsilon=0.1)
    np.testing.assert_array_equal(restored_indices, cached_indices)

    # Querying before prepare() also uses the cached search function
    fresh = NNDescent(nn_data[:600], n_neighbors=10, random_state=42)
    fresh.query(nn_data[600:610], k=8)
    assert fresh._uses_cached_search

    # And switching back
    index.prepare(compile_search=True)
    assert not index._uses_cached_search


def test_hellinger_distances_are_not_nan():
    # Duplicate and proportional rows can give a float32 overlap a hair above 1,
    # which came out of the square root as NaN.
    rng = np.random.default_rng(0)
    base = rng.random((300, 10)).astype(np.float32)
    data = np.vstack([base, base, 3 * base])
    index = NNDescent(data, "hellinger", n_neighbors=10, random_state=42)

    assert not np.isnan(index.neighbor_graph[1]).any()
    assert not np.isnan(index.query(data[:100], k=10)[1]).any()


@pytest.mark.parametrize(
    "metric", ["euclidean", "cosine", "proxy_inner_product", "inner_product"]
)
def test_build_matches_generic_nn_descent(metric):
    from pynndescent.pynndescent_ import nn_descent, INT32_MIN, INT32_MAX

    data = np.random.RandomState(0).normal(size=(300, 10)).astype(np.float32)
    index = NNDescent(data, metric=metric, n_neighbors=10, random_state=42)

    rng_state = (
        np.random.RandomState(42).randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    )
    expected = nn_descent(
        index._raw_data,
        index.n_neighbors,
        rng_state,
        10,
        index._distance_func,
        index.n_iters,
        index.delta,
        leaf_array=index._rp_forest,
        kernel=select_kernel(index._distance_func, index._raw_data),
    )
    np.testing.assert_array_equal(index._neighbor_graph[0], expected[0])
    np.testing.assert_array_equal(index._neighbor_graph[1], expected[1])


CACHE_HITS_IN_FRESH_PROCESS = """
import numpy as np
from pynndescent import NNDescent, pynndescent_

data = np.random.RandomState(0).normal(size=(300, 10)).astype(np.float32)
for metric in ("euclidean", "cosine", "proxy_inner_product", "inner_product"):
    NNDescent(data, metric=metric, n_neighbors=10, random_state=42).prepare()
cache_hits = [
    sum(getattr(pynndescent_, name).stats.cache_hits.values())
    for name in (
        "nn_descent_squared_euclidean",
        "nn_descent_alternative_dot",
        "nn_descent_proxy_inner_product",
        "nn_descent_alternative_inner_product",
        "diversify_squared_euclidean",
        "diversify_alternative_dot",
        "diversify_proxy_inner_product",
        "diversify_alternative_inner_product",
    )
]
print("RESULT", pynndescent_.__file__, *cache_hits)
"""


@pytest.mark.skipif(
    "NUMBA_DISABLE_JIT" in os.environ, reason="Caching needs Numba compilation"
)
def test_dense_build_is_reused_from_disk_cache():
    from pynndescent import pynndescent_

    data = np.random.RandomState(0).normal(size=(300, 10)).astype(np.float32)
    for metric in ("euclidean", "cosine", "proxy_inner_product", "inner_product"):
        NNDescent(data, metric=metric, n_neighbors=10, random_state=42).prepare()

    checkout = str(pathlib.Path(pynndescent_.__file__).parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([checkout, os.environ.get("PYTHONPATH", "")])}
    output = subprocess.run(
        [sys.executable, "-c", CACHE_HITS_IN_FRESH_PROCESS],
        env=env, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    _, child_module, *cache_hits = next(line for line in output if line.startswith("RESULT ")).split()

    assert child_module == pynndescent_.__file__
    assert [int(hits) > 0 for hits in cache_hits] == [True] * 8


@pytest.mark.parametrize(
    "data, match",
    [
        (np.zeros(10, dtype=np.float32), "Expected 2D array"),
        (np.zeros((0, 5), dtype=np.float32), "0 sample"),
        (np.zeros((5, 0), dtype=np.float32), "0 feature"),
        ([[1.0, 2.0], [3.0]], "inhomogeneous|sequence"),
    ],
)
def test_invalid_data_raises_value_error(data, match):
    """https://github.com/lmcinnes/pynndescent/issues/294 (11)"""
    with pytest.raises(ValueError, match=match):
        NNDescent(data, n_neighbors=5)


@pytest.mark.parametrize("n_neighbors", [0, -3])
def test_invalid_n_neighbors_raises_value_error(nn_data, n_neighbors):
    """https://github.com/lmcinnes/pynndescent/issues/294 (11)"""
    with pytest.raises(ValueError, match="n_neighbors must be at least 1"):
        NNDescent(nn_data[:50], n_neighbors=n_neighbors)


def test_list_input_is_accepted():
    data = np.random.RandomState(0).normal(size=(50, 4)).tolist()
    indices, _ = NNDescent(data, n_neighbors=5, random_state=0).neighbor_graph
    assert indices.shape == (50, 5)


def test_transformer_transform_none_after_query(nn_data):
    """https://github.com/lmcinnes/pynndescent/issues/294 (10)"""
    data = nn_data[:200]

    # An uncompressed index keeps the training graph, so transform(None) keeps working.
    transformer = PyNNDescentTransformer(n_neighbors=5, random_state=0)
    transformer.fit(data, compress_index=False)
    expected = transformer.transform(None)
    transformer.transform(data[:10])
    np.testing.assert_array_equal(
        transformer.transform(None).toarray(), expected.toarray()
    )

    # A compressed index discards it on the first query: fail with a clear error.
    transformer = PyNNDescentTransformer(n_neighbors=5, random_state=0).fit(data)
    transformer.transform(data[:10])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="compress_index=False"):
            transformer.transform(None)


@pytest.mark.parametrize("compile_search", [False, True])
def test_zero_epsilon_searches_past_the_seed_leaf(compile_search):
    # With k larger than the search tree leaf the result heap is not full after
    # seeding; epsilon=0 must still search the graph (inf + 0 * inf is NaN,
    # which used to end the search at once).
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3000, 16)).astype(np.float32)
    queries = rng.normal(size=(40, 16)).astype(np.float32)
    index = NNDescent(data, n_neighbors=15, random_state=42)
    index.prepare(compile_search=compile_search)
    indices = index.query_indices(queries, k=60, epsilon=0.0)
    true = np.argsort(((queries[:, None] - data[None]) ** 2).sum(-1), axis=1)[:, :60]
    recall = np.mean([len(set(a) & set(b)) / 60 for a, b in zip(indices, true)])
    assert recall > 0.8


@pytest.mark.parametrize("quantization", ["sq8", "sq4"])
@pytest.mark.parametrize("metric", ["euclidean", "cosine", "dot"])
def test_scalar_quantized_search(quantization, metric):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(30, 24)).astype(np.float32) * 4
    data = centers[rng.integers(0, 30, 4000)] + rng.normal(size=(4000, 24))
    queries = centers[rng.integers(0, 30, 60)] + rng.normal(size=(60, 24))
    data = data.astype(np.float32)
    queries = queries.astype(np.float32)
    index = NNDescent(
        data, metric=metric, n_neighbors=15, random_state=42, quantization=quantization
    )
    indices, dists = index.query(queries, k=10, epsilon=0.05, candidate_width=40)

    if metric == "euclidean":
        true_dists = np.sqrt(((queries[:, None] - data[None]) ** 2).sum(-1))
        returned = np.sqrt(((queries[:, None] - data[indices]) ** 2).sum(-1))
    else:
        true_dists = cosine_distances(queries, data)
        returned = np.take_along_axis(true_dists, indices, axis=1)
    true = np.argsort(true_dists, axis=1)[:, :10]
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(indices, true)])
    assert recall > (0.9 if quantization == "sq8" else 0.6)
    # Candidates are reranked exactly, so the returned distances are exact.
    np.testing.assert_allclose(dists, returned, atol=1e-4)
    assert np.all(np.diff(dists, axis=1) >= -1e-6)

    single = np.vstack(
        [index.query_indices(q[None], k=10, epsilon=0.05, candidate_width=40) for q in queries]
    )
    np.testing.assert_array_equal(single, indices)

    restored = pickle.loads(pickle.dumps(index))
    np.testing.assert_array_equal(
        restored.query_indices(queries, k=10, epsilon=0.05, candidate_width=40), indices
    )
    with pytest.raises(ValueError):
        index.query(queries, k=10, candidate_width=5)


def test_scalar_quantization_rejects_unsupported_metric(nn_data):
    index = NNDescent(
        nn_data[:300].astype(np.float32),
        metric="manhattan",
        n_neighbors=10,
        random_state=42,
        quantization="sq8",
    )
    with pytest.raises(ValueError):
        index.prepare()


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "dot"])
def test_float16_rerank_matches_float32(metric):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(30, 24)).astype(np.float32) * 4
    data = (centers[rng.integers(0, 30, 4000)] + rng.normal(size=(4000, 24))).astype(np.float32)
    queries = (centers[rng.integers(0, 30, 60)] + rng.normal(size=(60, 24))).astype(np.float32)
    if metric == "euclidean":
        # Far outside the float16 range unless the copy is rescaled.
        data *= np.float32(1e5)
        queries *= np.float32(1e5)
    results = {}
    for precision in ("float32", "float16"):
        index = NNDescent(
            data,
            metric=metric,
            n_neighbors=15,
            random_state=42,
            quantization="sq8",
            rerank_precision=precision,
        )
        results[precision] = index.query(queries, k=10, epsilon=0.05, candidate_width=40)
    (i32, d32), (i16, d16) = results["float32"], results["float16"]
    assert np.all(np.isfinite(d16))
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(i32, i16)])
    assert overlap > 0.97
    np.testing.assert_allclose(d16, d32, rtol=2e-3, atol=1e-3)

    restored = pickle.loads(pickle.dumps(index))
    np.testing.assert_array_equal(
        restored.query_indices(queries, k=10, epsilon=0.05, candidate_width=40), i16
    )


def test_float16_rerank_rejects_invalid_settings(nn_data):
    with pytest.raises(ValueError):
        NNDescent(nn_data[:200], n_neighbors=10, rerank_precision="float16")
    with pytest.raises(ValueError):
        NNDescent(nn_data[:200], n_neighbors=10, quantization="sq8", rerank_precision="half")


@pytest.mark.parametrize("vnni", [True, False])
def test_integer_code_dots_match_numpy(vnni, monkeypatch):
    import numba

    from pynndescent import integer_dot

    if vnni and not integer_dot.TARGET_HAS_VNNI:
        pytest.skip("no AVX-512 VNNI on this CPU")
    # The intrinsics read TARGET_HAS_VNNI when a caller is compiled, so these
    # (freshly defined) functions exercise the chosen code path.
    monkeypatch.setattr(integer_dot, "TARGET_HAS_VNNI", vnni)

    @numba.njit
    def dots(codes, query, nibbles):
        out = np.empty((codes.shape[0], 2), dtype=np.int64)
        for i in range(0, codes.shape[0], 4):
            if nibbles:
                four = integer_dot.dot4_u4(codes, i, i + 1, i + 2, i + 3, query)
            else:
                four = integer_dot.dot4_u8(codes, i, i + 1, i + 2, i + 3, query)
            for s in range(4):
                out[i + s, 0] = four[s]
                if nibbles:
                    out[i + s, 1] = integer_dot.dot_u4(codes, i + s, query)
                else:
                    out[i + s, 1] = integer_dot.dot_u8(codes, i + s, query)
        return out

    rng = np.random.default_rng(0)
    codes = integer_dot.aligned_padded_codes(rng.integers(0, 256, (40, 150), dtype=np.uint8))
    assert codes.shape == (40, 192) and codes.ctypes.data % 64 == 0
    assert not codes[:, 150:].any()
    query = rng.integers(-127, 128, (2, 192), dtype=np.int8)
    wide = codes.astype(np.int64)
    expected8 = wide @ query[0].astype(np.int64)
    expected4 = (wide & 15) @ query[0].astype(np.int64) + (wide >> 4) @ query[1].astype(np.int64)
    for nibbles, expected in ((False, expected8), (True, expected4)):
        got = dots(codes, query, nibbles)
        np.testing.assert_array_equal(got[:, 0], expected)
        np.testing.assert_array_equal(got[:, 1], expected)


@pytest.mark.parametrize("quantization", ["sq8", "sq4"])
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_integer_query_matches_float_kernels(quantization, metric):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(30, 40)).astype(np.float32) * 4
    data = (centers[rng.integers(0, 30, 4000)] + rng.normal(size=(4000, 40))).astype(np.float32)
    queries = (centers[rng.integers(0, 30, 60)] + rng.normal(size=(60, 40))).astype(np.float32)
    results = {}
    for integer in (False, True):
        index = NNDescent(
            data, metric=metric, n_neighbors=15, random_state=42, quantization=quantization
        )
        index._sq_integer_query = integer
        index._sq_align_codes = integer
        results[integer] = index.query_indices(queries, k=10, epsilon=0.05, candidate_width=40)
        codes = index._sq_codes
        width = 40 if quantization == "sq8" else 20
        if integer:
            # Whole, aligned cache lines per code row; zero padding.
            assert codes.shape[1] == 64 and codes.ctypes.data % 64 == 0
            assert not codes[:, width:].any()
        else:
            assert codes.shape[1] == width
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(results[False], results[True])])
    assert overlap > 0.97


def test_sq4_code_range_clips_outliers_by_default():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3000, 8)).astype(np.float32)
    data[:3, 0] = 1000.0  # rare outliers in feature 0
    scales = {}
    for quantization, clip in (("sq8", "default"), ("sq4", "default"), ("sq4", None)):
        index = NNDescent(data, n_neighbors=10, random_state=42, quantization=quantization)
        if clip is None:
            index._sq_clip_quantile = None
        index.prepare()
        scales[quantization, clip] = index._sq_scales[0]
    full_range = data[:, 0].max() - data[:, 0].min()
    # sq8 and an explicit None span the full range; sq4 drops the outliers.
    np.testing.assert_allclose(scales["sq8", "default"], full_range / 255, rtol=1e-5)
    np.testing.assert_allclose(scales["sq4", None], full_range / 15, rtol=1e-5)
    assert scales["sq4", "default"] < 10.0 / 15


@pytest.mark.parametrize("quantization", [None, "sq8"])
def test_rcm_vertex_order_keeps_graph_neighbours_close(quantization):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(30, 24)).astype(np.float32) * 3
    data = (centers[rng.integers(0, 30, 6000)] + rng.normal(size=(6000, 24))).astype(np.float32)
    queries = (centers[rng.integers(0, 30, 60)] + rng.normal(size=(60, 24))).astype(np.float32)
    true = np.argsort(((queries[:, None] - data[None]) ** 2).sum(-1), axis=1)[:, :10]
    kwargs = {} if quantization is None else dict(quantization=quantization)
    query_kwargs = {} if quantization is None else dict(candidate_width=40)
    spans, recalls = {}, {}
    for method in ("tree", "rcm"):
        index = NNDescent(data, n_neighbors=15, random_state=42, **kwargs)
        index._vertex_order_method = method
        index.prepare()
        np.testing.assert_array_equal(np.sort(index._vertex_order), np.arange(len(data)))
        graph = index._search_graph.tocoo()
        spans[method] = np.abs(graph.row.astype(np.int64) - graph.col).mean()
        indices = index.query_indices(queries, k=10, epsilon=0.05, **query_kwargs)
        recalls[method] = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(indices, true)])
    assert spans["rcm"] < 0.5 * spans["tree"]
    assert recalls["rcm"] > 0.9
    assert abs(recalls["rcm"] - recalls["tree"]) < 0.02
    # rcm is the default
    default = NNDescent(data[:2000], n_neighbors=15, random_state=42, **kwargs)
    default.prepare()
    graph = default._search_graph.tocoo()
    assert np.abs(graph.row.astype(np.int64) - graph.col).mean() < 0.1 * 2000


@pytest.mark.parametrize("metric", ["euclidean", "dot"])
@pytest.mark.parametrize("dims", [32, 0.9])
def test_quantization_dims_projects_low_rank_data(metric, dims):
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(24, 96)).astype(np.float32)
    data = (rng.normal(size=(4000, 24)) @ basis + 0.1 * rng.normal(size=(4000, 96))).astype(
        np.float32
    )
    queries = (rng.normal(size=(60, 24)) @ basis + 0.1 * rng.normal(size=(60, 96))).astype(
        np.float32
    )
    index = NNDescent(
        data,
        metric=metric,
        n_neighbors=15,
        random_state=42,
        quantization="sq8",
        quantization_dims=dims,
    )
    indices, dists = index.query(queries, k=10, epsilon=0.05, candidate_width=40)
    # Code rows may be padded to whole cache lines; the projection keeps <= 32 components.
    assert index._sq_projection[1].shape[1] <= 32
    if metric == "euclidean":
        true_dists = np.sqrt(((queries[:, None] - data[None]) ** 2).sum(-1))
        returned = np.sqrt(((queries[:, None] - data[indices]) ** 2).sum(-1))
    else:
        true_dists = cosine_distances(queries, data)
        returned = np.take_along_axis(true_dists, indices, axis=1)
    true = np.argsort(true_dists, axis=1)[:, :10]
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(indices, true)])
    assert recall > 0.9
    # Reranking uses every feature, so the returned distances stay exact.
    np.testing.assert_allclose(dists, returned, atol=1e-3, rtol=1e-4)

    restored = pickle.loads(pickle.dumps(index))
    np.testing.assert_array_equal(
        restored.query_indices(queries, k=10, epsilon=0.05, candidate_width=40), indices
    )


def test_quantization_dims_skips_projection_for_spread_variance():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3000, 32)).astype(np.float32)
    queries = rng.normal(size=(30, 32)).astype(np.float32)
    results = []
    for dims in (None, 0.95):
        index = NNDescent(
            data, n_neighbors=15, random_state=42, quantization="sq8", quantization_dims=dims
        )
        results.append(index.query_indices(queries, k=10, epsilon=0.05, candidate_width=40))
        # Isotropic data needs nearly every feature for 95% of the variance.
        assert index._sq_projection is None
    np.testing.assert_array_equal(results[0], results[1])


@pytest.mark.parametrize("dims", [0, 1.0, 1.5, -3, True])
def test_quantization_dims_rejects_invalid_settings(nn_data, dims):
    with pytest.raises(ValueError):
        NNDescent(nn_data[:200], n_neighbors=10, quantization="sq8", quantization_dims=dims)


def test_quantization_dims_requires_scalar_quantization(nn_data):
    with pytest.raises(ValueError):
        NNDescent(nn_data[:200], n_neighbors=10, quantization_dims=8)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_diversify_alpha_keeps_more_edges(metric):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3000, 16)).astype(np.float32)
    queries = rng.normal(size=(400, 16)).astype(np.float32)
    graphs = {}
    for alpha in (None, 1.0, 1.3):
        kwargs = {} if alpha is None else {"diversify_alpha": alpha}
        index = NNDescent(data, metric=metric, n_neighbors=15, random_state=42, **kwargs)
        index.prepare()
        graphs[alpha] = index
    default, one, relaxed = graphs[None], graphs[1.0], graphs[1.3]
    np.testing.assert_array_equal(default._search_graph.indices, one._search_graph.indices)
    assert relaxed._search_graph.nnz > 1.2 * default._search_graph.nnz
    if metric == "euclidean":
        true = np.argsort(((queries[:, None] - data[None]) ** 2).sum(-1), axis=1)[:, :10]
    else:
        true = np.argsort(cosine_distances(queries, data), axis=1)[:, :10]

    def recall(index):
        found = index.query_indices(queries, k=10, epsilon=0.1)
        return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, true)])

    # The extra edges improve search (by ~0.09 at every thread count). Absolute
    # recall here varies by about +-0.02 with the thread count, which changes
    # how NN-descent splits its work, so the floor leaves room for that.
    relaxed_recall = recall(relaxed)
    assert relaxed_recall > recall(default) + 0.04
    assert relaxed_recall > 0.85


def test_diversify_alpha_below_one_rejected(nn_data):
    with pytest.raises(ValueError):
        NNDescent(nn_data[:200], n_neighbors=10, diversify_alpha=0.9)


def _out_of_distribution_problem(seed=0, n=8000, dim=24, n_sample=4000, n_queries=80):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, dim)).astype(np.float32)
    shift = rng.normal(size=dim).astype(np.float32)
    sample = (rng.normal(size=(n_sample, dim)) * 0.7 + shift).astype(np.float32)
    queries = (rng.normal(size=(n_queries, dim)) * 0.7 + shift).astype(np.float32)
    return data, sample, queries


def _exact_neighbors(data, queries, k, metric):
    if metric == "cosine":
        data, queries = normalize(data), normalize(queries)
    nn = NearestNeighbors(n_neighbors=k).fit(data)
    return nn.kneighbors(queries, return_distance=False)


def _recall(found, true):
    return np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(found, true)])


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("quantization", [None, "sq8"])
def test_query_sample_improves_out_of_distribution_recall(metric, quantization):
    data, sample, queries = _out_of_distribution_problem()
    sample_neighbors = _exact_neighbors(data, sample, 64, metric)
    true = _exact_neighbors(data, queries, 10, metric)
    query_kwargs = {} if quantization is None else dict(candidate_width=30)
    recalls = {}
    for mode in ("none", "given", "found"):
        index = NNDescent(
            data, metric=metric, n_neighbors=15, random_state=42, quantization=quantization
        )
        if mode == "given":
            index.prepare(query_sample=sample, query_neighbors=sample_neighbors)
        elif mode == "found":
            index.prepare(query_sample=sample)
        found = index.query_indices(queries, k=10, epsilon=0.0, **query_kwargs)
        recalls[mode] = _recall(found, true)
    assert recalls["given"] > recalls["none"] + 0.1
    assert recalls["found"] > recalls["none"] + 0.1
    assert abs(recalls["found"] - recalls["given"]) < 0.05


def test_query_sample_index_pickles_without_the_sample():
    data, sample, queries = _out_of_distribution_problem(n_sample=1000)
    index = NNDescent(data, n_neighbors=15, random_state=42, quantization="sq8",
                      quantization_dims=0.9)
    index.prepare(query_sample=sample, query_neighbors=_exact_neighbors(data, sample, 64, "euclidean"))
    expected = index.query_indices(queries, k=10, candidate_width=30)
    for name in ("_query_sample", "_query_neighbors", "_query_pca_sample"):
        assert not hasattr(index, name)
    restored = pickle.loads(pickle.dumps(index))
    np.testing.assert_array_equal(restored.query_indices(queries, k=10, candidate_width=30), expected)


def test_query_sample_fits_projection_to_queries():
    # Data spread over 8 directions, queries over 8 others: a projection fitted to
    # the data alone keeps none of the queries' directions.
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(64, 64)))[0].astype(np.float32)
    data = (rng.normal(size=(5000, 8)) * 5 @ basis[:8] + rng.normal(size=(5000, 64)) * 0.3).astype(np.float32)
    sample = (rng.normal(size=(2000, 8)) * 5 @ basis[8:16]).astype(np.float32)
    neighbors = _exact_neighbors(data, sample, 64, "euclidean")
    projections = {}
    for query_pca in (False, True):
        index = NNDescent(data, n_neighbors=10, random_state=42, quantization="sq8",
                          quantization_dims=12)
        index._query_pca = query_pca
        index.prepare(query_sample=sample, query_neighbors=neighbors)
        mean, projection8, scale = index._sq_projection
        projections[query_pca] = projection8.astype(np.float32) * scale
    # Share of the query directions' energy the projection keeps
    kept = {k: np.linalg.norm(basis[8:16] @ p) ** 2 / 8 for k, p in projections.items()}
    assert kept[True] > kept[False] + 0.2


def test_query_sample_validation():
    data, sample, _ = _out_of_distribution_problem(n=2000, n_sample=200)
    neighbors = _exact_neighbors(data, sample, 20, "euclidean")
    index = NNDescent(data, n_neighbors=10, random_state=42)
    with pytest.raises(ValueError, match="features"):
        index.prepare(query_sample=sample[:, :10])
    with pytest.raises(ValueError, match="requires query_sample"):
        index.prepare(query_neighbors=neighbors)
    with pytest.raises(ValueError, match="row for each"):
        index.prepare(query_sample=sample, query_neighbors=neighbors[:10])
    with pytest.raises(ValueError, match="beyond"):
        index.prepare(query_sample=sample, query_neighbors=neighbors + len(data))
    index.prepare()
    with pytest.raises(ValueError, match="before the search index is built"):
        index.prepare(query_sample=sample, query_neighbors=neighbors)
    sparse_index = NNDescent(scipy.sparse.csr_matrix(data), n_neighbors=10, random_state=42)
    with pytest.raises(NotImplementedError):
        sparse_index.prepare(query_sample=sample)


def test_update_with_query_sample():
    data, sample, queries = _out_of_distribution_problem()
    fresh = data[6000:]
    true = _exact_neighbors(data, queries, 10, "euclidean")
    recalls = {}
    for mode in ("dropped", "kept"):
        index = NNDescent(data[:6000], n_neighbors=15, random_state=42)
        index.prepare(query_sample=sample)
        if mode == "dropped":
            with pytest.warns(UserWarning, match="query_sample"):
                index.update(xs_fresh=fresh)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                index.update(xs_fresh=fresh, query_sample=sample)
        found = index.query_indices(queries, k=10, epsilon=0.0)
        recalls[mode] = _recall(found, true)
    assert recalls["kept"] > recalls["dropped"] + 0.1
    with pytest.raises(ValueError, match="requires query_sample"):
        index.update(xs_fresh=fresh[:10], query_neighbors=np.zeros((5, 5), dtype=np.int32))


@pytest.mark.parametrize(
    "metric,quantization",
    [
        ("euclidean", "binary"),
        ("euclidean", "uint8"),
        ("cosine", "int8"),
        ("euclidean", "sq8"),
        ("cosine", "sq4"),
    ],
)
def test_update_prepared_quantized_index(metric, quantization):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1500, 16)).astype(np.float32)
    index = NNDescent(
        data[:1000], metric=metric, n_neighbors=15, random_state=42,
        quantization=quantization,
    )
    index.prepare()
    index.update(xs_fresh=data[1000:])
    fresh_build = NNDescent(
        data, metric=metric, n_neighbors=15, random_state=42, quantization=quantization
    )
    # The fresh points must be quantized and searchable: each should find itself
    # about as often as in an index built with them from the start
    self_recall = {}
    for name, nnd in (("updated", index), ("fresh_build", fresh_build)):
        found = nnd.query_indices(data[1000:], k=1, epsilon=0.2)
        self_recall[name] = np.mean(found[:, 0] == np.arange(1000, 1500))
    assert self_recall["updated"] > self_recall["fresh_build"] - 0.05
    assert self_recall["updated"] > 0.7


def test_query_induced_edges_are_relative_neighbourhood_pruned():
    from pynndescent.distances import squared_euclidean
    from pynndescent.query_sample import query_induced_edges

    data = np.arange(6, dtype=np.float32)[:, None]
    # One query near point 0 (neighbours 0, 1, 2, 3) and one near point 4.
    neighbors = np.array([[0, 1, 2, 3], [4, 5, 3, 2]])
    indices, distances = query_induced_edges(
        data, squared_euclidean, neighbors, n_candidates=4, max_edges=3, n_pivots=2
    )
    edges = {p: sorted(c for c in indices[p] if c >= 0) for p in range(6)}
    # Point 0 is offered 1, 2 and 3, but 2 and 3 are closer to 1 than to 0.
    assert edges[0] == [1]
    # Point 1 (the first query's second nearest) is offered 0, 2 and 3.
    assert edges[1] == [0, 2]
    assert edges[4] == [3, 5]
    assert edges[5] == [4]
    assert distances[0, 0] == 1.0


def _clustered(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(40, dim)).astype(np.float32) * 3
    return (centers[rng.integers(0, 40, n)] + rng.normal(size=(n, dim))).astype(np.float32)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_progressive_build_matches_exact_build(metric):
    data = _clustered(3000, 320)
    k = 15
    true = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="brute").fit(data)
    true_dist, true_ind = true.kneighbors(data)

    def recall(index):
        ind = index.neighbor_graph[0]
        return np.mean([np.intersect1d(a, b).shape[0] / k for a, b in zip(ind, true_ind)])

    exact = NNDescent(data, metric=metric, n_neighbors=k, random_state=1,
                      progressive_build=False)
    progressive = NNDescent(data, metric=metric, n_neighbors=k, random_state=1,
                            progressive_build=True)
    assert recall(progressive) > 0.95
    assert recall(progressive) >= recall(exact) - 0.005
    # the graph carries exact distances, not code distances
    ind, dist = progressive.neighbor_graph
    if metric == "euclidean":
        exact_dist = np.linalg.norm(data[ind] - data[:, None, :], axis=2)
    else:
        unit = normalize(data)
        exact_dist = 1.0 - np.einsum("id,ikd->ik", unit, unit[ind])
    np.testing.assert_allclose(dist, exact_dist, rtol=1e-3, atol=1e-4)


def test_progressive_build_auto_selection(monkeypatch):
    import pynndescent.pynndescent_ as pynnd

    calls = []
    original = pynnd.progressive_nn_descent

    def recording(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(pynnd, "progressive_nn_descent", recording)
    wide = _clustered(1000, 600)
    narrow = _clustered(1000, 64)
    NNDescent(wide, n_neighbors=10, random_state=0)
    NNDescent(narrow, n_neighbors=10, random_state=0)
    NNDescent(wide, n_neighbors=10, random_state=0, progressive_build=False)
    NNDescent(wide, n_neighbors=10, random_state=0, metric="manhattan")
    NNDescent(wide, n_neighbors=10, random_state=0, tree_init=False)
    assert calls == [wide.shape]
    NNDescent(narrow, n_neighbors=10, random_state=0, progressive_build=True)
    assert calls == [wide.shape, narrow.shape]


def test_progressive_build_validation():
    with pytest.raises(ValueError, match="progressive_build"):
        NNDescent(_clustered(200, 300), n_neighbors=5, progressive_build="yes")


def test_projected_trees_for_wide_data(monkeypatch):
    import pynndescent.pynndescent_ as pynnd

    calls = []
    original = pynnd.project_for_trees

    def recording(data, *args, **kwargs):
        calls.append(data.shape)
        return original(data, *args, **kwargs)

    monkeypatch.setattr(pynnd, "project_for_trees", recording)
    wide = _clustered(2000, 2048)
    k = 10
    projected = NNDescent(wide, metric="cosine", n_neighbors=k, random_state=0)
    assert calls == [wide.shape]
    exact = NNDescent(wide, metric="cosine", n_neighbors=k, random_state=0,
                      progressive_build=False)
    NNDescent(_clustered(2000, 784), n_neighbors=k, random_state=0)
    assert calls == [wide.shape]
    true_ind = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute").fit(
        wide).kneighbors(wide)[1]

    def recall(index):
        return np.mean([np.intersect1d(a, b).shape[0] / k
                        for a, b in zip(index.neighbor_graph[0], true_ind)])

    assert recall(projected) >= recall(exact) - 0.005
    projected.prepare()
    assert projected.query(wide[:20], k=k)[0].shape == (20, k)


def test_cosine_normalization_copies_only_when_needed():
    data = _clustered(500, 40)
    data[7] = 0.0
    original = data.copy()
    index = NNDescent(data, metric="cosine", n_neighbors=5, random_state=0)
    np.testing.assert_array_equal(data, original)
    expected = normalize(original)
    np.testing.assert_allclose(index._raw_data, expected, rtol=1e-6, atol=1e-7)
    assert np.all(index._raw_data[7] == 0.0)

    unit = np.ascontiguousarray(expected, dtype=np.float32)
    index = NNDescent(unit, metric="cosine", n_neighbors=5, random_state=0)
    assert index._raw_data is unit

    # rounded through float16: close to, but not, unit norm
    rounded = unit.astype(np.float16).astype(np.float32)
    index = NNDescent(rounded, metric="cosine", n_neighbors=5, random_state=0)
    assert index._raw_data is not rounded
    np.testing.assert_allclose(np.linalg.norm(index._raw_data[:7], axis=1), 1.0, atol=1e-6)
