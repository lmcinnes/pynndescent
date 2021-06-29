import os
import io
import re
import pytest
from contextlib import redirect_stdout

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import pickle
import joblib
import scipy

from pynndescent import NNDescent, PyNNDescentTransformer


def test_nn_descent_neighbor_accuracy(nn_data, seed):
    knn_indices, _ = NNDescent(
        nn_data, "euclidean", {}, 10, random_state=np.random.RandomState(seed)
    )._neighbor_graph

    tree = KDTree(nn_data)
    true_indices = tree.query(nn_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (nn_data.shape[0] * 10)
    assert percent_correct >= 0.98, (
        "NN-descent did not get 99% " "accuracy on nearest neighbors"
    )


def test_angular_nn_descent_neighbor_accuracy(nn_data, seed):
    knn_indices, _ = NNDescent(
        nn_data, "cosine", {}, 10, random_state=np.random.RandomState(seed)
    )._neighbor_graph

    angular_data = normalize(nn_data, norm="l2")
    tree = KDTree(angular_data)
    true_indices = tree.query(angular_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (nn_data.shape[0] * 10)
    assert percent_correct >= 0.98, (
        "NN-descent did not get 99% " "accuracy on nearest neighbors"
    )


@pytest.mark.skipif(
    list(map(int, scipy.version.version.split("."))) < [1, 3, 0],
    reason="requires scipy >= 1.3.0",
)
def test_sparse_nn_descent_neighbor_accuracy(sparse_nn_data, seed):
    knn_indices, _ = NNDescent(
        sparse_nn_data, "euclidean", n_neighbors=20, random_state=None
    )._neighbor_graph

    tree = KDTree(sparse_nn_data.toarray())
    true_indices = tree.query(sparse_nn_data.toarray(), 10, return_distance=False)

    num_correct = 0.0
    for i in range(sparse_nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (sparse_nn_data.shape[0] * 10)
    assert percent_correct >= 0.85, (
        "Sparse NN-descent did not get 95% " "accuracy on nearest neighbors"
    )


@pytest.mark.skipif(
    list(map(int, scipy.version.version.split("."))) < [1, 3, 0],
    reason="requires scipy >= 1.3.0",
)
def test_sparse_angular_nn_descent_neighbor_accuracy(sparse_nn_data):
    knn_indices, _ = NNDescent(
        sparse_nn_data, "cosine", {}, 20, random_state=None
    )._neighbor_graph

    angular_data = normalize(sparse_nn_data, norm="l2").toarray()
    tree = KDTree(angular_data)
    true_indices = tree.query(angular_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(sparse_nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (sparse_nn_data.shape[0] * 10)
    assert percent_correct >= 0.85, (
        "Sparse angular NN-descent did not get 98% " "accuracy on nearest neighbors"
    )


def test_nn_descent_query_accuracy(nn_data):
    nnd = NNDescent(nn_data[200:], "euclidean", n_neighbors=10, random_state=None)
    knn_indices, _ = nnd.query(nn_data[:200], k=10, epsilon=0.2)

    tree = KDTree(nn_data[200:])
    true_indices = tree.query(nn_data[:200], 10, return_distance=False)

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert percent_correct >= 0.95, (
        "NN-descent query did not get 95% " "accuracy on nearest neighbors"
    )


def test_nn_descent_query_accuracy_angular(nn_data):
    nnd = NNDescent(nn_data[200:], "cosine", n_neighbors=30, random_state=None)
    knn_indices, _ = nnd.query(nn_data[:200], k=10, epsilon=0.32)

    nn = NearestNeighbors(metric="cosine").fit(nn_data[200:])
    true_indices = nn.kneighbors(nn_data[:200], n_neighbors=10, return_distance=False)

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert percent_correct >= 0.95, (
        "NN-descent query did not get 95% " "accuracy on nearest neighbors"
    )


def test_sparse_nn_descent_query_accuracy(sparse_nn_data):
    nnd = NNDescent(
        sparse_nn_data[200:], "euclidean", n_neighbors=15, random_state=None
    )
    knn_indices, _ = nnd.query(sparse_nn_data[:200], k=10, epsilon=0.24)

    tree = KDTree(sparse_nn_data[200:].toarray())
    true_indices = tree.query(sparse_nn_data[:200].toarray(), 10, return_distance=False)

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert percent_correct >= 0.95, (
        "Sparse NN-descent query did not get 95% " "accuracy on nearest neighbors"
    )


def test_sparse_nn_descent_query_accuracy_angular(sparse_nn_data):
    nnd = NNDescent(sparse_nn_data[200:], "cosine", n_neighbors=50, random_state=None)
    knn_indices, _ = nnd.query(sparse_nn_data[:200], k=10, epsilon=0.36)

    nn = NearestNeighbors(metric="cosine").fit(sparse_nn_data[200:].toarray())
    true_indices = nn.kneighbors(
        sparse_nn_data[:200].toarray(), n_neighbors=10, return_distance=False
    )

    num_correct = 0.0
    for i in range(true_indices.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (true_indices.shape[0] * 10)
    assert percent_correct >= 0.95, (
        "Sparse NN-descent query did not get 95% " "accuracy on nearest neighbors"
    )


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
    )._neighbor_graph

    tree = KDTree(nn_data)
    true_indices = tree.query(nn_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert percent_correct >= 0.99, (
        "NN-descent did not get 99% " "accuracy on nearest neighbors"
    )


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
    )._neighbor_graph

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
    )._neighbor_graph

    for i in range(data.shape[0]):
        assert len(knn_indices[i]) == len(
            np.unique(knn_indices[i])
        ), "Duplicate graph_indices in knn graph"

    angular_data = normalize(data, norm="l2")
    tree = KDTree(angular_data)
    true_indices = tree.query(angular_data, n_neighbors, return_distance=False)

    num_correct = 0
    for i in range(data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    proportion_correct = num_correct / (data.shape[0] * n_neighbors)
    assert proportion_correct >= 0.95, (
        "NN-descent did not get 95%" " accuracy on nearest neighbors"
    )


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
