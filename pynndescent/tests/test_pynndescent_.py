from nose.tools import assert_greater_equal

import os

import numpy as np
from pynndescent import NNDescent
from scipy import sparse
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize

np.random.seed(42)
spatial_data = np.random.randn(10, 20)
spatial_data = np.vstack(
    [spatial_data, np.zeros((2, 20))]
)  # Add some all zero data for corner case test

nn_data = np.random.uniform(0, 1, size=(1000, 5))
nn_data = np.vstack(
    [nn_data, np.zeros((2, 5))]
)  # Add some all zero data for corner case test
binary_nn_data = np.random.choice(a=[False, True], size=(1000, 5), p=[0.66, 1 - 0.66])
binary_nn_data = np.vstack(
    [binary_nn_data, np.zeros((2, 5))]
)  # Add some all zero data for corner case test
sparse_nn_data = sparse.csr_matrix(nn_data * binary_nn_data)

def test_nn_descent_neighbor_accuracy():
    knn_indices, knn_dists = NNDescent(
        nn_data, "euclidean", {}, 10, random_state=np.random
    )._neighbor_graph

    tree = KDTree(nn_data)
    true_indices = tree.query(nn_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert_greater_equal(
        percent_correct,
        0.99,
        "NN-descent did not get 99% " "accuracy on nearest neighbors",
    )


def test_angular_nn_descent_neighbor_accuracy():
    knn_indices, knn_dists = NNDescent(
        nn_data, "cosine", {}, 10, random_state=np.random
    )._neighbor_graph

    angular_data = normalize(nn_data, norm="l2")
    tree = KDTree(angular_data)
    true_indices = tree.query(angular_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert_greater_equal(
        percent_correct,
        0.99,
        "NN-descent did not get 99% " "accuracy on nearest neighbors",
    )


def test_sparse_nn_descent_neighbor_accuracy():
    knn_indices, knn_dists = NNDescent(
        sparse_nn_data, "euclidean", {}, 10, random_state=np.random
    )._neighbor_graph

    tree = KDTree(sparse_nn_data.todense())
    true_indices = tree.query(sparse_nn_data.todense(), 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert_greater_equal(
        percent_correct,
        0.99,
        "Sparse NN-descent did not get " "99% accuracy on nearest " "neighbors",
    )


def test_sparse_angular_nn_descent_neighbor_accuracy():    
    knn_indices, knn_dists = NNDescent(
        sparse_nn_data, "cosine", {}, 10, random_state=np.random
    )._neighbor_graph

    angular_data = normalize(sparse_nn_data, norm="l2").toarray()
    tree = KDTree(angular_data)
    true_indices = tree.query(angular_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(nn_data.shape[0]):
        num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (spatial_data.shape[0] * 10)
    assert_greater_equal(
        percent_correct,
        0.99,
        "NN-descent did not get 99% " "accuracy on nearest neighbors",
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
# data used is a cut-down version of that provided by @scharron
# It contains lots of all-zero vectors and some other duplicates
def test_rp_trees_should_not_stack_overflow_with_duplicate_data():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(THIS_DIR, 'test_data/cosine_hang.npy')
    data = np.load(data_path)
    
    n_neighbors = 10
    knn_indices, _ = NNDescent(data, "cosine", {}, n_neighbors, \
        random_state=np.random, n_trees=20)._neighbor_graph

    angular_data = normalize(data, norm="l2")
    tree = KDTree(angular_data)
    true_indices = tree.query(angular_data, n_neighbors, return_distance=False)
    
    # all-zero vectors are bad news for the cosine metric, so we're only
    # going to shoot for 95% accuracy and only look at the non-zero entries
    # (the real success of this test is if it doesn't cause a recursion error)
    non_zero_rows = ~np.all(data == 0, axis=1)
    num_correct = 0
    for i in range(data.shape[0]):
        if non_zero_rows[i]:        
            num_correct += np.sum(np.in1d(true_indices[i], knn_indices[i]))

    proportion_correct = num_correct / (np.sum(non_zero_rows) * n_neighbors)
    assert_greater_equal(
        proportion_correct,
        0.95,
        "NN-descent did not get 95%" "accuracy on nearest neighbors",
    )
