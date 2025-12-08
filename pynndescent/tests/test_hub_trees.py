"""Unit tests for hub tree implementations.

Tests hub-based tree construction for all data types:
- Dense euclidean
- Dense angular (cosine)
- Sparse euclidean
- Sparse angular (cosine)
- Bit-packed
"""

import numpy as np
import pytest
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.preprocessing import normalize
import scipy.sparse as sparse

from pynndescent import NNDescent
from pynndescent.rp_trees import (
    euclidean_hub_split,
    angular_hub_split,
    sparse_euclidean_hub_split,
    sparse_angular_hub_split,
    bit_hub_split,
    compute_global_degrees,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hub_tree_data():
    """Generate test data for hub tree tests."""
    np.random.seed(42)
    return np.random.uniform(0, 1, size=(500, 20)).astype(np.float32)


@pytest.fixture
def hub_tree_sparse_data():
    """Generate sparse test data for hub tree tests."""
    np.random.seed(42)
    return sparse.random(500, 50, density=0.5, format="csr", dtype=np.float32)


@pytest.fixture
def hub_tree_bit_data():
    """Generate bit-packed test data for hub tree tests."""
    np.random.seed(42)
    data = np.random.uniform(0, 1, size=(500, 20)).astype(np.float32)
    return (data * 256).astype(np.uint8)


# ============================================================================
# Test hub split functions directly
# ============================================================================


def test_euclidean_hub_split_produces_valid_split(hub_tree_data):
    """Test that euclidean_hub_split produces a valid split."""
    # Build a simple neighbor graph
    nnd = NNDescent(hub_tree_data, n_neighbors=15, random_state=42)
    neighbor_indices = nnd._neighbor_graph[0]

    indices = np.arange(100, dtype=np.int32)  # Test with first 100 points
    rng_state = np.array([42, 12345, 67890], dtype=np.int64)
    global_degrees = compute_global_degrees(neighbor_indices)

    left, right, hyperplane, offset, balance = euclidean_hub_split(
        hub_tree_data, indices, neighbor_indices, global_degrees, rng_state
    )

    # Check that split is valid
    assert len(left) > 0, "Left partition should not be empty"
    assert len(right) > 0, "Right partition should not be empty"
    assert len(left) + len(right) == len(indices), "Split should preserve all points"
    assert len(set(left) & set(right)) == 0, "Partitions should not overlap"
    assert (
        hyperplane.shape[0] == hub_tree_data.shape[1]
    ), "Hyperplane dim should match data"


def test_angular_hub_split_produces_valid_split(hub_tree_data):
    """Test that angular_hub_split produces a valid split."""
    angular_data = normalize(hub_tree_data, norm="l2").astype(np.float32)

    nnd = NNDescent(angular_data, metric="cosine", n_neighbors=15, random_state=42)
    neighbor_indices = nnd._neighbor_graph[0]

    indices = np.arange(100, dtype=np.int32)
    rng_state = np.array([42, 12345, 67890], dtype=np.int64)
    global_degrees = compute_global_degrees(neighbor_indices)

    left, right, hyperplane, offset, balance = angular_hub_split(
        angular_data, indices, neighbor_indices, global_degrees, rng_state
    )

    assert len(left) > 0, "Left partition should not be empty"
    assert len(right) > 0, "Right partition should not be empty"
    assert len(left) + len(right) == len(indices), "Split should preserve all points"
    assert offset == 0.0, "Angular split should have zero offset"


def test_sparse_euclidean_hub_split_produces_valid_split(hub_tree_sparse_data):
    """Test that sparse_euclidean_hub_split produces a valid split."""
    nnd = NNDescent(hub_tree_sparse_data, n_neighbors=15, random_state=42)
    neighbor_indices = nnd._neighbor_graph[0]

    indices = np.arange(100, dtype=np.int32)
    rng_state = np.array([42, 12345, 67890], dtype=np.int64)
    global_degrees = compute_global_degrees(neighbor_indices)

    sp_data = hub_tree_sparse_data.tocsr()
    left, right, hyperplane, offset = sparse_euclidean_hub_split(
        sp_data.indices,
        sp_data.indptr,
        sp_data.data,
        indices,
        neighbor_indices,
        global_degrees,
        rng_state,
    )

    assert len(left) > 0, "Left partition should not be empty"
    assert len(right) > 0, "Right partition should not be empty"
    assert len(left) + len(right) == len(indices), "Split should preserve all points"


def test_sparse_angular_hub_split_produces_valid_split(hub_tree_sparse_data):
    """Test that sparse_angular_hub_split produces a valid split."""
    normalized_data = normalize(hub_tree_sparse_data, norm="l2")

    nnd = NNDescent(normalized_data, metric="cosine", n_neighbors=15, random_state=42)
    neighbor_indices = nnd._neighbor_graph[0]

    indices = np.arange(100, dtype=np.int32)
    rng_state = np.array([42, 12345, 67890], dtype=np.int64)
    global_degrees = compute_global_degrees(neighbor_indices)

    sp_data = normalized_data.tocsr()
    left, right, hyperplane, offset = sparse_angular_hub_split(
        sp_data.indices,
        sp_data.indptr,
        sp_data.data,
        indices,
        neighbor_indices,
        global_degrees,
        rng_state,
    )

    assert len(left) > 0, "Left partition should not be empty"
    assert len(right) > 0, "Right partition should not be empty"
    assert len(left) + len(right) == len(indices), "Split should preserve all points"


def test_bitpacked_hub_split_produces_valid_split(hub_tree_bit_data):
    """Test that bit_hub_split produces a valid split."""
    nnd = NNDescent(
        hub_tree_bit_data, metric="bit_jaccard", n_neighbors=15, random_state=42
    )
    neighbor_indices = nnd._neighbor_graph[0]

    indices = np.arange(100, dtype=np.int32)
    rng_state = np.array([42, 12345, 67890], dtype=np.int64)
    global_degrees = compute_global_degrees(neighbor_indices)

    left, right, hyperplane, offset = bit_hub_split(
        hub_tree_bit_data, indices, neighbor_indices, global_degrees, rng_state
    )

    assert len(left) > 0, "Left partition should not be empty"
    assert len(right) > 0, "Right partition should not be empty"
    assert len(left) + len(right) == len(indices), "Split should preserve all points"
    assert (
        hyperplane.shape[0] == hub_tree_bit_data.shape[1] * 2
    ), "Bit hyperplane should be 2x dim"


# ============================================================================
# Test hub tree construction
# ============================================================================


# ============================================================================
# Test hub tree construction (via end-to-end tests)
# Note: Direct tree construction tests removed due to Numba typed list
# serialization issues. The functionality is tested through query accuracy tests.
# ============================================================================


# ============================================================================
# Test end-to-end query accuracy with hub trees
# ============================================================================


def test_dense_euclidean_hub_tree_query_accuracy(hub_tree_data):
    """Test query accuracy after prepare() with dense euclidean hub tree."""
    train_data = hub_tree_data[100:]
    query_data = hub_tree_data[:100]

    nnd = NNDescent(train_data, metric="euclidean", n_neighbors=15, random_state=42)
    nnd.prepare()  # This builds the hub tree

    knn_indices, _ = nnd.query(query_data, k=10, epsilon=0.2)

    # Get true neighbors
    tree = KDTree(train_data)
    true_indices = tree.query(query_data, 10, return_distance=False)

    num_correct = 0.0
    for i in range(query_data.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (query_data.shape[0] * 10)
    assert percent_correct >= 0.90, f"Query accuracy too low: {percent_correct:.2%}"


def test_dense_angular_hub_tree_query_accuracy(hub_tree_data):
    """Test query accuracy after prepare() with dense angular hub tree."""
    angular_data = normalize(hub_tree_data, norm="l2").astype(np.float32)
    train_data = angular_data[100:]
    query_data = angular_data[:100]

    nnd = NNDescent(train_data, metric="cosine", n_neighbors=15, random_state=42)
    nnd.prepare()

    knn_indices, _ = nnd.query(query_data, k=10, epsilon=0.2)

    nn = NearestNeighbors(metric="cosine").fit(train_data)
    true_indices = nn.kneighbors(query_data, n_neighbors=10, return_distance=False)

    num_correct = 0.0
    for i in range(query_data.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (query_data.shape[0] * 10)
    assert percent_correct >= 0.90, f"Query accuracy too low: {percent_correct:.2%}"


def test_sparse_euclidean_hub_tree_query_accuracy(hub_tree_sparse_data):
    """Test query accuracy after prepare() with sparse euclidean hub tree."""
    train_data = hub_tree_sparse_data[100:]
    query_data = hub_tree_sparse_data[:100]

    nnd = NNDescent(train_data, metric="euclidean", n_neighbors=15, random_state=42)
    nnd.prepare()

    knn_indices, _ = nnd.query(query_data, k=10, epsilon=0.2)

    tree = KDTree(train_data.toarray())
    true_indices = tree.query(query_data.toarray(), 10, return_distance=False)

    num_correct = 0.0
    for i in range(query_data.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (query_data.shape[0] * 10)
    assert percent_correct >= 0.85, f"Query accuracy too low: {percent_correct:.2%}"


def test_sparse_angular_hub_tree_query_accuracy(hub_tree_sparse_data):
    """Test query accuracy after prepare() with sparse angular hub tree."""
    normalized_data = normalize(hub_tree_sparse_data, norm="l2")
    train_data = normalized_data[100:]
    query_data = normalized_data[:100]

    nnd = NNDescent(train_data, metric="cosine", n_neighbors=15, random_state=42)
    nnd.prepare()

    knn_indices, _ = nnd.query(query_data, k=10, epsilon=0.2)

    nn = NearestNeighbors(metric="cosine").fit(train_data.toarray())
    true_indices = nn.kneighbors(
        query_data.toarray(), n_neighbors=10, return_distance=False
    )

    num_correct = 0.0
    for i in range(query_data.shape[0]):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (query_data.shape[0] * 10)
    assert percent_correct >= 0.85, f"Query accuracy too low: {percent_correct:.2%}"


def test_bitpacked_hub_tree_query_accuracy(hub_tree_bit_data):
    """Test query accuracy after prepare() with bit-packed hub tree."""
    # Unpack for ground truth computation
    unpacked_data = np.zeros(
        (hub_tree_bit_data.shape[0], hub_tree_bit_data.shape[1] * 8), dtype=np.float32
    )
    for i in range(unpacked_data.shape[0]):
        for j in range(unpacked_data.shape[1]):
            unpacked_data[i, j] = (hub_tree_bit_data[i, j // 8] & (1 << (j % 8))) > 0

    train_idx = slice(100, None)
    query_idx = slice(0, 100)

    nnd = NNDescent(
        hub_tree_bit_data[train_idx],
        metric="bit_jaccard",
        n_neighbors=15,
        random_state=42,
    )
    nnd.prepare()

    knn_indices, _ = nnd.query(hub_tree_bit_data[query_idx], k=10, epsilon=0.3)

    nn = NearestNeighbors(metric="jaccard").fit(unpacked_data[train_idx])
    true_indices = nn.kneighbors(
        unpacked_data[query_idx], n_neighbors=10, return_distance=False
    )

    num_correct = 0.0
    for i in range(100):
        num_correct += np.sum(np.isin(true_indices[i], knn_indices[i]))

    percent_correct = num_correct / (100 * 10)
    assert percent_correct >= 0.70, f"Query accuracy too low: {percent_correct:.2%}"


# ============================================================================
# Test self-query accuracy (points should find themselves)
# ============================================================================


def test_dense_euclidean_hub_tree_self_query(hub_tree_data):
    """Test that points can find themselves after prepare() with hub tree."""
    nnd = NNDescent(hub_tree_data, metric="euclidean", n_neighbors=15, random_state=42)
    nnd.prepare()

    # Query first 50 points
    knn_indices, knn_distances = nnd.query(hub_tree_data[:50], k=1)

    self_found = sum(1 for i in range(50) if knn_indices[i, 0] == i)
    assert self_found >= 45, f"Self-query accuracy too low: {self_found}/50"


def test_dense_angular_hub_tree_self_query(hub_tree_data):
    """Test self-query with angular hub tree."""
    angular_data = normalize(hub_tree_data, norm="l2").astype(np.float32)

    nnd = NNDescent(angular_data, metric="cosine", n_neighbors=15, random_state=42)
    nnd.prepare()

    knn_indices, _ = nnd.query(angular_data[:50], k=1)

    self_found = sum(1 for i in range(50) if knn_indices[i, 0] == i)
    assert self_found >= 45, f"Self-query accuracy too low: {self_found}/50"


def test_sparse_euclidean_hub_tree_self_query(hub_tree_sparse_data):
    """Test self-query with sparse euclidean hub tree."""
    nnd = NNDescent(
        hub_tree_sparse_data, metric="euclidean", n_neighbors=15, random_state=42
    )
    nnd.prepare()

    knn_indices, _ = nnd.query(hub_tree_sparse_data[:50], k=1)

    self_found = sum(1 for i in range(50) if knn_indices[i, 0] == i)
    assert self_found >= 40, f"Self-query accuracy too low: {self_found}/50"


def test_sparse_angular_hub_tree_self_query(hub_tree_sparse_data):
    """Test self-query with sparse angular hub tree."""
    normalized_data = normalize(hub_tree_sparse_data, norm="l2")

    nnd = NNDescent(normalized_data, metric="cosine", n_neighbors=15, random_state=42)
    nnd.prepare()

    knn_indices, _ = nnd.query(normalized_data[:50], k=1)

    self_found = sum(1 for i in range(50) if knn_indices[i, 0] == i)
    assert self_found >= 40, f"Self-query accuracy too low: {self_found}/50"


def test_bitpacked_hub_tree_self_query(hub_tree_bit_data):
    """Test self-query with bit-packed hub tree."""
    nnd = NNDescent(
        hub_tree_bit_data, metric="bit_jaccard", n_neighbors=15, random_state=42
    )
    nnd.prepare()

    knn_indices, _ = nnd.query(hub_tree_bit_data[:50], k=1)

    self_found = sum(1 for i in range(50) if knn_indices[i, 0] == i)
    assert self_found >= 40, f"Self-query accuracy too low: {self_found}/50"
