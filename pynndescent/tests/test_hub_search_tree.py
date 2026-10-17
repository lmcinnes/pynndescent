import numba
import numpy as np
import pytest
from scipy import sparse

from pynndescent import NNDescent
from pynndescent.distances import popcnt_u8
from pynndescent.hub_tree import (
    CRITERION_BALANCE,
    CRITERION_EDGE_CUT,
    FALLBACK_LEAF,
    FALLBACK_MEDIAN,
    FALLBACK_RANDOM,
    make_hub_search_tree,
)
from pynndescent.rp_trees import (
    convert_tree_format,
    make_bit_hub_tree,
    make_hub_tree,
    make_sparse_hub_tree,
)

@numba.njit()
def _bit_margin(plane, point):
    dim = point.shape[0]
    margin = 0
    for d in range(dim):
        margin += popcnt_u8(plane[d] & point[d])
        margin -= popcnt_u8(plane[dim + d] & point[d])
    return margin


# local_bytes=0 sends every split through the level-synchronous kernels, a huge
# value sends every split through the depth-first local builder.
LOCAL_BYTES = [0, None, 10**12]


def _data(kind, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "dense":
        return rng.normal(size=(n, 12)).astype(np.float32)
    if kind == "sparse":
        X = rng.normal(size=(n, 60)).astype(np.float32)
        X[rng.random(X.shape) < 0.8] = 0.0
        X[np.abs(X).sum(axis=1) == 0, 0] = 1.0
        X = sparse.csr_matrix(X)
        X.sort_indices()
        return X
    return np.packbits(rng.random((n, 96)) < 0.3, axis=1)


def _graph(data, metric):
    return NNDescent(data, metric=metric, n_neighbors=15, random_state=0)._neighbor_graph[0]


def _serial_tree(data, graph, angular, leaf_size=30):
    rng_state = np.array([1, 2, 3], dtype=np.int64)
    if sparse.issparse(data):
        tree = make_sparse_hub_tree(
            data.indices, data.indptr, data.data, graph, rng_state,
            leaf_size=leaf_size, angular=angular, max_depth=200,
        )
    elif data.dtype == np.uint8:
        tree = make_bit_hub_tree(data, graph, rng_state, leaf_size=leaf_size, max_depth=200)
    else:
        tree = make_hub_tree(
            data, graph, rng_state, leaf_size=leaf_size, angular=angular, max_depth=200
        )
    return convert_tree_format(tree, data.shape[0], data.shape[1])


def _assert_same_tree(a, b):
    for name in ("children", "indices", "offsets", "hyperplanes"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert a.leaf_size == b.leaf_size


def _check_structure(tree, n, leaf_size, allow_large_leaves=False):
    children = tree.children
    assert np.array_equal(np.sort(tree.indices), np.arange(n))
    is_leaf = children[:, 0] <= 0
    bounds = -children[is_leaf]
    order = np.argsort(bounds[:, 0])
    starts, ends = bounds[order, 0], bounds[order, 1]
    assert starts[0] == 0 and ends[-1] == n
    assert np.all(starts[1:] == ends[:-1])
    assert np.all(ends > starts)  # no empty leaves
    if not allow_large_leaves:
        assert np.all(ends - starts <= leaf_size)
    internal = np.flatnonzero(~is_leaf)
    assert np.all(children[internal] > internal[:, None])  # pre-order
    assert np.all(children[internal] < children.shape[0])


@pytest.mark.parametrize("local_bytes", LOCAL_BYTES)
@pytest.mark.parametrize(
    "kind,metric,angular",
    [
        ("dense", "euclidean", False),
        ("dense", "cosine", True),
        ("sparse", "euclidean", False),
        ("sparse", "cosine", True),
        ("bit", "bit_jaccard", False),
    ],
)
def test_matches_serial_hub_tree(kind, metric, angular, local_bytes):
    data = _data(kind)
    if kind == "dense" and angular:
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
    graph = _graph(data, metric)
    kwargs = {} if local_bytes is None else {"local_bytes": local_bytes}
    tree = make_hub_search_tree(data, graph, angular=angular, **kwargs)
    _assert_same_tree(_serial_tree(data, graph, angular), tree)


@pytest.mark.parametrize("fallback", [FALLBACK_LEAF, FALLBACK_RANDOM, FALLBACK_MEDIAN])
@pytest.mark.parametrize("criterion", [CRITERION_BALANCE, CRITERION_EDGE_CUT])
@pytest.mark.parametrize("kind", ["dense", "sparse", "bit"])
def test_independent_of_threads_and_code_path(kind, criterion, fallback):
    data = _data(kind, n=4000, seed=1)
    graph = _graph(data, "bit_jaccard" if kind == "bit" else "euclidean")
    kwargs = dict(criterion=criterion, min_balance=0.25, fallback=fallback, seed=7)
    reference = make_hub_search_tree(data, graph, local_bytes=10**12, **kwargs)
    _check_structure(
        reference, data.shape[0], 30, allow_large_leaves=fallback == FALLBACK_LEAF
    )
    original_threads = numba.get_num_threads()
    try:
        for threads in sorted({1, 2, original_threads}):
            numba.set_num_threads(threads)
            for local_bytes in (0, None):
                kw = dict(kwargs)
                if local_bytes is not None:
                    kw["local_bytes"] = local_bytes
                _assert_same_tree(reference, make_hub_search_tree(data, graph, **kw))
    finally:
        numba.set_num_threads(original_threads)


@pytest.mark.parametrize("kind", ["dense", "bit"])
def test_training_points_route_to_their_own_leaf(kind):
    data = _data(kind, n=3000, seed=2)
    graph = _graph(data, "bit_jaccard" if kind == "bit" else "euclidean")
    tree = make_hub_search_tree(data, graph, fallback=FALLBACK_MEDIAN)
    leaf_of = np.empty(data.shape[0], dtype=np.int64)
    for node in np.flatnonzero(tree.children[:, 0] <= 0):
        start, end = -tree.children[node]
        leaf_of[tree.indices[start:end]] = node

    n_untied = 0
    for i in range(data.shape[0]):
        node = 0
        tied = False
        while tree.children[node, 0] > 0:
            plane = tree.hyperplanes[node]
            if kind == "bit":
                margin = float(tree.offsets[node]) + _bit_margin(plane, data[i])
                tied |= abs(margin) < 1e-8
            else:
                margin = float(tree.offsets[node]) + float(
                    plane.astype(np.float64) @ data[i].astype(np.float64)
                )
                tied |= abs(margin) < 1e-4
            node = tree.children[node, 0] if margin > 0 else tree.children[node, 1]
        if not tied:
            n_untied += 1
            assert node == leaf_of[i], f"point {i} routed to the wrong leaf"
    # Ties (points on a hyperplane) are broken at random by the search code
    assert n_untied > (0.95 if kind == "dense" else 0.3) * data.shape[0]


def test_median_fallback_avoids_oversized_leaves():
    rng = np.random.default_rng(3)
    # A tight cluster far from the origin: angular hub splits through the origin
    # cannot separate it, which used to leave the whole cluster in one leaf.
    data = rng.normal(size=(3000, 8)).astype(np.float32) * 0.01 + 1.0
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    graph = _graph(data, "cosine")
    leafy = make_hub_search_tree(data, graph, angular=True, fallback=FALLBACK_LEAF)
    median = make_hub_search_tree(data, graph, angular=True, fallback=FALLBACK_MEDIAN)
    _check_structure(median, data.shape[0], 30)
    assert median.leaf_size == 30
    assert leafy.leaf_size >= median.leaf_size


@pytest.mark.parametrize("kind", ["dense", "sparse", "bit"])
@pytest.mark.parametrize("fallback", [FALLBACK_LEAF, FALLBACK_RANDOM, FALLBACK_MEDIAN])
def test_degenerate_inputs(kind, fallback):
    data = _data(kind, n=200, seed=4)
    graph = _graph(data, "bit_jaccard" if kind == "bit" else "euclidean")
    # Fewer points than a leaf
    tiny = data[:10]
    tree = make_hub_search_tree(tiny, graph[:10] % 10, fallback=fallback)
    assert tree.children.shape == (1, 2)
    np.testing.assert_array_equal(np.sort(tree.indices), np.arange(10))
    # All rows identical: no hyperplane separates them
    if sparse.issparse(data):
        dup = sparse.csr_matrix(np.repeat(data[:1].toarray(), 200, axis=0))
    else:
        dup = np.repeat(data[:1], 200, axis=0)
    tree = make_hub_search_tree(dup, graph, fallback=fallback)
    _check_structure(tree, 200, 30, allow_large_leaves=fallback != FALLBACK_RANDOM)
    # Depth limit
    tree = make_hub_search_tree(data, graph, max_depth=2, fallback=fallback)
    _check_structure(tree, data.shape[0], 30, allow_large_leaves=True)
    assert tree.children.shape[0] <= 7
