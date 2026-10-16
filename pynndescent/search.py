"""Disk-cached graph search for dense float32 data.

Historically the search function was compiled as a closure over each index's
data, graph and tree, so that a call passed almost nothing and the compiled code
could treat those arrays as constants. The cost was a fresh compilation of the
search function in every ``prepare()``.

The functions here are compiled once, cached on disk, and shared by every index
whose distance is squared euclidean or alternative dot on dense float32 data.
Two things keep them as fast as the closure:

* the data, graph and tree arrays are typed read-only, which lets LLVM assume
  that writes to the result heaps and visited bitset never alias them, as it can
  for a closure's captured constants;
* queries call the compiled entry point for the one exact signature directly,
  skipping the dispatcher's per-argument type resolution (which would otherwise
  cost one to two microseconds per single-vector query). Argument types are
  fixed when the search function is built, and the query is checked by a cheap
  guard before each call.
"""
import heapq

import numba
import numpy as np
from numba import types

from pynndescent.distances import alternative_dot, squared_euclidean
from pynndescent.kernels import KERNEL_ALT_DOT, KERNEL_SQEUCLIDEAN, four_distances
from pynndescent.rp_trees import select_side
from pynndescent.utils import (
    check_and_mark_visited,
    make_heap,
    mark_visited,
    simple_heap_push,
    tau_rand_int,
)


def _array(dtype, ndim, readonly=False):
    return types.Array(dtype, ndim, "C", readonly=readonly)


_HEAP_TYPE = types.Tuple(
    (_array(types.int32, 2), _array(types.float32, 2), _array(types.uint8, 2))
)

SEARCH_SIGNATURE = _HEAP_TYPE(
    _array(types.float32, 2),  # query_points
    _array(types.float32, 2),  # tree_query_points
    _HEAP_TYPE,  # result (workspace heap, reset per query)
    types.int64,  # k
    types.float32,  # epsilon
    _array(types.uint8, 1),  # visited
    types.boolean,  # query_is_normalized
    _array(types.int64, 1),  # rng_state
    _array(types.float32, 2, readonly=True),  # data
    _array(types.int32, 1, readonly=True),  # indptr
    _array(types.int32, 1, readonly=True),  # indices
    _array(types.float32, 3, readonly=True),  # tree_hyperplanes (tree, node, dim)
    _array(types.float32, 2, readonly=True),  # tree_offsets (tree, node)
    _array(types.int32, 3, readonly=True),  # tree_children (tree, node, 2)
    _array(types.int32, 2, readonly=True),  # tree_indices (tree, point)
    types.int64,  # n_search_trees
    types.int64,  # tree_leaf_budget
    types.int64,  # n_neighbors
    types.float32,  # min_distance
    types.boolean,  # normalize_query
    types.boolean,  # tree_normalize_query
    types.int64,  # kernel (KERNEL_SQEUCLIDEAN or KERNEL_ALT_DOT)
)


@numba.njit(inline="always", fastmath=True)
def _distance(kernel, x, y):
    if kernel == KERNEL_SQEUCLIDEAN:
        return squared_euclidean(x, y)
    return alternative_dot(x, y)


@numba.njit(inline="always", locals={"node": types.uint32})
def _tree_leaf(point, tree_num, tree_hyperplanes, tree_offsets, tree_children, rng_state):
    node = 0
    while tree_children[tree_num, node, 0] > 0:
        side = select_side(
            tree_hyperplanes[tree_num, node], tree_offsets[tree_num, node], point, rng_state
        )
        if side == 0:
            node = tree_children[tree_num, node, 0]
        else:
            node = tree_children[tree_num, node, 1]
    return -tree_children[tree_num, node, 0], -tree_children[tree_num, node, 1]


@numba.njit(fastmath=True)
def _tree_spill_leaves(point, rng_state, leaves, tree_hyperplanes, tree_offsets, tree_children, leaf_budget):
    """Best-first descent of the first tree, collecting up to leaf_budget leaves."""
    max_tree_nodes = tree_children.shape[1]
    frontier_priorities = np.empty(max_tree_nodes, dtype=np.float32)
    frontier_nodes = np.empty(max_tree_nodes, dtype=np.int32)
    frontier_size = 1
    frontier_priorities[0] = 0.0
    frontier_nodes[0] = 0
    leaf_count = 0
    while leaf_count < leaf_budget and frontier_size > 0:
        best = 0
        for frontier_index in range(1, frontier_size):
            if frontier_priorities[frontier_index] < frontier_priorities[best]:
                best = frontier_index
        priority = frontier_priorities[best]
        node = frontier_nodes[best]
        frontier_size -= 1
        frontier_priorities[best] = frontier_priorities[frontier_size]
        frontier_nodes[best] = frontier_nodes[frontier_size]
        if tree_children[0, node, 0] <= 0:
            leaves[leaf_count, 0] = -tree_children[0, node, 0]
            leaves[leaf_count, 1] = -tree_children[0, node, 1]
            leaf_count += 1
            continue
        margin = tree_offsets[0, node]
        for dimension in range(point.shape[0]):
            margin += tree_hyperplanes[0, node, dimension] * point[dimension]
        if abs(margin) < 1.0e-8:
            primary_side = np.abs(tau_rand_int(rng_state)) % 2
        elif margin > 0.0:
            primary_side = 0
        else:
            primary_side = 1
        alternate_side = 1 - primary_side
        frontier_priorities[frontier_size] = priority
        frontier_nodes[frontier_size] = tree_children[0, node, primary_side]
        frontier_size += 1
        frontier_priorities[frontier_size] = max(priority, abs(margin))
        frontier_nodes[frontier_size] = tree_children[0, node, alternate_side]
        frontier_size += 1
    return leaf_count


@numba.njit(inline="always", fastmath=True)
def _push_seed(heap_priorities, heap_indices, seed_set, d, candidate):
    simple_heap_push(heap_priorities, heap_indices, d, candidate)
    heapq.heappush(seed_set, (d, candidate))


@numba.njit(
    inline="always",
    fastmath=True,
    locals={
        "candidate": types.int32,
        "vertex": types.int32,
        "d": types.float32,
        "d_vertex": types.float32,
        "distance_bound": types.float32,
    },
)
def _search_one(
    current_query,
    tree_current_query,
    heap_priorities,
    heap_indices,
    k,
    epsilon,
    visited,
    rng_state,
    data,
    indptr,
    indices,
    tree_hyperplanes,
    tree_offsets,
    tree_children,
    tree_indices,
    n_search_trees,
    tree_leaf_budget,
    n_neighbors,
    min_distance,
    kernel,
):
    seed_set = [(np.float32(np.inf), np.int32(-1)) for j in range(0)]

    # Initialise from all retained trees (or several leaves of the first tree)
    n_initial_points = 0
    if tree_leaf_budget == 1:
        for tree_num in range(n_search_trees):
            leaf_start, leaf_end = _tree_leaf(
                tree_current_query, tree_num, tree_hyperplanes, tree_offsets,
                tree_children, rng_state,
            )
            for j in range(leaf_start, leaf_end):
                candidate = tree_indices[tree_num, j]
                if check_and_mark_visited(visited, candidate) == 0:
                    d = np.float32(_distance(kernel, current_query, data[candidate]))
                    _push_seed(heap_priorities, heap_indices, seed_set, d, candidate)
                    n_initial_points += 1
    else:
        leaf_bounds = np.empty((tree_leaf_budget, 2), dtype=np.int32)
        leaf_count = _tree_spill_leaves(
            tree_current_query, rng_state, leaf_bounds, tree_hyperplanes,
            tree_offsets, tree_children, tree_leaf_budget,
        )
        for leaf_num in range(leaf_count):
            for j in range(leaf_bounds[leaf_num, 0], leaf_bounds[leaf_num, 1]):
                candidate = tree_indices[0, j]
                if check_and_mark_visited(visited, candidate) == 0:
                    d = np.float32(_distance(kernel, current_query, data[candidate]))
                    _push_seed(heap_priorities, heap_indices, seed_set, d, candidate)
                    n_initial_points += 1

    # Random samples if needed
    n_random_samples = min(k, n_neighbors) - n_initial_points
    if n_random_samples > 0:
        for j in range(n_random_samples):
            candidate = np.int32(np.abs(tau_rand_int(rng_state)) % data.shape[0])
            if check_and_mark_visited(visited, candidate) == 0:
                d = np.float32(_distance(kernel, current_query, data[candidate]))
                _push_seed(heap_priorities, heap_indices, seed_set, d, candidate)

    if len(seed_set) == 0:
        return

    # Graph search
    distance_bound = heap_priorities[0] + (
        epsilon * (heap_priorities[0] - min_distance)
    )
    d_vertex, vertex = heapq.heappop(seed_set)
    pending = np.empty(4, dtype=np.int32)
    pending_d = np.empty(4, dtype=np.float32)

    while d_vertex < distance_bound:
        # Unvisited neighbours are compared four at a time and then considered
        # in their original order, so the search path is unchanged.
        n_pending = 0
        for j in range(indptr[vertex], indptr[vertex + 1]):
            candidate = indices[j]
            if check_and_mark_visited(visited, candidate) == 0:
                pending[n_pending] = candidate
                n_pending += 1
                if n_pending == 4:
                    four_distances(current_query, data, pending, pending_d, kernel)
                    for r in range(4):
                        d = pending_d[r]
                        if d < distance_bound:
                            _push_seed(heap_priorities, heap_indices, seed_set, d, pending[r])
                            distance_bound = heap_priorities[0] + (
                                epsilon * (heap_priorities[0] - min_distance)
                            )
                    n_pending = 0
        for r in range(n_pending):
            d = np.float32(_distance(kernel, current_query, data[pending[r]]))
            if d < distance_bound:
                _push_seed(heap_priorities, heap_indices, seed_set, d, pending[r])
                distance_bound = heap_priorities[0] + (
                    epsilon * (heap_priorities[0] - min_distance)
                )

        if len(seed_set) == 0:
            break
        d_vertex, vertex = heapq.heappop(seed_set)


@numba.njit(inline="always")
def _prepare_queries(i, query_points, tree_query_points, query_is_normalized, normalize_query, tree_normalize_query):
    """Return (ok, current_query, tree_current_query) for query i."""
    if normalize_query and not query_is_normalized:
        norm = np.sqrt((query_points[i] ** 2).sum())
        if norm > 0.0:
            current_query = query_points[i] / norm
        else:
            return False, query_points[i], tree_query_points[i]
    else:
        current_query = query_points[i]
    if tree_normalize_query:
        # Zero queries have no defined angle; leave their row empty.
        tree_norm = np.sqrt((tree_query_points[i] ** 2).sum())
        if tree_norm == 0.0:
            return False, current_query, tree_query_points[i]
        if query_is_normalized:
            tree_current_query = tree_query_points[i]
        else:
            tree_current_query = tree_query_points[i] / tree_norm
    else:
        tree_current_query = tree_query_points[i]
    return True, current_query, tree_current_query


@numba.njit(fastmath=True, cache=True)
def search_dense(
    query_points, tree_query_points, result, k, epsilon, visited, query_is_normalized,
    rng_state, data, indptr, indices, tree_hyperplanes, tree_offsets, tree_children,
    tree_indices, n_search_trees, tree_leaf_budget, n_neighbors, min_distance,
    normalize_query, tree_normalize_query, kernel,
):
    internal_rng_state = np.copy(rng_state)
    for i in range(query_points.shape[0]):
        visited[:] = 0
        heap_priorities = result[1][i]
        heap_indices = result[0][i]
        heap_priorities[:] = np.inf
        heap_indices[:] = -1
        ok, current_query, tree_current_query = _prepare_queries(
            i, query_points, tree_query_points, query_is_normalized,
            normalize_query, tree_normalize_query,
        )
        if not ok:
            continue
        _search_one(
            current_query, tree_current_query, heap_priorities, heap_indices, k,
            epsilon, visited, internal_rng_state, data, indptr, indices,
            tree_hyperplanes, tree_offsets, tree_children, tree_indices,
            n_search_trees, tree_leaf_budget, n_neighbors, min_distance, kernel,
        )
    return result


@numba.njit(fastmath=True, cache=True, parallel=True)
def search_dense_parallel(
    query_points, tree_query_points, result, k, epsilon, visited, query_is_normalized,
    rng_state, data, indptr, indices, tree_hyperplanes, tree_offsets, tree_children,
    tree_indices, n_search_trees, tree_leaf_budget, n_neighbors, min_distance,
    normalize_query, tree_normalize_query, kernel,
):
    internal_rng_state = np.copy(rng_state)
    for i in numba.prange(query_points.shape[0]):
        # Each query needs its own visited set when searched in parallel
        visited_nodes = np.zeros_like(visited)
        heap_priorities = result[1][i]
        heap_indices = result[0][i]
        heap_priorities[:] = np.inf
        heap_indices[:] = -1
        ok, current_query, tree_current_query = _prepare_queries(
            i, query_points, tree_query_points, query_is_normalized,
            normalize_query, tree_normalize_query,
        )
        if not ok:
            continue
        _search_one(
            current_query, tree_current_query, heap_priorities, heap_indices, k,
            epsilon, visited_nodes, internal_rng_state, data, indptr, indices,
            tree_hyperplanes, tree_offsets, tree_children, tree_indices,
            n_search_trees, tree_leaf_budget, n_neighbors, min_distance, kernel,
        )
    return result


def _readonly(array, dtype):
    view = np.ascontiguousarray(array, dtype=dtype).view()
    view.flags.writeable = False
    return view


def stack_search_forest(search_forest, n_points):
    """Stack the search trees into (tree, ...) arrays, padding smaller trees."""
    n_trees = len(search_forest)
    max_nodes = max(tree.children.shape[0] for tree in search_forest)
    dim = search_forest[0].hyperplanes.shape[1]
    hyperplanes = np.zeros((n_trees, max_nodes, dim), dtype=np.float32)
    offsets = np.zeros((n_trees, max_nodes), dtype=np.float32)
    children = np.full((n_trees, max_nodes, 2), -1, dtype=np.int32)
    indices = np.empty((n_trees, n_points), dtype=np.int32)
    for tree_num, tree in enumerate(search_forest):
        n_nodes = tree.children.shape[0]
        hyperplanes[tree_num, :n_nodes] = tree.hyperplanes
        offsets[tree_num, :n_nodes] = tree.offsets
        children[tree_num, :n_nodes] = tree.children
        indices[tree_num] = tree.indices
    return hyperplanes, offsets, children, indices


def make_cached_search_function(
    data, indptr, indices, search_forest, tree_leaf_budget, n_neighbors,
    min_distance, normalize_query, tree_normalize_query, kernel, parallel,
):
    """Bind an index's arrays to the cached search function.

    Returns a callable with the dense search closure's interface,
    ``search(query_points, tree_query_points, query_inv_norms, result, k,
    epsilon, visited, query_is_normalized, rng_state) -> result``.
    ``query_inv_norms`` is only used by int8 quantised search, which does not
    use this path, and is ignored.
    """
    dispatcher = search_dense_parallel if parallel else search_dense
    hyperplanes, offsets, children, tree_indices = stack_search_forest(
        search_forest, data.shape[0]
    )
    bound = (
        _readonly(data, np.float32),
        _readonly(indptr, np.int32),
        _readonly(indices, np.int32),
        _readonly(hyperplanes, np.float32),
        _readonly(offsets, np.float32),
        _readonly(children, np.int32),
        _readonly(tree_indices, np.int32),
        len(search_forest),
        int(tree_leaf_budget),
        int(n_neighbors),
        np.float32(min_distance),
        bool(normalize_query),
        bool(tree_normalize_query),
        int(kernel),
    )

    # Compile (or load from the on-disk cache) exactly this signature, then call
    # its entry point directly to skip per-call type resolution.
    entry_point = None
    try:
        dispatcher.compile(SEARCH_SIGNATURE.args)
        entry_point = dispatcher.overloads[SEARCH_SIGNATURE.args].entry_point
    except (AttributeError, KeyError, TypeError):
        entry_point = None

    float32, uint8, int64, int32 = np.float32, np.uint8, np.int64, np.int32

    def _checked(query_points, tree_query_points, result, visited, rng_state):
        return (
            query_points.dtype == float32
            and query_points.ndim == 2
            and query_points.flags.c_contiguous
            and query_points.flags.writeable
            and tree_query_points.dtype == float32
            and tree_query_points.ndim == 2
            and tree_query_points.flags.c_contiguous
            and tree_query_points.flags.writeable
            and result[0].dtype == int32
            and result[0].flags.c_contiguous
            and result[1].dtype == float32
            and result[1].flags.c_contiguous
            and result[2].dtype == uint8
            and result[2].flags.c_contiguous
            and visited.dtype == uint8
            and rng_state.dtype == int64
        )

    def search(
        query_points, tree_query_points, query_inv_norms, result, k, epsilon,
        visited, query_is_normalized, rng_state,
    ):
        # The entry point does not check argument types; anything unexpected
        # goes through the dispatcher, which converts or raises as usual.
        if entry_point is not None and _checked(
            query_points, tree_query_points, result, visited, rng_state
        ):
            return entry_point(
                query_points, tree_query_points, result, k, epsilon, visited,
                bool(query_is_normalized), rng_state, *bound,
            )
        return dispatcher(
            np.ascontiguousarray(query_points, dtype=float32),
            np.ascontiguousarray(tree_query_points, dtype=float32),
            result, k, epsilon, visited, bool(query_is_normalized), rng_state,
            *bound,
        )

    return search


# deheap_sort variants, created once per process. Building them per index (as a
# fresh numba.njit of deheap_sort.py_func) recompiled them in every prepare().
from pynndescent.utils import deheap_sort as _deheap_sort

deheap_sort_serial = numba.njit()(_deheap_sort.py_func)
deheap_sort_parallel = numba.njit(parallel=True)(_deheap_sort.py_func)
