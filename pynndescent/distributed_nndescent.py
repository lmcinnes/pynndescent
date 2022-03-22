import numpy as np
import numba
import itertools
import scipy.sparse

from warnings import warn
from sklearn.utils import check_random_state

try:
    from dask import delayed
    import dask.array as da
except ImportError:
    warn("Distributed NNDescent requires dask to run in parallel. Please install dask to get the"
         "most out of the distributed code")
    # TODO: define an empty decorator for delayed

from pynndescent.utils import (
    simple_heap_push,
    mark_visited,
    has_been_visited,
    make_heap,
    tau_rand_int,
    checked_heap_push,
    deheap_sort,
)
from pynndescent.rp_trees import (
    select_side,
    make_forest,
    convert_tree_format,
    rptree_leaf_array,
)
from pynndescent.pynndescent_ import nn_descent, diversify, diversify_csr, degree_prune

import pynndescent.sparse as sparse
import pynndescent.distances as pynnd_dist

import heapq

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1
FLOAT32_EPS = np.finfo(np.float32).eps
alternative_dot = pynnd_dist.alternative_dot
alternative_cosine = pynnd_dist.alternative_cosine


@numba.njit(
    fastmath=True,
    locals={
        "current_query": numba.types.float32[::1],
        "i": numba.types.uint32,
        "j": numba.types.uint32,
        "heap_priorities": numba.types.float32[::1],
        "heap_indices": numba.types.int32[::1],
        "candidate": numba.types.int32,
        "vertex": numba.types.int32,
        "d": numba.types.float32,
        "d_vertex": numba.types.float32,
        "visited": numba.types.uint8[::1],
        "indices": numba.types.int32[::1],
        "indptr": numba.types.int32[::1],
        "data": numba.types.float32[:, ::1],
        "heap_size": numba.types.int16,
        "distance_scale": numba.types.float32,
        "distance_bound": numba.types.float32,
        "seed_scale": numba.types.float32,
    },
    parallel=True,
)
def search_with_tree(
    query_points,
    k,
    epsilon,
    visited,
    search_tree,
    data,
    indptr,
    indices,
    dist,
    n_neighbors,
    rng_state,
):
    result = make_heap(query_points.shape[0], k)
    distance_scale = 1.0 + epsilon
    tree_hyperplanes = search_tree.hyperplanes
    tree_offsets = search_tree.offsets
    tree_children = search_tree.children
    tree_indices = np.arange(data.shape[0]).astype(np.int32)

    for i in numba.prange(query_points.shape[0]):
        internal_rng_state = np.copy(rng_state)
        visited_nodes = np.zeros_like(visited)

        if dist == alternative_dot or dist == alternative_cosine:
            norm = np.sqrt((query_points[i] ** 2).sum())
            if norm > 0.0:
                current_query = query_points[i] / norm
            else:
                continue
        else:
            current_query = query_points[i]

        heap_priorities = result[1][i]
        heap_indices = result[0][i]
        seed_set = [(np.float32(np.inf), np.int32(-1)) for j in range(0)]

        # ---------- tree search ---------#
        node = 0
        while tree_children[node, 0] > 0:
            side = select_side(
                tree_hyperplanes[node],
                tree_offsets[node],
                current_query,
                internal_rng_state,
            )
            if side == 0:
                node = tree_children[node, 0]
            else:
                node = tree_children[node, 1]

        index_bounds = -tree_children[node]

        ############ Init ################
        candidate_indices = tree_indices[index_bounds[0] : index_bounds[1]]

        n_initial_points = candidate_indices.shape[0]
        n_random_samples = min(k, n_neighbors) - n_initial_points

        for j in range(n_initial_points):
            candidate = candidate_indices[j]
            d = np.float32(dist(data[candidate], current_query))
            # indices are guaranteed different
            simple_heap_push(heap_priorities, heap_indices, d, candidate)
            heapq.heappush(seed_set, (d, np.int32(candidate)))
            mark_visited(visited_nodes, candidate)

        if n_random_samples > 0:
            for j in range(n_random_samples):
                candidate = np.int32(
                    np.abs(tau_rand_int(internal_rng_state)) % data.shape[0]
                )
                if has_been_visited(visited_nodes, candidate) == 0:
                    d = np.float32(dist(data[candidate], current_query))
                    simple_heap_push(heap_priorities, heap_indices, d, candidate)
                    heapq.heappush(seed_set, (d, candidate))
                    mark_visited(visited_nodes, candidate)

        ############ Search ##############
        distance_bound = distance_scale * heap_priorities[0]

        # Find smallest seed point
        d_vertex, vertex = heapq.heappop(seed_set)

        while d_vertex < distance_bound:
            for j in range(indptr[vertex], indptr[vertex + 1]):

                candidate = indices[j]

                if has_been_visited(visited_nodes, candidate) == 0:
                    mark_visited(visited_nodes, candidate)

                    d = np.float32(dist(data[candidate], current_query))

                    if d < distance_bound:
                        simple_heap_push(heap_priorities, heap_indices, d, candidate)
                        heapq.heappush(seed_set, (d, candidate))
                        # Update bound
                        distance_bound = distance_scale * heap_priorities[0]

            # find new smallest seed point
            if len(seed_set) == 0:
                break
            else:
                d_vertex, vertex = heapq.heappop(seed_set)

    return result


@numba.njit(
    fastmath=True,
    locals={
        "current_query": numba.types.float32[::1],
        "i": numba.types.uint32,
        "j": numba.types.uint32,
        "heap_priorities": numba.types.float32[::1],
        "heap_indices": numba.types.int32[::1],
        "candidate": numba.types.int32,
        "vertex": numba.types.int32,
        "d": numba.types.float32,
        "d_vertex": numba.types.float32,
        "visited": numba.types.uint8[::1],
        "indices": numba.types.int32[::1],
        "indptr": numba.types.int32[::1],
        "data": numba.types.float32[:, ::1],
        "heap_size": numba.types.int16,
        "distance_scale": numba.types.float32,
        "distance_bound": numba.types.float32,
        "seed_scale": numba.types.float32,
    },
    parallel=True,
)
def search_with_reverse_neighbors(
    query_points,
    k,
    epsilon,
    init_indices,
    init_dists,
    search_tree,
    visited_,
    data,
    indptr,
    indices,
    dist,
    n_neighbors,
    rng_state,
):
    result = make_heap(query_points.shape[0], k)
    distance_scale = 1.0 + epsilon
    internal_rng_state = np.copy(rng_state)
    tree_hyperplanes = search_tree.hyperplanes
    tree_offsets = search_tree.offsets
    tree_children = search_tree.children
    tree_indices = np.arange(data.shape[0]).astype(np.int32)

    for i in range(init_indices.shape[0]):
        for j in range(init_indices.shape[1]):
            l = init_indices[i, j]
            if l >= 0 and l < result[0].shape[0]:
                d = init_dists[i, j]
                simple_heap_push(result[1][l], result[0][l], d, i)

    for i in numba.prange(query_points.shape[0]):
        visited = np.zeros_like(visited_)
        if dist == alternative_dot or dist == alternative_cosine:
            norm = np.sqrt((query_points[i] ** 2).sum())
            if norm > 0.0:
                current_query = query_points[i] / norm
            else:
                continue
        else:
            current_query = query_points[i]

        heap_priorities = result[1][i]
        heap_indices = result[0][i]
        seed_set = [(np.float32(np.inf), np.int32(-1)) for j in range(0)]

        ############ Init ################
        n_initial_points = 0
        for j in range(heap_indices.shape[0]):
            candidate = heap_indices[j]
            d = heap_priorities[j]
            if candidate >= 0:
                mark_visited(visited, candidate)
                heapq.heappush(seed_set, (d, candidate))
                n_initial_points += 1

        if n_initial_points == 0:
            # ---------- tree search ---------#
            node = 0
            while tree_children[node, 0] > 0:
                side = select_side(
                    tree_hyperplanes[node],
                    tree_offsets[node],
                    current_query,
                    internal_rng_state,
                )
                if side == 0:
                    node = tree_children[node, 0]
                else:
                    node = tree_children[node, 1]

            index_bounds = -tree_children[node]

            candidate_indices = tree_indices[index_bounds[0] : index_bounds[1]]

            n_initial_points = candidate_indices.shape[0]
            n_random_samples = min(k, n_neighbors) - n_initial_points

            for j in range(candidate_indices.shape[0]):
                candidate = candidate_indices[j]
                d = np.float32(dist(data[candidate], current_query))
                if d < heap_priorities[0] and has_been_visited(visited, candidate) == 0:
                    # indices are guaranteed different
                    simple_heap_push(heap_priorities, heap_indices, d, candidate)
                    heapq.heappush(seed_set, (d, np.int32(candidate)))
                    mark_visited(visited, candidate)
                    n_initial_points += 1

            n_random_samples = min(k, n_neighbors) - n_initial_points

            if n_random_samples > 0:
                for j in range(n_random_samples):
                    candidate = np.int32(
                        np.abs(tau_rand_int(internal_rng_state)) % data.shape[0]
                    )
                    if has_been_visited(visited, candidate) == 0:
                        d = dist(data[candidate], current_query)
                        simple_heap_push(heap_priorities, heap_indices, d, candidate)
                        heapq.heappush(seed_set, (d, candidate))
                        mark_visited(visited, candidate)

        ############ Search ##############
        distance_bound = distance_scale * heap_priorities[0]

        # Find smallest seed point
        d_vertex, vertex = heapq.heappop(seed_set)

        while d_vertex < distance_bound:

            for j in range(indptr[vertex], indptr[vertex + 1]):

                candidate = indices[j]

                if has_been_visited(visited, candidate) == 0:
                    mark_visited(visited, candidate)

                    d = dist(data[candidate], current_query)

                    if d < distance_bound:
                        simple_heap_push(heap_priorities, heap_indices, d, candidate)
                        heapq.heappush(seed_set, (d, candidate))
                        # Update bound
                        distance_bound = distance_scale * heap_priorities[0]

            # find new smallest seed point
            if len(seed_set) == 0:
                break
            else:
                d_vertex, vertex = heapq.heappop(seed_set)

    return result


@delayed
def create_forest(
    data, n_neighbors=30, n_trees=None, angular_trees=False, random_state=None
):
    current_random_state = check_random_state(random_state)
    rng_state = current_random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    if n_trees is None:
        n_trees = 5 + int(round(data.shape[0] ** 0.25))
        n_trees = min(32, n_trees)

    result = make_forest(
        data,
        n_neighbors,
        n_trees,
        None,
        rng_state,
        current_random_state,
        None,
        angular_trees,
    )
    leaf_array = rptree_leaf_array(result)
    return convert_tree_format(result[0], data.shape[0]), leaf_array


@delayed
def get_neighbor_graph(
    data,
    leaf_array,
    metric="euclidean",
    n_neighbors=30,
    n_iters=None,
    delta=0.001,
    random_state=None,
):
    current_random_state = check_random_state(random_state)
    rng_state = current_random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    if n_iters is None:
        n_iters = max(5, int(round(np.log2(data.shape[0]))))
    effective_max_candidates = min(60, n_neighbors)
    if metric in pynnd_dist.fast_distance_alternatives:
        _distance_func = pynnd_dist.fast_distance_alternatives[metric]["dist"]
    else:
        _distance_func = pynnd_dist.named_distances[metric]

    result = nn_descent(
        data,
        n_neighbors,
        rng_state,
        effective_max_candidates,
        _distance_func,
        n_iters,
        delta,
        low_memory=True,
        rp_tree_init=True,
        init_graph=make_heap(1, 1),
        leaf_array=leaf_array,
    )
    return result


@delayed
def build_search_graph(
    data,
    neighbor_indices,
    neighbor_dists,
    n_neighbors=30,
    metric="euclidean",
    pruning_degree_multiplier=1.5,
    random_state=None,
):
    current_random_state = check_random_state(random_state)
    rng_state = current_random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    if metric in pynnd_dist.fast_distance_alternatives:
        _distance_func = pynnd_dist.fast_distance_alternatives[metric]["dist"]
    else:
        _distance_func = pynnd_dist.named_distances[metric]

    diversified_rows, diversified_data = diversify(
        neighbor_indices.copy(),
        neighbor_dists.copy(),
        data,
        _distance_func,
        rng_state,
        1.0,
    )
    result = scipy.sparse.coo_matrix((data.shape[0], data.shape[0]), dtype=np.float32)
    diversified_data[diversified_data == 0.0] = FLOAT32_EPS
    result.row = np.repeat(
        np.arange(diversified_rows.shape[0], dtype=np.int32), diversified_rows.shape[1],
    )
    result.col = diversified_rows.ravel()
    result.data = diversified_data.ravel()

    result = result.tocsr()
    result.data[result.indices == -1] = 0.0
    result.eliminate_zeros()

    reverse_graph = result.transpose()
    diversify_csr(
        reverse_graph.indptr,
        reverse_graph.indices,
        reverse_graph.data,
        data,
        _distance_func,
        rng_state,
        1.0,
    )
    reverse_graph.eliminate_zeros()
    reverse_graph = reverse_graph.tocsr()
    reverse_graph.sort_indices()

    result = result.tocsr()
    result.sort_indices()

    result = result.maximum(reverse_graph).tocsr()

    result.setdiag(0.0)
    result.eliminate_zeros()
    result = degree_prune(
        result, int(np.round(pruning_degree_multiplier * n_neighbors)),
    )
    result.eliminate_zeros()
    result.sort_indices()

    return result.indptr, result.indices


@numba.njit()
def init_heap_numba(init_data, index_offset, size):
    result = make_heap(init_data[0].shape[0], size)
    for i in range(init_data[0].shape[0]):
        heap_indices = result[0][i]
        heap_priorities = result[1][i]
        other_indices = init_data[0][i] + index_offset
        other_priorities = init_data[1][i]
        for j in range(other_indices.shape[0]):
            if other_indices[j] >= 0 and other_priorities[j] < heap_priorities[0]:
                simple_heap_push(
                    heap_priorities, heap_indices, other_priorities[j], other_indices[j]
                )

    return result


@delayed
def init_heap(neighbors, offset, size=None):
    if size is None:
        _size = neighbors[0].shape[1]
    else:
        _size = size
    result = init_heap_numba(neighbors, offset, size)
    return result


@numba.njit(parallel=True, fastmath=True)
def update_heap(heap_to_update, other_heap, other_heap_index_offset):
    for i in numba.prange(other_heap[0].shape[0]):
        heap_indices = heap_to_update[0][i]
        heap_priorities = heap_to_update[1][i]
        other_indices = other_heap[0][i] + other_heap_index_offset
        other_priorities = other_heap[1][i]
        for j in range(other_indices.shape[0]):
            if other_indices[j] >= 0 and other_priorities[j] < heap_priorities[0]:
                checked_heap_push(
                    heap_priorities, heap_indices, other_priorities[j], other_indices[j]
                )
    return heap_to_update


@numba.njit()
def search_in_pair(
    left_data,
    right_data,
    left_tree,
    right_tree,
    left_indptr,
    right_indptr,
    left_indices,
    right_indices,
    distance,
    rng_state,
    n_neighbors=30,
    epsilon=0.01,
):
    right_visited = np.zeros((right_data.shape[0] // 8) + 1, dtype=np.uint8)
    left_visited = np.zeros((left_data.shape[0] // 8) + 1, dtype=np.uint8)

    left_neighbors_in_right = search_with_tree(
        left_data,
        n_neighbors,
        epsilon,
        right_visited,
        right_tree,
        right_data,
        right_indptr,
        right_indices,
        distance,
        n_neighbors,
        rng_state,
    )
    left_nn_inds, left_nn_dists = left_neighbors_in_right[:2]
    left_result = (left_nn_inds, left_nn_dists)

    init_inds = left_nn_inds
    init_dists = left_nn_dists
    right_neighbors_in_left = search_with_reverse_neighbors(
        right_data,
        n_neighbors,
        epsilon,
        init_inds,
        init_dists,
        left_tree,
        left_visited,
        left_data,
        left_indptr,
        left_indices,
        distance,
        n_neighbors,
        rng_state,
    )
    right_nn_inds, right_nn_dists = right_neighbors_in_left[:2]
    right_result = (right_nn_inds, right_nn_dists)

    return (left_result, right_result)


@delayed
def pair_updates(
    left_data,
    left_tree,
    left_graph,
    right_data,
    right_tree,
    right_graph,
    metric="euclidean",
    n_neighbors=30,
    epsilon=0.01,
    random_state=None,
):
    current_random_state = check_random_state(random_state)
    rng_state = current_random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    if metric in pynnd_dist.fast_distance_alternatives:
        _distance_func = pynnd_dist.fast_distance_alternatives[metric]["dist"]
    else:
        _distance_func = pynnd_dist.named_distances[metric]

    left_indptr = left_graph[0]
    left_indices = left_graph[1]
    right_indptr = right_graph[0]
    right_indices = right_graph[1]

    result = search_in_pair(
        left_data,
        right_data,
        left_tree,
        right_tree,
        left_indptr,
        right_indptr,
        left_indices,
        right_indices,
        _distance_func,
        rng_state,
        n_neighbors=n_neighbors,
        epsilon=epsilon,
    )

    return result


@delayed
def update_single_heap(base_heap, updates, offset):
    result = update_heap(base_heap, updates, offset)
    del base_heap
    del updates
    return result


def distributed_nndescent(
    dask_array,
    n_neighbors=30,
    metric="euclidean",
    n_internal_neighbors=30,
    n_trees=None,
    n_iters=None,
    delta=0.001,
    epsilon=0.01,
    pruning_degree_multiplier=1.5,
    random_state=None,
):
    chunk_size = dask_array.chunks[0][0]
    data_slices = dask_array.to_delayed().ravel()
    tree_and_leaf_arrays = [
        create_forest(
            local_data,
            n_neighbors=n_internal_neighbors,
            n_trees=n_trees,
            angular_trees=False,
            random_state=random_state,
        )
        for local_data in data_slices
    ]
    trees = [x[0] for x in tree_and_leaf_arrays]
    neighbor_graphs = [
        get_neighbor_graph(
            local_data,
            tree_and_leaf_arrays[i][1],
            metric=metric,
            n_neighbors=n_internal_neighbors,
            n_iters=n_iters,
            delta=delta,
            random_state=random_state,
        )
        for i, local_data in enumerate(data_slices)
    ]
    current_heaps = [
        init_heap(neighbors, i * chunk_size, size=n_neighbors)
        for i, neighbors in enumerate(neighbor_graphs)
    ]
    search_graphs = [
        build_search_graph(
            data_slices[i],
            n_graph[0],
            n_graph[1],
            metric=metric,
            pruning_degree_multiplier=pruning_degree_multiplier,
            random_state=random_state,
        )
        for i, n_graph in enumerate(neighbor_graphs)
    ]
    tmp_updates = [
        pair_updates(
            data_slices[i],
            trees[i],
            search_graphs[i],
            data_slices[j],
            trees[j],
            search_graphs[j],
            metric=metric,
            n_neighbors=n_neighbors,
            epsilon=epsilon,
            random_state=random_state,
        )
        for i, j in itertools.combinations(range(len(data_slices)), 2)
    ]
    for n, (i, j) in enumerate(itertools.combinations(range(len(data_slices)), 2)):
        current_heaps[i] = update_single_heap(
            current_heaps[i], tmp_updates[n][0], j * chunk_size
        )
        current_heaps[j] = update_single_heap(
            current_heaps[j], tmp_updates[n][1], i * chunk_size
        )

    sorted_heaps = [
        delayed(deheap_sort)(heap[0], heap[1]) for heap in current_heaps
    ]
    graph_indices = da.vstack(
        [
            da.from_delayed(
                heap[0],
                (dask_array.chunks[0][i], n_neighbors),
                dtype=np.int32,
            )
            for i, heap in enumerate(sorted_heaps)
        ]
    )
    graph_distances = da.vstack(
        [
            da.from_delayed(
                heap[0],
                (dask_array.chunks[0][i], n_neighbors),
                dtype=np.float32,
            )
            for i, heap in enumerate(sorted_heaps)
        ]
    )

    return graph_indices, graph_distances
