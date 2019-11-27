# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Enough simple sparse operations in numba to enable sparse UMAP
#
# License: BSD 3 clause
from __future__ import print_function
import locale
import numpy as np
import numba

import heapq

from pynndescent.utils import (
    tau_rand_int,
    make_heap,
    heap_push,
    new_build_candidates,
    deheap_sort,
    simple_heap_push,
    has_been_visited,
    mark_visited,
    apply_graph_updates_high_memory,
    apply_graph_updates_low_memory,
)

from pynndescent.sparse import sparse_euclidean
from pynndescent.rp_trees import search_sparse_flat_tree

locale.setlocale(locale.LC_NUMERIC, "C")


@numba.njit(
    fastmath=True,
    locals={
        "candidate": numba.types.int32,
        "d": numba.types.float32,
        "tried": numba.types.uint8[::1],
        #                     "tried": numba.types.Set(numba.types.int32),
        "indices": numba.types.int32[::1],
        "indptr": numba.types.int32[::1],
        "data": numba.types.float32[::1],
        "heap_size": numba.types.int16,
        "distance_scale": numba.types.float32,
        "seed_scale": numba.types.float32,
    },
)
def search_from_init(
    query_inds,
    query_data,
    inds,
    indptr,
    data,
    search_indptr,
    search_inds,
    heap_priorities,
    heap_indices,
    epsilon,
    tried,
    sparse_dist,
    dist_args,
):
    distance_scale = 1.0 + epsilon
    distance_bound = distance_scale * heap_priorities[0]
    heap_size = heap_priorities.shape[0]

    seed_set = [(heap_priorities[j], heap_indices[j]) for j in range(heap_size)]
    heapq.heapify(seed_set)

    # Find smallest seed point
    d_vertex, vertex = heapq.heappop(seed_set)

    while d_vertex < distance_bound:

        for j in range(search_indptr[vertex], search_indptr[vertex + 1]):

            candidate = search_inds[j]

            if has_been_visited(tried, candidate) == 0:
                mark_visited(tried, candidate)

                from_inds = inds[indptr[candidate] : indptr[candidate + 1]]
                from_data = data[indptr[candidate] : indptr[candidate + 1]]

                d = sparse_dist(
                    from_inds, from_data, query_inds, query_data, *dist_args
                )

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

    return heap_priorities, heap_indices


@numba.njit(
    fastmath=True,
    locals={
        "heap_priorities": numba.types.float32[::1],
        "heap_indices": numba.types.int32[::1],
        "indices": numba.types.int32[::1],
        "candidate": numba.types.int32,
        "current_query": numba.types.float32[::1],
        "d": numba.types.float32,
        "n_random_samples": numba.types.int32,
        "tried": numba.types.uint8[::1],
    },
)
def search_init(
    query_inds,
    query_data,
    k,
    inds,
    indptr,
    data,
    forest,
    n_neighbors,
    tried,
    sparse_dist,
    dist_args,
    rng_state,
):

    heap_priorities = np.float32(np.inf) + np.zeros(k, dtype=np.float32)
    heap_indices = np.int32(-1) + np.zeros(k, dtype=np.int32)
    n_samples = indptr.shape[0] - 1

    n_random_samples = min(k, n_neighbors)

    for tree in forest:
        indices = search_sparse_flat_tree(
            query_inds,
            query_data,
            tree.hyperplanes,
            tree.offsets,
            tree.children,
            tree.indices,
            rng_state,
        )

        n_initial_points = indices.shape[0]
        n_random_samples = min(k, n_neighbors) - n_initial_points

        for j in range(n_initial_points):
            candidate = indices[j]

            from_inds = inds[indptr[candidate] : indptr[candidate + 1]]
            from_data = data[indptr[candidate] : indptr[candidate + 1]]

            d = sparse_dist(from_inds, from_data, query_inds, query_data, *dist_args)

            # indices are guaranteed different
            simple_heap_push(heap_priorities, heap_indices, d, candidate)
            mark_visited(tried, candidate)

    if n_random_samples > 0:
        for i in range(n_random_samples):
            candidate = np.abs(tau_rand_int(rng_state)) % n_samples
            if has_been_visited(tried, candidate) == 0:
                from_inds = inds[indptr[candidate] : indptr[candidate + 1]]
                from_data = data[indptr[candidate] : indptr[candidate + 1]]

                d = sparse_dist(
                    from_inds, from_data, query_inds, query_data, *dist_args
                )

                simple_heap_push(heap_priorities, heap_indices, d, candidate)
                mark_visited(tried, candidate)

    return heap_priorities, heap_indices


@numba.njit()
def search(
    query_inds,
    query_indptr,
    query_data,
    k,
    inds,
    indptr,
    data,
    forest,
    search_indptr,
    search_indices,
    epsilon,
    n_neighbors,
    tried,
    sparse_dist,
    dist_args,
    rng_state,
):

    n_query_points = query_indptr.shape[0] - 1

    result = make_heap(n_query_points, k)
    for i in range(n_query_points):
        tried[:] = 0
        current_query_inds = query_inds[query_indptr[i] : query_indptr[i + 1]]
        current_query_data = query_data[query_indptr[i] : query_indptr[i + 1]]

        heap_priorities, heap_indices = search_init(
            current_query_inds,
            current_query_data,
            k,
            inds,
            indptr,
            data,
            forest,
            n_neighbors,
            tried,
            sparse_dist,
            dist_args,
            rng_state,
        )
        heap_priorities, heap_indices = search_from_init(
            current_query_inds,
            current_query_data,
            inds,
            indptr,
            data,
            search_indptr,
            search_indices,
            heap_priorities,
            heap_indices,
            epsilon,
            tried,
            sparse_dist,
            dist_args,
        )

        result[0, i] = heap_indices
        result[1, i] = heap_priorities

    return result


@numba.njit(parallel=True)
def generate_leaf_updates(
    leaf_block, dist_thresholds, inds, indptr, data, dist, dist_args
):

    updates = [[(-1, -1, np.inf)] for i in range(leaf_block.shape[0])]

    for n in numba.prange(leaf_block.shape[0]):
        for i in range(leaf_block.shape[1]):
            p = leaf_block[n, i]
            if p < 0:
                break

            for j in range(i + 1, leaf_block.shape[1]):
                q = leaf_block[n, j]
                if q < 0:
                    break

                from_inds = inds[indptr[p] : indptr[p + 1]]
                from_data = data[indptr[p] : indptr[p + 1]]

                to_inds = inds[indptr[q] : indptr[q + 1]]
                to_data = data[indptr[q] : indptr[q + 1]]
                d = dist(from_inds, from_data, to_inds, to_data, *dist_args)

                if d < dist_thresholds[p] or d < dist_thresholds[q]:
                    updates[n].append((p, q, d))

    return updates


@numba.njit()
def init_rp_tree(inds, indptr, data, dist, dist_args, current_graph, leaf_array):

    n_leaves = leaf_array.shape[0]
    block_size = 65536
    n_blocks = n_leaves // block_size

    for i in range(n_blocks + 1):
        block_start = i * block_size
        block_end = min(n_leaves, (i + 1) * block_size)

        leaf_block = leaf_array[block_start:block_end]
        dist_thresholds = current_graph[1, :, 0]

        updates = generate_leaf_updates(
            leaf_block, dist_thresholds, inds, indptr, data, dist, dist_args
        )

        for j in range(len(updates)):
            for k in range(len(updates[j])):
                p, q, d = updates[j][k]

                if p == -1 or q == -1:
                    continue

                heap_push(current_graph, p, d, q, 1)
                heap_push(current_graph, q, d, p, 1)


@numba.njit(fastmath=True)
def init_random(n_neighbors, inds, indptr, data, heap, dist, dist_args, rng_state):
    n_samples = indptr.shape[0] - 1
    for i in range(n_samples):
        if heap[0, i, 0] < 0.0:
            for j in range(n_neighbors - np.sum(heap[0, i] >= 0.0)):
                idx = np.abs(tau_rand_int(rng_state)) % n_samples

                from_inds = inds[indptr[idx] : indptr[idx + 1]]
                from_data = data[indptr[idx] : indptr[idx + 1]]

                to_inds = inds[indptr[i] : indptr[i + 1]]
                to_data = data[indptr[i] : indptr[i + 1]]
                d = dist(from_inds, from_data, to_inds, to_data, *dist_args)

                heap_push(heap, i, d, idx, 1)

    return


@numba.njit(parallel=True)
def generate_graph_updates(
    new_candidate_block,
    old_candidate_block,
    dist_thresholds,
    inds,
    indptr,
    data,
    dist,
    dist_args,
):

    block_size = new_candidate_block.shape[0]
    updates = [[(-1, -1, np.inf)] for i in range(block_size)]
    max_candidates = new_candidate_block.shape[1]

    for i in numba.prange(block_size):
        for j in range(max_candidates):
            p = int(new_candidate_block[i, j])
            if p < 0:
                continue
            for k in range(j, max_candidates):
                q = int(new_candidate_block[i, k])
                if q < 0:
                    continue

                from_inds = inds[indptr[p] : indptr[p + 1]]
                from_data = data[indptr[p] : indptr[p + 1]]

                to_inds = inds[indptr[q] : indptr[q + 1]]
                to_data = data[indptr[q] : indptr[q + 1]]
                d = dist(from_inds, from_data, to_inds, to_data, *dist_args)

                if d <= dist_thresholds[p] or d <= dist_thresholds[q]:
                    updates[i].append((p, q, d))

            for k in range(max_candidates):
                q = int(old_candidate_block[i, k])
                if q < 0:
                    continue

                from_inds = inds[indptr[p] : indptr[p + 1]]
                from_data = data[indptr[p] : indptr[p + 1]]

                to_inds = inds[indptr[q] : indptr[q + 1]]
                to_data = data[indptr[q] : indptr[q + 1]]
                d = dist(from_inds, from_data, to_inds, to_data, *dist_args)

                if d <= dist_thresholds[p] or d <= dist_thresholds[q]:
                    updates[i].append((p, q, d))

    return updates


@numba.njit()
def nn_descent_internal_low_memory_parallel(
    current_graph,
    inds,
    indptr,
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=sparse_euclidean,
    dist_args=(),
    n_iters=10,
    delta=0.001,
    verbose=False,
    seed_per_row=False,
):
    n_vertices = indptr.shape[0] - 1
    block_size = 16384
    n_blocks = n_vertices // block_size

    for n in range(n_iters):
        if verbose:
            print("\t", n, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph,
            n_vertices,
            n_neighbors,
            max_candidates,
            rng_state,
            seed_per_row,
        )

        c = 0
        for i in range(n_blocks + 1):
            block_start = i * block_size
            block_end = min(n_vertices, (i + 1) * block_size)

            new_candidate_block = new_candidate_neighbors[0, block_start:block_end]
            old_candidate_block = old_candidate_neighbors[0, block_start:block_end]
            dist_thresholds = current_graph[1, :, 0]

            updates = generate_graph_updates(
                new_candidate_block,
                old_candidate_block,
                dist_thresholds,
                inds,
                indptr,
                data,
                dist,
                dist_args,
            )

            c += apply_graph_updates_low_memory(current_graph, updates)

        if c <= delta * n_neighbors * n_vertices:
            return


@numba.njit()
def nn_descent_internal_high_memory_parallel(
    current_graph,
    inds,
    indptr,
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=sparse_euclidean,
    dist_args=(),
    n_iters=10,
    delta=0.001,
    verbose=False,
    seed_per_row=False,
):
    n_vertices = indptr.shape[0] - 1
    block_size = 16384
    n_blocks = n_vertices // block_size

    in_graph = [
        set(current_graph[0, i].astype(np.int64)) for i in range(current_graph.shape[1])
    ]

    for n in range(n_iters):
        if verbose:
            print("\t", n, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph,
            n_vertices,
            n_neighbors,
            max_candidates,
            rng_state,
            seed_per_row,
        )

        c = 0
        for i in range(n_blocks + 1):
            block_start = i * block_size
            block_end = min(n_vertices, (i + 1) * block_size)

            new_candidate_block = new_candidate_neighbors[0, block_start:block_end]
            old_candidate_block = old_candidate_neighbors[0, block_start:block_end]
            dist_thresholds = current_graph[1, :, 0]

            updates = generate_graph_updates(
                new_candidate_block,
                old_candidate_block,
                dist_thresholds,
                inds,
                indptr,
                data,
                dist,
                dist_args,
            )

            c += apply_graph_updates_high_memory(current_graph, updates, in_graph)

        if c <= delta * n_neighbors * n_vertices:
            return


@numba.njit()
def nn_descent(
    inds,
    indptr,
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=sparse_euclidean,
    dist_args=(),
    n_iters=10,
    delta=0.001,
    rp_tree_init=True,
    leaf_array=None,
    low_memory=False,
    verbose=False,
    seed_per_row=False,
):

    n_samples = indptr.shape[0] - 1
    current_graph = make_heap(n_samples, n_neighbors)

    if rp_tree_init:
        init_rp_tree(inds, indptr, data, dist, dist_args, current_graph, leaf_array)

    init_random(
        n_neighbors, inds, indptr, data, current_graph, dist, dist_args, rng_state
    )

    if low_memory:
        nn_descent_internal_low_memory_parallel(
            current_graph,
            inds,
            indptr,
            data,
            n_neighbors,
            rng_state,
            max_candidates=max_candidates,
            dist=dist,
            dist_args=dist_args,
            n_iters=n_iters,
            delta=delta,
            verbose=verbose,
            seed_per_row=seed_per_row,
        )
    else:
        nn_descent_internal_high_memory_parallel(
            current_graph,
            inds,
            indptr,
            data,
            n_neighbors,
            rng_state,
            max_candidates=max_candidates,
            dist=dist,
            dist_args=dist_args,
            n_iters=n_iters,
            delta=delta,
            verbose=verbose,
            seed_per_row=seed_per_row,
        )

    return deheap_sort(current_graph)
