# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Enough simple sparse operations in numba to enable sparse UMAP
#
# License: BSD 3 clause
from __future__ import print_function
import numpy as np
import numba

from pynndescent.utils import (
    tau_rand_int,
    make_heap,
    new_build_candidates,
    deheap_sort,
    checked_flagged_heap_push,
    sparse_generate_graph_update_array,
    apply_graph_update_array,
    EMPTY_GRAPH,
)

from pynndescent.sparse import sparse_euclidean


@numba.njit(parallel=True, cache=False)
def generate_leaf_updates(
    updates,
    n_updates_per_thread,
    leaf_block,
    dist_thresholds,
    inds,
    indptr,
    data,
    dist,
    n_threads,
):
    """Generate leaf updates into pre-allocated arrays for parallel efficiency."""
    n_leaves = leaf_block.shape[0]
    leaves_per_thread = (n_leaves + n_threads - 1) // n_threads

    # Reset update counts
    for t in range(n_threads):
        n_updates_per_thread[t] = 0

    for t in numba.prange(n_threads):
        start_leaf = t * leaves_per_thread
        end_leaf = min(start_leaf + leaves_per_thread, n_leaves)
        max_updates = updates.shape[1]
        count = 0

        for leaf_idx in range(start_leaf, end_leaf):
            for i in range(leaf_block.shape[1]):
                p = leaf_block[leaf_idx, i]
                if p < 0:
                    break

                for j in range(i + 1, leaf_block.shape[1]):
                    q = leaf_block[leaf_idx, j]
                    if q < 0:
                        break

                    from_inds = inds[indptr[p] : indptr[p + 1]]
                    from_data = data[indptr[p] : indptr[p + 1]]

                    to_inds = inds[indptr[q] : indptr[q + 1]]
                    to_data = data[indptr[q] : indptr[q + 1]]
                    d = dist(from_inds, from_data, to_inds, to_data)

                    if d < dist_thresholds[p] or d < dist_thresholds[q]:
                        if count < max_updates:
                            updates[t, count, 0] = np.float32(p)
                            updates[t, count, 1] = np.float32(q)
                            updates[t, count, 2] = d
                            count += 1

        n_updates_per_thread[t] = count

    return updates


@numba.njit(
    locals={"d": numba.float32, "p": numba.int32, "q": numba.int32},
    cache=False,
    parallel=True,
)
def init_rp_tree(inds, indptr, data, dist, current_graph, leaf_array, n_threads=8):
    n_leaves = leaf_array.shape[0]
    block_size = n_threads * 64
    n_blocks = n_leaves // block_size

    max_leaf_size = leaf_array.shape[1]
    updates_per_thread = (
        int(block_size * max_leaf_size * (max_leaf_size - 1) / (2 * n_threads)) + 1
    )
    updates = np.zeros((n_threads, updates_per_thread, 3), dtype=np.float32)
    n_updates_per_thread = np.zeros(n_threads, dtype=np.int32)

    n_vertices = current_graph[0].shape[0]
    vertex_block_size = n_vertices // n_threads + 1

    for i in range(n_blocks + 1):
        block_start = i * block_size
        block_end = min(n_leaves, (i + 1) * block_size)

        leaf_block = leaf_array[block_start:block_end]
        dist_thresholds = current_graph[1][:, 0]

        generate_leaf_updates(
            updates,
            n_updates_per_thread,
            leaf_block,
            dist_thresholds,
            inds,
            indptr,
            data,
            dist,
            n_threads,
        )

        for t in numba.prange(n_threads):
            v_block_start = t * vertex_block_size
            v_block_end = min(v_block_start + vertex_block_size, n_vertices)

            for j in range(n_threads):
                for k in range(n_updates_per_thread[j]):
                    p = np.int32(updates[j, k, 0])
                    q = np.int32(updates[j, k, 1])
                    d = updates[j, k, 2]

                    if p >= v_block_start and p < v_block_end:
                        checked_flagged_heap_push(
                            current_graph[1][p],
                            current_graph[0][p],
                            current_graph[2][p],
                            d,
                            q,
                            np.uint8(1),
                        )
                    if q >= v_block_start and q < v_block_end:
                        checked_flagged_heap_push(
                            current_graph[1][q],
                            current_graph[0][q],
                            current_graph[2][q],
                            d,
                            p,
                            np.uint8(1),
                        )


@numba.njit(
    fastmath=True,
    locals={"d": numba.float32, "i": numba.int32, "idx": numba.int32},
    cache=False,
)
def init_random(n_neighbors, inds, indptr, data, heap, dist, rng_state):
    n_samples = indptr.shape[0] - 1
    for i in range(n_samples):
        if heap[0][i, 0] < 0.0:
            for j in range(n_neighbors - np.sum(heap[0][i] >= 0.0)):
                idx = np.abs(tau_rand_int(rng_state)) % n_samples

                from_inds = inds[indptr[idx] : indptr[idx + 1]]
                from_data = data[indptr[idx] : indptr[idx + 1]]

                to_inds = inds[indptr[i] : indptr[i + 1]]
                to_data = data[indptr[i] : indptr[i + 1]]
                d = dist(from_inds, from_data, to_inds, to_data)

                checked_flagged_heap_push(
                    heap[1][i], heap[0][i], heap[2][i], d, idx, np.uint8(1)
                )

    return


@numba.njit(cache=False)
def sparse_process_candidates(
    inds,
    indptr,
    data,
    dist,
    current_graph,
    new_candidate_neighbors,
    old_candidate_neighbors,
    n_blocks,
    block_size,
    n_threads,
    update_array,
    n_updates_per_thread,
):
    """Process candidate neighbors for sparse data using array-based updates."""
    c = 0
    n_vertices = new_candidate_neighbors.shape[0]
    for i in range(n_blocks + 1):
        block_start = i * block_size
        block_end = min(n_vertices, (i + 1) * block_size)

        new_candidate_block = new_candidate_neighbors[block_start:block_end]
        old_candidate_block = old_candidate_neighbors[block_start:block_end]

        dist_thresholds = current_graph[1][:, 0]

        sparse_generate_graph_update_array(
            update_array,
            n_updates_per_thread,
            new_candidate_block,
            old_candidate_block,
            dist_thresholds,
            inds,
            indptr,
            data,
            dist,
            n_threads,
        )

        c += apply_graph_update_array(
            current_graph, update_array, n_updates_per_thread, n_threads
        )

    return c


@numba.njit()
def nn_descent_internal(
    current_graph,
    inds,
    indptr,
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=sparse_euclidean,
    n_iters=10,
    delta=0.001,
    verbose=False,
):
    n_vertices = indptr.shape[0] - 1
    block_size = 16384
    n_blocks = n_vertices // block_size
    n_threads = numba.get_num_threads()

    # Pre-allocate update arrays
    max_updates_per_thread = (
        int(
            (max_candidates**2 + max_candidates * (max_candidates - 1) / 2)
            * block_size
            / n_threads
        )
        + 1024
    )
    update_array = np.empty((n_threads, max_updates_per_thread, 3), dtype=np.float32)
    n_updates_per_thread = np.zeros(n_threads, dtype=np.int32)

    for n in range(n_iters):
        if verbose:
            print("\t", n + 1, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph, max_candidates, rng_state, n_threads
        )

        c = sparse_process_candidates(
            inds,
            indptr,
            data,
            dist,
            current_graph,
            new_candidate_neighbors,
            old_candidate_neighbors,
            n_blocks,
            block_size,
            n_threads,
            update_array,
            n_updates_per_thread,
        )

        if c <= delta * n_neighbors * n_vertices:
            if verbose:
                print("\tStopping threshold met -- exiting after", n + 1, "iterations")
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
    n_iters=10,
    delta=0.001,
    init_graph=EMPTY_GRAPH,
    rp_tree_init=True,
    leaf_array=None,
    low_memory=False,
    verbose=False,
):

    n_samples = indptr.shape[0] - 1

    if init_graph[0].shape[0] == 1:  # EMPTY_GRAPH
        current_graph = make_heap(n_samples, n_neighbors)

        if rp_tree_init:
            init_rp_tree(inds, indptr, data, dist, current_graph, leaf_array)

        init_random(n_neighbors, inds, indptr, data, current_graph, dist, rng_state)
    elif init_graph[0].shape[0] == n_samples and init_graph[0].shape[1] == n_neighbors:
        current_graph = init_graph
    else:
        raise ValueError("Invalid initial graph specified!")

    # Note: low_memory parameter is kept for API compatibility but
    # now uses the efficient array-based implementation
    nn_descent_internal(
        current_graph,
        inds,
        indptr,
        data,
        n_neighbors,
        rng_state,
        max_candidates=max_candidates,
        dist=dist,
        n_iters=n_iters,
        delta=delta,
        verbose=verbose,
    )

    return deheap_sort(current_graph[0], current_graph[1])
