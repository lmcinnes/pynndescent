import joblib
import math
import numba
import numpy as np

import pynndescent.sparse as sparse

from pynndescent.utils import heap_push, make_heap, seed

from pynndescent.threaded import (
    new_rng_state,
    per_thread_rng_state,
    parallel_calls,
    effective_n_jobs_with_context,
    chunk_rows,
    shuffle_jit,
    init_rp_tree_reduce_jit,
    new_build_candidates,
    nn_decent_reduce_jit,
    deheap_sort_map_jit,
)

# NNDescent algorithm

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


# Map Reduce functions to be jitted


@numba.njit(nogil=True)
def sparse_current_graph_map_jit(
    heap,
    rows,
    n_neighbors,
    inds,
    indptr,
    data,
    rng_state,
    seed_per_row,
    sparse_dist,
    dist_args,
):
    rng_state_local = rng_state.copy()
    for i in rows:
        if seed_per_row:
            seed(rng_state_local, i)
        if heap[0, i, 0] < 0.0:
            for j in range(n_neighbors - np.sum(heap[0, i] >= 0.0)):
                idx = np.abs(tau_rand_int(rng_state_local)) % data.shape[0]

                from_inds = inds[indptr[i] : indptr[i + 1]]
                from_data = data[indptr[i] : indptr[i + 1]]

                to_inds = inds[indptr[idx] : indptr[idx + 1]]
                to_data = data[indptr[idx] : indptr[idx + 1]]

                d = sparse_dist(from_inds, from_data, to_inds, to_data, *dist_args)

                heap_push(heap, i, d, idx, 1)

    return True


def sparse_init_random(
    current_graph,
    inds,
    indptr,
    data,
    dist,
    dist_args,
    n_neighbors,
    chunk_size,
    rng_state,
    parallel,
    seed_per_row=False,
):

    n_vertices = data.shape[0]
    n_tasks = int(math.ceil(float(n_vertices) / chunk_size))

    # store the updates in an array
    max_heap_update_count = chunk_size * n_neighbors * 2
    heap_updates = np.zeros((n_tasks, max_heap_update_count, 4), dtype=np.float32)
    heap_update_counts = np.zeros((n_tasks,), dtype=np.int64)
    rng_state_threads = per_thread_rng_state(n_tasks, rng_state)

    def current_graph_map(index):
        rows = chunk_rows(chunk_size, index, n_vertices)
        return (
            index,
            sparse_current_graph_map_jit(
                current_graph,
                rows,
                n_neighbors,
                inds,
                indptr,
                data,
                rng_state_threads[index],
                seed_per_row=seed_per_row,
                sparse_dist=dist,
                dist_args=dist_args,
            ),
        )

    # run map functions
    for index, status in parallel(parallel_calls(current_graph_map, n_tasks)):
        if status is False:
            raise ValueError("Failed in random initialization")

    return


@numba.njit(nogil=True, fastmath=True)
def sparse_init_rp_tree_map_jit(
    rows, leaf_array, inds, indptr, data, heap_updates, sparse_dist, dist_args
):
    count = 0
    for n in rows:
        if n >= leaf_array.shape[0]:
            break
        tried = set([(-1, -1)])
        for i in range(leaf_array.shape[1]):
            la_n_i = leaf_array[n, i]
            if la_n_i < 0:
                break
            for j in range(i + 1, leaf_array.shape[1]):
                la_n_j = leaf_array[n, j]
                if la_n_j < 0:
                    break
                if (la_n_i, la_n_j) in tried:
                    continue

                from_inds = inds[indptr[la_n_i] : indptr[la_n_i + 1]]
                from_data = data[indptr[la_n_i] : indptr[la_n_i + 1]]

                to_inds = inds[indptr[la_n_j] : indptr[la_n_j + 1]]
                to_data = data[indptr[la_n_j] : indptr[la_n_j + 1]]

                d = sparse_dist(from_inds, from_data, to_inds, to_data, *dist_args)

                hu = heap_updates[count]
                hu[0] = la_n_i
                hu[1] = d
                hu[2] = la_n_j
                hu[3] = 1
                count += 1
                hu = heap_updates[count]
                hu[0] = la_n_j
                hu[1] = d
                hu[2] = la_n_i
                hu[3] = 1
                count += 1
                tried.add((la_n_i, la_n_j))
                tried.add((la_n_j, la_n_i))

    return count


def sparse_init_rp_tree(
    inds, indptr, data, dist, dist_args, current_graph, leaf_array, chunk_size, parallel
):
    n_vertices = data.shape[0]
    n_tasks = int(math.ceil(float(n_vertices) / chunk_size))

    # store the updates in an array
    max_heap_update_count = chunk_size * leaf_array.shape[1] * leaf_array.shape[1] * 2
    heap_updates = np.zeros((n_tasks, max_heap_update_count, 4), dtype=np.float32)
    heap_update_counts = np.zeros((n_tasks,), dtype=np.int64)

    def init_rp_tree_map(index):
        rows = chunk_rows(chunk_size, index, n_vertices)
        return (
            index,
            sparse_init_rp_tree_map_jit(
                rows,
                leaf_array,
                inds,
                indptr,
                data,
                heap_updates[index],
                dist,
                dist_args,
            ),
        )

    def init_rp_tree_reduce(index):
        return init_rp_tree_reduce_jit(
            n_tasks, current_graph, heap_updates, offsets, index
        )

    # run map functions
    for index, count in parallel(parallel_calls(init_rp_tree_map, n_tasks)):
        heap_update_counts[index] = count

    # sort and chunk heap updates so they can be applied in the reduce
    max_count = heap_update_counts.max()
    offsets = np.zeros((n_tasks, max_count), dtype=np.int64)

    def shuffle(index):
        return shuffle_jit(
            heap_updates, heap_update_counts, offsets, chunk_size, n_vertices, index
        )

    parallel(parallel_calls(shuffle, n_tasks))

    # then run reduce functions
    parallel(parallel_calls(init_rp_tree_reduce, n_tasks))


@numba.njit(nogil=True, fastmath=True)
def sparse_nn_descent_map_jit(
    rows,
    max_candidates,
    inds,
    indptr,
    data,
    new_candidate_neighbors,
    old_candidate_neighbors,
    heap_updates,
    offset,
    sparse_dist,
    dist_args,
):
    count = 0
    for i in rows:
        i -= offset
        for j in range(max_candidates):
            p = int(new_candidate_neighbors[0, i, j])
            if p < 0:
                continue
            for k in range(j, max_candidates):
                q = int(new_candidate_neighbors[0, i, k])
                if q < 0:
                    continue

                from_inds = inds[indptr[p] : indptr[p + 1]]
                from_data = data[indptr[p] : indptr[p + 1]]

                to_inds = inds[indptr[q] : indptr[q + 1]]
                to_data = data[indptr[q] : indptr[q + 1]]

                d = sparse_dist(from_inds, from_data, to_inds, to_data, *dist_args)

                hu = heap_updates[count]
                hu[0] = p
                hu[1] = d
                hu[2] = q
                hu[3] = 1
                count += 1
                hu = heap_updates[count]
                hu[0] = q
                hu[1] = d
                hu[2] = p
                hu[3] = 1
                count += 1

            for k in range(max_candidates):
                q = int(old_candidate_neighbors[0, i, k])
                if q < 0:
                    continue

                from_inds = inds[indptr[p] : indptr[p + 1]]
                from_data = data[indptr[p] : indptr[p + 1]]

                to_inds = inds[indptr[q] : indptr[q + 1]]
                to_data = data[indptr[q] : indptr[q + 1]]

                d = sparse_dist(from_inds, from_data, to_inds, to_data, *dist_args)

                hu = heap_updates[count]
                hu[0] = p
                hu[1] = d
                hu[2] = q
                hu[3] = 1
                count += 1
                hu = heap_updates[count]
                hu[0] = q
                hu[1] = d
                hu[2] = p
                hu[3] = 1
                count += 1
    return count


def sparse_nn_descent(
    inds,
    indptr,
    data,
    n_vertices,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=sparse.sparse_euclidean,
    dist_args=(),
    n_iters=10,
    delta=0.001,
    rp_tree_init=False,
    leaf_array=None,
    verbose=False,
    n_jobs=None,
    seed_per_row=False,
):

    if rng_state is None:
        rng_state = new_rng_state()

    with joblib.Parallel(prefer="threads", n_jobs=n_jobs) as parallel:

        n_tasks = effective_n_jobs_with_context(n_jobs)
        chunk_size = int(math.ceil(n_vertices / n_tasks))

        current_graph = make_heap(n_vertices, n_neighbors)

        if rp_tree_init:
            sparse_init_rp_tree(
                inds,
                indptr,
                data,
                dist,
                dist_args,
                current_graph,
                leaf_array,
                chunk_size,
                parallel,
            )

        sparse_init_random(
            current_graph,
            inds,
            indptr,
            data,
            dist,
            dist_args,
            n_neighbors,
            chunk_size,
            rng_state,
            parallel,
            seed_per_row=seed_per_row,
        )

        # store the updates in an array
        # note that the factor here is `n_neighbors * n_neighbors`, not `max_candidates * max_candidates`
        # since no more than `n_neighbors` candidates are added for each row
        max_heap_update_count = chunk_size * n_neighbors * n_neighbors * 4
        heap_updates = np.zeros((n_tasks, max_heap_update_count, 4), dtype=np.float32)
        heap_update_counts = np.zeros((n_tasks,), dtype=np.int64)

        for n in range(n_iters):
            if verbose:
                print("\t", n, " / ", n_iters)

            (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
                current_graph,
                n_vertices,
                n_neighbors,
                max_candidates,
                chunk_size,
                rng_state,
                parallel,
                seed_per_row=seed_per_row,
            )

            def nn_descent_map(index):
                rows = chunk_rows(chunk_size, index, n_vertices)
                return (
                    index,
                    sparse_nn_descent_map_jit(
                        rows,
                        max_candidates,
                        inds,
                        indptr,
                        data,
                        new_candidate_neighbors,
                        old_candidate_neighbors,
                        heap_updates[index],
                        offset=0,
                        sparse_dist=dist,
                        dist_args=dist_args,
                    ),
                )

            def nn_decent_reduce(index):
                return nn_decent_reduce_jit(
                    n_tasks, current_graph, heap_updates, offsets, index
                )

            # run map functions
            for index, count in parallel(parallel_calls(nn_descent_map, n_tasks)):
                heap_update_counts[index] = count

            # sort and chunk heap updates so they can be applied in the reduce
            max_count = heap_update_counts.max()
            offsets = np.zeros((n_tasks, max_count), dtype=np.int64)

            def shuffle(index):
                return shuffle_jit(
                    heap_updates,
                    heap_update_counts,
                    offsets,
                    chunk_size,
                    n_vertices,
                    index,
                )

            parallel(parallel_calls(shuffle, n_tasks))

            # then run reduce functions
            c = 0
            for c_part in parallel(parallel_calls(nn_decent_reduce, n_tasks)):
                c += c_part

            if c <= delta * n_neighbors * data.shape[0]:
                break

        def deheap_sort_map(index):
            rows = chunk_rows(chunk_size, index, n_vertices)
            return index, deheap_sort_map_jit(rows, current_graph)

        parallel(parallel_calls(deheap_sort_map, n_tasks))
        return current_graph[0].astype(np.int64), current_graph[1]
