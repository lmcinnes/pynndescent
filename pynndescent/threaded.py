import joblib
import math
import numba
import numpy as np

import pynndescent.distances as dst

from pynndescent.utils import (
    heap_push,
    make_heap,
    rejection_sample,
    seed,
    siftdown,
    tau_rand,
    tau_rand_int,
)

# NNDescent algorithm

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def new_rng_state():
    return np.random.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)


def per_thread_rng_state(threads, rng_state):
    """Create an array of per-thread RNG states, seeded from an initial rng_state."""
    return rejection_sample(threads * 3, INT32_MAX, rng_state).reshape((threads, 3))


def parallel_calls(fn, n_tasks):
    return [(fn, [i], {}) for i in range(n_tasks)]


def effective_n_jobs_with_context(n_jobs=None):
    """Find the effective number of jobs, either specified directly, or from the joblib.parallel_backend context."""
    if n_jobs is None:
        _, n_jobs_from_context = joblib.parallel.get_active_backend()
        n_jobs = n_jobs_from_context
    return joblib.effective_n_jobs(n_jobs)


@numba.njit(nogil=True)
def chunk_rows(chunk_size, index, n_vertices):
    return np.arange(chunk_size * index, min(chunk_size * (index + 1), n_vertices))


@numba.njit(nogil=True)
def sort_heap_updates(heap_updates, num_heap_updates):
    """Take an array of unsorted heap updates and sort by row number."""
    row_numbers = heap_updates[:num_heap_updates, 0]
    # use mergesort since it is stable (and supported by numba)
    heap_updates[:num_heap_updates] = heap_updates[:num_heap_updates][
        row_numbers.argsort(kind="mergesort")
    ]
    return heap_updates


@numba.njit(nogil=True)
def chunk_heap_updates(heap_updates, num_heap_updates, n_vertices, chunk_size):
    """Return the offsets for each chunk of sorted heap updates."""
    chunk_boundaries = (
        np.arange(int(math.ceil(float(n_vertices) / chunk_size)) + 1) * chunk_size
    )
    offsets = np.searchsorted(
        heap_updates[:num_heap_updates, 0], chunk_boundaries, side="left"
    )
    return offsets


@numba.njit(nogil=True)
def shuffle_jit(
    heap_updates, heap_update_counts, offsets, chunk_size, n_vertices, index
):
    sorted_heap_updates = sort_heap_updates(
        heap_updates[index], heap_update_counts[index]
    )
    o = chunk_heap_updates(
        sorted_heap_updates, heap_update_counts[index], n_vertices, chunk_size
    )
    offsets[index, : o.shape[0]] = o


# Map Reduce functions to be jitted


@numba.njit(nogil=True)
def current_graph_map_jit(
    heap, rows, n_neighbors, data, rng_state, seed_per_row, dist,
):
    rng_state_local = rng_state.copy()
    for i in rows:
        if seed_per_row:
            seed(rng_state_local, i)
        if heap[0][i, 0] < 0.0:
            for j in range(n_neighbors - np.sum(heap[0][i] >= 0.0)):
                idx = np.abs(tau_rand_int(rng_state_local)) % data.shape[0]
                d = dist(data[i], data[idx])
                heap_push(heap, i, d, idx, 1)

    return True


@numba.njit(nogil=True)
def current_graph_reduce_jit(n_tasks, current_graph, heap_updates, offsets, index):
    for update_i in range(n_tasks):
        o = offsets[update_i]
        for j in range(o[index], o[index + 1]):
            heap_update = heap_updates[update_i, j]
            heap_push(
                current_graph,
                int(heap_update[0]),
                heap_update[1],
                int(heap_update[2]),
                int(heap_update[3]),
            )


def init_random(
    current_graph,
    data,
    dist,
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
    rng_state_threads = per_thread_rng_state(n_tasks, rng_state)

    def current_graph_map(index):
        rows = chunk_rows(chunk_size, index, n_vertices)
        return (
            index,
            current_graph_map_jit(
                current_graph,
                rows,
                n_neighbors,
                data,
                rng_state_threads[index],
                seed_per_row=seed_per_row,
                dist=dist,
            ),
        )

    # run map functions
    for index, status in parallel(parallel_calls(current_graph_map, n_tasks)):
        if status is False:
            raise ValueError("Failed in random initialization")

    return


@numba.njit(nogil=True, fastmath=True)
def init_rp_tree_map_jit(rows, leaf_array, data, heap_updates, dist):
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
                d = dist(data[la_n_i], data[la_n_j])
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


@numba.njit(nogil=True)
def init_rp_tree_reduce_jit(n_tasks, current_graph, heap_updates, offsets, index):
    for update_i in range(n_tasks):
        o = offsets[update_i]
        for j in range(o[index], o[index + 1]):
            heap_update = heap_updates[update_i, j]
            heap_push(
                current_graph,
                int(heap_update[0]),
                heap_update[1],
                int(heap_update[2]),
                int(heap_update[3]),
            )


def init_rp_tree(data, dist, current_graph, leaf_array, chunk_size, parallel):
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
            init_rp_tree_map_jit(rows, leaf_array, data, heap_updates[index], dist),
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


@numba.njit(nogil=True)
def candidates_map_jit(
    rows, n_neighbors, current_graph, heap_updates, offset, rng_state, seed_per_row
):
    rng_state_local = rng_state.copy()
    count = 0
    for i in rows:
        if seed_per_row:
            seed(rng_state_local, i)
        for j in range(n_neighbors):
            if current_graph[0][i - offset, j] < 0:
                continue
            idx = current_graph[0][i - offset, j]
            isn = current_graph[2][i - offset, j]
            d = tau_rand(rng_state_local)
            # if tau_rand(rng_state_local) < rho:
            # updates are common to old and new - decided by 'isn' flag
            hu = heap_updates[count]
            hu[0] = i
            hu[1] = d
            hu[2] = idx
            hu[3] = isn
            count += 1

            hu = heap_updates[count]
            hu[0] = idx
            hu[1] = d
            hu[2] = i
            hu[3] = isn
            count += 1
    return count


@numba.njit(nogil=True)
def candidates_reduce_jit(
    n_tasks,
    current_graph,
    new_candidate_neighbors,
    old_candidate_neighbors,
    heap_updates,
    offsets,
    index,
):
    for update_i in range(n_tasks):
        o = offsets[update_i]
        for j in range(o[index], o[index + 1]):
            heap_update = heap_updates[update_i, j]
            if heap_update[3] == 1:
                heap_push(
                    new_candidate_neighbors,
                    int(heap_update[0]),
                    heap_update[1],
                    int(heap_update[2]),
                    int(heap_update[3]),
                )
            else:
                heap_push(
                    old_candidate_neighbors,
                    int(heap_update[0]),
                    heap_update[1],
                    int(heap_update[2]),
                    int(heap_update[3]),
                )


@numba.njit(nogil=True)
def mark_candidate_results_map(
    rows, current_graph, n_neighbors, max_candidates, new_candidate_neighbors
):
    for i in rows:
        for j in range(n_neighbors):
            idx = current_graph[0][i, j]

            for k in range(max_candidates):
                if new_candidate_neighbors[0][i, k] == idx:
                    current_graph[2][i, j] = 0
                    break

    return


def new_build_candidates(
    current_graph,
    n_vertices,
    n_neighbors,
    max_candidates,
    chunk_size,
    rng_state,
    parallel,
    seed_per_row=False,
):

    n_tasks = int(math.ceil(float(n_vertices) / chunk_size))

    new_candidate_neighbors = make_heap(n_vertices, max_candidates)
    old_candidate_neighbors = make_heap(n_vertices, max_candidates)

    # store the updates in an array
    max_heap_update_count = chunk_size * n_neighbors * 2
    heap_updates = np.zeros((n_tasks, max_heap_update_count, 4), dtype=np.float32)
    heap_update_counts = np.zeros((n_tasks,), dtype=np.int64)
    rng_state_threads = per_thread_rng_state(n_tasks, rng_state)

    def candidates_map(index):
        rows = chunk_rows(chunk_size, index, n_vertices)
        return (
            index,
            candidates_map_jit(
                rows,
                n_neighbors,
                current_graph,
                heap_updates[index],
                offset=0,
                rng_state=rng_state_threads[index],
                seed_per_row=seed_per_row,
            ),
        )

    def candidates_reduce(index):
        return candidates_reduce_jit(
            n_tasks,
            current_graph,
            new_candidate_neighbors,
            old_candidate_neighbors,
            heap_updates,
            offsets,
            index,
        )

    # run map functions
    for index, count in parallel(parallel_calls(candidates_map, n_tasks)):
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
    parallel(parallel_calls(candidates_reduce, n_tasks))

    def mark_candidate_results(index):
        rows = chunk_rows(chunk_size, index, n_vertices)
        return mark_candidate_results_map(
            rows, current_graph, n_neighbors, max_candidates, new_candidate_neighbors
        )

    # Now mark whether things were used correctly
    parallel(parallel_calls(mark_candidate_results, n_tasks))

    return new_candidate_neighbors, old_candidate_neighbors


@numba.njit(nogil=True, fastmath=True)
def nn_descent_map_jit(
    rows,
    max_candidates,
    data,
    new_candidate_neighbors,
    old_candidate_neighbors,
    heap_updates,
    offset,
    dist,
):
    count = 0
    for i in rows:
        i -= offset
        for j in range(max_candidates):
            p = int(new_candidate_neighbors[0][i, j])
            if p < 0:
                continue
            for k in range(j, max_candidates):
                q = int(new_candidate_neighbors[0][i, k])
                if q < 0:
                    continue

                d = dist(data[p], data[q])
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
                q = int(old_candidate_neighbors[0][i, k])
                if q < 0:
                    continue

                d = dist(data[p], data[q])
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


@numba.njit(nogil=True)
def nn_decent_reduce_jit(n_tasks, current_graph, heap_updates, offsets, index):
    c = 0
    for update_i in range(n_tasks):
        o = offsets[update_i]
        for j in range(o[index], o[index + 1]):
            heap_update = heap_updates[update_i, j]
            c += heap_push(
                current_graph,
                heap_update[0],
                heap_update[1],
                heap_update[2],
                heap_update[3],
            )
    return c


@numba.njit(nogil=True)
def deheap_sort_map_jit(rows, heap):
    indices = heap[0]
    weights = heap[1]

    for i in rows:

        ind_heap = indices[i]
        dist_heap = weights[i]

        for j in range(ind_heap.shape[0] - 1):
            ind_heap[0], ind_heap[ind_heap.shape[0] - j - 1] = (
                ind_heap[ind_heap.shape[0] - j - 1],
                ind_heap[0],
            )
            dist_heap[0], dist_heap[dist_heap.shape[0] - j - 1] = (
                dist_heap[dist_heap.shape[0] - j - 1],
                dist_heap[0],
            )

            siftdown(
                dist_heap[: dist_heap.shape[0] - j - 1],
                ind_heap[: ind_heap.shape[0] - j - 1],
                0,
            )


def nn_descent(
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=dst.euclidean,
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

        n_vertices = data.shape[0]
        n_tasks = effective_n_jobs_with_context(n_jobs)
        chunk_size = int(math.ceil(n_vertices / n_tasks))

        current_graph = make_heap(data.shape[0], n_neighbors)

        if rp_tree_init:
            init_rp_tree(data, dist, current_graph, leaf_array, chunk_size, parallel)

        init_random(
            current_graph,
            data,
            dist,
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
                    nn_descent_map_jit(
                        rows,
                        max_candidates,
                        data,
                        new_candidate_neighbors,
                        old_candidate_neighbors,
                        heap_updates[index],
                        offset=0,
                        dist=dist,
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
