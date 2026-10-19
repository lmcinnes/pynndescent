# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 2 clause

import time

import numba
import numpy as np



# ---------------------------------------------------------------------------
# Graph update buffers are float32 arrays of (p, q, d) records. The vertex ids p
# and q are stored as the raw bits of an int32, not converted to float32 values,
# which could not represent ids above 2**24 exactly. These intrinsics
# reinterpret the bits (an LLVM bitcast, which costs nothing). Never perform
# arithmetic or comparisons on the float32 form of an id.
# ---------------------------------------------------------------------------
from numba import types as _nb_types
from numba.extending import intrinsic as _nb_intrinsic
from llvmlite import ir as _llvm_ir


@_nb_intrinsic
def int32_bits_as_float32(typingctx, x):
    """Store an integer vertex id in a float32 slot without changing its bits."""
    if not isinstance(x, _nb_types.Integer):
        return None

    def codegen(context, builder, sig, args):
        val = context.cast(builder, args[0], sig.args[0], _nb_types.int32)
        return builder.bitcast(val, _llvm_ir.FloatType())

    return _nb_types.float32(x), codegen


@_nb_intrinsic
def float32_bits_as_int32(typingctx, x):
    """Recover a vertex id stored with int32_bits_as_float32."""
    if not isinstance(x, _nb_types.Float) or x.bitwidth != 32:
        return None

    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], _llvm_ir.IntType(32))

    return _nb_types.int32(x), codegen


@numba.njit("void(i8[:], i8)", cache=True)
def seed(rng_state, seed):
    """Seed the random number generator with a given seed."""
    rng_state.fill(seed + 0xFFFF)


@numba.njit("i4(i8[:])", cache=True)
def tau_rand_int(state):
    """A fast (pseudo)-random number generator.

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random int32 value
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]


@numba.njit(inline="always", cache=True)
def _splitmix64(x, v):
    x = x ^ np.uint64(v)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@numba.njit(inline="always", cache=True)
def hash_uniform(seed, a, b, c):
    """A stateless uniform float in [0, 1) derived from (seed, a, b, c).

    Unlike tau_rand there is no shared state, so it can be used from parallel
    loops and gives the same result for any number of threads.
    """
    x = _splitmix64(np.uint64(seed), a)
    x = _splitmix64(x, b)
    x = _splitmix64(x, c)
    return np.float64(x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@numba.njit("f4(i8[:])", cache=True)
def tau_rand(state):
    """A fast (pseudo)-random number generator for floats in the range [0,1]

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random float32 in the interval [0, 1]
    """
    integer = tau_rand_int(state)
    return abs(float(integer) / 0x7FFFFFFF)


@numba.njit(
    [
        "f4(f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True)
        ),
    ],
    locals={
        "dim": numba.types.intp,
        "i": numba.types.uint32,
        # "result": numba.types.float32, # This provides speed, but causes errors in corner cases
    },
    fastmath=True,
    cache=True,
)
def norm(vec):
    """Compute the (standard l2) norm of a vector.

    Parameters
    ----------
    vec: array of shape (dim,)

    Returns
    -------
    The l2 norm of vec.
    """
    result = 0.0
    dim = vec.shape[0]
    for i in range(dim):
        result += vec[i] * vec[i]
    return np.sqrt(result)


@numba.njit(cache=True)
def rejection_sample(n_samples, pool_size, rng_state):
    """Generate n_samples many integers from 0 to pool_size such that no
    integer is selected twice. The duplication constraint is achieved via
    rejection sampling.

    Parameters
    ----------
    n_samples: int
        The number of random samples to select from the pool

    pool_size: int
        The size of the total pool of candidates to sample from

    rng_state: array of int64, shape (3,)
        Internal state of the random number generator

    Returns
    -------
    sample: array of shape(n_samples,)
        The ``n_samples`` randomly selected elements from the pool.
    """
    result = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        reject_sample = True
        j = 0
        while reject_sample:
            j = tau_rand_int(rng_state) % pool_size
            for k in range(i):
                if j == result[k]:
                    break
            else:
                reject_sample = False
        result[i] = j
    return result


@numba.njit(cache=True)
def make_heap(n_points, size):
    """Constructor for the numba enabled heap objects. The heaps are used
    for approximate nearest neighbor search, maintaining a list of potential
    neighbors sorted by their distance. We also flag if potential neighbors
    are newly added to the list or not. Internally this is stored as
    a single ndarray; the first axis determines whether we are looking at the
    array of candidate graph_indices, the array of distances, or the flag array for
    whether elements are new or not. Each of these arrays are of shape
    (``n_points``, ``size``)

    Parameters
    ----------
    n_points: int
        The number of graph_data points to track in the heap.

    size: int
        The number of items to keep on the heap for each graph_data point.

    Returns
    -------
    heap: An ndarray suitable for passing to other numba enabled heap functions.
    """
    indices = np.full((int(n_points), int(size)), -1, dtype=np.int32)
    distances = np.full((int(n_points), int(size)), np.inf, dtype=np.float32)
    flags = np.zeros((int(n_points), int(size)), dtype=np.uint8)
    result = (indices, distances, flags)

    return result


# Sentinel value for empty/uninitialized graphs
EMPTY_GRAPH = make_heap(1, 1)


@numba.njit(cache=True)
def siftdown(heap1, heap2, elt):
    """Restore the heap property for a heap with an out of place element
    at position ``elt``. This works with a heap pair where heap1 carries
    the weights and heap2 holds the corresponding elements."""
    while elt * 2 + 1 < heap1.shape[0]:
        left_child = elt * 2 + 1
        right_child = left_child + 1
        swap = elt

        if heap1[swap] < heap1[left_child]:
            swap = left_child

        if right_child < heap1.shape[0] and heap1[swap] < heap1[right_child]:
            swap = right_child

        if swap == elt:
            break
        else:
            heap1[elt], heap1[swap] = heap1[swap], heap1[elt]
            heap2[elt], heap2[swap] = heap2[swap], heap2[elt]
            elt = swap


@numba.njit(parallel=True, cache=False)
def deheap_sort(indices, distances):
    """Given two arrays representing a heap (indices and distances), reorder the
     arrays by increasing distance. This is effectively just the second half of
     heap sort (the first half not being required since we already have the
     graph_data in a heap).

     Note that this is done in-place.

    Parameters
    ----------
    indices : array of shape (n_samples, n_neighbors)
        The graph indices to sort by distance.
    distances : array of shape (n_samples, n_neighbors)
        The corresponding edge distance.

    Returns
    -------
    indices, distances: arrays of shape (n_samples, n_neighbors)
        The indices and distances sorted by increasing distance.
    """
    for i in numba.prange(indices.shape[0]):
        # starting from the end of the array and moving back
        for j in range(indices.shape[1] - 1, 0, -1):
            indices[i, 0], indices[i, j] = indices[i, j], indices[i, 0]
            distances[i, 0], distances[i, j] = distances[i, j], distances[i, 0]

            siftdown(distances[i, :j], indices[i, :j], 0)

    return indices, distances


@numba.njit(parallel=True, locals={"idx": numba.types.int64}, cache=False)
def new_build_candidates(current_graph, max_candidates, rng_state, n_threads):
    """Build a heap of candidate neighbors for nearest neighbor descent. For
    each vertex the candidate neighbors are any current neighbors, and any
    vertices that have the vertex as one of their nearest neighbors.

    Parameters
    ----------
    current_graph: heap
        The current state of the graph for nearest neighbor descent.

    max_candidates: int
        The maximum number of new candidate neighbors.

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    candidate_neighbors: A heap with an array of (randomly sorted) candidate
    neighbors for each vertex in the graph.
    """
    current_indices = current_graph[0]
    current_flags = current_graph[2]

    n_vertices = current_indices.shape[0]
    n_neighbors = current_indices.shape[1]

    new_candidate_indices = np.full((n_vertices, max_candidates), -1, dtype=np.int32)
    new_candidate_priority = np.full(
        (n_vertices, max_candidates), np.inf, dtype=np.float32
    )

    old_candidate_indices = np.full((n_vertices, max_candidates), -1, dtype=np.int32)
    old_candidate_priority = np.full(
        (n_vertices, max_candidates), np.inf, dtype=np.float32
    )

    block_size = n_vertices // n_threads + 1

    for n in numba.prange(n_threads):
        local_rng_state = rng_state + n
        block_start = n * block_size
        block_end = min(block_start + block_size, n_vertices)

        for i in range(n_vertices):
            for j in range(n_neighbors):
                idx = current_indices[i, j]

                if idx >= 0 and (
                    (i >= block_start and i < block_end)
                    or (idx >= block_start and idx < block_end)
                ):
                    isn = current_flags[i, j]
                    d = tau_rand(local_rng_state)

                    if isn:
                        if i >= block_start and i < block_end:
                            checked_heap_push(
                                new_candidate_priority[i],
                                new_candidate_indices[i],
                                d,
                                idx,
                            )
                        if idx >= block_start and idx < block_end:
                            checked_heap_push(
                                new_candidate_priority[idx],
                                new_candidate_indices[idx],
                                d,
                                i,
                            )
                    else:
                        if i >= block_start and i < block_end:
                            checked_heap_push(
                                old_candidate_priority[i],
                                old_candidate_indices[i],
                                d,
                                idx,
                            )
                        if idx >= block_start and idx < block_end:
                            checked_heap_push(
                                old_candidate_priority[idx],
                                old_candidate_indices[idx],
                                d,
                                i,
                            )

    indices = current_graph[0]
    flags = current_graph[2]

    for i in numba.prange(n_vertices):
        for j in range(n_neighbors):
            idx = indices[i, j]

            for k in range(max_candidates):
                if new_candidate_indices[i, k] == idx:
                    flags[i, j] = 0
                    break

    return new_candidate_indices, old_candidate_indices


@numba.njit("b1(u1[::1],i4)", cache=True)
def has_been_visited(table, candidate):
    loc = candidate >> 3
    mask = 1 << (candidate & 7)
    return table[loc] & mask


@numba.njit("void(u1[::1],i4)", cache=True)
def mark_visited(table, candidate):
    loc = candidate >> 3
    mask = 1 << (candidate & 7)
    table[loc] |= mask
    return


@numba.njit("b1(u1[::1],i4)", cache=True)
def check_and_mark_visited(table, candidate):
    """Check if candidate was visited and mark it as visited in one operation.

    Returns True if the candidate was already visited, False otherwise.
    More efficient than separate has_been_visited + mark_visited calls.
    """
    loc = candidate >> 3
    mask = numba.uint8(1 << (candidate & 7))
    was_visited = table[loc] & mask
    table[loc] |= mask
    return was_visited


@numba.njit(
    "i4(f4[::1],i4[::1],f4,i4)",
    fastmath=True,
    locals={
        "size": numba.types.intp,
        "i": numba.types.uint16,
        "ic1": numba.types.uint16,
        "ic2": numba.types.uint16,
        "i_swap": numba.types.uint16,
    },
    cache=True,
)
def simple_heap_push(priorities, indices, p, n):
    if p >= priorities[0]:
        return 0

    size = priorities.shape[0]

    # insert val at position zero
    priorities[0] = p
    indices[0] = n

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= size:
            break
        elif ic2 >= size:
            if priorities[ic1] > p:
                i_swap = ic1
            else:
                break
        elif priorities[ic1] >= priorities[ic2]:
            if p < priorities[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if p < priorities[ic2]:
                i_swap = ic2
            else:
                break

        priorities[i] = priorities[i_swap]
        indices[i] = indices[i_swap]

        i = i_swap

    priorities[i] = p
    indices[i] = n

    return 1


@numba.njit(
    "i4(f4[::1],i4[::1],f4,i4)",
    fastmath=True,
    locals={
        "size": numba.types.intp,
        "i": numba.types.uint16,
        "ic1": numba.types.uint16,
        "ic2": numba.types.uint16,
        "i_swap": numba.types.uint16,
    },
    cache=True,
)
def checked_heap_push(priorities, indices, p, n):
    if p >= priorities[0]:
        return 0

    size = priorities.shape[0]

    # break if we already have this element. For wider heaps this is written as a
    # branch-free reduction (no early exit) so that LLVM can vectorise the scan.
    if size >= 16:
        found = False
        for j in range(size):
            found |= indices[j] == n
        if found:
            return 0
    else:
        for j in range(size):
            if n == indices[j]:
                return 0

    # insert val at position zero
    priorities[0] = p
    indices[0] = n

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= size:
            break
        elif ic2 >= size:
            if priorities[ic1] > p:
                i_swap = ic1
            else:
                break
        elif priorities[ic1] >= priorities[ic2]:
            if p < priorities[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if p < priorities[ic2]:
                i_swap = ic2
            else:
                break

        priorities[i] = priorities[i_swap]
        indices[i] = indices[i_swap]

        i = i_swap

    priorities[i] = p
    indices[i] = n

    return 1


@numba.njit(
    "i4(f4[::1],i4[::1],u1[::1],f4,i4,u1)",
    fastmath=True,
    locals={
        "size": numba.types.intp,
        "i": numba.types.uint16,
        "ic1": numba.types.uint16,
        "ic2": numba.types.uint16,
        "i_swap": numba.types.uint16,
    },
    cache=True,
)
def checked_flagged_heap_push(priorities, indices, flags, p, n, f):
    if p >= priorities[0]:
        return 0

    size = priorities.shape[0]

    # break if we already have this element. For wider heaps this is written as a
    # branch-free reduction (no early exit) so that LLVM can vectorise the scan.
    if size >= 16:
        found = False
        for j in range(size):
            found |= indices[j] == n
        if found:
            return 0
    else:
        for j in range(size):
            if n == indices[j]:
                return 0

    # insert val at position zero
    priorities[0] = p
    indices[0] = n
    flags[0] = f

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= size:
            break
        elif ic2 >= size:
            if priorities[ic1] > p:
                i_swap = ic1
            else:
                break
        elif priorities[ic1] >= priorities[ic2]:
            if p < priorities[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if p < priorities[ic2]:
                i_swap = ic2
            else:
                break

        priorities[i] = priorities[i_swap]
        indices[i] = indices[i_swap]
        flags[i] = flags[i_swap]

        i = i_swap

    priorities[i] = p
    indices[i] = n
    flags[i] = f

    return 1


@numba.njit(inline="always")
def _push_candidate(priorities, indices, row, d, n):
    checked_heap_push(priorities[row], indices[row], d, n)


@numba.njit(
    parallel=True,
    cache=False,
    locals={"idx": numba.types.int64, "owner": numba.types.int64, "k": numba.types.int64},
)
def build_candidates_owner(
    current_graph,
    max_candidates,
    rng_state,
    n_threads,
    new_candidate_indices,
    new_candidate_priority,
    old_candidate_indices,
    old_candidate_priority,
    active,
    reverse_dst,
    reverse_src,
    bucket_bounds,
):
    """Owner-computes candidate construction with an active set.

    Each thread scans only the rows of its own vertex block. Forward edges and
    reverse edges whose endpoint is in the same block are pushed directly;
    other reverse edges are counting-sorted into buckets addressed to the
    owning thread, which drains them in a second parallel pass. Per-thread work
    is therefore O(n_vertices * n_neighbors / n_threads), rather than a scan of
    the whole graph per thread.

    New candidates are built first. Old candidates are then built only for
    vertices that received at least one new candidate, since the local join
    never reads the old list of a vertex with no new candidates. The candidate
    arrays are a workspace reused across calls; only rows written by the
    previous call (``active``) are reset. On the first call the workspace must
    be (-1, inf) and ``active`` all ones.

    ``reverse_dst`` and ``reverse_src`` need room for n_vertices * n_neighbors
    entries; ``bucket_bounds`` has shape (n_threads, n_threads + 1).
    """
    current_indices = current_graph[0]
    current_flags = current_graph[2]
    n_vertices = current_indices.shape[0]
    n_neighbors = current_indices.shape[1]
    block_size = n_vertices // n_threads + 1

    for i in numba.prange(n_vertices):
        if active[i]:
            for k in range(max_candidates):
                new_candidate_indices[i, k] = -1
                new_candidate_priority[i, k] = np.inf
                old_candidate_indices[i, k] = -1
                old_candidate_priority[i, k] = np.inf

    for phase in range(2):
        # phase 0 gathers new candidates (flag 1); phase 1 gathers old ones (flag 0)
        # for active vertices only.
        want_flag = 1 - phase
        if phase == 0:
            cand_idx = new_candidate_indices
            cand_pri = new_candidate_priority
        else:
            cand_idx = old_candidate_indices
            cand_pri = old_candidate_priority

        # 1. count cross-block reverse edges per destination block
        for t in numba.prange(n_threads):
            for b in range(n_threads + 1):
                bucket_bounds[t, b] = 0
            row_start = t * block_size
            row_end = min(row_start + block_size, n_vertices)
            for i in range(row_start, row_end):
                for j in range(n_neighbors):
                    idx = current_indices[i, j]
                    if idx >= 0 and current_flags[i, j] == want_flag:
                        owner = idx // block_size
                        if owner != t and (phase == 0 or active[idx]):
                            bucket_bounds[t, owner + 1] += 1

        # 2. prefix sum: bucket (t, b) occupies bucket_bounds[t, b]:bucket_bounds[t, b + 1]
        total = 0
        for t in range(n_threads):
            for b in range(n_threads):
                count = bucket_bounds[t, b + 1]
                bucket_bounds[t, b] = total
                total += count
            bucket_bounds[t, n_threads] = total

        # 3. scan own rows: push local edges, scatter cross-block reverse edges
        for t in numba.prange(n_threads):
            local_rng_state = rng_state + (phase * n_threads + t)
            cursor = bucket_bounds[t, :n_threads].copy()
            row_start = t * block_size
            row_end = min(row_start + block_size, n_vertices)
            for i in range(row_start, row_end):
                for j in range(n_neighbors):
                    idx = current_indices[i, j]
                    if idx < 0 or current_flags[i, j] != want_flag:
                        continue
                    if phase == 0 or active[i]:
                        d = tau_rand(local_rng_state)
                        _push_candidate(cand_pri, cand_idx, i, d, idx)
                    if phase == 0 or active[idx]:
                        owner = idx // block_size
                        if owner == t:
                            d = tau_rand(local_rng_state)
                            _push_candidate(cand_pri, cand_idx, idx, d, i)
                        else:
                            k = cursor[owner]
                            reverse_dst[k] = idx
                            reverse_src[k] = i
                            cursor[owner] = k + 1

        # 4. each owner drains the buckets addressed to it, in producer order
        for b in numba.prange(n_threads):
            local_rng_state = rng_state + ((2 + phase) * n_threads + b)
            for t in range(n_threads):
                end = bucket_bounds[t, b + 1]
                for k in range(bucket_bounds[t, b], end):
                    d = tau_rand(local_rng_state)
                    _push_candidate(cand_pri, cand_idx, reverse_dst[k], d, reverse_src[k])

        if phase == 0:
            for i in numba.prange(n_vertices):
                has_new = 0
                for k in range(max_candidates):
                    if new_candidate_indices[i, k] >= 0:
                        has_new = 1
                active[i] = has_new

    for i in numba.prange(n_vertices):
        if active[i]:
            for j in range(n_neighbors):
                idx = current_indices[i, j]
                for k in range(max_candidates):
                    if new_candidate_indices[i, k] == idx:
                        current_flags[i, j] = 0
                        break

    return new_candidate_indices, old_candidate_indices


@numba.njit(
    parallel=True,
    locals={
        "dist_thresh_p": numba.float32,
        "dist_thresh_q": numba.float32,
        "p": numba.int32,
        "q": numba.int32,
        "d": numba.float32,
        "max_updates": numba.int32,
        "max_threshold": numba.float32,
    },
    cache=False,
    fastmath=True,
)
def generate_graph_update_array(
    update_array,
    n_updates_per_thread,
    new_candidate_block,
    old_candidate_block,
    current_graph,
    data,
    dist,
    n_threads,
):
    """Generate graph updates into a pre-allocated array.

    This is more efficient than generating lists of tuples because:
    1. No dynamic memory allocation during the parallel loop
    2. Better cache locality with contiguous array storage
    3. Each thread writes to its own section of the array

    Parameters
    ----------
    update_array : ndarray of shape (n_threads, max_updates_per_thread, 3)
        Pre-allocated array to store updates. Each row stores (p, q, d).

    n_updates_per_thread : ndarray of shape (n_threads,)
        Output array to store the number of updates generated by each thread.

    new_candidate_block : ndarray of shape (block_size, max_candidates)
        New candidate indices for this block.

    old_candidate_block : ndarray of shape (block_size, max_candidates)
        Old candidate indices for this block.

    current_graph : tuple of (indices, distances, flags)
        The current graph heap. distances[:, 0] gives max heap distance per vertex.

    data : ndarray of shape (n_vertices, n_features)
        The data points.

    dist : callable
        Distance function.

    n_threads : int
        Number of threads to use.
    """
    heap_distances = current_graph[1]
    block_size = new_candidate_block.shape[0]
    max_new_candidates = new_candidate_block.shape[1]
    max_old_candidates = old_candidate_block.shape[1]
    rows_per_thread = (block_size // n_threads) + 1

    for t in numba.prange(n_threads):
        idx = 0
        max_updates = update_array.shape[1]

        for r in range(rows_per_thread):
            i = t * rows_per_thread + r
            if i >= block_size or idx >= max_updates:
                break

            for j in range(max_new_candidates):
                if idx >= max_updates:
                    break

                p = new_candidate_block[i, j]
                if p < 0:
                    continue

                data_p = data[p]
                dist_thresh_p = heap_distances[p, 0]

                # Compare with other new candidates (start at j+1 to avoid self-loops)
                for k in range(j + 1, max_new_candidates):
                    if idx >= max_updates:
                        break

                    q = new_candidate_block[i, k]
                    if q < 0:
                        continue

                    d = dist(data_p, data[q])

                    dist_thresh_q = heap_distances[q, 0]
                    max_threshold = max(dist_thresh_p, dist_thresh_q)

                    if d <= max_threshold:
                        update_array[t, idx, 0] = int32_bits_as_float32(p)
                        update_array[t, idx, 1] = int32_bits_as_float32(q)
                        update_array[t, idx, 2] = d
                        idx += 1

                # Compare with old candidates
                for k in range(max_old_candidates):
                    if idx >= max_updates:
                        break

                    q = old_candidate_block[i, k]
                    if q < 0 or q == p:
                        continue

                    d = dist(data_p, data[q])
                    dist_thresh_q = heap_distances[q, 0]
                    max_threshold = max(dist_thresh_p, dist_thresh_q)

                    if d <= max_threshold:
                        update_array[t, idx, 0] = int32_bits_as_float32(p)
                        update_array[t, idx, 1] = int32_bits_as_float32(q)
                        update_array[t, idx, 2] = d
                        idx += 1

        n_updates_per_thread[t] = idx


@numba.njit(
    parallel=True,
    cache=True,
    locals={
        "p": numba.int32,
        "q": numba.int32,
        "d": numba.float32,
        "added": numba.uint8,
        "n": numba.uint32,
        "t": numba.uint32,
        "j": numba.uint32,
    },
)
def apply_graph_update_array(
    current_graph, update_array, n_updates_per_thread, n_threads
):
    """Apply graph updates from a pre-allocated array.

    Uses block-based processing where each thread only updates vertices
    in its assigned block, avoiding the need for duplicate checking.

    Parameters
    ----------
    current_graph : tuple of (indices, distances, flags)
        The current nearest neighbor graph heap.

    update_array : ndarray of shape (n_threads, max_updates_per_thread, 3)
        Array of updates where each row is (p, q, d).

    n_updates_per_thread : ndarray of shape (n_threads,)
        Number of valid updates from each generating thread.

    n_threads : int
        Number of threads.

    Returns
    -------
    n_changes : int
        Total number of updates that modified the graph.
    """
    n_changes = 0
    priorities = current_graph[1]
    indices = current_graph[0]
    flags = current_graph[2]

    n_vertices = priorities.shape[0]
    vertex_block_size = n_vertices // n_threads + 1

    for n in numba.prange(n_threads):
        block_start = n * vertex_block_size
        block_end = min(block_start + vertex_block_size, n_vertices)

        # Each thread scans all updates but only applies those
        # where p or q falls in its block
        for t in range(n_threads):
            for j in range(n_updates_per_thread[t]):
                p = float32_bits_as_int32(update_array[t, j, 0])
                q = float32_bits_as_int32(update_array[t, j, 1])
                d = np.float32(update_array[t, j, 2])

                if p >= block_start and p < block_end:
                    added = checked_flagged_heap_push(
                        priorities[p], indices[p], flags[p], d, q, 1
                    )
                    n_changes += added

                if q >= block_start and q < block_end:
                    added = checked_flagged_heap_push(
                        priorities[q], indices[q], flags[q], d, p, 1
                    )
                    n_changes += added

    return n_changes


@numba.njit(
    parallel=True,
    cache=False,
    locals={
        "p": numba.int32,
        "q": numba.int32,
        "bp": numba.intp,
        "bq": numba.intp,
    },
)
def sort_updates_by_block(
    update_array, n_updates_per_thread, sorted_updates, block_offsets, n_vertices, n_threads
):
    """Counting-sort each thread's updates by destination vertex block.

    Each producer thread ``t`` sorts its own updates into ``sorted_updates[t]`` so
    that the updates destined for vertex block ``b`` occupy the half-open range
    ``block_offsets[t, b]:block_offsets[t, b + 1]``. An update whose endpoints
    lie in different blocks is written to both blocks. Producer order is kept
    within each range, so applying the ranges in thread order reproduces the
    exact push sequence of :func:`apply_graph_update_array`.

    ``sorted_updates`` must have room for twice the per-thread update capacity.
    """
    vertex_block_size = n_vertices // n_threads + 1

    for t in numba.prange(n_threads):
        positions = np.zeros(n_threads + 1, dtype=np.int64)
        n_updates = n_updates_per_thread[t]

        for j in range(n_updates):
            p = float32_bits_as_int32(update_array[t, j, 0])
            if p < 0:
                continue
            q = float32_bits_as_int32(update_array[t, j, 1])
            bp = p // vertex_block_size
            bq = q // vertex_block_size
            positions[bp + 1] += 1
            if bq != bp:
                positions[bq + 1] += 1

        for b in range(n_threads):
            positions[b + 1] += positions[b]
        for b in range(n_threads + 1):
            block_offsets[t, b] = positions[b]

        for j in range(n_updates):
            p = float32_bits_as_int32(update_array[t, j, 0])
            if p < 0:
                continue
            q = float32_bits_as_int32(update_array[t, j, 1])
            bp = p // vertex_block_size
            bq = q // vertex_block_size
            k = positions[bp]
            sorted_updates[t, k, 0] = update_array[t, j, 0]
            sorted_updates[t, k, 1] = update_array[t, j, 1]
            sorted_updates[t, k, 2] = update_array[t, j, 2]
            positions[bp] = k + 1
            if bq != bp:
                k = positions[bq]
                sorted_updates[t, k, 0] = update_array[t, j, 0]
                sorted_updates[t, k, 1] = update_array[t, j, 1]
                sorted_updates[t, k, 2] = update_array[t, j, 2]
                positions[bq] = k + 1


@numba.njit(
    parallel=True,
    cache=False,
    locals={
        "p": numba.int32,
        "q": numba.int32,
        "d": numba.float32,
        "added": numba.uint8,
        "n": numba.uint32,
        "t": numba.uint32,
        "j": numba.intp,
    },
)
def apply_sorted_graph_update_array(current_graph, sorted_updates, block_offsets, n_threads):
    """Apply updates bucketed by :func:`sort_updates_by_block`.

    Each thread reads only the updates destined for its own vertex block, rather
    than scanning every update produced by every thread.
    """
    n_changes = 0
    priorities = current_graph[1]
    indices = current_graph[0]
    flags = current_graph[2]

    n_vertices = priorities.shape[0]
    vertex_block_size = n_vertices // n_threads + 1

    for n in numba.prange(n_threads):
        block_start = n * vertex_block_size
        block_end = min(block_start + vertex_block_size, n_vertices)

        for t in range(n_threads):
            for j in range(block_offsets[t, n], block_offsets[t, n + 1]):
                p = float32_bits_as_int32(sorted_updates[t, j, 0])
                q = float32_bits_as_int32(sorted_updates[t, j, 1])
                d = np.float32(sorted_updates[t, j, 2])

                if p >= block_start and p < block_end:
                    added = checked_flagged_heap_push(
                        priorities[p], indices[p], flags[p], d, q, 1
                    )
                    n_changes += added

                if q >= block_start and q < block_end:
                    added = checked_flagged_heap_push(
                        priorities[q], indices[q], flags[q], d, p, 1
                    )
                    n_changes += added

    return n_changes


@numba.njit(
    parallel=True,
    cache=False,
    fastmath=True,
    locals={
        "dist_thresh_p": numba.float32,
        "dist_thresh_q": numba.float32,
        "p": numba.int32,
        "q": numba.int32,
        "d": numba.float32,
        "max_updates": numba.int32,
        "max_threshold": numba.float32,
    },
)
def sparse_generate_graph_update_array(
    update_array,
    n_updates_per_thread,
    new_candidate_block,
    old_candidate_block,
    current_graph,
    inds,
    indptr,
    data,
    dist,
    n_threads,
):
    """Generate graph updates for sparse data into a pre-allocated array."""
    heap_distances = current_graph[1]
    block_size = new_candidate_block.shape[0]
    max_new_candidates = new_candidate_block.shape[1]
    max_old_candidates = old_candidate_block.shape[1]
    rows_per_thread = (block_size // n_threads) + 1

    for t in numba.prange(n_threads):
        idx = 0
        max_updates = update_array.shape[1]

        for r in range(rows_per_thread):
            i = t * rows_per_thread + r
            if i >= block_size or idx >= max_updates:
                break

            for j in range(max_new_candidates):
                if idx >= max_updates:
                    break

                p = new_candidate_block[i, j]
                if p < 0:
                    continue

                from_inds = inds[indptr[p] : indptr[p + 1]]
                from_data = data[indptr[p] : indptr[p + 1]]
                dist_thresh_p = heap_distances[p, 0]

                # Compare with other new candidates (start at j+1 to avoid self-loops)
                for k in range(j + 1, max_new_candidates):
                    if idx >= max_updates:
                        break

                    q = new_candidate_block[i, k]
                    if q < 0:
                        continue

                    to_inds = inds[indptr[q] : indptr[q + 1]]
                    to_data = data[indptr[q] : indptr[q + 1]]
                    d = dist(from_inds, from_data, to_inds, to_data)

                    dist_thresh_q = heap_distances[q, 0]
                    max_threshold = max(dist_thresh_p, dist_thresh_q)

                    if d <= max_threshold:
                        update_array[t, idx, 0] = int32_bits_as_float32(p)
                        update_array[t, idx, 1] = int32_bits_as_float32(q)
                        update_array[t, idx, 2] = d
                        idx += 1

                # Compare with old candidates
                for k in range(max_old_candidates):
                    if idx >= max_updates:
                        break

                    q = old_candidate_block[i, k]
                    if q < 0 or q == p:
                        continue

                    to_inds = inds[indptr[q] : indptr[q + 1]]
                    to_data = data[indptr[q] : indptr[q + 1]]
                    d = dist(from_inds, from_data, to_inds, to_data)

                    dist_thresh_q = heap_distances[q, 0]
                    max_threshold = max(dist_thresh_p, dist_thresh_q)

                    if d <= max_threshold:
                        update_array[t, idx, 0] = int32_bits_as_float32(p)
                        update_array[t, idx, 1] = int32_bits_as_float32(q)
                        update_array[t, idx, 2] = d
                        idx += 1

        n_updates_per_thread[t] = idx


@numba.njit(cache=False)
def initalize_heap_from_graph_indices(heap, graph_indices, data, metric):

    for i in range(graph_indices.shape[0]):
        for idx in range(graph_indices.shape[1]):
            j = graph_indices[i, idx]
            if j >= 0 and j != i:
                d = metric(data[i], data[j])
                checked_flagged_heap_push(heap[1][i], heap[0][i], heap[2][i], d, j, 1)

    return heap


@numba.njit(cache=True)
def initalize_heap_from_graph_indices_and_distances(
    heap, graph_indices, graph_distances
):
    for i in range(graph_indices.shape[0]):
        for idx in range(graph_indices.shape[1]):
            j = graph_indices[i, idx]
            if j >= 0 and j != i:
                d = graph_distances[i, idx]
                checked_flagged_heap_push(heap[1][i], heap[0][i], heap[2][i], d, j, 1)

    return heap


@numba.njit(parallel=True, cache=False)
def sparse_initalize_heap_from_graph_indices(
    heap, graph_indices, data_indptr, data_indices, data_vals, metric
):

    for i in numba.prange(graph_indices.shape[0]):
        for idx in range(graph_indices.shape[1]):
            j = graph_indices[i, idx]
            if j < 0 or j == i:
                continue
            ind1 = data_indices[data_indptr[i] : data_indptr[i + 1]]
            data1 = data_vals[data_indptr[i] : data_indptr[i + 1]]
            ind2 = data_indices[data_indptr[j] : data_indptr[j + 1]]
            data2 = data_vals[data_indptr[j] : data_indptr[j + 1]]
            d = metric(ind1, data1, ind2, data2)
            checked_flagged_heap_push(heap[1][i], heap[0][i], heap[2][i], d, j, 1)

    return heap


# Generates a timestamp for use in logging messages when verbose=True
def ts():
    return time.ctime(time.time())


# ---------------------------------------------------------------------------
# Finishing a dense query in one compiled call: sort each result heap (exactly
# as deheap_sort does), map indices back to the original data order, and write
# corrected distances into new arrays. This replaces three separately
# dispatched steps (deheap, fancy indexing, a numba-vectorized or numpy
# correction) and an extra copy on the per-query path.
# ---------------------------------------------------------------------------


@numba.njit(inline="always")
def _deheap_rows_inplace(indices, distances):
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1] - 1, 0, -1):
            indices[i, 0], indices[i, j] = indices[i, j], indices[i, 0]
            distances[i, 0], distances[i, j] = distances[i, j], distances[i, 0]
            siftdown(distances[i, :j], indices[i, :j], 0)


@numba.njit(cache=True)
def finish_query_no_correction(indices, distances, vertex_order):
    """deheap_sort, then vertex_order[indices] and a float32 copy of distances."""
    _deheap_rows_inplace(indices, distances)
    n, k = indices.shape
    out_indices = np.empty((n, k), dtype=vertex_order.dtype)
    out_distances = np.empty((n, k), dtype=np.float32)
    for i in range(n):
        for j in range(k):
            out_indices[i, j] = vertex_order[indices[i, j]]
            out_distances[i, j] = distances[i, j]
    return out_indices, out_distances


@numba.njit(cache=True)
def finish_query_sqrt(indices, distances, vertex_order):
    """As finish_query_no_correction, with distances square-rooted (float32)."""
    _deheap_rows_inplace(indices, distances)
    n, k = indices.shape
    out_indices = np.empty((n, k), dtype=vertex_order.dtype)
    out_distances = np.empty((n, k), dtype=np.float32)
    for i in range(n):
        for j in range(k):
            out_indices[i, j] = vertex_order[indices[i, j]]
            out_distances[i, j] = np.sqrt(distances[i, j])
    return out_indices, out_distances


@numba.njit(cache=True, fastmath=True)
def finish_query_alternative_cosine(indices, distances, vertex_order):
    """As finish_query_no_correction, with distances mapped by 1 - 2**-d (float64,
    as correct_alternative_cosine returns)."""
    _deheap_rows_inplace(indices, distances)
    n, k = indices.shape
    out_indices = np.empty((n, k), dtype=vertex_order.dtype)
    out_distances = np.empty((n, k), dtype=np.float64)
    for i in range(n):
        for j in range(k):
            out_indices[i, j] = vertex_order[indices[i, j]]
            out_distances[i, j] = 1.0 - pow(2.0, -np.float64(distances[i, j]))
    return out_indices, out_distances


def l2_normalize_rows(X):
    """L2-normalise the rows of a 2-D float array (zero rows are left as zeros).

    Equivalent to sklearn.preprocessing.normalize(X, norm="l2") for dense input,
    without its per-call validation overhead.
    """
    norms = np.sqrt(np.einsum("ij,ij->i", X, X))
    norms[norms == 0.0] = 1.0
    return X / norms[:, np.newaxis]
