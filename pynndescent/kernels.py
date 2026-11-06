"""Specialised distance kernels for index construction.

The generic construction code calls ``dist(x, y)`` one pair at a time. For the
two most common dense metric families this module provides a 1x4 kernel for
the NN-descent local join that compares one point against four candidates at
once, sharing the loads of the first point. It is written as a plain loop so
that numba/LLVM vectorises it along the feature dimension. Results agree with
the pairwise kernels up to floating point reassociation (differences at the
level of a few ULPs).

Kernel ids
----------
0 : generic -- use the pairwise ``dist`` function
1 : squared euclidean (``distances.squared_euclidean``)
2 : alternative dot (``distances.alternative_dot``), i.e. -log2(<x, y>)
"""
import numpy as np
import numba

from pynndescent.distances import squared_euclidean, alternative_dot
from pynndescent.utils import int32_bits_as_float32

FLOAT32_MAX = np.finfo(np.float32).max

KERNEL_GENERIC = 0
KERNEL_SQEUCLIDEAN = 1
KERNEL_ALT_DOT = 2


def select_kernel(dist, data):
    """Return the specialised kernel id for ``dist`` on ``data``, or 0."""
    if (
        not isinstance(data, np.ndarray)
        or data.ndim != 2
        or data.dtype != np.float32
        or not data.flags.c_contiguous
    ):
        return KERNEL_GENERIC
    if dist is squared_euclidean:
        return KERNEL_SQEUCLIDEAN
    if dist is alternative_dot:
        return KERNEL_ALT_DOT
    return KERNEL_GENERIC


@numba.njit(fastmath=True, inline="always")
def alt_dot_from_dot(r):
    if r <= 0.0:
        return np.float32(FLOAT32_MAX)
    return np.float32(-np.log2(np.float32(r)))


@numba.njit(fastmath=True, inline="always", boundscheck=False)
def _four_sqeuclidean(x, data, q0, q1, q2, q3):
    s0 = s1 = s2 = s3 = np.float32(0.0)
    y0 = data[q0]
    y1 = data[q1]
    y2 = data[q2]
    y3 = data[q3]
    for t in range(x.shape[0]):
        xv = x[t]
        e = xv - y0[t]
        s0 += e * e
        e = xv - y1[t]
        s1 += e * e
        e = xv - y2[t]
        s2 += e * e
        e = xv - y3[t]
        s3 += e * e
    return s0, s1, s2, s3


@numba.njit(fastmath=True, inline="always", boundscheck=False)
def _four_alt_dot(x, data, q0, q1, q2, q3):
    s0 = s1 = s2 = s3 = np.float32(0.0)
    y0 = data[q0]
    y1 = data[q1]
    y2 = data[q2]
    y3 = data[q3]
    for t in range(x.shape[0]):
        xv = x[t]
        s0 += xv * y0[t]
        s1 += xv * y1[t]
        s2 += xv * y2[t]
        s3 += xv * y3[t]
    return (
        alt_dot_from_dot(s0),
        alt_dot_from_dot(s1),
        alt_dot_from_dot(s2),
        alt_dot_from_dot(s3),
    )


@numba.njit(fastmath=True, inline="always")
def _emit(update_array, t, idx, p, q, d, dist_thresh_p, heap_distances):
    if d <= max(dist_thresh_p, heap_distances[q, 0]):
        update_array[t, idx, 0] = int32_bits_as_float32(p)
        update_array[t, idx, 1] = int32_bits_as_float32(q)
        update_array[t, idx, 2] = d
        return idx + 1
    return idx


@numba.njit(fastmath=True, inline="always")
def _join_row(
    update_array,
    t,
    idx,
    max_updates,
    p,
    data_p,
    dist_thresh_p,
    candidates,
    start,
    heap_distances,
    data,
    kernel,
):
    """Compare p with candidates[start:], four at a time where possible.

    Emits exactly the updates (and in the same order) that the pairwise loop in
    ``generate_graph_update_array`` would, including its capacity cut-off.
    """
    end = candidates.shape[0]
    k = start
    while k < end:
        if idx >= max_updates:
            break
        if (
            k + 4 <= end
            and idx + 4 <= max_updates
            and candidates[k] >= 0
            and candidates[k + 1] >= 0
            and candidates[k + 2] >= 0
            and candidates[k + 3] >= 0
            and candidates[k] != p
            and candidates[k + 1] != p
            and candidates[k + 2] != p
            and candidates[k + 3] != p
        ):
            q0 = candidates[k]
            q1 = candidates[k + 1]
            q2 = candidates[k + 2]
            q3 = candidates[k + 3]
            if kernel == 1:
                d0, d1, d2, d3 = _four_sqeuclidean(data_p, data, q0, q1, q2, q3)
            else:
                d0, d1, d2, d3 = _four_alt_dot(data_p, data, q0, q1, q2, q3)
            idx = _emit(update_array, t, idx, p, q0, d0, dist_thresh_p, heap_distances)
            idx = _emit(update_array, t, idx, p, q1, d1, dist_thresh_p, heap_distances)
            idx = _emit(update_array, t, idx, p, q2, d2, dist_thresh_p, heap_distances)
            idx = _emit(update_array, t, idx, p, q3, d3, dist_thresh_p, heap_distances)
            k += 4
        else:
            q = candidates[k]
            if q >= 0 and q != p:
                if kernel == 1:
                    d = squared_euclidean(data_p, data[q])
                else:
                    d = alternative_dot(data_p, data[q])
                idx = _emit(update_array, t, idx, p, q, d, dist_thresh_p, heap_distances)
            k += 1
    return idx


@numba.njit(
    parallel=True,
    fastmath=True,
    cache=False,
    locals={"p": numba.int32, "dist_thresh_p": numba.float32},
)
def generate_graph_update_array_batched(
    update_array,
    n_updates_per_thread,
    new_candidate_block,
    old_candidate_block,
    current_graph,
    data,
    kernel,
    n_threads,
):
    """Batched equivalent of ``utils.generate_graph_update_array`` for kernels 1 and 2."""
    heap_distances = current_graph[1]
    block_size = new_candidate_block.shape[0]
    max_new_candidates = new_candidate_block.shape[1]
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

                # new-new pairs start at j + 1 (and _join_row skips q == p) to
                # avoid self-loops, matching the pairwise implementation
                idx = _join_row(
                    update_array,
                    t,
                    idx,
                    max_updates,
                    p,
                    data_p,
                    dist_thresh_p,
                    new_candidate_block[i],
                    j + 1,
                    heap_distances,
                    data,
                    kernel,
                )
                idx = _join_row(
                    update_array,
                    t,
                    idx,
                    max_updates,
                    p,
                    data_p,
                    dist_thresh_p,
                    old_candidate_block[i],
                    0,
                    heap_distances,
                    data,
                    kernel,
                )

        n_updates_per_thread[t] = idx


@numba.njit(fastmath=True, inline="always")
def four_distances(query, data, candidates, out, kernel):
    """Distances from ``query`` to the four rows ``data[candidates]`` (kernels 1, 2)."""
    if kernel == 1:
        d0, d1, d2, d3 = _four_sqeuclidean(
            query, data, candidates[0], candidates[1], candidates[2], candidates[3]
        )
    else:
        d0, d1, d2, d3 = _four_alt_dot(
            query, data, candidates[0], candidates[1], candidates[2], candidates[3]
        )
    out[0] = d0
    out[1] = d1
    out[2] = d2
    out[3] = d3
