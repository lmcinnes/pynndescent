"""Parallel assembly of the search graph used by ``prepare()``.

Given the diversified k-NN graph (rows of neighbour indices and distances), the
search graph is the union of forward and reverse edges, without self loops,
degree-pruned, binarised, and finally relabelled by the search tree's leaf
order. This used to be a chain of single-threaded scipy operations (COO to CSR,
transpose, ``maximum``, ``setdiag``, fancy-index permutation); these kernels do
the same in parallel and produce the same graph:

* union values follow ``scipy.sparse`` ``maximum`` (an edge present in one
  direction only is compared with an implicit zero, and zero results are
  dropped);
* degree pruning follows ``degree_prune_internal`` (keep entries no larger than
  the ``max_degree + 1``-th smallest value of the row).
"""
import numpy as np
import numba


@numba.njit(inline="always")
def _insertion_sort_pairs(keys, values, start, end):
    for i in range(start + 1, end):
        key = keys[i]
        value = values[i]
        j = i - 1
        while j >= start and keys[j] > key:
            keys[j + 1] = keys[j]
            values[j + 1] = values[j]
            j -= 1
        keys[j + 1] = key
        values[j + 1] = value


@numba.njit(parallel=True, cache=True)
def _forward_csr(rows, dists):
    """CSR of the -1 padded neighbour rows, each row sorted by column."""
    n = rows.shape[0]
    counts = np.zeros(n + 1, dtype=np.int64)
    for i in numba.prange(n):
        c = 0
        for j in range(rows.shape[1]):
            if rows[i, j] >= 0:
                c += 1
        counts[i + 1] = c
    indptr = np.cumsum(counts)
    indices = np.empty(indptr[n], dtype=np.int32)
    data = np.empty(indptr[n], dtype=np.float32)
    for i in numba.prange(n):
        pos = indptr[i]
        for j in range(rows.shape[1]):
            if rows[i, j] >= 0:
                indices[pos] = rows[i, j]
                data[pos] = dists[i, j]
                pos += 1
        _insertion_sort_pairs(indices, data, indptr[i], indptr[i + 1])
    return indptr, indices, data


@numba.njit(parallel=True, cache=True)
def _transpose_csr(indptr, indices, data, n_threads):
    """Transpose without atomics: producers bucket edges by owning thread.

    Rows of the result list their columns in ascending order.
    """
    n = indptr.shape[0] - 1
    block = n // n_threads + 1
    bounds = np.zeros((n_threads, n_threads + 1), dtype=np.int64)
    for t in numba.prange(n_threads):
        for i in range(t * block, min(n, (t + 1) * block)):
            for k in range(indptr[i], indptr[i + 1]):
                bounds[t, indices[k] // block + 1] += 1
    total = 0
    for t in range(n_threads):
        for b in range(n_threads):
            count = bounds[t, b + 1]
            bounds[t, b] = total
            total += count
        bounds[t, n_threads] = total
    bucket_dst = np.empty(total, dtype=np.int32)
    bucket_src = np.empty(total, dtype=np.int32)
    bucket_val = np.empty(total, dtype=np.float32)
    for t in numba.prange(n_threads):
        cursor = bounds[t, :n_threads].copy()
        for i in range(t * block, min(n, (t + 1) * block)):
            for k in range(indptr[i], indptr[i + 1]):
                owner = indices[k] // block
                pos = cursor[owner]
                bucket_dst[pos] = indices[k]
                bucket_src[pos] = i
                bucket_val[pos] = data[k]
                cursor[owner] = pos + 1

    counts = np.zeros(n + 1, dtype=np.int64)
    for b in numba.prange(n_threads):
        for t in range(n_threads):
            for pos in range(bounds[t, b], bounds[t, b + 1]):
                counts[bucket_dst[pos] + 1] += 1
    t_indptr = np.cumsum(counts)
    t_indices = np.empty(total, dtype=np.int32)
    t_data = np.empty(total, dtype=np.float32)
    for b in numba.prange(n_threads):
        cursor = np.zeros(block, dtype=np.int64)
        for t in range(n_threads):  # producers in order, so sources ascend
            for pos in range(bounds[t, b], bounds[t, b + 1]):
                row = bucket_dst[pos]
                k = t_indptr[row] + cursor[row - b * block]
                t_indices[k] = bucket_src[pos]
                t_data[k] = bucket_val[pos]
                cursor[row - b * block] += 1
    return t_indptr, t_indices, t_data


@numba.njit(inline="always")
def _merge_row(
    row, f_indptr, f_indices, f_data, r_indptr, r_indices, r_data,
    out_indices, out_data,
):
    """Union of forward and reverse row with scipy maximum semantics, no diagonal."""
    a = f_indptr[row]
    a_end = f_indptr[row + 1]
    b = r_indptr[row]
    b_end = r_indptr[row + 1]
    count = 0
    while a < a_end or b < b_end:
        if b >= b_end or (a < a_end and f_indices[a] < r_indices[b]):
            col = f_indices[a]
            value = max(f_data[a], np.float32(0.0))
            a += 1
        elif a >= a_end or r_indices[b] < f_indices[a]:
            col = r_indices[b]
            value = max(r_data[b], np.float32(0.0))
            b += 1
        else:
            col = f_indices[a]
            value = max(f_data[a], r_data[b])
            a += 1
            b += 1
        if col != row and value != 0.0:
            out_indices[count] = col
            out_data[count] = value
            count += 1
    return count


@numba.njit(inline="always")
def _prune_row(out_data, count, max_degree):
    """degree_prune_internal: keep values <= the (max_degree+1)-th smallest."""
    if count <= max_degree:
        return np.inf
    return np.sort(out_data[:count])[max_degree]


@numba.njit(parallel=True, cache=True)
def _merge_and_prune(
    f_indptr, f_indices, f_data, r_indptr, r_indices, r_data, max_degree
):
    n = f_indptr.shape[0] - 1
    counts = np.zeros(n + 1, dtype=np.int64)
    for i in numba.prange(n):
        width = (f_indptr[i + 1] - f_indptr[i]) + (r_indptr[i + 1] - r_indptr[i])
        tmp_indices = np.empty(width, dtype=np.int32)
        tmp_data = np.empty(width, dtype=np.float32)
        count = _merge_row(
            i, f_indptr, f_indices, f_data, r_indptr, r_indices, r_data,
            tmp_indices, tmp_data,
        )
        cut = _prune_row(tmp_data, count, max_degree)
        kept = 0
        for j in range(count):
            if tmp_data[j] <= cut:
                kept += 1
        counts[i + 1] = kept
    indptr = np.cumsum(counts)
    indices = np.empty(indptr[n], dtype=np.int32)
    for i in numba.prange(n):
        width = (f_indptr[i + 1] - f_indptr[i]) + (r_indptr[i + 1] - r_indptr[i])
        tmp_indices = np.empty(width, dtype=np.int32)
        tmp_data = np.empty(width, dtype=np.float32)
        count = _merge_row(
            i, f_indptr, f_indices, f_data, r_indptr, r_indices, r_data,
            tmp_indices, tmp_data,
        )
        cut = _prune_row(tmp_data, count, max_degree)
        pos = indptr[i]
        for j in range(count):
            if tmp_data[j] <= cut:
                indices[pos] = tmp_indices[j]
                pos += 1
    return indptr, indices


@numba.njit(parallel=True, cache=True)
def _permute_csr(indptr, indices, order):
    """Rows and columns relabelled so that new vertex a is old vertex order[a]."""
    n = indptr.shape[0] - 1
    inverse = np.empty(n, dtype=np.int32)
    for a in numba.prange(n):
        inverse[order[a]] = a
    counts = np.zeros(n + 1, dtype=np.int64)
    for a in numba.prange(n):
        counts[a + 1] = indptr[order[a] + 1] - indptr[order[a]]
    new_indptr = np.cumsum(counts)
    new_indices = np.empty(indices.shape[0], dtype=np.int32)
    for a in numba.prange(n):
        old = order[a]
        start = new_indptr[a]
        length = indptr[old + 1] - indptr[old]
        for j in range(length):
            new_indices[start + j] = inverse[indices[indptr[old] + j]]
        # insertion sort (rows are short)
        for i in range(start + 1, start + length):
            key = new_indices[i]
            j = i - 1
            while j >= start and new_indices[j] > key:
                new_indices[j + 1] = new_indices[j]
                j -= 1
            new_indices[j + 1] = key
    return new_indptr, new_indices


@numba.njit(parallel=True, cache=True)
def permute_rows(data, order):
    """Parallel ``data[order]`` for a 2D C-contiguous array."""
    out = np.empty_like(data)
    for a in numba.prange(order.shape[0]):
        out[a] = data[order[a]]
    return out


def build_search_graph(neighbor_indices, neighbor_distances, max_degree):
    """Symmetrised, degree-pruned search graph from diversified k-NN rows.

    Returns ``(indptr, indices, min_distance)``; rows have sorted indices.
    ``min_distance`` is the smallest forward edge distance.
    """
    n_threads = numba.get_num_threads()
    f_indptr, f_indices, f_data = _forward_csr(neighbor_indices, neighbor_distances)
    min_distance = f_data.min() if f_data.shape[0] > 0 else np.float32(0.0)
    r_indptr, r_indices, r_data = _transpose_csr(f_indptr, f_indices, f_data, n_threads)
    indptr, indices = _merge_and_prune(
        f_indptr, f_indices, f_data, r_indptr, r_indices, r_data, max_degree
    )
    return indptr, indices, min_distance


def permute_graph(indptr, indices, order):
    """CSR relabelled so that new vertex ``a`` is old vertex ``order[a]``."""
    return _permute_csr(indptr, indices, np.ascontiguousarray(order, dtype=np.int64))
