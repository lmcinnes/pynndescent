"""Graph search over scalar-quantized vectors with exact reranking.

``NNDescent(..., quantization="sq8")`` (or ``"sq4"``) stores, next to the float32
data, one code per feature: 8 bits, or 4 bits packed two to a byte. Each feature
gets its own affine map ``x ~ offset + scale * code`` fitted to the data's range.

A query is searched over the codes with an asymmetric distance (the query stays
in float32 and is transformed once, so no per-candidate decoding table is
needed), keeping the best ``candidate_width`` candidates; the best
``rerank_width`` of those are then scored with the exact float32 distance. The
whole query runs in one compiled call, compiled once and cached on disk, like
the float32 search in ``search.py``.

Supported exact distances are squared euclidean (euclidean metrics) and
alternative dot (dot, and cosine on normalized dense data). For squared
euclidean the code distance is ``sum_i scale_i**2 * (z_i - c_i)**2`` with
``z = (q - offset) / scale``; for dot it is ``1 - (sum_i q_i offset_i +
sum_i q_i scale_i c_i)``, which approximates ``1 - <q, x>``.
"""
import functools
import heapq

import numba
import numpy as np
from numba import types
from numba.experimental import structref

from pynndescent.distances import alternative_dot, squared_euclidean
from pynndescent.kernels import KERNEL_ALT_DOT, KERNEL_SQEUCLIDEAN, four_distances_at
from pynndescent.search import (
    _HEAP_TYPE,
    _array,
    _distance_bound,
    _prepare_queries,
    _readonly,
    _tree_leaf,
    _tree_spill_leaves,
    default_prefetch_lines,
    stack_search_forest,
)
from pynndescent.utils import check_and_mark_visited, prefetch_row, simple_heap_push, siftdown

SQ_MODES = ("sq8", "sq4")


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


@numba.njit(parallel=True, cache=True)
def _column_ranges(data):
    n, dim = data.shape
    n_chunks = min(n, 256)
    chunk = (n + n_chunks - 1) // n_chunks
    mins = np.full((n_chunks, dim), np.inf, dtype=np.float32)
    maxs = np.full((n_chunks, dim), -np.inf, dtype=np.float32)
    for c in numba.prange(n_chunks):
        for i in range(c * chunk, min(n, (c + 1) * chunk)):
            for j in range(dim):
                v = data[i, j]
                if v < mins[c, j]:
                    mins[c, j] = v
                if v > maxs[c, j]:
                    maxs[c, j] = v
    lo = np.empty(dim, dtype=np.float32)
    hi = np.empty(dim, dtype=np.float32)
    for j in range(dim):
        lo[j] = mins[:, j].min()
        hi[j] = maxs[:, j].max()
    return lo, hi


@numba.njit(parallel=True, cache=True)
def _encode_sq8(data, offsets, inv_scales, codes):
    for i in numba.prange(data.shape[0]):
        for j in range(data.shape[1]):
            v = np.rint((data[i, j] - offsets[j]) * inv_scales[j])
            codes[i, j] = np.uint8(min(255.0, max(0.0, v)))


@numba.njit(parallel=True, cache=True)
def _encode_sq4(data, offsets, inv_scales, codes):
    dim = data.shape[1]
    for i in numba.prange(data.shape[0]):
        for t in range(codes.shape[1]):
            j = 2 * t
            lo = np.rint((data[i, j] - offsets[j]) * inv_scales[j])
            lo = min(15.0, max(0.0, lo))
            if j + 1 < dim:
                hi = np.rint((data[i, j + 1] - offsets[j + 1]) * inv_scales[j + 1])
                hi = min(15.0, max(0.0, hi))
            else:
                hi = 0.0
            codes[i, t] = np.uint8(np.uint8(lo) | (np.uint8(hi) << 4))


def fit_scalar_quantizer(data, bits, sample_size=100_000, clip_quantile=None, random_state=None):
    """Per-feature (offset, scale) for ``bits``-bit codes.

    With ``clip_quantile`` set (e.g. 1e-4) the range is taken from those
    quantiles of a sample instead of the full min/max, so rare outliers do not
    stretch the code range.
    """
    levels = (1 << bits) - 1
    if clip_quantile:
        rng = np.random.default_rng(random_state)
        n = data.shape[0]
        rows = rng.choice(n, min(n, sample_size), replace=False) if n > sample_size else slice(None)
        sample = np.asarray(data[rows], dtype=np.float32)
        lo = np.quantile(sample, clip_quantile, axis=0).astype(np.float32)
        hi = np.quantile(sample, 1.0 - clip_quantile, axis=0).astype(np.float32)
    else:
        lo, hi = _column_ranges(data)
    scales = (hi - lo) / np.float32(levels)
    scales = np.where(scales > 0, scales, np.float32(1.0)).astype(np.float32)
    return lo.astype(np.float32), scales


def encode(data, offsets, scales, bits):
    inv_scales = (np.float32(1.0) / scales).astype(np.float32)
    if bits == 8:
        codes = np.empty(data.shape, dtype=np.uint8)
        _encode_sq8(data, offsets, inv_scales, codes)
    else:
        codes = np.empty((data.shape[0], (data.shape[1] + 1) // 2), dtype=np.uint8)
        _encode_sq4(data, offsets, inv_scales, codes)
    return codes


# --------------------------------------------------------------------------
# Search index (one pointer for all arrays, as in search.py)
# --------------------------------------------------------------------------


@structref.register
class QuantizedSearchIndexType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class QuantizedSearchIndex(structref.StructRefProxy):
    pass


_FIELDS = [
    "codes",
    "data",
    "q_offsets",
    "q_scales",
    "indptr",
    "indices",
    "tree_hyperplanes",
    "tree_offsets",
    "tree_children",
    "tree_indices",
    "n_search_trees",
    "tree_leaf_budget",
    "n_neighbors",
    "min_distance",
    "normalize_query",
    "tree_normalize_query",
    "max_degree",
    "prefetch_lines",
    "rerank_prefetch_lines",
]
structref.define_proxy(QuantizedSearchIndex, QuantizedSearchIndexType, _FIELDS)


@numba.njit(cache=True)
def _new_quantized_search_index(
    codes, data, q_offsets, q_scales, indptr, indices, tree_hyperplanes, tree_offsets,
    tree_children, tree_indices, n_search_trees, tree_leaf_budget, n_neighbors,
    min_distance, normalize_query, tree_normalize_query, max_degree, prefetch_lines,
    rerank_prefetch_lines,
):
    return QuantizedSearchIndex(
        codes, data, q_offsets, q_scales, indptr, indices, tree_hyperplanes, tree_offsets,
        tree_children, tree_indices, n_search_trees, tree_leaf_budget, n_neighbors,
        min_distance, normalize_query, tree_normalize_query, max_degree, prefetch_lines,
        rerank_prefetch_lines,
    )


# --------------------------------------------------------------------------
# Code distances. ``a`` and ``b`` hold the transformed query:
#   sq8 euclidean: a = z, b = w;  sq8 dot: a = u (b unused), ``base`` = <q, offset>,
#   all as (1, dim) arrays; sq4: (2, n_bytes) arrays for the low/high nibbles
# --------------------------------------------------------------------------


@numba.njit(inline="always", fastmath=True, boundscheck=False)
def _one_sq8_l2(z, w, c):
    s = np.float32(0.0)
    z0 = z[0]
    w0 = w[0]
    for t in range(z0.shape[0]):
        e = z0[t] - np.float32(c[t])
        s += w0[t] * e * e
    return s


@numba.njit(inline="always", fastmath=True, boundscheck=False)
def _four_sq8_l2(z, w, c0, c1, c2, c3):
    s0 = s1 = s2 = s3 = np.float32(0.0)
    z0 = z[0]
    w0 = w[0]
    for t in range(z0.shape[0]):
        zt = z0[t]
        wt = w0[t]
        e = zt - np.float32(c0[t])
        s0 += wt * e * e
        e = zt - np.float32(c1[t])
        s1 += wt * e * e
        e = zt - np.float32(c2[t])
        s2 += wt * e * e
        e = zt - np.float32(c3[t])
        s3 += wt * e * e
    return s0, s1, s2, s3


@numba.njit(inline="always", fastmath=True, boundscheck=False)
def _one_sq8_dot(u, base, c):
    s = np.float32(0.0)
    u0 = u[0]
    for t in range(u0.shape[0]):
        s += u0[t] * np.float32(c[t])
    return np.float32(1.0) - (base + s)


@numba.njit(inline="always", fastmath=True, boundscheck=False)
def _four_sq8_dot(u, base, c0, c1, c2, c3):
    s0 = s1 = s2 = s3 = np.float32(0.0)
    u0 = u[0]
    for t in range(u0.shape[0]):
        ut = u0[t]
        s0 += ut * np.float32(c0[t])
        s1 += ut * np.float32(c1[t])
        s2 += ut * np.float32(c2[t])
        s3 += ut * np.float32(c3[t])
    one = np.float32(1.0)
    return one - (base + s0), one - (base + s1), one - (base + s2), one - (base + s3)


@numba.njit(inline="always", fastmath=True, boundscheck=False)
def _one_sq4_l2(z, w, c):
    s = np.float32(0.0)
    z_lo = z[0]
    z_hi = z[1]
    w_lo = w[0]
    w_hi = w[1]
    for t in range(z_lo.shape[0]):
        byte = c[t]
        e = z_lo[t] - np.float32(byte & 15)
        s += w_lo[t] * e * e
        e = z_hi[t] - np.float32(byte >> 4)
        s += w_hi[t] * e * e
    return s


@numba.njit(inline="always", fastmath=True, boundscheck=False)
def _four_sq4_l2(z, w, c0, c1, c2, c3):
    s0 = s1 = s2 = s3 = np.float32(0.0)
    z_lo = z[0]
    z_hi = z[1]
    w_lo = w[0]
    w_hi = w[1]
    for t in range(z_lo.shape[0]):
        zl = z_lo[t]
        zh = z_hi[t]
        wl = w_lo[t]
        wh = w_hi[t]
        b = c0[t]
        e = zl - np.float32(b & 15)
        f = zh - np.float32(b >> 4)
        s0 += wl * e * e + wh * f * f
        b = c1[t]
        e = zl - np.float32(b & 15)
        f = zh - np.float32(b >> 4)
        s1 += wl * e * e + wh * f * f
        b = c2[t]
        e = zl - np.float32(b & 15)
        f = zh - np.float32(b >> 4)
        s2 += wl * e * e + wh * f * f
        b = c3[t]
        e = zl - np.float32(b & 15)
        f = zh - np.float32(b >> 4)
        s3 += wl * e * e + wh * f * f
    return s0, s1, s2, s3


@numba.njit(inline="always", fastmath=True, boundscheck=False)
def _one_sq4_dot(u, base, c):
    s = np.float32(0.0)
    u_lo = u[0]
    u_hi = u[1]
    for t in range(u_lo.shape[0]):
        byte = c[t]
        s += u_lo[t] * np.float32(byte & 15) + u_hi[t] * np.float32(byte >> 4)
    return np.float32(1.0) - (base + s)


@numba.njit(inline="always", fastmath=True, boundscheck=False)
def _four_sq4_dot(u, base, c0, c1, c2, c3):
    s0 = s1 = s2 = s3 = np.float32(0.0)
    u_lo = u[0]
    u_hi = u[1]
    for t in range(u_lo.shape[0]):
        ul = u_lo[t]
        uh = u_hi[t]
        b = c0[t]
        s0 += ul * np.float32(b & 15) + uh * np.float32(b >> 4)
        b = c1[t]
        s1 += ul * np.float32(b & 15) + uh * np.float32(b >> 4)
        b = c2[t]
        s2 += ul * np.float32(b & 15) + uh * np.float32(b >> 4)
        b = c3[t]
        s3 += ul * np.float32(b & 15) + uh * np.float32(b >> 4)
    one = np.float32(1.0)
    return one - (base + s0), one - (base + s1), one - (base + s2), one - (base + s3)


@numba.njit(inline="always", fastmath=True)
def _code_distance(kernel, bits, a, b, base, c):
    if bits == 8:
        if kernel == KERNEL_SQEUCLIDEAN:
            return _one_sq8_l2(a, b, c)
        return _one_sq8_dot(a, base, c)
    if kernel == KERNEL_SQEUCLIDEAN:
        return _one_sq4_l2(a, b, c)
    return _one_sq4_dot(a, base, c)


@numba.njit(inline="always", fastmath=True)
def _four_code_distances(kernel, bits, a, b, base, codes, pending, r, out):
    c0 = codes[pending[r]]
    c1 = codes[pending[r + 1]]
    c2 = codes[pending[r + 2]]
    c3 = codes[pending[r + 3]]
    if bits == 8:
        if kernel == KERNEL_SQEUCLIDEAN:
            d0, d1, d2, d3 = _four_sq8_l2(a, b, c0, c1, c2, c3)
        else:
            d0, d1, d2, d3 = _four_sq8_dot(a, base, c0, c1, c2, c3)
    else:
        if kernel == KERNEL_SQEUCLIDEAN:
            d0, d1, d2, d3 = _four_sq4_l2(a, b, c0, c1, c2, c3)
        else:
            d0, d1, d2, d3 = _four_sq4_dot(a, base, c0, c1, c2, c3)
    out[0] = d0
    out[1] = d1
    out[2] = d2
    out[3] = d3


@numba.njit(inline="always", fastmath=True)
def _transform_query(kernel, bits, query, q_offsets, q_scales):
    """Return (a, b, base): the query in the form the code distances use."""
    dim = query.shape[0]
    if bits == 8:
        a = np.empty(dim, dtype=np.float32)
        b = np.empty(dim, dtype=np.float32)
        base = np.float32(0.0)
        for t in range(dim):
            if kernel == KERNEL_SQEUCLIDEAN:
                a[t] = (query[t] - q_offsets[t]) / q_scales[t]
                b[t] = q_scales[t] * q_scales[t]
            else:
                a[t] = query[t] * q_scales[t]
                b[t] = 0.0
                base += query[t] * q_offsets[t]
        return a.reshape((1, dim)), b.reshape((1, dim)), base
    n_bytes = (dim + 1) // 2
    a = np.zeros((2, n_bytes), dtype=np.float32)
    b = np.zeros((2, n_bytes), dtype=np.float32)
    base = np.float32(0.0)
    for j in range(dim):
        half = j & 1
        t = j >> 1
        if kernel == KERNEL_SQEUCLIDEAN:
            a[half, t] = (query[j] - q_offsets[j]) / q_scales[j]
            b[half, t] = q_scales[j] * q_scales[j]
        else:
            a[half, t] = query[j] * q_scales[j]
            base += query[j] * q_offsets[j]
    return a, b, base


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


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
def _search_codes(
    a, b, base, tree_current_query, heap_priorities, heap_indices, k, epsilon, visited,
    rng_state, codes, indptr, indices, tree_hyperplanes, tree_offsets, tree_children,
    tree_indices, n_search_trees, tree_leaf_budget, n_neighbors, min_distance, kernel,
    bits, max_degree, prefetch_lines,
):
    # a and b are 2-D for every code type (1 x dim for sq8, 2 x n_bytes for
    # sq4) so that all branches type the same.
    qa = a
    qb = b
    seed_set = [(np.float32(np.inf), np.int32(-1)) for j in range(0)]

    n_initial_points = 0
    if tree_leaf_budget == 1:
        for tree_num in range(n_search_trees):
            leaf_start, leaf_end = _tree_leaf(
                tree_current_query, tree_num, tree_hyperplanes, tree_offsets, tree_children,
                rng_state,
            )
            for j in range(leaf_start, leaf_end):
                candidate = tree_indices[tree_num, j]
                if check_and_mark_visited(visited, candidate) == 0:
                    d = np.float32(_code_distance(kernel, bits, qa, qb, base, codes[candidate]))
                    simple_heap_push(heap_priorities, heap_indices, d, candidate)
                    heapq.heappush(seed_set, (d, candidate))
                    n_initial_points += 1
    else:
        leaf_bounds = np.empty((tree_leaf_budget, 2), dtype=np.int32)
        leaf_count = _tree_spill_leaves(
            tree_current_query, rng_state, leaf_bounds, tree_hyperplanes, tree_offsets,
            tree_children, tree_leaf_budget,
        )
        for leaf_num in range(leaf_count):
            for j in range(leaf_bounds[leaf_num, 0], leaf_bounds[leaf_num, 1]):
                candidate = tree_indices[0, j]
                if check_and_mark_visited(visited, candidate) == 0:
                    d = np.float32(_code_distance(kernel, bits, qa, qb, base, codes[candidate]))
                    simple_heap_push(heap_priorities, heap_indices, d, candidate)
                    heapq.heappush(seed_set, (d, candidate))
                    n_initial_points += 1

    if len(seed_set) == 0:
        return

    distance_bound = _distance_bound(heap_priorities, heap_indices, epsilon, min_distance)
    d_vertex, vertex = heapq.heappop(seed_set)
    pending = np.empty(max_degree + 4, dtype=np.int32)
    pending_d = np.empty(4, dtype=np.float32)

    while d_vertex < distance_bound:
        n_pending = 0
        for j in range(indptr[vertex], indptr[vertex + 1]):
            candidate = indices[j]
            if check_and_mark_visited(visited, candidate) == 0:
                pending[n_pending] = candidate
                n_pending += 1
                if prefetch_lines > 0:
                    prefetch_row(codes, candidate, prefetch_lines)

        r = 0
        while r + 4 <= n_pending:
            _four_code_distances(kernel, bits, qa, qb, base, codes, pending, r, pending_d)
            for s in range(4):
                d = pending_d[s]
                if d < distance_bound:
                    candidate = pending[r + s]
                    simple_heap_push(heap_priorities, heap_indices, d, candidate)
                    heapq.heappush(seed_set, (d, candidate))
                    distance_bound = _distance_bound(
                        heap_priorities, heap_indices, epsilon, min_distance
                    )
            r += 4
        while r < n_pending:
            candidate = pending[r]
            d = np.float32(_code_distance(kernel, bits, qa, qb, base, codes[candidate]))
            if d < distance_bound:
                simple_heap_push(heap_priorities, heap_indices, d, candidate)
                heapq.heappush(seed_set, (d, candidate))
                distance_bound = _distance_bound(
                    heap_priorities, heap_indices, epsilon, min_distance
                )
            r += 1

        if len(seed_set) == 0:
            break
        d_vertex, vertex = heapq.heappop(seed_set)


@numba.njit(inline="always", fastmath=True)
def _rerank(
    query, candidates, n_rerank, data, out_priorities, out_indices, kernel, prefetch_lines
):
    """Exact distances for candidates[:n_rerank] (all valid), kept in a k-heap."""
    if prefetch_lines > 0:
        for j in range(n_rerank):
            prefetch_row(data, candidates[j], prefetch_lines)
    pending_d = np.empty(4, dtype=np.float32)
    r = 0
    while r + 4 <= n_rerank:
        four_distances_at(query, data, candidates, r, pending_d, kernel)
        for s in range(4):
            simple_heap_push(out_priorities, out_indices, pending_d[s], candidates[r + s])
        r += 4
    while r < n_rerank:
        c = candidates[r]
        if kernel == KERNEL_SQEUCLIDEAN:
            d = squared_euclidean(query, data[c])
        else:
            d = alternative_dot(query, data[c])
        simple_heap_push(out_priorities, out_indices, np.float32(d), c)
        r += 1


@numba.njit(inline="always", fastmath=True)
def _search_quantized_serial(
    index, query_points, tree_query_points, candidates, result, k, epsilon, rerank_width,
    visited, query_is_normalized, rng_state, kernel, bits,
):
    codes = index.codes
    data = index.data
    q_offsets = index.q_offsets
    q_scales = index.q_scales
    internal_rng_state = np.copy(rng_state)
    n_candidates = candidates[0].shape[1]
    for i in range(query_points.shape[0]):
        visited[:] = 0
        cand_priorities = candidates[1][i]
        cand_indices = candidates[0][i]
        cand_priorities[:] = np.inf
        cand_indices[:] = -1
        out_priorities = result[1][i]
        out_indices = result[0][i]
        out_priorities[:] = np.inf
        out_indices[:] = -1
        ok, current_query, tree_current_query = _prepare_queries(
            i, query_points, tree_query_points, query_is_normalized, index.normalize_query,
            index.tree_normalize_query,
        )
        if not ok:
            continue
        a, b, base = _transform_query(kernel, bits, current_query, q_offsets, q_scales)
        _search_codes(
            a, b, base, tree_current_query, cand_priorities, cand_indices, n_candidates,
            epsilon, visited, internal_rng_state, codes, index.indptr, index.indices,
            index.tree_hyperplanes, index.tree_offsets, index.tree_children,
            index.tree_indices, index.n_search_trees, index.tree_leaf_budget,
            index.n_neighbors, index.min_distance, kernel, bits, index.max_degree,
            index.prefetch_lines,
        )
        # Keep the best rerank_width candidates (by code distance), then rerank.
        n_valid = 0
        for j in range(n_candidates):
            if cand_indices[j] >= 0:
                cand_indices[n_valid] = cand_indices[j]
                cand_priorities[n_valid] = cand_priorities[j]
                n_valid += 1
        n_rerank = n_valid
        if 0 < rerank_width < n_valid:
            order = np.argsort(cand_priorities[:n_valid])
            best = cand_indices[:n_valid][order[:rerank_width]].copy()
            cand_indices[:rerank_width] = best
            n_rerank = rerank_width
        _rerank(
            current_query, cand_indices, n_rerank, data, out_priorities, out_indices, kernel,
            index.rerank_prefetch_lines,
        )
    return result


@numba.njit(fastmath=True, cache=True)
def search_sq8_sqeuclidean(
    index, query_points, tree_query_points, candidates, result, k, epsilon, rerank_width,
    visited, query_is_normalized, rng_state,
):
    return _search_quantized_serial(
        index, query_points, tree_query_points, candidates, result, k, epsilon, rerank_width,
        visited, query_is_normalized, rng_state, KERNEL_SQEUCLIDEAN, 8,
    )


@numba.njit(fastmath=True, cache=True)
def search_sq8_alternative_dot(
    index, query_points, tree_query_points, candidates, result, k, epsilon, rerank_width,
    visited, query_is_normalized, rng_state,
):
    return _search_quantized_serial(
        index, query_points, tree_query_points, candidates, result, k, epsilon, rerank_width,
        visited, query_is_normalized, rng_state, KERNEL_ALT_DOT, 8,
    )


@numba.njit(fastmath=True, cache=True)
def search_sq4_sqeuclidean(
    index, query_points, tree_query_points, candidates, result, k, epsilon, rerank_width,
    visited, query_is_normalized, rng_state,
):
    return _search_quantized_serial(
        index, query_points, tree_query_points, candidates, result, k, epsilon, rerank_width,
        visited, query_is_normalized, rng_state, KERNEL_SQEUCLIDEAN, 4,
    )


@numba.njit(fastmath=True, cache=True)
def search_sq4_alternative_dot(
    index, query_points, tree_query_points, candidates, result, k, epsilon, rerank_width,
    visited, query_is_normalized, rng_state,
):
    return _search_quantized_serial(
        index, query_points, tree_query_points, candidates, result, k, epsilon, rerank_width,
        visited, query_is_normalized, rng_state, KERNEL_ALT_DOT, 4,
    )


_DISPATCH = {
    (8, KERNEL_SQEUCLIDEAN): search_sq8_sqeuclidean,
    (8, KERNEL_ALT_DOT): search_sq8_alternative_dot,
    (4, KERNEL_SQEUCLIDEAN): search_sq4_sqeuclidean,
    (4, KERNEL_ALT_DOT): search_sq4_alternative_dot,
}


def default_code_prefetch_lines(codes):
    return default_prefetch_lines(codes)


def make_quantized_search_function(
    codes, data, q_offsets, q_scales, indptr, indices, search_forest, tree_leaf_budget,
    n_neighbors, min_distance, normalize_query, tree_normalize_query, kernel, bits,
    prefetch_lines=None, rerank_prefetch_lines=None,
):
    """Bind an index's arrays to the cached quantized search function.

    Returns ``search(query_points, tree_query_points, candidates, result, k,
    epsilon, rerank_width, visited, query_is_normalized, rng_state) -> result``
    where ``candidates`` is a (n_queries, candidate_width) heap workspace and
    ``result`` a (n_queries, k) heap that receives the exactly-scored neighbors.
    """
    dispatcher = _DISPATCH[(bits, kernel)]
    hyperplanes, offsets, children, tree_indices = stack_search_forest(
        search_forest, data.shape[0]
    )
    if prefetch_lines is None:
        prefetch_lines = default_code_prefetch_lines(codes)
    if rerank_prefetch_lines is None:
        rerank_prefetch_lines = default_prefetch_lines(data)
    index = _new_quantized_search_index(
        _readonly(codes, np.uint8),
        _readonly(data, np.float32),
        _readonly(q_offsets, np.float32),
        _readonly(q_scales, np.float32),
        _readonly(indptr, np.int32),
        _readonly(indices, np.int32),
        _readonly(hyperplanes, np.float32),
        _readonly(offsets, np.float32),
        _readonly(children, np.int32),
        _readonly(tree_indices, np.int32),
        np.int64(len(search_forest)),
        np.int64(tree_leaf_budget),
        np.int64(n_neighbors),
        np.float32(min_distance),
        bool(normalize_query),
        bool(tree_normalize_query),
        np.int64(int(np.max(np.diff(indptr))) if indptr.shape[0] > 1 else 0),
        np.int64(prefetch_lines),
        np.int64(rerank_prefetch_lines),
    )
    signature = (
        numba.typeof(index),
        _array(types.float32, 2),  # query_points
        _array(types.float32, 2),  # tree_query_points
        _HEAP_TYPE,  # candidates
        _HEAP_TYPE,  # result
        types.int64,  # k
        types.float32,  # epsilon
        types.int64,  # rerank_width
        _array(types.uint8, 1),  # visited
        types.boolean,  # query_is_normalized
        _array(types.int64, 1),  # rng_state
    )
    try:
        dispatcher.compile(signature)
        entry_point = dispatcher.overloads[signature].entry_point
        return functools.partial(entry_point, index)
    except (AttributeError, KeyError, TypeError):
        pass

    def search(
        query_points, tree_query_points, candidates, result, k, epsilon, rerank_width,
        visited, query_is_normalized, rng_state,
    ):
        return dispatcher(
            index,
            np.ascontiguousarray(query_points, dtype=np.float32),
            np.ascontiguousarray(tree_query_points, dtype=np.float32),
            candidates, result, k, epsilon, rerank_width, visited,
            bool(query_is_normalized), rng_state,
        )

    return search
