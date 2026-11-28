"""Progressive-precision NN-descent for high-dimensional dense data.

For dense float32 data with many features, most of the cost of building the
kNN graph is distance evaluations in the RP-tree leaf join and the first
NN-descent iterations. Those phases only need to find roughly the right
neighbours, so they run on 8-bit scalar-quantized codes, scored with the
integer dot product kernels of ``pynndescent.integer_dot`` (4-6x faster than
float32 at 784-4096 features). The approximate phase keeps a wider heap
(``APPROX_WIDTH * n_neighbors``); at the switch every heap entry is re-scored
exactly and the heaps are truncated to ``n_neighbors``, and float32
NN-descent runs to convergence from there. The wider heap is what keeps the
quality: pairs compared under the codes are not compared again, so the true
neighbours have to survive the approximate phase somewhere in the heap.

The codes and distances are those of the quantized search
(``quantized_search``): each point ``p`` acts as a query, with int8 weights
``w_p`` (from ``_integer_query``), and its distance to ``q`` is
``offset_p + scale_p * <w_p, c_q> (+ |c_q|^2_s)``, expressed in the exact
metric's units (squared euclidean, or alternative dot), so approximate and
exact distances can share a heap. Code and weight rows are zero-padded to
whole 64-byte cache lines and start on one; each row's scalars (code norm,
weight offset and scale) are packed into one aligned 16-byte record.

The returned graph has exact distances.
"""
import numba
import numpy as np

from pynndescent.distances import alternative_dot, squared_euclidean
from pynndescent.integer_dot import MAX_CODE_BYTES, dot4_u8, dot_u8
from pynndescent.kernels import (
    KERNEL_ALT_DOT,
    KERNEL_SQEUCLIDEAN,
    _emit,
    _four_alt_dot,
    _four_sqeuclidean,
    alt_dot_from_dot,
    generate_graph_update_array_batched,
)
from pynndescent.quantized_search import (
    _code_statistics,
    _encode_sq8,
    _integer_query,
    code_layout_scales,
)
from pynndescent.search import default_prefetch_lines
from pynndescent.utils import (
    apply_graph_update_array,
    apply_sorted_graph_update_array,
    build_candidates_owner,
    deheap_sort,
    make_heap,
    prefetch_element,
    prefetch_row,
    sort_updates_by_block,
    tau_rand_int,
    checked_flagged_heap_push,
)

# Below this many features the faster SQ8 distances do not pay for the wider
# approximate heaps (at 256 features the build was 11-15% slower).
MIN_DIMENSION = 512
APPROX_WIDTH = 1.25
# SQ8 iterations before switching to float32; None runs them until the update
# count converges on the codes (bounded by n_iters).
APPROX_ITERS = None
CACHE_LINE = 64
# The initialisation forest of wide data is built on a Gaussian random
# projection to TREE_DIMENSION features, with TREE_MULTIPLIER times the trees.
TREE_PROJECTION_MIN_DIMENSION = 1024
TREE_DIMENSION = 128
TREE_MULTIPLIER = 1.5


def project_for_trees(data, seed, n_threads, dims=TREE_DIMENSION):
    """``data`` times a Gaussian (n_features, dims) matrix, to build the
    initialisation forest on.

    The leaf join only needs leaves of roughly nearby points, and splits of a
    random low-dimensional projection give those about as well as splits of
    the data itself (Johnson-Lindenstrauss), at a fraction of the memory
    traffic: building a forest is memory bound on wide data, since every level
    reads every row and its node's hyperplane.
    """
    from threadpoolctl import threadpool_limits

    basis = np.random.default_rng(seed).standard_normal(
        (data.shape[1], dims), dtype=np.float32
    )
    out = np.empty((data.shape[0], dims), dtype=np.float32)
    with threadpool_limits(limits=n_threads, user_api="blas"):
        for start in range(0, data.shape[0], 65536):
            np.matmul(data[start : start + 65536], basis, out=out[start : start + 65536])
    return out


def progressive_build_applies(kernel, data):
    """Whether ``progressive_nn_descent`` supports (and helps) this build."""
    return (
        kernel in (KERNEL_SQEUCLIDEAN, KERNEL_ALT_DOT)
        and data.ndim == 2
        and MIN_DIMENSION <= data.shape[1] <= MAX_CODE_BYTES
        and data.shape[0] > 1
    )


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def _aligned_empty(shape, dtype):
    n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    buffer = np.empty(n_bytes + CACHE_LINE, dtype=np.uint8)
    offset = -buffer.ctypes.data % CACHE_LINE
    return buffer[offset : offset + n_bytes].view(dtype).reshape(shape)


@numba.njit(parallel=True, cache=True)
def _column_ranges(data):
    n, dim = data.shape
    n_chunks = min(n, 256)
    chunk = (n + n_chunks - 1) // n_chunks
    mins = np.empty((n_chunks, dim), dtype=np.float32)
    maxs = np.empty((n_chunks, dim), dtype=np.float32)
    for c in numba.prange(n_chunks):
        lo = mins[c]
        hi = maxs[c]
        lo[:] = np.inf
        hi[:] = -np.inf
        for i in range(c * chunk, min(n, (c + 1) * chunk)):
            row = data[i]
            for j in range(dim):
                lo[j] = min(lo[j], row[j])
                hi[j] = max(hi[j], row[j])
    lo = mins[0].copy()
    hi = maxs[0].copy()
    for c in range(1, n_chunks):
        for j in range(dim):
            lo[j] = min(lo[j], mins[c, j])
            hi[j] = max(hi[j], maxs[c, j])
    return lo, hi


@numba.njit(parallel=True, fastmath=True, cache=True)
def _row_weights(data, kernel, offsets, scales, lo, hi, code_means, code_norms, weights,
                 meta, n_chunks):
    """Each row as int8 query weights. The row is clamped to the code range
    first, as its codes are: an out-of-range value would otherwise set the int8
    step and round the weights of every other feature of the row to zero."""
    n, dim = data.shape
    with_norms = code_norms.shape[0] == n
    chunk = (n + n_chunks - 1) // n_chunks
    for c in numba.prange(n_chunks):
        row = np.empty(dim, dtype=np.float32)
        for i in range(c * chunk, min(n, (c + 1) * chunk)):
            for j in range(dim):
                row[j] = min(max(data[i, j], lo[j]), hi[j])
            # shift and mips_const are 0: the build only uses the squared
            # euclidean and alternative dot kernels (mips builds as euclidean).
            w, off, sc, _ = _integer_query(
                kernel, 8, row, np.float32(0.0), np.float32(0.0), offsets, scales, code_means,
                code_norms,
            )
            weights[i] = w
            meta[i, 0] = code_norms[i, 0] if with_norms else np.float32(0.0)
            meta[i, 1] = off
            meta[i, 2] = sc
            meta[i, 3] = np.float32(0.0)


def encode_sq8_build(data, kernel):
    """(codes, weights, meta) for the approximate build phase of ``data``.

    Each feature's code range spans its ``CLIP_QUANTILE``..``1 - CLIP_QUANTILE``
    quantiles (estimated on a row sample) rather than its full min/max: a few
    extreme values would otherwise stretch the range until nearly every row
    falls into a handful of codes (one value 1000 std out squeezes 98% of a
    feature into ~1 code, and the approximate phase goes blind). Values outside
    the range are clamped to the end codes; the exact re-score repairs the few
    rows this distorts.
    """
    n, dim = data.shape
    lo, hi = _clipped_column_ranges(data)
    scales = (hi - lo) / np.float32(255.0)
    # Features with a (near) constant value would get denormal scales whose
    # inverse overflows; they carry no distance information, so any scale does.
    scales = np.where(
        scales > np.finfo(np.float32).tiny * 255.0, scales, np.float32(1.0)
    ).astype(np.float32)
    offsets = lo.astype(np.float32)
    width = -(-dim // CACHE_LINE) * CACHE_LINE
    codes = _aligned_empty((n, width), np.uint8)
    codes[:, dim:] = 0
    _encode_sq8(data, offsets, (np.float32(1.0) / scales).astype(np.float32), codes[:, :dim])
    layout = code_layout_scales(scales, 8, width)
    means, norms = _code_statistics(codes, layout, kernel == KERNEL_SQEUCLIDEAN)
    weights = _aligned_empty((n, 1, width), np.int8)
    meta = _aligned_empty((n, 4), np.float32)
    _row_weights(data, kernel, offsets, scales, lo, hi, means, norms, weights, meta,
                 4 * numba.get_num_threads())
    return codes, weights, meta


CLIP_QUANTILE = 1e-3
CLIP_SAMPLE = 50_000
# A feature is clipped only when its full range is this many times its
# quantile range: an outlier, not a tail. A Gaussian feature's full range is
# about 1.5x its 0.1%..99.9% range; clamping genuine tails costs recall.
CLIP_RATIO = 4.0


@numba.njit(parallel=True, cache=True)
def _column_quantiles(sample, q):
    m, dim = sample.shape
    lo = np.empty(dim, dtype=np.float32)
    hi = np.empty(dim, dtype=np.float32)
    k_lo = int(q * (m - 1))
    k_hi = (m - 1) - k_lo
    for j in numba.prange(dim):
        column = np.sort(sample[:, j].copy())
        lo[j] = column[k_lo]
        hi[j] = column[k_hi]
    return lo, hi


def _clipped_column_ranges(data):
    """Per-feature (lo, hi): sample quantiles for features with outliers (full
    range over ``CLIP_RATIO`` times the quantile range), the full range for the
    rest, and for features whose quantile range is degenerate (e.g. a gene
    expressed in < 0.1% of cells)."""
    n = data.shape[0]
    full_lo, full_hi = _column_ranges(data)
    if n <= 1000:
        return full_lo, full_hi
    rows = np.sort(np.random.default_rng(0).choice(n, min(n, CLIP_SAMPLE), replace=False))
    lo, hi = _column_quantiles(np.ascontiguousarray(data[rows]), CLIP_QUANTILE)
    clip = (hi > lo) & ((full_hi - full_lo) > CLIP_RATIO * (hi - lo))
    lo = np.where(clip, lo, full_lo).astype(np.float32)
    hi = np.where(clip, hi, full_hi).astype(np.float32)
    return lo, hi


# --------------------------------------------------------------------------
# Approximate local joins
# --------------------------------------------------------------------------


@numba.njit(inline="always", fastmath=True)
def _to_distance(kernel, d, meta, q):
    if kernel == KERNEL_SQEUCLIDEAN:
        return d + meta[q, 0]
    return alt_dot_from_dot(np.float32(1.0) - d)


@numba.njit(inline="always", fastmath=True, boundscheck=False)
def _join_row(update_array, t, idx, max_updates, p, thresh_p, cands, start, heap_d, codes,
              weights, meta, kernel):
    """Score p against cands[start:] (four at a time) and emit improving pairs."""
    w = weights[p]
    off = meta[p, 1]
    sc = meta[p, 2]
    end = cands.shape[0]
    k = start
    while k < end:
        if idx >= max_updates:
            break
        if (
            k + 4 <= end
            and idx + 4 <= max_updates
            and cands[k] >= 0
            and cands[k + 1] >= 0
            and cands[k + 2] >= 0
            and cands[k + 3] >= 0
            and cands[k] != p
            and cands[k + 1] != p
            and cands[k + 2] != p
            and cands[k + 3] != p
        ):
            q0 = cands[k]
            q1 = cands[k + 1]
            q2 = cands[k + 2]
            q3 = cands[k + 3]
            s0, s1, s2, s3 = dot4_u8(codes, q0, q1, q2, q3, w)
            d0 = _to_distance(kernel, off + sc * np.float32(s0), meta, q0)
            d1 = _to_distance(kernel, off + sc * np.float32(s1), meta, q1)
            d2 = _to_distance(kernel, off + sc * np.float32(s2), meta, q2)
            d3 = _to_distance(kernel, off + sc * np.float32(s3), meta, q3)
            idx = _emit(update_array, t, idx, p, q0, d0, thresh_p, heap_d)
            idx = _emit(update_array, t, idx, p, q1, d1, thresh_p, heap_d)
            idx = _emit(update_array, t, idx, p, q2, d2, thresh_p, heap_d)
            idx = _emit(update_array, t, idx, p, q3, d3, thresh_p, heap_d)
            k += 4
        else:
            q = cands[k]
            if q >= 0 and q != p:
                s = dot_u8(codes, q, w)
                d = _to_distance(kernel, off + sc * np.float32(s), meta, q)
                idx = _emit(update_array, t, idx, p, q, d, thresh_p, heap_d)
            k += 1
    return idx


@numba.njit(inline="always")
def _prefetch_point(codes, weights, meta, q, lines, with_weights):
    prefetch_row(codes, q, lines)
    prefetch_element(meta, q, 0)
    if with_weights:
        prefetch_row(weights[q], 0, lines)


@numba.njit(inline="always", fastmath=True,
            locals={"p": numba.int32, "thresh_p": numba.float32})
def _join_candidates_thread(t, update_array, new_block, old_block, heap_d, codes, weights,
                            meta, lines, rows_per_thread, kernel):
    block_size = new_block.shape[0]
    max_new = new_block.shape[1]
    max_updates = update_array.shape[1]
    idx = 0
    for r in range(rows_per_thread):
        i = t * rows_per_thread + r
        if i >= block_size or idx >= max_updates:
            break
        # Row i's candidates are joined all-pairs: fetch them once up front.
        for j in range(max_new):
            q = new_block[i, j]
            if q >= 0:
                _prefetch_point(codes, weights, meta, q, lines, True)
        for j in range(old_block.shape[1]):
            q = old_block[i, j]
            if q >= 0:
                _prefetch_point(codes, weights, meta, q, lines, False)
        for j in range(max_new):
            if idx >= max_updates:
                break
            p = new_block[i, j]
            if p < 0:
                continue
            thresh_p = heap_d[p, 0]
            idx = _join_row(update_array, t, idx, max_updates, p, thresh_p, new_block[i],
                            j + 1, heap_d, codes, weights, meta, kernel)
            idx = _join_row(update_array, t, idx, max_updates, p, thresh_p, old_block[i], 0,
                            heap_d, codes, weights, meta, kernel)
    return idx


@numba.njit(inline="always", fastmath=True,
            locals={"p": numba.int32, "thresh_p": numba.float32})
def _join_leaves_thread(t, update_array, leaf_block, heap_d, codes, weights, meta, lines,
                        leaves_per_thread, kernel):
    n_leaves = leaf_block.shape[0]
    max_updates = update_array.shape[1]
    idx = 0
    for leaf in range(t * leaves_per_thread, min(n_leaves, (t + 1) * leaves_per_thread)):
        row = leaf_block[leaf]
        length = row.shape[0]
        for i in range(row.shape[0]):
            if row[i] < 0:
                length = i
                break
        row = row[:length]
        for i in range(length):
            _prefetch_point(codes, weights, meta, row[i], lines, True)
        for i in range(length - 1):
            p = row[i]
            idx = _join_row(update_array, t, idx, max_updates, p, heap_d[p, 0], row, i + 1,
                            heap_d, codes, weights, meta, kernel)
    return idx


# The metric is a literal in each entry point (not an argument), so the
# per-distance branches on it fold away; dispatching on it at run time costs
# the integer kernels ~20ns per distance.


@numba.njit(parallel=True, fastmath=True, cache=True)
def _join_candidates_sqeuclidean(update_array, n_updates_per_thread, new_block, old_block,
                                 heap_d, codes, weights, meta, lines, n_threads):
    rows_per_thread = new_block.shape[0] // n_threads + 1
    for t in numba.prange(n_threads):
        n_updates_per_thread[t] = _join_candidates_thread(
            t, update_array, new_block, old_block, heap_d, codes, weights, meta, lines,
            rows_per_thread, 1,
        )


@numba.njit(parallel=True, fastmath=True, cache=True)
def _join_candidates_alternative_dot(update_array, n_updates_per_thread, new_block,
                                     old_block, heap_d, codes, weights, meta, lines,
                                     n_threads):
    rows_per_thread = new_block.shape[0] // n_threads + 1
    for t in numba.prange(n_threads):
        n_updates_per_thread[t] = _join_candidates_thread(
            t, update_array, new_block, old_block, heap_d, codes, weights, meta, lines,
            rows_per_thread, 2,
        )


@numba.njit(parallel=True, fastmath=True, cache=True)
def _join_leaves_sqeuclidean(update_array, n_updates_per_thread, leaf_block, heap_d, codes,
                             weights, meta, lines, n_threads):
    leaves_per_thread = leaf_block.shape[0] // n_threads + 1
    for t in numba.prange(n_threads):
        n_updates_per_thread[t] = _join_leaves_thread(
            t, update_array, leaf_block, heap_d, codes, weights, meta, lines,
            leaves_per_thread, 1,
        )


@numba.njit(parallel=True, fastmath=True, cache=True)
def _join_leaves_alternative_dot(update_array, n_updates_per_thread, leaf_block, heap_d,
                                 codes, weights, meta, lines, n_threads):
    leaves_per_thread = leaf_block.shape[0] // n_threads + 1
    for t in numba.prange(n_threads):
        n_updates_per_thread[t] = _join_leaves_thread(
            t, update_array, leaf_block, heap_d, codes, weights, meta, lines,
            leaves_per_thread, 2,
        )


# --------------------------------------------------------------------------
# Switch to exact distances
# --------------------------------------------------------------------------


@numba.njit(inline="always", fastmath=True)
def _rescore_rows(t, rows_per_thread, order, indices, flags, data, k_out, out_i, out_d,
                  out_f, kernel):
    n, k_in = indices.shape
    d = np.empty(k_in, dtype=np.float32)
    for r in range(t * rows_per_thread, min(n, (t + 1) * rows_per_thread)):
        i = order[r]
        x = data[i]
        for j in range(k_in):
            q = indices[i, j]
            if q < 0:
                d[j] = np.inf
            elif kernel == KERNEL_SQEUCLIDEAN:
                d[j] = squared_euclidean(x, data[q])
            else:
                d[j] = alternative_dot(x, data[q])
        ranked = np.argsort(d)
        # Lay out as a valid max-heap: descending order.
        for s in range(k_out):
            src = ranked[k_out - 1 - s]
            out_i[i, s] = indices[i, src]
            out_d[i, s] = d[src]
            out_f[i, s] = flags[i, src]


@numba.njit(parallel=True, fastmath=True, cache=True)
def _rescore_sqeuclidean(indices, flags, data, k_out, order, out_i, out_d, out_f, n_threads):
    rows_per_thread = indices.shape[0] // n_threads + 1
    for t in numba.prange(n_threads):
        _rescore_rows(t, rows_per_thread, order, indices, flags, data, k_out, out_i, out_d,
                      out_f, 1)


@numba.njit(parallel=True, fastmath=True, cache=True)
def _rescore_alternative_dot(indices, flags, data, k_out, order, out_i, out_d, out_f,
                             n_threads):
    rows_per_thread = indices.shape[0] // n_threads + 1
    for t in numba.prange(n_threads):
        _rescore_rows(t, rows_per_thread, order, indices, flags, data, k_out, out_i, out_d,
                      out_f, 2)


@numba.njit(cache=True)
def leaf_order(leaf_array, n):
    """Points in order of first appearance in the leaves (a permutation of range(n)).

    Re-scoring is memory bound; visiting points in leaf order keeps the rows of
    their neighbours, mostly in the same leaves, in cache."""
    seen = np.zeros(n, dtype=np.uint8)
    order = np.empty(n, dtype=np.int64)
    m = 0
    for i in range(leaf_array.shape[0]):
        for j in range(leaf_array.shape[1]):
            q = leaf_array[i, j]
            if q < 0:
                break
            if not seen[q]:
                seen[q] = 1
                order[m] = q
                m += 1
    for q in range(n):
        if not seen[q]:
            order[m] = q
            m += 1
    return order


@numba.njit(fastmath=True, cache=True, locals={"d": numba.float32, "idx": numba.int32})
def _init_random_exact(n_neighbors, data, heap, rng_state, kernel):
    """``init_random`` for the approximate phase (exact distances are fine there)."""
    for i in range(data.shape[0]):
        if heap[0][i, 0] < 0.0:
            for j in range(n_neighbors - np.sum(heap[0][i] >= 0.0)):
                idx = np.abs(tau_rand_int(rng_state)) % data.shape[0]
                while idx == i:
                    idx = np.abs(tau_rand_int(rng_state)) % data.shape[0]
                if kernel == KERNEL_SQEUCLIDEAN:
                    d = squared_euclidean(data[idx], data[i])
                else:
                    d = alternative_dot(data[idx], data[i])
                checked_flagged_heap_push(heap[1][i], heap[0][i], heap[2][i], d, idx,
                                          np.uint8(1))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _apply(heap, updates, n_updates, sorted_updates, offsets, n, n_threads):
    if n_threads > 1:
        sort_updates_by_block(updates, n_updates, sorted_updates, offsets, n, n_threads)
        return apply_sorted_graph_update_array(heap, sorted_updates, offsets, n_threads)
    return apply_graph_update_array(heap, updates, n_updates, n_threads)


def progressive_nn_descent(data, n_neighbors, rng_state, max_candidates, kernel, n_iters,
                           delta, leaf_array, n_threads, verbose=False,
                           approx_iters=APPROX_ITERS, approx_width=APPROX_WIDTH,
                           exact_iters=None):
    """NN-descent with the leaf join and up to ``approx_iters`` iterations (None:
    until convergence) on SQ8 codes, with heaps ``approx_width`` times wider,
    then float32 to convergence (or for at most ``exact_iters`` iterations).

    Returns the deheap-sorted ``(indices, distances)`` graph, with exact distances.
    """
    n = data.shape[0]
    k = n_neighbors
    width = min(n - 1, max(k, int(round(approx_width * k))))
    if kernel == KERNEL_SQEUCLIDEAN:
        join_candidates = _join_candidates_sqeuclidean
        join_leaves = _join_leaves_sqeuclidean
        rescore = _rescore_sqeuclidean
    else:
        join_candidates = _join_candidates_alternative_dot
        join_leaves = _join_leaves_alternative_dot
        rescore = _rescore_alternative_dot

    codes, weights, meta = encode_sq8_build(data, kernel)
    lines = default_prefetch_lines(codes)
    heap = make_heap(n, width)
    offsets = np.zeros((n_threads, n_threads + 1), dtype=np.int64)
    n_updates = np.zeros(n_threads, dtype=np.int32)

    # Leaf join on the codes
    leaf_block_size = n_threads * 64
    max_leaf = leaf_array.shape[1]
    per_thread = int(leaf_block_size * max_leaf * (max_leaf - 1) / (2 * n_threads)) + 1
    updates = np.zeros((n_threads, per_thread, 3), dtype=np.float32)
    sorted_updates = np.empty((n_threads, 2 * per_thread, 3), dtype=np.float32)
    for b in range(leaf_array.shape[0] // leaf_block_size + 1):
        block = leaf_array[b * leaf_block_size : (b + 1) * leaf_block_size]
        join_leaves(updates, n_updates, block, heap[1], codes, weights, meta, lines,
                    n_threads)
        _apply(heap, updates, n_updates, sorted_updates, offsets, n, n_threads)
    del updates, sorted_updates
    _init_random_exact(width, data, heap, rng_state, kernel)

    # NN-descent; workspace sized for the wider approximate heap
    block_size = 16384
    n_blocks = n // block_size
    per_thread = int((max_candidates**2 + max_candidates * (max_candidates - 1) / 2)
                     * block_size / n_threads) + 1024
    updates = np.empty((n_threads, per_thread, 3), dtype=np.float32)
    if n_threads > 1:
        sorted_updates = np.empty((n_threads, 2 * per_thread, 3), dtype=np.float32)
    else:
        sorted_updates = np.empty((1, 1, 3), dtype=np.float32)
    new_i = np.full((n, max_candidates), -1, dtype=np.int32)
    new_p = np.full((n, max_candidates), np.inf, dtype=np.float32)
    old_i = np.full((n, max_candidates), -1, dtype=np.int32)
    old_p = np.full((n, max_candidates), np.inf, dtype=np.float32)
    active = np.ones(n, dtype=np.uint8)
    reverse_dst = np.empty(n * width, dtype=np.int32)
    reverse_src = np.empty(n * width, dtype=np.int32)
    bucket_bounds = np.zeros((n_threads, n_threads + 1), dtype=np.int64)

    if approx_iters is None:
        approx_iters = n_iters
    approximate = True
    n_exact = 0
    for it in range(n_iters):
        if approximate and it >= approx_iters:
            if verbose:
                print("\t Switching to exact distances")
            out_i = np.empty((n, k), dtype=np.int32)
            out_d = np.empty((n, k), dtype=np.float32)
            out_f = np.empty((n, k), dtype=np.uint8)
            rescore(heap[0], heap[2], data, k, leaf_order(leaf_array, n), out_i, out_d,
                    out_f, n_threads)
            heap = (out_i, out_d, out_f)
            approximate = False
            del codes, weights, meta
        if not approximate:
            if exact_iters is not None and n_exact >= exact_iters:
                break
            n_exact += 1
        if verbose:
            print("\t", it + 1, " / ", n_iters, "(sq8)" if approximate else "")
        new_c, old_c = build_candidates_owner(
            heap, max_candidates, rng_state, n_threads, new_i, new_p, old_i, old_p, active,
            reverse_dst, reverse_src, bucket_bounds,
        )
        c = 0
        for b in range(n_blocks + 1):
            s, e = b * block_size, min(n, (b + 1) * block_size)
            if approximate:
                join_candidates(updates, n_updates, new_c[s:e], old_c[s:e], heap[1], codes,
                                weights, meta, lines, n_threads)
            else:
                generate_graph_update_array_batched(updates, n_updates, new_c[s:e],
                                                    old_c[s:e], heap, data, kernel,
                                                    n_threads)
            c += _apply(heap, updates, n_updates, sorted_updates, offsets, n, n_threads)
        # Approximate update counts do not reflect exact convergence: only stop
        # early once on exact distances (converging on the codes just switches).
        if c <= delta * k * n:
            if approximate:
                approx_iters = it + 1
                continue
            if verbose:
                print("\tStopping threshold met -- exiting after", it + 1, "iterations")
            break

    if approximate:
        out_i = np.empty((n, k), dtype=np.int32)
        out_d = np.empty((n, k), dtype=np.float32)
        out_f = np.empty((n, k), dtype=np.uint8)
        rescore(heap[0], heap[2], data, k, leaf_order(leaf_array, n), out_i, out_d, out_f,
                n_threads)
        heap = (out_i, out_d, out_f)
    return deheap_sort(heap[0], heap[1])
