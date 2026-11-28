"""Graph search over scalar-quantized vectors with exact reranking.

``NNDescent(..., quantization="sq8")`` (or ``"sq4"``) stores, next to the float32
data, one code per feature: 8 bits, or 4 bits packed two to a byte. Each feature
gets its own affine map ``x ~ offset + scale * code`` fitted to the data's range
(for 4-bit codes, the range between its 0.3% and 99.7% quantiles; values
outside it get the end codes).

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

A ``metric='mips'`` index (``KERNEL_MIPS``) is a euclidean index of the data
augmented with ``a = sqrt(M**2 - |x|**2)``, searched with queries padded with a
zero, so its squared distances are ``|q|**2 + M**2 - 2 <q, x>``. Its codes are of
the original features only and its code distance is that expression, with the
dot code distance's estimate of ``<q, x>``: the appended feature, which the
queries never weight, takes no code space, and its quantization error (which
would enter a squared euclidean code distance through the per-row norm) none of
the distances.

Optionally (``quantization_dims``) the codes are of ``P^T (x - mean)`` for the
top principal axes ``P`` instead of ``x``, and the query is projected the same
way once (``<q, x> ~ <q, mean> + <P^T q, P^T (x - mean)>``); reranking still
uses every feature.

By default the code distances are computed in integer arithmetic: both
distances are linear in the codes (``sum_i u_i c_i`` plus per-query and, for
squared euclidean, per-row constants), so the query is rounded once to int8
weights ``u`` and each candidate costs an integer dot product
(``pynndescent.integer_dot``; AVX-512 VNNI where available). Code rows are then
padded to whole 64-byte cache lines and stored cache-line aligned.
"""
import functools
import numbers

import numba
import numpy as np
from numba import types
from numba.experimental import structref

from pynndescent.distances import alternative_dot, squared_euclidean
from pynndescent.integer_dot import aligned_padded_codes, dot4_u4, dot4_u8, dot_u4, dot_u8
from pynndescent.kernels import (
    KERNEL_ALT_DOT,
    KERNEL_MIPS,
    KERNEL_SQEUCLIDEAN,
    alt_dot_from_dot,
    four_distances_at,
)
from pynndescent.search import (
    SEED_CAPACITY,
    _array,
    _distance_bound,
    _grow_seeds,
    _prepare_queries,
    _readonly,
    _seed_pop,
    _seed_push,
    _tree_leaf,
    _tree_spill_leaves,
    default_adjacency_lines,
    default_prefetch_lines,
    padded_adjacency,
    stack_search_forest,
)
from pynndescent.utils import (
    _deheap_rows_inplace,
    check_and_mark_visited,
    float16_bits_as_float32,
    prefetch_element,
    prefetch_row,
    simple_heap_push,
    siftdown,
)

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


def encode_float16(data):
    """A float16 copy of ``data / scale`` for reranking, as uint16 bits, and ``scale``.

    ``scale`` is the largest absolute value, so every stored value lies in
    [-1, 1]: nothing overflows and only negligible values become subnormal.
    """
    max_abs = float(np.abs(data).max()) if data.size else 0.0
    scale = np.float32(max_abs if max_abs > 0.0 and np.isfinite(max_abs) else 1.0)
    rerank16 = np.empty(data.shape, dtype=np.float16)
    block = 16384
    for start in range(0, data.shape[0], block):
        rerank16[start : start + block] = np.asarray(
            data[start : start + block], dtype=np.float32
        ) / scale
    return rerank16.view(np.uint16), scale


MAX_PROJECTED_FRACTION = 2.0 / 3.0


def n_projected_components(eigenvalues, dims):
    """Components to keep: ``dims`` itself (int), or the fewest that keep a
    ``dims`` fraction of the variance (float in (0, 1)); ``eigenvalues`` sorted
    in decreasing order."""
    if isinstance(dims, numbers.Integral):
        return int(dims)
    explained = np.cumsum(np.maximum(eigenvalues, 0.0))
    if explained[-1] <= 0.0:
        return eigenvalues.shape[0]
    return int(np.searchsorted(explained / explained[-1], dims) + 1)


def fit_pca_projection(data, dims, sample_size=100_000, random_state=None, extra=None):
    """Project the codes onto the top principal axes of (a sample of) ``data``.

    With ``extra`` (sample queries, say), the axes are those of the data sample
    and (a sample of up to as many rows of) ``extra`` together.

    Returns ``(mean, projection8, column_scale)``, the projection being stored
    as an int8 (dim, n_components) matrix with a scale per column (column j is
    ``column_scale[j] * projection8[:, j]``), which keeps projecting a query
    cheap. Returns None when more than ``MAX_PROJECTED_FRACTION`` of the
    features would be kept, as the projection would then cost more than the
    shorter codes save.
    """
    dim = data.shape[1]
    rng = np.random.default_rng(random_state)
    n = data.shape[0]
    rows = np.sort(rng.choice(n, sample_size, replace=False)) if n > sample_size else slice(None)
    sample = np.asarray(data[rows], dtype=np.float64)
    if extra is not None and extra.shape[0] > 0:
        n_extra = extra.shape[0]
        extra_rows = (
            np.sort(rng.choice(n_extra, sample_size, replace=False))
            if n_extra > sample_size
            else slice(None)
        )
        sample = np.vstack([sample, np.asarray(extra[extra_rows], dtype=np.float64)])
    mean = sample.mean(axis=0)
    centered = sample - mean
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    order = np.argsort(eigenvalues)[::-1]
    n_components = n_projected_components(eigenvalues[order], dims)
    if n_components > MAX_PROJECTED_FRACTION * dim:
        return None
    projection = np.ascontiguousarray(eigenvectors[:, order[:n_components]], dtype=np.float32)
    projection8, column_scale = _int8_columns(projection)
    return mean.astype(np.float32), projection8, column_scale


def project_rows(data, mean, projection8, column_scale, block=65536):
    """``(data - mean) @ projection``, with the int8-stored projection the queries use."""
    projection = projection8.astype(np.float32) * column_scale
    out = np.empty((data.shape[0], projection.shape[1]), dtype=np.float32)
    for start in range(0, data.shape[0], block):
        rows = np.asarray(data[start : start + block], dtype=np.float32)
        out[start : start + block] = (rows - mean) @ projection
    return out


def _int8_columns(projection):
    """``projection`` as int8 with one scale per column: column j ~ scale[j] * int8[:, j]."""
    max_abs = np.abs(projection).max(axis=0)
    scale = np.where(max_abs > 0, max_abs / np.float32(127.0), np.float32(1.0)).astype(np.float32)
    quantized = np.clip(np.rint(projection / scale), -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized), scale


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
    "rerank16",
    "rerank_scale",
    "rerank_half",
    "projection8",
    "projection_scale",
    "projection_mean",
    "vertex_order",
    "q_offsets",
    "q_scales",
    "code_means",
    "code_norms",
    "adjacency",
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
    "prefetch_lines",
    "rerank_prefetch_lines",
    "adjacency_lines",
    "mips_norm2",
]
structref.define_proxy(QuantizedSearchIndex, QuantizedSearchIndexType, _FIELDS)


@numba.njit(cache=True)
def _new_quantized_search_index(
    codes, data, rerank16, rerank_scale, rerank_half, projection8,
    projection_scale, projection_mean, vertex_order, q_offsets, q_scales, code_means,
    code_norms, adjacency, tree_hyperplanes,
    tree_offsets, tree_children, tree_indices, n_search_trees, tree_leaf_budget,
    n_neighbors, min_distance, normalize_query, tree_normalize_query,
    prefetch_lines, rerank_prefetch_lines, adjacency_lines, mips_norm2,
):
    return QuantizedSearchIndex(
        codes, data, rerank16, rerank_scale, rerank_half, projection8,
        projection_scale, projection_mean, vertex_order, q_offsets, q_scales, code_means,
        code_norms, adjacency, tree_hyperplanes,
        tree_offsets, tree_children, tree_indices, n_search_trees, tree_leaf_budget,
        n_neighbors, min_distance, normalize_query, tree_normalize_query,
        prefetch_lines, rerank_prefetch_lines, adjacency_lines, mips_norm2,
    )


def code_layout_scales(q_scales, bits, width):
    """Per-feature scales laid out like a code row: (1, width) for 8-bit codes,
    (2, width) (low / high nibbles) for 4-bit; zero on padding."""
    dim = q_scales.shape[0]
    if bits == 8:
        out = np.zeros((1, width), dtype=np.float32)
        out[0, :dim] = q_scales
    else:
        out = np.zeros((2, width), dtype=np.float32)
        out[0, : (dim + 1) // 2] = q_scales[0::2]
        out[1, : dim // 2] = q_scales[1::2]
    return out


@numba.njit(parallel=True, cache=True)
def _code_statistics(codes, layout_scales, with_norms):
    """Mean code per (nibble, byte) and, if ``with_norms``, each row's
    ``sum_j scale_j**2 * code_j**2`` (the squared norm of its centred decoding)."""
    n, width = codes.shape
    halves = layout_scales.shape[0]
    n_chunks = max(1, min(n, 256))
    chunk = (n + n_chunks - 1) // n_chunks
    partial = np.zeros((n_chunks, halves, width), dtype=np.float64)
    norms = np.zeros((n if with_norms else 0, 1), dtype=np.float32)
    squares = layout_scales * layout_scales
    for c in numba.prange(n_chunks):
        for i in range(c * chunk, min(n, (c + 1) * chunk)):
            total = np.float32(0.0)
            for t in range(width):
                byte = codes[i, t]
                if halves == 1:
                    v = np.float32(byte)
                    partial[c, 0, t] += v
                    total += squares[0, t] * v * v
                else:
                    lo = np.float32(byte & 15)
                    hi = np.float32(byte >> 4)
                    partial[c, 0, t] += lo
                    partial[c, 1, t] += hi
                    total += squares[0, t] * lo * lo + squares[1, t] * hi * hi
            if with_norms:
                norms[i, 0] = total
    means = np.zeros((halves, width), dtype=np.float32)
    for h in range(halves):
        for t in range(width):
            s = 0.0
            for c in range(n_chunks):
                s += partial[c, h, t]
            means[h, t] = s / max(n, 1)
    return means, norms


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
def _float_code_distance(kernel, bits, a, b, base, c):
    if bits == 8:
        if kernel == KERNEL_SQEUCLIDEAN:
            return _one_sq8_l2(a, b, c)
        return _one_sq8_dot(a, base, c)
    if kernel == KERNEL_SQEUCLIDEAN:
        return _one_sq4_l2(a, b, c)
    return _one_sq4_dot(a, base, c)


@numba.njit(inline="always", fastmath=True)
def _four_float_code_distances(kernel, bits, a, b, base, codes, pending, r, out):
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
    dim = q_offsets.shape[0]  # a padded mips query has a feature more than the codes
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


@numba.njit(inline="always", fastmath=True)
def _integer_query(
    kernel, bits, query, shift, mips_const, q_offsets, q_scales, code_means, code_norms
):
    """The query as int8 weights ``w`` with ``d ~ offset + scale * <w, c> (+ norm)``.

    Both distances are linear in the codes: ``|q - x|^2 = |q - o|^2 - 2 <u, c> +
    |s c|^2`` and ``1 - <q, x> = 1 - <q, o> - <u, c>``, with ``u = s (q - o)``
    (euclidean) or ``s q`` (dot and mips; mips distances are
    ``mips_const - 2 <q, x>``). ``u`` is rounded to ``step * w``; the rounding's
    expected effect over the data, ``sum_j (step w_j - u_j) mean_j``, is
    subtracted, so the rounding mostly adds the same amount to every candidate.
    """
    dim = q_offsets.shape[0]  # a padded mips query has a feature more than the codes
    halves = code_means.shape[0]
    width = code_means.shape[1]
    u = np.zeros((halves, width), dtype=np.float32)
    total = np.float32(0.0)
    for j in range(dim):
        if bits == 8:
            h = 0
            t = j
        else:
            h = j & 1
            t = j >> 1
        if kernel == KERNEL_SQEUCLIDEAN:
            e = query[j] - q_offsets[j]
            total += e * e
            u[h, t] = e * q_scales[j]
        else:
            total += query[j] * q_offsets[j]
            u[h, t] = query[j] * q_scales[j]
    max_abs = np.float32(0.0)
    for h in range(halves):
        for t in range(width):
            max_abs = max(max_abs, abs(u[h, t]))
    step = max_abs / np.float32(127.0) if max_abs > np.float32(0.0) else np.float32(1.0)
    inv_step = np.float32(1.0) / step
    weights = np.empty((halves, width), dtype=np.int8)
    correction = np.float32(0.0)
    for h in range(halves):
        for t in range(width):
            w = np.rint(u[h, t] * inv_step)
            weights[h, t] = np.int8(w)
            correction += (step * w - u[h, t]) * code_means[h, t]
    if kernel == KERNEL_SQEUCLIDEAN:
        return weights, total + np.float32(2.0) * correction, np.float32(-2.0) * step, code_norms
    if kernel == KERNEL_MIPS:
        two = np.float32(2.0)
        return weights, mips_const - two * (total + shift - correction), -two * step, code_norms
    return weights, np.float32(1.0) - (total + shift) + correction, -step, code_norms


@numba.njit(inline="always", fastmath=True)
def _query_state(integer, kernel, bits, query, shift, mips_const, index):
    """The transformed query, ``(a, b, base, weights, offset, scale, norms)``: the
    float kernels use ``(a, b, base)`` (and, for mips, ``offset``), the integer
    kernels the rest (the unused entries are empty)."""
    if integer:
        weights, offset, scale, norms = _integer_query(
            kernel, bits, query, shift, mips_const, index.q_offsets, index.q_scales,
            index.code_means, index.code_norms,
        )
        a = np.zeros((1, 0), dtype=np.float32)
        b = np.zeros((1, 0), dtype=np.float32)
        base = np.float32(0.0)
    else:
        a, b, base = _transform_query(kernel, bits, query, index.q_offsets, index.q_scales)
        base += shift
        weights = np.zeros((1, 0), dtype=np.int8)
        # mips: the dot code distance d = 1 - <q, x> becomes 2 d + offset
        offset = mips_const - np.float32(2.0) if kernel == KERNEL_MIPS else np.float32(0.0)
        scale = np.float32(0.0)
        norms = index.code_norms
    return a, b, base, weights, offset, scale, norms


@numba.njit(inline="always", fastmath=True)
def _integer_code_distance(kernel, bits, weights, offset, scale, norms, codes, candidate):
    if bits == 8:
        dot = dot_u8(codes, candidate, weights)
    else:
        dot = dot_u4(codes, candidate, weights)
    d = offset + scale * np.float32(dot)
    if kernel == KERNEL_SQEUCLIDEAN:
        d += norms[candidate, 0]
    return d


@numba.njit(inline="always", fastmath=True)
def _code_distance(kernel, bits, integer, state, codes, candidate):
    """Code distance of ``codes[candidate]`` to the transformed query ``state``."""
    if integer:
        return _integer_code_distance(
            kernel, bits, state[3], state[4], state[5], state[6], codes, candidate
        )
    d = _float_code_distance(kernel, bits, state[0], state[1], state[2], codes[candidate])
    if kernel == KERNEL_MIPS:
        d = np.float32(2.0) * d + state[4]
    return d


@numba.njit(inline="always", fastmath=True)
def _four_integer_code_distances(
    kernel, bits, weights, offset, scale, norms, codes, pending, r, out
):
    p0 = pending[r]
    p1 = pending[r + 1]
    p2 = pending[r + 2]
    p3 = pending[r + 3]
    if bits == 8:
        s0, s1, s2, s3 = dot4_u8(codes, p0, p1, p2, p3, weights)
    else:
        s0, s1, s2, s3 = dot4_u4(codes, p0, p1, p2, p3, weights)
    out[0] = offset + scale * np.float32(s0)
    out[1] = offset + scale * np.float32(s1)
    out[2] = offset + scale * np.float32(s2)
    out[3] = offset + scale * np.float32(s3)
    if kernel == KERNEL_SQEUCLIDEAN:
        out[0] += norms[p0, 0]
        out[1] += norms[p1, 0]
        out[2] += norms[p2, 0]
        out[3] += norms[p3, 0]


@numba.njit(inline="always", fastmath=True)
def _four_code_distances(kernel, bits, integer, state, codes, pending, r, out):
    """Code distances of ``codes[pending[r:r + 4]]`` into ``out``."""
    if integer:
        _four_integer_code_distances(
            kernel, bits, state[3], state[4], state[5], state[6], codes, pending, r, out
        )
    else:
        _four_float_code_distances(
            kernel, bits, state[0], state[1], state[2], codes, pending, r, out
        )
        if kernel == KERNEL_MIPS:
            for s in range(4):
                out[s] = np.float32(2.0) * out[s] + state[4]


@numba.njit(inline="always")
def _prefetch_code(integer, state, codes, candidate, n_lines):
    """Prefetch what the code distance of ``candidate`` reads."""
    prefetch_row(codes, candidate, n_lines)
    if integer:
        norms = state[6]
        if norms.shape[0] > 0:
            prefetch_element(norms, candidate, 0)


@numba.njit(inline="always", fastmath=True)
def _project_query(kernel, query, projection8, projection_scale, mean):
    """The query in the projected code space, and the dot product term lost by
    centring (zero for squared euclidean, where the query is centred instead)."""
    dim = projection8.shape[0]  # a padded mips query has a feature more
    n_components = projection8.shape[1]
    out = np.zeros(n_components, dtype=np.float32)
    shift = np.float32(0.0)
    for i in range(dim):
        if kernel == KERNEL_SQEUCLIDEAN:
            qi = query[i] - mean[i]
        else:
            qi = query[i]
            shift += qi * mean[i]
        row8 = projection8[i]
        for j in range(n_components):
            out[j] += qi * np.float32(row8[j])
    for j in range(n_components):
        out[j] *= projection_scale[j]
    return out, shift


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
    state, tree_current_query, heap_priorities, heap_indices, k, epsilon, visited,
    rng_state, codes, adjacency, tree_hyperplanes, tree_offsets, tree_children,
    tree_indices, n_search_trees, tree_leaf_budget, n_neighbors, min_distance, kernel,
    bits, prefetch_lines, adjacency_lines, integer,
):
    seed_d = np.empty(SEED_CAPACITY, dtype=np.float32)
    seed_i = np.empty(SEED_CAPACITY, dtype=np.int32)
    n_seeds = 0

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
                    d = np.float32(_code_distance(kernel, bits, integer, state, codes, candidate))
                    simple_heap_push(heap_priorities, heap_indices, d, candidate)
                    if n_seeds == seed_d.shape[0]:
                        seed_d, seed_i = _grow_seeds(seed_d, seed_i, n_seeds, n_seeds + 1)
                    n_seeds = _seed_push(seed_d, seed_i, n_seeds, d, candidate)
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
                    d = np.float32(_code_distance(kernel, bits, integer, state, codes, candidate))
                    simple_heap_push(heap_priorities, heap_indices, d, candidate)
                    if n_seeds == seed_d.shape[0]:
                        seed_d, seed_i = _grow_seeds(seed_d, seed_i, n_seeds, n_seeds + 1)
                    n_seeds = _seed_push(seed_d, seed_i, n_seeds, d, candidate)
                    n_initial_points += 1

    if n_seeds == 0:
        return

    distance_bound = _distance_bound(heap_priorities, heap_indices, epsilon, min_distance)
    d_vertex = seed_d[0]
    vertex = seed_i[0]
    n_seeds = _seed_pop(seed_d, seed_i, n_seeds)
    pending = np.empty(adjacency.shape[1] + 4, dtype=np.int32)
    pending_d = np.empty(4, dtype=np.float32)

    while d_vertex < distance_bound:
        if n_seeds + adjacency.shape[1] > seed_d.shape[0]:
            seed_d, seed_i = _grow_seeds(
                seed_d, seed_i, n_seeds, n_seeds + adjacency.shape[1]
            )
        if adjacency_lines > 0 and n_seeds > 0:
            # The closest remaining candidate is likely expanded next: fetch its
            # adjacency row (at a fixed, cache-aligned offset) while this vertex
            # is processed.
            prefetch_row(adjacency, seed_i[0], adjacency_lines)
        n_pending = 0
        row = adjacency[vertex]
        for j in range(row.shape[0]):
            candidate = row[j]
            if candidate < 0:
                break
            if check_and_mark_visited(visited, candidate) == 0:
                pending[n_pending] = candidate
                n_pending += 1
                if prefetch_lines > 0:
                    _prefetch_code(integer, state, codes, candidate, prefetch_lines)

        r = 0
        while r + 4 <= n_pending:
            _four_code_distances(kernel, bits, integer, state, codes, pending, r, pending_d)
            for s in range(4):
                d = pending_d[s]
                if d < distance_bound:
                    candidate = pending[r + s]
                    simple_heap_push(heap_priorities, heap_indices, d, candidate)
                    n_seeds = _seed_push(seed_d, seed_i, n_seeds, d, candidate)
                    distance_bound = _distance_bound(
                        heap_priorities, heap_indices, epsilon, min_distance
                    )
            r += 4
        while r < n_pending:
            candidate = pending[r]
            d = np.float32(_code_distance(kernel, bits, integer, state, codes, candidate))
            if d < distance_bound:
                simple_heap_push(heap_priorities, heap_indices, d, candidate)
                n_seeds = _seed_push(seed_d, seed_i, n_seeds, d, candidate)
                distance_bound = _distance_bound(
                    heap_priorities, heap_indices, epsilon, min_distance
                )
            r += 1

        if n_seeds == 0:
            break
        d_vertex = seed_d[0]
        vertex = seed_i[0]
        n_seeds = _seed_pop(seed_d, seed_i, n_seeds)


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


@numba.njit(fastmath=True, inline="always", boundscheck=False)
def _four_half_sums(kernel, x, y0, y1, y2, y3):
    """Squared euclidean distances or dot products of x with four float16 rows."""
    s0 = s1 = s2 = s3 = np.float32(0.0)
    if kernel == KERNEL_SQEUCLIDEAN:
        for t in range(x.shape[0]):
            xv = x[t]
            e = xv - float16_bits_as_float32(y0[t])
            s0 += e * e
            e = xv - float16_bits_as_float32(y1[t])
            s1 += e * e
            e = xv - float16_bits_as_float32(y2[t])
            s2 += e * e
            e = xv - float16_bits_as_float32(y3[t])
            s3 += e * e
    else:
        for t in range(x.shape[0]):
            xv = x[t]
            s0 += xv * float16_bits_as_float32(y0[t])
            s1 += xv * float16_bits_as_float32(y1[t])
            s2 += xv * float16_bits_as_float32(y2[t])
            s3 += xv * float16_bits_as_float32(y3[t])
    return s0, s1, s2, s3


@numba.njit(fastmath=True, inline="always")
def _half_distance(kernel, s, factor, mips_const):
    if kernel == KERNEL_SQEUCLIDEAN:
        return np.float32(s * factor)
    if kernel == KERNEL_MIPS:
        return mips_const - np.float32(2.0) * s
    return alt_dot_from_dot(s)


@numba.njit(inline="always", fastmath=True)
def _rerank_half(
    query, candidates, n_rerank, rerank16, scale, out_priorities, out_indices, kernel,
    prefetch_lines, mips_const,
):
    """As ``_rerank``, from the float16 copy ``rerank16`` (= data / scale).

    mips distances are ``mips_const - 2 <q, y>``: the padded query's zero leaves the
    appended feature out, so its float16 rounding (large, as the feature is) does
    not enter them, as it would a squared euclidean distance."""
    if prefetch_lines > 0:
        for j in range(n_rerank):
            prefetch_row(rerank16, candidates[j], prefetch_lines)
    # Fold the scale into the query: |q - s y|^2 = s^2 |q / s - y|^2 and
    # <q, s y> = <s q, y>.
    x = np.empty(query.shape[0], dtype=np.float32)
    if kernel == KERNEL_SQEUCLIDEAN:
        inv_scale = np.float32(1.0) / scale
        for t in range(query.shape[0]):
            x[t] = query[t] * inv_scale
        factor = scale * scale
    else:
        for t in range(query.shape[0]):
            x[t] = query[t] * scale
        factor = np.float32(1.0)
    r = 0
    while r + 4 <= n_rerank:
        c0 = candidates[r]
        c1 = candidates[r + 1]
        c2 = candidates[r + 2]
        c3 = candidates[r + 3]
        s0, s1, s2, s3 = _four_half_sums(
            kernel, x, rerank16[c0], rerank16[c1], rerank16[c2], rerank16[c3]
        )
        simple_heap_push(
            out_priorities, out_indices, _half_distance(kernel, s0, factor, mips_const), c0
        )
        simple_heap_push(
            out_priorities, out_indices, _half_distance(kernel, s1, factor, mips_const), c1
        )
        simple_heap_push(
            out_priorities, out_indices, _half_distance(kernel, s2, factor, mips_const), c2
        )
        simple_heap_push(
            out_priorities, out_indices, _half_distance(kernel, s3, factor, mips_const), c3
        )
        r += 4
    while r < n_rerank:
        c = candidates[r]
        row = rerank16[c]
        s0, _, _, _ = _four_half_sums(kernel, x, row, row, row, row)
        simple_heap_push(
            out_priorities, out_indices, _half_distance(kernel, s0, factor, mips_const), c
        )
        r += 1


@numba.njit(inline="always", fastmath=True)
def _search_quantized_serial(
    index, query_points, tree_query_points, candidates, result, k, epsilon, rerank_width,
    visited, query_is_normalized, rng_state, kernel, bits, integer,
):
    codes = index.codes
    data = index.data
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
        mips_const = np.float32(0.0)
        if kernel == KERNEL_MIPS:
            # |q|**2 + M**2, so that distances are squared euclidean on the augmented data
            for t in range(current_query.shape[0]):
                mips_const += current_query[t] * current_query[t]
            mips_const += index.mips_norm2
        if index.projection8.shape[1] > 0:
            projected, shift = _project_query(
                kernel, current_query, index.projection8, index.projection_scale,
                index.projection_mean,
            )
            state = _query_state(integer, kernel, bits, projected, shift, mips_const, index)
        else:
            state = _query_state(
                integer, kernel, bits, current_query, np.float32(0.0), mips_const, index
            )
        _search_codes(
            state, tree_current_query, cand_priorities, cand_indices, n_candidates,
            epsilon, visited, internal_rng_state, codes, index.adjacency,
            index.tree_hyperplanes, index.tree_offsets, index.tree_children,
            index.tree_indices, index.n_search_trees, index.tree_leaf_budget,
            index.n_neighbors, index.min_distance, kernel, bits,
            index.prefetch_lines, index.adjacency_lines, integer,
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
        if index.rerank_half:
            _rerank_half(
                current_query, cand_indices, n_rerank, index.rerank16, index.rerank_scale,
                out_priorities, out_indices, kernel, index.rerank_prefetch_lines, mips_const,
            )
        else:
            # mips: exact squared euclidean distances on the augmented data
            _rerank(
                current_query, cand_indices, n_rerank, data, out_priorities, out_indices,
                KERNEL_SQEUCLIDEAN if kernel == KERNEL_MIPS else kernel,
                index.rerank_prefetch_lines,
            )
    return result


@numba.njit(inline="always", fastmath=True)
def _search_and_finish(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state, kernel, bits, integer,
):
    """Search, then sort each row's results and map them to the original data order,
    all in one compiled call (each call from Python costs microseconds).

    Returns ``(indices, distances)``: per row the k neighbours in the original data
    order, nearest first, and their uncorrected distances (the search kernel's).
    """
    n_queries = query_points.shape[0]
    # The search resets every row before use.
    candidates = (
        np.empty((n_queries, candidate_width), dtype=np.int32),
        np.empty((n_queries, candidate_width), dtype=np.float32),
    )
    result = (
        np.empty((n_queries, k), dtype=np.int32),
        np.empty((n_queries, k), dtype=np.float32),
    )
    _search_quantized_serial(
        index, query_points, query_points, candidates, result, k, epsilon, rerank_width,
        visited, query_is_normalized, rng_state, kernel, bits, integer,
    )
    _deheap_rows_inplace(result[0], result[1])
    vertex_order = index.vertex_order
    indices = np.empty((n_queries, k), dtype=vertex_order.dtype)
    for i in range(n_queries):
        for j in range(k):
            indices[i, j] = vertex_order[result[0][i, j]]
    return indices, result[1]


@numba.njit(fastmath=True, cache=True)
def search_sq8_sqeuclidean(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state,
):
    return _search_and_finish(
        index, query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state, KERNEL_SQEUCLIDEAN, 8, False,
    )


@numba.njit(fastmath=True, cache=True)
def search_sq8_alternative_dot(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state,
):
    return _search_and_finish(
        index, query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state, KERNEL_ALT_DOT, 8, False,
    )


@numba.njit(fastmath=True, cache=True)
def search_sq4_sqeuclidean(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state,
):
    return _search_and_finish(
        index, query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state, KERNEL_SQEUCLIDEAN, 4, False,
    )


@numba.njit(fastmath=True, cache=True)
def search_sq4_alternative_dot(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state,
):
    return _search_and_finish(
        index, query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state, KERNEL_ALT_DOT, 4, False,
    )


# The same searches over int8 query weights and integer code dot products
# (``integer_dot``); these need codes from ``aligned_padded_codes``.


@numba.njit(fastmath=True, cache=True)
def search_sq8_sqeuclidean_int(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state,
):
    return _search_and_finish(
        index, query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state, KERNEL_SQEUCLIDEAN, 8, True,
    )


@numba.njit(fastmath=True, cache=True)
def search_sq8_alternative_dot_int(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state,
):
    return _search_and_finish(
        index, query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state, KERNEL_ALT_DOT, 8, True,
    )


@numba.njit(fastmath=True, cache=True)
def search_sq4_sqeuclidean_int(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state,
):
    return _search_and_finish(
        index, query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state, KERNEL_SQEUCLIDEAN, 4, True,
    )


@numba.njit(fastmath=True, cache=True)
def search_sq4_alternative_dot_int(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state,
):
    return _search_and_finish(
        index, query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state, KERNEL_ALT_DOT, 4, True,
    )


@numba.njit(fastmath=True, cache=True)
def search_sq8_mips(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state,
):
    return _search_and_finish(
        index, query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state, KERNEL_MIPS, 8, False,
    )

@numba.njit(fastmath=True, cache=True)
def search_sq8_mips_int(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state,
):
    return _search_and_finish(
        index, query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state, KERNEL_MIPS, 8, True,
    )

@numba.njit(fastmath=True, cache=True)
def search_sq4_mips(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state,
):
    return _search_and_finish(
        index, query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state, KERNEL_MIPS, 4, False,
    )

@numba.njit(fastmath=True, cache=True)
def search_sq4_mips_int(
    index, query_points, k, candidate_width, epsilon, rerank_width, visited,
    query_is_normalized, rng_state,
):
    return _search_and_finish(
        index, query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state, KERNEL_MIPS, 4, True,
    )


_DISPATCH = {
    (8, KERNEL_SQEUCLIDEAN, False): search_sq8_sqeuclidean,
    (8, KERNEL_ALT_DOT, False): search_sq8_alternative_dot,
    (4, KERNEL_SQEUCLIDEAN, False): search_sq4_sqeuclidean,
    (4, KERNEL_ALT_DOT, False): search_sq4_alternative_dot,
    (8, KERNEL_SQEUCLIDEAN, True): search_sq8_sqeuclidean_int,
    (8, KERNEL_ALT_DOT, True): search_sq8_alternative_dot_int,
    (4, KERNEL_SQEUCLIDEAN, True): search_sq4_sqeuclidean_int,
    (4, KERNEL_ALT_DOT, True): search_sq4_alternative_dot_int,
    (8, KERNEL_MIPS, False): search_sq8_mips,
    (4, KERNEL_MIPS, False): search_sq4_mips,
    (8, KERNEL_MIPS, True): search_sq8_mips_int,
    (4, KERNEL_MIPS, True): search_sq4_mips_int,
}


def default_code_prefetch_lines(codes):
    return default_prefetch_lines(codes)


def full_row_prefetch_lines(data):
    """Cache lines spanned by one (possibly unaligned) row of ``data``."""
    return (data.shape[1] * data.itemsize + 63) // 64 + 1


def make_quantized_search_function(
    codes, data, q_offsets, q_scales, indptr, indices, search_forest, tree_leaf_budget,
    n_neighbors, min_distance, normalize_query, tree_normalize_query, kernel, bits,
    prefetch_lines=None, rerank_prefetch_lines=None, adjacency_lines=None,
    rerank16=None, rerank_scale=1.0, projection8=None, projection_scale=None,
    projection_mean=None, integer_query=False, vertex_order=None, mips_norm2=0.0,
):
    """Bind an index's arrays to the cached quantized search function.

    Returns ``search(query_points, k, candidate_width, epsilon, rerank_width,
    visited, query_is_normalized, rng_state) -> (indices, distances)``: per query
    the k exactly-scored nearest neighbours, nearest first, as indices into the
    original data order (``vertex_order[i]`` is the original index of row i of
    ``data``; the identity if None) and their uncorrected exact distances.
    With ``rerank16`` (float16 bits of ``data / rerank_scale``, from
    ``encode_float16``) the candidates are reranked from that copy instead.
    With ``integer_query`` the code distances are integer dot products with
    int8 query weights (``codes`` are then aligned and padded if they are not
    already, see ``aligned_padded_codes``).
    For ``KERNEL_MIPS`` ``data`` (and ``rerank16``) are the augmented data, the
    codes are of its features but the last, and ``mips_norm2`` is ``M**2``.
    """
    integer_query = bool(integer_query)
    dispatcher = _DISPATCH[(bits, kernel, integer_query)]
    if integer_query:
        codes = aligned_padded_codes(codes)
        layout_scales = code_layout_scales(
            np.asarray(q_scales, dtype=np.float32), bits, codes.shape[1]
        )
        code_means, code_norms = _code_statistics(
            codes, layout_scales, kernel == KERNEL_SQEUCLIDEAN
        )
    else:
        code_means = np.zeros((1, 0), dtype=np.float32)
        code_norms = np.zeros((0, 1), dtype=np.float32)
    hyperplanes, offsets, children, tree_indices = stack_search_forest(
        search_forest, data.shape[0]
    )
    adjacency = padded_adjacency(indptr, indices)
    if adjacency_lines is None:
        adjacency_lines = default_adjacency_lines(adjacency)
    if prefetch_lines is None:
        prefetch_lines = default_code_prefetch_lines(codes)
    if vertex_order is None:
        vertex_order = np.arange(data.shape[0], dtype=np.int64)
    rerank_half = rerank16 is not None
    if not rerank_half:
        rerank16 = np.empty((0, data.shape[1]), dtype=np.uint16)
    dim = q_offsets.shape[0] if kernel == KERNEL_MIPS else data.shape[1]
    if projection8 is None:
        # (dim, 0): the codes are in the data's own coordinates.
        projection8 = np.empty((dim, 0), dtype=np.int8)
        projection_scale = np.ones(0, dtype=np.float32)
        projection_mean = np.zeros(dim, dtype=np.float32)
    if rerank_prefetch_lines is None:
        # Whole float16 rows pay off (they are half the size, so all the
        # finalists fit in flight); for float32 rows that is too many lines.
        rerank_prefetch_lines = (
            full_row_prefetch_lines(rerank16) if rerank_half else default_prefetch_lines(data)
        )
    index = _new_quantized_search_index(
        _readonly(codes, np.uint8),
        _readonly(data, np.float32),
        _readonly(rerank16, np.uint16),
        np.float32(rerank_scale),
        bool(rerank_half),
        _readonly(projection8, np.int8),
        _readonly(projection_scale, np.float32),
        _readonly(projection_mean, np.float32),
        _readonly(vertex_order, vertex_order.dtype),
        _readonly(q_offsets, np.float32),
        _readonly(q_scales, np.float32),
        _readonly(code_means, np.float32),
        _readonly(code_norms, np.float32),
        _readonly(adjacency, np.int32),
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
        np.int64(prefetch_lines),
        np.int64(rerank_prefetch_lines),
        np.int64(adjacency_lines),
        np.float32(mips_norm2),
    )
    signature = (
        numba.typeof(index),
        _array(types.float32, 2),  # query_points
        types.int64,  # k
        types.int64,  # candidate_width
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
        query_points, k, candidate_width, epsilon, rerank_width, visited,
        query_is_normalized, rng_state,
    ):
        return dispatcher(
            index,
            np.ascontiguousarray(query_points, dtype=np.float32),
            k, candidate_width, epsilon, rerank_width, visited,
            bool(query_is_normalized), rng_state,
        )

    return search
