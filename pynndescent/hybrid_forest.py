"""Leaf-only random projection forests built breadth-first across all trees.

NN-descent only needs the leaves of its initialisation forest, so the trees'
hyperplanes are never kept. Large nodes of every tree are split level by level:
each level computes every point's side for every tree in a single parallel pass
over the data (in storage order, comparing each dense row against up to four
trees' hyperplanes at once), then partitions each node's range of a per-tree
permutation in parallel. Once a node's rows fit in a small working set it is
finished depth-first; all such subtrees, across all trees, are built in one
parallel pass. Unlike building one tree per thread, the work parallelises
across points and nodes, not only across trees.

Splits follow the rules of :mod:`pynndescent.rp_trees`: a hyperplane through
two random points of the node, a random side when the margin is within EPS of
zero, and a random split if every point lands on one side. Each node's random
choices are derived from a per-node seed, so the forest does not depend on the
order in which nodes are processed.
"""
import numpy as np
import numba

EPS = 1e-8
LOCAL_SUBTREE_BYTES = 1 << 20
_U = np.uint64


@numba.njit(inline="always")
def _mix64(x):
    # splitmix64 finaliser
    x = x + _U(0x9E3779B97F4A7C15)
    x = (x ^ (x >> _U(30))) * _U(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> _U(27))) * _U(0x94D049BB133111EB)
    return x ^ (x >> _U(31))


@numba.njit()
def _root_seeds(seed, n_trees):
    out = np.empty(n_trees, dtype=np.uint64)
    for t in range(n_trees):
        out[t] = _mix64(_U(seed) ^ _mix64(_U(t + 1)))
    return out


@numba.njit()
def _child_seeds(seeds):
    left = np.empty(seeds.shape[0], dtype=np.uint64)
    right = np.empty(seeds.shape[0], dtype=np.uint64)
    for i in range(seeds.shape[0]):
        left[i] = _mix64(seeds[i] * _U(2))
        right[i] = _mix64(seeds[i] * _U(2) + _U(1))
    return left, right


@numba.njit(inline="always")
def _two_points(seed, size):
    a = np.int64(_mix64(seed) % _U(size))
    b = np.int64(_mix64(seed ^ _U(0x5851F42D4C957F2D)) % _U(size - 1))
    if b >= a:
        b += 1
    return a, b


@numba.njit(inline="always")
def _side(margin, seed, point):
    if abs(margin) < EPS:
        return np.uint8(_mix64(seed ^ _U(point)) & _U(1))
    if margin > 0:
        return np.uint8(0)
    return np.uint8(1)


# --------------------------------------------------------------------------- dense


@numba.njit(fastmath=True, inline="always")
def _dense_plane(data, p, q, plane, angular):
    dim = data.shape[1]
    if angular:
        left_norm = np.float32(0.0)
        right_norm = np.float32(0.0)
        for d in range(dim):
            left_norm += data[p, d] * data[p, d]
            right_norm += data[q, d] * data[q, d]
        left_norm = np.sqrt(left_norm)
        right_norm = np.sqrt(right_norm)
        if abs(left_norm) < EPS:
            left_norm = np.float32(1.0)
        if abs(right_norm) < EPS:
            right_norm = np.float32(1.0)
        plane_norm = np.float32(0.0)
        for d in range(dim):
            h = data[p, d] / left_norm - data[q, d] / right_norm
            plane[d] = h
            plane_norm += h * h
        plane_norm = np.sqrt(plane_norm)
        if abs(plane_norm) < EPS:
            plane_norm = np.float32(1.0)
        for d in range(dim):
            plane[d] = plane[d] / plane_norm
        return np.float32(0.0)
    offset = np.float32(0.0)
    for d in range(dim):
        h = data[p, d] - data[q, d]
        plane[d] = h
        offset -= h * (data[p, d] + data[q, d]) * np.float32(0.5)
    return offset


@numba.njit(parallel=True, fastmath=True)
def _dense_planes(data, perm, s_tree, s_start, s_end, s_seed, planes, offsets, angular):
    for s in numba.prange(s_tree.shape[0]):
        t = s_tree[s]
        a, b = _two_points(s_seed[s], s_end[s] - s_start[s])
        offsets[s] = _dense_plane(
            data, perm[t, s_start[s] + a], perm[t, s_start[s] + b], planes[s], angular
        )


@numba.njit(parallel=True, fastmath=True, boundscheck=False)
def _dense_sides(data, assign, planes, offsets, s_seed, sides):
    n, dim = data.shape
    n_trees = assign.shape[0]
    chunk = 1024
    n_chunks = (n + chunk - 1) // chunk
    for c in numba.prange(n_chunks):
        trees = np.empty(4, np.int64)
        nodes = np.empty(4, np.int64)
        for p in range(c * chunk, min(n, (c + 1) * chunk)):
            x = data[p]
            t = 0
            while t < n_trees:
                # Collect up to four trees for which this point is in a split node
                n_active = 0
                while t < n_trees and n_active < 4:
                    s = assign[t, p]
                    if s >= 0:
                        trees[n_active] = t
                        nodes[n_active] = s
                        n_active += 1
                    t += 1
                if n_active == 4:
                    h0 = planes[nodes[0]]
                    h1 = planes[nodes[1]]
                    h2 = planes[nodes[2]]
                    h3 = planes[nodes[3]]
                    m0 = m1 = m2 = m3 = np.float32(0.0)
                    for d in range(dim):
                        xv = x[d]
                        m0 += xv * h0[d]
                        m1 += xv * h1[d]
                        m2 += xv * h2[d]
                        m3 += xv * h3[d]
                    sides[p, trees[0]] = _side(m0 + offsets[nodes[0]], s_seed[nodes[0]], p)
                    sides[p, trees[1]] = _side(m1 + offsets[nodes[1]], s_seed[nodes[1]], p)
                    sides[p, trees[2]] = _side(m2 + offsets[nodes[2]], s_seed[nodes[2]], p)
                    sides[p, trees[3]] = _side(m3 + offsets[nodes[3]], s_seed[nodes[3]], p)
                else:
                    for r in range(n_active):
                        h = planes[nodes[r]]
                        m = np.float32(0.0)
                        for d in range(dim):
                            m += x[d] * h[d]
                        sides[p, trees[r]] = _side(
                            m + offsets[nodes[r]], s_seed[nodes[r]], p
                        )


@numba.njit(parallel=True, fastmath=True, boundscheck=False)
def _dense_local_subtrees(
    data, perm, leaf_mark, d_tree, d_start, d_end, d_seed, d_depth, leaf_size, max_depth, angular
):
    dim = data.shape[1]
    for s in numba.prange(d_tree.shape[0]):
        t = d_tree[s]
        plane = np.empty(dim, np.float32)
        st_start = np.empty(256, np.int64)
        st_end = np.empty(256, np.int64)
        st_seed = np.empty(256, np.uint64)
        st_depth = np.empty(256, np.int64)
        st_start[0] = d_start[s]
        st_end[0] = d_end[s]
        st_seed[0] = d_seed[s]
        st_depth[0] = d_depth[s]
        top = 1
        while top > 0:
            top -= 1
            a0 = st_start[top]
            a1 = st_end[top]
            seed = st_seed[top]
            depth = st_depth[top]
            size = a1 - a0
            if size <= leaf_size or depth >= max_depth or top >= 254:
                leaf_mark[t, a0] = 1
                continue
            a, b = _two_points(seed, size)
            offset = _dense_plane(data, perm[t, a0 + a], perm[t, a0 + b], plane, angular)
            lo = a0
            hi = a1
            while lo < hi:
                point = perm[t, lo]
                m = offset
                for d in range(dim):
                    m += plane[d] * data[point, d]
                if _side(m, seed, point) == 0:
                    lo += 1
                else:
                    hi -= 1
                    perm[t, lo] = perm[t, hi]
                    perm[t, hi] = point
            left = lo - a0
            if left == 0 or left == size:
                left = size // 2
            st_start[top] = a0 + left
            st_end[top] = a1
            st_seed[top] = _mix64(seed * _U(2) + _U(1))
            st_depth[top] = depth + 1
            top += 1
            st_start[top] = a0
            st_end[top] = a0 + left
            st_seed[top] = _mix64(seed * _U(2))
            st_depth[top] = depth + 1
            top += 1


# --------------------------------------------------------------------------- sparse


@numba.njit(inline="always")
def _row_norm(indptr, values, p):
    r = 0.0
    for i in range(indptr[p], indptr[p + 1]):
        r += values[i] * values[i]
    r = np.sqrt(r)
    if abs(r) < EPS:
        r = 1.0
    return r


@numba.njit(inline="always")
def _sparse_plane_size(inds, indptr, p, q):
    i = indptr[p]
    ie = indptr[p + 1]
    j = indptr[q]
    je = indptr[q + 1]
    count = 0
    while i < ie and j < je:
        if inds[i] == inds[j]:
            i += 1
            j += 1
        elif inds[i] < inds[j]:
            i += 1
        else:
            j += 1
        count += 1
    return count + (ie - i) + (je - j)


@numba.njit(inline="always")
def _sparse_plane_fill(inds, indptr, values, p, q, out_inds, out_values, pos, angular):
    """Write the hyperplane for points p, q at out[pos:]; return its offset."""
    if angular:
        scale_p = 1.0 / _row_norm(indptr, values, p)
        scale_q = 1.0 / _row_norm(indptr, values, q)
    else:
        scale_p = 1.0
        scale_q = 1.0
    start = pos
    i = indptr[p]
    ie = indptr[p + 1]
    j = indptr[q]
    je = indptr[q + 1]
    offset = 0.0
    plane_norm = 0.0
    while i < ie or j < je:
        if j >= je or (i < ie and inds[i] < inds[j]):
            col = inds[i]
            a = values[i] * scale_p
            b = 0.0
            i += 1
        elif i >= ie or inds[j] < inds[i]:
            col = inds[j]
            a = 0.0
            b = values[j] * scale_q
            j += 1
        else:
            col = inds[i]
            a = values[i] * scale_p
            b = values[j] * scale_q
            i += 1
            j += 1
        h = a - b
        out_inds[pos] = col
        out_values[pos] = h
        offset -= h * (a + b) * 0.5
        plane_norm += h * h
        pos += 1
    if angular:
        plane_norm = np.sqrt(plane_norm)
        if abs(plane_norm) < EPS:
            plane_norm = 1.0
        for k in range(start, pos):
            out_values[k] = out_values[k] / plane_norm
        return np.float32(0.0)
    return np.float32(offset)


@numba.njit(inline="always")
def _sparse_margin(inds, indptr, values, p, plane_inds, plane_values, a, b):
    i = indptr[p]
    ie = indptr[p + 1]
    m = np.float32(0.0)
    while i < ie and a < b:
        if inds[i] == plane_inds[a]:
            m += values[i] * plane_values[a]
            i += 1
            a += 1
        elif inds[i] < plane_inds[a]:
            i += 1
        else:
            a += 1
    return m


@numba.njit()
def _sparse_planes(inds, indptr, values, perm, s_tree, s_start, s_end, s_seed, angular):
    n_split = s_tree.shape[0]
    points = np.empty((n_split, 2), np.int64)
    plane_ptr = np.zeros(n_split + 1, np.int64)
    for s in range(n_split):
        t = s_tree[s]
        a, b = _two_points(s_seed[s], s_end[s] - s_start[s])
        points[s, 0] = perm[t, s_start[s] + a]
        points[s, 1] = perm[t, s_start[s] + b]
        plane_ptr[s + 1] = plane_ptr[s] + _sparse_plane_size(
            inds, indptr, points[s, 0], points[s, 1]
        )
    plane_inds = np.empty(plane_ptr[n_split], np.int32)
    plane_values = np.empty(plane_ptr[n_split], np.float32)
    offsets = np.empty(n_split, np.float32)
    for s in range(n_split):
        offsets[s] = _sparse_plane_fill(
            inds, indptr, values, points[s, 0], points[s, 1],
            plane_inds, plane_values, plane_ptr[s], angular,
        )
    return plane_ptr, plane_inds, plane_values, offsets


@numba.njit(parallel=True)
def _sparse_sides(inds, indptr, values, assign, plane_ptr, plane_inds, plane_values, offsets, s_seed, sides):
    n = indptr.shape[0] - 1
    n_trees = assign.shape[0]
    for p in numba.prange(n):
        for t in range(n_trees):
            s = assign[t, p]
            if s >= 0:
                m = _sparse_margin(
                    inds, indptr, values, p, plane_inds, plane_values,
                    plane_ptr[s], plane_ptr[s + 1],
                ) + offsets[s]
                sides[p, t] = _side(m, s_seed[s], p)


@numba.njit(parallel=True)
def _sparse_local_subtrees(
    inds, indptr, values, perm, leaf_mark, d_tree, d_start, d_end, d_seed, d_depth,
    leaf_size, max_depth, angular,
):
    for s in numba.prange(d_tree.shape[0]):
        t = d_tree[s]
        st_start = np.empty(256, np.int64)
        st_end = np.empty(256, np.int64)
        st_seed = np.empty(256, np.uint64)
        st_depth = np.empty(256, np.int64)
        st_start[0] = d_start[s]
        st_end[0] = d_end[s]
        st_seed[0] = d_seed[s]
        st_depth[0] = d_depth[s]
        top = 1
        while top > 0:
            top -= 1
            a0 = st_start[top]
            a1 = st_end[top]
            seed = st_seed[top]
            depth = st_depth[top]
            size = a1 - a0
            if size <= leaf_size or depth >= max_depth or top >= 254:
                leaf_mark[t, a0] = 1
                continue
            a, b = _two_points(seed, size)
            p = perm[t, a0 + a]
            q = perm[t, a0 + b]
            nnz = _sparse_plane_size(inds, indptr, p, q)
            plane_inds = np.empty(nnz, np.int32)
            plane_values = np.empty(nnz, np.float32)
            offset = _sparse_plane_fill(
                inds, indptr, values, p, q, plane_inds, plane_values, 0, angular
            )
            lo = a0
            hi = a1
            while lo < hi:
                point = perm[t, lo]
                m = _sparse_margin(inds, indptr, values, point, plane_inds, plane_values, 0, nnz) + offset
                if _side(m, seed, point) == 0:
                    lo += 1
                else:
                    hi -= 1
                    perm[t, lo] = perm[t, hi]
                    perm[t, hi] = point
            left = lo - a0
            if left == 0 or left == size:
                left = size // 2
            st_start[top] = a0 + left
            st_end[top] = a1
            st_seed[top] = _mix64(seed * _U(2) + _U(1))
            st_depth[top] = depth + 1
            top += 1
            st_start[top] = a0
            st_end[top] = a0 + left
            st_seed[top] = _mix64(seed * _U(2))
            st_depth[top] = depth + 1
            top += 1


# --------------------------------------------------------------------------- shared


@numba.njit(parallel=True)
def _set_assignments(perm, assign, n_tree, n_start, n_end, values):
    for s in numba.prange(n_tree.shape[0]):
        t = n_tree[s]
        v = values[s]
        for i in range(n_start[s], n_end[s]):
            assign[t, perm[t, i]] = v


@numba.njit(parallel=True)
def _partition(perm, sides, s_tree, s_start, s_end, lefts):
    for s in numba.prange(s_tree.shape[0]):
        t = s_tree[s]
        lo = s_start[s]
        hi = s_end[s]
        while lo < hi:
            if sides[perm[t, lo], t] == 0:
                lo += 1
            else:
                hi -= 1
                tmp = perm[t, lo]
                perm[t, lo] = perm[t, hi]
                perm[t, hi] = tmp
        left = lo - s_start[s]
        size = s_end[s] - s_start[s]
        if left == 0 or left == size:
            # Every point fell on one side: split arbitrarily
            left = size // 2
        lefts[s] = left


@numba.njit()
def _leaves_to_array(perm, leaf_mark):
    n_trees, n = perm.shape
    n_leaves = 0
    max_size = 0
    for t in range(n_trees):
        last = -1
        for i in range(n):
            if leaf_mark[t, i]:
                n_leaves += 1
                if last >= 0:
                    max_size = max(max_size, i - last)
                last = i
        max_size = max(max_size, n - last)
    result = np.full((n_leaves, max_size), -1, np.int32)
    row = -1
    col = 0
    for t in range(n_trees):
        for i in range(n):
            if leaf_mark[t, i]:
                row += 1
                col = 0
            result[row, col] = perm[t, i]
            col += 1
    return result


def make_hybrid_leaf_array(
    data, n_trees, leaf_size, seed, angular=False, max_depth=200, local_bytes=LOCAL_SUBTREE_BYTES
):
    """Build a random projection forest and return only its leaves.

    Parameters
    ----------
    data: array of shape (n_samples, n_features) or scipy CSR matrix
        Dense float32 data (C-contiguous) or a CSR matrix with sorted indices.
    n_trees: int
    leaf_size: int
        Maximum number of points in a leaf.
    seed: int
        Seed from which every node's random choices are derived.
    angular: bool
        Use angular rather than euclidean splits.
    max_depth: int
    local_bytes: int
        Nodes whose rows fit in this many bytes are finished depth-first.

    Returns
    -------
    leaf_array: array of shape (n_leaves, max_leaf_size)
        Leaf membership, padded with -1, in the format of
        :func:`pynndescent.rp_trees.rptree_leaf_array`.
    """
    sparse = hasattr(data, "indptr")
    if sparse:
        inds = data.indices
        indptr = data.indptr
        values = data.data
        n = data.shape[0]
        row_bytes = max(1, int(8 * data.nnz / max(n, 1)))
    else:
        n, dim = data.shape
        row_bytes = 4 * dim
    local_points = max(leaf_size, local_bytes // row_bytes)

    perm = np.tile(np.arange(n, dtype=np.int32), (n_trees, 1))
    assign = np.full((n_trees, n), -1, dtype=np.int32)
    sides = np.zeros((n, n_trees), dtype=np.uint8)
    leaf_mark = np.zeros((n_trees, n), dtype=np.uint8)

    f_tree = np.arange(n_trees, dtype=np.int64)
    f_start = np.zeros(n_trees, dtype=np.int64)
    f_end = np.full(n_trees, n, dtype=np.int64)
    f_seed = _root_seeds(np.uint64(seed & 0xFFFFFFFFFFFFFFFF), n_trees)
    f_depth = np.zeros(n_trees, dtype=np.int64)
    deferred = []

    while f_tree.shape[0] > 0:
        size = f_end - f_start
        is_leaf = (size <= leaf_size) | (f_depth >= max_depth)
        is_deferred = ~is_leaf & (size <= local_points)
        is_split = ~is_leaf & ~is_deferred

        leaf_mark[f_tree[is_leaf], f_start[is_leaf]] = 1
        if is_deferred.any():
            deferred.append(
                (
                    f_tree[is_deferred],
                    f_start[is_deferred],
                    f_end[is_deferred],
                    f_seed[is_deferred],
                    f_depth[is_deferred],
                )
            )
        retired = ~is_split
        if retired.any():
            _set_assignments(
                perm, assign, f_tree[retired], f_start[retired], f_end[retired],
                np.full(int(retired.sum()), -1, dtype=np.int32),
            )
        if not is_split.any():
            break

        s_tree = f_tree[is_split]
        s_start = f_start[is_split]
        s_end = f_end[is_split]
        s_seed = f_seed[is_split]
        s_depth = f_depth[is_split]
        n_split = s_tree.shape[0]

        _set_assignments(
            perm, assign, s_tree, s_start, s_end, np.arange(n_split, dtype=np.int32)
        )
        if sparse:
            plane_ptr, plane_inds, plane_values, offsets = _sparse_planes(
                inds, indptr, values, perm, s_tree, s_start, s_end, s_seed, angular
            )
            _sparse_sides(
                inds, indptr, values, assign, plane_ptr, plane_inds, plane_values,
                offsets, s_seed, sides,
            )
        else:
            planes = np.empty((n_split, dim), dtype=np.float32)
            offsets = np.empty(n_split, dtype=np.float32)
            _dense_planes(data, perm, s_tree, s_start, s_end, s_seed, planes, offsets, angular)
            _dense_sides(data, assign, planes, offsets, s_seed, sides)

        lefts = np.empty(n_split, dtype=np.int64)
        _partition(perm, sides, s_tree, s_start, s_end, lefts)
        mid = s_start + lefts
        left_seeds, right_seeds = _child_seeds(s_seed)
        f_tree = np.concatenate([s_tree, s_tree])
        f_start = np.concatenate([s_start, mid])
        f_end = np.concatenate([mid, s_end])
        f_seed = np.concatenate([left_seeds, right_seeds])
        f_depth = np.concatenate([s_depth + 1, s_depth + 1])

    if deferred:
        d_tree, d_start, d_end, d_seed, d_depth = [np.concatenate(c) for c in zip(*deferred)]
        if sparse:
            _sparse_local_subtrees(
                inds, indptr, values, perm, leaf_mark, d_tree, d_start, d_end, d_seed,
                d_depth, leaf_size, max_depth, angular,
            )
        else:
            _dense_local_subtrees(
                data, perm, leaf_mark, d_tree, d_start, d_end, d_seed, d_depth,
                leaf_size, max_depth, angular,
            )

    return _leaves_to_array(perm, leaf_mark)
