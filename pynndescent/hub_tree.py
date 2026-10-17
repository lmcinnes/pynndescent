"""Parallel construction of the graph-informed (hub) search tree.

``prepare()`` builds one search tree whose splits are chosen from the highest
in-degree points ("hubs") of the k-NN graph. The original builders recurse
depth-first in a single thread. This module builds the same kind of tree level
by level so that all threads share the work:

* every node that is still large is cut into fixed-size segments, and each pass
  (hub selection, candidate margins, split counts, edge cuts, partition) runs in
  parallel over the segments of all nodes of the level;
* nodes whose rows fit in a small working set are finished depth-first, all of
  them in one parallel pass.

For every node the candidate split hyperplanes are those through the node's
top-3 hubs (all three pairs). The margins of every point against every candidate
are written to a ``(3, n)`` array indexed by permutation position, so choosing a
split, counting edge cuts, the median fallback and the stable partition are all
independent of the data kind. Only building hyperplanes and computing margins is
kind specific (dense, sparse CSR, or bit-packed); that code mirrors the serial
builders in :mod:`pynndescent.rp_trees` so that, with the matching options, the
same trees are produced.

Nodes are stored as the pair of hubs that defines their hyperplane, plus offset
corrections; hyperplanes are materialised once at the end, directly in the flat
pre-order format used by the search code. All decisions depend only on a node's
contents (and per-node seeds), so the tree does not depend on the number of
threads.
"""
import numpy as np
import numba

from pynndescent.distances import popcnt_u8
from pynndescent.hybrid_forest import _mix64
from pynndescent.rp_trees import (
    EPS,
    FlatTree,
    MIN_SPLIT_BALANCE,
    compute_global_degrees,
    get_top_k_hub_indices,
)
from pynndescent.sparse import (
    sparse_diff,
    sparse_dot_accumulate,
    sparse_mul,
    sparse_sum,
)
from pynndescent.utils import norm

KIND_DENSE = 0
KIND_SPARSE = 1
KIND_BIT = 2

CRITERION_BALANCE = 0
CRITERION_EDGE_CUT = 1

FALLBACK_LEAF = 0
FALLBACK_RANDOM = 1
FALLBACK_MEDIAN = 2

ACTION_LEAF = 0
ACTION_SPLIT = 1
ACTION_RANDOM = 2

# Split selection per data kind: (criterion, min_balance, fallback).
#
# The defaults pick the most balanced hub-pair split and, when none reaches
# MIN_SPLIT_BALANCE, move the best hub hyperplane to the median margin. Compared
# with the original builders (SERIAL_SPLIT_OPTIONS) this avoids oversized leaves
# for dense data (MNIST-cosine queries ~2.4x faster at equal recall) and the
# very deep, unbalanced trees edge-cut selection produced for sparse and bit
# data (20NG ~1.3-1.5x, binarised MNIST ~1.45x faster queries; GloVe and sparse
# MNIST unchanged), and keeps tree depth logarithmic.
DEFAULT_SPLIT_OPTIONS = {
    KIND_DENSE: (CRITERION_BALANCE, MIN_SPLIT_BALANCE, FALLBACK_MEDIAN),
    KIND_SPARSE: (CRITERION_BALANCE, MIN_SPLIT_BALANCE, FALLBACK_MEDIAN),
    KIND_BIT: (CRITERION_BALANCE, MIN_SPLIT_BALANCE, FALLBACK_MEDIAN),
}

# Options that reproduce the original serial hub tree builders in rp_trees
SERIAL_SPLIT_OPTIONS = {
    KIND_DENSE: (CRITERION_BALANCE, MIN_SPLIT_BALANCE, FALLBACK_LEAF),
    KIND_SPARSE: (CRITERION_EDGE_CUT, 0.0, FALLBACK_RANDOM),
    KIND_BIT: (CRITERION_EDGE_CUT, 0.0, FALLBACK_RANDOM),
}

SEGMENT_SIZE = 4096
LOCAL_SUBTREE_BYTES = 1 << 20
_U = np.uint64

# Candidate c uses hubs (_PAIR_A[c], _PAIR_B[c]) of the node's top-3 hubs, in the
# order the serial builders evaluate them.
_PAIR_A = np.array([0, 0, 1], dtype=np.int64)
_PAIR_B = np.array([1, 2, 2], dtype=np.int64)


# --------------------------------------------------------------------------- planes
# The plane and margin code below intentionally mirrors rp_trees'
# euclidean_hub_split, angular_hub_split, sparse_*_hub_split and bit_hub_split
# expression for expression, so that numba infers the same intermediate types.


@numba.njit(fastmath=True, inline="always")
def _dense_euclidean_plane(data, left, right, hyperplane_vector):
    dim = data.shape[1]
    hyperplane_offset = np.float32(0.0)
    for d in range(dim):
        hyperplane_vector[d] = data[left, d] - data[right, d]
        hyperplane_offset -= (
            hyperplane_vector[d] * (data[left, d] + data[right, d]) / 2.0
        )
    return hyperplane_offset


@numba.njit(fastmath=True, inline="always")
def _dense_angular_plane(data, left, right, hyperplane_vector):
    dim = data.shape[1]
    left_norm = norm(data[left])
    right_norm = norm(data[right])

    if abs(left_norm) < EPS:
        left_norm = 1.0
    if abs(right_norm) < EPS:
        right_norm = 1.0

    for d in range(dim):
        hyperplane_vector[d] = (data[left, d] / left_norm) - (
            data[right, d] / right_norm
        )

    hyperplane_norm = norm(hyperplane_vector)
    if abs(hyperplane_norm) < EPS:
        hyperplane_norm = 1.0
    for d in range(dim):
        hyperplane_vector[d] = hyperplane_vector[d] / hyperplane_norm
    return 0.0


@numba.njit(fastmath=True, inline="always")
def _dense_euclidean_margin(data, hyperplane_vector, hyperplane_offset, point):
    margin = hyperplane_offset
    for d in range(data.shape[1]):
        margin += hyperplane_vector[d] * data[point, d]
    return margin


@numba.njit(fastmath=True, inline="always")
def _dense_angular_margin(data, hyperplane_vector, point):
    margin = np.float32(0.0)
    for d in range(data.shape[1]):
        margin += hyperplane_vector[d] * data[point, d]
    return margin


@numba.njit(fastmath=True)
def _sparse_euclidean_plane(inds, indptr, spdata, left, right):
    left_inds = inds[indptr[left] : indptr[left + 1]]
    left_data = spdata[indptr[left] : indptr[left + 1]]
    right_inds = inds[indptr[right] : indptr[right + 1]]
    right_data = spdata[indptr[right] : indptr[right + 1]]

    hyperplane_offset = np.float64(0.0)
    hyperplane_inds, hyperplane_data = sparse_diff(
        left_inds, left_data, right_inds, right_data
    )
    offset_inds, offset_data = sparse_sum(left_inds, left_data, right_inds, right_data)
    offset_data = offset_data / 2.0
    offset_inds, offset_data = sparse_mul(
        hyperplane_inds,
        hyperplane_data,
        offset_inds,
        offset_data.astype(np.float32),
    )
    for val in offset_data:
        hyperplane_offset -= val
    return hyperplane_inds, hyperplane_data, hyperplane_offset


@numba.njit(fastmath=True)
def _sparse_angular_plane(inds, indptr, spdata, left, right):
    left_inds = inds[indptr[left] : indptr[left + 1]]
    left_data = spdata[indptr[left] : indptr[left + 1]]
    right_inds = inds[indptr[right] : indptr[right + 1]]
    right_data = spdata[indptr[right] : indptr[right + 1]]

    left_norm = norm(left_data)
    right_norm = norm(right_data)

    if abs(left_norm) < EPS:
        left_norm = 1.0
    if abs(right_norm) < EPS:
        right_norm = 1.0

    normalized_left_data = (left_data / left_norm).astype(np.float32)
    normalized_right_data = (right_data / right_norm).astype(np.float32)

    hyperplane_inds, hyperplane_data = sparse_diff(
        left_inds, normalized_left_data, right_inds, normalized_right_data
    )

    hyperplane_norm = norm(hyperplane_data)
    if abs(hyperplane_norm) < EPS:
        hyperplane_norm = 1.0
    for d in range(hyperplane_data.shape[0]):
        hyperplane_data[d] = hyperplane_data[d] / hyperplane_norm
    return hyperplane_inds, hyperplane_data, np.float64(0.0)


@numba.njit(inline="always")
def _sparse_plane(inds, indptr, spdata, left, right, angular):
    if angular:
        return _sparse_angular_plane(inds, indptr, spdata, left, right)
    return _sparse_euclidean_plane(inds, indptr, spdata, left, right)


@numba.njit(inline="always")
def _sparse_margin(inds, indptr, spdata, plane_inds, plane_data, offset, point):
    i_inds = inds[indptr[point] : indptr[point + 1]]
    i_data = spdata[indptr[point] : indptr[point + 1]]
    return sparse_dot_accumulate(offset, plane_inds, plane_data, i_inds, i_data)


@numba.njit(fastmath=True, inline="always")
def _bit_plane(data, left, right, hyperplane_vector):
    dim = data.shape[1]
    for d in range(dim):
        xor_vector = data[left, d] ^ data[right, d]
        hyperplane_vector[d] = xor_vector & data[left, d]
        hyperplane_vector[dim + d] = xor_vector & data[right, d]


@numba.njit(fastmath=True, inline="always")
def _bit_margin(data, hyperplane_vector, point):
    dim = data.shape[1]
    margin = np.float32(0.0)
    for d in range(dim):
        margin += np.float32(popcnt_u8(hyperplane_vector[d] & data[point, d]))
        margin -= np.float32(popcnt_u8(hyperplane_vector[dim + d] & data[point, d]))
    return margin


@numba.njit(parallel=True, fastmath=True, cache=True)
def _build_planes(kind, angular, ddata, inds, indptr, spdata, bdata, pairs):
    """Hyperplanes for each (left hub, right hub) pair; -1 marks an absent pair.

    Dense planes are rows of ``dplanes``; bit planes rows of ``bplanes``; sparse
    planes are CSR-style in ``(sptr, sinds, svals)``. ``offsets`` holds the
    (unrounded) plane offsets used for margins during construction.
    """
    n_slots = pairs.shape[0]
    offsets = np.zeros(n_slots, dtype=np.float64)
    if kind == KIND_DENSE:
        dplanes = np.zeros((n_slots, ddata.shape[1]), dtype=np.float32)
    else:
        dplanes = np.zeros((1, 1), dtype=np.float32)
    if kind == KIND_BIT:
        bplanes = np.zeros((n_slots, 2 * bdata.shape[1]), dtype=np.uint8)
    else:
        bplanes = np.zeros((1, 1), dtype=np.uint8)
    sptr = np.zeros(n_slots + 1, dtype=np.int64)

    if kind == KIND_DENSE:
        for s in numba.prange(n_slots):
            if pairs[s, 0] >= 0:
                if angular:
                    offsets[s] = _dense_angular_plane(
                        ddata, pairs[s, 0], pairs[s, 1], dplanes[s]
                    )
                else:
                    offsets[s] = _dense_euclidean_plane(
                        ddata, pairs[s, 0], pairs[s, 1], dplanes[s]
                    )
        sinds = np.zeros(0, dtype=np.int32)
        svals = np.zeros(0, dtype=np.float32)
    elif kind == KIND_BIT:
        for s in numba.prange(n_slots):
            if pairs[s, 0] >= 0:
                _bit_plane(bdata, pairs[s, 0], pairs[s, 1], bplanes[s])
        sinds = np.zeros(0, dtype=np.int32)
        svals = np.zeros(0, dtype=np.float32)
    else:
        sizes = np.zeros(n_slots, dtype=np.int64)
        for s in numba.prange(n_slots):
            if pairs[s, 0] >= 0:
                p_inds, p_data, p_off = _sparse_plane(
                    inds, indptr, spdata, pairs[s, 0], pairs[s, 1], angular
                )
                sizes[s] = p_inds.shape[0]
                offsets[s] = p_off
        for s in range(n_slots):
            sptr[s + 1] = sptr[s] + sizes[s]
        sinds = np.empty(sptr[n_slots], dtype=np.int32)
        svals = np.empty(sptr[n_slots], dtype=np.float32)
        for s in numba.prange(n_slots):
            if pairs[s, 0] >= 0:
                p_inds, p_data, p_off = _sparse_plane(
                    inds, indptr, spdata, pairs[s, 0], pairs[s, 1], angular
                )
                for j in range(p_inds.shape[0]):
                    sinds[sptr[s] + j] = p_inds[j]
                    svals[sptr[s] + j] = p_data[j]
    return dplanes, sptr, sinds, svals, bplanes, offsets


@numba.njit(fastmath=True, inline="always")
def _margin(
    kind, angular, ddata, inds, indptr, spdata, bdata,
    dplanes, sptr, sinds, svals, bplanes, offsets, slot, point,
):
    if kind == KIND_DENSE:
        if angular:
            return np.float64(_dense_angular_margin(ddata, dplanes[slot], point))
        return np.float64(
            _dense_euclidean_margin(ddata, dplanes[slot], offsets[slot], point)
        )
    elif kind == KIND_SPARSE:
        return _sparse_margin(
            inds, indptr, spdata,
            sinds[sptr[slot] : sptr[slot + 1]],
            svals[sptr[slot] : sptr[slot + 1]],
            offsets[slot],
            point,
        )
    else:
        return np.float64(_bit_margin(bdata, bplanes[slot], point))


# ------------------------------------------------------------- generic decisions


@numba.njit(inline="always")
def _side(margins, c, idx, rank, shift):
    m = margins[c, idx] - shift
    if m > EPS:
        return 0
    elif m < -EPS:
        return 1
    return rank % 2


@numba.njit(inline="always")
def _random_side(seed, point):
    return np.int64(_mix64(seed ^ _U(point)) & _U(1))


@numba.njit(inline="always")
def _node_seed(base_seed, start, end):
    return _mix64(_U(base_seed) ^ _mix64(_U(start) * _U(2) + _U(1)) ^ _U(end))


@numba.njit(inline="always")
def _final_side(action, margins, c, idx, rank, shift, seed, point, flip0):
    if action == ACTION_SPLIT:
        side = _side(margins, c, idx, rank, shift)
    else:
        side = _random_side(seed, point)
    if flip0 and rank == 0:
        side = 1 - side
    return side


@numba.njit()
def _choose(n_left, cuts, n_cand, n_points, criterion, min_balance):
    """Best candidate by balance or by edge cuts; returns (candidate, balance)."""
    best = -1
    best_balance = np.float32(0.0)
    best_cuts = np.int64(0xFFFFFFFF)
    for c in range(n_cand):
        nl = n_left[c]
        nr = n_points - nl
        if nl == 0 or nr == 0:
            continue
        balance = np.float32(min(nl, nr)) / np.float32(n_points)
        if criterion == CRITERION_BALANCE:
            if balance > best_balance:
                best = c
                best_balance = balance
        else:
            if balance < min_balance:
                continue
            c_cuts = cuts[c] // 2
            if c_cuts < best_cuts:
                best = c
                best_cuts = c_cuts
                best_balance = balance
    return best, best_balance


@numba.njit()
def _decide(
    margins, base, start, end, n_cand, n_left, cuts, perm,
    criterion, min_balance, fallback, seed,
):
    """Decide how to split one node.

    Returns (action, candidate, shift, n_left, flip0). ``margins[:, pos - base]``
    are the candidate margins of the point at permutation position ``pos``.
    """
    n_points = end - start
    best, balance = _choose(n_left, cuts, n_cand, n_points, criterion, min_balance)
    if best >= 0 and balance >= min_balance:
        return ACTION_SPLIT, best, 0.0, n_left[best], False

    if fallback == FALLBACK_LEAF:
        return ACTION_LEAF, -1, 0.0, 0, False

    if fallback == FALLBACK_MEDIAN and n_cand > 0:
        # Keep the most balanced hub direction but move the hyperplane to the
        # median margin, which splits the node evenly unless margins are tied.
        direction = 0
        top = -1
        for c in range(n_cand):
            nl = min(n_left[c], n_points - n_left[c])
            if nl > top:
                top = nl
                direction = c
        values = np.empty(n_points, dtype=np.float64)
        for i in range(n_points):
            values[i] = margins[direction, start - base + i]
        shift = np.median(values)
        nl = 0
        for i in range(n_points):
            nl += 1 - _side(margins, direction, start - base + i, i, shift)
        nr = n_points - nl
        if nl > 0 and nr > 0:
            if np.float32(min(nl, nr)) / np.float32(n_points) >= min_balance:
                return ACTION_SPLIT, direction, shift, nl, False
        return ACTION_LEAF, -1, 0.0, 0, False

    # Random split (the serial sparse/bit builders' fallback), never one-sided
    nl = 0
    for i in range(n_points):
        nl += 1 - _random_side(seed, perm[start + i])
    flip0 = False
    if nl == 0:
        nl = 1
        flip0 = True
    elif nl == n_points:
        nl = n_points - 1
        flip0 = True
    return ACTION_RANDOM, -1, 0.0, nl, flip0


@numba.njit(inline="always")
def _count_and_cut(
    margins, base, start, end, seg_start, seg_end, n_cand, perm, pos_of,
    neighbor_indices, need_cuts, n_left, cuts,
):
    for pos in range(seg_start, seg_end):
        idx = pos - base
        rank = pos - start
        for c in range(n_cand):
            n_left[c] += 1 - _side(margins, c, idx, rank, 0.0)
        if need_cuts:
            point = perm[pos]
            for j in range(neighbor_indices.shape[1]):
                neighbor = neighbor_indices[point, j]
                if neighbor < 0:
                    break
                q = pos_of[neighbor]
                if q >= start and q < end:
                    for c in range(n_cand):
                        if _side(margins, c, idx, rank, 0.0) != _side(
                            margins, c, q - base, q - start, 0.0
                        ):
                            cuts[c] += 1


# --------------------------------------------------------------- level kernels


@numba.njit(parallel=True, cache=True)
def _segment_top3(perm, degrees, seg_start, seg_end, top_deg, top_pos):
    for s in numba.prange(seg_start.shape[0]):
        for r in range(3):
            top_deg[s, r] = -1
            top_pos[s, r] = -1
        for pos in range(seg_start[s], seg_end[s]):
            deg = degrees[perm[pos]]
            if deg > top_deg[s, 2]:
                r = 2
                while r > 0 and deg > top_deg[s, r - 1]:
                    top_deg[s, r] = top_deg[s, r - 1]
                    top_pos[s, r] = top_pos[s, r - 1]
                    r -= 1
                top_deg[s, r] = deg
                top_pos[s, r] = pos


@numba.njit(parallel=True, cache=True)
def _node_pairs(perm, seg_ptr, top_deg, top_pos, node_size):
    """Merge segment top-3s (degree desc, position asc) into each node's pairs."""
    n_nodes = seg_ptr.shape[0] - 1
    pairs = np.full((n_nodes * 3, 2), -1, dtype=np.int64)
    n_cand = np.zeros(n_nodes, dtype=np.int64)
    for n in numba.prange(n_nodes):
        best_deg = np.full(3, -1, dtype=np.int64)
        best_pos = np.full(3, -1, dtype=np.int64)
        for s in range(seg_ptr[n], seg_ptr[n + 1]):
            for r in range(3):
                deg = top_deg[s, r]
                pos = top_pos[s, r]
                if pos < 0:
                    continue
                k = 3
                while k > 0 and (
                    best_pos[k - 1] < 0
                    or deg > best_deg[k - 1]
                    or (deg == best_deg[k - 1] and pos < best_pos[k - 1])
                ):
                    k -= 1
                if k < 3:
                    for j in range(2, k, -1):
                        best_deg[j] = best_deg[j - 1]
                        best_pos[j] = best_pos[j - 1]
                    best_deg[k] = deg
                    best_pos[k] = pos
        n_hubs = min(3, node_size[n])
        cand = n_hubs * (n_hubs - 1) // 2
        n_cand[n] = cand
        for c in range(cand):
            pairs[n * 3 + c, 0] = perm[best_pos[_PAIR_A[c]]]
            pairs[n * 3 + c, 1] = perm[best_pos[_PAIR_B[c]]]
    return pairs, n_cand


@numba.njit(parallel=True, fastmath=True, cache=True)
def _segment_margins(
    kind, angular, ddata, inds, indptr, spdata, bdata,
    dplanes, sptr, sinds, svals, bplanes, offsets,
    perm, seg_node, seg_start, seg_end, n_cand, margins,
):
    for s in numba.prange(seg_start.shape[0]):
        node = seg_node[s]
        for pos in range(seg_start[s], seg_end[s]):
            point = perm[pos]
            for c in range(n_cand[node]):
                margins[c, pos] = _margin(
                    kind, angular, ddata, inds, indptr, spdata, bdata,
                    dplanes, sptr, sinds, svals, bplanes, offsets,
                    node * 3 + c, point,
                )


@numba.njit(parallel=True, cache=True)
def _segment_counts(
    margins, perm, pos_of, neighbor_indices, need_cuts,
    seg_node, seg_start, seg_end, node_start, node_end, n_cand, seg_left, seg_cuts,
):
    for s in numba.prange(seg_start.shape[0]):
        node = seg_node[s]
        n_left = np.zeros(3, dtype=np.int64)
        cuts = np.zeros(3, dtype=np.int64)
        _count_and_cut(
            margins, 0, node_start[node], node_end[node], seg_start[s], seg_end[s],
            n_cand[node], perm, pos_of, neighbor_indices, need_cuts, n_left, cuts,
        )
        for c in range(3):
            seg_left[s, c] = n_left[c]
            seg_cuts[s, c] = cuts[c]


@numba.njit(parallel=True, cache=True)
def _decide_nodes(
    margins, perm, seg_ptr, seg_left, seg_cuts, node_start, node_end, n_cand,
    criterion, min_balance, fallback, base_seed,
):
    n_nodes = node_start.shape[0]
    action = np.zeros(n_nodes, dtype=np.int64)
    choice = np.zeros(n_nodes, dtype=np.int64)
    shift = np.zeros(n_nodes, dtype=np.float64)
    n_left_final = np.zeros(n_nodes, dtype=np.int64)
    flip0 = np.zeros(n_nodes, dtype=np.bool_)
    for n in numba.prange(n_nodes):
        n_left = np.zeros(3, dtype=np.int64)
        cuts = np.zeros(3, dtype=np.int64)
        for s in range(seg_ptr[n], seg_ptr[n + 1]):
            for c in range(3):
                n_left[c] += seg_left[s, c]
                cuts[c] += seg_cuts[s, c]
        seed = _node_seed(base_seed, node_start[n], node_end[n])
        a, c, sh, nl, f0 = _decide(
            margins, 0, node_start[n], node_end[n], n_cand[n], n_left, cuts, perm,
            criterion, min_balance, fallback, seed,
        )
        action[n] = a
        choice[n] = c
        shift[n] = sh
        n_left_final[n] = nl
        flip0[n] = f0
    return action, choice, shift, n_left_final, flip0


@numba.njit(parallel=True, cache=True)
def _segment_final_left(
    margins, perm, seg_node, seg_start, seg_end, node_start, node_end,
    action, choice, shift, flip0, base_seed, seg_final_left,
):
    for s in numba.prange(seg_start.shape[0]):
        node = seg_node[s]
        seg_final_left[s] = 0
        if action[node] == ACTION_LEAF:
            continue
        seed = _node_seed(base_seed, node_start[node], node_end[node])
        count = 0
        for pos in range(seg_start[s], seg_end[s]):
            count += 1 - _final_side(
                action[node], margins, choice[node], pos, pos - node_start[node],
                shift[node], seed, perm[pos], flip0[node],
            )
        seg_final_left[s] = count


@numba.njit(parallel=True, cache=True)
def _segment_scatter(
    margins, perm, perm_out, pos_of, seg_node, seg_start, seg_end,
    node_start, node_end, action, choice, shift, flip0, base_seed,
    seg_left_base, seg_right_base,
):
    for s in numba.prange(seg_start.shape[0]):
        node = seg_node[s]
        if action[node] == ACTION_LEAF:
            continue
        seed = _node_seed(base_seed, node_start[node], node_end[node])
        lpos = seg_left_base[s]
        rpos = seg_right_base[s]
        for pos in range(seg_start[s], seg_end[s]):
            point = perm[pos]
            side = _final_side(
                action[node], margins, choice[node], pos, pos - node_start[node],
                shift[node], seed, point, flip0[node],
            )
            if side == 0:
                perm_out[lpos] = point
                pos_of[point] = lpos
                lpos += 1
            else:
                perm_out[rpos] = point
                pos_of[point] = rpos
                rpos += 1


@numba.njit(parallel=True, cache=True)
def _segment_copy_back(perm, perm_out, seg_node, seg_start, seg_end, action):
    for s in numba.prange(seg_start.shape[0]):
        if action[seg_node[s]] != ACTION_LEAF:
            for pos in range(seg_start[s], seg_end[s]):
                perm[pos] = perm_out[pos]


# Sparse margins via a dense per-thread scatter of the node's planes: a margin then
# costs O(nnz(row)) instead of a merge over O(nnz(plane) + nnz(row)). The same
# float32 products are added in the same (ascending index) order as
# sparse_dot_accumulate, so margins are identical; these helpers deliberately do
# not use fastmath, matching sparse_dot_accumulate.

SPARSE_SCATTER_BUDGET_BYTES = 256 << 20


@numba.njit(cache=True)
def _load_planes(ws, sptr, sinds, svals, slot0, n_cand):
    for c in range(n_cand):
        for j in range(sptr[slot0 + c], sptr[slot0 + c + 1]):
            ws[c, sinds[j]] = svals[j]


@numba.njit(cache=True)
def _clear_planes(ws, sptr, sinds, slot0, n_cand):
    for c in range(n_cand):
        for j in range(sptr[slot0 + c], sptr[slot0 + c + 1]):
            ws[c, sinds[j]] = 0.0


@numba.njit(cache=True)
def _scattered_margins(
    ws, inds, indptr, spdata, offsets, slot0, n_cand, perm, pos_start, pos_end,
    margins, col0,
):
    for pos in range(pos_start, pos_end):
        point = perm[pos]
        row_start = indptr[point]
        row_end = indptr[point + 1]
        for c in range(n_cand):
            margin = offsets[slot0 + c]
            for j in range(row_start, row_end):
                margin += np.float32(ws[c, inds[j]] * spdata[j])
            margins[c, pos - col0] = margin


@numba.njit(parallel=True, cache=True)
def _sparse_segment_margins(
    inds, indptr, spdata, sptr, sinds, svals, offsets,
    perm, seg_node, seg_start, seg_end, n_cand, margins, workspace,
):
    n_threads = workspace.shape[0]
    n_segs = seg_start.shape[0]
    per_thread = (n_segs + n_threads - 1) // n_threads
    for t in numba.prange(n_threads):
        ws = workspace[t]
        current = -1
        for s in range(t * per_thread, min(n_segs, (t + 1) * per_thread)):
            node = seg_node[s]
            if node != current:
                if current >= 0:
                    _clear_planes(ws, sptr, sinds, current * 3, n_cand[current])
                _load_planes(ws, sptr, sinds, svals, node * 3, n_cand[node])
                current = node
            _scattered_margins(
                ws, inds, indptr, spdata, offsets, node * 3, n_cand[node], perm,
                seg_start[s], seg_end[s], margins, 0,
            )
        if current >= 0:
            _clear_planes(ws, sptr, sinds, current * 3, n_cand[current])


# ------------------------------------------------------------ local subtrees


@numba.njit(parallel=True, fastmath=True, cache=True)
def _local_subtrees(
    kind, angular, ddata, inds, indptr, spdata, bdata,
    perm, pos_of, degrees, neighbor_indices,
    d_rec, d_start, d_end, d_depth, d_next_rec,
    rec_leaf, rec_hub_l, rec_hub_r, rec_offset, rec_shift,
    rec_left, rec_right, rec_start, rec_end,
    leaf_size, max_depth, criterion, min_balance, fallback, base_seed,
    thread_ptr, thread_items, workspace, use_scatter,
):
    need_cuts = criterion == CRITERION_EDGE_CUT
    stack_size = max_depth + 4
    for thread in numba.prange(thread_ptr.shape[0] - 1):
      ws = workspace[thread]
      for item in range(thread_ptr[thread], thread_ptr[thread + 1]):
        t = thread_items[item]
        st_rec = np.empty(stack_size, dtype=np.int64)
        st_start = np.empty(stack_size, dtype=np.int64)
        st_end = np.empty(stack_size, dtype=np.int64)
        st_depth = np.empty(stack_size, dtype=np.int64)
        st_rec[0] = d_rec[t]
        st_start[0] = d_start[t]
        st_end[0] = d_end[t]
        st_depth[0] = d_depth[t]
        top = 1
        next_rec = d_next_rec[t]
        pairs = np.empty((3, 2), dtype=np.int64)
        while top > 0:
            top -= 1
            rec = st_rec[top]
            start = st_start[top]
            end = st_end[top]
            depth = st_depth[top]
            size = end - start
            rec_start[rec] = start
            rec_end[rec] = end
            if size <= leaf_size or depth >= max_depth:
                rec_leaf[rec] = 1
                continue

            hubs = get_top_k_hub_indices(perm[start:end], degrees, 3)
            n_hubs = hubs.shape[0]
            n_cand = n_hubs * (n_hubs - 1) // 2
            for c in range(3):
                if c < n_cand:
                    pairs[c, 0] = hubs[_PAIR_A[c]]
                    pairs[c, 1] = hubs[_PAIR_B[c]]
                else:
                    pairs[c, 0] = -1
                    pairs[c, 1] = -1
            dplanes, sptr, sinds, svals, bplanes, offsets = _build_planes_serial(
                kind, angular, ddata, inds, indptr, spdata, bdata, pairs
            )
            margins = np.empty((3, size), dtype=np.float64)
            if kind == KIND_SPARSE and use_scatter:
                _load_planes(ws, sptr, sinds, svals, 0, n_cand)
                _scattered_margins(
                    ws, inds, indptr, spdata, offsets, 0, n_cand, perm, start, end,
                    margins, start,
                )
                _clear_planes(ws, sptr, sinds, 0, n_cand)
            else:
                for pos in range(start, end):
                    point = perm[pos]
                    for c in range(n_cand):
                        margins[c, pos - start] = _margin(
                            kind, angular, ddata, inds, indptr, spdata, bdata,
                            dplanes, sptr, sinds, svals, bplanes, offsets, c, point,
                        )
            n_left = np.zeros(3, dtype=np.int64)
            cuts = np.zeros(3, dtype=np.int64)
            _count_and_cut(
                margins, start, start, end, start, end, n_cand, perm, pos_of,
                neighbor_indices, need_cuts, n_left, cuts,
            )
            seed = _node_seed(base_seed, start, end)
            action, c, shift, nl, flip0 = _decide(
                margins, start, start, end, n_cand, n_left, cuts, perm,
                criterion, min_balance, fallback, seed,
            )
            if action == ACTION_LEAF:
                rec_leaf[rec] = 1
                continue

            # Stable partition of [start, end) into left then right
            tmp = perm[start:end].copy()
            lpos = start
            rpos = start + nl
            for i in range(size):
                point = tmp[i]
                side = _final_side(
                    action, margins, c, i, i, shift, seed, point, flip0
                )
                if side == 0:
                    perm[lpos] = point
                    pos_of[point] = lpos
                    lpos += 1
                else:
                    perm[rpos] = point
                    pos_of[point] = rpos
                    rpos += 1

            if action == ACTION_SPLIT:
                rec_hub_l[rec] = pairs[c, 0]
                rec_hub_r[rec] = pairs[c, 1]
                rec_offset[rec] = offsets[c]
                rec_shift[rec] = shift
            left = next_rec
            right = next_rec + 1
            next_rec += 2
            rec_left[rec] = left
            rec_right[rec] = right
            st_rec[top] = right
            st_start[top] = start + nl
            st_end[top] = end
            st_depth[top] = depth + 1
            top += 1
            st_rec[top] = left
            st_start[top] = start
            st_end[top] = start + nl
            st_depth[top] = depth + 1
            top += 1


@numba.njit(fastmath=True, cache=True)
def _build_planes_serial(kind, angular, ddata, inds, indptr, spdata, bdata, pairs):
    n_slots = pairs.shape[0]
    offsets = np.zeros(n_slots, dtype=np.float64)
    if kind == KIND_DENSE:
        dplanes = np.zeros((n_slots, ddata.shape[1]), dtype=np.float32)
    else:
        dplanes = np.zeros((1, 1), dtype=np.float32)
    if kind == KIND_BIT:
        bplanes = np.zeros((n_slots, 2 * bdata.shape[1]), dtype=np.uint8)
    else:
        bplanes = np.zeros((1, 1), dtype=np.uint8)
    sptr = np.zeros(n_slots + 1, dtype=np.int64)
    sinds = np.zeros(0, dtype=np.int32)
    svals = np.zeros(0, dtype=np.float32)
    if kind == KIND_DENSE:
        for s in range(n_slots):
            if pairs[s, 0] >= 0:
                if angular:
                    offsets[s] = _dense_angular_plane(
                        ddata, pairs[s, 0], pairs[s, 1], dplanes[s]
                    )
                else:
                    offsets[s] = _dense_euclidean_plane(
                        ddata, pairs[s, 0], pairs[s, 1], dplanes[s]
                    )
    elif kind == KIND_BIT:
        for s in range(n_slots):
            if pairs[s, 0] >= 0:
                _bit_plane(bdata, pairs[s, 0], pairs[s, 1], bplanes[s])
    else:
        total = 0
        for s in range(n_slots):
            if pairs[s, 0] >= 0:
                p_inds, p_data, p_off = _sparse_plane(
                    inds, indptr, spdata, pairs[s, 0], pairs[s, 1], angular
                )
                total += p_inds.shape[0]
        sinds = np.empty(total, dtype=np.int32)
        svals = np.empty(total, dtype=np.float32)
        for s in range(n_slots):
            sptr[s + 1] = sptr[s]
            if pairs[s, 0] >= 0:
                p_inds, p_data, p_off = _sparse_plane(
                    inds, indptr, spdata, pairs[s, 0], pairs[s, 1], angular
                )
                offsets[s] = p_off
                for j in range(p_inds.shape[0]):
                    sinds[sptr[s] + j] = p_inds[j]
                    svals[sptr[s] + j] = p_data[j]
                sptr[s + 1] += p_inds.shape[0]
    return dplanes, sptr, sinds, svals, bplanes, offsets


# ---------------------------------------------------------------- finalising


@numba.njit(cache=True)
def _preorder(rec_leaf, rec_left, rec_right, n_records):
    """Pre-order ids (left first), matching rp_trees.convert_tree_format."""
    pre_of = np.full(n_records, -1, dtype=np.int64)
    order = np.empty(n_records, dtype=np.int64)
    stack = np.empty(n_records + 1, dtype=np.int64)
    stack[0] = 0
    top = 1
    count = 0
    while top > 0:
        top -= 1
        rec = stack[top]
        pre_of[rec] = count
        order[count] = rec
        count += 1
        if not rec_leaf[rec]:
            stack[top] = rec_right[rec]
            top += 1
            stack[top] = rec_left[rec]
            top += 1
    return pre_of, order[:count]


@numba.njit(parallel=True, cache=True)
def _flat_children(order, pre_of, rec_leaf, rec_left, rec_right, rec_start, rec_end):
    n_nodes = order.shape[0]
    children = np.empty((n_nodes, 2), dtype=np.int32)
    for p in numba.prange(n_nodes):
        rec = order[p]
        if rec_leaf[rec]:
            children[p, 0] = -rec_start[rec]
            children[p, 1] = -rec_end[rec]
        else:
            children[p, 0] = pre_of[rec_left[rec]]
            children[p, 1] = pre_of[rec_right[rec]]
    return children


def make_hub_search_tree(
    data,
    neighbor_indices,
    leaf_size=30,
    max_depth=200,
    angular=False,
    seed=0,
    criterion=None,
    min_balance=None,
    fallback=None,
    local_bytes=LOCAL_SUBTREE_BYTES,
    segment_size=SEGMENT_SIZE,
):
    """Build the hub-based search tree in parallel and return it as a flat tree.

    Parameters
    ----------
    data: array of shape (n_samples, n_features), CSR matrix, or uint8 array
        Dense float32 data, a CSR matrix with sorted indices, or bit-packed uint8
        data.
    neighbor_indices: array of shape (n_samples, n_neighbors)
        The k-NN graph (int32, -1 padded).
    leaf_size, max_depth, angular:
        As for the serial hub tree builders.
    seed: int
        Seed for the per-node random fallback split.
    criterion, min_balance, fallback:
        How splits are chosen and what happens when no acceptable split exists.
        Defaults come from ``DEFAULT_SPLIT_OPTIONS`` for the data kind.
    local_bytes: int
        Nodes whose rows fit in this many bytes are finished depth-first.
    segment_size: int
        Number of positions per unit of parallel work in large nodes.

    Returns
    -------
    tree: FlatTree
        In the format produced by :func:`pynndescent.rp_trees.convert_tree_format`.
    """
    if hasattr(data, "indptr"):
        kind = KIND_SPARSE
        n = data.shape[0]
        inds = data.indices
        indptr = data.indptr
        spdata = data.data
        ddata = np.zeros((1, 1), dtype=np.float32)
        bdata = np.zeros((1, 1), dtype=np.uint8)
        row_bytes = max(1, int(8 * data.nnz / max(n, 1)))
    elif data.dtype == np.uint8:
        kind = KIND_BIT
        n = data.shape[0]
        bdata = np.ascontiguousarray(data)
        ddata = np.zeros((1, 1), dtype=np.float32)
        inds = np.zeros(1, dtype=np.int32)
        indptr = np.zeros(2, dtype=np.int32)
        spdata = np.zeros(1, dtype=np.float32)
        row_bytes = max(1, bdata.shape[1])
        angular = False
    else:
        kind = KIND_DENSE
        n = data.shape[0]
        ddata = np.ascontiguousarray(data, dtype=np.float32)
        bdata = np.zeros((1, 1), dtype=np.uint8)
        inds = np.zeros(1, dtype=np.int32)
        indptr = np.zeros(2, dtype=np.int32)
        spdata = np.zeros(1, dtype=np.float32)
        row_bytes = 4 * ddata.shape[1]

    default_criterion, default_min_balance, default_fallback = DEFAULT_SPLIT_OPTIONS[
        kind
    ]
    if criterion is None:
        criterion = default_criterion
    if min_balance is None:
        min_balance = default_min_balance
    if fallback is None:
        fallback = default_fallback
    min_balance = float(min_balance)
    need_cuts = criterion == CRITERION_EDGE_CUT
    base_seed = np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
    local_points = max(leaf_size, local_bytes // row_bytes)

    neighbor_indices = np.ascontiguousarray(neighbor_indices, dtype=np.int32)
    degrees = compute_global_degrees(neighbor_indices)

    n_threads = numba.get_num_threads()
    use_scatter = (
        kind == KIND_SPARSE
        and n_threads * 3 * data.shape[1] * 4 <= SPARSE_SCATTER_BUDGET_BYTES
    )
    if use_scatter:
        workspace = np.zeros((n_threads, 3, data.shape[1]), dtype=np.float32)
    else:
        workspace = np.zeros((n_threads, 1, 1), dtype=np.float32)

    perm = np.arange(n, dtype=np.int32)
    pos_of = np.arange(n, dtype=np.int32)
    capacity = 2 * n + 2
    rec_leaf = np.zeros(capacity, dtype=np.uint8)
    rec_hub_l = np.full(capacity, -1, dtype=np.int64)
    rec_hub_r = np.full(capacity, -1, dtype=np.int64)
    rec_offset = np.zeros(capacity, dtype=np.float64)
    rec_shift = np.zeros(capacity, dtype=np.float64)
    rec_left = np.full(capacity, -1, dtype=np.int64)
    rec_right = np.full(capacity, -1, dtype=np.int64)
    rec_start = np.zeros(capacity, dtype=np.int64)
    rec_end = np.zeros(capacity, dtype=np.int64)
    n_records = 1

    f_rec = np.zeros(1, dtype=np.int64)
    f_start = np.zeros(1, dtype=np.int64)
    f_end = np.full(1, n, dtype=np.int64)
    f_depth = np.zeros(1, dtype=np.int64)
    deferred = []
    margins = None
    perm_out = None

    while f_rec.shape[0] > 0:
        rec_start[f_rec] = f_start
        rec_end[f_rec] = f_end
        size = f_end - f_start
        must_split = (size > leaf_size) & (f_depth < max_depth)
        rec_leaf[f_rec[~must_split]] = 1
        local = must_split & (size <= local_points)
        if local.any():
            deferred.append((f_rec[local], f_start[local], f_end[local], f_depth[local]))
        level = must_split & ~local
        if not level.any():
            break

        l_rec = f_rec[level]
        l_start = f_start[level]
        l_end = f_end[level]
        l_depth = f_depth[level]
        l_size = l_end - l_start
        if margins is None:
            margins = np.empty((3, n), dtype=np.float64)
            perm_out = np.empty(n, dtype=np.int32)

        n_segs = (l_size + segment_size - 1) // segment_size
        seg_ptr = np.zeros(l_rec.shape[0] + 1, dtype=np.int64)
        np.cumsum(n_segs, out=seg_ptr[1:])
        seg_node = np.repeat(np.arange(l_rec.shape[0], dtype=np.int64), n_segs)
        within = np.arange(seg_ptr[-1], dtype=np.int64) - seg_ptr[seg_node]
        seg_start = l_start[seg_node] + within * segment_size
        seg_end = np.minimum(seg_start + segment_size, l_end[seg_node])
        n_seg_total = seg_start.shape[0]

        top_deg = np.empty((n_seg_total, 3), dtype=np.int64)
        top_pos = np.empty((n_seg_total, 3), dtype=np.int64)
        _segment_top3(perm, degrees, seg_start, seg_end, top_deg, top_pos)
        pairs, n_cand = _node_pairs(perm, seg_ptr, top_deg, top_pos, l_size)
        dplanes, sptr, sinds, svals, bplanes, offsets = _build_planes(
            kind, angular, ddata, inds, indptr, spdata, bdata, pairs
        )
        if use_scatter:
            _sparse_segment_margins(
                inds, indptr, spdata, sptr, sinds, svals, offsets,
                perm, seg_node, seg_start, seg_end, n_cand, margins, workspace,
            )
        else:
            _segment_margins(
                kind, angular, ddata, inds, indptr, spdata, bdata,
                dplanes, sptr, sinds, svals, bplanes, offsets,
                perm, seg_node, seg_start, seg_end, n_cand, margins,
            )
        seg_left = np.empty((n_seg_total, 3), dtype=np.int64)
        seg_cuts = np.empty((n_seg_total, 3), dtype=np.int64)
        _segment_counts(
            margins, perm, pos_of, neighbor_indices, need_cuts,
            seg_node, seg_start, seg_end, l_start, l_end, n_cand, seg_left, seg_cuts,
        )
        action, choice, shift, n_left, flip0 = _decide_nodes(
            margins, perm, seg_ptr, seg_left, seg_cuts, l_start, l_end, n_cand,
            criterion, min_balance, fallback, base_seed,
        )
        seg_final_left = np.empty(n_seg_total, dtype=np.int64)
        _segment_final_left(
            margins, perm, seg_node, seg_start, seg_end, l_start, l_end,
            action, choice, shift, flip0, base_seed, seg_final_left,
        )
        # Per-node exclusive prefix of the segments' left/right counts
        seg_size = seg_end - seg_start
        left_before = np.cumsum(seg_final_left) - seg_final_left
        left_before -= left_before[seg_ptr[seg_node]]
        right_before = np.cumsum(seg_size - seg_final_left) - (seg_size - seg_final_left)
        right_before -= right_before[seg_ptr[seg_node]]
        seg_left_base = l_start[seg_node] + left_before
        seg_right_base = l_start[seg_node] + n_left[seg_node] + right_before
        _segment_scatter(
            margins, perm, perm_out, pos_of, seg_node, seg_start, seg_end,
            l_start, l_end, action, choice, shift, flip0, base_seed,
            seg_left_base, seg_right_base,
        )
        _segment_copy_back(perm, perm_out, seg_node, seg_start, seg_end, action)

        is_leaf = action == ACTION_LEAF
        rec_leaf[l_rec[is_leaf]] = 1
        split = ~is_leaf
        s_rec = l_rec[split]
        n_split = s_rec.shape[0]
        planned = split & (action == ACTION_SPLIT)
        slots = np.arange(l_rec.shape[0]) * 3 + choice
        rec_hub_l[l_rec[planned]] = pairs[slots[planned], 0]
        rec_hub_r[l_rec[planned]] = pairs[slots[planned], 1]
        rec_offset[l_rec[planned]] = offsets[slots[planned]]
        rec_shift[l_rec[planned]] = shift[planned]
        left_ids = n_records + 2 * np.arange(n_split, dtype=np.int64)
        rec_left[s_rec] = left_ids
        rec_right[s_rec] = left_ids + 1
        n_records += 2 * n_split

        mid = l_start[split] + n_left[split]
        f_rec = np.concatenate([left_ids, left_ids + 1])
        f_start = np.concatenate([l_start[split], mid])
        f_end = np.concatenate([mid, l_end[split]])
        f_depth = np.concatenate([l_depth[split] + 1, l_depth[split] + 1])

    if deferred:
        d_rec, d_start, d_end, d_depth = [np.concatenate(c) for c in zip(*deferred)]
        # A subtree of m points has at most 2m - 1 nodes (no empty leaves)
        reserve = 2 * (d_end - d_start) - 2
        d_next_rec = n_records + np.cumsum(reserve) - reserve
        n_records += int(reserve.sum())
        thread_ptr, thread_items = _assign_to_threads(d_end - d_start, n_threads)
        _local_subtrees(
            kind, angular, ddata, inds, indptr, spdata, bdata,
            perm, pos_of, degrees, neighbor_indices,
            d_rec, d_start, d_end, d_depth, d_next_rec,
            rec_leaf, rec_hub_l, rec_hub_r, rec_offset, rec_shift,
            rec_left, rec_right, rec_start, rec_end,
            leaf_size, max_depth, criterion, min_balance, fallback, base_seed,
            thread_ptr, thread_items, workspace, use_scatter,
        )

    pre_of, order = _preorder(rec_leaf, rec_left, rec_right, n_records)
    children = _flat_children(
        order, pre_of, rec_leaf, rec_left, rec_right, rec_start, rec_end
    )
    return _materialise(
        kind, angular, ddata, inds, indptr, spdata, bdata, order, rec_leaf,
        rec_hub_l, rec_hub_r, rec_offset, rec_shift, children, perm, leaf_size,
        rec_start, rec_end,
    )


def _assign_to_threads(sizes, n_threads):
    """Deal subtrees to threads largest first, alternating direction (snake order)."""
    order = np.argsort(-sizes, kind="stable")
    rank = np.arange(order.shape[0])
    lap, offset = np.divmod(rank, n_threads)
    thread = np.where(lap % 2 == 0, offset, n_threads - 1 - offset)
    by_thread = np.argsort(thread, kind="stable")
    thread_items = order[by_thread].astype(np.int64)
    counts = np.bincount(thread, minlength=n_threads)
    thread_ptr = np.zeros(n_threads + 1, dtype=np.int64)
    np.cumsum(counts, out=thread_ptr[1:])
    return thread_ptr, thread_items


def _materialise(
    kind, angular, ddata, inds, indptr, spdata, bdata, order, rec_leaf,
    rec_hub_l, rec_hub_r, rec_offset, rec_shift, children, perm, leaf_size,
    rec_start, rec_end,
):
    n_nodes = order.shape[0]
    internal = rec_leaf[order] == 0
    internal_pre = np.flatnonzero(internal)
    internal_rec = order[internal_pre]
    pairs = np.stack([rec_hub_l[internal_rec], rec_hub_r[internal_rec]], axis=1)
    dplanes, sptr, sinds, svals, bplanes, _ = _build_planes(
        kind, angular, ddata, inds, indptr, spdata, bdata, pairs
    )
    offsets = np.zeros(n_nodes, dtype=np.float32)
    offsets[internal_pre] = (
        rec_offset[internal_rec] - rec_shift[internal_rec]
    ).astype(np.float32)

    if kind == KIND_DENSE:
        hyperplanes = np.zeros((n_nodes, ddata.shape[1]), dtype=np.float32)
        hyperplanes[internal_pre] = dplanes
    elif kind == KIND_BIT:
        hyperplanes = np.zeros((n_nodes, 2 * bdata.shape[1]), dtype=np.uint8)
        hyperplanes[internal_pre] = bplanes
    else:
        nnz = np.diff(sptr)
        width = max(1, int(nnz.max()) if nnz.shape[0] > 0 else 1)
        hyperplanes = np.zeros((n_nodes, 2, width), dtype=np.float32)
        hyperplanes[:, 0, :] = -1
        _fill_sparse_planes(hyperplanes, internal_pre, pairs, sptr, sinds, svals)

    leaf_sizes = rec_end[order[~internal]] - rec_start[order[~internal]]
    max_leaf = max(leaf_size, int(leaf_sizes.max()) if leaf_sizes.shape[0] else 0)
    return FlatTree(hyperplanes, offsets, children, perm.copy(), max_leaf)


@numba.njit(parallel=True, cache=True)
def _fill_sparse_planes(hyperplanes, internal_pre, pairs, sptr, sinds, svals):
    for k in numba.prange(internal_pre.shape[0]):
        p = internal_pre[k]
        if pairs[k, 0] < 0:
            # Random split: the serial builders store a [[-1], [-1]] hyperplane
            hyperplanes[p, 0, 0] = -1.0
            hyperplanes[p, 1, 0] = -1.0
            continue
        for j in range(sptr[k + 1] - sptr[k]):
            hyperplanes[p, 0, j] = sinds[sptr[k] + j]
            hyperplanes[p, 1, j] = svals[sptr[k] + j]
