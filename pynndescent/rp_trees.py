# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 2 clause
from warnings import warn

import numpy as np
import numba
import scipy.sparse

from pynndescent.sparse import (
    sparse_mul,
    sparse_diff,
    sparse_sum,
    arr_intersect,
    sparse_dot_product,
)
from pynndescent.utils import tau_rand_int, norm
import joblib

from collections import namedtuple

# Used for a floating point "nearly zero" comparison
EPS = 1e-8
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

FlatTree = namedtuple(
    "FlatTree", ["hyperplanes", "offsets", "children", "indices", "leaf_size"]
)

dense_hyperplane_type = numba.float32[::1]
sparse_hyperplane_type = numba.float64[:, ::1]
bit_hyperplane_type = numba.uint8[::1]
offset_type = numba.float64
children_type = numba.typeof((np.int32(-1), np.int32(-1)))
point_indices_type = numba.int32[::1]

popcnt = np.array([bin(i).count("1") for i in range(256)], dtype=np.float32)


@numba.njit(
    numba.types.Tuple(
        (numba.int32[::1], numba.int32[::1], dense_hyperplane_type, offset_type)
    )(numba.float32[:, ::1], numba.int32[::1], numba.int64[::1]),
    locals={
        "n_left": numba.uint32,
        "n_right": numba.uint32,
        "hyperplane_vector": numba.float32[::1],
        "hyperplane_offset": numba.float32,
        "margin": numba.float32,
        "d": numba.uint32,
        "i": numba.uint32,
        "left_index": numba.uint32,
        "right_index": numba.uint32,
    },
    fastmath=True,
    nogil=True,
    cache=True,
)
def angular_random_projection_split(data, indices, rng_state):
    """Given a set of ``graph_indices`` for graph_data points from ``graph_data``, create
    a random hyperplane to split the graph_data, returning two arrays graph_indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.
    This particular split uses cosine distance to determine the hyperplane
    and which side each graph_data sample falls on.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original graph_data to be split
    indices: array of shape (tree_node_size,)
        The graph_indices of the elements in the ``graph_data`` array that are to
        be split in the current operation.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    indices_left: array
        The elements of ``graph_indices`` that fall on the "left" side of the
        random hyperplane.
    indices_right: array
        The elements of ``graph_indices`` that fall on the "left" side of the
        random hyperplane.
    """
    dim = data.shape[1]

    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_norm = norm(data[left])
    right_norm = norm(data[right])

    if abs(left_norm) < EPS:
        left_norm = 1.0

    if abs(right_norm) < EPS:
        right_norm = 1.0

    # Compute the normal vector to the hyperplane (the vector between
    # the two points)
    hyperplane_vector = np.empty(dim, dtype=np.float32)

    for d in range(dim):
        hyperplane_vector[d] = (data[left, d] / left_norm) - (
            data[right, d] / right_norm
        )

    hyperplane_norm = norm(hyperplane_vector)
    if abs(hyperplane_norm) < EPS:
        hyperplane_norm = 1.0

    for d in range(dim):
        hyperplane_vector[d] = hyperplane_vector[d] / hyperplane_norm

    # For each point compute the margin (project into normal vector)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = 0.0
        for d in range(dim):
            margin += hyperplane_vector[d] * data[indices[i], d]

        if abs(margin) < EPS:
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # If all points end up on one side, something went wrong numerically
    # In this case, assign points randomly; they are likely very close anyway
    if n_left == 0 or n_right == 0:
        n_left = 0
        n_right = 0
        for i in range(indices.shape[0]):
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int32)
    indices_right = np.empty(n_right, dtype=np.int32)

    # Populate the arrays with graph_indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right, hyperplane_vector, 0.0


@numba.njit(
    numba.types.Tuple(
        (numba.int32[::1], numba.int32[::1], bit_hyperplane_type, offset_type)
    )(numba.uint8[:, ::1], numba.int32[::1], numba.int64[::1]),
    locals={
        "n_left": numba.uint32,
        "n_right": numba.uint32,
        "hyperplane_vector": numba.uint8[::1],
        "hyperplane_offset": numba.float32,
        "margin": numba.float32,
        "d": numba.uint32,
        "i": numba.uint32,
        "left_index": numba.uint32,
        "right_index": numba.uint32,
    },
    fastmath=True,
    nogil=True,
    cache=True,
)
def angular_bitpacked_random_projection_split(data, indices, rng_state):
    """Given a set of ``graph_indices`` for graph_data points from ``graph_data``, create
    a random hyperplane to split the graph_data, returning two arrays graph_indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.
    This particular split uses cosine distance to determine the hyperplane
    and which side each graph_data sample falls on.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original graph_data to be split
    indices: array of shape (tree_node_size,)
        The graph_indices of the elements in the ``graph_data`` array that are to
        be split in the current operation.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    indices_left: array
        The elements of ``graph_indices`` that fall on the "left" side of the
        random hyperplane.
    indices_right: array
        The elements of ``graph_indices`` that fall on the "left" side of the
        random hyperplane.
    """
    dim = data.shape[1]

    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_norm = 0.0
    right_norm = 0.0

    # Compute the normal vector to the hyperplane (the vector between
    # the two points)
    hyperplane_vector = np.empty(dim * 2, dtype=np.uint8)
    positive_hyperplane_component = hyperplane_vector[:dim]
    negative_hyperplane_component = hyperplane_vector[dim:]

    for d in range(dim):
        xor_vector = (data[left, d]) ^ (data[right, d])
        positive_hyperplane_component[d] = xor_vector & (data[left, d])
        negative_hyperplane_component[d] = xor_vector & (data[right, d])

    hyperplane_norm = 0.0

    for d in range(dim):
        hyperplane_norm += popcnt[hyperplane_vector[d]]
        left_norm += popcnt[data[left, d]]
        right_norm += popcnt[data[right, d]]

    # For each point compute the margin (project into normal vector)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = 0.0
        for d in range(dim):
            margin += popcnt[positive_hyperplane_component[d] & data[indices[i], d]]
            margin -= popcnt[negative_hyperplane_component[d] & data[indices[i], d]]

        if abs(margin) < EPS:
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # If all points end up on one side, something went wrong numerically
    # In this case, assign points randomly; they are likely very close anyway
    if n_left == 0 or n_right == 0:
        n_left = 0
        n_right = 0
        for i in range(indices.shape[0]):
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int32)
    indices_right = np.empty(n_right, dtype=np.int32)

    # Populate the arrays with graph_indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right, hyperplane_vector, 0.0


@numba.njit(
    numba.types.Tuple(
        (numba.int32[::1], numba.int32[::1], dense_hyperplane_type, offset_type)
    )(numba.float32[:, ::1], numba.int32[::1], numba.int64[::1]),
    locals={
        "n_left": numba.uint32,
        "n_right": numba.uint32,
        "hyperplane_vector": numba.float32[::1],
        "hyperplane_offset": numba.float32,
        "margin": numba.float32,
        "d": numba.uint32,
        "i": numba.uint32,
        "left_index": numba.uint32,
        "right_index": numba.uint32,
    },
    fastmath=True,
    nogil=True,
    cache=True,
)
def euclidean_random_projection_split(data, indices, rng_state):
    """Given a set of ``graph_indices`` for graph_data points from ``graph_data``, create
    a random hyperplane to split the graph_data, returning two arrays graph_indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.
    This particular split uses euclidean distance to determine the hyperplane
    and which side each graph_data sample falls on.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original graph_data to be split
    indices: array of shape (tree_node_size,)
        The graph_indices of the elements in the ``graph_data`` array that are to
        be split in the current operation.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    indices_left: array
        The elements of ``graph_indices`` that fall on the "left" side of the
        random hyperplane.
    indices_right: array
        The elements of ``graph_indices`` that fall on the "left" side of the
        random hyperplane.
    """
    dim = data.shape[1]

    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    # Compute the normal vector to the hyperplane (the vector between
    # the two points) and the offset from the origin
    hyperplane_offset = 0.0
    hyperplane_vector = np.empty(dim, dtype=np.float32)

    for d in range(dim):
        hyperplane_vector[d] = data[left, d] - data[right, d]
        hyperplane_offset -= (
            hyperplane_vector[d] * (data[left, d] + data[right, d]) / 2.0
        )

    # For each point compute the margin (project into normal vector, add offset)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = hyperplane_offset
        for d in range(dim):
            margin += hyperplane_vector[d] * data[indices[i], d]

        if abs(margin) < EPS:
            side[i] = abs(tau_rand_int(rng_state)) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # If all points end up on one side, something went wrong numerically
    # In this case, assign points randomly; they are likely very close anyway
    if n_left == 0 or n_right == 0:
        n_left = 0
        n_right = 0
        for i in range(indices.shape[0]):
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int32)
    indices_right = np.empty(n_right, dtype=np.int32)

    # Populate the arrays with graph_indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right, hyperplane_vector, hyperplane_offset


@numba.njit(
    fastmath=True,
    nogil=True,
    cache=True,
    locals={
        "normalized_left_data": numba.types.float32[::1],
        "normalized_right_data": numba.types.float32[::1],
        "hyperplane_norm": numba.types.float32,
        "i": numba.types.uint32,
    },
)
def sparse_angular_random_projection_split(inds, indptr, data, indices, rng_state):
    """Given a set of ``graph_indices`` for graph_data points from a sparse graph_data set
    presented in csr sparse format as inds, graph_indptr and graph_data, create
    a random hyperplane to split the graph_data, returning two arrays graph_indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.
    This particular split uses cosine distance to determine the hyperplane
    and which side each graph_data sample falls on.
    Parameters
    ----------
    inds: array
        CSR format index array of the matrix
    indptr: array
        CSR format index pointer array of the matrix
    data: array
        CSR format graph_data array of the matrix
    indices: array of shape (tree_node_size,)
        The graph_indices of the elements in the ``graph_data`` array that are to
        be split in the current operation.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    indices_left: array
        The elements of ``graph_indices`` that fall on the "left" side of the
        random hyperplane.
    indices_right: array
        The elements of ``graph_indices`` that fall on the "left" side of the
        random hyperplane.
    """
    # Select two random points, set the hyperplane between them
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_inds = inds[indptr[left] : indptr[left + 1]]
    left_data = data[indptr[left] : indptr[left + 1]]
    right_inds = inds[indptr[right] : indptr[right + 1]]
    right_data = data[indptr[right] : indptr[right + 1]]

    left_norm = norm(left_data)
    right_norm = norm(right_data)

    if abs(left_norm) < EPS:
        left_norm = 1.0

    if abs(right_norm) < EPS:
        right_norm = 1.0

    # Compute the normal vector to the hyperplane (the vector between
    # the two points)
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

    # For each point compute the margin (project into normal vector)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = 0.0

        i_inds = inds[indptr[indices[i]] : indptr[indices[i] + 1]]
        i_data = data[indptr[indices[i]] : indptr[indices[i] + 1]]

        _, mul_data = sparse_mul(hyperplane_inds, hyperplane_data, i_inds, i_data)
        for val in mul_data:
            margin += val

        if abs(margin) < EPS:
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # If all points end up on one side, something went wrong numerically
    # In this case, assign points randomly; they are likely very close anyway
    if n_left == 0 or n_right == 0:
        n_left = 0
        n_right = 0
        for i in range(indices.shape[0]):
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int32)
    indices_right = np.empty(n_right, dtype=np.int32)

    # Populate the arrays with graph_indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    hyperplane = np.vstack((hyperplane_inds, hyperplane_data))

    return indices_left, indices_right, hyperplane, 0.0


@numba.njit(fastmath=True, nogil=True, cache=True)
def sparse_euclidean_random_projection_split(inds, indptr, data, indices, rng_state):
    """Given a set of ``graph_indices`` for graph_data points from a sparse graph_data set
    presented in csr sparse format as inds, graph_indptr and graph_data, create
    a random hyperplane to split the graph_data, returning two arrays graph_indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.
    This particular split uses cosine distance to determine the hyperplane
    and which side each graph_data sample falls on.
    Parameters
    ----------
    inds: array
        CSR format index array of the matrix
    indptr: array
        CSR format index pointer array of the matrix
    data: array
        CSR format graph_data array of the matrix
    indices: array of shape (tree_node_size,)
        The graph_indices of the elements in the ``graph_data`` array that are to
        be split in the current operation.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    indices_left: array
        The elements of ``graph_indices`` that fall on the "left" side of the
        random hyperplane.
    indices_right: array
        The elements of ``graph_indices`` that fall on the "left" side of the
        random hyperplane.
    """
    # Select two random points, set the hyperplane between them
    left_index = np.abs(tau_rand_int(rng_state)) % indices.shape[0]
    right_index = np.abs(tau_rand_int(rng_state)) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_inds = inds[indptr[left] : indptr[left + 1]]
    left_data = data[indptr[left] : indptr[left + 1]]
    right_inds = inds[indptr[right] : indptr[right + 1]]
    right_data = data[indptr[right] : indptr[right + 1]]

    # Compute the normal vector to the hyperplane (the vector between
    # the two points) and the offset from the origin
    hyperplane_offset = 0.0
    hyperplane_inds, hyperplane_data = sparse_diff(
        left_inds, left_data, right_inds, right_data
    )
    offset_inds, offset_data = sparse_sum(left_inds, left_data, right_inds, right_data)
    offset_data = offset_data / 2.0
    offset_inds, offset_data = sparse_mul(
        hyperplane_inds, hyperplane_data, offset_inds, offset_data.astype(np.float32)
    )

    for val in offset_data:
        hyperplane_offset -= val

    # For each point compute the margin (project into normal vector, add offset)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.int8)
    for i in range(indices.shape[0]):
        margin = hyperplane_offset
        i_inds = inds[indptr[indices[i]] : indptr[indices[i] + 1]]
        i_data = data[indptr[indices[i]] : indptr[indices[i] + 1]]

        _, mul_data = sparse_mul(hyperplane_inds, hyperplane_data, i_inds, i_data)
        for val in mul_data:
            margin += val

        if abs(margin) < EPS:
            side[i] = abs(tau_rand_int(rng_state)) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # If all points end up on one side, something went wrong numerically
    # In this case, assign points randomly; they are likely very close anyway
    if n_left == 0 or n_right == 0:
        n_left = 0
        n_right = 0
        for i in range(indices.shape[0]):
            side[i] = abs(tau_rand_int(rng_state)) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int32)
    indices_right = np.empty(n_right, dtype=np.int32)

    # Populate the arrays with graph_indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    hyperplane = np.vstack((hyperplane_inds, hyperplane_data))

    return indices_left, indices_right, hyperplane, hyperplane_offset


# ============================================================================
# Graph-informed tree construction
# ============================================================================

# Threshold below which we skip edge-cut optimization and just use hub-based split
# This greatly speeds up tree construction for large datasets
FAST_SPLIT_THRESHOLD = 5000


@numba.njit(
    fastmath=True,
    nogil=True,
    cache=False,
)
def binary_search(sorted_arr, value):
    """Binary search returning index if found, -1 otherwise."""
    lo = 0
    hi = sorted_arr.shape[0] - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_arr[mid] == value:
            return mid
        elif sorted_arr[mid] < value:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


@numba.njit(
    fastmath=True,
    nogil=True,
    cache=False,
)
def compute_global_degrees(neighbor_indices):
    """Compute global in-degree for all points in the graph.

    In-degree of a point is how many times it appears as a neighbor of other points.
    This is computed once and reused throughout tree construction.

    Parameters
    ----------
    neighbor_indices : array of shape (n_samples, n_neighbors)
        The neighbor graph indices.

    Returns
    -------
    global_degrees : array of shape (n_samples,)
        The in-degree of each point.
    """
    n_points = neighbor_indices.shape[0]
    global_degrees = np.zeros(n_points, dtype=np.int32)

    for i in range(n_points):
        for j in range(neighbor_indices.shape[1]):
            neighbor = neighbor_indices[i, j]
            if neighbor >= 0 and neighbor < n_points:
                global_degrees[neighbor] += 1

    return global_degrees


@numba.njit(
    fastmath=True,
    nogil=True,
    cache=False,
)
def get_top_k_hub_indices(indices, global_degrees, k=5):
    """Get the indices of the top k highest-degree points from a subset.

    Uses an efficient O(n) selection for small k by maintaining a min-heap of k elements.

    Parameters
    ----------
    indices : array of shape (n,)
        The point indices in the current split.
    global_degrees : array of shape (n_total,)
        Precomputed global degrees for all points.
    k : int
        Number of top hubs to return.

    Returns
    -------
    top_hubs : array of shape (min(k, n),)
        The actual point indices (not positions) of the top k hubs.
    """
    n_points = indices.shape[0]
    actual_k = min(k, n_points)

    # For small k, use a simple insertion-sort-based approach which is O(n*k)
    # but with very low constants for small k
    top_degrees = np.full(actual_k, np.int32(-1), dtype=np.int32)
    top_indices = np.empty(actual_k, dtype=np.int32)

    for i in range(n_points):
        deg = global_degrees[indices[i]]

        # Check if this degree is larger than the smallest in our top-k
        if deg > top_degrees[actual_k - 1]:
            # Find insertion point (sorted descending)
            insert_pos = actual_k - 1
            while insert_pos > 0 and deg > top_degrees[insert_pos - 1]:
                insert_pos -= 1

            # Shift elements down
            for j in range(actual_k - 1, insert_pos, -1):
                top_degrees[j] = top_degrees[j - 1]
                top_indices[j] = top_indices[j - 1]

            # Insert new element
            top_degrees[insert_pos] = deg
            top_indices[insert_pos] = indices[i]

    return top_indices


# Minimum split balance threshold - if best split is worse than this, don't split
# A balance of 0.1 means 10/90 split which is quite unbalanced
MIN_SPLIT_BALANCE = 0.1


@numba.njit(
    fastmath=True,
    nogil=True,
    cache=False,
)
def euclidean_hub_split(data, indices, neighbor_indices, global_degrees, rng_state):
    """Hub-based graph-informed split using balance-based selection.

    Uses the top 3 highest-degree nodes to generate all 3 possible hyperplanes,
    then selects the one with the best balance (closest to 50/50 split).
    This is much faster than edge-cut counting while still producing good quality trees.

    Parameters
    ----------
    data : array of shape (n_samples, n_features)
        The data array.
    indices : array of shape (n,)
        Indices of points in this node.
    neighbor_indices : array of shape (n_samples, n_neighbors)
        The neighbor graph.
    global_degrees : array of shape (n_samples,)
        Precomputed global in-degrees.
    rng_state : array of int64, shape (3,)
        RNG state (only used for fallback).

    Returns
    -------
    indices_left, indices_right, hyperplane, offset, balance
        The balance is returned so the caller can decide whether to accept the split.
    """
    dim = data.shape[1]
    n_points = indices.shape[0]

    # Get top 3 hubs from this subset (3 pairs)
    top_hubs = get_top_k_hub_indices(indices, global_degrees, 3)
    n_hubs = top_hubs.shape[0]

    # Storage for best result
    best_balance = np.float32(0.0)  # Closer to 0.5 is better
    best_n_left = np.uint32(0)
    best_n_right = np.uint32(0)
    best_hyperplane = np.zeros(dim, dtype=np.float32)
    best_offset = np.float32(0.0)
    best_side = np.zeros(n_points, dtype=np.int8)
    side = np.empty(n_points, dtype=np.int8)

    # Evaluate all hub pairs and pick the most balanced split
    for hi in range(n_hubs):
        for hj in range(hi + 1, n_hubs):
            left = top_hubs[hi]
            right = top_hubs[hj]

            # Compute the hyperplane between the two hub points
            hyperplane_offset = np.float32(0.0)
            hyperplane_vector = np.empty(dim, dtype=np.float32)

            for d in range(dim):
                hyperplane_vector[d] = data[left, d] - data[right, d]
                hyperplane_offset -= (
                    hyperplane_vector[d] * (data[left, d] + data[right, d]) / 2.0
                )

            # Project all points onto hyperplane
            n_left = np.uint32(0)
            n_right = np.uint32(0)

            for i in range(n_points):
                margin = hyperplane_offset
                for d in range(dim):
                    margin += hyperplane_vector[d] * data[indices[i], d]

                if margin > EPS:
                    side[i] = 0
                    n_left += 1
                elif margin < -EPS:
                    side[i] = 1
                    n_right += 1
                else:
                    side[i] = i % 2
                    if side[i] == 0:
                        n_left += 1
                    else:
                        n_right += 1

            # Skip invalid splits
            if n_left == 0 or n_right == 0:
                continue

            # Score by balance (how close to 50/50)
            balance = np.float32(min(n_left, n_right)) / np.float32(n_points)

            if balance > best_balance:
                best_balance = balance
                best_n_left = n_left
                best_n_right = n_right
                best_offset = hyperplane_offset
                for d in range(dim):
                    best_hyperplane[d] = hyperplane_vector[d]
                for i in range(n_points):
                    best_side[i] = side[i]

    # If no valid candidate found, fall back to random assignment
    if best_n_left == 0 or best_n_right == 0:
        best_n_left = np.uint32(0)
        best_n_right = np.uint32(0)
        for i in range(n_points):
            best_side[i] = np.abs(tau_rand_int(rng_state)) % 2
            if best_side[i] == 0:
                best_n_left += 1
            else:
                best_n_right += 1

    # Allocate and populate result arrays
    indices_left = np.empty(best_n_left, dtype=np.int32)
    indices_right = np.empty(best_n_right, dtype=np.int32)

    n_left = np.uint32(0)
    n_right = np.uint32(0)
    for i in range(n_points):
        if best_side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right, best_hyperplane, best_offset, best_balance


@numba.njit(
    fastmath=True,
    nogil=True,
    cache=False,
)
def angular_hub_split(data, indices, neighbor_indices, global_degrees, rng_state):
    """Angular hub-based split using balance-based selection.

    Uses the top 3 highest-degree nodes to generate all 3 possible hyperplanes,
    then selects the one with the best balance (closest to 50/50 split).

    Returns
    -------
    indices_left, indices_right, hyperplane, offset, balance
        The balance is returned so the caller can decide whether to accept the split.
    """
    dim = data.shape[1]
    n_points = indices.shape[0]

    # Get top 3 hubs from this subset (3 pairs)
    top_hubs = get_top_k_hub_indices(indices, global_degrees, 3)
    n_hubs = top_hubs.shape[0]

    # Storage for best result
    best_balance = np.float32(0.0)
    best_n_left = np.uint32(0)
    best_n_right = np.uint32(0)
    best_hyperplane = np.zeros(dim, dtype=np.float32)
    best_side = np.zeros(n_points, dtype=np.int8)
    side = np.empty(n_points, dtype=np.int8)

    # Evaluate all hub pairs and pick the most balanced split
    for hi in range(n_hubs):
        for hj in range(hi + 1, n_hubs):
            left = top_hubs[hi]
            right = top_hubs[hj]

            # Compute normalized hyperplane (angular distance)
            left_norm = norm(data[left])
            right_norm = norm(data[right])

            if abs(left_norm) < EPS:
                left_norm = 1.0
            if abs(right_norm) < EPS:
                right_norm = 1.0

            hyperplane_vector = np.empty(dim, dtype=np.float32)
            for d in range(dim):
                hyperplane_vector[d] = (data[left, d] / left_norm) - (
                    data[right, d] / right_norm
                )

            hyperplane_norm = norm(hyperplane_vector)
            if abs(hyperplane_norm) < EPS:
                hyperplane_norm = 1.0
            for d in range(dim):
                hyperplane_vector[d] = hyperplane_vector[d] / hyperplane_norm

            # Project all points onto hyperplane
            n_left = np.uint32(0)
            n_right = np.uint32(0)

            for i in range(n_points):
                margin = np.float32(0.0)
                for d in range(dim):
                    margin += hyperplane_vector[d] * data[indices[i], d]

                if margin > EPS:
                    side[i] = 0
                    n_left += 1
                elif margin < -EPS:
                    side[i] = 1
                    n_right += 1
                else:
                    side[i] = i % 2
                    if side[i] == 0:
                        n_left += 1
                    else:
                        n_right += 1

            # Skip invalid splits
            if n_left == 0 or n_right == 0:
                continue

            # Score by balance
            balance = np.float32(min(n_left, n_right)) / np.float32(n_points)

            if balance > best_balance:
                best_balance = balance
                best_n_left = n_left
                best_n_right = n_right
                for d in range(dim):
                    best_hyperplane[d] = hyperplane_vector[d]
                for i in range(n_points):
                    best_side[i] = side[i]

    # If no valid candidate found, fall back to random assignment
    if best_n_left == 0 or best_n_right == 0:
        best_n_left = np.uint32(0)
        best_n_right = np.uint32(0)
        for i in range(n_points):
            best_side[i] = np.abs(tau_rand_int(rng_state)) % 2
            if best_side[i] == 0:
                best_n_left += 1
            else:
                best_n_right += 1

    # Allocate and populate result arrays
    indices_left = np.empty(best_n_left, dtype=np.int32)
    indices_right = np.empty(best_n_right, dtype=np.int32)

    n_left = np.uint32(0)
    n_right = np.uint32(0)
    for i in range(n_points):
        if best_side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right, best_hyperplane, np.float32(0.0), best_balance


@numba.njit(
    nogil=True,
    cache=False,
    locals={"left_node_num": numba.types.int32, "right_node_num": numba.types.int32},
)
def make_hub_euclidean_tree(
    data,
    indices,
    neighbor_indices,
    global_degrees,
    hyperplanes,
    offsets,
    children,
    point_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    """Recursive tree builder using hub-based splits.

    Stops splitting if:
    - Node size <= leaf_size
    - max_depth reached
    - Best split balance < MIN_SPLIT_BALANCE (creates larger leaf instead of bad split)
    """
    if indices.shape[0] > leaf_size and max_depth > 0:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
            balance,
        ) = euclidean_hub_split(
            data, indices, neighbor_indices, global_degrees, rng_state
        )

        # If split is too unbalanced, make a leaf instead
        if balance < MIN_SPLIT_BALANCE:
            hyperplanes.append(np.array([-1.0], dtype=np.float32))
            offsets.append(-np.inf)
            children.append((np.int32(-1), np.int32(-1)))
            point_indices.append(indices)
            return

        make_hub_euclidean_tree(
            data,
            left_indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        left_node_num = len(point_indices) - 1

        make_hub_euclidean_tree(
            data,
            right_indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((np.int32(left_node_num), np.int32(right_node_num)))
        point_indices.append(np.array([-1], dtype=np.int32))
    else:
        hyperplanes.append(np.array([-1.0], dtype=np.float32))
        offsets.append(-np.inf)
        children.append((np.int32(-1), np.int32(-1)))
        point_indices.append(indices)

    return


@numba.njit(
    nogil=True,
    cache=False,
    locals={"left_node_num": numba.types.int32, "right_node_num": numba.types.int32},
)
def make_hub_angular_tree(
    data,
    indices,
    neighbor_indices,
    global_degrees,
    hyperplanes,
    offsets,
    children,
    point_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    """Recursive tree builder using angular hub-based splits.

    Stops splitting if:
    - Node size <= leaf_size
    - max_depth reached
    - Best split balance < MIN_SPLIT_BALANCE (creates larger leaf instead of bad split)
    """
    if indices.shape[0] > leaf_size and max_depth > 0:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
            balance,
        ) = angular_hub_split(
            data, indices, neighbor_indices, global_degrees, rng_state
        )

        # If split is too unbalanced, make a leaf instead
        if balance < MIN_SPLIT_BALANCE:
            hyperplanes.append(np.array([-1.0], dtype=np.float32))
            offsets.append(-np.inf)
            children.append((np.int32(-1), np.int32(-1)))
            point_indices.append(indices)
            return

        make_hub_angular_tree(
            data,
            left_indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        left_node_num = len(point_indices) - 1

        make_hub_angular_tree(
            data,
            right_indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((np.int32(left_node_num), np.int32(right_node_num)))
        point_indices.append(np.array([-1], dtype=np.int32))
    else:
        hyperplanes.append(np.array([-1.0], dtype=np.float32))
        offsets.append(-np.inf)
        children.append((np.int32(-1), np.int32(-1)))
        point_indices.append(indices)

    return


@numba.njit(nogil=True, cache=False)
def make_hub_tree(
    data,
    neighbor_indices,
    rng_state,
    leaf_size=30,
    angular=False,
    max_depth=200,
):
    """Build an RP tree using simplified hub-based hyperplane selection.

    This version precomputes global degrees once and uses the top 3 highest-degree
    nodes at each split to generate all 3 possible hyperplanes. This is simpler
    and significantly faster than the random sampling approach while maintaining
    or improving tree quality.

    Parameters
    ----------
    data : array of shape (n_samples, n_features)
        The data to build the tree on.
    neighbor_indices : array of shape (n_samples, n_neighbors)
        The neighbor graph indices.
    rng_state : array of int64, shape (3,)
        The internal state of the rng.
    leaf_size : int
        The maximum size of a leaf node.
    angular : bool
        Whether to use angular (cosine) or euclidean distance.
    max_depth : int
        Maximum tree depth.

    Returns
    -------
    tree : FlatTree
        The constructed tree.
    """
    # Precompute global degrees ONCE
    global_degrees = compute_global_degrees(neighbor_indices)

    indices = np.arange(data.shape[0]).astype(np.int32)
    hyperplanes = numba.typed.List.empty_list(dense_hyperplane_type)
    offsets = numba.typed.List.empty_list(offset_type)
    children = numba.typed.List.empty_list(children_type)
    point_indices = numba.typed.List.empty_list(point_indices_type)

    if angular:
        make_hub_angular_tree(
            data,
            indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth=max_depth,
        )
    else:
        make_hub_euclidean_tree(
            data,
            indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth=max_depth,
        )

    max_leaf_size = leaf_size
    for points in point_indices:
        if len(points) > max_leaf_size:
            max_leaf_size = numba.int32(len(points))

    result = FlatTree(hyperplanes, offsets, children, point_indices, max_leaf_size)
    return result


# ============================================================================
# Simplified Sparse Hub Trees
# ============================================================================


@numba.njit(
    fastmath=True,
    nogil=True,
    cache=False,
)
def sparse_euclidean_hub_split(
    inds, indptr, spdata, indices, neighbor_indices, global_degrees, rng_state
):
    """Simplified hub-based split for sparse euclidean data.

    Uses the top 3 highest-degree nodes to generate 3 possible hyperplanes,
    then selects the one that minimizes edge cuts.
    """
    n_points = indices.shape[0]

    # Get top 3 hubs from this subset
    top_hubs = get_top_k_hub_indices(indices, global_degrees, 3)
    n_hubs = top_hubs.shape[0]

    # Build lookup
    idx_to_pos = np.full(neighbor_indices.shape[0], -1, dtype=np.int32)
    for i in range(n_points):
        idx_to_pos[indices[i]] = i

    # Storage for best result
    best_hyperplane_inds = np.array([np.int32(-1)])
    best_hyperplane_data = np.array([np.float32(-1.0)])
    best_offset = np.float64(0.0)
    best_side = np.empty(n_points, dtype=np.int8)
    best_edge_cuts = np.uint32(0xFFFFFFFF)
    best_n_left = np.uint32(0)
    best_n_right = np.uint32(0)
    side = np.empty(n_points, dtype=np.int8)

    # Evaluate all hub pairs (only 3 pairs for k=3)
    for hi in range(n_hubs):
        for hj in range(hi + 1, n_hubs):
            left = top_hubs[hi]
            right = top_hubs[hj]

            left_inds = inds[indptr[left] : indptr[left + 1]]
            left_data = spdata[indptr[left] : indptr[left + 1]]
            right_inds = inds[indptr[right] : indptr[right + 1]]
            right_data = spdata[indptr[right] : indptr[right + 1]]

            # Compute hyperplane
            hyperplane_offset = np.float64(0.0)
            hyperplane_inds, hyperplane_data = sparse_diff(
                left_inds, left_data, right_inds, right_data
            )
            offset_inds, offset_data = sparse_sum(
                left_inds, left_data, right_inds, right_data
            )
            offset_data = offset_data / 2.0
            offset_inds, offset_data = sparse_mul(
                hyperplane_inds,
                hyperplane_data,
                offset_inds,
                offset_data.astype(np.float32),
            )
            for val in offset_data:
                hyperplane_offset -= val

            # Project all points
            n_left = np.uint32(0)
            n_right = np.uint32(0)

            for i in range(n_points):
                margin = hyperplane_offset
                i_inds = inds[indptr[indices[i]] : indptr[indices[i] + 1]]
                i_data = spdata[indptr[indices[i]] : indptr[indices[i] + 1]]

                _, mul_data = sparse_mul(
                    hyperplane_inds, hyperplane_data, i_inds, i_data
                )
                for val in mul_data:
                    margin += val

                if margin > EPS:
                    side[i] = 0
                    n_left += 1
                elif margin < -EPS:
                    side[i] = 1
                    n_right += 1
                else:
                    side[i] = i % 2
                    if side[i] == 0:
                        n_left += 1
                    else:
                        n_right += 1

            if n_left == 0 or n_right == 0:
                continue

            # Count edge cuts
            edge_cuts = np.uint32(0)
            for i in range(n_points):
                point_idx = indices[i]
                point_side = side[i]
                for j_nb in range(neighbor_indices.shape[1]):
                    neighbor = neighbor_indices[point_idx, j_nb]
                    if neighbor < 0:
                        break
                    neighbor_pos = idx_to_pos[neighbor]
                    if neighbor_pos >= 0:
                        if side[neighbor_pos] != point_side:
                            edge_cuts += 1

            edge_cuts = edge_cuts // 2

            if edge_cuts < best_edge_cuts:
                best_edge_cuts = edge_cuts
                best_n_left = n_left
                best_n_right = n_right
                best_hyperplane_inds = hyperplane_inds.copy()
                best_hyperplane_data = hyperplane_data.copy()
                best_offset = hyperplane_offset
                for i in range(n_points):
                    best_side[i] = side[i]

    # Fallback
    if best_n_left == 0 or best_n_right == 0:
        best_n_left = np.uint32(0)
        best_n_right = np.uint32(0)
        for i in range(n_points):
            best_side[i] = np.abs(tau_rand_int(rng_state)) % 2
            if best_side[i] == 0:
                best_n_left += 1
            else:
                best_n_right += 1

    indices_left = np.empty(best_n_left, dtype=np.int32)
    indices_right = np.empty(best_n_right, dtype=np.int32)

    n_left = np.uint32(0)
    n_right = np.uint32(0)
    for i in range(n_points):
        if best_side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    hyperplane = np.vstack((best_hyperplane_inds, best_hyperplane_data))
    return indices_left, indices_right, hyperplane, best_offset


@numba.njit(
    fastmath=True,
    nogil=True,
    cache=False,
)
def sparse_angular_hub_split(
    inds, indptr, spdata, indices, neighbor_indices, global_degrees, rng_state
):
    """Simplified hub-based split for sparse angular data.

    Uses the top 3 highest-degree nodes to generate 3 possible hyperplanes,
    then selects the one that minimizes edge cuts.
    """
    n_points = indices.shape[0]

    # Get top 3 hubs from this subset
    top_hubs = get_top_k_hub_indices(indices, global_degrees, 3)
    n_hubs = top_hubs.shape[0]

    # Build lookup
    idx_to_pos = np.full(neighbor_indices.shape[0], -1, dtype=np.int32)
    for i in range(n_points):
        idx_to_pos[indices[i]] = i

    # Storage for best result
    best_hyperplane_inds = np.array([np.int32(-1)])
    best_hyperplane_data = np.array([np.float32(-1.0)])
    best_side = np.empty(n_points, dtype=np.int8)
    best_edge_cuts = np.uint32(0xFFFFFFFF)
    best_n_left = np.uint32(0)
    best_n_right = np.uint32(0)
    side = np.empty(n_points, dtype=np.int8)

    # Evaluate all hub pairs (only 3 pairs for k=3)
    for hi in range(n_hubs):
        for hj in range(hi + 1, n_hubs):
            left = top_hubs[hi]
            right = top_hubs[hj]

            left_inds = inds[indptr[left] : indptr[left + 1]]
            left_data = spdata[indptr[left] : indptr[left + 1]]
            right_inds = inds[indptr[right] : indptr[right + 1]]
            right_data = spdata[indptr[right] : indptr[right + 1]]

            # Normalize for angular distance
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

            # Project all points
            n_left = np.uint32(0)
            n_right = np.uint32(0)

            for i in range(n_points):
                margin = np.float64(0.0)
                i_inds = inds[indptr[indices[i]] : indptr[indices[i] + 1]]
                i_data = spdata[indptr[indices[i]] : indptr[indices[i] + 1]]

                _, mul_data = sparse_mul(
                    hyperplane_inds, hyperplane_data, i_inds, i_data
                )
                for val in mul_data:
                    margin += val

                if margin > EPS:
                    side[i] = 0
                    n_left += 1
                elif margin < -EPS:
                    side[i] = 1
                    n_right += 1
                else:
                    side[i] = i % 2
                    if side[i] == 0:
                        n_left += 1
                    else:
                        n_right += 1

            if n_left == 0 or n_right == 0:
                continue

            # Count edge cuts
            edge_cuts = np.uint32(0)
            for i in range(n_points):
                point_idx = indices[i]
                point_side = side[i]
                for j_nb in range(neighbor_indices.shape[1]):
                    neighbor = neighbor_indices[point_idx, j_nb]
                    if neighbor < 0:
                        break
                    neighbor_pos = idx_to_pos[neighbor]
                    if neighbor_pos >= 0:
                        if side[neighbor_pos] != point_side:
                            edge_cuts += 1

            edge_cuts = edge_cuts // 2

            if edge_cuts < best_edge_cuts:
                best_edge_cuts = edge_cuts
                best_n_left = n_left
                best_n_right = n_right
                best_hyperplane_inds = hyperplane_inds.copy()
                best_hyperplane_data = hyperplane_data.copy()
                for i in range(n_points):
                    best_side[i] = side[i]

    # Fallback
    if best_n_left == 0 or best_n_right == 0:
        best_n_left = np.uint32(0)
        best_n_right = np.uint32(0)
        for i in range(n_points):
            best_side[i] = np.abs(tau_rand_int(rng_state)) % 2
            if best_side[i] == 0:
                best_n_left += 1
            else:
                best_n_right += 1

    indices_left = np.empty(best_n_left, dtype=np.int32)
    indices_right = np.empty(best_n_right, dtype=np.int32)

    n_left = np.uint32(0)
    n_right = np.uint32(0)
    for i in range(n_points):
        if best_side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    hyperplane = np.vstack((best_hyperplane_inds, best_hyperplane_data))
    return indices_left, indices_right, hyperplane, np.float64(0.0)


@numba.njit(
    nogil=True,
    cache=False,
    locals={"left_node_num": numba.types.int32, "right_node_num": numba.types.int32},
)
def make_sparse_hub_euclidean_tree(
    inds,
    indptr,
    spdata,
    indices,
    neighbor_indices,
    global_degrees,
    hyperplanes,
    offsets,
    children,
    point_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    """Recursive tree builder using simplified sparse euclidean hub splits."""
    if indices.shape[0] > leaf_size and max_depth > 0:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = sparse_euclidean_hub_split(
            inds, indptr, spdata, indices, neighbor_indices, global_degrees, rng_state
        )

        make_sparse_hub_euclidean_tree(
            inds,
            indptr,
            spdata,
            left_indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        left_node_num = len(point_indices) - 1

        make_sparse_hub_euclidean_tree(
            inds,
            indptr,
            spdata,
            right_indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((np.int32(left_node_num), np.int32(right_node_num)))
        point_indices.append(np.array([-1], dtype=np.int32))
    else:
        hyperplanes.append(np.array([[-1.0], [-1.0]], dtype=np.float64))
        offsets.append(-np.inf)
        children.append((np.int32(-1), np.int32(-1)))
        point_indices.append(indices)

    return


@numba.njit(
    nogil=True,
    cache=False,
    locals={"left_node_num": numba.types.int32, "right_node_num": numba.types.int32},
)
def make_sparse_hub_angular_tree(
    inds,
    indptr,
    spdata,
    indices,
    neighbor_indices,
    global_degrees,
    hyperplanes,
    offsets,
    children,
    point_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    """Recursive tree builder using simplified sparse angular hub splits."""
    if indices.shape[0] > leaf_size and max_depth > 0:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = sparse_angular_hub_split(
            inds, indptr, spdata, indices, neighbor_indices, global_degrees, rng_state
        )

        make_sparse_hub_angular_tree(
            inds,
            indptr,
            spdata,
            left_indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        left_node_num = len(point_indices) - 1

        make_sparse_hub_angular_tree(
            inds,
            indptr,
            spdata,
            right_indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((np.int32(left_node_num), np.int32(right_node_num)))
        point_indices.append(np.array([-1], dtype=np.int32))
    else:
        hyperplanes.append(np.array([[-1.0], [-1.0]], dtype=np.float64))
        offsets.append(-np.inf)
        children.append((np.int32(-1), np.int32(-1)))
        point_indices.append(indices)

    return


@numba.njit(nogil=True, cache=False)
def make_sparse_hub_tree(
    inds,
    indptr,
    spdata,
    neighbor_indices,
    rng_state,
    leaf_size=30,
    angular=False,
    max_depth=200,
):
    """Build a sparse RP tree using simplified hub-based hyperplane selection.

    This version precomputes global degrees once and uses the top 3 highest-degree
    nodes at each split to generate all 3 possible hyperplanes.

    Parameters
    ----------
    inds : array
        CSR format index array of the matrix.
    indptr : array
        CSR format index pointer array of the matrix.
    spdata : array
        CSR format data array of the matrix.
    neighbor_indices : array of shape (n_samples, n_neighbors)
        The neighbor graph indices.
    rng_state : array of int64, shape (3,)
        The internal state of the rng.
    leaf_size : int
        The maximum size of a leaf node.
    angular : bool
        Whether to use angular (cosine) or euclidean distance.
    max_depth : int
        Maximum tree depth.

    Returns
    -------
    tree : FlatTree
        The constructed tree.
    """
    # Precompute global degrees ONCE
    global_degrees = compute_global_degrees(neighbor_indices)

    indices = np.arange(indptr.shape[0] - 1).astype(np.int32)
    hyperplanes = numba.typed.List.empty_list(sparse_hyperplane_type)
    offsets = numba.typed.List.empty_list(offset_type)
    children = numba.typed.List.empty_list(children_type)
    point_indices = numba.typed.List.empty_list(point_indices_type)

    if angular:
        make_sparse_hub_angular_tree(
            inds,
            indptr,
            spdata,
            indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth=max_depth,
        )
    else:
        make_sparse_hub_euclidean_tree(
            inds,
            indptr,
            spdata,
            indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth=max_depth,
        )

    max_leaf_size = leaf_size
    for points in point_indices:
        if len(points) > max_leaf_size:
            max_leaf_size = numba.int32(len(points))

    result = FlatTree(hyperplanes, offsets, children, point_indices, max_leaf_size)
    return result


# ============================================================================
# Simplified Bit-packed Hub Trees
# ============================================================================


@numba.njit(
    fastmath=True,
    nogil=True,
    cache=False,
)
def get_top_k_hub_indices_bit(indices, data, global_degrees, k=3):
    """Get the indices of the top k highest-degree points for bit data.

    Also returns the pair with maximum Hamming distance among the top k.
    """
    n_points = indices.shape[0]
    actual_k = min(k, n_points)

    # Find top k by degree using insertion sort approach
    top_degrees = np.full(actual_k, np.int32(-1), dtype=np.int32)
    top_indices = np.empty(actual_k, dtype=np.int32)

    for i in range(n_points):
        deg = global_degrees[indices[i]]

        if deg > top_degrees[actual_k - 1]:
            insert_pos = actual_k - 1
            while insert_pos > 0 and deg > top_degrees[insert_pos - 1]:
                insert_pos -= 1

            for j in range(actual_k - 1, insert_pos, -1):
                top_degrees[j] = top_degrees[j - 1]
                top_indices[j] = top_indices[j - 1]

            top_degrees[insert_pos] = deg
            top_indices[insert_pos] = indices[i]

    return top_indices


@numba.njit(
    fastmath=True,
    nogil=True,
    cache=False,
)
def bit_hub_split(data, indices, neighbor_indices, global_degrees, rng_state):
    """Simplified hub-based split for bit-packed data.

    Uses the top 3 highest-degree nodes to generate 3 possible hyperplanes,
    then selects the one that minimizes edge cuts.
    """
    dim = data.shape[1]
    n_points = indices.shape[0]

    # Get top 3 hubs from this subset
    top_hubs = get_top_k_hub_indices_bit(indices, data, global_degrees, 3)
    n_hubs = top_hubs.shape[0]

    # Build lookup
    idx_to_pos = np.full(neighbor_indices.shape[0], -1, dtype=np.int32)
    for i in range(n_points):
        idx_to_pos[indices[i]] = i

    # Storage for best result
    best_hyperplane = np.zeros(dim * 2, dtype=np.uint8)
    best_side = np.empty(n_points, dtype=np.int8)
    best_edge_cuts = np.uint32(0xFFFFFFFF)
    best_n_left = np.uint32(0)
    best_n_right = np.uint32(0)
    side = np.empty(n_points, dtype=np.int8)

    # Evaluate all hub pairs (only 3 pairs for k=3)
    for hi in range(n_hubs):
        for hj in range(hi + 1, n_hubs):
            left = top_hubs[hi]
            right = top_hubs[hj]

            # Compute hyperplane for bit data
            hyperplane_vector = np.empty(dim * 2, dtype=np.uint8)
            positive_hyperplane_component = hyperplane_vector[:dim]
            negative_hyperplane_component = hyperplane_vector[dim:]

            for d in range(dim):
                xor_vector = data[left, d] ^ data[right, d]
                positive_hyperplane_component[d] = xor_vector & data[left, d]
                negative_hyperplane_component[d] = xor_vector & data[right, d]

            # Project all points onto hyperplane
            n_left = np.uint32(0)
            n_right = np.uint32(0)

            for i in range(n_points):
                margin = np.float32(0.0)
                for d in range(dim):
                    margin += popcnt[
                        positive_hyperplane_component[d] & data[indices[i], d]
                    ]
                    margin -= popcnt[
                        negative_hyperplane_component[d] & data[indices[i], d]
                    ]

                if margin > EPS:
                    side[i] = 0
                    n_left += 1
                elif margin < -EPS:
                    side[i] = 1
                    n_right += 1
                else:
                    side[i] = i % 2
                    if side[i] == 0:
                        n_left += 1
                    else:
                        n_right += 1

            if n_left == 0 or n_right == 0:
                continue

            # Count edge cuts
            edge_cuts = np.uint32(0)
            for i in range(n_points):
                point_idx = indices[i]
                point_side = side[i]
                for j_nb in range(neighbor_indices.shape[1]):
                    neighbor = neighbor_indices[point_idx, j_nb]
                    if neighbor < 0:
                        break
                    neighbor_pos = idx_to_pos[neighbor]
                    if neighbor_pos >= 0:
                        if side[neighbor_pos] != point_side:
                            edge_cuts += 1

            edge_cuts = edge_cuts // 2

            if edge_cuts < best_edge_cuts:
                best_edge_cuts = edge_cuts
                best_n_left = n_left
                best_n_right = n_right
                for d in range(dim * 2):
                    best_hyperplane[d] = hyperplane_vector[d]
                for i in range(n_points):
                    best_side[i] = side[i]

    # Fallback
    if best_n_left == 0 or best_n_right == 0:
        best_n_left = np.uint32(0)
        best_n_right = np.uint32(0)
        for i in range(n_points):
            best_side[i] = np.abs(tau_rand_int(rng_state)) % 2
            if best_side[i] == 0:
                best_n_left += 1
            else:
                best_n_right += 1

    indices_left = np.empty(best_n_left, dtype=np.int32)
    indices_right = np.empty(best_n_right, dtype=np.int32)

    n_left = np.uint32(0)
    n_right = np.uint32(0)
    for i in range(n_points):
        if best_side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right, best_hyperplane, np.float32(0.0)


@numba.njit(
    nogil=True,
    cache=False,
    locals={"left_node_num": numba.types.int32, "right_node_num": numba.types.int32},
)
def make_bit_hub_tree_recursive(
    data,
    indices,
    neighbor_indices,
    global_degrees,
    hyperplanes,
    offsets,
    children,
    point_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    """Recursive tree builder using simplified bit hub splits."""
    if indices.shape[0] > leaf_size and max_depth > 0:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = bit_hub_split(data, indices, neighbor_indices, global_degrees, rng_state)

        make_bit_hub_tree_recursive(
            data,
            left_indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        left_node_num = len(point_indices) - 1

        make_bit_hub_tree_recursive(
            data,
            right_indices,
            neighbor_indices,
            global_degrees,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((np.int32(left_node_num), np.int32(right_node_num)))
        point_indices.append(np.array([-1], dtype=np.int32))
    else:
        hyperplanes.append(np.array([np.uint8(0)], dtype=np.uint8))
        offsets.append(-np.inf)
        children.append((np.int32(-1), np.int32(-1)))
        point_indices.append(indices)

    return


@numba.njit(nogil=True, cache=False)
def make_bit_hub_tree(
    data,
    neighbor_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    """Build a bit-packed RP tree using simplified hub-based hyperplane selection.

    This version precomputes global degrees once and uses the top 3 highest-degree
    nodes at each split to generate all 3 possible hyperplanes.

    Parameters
    ----------
    data : array of shape (n_samples, n_features)
        The bit-packed data to build the tree on.
    neighbor_indices : array of shape (n_samples, n_neighbors)
        The neighbor graph indices.
    rng_state : array of int64, shape (3,)
        The internal state of the rng.
    leaf_size : int
        The maximum size of a leaf node.
    max_depth : int
        Maximum tree depth.

    Returns
    -------
    tree : FlatTree
        The constructed tree.
    """
    # Precompute global degrees ONCE
    global_degrees = compute_global_degrees(neighbor_indices)

    indices = np.arange(data.shape[0]).astype(np.int32)
    hyperplanes = numba.typed.List.empty_list(bit_hyperplane_type)
    offsets = numba.typed.List.empty_list(offset_type)
    children = numba.typed.List.empty_list(children_type)
    point_indices = numba.typed.List.empty_list(point_indices_type)

    make_bit_hub_tree_recursive(
        data,
        indices,
        neighbor_indices,
        global_degrees,
        hyperplanes,
        offsets,
        children,
        point_indices,
        rng_state,
        leaf_size,
        max_depth=max_depth,
    )

    max_leaf_size = leaf_size
    for points in point_indices:
        if len(points) > max_leaf_size:
            max_leaf_size = numba.int32(len(points))

    result = FlatTree(hyperplanes, offsets, children, point_indices, max_leaf_size)
    return result


@numba.njit(
    nogil=True,
    locals={"left_node_num": numba.types.int32, "right_node_num": numba.types.int32},
)
def make_euclidean_tree(
    data,
    indices,
    hyperplanes,
    offsets,
    children,
    point_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    if indices.shape[0] > leaf_size and max_depth > 0:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = euclidean_random_projection_split(data, indices, rng_state)

        make_euclidean_tree(
            data,
            left_indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        left_node_num = len(point_indices) - 1

        make_euclidean_tree(
            data,
            right_indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((np.int32(left_node_num), np.int32(right_node_num)))
        point_indices.append(np.array([-1], dtype=np.int32))
    else:
        hyperplanes.append(np.array([-1.0], dtype=np.float32))
        offsets.append(-np.inf)
        children.append((np.int32(-1), np.int32(-1)))
        point_indices.append(indices)

    return


@numba.njit(
    nogil=True,
    locals={
        "children": numba.types.ListType(children_type),
        "left_node_num": numba.types.int32,
        "right_node_num": numba.types.int32,
    },
)
def make_angular_tree(
    data,
    indices,
    hyperplanes,
    offsets,
    children,
    point_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    if indices.shape[0] > leaf_size and max_depth > 0:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = angular_random_projection_split(data, indices, rng_state)

        make_angular_tree(
            data,
            left_indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        left_node_num = len(point_indices) - 1

        make_angular_tree(
            data,
            right_indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((np.int32(left_node_num), np.int32(right_node_num)))
        point_indices.append(np.array([-1], dtype=np.int32))
    else:
        hyperplanes.append(np.array([-1.0], dtype=np.float32))
        offsets.append(-np.inf)
        children.append((np.int32(-1), np.int32(-1)))
        point_indices.append(indices)

    return


@numba.njit(
    nogil=True,
    locals={
        "children": numba.types.ListType(children_type),
        "left_node_num": numba.types.int32,
        "right_node_num": numba.types.int32,
    },
)
def make_bit_tree(
    data,
    indices,
    hyperplanes,
    offsets,
    children,
    point_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    if indices.shape[0] > leaf_size and max_depth > 0:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = angular_bitpacked_random_projection_split(data, indices, rng_state)

        make_bit_tree(
            data,
            left_indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        left_node_num = len(point_indices) - 1

        make_bit_tree(
            data,
            right_indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((np.int32(left_node_num), np.int32(right_node_num)))
        point_indices.append(np.array([-1], dtype=np.int32))
    else:
        hyperplanes.append(np.array([255], dtype=np.uint8))
        offsets.append(-np.inf)
        children.append((np.int32(-1), np.int32(-1)))
        point_indices.append(indices)

    return


@numba.njit(
    nogil=True,
    locals={"left_node_num": numba.types.int32, "right_node_num": numba.types.int32},
)
def make_sparse_euclidean_tree(
    inds,
    indptr,
    data,
    indices,
    hyperplanes,
    offsets,
    children,
    point_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    if indices.shape[0] > leaf_size and max_depth > 0:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = sparse_euclidean_random_projection_split(
            inds, indptr, data, indices, rng_state
        )

        make_sparse_euclidean_tree(
            inds,
            indptr,
            data,
            left_indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        left_node_num = len(point_indices) - 1

        make_sparse_euclidean_tree(
            inds,
            indptr,
            data,
            right_indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((np.int32(left_node_num), np.int32(right_node_num)))
        point_indices.append(np.array([-1], dtype=np.int32))
    else:
        hyperplanes.append(np.array([[-1.0], [-1.0]], dtype=np.float64))
        offsets.append(-np.inf)
        children.append((np.int32(-1), np.int32(-1)))
        point_indices.append(indices)

    return


@numba.njit(
    nogil=True,
    locals={"left_node_num": numba.types.int32, "right_node_num": numba.types.int32},
)
def make_sparse_angular_tree(
    inds,
    indptr,
    data,
    indices,
    hyperplanes,
    offsets,
    children,
    point_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    if indices.shape[0] > leaf_size and max_depth > 0:
        (
            left_indices,
            right_indices,
            hyperplane,
            offset,
        ) = sparse_angular_random_projection_split(
            inds, indptr, data, indices, rng_state
        )

        make_sparse_angular_tree(
            inds,
            indptr,
            data,
            left_indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        left_node_num = len(point_indices) - 1

        make_sparse_angular_tree(
            inds,
            indptr,
            data,
            right_indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((np.int32(left_node_num), np.int32(right_node_num)))
        point_indices.append(np.array([-1], dtype=np.int32))
    else:
        hyperplanes.append(np.array([[-1.0], [-1.0]], dtype=np.float64))
        offsets.append(-np.inf)
        children.append((np.int32(-1), np.int32(-1)))
        point_indices.append(indices)


@numba.njit(nogil=True)
def make_dense_tree(data, rng_state, leaf_size=30, angular=False, max_depth=200):
    indices = np.arange(data.shape[0]).astype(np.int32)
    hyperplanes = numba.typed.List.empty_list(dense_hyperplane_type)
    offsets = numba.typed.List.empty_list(offset_type)
    children = numba.typed.List.empty_list(children_type)
    point_indices = numba.typed.List.empty_list(point_indices_type)

    if angular:
        make_angular_tree(
            data,
            indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth=max_depth,
        )
    else:
        make_euclidean_tree(
            data,
            indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth=max_depth,
        )

    max_leaf_size = leaf_size
    for points in point_indices:
        if len(points) > max_leaf_size:
            max_leaf_size = numba.int32(len(points))

    result = FlatTree(hyperplanes, offsets, children, point_indices, max_leaf_size)
    return result


@numba.njit(nogil=True)
def make_sparse_tree(
    inds,
    indptr,
    spdata,
    rng_state,
    leaf_size=30,
    angular=False,
    max_depth=200,
):
    indices = np.arange(indptr.shape[0] - 1).astype(np.int32)

    hyperplanes = numba.typed.List.empty_list(sparse_hyperplane_type)
    offsets = numba.typed.List.empty_list(offset_type)
    children = numba.typed.List.empty_list(children_type)
    point_indices = numba.typed.List.empty_list(point_indices_type)

    if angular:
        make_sparse_angular_tree(
            inds,
            indptr,
            spdata,
            indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth=max_depth,
        )
    else:
        make_sparse_euclidean_tree(
            inds,
            indptr,
            spdata,
            indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth=max_depth,
        )

    max_leaf_size = leaf_size
    for points in point_indices:
        if len(points) > max_leaf_size:
            max_leaf_size = numba.int32(len(points))

    return FlatTree(hyperplanes, offsets, children, point_indices, max_leaf_size)


@numba.njit(nogil=True)
def make_dense_bit_tree(data, rng_state, leaf_size=30, angular=False, max_depth=200):
    indices = np.arange(data.shape[0]).astype(np.int32)

    hyperplanes = numba.typed.List.empty_list(bit_hyperplane_type)
    offsets = numba.typed.List.empty_list(offset_type)
    children = numba.typed.List.empty_list(children_type)
    point_indices = numba.typed.List.empty_list(point_indices_type)

    if angular:
        make_bit_tree(
            data,
            indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
            max_depth=max_depth,
        )
    else:
        raise NotImplementedError("Euclidean bit trees are not implemented yet.")

    max_leaf_size = leaf_size
    for points in point_indices:
        if len(points) > max_leaf_size:
            max_leaf_size = numba.int32(len(points))

    result = FlatTree(hyperplanes, offsets, children, point_indices, max_leaf_size)
    return result


@numba.njit(
    [
        "b1(f4[::1],f4,f4[::1],i8[::1])",
        numba.types.boolean(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.float32,
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.int64, 1, "C", readonly=False),
        ),
    ],
    fastmath=True,
    locals={
        "margin": numba.types.float32,
        "dim": numba.types.intp,
        "d": numba.types.uint16,
    },
    cache=True,
)
def select_side(hyperplane, offset, point, rng_state):
    margin = offset
    dim = point.shape[0]
    for d in range(dim):
        margin += hyperplane[d] * point[d]

    if abs(margin) < EPS:
        side = np.abs(tau_rand_int(rng_state)) % 2
        if side == 0:
            return 0
        else:
            return 1
    elif margin > 0:
        return 0
    else:
        return 1


@numba.njit(
    [
        "b1(u1[::1],f4,u1[::1],i8[::1])",
        numba.types.boolean(
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.float32,
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.Array(numba.types.int64, 1, "C", readonly=False),
        ),
    ],
    fastmath=True,
    locals={
        "margin": numba.types.float32,
        "dim": numba.types.intp,
        "d": numba.types.uint16,
    },
    cache=True,
)
def select_side_bit(hyperplane, offset, point, rng_state):
    margin = offset
    dim = point.shape[0]
    for d in range(dim):
        margin += popcnt[hyperplane[d] & point[d]]
        margin -= popcnt[hyperplane[dim + d] & point[d]]

    if abs(margin) < EPS:
        side = np.abs(tau_rand_int(rng_state)) % 2
        if side == 0:
            return 0
        else:
            return 1
    elif margin > 0:
        return 0
    else:
        return 1


@numba.njit(
    [
        "i4[::1](f4[::1],f4[:,::1],f4[::1],i4[:,::1],i4[::1],i8[::1])",
        numba.types.Array(numba.types.int32, 1, "C", readonly=True)(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 2, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.int32, 2, "C", readonly=True),
            numba.types.Array(numba.types.int32, 1, "C", readonly=True),
            numba.types.Array(numba.types.int64, 1, "C", readonly=False),
        ),
    ],
    locals={"node": numba.types.uint32, "side": numba.types.boolean},
    cache=True,
)
def search_flat_tree(point, hyperplanes, offsets, children, indices, rng_state):
    node = 0
    while children[node, 0] > 0:
        side = select_side(hyperplanes[node], offsets[node], point, rng_state)
        if side == 0:
            node = children[node, 0]
        else:
            node = children[node, 1]

    return indices[-children[node, 0] : -children[node, 1]]


@numba.njit(
    [
        "i4[::1](u1[::1],u1[:,::1],f4[::1],i4[:,::1],i4[::1],i8[::1])",
        numba.types.Array(numba.types.int32, 1, "C", readonly=True)(
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.Array(numba.types.uint8, 2, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.int32, 2, "C", readonly=True),
            numba.types.Array(numba.types.int32, 1, "C", readonly=True),
            numba.types.Array(numba.types.int64, 1, "C", readonly=False),
        ),
    ],
    locals={"node": numba.types.uint32, "side": numba.types.boolean},
    cache=True,
)
def search_flat_bit_tree(point, hyperplanes, offsets, children, indices, rng_state):
    node = 0
    while children[node, 0] > 0:
        side = select_side_bit(hyperplanes[node], offsets[node], point, rng_state)
        if side == 0:
            node = children[node, 0]
        else:
            node = children[node, 1]

    return indices[-children[node, 0] : -children[node, 1]]


@numba.njit(fastmath=True, cache=True)
def sparse_select_side(hyperplane, offset, point_inds, point_data, rng_state):
    margin = offset

    hyperplane_size = hyperplane.shape[1]
    while hyperplane[0, hyperplane_size - 1] < 0.0:
        hyperplane_size -= 1

    hyperplane_inds = hyperplane[0, :hyperplane_size].astype(np.int32)
    hyperplane_data = hyperplane[1, :hyperplane_size]

    margin += sparse_dot_product(
        hyperplane_inds, hyperplane_data, point_inds, point_data
    )

    if abs(margin) < EPS:
        side = tau_rand_int(rng_state) % 2
        if side == 0:
            return 0
        else:
            return 1
    elif margin > 0:
        return 0
    else:
        return 1


@numba.njit(locals={"node": numba.types.uint32}, cache=True)
def search_sparse_flat_tree(
    point_inds, point_data, hyperplanes, offsets, children, indices, rng_state
):
    node = 0
    while children[node, 0] > 0:
        side = sparse_select_side(
            hyperplanes[node], offsets[node], point_inds, point_data, rng_state
        )
        if side == 0:
            node = children[node, 0]
        else:
            node = children[node, 1]

    return indices[-children[node, 0] : -children[node, 1]]


def make_forest(
    data,
    n_neighbors,
    n_trees,
    leaf_size,
    rng_state,
    random_state,
    n_jobs=None,
    angular=False,
    bit_tree=False,
    max_depth=200,
):
    """Build a random projection forest with ``n_trees``.

    Parameters
    ----------
    data
    n_neighbors
    n_trees
    leaf_size
    rng_state
    angular

    Returns
    -------
    forest: list
        A list of random projection trees.
    """
    # print(ts(), "Started forest construction")
    result = []
    if leaf_size is None:
        leaf_size = max(60, min(256, 5 * np.int32(n_neighbors)))
    if n_jobs is None:
        n_jobs = -1

    rng_states = random_state.randint(INT32_MIN, INT32_MAX, size=(n_trees, 3)).astype(
        np.int64
    )
    try:
        if scipy.sparse.isspmatrix_csr(data):
            result = joblib.Parallel(n_jobs=n_jobs, require="sharedmem")(
                joblib.delayed(make_sparse_tree)(
                    data.indices,
                    data.indptr,
                    data.data,
                    rng_states[i],
                    leaf_size,
                    angular,
                    max_depth=max_depth,
                )
                for i in range(n_trees)
            )
        elif bit_tree:
            result = joblib.Parallel(n_jobs=n_jobs, require="sharedmem")(
                joblib.delayed(make_dense_bit_tree)(
                    data, rng_states[i], leaf_size, angular, max_depth=max_depth
                )
                for i in range(n_trees)
            )
        else:
            result = joblib.Parallel(n_jobs=n_jobs, require="sharedmem")(
                joblib.delayed(make_dense_tree)(
                    data, rng_states[i], leaf_size, angular, max_depth=max_depth
                )
                for i in range(n_trees)
            )
    except (RuntimeError, RecursionError, SystemError):
        warn(
            "Random Projection forest initialisation failed due to recursion"
            "limit being reached. Something is a little strange with your "
            "graph_data, and this may take longer than normal to compute."
        )

    return tuple(result)


@numba.njit(nogil=True)
def get_leaves_from_tree(tree, max_leaf_size):
    n_leaves = 0
    for i in range(len(tree.children)):
        if tree.children[i][0] == -1 and tree.children[i][1] == -1:
            n_leaves += 1

    result = np.full((n_leaves, max_leaf_size), -1, dtype=np.int32)
    leaf_index = 0
    for i in range(len(tree.indices)):
        if tree.children[i][0] == -1 or tree.children[i][1] == -1:
            leaf_size = tree.indices[i].shape[0]
            result[leaf_index, :leaf_size] = tree.indices[i]
            leaf_index += 1

    return result


def rptree_leaf_array_parallel(rp_forest):
    max_leaf_size = np.max([rp_tree.leaf_size for rp_tree in rp_forest])
    result = joblib.Parallel(n_jobs=-1, require="sharedmem")(
        joblib.delayed(get_leaves_from_tree)(rp_tree, max_leaf_size)
        for rp_tree in rp_forest
    )
    return result


def rptree_leaf_array(rp_forest):
    if len(rp_forest) > 0:
        return np.vstack(rptree_leaf_array_parallel(rp_forest))
    else:
        return np.array([[-1]])


# @numba.njit()
def recursive_convert(
    tree, hyperplanes, offsets, children, indices, node_num, leaf_start, tree_node
):
    if tree.children[tree_node][0] < 0:
        leaf_end = leaf_start + len(tree.indices[tree_node])
        children[node_num, 0] = -leaf_start
        children[node_num, 1] = -leaf_end
        indices[leaf_start:leaf_end] = tree.indices[tree_node]
        return node_num, leaf_end
    else:
        hyperplanes[node_num] = tree.hyperplanes[tree_node]
        offsets[node_num] = tree.offsets[tree_node]
        children[node_num, 0] = node_num + 1
        old_node_num = node_num
        node_num, leaf_start = recursive_convert(
            tree,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_start,
            tree.children[tree_node][0],
        )
        children[old_node_num, 1] = node_num + 1
        node_num, leaf_start = recursive_convert(
            tree,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_start,
            tree.children[tree_node][1],
        )
        return node_num, leaf_start


@numba.njit()
def recursive_convert_sparse(
    tree, hyperplanes, offsets, children, indices, node_num, leaf_start, tree_node
):
    if tree.children[tree_node][0] < 0:
        leaf_end = leaf_start + len(tree.indices[tree_node])
        children[node_num, 0] = -leaf_start
        children[node_num, 1] = -leaf_end
        indices[leaf_start:leaf_end] = tree.indices[tree_node]
        return node_num, leaf_end
    else:
        hyperplanes[node_num, :, : tree.hyperplanes[tree_node].shape[1]] = (
            tree.hyperplanes[tree_node]
        )
        offsets[node_num] = tree.offsets[tree_node]
        children[node_num, 0] = node_num + 1
        old_node_num = node_num
        node_num, leaf_start = recursive_convert_sparse(
            tree,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_start,
            tree.children[tree_node][0],
        )
        children[old_node_num, 1] = node_num + 1
        node_num, leaf_start = recursive_convert_sparse(
            tree,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_start,
            tree.children[tree_node][1],
        )
        return node_num, leaf_start


@numba.njit(cache=True)
def num_nodes_and_leaves(tree):
    n_nodes = 0
    n_leaves = 0
    for i in range(len(tree.children)):
        if tree.children[i][0] < 0:
            n_leaves += 1
            n_nodes += 1
        else:
            n_nodes += 1

    return n_nodes, n_leaves


def convert_tree_format(tree, data_size, data_dim):
    n_nodes, n_leaves = num_nodes_and_leaves(tree)
    is_sparse = False
    if tree.hyperplanes[0].ndim == 1:
        # dense hyperplanes
        if tree.hyperplanes[0].dtype == np.uint8:
            hyperplane_dim = data_dim * 2
        else:
            hyperplane_dim = data_dim
        hyperplanes = np.zeros(
            (n_nodes, hyperplane_dim), dtype=tree.hyperplanes[0].dtype
        )
    else:
        # sparse hyperplanes
        is_sparse = True
        hyperplane_dim = data_dim
        hyperplanes = np.zeros((n_nodes, 2, hyperplane_dim), dtype=np.float32)
        hyperplanes[:, 0, :] = -1

    offsets = np.zeros(n_nodes, dtype=np.float32)
    children = np.int32(-1) * np.ones((n_nodes, 2), dtype=np.int32)
    indices = np.int32(-1) * np.ones(data_size, dtype=np.int32)
    if is_sparse:
        recursive_convert_sparse(
            tree, hyperplanes, offsets, children, indices, 0, 0, len(tree.children) - 1
        )
    else:
        recursive_convert(
            tree, hyperplanes, offsets, children, indices, 0, 0, len(tree.children) - 1
        )
    return FlatTree(hyperplanes, offsets, children, indices, tree.leaf_size)


# Indices for tuple version of flat tree for pickle serialization
FLAT_TREE_HYPERPLANES = 0
FLAT_TREE_OFFSETS = 1
FLAT_TREE_CHILDREN = 2
FLAT_TREE_INDICES = 3
FLAT_TREE_LEAF_SIZE = 4


def denumbaify_tree(tree):
    result = (
        tree.hyperplanes,
        tree.offsets,
        tree.children,
        tree.indices,
        tree.leaf_size,
    )

    return result


def renumbaify_tree(tree):
    result = FlatTree(
        tree[FLAT_TREE_HYPERPLANES],
        tree[FLAT_TREE_OFFSETS],
        tree[FLAT_TREE_CHILDREN],
        tree[FLAT_TREE_INDICES],
        tree[FLAT_TREE_LEAF_SIZE],
    )

    return result


@numba.njit(
    parallel=True,
    locals={
        "intersection": numba.int64[::1],
        "result": numba.float32,
        "i": numba.uint32,
    },
    cache=False,
)
def score_tree(tree, neighbor_indices, data, rng_state):
    result = 0.0
    for i in numba.prange(neighbor_indices.shape[0]):
        leaf_indices = search_flat_tree(
            data[i],
            tree.hyperplanes,
            tree.offsets,
            tree.children,
            tree.indices,
            rng_state,
        )
        intersection = arr_intersect(neighbor_indices[i], leaf_indices)
        result += numba.float32(intersection.shape[0] > 1)
    return result / numba.float32(neighbor_indices.shape[0])


@numba.njit(
    nogil=True,
    locals={"node": numba.int32, "count": numba.int32},
    cache=False,
)
def score_linked_tree(tree, neighbor_indices):
    """Score a tree by measuring how well leaves contain nearest neighbors.

    For each point, computes the fraction of its k nearest neighbors that
    are in the same leaf. Returns the average of this fraction across all points.

    A score of 1.0 means all neighbors are always in the same leaf (perfect).
    A score of 0.0 means no neighbors are ever in the same leaf (worst).
    """
    n_points = neighbor_indices.shape[0]
    k = neighbor_indices.shape[1]
    total_score = 0.0
    n_nodes = len(tree.children)

    for i in range(n_nodes):
        node = numba.int32(i)
        left_child = tree.children[node][0]
        right_child = tree.children[node][1]

        # Only process leaf nodes
        if left_child == -1 and right_child == -1:
            leaf_indices = tree.indices[node]
            leaf_size = leaf_indices.shape[0]

            # Build a lookup set for the leaf (max value we need to check)
            # Use a simple approach: for each point in leaf, count neighbors in leaf
            for j in range(leaf_size):
                idx = leaf_indices[j]
                neighbors = neighbor_indices[idx]

                # Count how many neighbors are in this leaf
                count = 0
                for ni in range(k):
                    neighbor = neighbors[ni]
                    # Check if neighbor is in leaf (linear scan - leaf is small)
                    for li in range(leaf_size):
                        if leaf_indices[li] == neighbor:
                            count += 1
                            break

                # Subtract 1 if point itself is counted as neighbor (self-loop)
                # and normalize by k
                # Actually, neighbor_indices typically doesn't include self,
                # so we just use count directly
                total_score += numba.float32(count) / numba.float32(k)

    return total_score / numba.float32(n_points)
