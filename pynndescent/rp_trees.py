# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 2 clause
from __future__ import print_function
from collections import namedtuple
from warnings import warn

import locale
import numpy as np
import numba
import scipy.sparse

from pynndescent.sparse import sparse_mul, sparse_diff, sparse_sum, arr_unique
from pynndescent.utils import tau_rand_int, norm, seed, ts
import joblib

locale.setlocale(locale.LC_NUMERIC, "C")

# Used for a floating point "nearly zero" comparison
EPS = 1e-8
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

RandomProjectionTreeNode = namedtuple(
    "RandomProjectionTreeNode",
    ["graph_indices", "is_leaf", "hyperplane", "offset", "left_child", "right_child"],
)

FlatTree = namedtuple(
    "FlatTree", ["hyperplanes", "offsets", "children", "indices", "leaf_size"]
)

dense_hyperplane_type = numba.float32[::1]
sparse_hyperplane_type = numba.float64[:, ::1]
offset_type = numba.float64
children_type = numba.typeof((-1, -1))
point_indices_type = numba.int64[::1]


@numba.njit(fastmath=True, nogil=True, cache=True)
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

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

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


@numba.njit(fastmath=True, nogil=True, cache=True)
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

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

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


@numba.njit(fastmath=True, nogil=True, cache=True)
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
    normalized_left_data = left_data / left_norm
    normalized_right_data = right_data / right_norm
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
        for d in range(mul_data.shape[0]):
            margin += mul_data[d]

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

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

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

    # Compute the normal vector to the hyperplane (the vector between
    # the two points) and the offset from the origin
    hyperplane_offset = 0.0
    hyperplane_inds, hyperplane_data = sparse_diff(
        left_inds, left_data, right_inds, right_data
    )
    offset_inds, offset_data = sparse_sum(left_inds, left_data, right_inds, right_data)
    offset_data = offset_data / 2.0
    offset_inds, offset_data = sparse_mul(
        hyperplane_inds, hyperplane_data, offset_inds, offset_data
    )

    for d in range(offset_data.shape[0]):
        hyperplane_offset -= offset_data[d]

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
        for d in range(mul_data.shape[0]):
            margin += mul_data[d]

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

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int64)
    indices_right = np.empty(n_right, dtype=np.int64)

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


@numba.njit(nogil=True, cache=True)
def make_euclidean_tree(
    data,
    indices,
    hyperplanes,
    offsets,
    children,
    point_indices,
    rng_state,
    leaf_size=30,
):
    if indices.shape[0] > leaf_size:
        left_indices, right_indices, hyperplane, offset = euclidean_random_projection_split(
            data, indices, rng_state
        )

        make_euclidean_tree(
            data,
            left_indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
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
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((left_node_num, right_node_num))
        point_indices.append(np.array([-1], dtype=np.int64))
        # print("Made a node in tree with", len(point_indices), "nodes")
    else:
        hyperplanes.append(np.array([-1.0], dtype=np.float32))
        offsets.append(-np.inf)
        children.append((-1, -1))
        point_indices.append(indices)
        # print("Made a leaf in tree with", len(point_indices), "nodes")

    return


@numba.njit(nogil=True, cache=True)
def make_angular_tree(
    data,
    indices,
    hyperplanes,
    offsets,
    children,
    point_indices,
    rng_state,
    leaf_size=30,
):
    if indices.shape[0] > leaf_size:
        left_indices, right_indices, hyperplane, offset = angular_random_projection_split(
            data, indices, rng_state
        )

        make_angular_tree(
            data,
            left_indices,
            hyperplanes,
            offsets,
            children,
            point_indices,
            rng_state,
            leaf_size,
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
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((left_node_num, right_node_num))
        point_indices.append(np.array([-1], dtype=np.int64))
    else:
        hyperplanes.append(np.array([-1.0], dtype=np.float32))
        offsets.append(-np.inf)
        children.append((-1, -1))
        point_indices.append(indices)

    return


@numba.njit(nogil=True, cache=True)
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
):
    if indices.shape[0] > leaf_size:
        left_indices, right_indices, hyperplane, offset = sparse_euclidean_random_projection_split(
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
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((left_node_num, right_node_num))
        point_indices.append(np.array([-1], dtype=np.int64))
    else:
        hyperplanes.append(np.array([[-1.0], [-1.0]], dtype=np.float64))
        offsets.append(-np.inf)
        children.append((-1, -1))
        point_indices.append(indices)

    return


@numba.njit(nogil=True, cache=True)
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
):
    if indices.shape[0] > leaf_size:
        left_indices, right_indices, hyperplane, offset = sparse_angular_random_projection_split(
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
        )

        right_node_num = len(point_indices) - 1

        hyperplanes.append(hyperplane)
        offsets.append(offset)
        children.append((left_node_num, right_node_num))
        point_indices.append(np.array([-1], dtype=np.int64))
    else:
        hyperplanes.append(np.array([[-1.0], [-1.0]], dtype=np.float64))
        offsets.append(-np.inf)
        children.append((-1, -1))
        point_indices.append(indices)


@numba.njit(nogil=True, cache=True)
def make_dense_tree(data, rng_state, leaf_size=30, angular=False):
    indices = np.arange(data.shape[0])

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
        )

    # print("Completed a tree")
    result = FlatTree(hyperplanes, offsets, children, point_indices, leaf_size)
    # print("Tree type is:", numba.typeof(result))
    return result


@numba.njit(nogil=True, cache=True)
def make_sparse_tree(inds, indptr, spdata, rng_state, leaf_size=30, angular=False):
    indices = np.arange(indptr.shape[0] - 1)

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
        )

    return FlatTree(hyperplanes, offsets, children, point_indices, leaf_size)


@numba.njit('b1(f4[::1],f4,f4[::1],i8[::1])',
    fastmath=True,
    locals={
        "margin": numba.types.float32,
        "dim": numba.types.uint16,
        "d": numba.types.uint16,
    }
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


@numba.njit('i4[::1](f4[::1],f4[:,::1],f4[::1],i8[:,::1],i8[:,::1],i8[::1])')
def search_flat_tree(point, hyperplanes, offsets, children, indices, rng_state):
    node = 0
    while children[node, 0] > 0:
        side = select_side(hyperplanes[node], offsets[node], point, rng_state)
        if side == 0:
            node = children[node, 0]
        else:
            node = children[node, 1]

    return indices[-children[node, 0]].astype(np.int32)


@numba.njit(fastmath=True)
def sparse_select_side(hyperplane, offset, point_inds, point_data, rng_state):
    margin = offset

    hyperplane_inds = arr_unique(hyperplane[0])
    hyperplane_data = hyperplane[1, : hyperplane_inds.shape[0]]

    _, aux_data = sparse_mul(hyperplane_inds, hyperplane_data, point_inds, point_data)

    for d in range(aux_data.shape[0]):
        margin += aux_data[d]

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


@numba.njit()
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

    return indices[-children[node, 0]]


def make_forest(data, n_neighbors, n_trees, leaf_size, rng_state,
                random_state, n_jobs=None, angular=False):
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
        leaf_size = max(10, n_neighbors)
    if n_jobs is None:
        n_jobs = -1

    rng_states = random_state.randint(INT32_MIN, INT32_MAX, size=(n_trees,
                                                                  3)).astype(
        np.int64)
    try:
        if scipy.sparse.isspmatrix_csr(data):
            result = joblib.Parallel(n_jobs=-1, prefer="threads")(
                joblib.delayed(make_sparse_tree)(data.indices,
                                                 data.indptr,
                                                 data.data,
                                                 rng_states[i],
                                                 leaf_size,
                                                 angular)
                for i in range(n_trees)
            )
        else:
            result = joblib.Parallel(n_jobs=-1, prefer="threads")(
                joblib.delayed(make_dense_tree)(data,
                                                rng_states[i],
                                                leaf_size,
                                                angular)
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
def get_leaves_from_tree(tree):
    n_leaves = 0
    for i in range(len(tree.children)):
        if tree.children[i][0] == -1 and tree.children[i][1] == -1:
            n_leaves += 1

    result = -1 * np.ones((n_leaves, tree.leaf_size), dtype=np.int64)
    leaf_index = 0
    for i in range(len(tree.indices)):
        if tree.children[i][0] == -1 or tree.children[i][1] == -1:
            leaf_size = tree.indices[i].shape[0]
            result[leaf_index, : leaf_size] = tree.indices[i]
            leaf_index += 1

    return result


def rptree_leaf_array_parallel(rp_forest):
    result = joblib.Parallel(n_jobs=-1, prefer="threads")(
        joblib.delayed(get_leaves_from_tree)(rp_tree)
        for rp_tree in rp_forest
    )
    # result = [get_leaves_from_tree(rp_tree) for rp_tree in rp_forest]
    return result


def rptree_leaf_array(rp_forest):
    if len(rp_forest) > 0:
        return np.vstack(rptree_leaf_array_parallel(rp_forest))
    else:
        return np.array([[-1]])
# def rptree_leaf_array(rp_forest):
#     """Generate an array of sets of candidate nearest neighbors by
#     constructing a random projection forest and taking the leaves of all the
#     trees. Any given tree has leaves that are a set of potential nearest
#     neighbors. Given enough trees the set of all such leaves gives a good
#     likelihood of getting a good set of nearest neighbors in composite. Since
#     such a random projection forest is inexpensive to compute, this can be a
#     useful means of seeding other nearest neighbor algorithms.
#     Parameters
#     ----------
#     graph_data: array of shape (n_samples, n_features)
#         The graph_data for which to generate nearest neighbor approximations.
#     n_neighbors: int
#         The number of nearest neighbors to attempt to approximate.
#     rng_state: array of int64, shape (3,)
#         The internal state of the rng
#     n_trees: int (optional, default 10)
#         The number of trees to build in the forest construction.
#     angular: bool (optional, default False)
#         Whether to use angular/cosine distance for random projection tree
#         construction.
#     Returns
#     -------
#     leaf_array: array of shape (n_leaves, max(10, n_neighbors))
#         Each row of leaf array is a list of graph_indices found in a given leaf.
#         Since not all leaves are the same size the arrays are padded out with -1
#         to ensure we can return a single ndarray.
#     """
#     if rp_forest:
#         # leaf_array = np.vstack([tree.graph_indices for tree in rp_forest])
#         leaf_array = np.vstack([get_leaves_from_tree(tree) for tree in rp_forest])
#     else:
#         leaf_array = np.array([[-1]])
#
#     return leaf_array


@numba.njit()
def recursive_convert(tree, hyperplanes, offsets, children, indices, node_num,
                      leaf_num, tree_node):

    if tree.children[tree_node][0] < 0:
        children[node_num, 0] = -leaf_num
        indices[leaf_num, : len(tree.indices[tree_node])] = tree.indices[tree_node]
        leaf_num += 1
        return node_num, leaf_num
    else:
        hyperplanes[node_num] = tree.hyperplanes[tree_node]
        offsets[node_num] = tree.offsets[tree_node]
        children[node_num, 0] = node_num + 1
        old_node_num = node_num
        node_num, leaf_num = recursive_convert(
            tree,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_num,
            tree.children[tree_node][0]
        )
        children[old_node_num, 1] = node_num + 1
        node_num, leaf_num = recursive_convert(
            tree,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_num,
            tree.children[tree_node][1]
        )
        return node_num, leaf_num

@numba.njit()
def recursive_convert_sparse(tree, hyperplanes, offsets, children, indices, node_num,
                      leaf_num, tree_node):

    if tree.children[tree_node][0] < 0:
        children[node_num, 0] = -leaf_num
        indices[leaf_num, : len(tree.indices[tree_node])] = tree.indices[tree_node]
        leaf_num += 1
        return node_num, leaf_num
    else:
        hyperplanes[node_num, :, :tree.hyperplanes[tree_node].shape[1]] = tree.hyperplanes[
            tree_node]
        offsets[node_num] = tree.offsets[tree_node]
        children[node_num, 0] = node_num + 1
        old_node_num = node_num
        node_num, leaf_num = recursive_convert_sparse(
            tree,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_num,
            tree.children[tree_node][0]
        )
        children[old_node_num, 1] = node_num + 1
        node_num, leaf_num = recursive_convert_sparse(
            tree,
            hyperplanes,
            offsets,
            children,
            indices,
            node_num + 1,
            leaf_num,
            tree.children[tree_node][1]
        )
        return node_num, leaf_num

@numba.njit()
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

# #@numba.njit()
# def convert_tree_format(tree):
#     n_nodes, n_leaves = num_nodes_and_leaves(tree)
#     print(n_nodes, n_leaves, len(tree.children))
#     hyperplane_dim = np.max([tree.hyperplanes[i].shape[0] for i in range(len(
#         tree.hyperplanes))])
#
#     hyperplanes = np.zeros((n_nodes, hyperplane_dim), dtype=np.float32)
#     offsets = np.zeros(n_nodes, dtype=np.float32)
#     children = -1 * np.ones((n_nodes, 2), dtype=np.int64)
#     graph_indices = -1 * np.ones((n_leaves, tree.leaf_size), dtype=np.int64)
#     recursive_convert(tree, hyperplanes, offsets, children, graph_indices, 0, 0,
#                       len(tree.children) - 1)
#     return FlatTree(hyperplanes, offsets, children, graph_indices, tree.leaf_size)


@numba.njit()
def dense_hyperplane_dim(hyperplanes):
    for i in range(len(hyperplanes)):
        if hyperplanes[i].shape[0] > 1:
            return hyperplanes[i].shape[0]
    else:
        raise ValueError("No hyperplanes of adequate size were found!")



@numba.njit()
def sparse_hyperplane_dim(hyperplanes):
    max_dim = 0
    for i in range(len(hyperplanes)):
        if hyperplanes[i].shape[1] > max_dim:
            max_dim = hyperplanes[i].shape[1]
    return max_dim


def convert_tree_format(tree):

    n_nodes, n_leaves = num_nodes_and_leaves(tree)
    is_sparse = False
    if tree.hyperplanes[0].ndim == 1:
        # dense hyperplanes
        hyperplane_dim = dense_hyperplane_dim(tree.hyperplanes)
        hyperplanes = np.zeros((n_nodes, hyperplane_dim), dtype=np.float32)
    else:
        # sparse hyperplanes
        is_sparse = True
        hyperplane_dim = sparse_hyperplane_dim(tree.hyperplanes)
        hyperplanes = np.zeros((n_nodes, 2, hyperplane_dim), dtype=np.float32)
        hyperplanes[:, 0, :] = -1

    offsets = np.zeros(n_nodes, dtype=np.float32)
    children = -1 * np.ones((n_nodes, 2), dtype=np.int64)
    indices = -1 * np.ones((n_leaves, tree.leaf_size), dtype=np.int64)
    if is_sparse:
        recursive_convert_sparse(tree, hyperplanes, offsets, children, indices, 0, 0,
                          len(tree.children) - 1)
    else:
        recursive_convert(tree, hyperplanes, offsets, children, indices, 0, 0,
                          len(tree.children) - 1)
    return FlatTree(hyperplanes, offsets, children, indices, tree.leaf_size)
