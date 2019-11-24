# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Enough simple sparse operations in numba to enable sparse UMAP
#
# License: BSD 3 clause
from __future__ import print_function
import locale
import numpy as np
import numba

from pynndescent.utils import norm

locale.setlocale(locale.LC_NUMERIC, "C")

FLOAT32_EPS = np.finfo(np.float32).eps

# Just reproduce a simpler version of numpy unique (not numba supported yet)
@numba.njit()
def arr_unique(arr):
    aux = np.sort(arr)
    flag = np.concatenate((np.ones(1, dtype=np.bool_), aux[1:] != aux[:-1]))
    return aux[flag]


# Just reproduce a simpler version of numpy union1d (not numba supported yet)
@numba.njit()
def arr_union(ar1, ar2):
    if ar1.shape[0] == 0:
        return ar2
    elif ar2.shape[0] == 0:
        return ar1
    else:
        return arr_unique(np.concatenate((ar1, ar2)))


# Just reproduce a simpler version of numpy intersect1d (not numba supported
# yet)
@numba.njit()
def arr_intersect(ar1, ar2):
    aux = np.concatenate((ar1, ar2))
    aux.sort()
    return aux[:-1][aux[1:] == aux[:-1]]


@numba.njit(
    fastmath=True,
    locals={
        "ind1": numba.types.int32[::1],
        "data1": numba.types.float32[::1],
        "ind2": numba.types.int32[::1],
        "data2": numba.types.float32[::1],
        "result_ind": numba.types.int32[::1],
        "result_data": numba.types.float32[::1],
        "val": numba.types.float32,
        "i1": numba.types.uint32,
        "i2": numba.types.uint32,
        "j1": numba.types.uint32,
        "j2": numba.types.uint32,
    },
)
def sparse_sum(ind1, data1, ind2, data2):
    result_size = len(set(np.concatenate((ind1, ind2))))
    result_ind = np.zeros(result_size, dtype=np.int32)
    result_data = np.zeros(result_size, dtype=np.float32)

    i1 = 0
    i2 = 0
    nnz = 0

    # pass through both index lists
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] + data2[i2]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            val = data1[i1]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
        else:
            val = data2[i2]
            if val != 0:
                result_ind[nnz] = j2
                result_data[nnz] = val
                nnz += 1
            i2 += 1

    # pass over the tails
    while i1 < ind1.shape[0]:
        j1 = ind1[i1]
        val = data1[i1]
        if val != 0:
            result_ind[nnz] = j1
            result_data[nnz] = val
            nnz += 1
        i1 += 1

    while i2 < ind2.shape[0]:
        j2 = ind2[i2]
        val = data2[i2]
        if val != 0:
            result_ind[nnz] = j2
            result_data[nnz] = val
            nnz += 1
        i2 += 1

    # truncate to the correct length in case there were zeros created
    result_ind = result_ind[:nnz]
    result_data = result_data[:nnz]

    return result_ind, result_data


@numba.njit()
def sparse_diff(ind1, data1, ind2, data2):
    return sparse_sum(ind1, data1, ind2, -data2)


@numba.njit(
    fastmath=True,
    locals={
        "ind1": numba.types.int32[::1],
        "data1": numba.types.float32[::1],
        "ind2": numba.types.int32[::1],
        "data2": numba.types.float32[::1],
        "val": numba.types.float32,
        "i1": numba.types.uint32,
        "i2": numba.types.uint32,
        "j1": numba.types.uint32,
        "j2": numba.types.uint32,
    },
)
def sparse_mul(ind1, data1, ind2, data2):
    result_ind = numba.typed.List.empty_list(numba.types.int32)
    result_data = numba.typed.List.empty_list(numba.types.float32)

    i1 = 0
    i2 = 0

    # pass through both index lists
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] * data2[i2]
            if val != 0:
                result_ind.append(j1)
                result_data.append(val)
            i1 += 1
            i2 += 1
        elif j1 < j2:
            i1 += 1
        else:
            i2 += 1

    return result_ind, result_data


# @numba.njit()
# def sparse_mul(ind1, data1, ind2, data2):
#     result_ind = arr_intersect(ind1, ind2)
#     result_data = np.zeros(result_ind.shape[0], dtype=np.float32)
#
#     i1 = 0
#     i2 = 0
#     nnz = 0
#
#     # pass through both index lists
#     while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
#         j1 = ind1[i1]
#         j2 = ind2[i2]
#
#         if j1 == j2:
#             val = data1[i1] * data2[i2]
#             if val != 0:
#                 result_ind[nnz] = j1
#                 result_data[nnz] = val
#                 nnz += 1
#             i1 += 1
#             i2 += 1
#         elif j1 < j2:
#             i1 += 1
#         else:
#             i2 += 1
#
#     # truncate to the correct length in case there were zeros created
#     result_ind = result_ind[:nnz]
#     result_data = result_data[:nnz]
#
#     return result_ind, result_data


@numba.njit()
def sparse_euclidean(ind1, data1, ind2, data2):
    _, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += aux_data[i] ** 2
    return np.sqrt(result)


@numba.njit()
def sparse_manhattan(ind1, data1, ind2, data2):
    _, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += np.abs(aux_data[i])
    return result


@numba.njit()
def sparse_chebyshev(ind1, data1, ind2, data2):
    _, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result = max(result, np.abs(aux_data[i]))
    return result


@numba.njit()
def sparse_minkowski(ind1, data1, ind2, data2, p=2.0):
    _, aux_data = sparse_diff(ind1, data1, ind2, data2)
    result = 0.0
    for i in range(aux_data.shape[0]):
        result += np.abs(aux_data[i]) ** p
    return result ** (1.0 / p)


@numba.njit()
def sparse_hamming(ind1, data1, ind2, data2, n_features):
    num_not_equal = sparse_diff(ind1, data1, ind2, data2)[0].shape[0]
    return float(num_not_equal) / n_features


@numba.njit()
def sparse_canberra(ind1, data1, ind2, data2):
    abs_data1 = np.abs(data1)
    abs_data2 = np.abs(data2)
    denom_inds, denom_data = sparse_sum(ind1, abs_data1, ind2, abs_data2)
    denom_data = (1.0 / denom_data).astype(np.float32)
    numer_inds, numer_data = sparse_diff(ind1, data1, ind2, data2)
    numer_data = np.abs(numer_data)

    _, val_data = sparse_mul(numer_inds, numer_data, denom_inds, denom_data)
    result = 0.0
    for val in val_data:
        result += val

    return result


@numba.njit()
def sparse_bray_curtis(ind1, data1, ind2, data2):  # pragma: no cover
    abs_data1 = np.abs(data1)
    abs_data2 = np.abs(data2)
    _, denom_data = sparse_sum(ind1, abs_data1, ind2, abs_data2)

    if denom_data.shape[0] == 0:
        return 0.0

    denominator = np.sum(denom_data)

    _, numer_data = sparse_diff(ind1, data1, ind2, data2)
    numer_data = np.abs(numer_data)

    numerator = np.sum(numer_data)

    return float(numerator) / denominator


@numba.njit()
def sparse_jaccard(ind1, data1, ind2, data2):
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_equal = arr_intersect(ind1, ind2).shape[0]

    if num_non_zero == 0:
        return 0.0
    else:
        return float(num_non_zero - num_equal) / num_non_zero


@numba.njit()
def sparse_matching(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    return float(num_not_equal) / n_features


@numba.njit()
def sparse_dice(ind1, data1, ind2, data2):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (2.0 * num_true_true + num_not_equal)


@numba.njit()
def sparse_kulsinski(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    if num_not_equal == 0:
        return 0.0
    else:
        return float(num_not_equal - num_true_true + n_features) / (
            num_not_equal + n_features
        )


@numba.njit()
def sparse_rogers_tanimoto(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    return (2.0 * num_not_equal) / (n_features + num_not_equal)


@numba.njit()
def sparse_russellrao(ind1, data1, ind2, data2, n_features):
    if ind1.shape[0] == ind2.shape[0] and np.all(ind1 == ind2):
        return 0.0

    num_true_true = arr_intersect(ind1, ind2).shape[0]

    if num_true_true == np.sum(data1 != 0) and num_true_true == np.sum(data2 != 0):
        return 0.0
    else:
        return float(n_features - num_true_true) / (n_features)


@numba.njit()
def sparse_sokal_michener(ind1, data1, ind2, data2, n_features):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    return (2.0 * num_not_equal) / (n_features + num_not_equal)


@numba.njit()
def sparse_sokal_sneath(ind1, data1, ind2, data2):
    num_true_true = arr_intersect(ind1, ind2).shape[0]
    num_non_zero = arr_union(ind1, ind2).shape[0]
    num_not_equal = num_non_zero - num_true_true

    if num_not_equal == 0.0:
        return 0.0
    else:
        return num_not_equal / (0.5 * num_true_true + num_not_equal)


@numba.njit()
def sparse_cosine(ind1, data1, ind2, data2):
    _, aux_data = sparse_mul(ind1, data1, ind2, data2)
    result = 0.0
    norm1 = norm(data1)
    norm2 = norm(data2)

    for val in aux_data:
        result += val

    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    elif norm1 == 0.0 or norm2 == 0.0:
        return 1.0
    else:
        return 1.0 - (result / (norm1 * norm2))


@numba.njit()
def sparse_correlation(ind1, data1, ind2, data2, n_features):

    mu_x = 0.0
    mu_y = 0.0
    dot_product = 0.0

    if ind1.shape[0] == 0 and ind2.shape[0] == 0:
        return 0.0
    elif ind1.shape[0] == 0 or ind2.shape[0] == 0:
        return 1.0

    for i in range(data1.shape[0]):
        mu_x += data1[i]
    for i in range(data2.shape[0]):
        mu_y += data2[i]

    mu_x /= n_features
    mu_y /= n_features

    shifted_data1 = np.empty(data1.shape[0], dtype=np.float32)
    shifted_data2 = np.empty(data2.shape[0], dtype=np.float32)

    for i in range(data1.shape[0]):
        shifted_data1[i] = data1[i] - mu_x
    for i in range(data2.shape[0]):
        shifted_data2[i] = data2[i] - mu_y

    norm1 = np.sqrt(
        (norm(shifted_data1) ** 2) + (n_features - ind1.shape[0]) * (mu_x ** 2)
    )
    norm2 = np.sqrt(
        (norm(shifted_data2) ** 2) + (n_features - ind2.shape[0]) * (mu_y ** 2)
    )

    dot_prod_inds, dot_prod_data = sparse_mul(ind1, shifted_data1, ind2, shifted_data2)

    common_indices = set(dot_prod_inds)

    for val in dot_prod_data:
        dot_product += val

    for i in range(ind1.shape[0]):
        if ind1[i] not in common_indices:
            dot_product -= shifted_data1[i] * (mu_y)

    for i in range(ind2.shape[0]):
        if ind2[i] not in common_indices:
            dot_product -= shifted_data2[i] * (mu_x)

    all_indices = arr_union(ind1, ind2)
    dot_product += mu_x * mu_y * (n_features - all_indices.shape[0])

    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    elif dot_product == 0.0:
        return 1.0
    else:
        return 1.0 - (dot_product / (norm1 * norm2))


@numba.njit()
def sparse_hellinger(ind1, data1, ind2, data2):
    aux_inds, aux_data = sparse_mul(ind1, data1, ind2, data2)
    result = 0.0
    norm1 = np.sum(data1)
    norm2 = np.sum(data2)
    sqrt_norm_prod = np.sqrt(norm1 * norm2)

    for val in aux_data:
        result += np.sqrt(val)

    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    elif norm1 == 0.0 or norm2 == 0.0:
        return 1.0
    elif result > sqrt_norm_prod:
        return 0.0
    else:
        return np.sqrt(1.0 - (result / sqrt_norm_prod))


@numba.njit(parallel=True)
def diversify(
    indices,
    distances,
    data_indices,
    data_indptr,
    data_data,
    dist,
    dist_args,
    epsilon=0.01,
):

    for i in numba.prange(indices.shape[0]):

        new_indices = [indices[i, 0]]
        new_distances = [distances[i, 0]]
        for j in range(1, indices.shape[1]):
            if indices[i, j] < 0:
                break

            flag = True
            for k in range(len(new_indices)):
                c = new_indices[k]

                from_ind = data_indices[
                    data_indptr[indices[i, j]] : data_indptr[indices[i, j] + 1]
                ]
                from_data = data_data[
                    data_indptr[indices[i, j]] : data_indptr[indices[i, j] + 1]
                ]

                to_ind = data_indices[data_indptr[c] : data_indptr[c + 1]]
                to_data = data_data[data_indptr[c] : data_indptr[c + 1]]

                d = dist(from_ind, from_data, to_ind, to_data, *dist_args)
                if new_distances[k] > FLOAT32_EPS and d < epsilon * distances[i, j]:
                    flag = False
                    break

            if flag:
                new_indices.append(indices[i, j])
                new_distances.append(distances[i, j])

        for j in range(indices.shape[1]):
            if j < len(new_indices):
                indices[i, j] = new_indices[j]
                distances[i, j] = new_distances[j]
            else:
                indices[i, j] = -1
                distances[i, j] = np.inf

    return indices, distances


@numba.njit(parallel=True)
def diversify_csr(
    graph_indptr,
    graph_indices,
    graph_data,
    data_indptr,
    data_indices,
    data_data,
    dist,
    dist_args,
    epsilon=0.01,
):

    n_nodes = graph_indptr.shape[0] - 1

    for i in numba.prange(n_nodes):

        current_indices = graph_indices[graph_indptr[i] : graph_indptr[i + 1]]
        current_data = graph_data[graph_indptr[i] : graph_indptr[i + 1]]

        order = np.argsort(current_data)
        retained = np.ones(order.shape[0], dtype=np.int8)

        for idx in range(1, order.shape[0]):

            j = order[idx]

            for k in range(idx):
                if retained[k] == 1:
                    p = current_indices[j]
                    q = current_indices[k]

                    from_inds = data_indices[data_indptr[p] : data_indptr[p + 1]]
                    from_data = data_data[data_indptr[p] : data_indptr[p + 1]]

                    to_inds = data_indices[data_indptr[q] : data_indptr[q + 1]]
                    to_data = data_data[data_indptr[q] : data_indptr[q + 1]]
                    d = dist(from_inds, from_data, to_inds, to_data, *dist_args)

                    if current_data[k] > FLOAT32_EPS and d < epsilon * current_data[j]:
                        retained[j] = 0
                        break

        for idx in range(order.shape[0]):
            j = order[idx]
            if retained[j] == 0:
                graph_data[graph_indptr[i] + j] = 0

    return


sparse_named_distances = {
    # general minkowski distances
    "euclidean": sparse_euclidean,
    "manhattan": sparse_manhattan,
    "l1": sparse_manhattan,
    "taxicab": sparse_manhattan,
    "chebyshev": sparse_chebyshev,
    "linf": sparse_chebyshev,
    "linfty": sparse_chebyshev,
    "linfinity": sparse_chebyshev,
    "minkowski": sparse_minkowski,
    # Other distances
    "canberra": sparse_canberra,
    # 'braycurtis': sparse_bray_curtis,
    # Binary distances
    "hamming": sparse_hamming,
    "jaccard": sparse_jaccard,
    "dice": sparse_dice,
    "matching": sparse_matching,
    "kulsinski": sparse_kulsinski,
    "rogerstanimoto": sparse_rogers_tanimoto,
    "russellrao": sparse_russellrao,
    "sokalmichener": sparse_sokal_michener,
    "sokalsneath": sparse_sokal_sneath,
    "cosine": sparse_cosine,
    "correlation": sparse_correlation,
    "hellinger": sparse_hellinger,
}

sparse_need_n_features = (
    "hamming",
    "matching",
    "kulsinski",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "correlation",
)
