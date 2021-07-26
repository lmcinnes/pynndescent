import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pynndescent.distances as dist
import pynndescent.sparse as spdist
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
from sklearn.preprocessing import normalize


@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "manhattan",
        "chebyshev",
        "minkowski",
        "hamming",
        "canberra",
        "braycurtis",
        "cosine",
        "correlation",
    ],
)
def test_spatial_check(spatial_data, metric):
    dist_matrix = pairwise_distances(spatial_data, metric=metric)
    # scipy is bad sometimes
    if metric == "braycurtis":
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0
    if metric in ("cosine", "correlation"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 1.0
        # And because distance between all zero vectors should be zero
        dist_matrix[10, 11] = 0.0
        dist_matrix[11, 10] = 0.0
    dist_function = dist.named_distances[metric]
    test_matrix = np.array(
        [
            [
                dist_function(spatial_data[i], spatial_data[j])
                for j in range(spatial_data.shape[0])
            ]
            for i in range(spatial_data.shape[0])
        ]
    )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric {}".format(metric),
    )


@pytest.mark.parametrize(
    "metric",
    [
        "jaccard",
        "matching",
        "dice",
        "kulsinski",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "yule",
    ],
)
def test_binary_check(binary_data, metric):
    dist_matrix = pairwise_distances(binary_data, metric=metric)
    if metric in ("jaccard", "dice", "sokalsneath", "yule"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0
    if metric in ("kulsinski", "russellrao"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0
        # And because distance between all zero vectors should be zero
        dist_matrix[10, 11] = 0.0
        dist_matrix[11, 10] = 0.0
    dist_function = dist.named_distances[metric]
    test_matrix = np.array(
        [
            [
                dist_function(binary_data[i], binary_data[j])
                for j in range(binary_data.shape[0])
            ]
            for i in range(binary_data.shape[0])
        ]
    )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric {}".format(metric),
    )


@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "manhattan",
        "chebyshev",
        "minkowski",
        "hamming",
        "canberra",
        "cosine",
        "braycurtis",
        "correlation",
    ],
)
def test_sparse_spatial_check(sparse_spatial_data, metric, decimal=6):
    if metric in spdist.sparse_named_distances:
        dist_matrix = pairwise_distances(
            sparse_spatial_data.todense().astype(np.float32), metric=metric
        )
    if metric in ("braycurtis", "dice", "sokalsneath", "yule"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0
    if metric in ("cosine", "correlation", "kulsinski", "russellrao"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 1.0
        # And because distance between all zero vectors should be zero
        dist_matrix[10, 11] = 0.0
        dist_matrix[11, 10] = 0.0

    dist_function = spdist.sparse_named_distances[metric]
    if metric in spdist.sparse_need_n_features:
        test_matrix = np.array(
            [
                [
                    dist_function(
                        sparse_spatial_data[i].indices,
                        sparse_spatial_data[i].data,
                        sparse_spatial_data[j].indices,
                        sparse_spatial_data[j].data,
                        sparse_spatial_data.shape[1],
                    )
                    for j in range(sparse_spatial_data.shape[0])
                ]
                for i in range(sparse_spatial_data.shape[0])
            ]
        )
    else:
        test_matrix = np.array(
            [
                [
                    dist_function(
                        sparse_spatial_data[i].indices,
                        sparse_spatial_data[i].data,
                        sparse_spatial_data[j].indices,
                        sparse_spatial_data[j].data,
                    )
                    for j in range(sparse_spatial_data.shape[0])
                ]
                for i in range(sparse_spatial_data.shape[0])
            ]
        )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Sparse distances don't match " "for metric {}".format(metric),
        decimal=decimal,
    )


@pytest.mark.parametrize(
    "metric",
    [
        "jaccard",
        "matching",
        "dice",
        "kulsinski",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
    ],
)
def test_sparse_binary_check(sparse_binary_data, metric):
    if metric in spdist.sparse_named_distances:
        dist_matrix = pairwise_distances(sparse_binary_data.todense(), metric=metric)
    if metric in ("jaccard", "dice", "sokalsneath"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 0.0
    if metric in ("kulsinski", "russellrao"):
        dist_matrix[np.where(~np.isfinite(dist_matrix))] = 1.0
        # And because distance between all zero vectors should be zero
        dist_matrix[10, 11] = 0.0
        dist_matrix[11, 10] = 0.0

    dist_function = spdist.sparse_named_distances[metric]
    if metric in spdist.sparse_need_n_features:
        test_matrix = np.array(
            [
                [
                    dist_function(
                        sparse_binary_data[i].indices,
                        sparse_binary_data[i].data,
                        sparse_binary_data[j].indices,
                        sparse_binary_data[j].data,
                        sparse_binary_data.shape[1],
                    )
                    for j in range(sparse_binary_data.shape[0])
                ]
                for i in range(sparse_binary_data.shape[0])
            ]
        )
    else:
        test_matrix = np.array(
            [
                [
                    dist_function(
                        sparse_binary_data[i].indices,
                        sparse_binary_data[i].data,
                        sparse_binary_data[j].indices,
                        sparse_binary_data[j].data,
                    )
                    for j in range(sparse_binary_data.shape[0])
                ]
                for i in range(sparse_binary_data.shape[0])
            ]
        )

    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Sparse distances don't match " "for metric {}".format(metric),
    )


def test_seuclidean(spatial_data):
    v = np.abs(np.random.randn(spatial_data.shape[1]))
    dist_matrix = pairwise_distances(spatial_data, metric="seuclidean", V=v)
    test_matrix = np.array(
        [
            [
                dist.standardised_euclidean(spatial_data[i], spatial_data[j], v)
                for j in range(spatial_data.shape[0])
            ]
            for i in range(spatial_data.shape[0])
        ]
    )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric seuclidean",
    )


def test_weighted_minkowski(spatial_data):
    v = np.abs(np.random.randn(spatial_data.shape[1]))
    dist_matrix = pairwise_distances(spatial_data, metric="wminkowski", w=v, p=3)
    test_matrix = np.array(
        [
            [
                dist.weighted_minkowski(spatial_data[i], spatial_data[j], v, p=3)
                for j in range(spatial_data.shape[0])
            ]
            for i in range(spatial_data.shape[0])
        ]
    )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric weighted_minkowski",
    )


def test_mahalanobis(spatial_data):
    v = np.cov(np.transpose(spatial_data))
    dist_matrix = pairwise_distances(spatial_data, metric="mahalanobis", VI=v)
    test_matrix = np.array(
        [
            [
                dist.mahalanobis(spatial_data[i], spatial_data[j], v)
                for j in range(spatial_data.shape[0])
            ]
            for i in range(spatial_data.shape[0])
        ]
    )
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric mahalanobis",
    )


def test_haversine(spatial_data):
    tree = BallTree(spatial_data[:, :2], metric="haversine")
    dist_matrix, _ = tree.query(spatial_data[:, :2], k=spatial_data.shape[0])
    test_matrix = np.array(
        [
            [
                dist.haversine(spatial_data[i, :2], spatial_data[j, :2])
                for j in range(spatial_data.shape[0])
            ]
            for i in range(spatial_data.shape[0])
        ]
    )
    test_matrix.sort(axis=1)
    assert_array_almost_equal(
        test_matrix,
        dist_matrix,
        err_msg="Distances don't match " "for metric haversine",
    )


def test_spearmanr():
    x = np.random.randn(100)
    y = np.random.randn(100)

    scipy_expected = stats.spearmanr(x, y)
    r = dist.spearmanr(x, y)
    assert_array_equal(r, scipy_expected.correlation)


def test_alternative_distances():

    for distname in dist.fast_distance_alternatives:

        true_dist = dist.named_distances[distname]
        alt_dist = dist.fast_distance_alternatives[distname]["dist"]
        correction = dist.fast_distance_alternatives[distname]["correction"]

        for i in range(100):
            x = np.random.random(30).astype(np.float32)
            y = np.random.random(30).astype(np.float32)
            x[x < 0.25] = 0.0
            y[y < 0.25] = 0.0

            true_distance = true_dist(x, y)
            corrected_alt_distance = correction(alt_dist(x, y))

            assert np.isclose(true_distance, corrected_alt_distance)


def test_jensen_shannon():
    test_data = np.random.random(size=(10, 50))
    test_data = normalize(test_data, norm="l1")
    for i in range(test_data.shape[0]):
        for j in range(i + 1, test_data.shape[0]):
            m = (test_data[i] + test_data[j]) / 2.0
            p = test_data[i]
            q = test_data[j]
            d1 = (
                -np.sum(m * np.log(m))
                + (np.sum(p * np.log(p)) + np.sum(q * np.log(q))) / 2.0
            )
            d2 = dist.jensen_shannon_divergence(p, q)
            assert np.isclose(d1, d2, rtol=1e-4)


def test_sparse_jensen_shannon():
    test_data = np.random.random(size=(10, 100))
    # sparsify
    test_data[test_data <= 0.5] = 0.0
    sparse_test_data = csr_matrix(test_data)
    sparse_test_data = normalize(sparse_test_data, norm="l1")
    test_data = normalize(test_data, norm="l1")

    for i in range(test_data.shape[0]):
        for j in range(i + 1, test_data.shape[0]):
            m = (test_data[i] + test_data[j]) / 2.0
            p = test_data[i]
            q = test_data[j]
            d1 = (
                -np.sum(m[m > 0] * np.log(m[m > 0]))
                + (
                    np.sum(p[p > 0] * np.log(p[p > 0]))
                    + np.sum(q[q > 0] * np.log(q[q > 0]))
                )
                / 2.0
            )
            d2 = spdist.sparse_jensen_shannon_divergence(
                    sparse_test_data[i].indices,
                    sparse_test_data[i].data,
                    sparse_test_data[j].indices,
                    sparse_test_data[j].data,
                )
            assert np.isclose(d1, d2, rtol=1e-3)
