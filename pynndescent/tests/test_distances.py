import numpy as np
from numpy.testing import assert_array_equal
import pynndescent.distances as dist
import pynndescent.sparse as spdist
from scipy import sparse, stats
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
from sklearn.utils.testing import assert_array_almost_equal

np.random.seed(42)
spatial_data = np.random.randn(10, 20)
spatial_data = np.vstack([spatial_data, np.zeros((2, 20))]).astype(
    np.float32, order="C"
)  # Add some all zero graph_data for corner case test
binary_data = np.random.choice(a=[False, True], size=(10, 20), p=[0.66, 1 - 0.66])
binary_data = np.vstack(
    [binary_data, np.zeros((2, 20), dtype="bool")]
)  # Add some all zero graph_data for corner case test
sparse_spatial_data = sparse.csr_matrix(spatial_data * binary_data, dtype=np.float32)
sparse_spatial_data.sort_indices()
sparse_binary_data = sparse.csr_matrix(binary_data)
sparse_binary_data.sort_indices()


def spatial_check(metric):
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


def binary_check(metric):
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


def sparse_spatial_check(metric, decimal=6):
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


def sparse_binary_check(metric):
    if metric in spdist.sparse_named_distances:
        dist_matrix = pairwise_distances(sparse_binary_data.todense(), metric=metric)
    if metric in ("jaccard", "dice", "sokalsneath", "yule"):
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


def test_euclidean():
    spatial_check("euclidean")


def test_manhattan():
    spatial_check("manhattan")


def test_chebyshev():
    spatial_check("chebyshev")


def test_minkowski():
    spatial_check("minkowski")


def test_hamming():
    spatial_check("hamming")


def test_canberra():
    spatial_check("canberra")


def test_braycurtis():
    spatial_check("braycurtis")


def test_cosine():
    spatial_check("cosine")


def test_correlation():
    spatial_check("correlation")


def test_jaccard():
    binary_check("jaccard")


def test_matching():
    binary_check("matching")


def test_dice():
    binary_check("dice")


def test_kulsinski():
    binary_check("kulsinski")


def test_rogerstanimoto():
    binary_check("rogerstanimoto")


def test_russellrao():
    binary_check("russellrao")


def test_sokalmichener():
    binary_check("sokalmichener")


def test_sokalsneath():
    binary_check("sokalsneath")


def test_yule():
    binary_check("yule")


def test_sparse_euclidean():
    sparse_spatial_check("euclidean")


def test_sparse_manhattan():
    sparse_spatial_check("manhattan")


def test_sparse_chebyshev():
    sparse_spatial_check("chebyshev")


def test_sparse_minkowski():
    sparse_spatial_check("minkowski")


def test_sparse_hamming():
    sparse_spatial_check("hamming")


def test_sparse_canberra():
    sparse_spatial_check("canberra")  # Be a little forgiving


def test_sparse_cosine():
    sparse_spatial_check("cosine")


def test_sparse_correlation():
    sparse_spatial_check("correlation")


def test_sparse_jaccard():
    sparse_binary_check("jaccard")


def test_sparse_matching():
    sparse_binary_check("matching")


def test_sparse_dice():
    sparse_binary_check("dice")


def test_sparse_kulsinski():
    sparse_binary_check("kulsinski")


def test_sparse_rogerstanimoto():
    sparse_binary_check("rogerstanimoto")


def test_sparse_russellrao():
    sparse_binary_check("russellrao")


def test_sparse_sokalmichener():
    sparse_binary_check("sokalmichener")


def test_sparse_sokalsneath():
    sparse_binary_check("sokalsneath")


def test_seuclidean():
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


def test_weighted_minkowski():
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


def test_mahalanobis():
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


def test_haversine():
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
