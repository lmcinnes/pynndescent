import numpy as np
import pynndescent.distances as dist
import pynndescent.sparse as spdist
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import BallTree
from sklearn.utils.testing import assert_array_almost_equal

np.random.seed(42)
spatial_data = np.random.randn(10, 20)
spatial_data = np.vstack(
    [spatial_data, np.zeros((2, 20))]
)  # Add some all zero data for corner case test
binary_data = np.random.choice(a=[False, True], size=(10, 20), p=[0.66, 1 - 0.66])
binary_data = np.vstack(
    [binary_data, np.zeros((2, 20), dtype="bool")]
)  # Add some all zero data for corner case test
sparse_spatial_data = sparse.csr_matrix(spatial_data * binary_data)
sparse_binary_data = sparse.csr_matrix(binary_data)

spatial_distances = (
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "hamming",
    "canberra",
    "braycurtis",
    "cosine",
    "correlation",
)

binary_distances = (
    "jaccard",
    "matching",
    "dice",
    "kulsinski",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
    "yule",
)

def test_metrics():
    for metric in spatial_distances:
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

    for metric in binary_distances:
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

    # Handle the few special distances separately
    # SEuclidean
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

    # Weighted minkowski
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
    # Mahalanobis
    v = np.abs(np.random.randn(spatial_data.shape[1], spatial_data.shape[1]))
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
    # Haversine
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


def test_sparse_metrics():
    for metric in spatial_distances:
        if metric in spdist.sparse_named_distances:
            dist_matrix = pairwise_distances(
                sparse_spatial_data.todense(), metric=metric
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
            )

    for metric in binary_distances:
        if metric in spdist.sparse_named_distances:
            dist_matrix = pairwise_distances(
                sparse_binary_data.todense(), metric=metric
            )
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
