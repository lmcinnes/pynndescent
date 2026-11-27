import numba
import numpy as np
import pytest

from pynndescent import NNDescent
from pynndescent.distances import bit_hamming


@numba.njit
def custom_bit_hamming(x, y):
    distance = 0
    for i in range(x.shape[0]):
        value = x[i] ^ y[i]
        for bit in range(8):
            distance += (value >> bit) & 1
    return np.float32(distance)


@pytest.fixture(scope="module", params=["bit_hamming", bit_hamming])
def bit_index(request):
    data = np.random.RandomState(42).randint(0, 256, size=(12, 4)).astype(np.uint8)
    index = NNDescent(
        data,
        metric=request.param,
        bit_metric=True,
        angular_trees=True,
        n_neighbors=4,
        n_trees=2,
        n_iters=2,
        n_jobs=1,
        random_state=42,
    )
    return data, index


@pytest.fixture(scope="module", params=[False, True], ids=["serial", "parallel"])
def custom_bit_index(request):
    data = np.random.RandomState(42).randint(0, 256, size=(12, 4)).astype(np.uint8)
    parallel = request.param
    index = NNDescent(
        data,
        metric=custom_bit_hamming,
        bit_metric=True,
        angular_trees=True,
        n_neighbors=4,
        n_trees=2,
        n_iters=2,
        n_jobs=2 if parallel else 1,
        parallel_batch_queries=parallel,
        random_state=42,
    )
    return data, index, parallel


@pytest.mark.parametrize("query_options", [{}, {"candidate_width": 4}])
def test_bit_metric_query_preserves_packed_dtype(bit_index, query_options):
    data, index = bit_index
    queries = data[:2].copy()

    indices, distances = index.query(queries, k=3, **query_options)

    assert indices.shape == distances.shape == (2, 3)
    assert np.all((indices >= 0) & (indices < len(data)))
    expected = np.unpackbits(queries[:, None, :] ^ data[indices], axis=2).sum(axis=2)
    np.testing.assert_array_equal(distances, expected)
    np.testing.assert_array_equal(queries, data[:2])


@pytest.mark.parametrize(
    "query_options",
    [pytest.param({}, id="fast"), pytest.param({"candidate_width": 4}, id="fallback")],
)
def test_custom_bit_metric_query_matches_unpacked_hamming(
    custom_bit_index, query_options
):
    data, index, parallel = custom_bit_index
    original_data = data.copy()
    queries = data[[0, 2, 4]].copy()
    original_queries = queries.copy()

    index.prepare()
    assert (getattr(index, "_fast_plan", None) is not None) is (not parallel)
    indices, distances = index.query(queries, k=3, **query_options)

    assert indices.shape == distances.shape == (3, 3)
    assert np.all((indices >= 0) & (indices < len(data)))
    expected = np.unpackbits(queries[:, None, :] ^ data[indices], axis=2).sum(axis=2)
    np.testing.assert_array_equal(distances, expected)
    np.testing.assert_array_equal(queries, original_queries)
    np.testing.assert_array_equal(data, original_data)


@pytest.mark.parametrize("query_options", [{}, {"candidate_width": 4}])
@pytest.mark.parametrize("shape", [(2, 2), (2, 5), (4,), (1, 2, 4)])
def test_bit_metric_query_rejects_invalid_shape(bit_index, query_options, shape):
    _, index = bit_index
    queries = np.zeros(shape, dtype=np.uint8)

    with pytest.raises(
        ValueError, match="query_data must be a 2-D array with 4 features"
    ):
        index.query(queries, k=3, **query_options)
