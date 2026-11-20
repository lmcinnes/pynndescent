import numba
import numpy as np
import pytest
from scipy import sparse

from pynndescent.hybrid_forest import make_hybrid_leaf_array


@pytest.mark.parametrize("angular", [False, True])
@pytest.mark.parametrize("is_sparse", [False, True])
@pytest.mark.parametrize("local_bytes", [0, 4096, 1 << 20])
def test_hybrid_leaf_membership_and_thread_independence(
    angular, is_sparse, local_bytes
):
    data = np.random.RandomState(42).normal(size=(300, 16)).astype(np.float32)
    if is_sparse:
        data[np.abs(data) < 1] = 0
        data = sparse.csr_matrix(data)
    original_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        expected = make_hybrid_leaf_array(
            data, 3, 15, 42, angular=angular, local_bytes=local_bytes
        )
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        actual = make_hybrid_leaf_array(
            data, 3, 15, 42, angular=angular, local_bytes=local_bytes
        )
    finally:
        numba.set_num_threads(original_threads)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape[1] <= 15
    members = actual[actual >= 0]
    assert members.size == 3 * data.shape[0]
    for tree in members.reshape(3, data.shape[0]):
        np.testing.assert_array_equal(np.sort(tree), np.arange(data.shape[0]))
    for leaf in actual:
        size = np.count_nonzero(leaf >= 0)
        assert size > 0
        assert np.all(leaf[size:] == -1)
