import os
import pytest
import numpy as np
from scipy import sparse

# Making Random Seed as a fixture in case it would be
# needed in tests for random states
@pytest.fixture
def seed():
    return 189212  # 0b101110001100011100


np.random.seed(189212)


@pytest.fixture
def spatial_data():
    sp_data = np.random.randn(10, 20)
    # Add some all zero graph_data for corner case test
    sp_data = np.vstack([sp_data, np.zeros((2, 20))]).astype(np.float32, order="C")
    return sp_data


@pytest.fixture
def binary_data():
    bin_data = np.random.choice(a=[False, True], size=(10, 20), p=[0.66, 1 - 0.66])
    # Add some all zero graph_data for corner case test
    bin_data = np.vstack([bin_data, np.zeros((2, 20), dtype="bool")])
    return bin_data


@pytest.fixture
def sparse_spatial_data(spatial_data, binary_data):
    sp_sparse_data = sparse.csr_matrix(spatial_data * binary_data, dtype=np.float32)
    sp_sparse_data.sort_indices()
    return sp_sparse_data


@pytest.fixture
def sparse_binary_data(binary_data):
    bin_sparse_data = sparse.csr_matrix(binary_data)
    bin_sparse_data.sort_indices()
    return bin_sparse_data


@pytest.fixture
def nn_data():
    nndata = np.random.uniform(0, 1, size=(1000, 5))
    # Add some all zero graph_data for corner case test
    nndata = np.vstack([nndata, np.zeros((2, 5))])
    return nndata


@pytest.fixture
def sparse_nn_data():
    return sparse.random(1000, 50, density=0.5, format="csr")


@pytest.fixture
def cosine_hang_data():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir, "test_data/cosine_hang.npy")
    return np.load(data_path)


@pytest.fixture
def cosine_near_duplicates_data():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir, "test_data/cosine_near_duplicates.npy")
    return np.load(data_path)


@pytest.fixture
def small_data():
    return np.random.uniform(40, 5, size=(20, 5))


@pytest.fixture
def sparse_small_data():
    # Too low dim might cause more than one empty row,
    # which might decrease the computed performance
    return sparse.random(40, 32, density=0.5, format="csr")


@pytest.fixture
def update_data():
    np.random.seed(12345)
    xs_orig = np.random.uniform(0, 1, size=(1000, 5))
    xs_fresh = np.random.uniform(0, 1, size=xs_orig.shape)
    xs_fresh_small = np.random.uniform(0, 1, size=(100, xs_orig.shape[1]))
    xs_for_complete_update = np.random.uniform(0, 1, size=xs_orig.shape)
    updates = [
        (xs_orig, None, None, None),
        (xs_orig, xs_fresh, None, None),
        (xs_orig, None, xs_for_complete_update, list(range(xs_orig.shape[0]))),
        (xs_orig, None, -xs_orig[0:50:2], list(range(0, 50, 2))),
        (xs_orig, None, -xs_orig[0:500:2], list(range(0, 500, 2))),
        (xs_orig, xs_fresh, xs_for_complete_update, list(range(xs_orig.shape[0]))),
        (xs_orig, xs_fresh_small, -xs_orig[0:50:2], list(range(0, 50, 2))),
        (xs_orig, xs_fresh, -xs_orig[0:500:2], list(range(0, 500, 2))),
    ]
    return updates
