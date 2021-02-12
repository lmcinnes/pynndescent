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
