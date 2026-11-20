import numba
import numpy as np
import pytest

from pynndescent import rp_trees

F4 = numba.types.Array(numba.float32, 1, "C")
F4_RO = numba.types.Array(numba.float32, 1, "C", readonly=True)
U1 = numba.types.Array(numba.uint8, 1, "C")
U1_RO = numba.types.Array(numba.uint8, 1, "C", readonly=True)
I4 = numba.types.Array(numba.int32, 1, "C")
I8 = numba.types.Array(numba.int64, 1, "C")
F4_2D = numba.types.Array(numba.float32, 2, "C")
I4_2D = numba.types.Array(numba.int32, 2, "C")


@pytest.mark.parametrize(
    "kernel,args,return_type",
    [
        (rp_trees.binary_search, (I4, numba.int32), numba.int32),
        (rp_trees.select_side, (F4, numba.float32, F4, I8), numba.boolean),
        (rp_trees.select_side, (F4_RO, numba.float32, F4_RO, I8), numba.boolean),
        (rp_trees.select_side_bit, (U1, numba.float32, U1, I8), numba.boolean),
        (rp_trees.select_side_bit, (U1_RO, numba.float32, U1_RO, I8), numba.boolean),
    ],
)
def test_kernel_return_types(kernel, args, return_type):
    kernel.compile(args)
    assert kernel.overloads[args].signature.return_type == return_type


@pytest.mark.parametrize(
    "kernel,args,offset_type",
    [
        (rp_trees.euclidean_random_projection_split, (F4_2D, I4, I8), numba.float64),
        (
            rp_trees.euclidean_hub_split,
            (F4_2D, I4, I4_2D, I4, I8),
            numba.float32,
        ),
    ],
)
def test_split_offset_types(kernel, args, offset_type):
    kernel.compile(args)
    assert kernel.overloads[args].signature.return_type[3] == offset_type


def _dense_data():
    return np.random.default_rng(0).normal(size=(300, 8)).astype(np.float32)


def _bit_data():
    return np.random.default_rng(0).integers(0, 256, size=(300, 8), dtype=np.uint8)


@pytest.mark.parametrize(
    "make_data,make_tree,angular",
    [
        (_dense_data, rp_trees.make_dense_tree, False),
        (_dense_data, rp_trees.make_dense_tree, True),
        (_bit_data, rp_trees.make_dense_bit_tree, True),
    ],
)
def test_tree_leaves_cover_all_points(make_data, make_tree, angular):
    rng_state = np.array([42, 12345, 67890], dtype=np.int64)
    tree = make_tree(make_data(), rng_state, leaf_size=10, angular=angular)
    leaves = np.concatenate(list(tree.indices))
    assert sum((leaf >= 0).any() for leaf in tree.indices) >= 2
    assert np.array_equal(np.sort(leaves[leaves >= 0]), np.arange(300))
