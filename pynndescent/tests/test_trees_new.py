"""Tests for random projection trees (rp_trees) and the threaded hyperplane helpers.

Expected values come from hand-computed tiny cases or from short numpy
reference implementations; partitions and hyperplane sides are checked against
the data directly.
"""

import itertools

import numpy as np
import pytest
import scipy.sparse as sp
import numba

import pynndescent.rp_trees as rpt
from pynndescent.rp_trees import (
	EPS,
	FlatTree,
	angular_bitpacked_random_projection_split,
	angular_random_projection_split,
	binary_search,
	compute_global_degrees,
	convert_tree_format,
	denumbaify_tree,
	euclidean_random_projection_split,
	get_leaves_from_tree,
	get_top_k_hub_indices,
	get_top_k_hub_indices_bit,
	make_dense_bit_tree,
	make_dense_tree,
	make_forest,
	make_sparse_tree,
	num_nodes_and_leaves,
	renumbaify_tree,
	rptree_leaf_array,
	rptree_leaf_array_parallel,
	score_linked_tree,
	score_tree,
	search_flat_bit_tree,
	search_flat_tree,
	search_sparse_flat_tree,
	select_side,
	select_side_bit,
	sparse_angular_random_projection_split,
	sparse_euclidean_random_projection_split,
	sparse_hyperplanes_to_ragged,
	sparse_select_side,
	sparse_select_side_ragged,
)
from pynndescent.threaded_rp_trees import (
	apply_hyperplane,
	make_angular_hyperplane,
	make_euclidean_hyperplane,
)

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def _rng_state(seed=0):
	rng = np.random.RandomState(seed)
	return rng.randint(-(2**31) + 1, 2**31 - 1, size=3).astype(np.int64)


def _dense(n=400, d=8, seed=0):
	return np.random.RandomState(seed).randn(n, d).astype(np.float32)


def _sparse(n=400, d=40, seed=0, density=0.3):
	m = sp.random(n, d, density=density, format="csr", dtype=np.float32, random_state=seed)
	m.data = m.data + np.float32(0.5)
	m.sort_indices()
	m.eliminate_zeros()
	return m


def _bits(n=400, d=6, seed=0):
	return np.random.RandomState(seed).randint(0, 256, size=(n, d)).astype(np.uint8)


def _margins(kind, plane, offset, X):
	"""Margins of all rows of X (dense float64 array, or uint8 array for bits)."""
	plane = np.asarray(plane)
	if kind == "dense":
		return offset + X.astype(np.float64) @ plane.astype(np.float64)
	if kind == "sparse":
		inds = plane[0]
		keep = inds >= 0
		return offset + X[:, inds[keep].astype(np.int64)].astype(np.float64) @ plane[1][keep].astype(np.float64)
	dim = X.shape[1]
	pos = POPCOUNT[plane[:dim][None, :] & X].sum(axis=1)
	neg = POPCOUNT[plane[dim:][None, :] & X].sum(axis=1)
	return offset + (pos - neg).astype(np.float64)


def _dense_view(data, kind):
	return data.toarray() if kind == "sparse" else data


def _check_split_sides(kind, data, indices, left, right, plane, offset, tol=1e-3):
	"""Every point sent left has margin >= -tol and every point sent right <= tol."""
	X = _dense_view(data, kind)
	assert np.array_equal(
		np.sort(np.concatenate([left, right])), np.sort(indices)
	), "left and right must partition the node"
	if len(left):
		assert np.all(_margins(kind, plane, offset, X[left]) >= -tol)
	if len(right):
		assert np.all(_margins(kind, plane, offset, X[right]) <= tol)


# ---------------------------------------------------------------- linked trees


def _linked(tree):
	children = [(int(a), int(b)) for a, b in tree.children]
	indices = [np.asarray(x) for x in tree.indices]
	hyperplanes = [np.asarray(x) for x in tree.hyperplanes]
	offsets = [float(x) for x in tree.offsets]
	return children, indices, hyperplanes, offsets


def _check_linked(tree, data, kind, leaf_size, tol=1e-3, bounded=True):
	"""Validate a linked (list based) tree; return the list of leaf index arrays."""
	n = data.shape[0]
	X = _dense_view(data, kind)
	children, indices, hyperplanes, offsets = _linked(tree)
	root = len(children) - 1
	visited = set()
	leaves = []

	def walk(t):
		assert t not in visited, "node reached twice"
		visited.add(t)
		left, right = children[t]
		if left == -1 and right == -1:
			leaves.append(indices[t])
			return indices[t]
		assert 0 <= left < t and 0 <= right < t and left != right
		below_left = walk(left)
		below_right = walk(right)
		if len(below_left):
			assert np.all(_margins(kind, hyperplanes[t], offsets[t], X[below_left]) >= -tol)
		if len(below_right):
			assert np.all(_margins(kind, hyperplanes[t], offsets[t], X[below_right]) <= tol)
		return np.concatenate([below_left, below_right])

	everything = walk(root)
	assert len(visited) == len(children)
	assert np.array_equal(np.sort(everything), np.arange(n))
	assert np.array_equal(np.sort(np.concatenate(leaves)), np.arange(n))
	largest = max(len(x) for x in leaves)
	if bounded:
		assert largest <= leaf_size
	assert tree.leaf_size == max(leaf_size, largest)
	return leaves


def _check_flat_structure(tree, n):
	children = np.asarray(tree.children)
	assert np.array_equal(np.sort(tree.indices), np.arange(n))
	is_leaf = children[:, 0] <= 0
	bounds = -children[is_leaf]
	bounds = bounds[np.argsort(bounds[:, 0])]
	assert bounds[0, 0] == 0 and bounds[-1, 1] == n
	assert np.all(bounds[1:, 0] == bounds[:-1, 1])
	assert np.all(bounds[:, 1] > bounds[:, 0])
	internal = np.flatnonzero(~is_leaf)
	assert np.all(children[internal] > internal[:, None])
	assert np.all(children[internal] < children.shape[0])


def _flat_plane(tree, node):
	hyperplanes = tree.hyperplanes
	if isinstance(hyperplanes, tuple):
		ptr, inds, vals = hyperplanes
		lo, hi = ptr[node], ptr[node + 1]
		return np.vstack([inds[lo:hi], vals[lo:hi]])
	return hyperplanes[node]


def _check_flat_sides(tree, data, kind, tol=1e-3):
	"""Hyperplane side consistency for every internal node of a flat tree."""
	X = _dense_view(data, kind)
	children = np.asarray(tree.children)

	def walk(node):
		if children[node, 0] <= 0:
			return tree.indices[-children[node, 0] : -children[node, 1]]
		below_left = walk(children[node, 0])
		below_right = walk(children[node, 1])
		plane = _flat_plane(tree, node)
		offset = float(tree.offsets[node])
		if len(below_left):
			assert np.all(_margins(kind, plane, offset, X[below_left]) >= -tol)
		if len(below_right):
			assert np.all(_margins(kind, plane, offset, X[below_right]) <= tol)
		return np.concatenate([below_left, below_right])

	assert np.array_equal(np.sort(walk(0)), np.arange(X.shape[0]))


def _route(kind, tree, data, i, rng_state):
	if kind == "dense":
		return search_flat_tree(
			data[i], tree.hyperplanes, tree.offsets, tree.children, tree.indices, rng_state
		)
	if kind == "bit":
		return search_flat_bit_tree(
			data[i], tree.hyperplanes, tree.offsets, tree.children, tree.indices, rng_state
		)
	row = data[i]
	return search_sparse_flat_tree(
		row.indices, row.data, tree.hyperplanes, tree.offsets, tree.children, tree.indices, rng_state
	)


def _make_tree(kind, data, rng_state, leaf_size, angular, max_depth=200):
	if kind == "dense":
		return make_dense_tree(data, rng_state, leaf_size, angular, max_depth=max_depth)
	if kind == "sparse":
		return make_sparse_tree(
			data.indices, data.indptr, data.data, rng_state, leaf_size, angular, max_depth=max_depth
		)
	return make_dense_bit_tree(data, rng_state, leaf_size, True, max_depth=max_depth)


def _make_data(kind, **kw):
	return {"dense": _dense, "sparse": _sparse, "bit": _bits}[kind](**kw)


KINDS = [
	("dense", False),
	("dense", True),
	("sparse", False),
	("sparse", True),
	("bit", True),
]
KIND_IDS = ["dense-euclid", "dense-angular", "sparse-euclid", "sparse-angular", "bit-angular"]


# ------------------------------------------------------------ single splits


def test_euclidean_split_hand_case():
	"""Two points: the hyperplane is the perpendicular bisector, one point per side."""
	data = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
	indices = np.array([0, 1], dtype=np.int32)
	left, right, plane, offset = euclidean_random_projection_split(data, indices, _rng_state(1))
	assert plane.dtype == np.float32 and plane.shape == (2,)
	assert len(left) == 1 and len(right) == 1
	if left[0] == 0:
		expected_plane, expected_offset = np.array([-2.0, 0.0]), 2.0
	else:
		expected_plane, expected_offset = np.array([2.0, 0.0]), -2.0
	np.testing.assert_allclose(plane, expected_plane)
	assert offset == pytest.approx(expected_offset)
	assert plane[0] * data[left[0], 0] + offset > 0


def test_euclidean_split_is_bisector_of_two_node_points():
	data = _dense(60, 5, seed=3)
	indices = np.arange(10, 50, dtype=np.int32)
	left, right, plane, offset = euclidean_random_projection_split(data, indices, _rng_state(2))
	found = False
	for a, b in itertools.permutations(indices, 2):
		if np.allclose(plane, data[a] - data[b], atol=1e-6):
			found = True
			midpoint = (data[a].astype(np.float64) + data[b]) / 2
			assert offset == pytest.approx(-plane.astype(np.float64) @ midpoint, abs=1e-4)
			break
	assert found, "hyperplane normal must be the difference of two node points"
	_check_split_sides("dense", data, indices, left, right, plane, offset)
	assert set(left) <= set(indices) and set(right) <= set(indices)
	assert left.dtype == np.int32 and right.dtype == np.int32


def test_angular_split_unit_normal_zero_offset():
	data = _dense(80, 6, seed=4)
	indices = np.arange(5, 75, dtype=np.int32)
	left, right, plane, offset = angular_random_projection_split(data, indices, _rng_state(3))
	assert offset == 0.0
	assert np.linalg.norm(plane) == pytest.approx(1.0, abs=1e-5)
	_check_split_sides("dense", data, indices, left, right, plane, offset)
	unit = data / np.linalg.norm(data, axis=1, keepdims=True)
	found = False
	for a, b in itertools.permutations(indices, 2):
		v = unit[a] - unit[b]
		if np.allclose(plane, v / np.linalg.norm(v), atol=1e-4):
			found = True
			break
	assert found


def test_angular_split_zero_row_does_not_produce_nan():
	data = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)
	indices = np.array([0, 1], dtype=np.int32)
	left, right, plane, offset = angular_random_projection_split(data, indices, _rng_state(5))
	assert np.all(np.isfinite(plane))
	assert len(left) + len(right) == 2
	assert np.linalg.norm(plane) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bitpacked_angular_split(seed):
	data = _bits(120, 5, seed=seed)
	indices = np.arange(0, 120, 2, dtype=np.int32)
	left, right, plane, offset = angular_bitpacked_random_projection_split(
		data, indices, _rng_state(seed)
	)
	dim = data.shape[1]
	assert plane.shape == (2 * dim,) and plane.dtype == np.uint8
	assert offset == 0.0
	assert np.all(plane[:dim] & plane[dim:] == 0), "positive and negative bits are disjoint"
	_check_split_sides("bit", data, indices, left, right, plane, offset, tol=0)
	found = False
	for a, b in itertools.permutations(indices, 2):
		x = data[a] ^ data[b]
		if np.array_equal(plane[:dim], x & data[a]) and np.array_equal(plane[dim:], x & data[b]):
			found = True
			break
	assert found


def test_sparse_euclidean_split():
	data = _sparse(120, 30, seed=2)
	indices = np.arange(10, 110, dtype=np.int32)
	left, right, plane, offset = sparse_euclidean_random_projection_split(
		data.indices, data.indptr, data.data, indices, _rng_state(4)
	)
	assert plane.shape[0] == 2
	_check_split_sides("sparse", data, indices, left, right, plane, offset)
	X = data.toarray().astype(np.float64)
	found = False
	for a, b in itertools.permutations(indices, 2):
		v = X[a] - X[b]
		got = np.zeros(X.shape[1])
		keep = plane[0] >= 0
		got[plane[0][keep].astype(int)] = plane[1][keep]
		if np.allclose(got, v, atol=1e-5):
			found = True
			assert offset == pytest.approx(-v @ (X[a] + X[b]) / 2, abs=1e-4)
			break
	assert found


def test_sparse_angular_split():
	data = _sparse(120, 30, seed=5)
	indices = np.arange(0, 120, dtype=np.int32)
	left, right, plane, offset = sparse_angular_random_projection_split(
		data.indices, data.indptr, data.data, indices, _rng_state(6)
	)
	assert offset == 0.0
	assert np.linalg.norm(plane[1]) == pytest.approx(1.0, abs=1e-4)
	_check_split_sides("sparse", data, indices, left, right, plane, offset)


@pytest.mark.parametrize(
	"kind,angular", KINDS, ids=KIND_IDS
)
def test_split_deterministic_for_fixed_rng_state(kind, angular):
	data = _make_data(kind, n=100)
	indices = np.arange(100, dtype=np.int32)

	def split(state):
		if kind == "dense":
			f = angular_random_projection_split if angular else euclidean_random_projection_split
			return f(data, indices, state)
		if kind == "bit":
			return angular_bitpacked_random_projection_split(data, indices, state)
		f = sparse_angular_random_projection_split if angular else sparse_euclidean_random_projection_split
		return f(data.indices, data.indptr, data.data, indices, state)

	a = split(_rng_state(11))
	b = split(_rng_state(11))
	for x, y in zip(a, b):
		np.testing.assert_array_equal(x, y)
	others = [split(_rng_state(s)) for s in range(12, 18)]
	assert any(not np.array_equal(o[0], a[0]) for o in others), "seed must influence the split"


@pytest.mark.parametrize("fn", [euclidean_random_projection_split, angular_random_projection_split])
def test_split_identical_points_still_partitions(fn):
	data = np.ones((40, 3), dtype=np.float32)
	indices = np.arange(40, dtype=np.int32)
	left, right, plane, offset = fn(data, indices, _rng_state(9))
	assert len(left) + len(right) == 40
	assert np.array_equal(np.sort(np.concatenate([left, right])), indices)


# -------------------------------------------------------- small helper kernels


def test_binary_search_matches_searchsorted():
	arr = np.array([-5, -1, 0, 3, 8, 12, 40], dtype=np.int32)
	for v in arr:
		pos = binary_search(arr, np.int32(v))
		assert pos == np.searchsorted(arr, v)
	for v in [-10, -2, 1, 5, 13, 100]:
		assert binary_search(arr, np.int32(v)) == -1


def test_binary_search_edge_cases():
	assert binary_search(np.array([], dtype=np.int32), np.int32(3)) == -1
	one = np.array([7], dtype=np.int32)
	assert binary_search(one, np.int32(7)) == 0
	assert binary_search(one, np.int32(6)) == -1
	dup = np.array([1, 2, 2, 2, 3], dtype=np.int32)
	assert dup[binary_search(dup, np.int32(2))] == 2


def test_compute_global_degrees_hand_case():
	graph = np.array([[1, 2], [0, 2], [1, 1], [2, -1]], dtype=np.int32)
	degrees = compute_global_degrees(graph)
	assert degrees.dtype == np.int32
	np.testing.assert_array_equal(degrees, [1, 3, 3, 0])


def test_compute_global_degrees_matches_bincount_and_ignores_out_of_range():
	rng = np.random.RandomState(0)
	graph = rng.randint(0, 50, size=(50, 7)).astype(np.int32)
	np.testing.assert_array_equal(
		compute_global_degrees(graph), np.bincount(graph.ravel(), minlength=50)
	)
	graph[0, 0] = 50
	graph[1, 1] = -1
	valid = graph[(graph >= 0) & (graph < 50)]
	np.testing.assert_array_equal(
		compute_global_degrees(graph), np.bincount(valid, minlength=50)
	)


@pytest.mark.parametrize("bit", [False, True])
def test_get_top_k_hub_indices_orders_by_degree(bit):
	rng = np.random.RandomState(1)
	degrees = rng.permutation(100).astype(np.int32)
	indices = np.sort(rng.choice(100, size=40, replace=False)).astype(np.int32)
	data = np.zeros((100, 2), dtype=np.uint8)
	for k in (1, 3, 5):
		if bit:
			top = get_top_k_hub_indices_bit(indices, data, degrees, k)
		else:
			top = get_top_k_hub_indices(indices, degrees, k)
		expected = indices[np.argsort(-degrees[indices], kind="stable")[:k]]
		np.testing.assert_array_equal(top, expected)


@pytest.mark.parametrize("bit", [False, True])
def test_get_top_k_hub_indices_small_node(bit):
	degrees = np.array([5, 1, 9, 3], dtype=np.int32)
	indices = np.array([3, 1], dtype=np.int32)
	data = np.zeros((4, 1), dtype=np.uint8)
	if bit:
		top = get_top_k_hub_indices_bit(indices, data, degrees, 3)
	else:
		top = get_top_k_hub_indices(indices, degrees, 5)
	np.testing.assert_array_equal(top, [3, 1])


# ------------------------------------------------------------- select_side


def test_select_side_hand_cases():
	plane = np.array([1.0, 0.0], dtype=np.float32)
	state = _rng_state(0)
	assert select_side(plane, np.float32(-1.0), np.array([3.0, 5.0], dtype=np.float32), state) == 0
	assert select_side(plane, np.float32(-1.0), np.array([0.0, 5.0], dtype=np.float32), state) == 1
	assert select_side(plane, np.float32(2.5), np.array([-2.0, 0.0], dtype=np.float32), state) == 0


def test_select_side_tie_is_random_but_reproducible():
	plane = np.array([1.0, 0.0], dtype=np.float32)
	point = np.array([1.0, 9.0], dtype=np.float32)
	state = _rng_state(3)
	sides = [bool(select_side(plane, np.float32(-1.0), point, state)) for _ in range(64)]
	assert set(sides) == {False, True}
	state = _rng_state(3)
	again = [bool(select_side(plane, np.float32(-1.0), point, state)) for _ in range(64)]
	assert sides == again


def test_select_side_bit_hand_cases():
	plane = np.array([0b0011, 0b1100], dtype=np.uint8)
	state = _rng_state(0)
	zero = np.float32(0.0)
	assert select_side_bit(plane, zero, np.array([0b0001], dtype=np.uint8), state) == 0
	assert select_side_bit(plane, zero, np.array([0b0100], dtype=np.uint8), state) == 1
	assert select_side_bit(plane, zero, np.array([0b1110], dtype=np.uint8), state) == 1
	assert select_side_bit(plane, zero, np.array([0b0011], dtype=np.uint8), state) == 0
	assert select_side_bit(plane, np.float32(5.0), np.array([0b1100], dtype=np.uint8), state) == 0
	tie = [select_side_bit(plane, zero, np.array([0], dtype=np.uint8), state) for _ in range(64)]
	assert set(map(bool, tie)) == {False, True}


def test_sparse_select_side_and_ragged_agree_with_hand_values():
	state = _rng_state(0)
	padded = np.array([[0.0, 2.0, -1.0], [1.0, -1.0, 0.0]], dtype=np.float32)
	p_inds = np.array([0, 2], dtype=np.int32)
	p_vals = np.array([2.0, 1.0], dtype=np.float32)
	assert sparse_select_side(padded, 0.5, p_inds, p_vals, state) == 0
	q_vals = np.array([-2.0, 1.0], dtype=np.float32)
	assert sparse_select_side(padded, 0.5, p_inds, q_vals, state) == 1
	h_inds = np.array([0, 2], dtype=np.int32)
	h_vals = np.array([1.0, -1.0], dtype=np.float32)
	assert sparse_select_side_ragged(h_inds, h_vals, 0.5, p_inds, p_vals, state) == 0
	assert sparse_select_side_ragged(h_inds, h_vals, 0.5, p_inds, q_vals, state) == 1
	assert sparse_select_side(padded, 0.5, np.array([5], dtype=np.int32), np.array([9.0], dtype=np.float32), state) == 0
	assert sparse_select_side(padded, -0.5, np.array([5], dtype=np.int32), np.array([9.0], dtype=np.float32), state) == 1


def test_sparse_hyperplanes_to_ragged_hand_case():
	padded = np.full((3, 2, 3), 0.0, dtype=np.float32)
	padded[:, 0, :] = -1
	padded[0, 0, :2] = [0, 2]
	padded[0, 1, :2] = [1.0, -1.0]
	padded[2, 0, :3] = [1, 4, 5]
	padded[2, 1, :3] = [0.5, 0.25, -2.0]
	ptr, inds, vals = sparse_hyperplanes_to_ragged(padded)
	np.testing.assert_array_equal(ptr, [0, 2, 2, 5])
	np.testing.assert_array_equal(inds, [0, 2, 1, 4, 5])
	np.testing.assert_array_equal(vals, [1.0, -1.0, 0.5, 0.25, -2.0])
	assert inds.dtype == np.int32 and vals.dtype == np.float32
	as_tuple = (ptr, inds, vals)
	assert sparse_hyperplanes_to_ragged(as_tuple) is as_tuple


# ----------------------------------------------------------- flat tree search


def _hand_flat_tree():
	hyperplanes = np.array([[1.0], [0.0], [0.0]], dtype=np.float32)
	offsets = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
	children = np.array([[1, 2], [0, -2], [-2, -4]], dtype=np.int32)
	indices = np.array([10, 11, 12, 13], dtype=np.int32)
	return hyperplanes, offsets, children, indices


def test_search_flat_tree_hand_case():
	hyperplanes, offsets, children, indices = _hand_flat_tree()
	state = _rng_state(0)
	left = search_flat_tree(np.array([3.0], dtype=np.float32), hyperplanes, offsets, children, indices, state)
	right = search_flat_tree(np.array([0.0], dtype=np.float32), hyperplanes, offsets, children, indices, state)
	np.testing.assert_array_equal(left, [10, 11])
	np.testing.assert_array_equal(right, [12, 13])


def test_search_flat_bit_tree_hand_case():
	hyperplanes = np.array([[0b0011, 0b1100], [0, 0], [0, 0]], dtype=np.uint8)
	offsets = np.zeros(3, dtype=np.float32)
	children = np.array([[1, 2], [0, -2], [-2, -4]], dtype=np.int32)
	indices = np.array([10, 11, 12, 13], dtype=np.int32)
	state = _rng_state(0)
	a = search_flat_bit_tree(np.array([0b0001], dtype=np.uint8), hyperplanes, offsets, children, indices, state)
	b = search_flat_bit_tree(np.array([0b0100], dtype=np.uint8), hyperplanes, offsets, children, indices, state)
	np.testing.assert_array_equal(a, [10, 11])
	np.testing.assert_array_equal(b, [12, 13])


def test_search_sparse_flat_tree_hand_case():
	hyperplanes = np.zeros((3, 2, 1), dtype=np.float32)
	hyperplanes[:, 0, :] = -1
	hyperplanes[0, 0, 0] = 0
	hyperplanes[0, 1, 0] = 1.0
	offsets = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
	children = np.array([[1, 2], [0, -2], [-2, -4]], dtype=np.int32)
	indices = np.array([10, 11, 12, 13], dtype=np.int32)
	state = _rng_state(0)
	inds = np.array([0], dtype=np.int32)
	hi = search_sparse_flat_tree(inds, np.array([3.0], dtype=np.float32), hyperplanes, offsets, children, indices, state)
	lo = search_sparse_flat_tree(inds, np.array([0.5], dtype=np.float32), hyperplanes, offsets, children, indices, state)
	np.testing.assert_array_equal(hi, [10, 11])
	np.testing.assert_array_equal(lo, [12, 13])


# ------------------------------------------------------------- tree builders


@pytest.mark.parametrize("kind,angular", KINDS, ids=KIND_IDS)
@pytest.mark.parametrize("leaf_size", [10, 40])
def test_random_tree_is_valid_partition(kind, angular, leaf_size):
	data = _make_data(kind, n=300)
	tree = _make_tree(kind, data, _rng_state(1), leaf_size, angular)
	_check_linked(tree, data, kind, leaf_size)
	assert isinstance(tree, FlatTree)


@pytest.mark.parametrize("kind,angular", KINDS, ids=KIND_IDS)
def test_random_tree_max_depth(kind, angular):
	data = _make_data(kind, n=200)
	tree0 = _make_tree(kind, data, _rng_state(1), 5, angular, max_depth=0)
	leaves = _check_linked(tree0, data, kind, 5, bounded=False)
	assert len(leaves) == 1 and len(leaves[0]) == 200
	assert tree0.leaf_size == 200
	tree1 = _make_tree(kind, data, _rng_state(1), 5, angular, max_depth=1)
	leaves = _check_linked(tree1, data, kind, 5, bounded=False)
	assert len(leaves) == 2
	assert len(tree1.children) == 3


@pytest.mark.parametrize("kind,angular", KINDS, ids=KIND_IDS)
def test_random_tree_small_data_is_single_leaf(kind, angular):
	data = _make_data(kind, n=20)
	tree = _make_tree(kind, data, _rng_state(1), 30, angular)
	leaves = _check_linked(tree, data, kind, 30)
	assert len(leaves) == 1
	assert len(tree.children) == 1
	assert tree.leaf_size == 30


@pytest.mark.parametrize("kind,angular", KINDS, ids=KIND_IDS)
def test_random_tree_deterministic_and_seed_sensitive(kind, angular):
	data = _make_data(kind, n=250)
	a = _make_tree(kind, data, _rng_state(7), 20, angular)
	b = _make_tree(kind, data, _rng_state(7), 20, angular)
	assert [tuple(c) for c in a.children] == [tuple(c) for c in b.children]
	for x, y in zip(a.indices, b.indices):
		np.testing.assert_array_equal(x, y)
	for x, y in zip(a.offsets, b.offsets):
		assert x == y or (np.isinf(x) and np.isinf(y))
	others = [_make_tree(kind, data, _rng_state(s), 20, angular) for s in range(8, 12)]
	assert any(
		not np.array_equal(np.concatenate(list(o.indices)), np.concatenate(list(a.indices)))
		for o in others
	)


def test_dense_bit_tree_euclidean_not_implemented():
	data = _bits(100, 4)
	with pytest.raises(NotImplementedError):
		make_dense_bit_tree(data, _rng_state(0), 10, False)


def test_random_tree_duplicate_points_terminates():
	data = np.ones((120, 4), dtype=np.float32)
	tree = make_dense_tree(data, _rng_state(0), 10, False, max_depth=40)
	children, indices, _, _ = _linked(tree)
	leaf_points = [x for x, c in zip(indices, children) if c == (-1, -1)]
	assert np.array_equal(np.sort(np.concatenate(leaf_points)), np.arange(120))


@pytest.mark.parametrize("kind,angular", KINDS, ids=KIND_IDS)
def test_converted_tree_matches_linked_tree(kind, angular):
	data = _make_data(kind, n=300)
	tree = _make_tree(kind, data, _rng_state(2), 15, angular)
	flat = convert_tree_format(tree, data.shape[0], data.shape[1])
	_check_flat_structure(flat, 300)
	assert flat.leaf_size == tree.leaf_size
	assert flat.offsets.dtype == np.float32
	assert flat.children.dtype == np.int32 and flat.indices.dtype == np.int32
	children, indices, hyperplanes, offsets = _linked(tree)
	fchildren = np.asarray(flat.children)

	def compare(t, node):
		if children[t] == (-1, -1):
			start, end = -fchildren[node]
			np.testing.assert_array_equal(flat.indices[start:end], indices[t])
			return node
		assert fchildren[node, 0] == node + 1
		fplane = _flat_plane(flat, node)
		if kind == "sparse":
			inds, vals = hyperplanes[t][0], hyperplanes[t][1]
			keep = fplane[0] >= 0
			np.testing.assert_array_equal(fplane[0][keep], inds)
			np.testing.assert_allclose(fplane[1][keep], vals, rtol=1e-6)
		else:
			np.testing.assert_array_equal(fplane, hyperplanes[t])
		assert flat.offsets[node] == pytest.approx(np.float32(offsets[t]))
		last_left = compare(children[t][0], node + 1)
		assert fchildren[node, 1] == last_left + 1
		return compare(children[t][1], last_left + 1)

	last = compare(len(children) - 1, 0)
	assert last == len(children) - 1
	_check_flat_sides(flat, data, kind)


def test_converted_dense_hyperplane_shapes():
	data = _dense(200, 7)
	tree = make_dense_tree(data, _rng_state(0), 20, False)
	flat = convert_tree_format(tree, 200, 7)
	n_nodes, n_leaves = num_nodes_and_leaves(tree)
	assert flat.hyperplanes.shape == (n_nodes, 7)
	bit_tree = make_dense_bit_tree(_bits(200, 3), _rng_state(0), 20, True)
	assert convert_tree_format(bit_tree, 200, 3).hyperplanes.shape[1] == 6


def test_num_nodes_and_leaves():
	data = _dense(300, 5)
	tree = make_dense_tree(data, _rng_state(0), 25, False)
	n_nodes, n_leaves = num_nodes_and_leaves(tree)
	children, _, _, _ = _linked(tree)
	assert n_nodes == len(children)
	assert n_leaves == sum(c == (-1, -1) for c in children)
	assert n_nodes == 2 * n_leaves - 1
	single = make_dense_tree(data[:10].copy(), _rng_state(0), 25, False)
	assert tuple(num_nodes_and_leaves(single)) == (1, 1)


@pytest.mark.parametrize("kind,angular", KINDS, ids=KIND_IDS)
def test_search_flat_tree_returns_leaf_of_training_point(kind, angular):
	data = _make_data(kind, n=300)
	tree = _make_tree(kind, data, _rng_state(3), 15, angular)
	flat = convert_tree_format(tree, 300, data.shape[1])
	state = _rng_state(4)
	X = _dense_view(data, kind)
	children = np.asarray(flat.children)
	checked = 0
	for i in range(300):
		leaf = _route(kind, flat, data, i, state)
		assert len(leaf) <= tree.leaf_size
		node, tied = 0, False
		while children[node, 0] > 0:
			m = _margins(kind, _flat_plane(flat, node), float(flat.offsets[node]), X[i : i + 1])[0]
			tied |= abs(m) < 1e-4
			node = children[node, 0] if m > 0 else children[node, 1]
		if tied:
			continue
		checked += 1
		np.testing.assert_array_equal(leaf, flat.indices[-children[node, 0] : -children[node, 1]])
		assert i in leaf
	assert checked >= (50 if kind == "bit" else 290)


def test_denumbaify_renumbaify_roundtrip():
	data = _dense(120, 4)
	flat = convert_tree_format(make_dense_tree(data, _rng_state(0), 20, False), 120, 4)
	as_tuple = denumbaify_tree(flat)
	assert isinstance(as_tuple, tuple) and len(as_tuple) == 5
	assert as_tuple[rpt.FLAT_TREE_HYPERPLANES] is flat.hyperplanes
	assert as_tuple[rpt.FLAT_TREE_OFFSETS] is flat.offsets
	assert as_tuple[rpt.FLAT_TREE_CHILDREN] is flat.children
	assert as_tuple[rpt.FLAT_TREE_INDICES] is flat.indices
	assert as_tuple[rpt.FLAT_TREE_LEAF_SIZE] == flat.leaf_size
	back = renumbaify_tree(as_tuple)
	assert isinstance(back, FlatTree)
	for name in FlatTree._fields:
		a, b = getattr(flat, name), getattr(back, name)
		np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------- scoring


def _reference_linked_score(tree, graph):
	children, indices, _, _ = _linked(tree)
	total = 0.0
	k = graph.shape[1]
	for c, idx in zip(children, indices):
		if c != (-1, -1):
			continue
		members = set(int(i) for i in idx)
		for i in idx:
			total += sum(int(nb) in members for nb in graph[i]) / k
	return total / graph.shape[0]


def _brute_knn(data, k):
	X = _dense_view(data, "sparse" if sp.issparse(data) else "dense").astype(np.float64)
	d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
	np.fill_diagonal(d, np.inf)
	return np.argsort(d, axis=1)[:, :k].astype(np.int32)


def test_score_linked_tree_hand_case():
	graph = np.array([[1, 2], [0, 3], [3, 0], [2, 1]], dtype=np.int32)
	data = _dense(4, 2)
	single = make_dense_tree(data, _rng_state(0), 10, False)
	assert score_linked_tree(single, graph) == pytest.approx(1.0)
	with_missing = graph.copy()
	with_missing[0, 1] = -1
	assert score_linked_tree(single, with_missing) == pytest.approx(7 / 8)


@pytest.mark.parametrize("leaf_size", [5, 20])
def test_score_linked_tree_matches_reference(leaf_size):
	data = _dense(200, 5, seed=2)
	graph = _brute_knn(data, 6)
	tree = make_dense_tree(data, _rng_state(1), leaf_size, False)
	score = score_linked_tree(tree, graph)
	assert 0.0 <= score <= 1.0
	assert score == pytest.approx(_reference_linked_score(tree, graph), abs=1e-5)


def test_score_linked_tree_prefers_larger_leaves():
	data = _dense(200, 5, seed=2)
	graph = _brute_knn(data, 6)
	small = score_linked_tree(make_dense_tree(data, _rng_state(1), 5, False), graph)
	big = score_linked_tree(make_dense_tree(data, _rng_state(1), 100, False), graph)
	assert big >= small


def test_score_tree_matches_reference():
	"""Recomputes the score from search_flat_tree leaves with the code's rule: a
	point counts when its leaf holds more than one of its neighbours. Pins the
	current output."""
	data = _dense(250, 5, seed=3)
	graph = _brute_knn(data, 8).astype(np.int64)
	flat = convert_tree_format(make_dense_tree(data, _rng_state(1), 25, False), 250, 5)
	state = _rng_state(2)
	got = score_tree(flat, graph, data, state)
	expected = 0.0
	for i in range(250):
		leaf = np.asarray(search_flat_tree(data[i], flat.hyperplanes, flat.offsets, flat.children, flat.indices, _rng_state(2)))
		expected += len(set(graph[i].tolist()) & set(leaf.tolist())) > 1
	assert 0.0 <= got <= 1.0
	assert got == pytest.approx(expected / 250, abs=1e-6)


def test_score_tree_counts_a_point_only_with_two_leaf_neighbours():
	"""Hand case: one leaf holding points 0-3. A point scores when at least two of
	its listed neighbours are in its leaf; neighbours outside the leaf (7, 9) do
	not count, so rows with one in-leaf neighbour score 0 and the total is 2/4."""
	data = _dense(4, 2)
	flat = convert_tree_format(make_dense_tree(data, _rng_state(1), 100, False), 4, 2)
	assert sorted(flat.indices.tolist()) == [0, 1, 2, 3]
	graph = np.array([[1, 7], [0, 2], [3, 9], [0, 1]], dtype=np.int64)
	assert score_tree(flat, graph, data, _rng_state(0)) == pytest.approx(0.5)
	graph_all = np.array([[1, 2], [0, 2], [3, 0], [0, 1]], dtype=np.int64)
	assert score_tree(flat, graph_all, data, _rng_state(0)) == pytest.approx(1.0)


def test_score_tree_single_leaf_is_one():
	data = _dense(30, 4)
	graph = _brute_knn(data, 5).astype(np.int64)
	flat = convert_tree_format(make_dense_tree(data, _rng_state(1), 100, False), 30, 4)
	assert score_tree(flat, graph, data, _rng_state(0)) == pytest.approx(1.0)


# ----------------------------------------------------------- leaf extraction


def test_get_leaves_from_tree():
	data = _dense(200, 5)
	tree = make_dense_tree(data, _rng_state(1), 20, False)
	leaves = get_leaves_from_tree(tree, tree.leaf_size)
	children, indices, _, _ = _linked(tree)
	expected = [idx for c, idx in zip(children, indices) if c == (-1, -1)]
	assert leaves.shape == (len(expected), tree.leaf_size)
	assert leaves.dtype == np.int32
	for row, idx in zip(leaves, expected):
		np.testing.assert_array_equal(row[: len(idx)], idx)
		assert np.all(row[len(idx) :] == -1)


def test_rptree_leaf_array_covers_every_point_once_per_tree():
	data = _dense(150, 4)
	trees = [make_dense_tree(data, _rng_state(s), 15, False) for s in range(3)]
	result = rptree_leaf_array(trees)
	assert result.dtype == np.int32
	assert result.shape[1] == max(t.leaf_size for t in trees)
	valid = result[result >= 0]
	counts = np.bincount(valid, minlength=150)
	assert np.all(counts == 3)
	rows = [row[row >= 0] for row in result]
	assert all(len(r) > 0 for r in rows)
	for row in result:
		pad = np.flatnonzero(row < 0)
		assert len(pad) == 0 or np.all(row[pad[0] :] == -1)


def test_rptree_leaf_array_parallel_returns_one_array_per_tree():
	data = _dense(150, 4)
	trees = [make_dense_tree(data, _rng_state(s), 15, False) for s in range(3)]
	parts = rptree_leaf_array_parallel(trees)
	assert len(parts) == 3
	for part, tree in zip(parts, trees):
		assert part.shape[1] == max(t.leaf_size for t in trees)
		assert np.array_equal(np.sort(part[part >= 0]), np.arange(150))


def test_rptree_leaf_array_empty_forest():
	result = rptree_leaf_array(())
	np.testing.assert_array_equal(result, np.array([[-1]]))


# ------------------------------------------------------------------- forest


@pytest.mark.parametrize("kind,angular", KINDS, ids=KIND_IDS)
def test_make_forest_trees_are_valid(kind, angular):
	data = _make_data(kind, n=250)
	forest = make_forest(
		data, 10, 3, 20, None, np.random.RandomState(0), n_jobs=1,
		angular=angular, bit_tree=(kind == "bit"),
	)
	assert isinstance(forest, tuple) and len(forest) == 3
	for tree in forest:
		_check_linked(tree, data, kind, 20)


def test_make_forest_deterministic_with_random_state():
	data = _dense(250, 5)
	a = make_forest(data, 10, 4, 20, None, np.random.RandomState(5), n_jobs=1)
	b = make_forest(data, 10, 4, 20, None, np.random.RandomState(5), n_jobs=1)
	c = make_forest(data, 10, 4, 20, None, np.random.RandomState(6), n_jobs=1)
	la, lb, lc = (rptree_leaf_array(f) for f in (a, b, c))
	np.testing.assert_array_equal(la, lb)
	assert la.shape != lc.shape or not np.array_equal(la, lc)


def test_make_forest_trees_differ_from_each_other():
	data = _dense(250, 5)
	forest = make_forest(data, 10, 4, 20, None, np.random.RandomState(5), n_jobs=1)
	first_splits = {tuple(np.asarray(t.hyperplanes[-1]).round(5)) for t in forest}
	assert len(first_splits) == 4


def test_make_forest_default_leaf_size_and_threads():
	"""leaf_size=None means max(60, min(256, 5 * n_neighbors)): 60 at 10 neighbours,
	256 at 100 (the formula is copied from make_forest, so this pins it)."""
	data = _dense(400, 5)
	for n_neighbors, n_jobs in ((10, 2), (100, 1)):
		expected = max(60, min(256, 5 * n_neighbors))
		forest = make_forest(
			data, n_neighbors, 2, None, None, np.random.RandomState(0), n_jobs=n_jobs
		)
		for tree in forest:
			_check_linked(tree, data, "dense", expected)
			assert tree.leaf_size == expected


def test_make_forest_accepts_csr_array_and_max_depth():
	matrix = _sparse(200, 30)
	array = sp.csr_array(matrix)
	forest = make_forest(array, 10, 2, 15, None, np.random.RandomState(0), n_jobs=1, max_depth=0)
	for tree in forest:
		assert len(tree.children) == 1
		assert tree.leaf_size == 200


def test_make_forest_zero_trees():
	assert make_forest(_dense(50, 3), 10, 0, 20, None, np.random.RandomState(0), n_jobs=1) == ()


# ------------------------------------------------------------ threaded helpers


def test_make_euclidean_hyperplane_hand_case():
	data = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
	indices = np.array([0, 1], dtype=np.int32)
	plane, offset = make_euclidean_hyperplane(data, indices, _rng_state(2))
	assert plane.dtype == np.float32
	if plane[0] > 0:
		np.testing.assert_allclose(plane, [3.0, -4.0])
		assert offset == pytest.approx(3.5)
	else:
		np.testing.assert_allclose(plane, [-3.0, 4.0])
		assert offset == pytest.approx(-3.5)


def test_make_euclidean_hyperplane_bisects_two_node_points():
	data = _dense(50, 5, seed=8)
	indices = np.arange(5, 45, dtype=np.int32)
	plane, offset = make_euclidean_hyperplane(data, indices, _rng_state(3))
	for a, b in itertools.permutations(indices, 2):
		if np.allclose(plane, data[a] - data[b], atol=1e-6):
			mid = (data[a].astype(np.float64) + data[b]) / 2
			assert offset == pytest.approx(-plane.astype(np.float64) @ mid, abs=1e-4)
			assert plane @ data[a] + offset > 0 > plane @ data[b] + offset
			return
	pytest.fail("no pair of node points reproduces the hyperplane")


def test_make_angular_hyperplane_hand_case():
	data = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
	indices = np.array([0, 1], dtype=np.int32)
	plane, offset = make_angular_hyperplane(data, indices, _rng_state(2))
	assert offset == 0.0
	if plane[0] > 0:
		np.testing.assert_allclose(plane, [1.0, -1.0], atol=1e-6)
	else:
		np.testing.assert_allclose(plane, [-1.0, 1.0], atol=1e-6)


def test_make_angular_hyperplane_zero_row():
	data = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)
	indices = np.array([0, 1], dtype=np.int32)
	plane, offset = make_angular_hyperplane(data, indices, _rng_state(2))
	assert abs(plane[0]) == 0 and abs(plane[1]) == pytest.approx(1.0)
	assert offset == 0.0


def test_apply_hyperplane_hand_case():
	data = np.array([[0.0], [1.5], [2.0], [3.0], [9.0]], dtype=np.float32)
	loc = np.array([0, 0, 0, 0, 7], dtype=np.int64)
	apply_hyperplane(
		data, np.array([1.0], dtype=np.float32), -1.0, 0, 8, loc, _rng_state(0)
	)
	np.testing.assert_array_equal(loc, [9, 8, 8, 8, 7])


def test_apply_hyperplane_tie_goes_to_a_child():
	data = np.array([[1.0], [1.0]], dtype=np.float32)
	loc = np.array([3, 3], dtype=np.int64)
	apply_hyperplane(data, np.array([1.0], dtype=np.float32), -1.0, 3, 10, loc, _rng_state(0))
	assert set(loc) <= {10, 11}


@pytest.mark.parametrize("angular", [False, True])
def test_threaded_builder_produces_valid_partition(angular):
	"""Grow a tree breadth-first with the threaded helpers and check the partition."""
	data = _dense(300, 6, seed=9)
	state = _rng_state(5)
	loc = np.zeros(300, dtype=np.int64)
	planes = {}
	n_nodes = 1
	frontier = [0]
	leaf_size = 20
	for _ in range(30):
		next_frontier = []
		for node in frontier:
			members = np.flatnonzero(loc == node).astype(np.int32)
			if len(members) <= leaf_size:
				continue
			make = make_angular_hyperplane if angular else make_euclidean_hyperplane
			plane, offset = make(data, members, state)
			apply_hyperplane(data, plane, offset, node, n_nodes, loc, state)
			planes[node] = (plane, offset, n_nodes, n_nodes + 1, members)
			next_frontier += [n_nodes, n_nodes + 1]
			n_nodes += 2
		frontier = next_frontier
		if not frontier:
			break
	assert not frontier
	sizes = np.bincount(loc, minlength=n_nodes)
	assert sizes.sum() == 300
	assert np.all(np.isin(np.unique(loc), [n for n in range(n_nodes) if n not in planes]))
	for node, (plane, offset, left, right, members) in planes.items():
		margin = offset + data[members].astype(np.float64) @ plane.astype(np.float64)
		in_left = np.isin(members, np.flatnonzero(_descendants(loc, planes, left)))
		in_right = np.isin(members, np.flatnonzero(_descendants(loc, planes, right)))
		assert np.all(in_left ^ in_right)
		assert np.all(margin[in_left] >= -1e-4)
		assert np.all(margin[in_right] <= 1e-4)


def _descendants(loc, planes, node):
	"""Boolean mask of points whose final leaf is node or below it."""
	mask = loc == node
	if node in planes:
		_, _, left, right, _ = planes[node]
		mask = _descendants(loc, planes, left) | _descendants(loc, planes, right)
	return mask
