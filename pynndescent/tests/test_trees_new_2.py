"""Tests for graph-informed hub trees, the hybrid leaf-only forest and the hub search tree.

Hub splits are compared with short numpy reference implementations of the
documented selection rule (top-3 in-degree hubs, three candidate hyperplanes,
best balance for dense data and fewest cut edges for sparse and bit data).
"""

import numba
import numpy as np
import pytest
import scipy.sparse as sp

from pynndescent.hub_tree import (
	CRITERION_BALANCE,
	CRITERION_EDGE_CUT,
	FALLBACK_LEAF,
	FALLBACK_MEDIAN,
	FALLBACK_RANDOM,
	make_hub_search_tree,
)
from pynndescent.hybrid_forest import make_hybrid_leaf_array
from pynndescent.rp_trees import (
	EPS,
	MIN_SPLIT_BALANCE,
	FlatTree,
	angular_hub_split,
	bit_hub_split,
	compute_global_degrees,
	convert_tree_format,
	euclidean_hub_split,
	make_bit_hub_tree,
	make_hub_tree,
	make_sparse_hub_tree,
	sparse_angular_hub_split,
	sparse_euclidean_hub_split,
)
from pynndescent.tests.test_trees_new import (
	POPCOUNT,
	_bits,
	_brute_knn,
	_check_flat_sides,
	_check_flat_structure,
	_check_linked,
	_dense,
	_dense_view,
	_flat_plane,
	_linked,
	_make_data,
	_margins,
	_rng_state,
	_route,
	_sparse,
)

HUB_KINDS = [
	("dense", False),
	("dense", True),
	("sparse", False),
	("sparse", True),
	("bit", False),
]
HUB_IDS = ["dense-euclid", "dense-angular", "sparse-euclid", "sparse-angular", "bit"]


def _graph(data, k=8, pad=True):
	graph = _brute_knn(data, k)
	if pad:
		graph[::7, -1] = -1
	return np.ascontiguousarray(graph, dtype=np.int32)


def _candidates(kind, angular, X, indices, degrees):
	"""The three hub-pair hyperplanes of the node, as (plane, offset) in numpy."""
	order = np.argsort(-degrees[indices], kind="stable")[:3]
	hubs = indices[order]
	out = []
	for i, j in [(0, 1), (0, 2), (1, 2)]:
		a, b = X[hubs[i]], X[hubs[j]]
		if kind == "bit":
			x = a ^ b
			out.append((np.concatenate([x & a, x & b]), 0.0))
		elif angular:
			na = np.linalg.norm(a.astype(np.float64)) if np.linalg.norm(a) >= EPS else 1.0
			nb = np.linalg.norm(b.astype(np.float64)) if np.linalg.norm(b) >= EPS else 1.0
			v = a / na - b / nb
			out.append((v / np.linalg.norm(v), 0.0))
		else:
			v = a.astype(np.float64) - b
			out.append((v, -v @ (a.astype(np.float64) + b) / 2))
	return out


def _reference_split(kind, angular, data, indices, graph, degrees):
	"""Expected (left, right, plane, offset, score) following the hub selection rule.

	The reference restates the code's rule, including its quirks (points within
	EPS of the plane go to arange % 2 sides; the first best candidate wins), so
	it pins the current output."""
	X = _dense_view(data, kind)
	pos = {int(p): i for i, p in enumerate(indices)}
	best = None
	for plane, offset in _candidates(kind, angular, X, indices, degrees):
		m = _margins("bit" if kind == "bit" else "dense", plane, offset, X[indices])
		side = np.where(m > EPS, 0, np.where(m < -EPS, 1, np.arange(len(indices)) % 2))
		n_left = int((side == 0).sum())
		if n_left == 0 or n_left == len(indices):
			continue
		if kind == "dense":
			score = min(n_left, len(indices) - n_left) / len(indices)
			better = best is None or score > best[0]
		else:
			cuts = 0
			for i, p in enumerate(indices):
				for nb in graph[p]:
					if nb < 0:
						break
					if nb in pos and side[pos[nb]] != side[i]:
						cuts += 1
			score = cuts // 2
			better = best is None or score < best[0]
		if better:
			best = (score, side, plane, offset)
	assert best is not None
	score, side, plane, offset = best
	return indices[side == 0], indices[side == 1], plane, offset, score


def _run_split(kind, angular, data, indices, graph, degrees, state):
	if kind == "dense":
		f = angular_hub_split if angular else euclidean_hub_split
		return f(data, indices, graph, degrees, state)
	if kind == "bit":
		return bit_hub_split(data, indices, graph, degrees, state)
	f = sparse_angular_hub_split if angular else sparse_euclidean_hub_split
	return f(data.indices, data.indptr, data.data, indices, graph, degrees, state)


@pytest.mark.parametrize("kind,angular", HUB_KINDS, ids=HUB_IDS)
@pytest.mark.parametrize("seed", [0, 1])
def test_hub_split_matches_reference_selection(kind, angular, seed):
	data = _make_data(kind, n=300, seed=seed)
	graph = _graph(data)
	degrees = compute_global_degrees(graph)
	rng = np.random.RandomState(seed)
	indices = np.sort(rng.choice(300, size=150, replace=False)).astype(np.int32)
	result = _run_split(kind, angular, data, indices, graph, degrees, _rng_state(seed))
	left, right, plane, offset = result[:4]
	exp_left, exp_right, exp_plane, exp_offset, score = _reference_split(
		kind, angular, data, indices, graph, degrees
	)
	np.testing.assert_array_equal(left, exp_left)
	np.testing.assert_array_equal(right, exp_right)
	if kind == "sparse":
		dense_plane = np.zeros(data.shape[1])
		keep = plane[0] >= 0
		dense_plane[plane[0][keep].astype(int)] = plane[1][keep]
		np.testing.assert_allclose(dense_plane, exp_plane, atol=1e-5)
	elif kind == "bit":
		np.testing.assert_array_equal(plane, exp_plane)
	else:
		np.testing.assert_allclose(plane, exp_plane, atol=1e-5)
	assert offset == pytest.approx(exp_offset, abs=1e-4)
	if kind == "dense":
		assert result[4] == pytest.approx(score, abs=1e-6)
		assert result[4] == pytest.approx(min(len(left), len(right)) / len(indices))


@pytest.mark.parametrize("kind,angular", HUB_KINDS, ids=HUB_IDS)
def test_hub_split_has_no_rng_dependence_when_candidates_valid(kind, angular):
	data = _make_data(kind, n=200)
	graph = _graph(data)
	degrees = compute_global_degrees(graph)
	indices = np.arange(200, dtype=np.int32)
	a = _run_split(kind, angular, data, indices, graph, degrees, _rng_state(1))
	b = _run_split(kind, angular, data, indices, graph, degrees, _rng_state(99))
	np.testing.assert_array_equal(a[0], b[0])
	np.testing.assert_array_equal(a[1], b[1])


def test_euclidean_hub_split_hand_case():
	"""Points on a line; hubs 0, 3 and 4 give one clearly most balanced bisector."""
	data = np.arange(8, dtype=np.float32).reshape(-1, 1)
	graph = np.zeros((8, 3), dtype=np.int32)
	graph[:] = [0, 3, 4]
	degrees = compute_global_degrees(graph)
	indices = np.arange(8, dtype=np.int32)
	left, right, plane, offset, balance = euclidean_hub_split(data, indices, graph, degrees, _rng_state(0))
	assert set(left) | set(right) == set(range(8))
	assert {tuple(sorted(left)), tuple(sorted(right))} == {(0, 1, 2, 3), (4, 5, 6, 7)}
	assert balance == pytest.approx(0.5)
	assert offset == pytest.approx(-plane[0] * 3.5)


def test_hub_tree_rejects_unbalanced_splits():
	data = np.arange(50, dtype=np.float32).reshape(-1, 1)
	graph = np.tile(np.array([0, 1, 2], dtype=np.int32), (50, 1))
	degrees = compute_global_degrees(graph)
	indices = np.arange(50, dtype=np.int32)
	_, _, _, _, balance = euclidean_hub_split(data, indices, graph, degrees, _rng_state(0))
	assert balance < MIN_SPLIT_BALANCE
	tree = make_hub_tree(data, graph, _rng_state(0), leaf_size=10)
	children, indices_list, _, _ = _linked(tree)
	assert children == [(-1, -1)]
	assert len(indices_list[0]) == 50
	assert tree.leaf_size == 50


def _build_hub_tree(kind, angular, data, graph, state, leaf_size=20, max_depth=200):
	if kind == "dense":
		return make_hub_tree(data, graph, state, leaf_size=leaf_size, angular=angular, max_depth=max_depth)
	if kind == "sparse":
		return make_sparse_hub_tree(
			data.indices, data.indptr, data.data, graph, state,
			leaf_size=leaf_size, angular=angular, max_depth=max_depth,
		)
	return make_bit_hub_tree(data, graph, state, leaf_size=leaf_size, max_depth=max_depth)


@pytest.mark.parametrize("kind,angular", HUB_KINDS, ids=HUB_IDS)
def test_hub_tree_is_valid_partition(kind, angular):
	data = _make_data(kind, n=300)
	graph = _graph(data)
	tree = _build_hub_tree(kind, angular, data, graph, _rng_state(1))
	assert isinstance(tree, FlatTree)
	_check_linked(tree, data, kind, 20, bounded=False)
	children, indices, _, _ = _linked(tree)
	leaves = [x for c, x in zip(children, indices) if c == (-1, -1)]
	assert sum(len(x) for x in leaves) == 300
	assert tree.leaf_size == max(20, max(len(x) for x in leaves))
	assert sum(len(x) <= 20 for x in leaves) >= 0.9 * len(leaves)


@pytest.mark.parametrize("kind,angular", HUB_KINDS, ids=HUB_IDS)
def test_hub_tree_deterministic(kind, angular):
	data = _make_data(kind, n=300)
	graph = _graph(data)
	a = _build_hub_tree(kind, angular, data, graph, _rng_state(1))
	b = _build_hub_tree(kind, angular, data, graph, _rng_state(1))
	assert [tuple(c) for c in a.children] == [tuple(c) for c in b.children]
	for x, y in zip(a.indices, b.indices):
		np.testing.assert_array_equal(x, y)
	for x, y in zip(a.hyperplanes, b.hyperplanes):
		np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("kind,angular", HUB_KINDS, ids=HUB_IDS)
def test_hub_tree_max_depth_and_single_leaf(kind, angular):
	data = _make_data(kind, n=200)
	graph = _graph(data)
	tree = _build_hub_tree(kind, angular, data, graph, _rng_state(1), leaf_size=10, max_depth=0)
	assert len(tree.children) == 1 and tree.leaf_size == 200
	tree = _build_hub_tree(kind, angular, data, graph, _rng_state(1), leaf_size=10, max_depth=1)
	assert len(tree.children) == 3
	_check_linked(tree, data, kind, 10, bounded=False)
	tree = _build_hub_tree(kind, angular, data, graph, _rng_state(1), leaf_size=500)
	assert len(tree.children) == 1 and tree.leaf_size == 500


@pytest.mark.parametrize("kind,angular", HUB_KINDS, ids=HUB_IDS)
def test_hub_tree_converts_and_routes_training_points(kind, angular):
	data = _make_data(kind, n=300)
	graph = _graph(data)
	tree = _build_hub_tree(kind, angular, data, graph, _rng_state(1))
	flat = convert_tree_format(tree, 300, data.shape[1])
	_check_flat_structure(flat, 300)
	_check_flat_sides(flat, data, kind)
	X = _dense_view(data, kind)
	children = np.asarray(flat.children)
	state = _rng_state(2)
	checked = 0
	for i in range(300):
		leaf = _route(kind, flat, data, i, state)
		node, tied = 0, False
		while children[node, 0] > 0:
			m = _margins(kind, _flat_plane(flat, node), float(flat.offsets[node]), X[i : i + 1])[0]
			tied |= abs(m) < 1e-4
			node = children[node, 0] if m > 0 else children[node, 1]
		if tied:
			continue
		checked += 1
		assert i in leaf
	assert checked >= 50


# ------------------------------------------------------------ hybrid forest


def _split_trees(leaf_array, n, n_trees):
	"""Group the rows of a leaf array into trees; each tree covers all n points once."""
	trees, current, seen = [], [], 0
	for row in leaf_array:
		members = row[row >= 0]
		current.append(members)
		seen += len(members)
		if seen == n:
			trees.append(current)
			current, seen = [], 0
		assert seen < n or seen == n
	assert current == []
	assert len(trees) == n_trees
	return trees


def _check_leaf_array(leaf_array, n, n_trees, leaf_size, bounded=True):
	assert leaf_array.dtype == np.int32 and leaf_array.ndim == 2
	for row in leaf_array:
		pad = np.flatnonzero(row < 0)
		assert len(pad) == 0 or np.all(row[pad[0] :] == -1)
		assert row[0] >= 0
	for tree in _split_trees(leaf_array, n, n_trees):
		everything = np.concatenate(tree)
		assert np.array_equal(np.sort(everything), np.arange(n))
		if bounded:
			assert max(len(m) for m in tree) <= leaf_size


def _hybrid_data(kind, n=500):
	if kind == "dense":
		return _dense(n, 6)
	return _sparse(n, 30)


@pytest.mark.parametrize("kind", ["dense", "sparse"])
@pytest.mark.parametrize("angular", [False, True])
@pytest.mark.parametrize("local_bytes", [0, 1 << 20, 1 << 40])
def test_hybrid_leaf_array_is_valid_forest(kind, angular, local_bytes):
	data = _hybrid_data(kind)
	leaves = make_hybrid_leaf_array(data, 4, 25, 3, angular=angular, local_bytes=local_bytes)
	_check_leaf_array(leaves, 500, 4, 25)


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_hybrid_leaf_array_deterministic_and_seed_sensitive(kind):
	data = _hybrid_data(kind)
	a = make_hybrid_leaf_array(data, 3, 25, 5)
	b = make_hybrid_leaf_array(data, 3, 25, 5)
	c = make_hybrid_leaf_array(data, 3, 25, 6)
	np.testing.assert_array_equal(a, b)
	assert a.shape != c.shape or not np.array_equal(a, c)


@pytest.mark.skipif(numba.config.NUMBA_NUM_THREADS < 2, reason="needs 2 numba threads")
@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_hybrid_leaf_array_independent_of_thread_count(kind):
	"""One thread and several threads give the same leaf array."""
	data = _hybrid_data(kind)
	previous = numba.get_num_threads()
	try:
		numba.set_num_threads(1)
		one = make_hybrid_leaf_array(data, 3, 25, 5)
		numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
		many = make_hybrid_leaf_array(data, 3, 25, 5)
	finally:
		numba.set_num_threads(previous)
	np.testing.assert_array_equal(one, many)


def test_hybrid_leaf_array_trees_differ():
	leaves = make_hybrid_leaf_array(_dense(300, 5), 2, 20, 1)
	first, second = _split_trees(leaves, 300, 2)
	signature = lambda tree: sorted(tuple(sorted(m)) for m in tree)
	assert signature(first) != signature(second)


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_hybrid_leaf_array_small_and_depth_limited(kind):
	data = _hybrid_data(kind, n=50)
	leaves = make_hybrid_leaf_array(data, 3, 100, 0)
	assert leaves.shape == (3, 50)
	for row in leaves:
		assert np.array_equal(np.sort(row), np.arange(50))
	data = _hybrid_data(kind, n=300)
	leaves = make_hybrid_leaf_array(data, 2, 5, 0, max_depth=0)
	assert leaves.shape == (2, 300)
	leaves = make_hybrid_leaf_array(data, 2, 5, 0, max_depth=1)
	assert leaves.shape[0] == 4
	_check_leaf_array(leaves, 300, 2, 300, bounded=False)
	sizes = (leaves >= 0).sum(axis=1)
	assert sizes[0] + sizes[1] == 300


def test_hybrid_leaf_array_identical_points_terminates():
	data = np.ones((200, 4), dtype=np.float32)
	leaves = make_hybrid_leaf_array(data, 2, 10, 0, max_depth=30)
	_check_leaf_array(leaves, 200, 2, 10, bounded=False)


def test_hybrid_leaf_array_locality_on_clusters():
	"""Points of well separated clusters mostly share a leaf with their nearest neighbour."""
	rng = np.random.RandomState(0)
	centers = rng.randn(10, 6) * 20
	data = (centers[rng.randint(0, 10, 500)] + rng.randn(500, 6)).astype(np.float32)
	leaves = make_hybrid_leaf_array(data, 6, 30, 2)
	nn = _brute_knn(data, 1)[:, 0]
	shared = np.zeros(500, dtype=bool)
	for row in leaves:
		members = row[row >= 0]
		inside = np.zeros(500, dtype=bool)
		inside[members] = True
		shared |= inside & inside[nn]
	assert shared.mean() >= 0.99


# ----------------------------------------------------------- hub search tree


@pytest.mark.parametrize("kind,angular", HUB_KINDS, ids=HUB_IDS)
def test_hub_search_tree_hyperplane_sides_and_structure(kind, angular):
	data = _make_data(kind, n=1200)
	graph = _graph(data)
	tree = make_hub_search_tree(data, graph, leaf_size=30, angular=angular, seed=3)
	assert isinstance(tree, FlatTree)
	_check_flat_structure(tree, 1200)
	_check_flat_sides(tree, data, kind)
	assert tree.leaf_size >= 30


@pytest.mark.parametrize("kind,angular", HUB_KINDS, ids=HUB_IDS)
def test_hub_search_tree_deterministic_for_seed(kind, angular):
	data = _make_data(kind, n=800)
	graph = _graph(data)
	a = make_hub_search_tree(data, graph, leaf_size=30, angular=angular, seed=7)
	b = make_hub_search_tree(data, graph, leaf_size=30, angular=angular, seed=7)
	np.testing.assert_array_equal(a.children, b.children)
	np.testing.assert_array_equal(a.indices, b.indices)
	np.testing.assert_array_equal(a.offsets, b.offsets)


@pytest.mark.parametrize("kind,angular", HUB_KINDS, ids=HUB_IDS)
def test_hub_search_tree_small_data_and_depth_limits(kind, angular):
	data = _make_data(kind, n=100)
	graph = _graph(data)
	single = make_hub_search_tree(data, graph, leaf_size=200, angular=angular)
	assert np.asarray(single.children).shape == (1, 2)
	assert np.array_equal(np.sort(single.indices), np.arange(100))
	shallow = make_hub_search_tree(data, graph, leaf_size=10, max_depth=1, angular=angular)
	assert np.asarray(shallow.children).shape == (3, 2)
	_check_flat_structure(shallow, 100)
	_check_flat_sides(shallow, data, kind)
	flat = make_hub_search_tree(data, graph, leaf_size=10, max_depth=0, angular=angular)
	assert np.asarray(flat.children).shape == (1, 2)


@pytest.mark.parametrize(
	"criterion,fallback",
	[
		(CRITERION_BALANCE, FALLBACK_LEAF),
		(CRITERION_BALANCE, FALLBACK_RANDOM),
		(CRITERION_BALANCE, FALLBACK_MEDIAN),
		(CRITERION_EDGE_CUT, FALLBACK_RANDOM),
		(CRITERION_EDGE_CUT, FALLBACK_MEDIAN),
	],
)
def test_hub_search_tree_options_keep_a_valid_partition(criterion, fallback):
	data = _dense(800, 8)
	graph = _graph(data)
	tree = make_hub_search_tree(
		data, graph, leaf_size=30, criterion=criterion, fallback=fallback, min_balance=0.1, seed=1
	)
	_check_flat_structure(tree, 800)
	_check_flat_sides(tree, data, "dense")
	if fallback != FALLBACK_LEAF:
		sizes = np.diff(np.sort(np.concatenate([-np.asarray(tree.children)[np.asarray(tree.children)[:, 0] <= 0][:, 0], [800]])))
		assert sizes.max() <= 30


def test_hub_search_tree_unbalanced_hubs_leaf_vs_median_fallback():
	data = np.arange(60, dtype=np.float32).reshape(-1, 1)
	graph = np.tile(np.array([0, 1, 2], dtype=np.int32), (60, 1))
	leaf = make_hub_search_tree(
		data, graph, leaf_size=10, criterion=CRITERION_BALANCE, min_balance=0.1, fallback=FALLBACK_LEAF
	)
	assert np.asarray(leaf.children).shape == (1, 2)
	median = make_hub_search_tree(
		data, graph, leaf_size=10, criterion=CRITERION_BALANCE, min_balance=0.1, fallback=FALLBACK_MEDIAN
	)
	children = np.asarray(median.children)
	is_leaf = children[:, 0] <= 0
	sizes = -children[is_leaf, 1] + children[is_leaf, 0]
	assert children.shape[0] > 1 and np.all(sizes <= 10)
	_check_flat_sides(median, data, "dense")


def test_hub_search_tree_accepts_padded_graph():
	data = _dense(600, 5)
	graph = _graph(data, pad=True)
	assert (graph < 0).any()
	tree = make_hub_search_tree(data, graph, leaf_size=40)
	_check_flat_structure(tree, 600)
