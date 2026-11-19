"""Tests for the module-level helpers of pynndescent_.py, search.py and
search_graph.py, and for the package namespace."""
import numba
import numpy as np
import pytest
from scipy.spatial.distance import cdist

import pynndescent
import pynndescent.distances as pynnd_dist
from pynndescent import NNDescent
from pynndescent.kernels import KERNEL_SQEUCLIDEAN
from pynndescent.pynndescent_ import (
	compute_degrees,
	compute_degrees_csr,
	diversify,
	diversify_alternative_dot,
	diversify_degree_aware,
	diversify_squared_euclidean,
	find_distance,
	init_from_neighbor_graph,
	init_random,
	init_rp_tree,
	is_c_contiguous,
	nn_descent,
	nn_descent_internal,
	rerank,
	resort_tree_indices,
)
from pynndescent.rp_trees import FlatTree
from pynndescent.search import (
	deheap_sort_parallel,
	deheap_sort_serial,
	make_cached_search_function,
	stack_search_forest,
)
from pynndescent.search_graph import build_search_graph, permute_graph, permute_rows
from pynndescent.utils import deheap_sort, make_heap, simple_heap_push


def _rng(seed=0):
	"""A well-seeded Tausworthe state; tiny seeds such as [1, 2, 3] never advance it."""
	return np.random.RandomState(seed).randint(-(2**31) + 1, 2**31 - 1, 3).astype(np.int64)


def _rows_recall(found, truth):
	return np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(found, truth)])


def _knn_rows(distance_matrix, k):
	"""Exact neighbour rows (self excluded) and their distances."""
	order = np.argsort(distance_matrix, axis=1, kind="stable")
	rows = order[:, 1 : k + 1].astype(np.int32)
	dists = np.take_along_axis(distance_matrix, rows, axis=1).astype(np.float32)
	return rows, dists


def _reference_diversify(rows, dists, distance):
	"""Relative-neighbourhood pruning with probability one, in plain Python. The loop
	restates the one in diversify, so it pins the current pruning rule."""
	eps = np.finfo(np.float32).eps
	out_rows = np.full_like(rows, -1)
	out_dists = np.full_like(dists, np.inf)
	for i in range(rows.shape[0]):
		kept = [(rows[i, 0], dists[i, 0])]
		for j in range(1, rows.shape[1]):
			u = rows[i, j]
			if u < 0:
				break
			if not any(dc > eps and distance(u, c) < dists[i, j] for c, dc in kept):
				kept.append((u, dists[i, j]))
		for m, (c, dc) in enumerate(kept):
			out_rows[i, m] = c
			out_dists[i, m] = dc
	return out_rows, out_dists


@pytest.fixture(scope="module")
def points():
	return np.random.RandomState(11).normal(size=(60, 3)).astype(np.float32)


# ---------------------------------------------------------------- namespace


def test_package_exports_and_version():
	assert pynndescent.NNDescent is NNDescent
	assert pynndescent.PyNNDescentTransformer.__name__ == "PyNNDescentTransformer"
	assert isinstance(pynndescent.__version__, str)
	assert pynndescent.__version__.split(".")[0].isdigit()


# ------------------------------------------------------------ small helpers


def test_is_c_contiguous():
	a = np.zeros((4, 5), dtype=np.float32)
	assert is_c_contiguous(a)
	assert not is_c_contiguous(np.asfortranarray(a))
	assert not is_c_contiguous(a[:, ::2])
	assert not is_c_contiguous([[1.0, 2.0]])
	assert not is_c_contiguous(None)


def test_compute_degrees_hand_example():
	indices = np.array([[0, 1, 2], [1, 0, -1], [2, -1, -1]], dtype=np.int32)
	# every listed neighbour adds one to the row's vertex and one to the neighbour
	expected = np.zeros(3, dtype=np.int32)
	for i in range(3):
		for j in range(3):
			if indices[i, j] >= 0:
				expected[i] += 1
				expected[indices[i, j]] += 1
	np.testing.assert_array_equal(expected, [5, 4, 3])
	result = compute_degrees(indices)
	assert result.dtype == np.int32
	np.testing.assert_array_equal(result, expected)


def test_find_distance():
	indices = np.array([[0, 3, 5, -1], [1, 2, -1, 7]], dtype=np.int32)
	distances = np.array([[0.0, 1.5, 2.5, np.inf], [0.0, 4.0, np.inf, 9.0]], dtype=np.float32)
	assert find_distance(indices, distances, 0, 3) == np.float32(1.5)
	assert find_distance(indices, distances, 0, 5) == np.float32(2.5)
	assert find_distance(indices, distances, 1, 2) == np.float32(4.0)
	assert find_distance(indices, distances, 0, 9) == np.inf
	# the scan stops at the first negative index, so 7 behind it is not found
	assert find_distance(indices, distances, 1, 7) == np.inf


def test_compute_degrees_csr_hand_example():
	indptr = np.array([0, 2, 3, 3], dtype=np.int32)
	indices = np.array([1, 2, 0], dtype=np.int32)
	# out-degrees 2, 1, 0 plus in-degrees 1, 1, 1
	np.testing.assert_array_equal(compute_degrees_csr(indptr, indices), [3, 2, 1])
	# targets outside the vertex range add no in-degree
	np.testing.assert_array_equal(
		compute_degrees_csr(indptr, np.array([1, 2, 9], dtype=np.int32)), [2, 2, 1]
	)


def test_init_from_neighbor_graph_fills_heap(points):
	n, k = 8, 3
	data = points[:n]
	D = cdist(data, data)
	rows, dists = _knn_rows(D, k)
	heap = make_heap(n, k)
	init_from_neighbor_graph(heap, rows, dists)
	for i in range(n):
		assert set(heap[0][i]) == set(rows[i])
		np.testing.assert_allclose(sorted(heap[1][i]), sorted(dists[i]))
		# a max-heap: the root holds the largest distance
		assert heap[1][i, 0] == heap[1][i].max()
	assert (heap[2] == 0).all()


def test_init_from_neighbor_graph_keeps_best_when_oversubscribed():
	rows = np.array([[1, 2, 3, 4]], dtype=np.int32)
	dists = np.array([[4.0, 1.0, 3.0, 2.0]], dtype=np.float32)
	heap = make_heap(5, 2)
	init_from_neighbor_graph(heap, rows, dists)
	assert set(heap[0][0]) == {2, 4}
	assert heap[1][0].max() == 2.0


def test_init_random_fills_rows_with_true_distances(points):
	n, k = 30, 4
	data = points[:n].copy()
	heap = make_heap(n, k)
	init_random(k, data, heap, pynnd_dist.euclidean, _rng())
	D = cdist(data, data)
	for i in range(n):
		filled = heap[0][i][heap[0][i] >= 0]
		assert 1 <= len(filled) <= k
		assert len(set(filled)) == len(filled)
		assert i not in filled
		for slot in np.nonzero(heap[0][i] >= 0)[0]:
			np.testing.assert_allclose(heap[1][i, slot], D[i, heap[0][i, slot]], rtol=1e-5)
			assert heap[2][i, slot] == 1


def test_init_random_skips_rows_that_already_have_neighbours(points):
	n, k = 10, 3
	data = points[:n].copy()
	heap = make_heap(n, k)
	rows = np.array([[1, 2, 3]], dtype=np.int32)
	dists = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
	init_from_neighbor_graph(heap, rows, dists)
	before = heap[0][0].copy()
	init_random(k, data, heap, pynnd_dist.euclidean, _rng())
	np.testing.assert_array_equal(heap[0][0], before)
	assert (heap[0][5] >= 0).any()


def test_init_rp_tree_joins_all_pairs_within_leaves(points):
	n = 20
	data = points[:n, :2].copy()
	leaf_array = np.array(
		[[0, 1, 2, 3], [4, 5, 6, -1], [7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]],
		dtype=np.int32,
	)
	heap = make_heap(n, 4)
	init_rp_tree(data, pynnd_dist.euclidean, heap, leaf_array, n_threads=2)
	D = cdist(data, data)
	for leaf in leaf_array:
		members = [int(p) for p in leaf if p >= 0]
		for p in members:
			expected = {q for q in members if q != p}
			found = {int(q) for q in heap[0][p] if q >= 0}
			assert found == expected
			for slot in range(4):
				q = heap[0][p, slot]
				if q >= 0:
					np.testing.assert_allclose(heap[1][p, slot], D[p, q], rtol=1e-5)
	# point 19 is in no leaf
	assert (heap[0][19] < 0).all()


# ---------------------------------------------------------------- NN descent


def _descent_inputs(n=80, d=4, k=6, seed=3):
	data = np.random.RandomState(seed).normal(size=(n, d)).astype(np.float32)
	D = cdist(data, data)
	rows, dists = _knn_rows(D, k)
	return data, rows, dists


def test_nn_descent_matches_brute_force():
	data, truth, truth_dists = _descent_inputs()
	k = 6
	indices, dists = nn_descent(
		data, k, _rng(), 10, pynnd_dist.euclidean, 10, 0.001,
		rp_tree_init=False, leaf_array=np.array([[-1]]),
	)
	assert indices.shape == (80, k) and dists.shape == (80, k)
	assert indices.dtype == np.int32 and dists.dtype == np.float32
	assert (np.diff(dists, axis=1) >= 0).all()
	assert (indices != np.arange(80)[:, None]).all()
	assert _rows_recall(indices, truth) >= 0.95
	D = cdist(data, data)
	np.testing.assert_allclose(dists, np.take_along_axis(D, indices, axis=1), rtol=1e-4, atol=1e-5)


def test_nn_descent_uses_rp_leaves_when_asked():
	data, truth, _ = _descent_inputs()
	k = 6
	order = np.random.RandomState(0).permutation(80).astype(np.int32)
	leaf_array = order.reshape(8, 10)
	indices, dists = nn_descent(
		data, k, _rng(), 10, pynnd_dist.euclidean, 0, 0.001,
		rp_tree_init=True, leaf_array=leaf_array,
	)
	# zero iterations: each point only holds neighbours from its own leaf, and
	# the k kept are the k closest of the 9 leaf peers
	D = cdist(data, data)
	for leaf in leaf_array:
		for p in leaf:
			peers = [int(q) for q in leaf if q != p]
			expected = sorted(peers, key=lambda q: D[p, q])[:k]
			assert sorted(indices[p]) == sorted(expected)


def test_nn_descent_accepts_init_graph_and_rejects_wrong_shape():
	data, truth, truth_dists = _descent_inputs()
	k = 6
	heap = make_heap(80, k)
	init_from_neighbor_graph(heap, truth, truth_dists)
	indices, dists = nn_descent(
		data, k, _rng(), 10, pynnd_dist.euclidean, 0, 0.001,
		init_graph=heap, rp_tree_init=False, leaf_array=np.array([[-1]]),
	)
	np.testing.assert_array_equal(np.sort(indices, axis=1), np.sort(truth, axis=1))
	with pytest.raises(ValueError):
		nn_descent(
			data, k, _rng(), 10, pynnd_dist.euclidean, 5, 0.001,
			init_graph=make_heap(80, k - 1), rp_tree_init=False,
			leaf_array=np.array([[-1]]),
		)


def test_nn_descent_internal_refines_random_graph():
	data, truth, _ = _descent_inputs()
	k = 6
	rng_state = _rng()
	graph = make_heap(80, k)
	init_random(k, data, graph, pynnd_dist.euclidean, rng_state)
	untouched = graph[0].copy()
	nn_descent_internal(
		graph, data, k, rng_state, max_candidates=10, dist=pynnd_dist.euclidean,
		n_iters=0, delta=0.001,
	)
	np.testing.assert_array_equal(graph[0], untouched)
	nn_descent_internal(
		graph, data, k, rng_state, max_candidates=10, dist=pynnd_dist.euclidean,
		n_iters=10, delta=0.001,
	)
	indices, dists = deheap_sort(graph[0], graph[1])
	assert _rows_recall(indices, truth) >= 0.95
	assert (np.diff(dists, axis=1) >= 0).all()


def test_nn_descent_internal_stops_early_with_large_delta():
	data, truth, _ = _descent_inputs()
	k = 6
	rng_state = _rng()

	def run(delta):
		graph = make_heap(80, k)
		init_random(k, data, graph, pynnd_dist.euclidean, _rng())
		nn_descent_internal(
			graph, data, k, rng_state.copy(), max_candidates=10, dist=pynnd_dist.euclidean,
			n_iters=10, delta=delta,
		)
		return _rows_recall(deheap_sort(graph[0], graph[1])[0], truth)

	assert run(1e9) < run(0.0)


# ---------------------------------------------------------------- diversify


def _squared_rows(data, k):
	D = cdist(data, data, "sqeuclidean")
	rows, dists = _knn_rows(D, k)
	rows = np.hstack([np.arange(len(data), dtype=np.int32)[:, None], rows[:, :-1]])
	dists = np.hstack([np.zeros((len(data), 1), dtype=np.float32), dists[:, :-1]])
	return rows, dists, D


def test_diversify_squared_euclidean_matches_reference(points):
	rows, dists, D = _squared_rows(points, 10)
	expected = _reference_diversify(rows, dists, lambda a, b: np.float32(D[a, b]))
	result = diversify_squared_euclidean(
		rows.copy(), dists.copy(), points, _rng(), 1.0
	)
	np.testing.assert_array_equal(result[0], expected[0])
	np.testing.assert_array_equal(result[1], expected[1])
	assert (result[0] >= 0).sum() < (rows >= 0).sum()
	np.testing.assert_array_equal(result[0][:, 0], np.arange(len(points)))


def test_diversify_squared_euclidean_probability_zero_keeps_everything(points):
	rows, dists, _ = _squared_rows(points, 10)
	result = diversify_squared_euclidean(
		rows.copy(), dists.copy(), points, _rng(), 0.0
	)
	np.testing.assert_array_equal(result[0], rows)
	np.testing.assert_array_equal(result[1], dists)


def test_diversify_alternative_dot_matches_generic_diversify():
	rng = np.random.RandomState(5)
	data = np.abs(rng.normal(size=(60, 5))).astype(np.float32) + 0.1
	data /= np.linalg.norm(data, axis=1, keepdims=True)
	D = np.array(
		[[pynnd_dist.alternative_dot(a, b) for b in data] for a in data], dtype=np.float32
	)
	order = np.argsort(D, axis=1, kind="stable")
	rows = order[:, :10].astype(np.int32)
	dists = np.take_along_axis(D, rows, axis=1).astype(np.float32)
	expected = _reference_diversify(rows, dists, lambda a, b: D[a, b])
	result = diversify_alternative_dot(
		rows.copy(), dists.copy(), data, _rng(), 1.0
	)
	np.testing.assert_array_equal(result[0], expected[0])
	np.testing.assert_array_equal(result[1], expected[1])
	assert (result[0] >= 0).sum() < (rows >= 0).sum()


def test_diversify_degree_aware_zero_aggressiveness_is_standard(points):
	rows, dists, D = _squared_rows(points, 10)
	standard = diversify(
		rows.copy(), dists.copy(), points, pynnd_dist.squared_euclidean,
		_rng(), 1.0,
	)
	aware = diversify_degree_aware(
		rows.copy(), dists.copy(), points, pynnd_dist.squared_euclidean, 5, 0.0, 1.0
	)
	np.testing.assert_array_equal(aware[0], standard[0])
	np.testing.assert_array_equal(aware[1], standard[1])


def test_diversify_degree_aware_prunes_hub_edges_more(points):
	rows, dists, _ = _squared_rows(points, 12)

	def run(aggressiveness, alpha=1.0):
		return diversify_degree_aware(
			rows.copy(), dists.copy(), points, pynnd_dist.squared_euclidean, 3,
			aggressiveness, alpha,
		)

	n_standard = (run(0.0)[0] >= 0).sum()
	n_aware = (run(3.0)[0] >= 0).sum()
	assert n_aware < n_standard
	# alpha 0 never prunes, and rows keep their padding layout
	unpruned = run(1.0, alpha=0.0)
	np.testing.assert_array_equal(unpruned[0], rows)
	pruned = run(3.0)
	for i in range(len(points)):
		valid = pruned[0][i] >= 0
		assert valid[: valid.sum()].all()
		assert np.isinf(pruned[1][i][~valid]).all()


# ------------------------------------------------------------ tree and rerank


def test_resort_tree_indices_reorders_and_keeps_other_fields():
	tree = FlatTree(
		np.zeros((1, 2), dtype=np.float32),
		np.zeros(1, dtype=np.float32),
		np.array([[-1, -1]], dtype=np.int32),
		np.array([3, 1, 2, 0], dtype=np.int32),
		4,
	)
	order = np.array([2, 0, 3, 1])
	new = resort_tree_indices(tree, order)
	np.testing.assert_array_equal(new.indices, tree.indices[order])
	assert new.indices.dtype == np.int32 and new.indices.flags["C_CONTIGUOUS"]
	assert new.hyperplanes is tree.hyperplanes
	assert new.offsets is tree.offsets
	assert new.children is tree.children
	assert new.leaf_size == 4


def test_rerank_picks_closest_candidates_and_ignores_padding():
	rng = np.random.RandomState(2)
	data = rng.normal(size=(30, 4)).astype(np.float32)
	queries = rng.normal(size=(3, 4)).astype(np.float32)
	candidates = np.array(
		[[5, 9, 12, 20, 21, -1], [0, 1, 2, 3, -1, -1], [29, 28, 27, 26, 25, 24]],
		dtype=np.int32,
	)
	inds, dists = rerank(
		candidates, queries, data, pynnd_dist.euclidean, 3, deheap_sort_serial
	)
	assert inds.shape == (3, 3)
	for i in range(3):
		valid = candidates[i][candidates[i] >= 0]
		d = cdist(queries[i : i + 1], data[valid])[0]
		best = np.argsort(d)[:3]
		np.testing.assert_array_equal(inds[i], valid[best])
		np.testing.assert_allclose(dists[i], d[best], rtol=1e-5)


def test_rerank_with_fewer_candidates_than_k_pads_with_minus_one():
	data = np.eye(4, dtype=np.float32)
	queries = np.zeros((1, 4), dtype=np.float32)
	candidates = np.array([[2, -1, -1]], dtype=np.int32)
	inds, dists = rerank(candidates, queries, data, pynnd_dist.euclidean, 3, deheap_sort_serial)
	assert inds[0, 0] == 2
	np.testing.assert_allclose(dists[0, 0], 1.0)
	assert (inds[0, 1:] == -1).all()
	assert np.isinf(dists[0, 1:]).all()


# ------------------------------------------------------------------ search.py


@pytest.mark.parametrize("sorter", [deheap_sort_serial, deheap_sort_parallel])
def test_deheap_sort_variants_sort_rows_ascending(sorter):
	rng = np.random.RandomState(4)
	n_rows, size = 7, 5
	heap = make_heap(n_rows, size)
	pushed = rng.uniform(0, 10, size=(n_rows, 12)).astype(np.float32)
	for i in range(n_rows):
		for j in range(12):
			simple_heap_push(heap[1][i], heap[0][i], pushed[i, j], j)
	inds, dists = sorter(heap[0], heap[1])
	for i in range(n_rows):
		order = np.argsort(pushed[i], kind="stable")[:size]
		np.testing.assert_array_equal(inds[i], order)
		np.testing.assert_array_equal(dists[i], pushed[i][order])


def test_stack_search_forest_pads_smaller_trees():
	def tree(n_nodes, indices):
		return FlatTree(
			np.full((n_nodes, 2), n_nodes, dtype=np.float32),
			np.arange(n_nodes, dtype=np.float32),
			np.full((n_nodes, 2), 7, dtype=np.int32),
			np.array(indices, dtype=np.int32),
			3,
		)

	small, large = tree(3, [2, 0, 1, 3]), tree(5, [3, 2, 1, 0])
	hyperplanes, offsets, children, indices = stack_search_forest([small, large], 4)
	assert hyperplanes.shape == (2, 5, 2) and offsets.shape == (2, 5)
	assert children.shape == (2, 5, 2) and indices.shape == (2, 4)
	assert hyperplanes.dtype == np.float32 and children.dtype == np.int32
	np.testing.assert_array_equal(hyperplanes[0, :3], small.hyperplanes)
	np.testing.assert_array_equal(hyperplanes[0, 3:], 0.0)
	np.testing.assert_array_equal(offsets[0], [0, 1, 2, 0, 0])
	np.testing.assert_array_equal(children[0, :3], 7)
	np.testing.assert_array_equal(children[0, 3:], -1)
	np.testing.assert_array_equal(children[1], 7)
	np.testing.assert_array_equal(indices, [[2, 0, 1, 3], [3, 2, 1, 0]])


@pytest.fixture(scope="module")
def prepared_euclidean():
	data = np.random.RandomState(21).normal(size=(250, 6)).astype(np.float32)
	index = NNDescent(data, n_neighbors=12, random_state=0, n_jobs=1)
	index.prepare()
	return data, index


@pytest.mark.parametrize("parallel", [False, True])
def test_make_cached_search_function_finds_neighbours(prepared_euclidean, parallel):
	data, index = prepared_euclidean
	queries = np.random.RandomState(22).normal(size=(40, 6)).astype(np.float32)
	k = 5
	search = make_cached_search_function(
		index._raw_data, index._search_graph.indptr, index._search_graph.indices,
		index._search_forest, 1, index.n_neighbors, index._min_distance, False, False,
		KERNEL_SQEUCLIDEAN, parallel,
	)
	result = make_heap(len(queries), k)
	visited = np.zeros_like(index._visited)
	inds, dists, _ = search(
		queries, queries, np.ones(len(queries), dtype=np.float32), result, k, 0.2,
		visited, False, _rng(),
	)
	sorter = deheap_sort_parallel if parallel else deheap_sort_serial
	inds, dists = sorter(inds, dists)
	inds = index._vertex_order[inds]
	D = cdist(queries, data, "sqeuclidean")
	truth = np.argsort(D, axis=1)[:, :k]
	assert _rows_recall(inds, truth) >= 0.9
	np.testing.assert_allclose(
		dists, np.take_along_axis(D, inds, axis=1), rtol=1e-4, atol=1e-5
	)
	assert (np.diff(dists, axis=1) >= 0).all()


# ------------------------------------------------------------- search_graph


def test_build_search_graph_hand_example():
	rows = np.array([[1, 2], [0, -1], [3, -1], [-1, -1]], dtype=np.int32)
	dists = np.array(
		[[1.0, 2.0], [1.0, np.inf], [5.0, np.inf], [np.inf, np.inf]], dtype=np.float32
	)
	indptr, indices, min_distance = build_search_graph(rows, dists, 10)
	# undirected union of 0-1, 0-2, 2-3
	np.testing.assert_array_equal(indptr, [0, 2, 3, 5, 6])
	np.testing.assert_array_equal(indices, [1, 2, 0, 0, 3, 2])
	assert min_distance == np.float32(1.0)


def test_build_search_graph_drops_self_loops_and_prunes_long_edges():
	rows = np.array([[0, 1, 2, 3], [-1] * 4, [-1] * 4, [-1] * 4], dtype=np.int32)
	dists = np.array(
		[[0.5, 1.0, 2.0, 3.0]] + [[np.inf] * 4] * 3, dtype=np.float32
	)
	indptr, indices, _ = build_search_graph(rows, dists, 100)
	np.testing.assert_array_equal(indptr, [0, 3, 4, 5, 6])
	assert 0 not in indices[indptr[0] : indptr[1]]
	pruned_ptr, pruned_idx, _ = build_search_graph(rows, dists, 1)
	# vertex 0 keeps edges up to its second-smallest value, so the farthest goes
	np.testing.assert_array_equal(pruned_idx[pruned_ptr[0] : pruned_ptr[1]], [1, 2])
	assert pruned_ptr[-1] < indptr[-1]


def test_build_search_graph_without_edges():
	rows = np.full((3, 2), -1, dtype=np.int32)
	dists = np.full((3, 2), np.inf, dtype=np.float32)
	indptr, indices, min_distance = build_search_graph(rows, dists, 4)
	np.testing.assert_array_equal(indptr, [0, 0, 0, 0])
	assert indices.shape == (0,)
	assert min_distance == 0.0


def test_permute_graph_relabels_vertices():
	"""New vertex a is old vertex order[a]; the expected CSR is written out by hand
	for a non-involutive order, so an order/inverse mix-up changes the result."""
	indptr = np.array([0, 2, 3, 5, 6], dtype=np.int32)
	indices = np.array([1, 2, 0, 0, 3, 2], dtype=np.int32)
	order = np.array([1, 3, 0, 2])
	# old adjacency 0:{1,2} 1:{0} 2:{0,3} 3:{2}; old->new labels 0->2 1->0 2->3 3->1
	expected = {0: {2}, 1: {3}, 2: {0, 3}, 3: {1, 2}}
	new_ptr, new_idx = permute_graph(indptr, indices, order)
	np.testing.assert_array_equal(new_ptr, [0, 1, 2, 4, 6])
	for a in range(4):
		assert set(new_idx[new_ptr[a] : new_ptr[a + 1]]) == expected[a]


@pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.int32])
def test_permute_rows_dtypes_and_shape(dtype):
	data = (np.arange(36).reshape(9, 4) % 7).astype(dtype)
	order = np.array([8, 0, 7, 1, 6, 2, 5, 3, 4])
	result = permute_rows(data, order)
	assert result.dtype == data.dtype and result.shape == data.shape
	np.testing.assert_array_equal(result, data[order])
