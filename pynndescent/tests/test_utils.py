"""Tests for pynndescent.utils, graph_utils and kernels.

Expected values come from short reference implementations written in numpy,
Python integers and heapq, or from hand-computable inputs.
"""

import heapq
import time
from itertools import combinations

import numba
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components

import pynndescent.distances as pdist
import pynndescent.kernels as kernels
import pynndescent.sparse as psparse
import pynndescent.utils as U
from pynndescent import NNDescent
from pynndescent.graph_utils import (
	FLOAT32_EPS,
	adjacency_matrix_representation,
	connect_graph,
	create_component_search,
	find_component_connection_edge,
)

F32_MAX = np.finfo(np.float32).max


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@numba.njit
def _to_bits_arr(x):
	out = np.empty(1, dtype=np.float32)
	out[0] = U.int32_bits_as_float32(x)
	return out


def _to_bits(x):
	return _to_bits_arr(x)[0]


@numba.njit
def _from_bits(x):
	return U.float32_bits_as_int32(x)


@numba.njit
def _store_and_load(ids):
	buf = np.empty(ids.shape[0], dtype=np.float32)
	out = np.empty(ids.shape[0], dtype=np.int32)
	for i in range(ids.shape[0]):
		buf[i] = U.int32_bits_as_float32(ids[i])
	for i in range(ids.shape[0]):
		out[i] = U.float32_bits_as_int32(buf[i])
	return out, buf


def _new_state(seed_value=0):
	state = np.zeros(3, dtype=np.int64)
	U.seed(state, seed_value)
	return state


def _int32_wrap(value):
	return ((value & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000


def _ref_tau_rand_int(state):
	"""Restates the Tausworthe update with Python integers; also see the hard-coded streams."""
	s0, s1, s2 = (int(s) for s in state)
	s0 = (((s0 & 4294967294) << 12) & 0xFFFFFFFF) ^ (
		(((s0 << 13) & 0xFFFFFFFF) ^ s0) >> 19
	)
	s1 = (((s1 & 4294967288) << 4) & 0xFFFFFFFF) ^ (
		(((s1 << 2) & 0xFFFFFFFF) ^ s1) >> 25
	)
	s2 = (((s2 & 4294967280) << 17) & 0xFFFFFFFF) ^ (
		(((s2 << 3) & 0xFFFFFFFF) ^ s2) >> 11
	)
	state[:] = (s0, s1, s2)
	return _int32_wrap(s0 ^ s1 ^ s2)


_MASK64 = (1 << 64) - 1


def _ref_splitmix64(x, v):
	x ^= v
	x = (x + 0x9E3779B97F4A7C15) & _MASK64
	x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
	x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
	return x ^ (x >> 31)


def _ref_hash_uniform(seed_value, a, b, c):
	"""Restates the splitmix64 chain; also see the hard-coded values."""
	x = _ref_splitmix64(seed_value, a)
	x = _ref_splitmix64(x, b)
	x = _ref_splitmix64(x, c)
	return (x >> 11) / 9007199254740992.0


@numba.njit
def _hash_uniform(seed_value, a, b, c):
	return U.hash_uniform(seed_value, a, b, c)


def _is_max_heap(priorities):
	for i in range(1, priorities.shape[0]):
		if priorities[(i - 1) // 2] < priorities[i]:
			return False
	return True


def _ref_push(entries, p, n, f, check_duplicates):
	"""Model of a bounded max-heap of (priority, index, flag) triples."""
	worst = max(e[0] for e in entries)
	if p >= worst:
		return 0
	if check_duplicates and any(e[1] == n for e in entries):
		return 0
	pos = [e[0] for e in entries].index(worst)
	entries[pos] = (p, n, f)
	return 1


def _row_entries(indices, priorities, flags=None):
	if flags is None:
		flags = np.zeros_like(indices, dtype=np.uint8)
	return [
		(priorities[i], int(indices[i]), int(flags[i])) for i in range(len(indices))
	]


def _rows_as_multisets(graph):
	indices, distances, flags = graph
	return [
		sorted(
			(float(distances[i, j]), int(indices[i, j]), int(flags[i, j]))
			for j in range(indices.shape[1])
		)
		for i in range(indices.shape[0])
	]


def _random_heap(rs, n_points, size, n_ids, n_pushes):
	"""Build a heap through checked_flagged_heap_push with distinct priorities."""
	graph = U.make_heap(n_points, size)
	for i in range(n_points):
		priorities = rs.permutation(n_pushes * 4)[:n_pushes].astype(np.float32) + 1.0
		for pri in priorities:
			n = np.int32(rs.randint(0, n_ids))
			U.checked_flagged_heap_push(
				graph[1][i], graph[0][i], graph[2][i], pri, n, np.uint8(rs.randint(0, 2))
			)
	return graph


def _random_knn_graph(rs, n, k, flag_probability=0.5):
	indices = np.empty((n, k), dtype=np.int32)
	for i in range(n):
		others = np.delete(np.arange(n), i)
		indices[i] = rs.choice(others, k, replace=False)
	distances = rs.rand(n, k).astype(np.float32)
	flags = (rs.rand(n, k) < flag_probability).astype(np.uint8)
	return indices, distances, flags


def _reference_candidates(indices, flags):
	"""Sets of new (flag 1) and old (flag 0) candidates for every vertex."""
	n = indices.shape[0]
	new = [set() for _ in range(n)]
	old = [set() for _ in range(n)]
	for i in range(n):
		for j in range(indices.shape[1]):
			idx = int(indices[i, j])
			target = new if flags[i, j] else old
			target[i].add(idx)
			target[idx].add(i)
	return new, old


def _row_set(row):
	values = [int(v) for v in row if v >= 0]
	assert len(values) == len(set(values))
	return set(values)


def _pack_updates(update_lists, max_updates):
	n_threads = len(update_lists)
	update_array = np.zeros((n_threads, max_updates, 3), dtype=np.float32)
	counts = np.zeros(n_threads, dtype=np.int64)
	ids = update_array.view(np.int32)
	for t, ups in enumerate(update_lists):
		counts[t] = len(ups)
		for j, (p, q, d) in enumerate(ups):
			ids[t, j, 0] = p
			ids[t, j, 1] = q
			update_array[t, j, 2] = d
	return update_array, counts


def _unpack_updates(update_array, counts):
	ids = update_array.view(np.int32)
	return [
		[
			(int(ids[t, j, 0]), int(ids[t, j, 1]), float(update_array[t, j, 2]))
			for j in range(counts[t])
		]
		for t in range(update_array.shape[0])
	]


def _random_updates(rs, n_vertices, n_threads, per_thread):
	lists = []
	for t in range(n_threads):
		ups = []
		for _ in range(per_thread):
			p, q = rs.choice(n_vertices, 2, replace=False)
			ups.append((int(p), int(q), np.float32(rs.rand() * 10)))
		lists.append(ups)
	return lists


def _copy_graph(graph):
	return tuple(a.copy() for a in graph)


# ---------------------------------------------------------------------------
# bit reinterpretation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, 1, 7, 255, 2**24, 2**24 + 1, 123456789, 2**30])
def test_int32_bits_as_float32_matches_numpy_view(value):
	"""Check int32 bits as float32 matches numpy view."""
	expected = np.array([value], dtype=np.int32).view(np.float32)[0]
	got = _to_bits(np.int32(value))
	assert got.dtype == np.float32
	assert got.view(np.int32) == value
	if np.isfinite(expected):
		assert got == expected


def test_int32_bits_as_float32_is_not_a_numeric_conversion():
	"""Check int32 bits as float32 is not a numeric conversion."""
	assert _to_bits(np.int32(1)) != np.float32(1.0)
	assert _to_bits(np.int32(1)).view(np.int32) == 1
	one_bits = np.float32(1.0).view(np.int32)
	assert _from_bits(np.float32(1.0)) == one_bits
	assert _from_bits(np.float32(-2.0)) == np.float32(-2.0).view(np.int32)


def test_float32_bits_as_int32_matches_numpy_view():
	"""Check float32 bits as int32 matches numpy view."""
	values = np.array([0.0, 1.0, -1.0, 0.5, 3.14159, 1e-30, -1e30, np.inf], dtype=np.float32)
	for v in values:
		assert _from_bits(v) == v.view(np.int32)


def test_vertex_ids_round_trip_through_float32_storage():
	"""Check vertex ids round trip through float32 storage."""
	ids = np.array(
		[0, 1, 2, 5, 2**24, 2**24 + 1, 2**24 + 3, 16777217, 100000007, 2**30 + 5, -1, -2],
		dtype=np.int32,
	)
	out, buf = _store_and_load(ids)
	np.testing.assert_array_equal(out, ids)
	np.testing.assert_array_equal(buf.view(np.int32), ids)


def test_int32_bits_accepts_other_integer_widths():
	"""Check int32 bits accepts other integer widths."""
	assert _to_bits(np.int64(2**24 + 1)).view(np.int32) == 2**24 + 1
	assert _to_bits(np.uint8(200)).view(np.int32) == 200


# ---------------------------------------------------------------------------
# random number generation
# ---------------------------------------------------------------------------


def test_seed_fills_state():
	"""Check seed sets all three state words to seed + 0xFFFF."""
	state = np.zeros(3, dtype=np.int64)
	U.seed(state, 0)
	np.testing.assert_array_equal(state, [0xFFFF] * 3)
	U.seed(state, 12345)
	np.testing.assert_array_equal(state, [12345 + 0xFFFF] * 3)
	U.seed(state, -5)
	np.testing.assert_array_equal(state, [0xFFFF - 5] * 3)


@pytest.mark.parametrize("seed_value", [0, 1, 189212, 2**31 - 1])
def test_tau_rand_int_matches_reference_stream(seed_value):
	"""Check tau rand int matches reference stream."""
	state = _new_state(seed_value)
	ref_state = state.copy()
	for _ in range(200):
		got = U.tau_rand_int(state)
		expected = _ref_tau_rand_int(ref_state)
		assert got == expected
		np.testing.assert_array_equal(state, ref_state)


@pytest.mark.parametrize(
	"seed_value,expected",
	[
		(0, [-267379553, -16770448, -937854944, 2143609379, -1036912402]),
		(12345, [1936754262, 995121266, 197582555, -1870100785, 1065744871]),
	],
)
def test_tau_rand_int_known_stream(seed_value, expected):
	"""Check the first five outputs against values computed once with Python integers from the Tausworthe constants."""
	state = _new_state(seed_value)
	assert [int(U.tau_rand_int(state)) for _ in range(5)] == expected


def test_tau_rand_int_is_deterministic_and_seed_dependent():
	"""Check tau rand int is deterministic and seed dependent."""
	a, b, c = _new_state(3), _new_state(3), _new_state(123456)
	seq_a = [U.tau_rand_int(a) for _ in range(20)]
	seq_b = [U.tau_rand_int(b) for _ in range(20)]
	seq_c = [U.tau_rand_int(c) for _ in range(20)]
	assert seq_a == seq_b
	assert seq_a != seq_c
	assert len(set(seq_a)) > 15


def test_tau_rand_int_returns_int32_range():
	"""Check tau rand int returns int32 range."""
	state = _new_state(9)
	vals = np.array([U.tau_rand_int(state) for _ in range(2000)], dtype=np.int64)
	assert vals.min() >= -(2**31) and vals.max() < 2**31
	assert (vals < 0).any() and (vals > 0).any()


def test_tau_rand_matches_scaled_integer_stream():
	"""Check tau rand matches scaled integer stream."""
	state = _new_state(21)
	ref_state = state.copy()
	for _ in range(200):
		got = U.tau_rand(state)
		expected = np.float32(abs(float(_ref_tau_rand_int(ref_state)) / 0x7FFFFFFF))
		assert got == expected
		assert 0.0 <= got <= 1.0


def test_tau_rand_is_roughly_uniform():
	"""Check tau rand is roughly uniform."""
	state = _new_state(5)
	vals = np.array([U.tau_rand(state) for _ in range(20000)])
	assert vals.min() >= 0.0 and vals.max() <= 1.0
	assert abs(vals.mean() - 0.5) < 0.02
	hist, _ = np.histogram(vals, bins=10, range=(0, 1))
	assert hist.min() > 0.7 * 2000 and hist.max() < 1.3 * 2000


@pytest.mark.parametrize("args", [(1, 2, 3, 4), (2**40, 7, 99, 5), (0, 0, 0, 0), (5, 6, 7, 8)])
def test_hash_uniform_matches_splitmix_reference(args):
	"""Check hash uniform matches splitmix reference."""
	got = _hash_uniform(*args)
	assert got == _ref_hash_uniform(*args)
	assert 0.0 <= got < 1.0


@pytest.mark.parametrize(
	"args,expected",
	[
		((1, 2, 3, 4), 0.004961322848166749),
		((0, 0, 0, 0), 0.13870941014555427),
		((5, 6, 7, 8), 0.8684104569363118),
	],
)
def test_hash_uniform_known_values(args, expected):
	"""Check hash_uniform against values computed once with Python integers from the splitmix64 constants."""
	assert _hash_uniform(*args) == pytest.approx(expected, rel=1e-15, abs=0.0)


def test_hash_uniform_is_stateless_and_sensitive_to_every_argument():
	"""Check hash uniform is stateless and sensitive to every argument."""
	base = _hash_uniform(1, 2, 3, 4)
	assert base == _hash_uniform(1, 2, 3, 4)
	others = {_hash_uniform(2, 2, 3, 4), _hash_uniform(1, 3, 3, 4), _hash_uniform(1, 2, 4, 4), _hash_uniform(1, 2, 3, 5)}
	assert base not in others
	assert len(others) == 4


def test_hash_uniform_is_roughly_uniform():
	"""Check hash uniform is roughly uniform."""
	vals = np.array([_hash_uniform(7, i, i % 13, 3) for i in range(20000)])
	assert vals.min() >= 0.0 and vals.max() < 1.0
	assert abs(vals.mean() - 0.5) < 0.01
	hist, _ = np.histogram(vals, bins=10, range=(0, 1))
	assert hist.min() > 0.85 * 2000 and hist.max() < 1.15 * 2000


# ---------------------------------------------------------------------------
# norm and rejection sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [1, 2, 5, 33, 100])
def test_norm_matches_numpy(dim):
	"""Check norm matches numpy."""
	rs = np.random.RandomState(dim)
	vec = rs.randn(dim).astype(np.float32)
	assert U.norm(vec) == pytest.approx(np.linalg.norm(vec.astype(np.float64)), rel=1e-5)


def test_norm_hand_values_and_readonly_input():
	"""Check norm hand values and readonly input."""
	assert U.norm(np.array([3.0, 4.0], dtype=np.float32)) == pytest.approx(5.0)
	assert U.norm(np.zeros(7, dtype=np.float32)) == 0.0
	assert U.norm(np.empty(0, dtype=np.float32)) == 0.0
	vec = np.array([1.0, 2.0, 2.0], dtype=np.float32)
	vec.flags.writeable = False
	assert U.norm(vec) == pytest.approx(3.0)


@pytest.mark.parametrize("n_samples,pool_size", [(1, 1), (3, 10), (10, 10), (50, 60), (7, 1000)])
def test_rejection_sample_distinct_and_in_range(n_samples, pool_size):
	"""Check rejection sample distinct and in range."""
	state = _new_state(n_samples + pool_size)
	for _ in range(20):
		sample = U.rejection_sample(n_samples, pool_size, state)
		assert sample.dtype == np.int64
		assert sample.shape == (n_samples,)
		assert len(set(sample.tolist())) == n_samples
		assert sample.min() >= 0 and sample.max() < pool_size


def test_rejection_sample_full_pool_is_a_permutation():
	"""Check rejection sample full pool is a permutation."""
	state = _new_state(0)
	sample = U.rejection_sample(12, 12, state)
	assert sorted(sample.tolist()) == list(range(12))


def test_rejection_sample_deterministic_given_state_and_advances_it():
	"""Check rejection sample deterministic given state and advances it."""
	a, b = _new_state(8), _new_state(8)
	np.testing.assert_array_equal(
		U.rejection_sample(5, 100, a), U.rejection_sample(5, 100, b)
	)
	np.testing.assert_array_equal(a, b)
	before = a.copy()
	U.rejection_sample(5, 100, a)
	assert not np.array_equal(a, before)


def test_rejection_sample_zero_samples_and_state_untouched():
	"""Check rejection sample zero samples and state untouched."""
	state = _new_state(2)
	before = state.copy()
	sample = U.rejection_sample(0, 10, state)
	assert sample.shape == (0,)
	np.testing.assert_array_equal(state, before)


def test_rejection_sample_first_draw_is_roughly_uniform():
	"""Check rejection sample first draw is roughly uniform."""
	state = _new_state(77)
	counts = np.zeros(5)
	for _ in range(5000):
		counts[U.rejection_sample(1, 5, state)[0]] += 1
	assert counts.min() > 0.8 * 1000 and counts.max() < 1.2 * 1000


# ---------------------------------------------------------------------------
# heaps
# ---------------------------------------------------------------------------


def test_make_heap_shapes_dtypes_and_fill():
	"""Check make_heap shapes, dtypes (int32, float32, uint8) and fill values (-1, inf, 0)."""
	indices, distances, flags = U.make_heap(4, 3)
	assert indices.shape == distances.shape == flags.shape == (4, 3)
	assert indices.dtype == np.int32 and distances.dtype == np.float32
	assert flags.dtype == np.uint8
	assert (indices == -1).all() and np.isinf(distances).all() and (flags == 0).all()
	i2, d2, f2 = U.make_heap(2.0, 5.0)
	assert i2.shape == (2, 5)


def test_empty_graph_sentinel():
	"""Check EMPTY_GRAPH is a 1x1 heap holding index -1 and an infinite distance."""
	indices, distances, flags = U.EMPTY_GRAPH
	assert indices.shape == distances.shape == flags.shape == (1, 1)
	assert indices[0, 0] == -1 and np.isinf(distances[0, 0])


@pytest.mark.parametrize("size", [1, 2, 3, 7, 16, 31])
def test_siftdown_restores_heap_and_keeps_pairs(size):
	"""Check siftdown restores heap and keeps pairs."""
	rs = np.random.RandomState(size)
	for trial in range(10):
		weights = rs.permutation(size * 3)[:size].astype(np.float32)
		labels = (weights * 10).astype(np.int32)
		heap = list(zip((-weights).tolist(), range(size)))
		heapq.heapify(heap)
		order = [i for _, i in heap]
		weights = weights[order].copy()
		labels = labels[order].copy()
		assert _is_max_heap(weights)
		weights[0] = rs.randint(0, size * 3) + 0.5
		labels[0] = int(weights[0] * 10)
		expected = sorted(zip(weights.tolist(), labels.tolist()))
		U.siftdown(weights, labels, 0)
		assert _is_max_heap(weights)
		assert sorted(zip(weights.tolist(), labels.tolist())) == expected
		assert (labels == (weights * 10).astype(np.int32)).all()


def test_siftdown_from_an_interior_position():
	"""Check siftdown from an interior position."""
	weights = np.array([9, 8, 7, 1, 2, 6, 5], dtype=np.float32)
	labels = np.arange(7, dtype=np.int32)
	weights[1] = 0.0
	U.siftdown(weights, labels, 1)
	assert _is_max_heap(weights)
	assert sorted(weights.tolist()) == [0, 1, 2, 5, 6, 7, 9]


def _heap_graph_distinct(rs, n, k, n_ids):
	graph = U.make_heap(n, k)
	for i in range(n):
		priorities = rs.permutation(4 * k)[:k].astype(np.float32) + 1.0
		ids = rs.choice(n_ids, k, replace=False).astype(np.int32)
		for pri, idx in zip(priorities, ids):
			U.checked_heap_push(graph[1][i], graph[0][i], pri, idx)
	return graph


@pytest.mark.parametrize("k", [1, 2, 5, 17])
def test_deheap_sort_orders_rows_by_increasing_distance(k):
	"""Check deheap sort orders rows by increasing distance."""
	rs = np.random.RandomState(k)
	indices, distances, _ = _heap_graph_distinct(rs, 9, k, 50)
	expected = [
		sorted(zip(distances[i].tolist(), indices[i].tolist())) for i in range(9)
	]
	out_i, out_d = U.deheap_sort(indices.copy(), distances.copy())
	for i in range(9):
		assert np.all(np.diff(out_d[i]) >= 0)
		assert list(zip(out_d[i].tolist(), out_i[i].tolist())) == expected[i]


def test_deheap_sort_is_in_place_and_handles_unfilled_slots():
	"""Check deheap sort is in place and handles unfilled slots."""
	indices, distances, _ = U.make_heap(2, 4)
	for pri, idx in [(3.0, 30), (1.0, 10)]:
		U.checked_heap_push(distances[0], indices[0], np.float32(pri), np.int32(idx))
	out_i, out_d = U.deheap_sort(indices, distances)
	assert out_i is indices and out_d is distances
	np.testing.assert_array_equal(out_d[0], [1.0, 3.0, np.inf, np.inf])
	np.testing.assert_array_equal(out_i[0][:2], [10, 30])
	assert (out_i[0][2:] == -1).all()
	assert np.isinf(out_d[1]).all()


# ---------------------------------------------------------------------------
# visited bit tables
# ---------------------------------------------------------------------------


def test_visited_table_bit_layout_and_semantics():
	"""Check visited table bit layout and semantics."""
	table = np.zeros(4, dtype=np.uint8)
	assert U.has_been_visited(table, np.int32(0)) == 0
	U.mark_visited(table, np.int32(0))
	assert table[0] == 1
	U.mark_visited(table, np.int32(3))
	assert table[0] == 1 + 8
	U.mark_visited(table, np.int32(9))
	assert table[1] == 2
	U.mark_visited(table, np.int32(31))
	assert table[3] == 128
	for c in (0, 3, 9, 31):
		assert U.has_been_visited(table, np.int32(c))
	for c in (1, 2, 8, 10, 30):
		assert not U.has_been_visited(table, np.int32(c))
	before = table.copy()
	U.mark_visited(table, np.int32(3))
	np.testing.assert_array_equal(table, before)


def test_check_and_mark_visited_matches_separate_calls():
	"""Check check and mark visited matches separate calls."""
	rs = np.random.RandomState(0)
	t1 = np.zeros(20, dtype=np.uint8)
	t2 = np.zeros(20, dtype=np.uint8)
	seen = set()
	for c in rs.randint(0, 160, size=400):
		was = bool(U.check_and_mark_visited(t1, np.int32(c)))
		assert was == (int(c) in seen)
		seen.add(int(c))
		assert bool(U.has_been_visited(t2, np.int32(c))) == was
		U.mark_visited(t2, np.int32(c))
		np.testing.assert_array_equal(t1, t2)
	for c in range(160):
		assert bool(U.has_been_visited(t1, np.int32(c))) == (c in seen)


# ---------------------------------------------------------------------------
# heap push variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 15, 16, 20, 33])
@pytest.mark.parametrize("variant", ["simple", "checked", "flagged"])
def test_heap_push_matches_bounded_heap_model(size, variant):
	"""Check heap push matches bounded heap model."""
	rs = np.random.RandomState(size * 3 + len(variant))
	priorities = np.full(size, np.inf, dtype=np.float32)
	indices = np.full(size, -1, dtype=np.int32)
	flags = np.zeros(size, dtype=np.uint8)
	model = _row_entries(indices, priorities, flags)
	for step in range(300):
		p = np.float32(((step * 7919) % 10007 + 1) / 8.0)
		n = np.int32(rs.randint(0, 3 * size + 2))
		f = np.uint8(rs.randint(0, 2))
		if variant == "simple":
			got = U.simple_heap_push(priorities, indices, p, n)
			expected = _ref_push(model, p, n, 0, False)
		elif variant == "checked":
			got = U.checked_heap_push(priorities, indices, p, n)
			expected = _ref_push(model, p, n, 0, True)
		else:
			got = U.checked_flagged_heap_push(priorities, indices, flags, p, n, f)
			expected = _ref_push(model, p, n, int(f), True)
		assert got == expected
		assert _is_max_heap(priorities)
		assert priorities[0] == priorities.max()
		have = sorted(_row_entries(indices, priorities, flags if variant == "flagged" else None))
		assert have == sorted(model)


@pytest.mark.parametrize("size", [4, 20])
def test_checked_pushes_reject_duplicate_index_even_with_smaller_priority(size):
	"""Check checked pushes reject duplicate index even with smaller priority."""
	priorities = np.full(size, np.inf, dtype=np.float32)
	indices = np.full(size, -1, dtype=np.int32)
	flags = np.zeros(size, dtype=np.uint8)
	assert U.checked_heap_push(priorities, indices, np.float32(5.0), np.int32(7)) == 1
	assert U.checked_heap_push(priorities, indices, np.float32(1.0), np.int32(7)) == 0
	assert sorted(priorities[np.isfinite(priorities)].tolist()) == [5.0]
	assert U.checked_flagged_heap_push(priorities, indices, flags, np.float32(2.0), np.int32(7), np.uint8(1)) == 0
	assert U.simple_heap_push(priorities, indices, np.float32(1.0), np.int32(7)) == 1
	assert (indices == 7).sum() == 2


@pytest.mark.parametrize("variant", ["simple", "checked", "flagged"])
def test_heap_push_rejects_priority_not_below_the_maximum(variant):
	"""Check heap push rejects priority not below the maximum."""
	size = 4
	priorities = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
	indices = np.array([4, 3, 2, 1], dtype=np.int32)
	flags = np.zeros(size, dtype=np.uint8)

	def push(p, n):
		if variant == "simple":
			return U.simple_heap_push(priorities, indices, np.float32(p), np.int32(n))
		if variant == "checked":
			return U.checked_heap_push(priorities, indices, np.float32(p), np.int32(n))
		return U.checked_flagged_heap_push(priorities, indices, flags, np.float32(p), np.int32(n), np.uint8(1))

	before = (priorities.copy(), indices.copy())
	assert push(4.0, 50) == 0
	assert push(9.0, 51) == 0
	np.testing.assert_array_equal(priorities, before[0])
	np.testing.assert_array_equal(indices, before[1])
	assert push(3.5, 52) == 1
	assert priorities.max() == 3.5
	assert sorted(priorities.tolist()) == [1.0, 2.0, 3.0, 3.5]
	assert dict(zip(indices.tolist(), priorities.tolist()))[52] == 3.5


def test_checked_flagged_heap_push_flags_travel_with_entries():
	"""Check checked flagged heap push flags travel with entries."""
	size = 6
	priorities = np.full(size, np.inf, dtype=np.float32)
	indices = np.full(size, -1, dtype=np.int32)
	flags = np.zeros(size, dtype=np.uint8)
	for n in range(size):
		U.checked_flagged_heap_push(
			priorities, indices, flags, np.float32(10 + n), np.int32(n), np.uint8(n % 2 + 1)
		)
	for n in range(6):
		pos = indices.tolist().index(n)
		assert flags[pos] == n % 2 + 1
		assert priorities[pos] == 10 + n
	U.checked_flagged_heap_push(priorities, indices, flags, np.float32(1.0), np.int32(99), np.uint8(7))
	pos = indices.tolist().index(99)
	assert flags[pos] == 7 and 15 not in priorities.tolist()
	assert 5 not in indices.tolist()


# ---------------------------------------------------------------------------
# candidate construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_threads", [1, 3])
def test_new_build_candidates_matches_reference_sets(n_threads):
	"""Check new build candidates matches reference sets."""
	rs = np.random.RandomState(1)
	n, k = 24, 5
	indices, distances, flags = _random_knn_graph(rs, n, k)
	exp_new, exp_old = _reference_candidates(indices, flags)
	graph = (indices.copy(), distances.copy(), flags.copy())
	new_c, old_c = U.new_build_candidates(graph, 2 * n, _new_state(3), n_threads)
	assert new_c.shape == old_c.shape == (n, 2 * n)
	assert new_c.dtype == old_c.dtype == np.int32
	for v in range(n):
		assert _row_set(new_c[v]) == exp_new[v]
		assert _row_set(old_c[v]) == exp_old[v]
	expected_flags = flags.copy()
	for i in range(n):
		for j in range(k):
			if int(indices[i, j]) in exp_new[i]:
				expected_flags[i, j] = 0
	np.testing.assert_array_equal(graph[2], expected_flags)
	np.testing.assert_array_equal(graph[0], indices)
	np.testing.assert_array_equal(graph[1], distances)


def test_new_build_candidates_respects_max_candidates():
	"""Check new build candidates respects max candidates."""
	rs = np.random.RandomState(2)
	n, k, mc = 30, 6, 4
	indices, distances, flags = _random_knn_graph(rs, n, k)
	exp_new, exp_old = _reference_candidates(indices, flags)
	graph = (indices.copy(), distances.copy(), flags.copy())
	new_c, old_c = U.new_build_candidates(graph, mc, _new_state(4), 2)
	assert new_c.shape == (n, mc)
	for v in range(n):
		assert _row_set(new_c[v]) <= exp_new[v]
		assert _row_set(old_c[v]) <= exp_old[v]
		assert len(_row_set(new_c[v])) == min(mc, len(exp_new[v]))
		assert len(_row_set(old_c[v])) == min(mc, len(exp_old[v]))


def test_new_build_candidates_all_old_graph_has_no_new_candidates():
	"""Check new build candidates all old graph has no new candidates."""
	rs = np.random.RandomState(3)
	indices, distances, flags = _random_knn_graph(rs, 12, 3, flag_probability=0.0)
	graph = (indices.copy(), distances.copy(), flags.copy())
	new_c, old_c = U.new_build_candidates(graph, 12, _new_state(1), 2)
	assert (new_c == -1).all()
	exp_new, exp_old = _reference_candidates(indices, flags)
	for v in range(12):
		assert _row_set(old_c[v]) == exp_old[v]
	np.testing.assert_array_equal(graph[2], flags)


def _owner_workspace(n, mc, k, n_threads):
	return dict(
		new_idx=np.full((n, mc), -1, dtype=np.int32),
		new_pri=np.full((n, mc), np.inf, dtype=np.float32),
		old_idx=np.full((n, mc), -1, dtype=np.int32),
		old_pri=np.full((n, mc), np.inf, dtype=np.float32),
		active=np.ones(n, dtype=np.uint8),
		reverse_dst=np.zeros(n * k, dtype=np.int32),
		reverse_src=np.zeros(n * k, dtype=np.int32),
		bounds=np.zeros((n_threads, n_threads + 1), dtype=np.int64),
	)


def _run_owner(graph, mc, rng, n_threads, ws):
	return U.build_candidates_owner(
		graph, mc, rng, n_threads,
		ws["new_idx"], ws["new_pri"], ws["old_idx"], ws["old_pri"],
		ws["active"], ws["reverse_dst"], ws["reverse_src"], ws["bounds"],
	)


@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_build_candidates_owner_matches_reference_sets(n_threads):
	"""Check build candidates owner matches reference sets."""
	rs = np.random.RandomState(10 + n_threads)
	n, k = 26, 5
	indices, distances, flags = _random_knn_graph(rs, n, k, flag_probability=0.4)
	exp_new, exp_old = _reference_candidates(indices, flags)
	graph = (indices.copy(), distances.copy(), flags.copy())
	ws = _owner_workspace(n, 2 * n, k, n_threads)
	new_c, old_c = _run_owner(graph, 2 * n, _new_state(2), n_threads, ws)
	assert new_c is ws["new_idx"] and old_c is ws["old_idx"]
	for v in range(n):
		assert _row_set(new_c[v]) == exp_new[v]
		if exp_new[v]:
			assert _row_set(old_c[v]) == exp_old[v]
			assert ws["active"][v] == 1
		else:
			assert ws["active"][v] == 0
			assert (old_c[v] == -1).all()
	expected_flags = flags.copy()
	for i in range(n):
		for j in range(k):
			if int(indices[i, j]) in exp_new[i]:
				expected_flags[i, j] = 0
	np.testing.assert_array_equal(graph[2], expected_flags)


def test_build_candidates_owner_workspace_reuse_across_calls():
	"""Check build candidates owner workspace reuse across calls."""
	rs = np.random.RandomState(5)
	n, k, nt = 30, 4, 3
	ws = _owner_workspace(n, n, k, nt)
	for round_number in range(3):
		indices, distances, flags = _random_knn_graph(rs, n, k, flag_probability=0.15)
		exp_new, exp_old = _reference_candidates(indices, flags)
		graph = (indices.copy(), distances.copy(), flags.copy())
		new_c, old_c = _run_owner(graph, n, _new_state(round_number), nt, ws)
		for v in range(n):
			assert _row_set(new_c[v]) == exp_new[v]
			if exp_new[v]:
				assert _row_set(old_c[v]) == exp_old[v]
			else:
				assert (old_c[v] == -1).all()
			assert ws["active"][v] == (1 if exp_new[v] else 0)


def test_build_candidates_owner_respects_max_candidates():
	"""Check build candidates owner respects max candidates."""
	rs = np.random.RandomState(6)
	n, k, mc, nt = 30, 6, 3, 2
	indices, distances, flags = _random_knn_graph(rs, n, k)
	exp_new, exp_old = _reference_candidates(indices, flags)
	graph = (indices.copy(), distances.copy(), flags.copy())
	ws = _owner_workspace(n, mc, k, nt)
	new_c, old_c = _run_owner(graph, mc, _new_state(9), nt, ws)
	for v in range(n):
		assert _row_set(new_c[v]) <= exp_new[v]
		assert len(_row_set(new_c[v])) == min(mc, len(exp_new[v]))
		assert _row_set(old_c[v]) <= exp_old[v]


# ---------------------------------------------------------------------------
# graph updates
# ---------------------------------------------------------------------------


def _ref_join_updates(new_blk, old_blk, hd, data, n_threads, max_updates, dist, skip_self_new):
	"""Restates the pair-enumeration loop of the update generators in plain Python."""
	block = new_blk.shape[0]
	rpt = block // n_threads + 1
	result = []
	for t in range(n_threads):
		ups = []

		def full():
			return len(ups) >= max_updates

		for r in range(rpt):
			i = t * rpt + r
			if i >= block or full():
				break
			for j in range(new_blk.shape[1]):
				if full():
					break
				p = int(new_blk[i, j])
				if p < 0:
					continue
				for k in range(j + 1, new_blk.shape[1]):
					if full():
						break
					q = int(new_blk[i, k])
					if q < 0 or (skip_self_new and q == p):
						continue
					d = dist(data[p], data[q])
					if d <= max(float(hd[p, 0]), float(hd[q, 0])):
						ups.append((p, q, d))
				for k in range(old_blk.shape[1]):
					if full():
						break
					q = int(old_blk[i, k])
					if q < 0 or q == p:
						continue
					d = dist(data[p], data[q])
					if d <= max(float(hd[p, 0]), float(hd[q, 0])):
						ups.append((p, q, d))
		result.append(ups)
	return result


def _ref_sqeuclid(x, y):
	diff = x.astype(np.float64) - y.astype(np.float64)
	return float(diff @ diff)


def _ref_alt_dot(x, y):
	r = float(x.astype(np.float64) @ y.astype(np.float64))
	if r <= 0.0:
		return float(F32_MAX)
	return float(-np.log2(r))


def _candidate_blocks(rs, block, n, mc_new, mc_old):
	new_blk = np.full((block, mc_new), -1, dtype=np.int32)
	old_blk = np.full((block, mc_old), -1, dtype=np.int32)
	for i in range(block):
		nn = rs.randint(0, mc_new + 1)
		no = rs.randint(0, mc_old + 1)
		new_blk[i, :nn] = rs.choice(n, nn, replace=False)
		old_blk[i, :no] = rs.choice(n, no, replace=False)
		if rs.rand() < 0.3 and nn > 2:
			new_blk[i, 1] = -1
	return new_blk, old_blk


def _assert_updates_match(got, expected, rtol=2e-4):
	assert len(got) == len(expected)
	for ups_got, ups_exp in zip(got, expected):
		assert len(ups_got) == len(ups_exp)
		for (p1, q1, d1), (p2, q2, d2) in zip(ups_got, ups_exp):
			assert (p1, q1) == (p2, q2)
			assert d1 == pytest.approx(d2, rel=rtol, abs=1e-5)


def _thresholds(rs, n, k, scale):
	hd = np.full((n, k), np.inf, dtype=np.float32)
	finite = rs.rand(n) < 0.7
	hd[finite, 0] = (rs.rand(finite.sum()) * scale).astype(np.float32)
	return hd


@pytest.mark.parametrize("n_threads", [1, 3])
@pytest.mark.parametrize("max_updates", [4, 10000])
def test_generate_graph_update_array_matches_reference(n_threads, max_updates):
	"""Check generate graph update array matches reference."""
	rs = np.random.RandomState(31)
	n, dim, block = 40, 6, 17
	data = rs.randn(n, dim).astype(np.float32)
	new_blk, old_blk = _candidate_blocks(rs, block, n, 6, 5)
	hd = _thresholds(rs, n, 3, 14.0)
	graph = (np.zeros((n, 3), dtype=np.int32), hd, np.zeros((n, 3), dtype=np.uint8))
	update_array = np.zeros((n_threads, max_updates if max_updates < 100 else 400, 3), dtype=np.float32)
	counts = np.zeros(n_threads, dtype=np.int64)
	U.generate_graph_update_array(
		update_array, counts, new_blk, old_blk, graph, data, pdist.squared_euclidean, n_threads
	)
	expected = _ref_join_updates(
		new_blk, old_blk, hd, data, n_threads, update_array.shape[1], _ref_sqeuclid, False
	)
	assert sum(len(e) for e in expected) > 0
	_assert_updates_match(_unpack_updates(update_array, counts), expected)
	if max_updates == 4:
		assert (counts == 4).any()


def test_generate_graph_update_array_threshold_filters_everything_or_nothing():
	"""Check generate graph update array threshold filters everything or nothing."""
	rs = np.random.RandomState(32)
	n, dim, block = 20, 4, 8
	data = rs.randn(n, dim).astype(np.float32)
	new_blk, old_blk = _candidate_blocks(rs, block, n, 5, 4)
	n_pairs_possible = sum(
		(new_blk[i] >= 0).sum() * ((new_blk[i] >= 0).sum() - 1) // 2
		+ sum(1 for p in new_blk[i] if p >= 0) * sum(1 for q in old_blk[i] if q >= 0)
		for i in range(block)
	)
	for scale_value, check in [(np.inf, "all"), (-1.0, "none")]:
		hd = np.full((n, 2), scale_value, dtype=np.float32)
		graph = (np.zeros((n, 2), dtype=np.int32), hd, np.zeros((n, 2), dtype=np.uint8))
		update_array = np.zeros((2, 500, 3), dtype=np.float32)
		counts = np.zeros(2, dtype=np.int64)
		U.generate_graph_update_array(
			update_array, counts, new_blk, old_blk, graph, data, pdist.squared_euclidean, 2
		)
		if check == "none":
			assert counts.sum() == 0
		else:
			overlap = sum(
				1
				for i in range(block)
				for p in new_blk[i] if p >= 0
				for q in old_blk[i] if q == p
			)
			assert counts.sum() == n_pairs_possible - overlap


def test_generate_graph_update_array_stores_ids_as_int32_bits():
	"""Check generate graph update array stores ids as int32 bits."""
	rs = np.random.RandomState(1)
	data = rs.randn(6, 2).astype(np.float32)
	new_blk = np.array([[0, 1, -1]], dtype=np.int32)
	old_blk = np.array([[5, -1]], dtype=np.int32)
	hd = np.full((6, 2), np.inf, dtype=np.float32)
	graph = (np.zeros((6, 2), dtype=np.int32), hd, np.zeros((6, 2), dtype=np.uint8))
	update_array = np.zeros((1, 10, 3), dtype=np.float32)
	counts = np.zeros(1, dtype=np.int64)
	U.generate_graph_update_array(
		update_array, counts, new_blk, old_blk, graph, data, pdist.squared_euclidean, 1
	)
	ids = update_array.view(np.int32)
	got = [(ids[0, j, 0], ids[0, j, 1]) for j in range(counts[0])]
	assert got == [(0, 1), (0, 5), (1, 5)]


def test_sparse_generate_graph_update_array_matches_dense_reference():
	"""Check sparse generate graph update array matches dense reference."""
	rs = np.random.RandomState(41)
	n, dim, block, n_threads = 36, 12, 15, 3
	dense = rs.randn(n, dim).astype(np.float32) * (rs.rand(n, dim) < 0.5)
	csr = sparse.csr_matrix(dense)
	csr.sort_indices()
	new_blk, old_blk = _candidate_blocks(rs, block, n, 5, 4)
	hd = _thresholds(rs, n, 3, 20.0)
	graph = (np.zeros((n, 3), dtype=np.int32), hd, np.zeros((n, 3), dtype=np.uint8))
	update_array = np.zeros((n_threads, 300, 3), dtype=np.float32)
	counts = np.zeros(n_threads, dtype=np.int64)
	U.sparse_generate_graph_update_array(
		update_array, counts, new_blk, old_blk, graph,
		csr.indices, csr.indptr, csr.data, psparse.sparse_squared_euclidean, n_threads,
	)
	expected = _ref_join_updates(
		new_blk, old_blk, hd, dense, n_threads, 300, _ref_sqeuclid, False
	)
	assert sum(len(e) for e in expected) > 0
	_assert_updates_match(_unpack_updates(update_array, counts), expected)


def _apply_reference(graph, update_lists):
	rows = [
		_row_entries(graph[0][i], graph[1][i], graph[2][i]) for i in range(graph[0].shape[0])
	]
	changes = 0
	for ups in update_lists:
		for p, q, d in ups:
			changes += _ref_push(rows[p], np.float32(d), q, 1, True)
			changes += _ref_push(rows[q], np.float32(d), p, 1, True)
	return rows, changes


@pytest.mark.parametrize("n_threads", [1, 3])
def test_apply_graph_update_array_matches_sequential_model(n_threads):
	"""Check apply graph update array matches sequential model."""
	rs = np.random.RandomState(50 + n_threads)
	n, k = 15, 4
	graph = _random_heap(rs, n, k, n, 3)
	ups = _random_updates(rs, n, n_threads, 40)
	update_array, counts = _pack_updates(ups, 64)
	expected_rows, expected_changes = _apply_reference(graph, ups)
	got_changes = U.apply_graph_update_array(graph, update_array, counts, n_threads)
	assert got_changes == expected_changes
	assert expected_changes > 0
	for i in range(n):
		assert _is_max_heap(graph[1][i])
		have = sorted(_row_entries(graph[0][i], graph[1][i], graph[2][i]))
		assert have == sorted(expected_rows[i])


def test_apply_graph_update_array_ignores_slots_past_the_counts():
	"""Check apply graph update array ignores slots past the counts."""
	rs = np.random.RandomState(7)
	n, k = 10, 3
	graph = _random_heap(rs, n, k, n, 3)
	ups = _random_updates(rs, n, 2, 6)
	update_array, counts = _pack_updates(ups, 6)
	counts[:] = [2, 0]
	expected_rows, expected_changes = _apply_reference(graph, [ups[0][:2]])
	assert U.apply_graph_update_array(graph, update_array, counts, 2) == expected_changes
	for i in range(n):
		assert sorted(_row_entries(graph[0][i], graph[1][i], graph[2][i])) == sorted(expected_rows[i])


def test_apply_graph_update_array_hand_case():
	"""Check apply graph update array hand case."""
	graph = U.make_heap(3, 2)
	update_array, counts = _pack_updates([[(0, 1, 1.5), (1, 2, 0.5), (0, 1, 0.25)]], 3)
	n_changes = U.apply_graph_update_array(graph, update_array, counts, 1)
	assert n_changes == 4
	rows = _rows_as_multisets(graph)
	assert rows[0] == [(1.5, 1, 1), (np.inf, -1, 0)]
	assert rows[1] == [(0.5, 2, 1), (1.5, 0, 1)]
	assert rows[2] == [(0.5, 1, 1), (np.inf, -1, 0)]


@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_sort_updates_by_block_buckets_and_apply_sorted_equals_apply(n_threads):
	"""Check sort updates by block buckets and apply sorted equals apply."""
	rs = np.random.RandomState(60 + n_threads)
	n, k, per = 23, 4, 30
	graph = _random_heap(rs, n, k, n, 3)
	ups = _random_updates(rs, n, n_threads, per)
	update_array, counts = _pack_updates(ups, 40)
	sorted_updates = np.zeros((n_threads, 80, 3), dtype=np.float32)
	offsets = np.zeros((n_threads, n_threads + 1), dtype=np.int64)
	U.sort_updates_by_block(update_array, counts, sorted_updates, offsets, n, n_threads)
	block_size = n // n_threads + 1
	ids = sorted_updates.view(np.int32)
	for t in range(n_threads):
		assert offsets[t, 0] == 0
		assert np.all(np.diff(offsets[t]) >= 0)
		for b in range(n_threads):
			expected = [
				(p, q, d) for p, q, d in ups[t]
				if p // block_size == b or q // block_size == b
			]
			got = [
				(int(ids[t, j, 0]), int(ids[t, j, 1]), float(sorted_updates[t, j, 2]))
				for j in range(offsets[t, b], offsets[t, b + 1])
			]
			assert got == expected
	g1, g2 = _copy_graph(graph), _copy_graph(graph)
	c1 = U.apply_graph_update_array(g1, update_array, counts, n_threads)
	c2 = U.apply_sorted_graph_update_array(g2, sorted_updates, offsets, n_threads)
	assert c1 == c2 and c1 > 0
	for a, b in zip(g1, g2):
		np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# heap initialisation from an existing graph
# ---------------------------------------------------------------------------


def _expected_init_rows(graph_indices, dist_of, k):
	n = graph_indices.shape[0]
	rows = []
	for i in range(n):
		seen = {}
		for j in graph_indices[i]:
			j = int(j)
			if j >= 0 and j != i and j not in seen:
				seen[j] = dist_of(i, j)
		best = sorted((np.float32(d), j) for j, d in seen.items())[:k]
		best = [(float(d), j, 1) for d, j in best]
		best += [(float(np.inf), -1, 0)] * (k - len(best))
		rows.append(sorted(best))
	return rows


def _graph_with_junk(rs, n, k):
	graph_indices = np.empty((n, k), dtype=np.int32)
	for i in range(n):
		graph_indices[i] = rs.randint(0, n, size=k)
	graph_indices[0, 0] = -1
	graph_indices[1, 1] = 1
	graph_indices[2] = [2, 2, 2, 2][:k] if k >= 4 else graph_indices[2]
	return graph_indices


def test_initalize_heap_from_graph_indices_matches_reference():
	"""Check initalize heap from graph indices matches reference."""
	rs = np.random.RandomState(70)
	n, k, dim = 12, 4, 5
	data = rs.randn(n, dim).astype(np.float32)
	graph_indices = _graph_with_junk(rs, n, k)
	heap = U.make_heap(n, k)
	out = U.initalize_heap_from_graph_indices(heap, graph_indices, data, pdist.squared_euclidean)
	expected = _expected_init_rows(graph_indices, lambda i, j: _ref_sqeuclid(data[i], data[j]), k)
	got = _rows_as_multisets(out)
	for i in range(n):
		assert sorted(g[1] for g in got[i]) == sorted(e[1] for e in expected[i])
		for g, e in zip(got[i], expected[i]):
			assert g[0] == pytest.approx(e[0], rel=1e-5, abs=1e-6) or (np.isinf(g[0]) and np.isinf(e[0]))
	assert all(_is_max_heap(out[1][i]) for i in range(n))


def test_initalize_heap_from_graph_indices_and_distances_matches_reference():
	"""Check initalize heap from graph indices and distances matches reference."""
	rs = np.random.RandomState(71)
	n, k = 12, 4
	graph_indices = _graph_with_junk(rs, n, k)
	graph_distances = (rs.permutation(n * k).reshape(n, k) + 1).astype(np.float32)
	heap = U.make_heap(n, k)
	out = U.initalize_heap_from_graph_indices_and_distances(heap, graph_indices, graph_distances)
	lookup = {(i, int(graph_indices[i, idx])): graph_distances[i, idx] for i in range(n) for idx in range(k - 1, -1, -1)}
	expected = _expected_init_rows(graph_indices, lambda i, j: lookup[(i, j)], k)
	assert _rows_as_multisets(out) == expected
	assert np.shares_memory(out[0], heap[0])
	assert all(_is_max_heap(out[1][i]) for i in range(n))


def test_initalize_heap_skips_self_and_negative_entries():
	"""Check initalize heap skips self and negative entries."""
	graph_indices = np.array([[0, -1, 2], [1, 1, 0], [2, 0, 1]], dtype=np.int32)
	graph_distances = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
	heap = U.make_heap(3, 3)
	out = U.initalize_heap_from_graph_indices_and_distances(heap, graph_indices, graph_distances)
	rows = _rows_as_multisets(out)
	assert rows[0] == [(3.0, 2, 1), (np.inf, -1, 0), (np.inf, -1, 0)]
	assert rows[1] == [(6.0, 0, 1), (np.inf, -1, 0), (np.inf, -1, 0)]
	assert rows[2] == [(8.0, 0, 1), (9.0, 1, 1), (np.inf, -1, 0)]


def test_sparse_initalize_heap_from_graph_indices_matches_dense_metric():
	"""Check sparse initalize heap from graph indices matches dense metric."""
	rs = np.random.RandomState(72)
	n, k, dim = 14, 4, 9
	dense = rs.randn(n, dim).astype(np.float32) * (rs.rand(n, dim) < 0.6)
	csr = sparse.csr_matrix(dense)
	csr.sort_indices()
	graph_indices = _graph_with_junk(rs, n, k)
	heap = U.make_heap(n, k)
	out = U.sparse_initalize_heap_from_graph_indices(
		heap, graph_indices, csr.indptr, csr.indices, csr.data, psparse.sparse_squared_euclidean
	)
	expected = _expected_init_rows(graph_indices, lambda i, j: _ref_sqeuclid(dense[i], dense[j]), k)
	got = _rows_as_multisets(out)
	for i in range(n):
		assert sorted(g[1] for g in got[i]) == sorted(e[1] for e in expected[i])
		for g, e in zip(got[i], expected[i]):
			assert g[0] == pytest.approx(e[0], rel=1e-4, abs=1e-5) or (np.isinf(g[0]) and np.isinf(e[0]))


def test_ts_returns_ctime_string():
	"""Call ts() and check it equals time.ctime() of some second between the calls around it."""
	before = int(time.time())
	text = U.ts()
	after = int(time.time()) + 1
	assert isinstance(text, str)
	assert text in {time.ctime(t) for t in range(before, after + 1)}


# ---------------------------------------------------------------------------
# finishing queries
# ---------------------------------------------------------------------------


def _finish_inputs(seed_value, n=7, k=5, scale=1.0):
	rs = np.random.RandomState(seed_value)
	graph = U.make_heap(n, k)
	for i in range(n):
		priorities = (rs.permutation(60)[:k] + 1).astype(np.float32) * np.float32(scale) / 10
		ids = rs.choice(30, k, replace=False).astype(np.int32)
		for pri, idx in zip(priorities, ids):
			U.checked_heap_push(graph[1][i], graph[0][i], pri, idx)
	order = rs.permutation(30)
	return graph[0], graph[1], order


def _sorted_reference(indices, distances, order, transform):
	exp_i, exp_d = [], []
	for i in range(indices.shape[0]):
		pairs = sorted(zip(distances[i].tolist(), indices[i].tolist()))
		exp_i.append([order[j] for _, j in pairs])
		exp_d.append([transform(np.float64(d)) for d, _ in pairs])
	return np.array(exp_i), np.array(exp_d)


@pytest.mark.parametrize("order_dtype", [np.int64, np.int32])
def test_finish_query_no_correction(order_dtype):
	"""Check rows are sorted by distance and indices mapped through the order array, against a Python sort."""
	indices, distances, order = _finish_inputs(1)
	order = order.astype(order_dtype)
	exp_i, exp_d = _sorted_reference(indices, distances, order, lambda d: d)
	out_i, out_d = U.finish_query_no_correction(indices, distances, order)
	assert out_i.dtype == order_dtype and out_d.dtype == np.float32
	np.testing.assert_array_equal(out_i, exp_i)
	np.testing.assert_array_equal(out_d, exp_d.astype(np.float32))
	assert out_i is not indices and out_d is not distances


def test_finish_query_sqrt():
	"""Check sorting, order mapping and the square root of distances against a Python reference."""
	indices, distances, order = _finish_inputs(2, scale=30.0)
	exp_i, exp_d = _sorted_reference(indices, distances, order, np.sqrt)
	out_i, out_d = U.finish_query_sqrt(indices, distances, order)
	assert out_d.dtype == np.float32 and out_i.dtype == order.dtype
	np.testing.assert_array_equal(out_i, exp_i)
	np.testing.assert_allclose(out_d, exp_d, rtol=1e-6)
	assert np.all(np.diff(out_d, axis=1) >= 0)


def test_finish_query_alternative_cosine():
	"""Check sorting, order mapping and 1 - 2**-d of distances against a Python reference."""
	indices, distances, order = _finish_inputs(3, scale=4.0)
	exp_i, exp_d = _sorted_reference(
		indices, distances, order, lambda d: max(0.0, 1.0 - 2.0 ** (-d))
	)
	out_i, out_d = U.finish_query_alternative_cosine(indices, distances, order)
	assert out_d.dtype == np.float64
	np.testing.assert_array_equal(out_i, exp_i)
	np.testing.assert_allclose(out_d, exp_d, rtol=1e-9, atol=1e-12)


def test_finish_query_alternative_cosine_clamps_at_zero():
	"""Pins the current output: negative alternative distances map to 0."""
	indices = np.array([[0, 1]], dtype=np.int32)
	distances = np.array([[-0.5, 0.0]], dtype=np.float32)
	order = np.array([10, 11], dtype=np.int64)
	_, out_d = U.finish_query_alternative_cosine(indices.copy(), distances.copy(), order)
	np.testing.assert_allclose(out_d, [[0.0, 0.0]], atol=1e-12)


def test_finish_query_alternative_hellinger():
	"""Check sorting, order mapping and sqrt(1 - 2**-d) of distances against a Python reference."""
	indices, distances, order = _finish_inputs(4, scale=4.0)
	exp_i, exp_d = _sorted_reference(
		indices, distances, order, lambda d: np.sqrt(max(0.0, 1.0 - 2.0 ** (-d)))
	)
	out_i, out_d = U.finish_query_alternative_hellinger(indices, distances, order)
	assert out_d.dtype == np.float64
	np.testing.assert_array_equal(out_i, exp_i)
	np.testing.assert_allclose(out_d, exp_d, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_l2_normalize_rows(dtype):
	"""Check rows are scaled in place to unit norm and zero rows left alone, against numpy norms."""
	rs = np.random.RandomState(0)
	X = rs.randn(9, 6).astype(dtype)
	X[3] = 0
	original = X.copy()
	out = U.l2_normalize_rows(X)
	assert out is X
	norms = np.linalg.norm(X.astype(np.float64), axis=1)
	expected_norms = np.where(np.linalg.norm(original.astype(np.float64), axis=1) == 0, 0.0, 1.0)
	np.testing.assert_allclose(norms, expected_norms, atol=1e-6)
	np.testing.assert_allclose(
		X[0], original[0] / np.linalg.norm(original[0].astype(np.float64)), rtol=1e-5
	)
	assert (X[3] == 0).all()


# ---------------------------------------------------------------------------
# graph_utils
# ---------------------------------------------------------------------------


def test_adjacency_matrix_representation_hand_case():
	"""Check a 3-vertex hand case; pins the current output that zero distances are replaced by FLOAT32_EPS in the input array."""
	indices = np.array([[1, 2], [0, 2], [0, -1]], dtype=np.int64)
	distances = np.array([[1.0, 0.0], [1.0, 2.0], [0.0, 5.0]], dtype=np.float32)
	result = adjacency_matrix_representation(indices, distances)
	assert sparse.isspmatrix_csr(result) or isinstance(result, sparse.csr_array)
	dense = result.toarray()
	expected = np.array(
		[
			[0.0, 1.0, FLOAT32_EPS],
			[1.0, 0.0, 2.0],
			[FLOAT32_EPS, 2.0, 0.0],
		],
		dtype=np.float32,
	)
	np.testing.assert_array_equal(dense, expected)
	assert result.dtype == np.float32
	assert not (distances == 0).any()


def test_adjacency_matrix_representation_is_symmetric_max_of_directed_edges():
	"""Check adjacency matrix representation is symmetric max of directed edges."""
	rs = np.random.RandomState(3)
	n, k = 20, 4
	indices, distances, _ = _random_knn_graph(rs, n, k)
	distances = distances + 0.1
	directed = np.zeros((n, n), dtype=np.float32)
	for i in range(n):
		for j in range(k):
			directed[i, indices[i, j]] = distances[i, j]
	result = adjacency_matrix_representation(indices.astype(np.int64), distances.copy())
	expected = np.maximum(directed, directed.T)
	np.testing.assert_array_equal(result.toarray(), expected)
	assert (result != result.T).nnz == 0
	assert result.diagonal().sum() == 0


def test_adjacency_matrix_representation_preserves_zero_distance_edges():
	"""Check zero-distance edges stay stored as FLOAT32_EPS (pins the current output)."""
	indices = np.array([[1], [0]], dtype=np.int64)
	distances = np.array([[0.0], [0.0]], dtype=np.float32)
	result = adjacency_matrix_representation(indices, distances)
	assert result.nnz == 2
	assert result[0, 1] == FLOAT32_EPS and result[1, 0] == FLOAT32_EPS
	assert FLOAT32_EPS == np.finfo(np.float32).eps


@pytest.fixture(scope="module")
def clustered_index():
	rs = np.random.RandomState(0)
	X = np.vstack(
		[rs.randn(30, 4), rs.randn(30, 4) + 30, rs.randn(20, 4) - 30]
	).astype(np.float32)
	index = NNDescent(X, n_neighbors=8, random_state=0, n_jobs=1)
	index._init_search_graph()
	return X, index


@pytest.fixture(scope="module")
def cosine_index():
	rs = np.random.RandomState(1)
	X = (rs.randn(60, 5) + 2.0).astype(np.float32)
	index = NNDescent(X, metric="cosine", n_neighbors=8, random_state=0, n_jobs=1)
	index._init_search_graph()
	return X, index


def test_create_component_search_is_exact_when_every_point_is_a_candidate(clustered_index):
	"""Check create component search is exact when every point is a candidate."""
	X, index = clustered_index
	search = create_component_search(index)
	rs = np.random.RandomState(5)
	queries = rs.randn(6, 4).astype(np.float32) * 20
	data = index._raw_data
	visited = index._visited.copy()
	indices, distances, flags = search(queries, np.arange(data.shape[0]), 4, 0.0, visited)
	brute = ((queries[:, None, :].astype(np.float64) - data[None].astype(np.float64)) ** 2).sum(-1)
	for i in range(queries.shape[0]):
		order = np.argsort(brute[i])[:4]
		assert sorted(indices[i].tolist()) == sorted(order.tolist())
		np.testing.assert_allclose(
			sorted(distances[i].tolist()), np.sort(brute[i])[:4], rtol=1e-4
		)


def test_create_component_search_distances_match_returned_ids(clustered_index):
	"""Check create component search distances match returned ids."""
	X, index = clustered_index
	search = create_component_search(index)
	data = index._raw_data
	rs = np.random.RandomState(6)
	queries = data[rs.choice(80, 5, replace=False)] + 0.01
	visited = index._visited.copy()
	indices, distances, _ = search(queries, np.array([0]), 3, 0.2, visited)
	for i in range(5):
		assert (indices[i] >= 0).all() and len(set(indices[i].tolist())) == 3
		for j in range(3):
			true = ((queries[i].astype(np.float64) - data[indices[i, j]]) ** 2).sum()
			assert distances[i, j] == pytest.approx(true, rel=1e-4)


def test_create_component_search_cosine_normalises_queries_and_skips_zero(cosine_index):
	"""Check create component search cosine normalises queries and skips zero."""
	X, index = cosine_index
	search = create_component_search(index)
	data = index._raw_data
	rs = np.random.RandomState(8)
	queries = (rs.rand(4, 5) * 3 + 0.5).astype(np.float32)
	queries[2] = 0.0
	visited = index._visited.copy()
	indices, distances, _ = search(queries, np.arange(data.shape[0]), 3, 0.0, visited)
	for i in range(4):
		if i == 2:
			assert (indices[i] == -1).all() and np.isinf(distances[i]).all()
			continue
		unit = queries[i].astype(np.float64) / np.linalg.norm(queries[i].astype(np.float64))
		dots = data.astype(np.float64) @ unit
		top = np.argsort(-dots)[:3]
		assert sorted(indices[i].tolist()) == sorted(top.tolist())
		np.testing.assert_allclose(
			sorted(distances[i].tolist()), np.sort(-np.log2(dots[top])), rtol=1e-4, atol=1e-6
		)


def _internal_cluster_ids(index):
	"""Blob label (0, 1, 2) of every row of the index's internal data, read from the blob centres 0, +30 and -30."""
	total = np.asarray(index._raw_data, dtype=np.float64).sum(axis=1)
	return np.where(total > 60, 1, np.where(total < -60, 2, 0))


def test_find_component_connection_edge_returns_a_cross_component_pair(clustered_index):
	"""Check the returned pair spans the two blobs and its distance is the true squared distance of that pair, at least the brute-force minimum."""
	X, index = clustered_index
	search = create_component_search(index)
	data = index._raw_data
	internal_ids = _internal_cluster_ids(index)
	c1 = np.where(internal_ids == 0)[0]
	c2 = np.where(internal_ids == 1)[0]
	assert len(c1) == 30 and len(c2) == 30
	i, j, d = find_component_connection_edge(
		c1, c2, search, data, index._visited, index.rng_state.copy()
	)
	assert (i in c1 and j in c2) or (i in c2 and j in c1)
	true_pair = ((data[i].astype(np.float64) - data[j]) ** 2).sum()
	assert d == pytest.approx(true_pair, rel=1e-4)
	brute = ((data[c1][:, None].astype(np.float64) - data[c2][None]) ** 2).sum(-1)
	assert d >= brute.min() * (1 - 1e-4)


def test_find_component_connection_edge_works_in_either_argument_order(clustered_index):
	"""Check both argument orders return a pair spanning the two blobs with the true distance of that pair."""
	X, index = clustered_index
	data = index._raw_data
	internal_ids = _internal_cluster_ids(index)
	search = create_component_search(index)
	c1 = np.where(internal_ids == 1)[0]
	c2 = np.where(internal_ids == 2)[0]
	for first, second in [(c1, c2), (c2, c1)]:
		i, j, d = find_component_connection_edge(
			first, second, search, data, index._visited, index.rng_state.copy()
		)
		assert (i in c1 and j in c2) or (i in c2 and j in c1)
		true_pair = ((data[i].astype(np.float64) - data[j]) ** 2).sum()
		assert d == pytest.approx(true_pair, rel=1e-4)


def test_connect_graph_joins_components_with_cross_component_edges(clustered_index):
	"""Check connect_graph yields one component, keeps the old edges, and adds one symmetric edge per blob pair weighted by the true distance."""
	X, index = clustered_index
	nb = index.neighbor_graph
	graph = adjacency_matrix_representation(nb[0].copy(), nb[1].copy())
	n_before, comp = connected_components(graph)
	assert n_before == 3
	result = connect_graph(graph, index, n_jobs=1)
	assert sparse.issparse(result) and result.format == "csr"
	assert connected_components(result)[0] == 1
	original = graph.toarray()
	merged = result.toarray()
	new_mask = (merged != 0) & (original == 0)
	assert new_mask.sum() == 2 * 3
	np.testing.assert_array_equal(merged[original != 0], original[original != 0])
	np.testing.assert_array_equal(merged, merged.T)
	X64 = X.astype(np.float64)
	pairs = set()
	for i, j in zip(*np.nonzero(np.triu(new_mask))):
		assert comp[i] != comp[j]
		pairs.add(tuple(sorted((comp[i], comp[j]))))
		true = np.linalg.norm(X64[i] - X64[j])
		assert merged[i, j] == pytest.approx(true, rel=1e-4)
	assert pairs == {(0, 1), (0, 2), (1, 2)}
	for c1, c2 in combinations(range(3), 2):
		a, b = np.where(comp == c1)[0], np.where(comp == c2)[0]
		brute = np.linalg.norm(X64[a][:, None] - X64[b][None], axis=-1).min()
		cross = merged[np.ix_(a, b)]
		assert cross[cross != 0].min() >= brute * (1 - 1e-4)


def test_connect_graph_leaves_a_connected_graph_unchanged(clustered_index):
	"""Check connect graph leaves a connected graph unchanged."""
	X, index = clustered_index
	nb = index.neighbor_graph
	graph = adjacency_matrix_representation(nb[0].copy(), nb[1].copy())
	full = connect_graph(graph, index, n_jobs=1)
	again = connect_graph(full, index, n_jobs=1)
	np.testing.assert_array_equal(again.toarray(), full.toarray())


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def test_kernel_ids():
	"""Pins the current output: the kernel id constants and FLOAT32_MAX."""
	assert kernels.KERNEL_GENERIC == 0
	assert kernels.KERNEL_SQEUCLIDEAN == 1
	assert kernels.KERNEL_ALT_DOT == 2
	assert kernels.FLOAT32_MAX == F32_MAX


def test_select_kernel_picks_specialised_kernels_for_c_contiguous_float32():
	"""Check select kernel picks specialised kernels for c contiguous float32."""
	data = np.zeros((5, 4), dtype=np.float32)
	assert kernels.select_kernel(pdist.squared_euclidean, data) == 1
	assert kernels.select_kernel(pdist.alternative_dot, data) == 2


def test_select_kernel_falls_back_to_generic():
	"""Check select kernel falls back to generic."""
	data = np.zeros((5, 4), dtype=np.float32)
	assert kernels.select_kernel(pdist.euclidean, data) == 0
	assert kernels.select_kernel(pdist.cosine, data) == 0
	assert kernels.select_kernel(pdist.alternative_cosine, data) == 0
	assert kernels.select_kernel(pdist.squared_euclidean, data.astype(np.float64)) == 0
	assert kernels.select_kernel(pdist.squared_euclidean, np.asfortranarray(data)) == 0
	assert kernels.select_kernel(pdist.squared_euclidean, np.zeros((5, 8), dtype=np.float32)[:, ::2]) == 0
	assert kernels.select_kernel(pdist.squared_euclidean, np.zeros(5, dtype=np.float32)) == 0
	assert kernels.select_kernel(pdist.squared_euclidean, np.zeros((2, 3, 4), dtype=np.float32)) == 0
	assert kernels.select_kernel(pdist.squared_euclidean, data.astype(np.int32)) == 0
	assert kernels.select_kernel(pdist.squared_euclidean, data.tolist()) == 0
	assert kernels.select_kernel(pdist.squared_euclidean, sparse.csr_matrix(data)) == 0
	assert kernels.select_kernel(pdist.alternative_dot, data.astype(np.float64)) == 0


def test_select_kernel_accepts_a_row_slice_that_stays_contiguous():
	"""Check select kernel accepts a row slice that stays contiguous."""
	data = np.zeros((10, 4), dtype=np.float32)
	assert kernels.select_kernel(pdist.squared_euclidean, data[2:7]) == 1


@pytest.mark.parametrize(
	"r,expected",
	[
		(1.0, 0.0),
		(0.5, 1.0),
		(0.25, 2.0),
		(2.0, -1.0),
		(0.1, -np.log2(0.1)),
		(0.0, F32_MAX),
		(-0.5, F32_MAX),
		(-1e-30, F32_MAX),
	],
)
def test_alt_dot_from_dot(r, expected):
	"""Check -log2(r) for positive r; pins the current output of FLOAT32_MAX for r <= 0."""
	got = kernels.alt_dot_from_dot(np.float32(r))
	assert got == pytest.approx(np.float32(expected), rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("kernel", [1, 2])
def test_four_distances_matches_reference(kernel):
	"""Check four distances matches reference."""
	rs = np.random.RandomState(kernel)
	dim = 11
	data = rs.randn(20, dim).astype(np.float32)
	data /= np.linalg.norm(data, axis=1, keepdims=True)
	query = data[0]
	candidates = np.array([3, 7, 7, 19], dtype=np.int32)
	out = np.zeros(4, dtype=np.float32)
	kernels.four_distances(query, data, candidates, out, kernel)
	ref = _ref_sqeuclid if kernel == 1 else _ref_alt_dot
	for slot, q in enumerate(candidates):
		assert out[slot] == pytest.approx(ref(query, data[q]), rel=1e-4, abs=1e-5)


def test_four_distances_alt_dot_marks_non_positive_dots_as_far():
	"""Pins the current output: non-positive dots give FLOAT32_MAX, positive ones -log2(dot)."""
	query = np.array([1.0, 0.0], dtype=np.float32)
	data = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.5, 0.0]], dtype=np.float32)
	out = np.zeros(4, dtype=np.float32)
	kernels.four_distances(query, data, np.arange(4, dtype=np.int32), out, 2)
	np.testing.assert_allclose(out, [0.0, F32_MAX, F32_MAX, 1.0], rtol=1e-6)


def test_four_distances_hand_squared_euclidean():
	"""Check four distances hand squared euclidean."""
	query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
	data = np.array(
		[[1, 2, 3], [2, 2, 3], [0, 0, 0], [1, 4, 3]], dtype=np.float32
	)
	out = np.zeros(4, dtype=np.float32)
	kernels.four_distances(query, data, np.arange(4, dtype=np.int32), out, 1)
	np.testing.assert_allclose(out, [0.0, 1.0, 14.0, 4.0], rtol=1e-6)


def _kernel_data(kernel, rs, n, dim):
	data = rs.randn(n, dim).astype(np.float32)
	if kernel == 2:
		data /= np.linalg.norm(data, axis=1, keepdims=True)
	return data


@pytest.mark.parametrize("kernel", [1, 2])
@pytest.mark.parametrize("max_updates", [5, 6, 7, 400])
def test_generate_graph_update_array_batched_matches_reference(kernel, max_updates):
	"""Check generate graph update array batched matches reference."""
	rs = np.random.RandomState(80 + kernel)
	n, dim, block, n_threads = 45, 7, 13, 3
	data = _kernel_data(kernel, rs, n, dim)
	new_blk, old_blk = _candidate_blocks(rs, block, n, 9, 8)
	new_blk[0] = np.arange(9)
	old_blk[0] = np.arange(10, 18)
	scale = 14.0 if kernel == 1 else 4.0
	hd = _thresholds(rs, n, 3, scale)
	graph = (np.zeros((n, 3), dtype=np.int32), hd, np.zeros((n, 3), dtype=np.uint8))
	update_array = np.zeros((n_threads, max_updates, 3), dtype=np.float32)
	counts = np.zeros(n_threads, dtype=np.int64)
	kernels.generate_graph_update_array_batched(
		update_array, counts, new_blk, old_blk, graph, data, kernel, n_threads
	)
	ref = _ref_sqeuclid if kernel == 1 else _ref_alt_dot
	expected = _ref_join_updates(new_blk, old_blk, hd, data, n_threads, max_updates, ref, True)
	assert sum(len(e) for e in expected) > 0
	_assert_updates_match(_unpack_updates(update_array, counts), expected)
	if max_updates < 20:
		assert (counts == max_updates).any()


@pytest.mark.parametrize("kernel,dist", [(1, pdist.squared_euclidean), (2, pdist.alternative_dot)])
def test_generate_graph_update_array_batched_agrees_with_pairwise_generator(kernel, dist):
	"""Check generate graph update array batched agrees with pairwise generator."""
	rs = np.random.RandomState(90 + kernel)
	n, dim, block, n_threads = 40, 6, 12, 2
	data = _kernel_data(kernel, rs, n, dim)
	new_blk, old_blk = _candidate_blocks(rs, block, n, 8, 6)
	hd = _thresholds(rs, n, 2, 12.0 if kernel == 1 else 3.0)
	graph = (np.zeros((n, 2), dtype=np.int32), hd, np.zeros((n, 2), dtype=np.uint8))
	ua1 = np.zeros((n_threads, 300, 3), dtype=np.float32)
	c1 = np.zeros(n_threads, dtype=np.int64)
	ua2 = np.zeros((n_threads, 300, 3), dtype=np.float32)
	c2 = np.zeros(n_threads, dtype=np.int64)
	U.generate_graph_update_array(ua1, c1, new_blk, old_blk, graph, data, dist, n_threads)
	kernels.generate_graph_update_array_batched(ua2, c2, new_blk, old_blk, graph, data, kernel, n_threads)
	_assert_updates_match(_unpack_updates(ua2, c2), _unpack_updates(ua1, c1), rtol=1e-3)
