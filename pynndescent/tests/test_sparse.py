"""Tests for pynndescent.sparse and pynndescent.sparse_nndescent.

Sparse metrics and helpers are compared against dense numpy/scipy references
computed from the same CSR rows; the sparse NN-descent is compared against
brute force.
"""

import itertools

import numba
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.spatial.distance as scd
import scipy.stats

import pynndescent.sparse as spdist
import pynndescent.sparse_nndescent as sparse_nnd
import pynndescent.utils as nnutils
from pynndescent.distances import kantorovich as dense_kantorovich

FLOAT32_MAX = np.finfo(np.float32).max
FLOAT32_EPS = np.finfo(np.float32).eps


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rng_state(seed):
	"""A valid 3-element tausworthe state (components must exceed 1, 7, 15)."""
	r = np.random.RandomState(seed)
	return r.randint(1000, 2**30, size=3).astype(np.int64)


def _vec(indices, data=None):
	ind = np.asarray(indices, dtype=np.int32)
	if data is None:
		data = np.ones(ind.shape[0])
	return ind, np.asarray(data, dtype=np.float32)


def _random_csr(n_rows, n_cols, density, seed, positive=False, explicit_zeros=0,
	empty_rows=(), binary=False):
	"""Sorted CSR matrix (float32, int32 indices) with chosen empty rows and a
	number of stored entries overwritten with explicit zeros."""
	rng = np.random.RandomState(seed)
	dense = rng.randn(n_rows, n_cols)
	if positive:
		dense = np.abs(dense) + 0.1
	if binary:
		dense = np.ones((n_rows, n_cols))
	dense = dense * (rng.uniform(size=(n_rows, n_cols)) < density)
	for r in empty_rows:
		dense[r] = 0.0
	m = sp.csr_matrix(dense.astype(np.float32))
	m.sort_indices()
	m.indices = m.indices.astype(np.int32)
	m.indptr = m.indptr.astype(np.int32)
	if explicit_zeros:
		stored = np.flatnonzero(m.data != 0)
		pick = rng.choice(stored, size=min(explicit_zeros, stored.shape[0]),
			replace=False)
		m.data[pick] = 0.0
	return m


def _row(m, i):
	s, e = m.indptr[i], m.indptr[i + 1]
	return m.indices[s:e], m.data[s:e]


def _pairs(m):
	n = m.shape[0]
	for i in range(n):
		for j in range(n):
			yield i, j, _row(m, i), _row(m, j)


def _check_pairs(m, func, ref, *extra, rtol=2e-4, atol=1e-5, skip=None):
	dense = m.toarray().astype(np.float64)
	n_checked = 0
	for i, j, (i1, d1), (i2, d2) in _pairs(m):
		x, y = dense[i], dense[j]
		if skip is not None and skip(x, y):
			continue
		expected = ref(x, y)
		got = func(i1, d1, i2, d2, *extra)
		np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol,
			err_msg="rows %d, %d" % (i, j))
		n_checked += 1
	assert n_checked > 0


def _support_counts(x, y):
	bx, by = x != 0, y != 0
	return (int(np.sum(bx & by)), int(np.sum(bx & ~by)), int(np.sum(~bx & by)),
		int(np.sum(~bx & ~by)))


@pytest.fixture(scope="module")
def real_csr():
	return _random_csr(14, 30, 0.3, seed=1, explicit_zeros=6, empty_rows=(3, 11))


@pytest.fixture(scope="module")
def positive_csr():
	return _random_csr(14, 30, 0.3, seed=2, positive=True, empty_rows=(5,))


@pytest.fixture(scope="module")
def binary_csr():
	return _random_csr(14, 30, 0.3, seed=3, binary=True, empty_rows=(0, 8))


# ---------------------------------------------------------------------------
# array helpers
# ---------------------------------------------------------------------------


def test_isclose_matches_numpy():
	a = np.array([0.0, 1.0, 1.0, 1e-9, 1.0, -2.0, 100.0])
	b = np.array([0.0, 1.0 + 5e-6, 1.0 + 1e-3, 0.0, 1.0 + 2e-5, -2.0, 100.0005])
	got = np.array([spdist.isclose(x, y) for x, y in zip(a, b)])
	assert np.array_equal(got, np.isclose(a, b))
	assert spdist.isclose(0.0, 1e-7, atol=1e-6)
	assert not spdist.isclose(0.0, 1e-7, atol=1e-8)
	assert spdist.isclose(100.0, 101.0, rtol=0.02)
	assert not spdist.isclose(100.0, 101.0, rtol=0.001)


def test_arr_unique_sorted_unique_unsorted_input():
	rng = np.random.RandomState(0)
	arr = rng.randint(0, 20, size=60).astype(np.int32)
	got = spdist.arr_unique(arr)
	assert np.array_equal(got, np.unique(arr))
	assert got.dtype == np.int32
	assert np.array_equal(spdist.arr_unique(np.array([4], dtype=np.int32)), [4])
	assert np.array_equal(spdist.arr_unique(np.array([2, 2, 2], dtype=np.int32)), [2])


def test_arr_unique_floats_and_input_untouched():
	arr = np.array([3.5, 1.0, 3.5, -1.0, 1.0])
	orig = arr.copy()
	assert np.array_equal(spdist.arr_unique(arr), np.unique(arr))
	assert np.array_equal(arr, orig)


def test_arr_union_matches_union1d():
	rng = np.random.RandomState(1)
	for _ in range(20):
		a = rng.randint(0, 15, size=rng.randint(1, 10)).astype(np.int32)
		b = rng.randint(0, 15, size=rng.randint(1, 10)).astype(np.int32)
		assert np.array_equal(spdist.arr_union(a, b), np.union1d(a, b))


def test_arr_union_with_empty_returns_other_array_as_given():
	a = np.array([5, 1, 3, 1], dtype=np.int32)
	empty = np.empty(0, dtype=np.int32)
	assert np.array_equal(spdist.arr_union(empty, a), a)
	assert np.array_equal(spdist.arr_union(a, empty), a)
	assert spdist.arr_union(empty, empty).shape == (0,)


def test_arr_intersect_sorted_unique_inputs():
	rng = np.random.RandomState(2)
	for _ in range(20):
		a = np.unique(rng.randint(0, 25, size=rng.randint(2, 15))).astype(np.int32)
		b = np.unique(rng.randint(0, 25, size=rng.randint(2, 15))).astype(np.int32)
		assert np.array_equal(spdist.arr_intersect(a, b), np.intersect1d(a, b))


def test_arr_intersect_disjoint_is_empty():
	a = np.array([1, 3, 5], dtype=np.int32)
	b = np.array([0, 2, 4], dtype=np.int32)
	assert spdist.arr_intersect(a, b).shape == (0,)


def test_fast_intersection_size_matches_intersect1d():
	rng = np.random.RandomState(3)
	for _ in range(60):
		a = np.unique(rng.randint(0, 40, size=rng.randint(0, 20))).astype(np.int32)
		b = np.unique(rng.randint(0, 40, size=rng.randint(0, 20))).astype(np.int32)
		assert spdist.fast_intersection_size(a, b) == np.intersect1d(a, b).shape[0]


@pytest.mark.parametrize(
	"a,b,expected",
	[
		([], [], 0),
		([1, 2], [], 0),
		([], [1, 2], 0),
		([4], [4], 1),
		([4], [5], 0),
		([0, 1, 2, 3], [0, 1, 2, 3], 4),
		([0, 1, 2], [3, 4, 5], 0),
		([3, 4, 5], [0, 1, 2], 0),
		([0, 9], [0, 3, 9], 2),
		([1, 5, 9], [9], 1),
		([1, 5, 9], [1], 1),
		([2, 4, 6, 8], [1, 3, 5, 7, 9], 0),
	],
)
def test_fast_intersection_size_hand_cases(a, b, expected):
	a = np.array(a, dtype=np.int32)
	b = np.array(b, dtype=np.int32)
	assert spdist.fast_intersection_size(a, b) == expected
	assert spdist.fast_intersection_size(b, a) == expected


# ---------------------------------------------------------------------------
# sparse arithmetic
# ---------------------------------------------------------------------------


def test_sparse_sum_hand_case_drops_cancelled_entries():
	ind, data = spdist.sparse_sum(
		*_vec([0, 2, 7], [1.0, 3.0, 2.0]), *_vec([2, 5, 7], [-3.0, 4.0, 1.0])
	)
	assert np.array_equal(ind, [0, 5, 7])
	assert np.allclose(data, [1.0, 4.0, 3.0])
	assert ind.dtype == np.int32 and data.dtype == np.float32


def test_sparse_sum_and_diff_match_dense(real_csr):
	dense = real_csr.toarray().astype(np.float64)
	for i, j, (i1, d1), (i2, d2) in _pairs(real_csr):
		for func, ref in ((spdist.sparse_sum, dense[i] + dense[j]),
			(spdist.sparse_diff, dense[i] - dense[j])):
			ind, data = func(i1, d1, i2, d2)
			out = np.zeros(dense.shape[1])
			out[ind] = data
			np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)
			assert np.all(np.diff(ind) > 0)
			assert np.all(data != 0)
			assert ind.shape == data.shape


def test_sparse_sum_empty_operands():
	e = _vec([], [])
	x = _vec([1, 4], [2.0, -1.0])
	for a, b in ((e, x), (x, e)):
		ind, data = spdist.sparse_sum(*a, *b)
		assert np.array_equal(ind, [1, 4])
		assert np.allclose(data, [2.0, -1.0])
	ind, data = spdist.sparse_sum(*e, *e)
	assert ind.shape == (0,) and data.shape == (0,)


def test_sparse_diff_of_vector_with_itself_is_empty():
	x = _vec([1, 4, 9], [2.0, -1.0, 5.0])
	ind, data = spdist.sparse_diff(*x, *x)
	assert ind.shape == (0,) and data.shape == (0,)


def test_sparse_diff_hand_case():
	ind, data = spdist.sparse_diff(
		*_vec([0, 3], [1.0, 2.0]), *_vec([3, 4], [0.5, 2.0])
	)
	assert np.array_equal(ind, [0, 3, 4])
	assert np.allclose(data, [1.0, 1.5, -2.0])


def test_sparse_mul_matches_dense(real_csr):
	dense = real_csr.toarray().astype(np.float64)
	for i, j, (i1, d1), (i2, d2) in _pairs(real_csr):
		ind, data = spdist.sparse_mul(i1, d1, i2, d2)
		ind, data = np.array(list(ind), dtype=np.int64), np.array(list(data))
		out = np.zeros(dense.shape[1])
		out[ind] = data
		np.testing.assert_allclose(out, dense[i] * dense[j], rtol=1e-5, atol=1e-6)
		assert np.all(data != 0)


def test_sparse_mul_hand_case_and_empty():
	ind, data = spdist.sparse_mul(
		*_vec([0, 2, 5], [2.0, 3.0, 4.0]), *_vec([2, 3, 5], [-1.0, 7.0, 0.5])
	)
	assert list(ind) == [2, 5]
	assert np.allclose(list(data), [-3.0, 2.0])
	ind, data = spdist.sparse_mul(*_vec([], []), *_vec([1], [1.0]))
	assert len(ind) == 0 and len(data) == 0


def test_sparse_dot_product_matches_dense_dot(real_csr):
	_check_pairs(real_csr, spdist.sparse_dot_product, lambda x, y: np.dot(x, y))


def test_sparse_dot_product_empty_and_disjoint():
	x = _vec([1, 3], [2.0, 2.0])
	assert spdist.sparse_dot_product(*_vec([], []), *x) == 0.0
	assert spdist.sparse_dot_product(*x, *_vec([], [])) == 0.0
	assert spdist.sparse_dot_product(*x, *_vec([0, 2, 4], [1.0, 1.0, 1.0])) == 0.0
	assert spdist.sparse_dot_product(*x, *x) == 8.0


def test_sparse_dot_accumulate_adds_start_to_dot(real_csr):
	dense = real_csr.toarray().astype(np.float64)
	for start in (np.float32(0.0), np.float32(1.5), np.float32(-2.25)):
		for i, j, (i1, d1), (i2, d2) in _pairs(real_csr):
			got = spdist.sparse_dot_accumulate(start, i1, d1, i2, d2)
			np.testing.assert_allclose(
				got, float(start) + np.dot(dense[i], dense[j]), rtol=1e-4, atol=1e-5
			)


def test_sparse_dot_accumulate_equals_start_plus_sum_of_sparse_mul():
	x = _vec([0, 2, 5, 9], [0.1, 0.2, 0.3, 0.4])
	y = _vec([2, 5, 9, 10], [1.0, 2.0, 3.0, 4.0])
	_, prods = spdist.sparse_mul(*x, *y)
	acc = np.float32(0.5)
	for p in prods:
		acc = np.float32(acc + np.float32(p))
	np.testing.assert_allclose(
		spdist.sparse_dot_accumulate(np.float32(0.5), *x, *y), acc, rtol=1e-5)


def test_dense_union_hand_case():
	d1, d2 = spdist.dense_union(
		*_vec([0, 2, 5], [1.0, 2.0, 3.0]), *_vec([2, 3], [4.0, 5.0])
	)
	assert np.allclose(d1, [1.0, 2.0, 0.0, 3.0])
	assert np.allclose(d2, [0.0, 4.0, 5.0, 0.0])
	assert d1.dtype == np.float32 and d2.dtype == np.float32


def test_dense_union_matches_dense_restricted_to_union_support(positive_csr):
	dense = positive_csr.toarray()
	for i, j, (i1, d1), (i2, d2) in _pairs(positive_csr):
		u1, u2 = spdist.dense_union(i1, d1, i2, d2)
		mask = (dense[i] != 0) | (dense[j] != 0)
		assert np.allclose(u1, dense[i][mask])
		assert np.allclose(u2, dense[j][mask])


def test_dense_union_skips_explicit_zeros():
	d1, d2 = spdist.dense_union(
		*_vec([0, 1, 4], [0.0, 2.0, 1.0]), *_vec([1, 3], [3.0, 0.0])
	)
	assert np.allclose(d1, [2.0, 1.0])
	assert np.allclose(d2, [3.0, 0.0])


def test_dense_union_empty_vectors():
	e = _vec([], [])
	d1, d2 = spdist.dense_union(*e, *e)
	assert d1.shape == (0,) and d2.shape == (0,)
	d1, d2 = spdist.dense_union(*e, *_vec([2, 7], [1.0, 2.0]))
	assert np.allclose(d1, [0.0, 0.0])
	assert np.allclose(d2, [1.0, 2.0])


# ---------------------------------------------------------------------------
# real-valued metrics against dense references
# ---------------------------------------------------------------------------


def test_sparse_euclidean_and_squared(real_csr):
	_check_pairs(real_csr, spdist.sparse_euclidean, lambda x, y: np.linalg.norm(x - y))
	_check_pairs(real_csr, spdist.sparse_squared_euclidean,
		lambda x, y: np.sum((x - y) ** 2))


def test_sparse_euclidean_hand_case():
	d = spdist.sparse_euclidean(
		*_vec([0, 2], [3.0, 0.0]), *_vec([0, 3], [0.0, 4.0])
	)
	assert np.isclose(d, 5.0)
	assert np.isclose(
		spdist.sparse_squared_euclidean(*_vec([0, 2], [3.0, 0.0]),
			*_vec([0, 3], [0.0, 4.0])), 25.0)


def test_sparse_manhattan_and_chebyshev(real_csr):
	_check_pairs(real_csr, spdist.sparse_manhattan, lambda x, y: scd.cityblock(x, y))
	_check_pairs(real_csr, spdist.sparse_chebyshev, lambda x, y: scd.chebyshev(x, y))


def test_sparse_minkowski(real_csr):
	for p in (1.0, 2.0, 3.0, 0.5):
		_check_pairs(real_csr, spdist.sparse_minkowski,
			lambda x, y: np.sum(np.abs(x - y) ** p) ** (1.0 / p), p)


def test_sparse_minkowski_default_p_is_two(real_csr):
	for _, _, (i1, d1), (i2, d2) in _pairs(real_csr):
		assert np.isclose(spdist.sparse_minkowski(i1, d1, i2, d2),
			spdist.sparse_euclidean(i1, d1, i2, d2), rtol=1e-5, atol=1e-6)


def test_minkowski_family_empty_rows_and_identity():
	e = _vec([], [])
	x = _vec([1, 6], [-2.0, 3.0])
	for func in (spdist.sparse_euclidean, spdist.sparse_squared_euclidean,
		spdist.sparse_manhattan, spdist.sparse_chebyshev, spdist.sparse_minkowski):
		assert func(*e, *e) == 0.0
		assert func(*x, *x) == 0.0
	assert np.isclose(spdist.sparse_manhattan(*e, *x), 5.0)
	assert np.isclose(spdist.sparse_chebyshev(*x, *e), 3.0)
	assert np.isclose(spdist.sparse_euclidean(*e, *x), np.sqrt(13.0))


def test_sparse_canberra(real_csr):
	def ref(x, y):
		den = np.abs(x) + np.abs(y)
		keep = den != 0
		return np.sum(np.abs(x - y)[keep] / den[keep])

	_check_pairs(real_csr, spdist.sparse_canberra, ref)


def test_sparse_canberra_hand_case():
	d = spdist.sparse_canberra(
		*_vec([0, 1, 4], [1.0, 2.0, 3.0]), *_vec([1, 2], [6.0, 1.0])
	)
	assert np.isclose(d, 1.0 + 4.0 / 8.0 + 1.0 + 1.0)


def test_sparse_bray_curtis(real_csr):
	def ref(x, y):
		den = np.sum(np.abs(x + y))
		return 0.0 if den == 0 else np.sum(np.abs(x - y)) / den

	_check_pairs(real_csr, spdist.sparse_bray_curtis, ref)


def test_sparse_bray_curtis_positive_data_matches_scipy(positive_csr):
	_check_pairs(positive_csr, spdist.sparse_bray_curtis,
		lambda x, y: scd.braycurtis(x, y) if np.any(x + y) else 0.0)


def test_sparse_hamming_counts_differing_positions(real_csr):
	n = real_csr.shape[1]
	_check_pairs(real_csr, spdist.sparse_hamming, lambda x, y: np.mean(x != y), n)


def test_sparse_hamming_explicit_zero_is_not_a_difference():
	d = spdist.sparse_hamming(*_vec([1, 3], [0.0, 2.0]), *_vec([3], [2.0]), 10)
	assert d == 0.0


def test_sparse_cosine(real_csr):
	def ref(x, y):
		nx, ny = np.linalg.norm(x), np.linalg.norm(y)
		return 1.0 - np.dot(x, y) / (nx * ny)

	_check_pairs(real_csr, spdist.sparse_cosine, ref,
		skip=lambda x, y: not x.any() or not y.any())


def test_sparse_cosine_zero_vector_conventions():
	"""Zero-vector and nonpositive-dot conventions; pins the current output."""
	e = _vec([], [])
	x = _vec([2], [1.5])
	assert spdist.sparse_cosine(*e, *e) == 0.0
	assert spdist.sparse_cosine(*e, *x) == 1.0
	assert spdist.sparse_cosine(*x, *e) == 1.0
	assert abs(spdist.sparse_cosine(*x, *x)) < 1e-6
	assert np.isclose(spdist.sparse_cosine(*x, *_vec([2], [-1.5])), 2.0)
	assert np.isclose(spdist.sparse_cosine(*x, *_vec([3], [1.0])), 1.0)


def test_sparse_cosine_matches_scipy(real_csr):
	dense = real_csr.toarray().astype(np.float64)
	for i, j, (i1, d1), (i2, d2) in _pairs(real_csr):
		if dense[i].any() and dense[j].any():
			np.testing.assert_allclose(spdist.sparse_cosine(i1, d1, i2, d2),
				scd.cosine(dense[i], dense[j]), rtol=1e-4, atol=1e-5)


def test_sparse_correlation_matches_scipy(real_csr):
	n = real_csr.shape[1]
	_check_pairs(real_csr, spdist.sparse_correlation,
		lambda x, y: scd.correlation(x, y), n, rtol=1e-3, atol=1e-4,
		skip=lambda x, y: not x.any() or not y.any())


def test_sparse_correlation_perfectly_correlated_and_anti():
	n = 12
	x = _vec([1, 4, 7], [1.0, 2.0, 3.0])
	assert abs(spdist.sparse_correlation(*x, *_vec([1, 4, 7], [2.0, 4.0, 6.0]), n)) < 1e-5
	assert np.isclose(
		spdist.sparse_correlation(*x, *_vec([1, 4, 7], [-1.0, -2.0, -3.0]), n), 2.0,
		atol=1e-5)


def test_sparse_correlation_empty_row_conventions():
	"""Pins the current output: undefined correlation maps to 0 or 1."""
	e = _vec([], [])
	x = _vec([1, 4], [1.0, 2.0])
	assert spdist.sparse_correlation(*e, *e, 8) == 0.0
	assert spdist.sparse_correlation(*e, *x, 8) == 1.0
	assert spdist.sparse_correlation(*x, *e, 8) == 1.0


def test_sparse_dot_is_one_minus_dot_product(real_csr):
	_check_pairs(real_csr, spdist.sparse_dot, lambda x, y: 1.0 - np.dot(x, y))


def test_sparse_alternative_dot(real_csr):
	def ref(x, y):
		d = np.dot(x, y)
		return FLOAT32_MAX if d <= 0 else -np.log2(d)

	_check_pairs(real_csr, spdist.sparse_alternative_dot, ref, rtol=1e-3, atol=1e-4)


def test_sparse_alternative_dot_nonpositive_is_float32_max():
	"""Zero-vector and nonpositive-dot conventions; pins the current output."""
	x = _vec([1], [1.0])
	assert spdist.sparse_alternative_dot(*x, *_vec([2], [1.0])) == FLOAT32_MAX
	assert spdist.sparse_alternative_dot(*x, *_vec([1], [-1.0])) == FLOAT32_MAX
	assert spdist.sparse_alternative_dot(*x, *_vec([], [])) == FLOAT32_MAX


def test_sparse_hellinger(positive_csr):
	def ref(x, y):
		sx, sy = x.sum(), y.sum()
		if sx == 0 and sy == 0:
			return 0.0
		if sx == 0 or sy == 0:
			return 1.0
		bc = np.sum(np.sqrt(x * y)) / np.sqrt(sx * sy)
		return np.sqrt(max(1.0 - bc, 0.0))

	_check_pairs(positive_csr, spdist.sparse_hellinger, ref, rtol=1e-5, atol=1e-3)


def test_sparse_hellinger_hand_cases():
	"""Zero-vector and nonpositive-dot conventions; pins the current output."""
	x = _vec([0, 1], [1.0, 1.0])
	assert np.isclose(spdist.sparse_hellinger(*x, *_vec([2, 3], [1.0, 1.0])), 1.0)
	assert abs(spdist.sparse_hellinger(*x, *x)) < 1e-3
	assert np.isclose(
		spdist.sparse_hellinger(*_vec([0, 1], [1.0, 0.0]), *_vec([0, 1], [0.5, 0.5])),
		np.sqrt(1.0 - np.sqrt(0.5)), atol=1e-4)
	e = _vec([], [])
	assert spdist.sparse_hellinger(*e, *e) == 0.0
	assert spdist.sparse_hellinger(*e, *x) == 1.0


def test_sparse_alternative_hellinger_and_correction(positive_csr):
	dense = positive_csr.toarray().astype(np.float64)
	vec_correct = spdist.sparse_correct_alternative_hellinger
	for i, j, (i1, d1), (i2, d2) in _pairs(positive_csr):
		x, y = dense[i], dense[j]
		alt = spdist.sparse_alternative_hellinger(i1, d1, i2, d2)
		if not x.any() and not y.any():
			assert alt == 0.0
		elif not x.any() or not y.any():
			assert alt == FLOAT32_MAX
			assert np.isclose(vec_correct(np.float32(alt)), 1.0)
		else:
			bc = np.sum(np.sqrt(x * y)) / np.sqrt(x.sum() * y.sum())
			if bc <= 0:
				assert alt == FLOAT32_MAX
			else:
				np.testing.assert_allclose(alt, -np.log2(bc), rtol=1e-3, atol=1e-4)
				np.testing.assert_allclose(
					vec_correct(np.float32(alt)), np.sqrt(max(1.0 - bc, 0.0)),
					rtol=1e-3, atol=2e-3)


def test_sparse_correct_alternative_hellinger_values():
	d = np.array([0.0, 1e-9, -1.0, 1.0, 3.0, FLOAT32_MAX], dtype=np.float32)
	out = spdist.sparse_correct_alternative_hellinger(d)
	expected = np.array([0.0, 0.0, 0.0, np.sqrt(0.5), np.sqrt(1 - 2.0**-3), 1.0])
	assert np.allclose(out, expected, atol=1e-6)


def test_sparse_correct_alternative_cosine_values():
	d = np.array([0.0, 1e-9, -1.0, 1.0, 3.0, FLOAT32_MAX], dtype=np.float32)
	out = spdist.sparse_correct_alternative_cosine(d)
	assert np.allclose(out, [0.0, 0.0, 0.0, 0.5, 0.875, 1.0], atol=1e-6)
	assert np.isclose(spdist.sparse_correct_alternative_cosine(2.0), 0.75)


def test_sparse_alternative_cosine_and_correction(positive_csr):
	dense = positive_csr.toarray().astype(np.float64)
	for i, j, (i1, d1), (i2, d2) in _pairs(positive_csr):
		x, y = dense[i], dense[j]
		alt = spdist.sparse_alternative_cosine(i1, d1, i2, d2)
		if not x.any() and not y.any():
			assert alt == 0.0
		elif not x.any() or not y.any():
			assert alt == FLOAT32_MAX
		else:
			cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
			if cos_sim <= 0:
				assert alt == FLOAT32_MAX
			else:
				np.testing.assert_allclose(alt, -np.log2(cos_sim), rtol=1e-3, atol=1e-4)
				np.testing.assert_allclose(
					spdist.sparse_correct_alternative_cosine(np.float32(alt)),
					1.0 - cos_sim, rtol=1e-3, atol=2e-4)


def test_sparse_alternative_cosine_nonpositive_dot_is_float32_max():
	"""Zero-vector and nonpositive-dot conventions; pins the current output."""
	x = _vec([1], [1.0])
	assert spdist.sparse_alternative_cosine(*x, *_vec([2], [3.0])) == FLOAT32_MAX
	assert spdist.sparse_alternative_cosine(*x, *_vec([1], [-3.0])) == FLOAT32_MAX


# ---------------------------------------------------------------------------
# binary metrics
# ---------------------------------------------------------------------------


def _ref_jaccard(x, y):
	ntt, ntf, nft, _ = _support_counts(x, y)
	return 0.0 if ntt + ntf + nft == 0 else (ntf + nft) / (ntt + ntf + nft)


def _ref_dice(x, y):
	ntt, ntf, nft, _ = _support_counts(x, y)
	r = ntf + nft
	return 0.0 if r == 0 else r / (2.0 * ntt + r)


def _ref_matching(x, y):
	ntt, ntf, nft, nff = _support_counts(x, y)
	return (ntf + nft) / (ntt + ntf + nft + nff)


def _ref_kulsinski(x, y):
	ntt, ntf, nft, nff = _support_counts(x, y)
	n = ntt + ntf + nft + nff
	r = ntf + nft
	return 0.0 if r == 0 else (r - ntt + n) / (r + n)


def _ref_rogerstanimoto(x, y):
	ntt, ntf, nft, nff = _support_counts(x, y)
	n = ntt + ntf + nft + nff
	r = ntf + nft
	return 2.0 * r / (n + r)


def _ref_russellrao(x, y):
	ntt, ntf, nft, nff = _support_counts(x, y)
	n = ntt + ntf + nft + nff
	return 0.0 if ntf + nft == 0 else (n - ntt) / n


def _ref_sokalsneath(x, y):
	ntt, ntf, nft, _ = _support_counts(x, y)
	r = ntf + nft
	return 0.0 if r == 0 else r / (0.5 * ntt + r)


def test_sparse_jaccard(binary_csr):
	_check_pairs(binary_csr, spdist.sparse_jaccard, _ref_jaccard)
	_check_pairs(binary_csr, spdist.sparse_jaccard,
		lambda x, y: scd.jaccard(x != 0, y != 0), skip=lambda x, y: not (x.any() or y.any()))


def test_sparse_jaccard_ignores_values_and_hand_case():
	d = spdist.sparse_jaccard(*_vec([0, 1, 2], [5.0, -1.0, 0.1]), *_vec([1, 2, 3], [9.0, 9.0, 9.0]))
	assert np.isclose(d, 2.0 / 4.0)
	e = _vec([], [])
	assert spdist.sparse_jaccard(*e, *e) == 0.0
	assert spdist.sparse_jaccard(*e, *_vec([1])) == 1.0


def test_sparse_alternative_jaccard_and_correction(binary_csr):
	dense = binary_csr.toarray()
	for i, j, (i1, d1), (i2, d2) in _pairs(binary_csr):
		x, y = dense[i] != 0, dense[j] != 0
		alt = spdist.sparse_alternative_jaccard(i1, d1, i2, d2)
		ntt, ntf, nft, _ = _support_counts(x, y)
		union = ntt + ntf + nft
		if union == 0:
			assert alt == 0.0
		elif ntt == 0:
			assert alt == FLOAT32_MAX
		else:
			np.testing.assert_allclose(alt, -np.log2(ntt / union), rtol=1e-5, atol=1e-6)
		corrected = spdist.correct_alternative_jaccard(np.float32(alt))
		np.testing.assert_allclose(corrected, _ref_jaccard(x, y), rtol=1e-4, atol=1e-5)


def test_correct_alternative_jaccard_values():
	out = spdist.correct_alternative_jaccard(np.array([0.0, 1.0, 2.0, 10.0]))
	assert np.allclose(out, [0.0, 0.5, 0.75, 1.0 - 2.0**-10])
	assert np.isclose(spdist.correct_alternative_jaccard(1.0), 0.5)


@pytest.mark.parametrize(
	"name,func,ref,needs_n",
	[
		("matching", spdist.sparse_matching, _ref_matching, True),
		("dice", spdist.sparse_dice, _ref_dice, False),
		("kulsinski", spdist.sparse_kulsinski, _ref_kulsinski, True),
		("rogerstanimoto", spdist.sparse_rogers_tanimoto, _ref_rogerstanimoto, True),
		("russellrao", spdist.sparse_russellrao, _ref_russellrao, True),
		("sokalmichener", spdist.sparse_sokal_michener, _ref_rogerstanimoto, True),
		("sokalsneath", spdist.sparse_sokal_sneath, _ref_sokalsneath, False),
	],
)
def test_sparse_binary_metrics_match_count_formulas(binary_csr, name, func, ref, needs_n):
	extra = (binary_csr.shape[1],) if needs_n else ()
	_check_pairs(binary_csr, func, ref, *extra)


def test_sparse_binary_metrics_match_scipy(binary_csr):
	n = binary_csr.shape[1]
	cases = (
		(spdist.sparse_rogers_tanimoto, scd.rogerstanimoto, True),
		(spdist.sparse_hamming, scd.hamming, True),
	)
	for func, sref, needs_n in cases:
		_check_pairs(binary_csr, func, lambda x, y: sref(x != 0, y != 0), n)
	_check_pairs(binary_csr, spdist.sparse_dice, lambda x, y: scd.dice(x != 0, y != 0),
		skip=lambda x, y: not (x.any() or y.any()))
	_check_pairs(binary_csr, spdist.sparse_sokal_sneath,
		lambda x, y: scd.sokalsneath(x != 0, y != 0),
		skip=lambda x, y: not (x.any() or y.any()))


def test_sparse_russellrao_and_kulsinski_match_scipy_when_supports_differ(binary_csr):
	"""Against scipy where it agrees: pairs with ntf + nft > 0 (scipy and the
	sparse code differ on identical supports)."""
	n = binary_csr.shape[1]
	differ = lambda x, y: not np.array_equal(x != 0, y != 0)
	_check_pairs(binary_csr, spdist.sparse_russellrao,
		lambda x, y: scd.russellrao(x != 0, y != 0), n,
		skip=lambda x, y: not differ(x, y))
	if hasattr(scd, "kulsinski"):
		_check_pairs(binary_csr, spdist.sparse_kulsinski,
			lambda x, y: scd.kulsinski(x != 0, y != 0), n,
			skip=lambda x, y: not differ(x, y))
	_check_pairs(binary_csr, spdist.sparse_russellrao, _ref_russellrao, n)
	_check_pairs(binary_csr, spdist.sparse_kulsinski, _ref_kulsinski, n)


def test_sparse_binary_metrics_hand_case():
	a = _vec([0, 1, 2])
	b = _vec([1, 2, 3, 4])
	n = 10
	ntt, ntf, nft, nff = 2, 1, 2, 5
	assert np.isclose(spdist.sparse_matching(*a, *b, n), 3 / 10)
	assert np.isclose(spdist.sparse_dice(*a, *b), 3 / 7)
	assert np.isclose(spdist.sparse_kulsinski(*a, *b, n), (3 - 2 + 10) / (3 + 10))
	assert np.isclose(spdist.sparse_rogers_tanimoto(*a, *b, n), 6 / 13)
	assert np.isclose(spdist.sparse_sokal_michener(*a, *b, n), 6 / 13)
	assert np.isclose(spdist.sparse_russellrao(*a, *b, n), 8 / 10)
	assert np.isclose(spdist.sparse_sokal_sneath(*a, *b), 3 / 4.0)
	assert ntt + ntf + nft + nff == n


def test_sparse_binary_metrics_identical_sets_are_zero():
	"""Identical supports give 0 for every binary metric; pins the current output
	(scipy.spatial.distance.russellrao gives a nonzero value for identical vectors)."""
	a = _vec([1, 5, 8])
	n = 12
	assert spdist.sparse_matching(*a, *a, n) == 0.0
	assert spdist.sparse_dice(*a, *a) == 0.0
	assert spdist.sparse_kulsinski(*a, *a, n) == 0.0
	assert spdist.sparse_rogers_tanimoto(*a, *a, n) == 0.0
	assert spdist.sparse_sokal_michener(*a, *a, n) == 0.0
	assert spdist.sparse_russellrao(*a, *a, n) == 0.0
	assert spdist.sparse_sokal_sneath(*a, *a) == 0.0
	e = _vec([], [])
	assert spdist.sparse_dice(*e, *e) == 0.0
	assert spdist.sparse_sokal_sneath(*e, *e) == 0.0
	assert spdist.sparse_russellrao(*e, *e, n) == 0.0


def test_sparse_russellrao_subset_support_is_not_zero():
	a = _vec([1, 5])
	b = _vec([1, 5, 8])
	assert np.isclose(spdist.sparse_russellrao(*a, *b, 10), (10 - 2) / 10)
	assert np.isclose(spdist.sparse_russellrao(*b, *a, 10), (10 - 2) / 10)


# ---------------------------------------------------------------------------
# distribution metrics
# ---------------------------------------------------------------------------


def test_sparse_kantorovich_default_ground_metric_is_total_variation(positive_csr):
	def ref(x, y):
		if not x.any() or not y.any():
			return None
		return 0.5 * np.sum(np.abs(x / x.sum() - y / y.sum()))

	dense = positive_csr.toarray().astype(np.float64)
	n_checked = 0
	for i, j, (i1, d1), (i2, d2) in _pairs(positive_csr):
		expected = ref(dense[i], dense[j])
		if expected is None:
			continue
		got = spdist.sparse_kantorovich(i1, d1, i2, d2)
		np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)
		n_checked += 1
	assert n_checked > 0


def test_dummy_ground_metric():
	assert spdist.dummy_ground_metric(3, 3) == 0.0
	assert spdist.dummy_ground_metric(3, 4) == 1.0


def test_create_ground_metric_applies_metric_to_ground_vectors():
	@numba.njit()
	def l1(a, b):
		return np.sum(np.abs(a - b))

	vectors = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 1.0]], dtype=np.float32)
	gm = spdist.create_ground_metric(vectors, l1)
	assert np.isclose(gm(0, 1), 3.0)
	assert np.isclose(gm(1, 2), 4.0 + 1.0)
	assert gm(2, 2) == 0.0


def test_sparse_kantorovich_with_1d_ground_vectors_matches_scipy_wasserstein():
	positions = np.arange(20, dtype=np.float32)[:, None] * 0.5
	gm = spdist.create_ground_metric(positions, numba.njit()(lambda a, b: np.abs(a[0] - b[0])))
	rng = np.random.RandomState(5)
	for _ in range(5):
		i1 = np.sort(rng.choice(20, 5, replace=False)).astype(np.int32)
		i2 = np.sort(rng.choice(20, 6, replace=False)).astype(np.int32)
		d1 = rng.uniform(0.1, 1.0, 5).astype(np.float32)
		d2 = rng.uniform(0.1, 1.0, 6).astype(np.float32)
		got = spdist.sparse_kantorovich(i1, d1, i2, d2, gm)
		expected = scipy.stats.wasserstein_distance(
			positions[i1, 0], positions[i2, 0], d1.astype(np.float64), d2.astype(np.float64)
		)
		np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_sparse_kantorovich_equals_dense_kantorovich_on_stored_entries():
	i1, d1 = _vec([1, 3, 6], [0.2, 0.5, 0.3])
	i2, d2 = _vec([3, 4], [0.6, 0.4])
	cost = np.array([[abs(a - b) for b in i2] for a in i1], dtype=np.float64)
	expected = dense_kantorovich(d1, d2, cost)
	@numba.njit()
	def gm(a, b):
		return np.float32(abs(a - b))

	assert np.isclose(spdist.sparse_kantorovich(i1, d1, i2, d2, gm), expected, rtol=1e-5)


def _wasserstein_cdf_ref(x, y, p):
	fx = np.cumsum(x / x.sum())
	fy = np.cumsum(y / y.sum())
	return np.sum(np.abs(fx[:-1] - fy[:-1]) ** p) ** (1.0 / p)


def test_sparse_wasserstein_1d_matches_scipy_for_p1(positive_csr):
	dense = positive_csr.toarray().astype(np.float64)
	grid = np.arange(dense.shape[1], dtype=np.float64)
	n_checked = 0
	for i, j, (i1, d1), (i2, d2) in _pairs(positive_csr):
		if not dense[i].any() or not dense[j].any():
			continue
		expected = scipy.stats.wasserstein_distance(grid, grid, dense[i], dense[j])
		got = spdist.sparse_wasserstein_1d(i1, d1, i2, d2)
		np.testing.assert_allclose(got, expected, rtol=2e-4, atol=1e-5)
		n_checked += 1
	assert n_checked > 0


@pytest.mark.parametrize("p", [1, 2, 3])
def test_sparse_wasserstein_1d_matches_cdf_reference(positive_csr, p):
	dense = positive_csr.toarray().astype(np.float64)
	for i, j, (i1, d1), (i2, d2) in _pairs(positive_csr):
		if not dense[i].any() or not dense[j].any():
			continue
		got = spdist.sparse_wasserstein_1d(i1, d1, i2, d2, p)
		np.testing.assert_allclose(got, _wasserstein_cdf_ref(dense[i], dense[j], p),
			rtol=2e-4, atol=1e-5)


def test_sparse_wasserstein_1d_hand_case():
	d = spdist.sparse_wasserstein_1d(*_vec([0], [3.0]), *_vec([4], [0.5]))
	assert np.isclose(d, 4.0)
	d = spdist.sparse_wasserstein_1d(*_vec([2, 5], [1.0, 1.0]), *_vec([2, 5], [2.0, 2.0]))
	assert np.isclose(d, 0.0)


def _dist_eps(x, y, kind):
	dim = x.shape[0]
	px = (x + FLOAT32_EPS) / (x.sum() + FLOAT32_EPS * dim)
	py = (y + FLOAT32_EPS) / (y.sum() + FLOAT32_EPS * dim)
	if kind == "js":
		m = 0.5 * (px + py)
		return np.sum(0.5 * (px * np.log(px / m) + py * np.log(py / m)))
	return np.sum(px * np.log(px / py) + py * np.log(py / px))


def test_sparse_jensen_shannon_and_symmetric_kl_use_union_support(positive_csr):
	dense = positive_csr.toarray().astype(np.float64)
	for i, j, (i1, d1), (i2, d2) in _pairs(positive_csr):
		mask = (dense[i] != 0) | (dense[j] != 0)
		if mask.sum() == 0:
			continue
		x, y = dense[i][mask], dense[j][mask]
		np.testing.assert_allclose(
			spdist.sparse_jensen_shannon_divergence(i1, d1, i2, d2),
			_dist_eps(x, y, "js"), rtol=1e-3, atol=1e-5)
		np.testing.assert_allclose(
			spdist.sparse_symmetric_kl_divergence(i1, d1, i2, d2),
			_dist_eps(x, y, "kl"), rtol=1e-3, atol=1e-4)


def test_sparse_jensen_shannon_properties():
	a = _vec([0, 1, 2], [1.0, 2.0, 3.0])
	b = _vec([1, 2, 5], [3.0, 1.0, 2.0])
	js_ab = spdist.sparse_jensen_shannon_divergence(*a, *b)
	assert js_ab > 0.0
	assert js_ab <= np.log(2.0) + 1e-6
	assert np.isclose(js_ab, spdist.sparse_jensen_shannon_divergence(*b, *a), rtol=1e-5)
	assert abs(spdist.sparse_jensen_shannon_divergence(*a, *a)) < 1e-6
	assert abs(spdist.sparse_symmetric_kl_divergence(*a, *a)) < 1e-6
	assert np.isclose(spdist.sparse_symmetric_kl_divergence(*a, *b),
		spdist.sparse_symmetric_kl_divergence(*b, *a), rtol=1e-5)


# ---------------------------------------------------------------------------
# registries
# ---------------------------------------------------------------------------


def test_named_distance_aliases_share_a_function():
	nd = spdist.sparse_named_distances
	assert nd["l2"] is nd["euclidean"]
	assert nd["l1"] is nd["manhattan"] is nd["taxicab"]
	for alias in ("linf", "linfty", "linfinity"):
		assert nd[alias] is nd["chebyshev"]
	assert nd["wasserstein"] is nd["kantorovich"]
	for alias in ("wasserstein-1d", "kantorovich-1d"):
		assert nd[alias] is nd["wasserstein_1d"]
	assert nd["jensen-shannon"] is nd["jensen_shannon"]
	assert nd["symmetric-kl"] is nd["symmetric_kl"] is nd["symmetric_kullback_liebler"]


def test_sparse_need_n_features_lists_exactly_the_metrics_taking_n_features():
	"""The registry holds the seven metrics that take n_features (hard-coded
	list); each of them needs the fifth argument, and the others run with four."""
	expected = {"hamming", "matching", "kulsinski", "rogerstanimoto", "russellrao",
		"sokalmichener", "correlation"}
	assert set(spdist.sparse_need_n_features) == expected
	x = _vec([0, 1, 3], [1.0, 2.0, 0.5])
	y = _vec([1, 2, 3], [0.5, 1.0, 2.0])
	for name, func in spdist.sparse_named_distances.items():
		if name in expected:
			assert np.isfinite(func(*x, *y, 6)), name
			with pytest.raises(TypeError):
				func(*x, *y)
		else:
			assert np.isfinite(func(*x, *y)), name


@pytest.mark.parametrize("name", sorted(spdist.sparse_named_distances))
def test_named_distances_are_symmetric_and_self_distance_zero(name, positive_csr):
	func = spdist.sparse_named_distances[name]
	extra = (positive_csr.shape[1],) if name in spdist.sparse_need_n_features else ()
	for i, j in itertools.combinations(range(0, positive_csr.shape[0], 2), 2):
		r1, r2 = _row(positive_csr, i), _row(positive_csr, j)
		if r1[0].shape[0] == 0 or r2[0].shape[0] == 0:
			continue
		forward = func(*r1, *r2, *extra)
		backward = func(*r2, *r1, *extra)
		assert np.isfinite(forward)
		assert np.isclose(forward, backward, rtol=1e-4, atol=1e-5)
		assert abs(func(*r1, *r1, *extra)) < 1e-3


@pytest.mark.parametrize("name", sorted(spdist.sparse_fast_distance_alternatives))
def test_fast_alternatives_corrected_equal_true_distance(name, positive_csr):
	entry = spdist.sparse_fast_distance_alternatives[name]
	alt, correction = entry["dist"], entry["correction"]
	dense = positive_csr.toarray().astype(np.float64)
	n_checked = 0
	for i, j, (i1, d1), (i2, d2) in _pairs(positive_csr):
		if not dense[i].any() or not dense[j].any() or i == j:
			continue
		if name == "dot":
			x = dense[i] / np.linalg.norm(dense[i]) * 0.5
			y = dense[j] / np.linalg.norm(dense[j]) * 0.5
			xr = sp.csr_matrix(x.astype(np.float32))
			yr = sp.csr_matrix(y.astype(np.float32))
			xr.sort_indices()
			yr.sort_indices()
			a = alt(xr.indices.astype(np.int32), xr.data, yr.indices.astype(np.int32), yr.data)
			expected = 1.0 - float(np.dot(x.astype(np.float32), y.astype(np.float32)))
			tol = 1e-4
		else:
			base = {"l2": "euclidean"}.get(name, name)
			a = alt(i1, d1, i2, d2)
			expected = spdist.sparse_named_distances[base](i1, d1, i2, d2)
			tol = 1e-5
		got = float(correction(np.float32(a)))
		np.testing.assert_allclose(got, expected, rtol=tol, atol=tol)
		n_checked += 1
	assert n_checked > 0


def test_fast_alternative_orderings_match_true_distance_orderings(positive_csr):
	q = _row(positive_csr, 0)
	others = [_row(positive_csr, i) for i in range(1, positive_csr.shape[0])
		if _row(positive_csr, i)[0].shape[0] > 0]
	for name in ("euclidean", "cosine", "hellinger", "jaccard"):
		entry = spdist.sparse_fast_distance_alternatives[name]
		true = [spdist.sparse_named_distances[name](*q, *o) for o in others]
		alt = [entry["dist"](*q, *o) for o in others]
		assert np.array_equal(np.argsort(true, kind="stable"),
			np.argsort(alt, kind="stable")), name


# ---------------------------------------------------------------------------
# finish_query_* helpers
# ---------------------------------------------------------------------------


def _descending_heap(rng, n, k, hi):
	dist = -np.sort(-rng.uniform(0.0, hi, size=(n, k)).astype(np.float32), axis=1)
	idx = np.stack([rng.permutation(k) for _ in range(n)]).astype(np.int32)
	return idx, dist


def test_finish_query_alternative_cosine_sorts_maps_and_corrects():
	rng = np.random.RandomState(0)
	idx, dist = _descending_heap(rng, 6, 5, 4.0)
	dist[0, :3] = [FLOAT32_MAX, 1e-9, -0.5]
	dist[0] = -np.sort(-dist[0])
	order = rng.permutation(5).astype(np.int64) + 100
	idx0, dist0 = idx.copy(), dist.copy()
	out_i, out_d = spdist.finish_query_sparse_alternative_cosine(idx, dist, order)
	assert out_i.dtype == np.int64 and out_d.dtype == np.float64
	for r in range(6):
		asc = np.argsort(dist0[r], kind="stable")
		assert np.array_equal(out_i[r], order[idx0[r][asc]])
		d = dist0[r][asc].astype(np.float64)
		expected = np.where((np.abs(d) <= 1e-7) | (d < 0), 0.0, 1.0 - 2.0 ** (-d))
		np.testing.assert_allclose(out_d[r], expected, atol=1e-6)
		assert np.all(np.diff(out_d[r]) >= -1e-12)


def test_finish_query_alternative_hellinger_sorts_maps_and_corrects():
	rng = np.random.RandomState(1)
	idx, dist = _descending_heap(rng, 6, 5, 4.0)
	dist[1, :2] = [FLOAT32_MAX, -1.0]
	dist[1] = -np.sort(-dist[1])
	order = rng.permutation(5).astype(np.int32)
	idx0, dist0 = idx.copy(), dist.copy()
	out_i, out_d = spdist.finish_query_sparse_alternative_hellinger(idx, dist, order)
	assert out_i.dtype == np.int32 and out_d.dtype == np.float64
	for r in range(6):
		asc = np.argsort(dist0[r], kind="stable")
		assert np.array_equal(out_i[r], order[idx0[r][asc]])
		d = dist0[r][asc].astype(np.float64)
		with np.errstate(invalid="ignore"):
			expected = np.where((np.abs(d) <= 1e-7) | (d < 0), 0.0, np.sqrt(1.0 - 2.0 ** (-d)))
		np.testing.assert_allclose(out_d[r], expected, atol=1e-6)


def test_finish_query_matches_vectorized_corrections():
	rng = np.random.RandomState(2)
	idx, dist = _descending_heap(rng, 4, 6, 5.0)
	order = np.arange(6, dtype=np.int32)
	d_sorted = np.sort(dist, axis=1)
	_, out_cos = spdist.finish_query_sparse_alternative_cosine(idx.copy(), dist.copy(), order)
	_, out_hel = spdist.finish_query_sparse_alternative_hellinger(idx.copy(), dist.copy(), order)
	np.testing.assert_allclose(
		out_cos, spdist.sparse_correct_alternative_cosine(d_sorted), atol=1e-6)
	np.testing.assert_allclose(
		out_hel, spdist.sparse_correct_alternative_hellinger(d_sorted), atol=1e-6)


# ---------------------------------------------------------------------------
# diversify
# ---------------------------------------------------------------------------


def _ref_diversify(indices, distances, dense, metric):
	"""Probability-one pruning in plain Python. The loop restates the one in
	diversify, so it pins the current pruning rule."""
	indices, distances = indices.copy(), distances.copy()
	n, k = indices.shape
	for i in range(n):
		n_ret = 1
		for j in range(1, k):
			u = indices[i, j]
			if u < 0:
				break
			d_iu = distances[i, j]
			keep = True
			for m in range(n_ret):
				if distances[i, m] > FLOAT32_EPS:
					c = indices[i, m]
					if metric(dense[u], dense[c]) < d_iu:
						keep = False
						break
			if keep:
				indices[i, n_ret] = u
				distances[i, n_ret] = d_iu
				n_ret += 1
		indices[i, n_ret:] = -1
		distances[i, n_ret:] = np.inf
	return indices, distances


def _knn_arrays(m, k):
	dense = m.toarray().astype(np.float64)
	d = scd.cdist(dense, dense)
	np.fill_diagonal(d, np.inf)
	idx = np.argsort(d, axis=1, kind="stable")[:, :k].astype(np.int32)
	dist = np.take_along_axis(d, idx, axis=1).astype(np.float32)
	return dense, idx, dist


def test_diversify_matches_reference_pruning():
	m = _random_csr(40, 12, 0.5, seed=7)
	dense, idx, dist = _knn_arrays(m, 8)
	ref_i, ref_d = _ref_diversify(
		idx, dist, dense, lambda a, b: float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32))))
	got_i, got_d = spdist.diversify(
		idx.copy(), dist.copy(), m.indices, m.indptr, m.data,
		spdist.sparse_euclidean, np.array([1, 2, 3], dtype=np.int64))
	assert np.array_equal(got_i, ref_i)
	assert np.array_equal(got_d, ref_d)
	assert np.any(got_i < 0)


def test_diversify_zero_probability_keeps_everything():
	m = _random_csr(30, 10, 0.5, seed=8)
	_, idx, dist = _knn_arrays(m, 6)
	got_i, got_d = spdist.diversify(
		idx.copy(), dist.copy(), m.indices, m.indptr, m.data,
		spdist.sparse_euclidean, np.array([1, 2, 3], dtype=np.int64),
		prune_probability=0.0)
	assert np.array_equal(got_i, idx)
	assert np.array_equal(got_d, dist)


def test_diversify_partial_probability_is_between_none_and_full():
	m = _random_csr(60, 10, 0.5, seed=9)
	_, idx, dist = _knn_arrays(m, 8)
	rng_state = np.array([5, 6, 7], dtype=np.int64)
	kept = {}
	for prob in (0.0, 0.5, 1.0):
		out_i, _ = spdist.diversify(idx.copy(), dist.copy(), m.indices, m.indptr,
			m.data, spdist.sparse_euclidean, rng_state, prune_probability=prob)
		kept[prob] = int(np.sum(out_i >= 0))
	assert kept[0.0] == idx.size
	assert kept[1.0] < kept[0.5] < kept[0.0]
	again, _ = spdist.diversify(idx.copy(), dist.copy(), m.indices, m.indptr,
		m.data, spdist.sparse_euclidean, rng_state, prune_probability=0.5)
	assert int(np.sum(again >= 0)) == kept[0.5]


def test_diversify_zero_distance_guard_with_duplicate_rows():
	"""Duplicate points sit at distance 0 from each other; the pruning loop must
	match the reference (which skips kept neighbours at distance <= FLOAT32_EPS)."""
	base = _random_csr(25, 10, 0.5, seed=12)
	dense0 = base.toarray()
	dense0[10:15] = dense0[0:5]
	dense0[20:23] = dense0[0:3]
	m = sp.csr_matrix(dense0)
	m.sort_indices()
	m.indices = m.indices.astype(np.int32)
	m.indptr = m.indptr.astype(np.int32)
	dense, idx, dist = _knn_arrays(m, 8)
	assert (dist == 0).any()
	ref_i, ref_d = _ref_diversify(
		idx, dist, dense, lambda a, b: float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32))))
	got_i, got_d = spdist.diversify(
		idx.copy(), dist.copy(), m.indices, m.indptr, m.data,
		spdist.sparse_euclidean, np.array([1, 2, 3], dtype=np.int64))
	assert np.array_equal(got_i, ref_i)
	assert np.array_equal(got_d, ref_d)


def test_diversify_distance_guard_on_hand_built_rows():
	"""One-dimensional points 0.0, 0.3, 1.0, 0.9. Row 0 keeps 0.3 away (a kept
	neighbour at nonzero distance prunes the point at 1.0, which is nearer to it
	than to the row's own point). Row 1 lists neighbour 3 at distance 0.0, so
	that kept neighbour is skipped and neighbour 2 survives although it is nearer
	to 3 than to the row's own point."""
	x = np.array([[0.0], [0.3], [1.0], [0.9]], dtype=np.float32)
	m = sp.csr_matrix(x)
	m.sort_indices()
	idx = np.array([[1, 2], [3, 2], [1, -1], [1, -1]], dtype=np.int32)
	dist = np.array(
		[[0.3, 1.0], [0.0, 0.7], [0.7, np.inf], [0.6, np.inf]], dtype=np.float32)
	got_i, got_d = spdist.diversify(
		idx.copy(), dist.copy(), m.indices.astype(np.int32), m.indptr.astype(np.int32),
		m.data, spdist.sparse_euclidean, np.array([1, 2, 3], dtype=np.int64))
	np.testing.assert_array_equal(got_i, [[1, -1], [3, 2], [1, -1], [1, -1]])
	np.testing.assert_array_equal(got_d[0], [np.float32(0.3), np.inf])
	np.testing.assert_array_equal(got_d[1], dist[1])


def test_diversify_first_neighbour_always_kept_and_padding_respected():
	m = _random_csr(20, 8, 0.6, seed=10)
	_, idx, dist = _knn_arrays(m, 5)
	idx[:, 3:] = -1
	dist[:, 3:] = np.inf
	out_i, out_d = spdist.diversify(
		idx.copy(), dist.copy(), m.indices, m.indptr, m.data,
		spdist.sparse_euclidean, np.array([1, 2, 3], dtype=np.int64))
	assert np.array_equal(out_i[:, 0], idx[:, 0])
	assert np.all(out_i[:, 3:] == -1)
	assert np.all(np.isinf(out_d[out_i < 0]))
	for r in range(20):
		kept = out_i[r][out_i[r] >= 0]
		assert set(kept) <= set(idx[r][idx[r] >= 0])


# ---------------------------------------------------------------------------
# utils helpers used by the sparse NN-descent
# ---------------------------------------------------------------------------


def test_sparse_initalize_heap_from_graph_indices():
	m = _random_csr(25, 10, 0.5, seed=11, empty_rows=(4,))
	dense = m.toarray().astype(np.float64)
	rng = np.random.RandomState(0)
	graph = rng.randint(0, 25, size=(25, 5)).astype(np.int32)
	graph[:, 4] = -1
	graph[3, 0] = 3
	heap = nnutils.make_heap(25, 6)
	out = nnutils.sparse_initalize_heap_from_graph_indices(
		heap, graph, m.indptr, m.indices, m.data, spdist.sparse_euclidean)
	for i in range(25):
		expected = {int(j) for j in graph[i] if j >= 0 and j != i}
		stored = {int(j) for j in out[0][i] if j >= 0}
		assert stored == expected
		for slot in range(6):
			j = out[0][i, slot]
			if j >= 0:
				np.testing.assert_allclose(
					out[1][i, slot], np.linalg.norm(dense[i] - dense[j]), rtol=1e-4, atol=1e-6)
				assert out[2][i, slot] == 1
			else:
				assert np.isinf(out[1][i, slot])


def _all_pairs_updates(update_array, counts):
	got = []
	for t in range(update_array.shape[0]):
		block = update_array[t, : counts[t]]
		for p, q, d in zip(block[:, 0].view(np.int32), block[:, 1].view(np.int32), block[:, 2]):
			got.append((int(p), int(q), float(d)))
	return got


def test_sparse_generate_graph_update_array_pairs_and_distances():
	m = _random_csr(20, 8, 0.5, seed=12)
	dense = m.toarray().astype(np.float64)
	graph = nnutils.make_heap(20, 4)
	new = np.array([[0, 1, 2, -1], [5, 6, -1, -1], [10, -1, -1, -1], [3, 4, 7, 8]], dtype=np.int32)
	old = np.array([[9, 0, -1], [1, 5, 6], [-1, -1, -1], [3, 12, 13]], dtype=np.int32)
	n_threads = 2
	updates = np.zeros((n_threads, 200, 3), dtype=np.float32)
	counts = np.zeros(n_threads, dtype=np.int32)
	nnutils.sparse_generate_graph_update_array(
		updates, counts, new, old, graph, m.indices, m.indptr, m.data,
		spdist.sparse_euclidean, n_threads)
	got = _all_pairs_updates(updates, counts)

	expected = []
	for r in range(new.shape[0]):
		for j in range(new.shape[1]):
			p = new[r, j]
			if p < 0:
				continue
			for k in range(j + 1, new.shape[1]):
				q = new[r, k]
				if q >= 0:
					expected.append((int(p), int(q)))
			for q in old[r]:
				if q >= 0 and q != p:
					expected.append((int(p), int(q)))
	assert sorted((p, q) for p, q, _ in got) == sorted(expected)
	for p, q, d in got:
		np.testing.assert_allclose(d, np.linalg.norm(dense[p] - dense[q]), rtol=1e-4, atol=1e-6)


def test_sparse_generate_graph_update_array_filters_by_heap_threshold():
	m = _random_csr(10, 6, 0.6, seed=13)
	graph = nnutils.make_heap(10, 3)
	graph[1][:] = 0.0
	new = np.array([[0, 1, 2, 3]], dtype=np.int32)
	old = np.full((1, 1), -1, dtype=np.int32)
	updates = np.zeros((2, 50, 3), dtype=np.float32)
	counts = np.zeros(2, dtype=np.int32)
	nnutils.sparse_generate_graph_update_array(
		updates, counts, new, old, graph, m.indices, m.indptr, m.data,
		spdist.sparse_euclidean, 2)
	assert counts.sum() == 0


# ---------------------------------------------------------------------------
# sparse NN-descent building blocks
# ---------------------------------------------------------------------------


def _leaf_array(n, leaf_size):
	n_leaves = (n + leaf_size - 1) // leaf_size
	leaves = np.full((n_leaves, leaf_size), -1, dtype=np.int32)
	for v in range(n):
		leaves[v // leaf_size, v % leaf_size] = v
	return leaves


def test_generate_leaf_updates_emits_all_leaf_pairs_with_distances():
	m = _random_csr(18, 8, 0.6, seed=14)
	dense = m.toarray().astype(np.float64)
	leaves = _leaf_array(18, 5)
	graph = nnutils.make_heap(18, 4)
	n_threads = 2
	updates = np.zeros((n_threads, 200, 3), dtype=np.float32)
	counts = np.zeros(n_threads, dtype=np.int32)
	sparse_nnd.generate_leaf_updates(
		updates, counts, leaves, graph, m.indices, m.indptr, m.data,
		spdist.sparse_euclidean, n_threads)
	got = _all_pairs_updates(updates, counts)
	expected = set()
	for leaf in leaves:
		members = [int(v) for v in leaf if v >= 0]
		expected.update(itertools.combinations(members, 2))
	assert sorted((p, q) for p, q, _ in got) == sorted(expected)
	for p, q, d in got:
		np.testing.assert_allclose(d, np.linalg.norm(dense[p] - dense[q]), rtol=1e-4, atol=1e-6)


def test_generate_leaf_updates_skips_pairs_not_improving_either_heap():
	m = _random_csr(10, 6, 0.6, seed=15)
	leaves = _leaf_array(10, 5)
	graph = nnutils.make_heap(10, 3)
	graph[1][:] = 0.0
	updates = np.zeros((2, 100, 3), dtype=np.float32)
	counts = np.zeros(2, dtype=np.int32)
	sparse_nnd.generate_leaf_updates(
		updates, counts, leaves, graph, m.indices, m.indptr, m.data,
		spdist.sparse_euclidean, 2)
	assert counts.sum() == 0


def test_generate_leaf_updates_respects_capacity():
	m = _random_csr(12, 6, 0.6, seed=16)
	leaves = _leaf_array(12, 6)
	graph = nnutils.make_heap(12, 4)
	updates = np.zeros((2, 3, 3), dtype=np.float32)
	counts = np.zeros(2, dtype=np.int32)
	sparse_nnd.generate_leaf_updates(
		updates, counts, leaves, graph, m.indices, m.indptr, m.data,
		spdist.sparse_euclidean, 2)
	assert np.all(counts <= 3)
	assert counts.sum() > 0


def test_init_rp_tree_puts_every_leaf_mate_in_the_heap():
	m = _random_csr(30, 8, 0.6, seed=17, empty_rows=(2,))
	dense = m.toarray().astype(np.float64)
	leaf_size = 5
	leaves = _leaf_array(30, leaf_size)
	graph = nnutils.make_heap(30, leaf_size)
	sparse_nnd.init_rp_tree(m.indices, m.indptr, m.data, spdist.sparse_euclidean,
		graph, leaves, n_threads=2)
	for i in range(30):
		mates = {int(v) for v in leaves[i // leaf_size] if v >= 0 and v != i}
		stored = {int(v) for v in graph[0][i] if v >= 0}
		assert stored == mates
		for slot in range(leaf_size):
			j = graph[0][i, slot]
			if j >= 0:
				np.testing.assert_allclose(
					graph[1][i, slot], np.linalg.norm(dense[i] - dense[j]),
					rtol=1e-4, atol=1e-6)
				assert graph[2][i, slot] == 1


def test_init_rp_tree_keeps_k_nearest_when_leaf_larger_than_k():
	m = _random_csr(20, 8, 0.6, seed=18)
	dense = m.toarray().astype(np.float64)
	leaves = np.arange(20, dtype=np.int32)[None, :]
	k = 4
	graph = nnutils.make_heap(20, k)
	sparse_nnd.init_rp_tree(m.indices, m.indptr, m.data, spdist.sparse_euclidean,
		graph, leaves, n_threads=2)
	d = scd.cdist(dense, dense)
	np.fill_diagonal(d, np.inf)
	expected = np.sort(d, axis=1)[:, :k]
	np.testing.assert_allclose(np.sort(graph[1], axis=1), expected, rtol=1e-4, atol=1e-6)


def test_init_random_fills_rows_with_valid_other_points():
	m = _random_csr(40, 10, 0.5, seed=19)
	dense = m.toarray().astype(np.float64)
	k = 5
	graph = nnutils.make_heap(40, k)
	prefilled = graph[0][7].copy()
	graph[0][7, 0] = 3
	graph[1][7, 0] = 0.25
	prefilled_row = (graph[0][7].copy(), graph[1][7].copy())
	sparse_nnd.init_random(k, m.indices, m.indptr, m.data, graph,
		spdist.sparse_euclidean, _rng_state(1))
	assert np.all(prefilled == -1)
	assert np.array_equal(graph[0][7], prefilled_row[0])
	assert np.array_equal(graph[1][7], prefilled_row[1])
	for i in range(40):
		if i == 7:
			continue
		stored = graph[0][i][graph[0][i] >= 0]
		assert stored.shape[0] >= 1
		assert len(set(stored.tolist())) == stored.shape[0]
		assert np.all((stored < 40) & (stored != i))
		for slot in range(k):
			j = graph[0][i, slot]
			if j >= 0:
				np.testing.assert_allclose(
					graph[1][i, slot], np.linalg.norm(dense[i] - dense[j]),
					rtol=1e-4, atol=1e-6)
				assert graph[2][i, slot] == 1


def test_init_random_single_sample_does_not_hang():
	"""With one sample the call returns; pins the current output (the point is its
	own only candidate)."""
	m = _random_csr(1, 5, 1.0, seed=20)
	graph = nnutils.make_heap(1, 3)
	sparse_nnd.init_random(3, m.indices, m.indptr, m.data, graph,
		spdist.sparse_euclidean, _rng_state(1))
	# the only point is its own sole candidate: one entry (itself, distance 0)
	# remains after duplicate rejection and the other slots stay empty
	assert sorted(graph[0][0].tolist()) == [-1, -1, 0]
	assert graph[1][0][graph[0][0] == 0][0] == 0.0
	assert np.all(np.isinf(graph[1][0][graph[0][0] == -1]))


def test_sparse_process_candidates_only_improves_the_graph():
	m = _random_csr(60, 12, 0.5, seed=21)
	dense = m.toarray().astype(np.float64)
	k, n_threads = 5, 2
	rng_state = _rng_state(4)
	graph = nnutils.make_heap(60, k)
	sparse_nnd.init_random(k, m.indices, m.indptr, m.data, graph,
		spdist.sparse_euclidean, rng_state)
	before = graph[1].copy()
	new_c, old_c = nnutils.new_build_candidates(graph, 10, rng_state, n_threads)
	update_array = np.empty((n_threads, 5000, 3), dtype=np.float32)
	counts = np.zeros(n_threads, dtype=np.int32)
	n_changed = sparse_nnd.sparse_process_candidates(
		m.indices, m.indptr, m.data, spdist.sparse_euclidean, graph, new_c, old_c,
		0, 16384, n_threads, update_array, counts)
	assert n_changed > 0
	assert np.all(np.sort(graph[1], axis=1) <= np.sort(before, axis=1) + 1e-7)
	assert np.sum(graph[1] < before) > 0
	for i in range(60):
		for slot in range(k):
			j = graph[0][i, slot]
			if j >= 0:
				assert j != i
				np.testing.assert_allclose(
					graph[1][i, slot], np.linalg.norm(dense[i] - dense[j]),
					rtol=1e-4, atol=1e-6)


def test_nn_descent_internal_improves_graph_toward_exact():
	m = _random_csr(80, 15, 0.4, seed=22)
	dense = m.toarray().astype(np.float64)
	k = 6
	rng_state = _rng_state(9)
	graph = nnutils.make_heap(80, k)
	sparse_nnd.init_random(k, m.indices, m.indptr, m.data, graph,
		spdist.sparse_euclidean, rng_state)
	before = np.sort(graph[1], axis=1).sum()
	result = sparse_nnd.nn_descent_internal(
		graph, m.indices, m.indptr, m.data, k, rng_state,
		max_candidates=20, dist=spdist.sparse_euclidean, n_iters=5, delta=0.0)
	assert result is None
	after = np.sort(graph[1], axis=1).sum()
	assert after < before
	d = scd.cdist(dense, dense)
	np.fill_diagonal(d, np.inf)
	exact = np.sort(d, axis=1)[:, :k].sum()
	assert after >= exact - 1e-3


# ---------------------------------------------------------------------------
# nn_descent against brute force
# ---------------------------------------------------------------------------


def _run_nn_descent(m, k, dist, leaf_array, seed=1, **kwargs):
	rng_state = _rng_state(seed)
	return sparse_nnd.nn_descent(
		m.indices, m.indptr, m.data, k, rng_state, dist=dist,
		leaf_array=leaf_array, **kwargs)


def _brute_force(dense, k, metric):
	d = scd.cdist(dense, dense, metric=metric)
	np.fill_diagonal(d, np.inf)
	idx = np.argsort(d, axis=1, kind="stable")[:, :k]
	return idx, np.take_along_axis(d, idx, axis=1)


def _recall(found, exact):
	hits = sum(len(set(f.tolist()) & set(e.tolist())) for f, e in zip(found, exact))
	return hits / exact.size


@pytest.mark.parametrize(
	"name,dist,metric",
	[
		("euclidean", spdist.sparse_euclidean, "euclidean"),
		("manhattan", spdist.sparse_manhattan, "cityblock"),
		("cosine", spdist.sparse_cosine, "cosine"),
	],
)
def test_nn_descent_single_leaf_init_is_exact(name, dist, metric):
	m = _random_csr(60, 20, 0.4, seed=23)
	m.data = np.abs(m.data) + 0.1
	dense = m.toarray().astype(np.float64)
	k = 6
	leaves = np.arange(60, dtype=np.int32)[None, :]
	idx, dists = _run_nn_descent(m, k, dist, leaves, n_iters=3, delta=0.0)
	_, exact_d = _brute_force(dense, k, metric)
	assert idx.shape == (60, k) and dists.shape == (60, k)
	assert idx.dtype == np.int32 and dists.dtype == np.float32
	np.testing.assert_allclose(dists, exact_d, rtol=1e-3, atol=1e-5)
	assert np.all(np.diff(dists, axis=1) >= 0)
	assert not np.any(idx == np.arange(60)[:, None])
	for i in range(60):
		assert len(set(idx[i].tolist())) == k


def test_nn_descent_rows_report_the_distance_to_the_listed_neighbours():
	m = _random_csr(70, 25, 0.3, seed=24)
	dense = m.toarray().astype(np.float64)
	idx, dists = _run_nn_descent(m, 7, spdist.sparse_euclidean,
		_leaf_array(70, 8), n_iters=6, max_candidates=30)
	assert np.all((idx >= 0) & (idx < 70))
	for i in range(70):
		np.testing.assert_allclose(
			dists[i], np.linalg.norm(dense[idx[i]] - dense[i], axis=1), rtol=1e-4, atol=1e-5)
	assert np.all(np.diff(dists, axis=1) >= 0)


@pytest.mark.parametrize("rp_tree_init", [True, False])
def test_nn_descent_recall_against_brute_force(rp_tree_init):
	m = _random_csr(300, 40, 0.15, seed=25)
	dense = m.toarray().astype(np.float64)
	k = 10
	if rp_tree_init:
		rng = np.random.RandomState(0)
		blocks = []
		for _ in range(6):
			order = np.argsort(dense @ rng.randn(dense.shape[1])).astype(np.int32)
			blocks.append(order.reshape(25, 12))
		leaves = np.vstack(blocks)
	else:
		leaves = np.array([[-1]], dtype=np.int32)
	idx, dists = _run_nn_descent(m, k, spdist.sparse_euclidean, leaves,
		n_iters=12, max_candidates=40, delta=0.0, rp_tree_init=rp_tree_init)
	exact_idx, exact_d = _brute_force(dense, k, "euclidean")
	assert _recall(idx, exact_idx) >= 0.9
	assert np.all(dists >= exact_d - 1e-4)


def test_nn_descent_handles_empty_rows_and_explicit_zeros():
	m = _random_csr(50, 20, 0.3, seed=26, explicit_zeros=15, empty_rows=(0, 10, 20))
	dense = m.toarray().astype(np.float64)
	k = 5
	leaves = np.arange(50, dtype=np.int32)[None, :]
	idx, dists = _run_nn_descent(m, k, spdist.sparse_euclidean, leaves, n_iters=3, delta=0.0)
	_, exact_d = _brute_force(dense, k, "euclidean")
	np.testing.assert_allclose(dists, exact_d, rtol=1e-3, atol=1e-5)


def test_nn_descent_init_graph_is_refined_in_place_and_returned_sorted():
	m = _random_csr(50, 20, 0.3, seed=28)
	dense = m.toarray().astype(np.float64)
	k = 5
	exact_idx, exact_d = _brute_force(dense, k, "euclidean")
	init = nnutils.make_heap(50, k)
	init_indices = exact_idx.astype(np.int32)
	nnutils.sparse_initalize_heap_from_graph_indices(
		init, init_indices, m.indptr, m.indices, m.data, spdist.sparse_euclidean)
	rng_state = _rng_state(1)
	idx, dists = sparse_nnd.nn_descent(
		m.indices, m.indptr, m.data, k, rng_state, n_iters=2, delta=0.0,
		dist=spdist.sparse_euclidean, init_graph=init,
		leaf_array=np.array([[-1]], dtype=np.int32), verbose=True)
	np.testing.assert_allclose(dists, exact_d, rtol=1e-3, atol=1e-5)
	np.testing.assert_array_equal(np.sort(init[0], axis=1), np.sort(exact_idx, axis=1))
	np.testing.assert_array_equal(np.sort(idx, axis=1), np.sort(exact_idx, axis=1))
	assert np.all(np.diff(dists, axis=1) >= 0)
	# from a random start the passed heap itself ends up holding the refined graph
	start = nnutils.make_heap(50, k)
	sparse_nnd.init_random(k, m.indices, m.indptr, m.data, start,
		spdist.sparse_euclidean, _rng_state(2))
	random_start = start[0].copy()
	idx2, dists2 = sparse_nnd.nn_descent(
		m.indices, m.indptr, m.data, k, _rng_state(3), n_iters=10, delta=0.0,
		dist=spdist.sparse_euclidean, init_graph=start,
		leaf_array=np.array([[-1]], dtype=np.int32))
	assert not np.array_equal(np.sort(random_start, axis=1), np.sort(exact_idx, axis=1))
	np.testing.assert_array_equal(np.sort(start[0], axis=1), np.sort(idx2, axis=1))
	np.testing.assert_allclose(dists2, exact_d, rtol=1e-3, atol=1e-5)


def test_nn_descent_rejects_init_graph_of_wrong_shape():
	m = _random_csr(30, 10, 0.4, seed=29)
	bad = nnutils.make_heap(30, 3)
	rng_state = _rng_state(1)
	with pytest.raises(ValueError):
		sparse_nnd.nn_descent(
			m.indices, m.indptr, m.data, 5, rng_state, dist=spdist.sparse_euclidean,
			init_graph=bad, leaf_array=np.array([[-1]], dtype=np.int32))
	bad = nnutils.make_heap(29, 5)
	with pytest.raises(ValueError):
		sparse_nnd.nn_descent(
			m.indices, m.indptr, m.data, 5, rng_state, dist=spdist.sparse_euclidean,
			init_graph=bad, leaf_array=np.array([[-1]], dtype=np.int32))


def test_nn_descent_default_distance_is_sparse_euclidean():
	m = _random_csr(40, 15, 0.4, seed=30)
	leaves = np.arange(40, dtype=np.int32)[None, :]
	rng_state = _rng_state(3)
	idx, dists = sparse_nnd.nn_descent(
		m.indices, m.indptr, m.data, 4, rng_state, n_iters=2, delta=0.0, leaf_array=leaves)
	dense = m.toarray().astype(np.float64)
	_, exact_d = _brute_force(dense, 4, "euclidean")
	np.testing.assert_allclose(dists, exact_d, rtol=1e-3, atol=1e-5)
