"""Tests for the dense distance functions in pynndescent.distances.

Expected values come from scipy.spatial.distance, scipy.stats, or short
reference implementations written with numpy inside this file.
"""

import numba
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats
from scipy.spatial import distance as spd

import pynndescent.distances as dist

FLOAT32_MAX = np.finfo(np.float32).max


def _rng(seed=0):
	return np.random.RandomState(seed)


def _pairs(n=8, dim=7, seed=0, dtype=np.float64, low=-1.0, high=2.0):
	rng = _rng(seed)
	data = rng.uniform(low, high, size=(n, dim)).astype(dtype)
	return [(data[i], data[(i + 1) % n]) for i in range(n)]


def _positive_pairs(n=8, dim=7, seed=0, dtype=np.float32):
	return _pairs(n, dim, seed, dtype, low=0.05, high=1.5)


def _binary_pairs(n=20, dim=11, seed=0, dtype=np.float64):
	rng = _rng(seed)
	data = (rng.uniform(size=(n, dim)) < 0.5).astype(dtype)
	return [(data[i], data[(i + 1) % n]) for i in range(n)]


def _binary_counts(x, y):
	xb = x != 0
	yb = y != 0
	ttt = float(np.sum(xb & yb))
	tf = float(np.sum(xb & ~yb))
	ft = float(np.sum(~xb & yb))
	ff = float(np.sum(~xb & ~yb))
	return ttt, tf, ft, ff


# ---------------------------------------------------------------- popcnt_u8


@numba.njit()
def _popcounts_all():
	out = np.empty(256, dtype=np.int64)
	for v in range(256):
		out[v] = dist.popcnt_u8(np.uint8(v))
	return out


def test_popcnt_u8_all_byte_values():
	"""Check popcnt_u8 for all 256 byte values against bin(v).count('1')."""
	expected = np.array([bin(v).count("1") for v in range(256)])
	assert_array_equal(_popcounts_all(), expected)


# ------------------------------------------------------- minkowski family


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_euclidean_matches_scipy(dtype):
	"""Check euclidean matches scipy."""
	for x, y in _pairs(dtype=dtype):
		assert_allclose(dist.euclidean(x, y), spd.euclidean(x, y), rtol=1e-5)


def test_euclidean_hand_value_zero_and_symmetry():
	"""Check euclidean hand value zero and symmetry."""
	x = np.array([0.0, 0.0], dtype=np.float64)
	y = np.array([3.0, 4.0], dtype=np.float64)
	assert dist.euclidean(x, y) == pytest.approx(5.0)
	assert dist.euclidean(y, x) == pytest.approx(5.0)
	assert dist.euclidean(y, y) == 0.0
	assert dist.euclidean(x, x) == 0.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_squared_euclidean_matches_scipy(dtype):
	"""Check squared euclidean matches scipy."""
	for x, y in _pairs(dtype=dtype):
		assert_allclose(dist.squared_euclidean(x, y), spd.sqeuclidean(x, y), rtol=1e-5)


def test_squared_euclidean_hand_value_and_identity():
	"""Check squared euclidean hand value and identity."""
	x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
	y = np.array([2.0, 0.0, 3.0], dtype=np.float32)
	assert dist.squared_euclidean(x, y) == pytest.approx(5.0)
	assert dist.squared_euclidean(y, x) == pytest.approx(5.0)
	assert dist.squared_euclidean(x, x) == 0.0


def test_standardised_euclidean_matches_scipy():
	"""Check standardised euclidean matches scipy."""
	rng = _rng(1)
	sigma = rng.uniform(0.5, 3.0, size=7).astype(np.float64)
	for x, y in _pairs():
		assert_allclose(
			dist.standardised_euclidean(x, y, sigma),
			spd.seuclidean(x, y, sigma),
			rtol=1e-6,
		)


def test_standardised_euclidean_default_sigma_is_euclidean():
	"""Check standardised euclidean default sigma is euclidean."""
	x = np.array([1.0, 5.0], dtype=np.float32)
	y = np.array([4.0, 1.0], dtype=np.float32)
	assert dist.standardised_euclidean(x, y) == pytest.approx(5.0, rel=1e-6)


def test_standardised_euclidean_hand_value():
	"""Check standardised euclidean hand value."""
	x = np.array([2.0, 0.0])
	y = np.array([0.0, 0.0])
	sigma = np.array([4.0, 1.0])
	assert dist.standardised_euclidean(x, y, sigma) == pytest.approx(1.0)
	assert dist.standardised_euclidean(x, x, sigma) == 0.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_manhattan_matches_scipy(dtype):
	"""Check manhattan matches scipy."""
	for x, y in _pairs(dtype=dtype):
		assert_allclose(dist.manhattan(x, y), spd.cityblock(x, y), rtol=1e-5)
		assert_allclose(dist.manhattan(y, x), dist.manhattan(x, y), rtol=1e-6)
	assert dist.manhattan(x, x) == 0.0


def test_manhattan_hand_value():
	"""Check manhattan hand value."""
	x = np.array([1.0, -2.0, 3.0])
	y = np.array([0.0, 2.0, 3.5])
	assert dist.manhattan(x, y) == pytest.approx(5.5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_chebyshev_matches_scipy(dtype):
	"""Check chebyshev matches scipy."""
	for x, y in _pairs(dtype=dtype):
		assert_allclose(dist.chebyshev(x, y), spd.chebyshev(x, y), rtol=1e-6)
		assert dist.chebyshev(y, x) == dist.chebyshev(x, y)
	assert dist.chebyshev(x, x) == 0.0


def test_chebyshev_hand_value_negative_difference():
	"""Check chebyshev hand value negative difference."""
	x = np.array([1.0, -5.0, 3.0])
	y = np.array([0.0, 2.0, 3.5])
	assert dist.chebyshev(x, y) == pytest.approx(7.0)


@pytest.mark.parametrize("p", [1, 2, 3, 4.5])
def test_minkowski_matches_scipy(p):
	"""Check minkowski matches scipy."""
	for x, y in _pairs():
		assert_allclose(dist.minkowski(x, y, p), spd.minkowski(x, y, p), rtol=1e-6)
		assert_allclose(dist.minkowski(y, x, p), dist.minkowski(x, y, p), rtol=1e-6)
	assert dist.minkowski(x, x, p) == 0.0


def test_minkowski_default_p_is_two():
	"""Check minkowski default p is two."""
	x, y = _pairs()[0]
	assert dist.minkowski(x, y) == pytest.approx(dist.euclidean(x, y), rel=1e-6)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_weighted_minkowski_matches_reference(p):
	"""Check weighted minkowski matches reference."""
	rng = _rng(2)
	w = rng.uniform(0.1, 2.0, size=7)
	for x, y in _pairs():
		expected = np.sum(w * np.abs(x - y) ** p) ** (1.0 / p)
		assert_allclose(dist.weighted_minkowski(x, y, w, p), expected, rtol=1e-6)
	assert dist.weighted_minkowski(x, x, w, p) == 0.0


def test_weighted_minkowski_unit_weights_equal_minkowski():
	"""Check weighted minkowski unit weights equal minkowski."""
	ones = np.ones(7)
	for x, y in _pairs():
		assert_allclose(
			dist.weighted_minkowski(x, y, ones, 3),
			dist.minkowski(x, y, 3),
			rtol=1e-6,
		)


def test_weighted_minkowski_defaults_on_two_dims():
	"""Check weighted minkowski defaults on two dims."""
	x = np.array([0.0, 0.0], dtype=np.float32)
	y = np.array([3.0, 4.0], dtype=np.float32)
	assert dist.weighted_minkowski(x, y) == pytest.approx(5.0, rel=1e-6)


def test_mahalanobis_matches_scipy():
	"""Check mahalanobis matches scipy."""
	rng = _rng(3)
	a = rng.normal(size=(7, 7))
	vinv = a @ a.T + np.eye(7)
	for x, y in _pairs(dtype=np.float32):
		assert_allclose(
			dist.mahalanobis(x, y, vinv.astype(np.float32)),
			spd.mahalanobis(x, y, vinv),
			rtol=1e-4,
		)


def test_mahalanobis_default_identity_and_diagonal_hand_value():
	"""Check mahalanobis default identity and diagonal hand value."""
	x = np.array([3.0, 0.0], dtype=np.float32)
	y = np.array([0.0, 4.0], dtype=np.float32)
	assert dist.mahalanobis(x, y) == pytest.approx(5.0, rel=1e-6)
	vinv = np.array([[4.0, 0.0], [0.0, 1.0]], dtype=np.float32)
	assert dist.mahalanobis(
		np.array([1.0, 0.0], dtype=np.float32),
		np.zeros(2, dtype=np.float32),
		vinv,
	) == pytest.approx(2.0, rel=1e-6)
	assert dist.mahalanobis(x, x, vinv) == 0.0


# ---------------------------------------------------- general elementwise


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hamming_matches_scipy_and_hand_value(dtype):
	"""Check hamming matches scipy and hand value."""
	x = np.array([1, 2, 3, 4], dtype=dtype)
	y = np.array([1, 0, 3, 5], dtype=dtype)
	assert dist.hamming(x, y) == pytest.approx(0.5)
	assert dist.hamming(x, x) == 0.0
	for a, b in _binary_pairs(dtype=dtype):
		assert_allclose(dist.hamming(a, b), spd.hamming(a, b), rtol=1e-6)
		assert dist.hamming(a, b) == dist.hamming(b, a)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_canberra_matches_scipy_and_zero_denominators(dtype):
	"""Check canberra matches scipy and zero denominators."""
	for x, y in _pairs(dtype=dtype):
		assert_allclose(dist.canberra(x, y), spd.canberra(x, y), rtol=1e-4)
	x = np.array([0.0, 1.0, 0.0, 2.0], dtype=dtype)
	y = np.array([0.0, 3.0, 0.0, 2.0], dtype=dtype)
	assert dist.canberra(x, y) == pytest.approx(0.5)
	assert dist.canberra(x, x) == 0.0
	assert dist.canberra(np.zeros(3, dtype=dtype), np.zeros(3, dtype=dtype)) == 0.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bray_curtis_matches_scipy_and_zero_vectors(dtype):
	"""Check bray curtis matches scipy and zero vectors."""
	for x, y in _positive_pairs(dtype=dtype):
		assert_allclose(dist.bray_curtis(x, y), spd.braycurtis(x, y), rtol=1e-5)
		assert_allclose(dist.bray_curtis(y, x), dist.bray_curtis(x, y), rtol=1e-6)
	z = np.zeros(4, dtype=dtype)
	assert dist.bray_curtis(z, z) == 0.0
	x = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
	assert dist.bray_curtis(x, x) == 0.0
	assert dist.bray_curtis(x, z) == pytest.approx(1.0)


def test_haversine_known_values():
	"""Check haversine known values."""
	origin = np.array([0.0, 0.0])
	antipode = np.array([0.0, np.pi])
	assert dist.haversine(origin, antipode) == pytest.approx(np.pi)
	pole_n = np.array([np.pi / 2, 0.0])
	pole_s = np.array([-np.pi / 2, 0.0])
	assert dist.haversine(pole_n, pole_s) == pytest.approx(np.pi)
	quarter = np.array([0.0, np.pi / 2])
	assert dist.haversine(origin, quarter) == pytest.approx(np.pi / 2)
	assert dist.haversine(quarter, quarter) == 0.0


def test_haversine_matches_sklearn_and_symmetric():
	"""Check haversine matches sklearn and symmetric."""
	from sklearn.metrics.pairwise import haversine_distances

	rng = _rng(4)
	pts = np.column_stack(
		[rng.uniform(-np.pi / 2, np.pi / 2, 10), rng.uniform(-np.pi, np.pi, 10)]
	)
	expected = haversine_distances(pts)
	for i in range(10):
		for j in range(10):
			assert_allclose(
				dist.haversine(pts[i], pts[j]), expected[i, j], rtol=1e-6, atol=1e-7
			)


def test_haversine_requires_two_dimensions():
	"""Check haversine requires two dimensions."""
	with pytest.raises(ValueError):
		dist.haversine(np.zeros(3), np.zeros(3))


# ---------------------------------------------------------- binary metrics


def _jaccard_ref(x, y):
	ttt, tf, ft, ff = _binary_counts(x, y)
	union = ttt + tf + ft
	return 0.0 if union == 0 else (tf + ft) / union


def _matching_ref(x, y):
	ttt, tf, ft, ff = _binary_counts(x, y)
	return (tf + ft) / len(x)


def _dice_ref(x, y):
	ttt, tf, ft, ff = _binary_counts(x, y)
	ne = tf + ft
	return 0.0 if ne == 0 else ne / (2 * ttt + ne)


def _kulsinski_ref(x, y):
	ttt, tf, ft, ff = _binary_counts(x, y)
	ne = tf + ft
	n = len(x)
	return 0.0 if ne == 0 else (ne - ttt + n) / (ne + n)


def _rogers_tanimoto_ref(x, y):
	ttt, tf, ft, ff = _binary_counts(x, y)
	ne = tf + ft
	return 2 * ne / (len(x) + ne)


def _russellrao_ref(x, y):
	ttt, tf, ft, ff = _binary_counts(x, y)
	return (len(x) - ttt) / len(x)


def _sokal_sneath_ref(x, y):
	ttt, tf, ft, ff = _binary_counts(x, y)
	ne = tf + ft
	return 0.0 if ne == 0 else ne / (0.5 * ttt + ne)


def _yule_ref(x, y):
	ttt, tf, ft, ff = _binary_counts(x, y)
	if tf == 0 or ft == 0:
		return 0.0
	return 2 * tf * ft / (ttt * ff + tf * ft)


_BINARY_CASES = [
	("jaccard", dist.jaccard, _jaccard_ref),
	("matching", dist.matching, _matching_ref),
	("dice", dist.dice, _dice_ref),
	("kulsinski", dist.kulsinski, _kulsinski_ref),
	("rogers_tanimoto", dist.rogers_tanimoto, _rogers_tanimoto_ref),
	("russellrao", dist.russellrao, _russellrao_ref),
	("sokal_michener", dist.sokal_michener, _rogers_tanimoto_ref),
	("sokal_sneath", dist.sokal_sneath, _sokal_sneath_ref),
	("yule", dist.yule, _yule_ref),
]


@pytest.mark.parametrize("name,func,ref", _BINARY_CASES, ids=[c[0] for c in _BINARY_CASES])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.bool_])
def test_binary_metrics_match_reference(name, func, ref, dtype):
	"""Check binary metrics match reference."""
	pairs = _binary_pairs(n=40, dim=11, seed=5, dtype=dtype)
	for x, y in pairs:
		assert_allclose(func(x, y), ref(x, y), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("name,func,ref", _BINARY_CASES, ids=[c[0] for c in _BINARY_CASES])
def test_binary_metrics_hand_value(name, func, ref):
	"""Check binary metrics hand value."""
	x = np.array([1, 1, 0, 0, 1, 0], dtype=np.float64)
	y = np.array([1, 0, 1, 0, 1, 0], dtype=np.float64)
	# ttt=2, tf=1, ft=1, ff=2, n=6
	expected = {
		"jaccard": 2.0 / 4.0,
		"matching": 2.0 / 6.0,
		"dice": 2.0 / 6.0,
		"kulsinski": (2 - 2 + 6) / (2 + 6),
		"rogers_tanimoto": 4.0 / 8.0,
		"russellrao": 4.0 / 6.0,
		"sokal_michener": 4.0 / 8.0,
		"sokal_sneath": 2.0 / 3.0,
		"yule": 2.0 / (2 * 2 + 1),
	}[name]
	assert func(x, y) == pytest.approx(expected)
	assert func(y, x) == pytest.approx(expected)


@pytest.mark.parametrize("name,func,ref", _BINARY_CASES, ids=[c[0] for c in _BINARY_CASES])
def test_binary_metrics_identity_and_zero_vectors(name, func, ref):
	"""Pins the current output: every binary metric is 0 for identical and for all-zero vectors (russellrao differs from scipy here)."""
	x = np.array([1, 0, 1, 1, 0], dtype=np.float64)
	z = np.zeros(5, dtype=np.float64)
	assert func(x, x) == pytest.approx(0.0)
	assert func(z, z) == pytest.approx(0.0)


@pytest.mark.parametrize("name,func,ref", _BINARY_CASES, ids=[c[0] for c in _BINARY_CASES])
def test_binary_metrics_treat_any_nonzero_as_true(name, func, ref):
	"""Check binary metrics treat any nonzero as true."""
	x = np.array([3.5, 0.0, -2.0, 0.0, 7.0])
	y = np.array([1.0, 1.0, 0.0, 0.0, 0.5])
	xb = (x != 0).astype(np.float64)
	yb = (y != 0).astype(np.float64)
	assert func(x, y) == pytest.approx(func(xb, yb))


def test_jaccard_matches_scipy():
	"""Check jaccard matches scipy."""
	for x, y in _binary_pairs(n=30, seed=6):
		xb, yb = x.astype(bool), y.astype(bool)
		if (xb | yb).any():
			assert_allclose(dist.jaccard(x, y), spd.jaccard(xb, yb), rtol=1e-6)


def test_matching_dice_rogers_sokal_match_scipy():
	"""Check matching dice rogers sokal match scipy."""
	rng = _rng(7)
	for _ in range(30):
		xb = rng.uniform(size=9) < 0.5
		yb = rng.uniform(size=9) < 0.5
		if (xb ^ yb).sum() == 0:
			continue
		x, y = xb.astype(np.float64), yb.astype(np.float64)
		assert_allclose(dist.matching(x, y), spd.hamming(xb, yb), rtol=1e-6)
		assert_allclose(dist.dice(x, y), spd.dice(xb, yb), rtol=1e-6)
		assert_allclose(dist.rogers_tanimoto(x, y), spd.rogerstanimoto(xb, yb), rtol=1e-6)
		assert_allclose(dist.sokal_sneath(x, y), spd.sokalsneath(xb, yb), rtol=1e-6)
		assert_allclose(dist.russellrao(x, y), spd.russellrao(xb, yb), rtol=1e-6)
		if (xb & ~yb).any() and (~xb & yb).any():
			assert_allclose(dist.yule(x, y), spd.yule(xb, yb), rtol=1e-6)


def test_russellrao_identical_vectors_are_distance_zero():
	"""Pins the current output: russellrao(x, x) is 0 for identical vectors (scipy gives a nonzero value), and a hand value for differing ones."""
	x = np.array([1, 0, 1, 0, 0], dtype=np.float64)
	assert dist.russellrao(x, x) == 0.0
	y = np.array([1, 0, 0, 0, 0], dtype=np.float64)
	assert dist.russellrao(x, y) == pytest.approx(0.8)


def test_yule_one_sided_disagreement_is_zero():
	"""Pins the current output: yule is 0 when only one of the two disagreement counts is nonzero."""
	x = np.array([1, 1, 0, 1], dtype=np.float64)
	y = np.array([1, 0, 0, 0], dtype=np.float64)
	assert dist.yule(x, y) == 0.0
	assert dist.yule(y, x) == 0.0


def test_alternative_jaccard_and_correction_round_trip():
	"""Check alternative jaccard and correction round trip."""
	for x, y in _binary_pairs(n=30, dim=13, seed=8, dtype=np.float32):
		ttt, tf, ft, ff = _binary_counts(x, y)
		if ttt == 0:
			continue
		alt = dist.alternative_jaccard(x, y)
		assert alt == pytest.approx(-np.log2(ttt / (ttt + tf + ft)), rel=1e-5, abs=1e-6)
		assert_allclose(
			dist.correct_alternative_jaccard(alt), _jaccard_ref(x, y), rtol=1e-4, atol=1e-6
		)


def test_alternative_jaccard_identity_and_zero_vectors():
	"""Pins the current output: identical vectors and two zero vectors both give 0."""
	x = np.array([1, 0, 1, 1], dtype=np.float32)
	z = np.zeros(4, dtype=np.float32)
	assert dist.alternative_jaccard(x, x) == 0.0
	assert dist.alternative_jaccard(z, z) == 0.0


def test_alternative_jaccard_hand_value():
	"""Check alternative jaccard hand value."""
	x = np.array([1, 1, 1, 1, 0, 0], dtype=np.float32)
	y = np.array([1, 1, 0, 0, 1, 1], dtype=np.float32)
	# intersection 2, union 6
	assert dist.alternative_jaccard(x, y) == pytest.approx(np.log2(3.0), rel=1e-5)


def test_correct_alternative_jaccard_values_and_array_input():
	"""Check correct alternative jaccard values and array input."""
	assert dist.correct_alternative_jaccard(0.0) == 0.0
	assert dist.correct_alternative_jaccard(1.0) == pytest.approx(0.5)
	assert dist.correct_alternative_jaccard(2.0) == pytest.approx(0.75)
	vals = np.array([0.0, 1.0, 3.0, 0.5])
	out = dist.correct_alternative_jaccard(vals)
	assert out.shape == vals.shape
	assert_allclose(out, 1.0 - 2.0 ** (-vals), rtol=1e-12)


# --------------------------------------------------- cosine and dot family


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cosine_matches_scipy_and_symmetry(dtype):
	"""Check cosine matches scipy and symmetry."""
	for x, y in _pairs(dtype=dtype):
		assert_allclose(dist.cosine(x, y), spd.cosine(x, y), rtol=1e-4, atol=1e-6)
		assert_allclose(dist.cosine(y, x), dist.cosine(x, y), rtol=1e-6)


def test_cosine_hand_values_and_zero_vectors():
	"""Check hand values; pins the current output for zero vectors (0 for two zero vectors, 1 for one)."""
	z = np.zeros(3)
	x = np.array([1.0, 2.0, 3.0])
	assert dist.cosine(z, z) == 0.0
	assert dist.cosine(x, z) == 1.0
	assert dist.cosine(z, x) == 1.0
	assert dist.cosine(x, x) == pytest.approx(0.0, abs=1e-12)
	assert dist.cosine(x, 5.0 * x) == pytest.approx(0.0, abs=1e-12)
	assert dist.cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
	assert dist.cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(2.0)


def test_alternative_cosine_matches_log_of_ratio():
	"""Check alternative cosine matches log of ratio."""
	for x, y in _positive_pairs():
		x64, y64 = x.astype(np.float64), y.astype(np.float64)
		expected = -np.log2(1.0 - spd.cosine(x64, y64))
		assert dist.alternative_cosine(x, y) == pytest.approx(expected, rel=2e-5, abs=1e-6)


def test_alternative_cosine_edge_cases():
	"""Pins the current output: zero, orthogonal and opposite inputs give FLOAT32_MAX; hand values for parallel vectors."""
	x = np.array([1.0, 0.0], dtype=np.float32)
	z = np.zeros(2, dtype=np.float32)
	orth = np.array([0.0, 1.0], dtype=np.float32)
	assert dist.alternative_cosine(z, z) == 0.0
	assert dist.alternative_cosine(x, z) == FLOAT32_MAX
	assert dist.alternative_cosine(z, x) == FLOAT32_MAX
	assert dist.alternative_cosine(x, orth) == FLOAT32_MAX
	assert dist.alternative_cosine(x, -x) == FLOAT32_MAX
	assert dist.alternative_cosine(x, 3.0 * x) == pytest.approx(0.0, abs=1e-6)
	half = np.array([1.0, 1.0], dtype=np.float32)
	assert dist.alternative_cosine(x, half) == pytest.approx(0.5, rel=1e-5)


def test_correct_alternative_cosine_round_trip_and_clamp():
	"""Check correct alternative cosine round trip and clamp."""
	for x, y in _positive_pairs():
		alt = dist.alternative_cosine(x, y)
		expected = spd.cosine(x.astype(np.float64), y.astype(np.float64))
		assert dist.correct_alternative_cosine(alt) == pytest.approx(
			expected, abs=2e-5
		)
	assert dist.correct_alternative_cosine(0.0) == 0.0
	assert dist.correct_alternative_cosine(1.0) == pytest.approx(0.5)
	assert dist.correct_alternative_cosine(-0.1) == 0.0
	vals = np.array([0.0, 1.0, 2.0])
	assert_allclose(dist.correct_alternative_cosine(vals), [0.0, 0.5, 0.75])


def _unit_pairs(seed=9, n=8, dim=6):
	rng = _rng(seed)
	data = np.abs(rng.normal(size=(n, dim))).astype(np.float32)
	data /= np.linalg.norm(data, axis=1, keepdims=True)
	return [(data[i], data[(i + 1) % n]) for i in range(n)]


def test_dot_on_unit_vectors_is_one_minus_dot():
	"""Check dot on unit vectors is one minus dot."""
	for x, y in _unit_pairs():
		assert dist.dot(x, y) == pytest.approx(1.0 - float(np.dot(x, y)), abs=1e-6)
		assert dist.dot(x, y) == pytest.approx(dist.dot(y, x))
	x = _unit_pairs()[0][0]
	assert dist.dot(x, x) == pytest.approx(0.0, abs=1e-6)


def test_dot_nonpositive_inner_product_is_one():
	"""Pins the current output: non-positive inner products give distance 1."""
	x = np.array([1.0, 0.0], dtype=np.float32)
	assert dist.dot(x, np.array([0.0, 1.0], dtype=np.float32)) == 1.0
	assert dist.dot(x, np.array([-1.0, 0.0], dtype=np.float32)) == 1.0


def test_direct_cosine_on_unit_vectors_matches_cosine():
	"""Check direct cosine on unit vectors matches cosine."""
	for x, y in _unit_pairs():
		assert dist.direct_cosine(x, y) == pytest.approx(
			spd.cosine(x.astype(np.float64), y.astype(np.float64)), abs=1e-5
		)
	x = _unit_pairs()[0][0]
	assert dist.direct_cosine(x, x) == pytest.approx(0.0, abs=1e-6)
	assert dist.direct_cosine(
		np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)
	) == 1.0


def test_alternative_dot_and_correction_round_trip():
	"""Check -log2(dot) and its correction on unit vectors; pins the current output of FLOAT32_MAX for non-positive dots."""
	for x, y in _unit_pairs():
		alt = dist.alternative_dot(x, y)
		assert alt == pytest.approx(-np.log2(float(np.dot(x, y))), rel=2e-5, abs=1e-6)
		assert dist.correct_alternative_cosine(alt) == pytest.approx(
			1.0 - float(np.dot(x, y)), abs=2e-5
		)
	x = np.array([1.0, 0.0], dtype=np.float32)
	assert dist.alternative_dot(x, np.array([0.0, 1.0], dtype=np.float32)) == FLOAT32_MAX
	assert dist.alternative_dot(x, -x) == FLOAT32_MAX
	half = np.array([0.5, 0.0], dtype=np.float32)
	assert dist.alternative_dot(x, half) == pytest.approx(1.0)


def test_inner_product_is_negative_dot():
	"""Check inner product is negative dot."""
	for x, y in _pairs(dtype=np.float32):
		assert dist.inner_product(x, y) == pytest.approx(-float(np.dot(x, y)), rel=2e-5, abs=1e-6)
		assert dist.inner_product(x, y) == pytest.approx(dist.inner_product(y, x))
	x = np.array([1.0, 2.0], dtype=np.float32)
	y = np.array([3.0, 4.0], dtype=np.float32)
	assert dist.inner_product(x, y) == pytest.approx(-11.0)
	assert dist.inner_product(x, np.zeros(2, dtype=np.float32)) == 0.0


def test_alternative_inner_product_and_correction():
	"""Check 1/ip hand values and the correction; pins the current output of FLOAT32_MAX for non-positive inner products."""
	x = np.array([1.0, 2.0], dtype=np.float32)
	y = np.array([3.0, 4.0], dtype=np.float32)
	assert dist.alternative_inner_product(x, y) == pytest.approx(1.0 / 11.0, rel=1e-6)
	assert dist.alternative_inner_product(x, -y) == FLOAT32_MAX
	assert dist.alternative_inner_product(x, np.zeros(2, dtype=np.float32)) == FLOAT32_MAX
	assert dist.correct_alternative_inner_product(dist.alternative_inner_product(x, y)) == pytest.approx(-11.0, rel=1e-5)
	assert dist.correct_alternative_inner_product(0.25) == pytest.approx(-4.0)
	assert dist.correct_alternative_inner_product(FLOAT32_MAX) == 0.0
	assert_allclose(
		dist.correct_alternative_inner_product(np.array([0.5, 2.0])), [-2.0, -0.5]
	)


def test_proxy_inner_product_matches_reference():
	"""Check proxy inner product matches reference."""
	for x, y in _positive_pairs():
		x64, y64 = x.astype(np.float64), y.astype(np.float64)
		ip = np.dot(x64, y64)
		expected = -np.log2(ip / np.sqrt(np.dot(x64, x64) * np.dot(y64, y64))) + 1.0 / np.sqrt(ip)
		assert dist.proxy_inner_product(x, y) == pytest.approx(expected, rel=2e-5, abs=1e-6)


def test_proxy_inner_product_edge_cases():
	"""Pins the current output: zero, orthogonal and opposite inputs give FLOAT32_MAX; identical unit vectors give 1."""
	x = np.array([1.0, 0.0], dtype=np.float32)
	z = np.zeros(2, dtype=np.float32)
	assert dist.proxy_inner_product(x, z) == FLOAT32_MAX
	assert dist.proxy_inner_product(z, x) == FLOAT32_MAX
	assert dist.proxy_inner_product(x, np.array([0.0, 1.0], dtype=np.float32)) == FLOAT32_MAX
	assert dist.proxy_inner_product(x, -x) == FLOAT32_MAX
	# identical unit vector: -log2(1) + 1/sqrt(1)
	assert dist.proxy_inner_product(x, x) == pytest.approx(1.0, rel=1e-5)


def test_proxy_inner_product_orders_like_dot_on_unit_vectors():
	"""Check the proxy ranks candidates in the same order as the dot product on unit vectors."""
	rng = _rng(10)
	q = np.abs(rng.normal(size=5)).astype(np.float32)
	q /= np.linalg.norm(q)
	cands = np.abs(rng.normal(size=(30, 5))).astype(np.float32)
	cands /= np.linalg.norm(cands, axis=1, keepdims=True)
	proxy = np.array([dist.proxy_inner_product(q, c) for c in cands])
	ips = cands @ q
	assert_array_equal(np.argsort(proxy), np.argsort(-ips))


# ------------------------------------------------------------ proxy family


def _normalized(x):
	x = x.astype(np.float64)
	return x / x.sum()


def test_proxy_wasserstein_1d_matches_cdf_l1():
	"""Check the proxy against the CDF L1 formula and scipy wasserstein_distance."""
	for x, y in _positive_pairs():
		expected = np.sum(np.abs(np.cumsum(_normalized(x)) - np.cumsum(_normalized(y))))
		assert dist.proxy_wasserstein_1d(x, y) == pytest.approx(expected, rel=2e-5, abs=1e-6)
		assert dist.proxy_wasserstein_1d(x, y) == pytest.approx(
			stats.wasserstein_distance(np.arange(7), np.arange(7), x, y), rel=1e-4, abs=1e-6
		)
		assert dist.proxy_wasserstein_1d(x, y) == pytest.approx(dist.proxy_wasserstein_1d(y, x), rel=1e-5)


def test_proxy_wasserstein_1d_edge_cases():
	"""Pins the current output: an all-zero input gives FLOAT32_MAX; hand values otherwise."""
	x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
	z = np.zeros(3, dtype=np.float32)
	assert dist.proxy_wasserstein_1d(x, x) == pytest.approx(0.0, abs=1e-6)
	assert dist.proxy_wasserstein_1d(x, z) == FLOAT32_MAX
	assert dist.proxy_wasserstein_1d(z, x) == FLOAT32_MAX
	a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
	b = np.array([0.0, 0.0, 1.0], dtype=np.float32)
	assert dist.proxy_wasserstein_1d(a, b) == pytest.approx(2.0)


def _bc(x, y):
	px, py = _normalized(x), _normalized(y)
	return np.sum(np.sqrt(px * py))


@pytest.mark.parametrize("func", [dist.proxy_kantorovich, dist.proxy_sinkhorn])
def test_proxy_tv_hellinger_matches_reference(func):
	"""Check total variation plus squared Hellinger of the normalised inputs, computed with scipy cityblock and sqeuclidean on square roots."""
	for x, y in _positive_pairs():
		px, py = _normalized(x), _normalized(y)
		expected = 0.5 * spd.cityblock(px, py) + 0.5 * spd.sqeuclidean(np.sqrt(px), np.sqrt(py))
		assert func(x, y) == pytest.approx(expected, rel=2e-5, abs=1e-6)
		assert func(x, y) == pytest.approx(func(y, x), rel=2e-5, abs=1e-6)
	x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
	z = np.zeros(3, dtype=np.float32)
	assert func(x, x) == pytest.approx(0.0, abs=1e-5)
	assert func(x, z) == FLOAT32_MAX
	assert func(z, x) == FLOAT32_MAX
	a = np.array([1.0, 0.0], dtype=np.float32)
	b = np.array([0.0, 1.0], dtype=np.float32)
	assert func(a, b) == pytest.approx(2.0, abs=1e-6)


def test_proxy_circular_kantorovich_matches_reference():
	"""Check the proxy against the mean-centred CDF-difference formula; pins the current output of FLOAT32_MAX for an all-zero input."""
	for x, y in _positive_pairs():
		diff = np.cumsum(_normalized(x)) - np.cumsum(_normalized(y))
		expected = np.abs(diff - diff.mean()).sum()
		assert dist.proxy_circular_kantorovich(x, y) == pytest.approx(expected, rel=1e-5, abs=1e-6)
	x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
	z = np.zeros(3, dtype=np.float32)
	assert dist.proxy_circular_kantorovich(x, x) == pytest.approx(0.0, abs=1e-6)
	assert dist.proxy_circular_kantorovich(x, z) == FLOAT32_MAX
	assert dist.proxy_circular_kantorovich(z, x) == FLOAT32_MAX


def test_proxy_circular_kantorovich_is_rotation_invariant():
	"""Check proxy circular kantorovich is rotation invariant."""
	x = np.array([0.5, 0.2, 0.2, 0.1], dtype=np.float32)
	y = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
	base = dist.proxy_circular_kantorovich(x, y)
	for shift in range(1, 4):
		assert dist.proxy_circular_kantorovich(np.roll(x, shift), np.roll(y, shift)) == pytest.approx(
			base, rel=1e-5, abs=1e-6
		)


def test_proxy_jensen_shannon_matches_reference():
	"""Check 1 - BC^2 (the reference restates the formula) and the hand value 0.25 for [1, 3] versus [3, 1]."""
	assert dist.proxy_jensen_shannon(
		np.array([1.0, 3.0], dtype=np.float32), np.array([3.0, 1.0], dtype=np.float32)
	) == pytest.approx(0.25, rel=1e-5)
	for x, y in _positive_pairs():
		expected = 1.0 - _bc(x, y) ** 2
		assert dist.proxy_jensen_shannon(x, y) == pytest.approx(expected, rel=1e-5, abs=1e-6)
		assert dist.proxy_jensen_shannon(x, y) == pytest.approx(dist.proxy_jensen_shannon(y, x), rel=1e-5, abs=1e-6)
	x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
	z = np.zeros(3, dtype=np.float32)
	assert dist.proxy_jensen_shannon(x, x) == pytest.approx(0.0, abs=1e-5)
	assert dist.proxy_jensen_shannon(x, z) == FLOAT32_MAX
	a = np.array([1.0, 0.0], dtype=np.float32)
	b = np.array([0.0, 1.0], dtype=np.float32)
	assert dist.proxy_jensen_shannon(a, b) == pytest.approx(1.0)


def test_proxy_symmetric_kl_matches_triangular_discrimination():
	"""Check sum((p-q)^2/(p+q)); pins the current output of FLOAT32_MAX for an all-zero input."""
	for x, y in _positive_pairs():
		px, py = _normalized(x), _normalized(y)
		expected = np.sum((px - py) ** 2 / (px + py))
		assert dist.proxy_symmetric_kl(x, y) == pytest.approx(expected, rel=1e-5, abs=1e-6)
		assert dist.proxy_symmetric_kl(x, y) == pytest.approx(dist.proxy_symmetric_kl(y, x), rel=1e-5, abs=1e-6)
	x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
	z = np.zeros(3, dtype=np.float32)
	assert dist.proxy_symmetric_kl(x, x) == 0.0
	assert dist.proxy_symmetric_kl(x, z) == FLOAT32_MAX
	a = np.array([1.0, 0.0], dtype=np.float32)
	b = np.array([0.0, 1.0], dtype=np.float32)
	assert dist.proxy_symmetric_kl(a, b) == pytest.approx(2.0)
	assert dist.proxy_symmetric_kl(np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)) == pytest.approx(2.0)


# ------------------------------------------------------------- angular


def _tsss_ref(x, y):
	"""Restates the TS-SS formula in float64 numpy; the hand value below is independent."""
	x = x.astype(np.float64)
	y = y.astype(np.float64)
	nx, ny = np.linalg.norm(x), np.linalg.norm(y)
	theta = np.arccos(np.dot(x, y) / (nx * ny)) + np.radians(10)
	sector = ((np.linalg.norm(x - y) + abs(nx - ny)) ** 2) * theta / 2.0
	triangle = nx * ny * np.sin(theta) / 2.0
	return triangle * sector


def test_tsss_matches_reference_and_identity():
	"""Check tsss against the restated formula and against the hand value theta * cos(10 deg) / 2 for orthogonal unit vectors."""
	theta = np.pi / 2 + np.radians(10)
	assert dist.tsss(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(
		theta * np.cos(np.radians(10)) / 2.0, rel=1e-9
	)
	for x, y in _positive_pairs(dtype=np.float64):
		assert dist.tsss(x, y) == pytest.approx(_tsss_ref(x, y), rel=1e-6)
		assert dist.tsss(x, y) == pytest.approx(dist.tsss(y, x), rel=1e-6)
	x = np.array([1.0, 2.0])
	y = np.array([2.0, 0.5])
	assert dist.tsss(x, y) == pytest.approx(_tsss_ref(x, y), rel=1e-6)
	assert dist.tsss(x, x) == pytest.approx(0.0, abs=1e-6)


def test_true_angular_known_angles():
	"""Check hand angles (0, 45 and 60 degrees as fractions of pi); pins the current output of FLOAT32_MAX for angles of 90 degrees or more."""
	x = np.array([1.0, 0.0])
	assert dist.true_angular(x, np.array([2.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
	assert dist.true_angular(x, np.array([1.0, 1.0])) == pytest.approx(0.25)
	assert dist.true_angular(x, np.array([1.0, np.sqrt(3.0)])) == pytest.approx(1.0 / 3.0)
	assert dist.true_angular(x, np.array([0.0, 1.0])) == FLOAT32_MAX
	assert dist.true_angular(x, np.array([-1.0, 0.0])) == FLOAT32_MAX


def test_true_angular_matches_arccos_cosine():
	"""Check true angular matches arccos cosine."""
	for x, y in _positive_pairs(dtype=np.float64):
		expected = np.arccos(1.0 - spd.cosine(x, y)) / np.pi
		assert dist.true_angular(x, y) == pytest.approx(expected, rel=1e-6)
		assert dist.true_angular(x, y) == pytest.approx(dist.true_angular(y, x), rel=1e-9)


def test_true_angular_zero_vectors():
	"""Pins the current output: two zero vectors give 0, one zero vector gives FLOAT32_MAX."""
	z = np.zeros(3)
	x = np.array([1.0, 2.0, 3.0])
	assert dist.true_angular(z, z) == 0.0
	assert dist.true_angular(x, z) == FLOAT32_MAX
	assert dist.true_angular(z, x) == FLOAT32_MAX


def test_true_angular_from_alt_cosine_values_and_round_trip():
	"""Check true angular from alt cosine values and round trip."""
	assert dist.true_angular_from_alt_cosine(0.0) == pytest.approx(0.0, abs=1e-7)
	assert dist.true_angular_from_alt_cosine(1.0) == pytest.approx(1.0 / 3.0)
	assert dist.true_angular_from_alt_cosine(0.5) == pytest.approx(
		np.arccos(2.0 ** -0.5) / np.pi
	)
	for x, y in _positive_pairs():
		alt = dist.alternative_cosine(x, y)
		expected = dist.true_angular(x.astype(np.float64), y.astype(np.float64))
		assert dist.true_angular_from_alt_cosine(alt) == pytest.approx(expected, abs=1e-4)
	vals = np.array([0.0, 1.0, 2.0])
	assert_allclose(
		dist.true_angular_from_alt_cosine(vals), np.arccos(2.0 ** -vals) / np.pi, atol=1e-7
	)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_correlation_matches_scipy(dtype):
	"""Check correlation matches scipy."""
	for x, y in _pairs(dtype=dtype):
		assert_allclose(dist.correlation(x, y), spd.correlation(x, y), rtol=1e-4, atol=1e-6)
		assert_allclose(dist.correlation(y, x), dist.correlation(x, y), rtol=1e-6)


def test_correlation_hand_values_and_degenerate_inputs():
	"""Check hand values; pins the current output for constant vectors (0 for two equal constants, 1 against a nonconstant vector)."""
	x = np.array([1.0, 2.0, 3.0, 4.0])
	assert dist.correlation(x, x) == pytest.approx(0.0, abs=1e-12)
	assert dist.correlation(x, 2.0 * x + 5.0) == pytest.approx(0.0, abs=1e-12)
	assert dist.correlation(x, -x) == pytest.approx(2.0)
	c = np.full(4, 2.0)
	assert dist.correlation(c, c) == 0.0
	assert dist.correlation(x, c) == 1.0
	assert dist.correlation(np.array([1.0, -1.0, 1.0, -1.0]), np.array([1.0, 1.0, -1.0, -1.0])) == pytest.approx(1.0)


# --------------------------------------------------------- distributions


def test_hellinger_matches_reference():
	"""Check hellinger matches reference."""
	for x, y in _positive_pairs():
		x64, y64 = x.astype(np.float64), y.astype(np.float64)
		expected = np.sqrt(1.0 - np.sum(np.sqrt(x64 * y64)) / np.sqrt(x64.sum() * y64.sum()))
		assert dist.hellinger(x, y) == pytest.approx(expected, rel=2e-5, abs=1e-6)
		assert dist.hellinger(x, y) == pytest.approx(dist.hellinger(y, x), rel=1e-5, abs=1e-6)


def test_hellinger_edge_cases():
	"""Check hand values; pins the current output for zero vectors (0 for two zero vectors, 1 for one) and the float32 noise floor of identical inputs."""
	x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
	z = np.zeros(3, dtype=np.float32)
	assert dist.hellinger(z, z) == 0.0
	assert dist.hellinger(x, z) == 1.0
	assert dist.hellinger(z, x) == 1.0
	assert dist.hellinger(x, x) == pytest.approx(0.0, abs=1e-3)
	assert dist.hellinger(x, 4.0 * x) == pytest.approx(0.0, abs=1e-3)
	a = np.array([1.0, 0.0], dtype=np.float32)
	b = np.array([0.0, 1.0], dtype=np.float32)
	assert dist.hellinger(a, b) == pytest.approx(1.0)
	assert dist.hellinger(np.array([0.5, 0.5], dtype=np.float32), np.array([1.0, 0.0], dtype=np.float32)) == pytest.approx(
		np.sqrt(1.0 - np.sqrt(0.5)), rel=1e-5
	)


def test_alternative_hellinger_and_correction():
	"""Check the log2 ratio and its correction against hellinger; pins the current output of FLOAT32_MAX for zero and disjoint inputs."""
	for x, y in _positive_pairs():
		x64, y64 = x.astype(np.float64), y.astype(np.float64)
		ratio = np.sqrt(x64.sum() * y64.sum()) / np.sum(np.sqrt(x64 * y64))
		alt = dist.alternative_hellinger(x, y)
		assert alt == pytest.approx(np.log2(ratio), rel=1e-5, abs=1e-6)
		assert dist.correct_alternative_hellinger(alt) == pytest.approx(
			dist.hellinger(x, y), abs=1e-4
		)
	x = np.array([1.0, 2.0], dtype=np.float32)
	z = np.zeros(2, dtype=np.float32)
	assert dist.alternative_hellinger(z, z) == 0.0
	assert dist.alternative_hellinger(x, z) == FLOAT32_MAX
	assert dist.alternative_hellinger(z, x) == FLOAT32_MAX
	assert dist.alternative_hellinger(
		np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)
	) == FLOAT32_MAX
	assert dist.alternative_hellinger(x, 3.0 * x) == pytest.approx(0.0, abs=1e-3)


def test_correct_alternative_hellinger_values_and_clamp():
	"""Check correct alternative hellinger values and clamp."""
	assert dist.correct_alternative_hellinger(0.0) == 0.0
	assert dist.correct_alternative_hellinger(1.0) == pytest.approx(np.sqrt(0.5))
	assert dist.correct_alternative_hellinger(2.0) == pytest.approx(np.sqrt(0.75))
	assert dist.correct_alternative_hellinger(-0.01) == 0.0
	vals = np.array([0.0, 1.0, 3.0])
	assert_allclose(dist.correct_alternative_hellinger(vals), np.sqrt(1.0 - 2.0 ** -vals))


@pytest.mark.parametrize("method", ["average", "min", "max", "dense", "ordinal"])
def test_rankdata_methods_match_scipy_with_ties(method):
	"""Check rankdata for every tie method against scipy.stats.rankdata."""
	rng = _rng(11)
	for _ in range(5):
		a = rng.randint(0, 6, size=15).astype(np.float64)
		assert_array_equal(dist.rankdata(a, method), stats.rankdata(a, method=method))
	a2 = np.array([[3.0, 1.0], [1.0, 2.0]])
	assert_array_equal(dist.rankdata(a2, method), stats.rankdata(a2.ravel(), method=method))


def test_spearmanr_matches_scipy():
	"""Check spearmanr matches scipy."""
	rng = _rng(12)
	for _ in range(6):
		x = rng.normal(size=12)
		y = rng.normal(size=12)
		assert dist.spearmanr(x, y) == pytest.approx(1.0 - stats.spearmanr(x, y)[0], abs=1e-9)
	xt = rng.randint(0, 4, size=12).astype(np.float64)
	yt = rng.randint(0, 4, size=12).astype(np.float64)
	assert dist.spearmanr(xt, yt) == pytest.approx(1.0 - stats.spearmanr(xt, yt)[0], abs=1e-9)


def test_spearmanr_monotone_and_reversed():
	"""Check spearmanr monotone and reversed."""
	x = np.array([1.0, 5.0, 2.0, 9.0, 4.0])
	assert dist.spearmanr(x, x) == pytest.approx(0.0, abs=1e-12)
	assert dist.spearmanr(x, np.exp(x)) == pytest.approx(0.0, abs=1e-12)
	assert dist.spearmanr(x, -x) == pytest.approx(2.0)


def _positions_cost(n):
	idx = np.arange(n, dtype=np.float64)
	return np.abs(idx[:, None] - idx[None, :])


def test_kantorovich_hand_values():
	"""Check kantorovich hand values."""
	cost = np.array([[0.0, 1.0], [1.0, 0.0]])
	x = np.array([1.0, 0.0])
	y = np.array([0.0, 1.0])
	assert dist.kantorovich(x, y, cost) == pytest.approx(1.0)
	assert dist.kantorovich(x, x, cost) == pytest.approx(0.0, abs=1e-12)
	half = np.array([0.5, 0.5])
	assert dist.kantorovich(x, half, cost) == pytest.approx(0.5)
	assert dist.kantorovich(2.0 * x, 5.0 * half, cost) == pytest.approx(0.5)


def test_kantorovich_matches_scipy_wasserstein_on_a_line():
	"""Check kantorovich matches scipy wasserstein on a line."""
	rng = _rng(13)
	n = 6
	cost = _positions_cost(n)
	for _ in range(8):
		x = rng.uniform(0.05, 1.0, n)
		y = rng.uniform(0.05, 1.0, n)
		expected = stats.wasserstein_distance(np.arange(n), np.arange(n), x, y)
		assert dist.kantorovich(x, y, cost) == pytest.approx(expected, rel=1e-6, abs=1e-9)
		assert dist.kantorovich(y, x, cost) == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_kantorovich_ignores_zero_mass_bins():
	"""Check kantorovich on inputs with zero bins against scipy wasserstein_distance on the nonzero support."""
	cost = _positions_cost(5)
	x = np.array([0.5, 0.0, 0.0, 0.5, 0.0])
	y = np.array([0.0, 0.5, 0.0, 0.0, 0.5])
	expected = stats.wasserstein_distance([0, 3], [1, 4], [0.5, 0.5], [0.5, 0.5])
	assert dist.kantorovich(x, y, cost) == pytest.approx(expected)


def test_kantorovich_matches_linprog():
	"""Check kantorovich matches linprog."""
	linprog = pytest.importorskip("scipy.optimize").linprog
	rng = _rng(14)
	n = 4
	cost = rng.uniform(0.1, 3.0, size=(n, n))
	np.fill_diagonal(cost, 0.0)
	x = rng.uniform(0.1, 1.0, n)
	y = rng.uniform(0.1, 1.0, n)
	a, b = x / x.sum(), y / y.sum()
	a_eq = np.zeros((2 * n, n * n))
	for i in range(n):
		a_eq[i, i * n : (i + 1) * n] = 1.0
		a_eq[n + i, i::n] = 1.0
	res = linprog(cost.ravel(), A_eq=a_eq, b_eq=np.concatenate([a, b]), bounds=(0, None))
	assert dist.kantorovich(x, y, cost) == pytest.approx(res.fun, rel=1e-6)


def test_sinkhorn_approximates_kantorovich():
	"""Check sinkhorn lies between the exact Kantorovich cost and cost + reg*log(n*m), and matches a plain numpy Sinkhorn loop."""
	n = 5
	cost = _positions_cost(n)
	rng = _rng(15)
	for _ in range(4):
		x = rng.uniform(0.2, 1.0, n)
		y = rng.uniform(0.2, 1.0, n)
		x, y = x / x.sum(), y / y.sum()
		exact = dist.kantorovich(x, y, cost)
		approx = dist.sinkhorn(x, y, cost, 0.05)
		assert approx >= exact - 1e-4
		assert approx <= exact + 0.05 * np.log(n * n) + 1e-6
		kernel = np.exp(-cost / 0.05)
		u = np.ones(n)
		for _ in range(2000):
			v = y / (kernel.T @ u)
			u = x / (kernel @ v)
		plan = u[:, None] * kernel * v[None, :]
		assert approx == pytest.approx((plan * cost).sum(), rel=1e-3)
		coarse = dist.sinkhorn(x, y, cost, 5.0)
		assert coarse >= approx - 1e-9


def test_sinkhorn_identical_inputs_are_near_zero_for_small_regularization():
	"""Check sinkhorn identical inputs are near zero for small regularization."""
	n = 4
	cost = _positions_cost(n)
	x = np.array([0.1, 0.2, 0.3, 0.4])
	assert dist.sinkhorn(x, x, cost, 0.01) == pytest.approx(0.0, abs=0.01 * np.log(n * n))


def _js_ref(x, y):
	eps = np.finfo(np.float32).eps
	d = len(x)
	px = (x + eps) / (x.sum() + eps * d)
	py = (y + eps) / (y.sum() + eps * d)
	m = 0.5 * (px + py)
	return 0.5 * np.sum(px * np.log(px / m) + py * np.log(py / m))


def _skl_ref(x, y):
	eps = np.finfo(np.float32).eps
	d = len(x)
	px = (x + eps) / (x.sum() + eps * d)
	py = (y + eps) / (y.sum() + eps * d)
	return np.sum(px * np.log(px / py) + py * np.log(py / px))


def test_jensen_shannon_matches_reference_and_scipy():
	"""Check jensen shannon matches reference and scipy."""
	for x, y in _positive_pairs(dtype=np.float64):
		assert dist.jensen_shannon_divergence(x, y) == pytest.approx(_js_ref(x, y), rel=1e-9)
		assert dist.jensen_shannon_divergence(x, y) == pytest.approx(
			spd.jensenshannon(x, y) ** 2, rel=1e-6
		)
		assert dist.jensen_shannon_divergence(x, y) == pytest.approx(
			dist.jensen_shannon_divergence(y, x), rel=1e-9
		)


def test_jensen_shannon_edge_cases():
	"""Check jensen shannon edge cases."""
	x = np.array([1.0, 2.0, 3.0])
	assert dist.jensen_shannon_divergence(x, x) == pytest.approx(0.0, abs=1e-12)
	assert dist.jensen_shannon_divergence(x, 7.0 * x) == pytest.approx(0.0, abs=1e-6)
	a = np.array([1.0, 0.0])
	b = np.array([0.0, 1.0])
	assert dist.jensen_shannon_divergence(a, b) == pytest.approx(np.log(2.0), abs=1e-5)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_wasserstein_1d_matches_cdf_reference(p):
	"""Check wasserstein 1d matches cdf reference."""
	for x, y in _positive_pairs(dtype=np.float64):
		diff = np.cumsum(x / x.sum()) - np.cumsum(y / y.sum())
		expected = np.sum(np.abs(diff) ** p) ** (1.0 / p)
		assert dist.wasserstein_1d(x, y, p) == pytest.approx(expected, rel=1e-9)


def test_wasserstein_1d_matches_scipy_for_p1_and_does_not_mutate_inputs():
	"""Check wasserstein 1d matches scipy for p1 and does not mutate inputs."""
	for x, y in _positive_pairs(dtype=np.float64):
		x0, y0 = x.copy(), y.copy()
		expected = stats.wasserstein_distance(np.arange(7), np.arange(7), x, y)
		assert dist.wasserstein_1d(x, y) == pytest.approx(expected, rel=1e-9)
		assert_array_equal(x, x0)
		assert_array_equal(y, y0)
	assert dist.wasserstein_1d(x, x) == 0.0
	a = np.array([1.0, 0.0, 0.0])
	b = np.array([0.0, 0.0, 1.0])
	assert dist.wasserstein_1d(a, b) == pytest.approx(2.0)


def _circular_w1_ref(x, y):
	diff = np.cumsum(x / x.sum()) - np.cumsum(y / y.sum())
	return min(np.abs(diff - mu).sum() for mu in diff)


def test_circular_kantorovich_p1_matches_minimum_over_shifts():
	"""Check circular kantorovich p1 matches minimum over shifts."""
	for x, y in _positive_pairs(dtype=np.float64):
		assert dist.circular_kantorovich(x, y, 1) == pytest.approx(_circular_w1_ref(x, y), rel=1e-9)
		assert dist.circular_kantorovich(x, y) == pytest.approx(_circular_w1_ref(x, y), rel=1e-9)


def test_circular_kantorovich_p1_rotation_invariance_and_hand_value():
	"""Check circular kantorovich p1 rotation invariance and hand value."""
	x = np.array([0.4, 0.3, 0.2, 0.1])
	y = np.array([0.1, 0.2, 0.3, 0.4])
	base = dist.circular_kantorovich(x, y, 1)
	for shift in range(1, 4):
		assert dist.circular_kantorovich(np.roll(x, shift), np.roll(y, shift), 1) == pytest.approx(base, rel=1e-9)
	# mass at bin 0 versus bin 3 of 4 are neighbours on the circle
	a = np.array([1.0, 0.0, 0.0, 0.0])
	b = np.array([0.0, 0.0, 0.0, 1.0])
	assert dist.circular_kantorovich(a, b, 1) == pytest.approx(1.0)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_circular_kantorovich_identical_inputs_are_zero(p):
	"""Check circular kantorovich identical inputs are zero."""
	x = np.array([0.1, 0.2, 0.3, 0.4])
	assert dist.circular_kantorovich(x, x, p) == 0.0


def test_circular_kantorovich_invalid_p_raises():
	"""Check circular kantorovich invalid p raises."""
	x = np.array([0.1, 0.2, 0.3, 0.4])
	y = np.array([0.4, 0.3, 0.2, 0.1])
	with pytest.raises(ValueError):
		dist.circular_kantorovich(x, y, 0)


def test_symmetric_kl_matches_reference():
	"""Check symmetric kl matches reference."""
	for x, y in _positive_pairs(dtype=np.float64):
		assert dist.symmetric_kl_divergence(x, y) == pytest.approx(_skl_ref(x, y), rel=1e-9)
		assert dist.symmetric_kl_divergence(x, y) == pytest.approx(
			dist.symmetric_kl_divergence(y, x), rel=1e-9
		)
		assert dist.symmetric_kl_divergence(x, y) >= 0.0
		px, py = x / x.sum(), y / y.sum()
		assert dist.symmetric_kl_divergence(x, y) == pytest.approx(
			stats.entropy(px, py) + stats.entropy(py, px), rel=1e-5
		)


def test_symmetric_kl_identity_and_scale_invariance():
	"""Check symmetric kl identity and scale invariance."""
	x = np.array([1.0, 2.0, 3.0])
	assert dist.symmetric_kl_divergence(x, x) == pytest.approx(0.0, abs=1e-12)
	assert dist.symmetric_kl_divergence(x, 5.0 * x) == pytest.approx(0.0, abs=1e-6)
