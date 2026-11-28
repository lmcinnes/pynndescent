"""Tests for the bit-packed and quantized distance functions and the lookup
tables in pynndescent.distances.

Expected values come from short numpy reference implementations that unpack
or dequantize the codes and apply the documented formula.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial import distance as spd

import pynndescent.distances as dist

FLOAT32_MAX = np.finfo(np.float32).max


def _rng(seed=0):
	return np.random.RandomState(seed)


def _packed_pairs(n=12, nbytes=5, seed=0):
	rng = _rng(seed)
	data = rng.randint(0, 256, size=(n, nbytes)).astype(np.uint8)
	return [(data[i], data[(i + 1) % n]) for i in range(n)]


# ---------------------------------------------------------- bit-packed


def test_bit_hamming_matches_unpacked_hamming_count():
	"""Check the popcount Hamming distance against the count of differing bits from np.unpackbits and scipy hamming."""
	for x, y in _packed_pairs():
		xb, yb = np.unpackbits(x), np.unpackbits(y)
		result = dist.bit_hamming(x, y)
		assert result == round(spd.hamming(xb, yb) * xb.size)
		assert result == float(np.sum(xb != yb))
		assert dist.bit_hamming(y, x) == result
	assert dist.bit_hamming(x, x) == 0.0


def test_bit_hamming_hand_values_and_dtype():
	"""Check bit hamming hand values and dtype."""
	x = np.array([0b11110000, 0b00000000], dtype=np.uint8)
	y = np.array([0b00001111, 0b10000001], dtype=np.uint8)
	assert dist.bit_hamming(x, y) == 10.0
	full = np.full(3, 255, dtype=np.uint8)
	zero = np.zeros(3, dtype=np.uint8)
	assert dist.bit_hamming(full, zero) == 24.0


def test_bit_jaccard_matches_unpacked_alternative_jaccard():
	"""Check bit jaccard matches unpacked alternative jaccard."""
	for x, y in _packed_pairs(seed=1):
		xb, yb = np.unpackbits(x).astype(bool), np.unpackbits(y).astype(bool)
		inter = np.sum(xb & yb)
		union = np.sum(xb | yb)
		result = dist.bit_jaccard(x, y)
		assert result == pytest.approx(-np.log2(inter / union), rel=1e-5, abs=1e-6)
		assert dist.bit_jaccard(y, x) == result
		assert dist.correct_alternative_jaccard(result) == pytest.approx(
			spd.jaccard(xb, yb), abs=1e-5
		)


def test_bit_jaccard_identity_zero_and_hand_value():
	"""Check bit jaccard identity zero and hand value."""
	x = np.array([0b10101010, 0b11110000], dtype=np.uint8)
	assert dist.bit_jaccard(x, x) == 0.0
	z = np.zeros(2, dtype=np.uint8)
	assert dist.bit_jaccard(z, z) == 0.0
	# 12 bits in the union, 4 bits shared
	y = np.array([0b10101010, 0b00001111], dtype=np.uint8)
	assert dist.bit_jaccard(x, y) == pytest.approx(
		-np.log2(np.sum(np.unpackbits(x & y)) / np.sum(np.unpackbits(x | y))), rel=1e-6
	)
	a = np.array([0b11110000], dtype=np.uint8)
	b = np.array([0b00111100], dtype=np.uint8)
	assert dist.bit_jaccard(a, b) == pytest.approx(np.log2(3.0), rel=1e-6)


# ------------------------------------------------------ uint8 codebook


def _codebook(seed=2, n=256, positive=False):
	rng = _rng(seed)
	vals = rng.uniform(0.0 if positive else -1.0, 1.0, size=n).astype(np.float32)
	return vals


def _uint8_data(seed=3, dim=9, positive=False):
	rng = _rng(seed)
	q = _codebook(positive=positive)
	x = rng.uniform(0.05 if positive else -1.0, 1.0, size=dim).astype(np.float32)
	y = rng.randint(0, 256, size=dim).astype(np.uint8)
	return x, y, q


def test_quantized_uint8_sq_euclidean_matches_dequantized():
	"""Check quantized uint8 sq euclidean matches dequantized."""
	for seed in range(6):
		x, y, q = _uint8_data(seed)
		expected = spd.sqeuclidean(x.astype(np.float64), q[y].astype(np.float64))
		result = dist.quantized_uint8_sq_euclidean(x, y, q)
		assert result == pytest.approx(expected, rel=1e-4, abs=1e-6)


def test_quantized_uint8_sq_euclidean_exact_codes_give_zero():
	"""Check quantized uint8 sq euclidean exact codes give zero."""
	q = np.linspace(-1.0, 1.0, 256).astype(np.float32)
	y = np.array([0, 17, 255, 128], dtype=np.uint8)
	assert dist.quantized_uint8_sq_euclidean(q[y], y, q) == 0.0
	x = np.zeros(4, dtype=np.float32)
	q2 = np.arange(256, dtype=np.float32)
	assert dist.quantized_uint8_sq_euclidean(x, y, q2) == pytest.approx(
		float(np.sum(y.astype(np.float64) ** 2))
	)


def _alt_cosine_quantized_ref(x, qy):
	"""Restates the quantized formula -log2((cos + 1) / 2), which differs from alternative_cosine's -log2(cos)."""
	x = x.astype(np.float64)
	qy = qy.astype(np.float64)
	nx, ny = np.dot(x, x), np.dot(qy, qy)
	ip = np.dot(x, qy)
	if nx == 0 and ny == 0:
		return 0.0
	if nx == 0 or ny == 0 or ip <= 0:
		return FLOAT32_MAX
	return -np.log2((ip / np.sqrt(nx * ny) + 1.0) / 2.0)


def test_quantized_uint8_alternative_cosine_matches_reference():
	"""Check against the restated quantized formula on dequantized codes; the reference restates the code's formula."""
	seen_finite = 0
	for seed in range(10):
		x, y, q = _uint8_data(seed, positive=True)
		expected = _alt_cosine_quantized_ref(x, q[y])
		result = dist.quantized_uint8_alternative_cosine(x, y, q)
		assert result == pytest.approx(expected, rel=2e-5, abs=1e-6)
		seen_finite += expected != FLOAT32_MAX
	assert seen_finite > 0


def test_quantized_uint8_alternative_cosine_edge_cases():
	"""Pins the current output: zero, negative and mismatched-zero inputs give 0 or FLOAT32_MAX; hand value 0 for parallel vectors."""
	q = np.linspace(-1.0, 1.0, 256).astype(np.float32)
	x = np.array([1.0, 0.0], dtype=np.float32)
	zero_code = np.array([127, 127], dtype=np.uint8)  # q[127] is close to but not exactly 0
	q_zero = np.zeros(256, dtype=np.float32)
	z = np.zeros(2, dtype=np.float32)
	assert dist.quantized_uint8_alternative_cosine(z, zero_code, q_zero) == 0.0
	assert dist.quantized_uint8_alternative_cosine(x, zero_code, q_zero) == FLOAT32_MAX
	assert dist.quantized_uint8_alternative_cosine(z, zero_code, q) == FLOAT32_MAX
	neg = np.array([0, 0], dtype=np.uint8)
	assert dist.quantized_uint8_alternative_cosine(np.ones(2, dtype=np.float32), neg, q) == FLOAT32_MAX
	# parallel vectors: cosine 1 -> -log2(1) = 0
	q_one = np.ones(256, dtype=np.float32)
	same = np.array([3, 3], dtype=np.uint8)
	assert dist.quantized_uint8_alternative_cosine(
		np.ones(2, dtype=np.float32), same, q_one
	) == pytest.approx(0.0, abs=1e-6)


def test_quantized_uint8_alternative_dot_matches_reference():
	"""Check -log2(ip / |y|) on dequantized codes; pins the current output of FLOAT32_MAX for a non-positive inner product."""
	for seed in range(10):
		x, y, q = _uint8_data(seed, positive=True)
		qy = q[y].astype(np.float64)
		ip = np.dot(x.astype(np.float64), qy)
		expected = FLOAT32_MAX if ip <= 0 else -np.log2(ip / np.sqrt(np.dot(qy, qy)))
		result = dist.quantized_uint8_alternative_dot(x, y, q)
		assert result == pytest.approx(expected, rel=2e-5, abs=1e-6)


def test_quantized_uint8_alternative_dot_hand_values():
	"""Check hand values of -log2(ip / |y|); pins the current output of FLOAT32_MAX for a non-positive inner product."""
	q = np.linspace(-1.0, 1.0, 256).astype(np.float32)
	x = np.array([1.0, 0.0], dtype=np.float32)
	q_unit = np.zeros(256, dtype=np.float32)
	q_unit[1] = 1.0
	y = np.array([1, 0], dtype=np.uint8)
	assert dist.quantized_uint8_alternative_dot(x, y, q_unit) == pytest.approx(0.0, abs=1e-6)
	q_half = np.zeros(256, dtype=np.float32)
	q_half[1] = 0.5
	assert dist.quantized_uint8_alternative_dot(x, y, q_half) == pytest.approx(0.0, abs=1e-6)
	q_neg = -np.ones(256, dtype=np.float32)
	assert dist.quantized_uint8_alternative_dot(x, np.array([0, 0], dtype=np.uint8), q_neg) == FLOAT32_MAX


# ------------------------------------------------------- uint4 codebook


def _unpack_nibbles(y, dim):
	out = np.empty(dim, dtype=np.int64)
	for i in range(dim):
		byte = int(y[i // 2])
		out[i] = (byte & 0x0F) if i % 2 == 0 else (byte >> 4) & 0x0F
	return out


def _uint4_data(seed, dim, positive=False):
	rng = _rng(seed)
	q = rng.uniform(0.05 if positive else -1.0, 1.0, size=16).astype(np.float32)
	x = rng.uniform(0.05 if positive else -1.0, 1.0, size=dim).astype(np.float32)
	y = rng.randint(0, 256, size=(dim + 1) // 2).astype(np.uint8)
	return x, y, q


@pytest.mark.parametrize("dim", [8, 9, 1])
def test_quantized_uint4_sq_euclidean_matches_dequantized(dim):
	"""Check quantized uint4 sq euclidean matches dequantized."""
	for seed in range(5):
		x, y, q = _uint4_data(seed, dim)
		codes = _unpack_nibbles(y, dim)
		expected = np.sum((x.astype(np.float64) - q[codes].astype(np.float64)) ** 2)
		result = dist.quantized_uint4_sq_euclidean(x, y, q)
		assert result == pytest.approx(expected, rel=1e-4, abs=1e-6)


def test_quantized_uint4_nibble_order_low_nibble_is_even_index():
	"""Check quantized uint4 nibble order low nibble is even index."""
	q = np.arange(16, dtype=np.float32)
	y = np.array([0x21], dtype=np.uint8)  # low nibble 1, high nibble 2
	x = np.array([1.0, 2.0], dtype=np.float32)
	assert dist.quantized_uint4_sq_euclidean(x, y, q) == 0.0
	x_swapped = np.array([2.0, 1.0], dtype=np.float32)
	assert dist.quantized_uint4_sq_euclidean(x_swapped, y, q) == pytest.approx(2.0)


@pytest.mark.parametrize("dim", [8, 9])
def test_quantized_uint4_alternative_cosine_matches_reference(dim):
	"""Check against the restated quantized formula on dequantized nibbles; the reference restates the code's formula."""
	for seed in range(8):
		x, y, q = _uint4_data(seed, dim, positive=True)
		qy = q[_unpack_nibbles(y, dim)]
		expected = _alt_cosine_quantized_ref(x, qy)
		result = dist.quantized_uint4_alternative_cosine(x, y, q)
		assert result == pytest.approx(expected, rel=2e-5, abs=1e-6)


def test_quantized_uint4_alternative_cosine_edge_cases():
	"""Pins the current output: zero and negative codebooks give 0 or FLOAT32_MAX; hand value 0 for parallel vectors."""
	q_zero = np.zeros(16, dtype=np.float32)
	y = np.array([0x11, 0x11], dtype=np.uint8)
	z = np.zeros(4, dtype=np.float32)
	x = np.ones(4, dtype=np.float32)
	assert dist.quantized_uint4_alternative_cosine(z, y, q_zero) == 0.0
	assert dist.quantized_uint4_alternative_cosine(x, y, q_zero) == FLOAT32_MAX
	q_one = np.ones(16, dtype=np.float32)
	assert dist.quantized_uint4_alternative_cosine(z, y, q_one) == FLOAT32_MAX
	assert dist.quantized_uint4_alternative_cosine(x, y, q_one) == pytest.approx(0.0, abs=1e-6)
	assert dist.quantized_uint4_alternative_cosine(x, y, -q_one) == FLOAT32_MAX


@pytest.mark.parametrize("dim", [8, 9])
def test_quantized_uint4_alternative_dot_matches_reference(dim):
	"""Check -log2(ip / |y|) on dequantized codes; pins the current output of FLOAT32_MAX for a non-positive inner product."""
	for seed in range(8):
		x, y, q = _uint4_data(seed, dim, positive=True)
		qy = q[_unpack_nibbles(y, dim)].astype(np.float64)
		ip = np.dot(x.astype(np.float64), qy)
		expected = FLOAT32_MAX if ip <= 0 else -np.log2(ip / np.sqrt(np.dot(qy, qy)))
		result = dist.quantized_uint4_alternative_dot(x, y, q)
		assert result == pytest.approx(expected, rel=2e-5, abs=1e-6)
	q_neg = -np.ones(16, dtype=np.float32)
	assert dist.quantized_uint4_alternative_dot(
		np.ones(2, dtype=np.float32), np.array([0x00], dtype=np.uint8), q_neg
	) == FLOAT32_MAX


# -------------------------------------------------------- int8 codes


def _int8_pairs(seed=4, n=10, dim=12, positive_bias=True):
	rng = _rng(seed)
	data = rng.randint(-127, 128, size=(n, dim)).astype(np.int8)
	if positive_bias:
		data = np.abs(data) * np.where(rng.uniform(size=(n, dim)) < 0.8, 1, -1).astype(np.int8)
		data = data.astype(np.int8)
	return [(np.ascontiguousarray(data[i]), np.ascontiguousarray(data[(i + 1) % n])) for i in range(n)]


def _int8_stats(x, y):
	xi, yi = x.astype(np.int64), y.astype(np.int64)
	return float(np.dot(xi, yi)), float(np.dot(xi, xi)), float(np.dot(yi, yi))


def _int8_similarity(x, y):
	dot, nx, ny = _int8_stats(x, y)
	return dot / np.sqrt(nx * ny)


def test_quantized_int8_alternative_cosine_matches_reference():
	"""Check -log2((cos + 1) / 2) against an int64 reference; pins the current output of FLOAT32_MAX for non-positive dots."""
	for x, y in _int8_pairs():
		dot, nx, ny = _int8_stats(x, y)
		result = dist.quantized_int8_alternative_cosine(x, y)
		if dot <= 0:
			assert result == FLOAT32_MAX
		else:
			sim = min(1.0, _int8_similarity(x, y))
			assert result == pytest.approx(-np.log2((sim + 1.0) / 2.0), rel=2e-5, abs=1e-6)
		assert dist.quantized_int8_alternative_cosine(y, x) == result


def test_quantized_int8_alternative_cosine_edge_cases():
	"""Pins the current output: zero vectors give 0 or FLOAT32_MAX, non-positive dots give FLOAT32_MAX."""
	x = np.array([10, -20, 30], dtype=np.int8)
	z = np.zeros(3, dtype=np.int8)
	assert dist.quantized_int8_alternative_cosine(z, z) == 0.0
	assert dist.quantized_int8_alternative_cosine(x, z) == FLOAT32_MAX
	assert dist.quantized_int8_alternative_cosine(z, x) == FLOAT32_MAX
	assert dist.quantized_int8_alternative_cosine(x, -x) == FLOAT32_MAX
	assert dist.quantized_int8_alternative_cosine(x, x) == 0.0
	orth = np.array([1, 0], dtype=np.int8)
	assert dist.quantized_int8_alternative_cosine(orth, np.array([0, 1], dtype=np.int8)) == FLOAT32_MAX


def test_quantized_int8_extreme_codes_do_not_overflow():
	"""Check quantized int8 extreme codes do not overflow."""
	x = np.full(1000, 127, dtype=np.int8)
	assert dist.quantized_int8_alternative_cosine(x, x) == pytest.approx(0.0, abs=1e-6)
	assert dist.quantized_int8_direct_cosine(x, x) == pytest.approx(0.0, abs=1e-6)
	assert dist.quantized_int8_alternative_dot(x, x) == pytest.approx(0.0, abs=1e-6)
	y = np.full(1000, -128, dtype=np.int8)
	assert dist.quantized_int8_alternative_cosine(y, y) == pytest.approx(0.0, abs=1e-6)


def test_quantized_int8_alternative_dot_matches_reference():
	"""Check -log2(cos) against an int64 reference; pins the current output of FLOAT32_MAX for non-positive dots and zero vectors."""
	for x, y in _int8_pairs(seed=5):
		dot, nx, ny = _int8_stats(x, y)
		result = dist.quantized_int8_alternative_dot(x, y)
		if dot <= 0:
			assert result == FLOAT32_MAX
		else:
			sim = min(1.0, _int8_similarity(x, y))
			assert result == pytest.approx(-np.log2(sim), rel=2e-5, abs=1e-6)
	x = np.array([10, -20, 30], dtype=np.int8)
	z = np.zeros(3, dtype=np.int8)
	assert dist.quantized_int8_alternative_dot(x, z) == FLOAT32_MAX
	assert dist.quantized_int8_alternative_dot(z, z) == FLOAT32_MAX
	assert dist.quantized_int8_alternative_dot(x, -x) == FLOAT32_MAX
	assert dist.quantized_int8_alternative_dot(x, x) == 0.0


def test_quantized_int8_direct_cosine_matches_reference():
	"""Check 1 - cos against an int64 reference; pins the current output of 1.0 for non-positive dots and zero vectors."""
	for x, y in _int8_pairs(seed=6):
		dot, nx, ny = _int8_stats(x, y)
		result = dist.quantized_int8_direct_cosine(x, y)
		if dot <= 0:
			assert result == 1.0
		else:
			assert result == pytest.approx(1.0 - min(1.0, _int8_similarity(x, y)), abs=1e-5)
	x = np.array([10, -20, 30], dtype=np.int8)
	z = np.zeros(3, dtype=np.int8)
	assert dist.quantized_int8_direct_cosine(x, z) == 1.0
	assert dist.quantized_int8_direct_cosine(z, z) == 1.0
	assert dist.quantized_int8_direct_cosine(x, -x) == 1.0
	assert dist.quantized_int8_direct_cosine(x, x) == pytest.approx(0.0, abs=1e-6)


def test_quantized_int8_distances_accept_readonly_arrays():
	"""Check quantized int8 distances accept readonly arrays."""
	x, y = _int8_pairs(seed=7)[0]
	expected = dist.quantized_int8_alternative_cosine(x, y)
	xr, yr = x.copy(), y.copy()
	xr.setflags(write=False)
	yr.setflags(write=False)
	assert dist.quantized_int8_alternative_cosine(xr, yr) == expected
	assert dist.quantized_int8_direct_cosine(xr, yr) == dist.quantized_int8_direct_cosine(x, y)
	assert dist.quantized_int8_alternative_dot(xr, yr) == dist.quantized_int8_alternative_dot(x, y)


def _inv_norms(x, y):
	return (
		np.float32(1.0 / np.sqrt(np.dot(x.astype(np.float64), x.astype(np.float64)))),
		np.float32(1.0 / np.sqrt(np.dot(y.astype(np.float64), y.astype(np.float64)))),
	)


def test_quantized_int8_alternative_cosine_with_norms_matches_reference():
	"""Check the with-norms variant against the reference formula and the plain variant; pins the current output of FLOAT32_MAX for non-positive dots."""
	for x, y in _int8_pairs(seed=8):
		inx, iny = _inv_norms(x, y)
		dot = _int8_stats(x, y)[0]
		result = dist.quantized_int8_alternative_cosine_with_norms(x, y, inx, iny)
		if dot <= 0:
			assert result == FLOAT32_MAX
		else:
			sim = min(1.0, dot * float(inx) * float(iny))
			assert result == pytest.approx(-np.log2((sim + 1.0) / 2.0), rel=2e-5, abs=1e-6)
			assert result == pytest.approx(
				dist.quantized_int8_alternative_cosine(x, y), rel=1e-4, abs=1e-5
			)


def test_quantized_int8_direct_cosine_with_norms_matches_reference():
	"""Check the with-norms variant against the reference formula and the plain variant; pins the current output of 1.0 for non-positive dots."""
	for x, y in _int8_pairs(seed=9):
		inx, iny = _inv_norms(x, y)
		dot = _int8_stats(x, y)[0]
		result = dist.quantized_int8_direct_cosine_with_norms(x, y, inx, iny)
		if dot <= 0:
			assert result == 1.0
		else:
			sim = min(1.0, dot * float(inx) * float(iny))
			assert result == pytest.approx(1.0 - sim, abs=1e-5)
			assert result == pytest.approx(dist.quantized_int8_direct_cosine(x, y), abs=1e-5)


def test_quantized_int8_alternative_dot_with_norms_matches_reference():
	"""Check the with-norms variant against the reference formula and the plain variant; pins the current output of FLOAT32_MAX for non-positive dots."""
	for x, y in _int8_pairs(seed=10):
		inx, iny = _inv_norms(x, y)
		dot = _int8_stats(x, y)[0]
		result = dist.quantized_int8_alternative_dot_with_norms(x, y, inx, iny)
		if dot <= 0:
			assert result == FLOAT32_MAX
		else:
			sim = min(1.0, dot * float(inx) * float(iny))
			assert result == pytest.approx(-np.log2(sim), rel=2e-5, abs=1e-6)
			assert result == pytest.approx(
				dist.quantized_int8_alternative_dot(x, y), rel=1e-4, abs=1e-5
			)


def test_quantized_int8_with_norms_scale_by_given_inverse_norms():
	"""Check quantized int8 with norms scale by given inverse norms."""
	x = np.array([3, 4], dtype=np.int8)
	y = np.array([3, 4], dtype=np.int8)
	one = np.float32(1.0 / 5.0)
	assert dist.quantized_int8_direct_cosine_with_norms(x, y, one, one) == pytest.approx(0.0, abs=1e-6)
	assert dist.quantized_int8_alternative_dot_with_norms(x, y, one, one) == pytest.approx(0.0, abs=1e-6)
	assert dist.quantized_int8_alternative_cosine_with_norms(x, y, one, one) == pytest.approx(0.0, abs=1e-6)
	half = np.float32(0.1)
	# similarity 25 * 0.1 * 0.1 = 0.25
	assert dist.quantized_int8_direct_cosine_with_norms(x, y, half, half) == pytest.approx(0.75)
	assert dist.quantized_int8_alternative_dot_with_norms(x, y, half, half) == pytest.approx(2.0)
	assert dist.quantized_int8_alternative_cosine_with_norms(x, y, half, half) == pytest.approx(
		-np.log2(0.625), rel=1e-6
	)


# ----------------------------------------------------- lookup tables


def test_fast_distance_alternatives_correct_back_to_true_distance():
	"""Check each fast alternative, once corrected, reproduces the true named distance on random inputs."""
	rng = _rng(20)
	data = np.abs(rng.normal(size=(6, 8))).astype(np.float32) + 0.1
	unit = (data / np.linalg.norm(data, axis=1, keepdims=True)).astype(np.float32)
	inputs = {"dot": unit, "inner_product": unit}
	for name, entry in dist.fast_distance_alternatives.items():
		if name in ("jaccard", "bit_jaccard"):
			continue
		d = inputs.get(name, data)
		true_fn = dist.named_distances[name]
		for i in range(5):
			x, y = d[i], d[i + 1]
			alt = entry["dist"](x, y)
			corrected = entry["correction"](alt)
			if name == "inner_product":
				expected = true_fn(x, y)
				assert corrected == pytest.approx(expected, rel=2e-5, abs=1e-6)
			else:
				expected = true_fn(x.astype(np.float64), y.astype(np.float64))
				assert corrected == pytest.approx(expected, rel=1e-4, abs=1e-5)


def test_fast_distance_alternatives_jaccard_entries():
	"""Check fast distance alternatives jaccard entries."""
	rng = _rng(21)
	x = (rng.uniform(size=16) < 0.6).astype(np.float32)
	y = (rng.uniform(size=16) < 0.6).astype(np.float32)
	entry = dist.fast_distance_alternatives["jaccard"]
	assert entry["correction"](entry["dist"](x, y)) == pytest.approx(
		dist.jaccard(x, y), abs=1e-5
	)
	xp, yp = np.packbits(x.astype(np.uint8)), np.packbits(y.astype(np.uint8))
	entry = dist.fast_distance_alternatives["bit_jaccard"]
	assert entry["correction"](entry["dist"](xp, yp)) == pytest.approx(
		dist.jaccard(x, y), abs=1e-5
	)


def test_fast_distance_alternatives_keys_are_named_distances():
	"""Check fast distance alternatives keys are named distances."""
	for name in dist.fast_distance_alternatives:
		assert name in dist.named_distances
		assert set(dist.fast_distance_alternatives[name]) == {"dist", "correction"}


def test_proxy_distances_entries_pair_proxy_with_true_distance():
	"""Check proxy distances entries pair proxy with true distance."""
	for name, entry in dist.proxy_distances.items():
		assert name.startswith("proxy_")
		assert set(entry) == {"proxy_dist", "true_dist"}
	assert dist.proxy_distances["proxy_wasserstein_1d"]["true_dist"] is dist.wasserstein_1d
	assert dist.proxy_distances["proxy_kantorovich"]["proxy_dist"] is dist.proxy_kantorovich
	assert dist.proxy_distances["proxy_sinkhorn"]["true_dist"] is dist.sinkhorn
	assert dist.proxy_distances["proxy_inner_product"]["true_dist"] is dist.inner_product


def test_named_distance_aliases_share_functions():
	"""Check named distance aliases share functions."""
	nd = dist.named_distances
	assert nd["euclidean"] is nd["l2"]
	assert nd["manhattan"] is nd["l1"] is nd["taxicab"]
	assert nd["chebyshev"] is nd["linf"] is nd["linfty"] is nd["linfinity"]
	assert nd["wasserstein"] is nd["kantorovich"]
	assert nd["sokalmichener"] is dist.sokal_michener
	assert nd["braycurtis"] is dist.bray_curtis


def test_quantized_distances_table_maps_to_documented_functions():
	"""Check quantized distances table maps to documented functions."""
	qd = dist.quantized_distances
	assert qd["binary"]["hamming"] is dist.bit_hamming
	assert qd["binary"]["cosine"] is dist.bit_jaccard
	assert qd["uint8"]["euclidean"] is dist.quantized_uint8_sq_euclidean
	assert qd["uint4"]["dot"] is dist.quantized_uint4_alternative_dot
	assert qd["int8"]["cosine"] is dist.quantized_int8_alternative_cosine
	assert qd["int8"]["direct_cosine"] is dist.quantized_int8_direct_cosine
