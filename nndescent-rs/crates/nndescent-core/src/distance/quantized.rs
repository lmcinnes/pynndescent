//! Quantized distance functions.
//!
//! These compute distances between a full-precision float query vector and
//! a quantized (uint8 or uint4) database vector, using a lookup table to
//! dequantize on the fly.
//!
//! Three distance types are supported for each quantization level:
//!   - Squared Euclidean
//!   - Alternative Cosine (log-transformed)
//!   - Alternative Dot (log-transformed)
//!
//! AVX2 SIMD is used where available, processing 8 floats at a time by
//! gathering dequantized values through the lookup table.

const FLOAT32_MAX: f32 = f32::MAX;

// ────────────────────────────────────────────────────────────────
//  uint8 quantized distances
// ────────────────────────────────────────────────────────────────

/// Squared Euclidean between float query `x` and quantized uint8 `y`.
///
/// `codebook[y[i]]` gives the dequantized float value for dimension i.
#[inline]
pub fn quantized_u8_sq_euclidean(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { quantized_u8_sq_euclidean_avx2(x, y, codebook) };
        }
    }

    let mut result = 0.0f32;
    for i in 0..x.len() {
        let yi = codebook[y[i] as usize];
        let diff = x[i] - yi;
        result += diff * diff;
    }
    result
}

/// Alternative cosine between float query `x` and quantized uint8 `y`.
///
/// Returns log₂((‖x‖·‖y‖) / (x·y)) mapped through (sim+1)/2 to keep
/// values non-negative even for negative cosine similarities.
#[inline]
pub fn quantized_u8_alternative_cosine(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { quantized_u8_alt_cosine_avx2(x, y, codebook) };
        }
    }

    let mut dot = 0.0f32;
    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;
    for i in 0..x.len() {
        let qy = codebook[y[i] as usize];
        dot += x[i] * qy;
        norm_x += x[i] * x[i];
        norm_y += qy * qy;
    }

    if norm_x == 0.0 && norm_y == 0.0 {
        return 0.0;
    } else if norm_x == 0.0 || norm_y == 0.0 {
        return FLOAT32_MAX;
    } else if dot <= 0.0 {
        return FLOAT32_MAX;
    }

    let sim = dot / (norm_x * norm_y).sqrt();
    -((sim + 1.0) / 2.0).log2()
}

/// Alternative dot between float query `x` and quantized uint8 `y`.
///
/// y is assumed to be from a normalized dataset.
/// Returns -log₂(x·y / ‖y‖).
#[inline]
pub fn quantized_u8_alternative_dot(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let mut dot = 0.0f32;
    let mut norm_y = 0.0f32;
    for i in 0..x.len() {
        let qy = codebook[y[i] as usize];
        dot += x[i] * qy;
        norm_y += qy * qy;
    }

    if dot <= 0.0 {
        FLOAT32_MAX
    } else {
        -(dot / norm_y.sqrt()).log2()
    }
}

// ────────────────────────────────────────────────────────────────
//  uint4 quantized distances (nibble-packed: 2 values per byte)
// ────────────────────────────────────────────────────────────────

/// Extract the float value for dimension `i` from a nibble-packed byte array.
#[inline(always)]
fn dequant_u4(y: &[u8], i: usize, codebook: &[f32]) -> f32 {
    let byte = y[i / 2];
    let idx = if i % 2 == 0 {
        byte & 0x0F // lower nibble
    } else {
        (byte >> 4) & 0x0F // upper nibble
    };
    codebook[idx as usize]
}

/// Squared Euclidean between float query `x` and quantized uint4 `y`.
#[inline]
pub fn quantized_u4_sq_euclidean(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        let diff = x[i] - dequant_u4(y, i, codebook);
        result += diff * diff;
    }
    result
}

/// Alternative cosine between float query `x` and quantized uint4 `y`.
#[inline]
pub fn quantized_u4_alternative_cosine(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;
    for i in 0..x.len() {
        let qy = dequant_u4(y, i, codebook);
        dot += x[i] * qy;
        norm_x += x[i] * x[i];
        norm_y += qy * qy;
    }

    if norm_x == 0.0 && norm_y == 0.0 {
        return 0.0;
    } else if norm_x == 0.0 || norm_y == 0.0 {
        return FLOAT32_MAX;
    } else if dot <= 0.0 {
        return FLOAT32_MAX;
    }

    let sim = dot / (norm_x * norm_y).sqrt();
    -((sim + 1.0) / 2.0).log2()
}

/// Alternative dot between float query `x` and quantized uint4 `y`.
#[inline]
pub fn quantized_u4_alternative_dot(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_y = 0.0f32;
    for i in 0..x.len() {
        let qy = dequant_u4(y, i, codebook);
        dot += x[i] * qy;
        norm_y += qy * qy;
    }

    if dot <= 0.0 {
        FLOAT32_MAX
    } else {
        -(dot / norm_y.sqrt()).log2()
    }
}

// ────────────────────────────────────────────────────────────────
//  AVX2+FMA SIMD implementations for uint8 quantized distances
// ────────────────────────────────────────────────────────────────

/// AVX2+FMA squared Euclidean for uint8 quantized vectors.
///
/// Strategy: process 8 dimensions at a time.
/// For each group of 8:
///   1. Gather 8 codebook entries via `_mm256_i32gather_ps` using u8 indices.
///   2. Subtract query floats, FMA to accumulate squared differences.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn quantized_u8_sq_euclidean_avx2(
    x: &[f32],
    y: &[u8],
    codebook: &[f32],
) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();
    let chunks = n / 8;
    let mut vsum = _mm256_setzero_ps();

    for c in 0..chunks {
        let idx = c * 8;

        // Load 8 indices and widen to i32 for gather
        let indices = _mm256_set_epi32(
            y[idx + 7] as i32,
            y[idx + 6] as i32,
            y[idx + 5] as i32,
            y[idx + 4] as i32,
            y[idx + 3] as i32,
            y[idx + 2] as i32,
            y[idx + 1] as i32,
            y[idx] as i32,
        );

        // Gather dequantized values from codebook
        let vy = _mm256_i32gather_ps::<4>(codebook.as_ptr(), indices);

        // Load query values
        let vx = _mm256_loadu_ps(x.as_ptr().add(idx));

        // diff = x - y_dequant
        let diff = _mm256_sub_ps(vx, vy);
        // accumulate diff²
        vsum = _mm256_fmadd_ps(diff, diff, vsum);
    }

    let mut result = hsum256_ps(vsum);

    // Scalar remainder
    let start = chunks * 8;
    for i in start..n {
        let yi = codebook[y[i] as usize];
        let diff = x[i] - yi;
        result += diff * diff;
    }

    result
}

/// AVX2+FMA alternative cosine for uint8 quantized vectors.
///
/// Gathers dequantized values and computes dot, norm_x, norm_y in one pass.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn quantized_u8_alt_cosine_avx2(
    x: &[f32],
    y: &[u8],
    codebook: &[f32],
) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();
    let chunks = n / 8;
    let mut vdot = _mm256_setzero_ps();
    let mut vnorm_x = _mm256_setzero_ps();
    let mut vnorm_y = _mm256_setzero_ps();

    for c in 0..chunks {
        let idx = c * 8;

        let indices = _mm256_set_epi32(
            y[idx + 7] as i32,
            y[idx + 6] as i32,
            y[idx + 5] as i32,
            y[idx + 4] as i32,
            y[idx + 3] as i32,
            y[idx + 2] as i32,
            y[idx + 1] as i32,
            y[idx] as i32,
        );

        let vy = _mm256_i32gather_ps::<4>(codebook.as_ptr(), indices);
        let vx = _mm256_loadu_ps(x.as_ptr().add(idx));

        vdot = _mm256_fmadd_ps(vx, vy, vdot);
        vnorm_x = _mm256_fmadd_ps(vx, vx, vnorm_x);
        vnorm_y = _mm256_fmadd_ps(vy, vy, vnorm_y);
    }

    let mut dot = hsum256_ps(vdot);
    let mut norm_x = hsum256_ps(vnorm_x);
    let mut norm_y = hsum256_ps(vnorm_y);

    let start = chunks * 8;
    for i in start..n {
        let qy = codebook[y[i] as usize];
        dot += x[i] * qy;
        norm_x += x[i] * x[i];
        norm_y += qy * qy;
    }

    if norm_x == 0.0 && norm_y == 0.0 {
        return 0.0;
    } else if norm_x == 0.0 || norm_y == 0.0 {
        return FLOAT32_MAX;
    } else if dot <= 0.0 {
        return FLOAT32_MAX;
    }

    let sim = dot / (norm_x * norm_y).sqrt();
    -((sim + 1.0) / 2.0).log2()
}

/// Horizontal sum of 8 floats in __m256.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_codebook_256() -> Vec<f32> {
        // Simple identity codebook: index i → value i as float
        (0..256).map(|i| i as f32 / 255.0).collect()
    }

    fn make_codebook_16() -> Vec<f32> {
        (0..16).map(|i| i as f32 / 15.0).collect()
    }

    #[test]
    fn test_u8_sq_euclidean_identical() {
        let cb = make_codebook_256();
        let x = vec![0.0 / 255.0, 128.0 / 255.0, 255.0 / 255.0];
        let y = vec![0u8, 128u8, 255u8];
        let d = quantized_u8_sq_euclidean(&x, &y, &cb);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn test_u8_sq_euclidean_basic() {
        let cb = make_codebook_256();
        let x = vec![0.0; 4];
        let y = vec![1u8, 1u8, 1u8, 1u8];
        // Each dequantized y_i = 1/255, diff = 1/255
        // sq_euc = 4 * (1/255)^2
        let expected = 4.0 * (1.0 / 255.0) * (1.0 / 255.0);
        let d = quantized_u8_sq_euclidean(&x, &y, &cb);
        assert!((d - expected).abs() < 1e-8);
    }

    #[test]
    fn test_u4_sq_euclidean_identical() {
        let cb = make_codebook_16();
        // 4 dimensions packed into 2 bytes
        // dim0 = idx 0 (lower nibble byte 0), dim1 = idx 7 (upper nibble byte 0)
        // dim2 = idx 15 (lower nibble byte 1), dim3 = idx 8 (upper nibble byte 1)
        let y = vec![0x70u8, 0x8Fu8]; // nibbles: 0, 7, 15, 8
        let x = vec![0.0 / 15.0, 7.0 / 15.0, 15.0 / 15.0, 8.0 / 15.0];
        let d = quantized_u4_sq_euclidean(&x, &y, &cb);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn test_u8_alt_cosine_identical() {
        let cb: Vec<f32> = (0..256).map(|i| (i as f32 + 1.0) / 256.0).collect();
        let y: Vec<u8> = (0..8).collect();
        let x: Vec<f32> = y.iter().map(|&i| cb[i as usize]).collect();
        let d = quantized_u8_alternative_cosine(&x, &y, &cb);
        // Cosine similarity of identical vectors should be ~1
        // -log₂((1+1)/2) = -log₂(1) = 0
        assert!(d.abs() < 1e-4, "expected ~0, got {}", d);
    }
}
