//! Cosine distance implementation.
//!
//! Cosine distance measures the angle between vectors, independent of magnitude.
//! It is commonly used for text embeddings and normalized vectors.

use super::traits::Distance;

/// Cosine distance: 1 - cosine_similarity.
///
/// For normalized vectors, this simplifies to 1 - dot(a, b).
///
/// Formula: d(a, b) = 1 - (a · b) / (‖a‖ × ‖b‖)
///
/// Range: [0, 2] where 0 means identical direction, 1 means orthogonal,
/// and 2 means opposite direction.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cosine;

/// Scalar cosine distance fallback.
#[inline]
fn scalar_cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-12 {
        return 1.0;
    }
    let similarity = (dot / denom).clamp(-1.0, 1.0);
    1.0 - similarity
}

/// AVX2+FMA optimized cosine distance.
/// Computes dot product and both norms in a single pass using three accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;

    let mut vdot = _mm256_setzero_ps();
    let mut vnorm_a = _mm256_setzero_ps();
    let mut vnorm_b = _mm256_setzero_ps();

    for i in 0..chunks {
        let idx = i * 8;
        let aa = _mm256_loadu_ps(a.as_ptr().add(idx));
        let bb = _mm256_loadu_ps(b.as_ptr().add(idx));
        vdot = _mm256_fmadd_ps(aa, bb, vdot);
        vnorm_a = _mm256_fmadd_ps(aa, aa, vnorm_a);
        vnorm_b = _mm256_fmadd_ps(bb, bb, vnorm_b);
    }

    // Horizontal sums
    let mut dot = hsum256_ps(vdot);
    let mut norm_a = hsum256_ps(vnorm_a);
    let mut norm_b = hsum256_ps(vnorm_b);

    // Handle remainder
    let start = chunks * 8;
    for i in start..n {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-12 {
        return 1.0;
    }
    let similarity = (dot / denom).clamp(-1.0, 1.0);
    1.0 - similarity
}

/// Horizontal sum of 8 floats in a __m256.
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

impl Distance<f32> for Cosine {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { cosine_avx2(a, b) };
            }
        }

        scalar_cosine(a, b)
    }

    fn name(&self) -> &'static str {
        "cosine"
    }
}

/// Alternative cosine distance for pre-normalized vectors.
///
/// When vectors are known to be unit-length, we can skip the normalization
/// step and just compute 1 - dot(a, b).
///
/// This is faster but only valid for normalized inputs!
#[derive(Clone, Copy, Debug, Default)]
pub struct AlternativeCosine;

impl Distance<f32> for AlternativeCosine {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut dot = 0.0f32;

        let chunks = a.len() / 4;
        let remainder = a.len() % 4;

        for i in 0..chunks {
            let idx = i * 4;
            dot += a[idx] * b[idx]
                + a[idx + 1] * b[idx + 1]
                + a[idx + 2] * b[idx + 2]
                + a[idx + 3] * b[idx + 3];
        }

        let start = chunks * 4;
        for i in 0..remainder {
            dot += a[start + i] * b[start + i];
        }

        1.0 - dot
    }

    fn name(&self) -> &'static str {
        "alternative_cosine"
    }
}

/// Normalize a vector to unit length in-place.
#[inline]
pub fn normalize_inplace(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv_norm = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }
}

/// Normalize a vector and return a new vector.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        let inv_norm = 1.0 / norm;
        v.iter().map(|x| x * inv_norm).collect()
    } else {
        v.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let d = Cosine.distance(&a, &a);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let d = Cosine.distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let d = Cosine.distance(&a, &b);
        assert!((d - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_scaled() {
        // Same direction, different magnitudes should have distance 0
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];
        let d = Cosine.distance(&a, &b);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_alternative_cosine_normalized() {
        let a = normalize(&[1.0, 2.0, 3.0]);
        let b = normalize(&[4.0, 5.0, 6.0]);

        let d_cosine = Cosine.distance(&a, &b);
        let d_alt = AlternativeCosine.distance(&a, &b);

        assert!((d_cosine - d_alt).abs() < 1e-5);
    }

    #[test]
    fn test_normalize() {
        let v = normalize(&[3.0, 4.0, 0.0]);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_known_angle() {
        // 45-degree angle
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 1.0];

        let d = Cosine.distance(&a, &b);
        // cos(45°) = 1/√2 ≈ 0.707
        // distance = 1 - 0.707 ≈ 0.293
        let expected = 1.0 - (std::f32::consts::FRAC_1_SQRT_2);
        assert!((d - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_large_vectors() {
        let dim = 768;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();

        let d = Cosine.distance(&a, &b);

        // Just verify it's in valid range
        assert!(d >= 0.0 && d <= 2.0);
    }
}
