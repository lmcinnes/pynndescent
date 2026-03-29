//! Euclidean (L2) and Squared Euclidean distance implementations.

use super::traits::{Distance, HasSquaredForm};

/// Squared Euclidean distance: sum of squared differences.
///
/// This is faster than Euclidean as it avoids the sqrt operation.
/// Use this during search and apply sqrt to final results.
///
/// Formula: d²(a, b) = Σ(aᵢ - bᵢ)²
#[derive(Clone, Copy, Debug, Default)]
pub struct SquaredEuclidean;

impl Distance<f32> for SquaredEuclidean {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { l2_sqr_avx2(a, b) };
            }
        }

        // Scalar fallback with loop unrolling
        scalar_l2_sqr(a, b)
    }

    fn name(&self) -> &'static str {
        "squared_euclidean"
    }
}

/// Scalar squared Euclidean distance (fallback).
#[inline]
fn scalar_l2_sqr(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;

    // Process in chunks of 4 for better pipelining
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let idx = i * 4;
        let d0 = a[idx] - b[idx];
        let d1 = a[idx + 1] - b[idx + 1];
        let d2 = a[idx + 2] - b[idx + 2];
        let d3 = a[idx + 3] - b[idx + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    // Handle remainder
    let start = chunks * 4;
    for i in 0..remainder {
        let d = a[start + i] - b[start + i];
        sum += d * d;
    }

    sum
}

/// AVX2+FMA optimized squared Euclidean distance.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_sqr_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let idx = i * 8;
        let aa = _mm256_loadu_ps(a.as_ptr().add(idx));
        let bb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm256_sub_ps(aa, bb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    let mut result = hsum256_ps_avx(sum);

    // Handle remainder
    let start = chunks * 8;
    for i in start..n {
        let d = a[i] - b[i];
        result += d * d;
    }

    result
}

/// Horizontal sum of 8 floats in a __m256.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn hsum256_ps_avx(v: std::arch::x86_64::__m256) -> f32 {
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

/// Euclidean (L2) distance: sqrt of sum of squared differences.
///
/// Formula: d(a, b) = √(Σ(aᵢ - bᵢ)²)
#[derive(Clone, Copy, Debug, Default)]
pub struct Euclidean;

impl Distance<f32> for Euclidean {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        SquaredEuclidean.distance(a, b).sqrt()
    }

    fn name(&self) -> &'static str {
        "euclidean"
    }
}

impl HasSquaredForm for Euclidean {
    type Squared = SquaredEuclidean;

    fn squared(&self) -> SquaredEuclidean {
        SquaredEuclidean
    }
}

/// SIMD-optimized squared Euclidean using portable SIMD when available.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub mod simd {
    use super::*;

    /// AVX2-optimized squared Euclidean distance.
    ///
    /// Processes 8 floats at a time using 256-bit SIMD.
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn squared_euclidean_avx2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        debug_assert_eq!(a.len(), b.len());

        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;

        for i in 0..chunks {
            let idx = i * 8;
            let aa = _mm256_loadu_ps(a.as_ptr().add(idx));
            let bb = _mm256_loadu_ps(b.as_ptr().add(idx));
            let diff = _mm256_sub_ps(aa, bb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        // Horizontal sum
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(hi, lo);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let mut result = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));

        // Handle remainder
        let start = chunks * 8;
        for i in start..a.len() {
            let d = a[i] - b[i];
            result += d * d;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_euclidean_basic() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let d = SquaredEuclidean.distance(&a, &b);
        assert!((d - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_basic() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let d = Euclidean.distance(&a, &b);
        assert!((d - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_same_point() {
        let a = vec![1.0, 2.0, 3.0];
        let d = Euclidean.distance(&a, &a);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_squared_euclidean_known_values() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];

        // 3² + 4² = 25, sqrt = 5
        let d_sq = SquaredEuclidean.distance(&a, &b);
        let d = Euclidean.distance(&a, &b);

        assert!((d_sq - 25.0).abs() < 1e-6);
        assert!((d - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_large_vectors() {
        let dim = 1024;
        let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01) + 0.5).collect();

        // Should be n * 0.25 where n = 1024
        let d_sq = SquaredEuclidean.distance(&a, &b);
        assert!((d_sq - 256.0).abs() < 1e-3);
    }

    #[test]
    fn test_has_squared_form() {
        let euclidean = Euclidean;
        let squared = euclidean.squared();

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let d_euc = euclidean.distance(&a, &b);
        let d_sq = squared.distance(&a, &b);

        assert!((d_euc * d_euc - d_sq).abs() < 1e-5);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn test_simd_matches_scalar() {
        let dim = 256;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32).cos()).collect();

        let scalar = SquaredEuclidean.distance(&a, &b);
        let simd = unsafe { simd::squared_euclidean_avx2(&a, &b) };

        assert!((scalar - simd).abs() < 1e-4);
    }
}
