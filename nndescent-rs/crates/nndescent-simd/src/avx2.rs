//! AVX2-optimized distance functions.
//!
//! These functions require AVX2 and FMA support.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Squared Euclidean distance using AVX2.
///
/// # Safety
/// Requires AVX2 and FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn l2_sqr_avx2(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let n = x.len();
    let chunks = n / 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let idx = i * 8;
        let xx = _mm256_loadu_ps(x.as_ptr().add(idx));
        let yy = _mm256_loadu_ps(y.as_ptr().add(idx));
        let diff = _mm256_sub_ps(xx, yy);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    let mut result = hsum256_ps_avx(sum);

    // Handle remainder
    let start = chunks * 8;
    for i in start..n {
        let d = x[i] - y[i];
        result += d * d;
    }

    result
}

/// Inner product (negative) using AVX2.
///
/// # Safety
/// Requires AVX2 and FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn inner_product_avx2(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let n = x.len();
    let chunks = n / 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let idx = i * 8;
        let xx = _mm256_loadu_ps(x.as_ptr().add(idx));
        let yy = _mm256_loadu_ps(y.as_ptr().add(idx));
        sum = _mm256_fmadd_ps(xx, yy, sum);
    }

    let mut result = hsum256_ps_avx(sum);

    // Handle remainder
    let start = chunks * 8;
    for i in start..n {
        result += x[i] * y[i];
    }

    -result // Negate for distance (higher similarity = lower distance)
}

/// Cosine distance using AVX2.
///
/// # Safety
/// Requires AVX2 and FMA CPU support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn cosine_avx2(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let n = x.len();
    let chunks = n / 8;

    let mut dot = _mm256_setzero_ps();
    let mut norm_x = _mm256_setzero_ps();
    let mut norm_y = _mm256_setzero_ps();

    for i in 0..chunks {
        let idx = i * 8;
        let xx = _mm256_loadu_ps(x.as_ptr().add(idx));
        let yy = _mm256_loadu_ps(y.as_ptr().add(idx));

        dot = _mm256_fmadd_ps(xx, yy, dot);
        norm_x = _mm256_fmadd_ps(xx, xx, norm_x);
        norm_y = _mm256_fmadd_ps(yy, yy, norm_y);
    }

    let mut dot_sum = hsum256_ps_avx(dot);
    let mut norm_x_sum = hsum256_ps_avx(norm_x);
    let mut norm_y_sum = hsum256_ps_avx(norm_y);

    // Handle remainder
    let start = chunks * 8;
    for i in start..n {
        dot_sum += x[i] * y[i];
        norm_x_sum += x[i] * x[i];
        norm_y_sum += y[i] * y[i];
    }

    let denom = (norm_x_sum * norm_y_sum).sqrt();
    if denom < 1e-12 {
        return 1.0;
    }

    let similarity = (dot_sum / denom).clamp(-1.0, 1.0);
    1.0 - similarity
}

/// Horizontal sum of 8 floats in a __m256.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn hsum256_ps_avx(v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);

    // Add high and low lanes
    let sum128 = _mm_add_ps(hi, lo);

    // Horizontal add within 128 bits
    let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
    let sums = _mm_add_ps(sum128, shuf); // [0+1, 1+1, 2+3, 3+3]
    let shuf2 = _mm_movehl_ps(sums, sums); // [2+3, 3+3, ...]
    let result = _mm_add_ss(sums, shuf2);

    _mm_cvtss_f32(result)
}

/// Prefetch data into L1 cache.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn prefetch_l1<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
}

/// Prefetch data into L2 cache.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn prefetch_l2<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_l2_sqr(x: &[f32], y: &[f32]) -> f32 {
        x.iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    fn scalar_ip(x: &[f32], y: &[f32]) -> f32 {
        -x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f32>()
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_l2_sqr_avx2() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not supported, skipping test");
            return;
        }

        let x: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let y: Vec<f32> = (0..256).map(|i| (i as f32).cos()).collect();

        let scalar = scalar_l2_sqr(&x, &y);
        let simd = unsafe { l2_sqr_avx2(&x, &y) };

        assert!(
            (scalar - simd).abs() < 1e-4,
            "scalar={}, simd={}",
            scalar,
            simd
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_inner_product_avx2() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not supported, skipping test");
            return;
        }

        let x: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let y: Vec<f32> = (0..256).map(|i| (i as f32).cos()).collect();

        let scalar = scalar_ip(&x, &y);
        let simd = unsafe { inner_product_avx2(&x, &y) };

        assert!(
            (scalar - simd).abs() < 1e-4,
            "scalar={}, simd={}",
            scalar,
            simd
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_non_aligned_lengths() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not supported, skipping test");
            return;
        }

        // Test with length not divisible by 8
        for len in [1, 7, 9, 15, 17, 33, 100] {
            let x: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
            let y: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1 + 0.5).collect();

            let scalar = scalar_l2_sqr(&x, &y);
            let simd = unsafe { l2_sqr_avx2(&x, &y) };

            assert!(
                (scalar - simd).abs() < 1e-4,
                "len={}, scalar={}, simd={}",
                len,
                scalar,
                simd
            );
        }
    }
}
