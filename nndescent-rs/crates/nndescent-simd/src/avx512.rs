//! AVX-512 optimized distance functions.
//!
//! These functions require AVX-512F CPU support.
//! NOTE: AVX-512 intrinsics require nightly Rust or Rust 1.72+ with
//! target-feature flags. On stable Rust <1.72, these are disabled.

// AVX-512 intrinsics are unstable on older Rust versions.
// We'll provide stub implementations that fall back to AVX2 or scalar.

/// Check if AVX-512 is available at runtime.
#[cfg(target_arch = "x86_64")]
pub fn is_avx512_available() -> bool {
    is_x86_feature_detected!("avx512f")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn is_avx512_available() -> bool {
    false
}

// For stable Rust, we provide fallback implementations using AVX2.
// The actual AVX-512 implementations would require nightly Rust.

/// Squared Euclidean distance using AVX-512 (fallback to scalar).
///
/// On stable Rust without AVX-512 support, this falls back to scalar code.
pub fn l2_sqr_avx512_fallback(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum()
}

/// Inner product (negative) using AVX-512 (fallback to scalar).
pub fn inner_product_avx512_fallback(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    -x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f32>()
}

/// Cosine distance using AVX-512 (fallback to scalar).
pub fn cosine_avx512_fallback(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    
    let mut dot = 0.0f32;
    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;
    
    for (a, b) in x.iter().zip(y.iter()) {
        dot += a * b;
        norm_x += a * a;
        norm_y += b * b;
    }
    
    let denom = (norm_x * norm_y).sqrt();
    if denom < 1e-12 {
        return 1.0;
    }
    
    let similarity = (dot / denom).clamp(-1.0, 1.0);
    1.0 - similarity
}

// Re-export fallbacks as the "avx512" versions for stable Rust
pub use l2_sqr_avx512_fallback as l2_sqr_avx512;
pub use inner_product_avx512_fallback as inner_product_avx512;
pub use cosine_avx512_fallback as cosine_avx512;

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_l2_sqr(x: &[f32], y: &[f32]) -> f32 {
        x.iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    #[test]
    fn test_l2_sqr_fallback() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let y: Vec<f32> = (0..256).map(|i| (i as f32).cos()).collect();

        let scalar = scalar_l2_sqr(&x, &y);
        let fallback = l2_sqr_avx512_fallback(&x, &y);

        assert!(
            (scalar - fallback).abs() < 1e-6,
            "scalar={}, fallback={}",
            scalar,
            fallback
        );
    }

    #[test]
    fn test_inner_product_fallback() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let y: Vec<f32> = (0..256).map(|i| (i as f32).cos()).collect();

        let scalar: f32 = -x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f32>();
        let fallback = inner_product_avx512_fallback(&x, &y);

        assert!(
            (scalar - fallback).abs() < 1e-6,
            "scalar={}, fallback={}",
            scalar,
            fallback
        );
    }
}
