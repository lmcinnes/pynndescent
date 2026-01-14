//! CPU feature detection for SIMD support.

/// Available SIMD levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// No SIMD (scalar fallback)
    None,
    /// SSE4.1 support
    Sse4,
    /// AVX2 + FMA support
    Avx2,
    /// AVX-512F support
    Avx512,
}

impl SimdLevel {
    /// Get a human-readable name for this SIMD level.
    pub fn name(&self) -> &'static str {
        match self {
            SimdLevel::None => "Scalar",
            SimdLevel::Sse4 => "SSE4.1",
            SimdLevel::Avx2 => "AVX2+FMA",
            SimdLevel::Avx512 => "AVX-512",
        }
    }
}

/// Detect the best SIMD level available on the current CPU.
pub fn detect_simd_level() -> SimdLevel {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdLevel::Avx2;
        }
        if is_x86_feature_detected!("sse4.1") {
            return SimdLevel::Sse4;
        }
    }
    SimdLevel::None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect() {
        let level = detect_simd_level();
        println!("Detected SIMD level: {}", level.name());
        // Just verify it doesn't crash
        assert!(level >= SimdLevel::None);
    }
}
