//! SIMD-optimized kernels for nndescent-core.
//!
//! This crate provides SIMD-optimized implementations of distance functions
//! using x86 AVX2 and AVX-512 instructions.
//!
//! The implementations follow patterns from the Glass library for maximum performance.

#![allow(clippy::missing_safety_doc)]

pub mod avx2;
pub mod avx512;
pub mod detect;

pub use detect::SimdLevel;

/// Get the best SIMD level available on this CPU.
pub fn best_simd_level() -> SimdLevel {
    detect::detect_simd_level()
}
