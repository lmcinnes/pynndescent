//! Distance functions and traits for nearest neighbor computation.
//!
//! This module provides:
//! - The `Distance` trait for distance computation
//! - Scalar implementations of common distance functions
//! - SIMD-optimized variants (via optional `simd` feature)
//!
//! # Distance Functions
//!
//! Primary supported distances (optimized):
//! - Euclidean (L2)
//! - Squared Euclidean (for performance)
//! - Cosine
//! - Inner Product (Dot)
//!
//! The module also supports proxy distances where a cheaper metric can be
//! used during graph construction with a correction applied for final results.

mod traits;
mod euclidean;
mod cosine;
mod inner_product;

pub use traits::{Distance, HasSquaredForm};
pub use euclidean::{Euclidean, SquaredEuclidean};
pub use cosine::Cosine;
pub use inner_product::InnerProduct;

/// Fast distance alternatives mapping.
///
/// For some distances, we can use a faster proxy during computation
/// and apply a correction at the end. For example, squared Euclidean
/// can be used instead of Euclidean during search, with sqrt applied
/// only to final results.
pub struct FastDistanceAlternatives;

impl FastDistanceAlternatives {
    /// Get the fast alternative for Euclidean (squared euclidean + sqrt correction).
    pub fn euclidean() -> (SquaredEuclidean, fn(f32) -> f32) {
        (SquaredEuclidean, |d| d.sqrt())
    }

    /// Get the fast alternative for Cosine (alternative cosine + identity correction).
    pub fn cosine() -> (Cosine, fn(f32) -> f32) {
        (Cosine, |d| d)
    }
}

/// Available metrics as an enum for runtime selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Euclidean,
    SquaredEuclidean,
    L2,  // Alias for Euclidean
    Cosine,
    InnerProduct,
    Dot,  // Alias for InnerProduct
}

impl Metric {
    /// Compute distance between two vectors using this metric.
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Euclidean | Metric::L2 => Euclidean.distance(a, b),
            Metric::SquaredEuclidean => SquaredEuclidean.distance(a, b),
            Metric::Cosine => Cosine.distance(a, b),
            Metric::InnerProduct | Metric::Dot => InnerProduct.distance(a, b),
        }
    }

    /// Check if this metric has a fast alternative.
    pub fn has_fast_alternative(&self) -> bool {
        matches!(self, Metric::Euclidean | Metric::L2)
    }

    /// Get the correction function for fast alternative (if any).
    pub fn correction(&self) -> Option<fn(f32) -> f32> {
        match self {
            Metric::Euclidean | Metric::L2 => Some(|d: f32| d.sqrt()),
            _ => None,
        }
    }

    /// Parse metric from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "euclidean" | "l2" => Some(Metric::Euclidean),
            "sqeuclidean" | "squared_euclidean" => Some(Metric::SquaredEuclidean),
            "cosine" => Some(Metric::Cosine),
            "inner_product" | "dot" | "ip" => Some(Metric::InnerProduct),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_from_str() {
        assert_eq!(Metric::from_str("euclidean"), Some(Metric::Euclidean));
        assert_eq!(Metric::from_str("l2"), Some(Metric::Euclidean));
        assert_eq!(Metric::from_str("cosine"), Some(Metric::Cosine));
        assert_eq!(Metric::from_str("dot"), Some(Metric::InnerProduct));
        assert_eq!(Metric::from_str("unknown"), None);
    }

    #[test]
    fn test_metric_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        // Euclidean distance should be sqrt(2)
        let d = Metric::Euclidean.distance(&a, &b);
        assert!((d - std::f32::consts::SQRT_2).abs() < 1e-6);

        // Squared Euclidean should be 2
        let d = Metric::SquaredEuclidean.distance(&a, &b);
        assert!((d - 2.0).abs() < 1e-6);
    }
}
