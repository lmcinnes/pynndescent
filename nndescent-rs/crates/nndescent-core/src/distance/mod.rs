//! Distance functions and traits for nearest neighbor computation.
//!
//! This module provides:
//! - The `Distance` trait for distance computation
//! - Scalar implementations of common distance functions
//! - SIMD-optimized variants (AVX2+FMA where available)
//!
//! # Distance Categories
//!
//! ## Minkowski-family (SIMD-optimized)
//! - Euclidean (L2), Squared Euclidean, Manhattan (L1), Chebyshev (L∞), Minkowski (Lp)
//! - Canberra, Bray-Curtis
//!
//! ## Angular / similarity
//! - Cosine, Correlation, True Angular, TS-SS
//! - Inner Product (Dot)
//!
//! ## Binary / set-based
//! - Hamming, Jaccard, Dice, Matching, Kulsinski
//! - Rogers-Tanimoto, Sokal-Michener, Sokal-Sneath, Russell-Rao, Yule
//! - Bit-packed: BitHamming, BitJaccard (u8 arrays)
//!
//! ## Probability distribution
//! - Hellinger, Jensen-Shannon, Symmetric KL
//!
//! ## Proxy / alternative (fast approximations for graph construction)
//! - AlternativeCosine, AlternativeDot, AlternativeInnerProduct
//! - AlternativeHellinger, AlternativeJaccard
//! - ProxyInnerProduct, ProxyJensenShannon, ProxySymmetricKL
//! - ProxyKantorovich, ProxyWasserstein1D, ProxyCircularKantorovich
//!
//! ## Quantized (float × quantized uint8/uint4, AVX2 gather)
//! - Squared Euclidean, Alternative Cosine, Alternative Dot

mod traits;
mod euclidean;
mod cosine;
mod inner_product;
mod minkowski;
mod binary;
mod alternatives;
mod hellinger;
mod angular;
pub mod quantized;

pub use traits::{Distance, HasSquaredForm};
pub use euclidean::{Euclidean, SquaredEuclidean};
pub use cosine::Cosine;
pub use inner_product::InnerProduct;

// Minkowski family
pub use minkowski::{Manhattan, Chebyshev, Minkowski, Canberra, BrayCurtis};

// Binary/set distances
pub use binary::{
    Hamming, Jaccard, AlternativeJaccard, Matching, Dice, Kulsinski,
    RogersTanimoto, RussellRao, SokalMichener, SokalSneath, Yule,
    BitHamming, BitJaccard,
    correct_alternative_jaccard,
};

// Alternative/proxy distances
pub use alternatives::{
    AlternativeCosine, AlternativeDot, AlternativeInnerProduct, ProxyInnerProduct,
    correct_alternative_cosine, correct_alternative_inner_product,
    true_angular_from_alt_cosine,
};

// Hellinger and distribution distances
pub use hellinger::{
    Hellinger, AlternativeHellinger,
    JensenShannon, SymmetricKL,
    ProxyJensenShannon, ProxySymmetricKL,
    ProxyKantorovich, ProxyWasserstein1D, ProxyCircularKantorovich,
    correct_alternative_hellinger,
};

// Angular/special distances
pub use angular::{Correlation, TrueAngular, TSSS};

/// Fast distance alternatives mapping.
///
/// For some distances, we can use a faster proxy during computation
/// and apply a correction at the end. For example, squared Euclidean
/// can be used instead of Euclidean during search, with sqrt applied
/// only to final results.
pub struct FastDistanceAlternatives;

impl FastDistanceAlternatives {
    pub fn euclidean() -> (SquaredEuclidean, fn(f32) -> f32) {
        (SquaredEuclidean, |d| d.sqrt())
    }

    pub fn cosine() -> (AlternativeCosine, fn(f32) -> f32) {
        (AlternativeCosine, correct_alternative_cosine)
    }

    pub fn dot() -> (AlternativeDot, fn(f32) -> f32) {
        (AlternativeDot, correct_alternative_cosine)
    }

    pub fn inner_product() -> (AlternativeInnerProduct, fn(f32) -> f32) {
        (AlternativeInnerProduct, correct_alternative_inner_product)
    }

    pub fn true_angular() -> (AlternativeCosine, fn(f32) -> f32) {
        (AlternativeCosine, true_angular_from_alt_cosine)
    }

    pub fn hellinger() -> (AlternativeHellinger, fn(f32) -> f32) {
        (AlternativeHellinger, correct_alternative_hellinger)
    }

    pub fn jaccard() -> (AlternativeJaccard, fn(f32) -> f32) {
        (AlternativeJaccard, correct_alternative_jaccard)
    }
}

/// Available metrics as an enum for runtime selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    // Minkowski family
    Euclidean,
    L2,
    SquaredEuclidean,
    Manhattan,
    Chebyshev,
    Canberra,
    BrayCurtis,
    // Angular / similarity
    Cosine,
    InnerProduct,
    Dot,
    Correlation,
    TrueAngular,
    TSSS,
    // Binary / set
    Hamming,
    Jaccard,
    Dice,
    Matching,
    Kulsinski,
    RogersTanimoto,
    RussellRao,
    SokalMichener,
    SokalSneath,
    Yule,
    // Distribution
    Hellinger,
    JensenShannon,
    SymmetricKL,
}

impl Metric {
    /// Compute distance between two vectors using this metric.
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Euclidean | Metric::L2 => Euclidean.distance(a, b),
            Metric::SquaredEuclidean => SquaredEuclidean.distance(a, b),
            Metric::Manhattan => Manhattan.distance(a, b),
            Metric::Chebyshev => Chebyshev.distance(a, b),
            Metric::Canberra => Canberra.distance(a, b),
            Metric::BrayCurtis => BrayCurtis.distance(a, b),
            Metric::Cosine => Cosine.distance(a, b),
            Metric::InnerProduct | Metric::Dot => InnerProduct.distance(a, b),
            Metric::Correlation => Correlation.distance(a, b),
            Metric::TrueAngular => TrueAngular.distance(a, b),
            Metric::TSSS => TSSS.distance(a, b),
            Metric::Hamming => Hamming.distance(a, b),
            Metric::Jaccard => Jaccard.distance(a, b),
            Metric::Dice => Dice.distance(a, b),
            Metric::Matching => Matching.distance(a, b),
            Metric::Kulsinski => Kulsinski.distance(a, b),
            Metric::RogersTanimoto => RogersTanimoto.distance(a, b),
            Metric::RussellRao => RussellRao.distance(a, b),
            Metric::SokalMichener => SokalMichener.distance(a, b),
            Metric::SokalSneath => SokalSneath.distance(a, b),
            Metric::Yule => Yule.distance(a, b),
            Metric::Hellinger => Hellinger.distance(a, b),
            Metric::JensenShannon => JensenShannon.distance(a, b),
            Metric::SymmetricKL => SymmetricKL.distance(a, b),
        }
    }

    /// Check if this metric has a fast alternative for graph construction.
    pub fn has_fast_alternative(&self) -> bool {
        matches!(
            self,
            Metric::Euclidean
                | Metric::L2
                | Metric::Cosine
                | Metric::Dot
                | Metric::InnerProduct
                | Metric::TrueAngular
                | Metric::Hellinger
                | Metric::Jaccard
        )
    }

    /// Get the correction function for fast alternative (if any).
    pub fn correction(&self) -> Option<fn(f32) -> f32> {
        match self {
            Metric::Euclidean | Metric::L2 => Some(|d: f32| d.sqrt()),
            Metric::Cosine | Metric::Dot => Some(correct_alternative_cosine),
            Metric::InnerProduct => Some(correct_alternative_inner_product),
            Metric::TrueAngular => Some(true_angular_from_alt_cosine),
            Metric::Hellinger => Some(correct_alternative_hellinger),
            Metric::Jaccard => Some(correct_alternative_jaccard),
            _ => None,
        }
    }

    /// Parse metric from string (case-insensitive, supports PyNND aliases).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            // Minkowski family
            "euclidean" | "l2" => Some(Metric::Euclidean),
            "sqeuclidean" | "squared_euclidean" => Some(Metric::SquaredEuclidean),
            "manhattan" | "taxicab" | "l1" => Some(Metric::Manhattan),
            "chebyshev" | "linfinity" | "linfty" | "linf" => Some(Metric::Chebyshev),
            "canberra" => Some(Metric::Canberra),
            "braycurtis" => Some(Metric::BrayCurtis),
            // Angular / similarity
            "cosine" => Some(Metric::Cosine),
            "inner_product" | "ip" => Some(Metric::InnerProduct),
            "dot" => Some(Metric::Dot),
            "correlation" => Some(Metric::Correlation),
            "true_angular" => Some(Metric::TrueAngular),
            "tsss" => Some(Metric::TSSS),
            // Binary / set
            "hamming" => Some(Metric::Hamming),
            "jaccard" => Some(Metric::Jaccard),
            "dice" => Some(Metric::Dice),
            "matching" => Some(Metric::Matching),
            "kulsinski" => Some(Metric::Kulsinski),
            "rogerstanimoto" => Some(Metric::RogersTanimoto),
            "russellrao" => Some(Metric::RussellRao),
            "sokalsneath" => Some(Metric::SokalSneath),
            "sokalmichener" => Some(Metric::SokalMichener),
            "yule" => Some(Metric::Yule),
            // Distribution
            "hellinger" => Some(Metric::Hellinger),
            "jensen_shannon" | "jensen-shannon" => Some(Metric::JensenShannon),
            "symmetric_kl" | "symmetric-kl" | "symmetric_kullback_liebler" => {
                Some(Metric::SymmetricKL)
            }
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
        assert_eq!(Metric::from_str("dot"), Some(Metric::Dot));
        assert_eq!(Metric::from_str("manhattan"), Some(Metric::Manhattan));
        assert_eq!(Metric::from_str("taxicab"), Some(Metric::Manhattan));
        assert_eq!(Metric::from_str("l1"), Some(Metric::Manhattan));
        assert_eq!(Metric::from_str("chebyshev"), Some(Metric::Chebyshev));
        assert_eq!(Metric::from_str("hellinger"), Some(Metric::Hellinger));
        assert_eq!(Metric::from_str("jaccard"), Some(Metric::Jaccard));
        assert_eq!(Metric::from_str("jensen-shannon"), Some(Metric::JensenShannon));
        assert_eq!(Metric::from_str("symmetric_kl"), Some(Metric::SymmetricKL));
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
