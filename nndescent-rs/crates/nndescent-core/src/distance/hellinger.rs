//! Hellinger distance and other probability distribution distances.
//!
//! These distances treat input vectors as (unnormalized) probability distributions.

use super::traits::Distance;

const FLOAT32_MAX: f32 = f32::MAX;
const FLOAT32_EPS: f32 = f32::EPSILON;

/// Hellinger distance for probability distributions.
///
/// Formula: d(a, b) = sqrt(1 - Σ√(aᵢ·bᵢ) / √(Σaᵢ · Σbᵢ))
#[derive(Clone, Copy, Debug, Default)]
pub struct Hellinger;

impl Distance<f32> for Hellinger {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut bc = 0.0f32;  // Bhattacharyya coefficient numerator
        let mut l1_a = 0.0f32;
        let mut l1_b = 0.0f32;

        for i in 0..a.len() {
            bc += (a[i] * b[i]).sqrt();
            l1_a += a[i];
            l1_b += b[i];
        }

        if l1_a == 0.0 && l1_b == 0.0 {
            0.0
        } else if l1_a == 0.0 || l1_b == 0.0 {
            1.0
        } else {
            let val = 1.0 - bc / (l1_a * l1_b).sqrt();
            // Clamp to avoid NaN from floating-point errors
            val.max(0.0).sqrt()
        }
    }

    fn name(&self) -> &'static str {
        "hellinger"
    }
}

/// Alternative Hellinger using log transform for bounded-radius search.
///
/// Formula: d_alt(a, b) = log₂(√(Σaᵢ·Σbᵢ) / Σ√(aᵢ·bᵢ))
/// Correction: sqrt(1 - 2^(-d_alt))
#[derive(Clone, Copy, Debug, Default)]
pub struct AlternativeHellinger;

impl Distance<f32> for AlternativeHellinger {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut bc = 0.0f32;
        let mut l1_a = 0.0f32;
        let mut l1_b = 0.0f32;

        for i in 0..a.len() {
            bc += (a[i] * b[i]).sqrt();
            l1_a += a[i];
            l1_b += b[i];
        }

        if l1_a == 0.0 && l1_b == 0.0 {
            0.0
        } else if l1_a == 0.0 || l1_b == 0.0 {
            FLOAT32_MAX
        } else if bc <= 0.0 {
            FLOAT32_MAX
        } else {
            ((l1_a * l1_b).sqrt() / bc).log2()
        }
    }

    fn name(&self) -> &'static str {
        "alternative_hellinger"
    }
}

/// Correction for alternative Hellinger: d = sqrt(1 - 2^(-d_alt))
#[inline]
pub fn correct_alternative_hellinger(d: f32) -> f32 {
    (1.0 - 2.0f32.powf(-d)).max(0.0).sqrt()
}

/// Jensen-Shannon divergence.
///
/// Formula: d(a, b) = 0.5 * (KL(p||m) + KL(q||m)) where m = (p+q)/2
/// Input vectors are normalized to probability distributions.
#[derive(Clone, Copy, Debug, Default)]
pub struct JensenShannon;

impl Distance<f32> for JensenShannon {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let dim = a.len();

        let mut l1_a = 0.0f32;
        let mut l1_b = 0.0f32;
        for i in 0..dim {
            l1_a += a[i];
            l1_b += b[i];
        }

        let eps_total = FLOAT32_EPS * dim as f32;
        l1_a += eps_total;
        l1_b += eps_total;

        let mut result = 0.0f32;
        for i in 0..dim {
            let px = (a[i] + FLOAT32_EPS) / l1_a;
            let py = (b[i] + FLOAT32_EPS) / l1_b;
            let m = 0.5 * (px + py);
            result += 0.5 * (px * (px / m).ln() + py * (py / m).ln());
        }
        result
    }

    fn name(&self) -> &'static str {
        "jensen_shannon"
    }
}

/// Symmetric KL divergence.
///
/// Formula: d(a, b) = KL(p||q) + KL(q||p)
/// Input vectors are normalized to probability distributions.
#[derive(Clone, Copy, Debug, Default)]
pub struct SymmetricKL;

impl Distance<f32> for SymmetricKL {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let dim = a.len();

        let mut l1_a = 0.0f32;
        let mut l1_b = 0.0f32;
        for i in 0..dim {
            l1_a += a[i];
            l1_b += b[i];
        }

        let eps_total = FLOAT32_EPS * dim as f32;
        l1_a += eps_total;
        l1_b += eps_total;

        let mut result = 0.0f32;
        for i in 0..dim {
            let px = (a[i] + FLOAT32_EPS) / l1_a;
            let py = (b[i] + FLOAT32_EPS) / l1_b;
            result += px * (px / py).ln() + py * (py / px).ln();
        }
        result
    }

    fn name(&self) -> &'static str {
        "symmetric_kl"
    }
}

/// Proxy for Jensen-Shannon divergence using squared Bhattacharyya coefficient.
///
/// Formula: d_proxy(a, b) = 1 - BC² where BC = Σ√(pᵢ·qᵢ)
/// Results should be reranked with true Jensen-Shannon.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProxyJensenShannon;

impl Distance<f32> for ProxyJensenShannon {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut l1_a = 0.0f32;
        let mut l1_b = 0.0f32;
        for i in 0..a.len() {
            l1_a += a[i];
            l1_b += b[i];
        }

        if l1_a == 0.0 || l1_b == 0.0 {
            return FLOAT32_MAX;
        }

        let mut bc = 0.0f32;
        for i in 0..a.len() {
            bc += ((a[i] / l1_a) * (b[i] / l1_b)).sqrt();
        }
        1.0 - bc * bc
    }

    fn name(&self) -> &'static str {
        "proxy_jensen_shannon"
    }
}

/// Proxy for symmetric KL divergence using triangular discrimination.
///
/// Formula: d_proxy(a, b) = Σ (pᵢ - qᵢ)² / (pᵢ + qᵢ)
/// Results should be reranked with true symmetric KL.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProxySymmetricKL;

impl Distance<f32> for ProxySymmetricKL {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut l1_a = 0.0f32;
        let mut l1_b = 0.0f32;
        for i in 0..a.len() {
            l1_a += a[i];
            l1_b += b[i];
        }

        if l1_a == 0.0 || l1_b == 0.0 {
            return FLOAT32_MAX;
        }

        let mut result = 0.0f32;
        for i in 0..a.len() {
            let px = a[i] / l1_a;
            let py = b[i] / l1_b;
            let denom = px + py;
            if denom > 0.0 {
                let diff = px - py;
                result += (diff * diff) / denom;
            }
        }
        result
    }

    fn name(&self) -> &'static str {
        "proxy_symmetric_kl"
    }
}

/// Proxy for Kantorovich/Wasserstein distance.
///
/// Uses total variation + Hellinger-like term.
/// Results should be reranked with true Kantorovich.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProxyKantorovich;

impl Distance<f32> for ProxyKantorovich {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut l1_a = 0.0f32;
        let mut l1_b = 0.0f32;
        for i in 0..a.len() {
            l1_a += a[i];
            l1_b += b[i];
        }

        if l1_a == 0.0 || l1_b == 0.0 {
            return FLOAT32_MAX;
        }

        let mut tv = 0.0f32;
        let mut hell = 0.0f32;
        for i in 0..a.len() {
            let px = a[i] / l1_a;
            let py = b[i] / l1_b;
            tv += (px - py).abs();
            hell += (px * py).sqrt();
        }
        0.5 * tv + (1.0 - hell)
    }

    fn name(&self) -> &'static str {
        "proxy_kantorovich"
    }
}

/// Proxy for 1D Wasserstein distance using CDF L1 distance.
///
/// Exact for Wasserstein-1; lower bound for p > 1.
/// Results should be reranked with true Wasserstein-1D.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProxyWasserstein1D;

impl Distance<f32> for ProxyWasserstein1D {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut l1_a = 0.0f32;
        let mut l1_b = 0.0f32;
        for i in 0..a.len() {
            l1_a += a[i];
            l1_b += b[i];
        }

        if l1_a == 0.0 || l1_b == 0.0 {
            return FLOAT32_MAX;
        }

        let mut cdf_a = 0.0f32;
        let mut cdf_b = 0.0f32;
        let mut result = 0.0f32;
        for i in 0..a.len() {
            cdf_a += a[i] / l1_a;
            cdf_b += b[i] / l1_b;
            result += (cdf_a - cdf_b).abs();
        }
        result
    }

    fn name(&self) -> &'static str {
        "proxy_wasserstein_1d"
    }
}

/// Proxy for circular Kantorovich using mean-shifted CDF L1 distance.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProxyCircularKantorovich;

impl Distance<f32> for ProxyCircularKantorovich {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let dim = a.len();

        let mut l1_a = 0.0f32;
        let mut l1_b = 0.0f32;
        for i in 0..dim {
            l1_a += a[i];
            l1_b += b[i];
        }

        if l1_a == 0.0 || l1_b == 0.0 {
            return FLOAT32_MAX;
        }

        // First pass: compute CDFs and mean of differences
        let mut cdf_a = 0.0f32;
        let mut cdf_b = 0.0f32;
        let mut mu = 0.0f32;
        for i in 0..dim {
            cdf_a += a[i] / l1_a;
            cdf_b += b[i] / l1_b;
            mu += cdf_a - cdf_b;
        }
        mu /= dim as f32;

        // Second pass: L1 on shifted CDFs
        cdf_a = 0.0;
        cdf_b = 0.0;
        let mut result = 0.0f32;
        for i in 0..dim {
            cdf_a += a[i] / l1_a;
            cdf_b += b[i] / l1_b;
            result += (cdf_a - cdf_b - mu).abs();
        }
        result
    }

    fn name(&self) -> &'static str {
        "proxy_circular_kantorovich"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hellinger_identical() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let d = Hellinger.distance(&a, &a);
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn test_hellinger_disjoint() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 1.0];
        let d = Hellinger.distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_alternative_hellinger_identical() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let d = AlternativeHellinger.distance(&a, &a);
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn test_correct_alternative_hellinger() {
        // d_alt = 0 → sqrt(1 - 1) = 0
        assert!(correct_alternative_hellinger(0.0).abs() < 1e-6);
    }

    #[test]
    fn test_jensen_shannon_identical() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let d = JensenShannon.distance(&a, &a);
        assert!(d.abs() < 1e-4);
    }

    #[test]
    fn test_symmetric_kl_identical() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let d = SymmetricKL.distance(&a, &a);
        assert!(d.abs() < 1e-4);
    }

    #[test]
    fn test_proxy_wasserstein_1d_identical() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let d = ProxyWasserstein1D.distance(&a, &a);
        assert!(d.abs() < 1e-5);
    }
}
