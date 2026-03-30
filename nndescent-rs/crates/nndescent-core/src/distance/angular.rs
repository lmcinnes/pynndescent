//! Angular and combined distances: Correlation, TrueAngular, TSSS.

use super::traits::Distance;

const FLOAT32_MAX: f32 = f32::MAX;

/// Correlation distance: 1 - Pearson correlation.
///
/// Equivalent to cosine distance on mean-centered data.
#[derive(Clone, Copy, Debug, Default)]
pub struct Correlation;

impl Distance<f32> for Correlation {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        if n == 0 {
            return 0.0;
        }

        let mut mu_a = 0.0f32;
        let mut mu_b = 0.0f32;
        for i in 0..n {
            mu_a += a[i];
            mu_b += b[i];
        }
        mu_a /= n as f32;
        mu_b /= n as f32;

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..n {
            let sa = a[i] - mu_a;
            let sb = b[i] - mu_b;
            dot += sa * sb;
            norm_a += sa * sa;
            norm_b += sb * sb;
        }

        if norm_a == 0.0 && norm_b == 0.0 {
            0.0
        } else if dot == 0.0 {
            1.0
        } else {
            1.0 - dot / (norm_a * norm_b).sqrt()
        }
    }

    fn name(&self) -> &'static str {
        "correlation"
    }
}

/// True angular distance: 1 - arccos(cosine_sim) / π.
///
/// Returns values in [0, 1] representing the normalized angle.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrueAngular;

impl Distance<f32> for TrueAngular {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if norm_a == 0.0 && norm_b == 0.0 {
            return 0.0;
        } else if norm_a == 0.0 || norm_b == 0.0 {
            return FLOAT32_MAX;
        } else if dot <= 0.0 {
            return FLOAT32_MAX;
        }

        let sim = (dot / (norm_a * norm_b).sqrt()).clamp(-1.0, 1.0);
        1.0 - sim.acos() / std::f32::consts::PI
    }

    fn name(&self) -> &'static str {
        "true_angular"
    }
}

/// Triangle Area Similarity × Sector Area Similarity (TS-SS) distance.
///
/// Combines both magnitude and angular information.
#[derive(Clone, Copy, Debug, Default)]
pub struct TSSS;

impl Distance<f32> for TSSS {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut d_euc_sq = 0.0f32;
        let mut d_cos = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..a.len() {
            let diff = a[i] - b[i];
            d_euc_sq += diff * diff;
            d_cos += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        norm_a = norm_a.sqrt();
        norm_b = norm_b.sqrt();
        let mag_diff = (norm_a - norm_b).abs();

        let denom = norm_a * norm_b;
        if denom < 1e-12 {
            return 0.0;
        }

        let cos_sim = (d_cos / denom).clamp(-1.0, 1.0);
        let theta = cos_sim.acos() + 10.0f32.to_radians();

        let sector = (d_euc_sq.sqrt() + mag_diff).powi(2) * theta;
        let triangle = norm_a * norm_b * theta.sin() / 2.0;

        triangle * sector
    }

    fn name(&self) -> &'static str {
        "tsss"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = Correlation.distance(&a, &a);
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn test_correlation_perfect_positive() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];
        let d = Correlation.distance(&a, &b);
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn test_true_angular_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let d = TrueAngular.distance(&a, &a);
        // arccos(1) = 0, so 1 - 0/π = 1... wait, this is similarity
        // For identical vectors: cos_sim = 1 → arccos(1) = 0 → 1 - 0/π = 1
        // But PyNND returns 0 for identical vectors... let me check
        // Actually: 1 - arccos(cosine_sim)/π → 1 - 0/π = 1
        // But in PyNND: it returns 1 - arccos(result)/pi for positive dot
        // Hmm, for identical vectors, this is 1 - 0 = 1... That's similarity not distance
        // Looking at PyNND: it returns 1 - arccos(result)/pi which is ~ 1 for identical
        // That seems wrong for a distance but that's what PyNND does
        // Actually PyNND's true_angular is unique - it returns a similarity-like value
        // where 1 = same direction.
        assert!((d - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tsss_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let d = TSSS.distance(&a, &a);
        // cos_sim = 1 → theta = arccos(1) + 10° = 10°
        // mag_diff = 0, d_euc = 0
        // sector = 0² * theta = 0
        // triangle * sector = 0
        assert!(d.abs() < 1e-5);
    }
}
