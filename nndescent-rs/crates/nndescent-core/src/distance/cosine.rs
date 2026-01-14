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

impl Distance<f32> for Cosine {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        // Compute dot product and norms in a single pass
        // Process in chunks of 4 for better pipelining
        let chunks = a.len() / 4;
        let remainder = a.len() % 4;

        for i in 0..chunks {
            let idx = i * 4;

            dot += a[idx] * b[idx]
                + a[idx + 1] * b[idx + 1]
                + a[idx + 2] * b[idx + 2]
                + a[idx + 3] * b[idx + 3];

            norm_a += a[idx] * a[idx]
                + a[idx + 1] * a[idx + 1]
                + a[idx + 2] * a[idx + 2]
                + a[idx + 3] * a[idx + 3];

            norm_b += b[idx] * b[idx]
                + b[idx + 1] * b[idx + 1]
                + b[idx + 2] * b[idx + 2]
                + b[idx + 3] * b[idx + 3];
        }

        // Handle remainder
        let start = chunks * 4;
        for i in 0..remainder {
            dot += a[start + i] * b[start + i];
            norm_a += a[start + i] * a[start + i];
            norm_b += b[start + i] * b[start + i];
        }

        // Avoid division by zero
        let denom = (norm_a * norm_b).sqrt();
        if denom < 1e-12 {
            return 1.0; // Maximum distance for zero vectors
        }

        // Clamp to valid range to handle floating point errors
        let similarity = (dot / denom).clamp(-1.0, 1.0);
        1.0 - similarity
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
