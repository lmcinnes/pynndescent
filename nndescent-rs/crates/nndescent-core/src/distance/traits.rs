//! Distance trait definitions.

/// Core trait for distance computation between vectors.
///
/// Implementations should be thread-safe (Send + Sync) to support
/// parallel computation.
pub trait Distance<T>: Send + Sync + Clone {
    /// Compute the distance between two vectors.
    ///
    /// # Arguments
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    /// The distance as a f32. Lower values indicate more similarity.
    fn distance(&self, a: &[T], b: &[T]) -> f32;

    /// Compute distances from one query to multiple data points.
    ///
    /// Default implementation calls `distance` in a loop.
    /// Override for SIMD-optimized batch computation.
    fn distance_batch(&self, query: &[T], data: &[T], dim: usize, results: &mut [f32]) {
        let n = results.len();
        for i in 0..n {
            let start = i * dim;
            let end = start + dim;
            results[i] = self.distance(query, &data[start..end]);
        }
    }

    /// Whether this distance requires a correction for final output.
    ///
    /// For example, squared Euclidean needs sqrt applied to get true Euclidean.
    fn needs_correction(&self) -> bool {
        false
    }

    /// Apply distance correction.
    ///
    /// Default is identity. Override for metrics like squared Euclidean.
    fn correct(&self, d: f32) -> f32 {
        d
    }

    /// Get the name of this distance metric.
    fn name(&self) -> &'static str;
}

/// Marker trait for distances that have a squared form.
///
/// This enables optimizations where we can skip the final sqrt
/// during search and only apply it to results.
pub trait HasSquaredForm: Distance<f32> {
    /// The squared form of this distance.
    type Squared: Distance<f32>;

    /// Get the squared form of this distance.
    fn squared(&self) -> Self::Squared;

    /// Get the correction function to convert squared distance to true distance.
    fn correction_fn() -> fn(f32) -> f32 {
        |d| d.sqrt()
    }
}
