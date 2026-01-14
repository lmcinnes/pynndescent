//! Main NNDescent index structure and builder.

use crate::distance::{Distance, SquaredEuclidean, Euclidean, Cosine, InnerProduct, Metric};
use crate::graph::{NeighborGraph, SearchGraph};
use crate::heap::NeighborHeap;
use crate::nndescent::{nn_descent, NNDescentParams};
use crate::rng::FastRng;
use crate::search::greedy_search;
use crate::tree::FlatTree;

/// The main NNDescent index for approximate nearest neighbor search.
///
/// This struct holds all the data needed for querying:
/// - The original data points
/// - The search graph (diversified k-NN graph)
/// - Search tree for initialization
/// - Distance function
pub struct NNDescentIndex<D: Distance<f32>> {
    /// Data points (flattened, n_points × dim)
    pub data: Vec<f32>,
    /// Number of data points
    pub n_points: usize,
    /// Dimension of data points
    pub dim: usize,
    /// Distance function
    pub distance: D,
    /// Distance correction function (e.g., sqrt for squared euclidean)
    pub distance_correction: Option<fn(f32) -> f32>,
    /// Number of neighbors in the graph
    pub n_neighbors: usize,
    /// The k-NN graph (indices, shape n_points × n_neighbors)
    pub neighbor_indices: Vec<i32>,
    /// The k-NN graph (distances, shape n_points × n_neighbors)
    pub neighbor_distances: Vec<f32>,
    /// Search graph (CSR format)
    pub search_graph: SearchGraph,
    /// Search tree
    pub search_tree: Option<FlatTree>,
    /// Vertex ordering (for tree leaf order)
    pub vertex_order: Vec<usize>,
    /// Minimum distance in graph (for epsilon scaling)
    pub min_distance: f32,
    /// RNG seed for search
    rng_seed: u64,
}

impl<D: Distance<f32>> NNDescentIndex<D> {
    /// Query for the k nearest neighbors of query points.
    ///
    /// # Arguments
    /// * `queries` - Query points (flattened, n_queries × dim)
    /// * `n_queries` - Number of query points
    /// * `k` - Number of neighbors to return
    /// * `epsilon` - Search expansion factor (0.0-0.5 recommended)
    ///
    /// # Returns
    /// (indices, distances) where:
    /// - indices: shape (n_queries × k), neighbor indices in original data order
    /// - distances: shape (n_queries × k), distances to neighbors
    pub fn query(
        &self,
        queries: &[f32],
        n_queries: usize,
        k: usize,
        epsilon: f32,
    ) -> (Vec<i32>, Vec<f32>) {
        let mut all_indices = Vec::with_capacity(n_queries * k);
        let mut all_distances = Vec::with_capacity(n_queries * k);

        for i in 0..n_queries {
            let query = &queries[i * self.dim..(i + 1) * self.dim];
            let mut rng = FastRng::new(self.rng_seed.wrapping_add(i as u64));

            let (mut indices, mut distances) = greedy_search(
                query,
                &self.data,
                self.dim,
                &self.search_graph,
                self.search_tree.as_ref(),
                &self.distance,
                k,
                epsilon,
                self.min_distance,
                &mut rng,
            );

            // Map back to original vertex order
            for idx in &mut indices {
                if *idx >= 0 {
                    *idx = self.vertex_order[*idx as usize] as i32;
                }
            }

            // Apply distance correction if needed
            if let Some(correction) = self.distance_correction {
                for d in &mut distances {
                    *d = correction(*d);
                }
            }

            // Pad to k if needed
            while indices.len() < k {
                indices.push(-1);
                distances.push(f32::INFINITY);
            }

            all_indices.extend_from_slice(&indices[..k]);
            all_distances.extend_from_slice(&distances[..k]);
        }

        (all_indices, all_distances)
    }

    /// Get the neighbor graph (k-NN graph before search preparation).
    pub fn neighbor_graph(&self) -> Option<NeighborGraph> {
        // This would need to be stored if we want to return it
        None
    }
}

/// Builder for NNDescentIndex.
pub struct NNDescentBuilder<'a> {
    data: &'a [f32],
    n_points: usize,
    dim: usize,
    metric: Metric,
    n_neighbors: usize,
    n_trees: usize,
    leaf_size: Option<usize>,
    max_candidates: Option<usize>,
    n_iters: Option<usize>,
    delta: f32,
    random_seed: u64,
    verbose: bool,
}

impl<'a> NNDescentBuilder<'a> {
    /// Create a new builder.
    ///
    /// # Arguments
    /// * `data` - Flattened data array (n_points × dim)
    /// * `n_points` - Number of data points
    /// * `dim` - Dimension of each point
    pub fn new(data: &'a [f32], n_points: usize, dim: usize) -> Self {
        Self {
            data,
            n_points,
            dim,
            metric: Metric::Euclidean,
            n_neighbors: 30,
            n_trees: 8,
            leaf_size: None,
            max_candidates: None,
            n_iters: None,
            delta: 0.001,
            random_seed: 42,
            verbose: false,
        }
    }

    /// Set the distance metric.
    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the distance metric from a string.
    pub fn metric_str(mut self, metric: &str) -> Self {
        if let Some(m) = Metric::from_str(metric) {
            self.metric = m;
        }
        self
    }

    /// Set the number of neighbors.
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// Set the number of RP trees.
    pub fn n_trees(mut self, n: usize) -> Self {
        self.n_trees = n;
        self
    }

    /// Set the leaf size for RP trees.
    pub fn leaf_size(mut self, size: usize) -> Self {
        self.leaf_size = Some(size);
        self
    }

    /// Set the maximum candidates per iteration.
    pub fn max_candidates(mut self, n: usize) -> Self {
        self.max_candidates = Some(n);
        self
    }

    /// Set the number of NN-descent iterations.
    pub fn n_iters(mut self, n: usize) -> Self {
        self.n_iters = Some(n);
        self
    }

    /// Set the convergence delta.
    pub fn delta(mut self, d: f32) -> Self {
        self.delta = d;
        self
    }

    /// Set the random seed.
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    /// Enable verbose output.
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Build the index with Euclidean distance.
    pub fn build_euclidean(self) -> NNDescentIndex<SquaredEuclidean> {
        self.build_with_distance(SquaredEuclidean, Some(|d: f32| d.sqrt()))
    }

    /// Build the index with Cosine distance.
    pub fn build_cosine(self) -> NNDescentIndex<Cosine> {
        self.build_with_distance(Cosine, None)
    }

    /// Build the index with Inner Product distance.
    pub fn build_inner_product(self) -> NNDescentIndex<InnerProduct> {
        self.build_with_distance(InnerProduct, None)
    }

    /// Build the index with a custom distance function.
    pub fn build_with_distance<D: Distance<f32>>(
        self,
        distance: D,
        correction: Option<fn(f32) -> f32>,
    ) -> NNDescentIndex<D> {
        let angular = matches!(self.metric, Metric::Cosine | Metric::InnerProduct);

        // Compute default parameters based on data size
        let leaf_size = self.leaf_size.unwrap_or_else(|| {
            (5 * self.n_neighbors).min(256).max(60)
        });
        let max_candidates = self.max_candidates.unwrap_or_else(|| {
            self.n_neighbors.min(60)
        });
        let n_iters = self.n_iters.unwrap_or_else(|| {
            ((self.n_points as f64).log2().ceil() as usize).max(5)
        });

        let params = NNDescentParams {
            n_neighbors: self.n_neighbors,
            n_trees: self.n_trees,
            leaf_size,
            max_candidates,
            n_iters,
            delta: self.delta,
            angular,
            max_depth: 200,
            verbose: self.verbose,
        };

        let mut rng = FastRng::new(self.random_seed);

        // Run NN-descent
        let (mut neighbor_heap, forest) = nn_descent(
            self.data,
            self.n_points,
            self.dim,
            &distance,
            &params,
            &mut rng,
        );

        // Sort the heap so neighbors are in ascending distance order
        // (matches PyNNDescent's deheap_sort behavior)
        neighbor_heap.sort_all();

        // Build search graph
        let search_graph = SearchGraph::from_dense(
            &neighbor_heap.indices,
            self.n_points,
            self.n_neighbors,
        );

        // Get min distance
        let min_distance = neighbor_heap.distances
            .iter()
            .filter(|&&d| d > 0.0 && d < f32::INFINITY)
            .copied()
            .fold(f32::INFINITY, f32::min);

        // Apply distance correction to stored distances if needed
        let neighbor_distances = if let Some(corr) = correction {
            neighbor_heap.distances.iter().map(|&d| corr(d)).collect()
        } else {
            neighbor_heap.distances.clone()
        };

        // Use first tree for search
        let search_tree = forest.into_iter().next();

        // For now, identity vertex order
        let vertex_order: Vec<usize> = (0..self.n_points).collect();

        NNDescentIndex {
            data: self.data.to_vec(),
            n_points: self.n_points,
            dim: self.dim,
            distance,
            distance_correction: correction,
            n_neighbors: self.n_neighbors,
            neighbor_indices: neighbor_heap.indices,
            neighbor_distances,
            search_graph,
            search_tree,
            vertex_order,
            min_distance,
            rng_seed: self.random_seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n: usize, dim: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                // Create data with some structure
                data.push(((i * dim + j) as f32 * 0.1).sin());
            }
        }
        data
    }

    #[test]
    fn test_builder_basic() {
        let n = 100;
        let dim = 10;
        let data = create_test_data(n, dim);

        let index = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(10)
            .n_trees(2)
            .n_iters(3)
            .verbose(false)
            .build_euclidean();

        assert_eq!(index.n_points, n);
        assert_eq!(index.dim, dim);
    }

    #[test]
    fn test_query_basic() {
        let n = 100;
        let dim = 10;
        let data = create_test_data(n, dim);

        let index = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(10)
            .n_trees(2)
            .n_iters(3)
            .build_euclidean();

        // Query with first point - should find itself or very close neighbors
        let query = &data[0..dim];
        let (indices, distances) = index.query(query, 1, 5, 0.1);

        assert_eq!(indices.len(), 5);
        assert_eq!(distances.len(), 5);

        // First result should be very close (ideally the query point itself)
        assert!(distances[0] < 1.0);
    }

    #[test]
    fn test_query_multiple() {
        let n = 100;
        let dim = 10;
        let data = create_test_data(n, dim);

        let index = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(10)
            .n_trees(2)
            .n_iters(3)
            .build_euclidean();

        // Query with multiple points
        let queries = &data[0..dim * 3]; // First 3 points
        let (indices, distances) = index.query(queries, 3, 5, 0.1);

        assert_eq!(indices.len(), 15); // 3 queries × 5 neighbors
        assert_eq!(distances.len(), 15);
    }
}
