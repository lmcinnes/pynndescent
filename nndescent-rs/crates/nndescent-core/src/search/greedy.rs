//! Greedy graph search for approximate nearest neighbors.

use crate::distance::Distance;
use crate::graph::SearchGraph;
use crate::heap::{BoundedHeap, CandidateHeap};
use crate::rng::FastRng;
use crate::tree::FlatTree;
use crate::visited::VisitedSet;

/// Greedy search on a k-NN graph.
///
/// # Arguments
/// * `query` - Query point
/// * `data` - All data points (flattened)
/// * `dim` - Dimension of points
/// * `graph` - Search graph (CSR format)
/// * `tree` - Search tree for initialization (optional)
/// * `distance` - Distance function
/// * `k` - Number of neighbors to return
/// * `epsilon` - Search expansion factor (0.0 = exact search on graph, higher = more exploration)
/// * `rng` - Random number generator
///
/// # Returns
/// (indices, distances) of the k nearest neighbors found.
pub fn greedy_search<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    tree: Option<&FlatTree>,
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
) -> (Vec<i32>, Vec<f32>) {
    let n_points = graph.n_vertices;
    let mut visited = VisitedSet::new(n_points);
    let mut result_heap = BoundedHeap::new(k);
    let mut seed_set = CandidateHeap::new();

    // Initialize from tree if available
    if let Some(tree) = tree {
        let (start, end) = tree.search(query, rng);
        for &idx in &tree.indices[start..end] {
            if idx >= 0 && !visited.check_and_mark(idx) {
                let point = &data[idx as usize * dim..(idx as usize + 1) * dim];
                let d = distance.distance(query, point);
                result_heap.push(d, idx);
                seed_set.push(d, idx);
            }
        }
    }

    // Add random seeds if we don't have enough
    let n_initial = seed_set.len();
    let n_random = k.saturating_sub(n_initial);
    for _ in 0..n_random {
        let idx = rng.next_index(n_points) as i32;
        if !visited.check_and_mark(idx) {
            let point = &data[idx as usize * dim..(idx as usize + 1) * dim];
            let d = distance.distance(query, point);
            result_heap.push(d, idx);
            seed_set.push(d, idx);
        }
    }

    // Greedy search
    while let Some((d_vertex, vertex)) = seed_set.pop() {
        // Compute distance bound
        let distance_bound = result_heap.max_distance() + epsilon * (result_heap.max_distance() - min_distance);
        
        if d_vertex >= distance_bound {
            break;
        }

        // Explore neighbors
        for &neighbor in graph.neighbors(vertex as usize) {
            if neighbor < 0 {
                continue;
            }

            if visited.check_and_mark(neighbor) {
                continue;
            }

            let point = &data[neighbor as usize * dim..(neighbor as usize + 1) * dim];
            let d = distance.distance(query, point);

            // Update distance bound
            let distance_bound = result_heap.max_distance() + epsilon * (result_heap.max_distance() - min_distance);

            if d < distance_bound {
                result_heap.push(d, neighbor);
                seed_set.push(d, neighbor);
            }
        }
    }

    result_heap.into_sorted()
}

/// Batch search for multiple queries.
#[cfg(feature = "rayon")]
pub fn batch_search<D: Distance<f32> + Sync>(
    queries: &[f32],
    n_queries: usize,
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    tree: Option<&FlatTree>,
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    seed: u64,
) -> (Vec<i32>, Vec<f32>) {
    use rayon::prelude::*;

    let results: Vec<(Vec<i32>, Vec<f32>)> = (0..n_queries)
        .into_par_iter()
        .map(|i| {
            let query = &queries[i * dim..(i + 1) * dim];
            let mut rng = FastRng::new(seed.wrapping_add(i as u64));
            greedy_search(query, data, dim, graph, tree, distance, k, epsilon, min_distance, &mut rng)
        })
        .collect();

    // Flatten results
    let mut all_indices = Vec::with_capacity(n_queries * k);
    let mut all_distances = Vec::with_capacity(n_queries * k);

    for (indices, distances) in results {
        // Pad with -1 if fewer than k results
        all_indices.extend(indices.iter().copied());
        all_distances.extend(distances.iter().copied());
        
        let padding = k.saturating_sub(indices.len());
        all_indices.extend(std::iter::repeat(-1).take(padding));
        all_distances.extend(std::iter::repeat(f32::INFINITY).take(padding));
    }

    (all_indices, all_distances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::SquaredEuclidean;

    fn create_simple_graph() -> (Vec<f32>, SearchGraph) {
        // 4 points in 2D arranged in a square
        let data = vec![
            0.0, 0.0,  // 0
            1.0, 0.0,  // 1
            0.0, 1.0,  // 2
            1.0, 1.0,  // 3
        ];

        // Each point connected to its neighbors
        let neighbors = vec![
            1, 2,  // 0 -> 1, 2
            0, 3,  // 1 -> 0, 3
            0, 3,  // 2 -> 0, 3
            1, 2,  // 3 -> 1, 2
        ];

        let graph = SearchGraph::from_dense(&neighbors, 4, 2);

        (data, graph)
    }

    #[test]
    fn test_greedy_search_basic() {
        let (data, graph) = create_simple_graph();
        let distance = SquaredEuclidean;
        let mut rng = FastRng::new(42);

        // Query near point 0
        let query = vec![0.1, 0.1];
        let (indices, distances) = greedy_search(
            &query, &data, 2, &graph, None, &distance, 2, 0.1, 0.0, &mut rng,
        );

        // Should find point 0 as closest
        assert!(!indices.is_empty());
        // Point 0 should be among the results
        assert!(indices.contains(&0) || distances[0] < 1.0);
    }

    #[test]
    fn test_greedy_search_exact_point() {
        let (data, graph) = create_simple_graph();
        let distance = SquaredEuclidean;
        let mut rng = FastRng::new(42);

        // Query exactly at point 3
        let query = vec![1.0, 1.0];
        let (indices, distances) = greedy_search(
            &query, &data, 2, &graph, None, &distance, 1, 0.1, 0.0, &mut rng,
        );

        // Should find point 3
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 3);
        assert!((distances[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_greedy_search_returns_k() {
        let (data, graph) = create_simple_graph();
        let distance = SquaredEuclidean;
        let mut rng = FastRng::new(42);

        let query = vec![0.5, 0.5];
        let (indices, distances) = greedy_search(
            &query, &data, 2, &graph, None, &distance, 4, 0.1, 0.0, &mut rng,
        );

        // Should find all 4 points
        assert_eq!(indices.len(), 4);
        assert_eq!(distances.len(), 4);

        // Distances should be sorted
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }
}
