//! Neighbor graph representation.
//!
//! This stores the k-nearest neighbor graph in a format suitable for
//! both construction (via NeighborHeap) and output.

use crate::heap::NeighborHeap;

/// A k-nearest neighbor graph stored as dense arrays.
///
/// Each point has exactly k neighbors stored in row-major order.
#[derive(Clone, Debug)]
pub struct NeighborGraph {
    /// Neighbor indices, shape (n_points × k)
    pub indices: Vec<i32>,
    /// Distances to neighbors, shape (n_points × k)
    pub distances: Vec<f32>,
    /// Number of points
    pub n_points: usize,
    /// Number of neighbors per point
    pub k: usize,
}

impl NeighborGraph {
    /// Create a new neighbor graph with uninitialized values.
    pub fn new(n_points: usize, k: usize) -> Self {
        Self {
            indices: vec![-1; n_points * k],
            distances: vec![f32::INFINITY; n_points * k],
            n_points,
            k,
        }
    }

    /// Create from a NeighborHeap, applying distance correction and sorting.
    pub fn from_heap(heap: &NeighborHeap, correction: Option<fn(f32) -> f32>) -> Self {
        let n_points = heap.n_points;
        let k = heap.k;
        let mut result = Self::new(n_points, k);

        for point in 0..n_points {
            let (indices, distances) = heap.deheap_sort(point);

            let offset = point * k;
            for (i, (&idx, &dist)) in indices.iter().zip(distances.iter()).enumerate() {
                result.indices[offset + i] = idx;
                result.distances[offset + i] = match correction {
                    Some(f) => f(dist),
                    None => dist,
                };
            }
        }

        result
    }

    /// Get neighbors for a point.
    #[inline]
    pub fn get_neighbors(&self, point: usize) -> (&[i32], &[f32]) {
        let start = point * self.k;
        let end = start + self.k;
        (&self.indices[start..end], &self.distances[start..end])
    }

    /// Get the neighbor index at a specific position.
    #[inline]
    pub fn get_neighbor(&self, point: usize, pos: usize) -> (i32, f32) {
        let idx = point * self.k + pos;
        (self.indices[idx], self.distances[idx])
    }

    /// Reorder graph according to a permutation.
    ///
    /// This updates indices to point to the new locations after reordering.
    pub fn reorder(&mut self, order: &[usize]) {
        let n = self.n_points;
        let k = self.k;

        // Create inverse mapping: new_position -> old_position becomes old_position -> new_position
        let mut inverse_order = vec![0usize; n];
        for (new_pos, &old_pos) in order.iter().enumerate() {
            inverse_order[old_pos] = new_pos;
        }

        // Create new arrays
        let mut new_indices = vec![-1i32; n * k];
        let mut new_distances = vec![f32::INFINITY; n * k];

        for new_point in 0..n {
            let old_point = order[new_point];
            let old_offset = old_point * k;
            let new_offset = new_point * k;

            for i in 0..k {
                let old_neighbor = self.indices[old_offset + i];
                let dist = self.distances[old_offset + i];

                if old_neighbor >= 0 {
                    // Map old neighbor index to new index
                    let new_neighbor = inverse_order[old_neighbor as usize] as i32;
                    new_indices[new_offset + i] = new_neighbor;
                } else {
                    new_indices[new_offset + i] = old_neighbor;
                }
                new_distances[new_offset + i] = dist;
            }
        }

        self.indices = new_indices;
        self.distances = new_distances;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_graph() {
        let graph = NeighborGraph::new(10, 5);
        assert_eq!(graph.n_points, 10);
        assert_eq!(graph.k, 5);
        assert_eq!(graph.indices.len(), 50);
        assert_eq!(graph.distances.len(), 50);
    }

    #[test]
    fn test_from_heap() {
        let mut heap = NeighborHeap::new(2, 3);

        // Point 0
        heap.simple_push(0, 1, 0.1);
        heap.simple_push(0, 2, 0.2);
        heap.simple_push(0, 3, 0.3);

        // Point 1
        heap.simple_push(1, 0, 0.1);
        heap.simple_push(1, 2, 0.4);
        heap.simple_push(1, 3, 0.5);

        let graph = NeighborGraph::from_heap(&heap, None);

        // Check point 0 neighbors are sorted
        let (indices, distances) = graph.get_neighbors(0);
        assert_eq!(indices[0], 1); // smallest distance
        assert!((distances[0] - 0.1).abs() < 1e-6);

        // Check point 1 neighbors are sorted
        let (indices, distances) = graph.get_neighbors(1);
        assert_eq!(indices[0], 0); // smallest distance
        assert!((distances[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_from_heap_with_correction() {
        let mut heap = NeighborHeap::new(1, 2);
        heap.simple_push(0, 1, 4.0);
        heap.simple_push(0, 2, 9.0);

        let graph = NeighborGraph::from_heap(&heap, Some(|d| d.sqrt()));

        let (_, distances) = graph.get_neighbors(0);
        assert!((distances[0] - 2.0).abs() < 1e-6);
        assert!((distances[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_reorder() {
        let mut graph = NeighborGraph::new(3, 2);

        // Point 0's neighbors: [1, 2]
        graph.indices[0] = 1;
        graph.indices[1] = 2;
        graph.distances[0] = 0.1;
        graph.distances[1] = 0.2;

        // Point 1's neighbors: [0, 2]
        graph.indices[2] = 0;
        graph.indices[3] = 2;
        graph.distances[2] = 0.1;
        graph.distances[3] = 0.3;

        // Point 2's neighbors: [0, 1]
        graph.indices[4] = 0;
        graph.indices[5] = 1;
        graph.distances[4] = 0.2;
        graph.distances[5] = 0.3;

        // Reorder: old position -> new position
        // order[new] = old, so order = [2, 0, 1] means:
        // new 0 = old 2, new 1 = old 0, new 2 = old 1
        let order = vec![2, 0, 1];
        graph.reorder(&order);

        // After reorder:
        // New point 0 (was point 2) should have neighbors pointing to new indices
        // Old point 2's neighbors were [0, 1] -> new indices [1, 2]
        let (indices, _) = graph.get_neighbors(0);
        assert!(indices.contains(&1) && indices.contains(&2));
    }
}
