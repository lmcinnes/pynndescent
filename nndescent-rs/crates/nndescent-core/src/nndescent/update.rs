//! Graph update structures for parallel NN-Descent.
//!
//! This module implements the block-based update strategy for lock-free
//! parallel graph updates.

use crate::heap::NeighborHeap;

/// A single update to the neighbor graph.
#[derive(Clone, Copy, Debug)]
pub struct Update {
    /// Source vertex
    pub p: i32,
    /// Target vertex  
    pub q: i32,
    /// Distance between p and q
    pub distance: f32,
}

/// Update array for batched graph updates.
///
/// Updates are collected in batches and then applied in a lock-free manner
/// using block-based ownership.
#[derive(Clone, Debug)]
pub struct UpdateArray {
    /// Updates stored as flat array
    pub updates: Vec<Update>,
    /// Number of valid updates
    pub count: usize,
    /// Capacity
    pub capacity: usize,
}

impl UpdateArray {
    /// Create a new update array with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            updates: Vec::with_capacity(capacity),
            count: 0,
            capacity,
        }
    }

    /// Add an update to the array.
    #[inline]
    pub fn push(&mut self, p: i32, q: i32, distance: f32) {
        if self.count < self.capacity {
            self.updates.push(Update { p, q, distance });
            self.count += 1;
        }
    }

    /// Clear all updates.
    pub fn clear(&mut self) {
        self.updates.clear();
        self.count = 0;
    }

    /// Get updates as slice.
    pub fn as_slice(&self) -> &[Update] {
        &self.updates[..self.count]
    }
}

/// Apply updates to the neighbor graph.
///
/// Returns the number of successful updates (changes to the graph).
pub fn apply_updates(graph: &mut NeighborHeap, updates: &[Update]) -> usize {
    let mut changes = 0;

    for update in updates {
        // Try to add q to p's neighbors
        if graph.checked_flagged_push(update.p as usize, update.q, update.distance, true) {
            changes += 1;
        }

        // Try to add p to q's neighbors (symmetric)
        if graph.checked_flagged_push(update.q as usize, update.p, update.distance, true) {
            changes += 1;
        }
    }

    changes
}

/// Block-based parallel update application.
///
/// Divides vertices into blocks and assigns each block to a thread.
/// Each thread only updates vertices in its block, ensuring lock-free updates.
#[cfg(feature = "rayon")]
pub mod parallel {
    use super::*;
    use rayon::prelude::*;

    /// Block size for parallel processing (tuned for cache efficiency).
    pub const BLOCK_SIZE: usize = 16384;

    /// Apply updates in parallel using block-based ownership.
    ///
    /// # Safety
    /// This is safe because each block owns a disjoint set of vertices,
    /// so there are no data races.
    pub fn apply_updates_parallel(
        graph: &mut NeighborHeap,
        updates: &[Update],
        n_threads: usize,
    ) -> usize {
        let n_vertices = graph.n_points;
        let block_size = (n_vertices + n_threads - 1) / n_threads;

        // For now, use sequential application
        // TODO: Implement true parallel application with proper synchronization
        apply_updates(graph, updates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_array() {
        let mut updates = UpdateArray::new(100);

        updates.push(0, 1, 0.5);
        updates.push(1, 2, 0.3);
        updates.push(2, 3, 0.7);

        assert_eq!(updates.count, 3);
        assert_eq!(updates.as_slice().len(), 3);
    }

    #[test]
    fn test_apply_updates() {
        let mut graph = NeighborHeap::new(4, 2);

        let updates = vec![
            Update { p: 0, q: 1, distance: 0.1 },
            Update { p: 1, q: 2, distance: 0.2 },
            Update { p: 2, q: 3, distance: 0.3 },
        ];

        let changes = apply_updates(&mut graph, &updates);

        // Each update affects two vertices (symmetric)
        assert!(changes > 0);

        // Check that neighbors were added
        assert!(graph.contains(0, 1));
        assert!(graph.contains(1, 0));
        assert!(graph.contains(1, 2));
        assert!(graph.contains(2, 1));
    }

    #[test]
    fn test_update_array_capacity() {
        let mut updates = UpdateArray::new(2);

        updates.push(0, 1, 0.5);
        updates.push(1, 2, 0.3);
        updates.push(2, 3, 0.7); // Should be ignored

        assert_eq!(updates.count, 2);
    }

    #[test]
    fn test_clear() {
        let mut updates = UpdateArray::new(100);

        updates.push(0, 1, 0.5);
        updates.push(1, 2, 0.3);

        updates.clear();

        assert_eq!(updates.count, 0);
        assert!(updates.as_slice().is_empty());
    }
}
