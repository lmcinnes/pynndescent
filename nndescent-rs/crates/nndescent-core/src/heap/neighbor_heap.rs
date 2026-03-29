//! 3-component neighbor heap matching PyNNDescent's heap structure.
//!
//! The heap maintains k-nearest neighbors per point with:
//! - Neighbor indices (i32)
//! - Distances (f32)
//! - Flags (u8) for new/old tracking in NN-Descent

/// A max-heap structure for maintaining k-nearest neighbors.
///
/// This is a 3-component heap matching PyNNDescent's structure:
/// - `indices`: neighbor indices (n_points × k)
/// - `distances`: distances to neighbors (n_points × k)
/// - `flags`: new/old flags (n_points × k), 1 = new, 0 = old
///
/// The heap is organized as a max-heap based on distances, so the
/// largest distance is always at position 0 for each row.
#[derive(Clone, Debug)]
pub struct NeighborHeap {
    /// Neighbor indices, shape (n_points × k), row-major
    pub indices: Vec<i32>,
    /// Distances to neighbors, shape (n_points × k), row-major
    pub distances: Vec<f32>,
    /// New/old flags, shape (n_points × k), 1 = new, 0 = old
    pub flags: Vec<u8>,
    /// Number of points
    pub n_points: usize,
    /// Number of neighbors per point (k)
    pub k: usize,
}

impl NeighborHeap {
    /// Create a new neighbor heap initialized with infinity distances.
    ///
    /// Matches PyNNDescent's `make_heap` function.
    pub fn new(n_points: usize, k: usize) -> Self {
        let size = n_points * k;
        Self {
            indices: vec![-1; size],
            distances: vec![f32::INFINITY; size],
            flags: vec![0; size],
            n_points,
            k,
        }
    }

    /// Get the offset into the flat arrays for a given point.
    #[inline]
    fn offset(&self, point: usize) -> usize {
        point * self.k
    }

    /// Get the maximum distance (root of heap) for a point.
    #[inline]
    pub fn max_distance(&self, point: usize) -> f32 {
        self.distances[self.offset(point)]
    }

    /// Get the neighbor at a given position for a point.
    #[inline]
    pub fn get(&self, point: usize, pos: usize) -> (i32, f32, bool) {
        let idx = self.offset(point) + pos;
        (self.indices[idx], self.distances[idx], self.flags[idx] != 0)
    }

    /// Get all neighbors for a point as slices.
    #[inline]
    pub fn get_row(&self, point: usize) -> (&[i32], &[f32], &[u8]) {
        let start = self.offset(point);
        let end = start + self.k;
        (
            &self.indices[start..end],
            &self.distances[start..end],
            &self.flags[start..end],
        )
    }

    /// Get mutable slices for a point's neighbors.
    #[inline]
    pub fn get_row_mut(&mut self, point: usize) -> (&mut [i32], &mut [f32], &mut [u8]) {
        let start = self.offset(point);
        let end = start + self.k;
        let (indices, rest) = self.indices.split_at_mut(end);
        let _ = rest; // silence warning
        let indices = &mut indices[start..];
        
        let (distances, rest) = self.distances.split_at_mut(end);
        let _ = rest;
        let distances = &mut distances[start..];
        
        let (flags, _) = self.flags.split_at_mut(end);
        let flags = &mut flags[start..];
        
        (indices, distances, flags)
    }

    /// Simple heap push without duplicate checking.
    ///
    /// Matches PyNNDescent's `simple_heap_push`.
    #[inline]
    pub fn simple_push(&mut self, point: usize, neighbor: i32, distance: f32) -> bool {
        let offset = self.offset(point);
        
        // Check if the new distance is smaller than the max (root)
        if distance >= self.distances[offset] {
            return false;
        }
        
        // Replace root and sift down
        self.distances[offset] = distance;
        self.indices[offset] = neighbor;
        self.flags[offset] = 0;
        
        self.sift_down(offset, 0);
        true
    }

    /// Heap push with duplicate checking.
    ///
    /// Matches PyNNDescent's `checked_heap_push`.
    #[inline]
    pub fn checked_push(&mut self, point: usize, neighbor: i32, distance: f32) -> bool {
        let offset = self.offset(point);
        
        // Check if the new distance is smaller than the max (root)
        if distance >= self.distances[offset] {
            return false;
        }
        
        // Check for duplicates
        for i in 0..self.k {
            if self.indices[offset + i] == neighbor {
                return false;
            }
        }
        
        // Replace root and sift down
        self.distances[offset] = distance;
        self.indices[offset] = neighbor;
        self.flags[offset] = 0;
        
        self.sift_down(offset, 0);
        true
    }

    /// Heap push with duplicate checking and flag setting.
    ///
    /// Matches PyNNDescent's `checked_flagged_heap_push`.
    #[inline]
    pub fn checked_flagged_push(
        &mut self,
        point: usize,
        neighbor: i32,
        distance: f32,
        is_new: bool,
    ) -> bool {
        let offset = self.offset(point);
        
        // Check if the new distance is smaller than the max (root)
        if distance >= self.distances[offset] {
            return false;
        }
        
        // Check for duplicates
        for i in 0..self.k {
            if self.indices[offset + i] == neighbor {
                return false;
            }
        }
        
        // Replace root and sift down
        self.distances[offset] = distance;
        self.indices[offset] = neighbor;
        self.flags[offset] = is_new as u8;
        
        self.sift_down(offset, 0);
        true
    }

    /// Unconditional push with flag (used during initialization).
    ///
    /// Matches PyNNDescent's `unchecked_heap_push`.
    #[inline]
    pub fn unchecked_flagged_push(
        &mut self,
        point: usize,
        neighbor: i32,
        distance: f32,
        is_new: bool,
    ) {
        let offset = self.offset(point);
        
        // Replace root and sift down
        self.distances[offset] = distance;
        self.indices[offset] = neighbor;
        self.flags[offset] = is_new as u8;
        
        self.sift_down(offset, 0);
    }

    /// Sift down operation for the heap at given offset.
    ///
    /// Uses the "shift-down" technique matching PyNNDescent's `siftdown`:
    /// instead of swapping parent↔child (3 ops per level × 3 arrays = 9 ops),
    /// copies child→parent and places the new value once at final position
    /// (2 ops per level × 3 arrays = 6 ops, plus 1 final write × 3 = 3 ops).
    #[inline]
    fn sift_down(&mut self, offset: usize, start_pos: usize) {
        let end = self.k;
        // Save the value being sifted
        let val_dist = self.distances[offset + start_pos];
        let val_idx = self.indices[offset + start_pos];
        let val_flag = self.flags[offset + start_pos];
        
        let mut pos = start_pos;
        let mut child = 2 * pos + 1;
        
        while child < end {
            let right = child + 1;
            
            // Find the larger child
            if right < end && self.distances[offset + child] < self.distances[offset + right] {
                child = right;
            }
            
            // If the value is larger than or equal to the larger child, we're done
            if val_dist >= self.distances[offset + child] {
                break;
            }
            
            // Move child up to parent position
            self.distances[offset + pos] = self.distances[offset + child];
            self.indices[offset + pos] = self.indices[offset + child];
            self.flags[offset + pos] = self.flags[offset + child];
            
            pos = child;
            child = 2 * pos + 1;
        }
        
        // Place value at final position
        self.distances[offset + pos] = val_dist;
        self.indices[offset + pos] = val_idx;
        self.flags[offset + pos] = val_flag;
    }

    /// Mark all neighbors for a point as "old".
    #[inline]
    pub fn mark_all_old(&mut self, point: usize) {
        let offset = self.offset(point);
        for i in 0..self.k {
            self.flags[offset + i] = 0;
        }
    }

    /// Mark a specific neighbor position as "old".
    #[inline]
    pub fn mark_old(&mut self, point: usize, neighbor_pos: usize) {
        let offset = self.offset(point);
        self.flags[offset + neighbor_pos] = 0;
    }

    /// Count the number of "new" neighbors for a point.
    #[inline]
    pub fn count_new(&self, point: usize) -> usize {
        let offset = self.offset(point);
        self.flags[offset..offset + self.k]
            .iter()
            .filter(|&&f| f != 0)
            .count()
    }

    /// Sort heap entries by distance (ascending) and return sorted arrays.
    ///
    /// Matches PyNNDescent's `deheap_sort`.
    pub fn deheap_sort(&self, point: usize) -> (Vec<i32>, Vec<f32>) {
        let (indices, distances, _) = self.get_row(point);
        
        // Create pairs and sort by distance
        let mut pairs: Vec<(f32, i32)> = distances
            .iter()
            .copied()
            .zip(indices.iter().copied())
            .collect();
        
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let sorted_indices: Vec<i32> = pairs.iter().map(|p| p.1).collect();
        let sorted_distances: Vec<f32> = pairs.iter().map(|p| p.0).collect();
        
        (sorted_indices, sorted_distances)
    }

    /// Sort all rows in-place by distance (ascending).
    pub fn sort_all(&mut self) {
        for point in 0..self.n_points {
            self.sort_row(point);
        }
    }

    /// Sort a single row in-place by distance (ascending).
    fn sort_row(&mut self, point: usize) {
        let offset = self.offset(point);
        
        // Create pairs with indices
        let mut pairs: Vec<(f32, i32, u8)> = (0..self.k)
            .map(|i| {
                let idx = offset + i;
                (self.distances[idx], self.indices[idx], self.flags[idx])
            })
            .collect();
        
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Write back
        for (i, (d, idx, f)) in pairs.into_iter().enumerate() {
            let pos = offset + i;
            self.distances[pos] = d;
            self.indices[pos] = idx;
            self.flags[pos] = f;
        }
    }

    /// Check if a neighbor exists for a point.
    #[inline]
    pub fn contains(&self, point: usize, neighbor: i32) -> bool {
        let offset = self.offset(point);
        for i in 0..self.k {
            if self.indices[offset + i] == neighbor {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_heap() {
        let heap = NeighborHeap::new(10, 5);
        assert_eq!(heap.n_points, 10);
        assert_eq!(heap.k, 5);
        assert_eq!(heap.indices.len(), 50);
        assert_eq!(heap.distances.len(), 50);
        assert_eq!(heap.flags.len(), 50);
        
        // All should be initialized to -1/infinity/0
        assert!(heap.indices.iter().all(|&x| x == -1));
        assert!(heap.distances.iter().all(|&x| x == f32::INFINITY));
        assert!(heap.flags.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_simple_push() {
        let mut heap = NeighborHeap::new(1, 3);
        
        // Push three elements
        assert!(heap.simple_push(0, 5, 1.0));
        assert!(heap.simple_push(0, 3, 0.5));
        assert!(heap.simple_push(0, 7, 0.8));
        
        // Max should be 1.0
        assert_eq!(heap.max_distance(0), 1.0);
        
        // Push a smaller distance
        assert!(heap.simple_push(0, 9, 0.3));
        
        // Max should now be 0.8 (1.0 was replaced)
        assert!((heap.max_distance(0) - 0.8).abs() < 1e-6);
        
        // Pushing a larger distance should fail
        assert!(!heap.simple_push(0, 11, 2.0));
    }

    #[test]
    fn test_checked_push_no_duplicates() {
        let mut heap = NeighborHeap::new(1, 3);
        
        assert!(heap.checked_push(0, 5, 1.0));
        assert!(heap.checked_push(0, 3, 0.5));
        
        // Duplicate should be rejected
        assert!(!heap.checked_push(0, 5, 0.1));
        
        // Same neighbor with larger distance should also be rejected
        assert!(!heap.checked_push(0, 3, 0.3));
    }

    #[test]
    fn test_flagged_push() {
        let mut heap = NeighborHeap::new(1, 3);
        
        heap.checked_flagged_push(0, 5, 1.0, true);
        heap.checked_flagged_push(0, 3, 0.5, false);
        heap.checked_flagged_push(0, 7, 0.8, true);
        
        assert_eq!(heap.count_new(0), 2);
        
        heap.mark_all_old(0);
        assert_eq!(heap.count_new(0), 0);
    }

    #[test]
    fn test_deheap_sort() {
        let mut heap = NeighborHeap::new(1, 5);
        
        heap.simple_push(0, 1, 0.5);
        heap.simple_push(0, 2, 0.2);
        heap.simple_push(0, 3, 0.8);
        heap.simple_push(0, 4, 0.1);
        heap.simple_push(0, 5, 0.3);
        
        let (indices, distances) = heap.deheap_sort(0);
        
        // Should be sorted by distance ascending
        assert_eq!(indices, vec![4, 2, 5, 1, 3]);
        assert!((distances[0] - 0.1).abs() < 1e-6);
        assert!((distances[4] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_heap_property() {
        let mut heap = NeighborHeap::new(1, 7);
        
        // Insert in random order
        let values = [(5, 0.5), (2, 0.2), (8, 0.8), (1, 0.1), (9, 0.9), (3, 0.3), (7, 0.7)];
        for (idx, dist) in values {
            heap.simple_push(0, idx, dist);
        }
        
        // Verify heap property: parent >= children
        let offset = heap.offset(0);
        for i in 0..heap.k {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < heap.k {
                assert!(heap.distances[offset + i] >= heap.distances[offset + left]);
            }
            if right < heap.k {
                assert!(heap.distances[offset + i] >= heap.distances[offset + right]);
            }
        }
    }

    #[test]
    fn test_multiple_points() {
        let mut heap = NeighborHeap::new(3, 2);
        
        // Point 0
        heap.simple_push(0, 1, 0.1);
        heap.simple_push(0, 2, 0.2);
        
        // Point 1
        heap.simple_push(1, 3, 0.3);
        heap.simple_push(1, 4, 0.4);
        
        // Point 2
        heap.simple_push(2, 5, 0.5);
        heap.simple_push(2, 6, 0.6);
        
        // Check that points are independent
        assert!((heap.max_distance(0) - 0.2).abs() < 1e-6);
        assert!((heap.max_distance(1) - 0.4).abs() < 1e-6);
        assert!((heap.max_distance(2) - 0.6).abs() < 1e-6);
    }
}
