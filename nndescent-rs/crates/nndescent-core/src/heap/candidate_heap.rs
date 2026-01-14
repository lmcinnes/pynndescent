//! Candidate heap for search operations using std BinaryHeap.

use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// A min-heap of (distance, index) pairs for search candidates.
///
/// Uses `Reverse` to create a min-heap from Rust's max-heap BinaryHeap.
#[derive(Clone, Debug)]
pub struct CandidateHeap {
    heap: BinaryHeap<Reverse<(OrderedFloat<f32>, i32)>>,
}

impl CandidateHeap {
    /// Create a new empty candidate heap.
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    /// Create a new candidate heap with specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
        }
    }

    /// Push a candidate onto the heap.
    #[inline]
    pub fn push(&mut self, distance: f32, index: i32) {
        self.heap.push(Reverse((OrderedFloat(distance), index)));
    }

    /// Pop the minimum distance candidate.
    #[inline]
    pub fn pop(&mut self) -> Option<(f32, i32)> {
        self.heap.pop().map(|Reverse((d, idx))| (d.0, idx))
    }

    /// Peek at the minimum distance candidate without removing it.
    #[inline]
    pub fn peek(&self) -> Option<(f32, i32)> {
        self.heap.peek().map(|Reverse((d, idx))| (d.0, *idx))
    }

    /// Check if the heap is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Get the number of elements in the heap.
    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Clear all elements from the heap.
    #[inline]
    pub fn clear(&mut self) {
        self.heap.clear();
    }
}

impl Default for CandidateHeap {
    fn default() -> Self {
        Self::new()
    }
}

/// A bounded max-heap for maintaining k-nearest neighbors during search.
///
/// Unlike `CandidateHeap`, this maintains a fixed size and returns the
/// maximum distance element, useful for the result heap in search.
#[derive(Clone, Debug)]
pub struct BoundedHeap {
    /// Stored as (distance, index) pairs, max-heap by distance
    heap: BinaryHeap<(OrderedFloat<f32>, i32)>,
    /// Maximum capacity
    k: usize,
}

impl BoundedHeap {
    /// Create a new bounded heap with capacity k.
    pub fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
        }
    }

    /// Push a candidate, maintaining at most k elements.
    /// Returns true if the element was inserted.
    #[inline]
    pub fn push(&mut self, distance: f32, index: i32) -> bool {
        if self.heap.len() < self.k {
            self.heap.push((OrderedFloat(distance), index));
            return true;
        }

        // Only insert if smaller than current max
        if let Some(&(max_d, _)) = self.heap.peek() {
            if distance < max_d.0 {
                self.heap.pop();
                self.heap.push((OrderedFloat(distance), index));
                return true;
            }
        }

        false
    }

    /// Get the maximum distance in the heap (the threshold).
    #[inline]
    pub fn max_distance(&self) -> f32 {
        self.heap
            .peek()
            .map(|(d, _)| d.0)
            .unwrap_or(f32::INFINITY)
    }

    /// Check if the heap is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    /// Get the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Extract all elements sorted by distance (ascending).
    pub fn into_sorted(self) -> (Vec<i32>, Vec<f32>) {
        let mut pairs: Vec<_> = self.heap.into_vec();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));

        let indices: Vec<i32> = pairs.iter().map(|(_, idx)| *idx).collect();
        let distances: Vec<f32> = pairs.iter().map(|(d, _)| d.0).collect();

        (indices, distances)
    }

    /// Clear the heap.
    #[inline]
    pub fn clear(&mut self) {
        self.heap.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_heap_basic() {
        let mut heap = CandidateHeap::new();

        heap.push(0.5, 1);
        heap.push(0.2, 2);
        heap.push(0.8, 3);

        // Should pop in ascending distance order
        assert_eq!(heap.pop(), Some((0.2, 2)));
        assert_eq!(heap.pop(), Some((0.5, 1)));
        assert_eq!(heap.pop(), Some((0.8, 3)));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_candidate_heap_peek() {
        let mut heap = CandidateHeap::new();

        heap.push(0.5, 1);
        heap.push(0.2, 2);

        // Peek should return minimum without removing
        assert_eq!(heap.peek(), Some((0.2, 2)));
        assert_eq!(heap.peek(), Some((0.2, 2)));
        assert_eq!(heap.len(), 2);
    }

    #[test]
    fn test_bounded_heap_basic() {
        let mut heap = BoundedHeap::new(3);

        assert!(heap.push(0.5, 1));
        assert!(heap.push(0.2, 2));
        assert!(heap.push(0.8, 3));

        // Max distance should be 0.8
        assert!((heap.max_distance() - 0.8).abs() < 1e-6);

        // Push smaller should succeed
        assert!(heap.push(0.1, 4));
        // Max distance should now be 0.5
        assert!((heap.max_distance() - 0.5).abs() < 1e-6);

        // Push larger should fail
        assert!(!heap.push(0.9, 5));
    }

    #[test]
    fn test_bounded_heap_sorted_output() {
        let mut heap = BoundedHeap::new(5);

        heap.push(0.5, 5);
        heap.push(0.2, 2);
        heap.push(0.8, 8);
        heap.push(0.1, 1);
        heap.push(0.3, 3);

        let (indices, distances) = heap.into_sorted();

        // Should be sorted by distance ascending
        assert_eq!(indices, vec![1, 2, 3, 5, 8]);
        assert!((distances[0] - 0.1).abs() < 1e-6);
        assert!((distances[4] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_bounded_heap_overflow() {
        let mut heap = BoundedHeap::new(3);

        heap.push(0.5, 1);
        heap.push(0.6, 2);
        heap.push(0.7, 3);

        // Now full, push smaller values
        heap.push(0.1, 4);
        heap.push(0.2, 5);
        heap.push(0.3, 6);

        let (indices, distances) = heap.into_sorted();

        // Should have the 3 smallest: 0.1, 0.2, 0.3
        assert_eq!(indices.len(), 3);
        assert!((distances[0] - 0.1).abs() < 1e-6);
        assert!((distances[1] - 0.2).abs() < 1e-6);
        assert!((distances[2] - 0.3).abs() < 1e-6);
    }
}
