//! Bit-packed visited set for efficient tracking during graph search.
//!
//! Uses a single bit per element to track whether a node has been visited,
//! matching PyNNDescent's visited set implementation.

/// A bit-packed set for tracking visited nodes during search.
///
/// Each bit represents whether the corresponding node index has been visited.
/// This is much more memory-efficient than a `HashSet` for dense indices.
#[derive(Clone)]
pub struct VisitedSet {
    bits: Vec<u8>,
    n_elements: usize,
}

impl VisitedSet {
    /// Create a new visited set that can track `n` elements.
    pub fn new(n: usize) -> Self {
        let n_bytes = (n >> 3) + 1;
        Self {
            bits: vec![0u8; n_bytes],
            n_elements: n,
        }
    }

    /// Check if an index has been visited.
    #[inline]
    pub fn is_visited(&self, idx: i32) -> bool {
        debug_assert!(idx >= 0 && (idx as usize) < self.n_elements);
        let loc = (idx >> 3) as usize;
        let mask = 1u8 << (idx & 7);
        (self.bits[loc] & mask) != 0
    }

    /// Mark an index as visited.
    #[inline]
    pub fn mark(&mut self, idx: i32) {
        debug_assert!(idx >= 0 && (idx as usize) < self.n_elements);
        let loc = (idx >> 3) as usize;
        let mask = 1u8 << (idx & 7);
        self.bits[loc] |= mask;
    }

    /// Check if visited and mark in one operation.
    /// Returns `true` if the index was already visited, `false` otherwise.
    ///
    /// This matches PyNNDescent's `check_and_mark_visited` function.
    #[inline]
    pub fn check_and_mark(&mut self, idx: i32) -> bool {
        debug_assert!(idx >= 0 && (idx as usize) < self.n_elements);
        let loc = (idx >> 3) as usize;
        let mask = 1u8 << (idx & 7);
        let was_visited = (self.bits[loc] & mask) != 0;
        self.bits[loc] |= mask;
        was_visited
    }

    /// Clear all visited flags.
    #[inline]
    pub fn clear(&mut self) {
        self.bits.fill(0);
    }

    /// Get the number of elements this set can track.
    pub fn capacity(&self) -> usize {
        self.n_elements
    }
}

impl std::fmt::Debug for VisitedSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisitedSet")
            .field("n_elements", &self.n_elements)
            .field("n_bytes", &self.bits.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut visited = VisitedSet::new(100);
        
        assert!(!visited.is_visited(0));
        assert!(!visited.is_visited(50));
        assert!(!visited.is_visited(99));
        
        visited.mark(50);
        assert!(!visited.is_visited(0));
        assert!(visited.is_visited(50));
        assert!(!visited.is_visited(99));
    }

    #[test]
    fn test_check_and_mark() {
        let mut visited = VisitedSet::new(100);
        
        // First check should return false and mark
        assert!(!visited.check_and_mark(42));
        
        // Second check should return true (already visited)
        assert!(visited.check_and_mark(42));
        
        // Different index should return false
        assert!(!visited.check_and_mark(43));
    }

    #[test]
    fn test_clear() {
        let mut visited = VisitedSet::new(100);
        
        visited.mark(10);
        visited.mark(20);
        visited.mark(30);
        
        assert!(visited.is_visited(10));
        assert!(visited.is_visited(20));
        assert!(visited.is_visited(30));
        
        visited.clear();
        
        assert!(!visited.is_visited(10));
        assert!(!visited.is_visited(20));
        assert!(!visited.is_visited(30));
    }

    #[test]
    fn test_boundary_indices() {
        let mut visited = VisitedSet::new(256);
        
        // Test byte boundaries
        for i in [0, 7, 8, 15, 16, 255] {
            assert!(!visited.check_and_mark(i));
            assert!(visited.is_visited(i));
        }
    }

    #[test]
    fn test_large_set() {
        let mut visited = VisitedSet::new(1_000_000);
        
        // Mark every 1000th element
        for i in (0..1_000_000).step_by(1000) {
            visited.mark(i as i32);
        }
        
        // Verify
        for i in 0..1_000_000 {
            let expected = i % 1000 == 0;
            assert_eq!(visited.is_visited(i as i32), expected);
        }
    }
}
