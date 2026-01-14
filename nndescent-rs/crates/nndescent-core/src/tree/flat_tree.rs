//! Flat (compressed) random projection tree structure.
//!
//! The flat tree format is optimized for search operations, storing
//! all tree data in contiguous arrays.

use crate::rng::FastRng;

/// A flattened random projection tree for efficient search.
///
/// This matches PyNNDescent's FlatTree structure.
#[derive(Clone, Debug)]
pub struct FlatTree {
    /// Hyperplane vectors for each internal node, shape (n_nodes × dim)
    pub hyperplanes: Vec<f32>,
    /// Hyperplane offsets for each internal node
    pub offsets: Vec<f32>,
    /// Child pointers: children[node] = [left_child, right_child]
    /// Negative values indicate leaves: -children[node] gives the range
    pub children: Vec<[i32; 2]>,
    /// Point indices in leaf order
    pub indices: Vec<i32>,
    /// Dimension of data
    pub dim: usize,
    /// Number of nodes in the tree
    pub n_nodes: usize,
}

impl FlatTree {
    /// Create a new empty flat tree.
    pub fn new(dim: usize) -> Self {
        Self {
            hyperplanes: Vec::new(),
            offsets: Vec::new(),
            children: Vec::new(),
            indices: Vec::new(),
            dim,
            n_nodes: 0,
        }
    }

    /// Search the tree to find the leaf containing a query point.
    ///
    /// Returns (start, end) indices into `self.indices` for the leaf.
    #[inline]
    pub fn search(&self, point: &[f32], rng: &mut FastRng) -> (usize, usize) {
        let mut node = 0usize;

        while self.children[node][0] > 0 {
            // Get hyperplane for this node
            let hp_start = node * self.dim;
            let hp_end = hp_start + self.dim;
            let hyperplane = &self.hyperplanes[hp_start..hp_end];
            let offset = self.offsets[node];

            // Compute margin (dot product with hyperplane + offset)
            let mut margin = offset;
            for i in 0..self.dim {
                margin += point[i] * hyperplane[i];
            }

            // Choose side (with random tie-breaking)
            let side = if margin.abs() < 1e-8 {
                rng.next_bool() as usize
            } else {
                (margin > 0.0) as usize
            };

            node = self.children[node][side] as usize;
        }

        // At a leaf node - extract bounds from children
        // children[node] = [-start, -end]
        let start = (-self.children[node][0]) as usize;
        let end = (-self.children[node][1]) as usize;

        (start, end)
    }

    /// Search tree for angular data (using normalized hyperplane comparison).
    #[inline]
    pub fn search_angular(&self, point: &[f32], rng: &mut FastRng) -> (usize, usize) {
        // For angular trees, the search is the same - normalization happens during construction
        self.search(point, rng)
    }

    /// Get the leaf indices for a query.
    pub fn get_leaf_indices(&self, point: &[f32], rng: &mut FastRng) -> &[i32] {
        let (start, end) = self.search(point, rng);
        &self.indices[start..end]
    }

    /// Get all leaf boundaries.
    ///
    /// Returns a vector of (start, end) pairs for each leaf.
    pub fn get_all_leaves(&self) -> Vec<(usize, usize)> {
        let mut leaves = Vec::new();
        self.collect_leaves(0, &mut leaves);
        leaves
    }

    fn collect_leaves(&self, node: usize, leaves: &mut Vec<(usize, usize)>) {
        if self.children[node][0] <= 0 {
            // Leaf node
            let start = (-self.children[node][0]) as usize;
            let end = (-self.children[node][1]) as usize;
            leaves.push((start, end));
        } else {
            // Internal node - recurse
            self.collect_leaves(self.children[node][0] as usize, leaves);
            self.collect_leaves(self.children[node][1] as usize, leaves);
        }
    }
}

/// Select which side of a hyperplane a point falls on.
///
/// Returns 0 for left, 1 for right.
#[inline]
pub fn select_side(hyperplane: &[f32], offset: f32, point: &[f32], rng: &mut FastRng) -> usize {
    let mut margin = offset;
    for i in 0..hyperplane.len() {
        margin += point[i] * hyperplane[i];
    }

    if margin.abs() < 1e-8 {
        rng.next_bool() as usize
    } else {
        (margin > 0.0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tree() -> FlatTree {
        // Create a simple tree with 3 nodes (1 internal + 2 leaves)
        // dim = 2
        let dim = 2;
        let mut tree = FlatTree::new(dim);

        // Root node (0): hyperplane [1, 0], offset = 0
        tree.hyperplanes.extend_from_slice(&[1.0, 0.0]);
        tree.offsets.push(0.0);
        tree.children.push([1, 2]); // left child = node 1, right child = node 2

        // Left leaf (node 1): points 0, 1
        tree.hyperplanes.extend_from_slice(&[0.0, 0.0]); // unused for leaf
        tree.offsets.push(0.0);
        tree.children.push([-0, -2]); // indices[0:2]

        // Right leaf (node 2): points 2, 3
        tree.hyperplanes.extend_from_slice(&[0.0, 0.0]); // unused for leaf
        tree.offsets.push(0.0);
        tree.children.push([-2, -4]); // indices[2:4]

        tree.indices = vec![0, 1, 2, 3];
        tree.n_nodes = 3;

        tree
    }

    #[test]
    fn test_search_left() {
        let tree = create_test_tree();
        let mut rng = FastRng::new(42);

        // Point on left side (x < 0)
        let point = vec![-1.0, 0.0];
        let (start, end) = tree.search(&point, &mut rng);

        assert_eq!(start, 0);
        assert_eq!(end, 2);
    }

    #[test]
    fn test_search_right() {
        let tree = create_test_tree();
        let mut rng = FastRng::new(42);

        // Point on right side (x > 0)
        let point = vec![1.0, 0.0];
        let (start, end) = tree.search(&point, &mut rng);

        assert_eq!(start, 2);
        assert_eq!(end, 4);
    }

    #[test]
    fn test_get_leaf_indices() {
        let tree = create_test_tree();
        let mut rng = FastRng::new(42);

        let point = vec![1.0, 0.0];
        let indices = tree.get_leaf_indices(&point, &mut rng);

        assert_eq!(indices, &[2, 3]);
    }

    #[test]
    fn test_get_all_leaves() {
        let tree = create_test_tree();
        let leaves = tree.get_all_leaves();

        assert_eq!(leaves.len(), 2);
        assert!(leaves.contains(&(0, 2)));
        assert!(leaves.contains(&(2, 4)));
    }
}
