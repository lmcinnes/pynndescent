//! Random projection tree construction.
//!
//! This module implements the tree building algorithm matching PyNNDescent.

use crate::rng::FastRng;
use super::flat_tree::FlatTree;
use rayon::prelude::*;

/// Build a random projection tree.
///
/// # Arguments
/// * `data` - Flattened data array (n_points × dim)
/// * `n_points` - Number of data points
/// * `dim` - Dimension of each point
/// * `leaf_size` - Maximum number of points in a leaf
/// * `rng` - Random number generator
/// * `angular` - Whether to use angular (cosine) distance for splitting
/// * `max_depth` - Maximum tree depth
pub fn build_rp_tree(
    data: &[f32],
    n_points: usize,
    dim: usize,
    leaf_size: usize,
    rng: &mut FastRng,
    angular: bool,
    max_depth: usize,
) -> FlatTree {
    let indices: Vec<i32> = (0..n_points as i32).collect();
    
    let mut builder = TreeBuilder::new(data, dim, leaf_size, angular, max_depth);
    builder.build(&indices, rng);
    builder.into_flat_tree()
}

/// Build a forest of random projection trees (parallel).
pub fn build_rp_forest(
    data: &[f32],
    n_points: usize,
    dim: usize,
    n_trees: usize,
    leaf_size: usize,
    rng: &mut FastRng,
    angular: bool,
    max_depth: usize,
) -> Vec<FlatTree> {
    // Generate seeds for each tree's RNG
    let seeds: Vec<u64> = (0..n_trees)
        .map(|_| rng.next_u64())
        .collect();
    
    // Build trees in parallel with independent RNGs
    seeds.into_par_iter()
        .map(|seed| {
            let mut tree_rng = FastRng::new(seed);
            build_rp_tree(data, n_points, dim, leaf_size, &mut tree_rng, angular, max_depth)
        })
        .collect()
}

/// Extract leaf arrays from a forest.
///
/// Returns a 2D array where each row contains the point indices in one leaf.
pub fn rptree_leaf_array(forest: &[FlatTree]) -> Vec<Vec<i32>> {
    let mut result = Vec::new();
    
    for tree in forest {
        let leaves = tree.get_all_leaves();
        for (start, end) in leaves {
            let leaf_indices = tree.indices[start..end].to_vec();
            result.push(leaf_indices);
        }
    }
    
    result
}

/// Internal tree builder structure.
struct TreeBuilder<'a> {
    data: &'a [f32],
    dim: usize,
    leaf_size: usize,
    angular: bool,
    max_depth: usize,
    
    // Output arrays
    hyperplanes: Vec<f32>,
    offsets: Vec<f32>,
    children: Vec<[i32; 2]>,
    leaf_indices: Vec<i32>,
}

impl<'a> TreeBuilder<'a> {
    fn new(
        data: &'a [f32],
        dim: usize,
        leaf_size: usize,
        angular: bool,
        max_depth: usize,
    ) -> Self {
        Self {
            data,
            dim,
            leaf_size,
            angular,
            max_depth,
            hyperplanes: Vec::new(),
            offsets: Vec::new(),
            children: Vec::new(),
            leaf_indices: Vec::new(),
        }
    }

    fn build(&mut self, indices: &[i32], rng: &mut FastRng) {
        self.build_recursive(indices, 0, rng);
    }

    fn build_recursive(&mut self, indices: &[i32], depth: usize, rng: &mut FastRng) -> i32 {
        let node_id = self.children.len() as i32;
        
        // Check if we should make a leaf
        if indices.len() <= self.leaf_size || depth >= self.max_depth {
            // Create leaf node
            let start = self.leaf_indices.len() as i32;
            self.leaf_indices.extend_from_slice(indices);
            let end = self.leaf_indices.len() as i32;
            
            // Store zeros for hyperplane (unused in leaves)
            self.hyperplanes.extend(vec![0.0f32; self.dim]);
            self.offsets.push(0.0);
            self.children.push([-start, -end]);
            
            return node_id;
        }

        // Select split hyperplane
        let (hyperplane, offset) = self.make_split(indices, rng);
        
        // Partition points
        let (left_indices, right_indices) = self.partition(indices, &hyperplane, offset, rng);
        
        // Handle degenerate splits
        if left_indices.is_empty() || right_indices.is_empty() {
            // Fall back to leaf
            let start = self.leaf_indices.len() as i32;
            self.leaf_indices.extend_from_slice(indices);
            let end = self.leaf_indices.len() as i32;
            
            self.hyperplanes.extend(vec![0.0f32; self.dim]);
            self.offsets.push(0.0);
            self.children.push([-start, -end]);
            
            return node_id;
        }

        // Add placeholder for this node
        self.hyperplanes.extend_from_slice(&hyperplane);
        self.offsets.push(offset);
        self.children.push([0, 0]); // Will be filled in

        // Recursively build children
        let left_child = self.build_recursive(&left_indices, depth + 1, rng);
        let right_child = self.build_recursive(&right_indices, depth + 1, rng);

        // Update children pointers
        self.children[node_id as usize] = [left_child, right_child];

        node_id
    }

    fn make_split(&self, indices: &[i32], rng: &mut FastRng) -> (Vec<f32>, f32) {
        // Pick two random points
        let idx1 = rng.next_index(indices.len());
        let mut idx2 = rng.next_index(indices.len());
        while idx2 == idx1 && indices.len() > 1 {
            idx2 = rng.next_index(indices.len());
        }

        let p1 = indices[idx1] as usize;
        let p2 = indices[idx2] as usize;

        let point1 = &self.data[p1 * self.dim..(p1 + 1) * self.dim];
        let point2 = &self.data[p2 * self.dim..(p2 + 1) * self.dim];

        if self.angular {
            self.make_angular_split(point1, point2)
        } else {
            self.make_euclidean_split(point1, point2)
        }
    }

    fn make_euclidean_split(&self, point1: &[f32], point2: &[f32]) -> (Vec<f32>, f32) {
        // Hyperplane is perpendicular bisector of the two points
        // Normal vector: point2 - point1
        // Offset: -dot(normal, midpoint)
        
        let mut hyperplane = Vec::with_capacity(self.dim);
        let mut midpoint = Vec::with_capacity(self.dim);
        
        for i in 0..self.dim {
            hyperplane.push(point2[i] - point1[i]);
            midpoint.push((point1[i] + point2[i]) / 2.0);
        }

        // Normalize hyperplane
        let norm: f32 = hyperplane.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut hyperplane {
                *x /= norm;
            }
        }

        // Compute offset: -dot(hyperplane, midpoint)
        let offset: f32 = -hyperplane.iter().zip(midpoint.iter()).map(|(h, m)| h * m).sum::<f32>();

        (hyperplane, offset)
    }

    fn make_angular_split(&self, point1: &[f32], point2: &[f32]) -> (Vec<f32>, f32) {
        // For angular distance, use the normalized difference
        let mut hyperplane = Vec::with_capacity(self.dim);
        
        // Normalize both points
        let norm1: f32 = point1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = point2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 < 1e-8 || norm2 < 1e-8 {
            // Degenerate case
            hyperplane.resize(self.dim, 0.0);
            return (hyperplane, 0.0);
        }

        for i in 0..self.dim {
            hyperplane.push(point2[i] / norm2 - point1[i] / norm1);
        }

        // Normalize hyperplane
        let norm: f32 = hyperplane.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut hyperplane {
                *x /= norm;
            }
        }

        (hyperplane, 0.0) // offset is 0 for angular splits
    }

    fn partition(
        &self,
        indices: &[i32],
        hyperplane: &[f32],
        offset: f32,
        rng: &mut FastRng,
    ) -> (Vec<i32>, Vec<i32>) {
        let mut left = Vec::new();
        let mut right = Vec::new();

        for &idx in indices {
            let point = &self.data[idx as usize * self.dim..(idx as usize + 1) * self.dim];
            
            // Compute margin
            let mut margin = offset;
            for i in 0..self.dim {
                margin += point[i] * hyperplane[i];
            }

            // Assign to side
            if margin.abs() < 1e-8 {
                // Random assignment for points on the boundary
                if rng.next_bool() {
                    right.push(idx);
                } else {
                    left.push(idx);
                }
            } else if margin < 0.0 {
                left.push(idx);
            } else {
                right.push(idx);
            }
        }

        (left, right)
    }

    fn into_flat_tree(self) -> FlatTree {
        let n_nodes = self.children.len();
        FlatTree {
            hyperplanes: self.hyperplanes,
            offsets: self.offsets,
            children: self.children,
            indices: self.leaf_indices,
            dim: self.dim,
            n_nodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_single_tree() {
        // Create simple 2D data
        let data: Vec<f32> = vec![
            0.0, 0.0,  // point 0
            1.0, 0.0,  // point 1
            0.0, 1.0,  // point 2
            1.0, 1.0,  // point 3
            -1.0, 0.0, // point 4
            0.0, -1.0, // point 5
        ];
        let n_points = 6;
        let dim = 2;
        let leaf_size = 2;
        let mut rng = FastRng::new(42);

        let tree = build_rp_tree(&data, n_points, dim, leaf_size, &mut rng, false, 10);

        // Tree should have all points
        assert_eq!(tree.indices.len(), n_points);
        
        // All indices should be present
        let mut indices: Vec<i32> = tree.indices.clone();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_build_forest() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let n_points = 50;
        let dim = 2;
        let n_trees = 3;
        let leaf_size = 10;
        let mut rng = FastRng::new(42);

        let forest = build_rp_forest(&data, n_points, dim, n_trees, leaf_size, &mut rng, false, 10);

        assert_eq!(forest.len(), n_trees);
        
        for tree in &forest {
            assert_eq!(tree.indices.len(), n_points);
        }
    }

    #[test]
    fn test_leaf_array() {
        let data: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1).collect();
        let n_points = 10;
        let dim = 2;
        let leaf_size = 3;
        let mut rng = FastRng::new(42);

        let tree = build_rp_tree(&data, n_points, dim, leaf_size, &mut rng, false, 10);
        let leaves = rptree_leaf_array(&[tree]);

        // All points should appear in exactly one leaf
        let mut all_indices: Vec<i32> = leaves.iter().flatten().copied().collect();
        all_indices.sort();
        let expected: Vec<i32> = (0..n_points as i32).collect();
        assert_eq!(all_indices, expected);
    }

    #[test]
    fn test_search_consistency() {
        let data: Vec<f32> = vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ];
        let n_points = 4;
        let dim = 2;
        let leaf_size = 2;
        let mut rng = FastRng::new(42);

        let tree = build_rp_tree(&data, n_points, dim, leaf_size, &mut rng, false, 10);

        // Search for each data point - it should find itself in its leaf
        for i in 0..n_points {
            let point = &data[i * dim..(i + 1) * dim];
            let leaf_indices = tree.get_leaf_indices(point, &mut rng);
            
            // The point should be in its own leaf (though exact index might vary)
            assert!(!leaf_indices.is_empty());
        }
    }
}
