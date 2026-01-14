//! CSR (Compressed Sparse Row) search graph for efficient traversal.

/// Search graph in CSR format for efficient neighbor traversal.
///
/// This format is memory-efficient and cache-friendly for graph search
/// operations where we iterate over neighbors of each vertex.
#[derive(Clone, Debug)]
pub struct SearchGraph {
    /// Row pointers: indptr[i] to indptr[i+1] gives the range of neighbors for vertex i
    pub indptr: Vec<i32>,
    /// Column indices: the neighbor indices
    pub indices: Vec<i32>,
    /// Number of vertices
    pub n_vertices: usize,
}

impl SearchGraph {
    /// Create a new empty search graph.
    pub fn new(n_vertices: usize) -> Self {
        Self {
            indptr: vec![0; n_vertices + 1],
            indices: Vec::new(),
            n_vertices,
        }
    }

    /// Create from dense neighbor arrays (after diversification/pruning).
    ///
    /// Takes row-major neighbor indices where -1 indicates no neighbor.
    pub fn from_dense(neighbor_indices: &[i32], n_points: usize, k: usize) -> Self {
        let mut indptr = Vec::with_capacity(n_points + 1);
        let mut indices = Vec::new();

        indptr.push(0);

        for point in 0..n_points {
            let start = point * k;
            let end = start + k;

            for &neighbor in &neighbor_indices[start..end] {
                if neighbor >= 0 {
                    indices.push(neighbor);
                }
            }

            indptr.push(indices.len() as i32);
        }

        Self {
            indptr,
            indices,
            n_vertices: n_points,
        }
    }

    /// Get the neighbors of a vertex.
    #[inline]
    pub fn neighbors(&self, vertex: usize) -> &[i32] {
        let start = self.indptr[vertex] as usize;
        let end = self.indptr[vertex + 1] as usize;
        &self.indices[start..end]
    }

    /// Get the degree (number of neighbors) of a vertex.
    #[inline]
    pub fn degree(&self, vertex: usize) -> usize {
        (self.indptr[vertex + 1] - self.indptr[vertex]) as usize
    }

    /// Get total number of edges.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.indices.len()
    }

    /// Compute the union with another graph (for undirected graph from directed).
    pub fn union_with(&self, other: &SearchGraph) -> SearchGraph {
        use std::collections::BTreeSet;

        let n = self.n_vertices;
        let mut adjacency: Vec<BTreeSet<i32>> = vec![BTreeSet::new(); n];

        // Add edges from self
        for v in 0..n {
            for &neighbor in self.neighbors(v) {
                adjacency[v].insert(neighbor);
            }
        }

        // Add edges from other
        for v in 0..n {
            for &neighbor in other.neighbors(v) {
                adjacency[v].insert(neighbor);
            }
        }

        // Convert back to CSR
        let mut indptr = Vec::with_capacity(n + 1);
        let mut indices = Vec::new();

        indptr.push(0);
        for neighbors in &adjacency {
            for &neighbor in neighbors {
                indices.push(neighbor);
            }
            indptr.push(indices.len() as i32);
        }

        SearchGraph {
            indptr,
            indices,
            n_vertices: n,
        }
    }

    /// Transpose the graph (reverse all edges).
    pub fn transpose(&self) -> SearchGraph {
        let n = self.n_vertices;

        // Count incoming edges for each vertex
        let mut in_degree = vec![0i32; n];
        for &neighbor in &self.indices {
            if neighbor >= 0 {
                in_degree[neighbor as usize] += 1;
            }
        }

        // Build indptr for transposed graph
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0);
        for &deg in &in_degree {
            indptr.push(indptr.last().unwrap() + deg);
        }

        // Fill indices
        let mut indices = vec![0i32; self.indices.len()];
        let mut current_pos = indptr[..n].to_vec();

        for v in 0..n {
            for &neighbor in self.neighbors(v) {
                if neighbor >= 0 {
                    let pos = current_pos[neighbor as usize] as usize;
                    indices[pos] = v as i32;
                    current_pos[neighbor as usize] += 1;
                }
            }
        }

        SearchGraph {
            indptr,
            indices,
            n_vertices: n,
        }
    }

    /// Reorder the graph according to a permutation.
    pub fn reorder(&mut self, order: &[usize]) {
        let n = self.n_vertices;

        // Create inverse mapping
        let mut inverse = vec![0usize; n];
        for (new_pos, &old_pos) in order.iter().enumerate() {
            inverse[old_pos] = new_pos;
        }

        // Build new graph
        let mut new_indptr = Vec::with_capacity(n + 1);
        let mut new_indices = Vec::new();

        new_indptr.push(0);
        for new_v in 0..n {
            let old_v = order[new_v];

            // Map neighbors to new indices
            for &old_neighbor in self.neighbors(old_v) {
                if old_neighbor >= 0 {
                    let new_neighbor = inverse[old_neighbor as usize] as i32;
                    new_indices.push(new_neighbor);
                }
            }

            new_indptr.push(new_indices.len() as i32);
        }

        self.indptr = new_indptr;
        self.indices = new_indices;
    }

    /// Prune edges to limit maximum degree.
    pub fn prune_degree(&mut self, max_degree: usize) {
        let n = self.n_vertices;

        let mut new_indptr = Vec::with_capacity(n + 1);
        let mut new_indices = Vec::new();

        new_indptr.push(0);
        for v in 0..n {
            let neighbors = self.neighbors(v);
            let keep = neighbors.len().min(max_degree);

            for &neighbor in &neighbors[..keep] {
                new_indices.push(neighbor);
            }

            new_indptr.push(new_indices.len() as i32);
        }

        self.indptr = new_indptr;
        self.indices = new_indices;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_dense() {
        // 3 points, k=2
        let neighbors = vec![1, 2, 0, 2, 0, 1];
        let graph = SearchGraph::from_dense(&neighbors, 3, 2);

        assert_eq!(graph.n_vertices, 3);
        assert_eq!(graph.neighbors(0), &[1, 2]);
        assert_eq!(graph.neighbors(1), &[0, 2]);
        assert_eq!(graph.neighbors(2), &[0, 1]);
    }

    #[test]
    fn test_from_dense_with_missing() {
        // Some -1 entries
        let neighbors = vec![1, -1, 0, 2, -1, -1];
        let graph = SearchGraph::from_dense(&neighbors, 3, 2);

        assert_eq!(graph.neighbors(0), &[1]);
        assert_eq!(graph.neighbors(1), &[0, 2]);
        assert_eq!(graph.neighbors(2).len(), 0);
    }

    #[test]
    fn test_transpose() {
        let neighbors = vec![1, 2, 2, -1, -1, -1];
        let graph = SearchGraph::from_dense(&neighbors, 3, 2);
        let transposed = graph.transpose();

        // Original: 0 -> [1, 2], 1 -> [2]
        // Transposed: 1 <- [0], 2 <- [0, 1]
        assert_eq!(transposed.neighbors(0).len(), 0);
        assert_eq!(transposed.neighbors(1), &[0]);
        assert!(transposed.neighbors(2).contains(&0) && transposed.neighbors(2).contains(&1));
    }

    #[test]
    fn test_union() {
        let neighbors1 = vec![1, -1, 0, -1, -1, -1];
        let neighbors2 = vec![-1, 2, -1, -1, 0, 1];

        let g1 = SearchGraph::from_dense(&neighbors1, 3, 2);
        let g2 = SearchGraph::from_dense(&neighbors2, 3, 2);

        let union = g1.union_with(&g2);

        assert_eq!(union.neighbors(0).len(), 2); // 1 from g1, 2 from g2
        assert_eq!(union.neighbors(2).len(), 2); // 0 and 1 from g2
    }

    #[test]
    fn test_prune_degree() {
        let neighbors = vec![1, 2, 0, 2, 0, 1];
        let mut graph = SearchGraph::from_dense(&neighbors, 3, 2);

        graph.prune_degree(1);

        assert_eq!(graph.degree(0), 1);
        assert_eq!(graph.degree(1), 1);
        assert_eq!(graph.degree(2), 1);
    }
}
