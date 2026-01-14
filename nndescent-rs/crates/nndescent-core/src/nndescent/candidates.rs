//! Candidate set management for NN-Descent.
//!
//! This implements the new/old candidate separation that is key to
//! NN-Descent's efficiency - we skip comparing (old, old) pairs.

use crate::heap::NeighborHeap;
use crate::rng::FastRng;
use rayon::prelude::*;

/// Candidate sets using flat arrays for cache efficiency.
///
/// This separation is crucial for NN-Descent performance:
/// - (new, new) pairs: always compare
/// - (new, old) pairs: always compare  
/// - (old, old) pairs: skip (already compared in previous iteration)
#[derive(Clone, Debug)]
pub struct CandidateSets {
    /// New candidate indices, flat array shape (n_vertices × max_candidates)
    /// -1 indicates empty slot
    pub new_indices: Vec<i32>,
    /// Old candidate indices, flat array shape (n_vertices × max_candidates)
    pub old_indices: Vec<i32>,
    /// Number of vertices
    pub n_vertices: usize,
    /// Maximum candidates per vertex
    pub max_candidates: usize,
}

impl CandidateSets {
    /// Get new candidates for a vertex as a slice.
    #[inline]
    pub fn get_new(&self, vertex: usize) -> &[i32] {
        let start = vertex * self.max_candidates;
        &self.new_indices[start..start + self.max_candidates]
    }

    /// Get old candidates for a vertex as a slice.
    #[inline]
    pub fn get_old(&self, vertex: usize) -> &[i32] {
        let start = vertex * self.max_candidates;
        &self.old_indices[start..start + self.max_candidates]
    }

    /// Build candidate sets from the current neighbor graph (parallel version).
    ///
    /// This uses a block-based parallel approach matching PyNNDescent:
    /// - Vertices are divided into blocks
    /// - Each thread handles one block but iterates all vertices
    /// - A thread only writes to heaps in its block
    /// - This avoids write conflicts without locks
    pub fn build_from_graph(
        graph: &mut NeighborHeap,
        max_candidates: usize,
        rng: &mut FastRng,
    ) -> Self {
        let n_vertices = graph.n_points;
        let k = graph.k;
        let n_threads = rayon::current_num_threads().max(1);
        
        // Use the parallel version if we have enough vertices
        if n_vertices >= 256 && n_threads > 1 {
            Self::build_from_graph_parallel(graph, max_candidates, rng, n_threads)
        } else {
            Self::build_from_graph_sequential(graph, max_candidates, rng)
        }
    }

    /// Sequential version of candidate building.
    fn build_from_graph_sequential(
        graph: &mut NeighborHeap,
        max_candidates: usize,
        rng: &mut FastRng,
    ) -> Self {
        let n_vertices = graph.n_points;
        let k = graph.k;

        // Flat arrays for cache efficiency
        let size = n_vertices * max_candidates;
        let mut new_indices = vec![-1i32; size];
        let mut new_priority = vec![f32::INFINITY; size];
        let mut old_indices = vec![-1i32; size];
        let mut old_priority = vec![f32::INFINITY; size];

        // Process all edges: forward (i -> neighbor) and reverse (neighbor -> i)
        for i in 0..n_vertices {
            let row_start = i * k;
            
            for j in 0..k {
                let neighbor = graph.indices[row_start + j];
                if neighbor < 0 {
                    continue;
                }
                let neighbor_idx = neighbor as usize;
                let is_new = graph.flags[row_start + j] != 0;
                let priority = rng.next_float();

                // Add forward edge (i -> neighbor) - neighbor is candidate for i
                let offset_i = i * max_candidates;
                if is_new {
                    checked_heap_push_flat(
                        &mut new_priority[offset_i..offset_i + max_candidates],
                        &mut new_indices[offset_i..offset_i + max_candidates],
                        priority,
                        neighbor,
                    );
                } else {
                    checked_heap_push_flat(
                        &mut old_priority[offset_i..offset_i + max_candidates],
                        &mut old_indices[offset_i..offset_i + max_candidates],
                        priority,
                        neighbor,
                    );
                }

                // Add reverse edge (neighbor -> i) - i is candidate for neighbor  
                let reverse_priority = rng.next_float();
                let offset_n = neighbor_idx * max_candidates;
                if is_new {
                    checked_heap_push_flat(
                        &mut new_priority[offset_n..offset_n + max_candidates],
                        &mut new_indices[offset_n..offset_n + max_candidates],
                        reverse_priority,
                        i as i32,
                    );
                } else {
                    checked_heap_push_flat(
                        &mut old_priority[offset_n..offset_n + max_candidates],
                        &mut old_indices[offset_n..offset_n + max_candidates],
                        reverse_priority,
                        i as i32,
                    );
                }
            }
        }

        // Mark neighbors that appear in new_candidates as old (flag=0) in the graph
        Self::mark_old_flags(graph, &new_indices, max_candidates);

        Self {
            new_indices,
            old_indices,
            n_vertices,
            max_candidates,
        }
    }

    /// Parallel version of candidate building using block-based approach.
    ///
    /// Matches PyNNDescent's algorithm: each thread handles a block but iterates
    /// all vertices, only writing to heaps within its block.
    fn build_from_graph_parallel(
        graph: &mut NeighborHeap,
        max_candidates: usize,
        rng: &mut FastRng,
        n_threads: usize,
    ) -> Self {
        let n_vertices = graph.n_points;
        let k = graph.k;
        let block_size = (n_vertices + n_threads - 1) / n_threads;

        // Flat arrays for all blocks
        let size = n_vertices * max_candidates;
        let mut new_indices = vec![-1i32; size];
        let mut new_priority = vec![f32::INFINITY; size];
        let mut old_indices = vec![-1i32; size];
        let mut old_priority = vec![f32::INFINITY; size];

        // Pre-generate random priorities for determinism
        // We need 2 priorities per edge (forward + reverse) per vertex per neighbor
        let n_random = n_vertices * k * 2;
        let priorities: Vec<f32> = (0..n_random).map(|_| rng.next_float()).collect();

        // Process blocks in parallel directly (without par_chunks_mut which has overhead)
        (0..n_threads).into_par_iter().for_each(|thread_idx| {
            let block_start = thread_idx * block_size;
            let block_end = ((thread_idx + 1) * block_size).min(n_vertices);
            
            if block_start >= n_vertices {
                return;
            }
            
            // SAFETY: Each thread writes to disjoint portions of the arrays
            // based on block_start..block_end
            let new_idx_ptr = new_indices.as_ptr() as *mut i32;
            let new_pri_ptr = new_priority.as_ptr() as *mut f32;
            let old_idx_ptr = old_indices.as_ptr() as *mut i32;
            let old_pri_ptr = old_priority.as_ptr() as *mut f32;
            
            // Each thread iterates ALL vertices, but only writes to heaps in its block
            for i in 0..n_vertices {
                let row_start = i * k;
                
                for j in 0..k {
                    let neighbor = graph.indices[row_start + j];
                    if neighbor < 0 {
                        continue;
                    }
                    let neighbor_idx = neighbor as usize;
                    let is_new = graph.flags[row_start + j] != 0;
                    
                    // Get pre-generated random priority
                    let priority = priorities[i * k * 2 + j * 2];
                    let reverse_priority = priorities[i * k * 2 + j * 2 + 1];

                    // Forward edge: i -> neighbor (neighbor is candidate for i)
                    // Only write if i is in our block
                    if i >= block_start && i < block_end {
                        let offset = i * max_candidates;
                        unsafe {
                            let idx_slice = std::slice::from_raw_parts_mut(
                                new_idx_ptr.add(offset), max_candidates);
                            let pri_slice = std::slice::from_raw_parts_mut(
                                new_pri_ptr.add(offset), max_candidates);
                            let old_idx_slice = std::slice::from_raw_parts_mut(
                                old_idx_ptr.add(offset), max_candidates);
                            let old_pri_slice = std::slice::from_raw_parts_mut(
                                old_pri_ptr.add(offset), max_candidates);
                            
                            if is_new {
                                checked_heap_push_flat(pri_slice, idx_slice, priority, neighbor);
                            } else {
                                checked_heap_push_flat(old_pri_slice, old_idx_slice, priority, neighbor);
                            }
                        }
                    }

                    // Reverse edge: neighbor -> i (i is candidate for neighbor)
                    // Only write if neighbor is in our block
                    if neighbor_idx >= block_start && neighbor_idx < block_end {
                        let offset = neighbor_idx * max_candidates;
                        unsafe {
                            let idx_slice = std::slice::from_raw_parts_mut(
                                new_idx_ptr.add(offset), max_candidates);
                            let pri_slice = std::slice::from_raw_parts_mut(
                                new_pri_ptr.add(offset), max_candidates);
                            let old_idx_slice = std::slice::from_raw_parts_mut(
                                old_idx_ptr.add(offset), max_candidates);
                            let old_pri_slice = std::slice::from_raw_parts_mut(
                                old_pri_ptr.add(offset), max_candidates);
                            
                            if is_new {
                                checked_heap_push_flat(pri_slice, idx_slice, reverse_priority, i as i32);
                            } else {
                                checked_heap_push_flat(old_pri_slice, old_idx_slice, reverse_priority, i as i32);
                            }
                        }
                    }
                }
            }
        });

        // Mark neighbors that appear in new_candidates as old (flag=0) in the graph
        Self::mark_old_flags(graph, &new_indices, max_candidates);

        Self {
            new_indices,
            old_indices,
            n_vertices,
            max_candidates,
        }
    }

    /// Mark flags in graph as old for neighbors that appear in new_indices.
    fn mark_old_flags(graph: &mut NeighborHeap, new_indices: &[i32], max_candidates: usize) {
        let n_vertices = graph.n_points;
        let k = graph.k;
        
        for i in 0..n_vertices {
            let row_offset = i * k;
            let new_offset = i * max_candidates;
            
            for j in 0..k {
                let neighbor = graph.indices[row_offset + j];
                if neighbor < 0 || graph.flags[row_offset + j] == 0 {
                    continue;
                }
                
                // Check if this neighbor appears in new_indices for vertex i
                for nc_idx in 0..max_candidates {
                    if new_indices[new_offset + nc_idx] == neighbor {
                        graph.flags[row_offset + j] = 0;
                        break;
                    }
                }
            }
        }
    }

    /// Get total number of new candidates across all vertices.
    pub fn total_new(&self) -> usize {
        self.new_indices.iter().filter(|&&x| x >= 0).count()
    }

    /// Get total number of old candidates across all vertices.
    pub fn total_old(&self) -> usize {
        self.old_indices.iter().filter(|&&x| x >= 0).count()
    }
}

/// Push to a bounded priority max-heap with duplicate checking (flat slice version).
/// Uses unsafe for bounds-check elimination in hot paths.
#[inline(always)]
fn checked_heap_push_flat(
    priorities: &mut [f32],
    indices: &mut [i32],
    priority: f32,
    index: i32,
) {
    // Early exit if priority is worse than current max
    // SAFETY: We know priorities is non-empty (max_candidates > 0)
    if priority >= unsafe { *priorities.get_unchecked(0) } {
        return;
    }

    // Check for duplicate (linear scan - OK since max_size is small ~30-60)
    let n = priorities.len();
    for i in 0..n {
        // SAFETY: i < n = priorities.len()
        if unsafe { *indices.get_unchecked(i) } == index {
            return;
        }
    }

    // Insert by replacing root
    // SAFETY: indices 0..n are valid
    unsafe {
        *priorities.get_unchecked_mut(0) = priority;
        *indices.get_unchecked_mut(0) = index;
    }
    
    // Sift down to maintain max-heap property
    let mut pos = 0usize;
    loop {
        let left = 2 * pos + 1;
        let right = 2 * pos + 2;
        let mut largest = pos;

        // SAFETY: left/right/pos/largest are all < n when accessed
        unsafe {
            if left < n && *priorities.get_unchecked(left) > *priorities.get_unchecked(largest) {
                largest = left;
            }
            if right < n && *priorities.get_unchecked(right) > *priorities.get_unchecked(largest) {
                largest = right;
            }
        }

        if largest != pos {
            priorities.swap(pos, largest);
            indices.swap(pos, largest);
            pos = largest;
        } else {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checked_heap_push_flat_basic() {
        let mut priorities = vec![f32::INFINITY; 3];
        let mut indices = vec![-1; 3];
        
        checked_heap_push_flat(&mut priorities, &mut indices, 0.5, 1);
        checked_heap_push_flat(&mut priorities, &mut indices, 0.3, 2);
        checked_heap_push_flat(&mut priorities, &mut indices, 0.7, 3);
        
        // All three should be in
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
        assert!(indices.contains(&3));
    }

    #[test]
    fn test_checked_heap_push_flat_duplicate() {
        let mut priorities = vec![f32::INFINITY; 3];
        let mut indices = vec![-1; 3];
        
        checked_heap_push_flat(&mut priorities, &mut indices, 0.5, 1);
        checked_heap_push_flat(&mut priorities, &mut indices, 0.3, 1);  // Duplicate - should be rejected
        checked_heap_push_flat(&mut priorities, &mut indices, 0.7, 1);  // Duplicate - should be rejected
        
        // Only one entry should have index 1
        assert_eq!(indices.iter().filter(|&&x| x == 1).count(), 1);
    }

    #[test]
    fn test_checked_heap_push_flat_overflow() {
        let mut priorities = vec![f32::INFINITY; 3];
        let mut indices = vec![-1; 3];
        
        checked_heap_push_flat(&mut priorities, &mut indices, 0.9, 1);
        checked_heap_push_flat(&mut priorities, &mut indices, 0.8, 2);
        checked_heap_push_flat(&mut priorities, &mut indices, 0.7, 3);
        // Heap full with priorities [0.9, 0.8, 0.7]
        
        checked_heap_push_flat(&mut priorities, &mut indices, 0.5, 4);  // Should replace 0.9
        
        // 4 should be in heap, 1 should not
        assert!(indices.contains(&4));
        assert!(!indices.contains(&1));
    }

    #[test]
    fn test_build_candidates() {
        let mut graph = NeighborHeap::new(5, 3);

        graph.unchecked_flagged_push(0, 1, 0.1, true);
        graph.unchecked_flagged_push(0, 2, 0.2, false);
        graph.unchecked_flagged_push(0, 3, 0.3, false);
        graph.unchecked_flagged_push(1, 0, 0.1, true);
        graph.unchecked_flagged_push(1, 2, 0.15, true);

        let mut rng = FastRng::new(42);
        let candidates = CandidateSets::build_from_graph(&mut graph, 10, &mut rng);

        // Check point 0's candidates include 1 as new
        let new_0 = candidates.get_new(0);
        assert!(new_0.contains(&1));
        
        // Check point 1's candidates include 0 and 2 as new
        let new_1 = candidates.get_new(1);
        assert!(new_1.contains(&0));
        assert!(new_1.contains(&2));
    }

    #[test]
    fn test_reverse_neighbors() {
        let mut graph = NeighborHeap::new(3, 2);

        graph.unchecked_flagged_push(0, 1, 0.1, true);
        graph.unchecked_flagged_push(1, 2, 0.2, true);

        let mut rng = FastRng::new(42);
        let candidates = CandidateSets::build_from_graph(&mut graph, 10, &mut rng);

        // Point 1 should have 0 as a reverse neighbor
        let new_1 = candidates.get_new(1);
        assert!(new_1.contains(&0));

        // Point 2 should have 1 as a reverse neighbor
        let new_2 = candidates.get_new(2);
        assert!(new_2.contains(&1));
    }

    #[test]
    fn test_max_candidates_limit() {
        let mut graph = NeighborHeap::new(10, 5);

        for i in 1..10 {
            graph.unchecked_flagged_push(0, i as i32, i as f32 * 0.1, true);
        }

        let mut rng = FastRng::new(42);
        let candidates = CandidateSets::build_from_graph(&mut graph, 3, &mut rng);

        // Point 0 should have at most 3 new candidates
        let new_0 = candidates.get_new(0);
        let count = new_0.iter().filter(|&&x| x >= 0).count();
        assert!(count <= 3);
    }
}
