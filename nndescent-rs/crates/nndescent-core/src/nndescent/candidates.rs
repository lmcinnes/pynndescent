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

    /// Parallel version of candidate building using two-phase approach.
    ///
    /// Phase 1: Each thread processes its own block's rows, pushing forward edges.
    ///          Also collects reverse edges bucketed by destination block.
    /// Phase 2: Each thread applies reverse edges destined for its block.
    ///
    /// This is O(n*k) total scan work instead of O(n_threads * n*k).
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

        // Create per-thread RNG seeds
        let thread_seeds: Vec<u64> = (0..n_threads).map(|_| rng.next_u64()).collect();

        // Read-only references
        let graph_indices = &graph.indices;
        let graph_flags = &graph.flags;

        // Phase 1: Each thread processes its own rows (forward edges) and collects
        // reverse edges bucketed by destination block.
        // reverse_buckets[src_thread][dest_block] = Vec of (dest_vertex, candidate, priority, is_new)
        let reverse_buckets: Vec<Vec<Vec<(usize, i32, f32, bool)>>> = (0..n_threads)
            .into_par_iter()
            .map(|thread_idx| {
                let block_start = thread_idx * block_size;
                let block_end = ((thread_idx + 1) * block_size).min(n_vertices);

                if block_start >= n_vertices {
                    return vec![Vec::new(); n_threads];
                }

                let mut local_rng = FastRng::new(thread_seeds[thread_idx]);

                // Per-destination-block reverse edge buckets
                let mut buckets: Vec<Vec<(usize, i32, f32, bool)>> = vec![Vec::new(); n_threads];

                // SAFETY: Each thread writes to disjoint portions [block_start*mc .. block_end*mc)
                let new_idx_ptr = new_indices.as_ptr() as *mut i32;
                let new_pri_ptr = new_priority.as_ptr() as *mut f32;
                let old_idx_ptr = old_indices.as_ptr() as *mut i32;
                let old_pri_ptr = old_priority.as_ptr() as *mut f32;

                for i in block_start..block_end {
                    let row_start = i * k;

                    for j in 0..k {
                        let flat_idx = row_start + j;
                        let neighbor = graph_indices[flat_idx];

                        if neighbor < 0 {
                            continue;
                        }
                        let neighbor_idx = neighbor as usize;
                        let is_new = graph_flags[flat_idx] != 0;
                        let priority = local_rng.next_float();

                        // Forward edge: push (neighbor) as candidate for vertex i
                        let offset = i * max_candidates;
                        if is_new {
                            unsafe {
                                let pri = std::slice::from_raw_parts_mut(new_pri_ptr.add(offset), max_candidates);
                                let idx = std::slice::from_raw_parts_mut(new_idx_ptr.add(offset), max_candidates);
                                checked_heap_push_flat(pri, idx, priority, neighbor);
                            }
                        } else {
                            unsafe {
                                let pri = std::slice::from_raw_parts_mut(old_pri_ptr.add(offset), max_candidates);
                                let idx = std::slice::from_raw_parts_mut(old_idx_ptr.add(offset), max_candidates);
                                checked_heap_push_flat(pri, idx, priority, neighbor);
                            }
                        }

                        // Reverse edge: push (i) as candidate for vertex neighbor
                        // Bucket by which block the neighbor belongs to
                        let dest_block = neighbor_idx / block_size;
                        if dest_block < n_threads {
                            if dest_block == thread_idx {
                                // Neighbor is in our own block - handle directly
                                let n_offset = neighbor_idx * max_candidates;
                                if is_new {
                                    unsafe {
                                        let pri = std::slice::from_raw_parts_mut(new_pri_ptr.add(n_offset), max_candidates);
                                        let idx = std::slice::from_raw_parts_mut(new_idx_ptr.add(n_offset), max_candidates);
                                        checked_heap_push_flat(pri, idx, priority, i as i32);
                                    }
                                } else {
                                    unsafe {
                                        let pri = std::slice::from_raw_parts_mut(old_pri_ptr.add(n_offset), max_candidates);
                                        let idx = std::slice::from_raw_parts_mut(old_idx_ptr.add(n_offset), max_candidates);
                                        checked_heap_push_flat(pri, idx, priority, i as i32);
                                    }
                                }
                            } else {
                                buckets[dest_block].push((neighbor_idx, i as i32, priority, is_new));
                            }
                        }
                    }
                }

                buckets
            })
            .collect();

        // Phase 2: Each thread applies reverse edges destined for its block
        (0..n_threads).into_par_iter().for_each(|thread_idx| {
            let block_start = thread_idx * block_size;
            if block_start >= n_vertices {
                return;
            }

            // SAFETY: Each thread writes to disjoint portions [block_start*mc .. block_end*mc)
            let new_idx_ptr = new_indices.as_ptr() as *mut i32;
            let new_pri_ptr = new_priority.as_ptr() as *mut f32;
            let old_idx_ptr = old_indices.as_ptr() as *mut i32;
            let old_pri_ptr = old_priority.as_ptr() as *mut f32;

            // Read reverse edges from all source threads destined for this block
            for src_thread in 0..n_threads {
                if src_thread == thread_idx {
                    continue; // Already handled in Phase 1
                }
                for &(dest_vertex, candidate, priority, is_new) in &reverse_buckets[src_thread][thread_idx] {
                    let offset = dest_vertex * max_candidates;
                    if is_new {
                        unsafe {
                            let pri = std::slice::from_raw_parts_mut(new_pri_ptr.add(offset), max_candidates);
                            let idx = std::slice::from_raw_parts_mut(new_idx_ptr.add(offset), max_candidates);
                            checked_heap_push_flat(pri, idx, priority, candidate);
                        }
                    } else {
                        unsafe {
                            let pri = std::slice::from_raw_parts_mut(old_pri_ptr.add(offset), max_candidates);
                            let idx = std::slice::from_raw_parts_mut(old_idx_ptr.add(offset), max_candidates);
                            checked_heap_push_flat(pri, idx, priority, candidate);
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
        let n_threads = rayon::current_num_threads().max(1);
        
        if n_vertices < 256 || n_threads <= 1 {
            // Sequential version
            for i in 0..n_vertices {
                let row_offset = i * k;
                let new_offset = i * max_candidates;
                
                for j in 0..k {
                    let neighbor = graph.indices[row_offset + j];
                    if neighbor < 0 || graph.flags[row_offset + j] == 0 {
                        continue;
                    }
                    
                    for nc_idx in 0..max_candidates {
                        if new_indices[new_offset + nc_idx] == neighbor {
                            graph.flags[row_offset + j] = 0;
                            break;
                        }
                    }
                }
            }
        } else {
            // Parallel version - each thread handles a disjoint range of vertices
            // SAFETY: Each thread writes to a disjoint range of flags (different rows).
            // We use a usize wrapper to pass the pointer safely across threads.
            let flags_base = graph.flags.as_mut_ptr() as usize;
            let indices_ref = &graph.indices;
            
            (0..n_threads).into_par_iter().for_each(|t| {
                let block_size = (n_vertices + n_threads - 1) / n_threads;
                let start = t * block_size;
                let end = ((t + 1) * block_size).min(n_vertices);
                let flags_ptr = flags_base as *mut u8;
                
                for i in start..end {
                    let row_offset = i * k;
                    let new_offset = i * max_candidates;
                    
                    for j in 0..k {
                        let neighbor = indices_ref[row_offset + j];
                        if neighbor < 0 {
                            continue;
                        }
                        if unsafe { *flags_ptr.add(row_offset + j) } == 0 {
                            continue;
                        }
                        
                        for nc_idx in 0..max_candidates {
                            if new_indices[new_offset + nc_idx] == neighbor {
                                unsafe { *flags_ptr.add(row_offset + j) = 0; }
                                break;
                            }
                        }
                    }
                }
            });
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
/// Uses the shift-down technique matching PyNNDescent's `checked_heap_push`.
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
        // SAFETY: i < n = indices.len()
        if unsafe { *indices.get_unchecked(i) } == index {
            return;
        }
    }

    // Insert at root and sift down using shift technique
    unsafe {
        *priorities.get_unchecked_mut(0) = priority;
        *indices.get_unchecked_mut(0) = index;
    }
    
    let mut pos = 0usize;
    loop {
        let left = 2 * pos + 1;
        let right = 2 * pos + 2;
        let mut largest = pos;

        unsafe {
            if left < n && *priorities.get_unchecked(left) > *priorities.get_unchecked(largest) {
                largest = left;
            }
            if right < n && *priorities.get_unchecked(right) > *priorities.get_unchecked(largest) {
                largest = right;
            }
        }

        if largest != pos {
            unsafe {
                let child_pri = *priorities.get_unchecked(largest);
                let child_idx = *indices.get_unchecked(largest);
                *priorities.get_unchecked_mut(pos) = child_pri;
                *indices.get_unchecked_mut(pos) = child_idx;
                *priorities.get_unchecked_mut(largest) = priority;
                *indices.get_unchecked_mut(largest) = index;
            }
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
