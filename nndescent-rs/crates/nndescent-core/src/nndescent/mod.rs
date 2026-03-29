//! NN-Descent algorithm implementation.
//!
//! This module implements the core NN-Descent algorithm for approximate
//! k-nearest neighbor graph construction.

mod candidates;
mod update;

pub use candidates::CandidateSets;
pub use update::{UpdateArray, apply_updates};

use crate::distance::Distance;
use crate::heap::NeighborHeap;
use crate::rng::FastRng;
use crate::tree::{build_rp_forest, rptree_leaf_array, FlatTree};

use rayon::prelude::*;

/// NN-Descent algorithm parameters.
#[derive(Clone, Debug)]
pub struct NNDescentParams {
    /// Number of neighbors to find
    pub n_neighbors: usize,
    /// Number of RP trees for initialization
    pub n_trees: usize,
    /// Maximum leaf size in RP trees
    pub leaf_size: usize,
    /// Maximum candidates per iteration
    pub max_candidates: usize,
    /// Maximum iterations
    pub n_iters: usize,
    /// Convergence threshold (fraction of updates)
    pub delta: f32,
    /// Whether to use angular trees
    pub angular: bool,
    /// Maximum tree depth
    pub max_depth: usize,
    /// Verbose output
    pub verbose: bool,
}

impl Default for NNDescentParams {
    fn default() -> Self {
        Self {
            n_neighbors: 30,
            n_trees: 8,
            leaf_size: 64,
            max_candidates: 60,
            n_iters: 10,
            delta: 0.001,
            angular: false,
            max_depth: 200,
            verbose: false,
        }
    }
}

impl NNDescentParams {
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            ..Default::default()
        }
    }
}

/// Run the NN-Descent algorithm to build a k-NN graph.
///
/// # Arguments
/// * `data` - Flattened data array (n_points × dim)
/// * `n_points` - Number of data points
/// * `dim` - Dimension of each point
/// * `distance` - Distance function
/// * `params` - Algorithm parameters
/// * `rng` - Random number generator (FastRng for performance)
///
/// # Returns
/// A `NeighborHeap` containing the k-nearest neighbors for each point.
pub fn nn_descent<D: Distance<f32> + Sync>(
    data: &[f32],
    n_points: usize,
    dim: usize,
    distance: &D,
    params: &NNDescentParams,
    rng: &mut FastRng,
) -> (NeighborHeap, Vec<FlatTree>) {
    use std::time::Instant;
    
    let effective_max_candidates = params.max_candidates.min(60).min(params.n_neighbors);

    let t_forest_start = Instant::now();
    
    if params.verbose {
        println!("Building RP forest with {} trees...", params.n_trees);
    }

    // Build random projection forest for initialization
    let forest = build_rp_forest(
        data,
        n_points,
        dim,
        params.n_trees,
        params.leaf_size,
        rng,
        params.angular,
        params.max_depth,
    );

    let t_forest = t_forest_start.elapsed();

    // Initialize neighbor graph from tree leaves
    let mut neighbor_graph = NeighborHeap::new(n_points, params.n_neighbors);
    let leaf_array = rptree_leaf_array(&forest);

    let t_init_start = Instant::now();
    
    if params.verbose {
        println!("Initializing graph from {} leaves...", leaf_array.len());
    }

    initialize_from_leaves(
        &mut neighbor_graph,
        &leaf_array,
        data,
        dim,
        distance,
    );

    let t_init = t_init_start.elapsed();

    if params.verbose {
        println!("Running NN-descent for up to {} iterations...", params.n_iters);
    }

    let mut t_candidates_total = std::time::Duration::ZERO;
    let mut t_updates_total = std::time::Duration::ZERO;
    let mut actual_iters = 0;

    // NN-descent iterations
    for iter in 0..params.n_iters {
        actual_iters = iter + 1;
        
        // Build candidate sets (separating new from old)
        // This also marks neighbors that appear in new_candidates as old
        let t_cand_start = Instant::now();
        let candidates = CandidateSets::build_from_graph(
            &mut neighbor_graph,
            effective_max_candidates,
            rng,
        );
        t_candidates_total += t_cand_start.elapsed();

        // Generate and apply updates
        let t_upd_start = Instant::now();
        let n_changes = update_iteration(
            &mut neighbor_graph,
            &candidates,
            data,
            dim,
            distance,
        );
        t_updates_total += t_upd_start.elapsed();

        if params.verbose {
            println!("Iteration {}: {} updates", iter + 1, n_changes);
        }

        // Check convergence
        let threshold = (params.delta * params.n_neighbors as f32 * n_points as f32) as usize;
        if n_changes <= threshold {
            if params.verbose {
                println!("Converged after {} iterations", iter + 1);
            }
            break;
        }
        
        // Note: Neighbors are marked as old inside CandidateSets::build_from_graph
        // (only those that appear in new_candidates, matching PyNNDescent behavior)
    }

    if params.verbose {
        println!("\n=== Timing Breakdown ===");
        println!("Forest building:    {:>8.3}ms", t_forest.as_secs_f64() * 1000.0);
        println!("Leaf initialization:{:>8.3}ms", t_init.as_secs_f64() * 1000.0);
        println!("Candidate building: {:>8.3}ms ({} iters)", t_candidates_total.as_secs_f64() * 1000.0, actual_iters);
        println!("Update iterations:  {:>8.3}ms ({} iters)", t_updates_total.as_secs_f64() * 1000.0, actual_iters);
        let total = t_forest + t_init + t_candidates_total + t_updates_total;
        println!("Total measured:     {:>8.3}ms", total.as_secs_f64() * 1000.0);
    }

    (neighbor_graph, forest)
}

/// Initialize the neighbor graph from RP tree leaves (parallel version).
/// 
/// This uses a block-based approach similar to PyNNDescent for efficient
/// parallel updates without locks.
fn initialize_from_leaves<D: Distance<f32> + Sync>(
    graph: &mut NeighborHeap,
    leaves: &[Vec<i32>],
    data: &[f32],
    dim: usize,
    distance: &D,
) {
    let n_points = graph.n_points;
    let n_leaves = leaves.len();
    
    // Get current distance thresholds for filtering
    let dist_thresholds: Vec<f32> = (0..n_points)
        .map(|i| graph.max_distance(i))
        .collect();
    
    // Process in blocks - generate updates in parallel, apply in parallel by vertex block
    let n_threads = rayon::current_num_threads();
    let block_size = (n_threads * 64).max(128);
    
    // Pre-allocate update storage per thread
    // Estimate max updates: block_size * leaf_size^2 / 2 per thread
    let max_leaf_size = leaves.iter().map(|l| l.len()).max().unwrap_or(0);
    let updates_per_thread = (block_size * max_leaf_size * max_leaf_size / (2 * n_threads)).max(1024);
    
    let vertex_block_size = (n_points + n_threads - 1) / n_threads;
    
    // Process leaves in blocks
    for block_start in (0..n_leaves).step_by(block_size) {
        let block_end = (block_start + block_size).min(n_leaves);
        let leaf_block = &leaves[block_start..block_end];
        
        // Generate updates in parallel (each thread handles a slice of leaves)
        let updates: Vec<Vec<(i32, i32, f32)>> = (0..n_threads)
            .into_par_iter()
            .map(|t| {
                let mut thread_updates = Vec::with_capacity(updates_per_thread);
                let leaves_per_thread = (leaf_block.len() + n_threads - 1) / n_threads;
                let start_leaf = t * leaves_per_thread;
                let end_leaf = (start_leaf + leaves_per_thread).min(leaf_block.len());
                
                for leaf_idx in start_leaf..end_leaf {
                    let leaf = &leaf_block[leaf_idx];
                    
                    for i in 0..leaf.len() {
                        let p = leaf[i];
                        if p < 0 {
                            break;
                        }
                        let p_usize = p as usize;
                        let point_p = &data[p_usize * dim..(p_usize + 1) * dim];
                        
                        for j in (i + 1)..leaf.len() {
                            let q = leaf[j];
                            if q < 0 {
                                break;
                            }
                            let q_usize = q as usize;
                            let point_q = &data[q_usize * dim..(q_usize + 1) * dim];
                            
                            let d = distance.distance(point_p, point_q);
                            
                            // Filter by threshold (like PyNNDescent)
                            let max_threshold = dist_thresholds[p_usize].max(dist_thresholds[q_usize]);
                            if d < max_threshold {
                                thread_updates.push((p, q, d));
                            }
                        }
                    }
                }
                
                thread_updates
            })
            .collect();
        
        // Apply updates in parallel by vertex block (avoid write conflicts)
        let updates_ref = &updates;
        (0..n_threads).into_par_iter().for_each(|t| {
            let v_block_start = t * vertex_block_size;
            let v_block_end = (v_block_start + vertex_block_size).min(n_points);
            
            // Get mutable access to this thread's vertex block
            // SAFETY: Each thread writes to disjoint vertex blocks
            let graph_ptr = graph as *const NeighborHeap as *mut NeighborHeap;
            let graph_mut = unsafe { &mut *graph_ptr };
            
            for thread_updates in updates_ref.iter() {
                for &(p, q, d) in thread_updates {
                    let p_usize = p as usize;
                    let q_usize = q as usize;
                    
                    if p_usize >= v_block_start && p_usize < v_block_end {
                        graph_mut.checked_flagged_push(p_usize, q, d, true);
                    }
                    if q_usize >= v_block_start && q_usize < v_block_end {
                        graph_mut.checked_flagged_push(q_usize, p, d, true);
                    }
                }
            }
        });
    }
}

/// A potential update to the neighbor graph, stored as compact 12-byte struct.
#[derive(Clone, Copy)]
#[repr(C)]
struct PotentialUpdate {
    point: i32,
    neighbor: i32,
    distance: f32,
}

/// Run one iteration of NN-descent updates using block-based processing.
///
/// Matches PyNNDescent's `process_candidates` / `generate_graph_update_array`:
/// - Processes candidate rows in blocks of BLOCK_SIZE (16384)
/// - Within each block, rows are split across threads
/// - Each thread writes to pre-allocated update storage (no dynamic allocation)
/// - Updates applied via block-based vertex ownership
///
/// This avoids the per-point Vec allocation of the previous approach, which
/// created n_points separate allocations per iteration.
fn update_iteration<D: Distance<f32> + Sync>(
    graph: &mut NeighborHeap,
    candidates: &CandidateSets,
    data: &[f32],
    dim: usize,
    distance: &D,
) -> usize {
    let n_points = graph.n_points;
    let n_threads = rayon::current_num_threads().max(1);
    let max_candidates = candidates.max_candidates;

    const BLOCK_SIZE: usize = 16384;

    let rows_per_block = BLOCK_SIZE.min(n_points);
    let max_updates_per_thread = 
        ((max_candidates * max_candidates + max_candidates * (max_candidates - 1) / 2)
            * rows_per_block / n_threads)
        + 1024;

    // Per-thread, per-destination-block update buckets
    // thread_buckets[gen_thread][dest_block] = Vec<PotentialUpdate>
    let vertex_block_size = (n_points + n_threads - 1) / n_threads;
    let n_vertex_blocks = n_threads; // One vertex block per thread for application elision
    
    let mut thread_buckets: Vec<Vec<Vec<PotentialUpdate>>> = (0..n_threads)
        .map(|_| {
            (0..n_vertex_blocks)
                .map(|_| Vec::with_capacity(max_updates_per_thread / n_vertex_blocks + 64))
                .collect()
        })
        .collect();

    use std::sync::atomic::{AtomicUsize, Ordering};
    let total_changes = AtomicUsize::new(0);

    let n_blocks = (n_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        let block_end = ((block_idx + 1) * BLOCK_SIZE).min(n_points);
        let block_len = block_end - block_start;
        let rows_per_thread = (block_len + n_threads - 1) / n_threads;

        let thresholds: Vec<f32> = (0..n_points)
            .map(|i| graph.max_distance(i))
            .collect();

        // Clear all buckets
        for thread_buck in thread_buckets.iter_mut() {
            for bucket in thread_buck.iter_mut() {
                bucket.clear();
            }
        }

        let thread_buckets_ref = &thread_buckets;
        
        // Generate updates and bucket by destination vertex block
        (0..n_threads).into_par_iter().for_each(|t| {
            let row_start = block_start + t * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(block_end);
            
            // SAFETY: Each thread writes to its own thread_buckets[t]
            let buckets_ptr = thread_buckets_ref.as_ptr() as *mut Vec<Vec<PotentialUpdate>>;
            let local_buckets = unsafe { &mut *buckets_ptr.add(t) };
            
            for i in row_start..row_end {
                let new_cands = candidates.get_new(i);
                let old_cands = candidates.get_old(i);

                for j in 0..max_candidates {
                    let p = new_cands[j];
                    if p < 0 {
                        continue;
                    }
                    let p_usize = p as usize;
                    let data_p = &data[p_usize * dim..(p_usize + 1) * dim];
                    let thresh_p = thresholds[p_usize];

                    for k in (j + 1)..max_candidates {
                        let q = new_cands[k];
                        if q < 0 {
                            continue;
                        }
                        let q_usize = q as usize;
                        let max_thresh = thresh_p.max(thresholds[q_usize]);
                        let d = distance.distance(data_p, &data[q_usize * dim..(q_usize + 1) * dim]);

                        if d <= max_thresh {
                            let update = PotentialUpdate { point: p, neighbor: q, distance: d };
                            // Bucket by p's vertex block
                            let p_block = p_usize / vertex_block_size;
                            let q_block = q_usize / vertex_block_size;
                            local_buckets[p_block.min(n_vertex_blocks - 1)].push(update);
                            if p_block != q_block {
                                local_buckets[q_block.min(n_vertex_blocks - 1)].push(update);
                            }
                        }
                    }

                    for k in 0..max_candidates {
                        let q = old_cands[k];
                        if q < 0 || p == q {
                            continue;
                        }
                        let q_usize = q as usize;
                        let max_thresh = thresh_p.max(thresholds[q_usize]);
                        let d = distance.distance(data_p, &data[q_usize * dim..(q_usize + 1) * dim]);

                        if d <= max_thresh {
                            let update = PotentialUpdate { point: p, neighbor: q, distance: d };
                            let p_block = p_usize / vertex_block_size;
                            let q_block = q_usize / vertex_block_size;
                            local_buckets[p_block.min(n_vertex_blocks - 1)].push(update);
                            if p_block != q_block {
                                local_buckets[q_block.min(n_vertex_blocks - 1)].push(update);
                            }
                        }
                    }
                }
            }
        });

        // Apply updates - each thread only reads buckets destined for its vertex block
        let thread_buckets_apply = &thread_buckets;

        (0..n_threads).into_par_iter().for_each(|t| {
            let v_block_start = t * vertex_block_size;
            let v_block_end = (v_block_start + vertex_block_size).min(n_points);
            
            let graph_ptr = graph as *const NeighborHeap as *mut NeighborHeap;
            let graph_mut = unsafe { &mut *graph_ptr };
            
            let mut local_changes = 0usize;
            
            // Only scan updates destined for this vertex block
            for src_thread in 0..n_threads {
                let bucket = &thread_buckets_apply[src_thread][t];
                
                for update in bucket.iter() {
                    let p = update.point as usize;
                    let q = update.neighbor;
                    let d = update.distance;
                    
                    // Apply symmetrically, but only for vertices in our block
                    if p >= v_block_start && p < v_block_end {
                        if graph_mut.checked_flagged_push(p, q, d, true) {
                            local_changes += 1;
                        }
                    }
                    let q_usize = q as usize;
                    if q_usize >= v_block_start && q_usize < v_block_end {
                        if graph_mut.checked_flagged_push(q_usize, update.point, d, true) {
                            local_changes += 1;
                        }
                    }
                }
            }
            
            total_changes.fetch_add(local_changes, Ordering::Relaxed);
        });
    }

    total_changes.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::SquaredEuclidean;

    fn create_test_data(n: usize, dim: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                data.push((i * dim + j) as f32 * 0.1);
            }
        }
        data
    }

    #[test]
    fn test_nn_descent_basic() {
        let n_points = 100;
        let dim = 10;
        let data = create_test_data(n_points, dim);
        let distance = SquaredEuclidean;
        let mut rng = FastRng::new(42);

        let params = NNDescentParams {
            n_neighbors: 10,
            n_trees: 2,
            leaf_size: 20,
            max_candidates: 20,
            n_iters: 5,
            delta: 0.001,
            angular: false,
            max_depth: 100,
            verbose: false,
        };

        let (graph, _forest) = nn_descent(&data, n_points, dim, &distance, &params, &mut rng);

        // Check that each point has neighbors
        for point in 0..n_points {
            let (indices, distances, _) = graph.get_row(point);
            
            // Should have some valid neighbors
            let valid_count = indices.iter().filter(|&&x| x >= 0).count();
            assert!(valid_count > 0, "Point {} has no valid neighbors", point);
            
            // Distances should be finite for valid neighbors
            for (&idx, &dist) in indices.iter().zip(distances.iter()) {
                if idx >= 0 {
                    assert!(dist.is_finite(), "Infinite distance for point {}", point);
                }
            }
        }
    }

    #[test]
    fn test_nn_descent_self_not_neighbor() {
        let n_points = 50;
        let dim = 5;
        let data = create_test_data(n_points, dim);
        let distance = SquaredEuclidean;
        let mut rng = FastRng::new(42);

        let params = NNDescentParams::new(5);

        let (graph, _) = nn_descent(&data, n_points, dim, &distance, &params, &mut rng);

        // No point should have itself as a neighbor
        for point in 0..n_points {
            let (indices, _, _) = graph.get_row(point);
            assert!(
                !indices.contains(&(point as i32)),
                "Point {} has itself as neighbor",
                point
            );
        }
    }
}
