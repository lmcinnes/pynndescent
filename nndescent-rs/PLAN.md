# Comprehensive Plan: Rust Port of PyNNDescent

## Executive Summary

This plan outlines the development of **`nndescent-rs`**, a high-performance Rust implementation of the NN-Descent algorithm for approximate k-nearest neighbor graph construction and search. The library will provide:

1. **Standalone Rust library** (`nndescent-core`) - Pure Rust with SIMD optimizations
2. **Python bindings** (`pynndescent-rs`) - PyO3-based bindings exposing both high-level and low-level APIs
3. **Modular components** for integration into existing PyNNDescent as optional accelerated backends

---

## Part 1: Architecture Overview

### 1.1 Crate Structure

```
nndescent-rs/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── nndescent-core/           # Core Rust library
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── distance/         # SIMD distance functions
│   │   │   │   ├── mod.rs
│   │   │   │   ├── traits.rs     # Distance trait definitions
│   │   │   │   ├── euclidean.rs  # L2 variants (f32, quantized)
│   │   │   │   ├── cosine.rs     # Cosine/dot variants
│   │   │   │   ├── inner_product.rs
│   │   │   │   └── quantized.rs  # SQ4, SQ8, binary
│   │   │   ├── heap/             # Heap data structures
│   │   │   │   ├── mod.rs
│   │   │   │   ├── neighbor_heap.rs
│   │   │   │   └── candidate_heap.rs
│   │   │   ├── graph/            # Graph structures
│   │   │   │   ├── mod.rs
│   │   │   │   ├── neighbor_graph.rs
│   │   │   │   └── search_graph.rs
│   │   │   ├── tree/             # Random projection trees
│   │   │   │   ├── mod.rs
│   │   │   │   ├── rp_tree.rs
│   │   │   │   └── hub_tree.rs
│   │   │   ├── nndescent/        # Core algorithm
│   │   │   │   ├── mod.rs
│   │   │   │   ├── builder.rs    # Index construction
│   │   │   │   ├── candidates.rs # Candidate management
│   │   │   │   └── update.rs     # Graph update logic
│   │   │   ├── search/           # Search algorithms
│   │   │   │   ├── mod.rs
│   │   │   │   └── greedy.rs
│   │   │   ├── diversify/        # Graph pruning
│   │   │   │   ├── mod.rs
│   │   │   │   └── degree_aware.rs
│   │   │   ├── rng.rs            # Tau-rand PRNG
│   │   │   ├── visited.rs        # Bit-packed visited set
│   │   │   └── index.rs          # Main NNDescentIndex struct
│   │   └── Cargo.toml
│   │
│   ├── nndescent-simd/           # SIMD kernels (separate for feature flags)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── avx2.rs
│   │   │   ├── avx512.rs
│   │   │   └── detect.rs         # CPU feature detection
│   │   └── Cargo.toml
│   │
│   └── pynndescent-rs/           # Python bindings
│       ├── src/
│       │   ├── lib.rs
│       │   ├── index.rs          # High-level NNDescent class
│       │   ├── distance.rs       # Distance function exports
│       │   └── utils.rs          # Conversion utilities
│       ├── Cargo.toml
│       └── pyproject.toml        # Maturin config
│
├── benches/                      # Criterion benchmarks
│   ├── distance_bench.rs
│   ├── nndescent_bench.rs
│   └── search_bench.rs
│
└── examples/
    ├── basic_usage.rs
    └── custom_distance.rs
```

### 1.2 Feature Flags

```toml
# nndescent-core/Cargo.toml
[features]
default = ["std", "rayon"]
std = []
rayon = ["dep:rayon"]           # Parallel execution
simd = ["nndescent-simd"]       # Explicit SIMD (auto-detected)
mmap = ["dep:memmap2"]          # Memory-mapped indexes
serde = ["dep:serde"]           # Serialization
```

---

## Part 2: Distance Function Implementations

### 2.1 Distance Trait Design

```rust
/// Core distance trait with compile-time metric selection
pub trait Distance<T>: Send + Sync {
    /// Compute distance between two vectors
    fn distance(&self, a: &[T], b: &[T]) -> f32;
    
    /// Batch distance computation (for SIMD optimization)
    fn distance_batch(&self, query: &[T], data: &[&[T]], results: &mut [f32]) {
        for (i, d) in data.iter().enumerate() {
            results[i] = self.distance(query, d);
        }
    }
    
    /// Whether this is a "proxy" distance requiring correction
    fn needs_correction(&self) -> bool { false }
    
    /// Apply distance correction (e.g., sqrt for squared euclidean)
    fn correct(&self, d: f32) -> f32 { d }
}

/// Marker trait for distances that can use squared form
pub trait HasSquaredForm: Distance<f32> {
    type Squared: Distance<f32>;
    fn squared(&self) -> Self::Squared;
}
```

### 2.2 SIMD Distance Implementations (Following Glass Patterns)

Based on the pyglass SIMD implementations, we'll implement:

#### L2 Squared (Euclidean)
```rust
// AVX-512 version (d must be multiple of 16)
#[cfg(target_feature = "avx512f")]
pub fn l2_sqr_avx512(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut sum = _mm512_setzero_ps();
        let chunks = x.len() / 16;
        
        for i in 0..chunks {
            let xx = _mm512_loadu_ps(x.as_ptr().add(i * 16));
            let yy = _mm512_loadu_ps(y.as_ptr().add(i * 16));
            let diff = _mm512_sub_ps(xx, yy);
            sum = _mm512_fmadd_ps(diff, diff, sum);  // Use FMA for better performance
        }
        
        _mm512_reduce_add_ps(sum)
    }
}

// AVX2 version (d must be multiple of 8)
#[cfg(target_feature = "avx2")]
pub fn l2_sqr_avx2(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut sum = _mm256_setzero_ps();
        let chunks = x.len() / 8;
        
        for i in 0..chunks {
            let xx = _mm256_loadu_ps(x.as_ptr().add(i * 8));
            let yy = _mm256_loadu_ps(y.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(xx, yy);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        
        // Horizontal sum
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(hi, lo);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    }
}
```

#### Inner Product (Negative for similarity)
```rust
#[cfg(target_feature = "avx512f")]
pub fn inner_product_avx512(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut sum = _mm512_setzero_ps();
        let chunks = x.len() / 16;
        
        for i in 0..chunks {
            let xx = _mm512_loadu_ps(x.as_ptr().add(i * 16));
            let yy = _mm512_loadu_ps(y.as_ptr().add(i * 16));
            sum = _mm512_fmadd_ps(xx, yy, sum);
        }
        
        -_mm512_reduce_add_ps(sum)  // Negate for distance (higher similarity = lower distance)
    }
}
```

#### Quantized Distances (SQ8, SQ4)

Following the glass pattern for asymmetric quantization:

```rust
/// SQ8 quantization: stores min/diff per dimension for dequantization
pub struct SQ8Quantizer {
    min_vals: Vec<f32>,    // Per-dimension minimum
    diff_vals: Vec<f32>,   // Per-dimension (max - min) / 255
    dim: usize,
}

/// Asymmetric distance: float query vs quantized data
#[cfg(target_feature = "avx512f")]  
pub fn l2_sqr_sq8_avx512(
    x: &[f32],           // Query (float)
    y: &[u8],            // Quantized data
    mi: &[f32],          // min_vals
    dif: &[f32],         // diff_vals  
) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut sum = _mm512_setzero_ps();
        let dot5 = _mm512_set1_ps(0.5);
        let const_256 = _mm512_set1_ps(256.0);
        
        for i in (0..x.len()).step_by(16) {
            // Load and convert quantized values
            let zz = _mm_loadu_si128(y.as_ptr().add(i) as *const __m128i);
            let zzz = _mm512_cvtepu8_epi32(zz);
            let mut yy = _mm512_cvtepi32_ps(zzz);
            
            // Dequantize: y = (code + 0.5) * diff + min * 256
            yy = _mm512_add_ps(yy, dot5);
            let mi512 = _mm512_loadu_ps(mi.as_ptr().add(i));
            let dif512 = _mm512_loadu_ps(dif.as_ptr().add(i));
            yy = _mm512_mul_ps(yy, dif512);
            yy = _mm512_add_ps(yy, _mm512_mul_ps(mi512, const_256));
            
            // Compute squared difference
            let xx = _mm512_loadu_ps(x.as_ptr().add(i));
            let d = _mm512_sub_ps(_mm512_mul_ps(xx, const_256), yy);
            sum = _mm512_fmadd_ps(d, d, sum);
        }
        
        _mm512_reduce_add_ps(sum)
    }
}
```

### 2.3 Runtime Dispatch

```rust
/// Distance function dispatcher with runtime CPU detection
pub struct DistanceDispatcher {
    l2_sqr: fn(&[f32], &[f32]) -> f32,
    inner_product: fn(&[f32], &[f32]) -> f32,
    // ... other metrics
}

impl DistanceDispatcher {
    pub fn new() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                return Self {
                    l2_sqr: l2_sqr_avx512,
                    inner_product: inner_product_avx512,
                };
            }
            if is_x86_feature_detected!("avx2") {
                return Self {
                    l2_sqr: l2_sqr_avx2,
                    inner_product: inner_product_avx2,
                };
            }
        }
        // Fallback to scalar
        Self {
            l2_sqr: l2_sqr_scalar,
            inner_product: inner_product_scalar,
        }
    }
}
```

---

## Part 3: Core Data Structures

### 3.1 Neighbor Heap (3-component structure matching PyNNDescent)

```rust
/// Neighbor heap maintaining k-nearest neighbors with new/old flags
/// Layout optimized for cache efficiency
pub struct NeighborHeap {
    /// Neighbor indices (n_points × k), row-major
    indices: Vec<i32>,
    /// Distances (n_points × k), row-major
    distances: Vec<f32>,
    /// Flags: 1 = new, 0 = old (n_points × k)
    flags: Vec<u8>,
    /// Number of points
    n_points: usize,
    /// Number of neighbors per point
    k: usize,
}

impl NeighborHeap {
    /// Push with duplicate checking and flag setting
    #[inline]
    pub fn checked_flagged_push(
        &mut self,
        point: usize,
        neighbor: i32,
        distance: f32,
        is_new: bool,
    ) -> bool {
        let offset = point * self.k;
        let priorities = &mut self.distances[offset..offset + self.k];
        let indices = &mut self.indices[offset..offset + self.k];
        let flags = &mut self.flags[offset..offset + self.k];
        
        // Early exit if distance is too large
        if distance >= priorities[0] {
            return false;
        }
        
        // Check for duplicates
        for i in 0..self.k {
            if indices[i] == neighbor {
                return false;
            }
        }
        
        // Insert at root and sift down
        priorities[0] = distance;
        indices[0] = neighbor;
        flags[0] = is_new as u8;
        
        self.sift_down(point, 0);
        true
    }
    
    #[inline]
    fn sift_down(&mut self, point: usize, mut pos: usize) {
        // ... heap sift-down implementation
    }
}
```

### 3.2 Candidate Sets (New/Old Separation)

```rust
/// Candidate neighbors separated into new and old sets
/// This enables the key optimization: skip (old, old) pairs
pub struct CandidateSets {
    new_candidates: Vec<Vec<i32>>,  // Per-vertex new candidates
    old_candidates: Vec<Vec<i32>>,  // Per-vertex old candidates
    max_candidates: usize,
}

impl CandidateSets {
    /// Build candidates from current graph, separating new from old
    pub fn build_from_graph(
        graph: &NeighborHeap,
        max_candidates: usize,
        rng: &mut TauRand,
    ) -> Self {
        let n_vertices = graph.n_points;
        let mut new_candidates = vec![Vec::with_capacity(max_candidates); n_vertices];
        let mut old_candidates = vec![Vec::with_capacity(max_candidates); n_vertices];
        
        // Parallel candidate collection using block-based ownership
        // (matches PyNNDescent's new_build_candidates)
        
        // ... implementation
        
        Self { new_candidates, old_candidates, max_candidates }
    }
}
```

### 3.3 Update Array (Block-Based Parallelism)

```rust
/// Pre-allocated update array for lock-free parallel graph updates
pub struct UpdateArray {
    /// Updates per thread: (source, target, distance)
    updates: Vec<Vec<(i32, i32, f32)>>,
    /// Number of valid updates per thread
    counts: Vec<usize>,
    n_threads: usize,
}

impl UpdateArray {
    /// Generate updates from candidate pairs
    pub fn generate_updates<D: Distance<f32>>(
        &mut self,
        new_candidates: &[Vec<i32>],
        old_candidates: &[Vec<i32>],
        data: &[f32],  // Flattened (n × d)
        dim: usize,
        dist_thresholds: &[f32],
        distance: &D,
    ) {
        // Process in parallel blocks
        self.updates.par_iter_mut()
            .zip(self.counts.par_iter_mut())
            .enumerate()
            .for_each(|(thread_id, (updates, count))| {
                *count = 0;
                let block_start = /* calculate block range */;
                let block_end = /* ... */;
                
                for i in block_start..block_end {
                    // Compare (new, new) pairs
                    for j in 0..new_candidates[i].len() {
                        let p = new_candidates[i][j];
                        if p < 0 { continue; }
                        
                        for k in j..new_candidates[i].len() {
                            let q = new_candidates[i][k];
                            if q < 0 { continue; }
                            
                            let d = distance.distance(
                                &data[p as usize * dim..(p as usize + 1) * dim],
                                &data[q as usize * dim..(q as usize + 1) * dim],
                            );
                            
                            let max_thresh = dist_thresholds[p as usize]
                                .max(dist_thresholds[q as usize]);
                            
                            if d <= max_thresh {
                                updates[*count] = (p, q, d);
                                *count += 1;
                            }
                        }
                    }
                    
                    // Compare (new, old) pairs
                    // ... similar logic
                }
            });
    }
    
    /// Apply updates to graph (lock-free via vertex block ownership)
    pub fn apply_updates(&self, graph: &mut NeighborHeap) -> usize {
        let n_vertices = graph.n_points;
        let block_size = (n_vertices + self.n_threads - 1) / self.n_threads;
        
        (0..self.n_threads).into_par_iter()
            .map(|thread_id| {
                let block_start = thread_id * block_size;
                let block_end = (block_start + block_size).min(n_vertices);
                let mut changes = 0;
                
                // Scan ALL updates, but only apply where p or q is in our block
                for t in 0..self.n_threads {
                    for &(p, q, d) in &self.updates[t][..self.counts[t]] {
                        if (p as usize) >= block_start && (p as usize) < block_end {
                            if graph.checked_flagged_push(p as usize, q, d, true) {
                                changes += 1;
                            }
                        }
                        if (q as usize) >= block_start && (q as usize) < block_end {
                            if graph.checked_flagged_push(q as usize, p, d, true) {
                                changes += 1;
                            }
                        }
                    }
                }
                
                changes
            })
            .sum()
    }
}
```

### 3.4 Visited Set (Bit-Packed)

```rust
/// Bit-packed visited set for search
pub struct VisitedSet {
    bits: Vec<u8>,
}

impl VisitedSet {
    #[inline]
    pub fn check_and_mark(&mut self, idx: i32) -> bool {
        let loc = (idx >> 3) as usize;
        let mask = 1u8 << (idx & 7);
        let was_visited = (self.bits[loc] & mask) != 0;
        self.bits[loc] |= mask;
        was_visited
    }
    
    #[inline]
    pub fn clear(&mut self) {
        self.bits.fill(0);
    }
}
```

---

## Part 4: Random Projection Trees

### 4.1 Tree Structure

```rust
/// Compact tree format for search (matches PyNNDescent's FlatTree)
pub struct FlatTree {
    /// Hyperplane vectors (n_nodes × dim)
    hyperplanes: Vec<f32>,
    /// Hyperplane offsets
    offsets: Vec<f32>,
    /// Child node indices (n_nodes × 2), negative = leaf with point range
    children: Vec<[i32; 2]>,
    /// Point indices in leaf order
    indices: Vec<i32>,
    dim: usize,
}

impl FlatTree {
    /// Search tree to find leaf containing query point
    #[inline]
    pub fn search(&self, point: &[f32], rng: &mut TauRand) -> (usize, usize) {
        let mut node = 0usize;
        
        while self.children[node][0] > 0 {
            let hp = &self.hyperplanes[node * self.dim..(node + 1) * self.dim];
            let offset = self.offsets[node];
            
            // Dot product with hyperplane
            let mut margin: f32 = offset;
            for i in 0..self.dim {
                margin += point[i] * hp[i];
            }
            
            // Choose side (with random tie-breaking near boundary)
            let side = if margin.abs() < 1e-8 {
                (rng.next_int() & 1) as usize
            } else {
                (margin > 0.0) as usize
            };
            
            node = self.children[node][side] as usize;
        }
        
        // Return leaf bounds
        let leaf_start = (-self.children[node][0]) as usize;
        let leaf_end = (-self.children[node][1]) as usize;
        (leaf_start, leaf_end)
    }
}
```

### 4.2 Hub Tree Construction

```rust
/// Build hub tree using graph structure to minimize edge cuts
pub fn build_hub_tree(
    data: &[f32],
    dim: usize,
    neighbor_graph: &NeighborHeap,
    leaf_size: usize,
    rng: &mut TauRand,
    angular: bool,
) -> FlatTree {
    // ... implementation matching PyNNDescent's make_hub_tree
}
```

---

## Part 5: NN-Descent Algorithm

### 5.1 Main Algorithm Structure

```rust
pub struct NNDescentBuilder<D: Distance<f32>> {
    data: Vec<f32>,          // Flattened n × d
    n_points: usize,
    dim: usize,
    distance: D,
    n_neighbors: usize,
    n_trees: usize,
    leaf_size: usize,
    max_candidates: usize,
    n_iters: usize,
    delta: f32,
    n_threads: usize,
    verbose: bool,
}

impl<D: Distance<f32>> NNDescentBuilder<D> {
    pub fn build(self) -> NNDescentIndex<D> {
        let mut rng = TauRand::new(self.random_seed);
        
        // 1. Build random projection forest
        let rp_forest = if self.tree_init {
            if self.verbose { println!("Building RP forest..."); }
            build_rp_forest(
                &self.data, self.dim, self.n_points,
                self.n_trees, self.leaf_size, &mut rng,
            )
        } else {
            vec![]
        };
        
        // 2. Initialize neighbor graph from RP forest leaves
        let mut neighbor_graph = NeighborHeap::new(self.n_points, self.n_neighbors);
        if !rp_forest.is_empty() {
            initialize_from_forest(&mut neighbor_graph, &rp_forest, &self.data, self.dim, &self.distance);
        }
        
        // 3. NN-descent iterations
        let effective_max_candidates = self.max_candidates.min(60).min(self.n_neighbors);
        let mut update_array = UpdateArray::new(self.n_threads, /* max_updates */);
        
        for iter in 0..self.n_iters {
            if self.verbose { println!("Iteration {}/{}", iter + 1, self.n_iters); }
            
            // Build new/old candidate sets
            let candidates = CandidateSets::build_from_graph(
                &neighbor_graph,
                effective_max_candidates,
                &mut rng,
            );
            
            // Generate updates in parallel
            let dist_thresholds: Vec<f32> = (0..self.n_points)
                .map(|i| neighbor_graph.max_distance(i))
                .collect();
            
            update_array.generate_updates(
                &candidates.new_candidates,
                &candidates.old_candidates,
                &self.data,
                self.dim,
                &dist_thresholds,
                &self.distance,
            );
            
            // Apply updates
            let n_changes = update_array.apply_updates(&mut neighbor_graph);
            
            // Check convergence
            let threshold = (self.delta * self.n_neighbors as f32 * self.n_points as f32) as usize;
            if n_changes <= threshold {
                if self.verbose { 
                    println!("Converged after {} iterations", iter + 1);
                }
                break;
            }
        }
        
        // 4. Build search structures
        // ... hub tree, diversified graph, etc.
        
        NNDescentIndex { /* ... */ }
    }
}
```

---

## Part 6: Search Implementation

### 6.1 Greedy Graph Search

```rust
impl<D: Distance<f32>> NNDescentIndex<D> {
    pub fn query(&self, queries: &[f32], k: usize, epsilon: f32) -> (Vec<i32>, Vec<f32>) {
        let n_queries = queries.len() / self.dim;
        let mut result_indices = vec![-1i32; n_queries * k];
        let mut result_distances = vec![f32::INFINITY; n_queries * k];
        
        // Parallel query processing
        result_indices.par_chunks_mut(k)
            .zip(result_distances.par_chunks_mut(k))
            .enumerate()
            .for_each(|(i, (indices, distances))| {
                let query = &queries[i * self.dim..(i + 1) * self.dim];
                self.search_single(query, k, epsilon, indices, distances);
            });
        
        (result_indices, result_distances)
    }
    
    fn search_single(
        &self,
        query: &[f32],
        k: usize,
        epsilon: f32,
        result_indices: &mut [i32],
        result_distances: &mut [f32],
    ) {
        let mut visited = VisitedSet::new(self.n_points);
        let mut heap = BinaryHeap::with_capacity(k);
        let mut seed_set = BinaryHeap::new();
        
        // Initialize from tree
        let (leaf_start, leaf_end) = self.search_tree.search(query, &mut self.rng.clone());
        for idx in self.search_tree.indices[leaf_start..leaf_end].iter() {
            let d = self.distance.distance(query, self.get_point(*idx as usize));
            heap.push(Reverse((OrderedFloat(d), *idx)));
            seed_set.push(Reverse((OrderedFloat(d), *idx)));
            visited.mark(*idx);
        }
        
        // Graph search
        let distance_bound = |heap: &BinaryHeap<_>| {
            if let Some(Reverse((OrderedFloat(d), _))) = heap.peek() {
                *d + epsilon * (*d - self.min_distance)
            } else {
                f32::INFINITY
            }
        };
        
        while let Some(Reverse((OrderedFloat(d_vertex), vertex))) = seed_set.pop() {
            if d_vertex >= distance_bound(&heap) {
                break;
            }
            
            // Expand neighbors
            for &neighbor in self.get_neighbors(vertex as usize) {
                if neighbor < 0 { continue; }
                if visited.check_and_mark(neighbor) { continue; }
                
                let d = self.distance.distance(query, self.get_point(neighbor as usize));
                
                if d < distance_bound(&heap) {
                    if heap.len() >= k {
                        heap.pop();
                    }
                    heap.push(Reverse((OrderedFloat(d), neighbor)));
                    seed_set.push(Reverse((OrderedFloat(d), neighbor)));
                }
            }
        }
        
        // Extract results
        for (i, Reverse((OrderedFloat(d), idx))) in heap.into_sorted_vec().into_iter().enumerate() {
            result_indices[i] = idx;
            result_distances[i] = if self.distance.needs_correction() {
                self.distance.correct(d)
            } else {
                d
            };
        }
    }
}
```

---

## Part 7: Python Bindings

### 7.1 High-Level API

```rust
// pynndescent-rs/src/lib.rs
use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};

#[pyclass]
pub struct NNDescent {
    inner: nndescent_core::NNDescentIndex<nndescent_core::distance::EuclideanSquared>,
}

#[pymethods]
impl NNDescent {
    #[new]
    #[pyo3(signature = (
        data,
        metric = "euclidean",
        n_neighbors = 30,
        n_trees = None,
        leaf_size = None,
        // ... other params
    ))]
    fn new(
        py: Python<'_>,
        data: PyReadonlyArray2<f32>,
        metric: &str,
        n_neighbors: usize,
        n_trees: Option<usize>,
        leaf_size: Option<usize>,
        // ...
    ) -> PyResult<Self> {
        // ... implementation
    }
    
    fn query<'py>(
        &self,
        py: Python<'py>,
        query_data: PyReadonlyArray2<f32>,
        k: usize,
        epsilon: f32,
    ) -> PyResult<(&'py PyArray2<i32>, &'py PyArray2<f32>)> {
        // ... implementation
    }
    
    #[getter]
    fn neighbor_graph<'py>(&self, py: Python<'py>) -> PyResult<(&'py PyArray2<i32>, &'py PyArray2<f32>)> {
        // ... implementation
    }
}

#[pymodule]
fn pynndescent_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<NNDescent>()?;
    
    // Also expose low-level components
    m.add_function(wrap_pyfunction!(euclidean_distance, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_distance, m)?)?;
    // ...
    
    Ok(())
}
```

### 7.2 Low-Level Component Exposure

```rust
/// Expose distance functions for use in existing PyNNDescent
#[pyfunction]
fn euclidean_distance_batch(
    py: Python<'_>,
    query: PyReadonlyArray1<f32>,
    data: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray1<f32>>> {
    let query = query.as_slice()?;
    let data = data.as_array();
    let n = data.nrows();
    
    let mut results = vec![0.0f32; n];
    
    // Use SIMD-optimized distance computation
    let dispatcher = DistanceDispatcher::new();
    for i in 0..n {
        results[i] = (dispatcher.l2_sqr)(query, data.row(i).as_slice().unwrap()).sqrt();
    }
    
    Ok(PyArray1::from_vec(py, results).to_owned())
}
```

---

## Part 8: Cache Optimization Strategies

### 8.1 Data Layout for Cache Efficiency

```rust
/// Align data to cache line boundaries
#[repr(C, align(64))]
pub struct AlignedData {
    data: Vec<f32>,
}

/// Prefetch hints for distance computation
#[inline]
pub fn prefetch_l1<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}

#[inline]
pub fn prefetch_l2<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T1);
    }
}
```

### 8.2 Blocking for Cache Locality

```rust
/// Process data in blocks that fit in L2 cache
const BLOCK_SIZE: usize = 16384;  // ~64KB of i32 indices

pub fn process_in_blocks<F>(n_items: usize, mut f: F)
where
    F: FnMut(usize, usize),
{
    let n_blocks = (n_items + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for block in 0..n_blocks {
        let start = block * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(n_items);
        f(start, end);
    }
}
```

### 8.3 Data Reordering by Tree Leaf Order

```rust
/// Reorder data according to tree leaf order for better locality
pub fn reorder_by_tree_order(
    data: &mut [f32],
    graph: &mut SearchGraph,
    tree: &FlatTree,
    dim: usize,
) -> Vec<usize> {
    let vertex_order = tree.indices.iter().map(|&i| i as usize).collect::<Vec<_>>();
    
    // Reorder data
    let mut new_data = vec![0.0f32; data.len()];
    for (new_idx, &old_idx) in vertex_order.iter().enumerate() {
        new_data[new_idx * dim..(new_idx + 1) * dim]
            .copy_from_slice(&data[old_idx * dim..(old_idx + 1) * dim]);
    }
    data.copy_from_slice(&new_data);
    
    // Reorder graph
    graph.reorder(&vertex_order);
    
    vertex_order
}
```

---

## Part 9: Serialization

### 9.1 Binary Format

```rust
/// Efficient binary serialization format
pub struct IndexHeader {
    magic: [u8; 8],      // "NNDIDX01"
    version: u32,
    n_points: u64,
    dim: u32,
    n_neighbors: u32,
    metric_id: u32,
    flags: u32,          // quantization type, etc.
}

impl<D: Distance<f32> + Serialize> NNDescentIndex<D> {
    pub fn save<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Write header
        let header = IndexHeader { /* ... */ };
        writer.write_all(bytemuck::bytes_of(&header))?;
        
        // Write data (optionally compressed)
        writer.write_all(bytemuck::cast_slice(&self.data))?;
        
        // Write graph
        writer.write_all(bytemuck::cast_slice(&self.search_graph.indices))?;
        
        // Write tree
        // ...
        
        Ok(())
    }
    
    pub fn load<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        // ...
    }
}
```

### 9.2 Memory-Mapped Index

```rust
#[cfg(feature = "mmap")]
pub struct MmapIndex {
    mmap: memmap2::Mmap,
    header: &'static IndexHeader,
    data: &'static [f32],
    graph_indices: &'static [i32],
    // ...
}

impl MmapIndex {
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        
        // Parse header and set up slices
        // ...
        
        Ok(Self { /* ... */ })
    }
}
```

---

## Part 10: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Project scaffolding with cargo workspace
- [ ] Basic data structures (NeighborHeap, VisitedSet, TauRand)
- [ ] Scalar distance functions with tests
- [ ] Basic heap operations

### Phase 2: SIMD Distance Functions (Weeks 3-4)
- [ ] AVX2 implementations (L2, IP, cosine)
- [ ] AVX-512 implementations
- [ ] SQ8/SQ4 quantized distances
- [ ] Runtime dispatch system
- [ ] Benchmarks against PyNNDescent/numpy

### Phase 3: NN-Descent Algorithm (Weeks 5-7)
- [ ] Random projection tree construction
- [ ] Candidate set building (new/old separation)
- [ ] Update array generation and application
- [ ] Block-based parallel processing
- [ ] Delta convergence checking
- [ ] Full NN-descent iteration loop

### Phase 4: Search Infrastructure (Weeks 8-9)
- [ ] Hub tree construction
- [ ] Graph diversification
- [ ] Degree pruning
- [ ] Greedy graph search
- [ ] Search tree leaf ordering

### Phase 5: Python Bindings (Weeks 10-11)
- [ ] PyO3 bindings for NNDescent class
- [ ] NumPy array interop
- [ ] Low-level function exports
- [ ] Documentation and examples

### Phase 6: Optimization & Polish (Weeks 12-13)
- [ ] Cache optimization tuning
- [ ] Memory-mapped index support
- [ ] Serialization format
- [ ] Comprehensive benchmarks
- [ ] Integration tests with PyNNDescent

---

## Part 11: Benchmark Strategy

### 11.1 Microbenchmarks
```rust
// benches/distance_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_l2_distance(c: &mut Criterion) {
    let dims = [128, 256, 512, 768, 1024];
    
    for &dim in &dims {
        let x = vec![0.5f32; dim];
        let y = vec![0.3f32; dim];
        
        c.bench_with_input(
            BenchmarkId::new("l2_sqr_avx2", dim),
            &(&x, &y),
            |b, (x, y)| b.iter(|| l2_sqr_avx2(x, y)),
        );
    }
}
```

### 11.2 End-to-End Benchmarks
- **Datasets**: MNIST, Fashion-MNIST, SIFT1M, GloVe
- **Metrics**: Index build time, query QPS, recall@k
- **Comparisons**: PyNNDescent, FAISS, hnswlib

---

## Design Decisions

### Answers to Clarification Questions

1. **Rayon vs. raw threads**: Use Rayon - excellent work-stealing parallelism and idiomatic Rust.

2. **Dimension padding**: Pad internally and handle user-facing dimensions transparently for SIMD efficiency.

3. **Product quantization (PQ/OPQ)**: Defer to later phase - focus on core algorithm first.

4. **Index update**: Lower priority - implement after core functionality is complete.

5. **Sparse data**: Design architecture to make sparse support easier to add later, but optimize purely for dense initially.
