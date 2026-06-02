//! Python bindings for nndescent-rs using PyO3.
//!
//! This crate provides Python-compatible classes that mirror the PyNNDescent API.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use nndescent_core::index::{NNDescentBuilder, NNDescentIndex};
use nndescent_core::distance::*;

/// NNDescent index for approximate nearest neighbor search.
///
/// This is the main class for building and querying k-NN graphs.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     2D array of shape (n_samples, n_features) containing the data points.
/// metric : str, default='euclidean'
///     Distance metric to use. Options: 'euclidean', 'l2', 'cosine',
///     'inner_product', 'dot'.
/// n_neighbors : int, default=30
///     Number of neighbors to compute.
/// n_trees : int, default=8
///     Number of random projection trees to build.
/// leaf_size : int, optional
///     Maximum leaf size for RP trees.
/// max_candidates : int, optional
///     Maximum number of candidates per iteration.
/// n_iters : int, optional
///     Number of NN-descent iterations.
/// delta : float, default=0.001
///     Convergence threshold (early stopping if fewer than delta*n*k updates).
/// random_state : int, optional
///     Random seed for reproducibility.
/// verbose : bool, default=False
///     Whether to print progress information.
///
/// Attributes
/// ----------
/// neighbor_graph : tuple of (indices, distances)
///     The k-NN graph as a tuple of 2D arrays.
///
/// Examples
/// --------
/// >>> from pynndescent import NNDescent
/// >>> import numpy as np
/// >>> data = np.random.randn(1000, 128).astype(np.float32)
/// >>> index = NNDescent(data, n_neighbors=15)
/// >>> indices, distances = index.query(data[:10], k=5)
#[pyclass(name = "NNDescent")]
pub struct PyNNDescent {
    /// Stored data
    data: Vec<f32>,
    n_points: usize,
    dim: usize,
    /// Metric type
    metric: Metric,
    /// Index parameters
    n_neighbors: usize,
    /// The internal index (type-erased)
    index_data: Box<dyn AnyIndex>,
}

/// Trait for type-erased index operations.
trait AnyIndex: Send + Sync {
    fn query(&self, queries: &[f32], n_queries: usize, k: usize, epsilon: f32) -> (Vec<i32>, Vec<f32>);
    fn neighbor_indices(&self) -> &[i32];
    fn neighbor_distances(&self) -> &[f32];
}

impl<D: Distance<f32> + Send + Sync> AnyIndex for NNDescentIndex<D> {
    fn query(&self, queries: &[f32], n_queries: usize, k: usize, epsilon: f32) -> (Vec<i32>, Vec<f32>) {
        NNDescentIndex::query(self, queries, n_queries, k, epsilon)
    }
    fn neighbor_indices(&self) -> &[i32] {
        &self.neighbor_indices
    }
    fn neighbor_distances(&self) -> &[f32] {
        &self.neighbor_distances
    }
}

#[pymethods]
impl PyNNDescent {
    #[new]
    #[pyo3(signature = (data, metric="euclidean", n_neighbors=30, n_trees=None, leaf_size=None, max_candidates=None, n_iters=None, delta=0.001, random_state=None, diversify_prob=1.0, pruning_degree_multiplier=1.5, verbose=false))]
    fn new(
        data: PyReadonlyArray2<f32>,
        metric: &str,
        n_neighbors: usize,
        n_trees: Option<usize>,
        leaf_size: Option<usize>,
        max_candidates: Option<usize>,
        n_iters: Option<usize>,
        delta: f32,
        random_state: Option<u64>,
        diversify_prob: f32,
        pruning_degree_multiplier: f32,
        verbose: bool,
    ) -> PyResult<Self> {
        let shape = data.shape();
        let n_points = shape[0];
        let dim = shape[1];

        // Copy data to owned vec in C-contiguous (row-major) order
        let data_vec: Vec<f32> = if data.is_c_contiguous() {
            data.as_slice().unwrap().to_vec()
        } else {
            // F-contiguous or strided: read element by element in row-major order
            let mut vec = Vec::with_capacity(n_points * dim);
            for i in 0..n_points {
                for j in 0..dim {
                    vec.push(*data.get([i, j]).unwrap());
                }
            }
            vec
        };

        // Parse metric
        let parsed_metric = Metric::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // Build index based on metric
        let index_data = Self::build_index(
            &data_vec,
            n_points,
            dim,
            parsed_metric,
            n_neighbors,
            n_trees,
            leaf_size,
            max_candidates,
            n_iters,
            delta,
            random_state.unwrap_or(42),
            diversify_prob,
            pruning_degree_multiplier,
            verbose,
        )?;

        Ok(Self {
            data: data_vec,
            n_points,
            dim,
            metric: parsed_metric,
            n_neighbors,
            index_data,
        })
    }

    /// Query for nearest neighbors.
    ///
    /// Parameters
    /// ----------
    /// query_data : numpy.ndarray
    ///     2D array of shape (n_queries, n_features) containing query points.
    /// k : int, default=10
    ///     Number of neighbors to return.
    /// epsilon : float, default=0.1
    ///     Search expansion factor. Larger values give more accurate results
    ///     but slower queries.
    ///
    /// Returns
    /// -------
    /// indices : numpy.ndarray
    ///     2D array of shape (n_queries, k) containing neighbor indices.
    /// distances : numpy.ndarray
    ///     2D array of shape (n_queries, k) containing distances to neighbors.
    #[pyo3(signature = (query_data, k=10, epsilon=0.1))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        query_data: PyReadonlyArray2<f32>,
        k: usize,
        epsilon: f32,
    ) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<f32>>)> {
        let shape = query_data.shape();
        let n_queries = shape[0];
        let query_dim = shape[1];

        if query_dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match data dimension {}",
                query_dim, self.dim
            )));
        }

        let query_vec: Vec<f32> = if query_data.is_c_contiguous() {
            query_data.as_slice().unwrap().to_vec()
        } else {
            let mut vec = Vec::with_capacity(n_queries * query_dim);
            for i in 0..n_queries {
                for j in 0..query_dim {
                    vec.push(*query_data.get([i, j]).unwrap());
                }
            }
            vec
        };

        let (indices, distances) = self.index_data.query(&query_vec, n_queries, k, epsilon);

        // Create 2D arrays directly
        let indices_arr = PyArray1::from_vec_bound(py, indices);
        let distances_arr = PyArray1::from_vec_bound(py, distances);

        let indices_2d = indices_arr.reshape([n_queries, k])?;
        let distances_2d = distances_arr.reshape([n_queries, k])?;

        Ok((indices_2d, distances_2d))
    }

    /// Get the computed neighbor graph.
    ///
    /// Returns
    /// -------
    /// indices : numpy.ndarray
    ///     2D array of shape (n_samples, n_neighbors) containing neighbor indices.
    /// distances : numpy.ndarray
    ///     2D array of shape (n_samples, n_neighbors) containing neighbor distances.
    #[getter]
    fn neighbor_graph<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<f32>>)> {
        // Return the stored neighbor graph (no re-query needed)
        let indices = self.index_data.neighbor_indices().to_vec();
        let distances = self.index_data.neighbor_distances().to_vec();

        let indices_arr = PyArray1::from_vec_bound(py, indices);
        let distances_arr = PyArray1::from_vec_bound(py, distances);

        let indices_2d = indices_arr.reshape([self.n_points, self.n_neighbors])?;
        let distances_2d = distances_arr.reshape([self.n_points, self.n_neighbors])?;

        Ok((indices_2d, distances_2d))
    }
}

impl PyNNDescent {
    fn build_index(
        data: &[f32],
        n_points: usize,
        dim: usize,
        metric: Metric,
        n_neighbors: usize,
        n_trees: Option<usize>,
        leaf_size: Option<usize>,
        max_candidates: Option<usize>,
        n_iters: Option<usize>,
        delta: f32,
        random_seed: u64,
        diversify_prob: f32,
        pruning_degree_multiplier: f32,
        verbose: bool,
    ) -> PyResult<Box<dyn AnyIndex>> {
        // Compute default n_trees matching PyNNDescent: max(3, min(12, round(2*log10(n))))
        let effective_n_trees = n_trees.unwrap_or_else(|| {
            let log_val = 2.0 * (n_points as f64).log10();
            (log_val.round() as usize).clamp(3, 12)
        });

        let mut builder = NNDescentBuilder::new(data, n_points, dim)
            .metric(metric)
            .n_neighbors(n_neighbors)
            .n_trees(effective_n_trees)
            .delta(delta)
            .random_seed(random_seed)
            .diversify_prob(diversify_prob)
            .pruning_degree_multiplier(pruning_degree_multiplier)
            .verbose(verbose);

        if let Some(ls) = leaf_size {
            builder = builder.leaf_size(ls);
        }
        if let Some(mc) = max_candidates {
            builder = builder.max_candidates(mc);
        }
        if let Some(ni) = n_iters {
            builder = builder.n_iters(ni);
        }

        // Dispatch to the correct concrete distance type for each metric.
        // Metrics with fast alternatives use proxy distances + correction.
        // Others use the direct distance function.
        macro_rules! build {
            ($dist:expr, $corr:expr) => {
                Box::new(builder.build_with_distance($dist, $corr)) as Box<dyn AnyIndex>
            };
        }

        let index_data: Box<dyn AnyIndex> = match metric {
            // Minkowski family
            Metric::Euclidean | Metric::L2 => build!(SquaredEuclidean, Some(|d: f32| d.sqrt())),
            Metric::SquaredEuclidean => build!(SquaredEuclidean, None),
            Metric::Manhattan => build!(Manhattan, None),
            Metric::Chebyshev => build!(Chebyshev, None),
            Metric::Canberra => build!(Canberra, None),
            Metric::BrayCurtis => build!(BrayCurtis, None),
            // Angular / similarity
            Metric::Cosine => build!(Cosine, None),
            Metric::Dot => build!(InnerProduct, None),
            Metric::InnerProduct => build!(InnerProduct, None),
            Metric::Correlation => build!(Correlation, None),
            Metric::TrueAngular => build!(TrueAngular, None),
            Metric::TSSS => build!(TSSS, None),
            // Binary / set
            Metric::Hamming => build!(Hamming, None),
            Metric::Jaccard => build!(Jaccard, None),
            Metric::Dice => build!(Dice, None),
            Metric::Matching => build!(Matching, None),
            Metric::Kulsinski => build!(Kulsinski, None),
            Metric::RogersTanimoto => build!(RogersTanimoto, None),
            Metric::RussellRao => build!(RussellRao, None),
            Metric::SokalMichener => build!(SokalMichener, None),
            Metric::SokalSneath => build!(SokalSneath, None),
            Metric::Yule => build!(Yule, None),
            // Distribution
            Metric::Hellinger => build!(Hellinger, None),
            Metric::JensenShannon => build!(JensenShannon, None),
            Metric::SymmetricKL => build!(SymmetricKL, None),
        };

        Ok(index_data)
    }
}

/// Get the version of the nndescent-rs library.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Check available SIMD support.
#[pyfunction]
fn simd_info() -> String {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let mut features = Vec::new();
        if is_x86_feature_detected!("avx512f") {
            features.push("AVX-512F");
        }
        if is_x86_feature_detected!("avx2") {
            features.push("AVX2");
        }
        if is_x86_feature_detected!("fma") {
            features.push("FMA");
        }
        if is_x86_feature_detected!("sse4.1") {
            features.push("SSE4.1");
        }

        if features.is_empty() {
            "Scalar (no SIMD)".to_string()
        } else {
            features.join(", ")
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        "Scalar (non-x86 platform)".to_string()
    }
}

/// Benchmark heap push operations.
/// 
/// This function simulates the heap push operations that occur during
/// candidate building in NN-Descent.
#[pyfunction]
fn benchmark_heap_push(
    n_vertices: usize,
    k: usize,
    max_candidates: usize,
    n_iters: usize,
    test_indices: PyReadonlyArray2<i32>,
    test_priorities: numpy::PyReadonlyArray4<f32>,
) -> PyResult<usize> {
    let indices_slice = test_indices.as_slice()?;
    let priorities_slice = test_priorities.as_slice()?;
    
    let mut total_pushes: usize = 0;
    
    // Allocate flat arrays for heaps
    let size = n_vertices * max_candidates;
    let mut heap_priorities = vec![f32::INFINITY; size];
    let mut heap_indices = vec![-1i32; size];
    
    for iter_idx in 0..n_iters {
        // Reset heaps
        for i in 0..size {
            heap_priorities[i] = f32::INFINITY;
            heap_indices[i] = -1;
        }
        
        // Simulate pushing edges (forward + reverse)
        for i in 0..n_vertices {
            for j in 0..k {
                let neighbor = indices_slice[i * k + j];
                if neighbor < 0 {
                    continue;
                }
                
                // Forward edge: push neighbor as candidate for vertex i
                let priority_idx = iter_idx * n_vertices * k * 2 + i * k * 2 + j * 2;
                let priority = priorities_slice[priority_idx];
                
                let offset_i = i * max_candidates;
                checked_heap_push_bench(
                    &mut heap_priorities[offset_i..offset_i + max_candidates],
                    &mut heap_indices[offset_i..offset_i + max_candidates],
                    priority,
                    neighbor,
                );
                total_pushes += 1;
                
                // Reverse edge: push i as candidate for neighbor
                let reverse_priority = priorities_slice[priority_idx + 1];
                let neighbor_idx = neighbor as usize;
                let offset_n = neighbor_idx * max_candidates;
                checked_heap_push_bench(
                    &mut heap_priorities[offset_n..offset_n + max_candidates],
                    &mut heap_indices[offset_n..offset_n + max_candidates],
                    reverse_priority,
                    i as i32,
                );
                total_pushes += 1;
            }
        }
    }
    
    Ok(total_pushes)
}

/// Push to a bounded priority max-heap with duplicate checking.
#[inline]
fn checked_heap_push_bench(
    priorities: &mut [f32],
    indices: &mut [i32],
    priority: f32,
    index: i32,
) {
    // Early exit if priority is worse than current max
    if priority >= priorities[0] {
        return;
    }

    // Check for duplicate (linear scan)
    let n = priorities.len();
    for i in 0..n {
        if indices[i] == index {
            return;
        }
    }

    // Insert by replacing root and sifting down
    priorities[0] = priority;
    indices[0] = index;
    
    // Sift down to maintain max-heap property
    let mut pos = 0;
    loop {
        let left = 2 * pos + 1;
        let right = 2 * pos + 2;
        let mut largest = pos;

        if left < n && priorities[left] > priorities[largest] {
            largest = left;
        }
        if right < n && priorities[right] > priorities[largest] {
            largest = right;
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

/// Benchmark candidate building from a graph.
///
/// Takes graph indices, distances, and flags, builds candidate sets.
/// Returns tuple of (new_candidates, old_candidates).
#[pyfunction]
fn benchmark_candidate_building<'py>(
    py: Python<'py>,
    graph_indices: PyReadonlyArray2<i32>,
    graph_distances: PyReadonlyArray2<f32>,
    graph_flags: PyReadonlyArray2<u8>,
    max_candidates: usize,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<i32>>)> {
    use nndescent_core::heap::NeighborHeap;
    use nndescent_core::nndescent::CandidateSets;
    use nndescent_core::rng::FastRng;
    
    let indices_view = graph_indices.as_array();
    let distances_view = graph_distances.as_array();
    let flags_view = graph_flags.as_array();
    
    let n_vertices = indices_view.shape()[0];
    let k = indices_view.shape()[1];
    
    // Create a NeighborHeap from the input data
    let mut heap = NeighborHeap::new(n_vertices, k);
    
    // Copy data into the heap
    for i in 0..n_vertices {
        for j in 0..k {
            heap.indices[i * k + j] = indices_view[[i, j]];
            heap.distances[i * k + j] = distances_view[[i, j]];
            heap.flags[i * k + j] = flags_view[[i, j]];
        }
    }
    
    let mut rng = FastRng::new(42);
    
    let candidates = CandidateSets::build_from_graph(&mut heap, max_candidates, &mut rng);
    
    // Convert to numpy arrays
    let new_indices = PyArray1::from_vec_bound(py, candidates.new_indices)
        .reshape([n_vertices, max_candidates])?;
    let old_indices = PyArray1::from_vec_bound(py, candidates.old_indices)
        .reshape([n_vertices, max_candidates])?;
    
    Ok((new_indices, old_indices))
}

/// Benchmark distance computations.
///
/// Computes squared Euclidean distance for given pairs.
#[pyfunction]
fn benchmark_distances<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f32>,
    pairs_i: numpy::PyReadonlyArray1<i32>,
    pairs_j: numpy::PyReadonlyArray1<i32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use nndescent_core::distance::SquaredEuclidean;
    
    let data_view = data.as_array();
    let _n_points = data_view.shape()[0];
    let dim = data_view.shape()[1];
    let data_slice = data.as_slice()?;
    
    let pairs_i_slice = pairs_i.as_slice()?;
    let pairs_j_slice = pairs_j.as_slice()?;
    let n_pairs = pairs_i_slice.len();
    
    let distance = SquaredEuclidean;
    
    let mut results = Vec::with_capacity(n_pairs);
    for k in 0..n_pairs {
        let i = pairs_i_slice[k] as usize;
        let j = pairs_j_slice[k] as usize;
        let vi = &data_slice[i * dim..(i + 1) * dim];
        let vj = &data_slice[j * dim..(j + 1) * dim];
        results.push(distance.distance(vi, vj));
    }
    
    Ok(PyArray1::from_vec_bound(py, results))
}

/// The pynndescent_rs Python module.
#[pymodule]
fn pynndescent_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNNDescent>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(simd_info, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_heap_push, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_candidate_building, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_distances, m)?)?;
    Ok(())
}
