//! Python bindings for nndescent-rs using PyO3.
//!
//! This crate provides Python-compatible classes that mirror the PyNNDescent API.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use nndescent_core::index::NNDescentBuilder;
use nndescent_core::distance::{Metric, SquaredEuclidean, Cosine, InnerProduct};

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
    index_data: IndexData,
}

/// Type-erased index storage
enum IndexData {
    Euclidean(nndescent_core::index::NNDescentIndex<SquaredEuclidean>),
    Cosine(nndescent_core::index::NNDescentIndex<Cosine>),
    InnerProduct(nndescent_core::index::NNDescentIndex<InnerProduct>),
}

#[pymethods]
impl PyNNDescent {
    #[new]
    #[pyo3(signature = (data, metric="euclidean", n_neighbors=30, n_trees=8, leaf_size=None, max_candidates=None, n_iters=None, delta=0.001, random_state=None, verbose=false))]
    fn new(
        data: PyReadonlyArray2<f32>,
        metric: &str,
        n_neighbors: usize,
        n_trees: usize,
        leaf_size: Option<usize>,
        max_candidates: Option<usize>,
        n_iters: Option<usize>,
        delta: f32,
        random_state: Option<u64>,
        verbose: bool,
    ) -> PyResult<Self> {
        let shape = data.shape();
        let n_points = shape[0];
        let dim = shape[1];

        // Copy data to owned vec
        let data_vec: Vec<f32> = data.as_slice()?.to_vec();

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

        let query_vec: Vec<f32> = query_data.as_slice()?.to_vec();

        let (indices, distances) = match &self.index_data {
            IndexData::Euclidean(idx) => idx.query(&query_vec, n_queries, k, epsilon),
            IndexData::Cosine(idx) => idx.query(&query_vec, n_queries, k, epsilon),
            IndexData::InnerProduct(idx) => idx.query(&query_vec, n_queries, k, epsilon),
        };

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
        let (indices, distances) = match &self.index_data {
            IndexData::Euclidean(idx) => (idx.neighbor_indices.clone(), idx.neighbor_distances.clone()),
            IndexData::Cosine(idx) => (idx.neighbor_indices.clone(), idx.neighbor_distances.clone()),
            IndexData::InnerProduct(idx) => (idx.neighbor_indices.clone(), idx.neighbor_distances.clone()),
        };

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
        n_trees: usize,
        leaf_size: Option<usize>,
        max_candidates: Option<usize>,
        n_iters: Option<usize>,
        delta: f32,
        random_seed: u64,
        verbose: bool,
    ) -> PyResult<IndexData> {
        let mut builder = NNDescentBuilder::new(data, n_points, dim)
            .metric(metric)
            .n_neighbors(n_neighbors)
            .n_trees(n_trees)
            .delta(delta)
            .random_seed(random_seed)
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

        let index_data = match metric {
            Metric::Euclidean | Metric::SquaredEuclidean | Metric::L2 => {
                IndexData::Euclidean(builder.build_euclidean())
            }
            Metric::Cosine => {
                IndexData::Cosine(builder.build_cosine())
            }
            Metric::InnerProduct | Metric::Dot => {
                IndexData::InnerProduct(builder.build_inner_product())
            }
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
    let new_indices = PyArray2::from_vec_bound(py, candidates.new_indices)
        .reshape([n_vertices, max_candidates])?;
    let old_indices = PyArray2::from_vec_bound(py, candidates.old_indices)
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
    pairs_i: PyReadonlyArray1<i32>,
    pairs_j: PyReadonlyArray1<i32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use nndescent_core::distance::SquaredEuclidean;
    
    let data_view = data.as_array();
    let n_points = data_view.shape()[0];
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
