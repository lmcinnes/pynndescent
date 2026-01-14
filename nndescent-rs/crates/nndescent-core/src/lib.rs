//! # nndescent-core
//!
//! High-performance Rust implementation of the NN-Descent algorithm for approximate
//! k-nearest neighbor graph construction and search.
//!
//! This crate provides:
//! - SIMD-optimized distance functions (Euclidean, Cosine, Inner Product)
//! - Efficient heap data structures for neighbor management
//! - Random projection tree construction
//! - NN-Descent algorithm with parallel execution
//! - Greedy graph search
//!
//! ## Example
//!
//! ```rust,ignore
//! use nndescent_core::{NNDescentBuilder, distance::Euclidean};
//!
//! // Build index from data (n points × d dimensions, flattened)
//! let data: Vec<f32> = /* your data */;
//! let n_points = 1000;
//! let dim = 128;
//!
//! let index = NNDescentBuilder::new(&data, n_points, dim)
//!     .n_neighbors(30)
//!     .n_trees(8)
//!     .build();
//!
//! // Query for k nearest neighbors
//! let query: Vec<f32> = /* your query */;
//! let (indices, distances) = index.query(&query, 10, 0.1);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod distance;
pub mod heap;
pub mod rng;
pub mod visited;
pub mod graph;
pub mod tree;
pub mod nndescent;
pub mod search;
pub mod index;

// Re-exports for convenience
pub use distance::{Distance, Euclidean, SquaredEuclidean, Cosine, InnerProduct, Metric};
pub use heap::NeighborHeap;
pub use index::{NNDescentIndex, NNDescentBuilder};
pub use rng::{TauRand, FastRng};
pub use visited::VisitedSet;

/// Constants matching PyNNDescent
pub mod constants {
    /// Minimum i32 value for RNG
    pub const INT32_MIN: i64 = i32::MIN as i64;
    /// Maximum i32 value for RNG
    pub const INT32_MAX: i64 = i32::MAX as i64;
    /// Small epsilon for float comparisons
    pub const FLOAT32_EPS: f32 = 1.0e-7;
    /// Block size for cache-efficient processing
    pub const BLOCK_SIZE: usize = 16384;
}
