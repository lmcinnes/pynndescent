//! Heap data structures for neighbor management.
//!
//! This module provides the `NeighborHeap` structure which maintains k-nearest
//! neighbors for each point. It uses a max-heap structure to efficiently track
//! the k smallest distances, with support for the "new/old" flag tracking used
//! in NN-Descent.

mod neighbor_heap;
mod candidate_heap;

pub use neighbor_heap::NeighborHeap;
pub use candidate_heap::{CandidateHeap, BoundedHeap};
