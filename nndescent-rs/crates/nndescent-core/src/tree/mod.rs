//! Random projection tree structures and construction.

mod flat_tree;
mod builder;

pub use flat_tree::FlatTree;
pub use builder::{build_rp_tree, build_rp_forest, rptree_leaf_array};
