//! Various CPU-based indices that run on the raw vectors. Leverage SIMD under
//! the hood.
//!
//! Provides:
//!
//! - Two tree-based versions: Annoy (memory-based) and BallTree.
//! - Two clustering-based versions: IVF and LSH (multi-probe) version
//! - Three graph-based versions: Vanama (memory-based version), HNSW and
//!   NNDescent
//!
//! And a flat exhaustive version

pub mod annoy;
pub mod ball_tree;
pub mod exhaustive;
pub mod hnsw;
pub mod ivf;
pub mod kd_forest;
pub mod lsh;
pub mod nndescent;
pub mod vamana;
