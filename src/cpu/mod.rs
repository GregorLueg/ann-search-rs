//! Various CPU-based indices that run on the raw vectors. Leverage SIMD under
//! the hood.
//!
//! Provides:
//!
//! - Three tree-based versions: Annoy (memory-based), Kd Forest and BallTree.
//! - Two clustering-based versions: IVF and LSH (multi-probe) version
//! - Three graph-based versions: Vanama (memory-based version), HNSW and
//!   NNDescent
//!
//! And a flat exhaustive version + an accelerated exhaustive version via KmKnn.

pub mod annoy;
pub mod ball_tree;
pub mod exhaustive;
pub mod hnsw;
pub mod ivf;
pub mod kd_forest;
pub mod kmknn;
pub mod lsh;
pub mod nndescent;
pub mod vamana;
