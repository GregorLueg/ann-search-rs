//! Flattened HNSW topology, built under an arbitrary distance.
//!
//! The hierarchy is only ever used to pick a good entry point; the beam search
//! runs entirely on layer 0. Splitting the two apart gives the base layer a
//! dense `n * degree` array with no per-node offset lookup, and keeps the
//! upper-layer lists out of the bytes the walk streams through.
//!
//! [`build::build_hierarchical_graph`] takes the distance as a closure over
//! node ids, so the topology is independent of how vectors are stored. That is
//! why this lives in `utils` rather than beside any one index: the uniformly
//! quantised graph and the RaBitQ graph both build on it, and they sit behind
//! different feature flags.

pub mod build;
pub mod flat_graph;
