//! Graph index over uniformly quantised vectors.
//!
//! Inspired by pyglass (<https://github.com/zilliztech/pyglass>), which pairs a
//! flattened HNSW graph with a swappable distance codec. The element taken from
//! it is the *uniform* scalar quantisation: one scale shared across every
//! dimension, which is what makes an integer distance preserve the exact
//! ordering of the float one, so the same kernel serves both graph construction
//! and query. Everything else here is this crate's own machinery.
//!
//! The quantiser and its integer kernels live one level up, in
//! [`crate::quantised::uniform_quant`] and [`crate::quantised::int_kernels`],
//! because the exhaustive and IVF indices use them too.

pub mod build;
pub mod codec;
pub mod flat_graph;
pub mod index;
