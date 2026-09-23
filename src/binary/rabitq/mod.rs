//! Implements the quantisation approach from RaBitQ, see:
//!
//! "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
//! for Approximate Nearest Neighbor Search" (Gao and Long, 2024).
//!
//! [`quantiser`] holds the one-bit encoder and its query side, [`rotator`] the
//! random rotation it encodes in, [`ex_bits`] the RaBitQ+ magnitude codes on
//! top of it, and [`fastscan`] the batched SIMD scoring path for the sign
//! bits. Under `quantised`, `codec` adds the
//! `GraphCodec` implementation that lets a quantised graph index run on RaBitQ
//! codes alone.

#[cfg(feature = "quantised")]
pub mod codec;
pub mod ex_bits;
pub mod fastscan;
pub mod quantiser;
pub mod rotator;

pub use quantiser::*;
