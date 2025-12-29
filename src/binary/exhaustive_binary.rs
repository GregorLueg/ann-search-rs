use crate::binary::binariser::*;
use crate::binary::dist_binary::*;

///////////////////////////
// ExhaustiveIndexBinary //
///////////////////////////

/// Exhaustive (brute-force) binary nearest neighbour index
///
/// ### Fields
///
/// * `vectors_flat` - Binarised vector data for distance calculations via
///   Hamming. Flattened for better cache locality.
/// * `dim` - Embedding dimensions
/// * `n` - Number of samples
pub struct ExhaustiveIndexBinary<T> {
    // shared ones
    pub vectors_flat: Vec<u8>,
    pub dim: usize,
    pub n: usize,
    binariser: Binariser<T>,
}
