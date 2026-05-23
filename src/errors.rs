//! Errors in `ann-search-rs`

use thiserror::Error;

use crate::utils::dist::Dist;

#[cfg(feature = "gpu")]
use cubecl::server::ServerError;

/// All error variants that can occur across `ann-search-rs` operations.
#[derive(Debug, Error)]
pub enum AnnSearchErrors {
    // -- general errors --
    /// Dimension mismatch error between index and query.
    #[error(
        "The query dimensions ({query_dim}) are not equal to the index dimensions ({index_dim})."
    )]
    DimensionMismatch {
        /// Dimension the index expects
        index_dim: usize,
        /// Provided query dimension
        query_dim: usize,
    },

    /// Distance type not supported
    #[error("Distance metric '{0}' is not supported for this method.")]
    DistanceNotSupported(Dist),

    // -- quantisation errors --
    /// Dimension must be divisible by m
    #[error("Dimension ({dim}) must be divisible by m ({m}).")]
    #[cfg(feature = "quantised")]
    DimensionNotDivisibleByM {
        /// Input dimensionality
        dim: usize,
        /// Number of subspaces
        m: usize,
    },
    /// Dimension too small for product quantisation
    #[cfg(feature = "quantised")]
    #[error("Dimension ({dim}) is too small for product quantisation; minimum is 32.")]
    DimensionTooSmallForPQ {
        /// Input dimensionality
        dim: usize,
    },
    /// Number of centroids exceeds PQ limit
    #[cfg(feature = "quantised")]
    #[error("The number of centroids ({n_centroids}) for PQ is limited to 256.")]
    TooManyCentroidsForPQ {
        /// Chosen number of centroids
        n_centroids: usize,
    },

    // -- binary errors --
    /// Asymmetric queries are only supported with sign-based binarisation
    #[cfg(feature = "binary")]
    #[error("Only sign-based binarisation is supported for asymmetric queries")]
    AsymmetricQueryMisMatch,

    /// Error when n-bits is not a multiple of 8
    #[cfg(feature = "binary")]
    #[error("n_bits must be multiple of 8; chosen n_bits is {n_bits}.")]
    NBitsMustBe8Multiple {
        /// Chosen n_bits
        n_bits: usize,
    },

    /// Vector store is not available
    #[cfg(feature = "binary")]
    #[error("Vector store is not available. Use build_with_vector_store() to enable reranking.")]
    VectorStoreNotAvailable,

    /// IO error
    #[cfg(feature = "binary")]
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Turbo quant error for invalid number of bits
    #[cfg(feature = "binary")]
    #[error("Turbu quantisation only allows bits of 2, 3 or 4. Chosen n of bits {n_bits}")]
    TQInvalidBits {
        /// Number of chosen bits
        n_bits: usize,
    },

    /// Turbo quant error for invalid dimensionality
    #[cfg(feature = "binary")]
    #[error("Turbu quantisation needs a minimum dimensionality of 2. Data set dimensionality is {dims}.")]
    TQInvalidDim {
        /// Dimensionality of the data
        dims: usize,
    },

    /// Error when dimensionality is not a multiple of 8
    #[cfg(feature = "binary")]
    #[error("Turbo quantisation: dimensions must be multiple of 8; dimensionality of the data is {dims}.")]
    TQDimMustBe8Multiple {
        /// Dimensionality of the data
        dims: usize,
    },

    /// Error when the output buffer is not the length of bytes per vec
    #[cfg(feature = "binary")]
    #[error("Turbo quantisation: output buffer must be length bytes_per_vec ({bytes_per_vec}); has length ({len}).")]
    TQBufferUnequalBytesPerVec {
        /// Bytes per vec
        bytes_per_vec: usize,
        /// Output buffer length
        len: usize,
    },

    // -- gpu errors --
    /// Propagate errors from the ann-search-rs crate
    #[cfg(feature = "gpu")]
    #[error("Error from the cubecl runtime: {0}")]
    CubeClServerError(#[from] ServerError),
}
