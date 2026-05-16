//! Errors in `ann-search-rs`

use thiserror::Error;

use crate::utils::dist::Dist;

/// All error variants that can occur across `ann-search-rs` operations.
#[derive(Debug, Error)]
pub enum AnnSearchErrors {
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

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Distance type not supported
    #[error("Distance metric '{0}' is not supported for this index.")]
    DistanceNotSupported(Dist),

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
}
