//! Errors in `ann-search-rs`

use thiserror::Error;

use crate::utils::dist::Dist;

#[cfg(feature = "gpu")]
use cubecl::server::ServerError;

/// All error variants that can occur across `ann-search-rs` operations.
///
/// Marked `#[non_exhaustive]`: variants appear and disappear with the optional
/// features, so a downstream exhaustive `match` would break every time a
/// feature flag moves. Match on the variants you care about and keep a `_` arm.
#[derive(Debug, Error)]
#[non_exhaustive]
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

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    // -- lsh errors --
    /// Requested hash width exceeds what a directly addressed bucket table can
    /// hold.
    #[error(
        "bits_per_hash ({bits_per_hash}) must be between 1 and {max}; the LSH bucket table is directly addressed."
    )]
    InvalidLshBits {
        /// Requested number of bits per hash code
        bits_per_hash: usize,
        /// Largest supported number of bits per hash code
        max: usize,
    },

    /// Dataset is too large for the `u32` bucket identifiers used by the LSH
    /// CSR layout.
    #[error("The LSH index stores bucket members as u32; {n} samples exceeds the limit of {max}.")]
    LshTooManySamples {
        /// Number of samples handed to the index
        n: usize,
        /// Largest supported number of samples
        max: usize,
    },

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
    /// Fewer training vectors than centroids to seed
    #[error(
        "Cannot train {n_centroids} centroids from {n_samples} vectors; \
         there must be at least as many vectors as centroids."
    )]
    TooFewSamplesForCentroids {
        /// Chosen number of centroids
        n_centroids: usize,
        /// Vectors available for training
        n_samples: usize,
    },

    // -- binary errors --
    /// Asymmetric queries are only supported with sign-based binarisation
    #[cfg(feature = "binary")]
    #[error("Only sign-based binarisation is supported for asymmetric queries")]
    AsymmetricQueryMisMatch,

    /// Residual encoding is only defined for sign-based binarisation
    #[cfg(feature = "binary")]
    #[error("Only sign-based binarisation supports residual encoding")]
    ResidualEncodingUnsupported,

    /// Residual codes cannot be compared across Voronoi cells
    #[cfg(feature = "binary")]
    #[error(
        "A sign-based IVF binary index stores codes relative to each cell's centroid, so \
         they are only comparable within a cell. Building a full kNN graph from it needs \
         the float vectors: build the index with build_with_vector_store()."
    )]
    ResidualCodesRequireVectorStore,

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

    /// Size mismatch error for locally stored files
    #[cfg(feature = "binary")]
    #[error("Size mismatch: expected {expected} bytes, got {actual} bytes.")]
    SizeMismatch {
        /// Expected size of the file
        expected: usize,
        /// Actual size of the file
        actual: usize,
    },

    /// A store file could not be opened
    #[cfg(feature = "binary")]
    #[error("Could not open the vector store file '{path}': {source}")]
    StoreFileUnavailable {
        /// The file that was being opened
        path: String,
        /// The underlying IO error
        #[source]
        source: std::io::Error,
    },

    /// The store found on disk does not have the shape the index recorded
    #[cfg(feature = "binary")]
    #[error(
        "The vector store holds {store_n} x {store_dim} vectors, but the index holds \
         {index_n} x {index_dim}."
    )]
    StoreShapeMismatch {
        /// Number of samples the index holds
        index_n: usize,
        /// Dimensionality the index holds
        index_dim: usize,
        /// Number of samples recorded for the store
        store_n: usize,
        /// Dimensionality recorded for the store
        store_dim: usize,
    },

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

    /// Error when LUT is being attempted with wrong bits
    #[cfg(feature = "binary")]
    #[error("Turbo quantisation: LUT scoring supports 2-bit and 4-bit only (chosen bit: {bit}")]
    TQLutError {
        /// The chosen bit
        bit: usize,
    },

    // -- gpu errors --
    /// Propagate errors from the CubeCL
    #[cfg(feature = "gpu")]
    #[error("Error from the cubecl runtime: {0}")]
    CubeClServerError(#[from] ServerError),

    /// Propagate device-limit errors from `cubecl-utils-rs`
    ///
    /// Covers cube counts, per-binding allocation sizes and the generic
    /// shared-memory budget check. The variants below are the ones specific to
    /// this crate's own kernels.
    #[cfg(feature = "gpu")]
    #[error("{0}")]
    CubeclUtils(#[from] cubecl_utils_rs::CubeclUtilsErrors),

    /// Error for a dimensionality whose per-workgroup staging cannot fit
    #[cfg(feature = "gpu")]
    #[error(
        "A padded dimensionality of {chosen_dim} needs {required} bytes of shared memory per \
         workgroup, but this device offers only {available}. Reduce the dimensionality."
    )]
    DimTooHighForSharedMemory {
        /// The chosen padded dimensionality
        chosen_dim: usize,
        /// Bytes the smallest viable staging plan would need
        required: usize,
        /// Shared memory the device reports, in bytes
        available: usize,
    },

    // -- serialisation errors --
    /// The file does not carry the `ann-search-rs` magic bytes
    #[cfg(feature = "serialise")]
    #[error("'{path}' is not an ann-search-rs index file.")]
    NotAnIndexFile {
        /// The file that was read
        path: String,
    },

    /// The file carries the magic bytes but ends inside the header
    #[cfg(feature = "serialise")]
    #[error("'{path}' is an ann-search-rs index file, but it is truncated.")]
    TruncatedIndexFile {
        /// The file that was read
        path: String,
    },

    /// The payload decoded, but the file carries on past its end
    #[cfg(feature = "serialise")]
    #[error("'{path}' has trailing bytes after the index payload.")]
    TrailingBytes {
        /// The file that was read
        path: String,
    },

    /// The file was written by an incompatible version of the format
    #[cfg(feature = "serialise")]
    #[error(
        "Index format version {found} is not supported; this build reads version {supported}."
    )]
    UnsupportedFormatVersion {
        /// Version tag found in the file
        found: u32,
        /// Version tag this build understands
        supported: u32,
    },

    /// The file holds a different index type than the one being loaded into
    #[cfg(feature = "serialise")]
    #[error("Index file holds a '{found}' index, but a '{expected}' index was requested.")]
    IndexKindMismatch {
        /// Index kind the caller asked for
        expected: &'static str,
        /// Index kind stored in the file
        found: String,
    },

    /// The file was written with a different float type
    #[cfg(feature = "serialise")]
    #[error("Index file was written with a {found}-byte float, but a {expected}-byte float was requested.")]
    FloatWidthMismatch {
        /// Size in bytes of the float the caller asked for
        expected: usize,
        /// Size in bytes of the float stored in the file
        found: usize,
    },

    /// Encoding the index failed
    #[cfg(feature = "serialise")]
    #[error("Failed to encode the index: {0}")]
    EncodeError(String),

    /// Decoding the index failed
    #[cfg(feature = "serialise")]
    #[error("Failed to decode the index: {0}")]
    DecodeError(String),
}
