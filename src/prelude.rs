//! Re-exports of commonly used types and traits for convenient glob importing.
//!
//! ```rust
//! use ann_search_rs::prelude::*;
//! ```

pub use crate::errors::AnnSearchErrors;
pub use crate::utils::dist::*;
pub use crate::utils::heap_structs::*;
pub use crate::utils::input::AnnMatrix;
pub use crate::utils::k_means_utils::{KMeansInit, KMeansTrainingParams, LloydPath, SoarRule};
pub use crate::utils::matrix_to_flat;
pub use crate::utils::parallelism::StripedLocks;
pub use crate::utils::prefetch_read;
pub use crate::utils::traits::AnnSearchFloat;
pub use crate::utils::DimensionValidation;
pub use crate::utils::FlattenData;

#[cfg(feature = "gpu")]
pub use crate::gpu::cagra_gpu_search::CagraGpuSearchParams;
// The GPU builders take their own parameter structs. Without these the prelude
// hands a caller `KMeansTrainingParams` and silently not the GPU one that
// `build_ivf_index_gpu` and the batched kNN build actually want.
#[cfg(feature = "gpu")]
pub use crate::gpu::clustered_nndescent_gpu::ClusteredBuildParams;
#[cfg(feature = "gpu")]
pub use crate::gpu::k_means_gpu::KMeansGpuParams;
#[cfg(feature = "gpu")]
pub use cubecl_utils_rs::CubeclFloat;

#[cfg(feature = "serialise")]
pub use crate::serialise::IndexIo;
#[cfg(feature = "serialise")]
pub use crate::utils::staging::StagedFiles;

///////////
// Types //
///////////

/// Results type for large approximate nearest neighbour searches. The distances
/// are options here.
pub type KnnOptionResult<T> = Result<(Vec<Vec<usize>>, Option<Vec<Vec<T>>>), AnnSearchErrors>;

/// Results type for large approximate nearest neighbour searches.
pub type KnnResult<T> = Result<(Vec<Vec<usize>>, Vec<Vec<T>>), AnnSearchErrors>;
