//! Re-exports of commonly used types and traits for convenient glob importing.
//!
//! ```rust
//! use ann_search_rs::prelude::*;
//! ```

pub use crate::errors::AnnSearchErrors;
pub use crate::utils::dist::*;
pub use crate::utils::heap_structs::*;
pub use crate::utils::k_means_utils::{KMeansInit, KMeansTrainingParams, LloydPath};
pub use crate::utils::parallelism::StripedLocks;
pub use crate::utils::prefetch_read;
pub use crate::utils::traits::AnnSearchFloat;
pub use crate::utils::DimensionValidation;

#[cfg(feature = "gpu")]
pub use crate::gpu::cagra_gpu_search::CagraGpuSearchParams;
#[cfg(feature = "gpu")]
pub use crate::gpu::traits_gpu::AnnSearchGpuFloat;

///////////
// Types //
///////////

/// Results type for large approximate nearest neighbour searches. The distances
/// are options here.
pub type KnnOptionResult<T> = Result<(Vec<Vec<usize>>, Option<Vec<Vec<T>>>), AnnSearchErrors>;

/// Results type for large approximate nearest neighbour searches.
pub type KnnResult<T> = Result<(Vec<Vec<usize>>, Vec<Vec<T>>), AnnSearchErrors>;
