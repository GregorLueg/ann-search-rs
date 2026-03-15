//! This module contains all of the helpers, structures and methods related
//! to GPU-accelerated indices.

pub mod cagra_gpu_search;
pub mod dist_gpu;
pub mod exhaustive_gpu;
pub mod forest_gpu;
pub mod ivf_gpu;
pub mod nndescent_gpu;
pub mod tensor;
pub mod traits_gpu;

///////////
// Const //
///////////

/// Size of the query chunks
pub const QUERY_CHUNK_SIZE: usize = 8192;

/// Size of the DB chunks
pub const DB_CHUNK_SIZE: usize = 16_384;

/// Work group size in the cubecl cube (X)
pub const WORKGROUP_SIZE_X: u32 = 32;

/// Work group size in the cubecl cube (Y)
pub const WORKGROUP_SIZE_Y: u32 = 32;

/// Line size for vectorisations in this crate
pub const LINE_SIZE: u32 = 4;
