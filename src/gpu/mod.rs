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

/////////////
// Helpers //
/////////////

/// Split a flat workgroup count into a 2D grid that respects the 65535 limit.
///
/// ### Params
///
/// * `total_cubes` - Total number of cubes
///
/// ### Returns
///
/// (x, y) in terms of size
pub fn grid_2d(total_cubes: u32) -> (u32, u32) {
    let x = total_cubes.min(65535);
    let y = total_cubes.div_ceil(x);
    (x, y)
}

/// Pad vectors to `dim_padded` by appending zeros to each row.
///
/// ### Params
///
/// * `flat` - Flattened row-major vector data of size `n * dim`
/// * `n` - Number of vectors
/// * `dim` - Original dimensionality
/// * `dim_padded` - Target dimensionality (must be >= `dim`)
///
/// ### Returns
///
/// Padded flat vector of size `n * dim_padded`
pub fn pad_vectors<T: num_traits::Float>(
    flat: &[T],
    n: usize,
    dim: usize,
    dim_padded: usize,
) -> Vec<T> {
    let mut padded = vec![T::zero(); n * dim_padded];
    for i in 0..n {
        let src = &flat[i * dim..(i + 1) * dim];
        let dst = &mut padded[i * dim_padded..i * dim_padded + dim];
        dst.copy_from_slice(src);
    }
    padded
}
