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

use crate::prelude::*;

///////////
// Const //
///////////

/// Size of the query chunks
pub const QUERY_CHUNK_SIZE: usize = 8192;

/// Size of the DB chunks
pub const DB_CHUNK_SIZE: usize = 16_384;

/// Work group size in the cubecl cube (X)
pub const WORKGROUP_SIZE_X: u32 = 32;

/// Line size for vectorisations in this crate
pub const LINE_SIZE: usize = 4;

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

/// Pick the largest workgroup Y size that fits within the per-workgroup
/// shared memory budget for the IVF/exhaustive distance kernels.
///
/// Targets a 32 KiB device budget with roughly 2x headroom. Worst-case
/// footprint is the cosine cached kernel:
///   smem_bytes = wg_y * (4 * dim + 20)   (f32)
/// where `4 * dim` is `s_query`, the constant 20 is four u32 task-metadata
/// arrays plus `s_query_norms` (each contributes `wg_y` slots).
///
/// ### Params
///
/// * `dim_padded` - Padded embedding dimensionality (multiple of LINE_SIZE)
///
/// ### Returns
///
/// Chosen workgroup Y size, or `DimensionNotSupported` if `dim_padded`
/// exceeds the largest dim covered by the table.
pub fn pick_wg_y(dim_padded: usize) -> Result<u32, AnnSearchErrors> {
    let wg_y = match dim_padded {
        0..=128 => 32,
        129..=256 => 16,
        257..=512 => 8,
        513..=1024 => 4,
        1025..=2048 => 2,
        _ => {
            return Err(AnnSearchErrors::DimTooHighForSharedMemory {
                chosen_dim: dim_padded,
            })
        }
    };
    Ok(wg_y)
}
