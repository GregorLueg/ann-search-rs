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

/// DB vectors per thread in the register-tiled distance kernel.
///
/// The untiled kernel issues one vectorised global load plus `LINE_SIZE`
/// shared loads per `LINE_SIZE` FMAs: 1.25 memory operations per FMA. It
/// measures a flat 520-580 GFLOP/s across dim=32..512 on an M1 Max, about
/// 5.5% of f32 peak, while using only ~9% of memory bandwidth, so it is bound
/// by memory-op issue count rather than by compute or bandwidth.
///
/// A `TILE_D x TILE_Q` tile drops the ratio to
/// `(TILE_D + TILE_Q * LINE_SIZE) / (TILE_D * TILE_Q * LINE_SIZE)`, which is
/// 0.3125 at 4x4. Measured against the untiled kernel, bit-exact: 2.21x at
/// dim=32, 1.94x at 64, 1.79x at 128, 1.43x at 512. The gain tapers at high
/// dim because `pick_wg_y` shrinks the query tile, leaving fewer threads per
/// cube.
pub const TILE_D: usize = 4;

/// Query vectors per thread in the register-tiled distance kernel.
///
/// `pick_wg_y` must return a multiple of this. At 4 that holds for every tier
/// up to dim=1024 (wg_y 32/16/8/4) but not for 1025..2048 (wg_y = 2), which
/// falls back to the untiled kernel via `tile_fits`.
pub const TILE_Q: usize = 4;

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

/// Whether the register-tiled distance kernel can be used for a given query
/// tile height.
///
/// The tiled kernel assigns `TILE_Q` consecutive shared-memory query rows to
/// each thread, so the tile height must divide evenly. Callers fall back to the
/// untiled kernel when this returns false.
///
/// ### Params
///
/// * `wg_y` - Query tile height as returned by `pick_wg_y`
///
/// ### Returns
///
/// True if `wg_y` is a multiple of `TILE_Q`
pub fn tile_fits(wg_y: u32) -> bool {
    wg_y as usize % TILE_Q == 0
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
/// Chosen workgroup Y size, or `DimTooHighForSharedMemory` if `dim_padded`
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
