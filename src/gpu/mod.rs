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

/// Shared-memory staging plan for the NNDescent local-join kernel.
///
/// Produced by [`plan_local_join_staging`] and handed to the kernel as comptime
/// arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalJoinStaging {
    /// Candidates whose vectors are staged per buffer.
    pub block: usize,
    /// Whether every candidate fits in a single buffer, enabling the unblocked
    /// fast path.
    pub single_block: bool,
    /// Scalar length of the first vector staging buffer.
    pub buf_a_len: usize,
    /// Scalar length of the second vector staging buffer. `1` (never touched)
    /// on the single-block path.
    pub buf_b_len: usize,
}

/// Per-cube shared-memory overhead of the local-join kernel that is *not* the
/// staged vectors.
///
/// `shared_compact` holds two u32s, `shared_rev_count` one.
const LOCAL_JOIN_FIXED_BYTES: usize = 3 * 4;

/// Plan how the NNDescent local join stages candidate vectors in shared memory.
///
/// `local_join_shared` used to stage all `2 * build_k` candidate vectors at
/// once, which costs `2 * build_k * ((dim_padded + 1) * size_of::<T>() + 8)`
/// bytes. Nothing checked that against the device limit (32768 bytes on an
/// M1 Max), and a `launch_unchecked` dispatch that busts it does no work,
/// reports no error and leaves the graph at its forest initialisation. Measured
/// on that machine before this helper existed:
///
/// | k  | dim | build_k | bytes | iteration-1 updates |
/// |----|-----|---------|-------|---------------------|
/// | 15 | 160 | 22      | 28688 | 142274              |
/// | 15 | 192 | 22      | 34320 | 0 (silent no-op)    |
/// | 30 | 64  | 45      | 24120 | 383725              |
/// | 30 | 128 | 45      | 47160 | 0 (silent no-op)    |
///
/// Since `k` defaults to 30, the default configuration was silently broken from
/// dim 128 upwards. The fix keeps the candidate metadata staged in full (it is
/// only `max_cands * (8 + size_of::<T>())` bytes) and blocks the vectors into
/// two buffers of `block` candidates each, walked as block pairs `(bi, bj)`
/// with `bi <= bj`. When all candidates fit in one buffer the kernel takes the
/// original unblocked path and the second buffer is allocated with length 1.
///
/// ### Params
///
/// * `dim_padded` - Padded embedding dimensionality (multiple of `LINE_SIZE`)
/// * `max_cands` - Maximum candidates per node, i.e. `2 * build_k`
/// * `elem_bytes` - Size of the float element type in bytes
/// * `max_shared_bytes` - Device shared-memory limit per workgroup, read from
///   `client.properties().hardware.max_shared_memory_size`
///
/// ### Returns
///
/// A [`LocalJoinStaging`] plan, or `DimTooHighForSharedMemory` when a single
/// candidate vector does not fit in the budget.
pub fn plan_local_join_staging(
    dim_padded: usize,
    max_cands: usize,
    elem_bytes: usize,
    max_shared_bytes: usize,
) -> Result<LocalJoinStaging, AnnSearchErrors> {
    // shared_pids + shared_is_new (u32 each) + shared_norms (F), all indexed by
    // absolute candidate id and therefore never blocked.
    let metadata_bytes = max_cands * (2 * 4 + elem_bytes) + LOCAL_JOIN_FIXED_BYTES;
    let vec_bytes = dim_padded * elem_bytes;

    let avail = max_shared_bytes.saturating_sub(metadata_bytes);
    if avail < 2 * vec_bytes {
        return Err(AnnSearchErrors::DimTooHighForSharedMemory {
            chosen_dim: dim_padded,
        });
    }

    if max_cands * vec_bytes <= avail {
        return Ok(LocalJoinStaging {
            block: max_cands,
            single_block: true,
            buf_a_len: max_cands * dim_padded,
            buf_b_len: 1,
        });
    }

    // Two buffers live simultaneously on the blocked path.
    let block = (avail / (2 * vec_bytes)).min(max_cands).max(1);
    Ok(LocalJoinStaging {
        block,
        single_block: false,
        buf_a_len: block * dim_padded,
        buf_b_len: block * dim_padded,
    })
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
