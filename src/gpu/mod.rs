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
    (wg_y as usize).is_multiple_of(TILE_Q)
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

/// Shared-memory merge plan for the coalesced top-k reducers.
///
/// Produced by [`plan_topk_merge`] and handed to the kernel as comptime
/// arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopkMergePlan {
    /// Lanes whose private top-k lists are staged per merge round. Always a
    /// power of two dividing `WORKGROUP_SIZE_X`.
    pub group: usize,
    /// Whether every lane fits in a single round, enabling the unchunked fast
    /// path.
    pub single_round: bool,
    /// Number of `k`-element lists held in shared memory. Equals
    /// `WORKGROUP_SIZE_X` on the single-round path and `group + 1` otherwise,
    /// the extra list being the running accumulator.
    pub slots: usize,
}

/// Plan how the coalesced top-k reducers stage per-lane lists in shared memory.
///
/// ### Params
///
/// * `k` - Number of neighbours per query
/// * `elem_bytes` - Size of the float element type in bytes
/// * `max_shared_bytes` - Device shared-memory limit per workgroup, read from
///   `client.properties().hardware.max_shared_memory_size`
///
/// ### Returns
///
/// A [`TopkMergePlan`], or `KTooHighForSharedMemory` when not even two
/// `k`-element lists fit in the budget, i.e. `k > 2048` for `f32` against a
/// 32 KiB limit. Below that the merge is always representable.
pub fn plan_topk_merge(
    k: usize,
    elem_bytes: usize,
    max_shared_bytes: usize,
) -> Result<TopkMergePlan, AnnSearchErrors> {
    let lanes = WORKGROUP_SIZE_X as usize;
    // One list is `k` distances plus `k` u32 indices.
    let list_bytes = k * (elem_bytes + 4);

    if lanes * list_bytes <= max_shared_bytes {
        return Ok(TopkMergePlan {
            group: lanes,
            single_round: true,
            slots: lanes,
        });
    }

    // Chunked path: `group` staged lists plus the accumulator live at once.
    if 2 * list_bytes > max_shared_bytes {
        return Err(AnnSearchErrors::KTooHighForSharedMemory {
            k,
            available: max_shared_bytes,
        });
    }

    let mut group = 1usize;
    while 2 * group < lanes && (2 * group + 1) * list_bytes <= max_shared_bytes {
        group *= 2;
    }

    Ok(TopkMergePlan {
        group,
        single_round: false,
        slots: group + 1,
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
