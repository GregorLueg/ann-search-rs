//! This module contains all of the helpers, structures and methods related
//! to GPU-accelerated indices.
//!
//! The generic CubeCL machinery, tensors, device limits and dispatch geometry,
//! lives in `cubecl-utils-rs`. What stays here are the staging plans, which
//! model the shared-memory footprint of *this crate's* kernels and are not
//! reusable outside them.

pub mod cagra_gpu_search;
pub mod clustered_nndescent_gpu;
pub mod dist_gpu;
pub mod exhaustive_gpu;
pub mod forest_gpu;
pub mod ivf_gpu;
pub mod k_means_gpu;
pub mod nndescent_gpu;
pub mod topk_gpu;

use cubecl_utils_rs::prelude::*;

use crate::prelude::*;

///////////
// Const //
///////////

/// Size of the query chunks
///
/// An upper bound rather than a fixed size: [`plan_db_chunk`] shrinks the
/// per-chunk transient when the device cannot bind it.
pub const QUERY_CHUNK_SIZE: usize = 8192;

/// Size of the DB chunks
///
/// See [`QUERY_CHUNK_SIZE`]; this is the other half of the transient's shape.
pub const DB_CHUNK_SIZE: usize = 16_384;

/// Work group size in the cubecl cube (X)
///
/// A tuning constant, not a plane-size assumption. This crate uses no plane
/// primitives: every cross-thread reduction goes through `sync_cube()` and
/// shared memory, so a device whose plane is 64 (AMD) or variable (Intel)
/// computes the same answers. It matching Apple Silicon's plane size of 32 is
/// coincidence.
pub const WORKGROUP_SIZE_X: u32 = 32;

/// Wide workgroup used by the k-means kernels.
///
/// The search kernels want 32 because they run one cube per query or per node
/// and a wider cube would leave most lanes idle. The k-means assignment and
/// segmented-reduction kernels are the opposite shape: one thread per point
/// over the whole dataset, where the wider cube hides memory latency and the
/// segmented centroid update needs enough lanes to cover `dim` in one pass.
pub const WORKGROUP_128: u32 = 128;

/// DB vectors per thread in the register-tiled distance kernel.
pub const TILE_D: usize = 4;

/// Query vectors per thread in the register-tiled distance kernel.
///
/// `pick_wg_y` must return a multiple of this. At 4 that holds for every tier
/// up to dim=1024 (wg_y 32/16/8/4) but not for 1025..2048 (wg_y = 2), which
/// falls back to the untiled kernel via `tile_fits`. A device that forces
/// `pick_wg_y` below the table value falls back the same way.
pub const TILE_Q: usize = 4;

/////////////
// Helpers //
/////////////

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

/// Per-cube shared-memory footprint of the widest distance kernel.
///
/// The worst case is `compute_ivf_mega_cosine_cached`: `s_query` holds
/// `wg_y * dim_padded` elements, `s_query_norms` another `wg_y`, and four u32
/// task-metadata arrays contribute `wg_y` slots each.
///
/// ### Params
///
/// * `wg_y` - Query tile height
/// * `dim_padded` - Padded embedding dimensionality
/// * `elem_bytes` - Size of the float element type in bytes
///
/// ### Returns
///
/// Bytes of shared memory one cube allocates.
pub fn mega_smem_bytes(wg_y: u32, dim_padded: usize, elem_bytes: usize) -> usize {
    wg_y as usize * (dim_padded * elem_bytes + elem_bytes + 4 * 4)
}

/// Pick the largest workgroup Y size that fits this device.
///
/// The table is an upper bound tuned against a 32 KiB budget with roughly 2x
/// headroom for `f32`. From there the height halves until the staging fits the
/// device's shared memory, the cube fits its unit budget, and the height fits
/// the y extent of a cube. On the machine the table was tuned for this returns
/// the table value unchanged; it only ever shrinks.
///
/// Dropping below `TILE_Q` costs the register-tiled kernel, not correctness:
/// `tile_fits` goes false and callers take the untiled path.
///
/// ### Params
///
/// * `dim_padded` - Padded embedding dimensionality (multiple of `LINE_SIZE`)
/// * `elem_bytes` - Size of the float element type in bytes
/// * `limits` - Device limits from `GpuLimits::from_client`
///
/// ### Returns
///
/// Chosen workgroup Y size, or `DimTooHighForSharedMemory` when not even a
/// single query row fits.
pub fn pick_wg_y(
    dim_padded: usize,
    elem_bytes: usize,
    limits: &GpuLimits,
) -> Result<u32, AnnSearchErrors> {
    let mut wg_y: u32 = match dim_padded {
        0..=128 => 32,
        129..=256 => 16,
        257..=512 => 8,
        513..=1024 => 4,
        1025..=2048 => 2,
        _ => 1,
    };

    while wg_y >= 1 {
        let fits_smem = mega_smem_bytes(wg_y, dim_padded, elem_bytes) <= limits.max_shared_bytes;
        let fits_units = wg_y * WORKGROUP_SIZE_X <= limits.max_units_per_cube;
        let fits_dim = wg_y <= limits.max_cube_dim.1;

        if fits_smem && fits_units && fits_dim {
            return Ok(wg_y);
        }
        wg_y /= 2;
    }

    Err(AnnSearchErrors::DimTooHighForSharedMemory {
        chosen_dim: dim_padded,
        required: mega_smem_bytes(1, dim_padded, elem_bytes),
        available: limits.max_shared_bytes,
    })
}

/// Largest DB chunk whose distance transient still fits one binding.
///
/// The exhaustive path holds an `n_q * db_chunk` distance matrix, which at the
/// full chunk sizes is 512 MiB for `f32`. Apple Silicon binds 4 GiB and does not
/// notice; a device reporting the WebGPU default of 128 MiB would silently
/// return zeros.
///
/// ### Params
///
/// * `n_q` - Queries in the current chunk
/// * `elem_bytes` - Size of the float element type in bytes
/// * `limits` - Device limits from `GpuLimits::from_client`
///
/// ### Returns
///
/// DB rows per chunk, capped at [`DB_CHUNK_SIZE`] and at least 1.
pub fn plan_db_chunk(n_q: usize, elem_bytes: usize, limits: &GpuLimits) -> usize {
    let per_query = (n_q * elem_bytes).max(1) as u64;
    let affordable = (limits.max_binding_bytes / per_query).max(1);
    DB_CHUNK_SIZE.min(affordable as usize).max(1)
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
/// * `limits` - Device limits from `GpuLimits::from_client`
///
/// ### Returns
///
/// A [`LocalJoinStaging`] plan, or `DimTooHighForSharedMemory` when a single
/// candidate vector does not fit in the budget.
pub fn plan_local_join_staging(
    dim_padded: usize,
    max_cands: usize,
    elem_bytes: usize,
    limits: &GpuLimits,
) -> Result<LocalJoinStaging, AnnSearchErrors> {
    let max_shared_bytes = limits.max_shared_bytes;

    // shared_pids + shared_is_new (u32 each) + shared_norms (F), all indexed by
    // absolute candidate id and therefore never blocked.
    let metadata_bytes = max_cands * (2 * 4 + elem_bytes) + LOCAL_JOIN_FIXED_BYTES;
    let vec_bytes = dim_padded * elem_bytes;

    let avail = max_shared_bytes.saturating_sub(metadata_bytes);
    if avail < 2 * vec_bytes {
        return Err(AnnSearchErrors::DimTooHighForSharedMemory {
            chosen_dim: dim_padded,
            required: metadata_bytes + 2 * vec_bytes,
            available: max_shared_bytes,
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

/// Shared-memory staging plan for the CAGRA beam-search kernel.
///
/// Produced by [`plan_beam_search_staging`] and handed to the kernel as
/// comptime arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeamSearchStaging {
    /// Slots in the visited-node hash table. Always a power of two.
    pub hash_size: usize,
}

/// Smallest visited-node hash table worth running.
///
/// Below this the table thrashes badly enough that the search revisits more
/// than it explores.
const MIN_HASH_SIZE: usize = 128;

/// Plan the CAGRA beam search's shared-memory footprint.
///
/// Every term but the hash table is fixed by the graph degree and the beam
/// width, so the table is the only thing that can give. Halving it costs
/// revisits, not correctness: a collision makes the search re-expand a node it
/// has already seen, which wastes work but returns the same neighbours.
///
/// ### Params
///
/// * `dim_padded` - Padded embedding dimensionality (multiple of `LINE_SIZE`)
/// * `k_graph` - Graph degree, i.e. neighbours stored per node
/// * `beam_width` - Active candidates maintained during the search
/// * `expand_per_iter` - Neighbours expanded per beam iteration
/// * `preferred_hash` - Hash table size to start from, a power of two
/// * `elem_bytes` - Size of the float element type in bytes
/// * `limits` - Device limits from `GpuLimits::from_client`
///
/// ### Returns
///
/// A [`BeamSearchStaging`] plan, or `DimTooHighForSharedMemory` when even the
/// smallest table leaves the fixed terms over budget.
pub fn plan_beam_search_staging(
    dim_padded: usize,
    k_graph: usize,
    beam_width: usize,
    expand_per_iter: usize,
    preferred_hash: usize,
    elem_bytes: usize,
    limits: &GpuLimits,
) -> Result<BeamSearchStaging, AnnSearchErrors> {
    // sq_vec + s_cand_{dist,idx,expanded} + s_nbr_{idx,dist}
    // + s_active_flag + s_num_cands + s_query_norm
    let fixed = dim_padded * elem_bytes
        + beam_width * (elem_bytes + 2 * 4)
        + k_graph * expand_per_iter * (4 + elem_bytes)
        + 2 * 4
        + elem_bytes;

    let mut hash_size = preferred_hash.max(MIN_HASH_SIZE);
    while hash_size >= MIN_HASH_SIZE {
        if fixed + hash_size * 4 <= limits.max_shared_bytes {
            return Ok(BeamSearchStaging { hash_size });
        }
        hash_size /= 2;
    }

    Err(AnnSearchErrors::DimTooHighForSharedMemory {
        chosen_dim: dim_padded,
        required: fixed + MIN_HASH_SIZE * 4,
        available: limits.max_shared_bytes,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Apple Silicon via wgpu, the machine the tuning tables were built on.
    fn apple() -> GpuLimits {
        GpuLimits {
            max_shared_bytes: 32_768,
            max_cube_count: (65_535, 65_535, 65_535),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 1024),
            max_binding_bytes: 4_294_967_292,
            plane_size_min: 32,
            plane_size_max: 32,
        }
    }

    /// A device with half the shared memory and a quarter of the units.
    fn small() -> GpuLimits {
        GpuLimits {
            max_shared_bytes: 16_384,
            max_units_per_cube: 256,
            max_cube_dim: (256, 256, 256),
            max_binding_bytes: 128 * 1024 * 1024,
            ..apple()
        }
    }

    // -- pick_wg_y --

    #[test]
    fn test_pick_wg_y_apple_table_is_unchanged() {
        // The regression guard for "correctness only": on the machine the
        // table was tuned for, every tier must come back exactly as before.
        let l = apple();
        for (dim, expected) in [(128, 32), (256, 16), (512, 8), (1024, 4), (2048, 2)] {
            assert_eq!(pick_wg_y(dim, 4, &l).unwrap(), expected, "dim {dim}");
        }
    }

    #[test]
    fn test_pick_wg_y_errors_above_the_table() {
        // 4096 f32 rows are 16 KiB each; two of them plus metadata bust 32 KiB.
        assert!(pick_wg_y(8192, 4, &apple()).is_err());
    }

    #[test]
    fn test_pick_wg_y_shrinks_on_a_smaller_device() {
        let l = small();
        for dim in [128usize, 256, 512, 1024] {
            let apple_y = pick_wg_y(dim, 4, &apple()).unwrap();
            let small_y = pick_wg_y(dim, 4, &l).unwrap();
            assert!(small_y <= apple_y, "grew at dim {dim}");
            assert!(
                mega_smem_bytes(small_y, dim, 4) <= l.max_shared_bytes,
                "over budget at dim {dim}"
            );
            assert!(small_y * WORKGROUP_SIZE_X <= l.max_units_per_cube);
        }
    }

    #[test]
    fn test_pick_wg_y_respects_units_per_cube() {
        // 256 units per cube at a 32-wide x leaves room for 8 rows, not 32.
        let l = GpuLimits {
            max_units_per_cube: 256,
            ..apple()
        };
        assert!(pick_wg_y(128, 4, &l).unwrap() <= 8);
    }

    #[test]
    fn test_pick_wg_y_f64_shrinks_against_the_same_budget() {
        // Unreachable on wgpu (WGSL has no f64) but live on CUDA/HIP.
        let l = apple();
        for dim in [128usize, 256, 512] {
            let a = pick_wg_y(dim, 4, &l).unwrap();
            let b = pick_wg_y(dim, 8, &l).unwrap();
            assert!(b <= a, "f64 did not shrink at dim {dim}");
            assert!(mega_smem_bytes(b, dim, 8) <= l.max_shared_bytes);
        }
    }

    #[test]
    fn test_pick_wg_y_never_exceeds_the_budget_it_is_given() {
        for shared in [16_384usize, 32_768, 49_152, 65_536] {
            for elem in [4usize, 8] {
                for dim in [4usize, 64, 128, 256, 512, 1024, 2048] {
                    let l = GpuLimits {
                        max_shared_bytes: shared,
                        ..apple()
                    };
                    if let Ok(wg_y) = pick_wg_y(dim, elem, &l) {
                        assert!(wg_y >= 1);
                        assert!(
                            mega_smem_bytes(wg_y, dim, elem) <= shared,
                            "over budget: shared {shared}, elem {elem}, dim {dim}"
                        );
                    }
                }
            }
        }
    }

    // -- plan_db_chunk --

    #[test]
    fn test_plan_db_chunk_apple_is_the_full_chunk() {
        assert_eq!(plan_db_chunk(QUERY_CHUNK_SIZE, 4, &apple()), DB_CHUNK_SIZE);
    }

    #[test]
    fn test_plan_db_chunk_shrinks_to_the_binding_limit() {
        let l = small();
        let chunk = plan_db_chunk(QUERY_CHUNK_SIZE, 4, &l);
        assert!(chunk < DB_CHUNK_SIZE, "did not shrink");
        assert!(chunk >= 1);
        let bytes = (QUERY_CHUNK_SIZE * chunk * 4) as u64;
        assert!(bytes <= l.max_binding_bytes, "still over the binding limit");
    }

    // -- staging plans --

    #[test]
    fn test_plan_local_join_fits_every_budget() {
        for shared in [16_384usize, 32_768, 49_152, 65_536] {
            for elem in [4usize, 8] {
                let l = GpuLimits {
                    max_shared_bytes: shared,
                    ..apple()
                };
                for dim in [32usize, 128, 512] {
                    let plan = plan_local_join_staging(dim, 90, elem, &l).unwrap();
                    let meta = 90 * (2 * 4 + elem) + LOCAL_JOIN_FIXED_BYTES;
                    let used = meta + (plan.buf_a_len + plan.buf_b_len) * elem;
                    assert!(used <= shared, "over budget: {shared}/{elem}/{dim}");
                }
            }
        }
    }

    #[test]
    fn test_plan_beam_search_shrinks_the_hash_rather_than_failing() {
        let big = plan_beam_search_staging(128, 32, 16, 3, 2048, 4, &apple()).unwrap();
        let small_dev = plan_beam_search_staging(128, 32, 16, 3, 2048, 4, &small()).unwrap();
        assert_eq!(big.hash_size, 2048);
        assert!(small_dev.hash_size <= big.hash_size);
        assert!(small_dev.hash_size >= MIN_HASH_SIZE);
        assert!(small_dev.hash_size.is_power_of_two());
    }

    #[test]
    fn test_plan_beam_search_errors_when_the_fixed_terms_do_not_fit() {
        // A 4096-wide f32 row alone is 16 KiB; against 16 KiB total there is
        // nothing left for the beam, the neighbour slots or the hash table.
        assert!(plan_beam_search_staging(4096, 32, 16, 3, 2048, 4, &small()).is_err());
    }
}
