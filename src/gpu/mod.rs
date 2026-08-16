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
/// The k-means assignment and segmented-reduction kernels are one thread per
/// point over the whole dataset, where a wide cube hides memory latency and the
/// segmented centroid update needs enough lanes to cover `dim` in one pass.
///
/// Most search kernels stay at [`WORKGROUP_SIZE_X`] because they run one cube
/// per query and the work per cube does not fill more. The local join is the
/// exception and sizes its own cube through `pick_local_join_cube`: idle
/// lanes turned out to be free there, and widening it was worth roughly 2x.
/// Treat "search kernels want 32" as a default, not a rule.
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
    /// Length of the first vector staging buffer in `Vector<F, N>` elements,
    /// i.e. `block * row_lines`. Lines, not scalars: the buffers are
    /// vectorised, and a scalar count here would over-allocate by `LINE_SIZE`
    /// and silently bust the shared-memory budget.
    pub buf_a_lines: usize,
    /// Length of the second vector staging buffer in `Vector<F, N>` elements.
    /// `1` (never touched) on the single-block path.
    pub buf_b_lines: usize,
    /// Stride between staged candidate rows, in `Vector<F, N>` elements. This
    /// is `dim_padded / LINE_SIZE` plus whatever `resolve_row_pad` adds to
    /// offset consecutive rows across shared-memory banks.
    pub row_lines: usize,
    /// Length of the candidate-norm buffer: `max_cands` under cosine, `1`
    /// otherwise. Euclidean never reads it, so allocating it at full width
    /// would spend shared memory the vector staging wants.
    pub norm_buf_len: usize,
    /// Lines the pair-distance loop processes per unrolled step, and therefore
    /// how many independent accumulator chains it keeps in flight.
    pub line_unroll: usize,
    /// Cube extent along the candidate-`j` axis.
    pub cube_x: u32,
    /// Cube extent along the candidate-`i` axis.
    pub cube_y: u32,
}

/// Cube shape for the local join.
///
/// The kernel is latency bound and its shared footprint sits near the whole
/// device budget, so exactly one cube is resident per core and the cube width
/// *is* the resident thread count. Shared memory is per cube, so widening costs
/// no footprint.
///
/// The shape is deliberately **not** a function of `block`. The obvious rule,
/// never dispatch more threads than the `block * block` pair slab has slots,
/// was measured and is wrong: 128 threads wins at dim 1024 where the slab holds
/// nine slots and 119 of 128 lanes retire immediately. Idle lanes are free; the
/// SIMD groups they sit in are what hides memory latency.
///
/// ### Params
///
/// * `limits` - Device limits from `GpuLimits::from_client`
///
/// ### Returns
///
/// Cube extent as `(x, y)`, candidate `j` on x and candidate `i` on y. Normally
/// `x >= y`, so one plane spans few distinct rows of the `i` operand; a device
/// with a small `max_cube_dim.0` can invert that when x is clamped, which costs
/// locality but stays legal and correct.
fn pick_local_join_cube(limits: &GpuLimits) -> (u32, u32) {
    // Round the device's unit budget down to a power of two so the squarish
    // split below stays exact.
    let mut unit_cap = limits.max_units_per_cube.max(1);
    if !unit_cap.is_power_of_two() {
        unit_cap = unit_cap.next_power_of_two() / 2;
    }
    let threads = (LOCAL_JOIN_THREADS as u32)
        .min(unit_cap)
        .max(WORKGROUP_SIZE_X.min(unit_cap));

    let mut x = 1u32;
    while x * x < threads {
        x *= 2;
    }
    let x = x.min(limits.max_cube_dim.0).max(1);
    let y = (threads / x).min(limits.max_cube_dim.1).max(1);
    (x, y)
}

/// Threads per local-join cube.
///
/// Swept on an M1 Max at n=25k, `build_k=45`, min-of-15, in milliseconds:
///
/// | dim | 8x4 (32) | 8x8 (64) | 16x8 (128) | 16x16 (256) |
/// |---|---|---|---|---|
/// | 128 | 55.9 | 35.7 | **28.4** | 38.9 |
/// | 256 | 98.0 | 59.4 | **49.1** | 61.8 |
/// | 512 | 224.2 | 127.7 | **117.3** | 175.3 |
/// | 1024 | 862.7 | 690.1 | **627.2** | 1100 |
///
/// 128 wins at every width and 256 regresses, which is the register-pressure
/// edge again: each thread carries [`LOCAL_JOIN_LINE_UNROLL`] vector
/// accumulators and that does not shrink as the cube grows.
const LOCAL_JOIN_THREADS: usize = 128;

/// Unroll depth for the local join's pair-distance loop.
///
/// The kernel is latency bound, so this is a real lever rather than a tidiness
/// knob: it sets how many loads are in flight per thread. Both extremes were
/// measured on an M1 Max at n=25k and both lose. Unrolling the whole row emits
/// `dim_padded` statements at each of three call sites and costs ~20% at dim
/// 1024; not unrolling at all costs ~20% at dim 64.
///
/// Sweep this rather than inheriting it. The knee has moved between kernels in
/// this crate before.
///
/// ### Params
///
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row
///
/// ### Returns
///
/// Lines per unrolled step, at least 1 and never more than the row length.
fn resolve_line_unroll(dim_lines: usize) -> usize {
    LOCAL_JOIN_LINE_UNROLL.min(dim_lines).max(1)
}

/// Padding for the local join's shared row stride, in `Vector<F, N>` elements.
///
/// Threads holding different candidate rows read the same line at the same
/// time, so an unpadded stride puts them all in the same shared-memory bank
/// group. One line of padding walks consecutive rows across banks instead.
///
/// It only pays while there are enough rows staged to conflict, which is why
/// this is a function of the row length rather than a constant. Measured on an
/// M1 Max at n=25k, min-of-15, padded against unpadded:
///
/// | dim | block | build_k 45 | build_k 64 |
/// |---|---|---|---|
/// | 128 | 29 | -4.7% | -0.6% |
/// | 256 | 15 | -7.1% | **-15.3%** |
/// | 512 | 7 | +1.8% | +3.0% |
/// | 1024 | 3 | **+29.4%** | **+33.0%** |
///
/// Past dim 256 the shared budget holds so few candidates that there is little
/// conflict left to remove, and the odd stride costs more than it saves.
///
/// ### Params
///
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row
///
/// ### Returns
///
/// Lines of padding to add to the row stride.
fn resolve_row_pad(dim_lines: usize) -> usize {
    if dim_lines <= LOCAL_JOIN_PAD_MAX_LINES {
        1
    } else {
        0
    }
}

/// Longest row, in lines, that still benefits from `resolve_row_pad`.
const LOCAL_JOIN_PAD_MAX_LINES: usize = 64;

/// Default unroll depth handed to `resolve_line_unroll`.
///
/// Swept on an M1 Max at n=25k, `build_k=45`, min-of-15, in milliseconds:
///
/// | dim | 1 | 2 | 4 | 8 | 16 |
/// |---|---|---|---|---|---|
/// | 64 | 68.7 | 58.5 | 51.3 | **47.9** | 51.3 |
/// | 128 | 114.2 | 88.6 | 81.6 | **74.0** | 141.3 |
/// | 1024 | 3652 | 2398 | 1956 | **1575** | 3661 |
///
/// 8 wins at every width. 16 falls off a cliff at dim 128 and above, which is
/// the register-pressure edge: 16 `Vector<F, N>` accumulators plus their
/// in-flight operands spill, and a spilled register array is global memory.
const LOCAL_JOIN_LINE_UNROLL: usize = 8;

/// Per-cube shared-memory overhead of the local-join kernel that is *not* the
/// staged vectors.
///
/// `shared_compact` holds two u32s, `shared_rev_count` one.
const LOCAL_JOIN_FIXED_BYTES: usize = 3 * 4;

/// Per-cube shared-memory footprint of the local-join kernel.
///
/// One formula, so the kernel, the planner, the tests and the benches cannot
/// drift apart. A footprint over the device budget makes the dispatch do
/// nothing and report no error, so this is the number every one of them has to
/// agree on.
///
/// ### Params
///
/// * `plan` - Staging plan under test
/// * `max_cands` - Maximum candidates per node, i.e. `2 * build_k`
/// * `elem_bytes` - Size of the float element type in bytes
///
/// ### Returns
///
/// Total bytes of shared memory one cube allocates.
pub fn local_join_smem_bytes(
    plan: &LocalJoinStaging,
    max_cands: usize,
    elem_bytes: usize,
) -> usize {
    local_join_meta_bytes(max_cands, plan.norm_buf_len, elem_bytes)
        + (plan.buf_a_lines + plan.buf_b_lines) * LINE_SIZE * elem_bytes
}

/// Shared-memory the local join spends on per-candidate scalars.
///
/// `shared_pids` and `shared_is_new` hold a u32 each, `shared_thresh` a float,
/// and `shared_norms` a float per candidate under cosine only. All four are
/// indexed by absolute candidate id and are therefore never blocked.
///
/// ### Params
///
/// * `max_cands` - Maximum candidates per node, i.e. `2 * build_k`
/// * `norm_buf_len` - Length of the candidate-norm buffer
/// * `elem_bytes` - Size of the float element type in bytes
///
/// ### Returns
///
/// Bytes of shared memory the non-vector staging occupies.
fn local_join_meta_bytes(max_cands: usize, norm_buf_len: usize, elem_bytes: usize) -> usize {
    max_cands * (2 * 4 + elem_bytes) + norm_buf_len * elem_bytes + LOCAL_JOIN_FIXED_BYTES
}

/// Plan how the NNDescent local join stages candidate vectors in shared memory.
///
/// Also picks the launch geometry and the two tuning knobs that go with it, so
/// every device-dependent decision the kernel makes lives in one pure function
/// of [`GpuLimits`]: the cube shape (`pick_local_join_cube`), the pair-loop
/// unroll depth (`resolve_line_unroll`) and the shared row padding
/// (`resolve_row_pad`).
///
/// ### Params
///
/// * `dim_padded` - Padded embedding dimensionality (multiple of `LINE_SIZE`)
/// * `max_cands` - Maximum candidates per node, i.e. `2 * build_k`
/// * `elem_bytes` - Size of the float element type in bytes
/// * `use_cosine` - Whether the kernel takes its cosine arm, which is the only
///   one that stages candidate norms
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
    use_cosine: bool,
    limits: &GpuLimits,
) -> Result<LocalJoinStaging, AnnSearchErrors> {
    let max_shared_bytes = limits.max_shared_bytes;

    let norm_buf_len = if use_cosine { max_cands } else { 1 };
    let metadata_bytes = local_join_meta_bytes(max_cands, norm_buf_len, elem_bytes);
    let dim_lines = dim_padded / LINE_SIZE;
    let row_lines = dim_lines + resolve_row_pad(dim_lines);
    let vec_bytes = row_lines * LINE_SIZE * elem_bytes;

    let avail = max_shared_bytes.saturating_sub(metadata_bytes);
    if avail < 2 * vec_bytes {
        return Err(AnnSearchErrors::DimTooHighForSharedMemory {
            chosen_dim: dim_padded,
            required: metadata_bytes + 2 * vec_bytes,
            available: max_shared_bytes,
        });
    }

    // The dummy `buf_b_lines: 1` below is still an allocation, so it has to be
    // admitted here too. It is one line rather than one scalar now that the
    // buffers are vectorised, and an unbudgeted allocation is a dispatch that
    // silently does nothing.
    let dummy_b_bytes = LINE_SIZE * elem_bytes;
    if max_cands * vec_bytes + dummy_b_bytes <= avail {
        let (cube_x, cube_y) = pick_local_join_cube(limits);
        return Ok(LocalJoinStaging {
            block: max_cands,
            single_block: true,
            buf_a_lines: max_cands * row_lines,
            buf_b_lines: 1,
            row_lines,
            norm_buf_len,
            line_unroll: resolve_line_unroll(dim_lines),
            cube_x,
            cube_y,
        });
    }

    // Two buffers live simultaneously on the blocked path.
    let block = (avail / (2 * vec_bytes)).min(max_cands).max(1);
    let (cube_x, cube_y) = pick_local_join_cube(limits);
    Ok(LocalJoinStaging {
        block,
        single_block: false,
        buf_a_lines: block * row_lines,
        buf_b_lines: block * row_lines,
        row_lines,
        norm_buf_len,
        line_unroll: resolve_line_unroll(dim_lines),
        cube_x,
        cube_y,
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
                for dim in [32usize, 64, 128, 256, 512, 1024, 2048] {
                    for max_cands in [60usize, 90, 128] {
                        for cosine in [false, true] {
                            let Ok(plan) =
                                plan_local_join_staging(dim, max_cands, elem, cosine, &l)
                            else {
                                continue;
                            };
                            let tag = format!("{shared}/{elem}/{dim}/{max_cands}/{cosine}");

                            assert!(
                                local_join_smem_bytes(&plan, max_cands, elem) <= shared,
                                "over budget: {tag}"
                            );

                            // Lengths are in lines, not scalars. Mixing the two
                            // up over-allocates by LINE_SIZE, which busts the
                            // budget and makes the dispatch silently do
                            // nothing, so assert the unit and not just the
                            // total.
                            assert_eq!(plan.buf_a_lines % plan.row_lines, 0, "ragged buf_a: {tag}");
                            assert_eq!(
                                plan.buf_a_lines / plan.row_lines,
                                plan.block,
                                "buf_a rows: {tag}"
                            );
                            assert!(
                                plan.row_lines >= dim / LINE_SIZE,
                                "row stride shorter than the row: {tag}"
                            );
                            assert!(plan.block >= 1 && plan.block <= max_cands, "block: {tag}");

                            assert_eq!(
                                plan.single_block,
                                plan.block == max_cands,
                                "flag disagrees with block: {tag}"
                            );
                            assert_eq!(
                                plan.buf_b_lines == 1,
                                plan.single_block,
                                "buf_b must be the dummy iff single-block: {tag}"
                            );

                            assert_eq!(
                                plan.norm_buf_len,
                                if cosine { max_cands } else { 1 },
                                "norm buffer: {tag}"
                            );

                            // A cube over a device limit is a dispatch that
                            // does nothing and reports no error.
                            let threads = plan.cube_x * plan.cube_y;
                            assert!(threads <= l.max_units_per_cube, "cube units: {tag}");
                            assert!(plan.cube_x <= l.max_cube_dim.0, "cube x: {tag}");
                            assert!(plan.cube_y <= l.max_cube_dim.1, "cube y: {tag}");
                            // x-major except when x itself is clamped by a
                            // narrow device, which is the one legal inversion.
                            assert!(
                                plan.cube_x >= plan.cube_y || plan.cube_x == l.max_cube_dim.0,
                                "cube not x-major: {tag}"
                            );
                            assert!(threads.is_power_of_two(), "cube not a power of two: {tag}");

                            assert!(plan.line_unroll >= 1, "unroll: {tag}");
                            assert!(
                                plan.line_unroll <= dim / LINE_SIZE,
                                "unroll past the row: {tag}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_plan_local_join_shrinks_monotonically() {
        // Anything that makes a candidate more expensive must never buy a
        // bigger block. These are the directions a retune could silently
        // invert.
        for dim in [128usize, 256, 512, 1024] {
            let big = plan_local_join_staging(dim, 90, 4, false, &apple()).unwrap();

            let half = GpuLimits {
                max_shared_bytes: apple().max_shared_bytes / 2,
                ..apple()
            };
            let smaller = plan_local_join_staging(dim, 90, 4, false, &half).unwrap();
            assert!(
                smaller.block <= big.block,
                "half budget grew block at {dim}"
            );

            let f64_plan = plan_local_join_staging(dim, 90, 8, false, &apple()).unwrap();
            assert!(f64_plan.block <= big.block, "f64 grew block at {dim}");

            let cos = plan_local_join_staging(dim, 90, 4, true, &apple()).unwrap();
            assert!(cos.block <= big.block, "cosine grew block at {dim}");
        }
    }

    #[test]
    fn test_plan_local_join_apple_table_is_unchanged() {
        // The block sizes and launch geometry the kernel's measured speedups
        // were scored against. Changing them is a deliberate retune, not a
        // drive-by: the cube width and unroll depth were each swept, and both
        // regress by ~2x one step past the chosen value.
        for (dim, block, single) in [
            (64usize, 90usize, true),
            (128, 29, false),
            (256, 15, false),
            (512, 7, false),
            (1024, 3, false),
        ] {
            let plan = plan_local_join_staging(dim, 90, 4, false, &apple()).unwrap();
            assert_eq!(plan.block, block, "block moved at dim {dim}");
            assert_eq!(plan.single_block, single, "path moved at dim {dim}");
            assert_eq!(
                (plan.cube_x, plan.cube_y),
                (16, 8),
                "cube moved at dim {dim}"
            );
            assert_eq!(plan.line_unroll, 8, "unroll moved at dim {dim}");
        }
    }

    #[test]
    fn test_plan_local_join_cube_shrinks_on_a_narrow_device() {
        // A device that cannot host 128 units per cube must get a legal cube,
        // not a dispatch that silently does nothing.
        let narrow = GpuLimits {
            max_units_per_cube: 64,
            max_cube_dim: (64, 64, 64),
            ..apple()
        };
        let plan = plan_local_join_staging(128, 90, 4, false, &narrow).unwrap();
        assert!(plan.cube_x * plan.cube_y <= 64);
        assert!(plan.cube_x >= plan.cube_y);
    }

    #[test]
    fn test_plan_local_join_errors_only_past_the_boundary() {
        // Two f32 vectors plus the metadata have to fit. At 32 KiB and 90
        // candidates, 2048 is the last width that does and 4096 is not.
        assert!(plan_local_join_staging(2048, 90, 4, false, &apple()).is_ok());
        assert!(plan_local_join_staging(4096, 90, 4, false, &apple()).is_err());
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
