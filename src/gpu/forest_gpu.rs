//! GPU-accelerated random partition forest for kNN graph initialisation.
//!
//! Replaces the CPU Annoy forest for populating the initial kNN graph. Builds
//! multiple random projection trees on GPU, computes intra-leaf pairwise
//! distances, and merges results via the existing proposal infrastructure from
//! nndescent_gpu.

#![allow(missing_docs)]

use cubecl::frontend::{Atomic, SharedMemory};
use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use std::time::Instant;

use crate::gpu::nndescent_gpu::{launch_merge_proposals, reset_proposals, MAX_PROPOSALS};
use crate::gpu::*;
use crate::prelude::*;
pub use crate::utils::rp_forest::ForestRouter;
use crate::utils::rp_forest::*;

/////////////
// Kernels //
/////////////

/// Project every point onto all of a tree's random vectors in one pass.
///
/// The projections do not depend on the partitioning at all: `random_vec` for
/// a level is derived purely from the tree seed and the level index, so all
/// `n_levels` of them are known before the first one is needed. Only the median-and-scatter step is
/// sequential across levels.
///
/// Reading each point's row once and accumulating `n_levels` dot products
/// reads the vector matrix once instead of once per level, with one launch and
/// one readback. The projection rows are read from global rather than staged in
/// shared memory: every thread reads the same element at the same time, so they
/// broadcast from cache, and staging would put a `n_levels * dim` ceiling on
/// the kernel.
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix `[n_pts, dim/N]` as `Vector<F, N>`
/// * `projections` - Random projection vectors `[n_trees * n_levels, dim/N]`,
///   tree-major
/// * `dot_values` - Output `[n_trees * n_levels, n_pts]`, level-major within
///   each tree so every level's block is contiguous and can be sliced for the
///   host-side median pass
/// * `n_pts` - Number of points
/// * `dim_lines` - `Vector<F, N>` elements per row (comptime)
/// * `n_levels` - Tree depth, i.e. number of projections (comptime)
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X`
///   -> point index
/// * `CUBE_POS_Z` -> tree index
#[cube(launch_unchecked)]
fn compute_dot_products_multi<F: CubeclFloat, N: Size>(
    vectors: &Tensor<Vector<F, N>>,
    projections: &Tensor<Vector<F, N>>,
    dot_values: &mut Tensor<F>,
    n_pts: u32,
    #[comptime] dim_lines: usize,
    #[comptime] n_levels: usize,
) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    if idx >= n_pts {
        terminate!();
    }
    let lanes = LINE_SIZE;
    let off = idx as usize * dim_lines;
    let tree = CUBE_POS_Z as usize;
    let proj_base = tree * n_levels * dim_lines;
    let out_base = tree * n_levels * n_pts as usize;

    let mut acc = Array::<F>::new(n_levels);
    #[unroll]
    for l in 0..n_levels {
        acc[l] = F::new(0.0_f32);
    }

    for i in 0..dim_lines {
        let v = vectors[off + i];
        #[unroll]
        for l in 0..n_levels {
            let r = projections[proj_base + l * dim_lines + i];
            let prod = v * r;
            #[unroll]
            for lane in 0..lanes {
                acc[l] += prod[lane];
            }
        }
    }

    // Level-major write: consecutive threads hit consecutive addresses.
    #[unroll]
    for l in 0..n_levels {
        dot_values[out_base + l * n_pts as usize + idx as usize] = acc[l];
    }
}

/// Points per leaf whose staging fits the device's shared-memory budget.
///
/// Per-point cost is `dim_padded * elem_bytes` for the vector, four bytes for
/// the pid and `elem_bytes` for the threshold, plus eight fixed bytes covering
/// `shared_leaf_start` and `shared_leaf_size`.
///
/// ### Params
///
/// * `dim_padded` - Vector dimensionality padded to a multiple of `LINE_SIZE`
/// * `elem_bytes` - Size of the float element type in bytes
/// * `limits` - Device limits from `GpuLimits::from_client`
///
/// ### Returns
///
/// Maximum points per leaf, capped at 256, or `DimTooHighForSharedMemory` when
/// fewer than two points fit. Two is the floor because the kernel computes
/// pairwise distances and a leaf of one has no pairs.
fn compute_max_leaf_size(
    dim_padded: usize,
    elem_bytes: usize,
    limits: &GpuLimits,
) -> Result<usize, AnnSearchErrors> {
    // `shared_leaf_start` + `shared_leaf_size`
    const OVERHEAD: usize = 8;

    // Per point: `shared_vecs` holds a row, `shared_pids` a u32 and
    // `shared_thresh` a float. Every `SharedMemory` in
    // `leaf_pairwise_proposals` is counted here; missing one busts the device
    // limit, and `launch_unchecked` then does no work and reports nothing.
    let per_point = dim_padded * elem_bytes + 4 + elem_bytes;
    let available = limits.max_shared_bytes.saturating_sub(OVERHEAD);
    let fits = available / per_point;

    if fits < 2 {
        return Err(AnnSearchErrors::DimTooHighForSharedMemory {
            chosen_dim: dim_padded,
            required: OVERHEAD + 2 * per_point,
            available: limits.max_shared_bytes,
        });
    }

    Ok(fits.min(256))
}

/// All-pairs distance computation within a leaf, emitting proposals.
///
/// One workgroup per leaf. Loads leaf vectors into scalar shared memory,
/// computes C(leaf_size, 2) pairwise distances, and writes proposals via
/// atomics. Overflow beyond `max_proposals` is dropped: a proposal is an
/// (index, distance) pair written as two separate stores, so a reservoir
/// sample that overwrites an already-claimed slot lets two threads interleave
/// their stores and pair one thread's index with the other's distance. The
/// atomic slot claim is the only thing that keeps the two halves together.
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, line-vectorised `[n, dim/LINE_SIZE]`
/// * `leaf_points` - Flat array of global point IDs in leaf order
/// * `leaf_offsets` - CSR-style offsets into `leaf_points`, length n_leaves + 1
/// * `graph_dist` - Current kNN graph distances [n, k], used for threshold
///   filtering
/// * `prop_idx` - Output proposal indices `[n, max_proposals]`
/// * `prop_dist` - Output proposal distances `[n, max_proposals]`
/// * `prop_count` - Atomic per-node proposal counter `[n]`
/// * `n_pts` - Total number of points in the dataset
/// * `n_leaves` - Number of leaves in the current batch
/// * `max_proposals` - Proposal buffer capacity per node (comptime)
/// * `use_cosine` - Whether to compute cosine distance instead of squared
///   Euclidean (comptime)
/// * `dim_lines` - Number of `Line<F>` elements per vector row (comptime)
/// * `max_leaf_size` - Maximum points per leaf for shared memory allocation
///   (comptime)
///
/// ### Grid mapping
///
/// * One Cube per leaf
#[cube(launch_unchecked)]
pub fn leaf_pairwise_proposals<F: CubeclFloat, N: Size>(
    vectors: &Tensor<Vector<F, N>>,
    leaf_points: &Tensor<u32>,
    leaf_offsets: &Tensor<u32>,
    graph_dist: &Tensor<F>,
    prop_idx: &mut Tensor<u32>,
    prop_dist: &mut Tensor<F>,
    prop_count: &Tensor<Atomic<u32>>,
    n_pts: u32,
    n_leaves: u32,
    #[comptime] max_proposals: u32,
    #[comptime] use_cosine: bool,
    #[comptime] dim_lines: usize,
    #[comptime] max_leaf_size: usize,
) {
    let leaf_idx = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if leaf_idx >= n_leaves {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let lanes = LINE_SIZE;
    let dim_scalars = dim_lines * lanes;

    let mut shared_leaf_start = SharedMemory::<u32>::new(1usize);
    let mut shared_leaf_size = SharedMemory::<u32>::new(1usize);

    if tx == 0u32 {
        let start = leaf_offsets[leaf_idx as usize];
        let end = leaf_offsets[(leaf_idx + 1u32) as usize];
        shared_leaf_start[0usize] = start;
        shared_leaf_size[0usize] = end - start;
    }
    sync_cube();

    let leaf_start = shared_leaf_start[0usize];
    // Truncate rather than run off the end of the staging. The caller sizes
    // `max_leaf_size` from the batch's real leaves, so this never binds in
    // practice; a partition that cannot be split (every dot value equal, or
    // duplicate points) is the case it exists for, and dropping its tail costs
    // some proposals from one tree rather than an out-of-bounds shared write
    // that no backend reports.
    let mut leaf_size = shared_leaf_size[0usize];
    if leaf_size > max_leaf_size as u32 {
        leaf_size = max_leaf_size as u32;
    }

    if leaf_size < 2u32 {
        terminate!();
    }

    // Scalar shared memory (never use SharedMemory<Line<F>> -- see post-mortem)
    let mut shared_vecs = SharedMemory::<F>::new(max_leaf_size * dim_scalars);
    let mut shared_pids = SharedMemory::<u32>::new(max_leaf_size);
    let mut shared_thresh = SharedMemory::<F>::new(max_leaf_size);

    let k = graph_dist.shape(1usize);

    let mut i = tx;
    while i < leaf_size {
        let global_pid = leaf_points[(leaf_start + i) as usize];
        shared_pids[i as usize] = global_pid;
        // The partner's acceptance threshold is a global read inside the pair
        // loop otherwise, so `leaf_size` reads become `leaf_size^2 / 2`.
        shared_thresh[i as usize] = graph_dist[global_pid as usize * k + k - 1usize];
        i += WORKGROUP_SIZE_X;
    }
    sync_cube();

    let total_scalars = leaf_size as usize * dim_scalars;
    let mut idx_load = tx as usize;
    while idx_load < total_scalars {
        let n_idx = idx_load / dim_scalars;
        let s_idx = idx_load % dim_scalars;
        let line_idx = s_idx / lanes;
        let lane = s_idx % lanes;
        let pid = shared_pids[n_idx];

        if pid < n_pts {
            let vec_offset = pid as usize * dim_lines + line_idx;
            let line_val = vectors[vec_offset];
            shared_vecs[idx_load] = line_val[lane];
        }
        idx_load += WORKGROUP_SIZE_X as usize;
    }
    sync_cube();

    let mut ii = tx as usize;

    while ii < leaf_size as usize {
        let pid_i = shared_pids[ii];
        let thresh_i = shared_thresh[ii];

        let mut jj = ii + 1usize;
        while jj < leaf_size as usize {
            let pid_j = shared_pids[jj];

            if pid_i != pid_j && pid_i < n_pts && pid_j < n_pts {
                let mut sum = F::new(0.0_f32);
                #[unroll]
                for s in 0..dim_scalars {
                    let va = shared_vecs[ii * dim_scalars + s];
                    let vb = shared_vecs[jj * dim_scalars + s];
                    if use_cosine {
                        sum += va * vb;
                    } else {
                        let diff = va - vb;
                        sum += diff * diff;
                    }
                }

                let dist = if use_cosine {
                    F::new(1.0_f32) - sum
                } else {
                    sum
                };

                if dist < thresh_i {
                    let slot = prop_count[pid_i as usize].fetch_add(1u32);
                    if slot < max_proposals {
                        let off = pid_i as usize * max_proposals as usize + slot as usize;
                        prop_idx[off] = pid_j;
                        prop_dist[off] = dist;
                    }
                }

                let thresh_j = shared_thresh[jj];
                if dist < thresh_j {
                    let slot = prop_count[pid_j as usize].fetch_add(1u32);
                    if slot < max_proposals {
                        let off = pid_j as usize * max_proposals as usize + slot as usize;
                        prop_idx[off] = pid_i;
                        prop_dist[off] = dist;
                    }
                }
            }

            jj += 1usize;
        }

        ii += WORKGROUP_SIZE_X as usize;
    }
}

/// Set the IS_NEW flag on all non-sentinel graph entries.
///
/// ### Params
///
/// * `graph_idx` - kNN graph index buffer `[n * k]`; entries are updated
///   in-place by setting bit 31
/// * `total_entries` - Total number of entries in `graph_idx` (`n * k`)
///
/// ### Grid mapping
///
/// * Flat index = `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WG + UNIT_POS_X`
#[cube(launch_unchecked)]
pub fn mark_all_new(graph_idx: &mut Tensor<u32>, total_entries: u32) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    if idx >= total_entries {
        terminate!();
    }

    let val = graph_idx[idx as usize];
    let pid = val & 0x7FFFFFFFu32;
    if pid < 0x7FFFFFFFu32 {
        graph_idx[idx as usize] = pid | (1u32 << 31);
    }
}

////////////////////////
// Main orchestration //
////////////////////////

/// Build the initial kNN graph via GPU random partition forest.
///
/// Split host/device. Every level's random projection is a function of the
/// tree seed and the level index alone, so all `max_depth` projections are
/// generated upfront and their dot products come off the device in one
/// `compute_dot_products_multi` launch per tree: one readback per tree rather
/// than one launch plus one blocking readback per level. Medians and the
/// partition scatter then run on the host under rayon, and the leaf pairwise
/// distances plus the proposal merge go back to the device.
///
/// ### Params
///
/// * `vectors_gpu` - GPU-resident vector matrix `[n, dim_padded/LINE_SIZE]`
/// * `graph_idx_gpu` - kNN graph index buffer `[n, k]`; updated in-place
/// * `graph_dist_gpu` - kNN graph distance buffer `[n, k]`; updated in-place
/// * `prop_idx_gpu` - Proposal index scratch buffer `[n, MAX_PROPOSALS]`
/// * `prop_dist_gpu` - Proposal distance scratch buffer `[n, MAX_PROPOSALS]`
/// * `prop_count_gpu` - Atomic proposal counter scratch buffer `[n]`
/// * `update_counter_gpu` - Global update counter used by
///   `merge_proposals` `[1]`
/// * `n` - Number of points
/// * `dim` - Original (unpadded) vector dimensionality
/// * `dim_padded` - Vector dimensionality padded to a multiple of `LINE_SIZE`
/// * `n_trees` - Number of random projection trees to build
/// * `seed` - Base random seed; each tree and level derives its own seed from
///   this
/// * `use_cosine` - Whether to compute cosine distance instead of squared
///   Euclidean
/// * `verbose` - Print timing information for each phase
/// * `client` - GPU compute client
///
/// ### Returns
///
/// A [`ForestRouter`] holding the projections and medians of the first few
/// trees, for query-time entry-point routing. The graph itself lands in
/// `graph_idx_gpu` / `graph_dist_gpu` in place.
#[allow(clippy::too_many_arguments)]
pub fn gpu_forest_init<T, R>(
    vectors_gpu: &GpuTensor<R, T>,
    graph_idx_gpu: &GpuTensor<R, u32>,
    graph_dist_gpu: &GpuTensor<R, T>,
    prop_idx_gpu: &GpuTensor<R, u32>,
    prop_dist_gpu: &GpuTensor<R, T>,
    prop_count_gpu: &GpuTensor<R, u32>,
    update_counter_gpu: &GpuTensor<R, u32>,
    n: usize,
    dim: usize,
    dim_padded: usize,
    n_trees: usize,
    seed: usize,
    use_cosine: bool,
    verbose: bool,
    client: &ComputeClient<R>,
) -> Result<ForestRouter<T>, AnnSearchErrors>
where
    R: Runtime,
    T: AnnSearchFloat + CubeclFloat,
{
    let limits = GpuLimits::from_client(client);
    let line = LINE_SIZE;
    let dim_vec = dim_padded / line;
    let (grid_n_x, grid_n_y) = grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X), &limits)?;

    let max_leaf_size = compute_max_leaf_size(dim_padded, size_of::<T>(), &limits)?;

    // Leaves of ~64, but never more than the leaf kernel can stage. At high
    // dim the capacity drops below 64 (10 points at dim 784 on 32 KiB) and the
    // kernel would truncate every leaf, leaving most points without
    // proposals. Deeper trees keep every point in play; at dim <= 64 the
    // capacity is 256 and nothing changes.
    let target_leaf_size = 64.0_f64.min(max_leaf_size as f64);
    let max_depth = if n as f64 <= target_leaf_size {
        0
    } else {
        ((n as f64) / target_leaf_size).log2().ceil() as usize
    };

    // How many trees to keep routing data for (query entry points)
    let n_router_trees = n_trees.min(N_ROUTER_TREES);

    if verbose {
        println!(
            "  GPU forest init: {} trees, max_depth={}, max_leaf={}, router_trees={}",
            n_trees, max_depth, max_leaf_size, n_router_trees
        );
    }

    let forest_start = Instant::now();

    // Build trees in parallel
    let cpu_start = Instant::now();

    let dot_grid = (n as u32).div_ceil(WORKGROUP_SIZE_X);
    let (dot_grid_x, dot_grid_y) = grid_2d(dot_grid, &limits)?;

    // Every tree's projections up front, so the dot products for the whole
    // forest are one upload, one launch and one readback. Per-tree launches
    // from the rayon pool each paid their own upload and readback sync, and
    // those serialised on the client.
    let (projections_flat, tree_level_vecs) =
        forest_projections::<T>(n_trees, max_depth, dim, dim_padded, seed);

    let all_dots = if max_depth > 0 {
        let projections_gpu = GpuTensor::<R, T>::from_slice(
            &projections_flat,
            vec![n_trees * max_depth, dim_padded],
            client,
        )?;
        let all_dots_gpu = GpuTensor::<R, T>::empty(vec![n_trees * max_depth, n], client)?;
        let dot_count = checked_cube_count(
            "compute_dot_products_multi",
            dot_grid_x,
            dot_grid_y,
            n_trees as u32,
            &limits,
        )?;
        unsafe {
            compute_dot_products_multi::launch_unchecked::<T, R>(
                client,
                dot_count,
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.clone().into_tensor_arg(),
                projections_gpu.into_tensor_arg(),
                all_dots_gpu.clone().into_tensor_arg(),
                n as u32,
                dim_vec,
                max_depth,
            );
        }
        all_dots_gpu.read(client)?
    } else {
        Vec::new()
    };

    let (leaf_structures, router) = partition_forest(&all_dots, tree_level_vecs, n, max_depth, dim);

    if verbose {
        println!("    Tree construction: {:.2?}", cpu_start.elapsed());
    }

    // GPU phase: batched pairwise + merge
    let gpu_start = Instant::now();
    let trees_per_batch = 5;
    let n_batches = n_trees.div_ceil(trees_per_batch);

    for batch_idx in 0..n_batches {
        let batch_start = batch_idx * trees_per_batch;
        let batch_end = (batch_start + trees_per_batch).min(n_trees);

        let mut batch_leaf_points: Vec<u32> = Vec::new();
        let mut batch_leaf_offsets: Vec<u32> = Vec::new();

        for tree_idx in batch_start..batch_end {
            let (leaf_points, leaf_offsets, n_leaves) = &leaf_structures[tree_idx];

            if *n_leaves == 0 {
                continue;
            }

            let base_offset = batch_leaf_points.len() as u32;
            for i in 0..*n_leaves {
                batch_leaf_offsets.push(leaf_offsets[i] + base_offset);
            }
            batch_leaf_points.extend_from_slice(leaf_points);
        }

        batch_leaf_offsets.push(batch_leaf_points.len() as u32);
        let batch_leaves = batch_leaf_offsets.len() - 1;

        // Size the staging to the leaves this batch actually has, not to what
        // the device could hold: `max_leaf_size` is a capacity bound and real
        // leaves come out well under it, so staging for it wastes shared memory
        // and costs resident cubes. Rounded to a power of two so only a handful
        // of kernel variants ever compile.
        let batch_max_leaf = batch_leaf_offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);
        // The floor keeps a tiny batch from compiling a degenerate variant, but
        // it cannot exceed the capacity: past dim 128 `max_leaf_size` is itself
        // below the workgroup width.
        let stage_floor = (WORKGROUP_SIZE_X as usize).min(max_leaf_size);
        let leaf_stage = batch_max_leaf
            .next_power_of_two()
            .clamp(stage_floor, max_leaf_size);

        if batch_leaves == 0 {
            continue;
        }

        let leaf_points_gpu = GpuTensor::<R, u32>::from_slice(
            &batch_leaf_points,
            vec![batch_leaf_points.len()],
            client,
        )?;
        let leaf_offsets_gpu = GpuTensor::<R, u32>::from_slice(
            &batch_leaf_offsets,
            vec![batch_leaf_offsets.len()],
            client,
        )?;

        unsafe {
            reset_proposals::launch_unchecked::<R>(
                client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                prop_count_gpu.clone().into_tensor_arg(),
                update_counter_gpu.clone().into_tensor_arg(),
                n as u32,
            );
        }

        let (cubes_x, cubes_y) = grid_2d(batch_leaves as u32, &limits)?;

        unsafe {
            leaf_pairwise_proposals::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(cubes_x, cubes_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.clone().into_tensor_arg(),
                leaf_points_gpu.into_tensor_arg(),
                leaf_offsets_gpu.into_tensor_arg(),
                graph_dist_gpu.clone().into_tensor_arg(),
                prop_idx_gpu.clone().into_tensor_arg(),
                prop_dist_gpu.clone().into_tensor_arg(),
                prop_count_gpu.clone().into_tensor_arg(),
                n as u32,
                batch_leaves as u32,
                MAX_PROPOSALS as u32,
                use_cosine,
                dim_vec,
                leaf_stage,
            );
        }

        launch_merge_proposals::<T, R>(
            client,
            &limits,
            graph_idx_gpu,
            graph_dist_gpu,
            prop_idx_gpu,
            prop_dist_gpu,
            prop_count_gpu,
            update_counter_gpu,
            n,
            graph_idx_gpu.shape()[1],
        )?;
    }

    if verbose {
        println!(
            "    GPU batched pairwise + merge ({} batches): {:.2?}",
            n_batches,
            gpu_start.elapsed()
        );
    }

    if verbose {
        let _ = update_counter_gpu.clone().read(client);
        println!("  GPU forest init: {:.2?}", forest_start.elapsed());
    }

    Ok(router)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod budget_tests {
    use super::*;

    /// Shared-memory footprint of `leaf_pairwise_proposals`, counted off the
    /// kernel body rather than off `compute_max_leaf_size`.
    fn kernel_smem_bytes(leaf_size: usize, dim_padded: usize, elem_bytes: usize) -> usize {
        // shared_leaf_start + shared_leaf_size
        8
            // shared_vecs
            + leaf_size * dim_padded * elem_bytes
            // shared_pids
            + leaf_size * 4
            // shared_thresh
            + leaf_size * elem_bytes
    }

    fn limits_with(shared: usize) -> GpuLimits {
        GpuLimits {
            max_shared_bytes: shared,
            max_cube_count: (65_535, 65_535, 65_535),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 1024),
            max_binding_bytes: 4_294_967_292,
            plane_size_min: 32,
            plane_size_max: 32,
        }
    }

    /// The regression this exists for: `shared_thresh` was added to the kernel
    /// without being counted here, so at dim 128 on a 32 KiB device the leaf
    /// size came out one point too large and every launch silently did nothing.
    #[test]
    fn test_max_leaf_size_fits_the_kernel_footprint() {
        for shared in [16_384usize, 32_768, 49_152, 65_536] {
            for elem in [4usize, 8] {
                for dim in [8usize, 32, 64, 128, 256, 512, 1024] {
                    let l = limits_with(shared);
                    let Ok(max_leaf) = compute_max_leaf_size(dim, elem, &l) else {
                        continue;
                    };
                    let used = kernel_smem_bytes(max_leaf, dim, elem);
                    assert!(
                        used <= shared,
                        "shared {shared}, elem {elem}, dim {dim}: \
                         max_leaf {max_leaf} needs {used}"
                    );
                    assert!(max_leaf >= 2);
                }
            }
        }
    }

    /// The staging bucket the batch loop computes must fit too, for every leaf
    /// size the data could produce.
    #[test]
    fn test_leaf_stage_bucket_fits() {
        let l = limits_with(32_768);
        for dim in [32usize, 64, 128, 256, 512] {
            let max_leaf = compute_max_leaf_size(dim, 4, &l).unwrap();
            for batch_max_leaf in 1..=max_leaf {
                let stage_floor = (WORKGROUP_SIZE_X as usize).min(max_leaf);
                let stage = batch_max_leaf
                    .next_power_of_two()
                    .clamp(stage_floor, max_leaf);
                assert!(
                    kernel_smem_bytes(stage, dim, 4) <= 32_768,
                    "dim {dim}, batch max {batch_max_leaf}: stage {stage} over budget"
                );
                assert!(stage >= batch_max_leaf.min(max_leaf));
            }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "gpu-tests")]
mod tests {
    use super::*;
    use cubecl::wgpu::WgpuDevice;
    use cubecl::wgpu::WgpuRuntime;

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cubecl::wgpu::WgpuRuntime::client(&device);
        }));
        result.ok().map(|_| device)
    }

    // For testing the code with 32 dimension
    const MAX_LEAF_SIZE: usize = 128;

    #[test]
    fn test_dot_products_multi_matches_host() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE;
        let n = 133usize;
        let dim = 16usize;
        let dim_vec = dim / line;
        let n_levels = 5usize;

        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 31 + 7) % 23) as f32 * 0.13)
            .collect();
        let projections: Vec<f32> = (0..n_levels * dim)
            .map(|i| ((i * 17 + 3) % 19) as f32 * 0.21 - 1.0)
            .collect();

        let vectors_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client).unwrap();
        let proj_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&projections, vec![n_levels, dim], &client)
                .unwrap();
        let multi_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n_levels, n], &client).unwrap();

        let grid = (n as u32).div_ceil(WORKGROUP_SIZE_X);
        unsafe {
            compute_dot_products_multi::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(grid, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.clone().into_tensor_arg(),
                proj_gpu.into_tensor_arg(),
                multi_gpu.clone().into_tensor_arg(),
                n as u32,
                dim_vec,
                n_levels,
            );
        }
        let multi = multi_gpu.read(&client).unwrap();

        // Host reference, one level at a time.
        for level in 0..n_levels {
            let rvec = &projections[level * dim..(level + 1) * dim];
            for i in 0..n {
                let b: f32 = data[i * dim..(i + 1) * dim]
                    .iter()
                    .zip(rvec)
                    .map(|(x, r)| x * r)
                    .sum();
                let a = multi[level * n + i];
                assert!(
                    (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                    "level {level} point {i}: multi {a}, host {b}"
                );
            }
        }
    }

    #[cube(launch_unchecked)]
    fn debug_leaf_shared_roundtrip<F: CubeclFloat, N: Size>(
        vectors: &Tensor<Vector<F, N>>,
        leaf_points: &Tensor<u32>,
        leaf_offsets: &Tensor<u32>,
        out_vecs: &mut Tensor<F>,
        n_pts: u32,
        #[comptime] dim_lines: usize,
        #[comptime] max_leaf_size: usize,
    ) {
        let leaf_idx = CUBE_POS_X;
        let tx = UNIT_POS_X;
        let lanes = LINE_SIZE;
        let dim_scalars = dim_lines * lanes;

        let mut shared_leaf_start = SharedMemory::<u32>::new(1usize);
        let mut shared_leaf_size = SharedMemory::<u32>::new(1usize);

        if tx == 0u32 {
            let start = leaf_offsets[leaf_idx as usize];
            let end = leaf_offsets[(leaf_idx + 1u32) as usize];
            shared_leaf_start[0usize] = start;
            shared_leaf_size[0usize] = end - start;
        }
        sync_cube();

        let leaf_start = shared_leaf_start[0usize];
        let leaf_size = shared_leaf_size[0usize];

        let mut shared_vecs = SharedMemory::<F>::new(max_leaf_size * dim_scalars);
        let mut shared_pids = SharedMemory::<u32>::new(max_leaf_size);

        let mut i = tx;
        while i < leaf_size {
            shared_pids[i as usize] = leaf_points[(leaf_start + i) as usize];
            i += WORKGROUP_SIZE_X;
        }
        sync_cube();

        let total_scalars = leaf_size as usize * dim_scalars;
        let mut idx_load = tx as usize;
        while idx_load < total_scalars {
            let n_idx = idx_load / dim_scalars;
            let s_idx = idx_load % dim_scalars;
            let line_idx = s_idx / lanes;
            let lane = s_idx % lanes;
            let pid = shared_pids[n_idx];

            if pid < n_pts {
                let vec_offset = pid as usize * dim_lines + line_idx;
                let line_val = vectors[vec_offset];
                shared_vecs[idx_load] = line_val[lane];
            }
            idx_load += WORKGROUP_SIZE_X as usize;
        }
        sync_cube();

        if tx == 0u32 {
            let mut w = 0usize;
            while w < total_scalars {
                out_vecs[w] = shared_vecs[w];
                w += 1usize;
            }
        }
    }

    #[test]
    fn test_leaf_shared_memory_roundtrip() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE;
        let n = 8usize;
        let dim = 8usize;
        let dim_vec = dim / line;

        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                data[i * dim + j] = (i * 100 + j) as f32;
            }
        }

        let leaf_points: Vec<u32> = vec![2, 5, 7];
        let leaf_offsets: Vec<u32> = vec![0, 3];

        let vectors_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client).unwrap();
        let lp_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_points, vec![3], &client).unwrap();
        let lo_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_offsets, vec![2], &client).unwrap();
        let out_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; MAX_LEAF_SIZE * dim],
            vec![MAX_LEAF_SIZE * dim],
            &client,
        )
        .unwrap();

        unsafe {
            debug_leaf_shared_roundtrip::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.into_tensor_arg(),
                lp_gpu.into_tensor_arg(),
                lo_gpu.into_tensor_arg(),
                out_gpu.clone().into_tensor_arg(),
                n as u32,
                dim_vec,
                MAX_LEAF_SIZE,
            );
        }

        let result = out_gpu.read(&client).unwrap();
        for (local_idx, &global_pid) in leaf_points.iter().enumerate() {
            let expected: Vec<f32> = (0..dim)
                .map(|j| (global_pid as usize * 100 + j) as f32)
                .collect();
            let got: Vec<f32> = result[local_idx * dim..(local_idx + 1) * dim].to_vec();
            assert_eq!(
                got, expected,
                "Leaf slot {local_idx} (pid={global_pid}) mismatch"
            );
        }
    }

    #[test]
    fn test_leaf_shared_memory_roundtrip_dim32() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE;
        let n = 64usize;
        let dim = 32usize;
        let dim_vec = dim / line;

        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                data[i * dim + j] = (i * 1000 + j) as f32;
            }
        }

        let leaf_points: Vec<u32> = vec![3, 10, 22, 31, 45, 50, 58, 63];
        let leaf_offsets: Vec<u32> = vec![0, 8];

        let vectors_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client).unwrap();
        let lp_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_points, vec![8], &client).unwrap();
        let lo_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_offsets, vec![2], &client).unwrap();
        let out_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; MAX_LEAF_SIZE * dim],
            vec![MAX_LEAF_SIZE * dim],
            &client,
        )
        .unwrap();

        unsafe {
            debug_leaf_shared_roundtrip::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.into_tensor_arg(),
                lp_gpu.into_tensor_arg(),
                lo_gpu.into_tensor_arg(),
                out_gpu.clone().into_tensor_arg(),
                n as u32,
                dim_vec,
                MAX_LEAF_SIZE,
            );
        }

        let result = out_gpu.read(&client).unwrap();
        for (local_idx, &global_pid) in leaf_points.iter().enumerate() {
            for j in 0..dim {
                let got = result[local_idx * dim + j];
                let expected = (global_pid as usize * 1000 + j) as f32;
                assert!(
                    (got - expected).abs() < 1e-4,
                    "pid={global_pid}, dim={j}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_leaf_pairwise_small_euclidean() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE;
        let n = 4usize;
        let dim = 4usize;
        let dim_vec = dim / line;
        let build_k = 3usize;

        let data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let leaf_points: Vec<u32> = vec![0, 1, 2, 3];
        let leaf_offsets: Vec<u32> = vec![0, 4];
        let graph_dist = vec![f32::MAX; n * build_k];

        let vectors_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client).unwrap();
        let lp_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_points, vec![4], &client).unwrap();
        let lo_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_offsets, vec![2], &client).unwrap();
        let gdist_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&graph_dist, vec![n, build_k], &client)
                .unwrap();
        let prop_idx_gpu =
            GpuTensor::<WgpuRuntime, u32>::empty(vec![n, MAX_PROPOSALS], &client).unwrap();
        let prop_dist_gpu =
            GpuTensor::<WgpuRuntime, f32>::empty(vec![n, MAX_PROPOSALS], &client).unwrap();
        let prop_count_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client).unwrap();

        unsafe {
            leaf_pairwise_proposals::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.into_tensor_arg(),
                lp_gpu.into_tensor_arg(),
                lo_gpu.into_tensor_arg(),
                gdist_gpu.into_tensor_arg(),
                prop_idx_gpu.clone().into_tensor_arg(),
                prop_dist_gpu.clone().into_tensor_arg(),
                prop_count_gpu.clone().into_tensor_arg(),
                n as u32,
                1u32,
                MAX_PROPOSALS as u32,
                false,
                dim_vec,
                MAX_LEAF_SIZE,
            );
        }

        let p_idx = prop_idx_gpu.read(&client).unwrap();
        let p_dist = prop_dist_gpu.read(&client).unwrap();
        let p_count = prop_count_gpu.read(&client).unwrap();

        for node in 0..n {
            assert_eq!(
                p_count[node] as usize, 3,
                "node {node} should have 3 proposals"
            );
        }

        let mut any_mismatch = false;
        for node in 0..n {
            let count = (p_count[node] as usize).min(MAX_PROPOSALS);
            for p in 0..count {
                let cand = p_idx[node * MAX_PROPOSALS + p] as usize;
                let gpu_dist = p_dist[node * MAX_PROPOSALS + p];
                let cpu_dist: f32 = data[node * dim..(node + 1) * dim]
                    .iter()
                    .zip(&data[cand * dim..(cand + 1) * dim])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                if (gpu_dist - cpu_dist).abs() > 1e-4 {
                    any_mismatch = true;
                }
            }
        }
        assert!(!any_mismatch, "Distance mismatches found");
    }

    #[test]
    fn test_leaf_pairwise_small_cosine() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE;
        let n = 4usize;
        let dim = 4usize;
        let dim_vec = dim / line;
        let build_k = 3usize;
        let data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let norms: Vec<f32> = (0..n)
            .map(|i| {
                data[i * dim..(i + 1) * dim]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();
        let leaf_points: Vec<u32> = vec![0, 1, 2, 3];
        let leaf_offsets: Vec<u32> = vec![0, 4];
        let graph_dist = vec![f32::MAX; n * build_k];
        // The kernel takes unit rows; the reference below keeps the raw ones.
        let mut unit = data.clone();
        crate::gpu::normalise_rows(&mut unit, dim);
        let vectors_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&unit, vec![n, dim], &client).unwrap();
        let lp_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_points, vec![4], &client).unwrap();
        let lo_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_offsets, vec![2], &client).unwrap();
        let gdist_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&graph_dist, vec![n, build_k], &client)
                .unwrap();
        let prop_idx_gpu =
            GpuTensor::<WgpuRuntime, u32>::empty(vec![n, MAX_PROPOSALS], &client).unwrap();
        let prop_dist_gpu =
            GpuTensor::<WgpuRuntime, f32>::empty(vec![n, MAX_PROPOSALS], &client).unwrap();
        let prop_count_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client).unwrap();

        eprintln!(
                "config: dim={dim} line={line} dim_vec={dim_vec} MAX_LEAF_SIZE={MAX_LEAF_SIZE} MAX_PROPOSALS={MAX_PROPOSALS}"
            );

        unsafe {
            leaf_pairwise_proposals::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.into_tensor_arg(),
                lp_gpu.into_tensor_arg(),
                lo_gpu.into_tensor_arg(),
                gdist_gpu.into_tensor_arg(),
                prop_idx_gpu.clone().into_tensor_arg(),
                prop_dist_gpu.clone().into_tensor_arg(),
                prop_count_gpu.clone().into_tensor_arg(),
                n as u32,
                1u32,
                MAX_PROPOSALS as u32,
                true,
                dim_vec,
                MAX_LEAF_SIZE,
            );
        }
        let p_idx = prop_idx_gpu.read(&client).unwrap();
        let p_dist = prop_dist_gpu.read(&client).unwrap();
        let p_count = prop_count_gpu.read(&client).unwrap();

        let mut any_negative = false;
        let mut any_mismatch = false;
        for node in 0..n {
            let count = (p_count[node] as usize).min(MAX_PROPOSALS);
            eprintln!("node {node}: count={} (raw={})", count, p_count[node]);
            for p in 0..count {
                let cand = p_idx[node * MAX_PROPOSALS + p] as usize;
                let gpu_dist = p_dist[node * MAX_PROPOSALS + p];
                let dot: f32 = data[node * dim..(node + 1) * dim]
                    .iter()
                    .zip(&data[cand * dim..(cand + 1) * dim])
                    .map(|(a, b)| a * b)
                    .sum();
                let cpu_dist = 1.0 - dot / (norms[node] * norms[cand]);
                let neg = gpu_dist < -1e-6;
                let mismatch = (gpu_dist - cpu_dist).abs() > 1e-4;
                eprintln!(
                        "  p{p}: cand={cand} gpu={gpu_dist:.6} cpu={cpu_dist:.6} neg={neg} mismatch={mismatch}"
                    );
                any_negative |= neg;
                any_mismatch |= mismatch;
            }
        }
        assert!(!any_negative, "Negative cosine distances");
        assert!(!any_mismatch, "Cosine distance mismatches");
    }

    #[test]
    fn test_leaf_pairwise_dim32() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE;
        let n = 32usize;
        let dim = 32usize;
        let dim_vec = dim / line;
        let build_k = 10usize;

        let data: Vec<f32> = (0..n * dim)
            .map(|idx| ((idx % 7) as f32) * 0.1 + (idx / dim) as f32)
            .collect();
        let leaf_points: Vec<u32> = (0..16).map(|i| i as u32).collect();
        let leaf_offsets: Vec<u32> = vec![0, 16];
        let graph_dist = vec![f32::MAX; n * build_k];

        let vectors_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client).unwrap();
        let lp_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_points, vec![16], &client).unwrap();
        let lo_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_offsets, vec![2], &client).unwrap();
        let gdist_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&graph_dist, vec![n, build_k], &client)
                .unwrap();
        let prop_idx_gpu =
            GpuTensor::<WgpuRuntime, u32>::empty(vec![n, MAX_PROPOSALS], &client).unwrap();
        let prop_dist_gpu =
            GpuTensor::<WgpuRuntime, f32>::empty(vec![n, MAX_PROPOSALS], &client).unwrap();
        let prop_count_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client).unwrap();

        unsafe {
            leaf_pairwise_proposals::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.into_tensor_arg(),
                lp_gpu.into_tensor_arg(),
                lo_gpu.into_tensor_arg(),
                gdist_gpu.into_tensor_arg(),
                prop_idx_gpu.clone().into_tensor_arg(),
                prop_dist_gpu.clone().into_tensor_arg(),
                prop_count_gpu.clone().into_tensor_arg(),
                n as u32,
                1u32,
                MAX_PROPOSALS as u32,
                false,
                dim_vec,
                MAX_LEAF_SIZE,
            );
        }

        let p_idx = prop_idx_gpu.read(&client).unwrap();
        let p_dist = prop_dist_gpu.read(&client).unwrap();
        let p_count = prop_count_gpu.read(&client).unwrap();

        let mut mismatch_count = 0;
        for node in 0..16 {
            let count = (p_count[node] as usize).min(5);
            for p in 0..count {
                let cand = p_idx[node * MAX_PROPOSALS + p] as usize;
                let gpu_dist = p_dist[node * MAX_PROPOSALS + p];
                let cpu_dist: f32 = data[node * dim..(node + 1) * dim]
                    .iter()
                    .zip(&data[cand * dim..(cand + 1) * dim])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                if (gpu_dist - cpu_dist).abs() > 1e-2 {
                    mismatch_count += 1;
                }
            }
        }
        assert!(
            mismatch_count == 0,
            "dim=32 leaf pairwise: {mismatch_count} mismatches"
        );
    }

    #[test]
    fn test_forest_init_recall() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let n = 500usize;
        let dim = 8usize;
        let dim_padded = dim;
        let build_k = 10usize;
        let n_trees = 5;

        let data: Vec<f32> = (0..n)
            .flat_map(|i| {
                let cluster = (i / 100) as f32 * 10.0;
                (0..dim).map(move |j| cluster + (i % 100) as f32 * 0.05 + j as f32 * 0.01)
            })
            .collect();

        let vectors_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim_padded], &client).unwrap();
        let graph_idx_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
            &vec![0x7FFFFFFFu32; n * build_k],
            vec![n, build_k],
            &client,
        )
        .unwrap();
        let graph_dist_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![f32::MAX; n * build_k],
            vec![n, build_k],
            &client,
        )
        .unwrap();
        let prop_idx_gpu =
            GpuTensor::<WgpuRuntime, u32>::empty(vec![n, MAX_PROPOSALS], &client).unwrap();
        let prop_dist_gpu =
            GpuTensor::<WgpuRuntime, f32>::empty(vec![n, MAX_PROPOSALS], &client).unwrap();
        let prop_count_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client).unwrap();
        let update_counter_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32], vec![1], &client).unwrap();

        let _ = gpu_forest_init(
            &vectors_gpu,
            &graph_idx_gpu,
            &graph_dist_gpu,
            &prop_idx_gpu,
            &prop_dist_gpu,
            &prop_count_gpu,
            &update_counter_gpu,
            n,
            dim,
            dim_padded,
            n_trees,
            42,
            false,
            true,
            &client,
        );

        let result_idx = graph_idx_gpu.read(&client).unwrap();
        let pid_mask = 0x7FFFFFFFu32;

        let mut total_hits = 0;
        let mut total_possible = 0;
        for i in 0..n {
            let mut dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f32 = data[i * dim..(i + 1) * dim]
                        .iter()
                        .zip(&data[j * dim..(j + 1) * dim])
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt_set: std::collections::HashSet<usize> =
                dists.iter().take(build_k).map(|&(j, _)| j).collect();
            let init_set: std::collections::HashSet<usize> = (0..build_k)
                .map(|j| (result_idx[i * build_k + j] & pid_mask) as usize)
                .filter(|&pid| pid < n)
                .collect();
            total_hits += gt_set.intersection(&init_set).count();
            total_possible += build_k;
        }

        let recall = total_hits as f64 / total_possible as f64;
        println!("Forest init recall@{build_k} ({n_trees} trees): {recall:.4}");
        assert!(recall > 0.3, "Forest init recall too low: {recall:.4}");
    }

    #[test]
    fn test_build_leaf_structure() {
        let partition_ids = vec![2u32, 0, 1, 0, 2, 1, 0, 2];
        let (leaf_points, leaf_offsets, n_leaves) = build_leaf_structure(&partition_ids, 8);

        assert_eq!(n_leaves, 3);
        let mut all_points: Vec<u32> = leaf_points.clone();
        all_points.sort();
        assert_eq!(all_points, vec![0, 1, 2, 3, 4, 5, 6, 7]);

        let mut sizes: Vec<u32> = (0..n_leaves)
            .map(|i| leaf_offsets[i + 1] - leaf_offsets[i])
            .collect();
        sizes.sort();
        assert_eq!(sizes, vec![2, 3, 3]);
    }

    #[test]
    fn test_compute_partition_medians() {
        let partition_ids = vec![0u32, 0, 0, 0, 1, 1, 1, 1];
        let dot_values = vec![1.0f32, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
        let medians = compute_partition_medians(&partition_ids, &dot_values, 2);
        assert!((medians[0] - 5.0).abs() < 1e-6);
        assert!((medians[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_mark_all_new() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);

        let sentinel = 0x7FFFFFFFu32;
        let data = vec![5u32, 10 | (1u32 << 31), sentinel, 42u32];
        let gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&data, vec![4], &client).unwrap();

        unsafe {
            mark_all_new::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                gpu.clone().into_tensor_arg(),
                4u32,
            );
        }

        let result = gpu.read(&client).unwrap();
        let is_new = 1u32 << 31;
        let pid_mask = 0x7FFFFFFFu32;

        assert_eq!(result[0] & pid_mask, 5);
        assert_ne!(result[0] & is_new, 0);
        assert_eq!(result[1] & pid_mask, 10);
        assert_ne!(result[1] & is_new, 0);
        assert_eq!(result[2], sentinel);
        assert_eq!(result[3] & pid_mask, 42);
        assert_ne!(result[3] & is_new, 0);
    }
}
