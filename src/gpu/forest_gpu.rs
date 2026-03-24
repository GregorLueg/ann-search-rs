//! GPU-accelerated random partition forest for kNN graph initialisation.
//!
//! Replaces the CPU Annoy forest for populating the initial kNN graph.
//! Builds multiple random projection trees on GPU, computes intra-leaf
//! pairwise distances, and merges results via the existing proposal
//! infrastructure from nndescent_gpu.

#![allow(missing_docs)]

use cubecl::frontend::{Atomic, SharedMemory};
use cubecl::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use crate::gpu::nndescent_gpu::{merge_proposals, reset_proposals, MAX_PROPOSALS};
use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::prelude::*;

///////////
// Const //
///////////

/// Shared memory budget in bytes (conservative for Apple Silicon / older GPUs)
const SMEM_BUDGET: usize = 32_768;

/// Result of parallel tree construction for a single tree.
///
/// Each element is a tuple of:
/// - `Vec<u32>` -- final partition ID per point (length `n`), used to build the
///   leaf structure for GPU pairwise distance computation.
/// - `Option<Vec<Vec<T>>>` -- per-level random projection vectors
///   (`[max_depth][dim]`). Present only for trees retained by the
///   `ForestRouter` for query-time entry point routing.
/// - `Option<Vec<Vec<T>>>` -- per-level partition medians
///   (`[max_depth][n_partitions_at_level]`). Present only for router
///   trees, paired with the projection vectors above.
type TreeResults<T> = Vec<(Vec<u32>, Option<Vec<Vec<T>>>, Option<Vec<Vec<T>>>)>;

////////////////////
// Kernel helpers //
////////////////////

/// Single xorshift step used to generate random node offsets during
/// reservoir sampling.
///
/// ### Params
///
/// * `state` - Current RNG state (must be non-zero)
///
/// ### Returns
///
/// Next RNG state
#[cube]
fn xorshift32(state: u32) -> u32 {
    let mut x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

/////////////
// Kernels //
/////////////

/// Compute dot product of each vector with a random projection vector.
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, line-vectorised `[n, dim/LINE_SIZE]`
/// * `random_vec` - Random projection vector `[dim/LINE_SIZE]`
/// * `dot_values` - Output dot products `[n]`
/// * `n` - Number of points
/// * `dim_lines` - Number of `Line<F>` elements per vector row (comptime)
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> point index
#[cube(launch_unchecked)]
fn compute_dot_products<F: AnnSearchGpuFloat>(
    vectors: &Tensor<Line<F>>,
    random_vec: &Tensor<Line<F>>,
    dot_values: &mut Tensor<F>,
    n: u32,
    #[comptime] dim_lines: usize,
) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    if idx >= n {
        terminate!();
    }

    let off = idx as usize * dim_lines;
    let mut sum = F::new(0.0);

    for i in 0..dim_lines {
        let v = vectors[off + i];
        let r = random_vec[i];
        let prod = v * r;
        sum += prod[0];
        sum += prod[1];
        sum += prod[2];
        sum += prod[3];
    }

    dot_values[idx as usize] = sum;
}

/// Partition points by comparing dot products against per-partition medians.
///
/// ### Params
///
/// * `partition_id` - Current partition ID per point `[n]`; updated in-place
///   to `pid * 2` (left) or `pid * 2 + 1` (right)
/// * `dot_values` - Dot product of each point with the projection vector `[n]`
/// * `medians` - Median dot value per partition `[n_partitions]`
/// * `n` - Number of points
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> point index
#[cube(launch_unchecked)]
fn partition_points<F: AnnSearchGpuFloat>(
    partition_id: &mut Tensor<u32>,
    dot_values: &Tensor<F>,
    medians: &Tensor<F>,
    n: u32,
) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    if idx >= n {
        terminate!();
    }

    let pid = partition_id[idx as usize];
    let dot = dot_values[idx as usize];
    let median = medians[pid as usize];

    let new_pid = if dot <= median {
        pid * 2u32
    } else {
        pid * 2u32 + 1u32
    };
    partition_id[idx as usize] = new_pid;
}

/// Compute the maximum leaf size that fits within the shared memory budget
/// for `leaf_pairwise_proposals`.
///
/// Per-point cost: `dim_scalars * sizeof(F)` (vectors) + `sizeof(u32)` (pid)
/// + `sizeof(F)` (norm).
///
/// ### Params
///
/// * `dim_padded` - Vector dimensionality padded to a multiple of `LINE_SIZE`
///
/// ### Returns
///
/// Maximum number of points per leaf, clamped to `[2, 256]`
fn compute_max_leaf_size(dim_padded: usize) -> usize {
    let line = LINE_SIZE as usize;
    let dim_scalars = (dim_padded / line) * 4;
    let per_point = dim_scalars * std::mem::size_of::<f32>() + 4 + 4;
    let overhead = 8; // shared_leaf_start + shared_leaf_size
    let available = SMEM_BUDGET.saturating_sub(overhead);
    (available / per_point).clamp(2, 256)
}

/// All-pairs distance computation within a leaf, emitting proposals.
///
/// One workgroup per leaf. Loads leaf vectors into scalar shared memory,
/// computes C(leaf_size, 2) pairwise distances, and writes proposals via
/// atomics. Overflow beyond `max_proposals` is handled via reservoir sampling.
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, line-vectorised [n, dim/LINE_SIZE]
/// * `norms` - Pre-computed L2 norms [n] (ignored when `use_cosine` is false)
/// * `leaf_points` - Flat array of global point IDs in leaf order
/// * `leaf_offsets` - CSR-style offsets into `leaf_points`, length n_leaves + 1
/// * `graph_dist` - Current kNN graph distances [n, k], used for threshold
///   filtering
/// * `prop_idx` - Output proposal indices [n, max_proposals]
/// * `prop_dist` - Output proposal distances [n, max_proposals]
/// * `prop_count` - Atomic per-node proposal counter [n]
/// * `n` - Total number of points in the dataset
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
pub fn leaf_pairwise_proposals<F: AnnSearchGpuFloat>(
    vectors: &Tensor<Line<F>>,
    norms: &Tensor<F>,
    leaf_points: &Tensor<u32>,
    leaf_offsets: &Tensor<u32>,
    graph_dist: &Tensor<F>,
    prop_idx: &mut Tensor<u32>,
    prop_dist: &mut Tensor<F>,
    prop_count: &Tensor<Atomic<u32>>,
    n: u32,
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
    let dim_scalars = dim_lines * 4usize;

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

    if leaf_size < 2u32 {
        terminate!();
    }

    // Scalar shared memory (never use SharedMemory<Line<F>> -- see post-mortem)
    let mut shared_vecs = SharedMemory::<F>::new(max_leaf_size * dim_scalars);
    let mut shared_pids = SharedMemory::<u32>::new(max_leaf_size);
    let mut shared_norms = SharedMemory::<F>::new(max_leaf_size);

    let mut i = tx;
    while i < leaf_size {
        let global_pid = leaf_points[(leaf_start + i) as usize];
        shared_pids[i as usize] = global_pid;
        if use_cosine {
            shared_norms[i as usize] = norms[global_pid as usize];
        }
        i += WORKGROUP_SIZE_X;
    }
    sync_cube();

    let total_scalars = leaf_size as usize * dim_scalars;
    let mut idx_load = tx as usize;
    while idx_load < total_scalars {
        let n_idx = idx_load / dim_scalars;
        let s_idx = idx_load % dim_scalars;
        let line_idx = s_idx / 4usize;
        let lane = s_idx % 4usize;
        let pid = shared_pids[n_idx];

        if pid < n {
            let vec_offset = pid as usize * dim_lines + line_idx;
            let line_val = vectors[vec_offset];
            shared_vecs[idx_load] = line_val[lane];
        }
        idx_load += WORKGROUP_SIZE_X as usize;
    }
    sync_cube();

    let k = graph_dist.shape(1usize);
    let num_pairs = (leaf_size * (leaf_size - 1u32)) / 2u32;
    let mut pair_idx = tx;

    while pair_idx < num_pairs {
        let mut rem = pair_idx as usize;
        let mut ii = 0usize;
        let mut step = leaf_size as usize - 1usize;

        while rem >= step {
            rem -= step;
            ii += 1usize;
            step = leaf_size as usize - 1usize - ii;
        }
        let jj = ii + 1usize + rem;

        let pid_i = shared_pids[ii];
        let pid_j = shared_pids[jj];

        if pid_i != pid_j && pid_i < n && pid_j < n {
            let mut sum = F::new(0.0);
            let mut s = 0usize;
            while s < dim_scalars {
                let va = shared_vecs[ii * dim_scalars + s];
                let vb = shared_vecs[jj * dim_scalars + s];
                if use_cosine {
                    sum += va * vb;
                } else {
                    let diff = va - vb;
                    sum += diff * diff;
                }
                s += 1usize;
            }

            let dist = if use_cosine {
                F::new(1.0) - (sum / (shared_norms[ii] * shared_norms[jj]))
            } else {
                sum
            };

            let thresh_i = graph_dist[pid_i as usize * k + k - 1usize];
            if dist < thresh_i {
                let slot = prop_count[pid_i as usize].fetch_add(1u32);
                if slot < max_proposals {
                    let off = pid_i as usize * max_proposals as usize + slot as usize;
                    prop_idx[off] = pid_j;
                    prop_dist[off] = dist;
                } else {
                    let rand = xorshift32(pid_i ^ slot ^ pid_j) % (slot + 1u32);
                    if rand < max_proposals {
                        let off = pid_i as usize * max_proposals as usize + rand as usize;
                        prop_idx[off] = pid_j;
                        prop_dist[off] = dist;
                    }
                }
            }

            let thresh_j = graph_dist[pid_j as usize * k + k - 1usize];
            if dist < thresh_j {
                let slot = prop_count[pid_j as usize].fetch_add(1u32);
                if slot < max_proposals {
                    let off = pid_j as usize * max_proposals as usize + slot as usize;
                    prop_idx[off] = pid_i;
                    prop_dist[off] = dist;
                } else {
                    let rand = xorshift32(pid_j ^ slot ^ pid_i) % (slot + 1u32);
                    if rand < max_proposals {
                        let off = pid_j as usize * max_proposals as usize + rand as usize;
                        prop_idx[off] = pid_i;
                        prop_dist[off] = dist;
                    }
                }
            }
        }

        pair_idx += WORKGROUP_SIZE_X;
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

/////////////////
// CPU helpers //
/////////////////

/// Build leaf-point arrays from final partition IDs.
///
/// Groups points by partition, sorts them, and builds a CSR-style offset
/// array for subsequent per-leaf GPU kernels.
///
/// ### Params
///
/// * `partition_ids` - Partition ID per point, length `n`
/// * `n` - Number of points
///
/// ### Returns
///
/// `(leaf_points, leaf_offsets, n_leaves)` where `leaf_points` is the
/// global point IDs sorted by partition, `leaf_offsets` is the CSR offset
/// array of length `n_leaves + 1`, and `n_leaves` is the number of distinct
/// partitions.
fn build_leaf_structure(partition_ids: &[u32], n: usize) -> (Vec<u32>, Vec<u32>, usize) {
    let mut sorted: Vec<(u32, u32)> = partition_ids
        .iter()
        .enumerate()
        .map(|(i, &pid)| (pid, i as u32))
        .collect();
    sorted.par_sort_unstable_by_key(|&(pid, _)| pid);

    let leaf_points: Vec<u32> = sorted.iter().map(|&(_, idx)| idx).collect();

    let mut leaf_offsets = vec![0u32];
    for i in 1..n {
        if sorted[i].0 != sorted[i - 1].0 {
            leaf_offsets.push(i as u32);
        }
    }
    leaf_offsets.push(n as u32);

    let n_leaves = leaf_offsets.len() - 1;
    (leaf_points, leaf_offsets, n_leaves)
}

/// Compute per-partition median dot values on CPU.
///
/// Used after each random projection step to determine the split threshold
/// for bisecting each partition.
///
/// ### Params
///
/// * `partition_ids` - Current partition ID per point, length `n`
/// * `dot_values` - Dot product of each point with the projection vector,
///   length `n`
/// * `n_partitions` - Number of active partitions at the current tree level
///
/// ### Returns
///
/// Vector of length `n_partitions` containing the median dot value for each
/// partition. Empty partitions retain `T::zero()`.
fn compute_partition_medians<T: AnnSearchFloat>(
    partition_ids: &[u32],
    dot_values: &[T],
    n_partitions: usize,
) -> Vec<T> {
    // add 50% slack to the expected capacity to accommodate uneven splits
    // and drastically reduce reallocation thrashing.
    let expected_cap = dot_values.len() / n_partitions + (dot_values.len() / n_partitions / 2);
    let mut buckets: Vec<Vec<T>> = vec![Vec::with_capacity(expected_cap); n_partitions];

    for (&pid, &dot) in partition_ids.iter().zip(dot_values.iter()) {
        let p = pid as usize;
        if p < n_partitions {
            buckets[p].push(dot);
        }
    }

    // parallelise the median finding using Rayon - hopefully faster ... ?
    buckets
        .into_par_iter()
        .map(|mut bucket| {
            if bucket.is_empty() {
                T::zero()
            } else {
                let mid = bucket.len() / 2;
                bucket.select_nth_unstable_by(mid, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                bucket[mid]
            }
        })
        .collect()
}

//////////////////
// ForestRouter //
//////////////////

/// Lightweight query-time router reusing GPU forest tree structure.
/// Replaces the Annoy index for beam search entry point selection.
pub struct ForestRouter<T: AnnSearchFloat> {
    /// Per tree, per level: random projection vector [n_trees][max_depth][dim]
    random_vecs: Vec<Vec<Vec<T>>>,
    /// Per tree, per level: median per partition [n_trees][max_depth][variable]
    medians: Vec<Vec<Vec<T>>>,
    /// Per tree: leaves[partition_id] -> point indices [n_trees][2^max_depth][]
    leaves: Vec<Vec<Vec<u32>>>,
    /// Tree depth
    max_depth: usize,
    /// Original (unpadded) dimensionality
    dim: usize,
    /// Number of trees stored for routing
    n_trees: usize,
}

impl<T: AnnSearchFloat> ForestRouter<T> {
    /// Route a query through every stored tree using priority-queue
    /// traversal (same strategy as Annoy) to find entry point candidates.
    ///
    /// Explores multiple leaves per tree by backtracking to the most
    /// promising unexplored branches, ranked by distance to the split
    /// hyperplane.
    ///
    /// ### Params
    ///
    /// * `query` - The query for which to identify the entry points
    ///
    /// ### Returns
    ///
    /// Leaf-co-members
    pub fn find_entry_points(&self, query: &[T], max_candidates: usize) -> Vec<usize> {
        let mut candidates = Vec::new();
        let q = &query[..self.dim];
        let per_tree = (max_candidates / self.n_trees).max(1);

        for t in 0..self.n_trees {
            // Priority queue: (margin to hyperplane, pid, level)
            // Smallest margin = most promising unexplored branch
            let mut pq: BinaryHeap<Reverse<(OrderedFloat<T>, u32, usize)>> = BinaryHeap::new();
            pq.push(Reverse((OrderedFloat(T::zero()), 0u32, 0usize)));

            let mut found = 0usize;

            while let Some(Reverse((_, pid, level))) = pq.pop() {
                if found >= per_tree {
                    break;
                }

                if level >= self.max_depth {
                    // Reached a leaf
                    if let Some(leaf) = self.leaves[t].get(pid as usize) {
                        candidates.extend(leaf.iter().map(|&p| p as usize));
                        found += leaf.len();
                    }
                    continue;
                }

                let dot = T::dot_simd(q, &self.random_vecs[t][level]);
                let median = self.medians[t][level]
                    .get(pid as usize)
                    .copied()
                    .unwrap_or_else(T::zero);
                let margin = if dot <= median {
                    median - dot
                } else {
                    dot - median
                };

                // Go to the preferred side first (margin = 0),
                // push the other side with its actual margin
                let (preferred, other) = if dot <= median {
                    (pid * 2, pid * 2 + 1)
                } else {
                    (pid * 2 + 1, pid * 2)
                };

                pq.push(Reverse((OrderedFloat(T::zero()), preferred, level + 1)));
                pq.push(Reverse((OrderedFloat(margin), other, level + 1)));
            }
        }

        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }
}

////////////////////////
// Main orchestration //
////////////////////////

/// Build the initial kNN graph via GPU random partition forest.
///
/// Tree construction (dot products + partitioning) runs entirely on CPU
/// with rayon parallelism. Only the expensive leaf pairwise distance
/// computation and proposal merge runs on GPU. This eliminates all per-level
/// GPU sync overhead.
///
/// ### Params
///
/// * `vectors_gpu` - GPU-resident vector matrix `[n, dim_padded/LINE_SIZE]`
/// * `norms_gpu` - GPU-resident L2 norms `[n]`; unused when `use_cosine` is
///   false
/// * `graph_idx_gpu` - kNN graph index buffer `[n, k]`; updated in-place
/// * `graph_dist_gpu` - kNN graph distance buffer `[n, k]`; updated in-place
/// * `prop_idx_gpu` - Proposal index scratch buffer `[n, MAX_PROPOSALS]`
/// * `prop_dist_gpu` - Proposal distance scratch buffer `[n, MAX_PROPOSALS]`
/// * `prop_count_gpu` - Atomic proposal counter scratch buffer `[n]`
/// * `update_counter_gpu` - Global update counter used by
///   `merge_proposals` `[1]`
/// * `vectors_flat` - CPU-side flattened vector data `[n * dim]`; used for
///   dot product and median computation during tree construction
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
#[allow(clippy::too_many_arguments)]
pub fn gpu_forest_init<T, R>(
    vectors_gpu: &GpuTensor<R, T>,
    norms_gpu: &GpuTensor<R, T>,
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
) -> ForestRouter<T>
where
    R: Runtime,
    T: AnnSearchFloat + AnnSearchGpuFloat,
{
    let line = LINE_SIZE as usize;
    let dim_vec = dim_padded / line;
    let (grid_n_x, grid_n_y) = grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X));

    let max_leaf_size = compute_max_leaf_size(dim_padded);
    let max_depth = if n <= max_leaf_size {
        0
    } else {
        ((n as f64) / (max_leaf_size as f64)).log2().ceil() as usize
    };

    // How many trees to keep routing data for (query entry points)
    let n_router_trees = n_trees.min(5);

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
    let (dot_grid_x, dot_grid_y) = grid_2d(dot_grid);

    // parallelise the outer tree loop to overlap GPU execution with CPU memory reads
    let all_tree_results: TreeResults<T> = (0..n_trees)
        .into_par_iter()
        .map(|tree_idx| {
            // allocate the GPU buffer inside the parallel closure so each tree
            // has an isolated buffer, preventing thread contention on the client.
            let dot_values_gpu = GpuTensor::<R, T>::empty(vec![n], client);

            let tree_seed =
                (seed as u64).wrapping_add((tree_idx as u64).wrapping_mul(0x9E3779B97F4A7C15u64));
            let save_routing = tree_idx < n_router_trees;

            let mut partition_ids = vec![0u32; n];
            let mut routing_vecs: Option<Vec<Vec<T>>> = if save_routing {
                Some(Vec::with_capacity(max_depth))
            } else {
                None
            };
            let mut routing_medians: Option<Vec<Vec<T>>> = if save_routing {
                Some(Vec::with_capacity(max_depth))
            } else {
                None
            };

            for level in 0..max_depth {
                let level_seed =
                    tree_seed.wrapping_add((level as u64).wrapping_mul(0x517CC1B727220A95u64));
                let mut rng = SmallRng::seed_from_u64(level_seed);

                // Generate and normalise random projection vector
                let mut random_vec = vec![T::zero(); dim];
                for v in random_vec.iter_mut() {
                    *v = T::from_f64(rng.random_range(-1.0..1.0)).unwrap();
                }
                let norm_sq: T = random_vec.iter().map(|x| *x * *x).sum();
                let norm = num_traits::Float::sqrt(norm_sq);
                if norm > T::zero() {
                    for x in random_vec.iter_mut() {
                        *x /= norm;
                    }
                }

                // Pad to dim_padded for the GPU kernel's Line<F> layout
                let mut random_vec_padded = vec![T::zero(); dim_padded];
                random_vec_padded[..dim].copy_from_slice(&random_vec);
                let random_vec_gpu =
                    GpuTensor::<R, T>::from_slice(&random_vec_padded, vec![dim_padded], client);

                // GPU dot products (2.8M threads, ~2ms per launch)
                unsafe {
                    let _ = compute_dot_products::launch_unchecked::<T, R>(
                        client,
                        CubeCount::Static(dot_grid_x, dot_grid_y, 1),
                        CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                        vectors_gpu.clone().into_tensor_arg(line),
                        random_vec_gpu.into_tensor_arg(line),
                        dot_values_gpu.clone().into_tensor_arg(1),
                        ScalarArg { elem: n as u32 },
                        dim_vec,
                    );
                }

                // Read back dot values (~11MB, blocking but overlapped by Rayon)
                let dot_values = dot_values_gpu.clone().read(client);

                // CPU median computation (fast, O(n), parallelized internally)
                let n_partitions = 1usize << level;
                let medians = compute_partition_medians(&partition_ids, &dot_values, n_partitions);

                // CPU partition update (trivial parallel scatter)
                partition_ids
                    .par_iter_mut()
                    .zip(dot_values.par_iter())
                    .for_each(|(pid, &dot)| {
                        let p = *pid as usize;
                        *pid = if dot <= medians[p] {
                            *pid * 2
                        } else {
                            *pid * 2 + 1
                        };
                    });

                if save_routing {
                    routing_vecs.as_mut().unwrap().push(random_vec);
                    routing_medians.as_mut().unwrap().push(medians);
                }
            }

            (partition_ids, routing_vecs, routing_medians)
        })
        .collect();

    if verbose {
        println!("    Tree construction: {:.2?}", cpu_start.elapsed());
    }

    // build forest router
    let mut router_rvecs = Vec::with_capacity(n_router_trees);
    let mut router_medians = Vec::with_capacity(n_router_trees);
    let mut router_leaves = Vec::with_capacity(n_router_trees);

    for (partition_ids, rvecs_opt, medians_opt) in &all_tree_results[..n_router_trees] {
        router_rvecs.push(rvecs_opt.clone().unwrap());
        router_medians.push(medians_opt.clone().unwrap());

        let max_pid = partition_ids
            .iter()
            .fold(0u32, |acc, &x| if x > acc { x } else { acc }) as usize;
        let mut leaves = vec![Vec::new(); max_pid + 1];
        for (i, &pid) in partition_ids.iter().enumerate() {
            leaves[pid as usize].push(i as u32);
        }
        router_leaves.push(leaves);
    }

    let router = ForestRouter {
        random_vecs: router_rvecs,
        medians: router_medians,
        leaves: router_leaves,
        max_depth,
        dim,
        n_trees: n_router_trees,
    };

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
            let (leaf_points, leaf_offsets, n_leaves) =
                build_leaf_structure(&all_tree_results[tree_idx].0, n);

            if n_leaves == 0 {
                continue;
            }

            let base_offset = batch_leaf_points.len() as u32;
            for i in 0..n_leaves {
                batch_leaf_offsets.push(leaf_offsets[i] + base_offset);
            }
            batch_leaf_points.extend_from_slice(&leaf_points);
        }

        batch_leaf_offsets.push(batch_leaf_points.len() as u32);
        let batch_leaves = batch_leaf_offsets.len() - 1;

        if batch_leaves == 0 {
            continue;
        }

        let leaf_points_gpu = GpuTensor::<R, u32>::from_slice(
            &batch_leaf_points,
            vec![batch_leaf_points.len()],
            client,
        );
        let leaf_offsets_gpu = GpuTensor::<R, u32>::from_slice(
            &batch_leaf_offsets,
            vec![batch_leaf_offsets.len()],
            client,
        );

        unsafe {
            let _ = reset_proposals::launch_unchecked::<R>(
                client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                prop_count_gpu.clone().into_tensor_arg(1),
                update_counter_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
            );
        }

        let cubes_x = (batch_leaves as u32).min(65535);
        let cubes_y = (batch_leaves as u32).div_ceil(cubes_x);

        unsafe {
            let _ = leaf_pairwise_proposals::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(cubes_x, cubes_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                vectors_gpu.clone().into_tensor_arg(line),
                norms_gpu.clone().into_tensor_arg(1),
                leaf_points_gpu.into_tensor_arg(1),
                leaf_offsets_gpu.into_tensor_arg(1),
                graph_dist_gpu.clone().into_tensor_arg(1),
                prop_idx_gpu.clone().into_tensor_arg(1),
                prop_dist_gpu.clone().into_tensor_arg(1),
                prop_count_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                ScalarArg {
                    elem: batch_leaves as u32,
                },
                MAX_PROPOSALS as u32,
                use_cosine,
                dim_vec,
                max_leaf_size,
            );
        }

        unsafe {
            let _ = merge_proposals::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                graph_idx_gpu.clone().into_tensor_arg(1),
                graph_dist_gpu.clone().into_tensor_arg(1),
                prop_idx_gpu.clone().into_tensor_arg(1),
                prop_dist_gpu.clone().into_tensor_arg(1),
                prop_count_gpu.clone().into_tensor_arg(1),
                update_counter_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                MAX_PROPOSALS as u32,
            );
        }
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

    router
}

///////////
// Tests //
///////////

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

    // ---------------------------------------------------------------
    // 1. Dot product kernel
    // ---------------------------------------------------------------

    #[test]
    fn test_dot_products_basic() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE as usize;
        let n = 4usize;
        let dim = 4usize;
        let dim_vec = dim / line;

        let data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5,
        ];
        let rvec: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let rvec_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&rvec, vec![dim], &client);
        let dots_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n], &client);

        let grid = (n as u32).div_ceil(WORKGROUP_SIZE_X);
        unsafe {
            let _ = compute_dot_products::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(grid, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                vectors_gpu.into_tensor_arg(line),
                rvec_gpu.into_tensor_arg(line),
                dots_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                dim_vec,
            );
        }

        let dots = dots_gpu.read(&client);
        let expected = [1.0f32, 0.0, 1.0, 0.5];
        for (i, (&got, &exp)) in dots.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-5,
                "dot[{i}]: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_dot_products_dim32() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE as usize;
        let n = 16usize;
        let dim = 32usize;
        let dim_vec = dim / line;

        let data: Vec<f32> = (0..n * dim).map(|idx| (idx / dim + 1) as f32).collect();
        let rvec: Vec<f32> = vec![1.0; dim];

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let rvec_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&rvec, vec![dim], &client);
        let dots_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n], &client);

        let grid = (n as u32).div_ceil(WORKGROUP_SIZE_X);
        unsafe {
            let _ = compute_dot_products::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(grid, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                vectors_gpu.into_tensor_arg(line),
                rvec_gpu.into_tensor_arg(line),
                dots_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                dim_vec,
            );
        }

        let dots = dots_gpu.read(&client);
        for i in 0..n {
            let expected = (i + 1) as f32 * dim as f32;
            assert!(
                (dots[i] - expected).abs() < 1e-2,
                "dot[{i}]: got {}, expected {}",
                dots[i],
                expected
            );
        }
    }

    // ---------------------------------------------------------------
    // 2. Partition kernel
    // ---------------------------------------------------------------

    #[test]
    fn test_partition_basic() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let n = 8usize;
        let pids = vec![0u32; n];
        let dots: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let medians: Vec<f32> = vec![3.5];

        let pid_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&pids, vec![n], &client);
        let dot_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&dots, vec![n], &client);
        let med_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&medians, vec![1], &client);

        let grid = (n as u32).div_ceil(WORKGROUP_SIZE_X);
        unsafe {
            let _ = partition_points::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(grid, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                pid_gpu.clone().into_tensor_arg(1),
                dot_gpu.into_tensor_arg(1),
                med_gpu.into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
            );
        }

        let result = pid_gpu.read(&client);
        for i in 0..4 {
            assert_eq!(result[i], 0, "point {i} should be in partition 0");
        }
        for i in 4..8 {
            assert_eq!(result[i], 1, "point {i} should be in partition 1");
        }
    }

    #[test]
    fn test_partition_multilevel_with_cpu_medians() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let n = 16usize;
        let line = LINE_SIZE as usize;
        let dim = 4usize;
        let dim_vec = dim / line;

        let data: Vec<f32> = (0..n).flat_map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();
        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let dots_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n], &client);
        let grid = (n as u32).div_ceil(WORKGROUP_SIZE_X);

        let mut cpu_pids = vec![0u32; n];
        let pid_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&cpu_pids, vec![n], &client);

        for level in 0..2usize {
            let rvec: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
            let rvec_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&rvec, vec![dim], &client);

            unsafe {
                let _ = compute_dot_products::launch_unchecked::<f32, WgpuRuntime>(
                    &client,
                    CubeCount::Static(grid, 1, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    vectors_gpu.clone().into_tensor_arg(line),
                    rvec_gpu.into_tensor_arg(line),
                    dots_gpu.clone().into_tensor_arg(1),
                    ScalarArg { elem: n as u32 },
                    dim_vec,
                );
            }

            let dots_cpu = dots_gpu.clone().read(&client);
            let n_partitions = 1usize << level;
            let medians = compute_partition_medians(&cpu_pids, &dots_cpu, n_partitions);
            let med_gpu =
                GpuTensor::<WgpuRuntime, f32>::from_slice(&medians, vec![n_partitions], &client);

            unsafe {
                let _ = partition_points::launch_unchecked::<f32, WgpuRuntime>(
                    &client,
                    CubeCount::Static(grid, 1, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    pid_gpu.clone().into_tensor_arg(1),
                    dots_gpu.clone().into_tensor_arg(1),
                    med_gpu.into_tensor_arg(1),
                    ScalarArg { elem: n as u32 },
                );
            }

            // Mirror on CPU
            for i in 0..n {
                let pid = cpu_pids[i] as usize;
                cpu_pids[i] = if dots_cpu[i] <= medians[pid] {
                    cpu_pids[i] * 2
                } else {
                    cpu_pids[i] * 2 + 1
                };
            }
        }

        // Verify GPU and CPU agree
        let gpu_pids = pid_gpu.read(&client);
        assert_eq!(gpu_pids, cpu_pids, "GPU and CPU partition IDs must match");

        let (_, leaf_offsets, n_leaves) = build_leaf_structure(&cpu_pids, n);
        assert_eq!(n_leaves, 4, "2 levels should produce 4 leaves");

        for i in 0..n_leaves {
            let size = leaf_offsets[i + 1] - leaf_offsets[i];
            assert!(
                (2..=8).contains(&size),
                "Leaf {i} has {size} points (expected ~4)"
            );
        }
    }

    // ---------------------------------------------------------------
    // 3. Leaf shared memory roundtrip
    // ---------------------------------------------------------------

    #[cube(launch_unchecked)]
    fn debug_leaf_shared_roundtrip<F: AnnSearchGpuFloat>(
        vectors: &Tensor<Line<F>>,
        leaf_points: &Tensor<u32>,
        leaf_offsets: &Tensor<u32>,
        out_vecs: &mut Tensor<F>,
        n: u32,
        #[comptime] dim_lines: usize,
        #[comptime] max_leaf_size: usize,
    ) {
        let leaf_idx = CUBE_POS_X;
        let tx = UNIT_POS_X;
        let dim_scalars = dim_lines * 4usize;

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
            let line_idx = s_idx / 4usize;
            let lane = s_idx % 4usize;
            let pid = shared_pids[n_idx];

            if pid < n {
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
        let line = LINE_SIZE as usize;
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

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let lp_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_points, vec![3], &client);
        let lo_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_offsets, vec![2], &client);
        let out_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; MAX_LEAF_SIZE * dim],
            vec![MAX_LEAF_SIZE * dim],
            &client,
        );

        unsafe {
            let _ = debug_leaf_shared_roundtrip::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                vectors_gpu.into_tensor_arg(line),
                lp_gpu.into_tensor_arg(1),
                lo_gpu.into_tensor_arg(1),
                out_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                dim_vec,
                MAX_LEAF_SIZE,
            );
        }

        let result = out_gpu.read(&client);
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
        let line = LINE_SIZE as usize;
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

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let lp_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_points, vec![8], &client);
        let lo_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_offsets, vec![2], &client);
        let out_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; MAX_LEAF_SIZE * dim],
            vec![MAX_LEAF_SIZE * dim],
            &client,
        );

        unsafe {
            let _ = debug_leaf_shared_roundtrip::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                vectors_gpu.into_tensor_arg(line),
                lp_gpu.into_tensor_arg(1),
                lo_gpu.into_tensor_arg(1),
                out_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                dim_vec,
                MAX_LEAF_SIZE,
            );
        }

        let result = out_gpu.read(&client);
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

    // ---------------------------------------------------------------
    // 4. Leaf pairwise proposals
    // ---------------------------------------------------------------

    #[test]
    fn test_leaf_pairwise_small_euclidean() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE as usize;
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

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32], vec![1], &client);
        let lp_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_points, vec![4], &client);
        let lo_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_offsets, vec![2], &client);
        let gdist_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&graph_dist, vec![n, build_k], &client);
        let prop_idx_gpu = GpuTensor::<WgpuRuntime, u32>::empty(vec![n, MAX_PROPOSALS], &client);
        let prop_dist_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, MAX_PROPOSALS], &client);
        let prop_count_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client);

        unsafe {
            let _ = leaf_pairwise_proposals::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                vectors_gpu.into_tensor_arg(line),
                norms_gpu.into_tensor_arg(1),
                lp_gpu.into_tensor_arg(1),
                lo_gpu.into_tensor_arg(1),
                gdist_gpu.into_tensor_arg(1),
                prop_idx_gpu.clone().into_tensor_arg(1),
                prop_dist_gpu.clone().into_tensor_arg(1),
                prop_count_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                ScalarArg { elem: 1u32 },
                MAX_PROPOSALS as u32,
                false,
                dim_vec,
                MAX_LEAF_SIZE,
            );
        }

        let p_idx = prop_idx_gpu.read(&client);
        let p_dist = prop_dist_gpu.read(&client);
        let p_count = prop_count_gpu.read(&client);

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
        let line = LINE_SIZE as usize;
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

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&norms, vec![n], &client);
        let lp_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_points, vec![4], &client);
        let lo_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_offsets, vec![2], &client);
        let gdist_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&graph_dist, vec![n, build_k], &client);
        let prop_idx_gpu = GpuTensor::<WgpuRuntime, u32>::empty(vec![n, MAX_PROPOSALS], &client);
        let prop_dist_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, MAX_PROPOSALS], &client);
        let prop_count_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client);

        unsafe {
            let _ = leaf_pairwise_proposals::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                vectors_gpu.into_tensor_arg(line),
                norms_gpu.into_tensor_arg(1),
                lp_gpu.into_tensor_arg(1),
                lo_gpu.into_tensor_arg(1),
                gdist_gpu.into_tensor_arg(1),
                prop_idx_gpu.clone().into_tensor_arg(1),
                prop_dist_gpu.clone().into_tensor_arg(1),
                prop_count_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                ScalarArg { elem: 1u32 },
                MAX_PROPOSALS as u32,
                true,
                dim_vec,
                MAX_LEAF_SIZE,
            );
        }

        let p_idx = prop_idx_gpu.read(&client);
        let p_dist = prop_dist_gpu.read(&client);
        let p_count = prop_count_gpu.read(&client);

        let mut any_negative = false;
        let mut any_mismatch = false;
        for node in 0..n {
            let count = (p_count[node] as usize).min(MAX_PROPOSALS);
            for p in 0..count {
                let cand = p_idx[node * MAX_PROPOSALS + p] as usize;
                let gpu_dist = p_dist[node * MAX_PROPOSALS + p];
                let dot: f32 = data[node * dim..(node + 1) * dim]
                    .iter()
                    .zip(&data[cand * dim..(cand + 1) * dim])
                    .map(|(a, b)| a * b)
                    .sum();
                let cpu_dist = 1.0 - dot / (norms[node] * norms[cand]);
                if gpu_dist < -1e-6 {
                    any_negative = true;
                }
                if (gpu_dist - cpu_dist).abs() > 1e-4 {
                    any_mismatch = true;
                }
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
        let line = LINE_SIZE as usize;
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

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32], vec![1], &client);
        let lp_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_points, vec![16], &client);
        let lo_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&leaf_offsets, vec![2], &client);
        let gdist_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&graph_dist, vec![n, build_k], &client);
        let prop_idx_gpu = GpuTensor::<WgpuRuntime, u32>::empty(vec![n, MAX_PROPOSALS], &client);
        let prop_dist_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, MAX_PROPOSALS], &client);
        let prop_count_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client);

        unsafe {
            let _ = leaf_pairwise_proposals::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                vectors_gpu.into_tensor_arg(line),
                norms_gpu.into_tensor_arg(1),
                lp_gpu.into_tensor_arg(1),
                lo_gpu.into_tensor_arg(1),
                gdist_gpu.into_tensor_arg(1),
                prop_idx_gpu.clone().into_tensor_arg(1),
                prop_dist_gpu.clone().into_tensor_arg(1),
                prop_count_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                ScalarArg { elem: 1u32 },
                MAX_PROPOSALS as u32,
                false,
                dim_vec,
                MAX_LEAF_SIZE,
            );
        }

        let p_idx = prop_idx_gpu.read(&client);
        let p_dist = prop_dist_gpu.read(&client);
        let p_count = prop_count_gpu.read(&client);

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

    // ---------------------------------------------------------------
    // 5. Full forest init quality
    // ---------------------------------------------------------------

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
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim_padded], &client);
        let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32], vec![1], &client);
        let graph_idx_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
            &vec![0x7FFFFFFFu32; n * build_k],
            vec![n, build_k],
            &client,
        );
        let graph_dist_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![f32::MAX; n * build_k],
            vec![n, build_k],
            &client,
        );
        let prop_idx_gpu = GpuTensor::<WgpuRuntime, u32>::empty(vec![n, MAX_PROPOSALS], &client);
        let prop_dist_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, MAX_PROPOSALS], &client);
        let prop_count_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client);
        let update_counter_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32], vec![1], &client);

        gpu_forest_init(
            &vectors_gpu,
            &norms_gpu,
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

        let result_idx = graph_idx_gpu.read(&client);
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

    // ---------------------------------------------------------------
    // 6. CPU helpers
    // ---------------------------------------------------------------

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

    // ---------------------------------------------------------------
    // 7. mark_all_new kernel
    // ---------------------------------------------------------------

    #[test]
    fn test_mark_all_new() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);

        let sentinel = 0x7FFFFFFFu32;
        let data = vec![5u32, 10 | (1u32 << 31), sentinel, 42u32];
        let gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&data, vec![4], &client);

        unsafe {
            let _ = mark_all_new::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: 4u32 },
            );
        }

        let result = gpu.read(&client);
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

    // ---------------------------------------------------------------
    // 8. CPU-GPU partition mirror consistency
    // ---------------------------------------------------------------

    #[test]
    fn test_cpu_gpu_partition_mirror() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };
        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE as usize;
        let n = 64usize;
        let dim = 8usize;
        let dim_vec = dim / line;

        // Random-ish data
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 7 + 3) % 100) as f32 / 10.0)
            .collect();
        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let dots_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n], &client);
        let grid = (n as u32).div_ceil(WORKGROUP_SIZE_X);

        let mut cpu_pids = vec![0u32; n];
        let pid_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&cpu_pids, vec![n], &client);

        // Run 4 levels with known random vecs
        for level in 0..4usize {
            let rvec: Vec<f32> = (0..dim)
                .map(|j| ((level * 3 + j * 5 + 1) % 11) as f32 / 5.0 - 1.0)
                .collect();
            let rvec_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&rvec, vec![dim], &client);

            unsafe {
                let _ = compute_dot_products::launch_unchecked::<f32, WgpuRuntime>(
                    &client,
                    CubeCount::Static(grid, 1, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    vectors_gpu.clone().into_tensor_arg(line),
                    rvec_gpu.into_tensor_arg(line),
                    dots_gpu.clone().into_tensor_arg(1),
                    ScalarArg { elem: n as u32 },
                    dim_vec,
                );
            }

            let dots_cpu = dots_gpu.clone().read(&client);
            let n_partitions = 1usize << level;
            let medians = compute_partition_medians(&cpu_pids, &dots_cpu, n_partitions);
            let med_gpu =
                GpuTensor::<WgpuRuntime, f32>::from_slice(&medians, vec![n_partitions], &client);

            unsafe {
                let _ = partition_points::launch_unchecked::<f32, WgpuRuntime>(
                    &client,
                    CubeCount::Static(grid, 1, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    pid_gpu.clone().into_tensor_arg(1),
                    dots_gpu.clone().into_tensor_arg(1),
                    med_gpu.into_tensor_arg(1),
                    ScalarArg { elem: n as u32 },
                );
            }

            for i in 0..n {
                let pid = cpu_pids[i] as usize;
                cpu_pids[i] = if dots_cpu[i] <= medians[pid] {
                    cpu_pids[i] * 2
                } else {
                    cpu_pids[i] * 2 + 1
                };
            }
        }

        let gpu_pids = pid_gpu.read(&client);
        assert_eq!(
            gpu_pids, cpu_pids,
            "GPU and CPU partitions diverged after 4 levels"
        );

        // Should have roughly 16 partitions (2^4) with ~4 points each
        let unique: std::collections::HashSet<u32> = cpu_pids.iter().copied().collect();
        println!(
            "{} unique partitions from 4 levels of 64 points",
            unique.len()
        );
        assert!(unique.len() >= 8 && unique.len() <= 16);
    }
}
