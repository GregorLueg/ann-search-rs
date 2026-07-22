//! GPU-accelerated NNDescent kNN graph construction via CubeCL.
//!
//! All vector data remains GPU-resident throughout construction. The host
//! loop only downloads a single u32 convergence counter per iteration.

#![allow(missing_docs)] // complains about cubecl macros...

use cubecl::frontend::{Atomic, CubePrimitive, Float, SharedMemory};
use cubecl::prelude::*;
use faer::{MatRef, RowRef};
use fixedbitset::FixedBitSet;
use rayon::prelude::*;
use std::time::Instant;
use std::{cell::RefCell, cmp::Reverse, collections::BinaryHeap};
use thousands::*;

use crate::cpu::nndescent::*;
use crate::cpu::vamana::compute_medoid;
use crate::gpu::cagra_gpu_search::*;
use crate::gpu::forest_gpu::*;
use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::prelude::*;
use crate::utils::nndescent_utils::SENTINEL_PID;

///////////
// Const //
///////////

/// Max proposals per node per iteration. Overflow is silently dropped.
pub const MAX_PROPOSALS: usize = 128;
/// Default maximum number of NNDescent iterations
const DEFAULT_MAX_ITERS: usize = 15;
/// Default convergence threshold (fraction of k*n edges updated)
const DEFAULT_DELTA: f32 = 0.001;
/// Default sampling rate for the local join
const DEFAULT_RHO: f32 = 0.5;

////////////////////
// Kernel helpers //
////////////////////

/// Simple xorshift32 PRNG for deterministic per-thread random decisions.
///
/// ### Params
///
/// * `state` - Current PRNG state
///
/// ### Returns
///
/// Next PRNG state
#[cube]
fn xorshift(state: u32) -> u32 {
    let mut x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

/// Deterministic hash for per-entry rho sampling decisions.
///
/// Same (node, entry, seed) triple always produces the same result, so no local
/// storage is needed for the participation decision.
///
/// ### Params
///
/// * `node` - Source node index
/// * `entry` - Neighbour slot index within the node's adjacency list
/// * `seed` - Per-iteration seed
///
/// ### Returns
///
/// A pseudo-random u32 for use in sampling decisions
#[cube]
fn entry_hash(node: u32, entry: u32, seed: u32) -> u32 {
    xorshift(node ^ (entry * 2654435769u32) ^ seed)
}

/// Squared Euclidean distance between vectors at indices `a` and `b`.
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, vectorised along the feature
///   dimension
/// * `a` - Row index of the first vector
/// * `b` - Row index of the second vector
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime)
///
/// ### Returns
///
/// Squared Euclidean distance between the two vectors
#[cube]
fn dist_sq_euclidean<F: Float + CubePrimitive, N: Size>(
    vectors: &Tensor<Vector<F, N>>,
    a: u32,
    b: u32,
    #[comptime] dim_lines: usize,
) -> F {
    let lanes = LINE_SIZE;
    let off_a = a as usize * dim_lines;
    let off_b = b as usize * dim_lines;
    let mut sum = F::new(0.0_f32);
    for i in 0..dim_lines {
        let va = vectors[off_a + i];
        let vb = vectors[off_b + i];
        let diff = va - vb;
        let sq = diff * diff;
        #[unroll]
        for lane in 0..lanes {
            sum += sq[lane];
        }
    }
    sum
}

/// Cosine distance (1 - cosine similarity) between vectors at `a` and `b`.
///
/// Requires pre-computed L2 norms.
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, vectorised along the feature
///   dimension
/// * `norms` - Pre-computed L2 norms, one per row
/// * `a` - Row index of the first vector
/// * `b` - Row index of the second vector
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime)
///
/// ### Returns
///
/// Cosine distance in the range [0, 2]
#[cube]
fn dist_cosine<F: Float, N: Size>(
    vectors: &Tensor<Vector<F, N>>,
    norms: &Tensor<F>,
    a: u32,
    b: u32,
    #[comptime] dim_lines: usize,
) -> F {
    let lanes = LINE_SIZE;
    let off_a = a as usize * dim_lines;
    let off_b = b as usize * dim_lines;
    let mut dot = F::new(0.0_f32);
    for i in 0..dim_lines {
        let va = vectors[off_a + i];
        let vb = vectors[off_b + i];
        let prod = va * vb;
        #[unroll]
        for lane in 0..lanes {
            dot += prod[lane];
        }
    }
    F::new(1.0_f32) - dot / (norms[a as usize] * norms[b as usize])
}

/////////////
// Kernels //
/////////////

///////////////
// NNDescent //
///////////////

/// Initialise the kNN graph with random neighbours.
///
/// One thread per node. Generates k random neighbours, computes distances,
/// and maintains a sorted (ascending by distance) list via insertion.
/// All entries are flagged as new (MSB set).
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, line-vectorised along the feature
///   dimension
/// * `norms` - Pre-computed L2 norms (ignored when `use_cosine` is false)
/// * `n_pts` - Number of vectors
/// * `seed` - Random seed for neighbour generation
/// * `use_cosine` - Whether to use cosine distance instead of squared Euclidean
/// * `dim_lines` - Number of `Line<F>` elements per vector row (comptime)
///
/// ### Returns
///
/// Writes an initialised sorted kNN graph into `graph_idx` and `graph_dist`.
/// All entries are flagged as new (MSB set).
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> node index
#[cube(launch_unchecked)]
fn init_random_graph<F: Float, N: Size>(
    vectors: &Tensor<Vector<F, N>>,
    norms: &Tensor<F>,
    graph_idx: &mut Tensor<u32>,
    graph_dist: &mut Tensor<F>,
    n_pts: u32,
    seed: u32,
    #[comptime] use_cosine: bool,
    #[comptime] dim_lines: usize,
) {
    let node = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    if node >= n_pts {
        terminate!();
    }

    let k = graph_idx.shape(1);
    let is_new_bit = 1u32 << 31;
    let base = node as usize * k;

    let mut rng = xorshift(node ^ seed ^ 0xDEADBEEFu32);

    for slot in 0..k {
        rng = xorshift(rng);
        let mut pid = rng % n_pts;
        if pid == node {
            pid = (pid + 1u32) % n_pts;
        }

        let dist = if use_cosine {
            dist_cosine(vectors, norms, node, pid, dim_lines)
        } else {
            dist_sq_euclidean(vectors, node, pid, dim_lines)
        };

        // sorted insertion into slots [0..slot].
        // find the first position where dist < existing, scanning left to right.
        let mut insert_pos = slot;
        for j in 0..slot {
            if dist < graph_dist[base + j] && insert_pos == slot {
                insert_pos = j;
            }
        }

        // shift right from insert_pos to slot-1
        for j in 0..slot {
            let src = slot - 1 - j;
            let dst = slot - j;
            if src >= insert_pos {
                graph_idx[base + dst] = graph_idx[base + src];
                graph_dist[base + dst] = graph_dist[base + src];
            }
        }

        graph_idx[base + insert_pos] = pid | is_new_bit;
        graph_dist[base + insert_pos] = dist;
    }
}

/// Zero out proposal counts and the global update counter.
///
/// One thread per node zeroes its entry in `prop_count`. Thread 0
/// additionally resets `update_counter[0]` to zero.
///
/// ### Params
///
/// * `prop_count` - Per-node proposal counter to reset `[n]`
/// * `update_counter` - Global update accumulator to reset `[1]`
/// * `n` - Number of nodes
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> node index
#[cube(launch_unchecked)]
pub fn reset_proposals(prop_count: &mut Tensor<u32>, update_counter: &mut Tensor<u32>, n: u32) {
    let idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    if idx < n {
        prop_count[idx as usize] = 0u32;
    }
    if idx == 0u32 {
        update_counter[0usize] = 0u32;
    }
}

/// Scatter forward edges into a reverse edge buffer.
///
/// Ensures symmetric information flow during the local join phase by
/// creating reverse (target -> source) copies of each forward edge.
///
/// ### Params
///
/// * `graph_idx` - Current kNN graph indices (with IS_NEW flag in MSB)
/// * `reverse_idx` - Output reverse edge buffer, row-major `[n, build_k]`
/// * `reverse_count` - Atomic per-node counter for reverse edge slots
/// * `n` - Number of nodes
/// * `build_k` - Degree of the build graph (comptime)
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> node index
#[cube(launch_unchecked)]
pub fn build_reverse_candidates(
    graph_idx: &Tensor<u32>,
    reverse_idx: &mut Tensor<u32>,
    reverse_count: &Tensor<Atomic<u32>>,
    n: u32,
    #[comptime] build_k: u32,
) {
    let node = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    if node >= n {
        terminate!();
    }

    let pid_mask = 0x7FFFFFFFu32;
    let base = node as usize * build_k as usize;

    let mut i = 0usize;
    while i < build_k as usize {
        let target_raw = graph_idx[base + i];
        let target = target_raw & pid_mask;

        // Sentinels and out-of-bounds check
        if target < n && target != node {
            let pos = reverse_count[target as usize].fetch_add(1u32);
            if pos < build_k {
                let rev_base = target as usize * build_k as usize;
                // Preserve the "is_new" flag from the forward edge
                let is_new_bit = target_raw & (1u32 << 31);
                reverse_idx[rev_base + pos as usize] = node | is_new_bit;
            }
        }
        i += 1usize;
    }
}

/// Core NNDescent local join kernel.
///
/// For each node, loads forward and reverse candidates into shared memory,
/// pre-filters by rho sampling and compacts the candidate list before loading
/// vectors, then evaluates all (new, new) and (new, old) pairs and emits
/// proposals for both endpoints.
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, line-vectorised along the feature dimension
/// * `norms` - Pre-computed L2 norms (ignored when `use_cosine` is false)
/// * `graph_idx` - Current kNN graph indices (with IS_NEW flag in MSB)
/// * `graph_dist` - Current kNN graph distances (used for threshold filtering)
/// * `reverse_idx` - Reverse edge buffer from `build_reverse_candidates`
/// * `reverse_count` - Number of valid reverse edges per node
/// * `prop_idx` - Output proposal indices, row-major `[n, max_proposals]`
/// * `prop_dist` - Output proposal distances, matching layout to `prop_idx`
/// * `prop_count` - Atomic per-node counter for proposal slots
/// * `n` - Number of nodes
/// * `rho_thresh` - Sampling threshold (scaled to 16-bit range)
/// * `iter_seed` - Per-iteration seed for deterministic sampling
/// * `max_proposals` - Proposal buffer capacity per node (comptime)
/// * `use_cosine` - Whether to use cosine distance (comptime)
/// * `dim_lines` - Number of `Line<F>` elements per vector row (comptime)
/// * `build_k` - Degree of the build graph (comptime)
///
/// ### Grid mapping
///
/// * One workgroup (cube) per node
#[cube(launch_unchecked)]
pub fn local_join_shared<F: Float, N: Size>(
    vectors: &Tensor<Vector<F, N>>,
    norms: &Tensor<F>,
    graph_idx: &Tensor<u32>,
    graph_dist: &Tensor<F>,
    reverse_idx: &Tensor<u32>,
    reverse_count: &Tensor<u32>,
    prop_idx: &mut Tensor<u32>,
    prop_dist: &mut Tensor<F>,
    prop_count: &Tensor<Atomic<u32>>,
    n_pts: u32,
    rho_thresh: u32,
    iter_seed: u32,
    #[comptime] max_proposals: u32,
    #[comptime] use_cosine: bool,
    #[comptime] dim_lines: usize,
    #[comptime] build_k: usize,
) {
    let node = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if node >= n_pts {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let k = graph_idx.shape(1usize) as u32;
    let pid_mask = 0x7FFFFFFFu32;

    let max_cands_comp = build_k * 2usize;
    let dim_scalars = dim_lines * 4usize;

    let mut shared_vecs = SharedMemory::<F>::new(max_cands_comp * dim_scalars);
    let mut shared_pids = SharedMemory::<u32>::new(max_cands_comp);
    let mut shared_is_new = SharedMemory::<u32>::new(max_cands_comp);
    let mut shared_norms = SharedMemory::<F>::new(max_cands_comp);

    // Compacted candidate count and whether any new candidates exist
    let mut shared_compact = SharedMemory::<u32>::new(2usize);

    let mut shared_rev_count = SharedMemory::<u32>::new(1usize);
    if tx == 0u32 {
        let rc = reverse_count[node as usize];
        let mut clamped = rc;
        if rc > k {
            clamped = k;
        }
        shared_rev_count[0usize] = clamped;
    }
    sync_cube();

    let rev_k = shared_rev_count[0usize];
    let raw_total = k + rev_k;

    let mut i_load = tx;
    while i_load < raw_total {
        #[allow(unused_assignments)]
        let mut entry = 0u32;
        if i_load < k {
            entry = graph_idx[(node * k + i_load) as usize];
        } else {
            entry = reverse_idx[(node * k + i_load - k) as usize];
        }

        shared_pids[i_load as usize] = entry & pid_mask;
        shared_is_new[i_load as usize] = entry >> 31;
        i_load += WORKGROUP_SIZE_X;
    }
    sync_cube();

    if tx == 0u32 {
        let mut write = 0u32;
        let mut has_new = 0u32;
        let mut read = 0u32;
        while read < raw_total {
            let hash = entry_hash(node, read, iter_seed);
            if (hash & 0xFFFFu32) < rho_thresh {
                shared_pids[write as usize] = shared_pids[read as usize];
                shared_is_new[write as usize] = shared_is_new[read as usize];
                if shared_is_new[read as usize] != 0u32 {
                    has_new = 1u32;
                }
                write += 1u32;
            }
            read += 1u32;
        }
        shared_compact[0usize] = write;
        shared_compact[1usize] = has_new;
    }
    sync_cube();

    let total_cands = shared_compact[0usize];
    let has_new = shared_compact[1usize];

    // early exit: fewer than 2 candidates or no new candidates at all
    if total_cands < 2u32 || has_new == 0u32 {
        terminate!();
    }

    if use_cosine {
        let mut i_norm = tx;
        while i_norm < total_cands {
            shared_norms[i_norm as usize] = norms[shared_pids[i_norm as usize] as usize];
            i_norm += WORKGROUP_SIZE_X;
        }
        sync_cube();
    }

    let total_scalars = total_cands as usize * dim_scalars;
    let mut idx_load = tx as usize;
    while idx_load < total_scalars {
        let n_idx = idx_load / dim_scalars;
        let s_idx = idx_load % dim_scalars;
        let line_idx = s_idx / 4usize;
        let lane = s_idx % 4usize;
        let pid = shared_pids[n_idx];

        if pid < n_pts {
            let vec_offset = pid as usize * dim_lines + line_idx;
            let line_val = vectors[vec_offset];
            shared_vecs[idx_load] = line_val[lane];
        }
        idx_load += WORKGROUP_SIZE_X as usize;
    }
    sync_cube();

    let num_pairs = (total_cands * (total_cands - 1u32)) / 2u32;
    let mut pair_idx = tx as usize;

    while pair_idx < num_pairs as usize {
        let mut rem = pair_idx;
        let mut i = 0usize;
        let mut step = total_cands as usize - 1usize;

        while rem >= step {
            rem -= step;
            i += 1usize;
            step = total_cands as usize - 1usize - i;
        }
        let j = i + 1usize + rem;

        let is_new_i = shared_is_new[i] != 0u32;
        let is_new_j = shared_is_new[j] != 0u32;
        let pid_i = shared_pids[i];
        let pid_j = shared_pids[j];

        if (is_new_i || is_new_j) && pid_i != pid_j {
            let mut sum = F::new(0.0_f32);
            let mut s = 0usize;
            while s < dim_scalars {
                let va = shared_vecs[i * dim_scalars + s];
                let vb = shared_vecs[j * dim_scalars + s];

                if use_cosine {
                    sum += va * vb;
                } else {
                    let diff = va - vb;
                    sum += diff * diff;
                }
                s += 1usize;
            }

            let dist = if use_cosine {
                F::new(1.0_f32) - (sum / (shared_norms[i] * shared_norms[j]))
            } else {
                sum
            };

            let thresh_i = graph_dist[pid_i as usize * k as usize + k as usize - 1usize];
            let thresh_j = graph_dist[pid_j as usize * k as usize + k as usize - 1usize];

            if dist < thresh_i {
                let slot_i = prop_count[pid_i as usize].fetch_add(1u32);
                if slot_i < max_proposals {
                    let off = pid_i as usize * max_proposals as usize + slot_i as usize;
                    prop_idx[off] = pid_j;
                    prop_dist[off] = dist;
                }
            }

            if dist < thresh_j {
                let slot_j = prop_count[pid_j as usize].fetch_add(1u32);
                if slot_j < max_proposals {
                    let off = pid_j as usize * max_proposals as usize + slot_j as usize;
                    prop_idx[off] = pid_i;
                    prop_dist[off] = dist;
                }
            }
        }
        pair_idx += WORKGROUP_SIZE_X as usize;
    }
}

/// Merge proposals into the sorted kNN graph.
///
/// One thread per node. For each node:
///
/// 1. Clears the IS_NEW flag on all existing neighbours (marks old).
/// 2. Iterates over received proposals (up to MAX_PROPOSALS).
/// 3. Skips duplicates already in the graph.
/// 4. Inserts improvements into the sorted list, flagged as new.
/// 5. Atomically accumulates the total improvement count.
///
/// ### Params
///
/// * `prop_idx` - Proposal candidate indices, row-major with `max_proposals`
///   columns
/// * `prop_dist` - Proposal candidate distances, matching layout to `prop_idx`
/// * `prop_count` - Number of valid proposals received per node
/// * `n` - Number of nodes
/// * `max_proposals` - Proposal buffer capacity per node (comptime)
///
/// ### Returns
///
/// Updates `graph_idx` and `graph_dist` in place with any improvements, flags
/// inserted entries as new, and accumulates the total improvement count into
/// `update_counter[0]`.
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> node index
#[cube(launch_unchecked)]
pub fn merge_proposals<F: Float>(
    graph_idx: &mut Tensor<u32>,
    graph_dist: &mut Tensor<F>,
    prop_idx: &Tensor<u32>,
    prop_dist: &Tensor<F>,
    prop_count: &Tensor<u32>,
    update_counter: &Tensor<Atomic<u32>>,
    n: u32,
    #[comptime] max_proposals: u32,
) {
    let node = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    if node >= n {
        terminate!();
    }

    let k = graph_idx.shape(1);
    let pid_mask = 0x7FFFFFFFu32;
    let is_new_bit = 1u32 << 31;
    let base = node as usize * k;

    // clear new flags on all existing entries
    for j in 0..k {
        graph_idx[base + j] = graph_idx[base + j] & pid_mask;
    }

    // read how many proposals this node received (capped at max_proposals)
    let raw_count = prop_count[node as usize];
    let prop_base = node as usize * max_proposals as usize;
    let mut improvements = 0u32;

    // Fixed loop bound (comptime); guard with runtime count
    for p in 0..max_proposals {
        if p < raw_count {
            let candidate = prop_idx[prop_base + p as usize];
            let dist = prop_dist[prop_base + p as usize];

            // Only process if better than current worst
            if dist < graph_dist[base + k - 1] {
                // Check for duplicates
                let mut exists: u32 = 0u32;
                for j in 0..k {
                    if (graph_idx[base + j] & pid_mask) == candidate {
                        exists = 1u32;
                    }
                }

                // Reject duplicates and self-loops
                if exists == 0u32 && candidate != node {
                    // Find insertion point (first slot where dist < current)
                    let mut insert_pos = k - 1;
                    for j in 0..k {
                        if dist < graph_dist[base + j] && insert_pos == k - 1 {
                            insert_pos = j;
                        }
                    }

                    // Shift right from insert_pos to k-2
                    for j in 0..k - 1 {
                        let src = k - 2 - j;
                        let dst = k - 1 - j;
                        if src >= insert_pos {
                            graph_idx[base + dst] = graph_idx[base + src];
                            graph_dist[base + dst] = graph_dist[base + src];
                        }
                    }

                    // Insert with new flag
                    graph_idx[base + insert_pos] = candidate | is_new_bit;
                    graph_dist[base + insert_pos] = dist;
                    improvements += 1u32;
                }
            }
        }
    }

    if improvements > 0u32 {
        update_counter[0usize].fetch_add(improvements);
    }
}

/// 2-hop refinement kernel.
///
/// Runs after NNDescent convergence. For each node, evaluates the k^2
/// second-degree neighbours. Filters aggressively against duplicates and
/// the current worst distance before pushing to the proposal buffer.
/// Overflow beyond `max_proposals` is handled via reservoir sampling.
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, line-vectorised along the feature
///   dimension
/// * `norms` - Pre-computed L2 norms (ignored when `use_cosine` is false)
/// * `graph_idx` - Current kNN graph indices (with IS_NEW flag in MSB)
/// * `graph_dist` - Current kNN graph distances
/// * `prop_idx` - Output proposal indices, row-major `[n, max_proposals]`
/// * `prop_dist` - Output proposal distances, matching layout to `prop_idx`
/// * `prop_count` - Atomic per-node counter for proposal slots
/// * `n` - Number of nodes
/// * `max_proposals` - Proposal buffer capacity per node (comptime)
/// * `use_cosine` - Whether to use cosine distance (comptime)
/// * `dim_lines` - Number of `Line<F>` elements per vector row (comptime)
///
/// ### Returns
///
/// Emits improvement proposals into `prop_idx` and `prop_dist` for subsequent
/// merging via `merge_proposals`. Does not modify the graph directly.
///
/// ### Grid mapping
///
/// * One workgroup (cube) per node
#[cube(launch_unchecked)]
pub fn two_hop_refinement<F: Float, N: Size>(
    vectors: &Tensor<Vector<F, N>>,
    norms: &Tensor<F>,
    graph_idx: &Tensor<u32>,
    graph_dist: &Tensor<F>,
    prop_idx: &mut Tensor<u32>,
    prop_dist: &mut Tensor<F>,
    prop_count: &Tensor<Atomic<u32>>,
    n_pts: u32,
    #[comptime] max_proposals: u32,
    #[comptime] use_cosine: bool,
    #[comptime] dim_lines: usize,
) {
    let node = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if node >= n_pts {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let k = graph_idx.shape(1usize);
    let pid_mask = 0x7FFFFFFFu32;
    let graph_base = node as usize * k;
    let dim_scalars = dim_lines * 4usize;

    // Load source vector into shared memory as scalars
    let mut shared_source = SharedMemory::<F>::new(dim_scalars);
    let mut shared_worst_dist = SharedMemory::<F>::new(1usize);

    let mut idx_load = tx as usize;
    while idx_load < dim_scalars {
        let line_idx = idx_load / 4usize;
        let lane = idx_load % 4usize;
        let vec_offset = node as usize * dim_lines + line_idx;
        let line_val = vectors[vec_offset];
        shared_source[idx_load] = line_val[lane];
        idx_load += WORKGROUP_SIZE_X as usize;
    }

    if tx == 0u32 {
        shared_worst_dist[0usize] = graph_dist[graph_base + k - 1usize];
    }
    sync_cube();

    let worst_dist = shared_worst_dist[0usize];
    let num_candidates = k * k;
    let mut cand_idx = tx as usize;

    while cand_idx < num_candidates {
        let n1_idx = cand_idx / k;
        let n2_idx = cand_idx % k;

        let n1_raw = graph_idx[graph_base + n1_idx];
        let n1_pid = n1_raw & pid_mask;

        if n1_pid < n_pts {
            let n2_raw = graph_idx[n1_pid as usize * k + n2_idx];
            let cand_pid = n2_raw & pid_mask;

            if cand_pid < n_pts && cand_pid != node {
                let mut is_dup: bool = false;
                let mut scan_idx = 0usize;
                while scan_idx < k {
                    if (graph_idx[graph_base + scan_idx] & pid_mask) == cand_pid {
                        is_dup = true;
                    }
                    scan_idx += 1usize;
                }

                if !is_dup {
                    let mut sum = F::new(0.0_f32);
                    let mut s = 0usize;
                    while s < dim_scalars {
                        let va = shared_source[s];
                        let line_idx = s / 4usize;
                        let lane = s % 4usize;
                        let line_val = vectors[cand_pid as usize * dim_lines + line_idx];
                        let vb = line_val[lane];

                        if use_cosine {
                            sum += va * vb;
                        } else {
                            let diff = va - vb;
                            sum += diff * diff;
                        }
                        s += 1usize;
                    }

                    let dist = if use_cosine {
                        F::new(1.0_f32) - (sum / (norms[node as usize] * norms[cand_pid as usize]))
                    } else {
                        sum
                    };

                    if dist < worst_dist {
                        let slot = prop_count[node as usize].fetch_add(1u32);
                        if slot < max_proposals {
                            let off = node as usize * max_proposals as usize + slot as usize;
                            prop_idx[off] = cand_pid;
                            prop_dist[off] = dist;
                        } else {
                            let rand_val = xorshift(node ^ slot ^ cand_pid) % (slot + 1u32);
                            if rand_val < max_proposals {
                                let off =
                                    node as usize * max_proposals as usize + rand_val as usize;
                                prop_idx[off] = cand_pid;
                                prop_dist[off] = dist;
                            }
                        }
                    }
                }
            }
        }
        cand_idx += WORKGROUP_SIZE_X as usize;
    }
}

///////////
// CAGRA //
///////////

/// Rank-based edge reordering and pruning (CAGRA graph optimisation step 1).
///
/// For each node, counts how many "detourable" routes exist for each neighbour
/// edge using rank-based approximation (position in distance-sorted list as
/// proxy for distance). Neighbours with fewer detours are more important and
/// are kept; the top `d` are written to `pruned_idx`.
///
/// Uses shared memory for the neighbour list and detour counts.
///
/// ### Params
///
/// * `graph_idx` - Current kNN graph indices `[n, k]`
/// * `pruned_idx` - Output pruned graph `[n, d]`
/// * `n` - Number of nodes
/// * `k` - Input graph degree (comptime)
/// * `d` - Output graph degree after pruning (comptime)
///
/// ### Grid mapping
///
/// * One workgroup (cube) per node
#[cube(launch_unchecked)]
pub fn cagra_rank_prune_shared(
    graph_idx: &Tensor<u32>,
    pruned_idx: &mut Tensor<u32>,
    n: u32,
    #[comptime] k: usize,
    #[comptime] d: usize,
) {
    let node = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if node >= n {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let k_u32 = k as u32;
    let d_u32 = d as u32;
    let graph_base = node * k_u32;

    let mut shared_neighbors = SharedMemory::<u32>::new(k);
    let mut shared_detours = SharedMemory::<u32>::new(k);

    let mut i = tx;
    while i < k_u32 {
        shared_neighbors[i as usize] = graph_idx[(graph_base + i) as usize] & 0x7FFFFFFFu32;
        shared_detours[i as usize] = 0u32;
        i += WORKGROUP_SIZE_X;
    }
    sync_cube();

    i = tx;
    while i < k_u32 {
        let y = shared_neighbors[i as usize];
        let mut detours = 0u32;

        let mut j = 0u32;
        while j < i {
            let z = shared_neighbors[j as usize];
            let z_base = z * k_u32;

            let mut found: bool = false;
            let mut m = 0u32;
            while m < i {
                let z_neighbor = graph_idx[(z_base + m) as usize] & 0x7FFFFFFFu32;
                if z_neighbor == y {
                    found = true;
                }
                m += 1u32;
            }

            if found {
                detours += 1u32;
            }
            j += 1u32;
        }
        // pack detours into top 16 bits, original rank into bottom 16 bits
        shared_detours[i as usize] = (detours << 16) | i;
        i += WORKGROUP_SIZE_X;
    }
    sync_cube();

    // thread 0 performs selection sort and commits the top D candidates
    if tx == 0u32 {
        let mut step = 0u32;
        while step < d_u32 {
            let mut min_val = 0xFFFFFFFFu32;
            let mut min_idx = 0u32;

            let mut scan = step;
            while scan < k_u32 {
                let val = shared_detours[scan as usize];
                if val < min_val {
                    min_val = val;
                    min_idx = scan;
                }
                scan += 1u32;
            }

            let temp = shared_detours[step as usize];
            shared_detours[step as usize] = shared_detours[min_idx as usize];
            shared_detours[min_idx as usize] = temp;

            let original_rank = min_val & 0xFFFFu32;
            pruned_idx[(node * d_u32 + step) as usize] = shared_neighbors[original_rank as usize];

            step += 1u32;
        }
    }
}

/// Build reverse edge lists for the CAGRA graph (optimisation step 2).
///
/// For each node, iterates over its pruned forward edges and atomically
/// appends itself as a reverse neighbour of each target node. Overflow
/// beyond degree `d` is silently dropped.
///
/// ### Params
///
/// * `pruned_idx` - Pruned forward graph from `cagra_rank_prune_shared`
///   `[n, d]`
/// * `reverse_idx` - Output reverse edge buffer `[n, d]`
/// * `reverse_counts` - Atomic per-node reverse edge counter `[n]`
/// * `n` - Number of nodes
/// * `d` - Graph degree (comptime)
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> node index
#[cube(launch_unchecked)]
pub fn cagra_build_reverse(
    pruned_idx: &Tensor<u32>,
    reverse_idx: &mut Tensor<u32>,
    reverse_counts: &Tensor<Atomic<u32>>,
    n: u32,
    #[comptime] d: usize,
) {
    let node = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    if node >= n {
        terminate!();
    }

    let d_u32 = d as u32;
    let mut i = 0u32;
    while i < d_u32 {
        let target = pruned_idx[(node * d_u32 + i) as usize];
        if target < n {
            let pos = reverse_counts[target as usize].fetch_add(1u32);
            if pos < d_u32 {
                reverse_idx[(target * d_u32 + pos) as usize] = node;
            }
        }
        i += 1u32;
    }
}

/// Merge pruned forward and reverse edge graphs (CAGRA optimisation step 3).
///
/// For each node, takes up to `d/2` reverse edges and fills the remainder
/// from the pruned forward graph, deduplicating. Pads with sentinels.
///
/// ### Params
///
/// * `pruned_idx` - Pruned forward graph `[n, d]`
/// * `reverse_idx` - Reverse edge buffer from `cagra_build_reverse` `[n, d]`
/// * `reverse_counts` - Number of valid reverse edges per node `[n]`
/// * `final_idx` - Output merged graph `[n, d]`
/// * `n` - Number of nodes
/// * `d` - Graph degree (comptime)
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> node index
#[cube(launch_unchecked)]
pub fn cagra_merge_graphs(
    pruned_idx: &Tensor<u32>,
    reverse_idx: &Tensor<u32>,
    reverse_counts: &Tensor<u32>,
    final_idx: &mut Tensor<u32>,
    n: u32,
    #[comptime] d: usize,
) {
    let node = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X;
    if node >= n {
        terminate!();
    }

    let d_u32 = d as u32;
    let half_d = d_u32 / 2u32;
    let rev_count = reverse_counts[node as usize];

    // Establish `take_rev` as a dynamic GPU variable first
    let mut take_rev = rev_count;
    if take_rev > half_d {
        // The macro overloads assignment to handle the Rust u32 -> CubeCL u32 cast
        take_rev = half_d;
    }

    let mut final_count = 0u32;
    let mut i = 0u32;

    while i < take_rev {
        final_idx[(node * d_u32 + final_count) as usize] = reverse_idx[(node * d_u32 + i) as usize];
        final_count += 1u32;
        i += 1u32;
    }

    let mut j = 0u32;
    while j < d_u32 {
        if final_count < d_u32 {
            let candidate = pruned_idx[(node * d_u32 + j) as usize];
            let mut is_dup: bool = false;

            let mut c = 0u32;
            while c < final_count {
                if final_idx[(node * d_u32 + c) as usize] == candidate {
                    is_dup = true;
                }
                c += 1u32;
            }

            if !is_dup {
                final_idx[(node * d_u32 + final_count) as usize] = candidate;
                final_count += 1u32;
            }
        }
        j += 1u32;
    }

    // Pad remaining with sentinels
    while final_count < d_u32 {
        final_idx[(node * d_u32 + final_count) as usize] = 0x7FFFFFFFu32;
        final_count += 1u32;
    }
}

/////////////
// Helpers //
/////////////

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
fn pad_vectors<T: Float>(flat: &[T], n: usize, dim: usize, dim_padded: usize) -> Vec<T> {
    let mut padded = vec![T::zero(); n * dim_padded];
    for i in 0..n {
        let src = &flat[i * dim..(i + 1) * dim];
        let dst = &mut padded[i * dim_padded..i * dim_padded + dim];
        dst.copy_from_slice(src);
    }
    padded
}

//////////////
// Querying //
//////////////

thread_local! {
static QUERY_VISITED: RefCell<FixedBitSet> = const { RefCell::new(FixedBitSet::new()) };
static QUERY_CANDIDATES_F32: QueryCandF32 =
    const { RefCell::new(BinaryHeap::new()) };
static QUERY_CANDIDATES_F64: QueryCandF64 =
    const { RefCell::new(BinaryHeap::new()) };
static QUERY_RESULTS_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize)>> =
    const { RefCell::new(BinaryHeap::new()) };
static QUERY_RESULTS_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize)>> =
    const { RefCell::new(BinaryHeap::new()) };
}

/// Generates the `NNDescentQuery` impl for a concrete float type.
macro_rules! impl_nndescent_gpu_query {
    ($float:ty, $cand_tls:ident, $res_tls:ident) => {
        impl<R: Runtime> NNDescentQuery<$float> for NNDescentGpu<$float, R> {
            fn query_internal(
                &self,
                query_vec: &[$float],
                query_norm: $float,
                k: usize,
                ef: usize,
            ) -> Result<(Vec<usize>, Vec<$float>), AnnSearchErrors> {
                QUERY_VISITED.with(|visited_cell| {
                    $cand_tls.with(|cand_cell| {
                        $res_tls.with(|res_cell| {
                            let mut visited = visited_cell.borrow_mut();
                            let mut candidates = cand_cell.borrow_mut();
                            let mut results = res_cell.borrow_mut();

                            visited.clear();
                            visited.grow(self.n);
                            candidates.clear();
                            results.clear();

                            match self.metric {
                                Dist::SquaredEuclidean => self.query_euclidean(
                                    query_vec,
                                    k,
                                    ef,
                                    &mut visited,
                                    &mut candidates,
                                    &mut results,
                                ),
                                Dist::Cosine => self.query_cosine(
                                    query_vec,
                                    query_norm,
                                    k,
                                    ef,
                                    &mut visited,
                                    &mut candidates,
                                    &mut results,
                                ),
                                Dist::Manhattan => unreachable!(),
                            }
                        })
                    })
                })
            }

            #[inline(always)]
            fn query_euclidean(
                &self,
                query_vec: &[$float],
                k: usize,
                ef: usize,
                visited: &mut FixedBitSet,
                candidates: &mut BinaryHeap<Reverse<(OrderedFloat<$float>, usize)>>,
                results: &mut BinaryHeap<(OrderedFloat<$float>, usize)>,
            ) -> Result<(Vec<usize>, Vec<$float>), AnnSearchErrors> {
                let init_indices = self
                    .router
                    .find_entry_points(query_vec, (ef / 2).max(2 * k).min(self.n));

                for &entry_idx in &init_indices {
                    if entry_idx >= self.n || visited.contains(entry_idx) {
                        continue;
                    }
                    visited.insert(entry_idx);
                    let dist = self.euclidean_distance_to_query(entry_idx, query_vec);
                    candidates.push(Reverse((OrderedFloat(dist), entry_idx)));
                    results.push((OrderedFloat(dist), entry_idx));
                }

                while results.len() > ef {
                    results.pop();
                }

                let mut lower_bound = if results.len() >= ef {
                    results.peek().unwrap().0 .0
                } else {
                    <$float>::MAX
                };

                while let Some(Reverse((OrderedFloat(curr_dist), curr_idx))) = candidates.pop() {
                    if curr_dist > lower_bound {
                        break;
                    }

                    for &(nbr_idx, _) in self.graph_neighbours(curr_idx) {
                        if nbr_idx == SENTINEL_PID || visited.contains(nbr_idx) {
                            continue;
                        }
                        visited.insert(nbr_idx);

                        let dist = self.euclidean_distance_to_query(nbr_idx, query_vec);

                        if dist < lower_bound || results.len() < ef {
                            candidates.push(Reverse((OrderedFloat(dist), nbr_idx)));

                            if results.len() < ef {
                                results.push((OrderedFloat(dist), nbr_idx));
                                if results.len() == ef {
                                    lower_bound = results.peek().unwrap().0 .0;
                                }
                            } else if dist < lower_bound {
                                results.pop();
                                results.push((OrderedFloat(dist), nbr_idx));
                                lower_bound = results.peek().unwrap().0 .0;
                            }
                        }
                    }
                }

                let mut final_results: Vec<_> = results.drain().collect();
                final_results.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                final_results.truncate(k);

                Ok(final_results
                    .into_iter()
                    .map(|(OrderedFloat(d), i)| (i, d))
                    .unzip())
            }

            #[inline(always)]
            fn query_cosine(
                &self,
                query_vec: &[$float],
                query_norm: $float,
                k: usize,
                ef: usize,
                visited: &mut FixedBitSet,
                candidates: &mut BinaryHeap<Reverse<(OrderedFloat<$float>, usize)>>,
                results: &mut BinaryHeap<(OrderedFloat<$float>, usize)>,
            ) -> Result<(Vec<usize>, Vec<$float>), AnnSearchErrors> {
                let init_indices = self
                    .router
                    .find_entry_points(query_vec, (ef / 2).max(2 * k).min(self.n));

                for &entry_idx in &init_indices {
                    if entry_idx >= self.n || visited.contains(entry_idx) {
                        continue;
                    }
                    visited.insert(entry_idx);
                    let dist = self.cosine_distance_to_query(entry_idx, query_vec, query_norm);
                    candidates.push(Reverse((OrderedFloat(dist), entry_idx)));
                    results.push((OrderedFloat(dist), entry_idx));
                }

                while results.len() > ef {
                    results.pop();
                }

                let mut lower_bound = if results.len() >= ef {
                    results.peek().unwrap().0 .0
                } else {
                    <$float>::MAX
                };

                while let Some(Reverse((OrderedFloat(curr_dist), curr_idx))) = candidates.pop() {
                    if curr_dist > lower_bound {
                        break;
                    }

                    for &(nbr_idx, _) in self.graph_neighbours(curr_idx) {
                        if nbr_idx == SENTINEL_PID || visited.contains(nbr_idx) {
                            continue;
                        }
                        visited.insert(nbr_idx);

                        let dist = self.cosine_distance_to_query(nbr_idx, query_vec, query_norm);

                        if dist < lower_bound || results.len() < ef {
                            candidates.push(Reverse((OrderedFloat(dist), nbr_idx)));

                            if results.len() < ef {
                                results.push((OrderedFloat(dist), nbr_idx));
                                if results.len() == ef {
                                    lower_bound = results.peek().unwrap().0 .0;
                                }
                            } else if dist < lower_bound {
                                results.pop();
                                results.push((OrderedFloat(dist), nbr_idx));
                                lower_bound = results.peek().unwrap().0 .0;
                            }
                        }
                    }
                }

                let mut final_results: Vec<_> = results.drain().collect();
                final_results.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                final_results.truncate(k);

                Ok(final_results
                    .into_iter()
                    .map(|(OrderedFloat(d), i)| (i, d))
                    .unzip())
            }

            #[inline(always)]
            fn query_manhattan(
                &self,
                _query_vec: &[$float],
                _k: usize,
                _ef: usize,
                _visited: &mut FixedBitSet,
                _candidates: &mut BinaryHeap<Reverse<(OrderedFloat<$float>, usize)>>,
                _results: &mut BinaryHeap<(OrderedFloat<$float>, usize)>,
            ) -> Result<(Vec<usize>, Vec<$float>), AnnSearchErrors> {
                unreachable!("NNDescentGpu does not support Manhattan distance")
            }
        }
    };
}

impl_nndescent_gpu_query!(f32, QUERY_CANDIDATES_F32, QUERY_RESULTS_F32);
impl_nndescent_gpu_query!(f64, QUERY_CANDIDATES_F64, QUERY_RESULTS_F64);

//////////////////
// NNDescentGpu //
//////////////////

/// GPU-accelerated NNDescent kNN graph builder with CAGRA graph optimisation.
///
/// Builds a k-NN graph on the GPU, optionally using an Annoy forest for
/// initialisation. The CAGRA rank-prune + reverse-edge optimisation produces
/// a fixed-degree directed graph with improved reachability.
///
/// The final graph has exactly `k` neighbours per node (the user-requested
/// degree). Internally, NNDescent runs at a higher degree (`build_k`, default
/// `2*k`) which CAGRA then prunes down to `k`.
pub struct NNDescentGpu<T: AnnSearchFloat + AnnSearchGpuFloat, R: Runtime> {
    /// Original (unpadded) vector data, flattened row-major
    pub vectors_flat: Vec<T>,
    /// Original embedding dimensionality
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Neighbours per node (final CAGRA degree)
    pub k: usize,
    /// Pre-computed L2 norms (Cosine only; empty for Euclidean)
    pub norms: Vec<T>,
    /// Distance metric
    metric: Dist,
    /// The medoid of the graph as entry point
    pub medoid: u32,
    /// True kNN graph of size n * k, sorted by distance per row.
    /// Extracted from NNDescent output before CAGRA pruning.
    knn_graph: Vec<(usize, T)>,
    /// CAGRA navigational graph of size n * k, used for beam search.
    /// NOT a faithful kNN graph -- edges are reordered for reachability.
    nav_graph: Vec<(usize, T)>,
    /// Whether NNDescent hit the delta threshold
    converged: bool,
    /// Forest router for query entry points (replaces Annoy)
    router: ForestRouter<T>,
    /// CubeCL runtime device
    _device: R::Device,
    /// Padded dimensionality (next multiple of LINE_SIZE)
    dim_padded: usize,
    /// GPU-resident CAGRA navigational graph [n, k] (raw u32 node IDs)
    nav_graph_gpu: Option<GpuTensor<R, u32>>,
    /// GPU-resident vectors [n, dim_padded]
    vectors_gpu: Option<GpuTensor<R, T>>,
    /// GPU-resident norms [n] (cosine) or [1] (euclidean)
    norms_gpu: Option<GpuTensor<R, T>>,
}

////////////////////
// VectorDistance //
////////////////////

/// VectorDistance implementation for NNDescentGPU
impl<T, R> VectorDistance<T> for NNDescentGpu<T, R>
where
    T: AnnSearchFloat + AnnSearchGpuFloat,
    R: Runtime,
{
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn norms(&self) -> &[T] {
        &self.norms
    }
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T, R> DimensionValidation for NNDescentGpu<T, R>
where
    R: Runtime,
    T: AnnSearchGpuFloat + AnnSearchFloat,
{
    // needs to be allowed here, because dim_padded is the relevant dim for GPU
    // indices
    #[allow(clippy::misnamed_getters)]
    fn dim(&self) -> usize {
        self.dim_padded
    }
}

/////////////////////////
// Lightweight getters //
/////////////////////////

impl<T, R> NNDescentGpu<T, R>
where
    R: Runtime,
    T: AnnSearchFloat + AnnSearchGpuFloat,
{
    /// Distance metric this index was built with.
    ///
    /// ### Returns
    ///
    /// The [`Dist`] metric embedded at build time.
    pub fn metric(&self) -> Dist {
        self.metric
    }

    /// Borrow the flat kNN graph produced by NN-Descent.
    ///
    /// Layout is `n * k` `(pid, distance)` pairs, sorted by distance
    /// ascending within each node's slot. Sentinel entries mark unused
    /// slots. This is the graph before the CAGRA rank-prune / reverse-edge
    /// optimisation is applied. It is the input a downstream index like
    /// NSG should consume.
    ///
    /// ### Returns
    ///
    /// A slice view over the raw NN-Descent graph.
    pub fn knn_graph(&self) -> &[(usize, T)] {
        &self.knn_graph
    }
}

/////////////////////////
// Main implementation //
/////////////////////////

impl<T, R> NNDescentGpu<T, R>
where
    R: Runtime,
    T: AnnSearchFloat + cubecl::frontend::Float + cubecl::CubeElement,
    Self: NNDescentQuery<T>,
{
    /// Build a kNN graph on the GPU via NNDescent + CAGRA optimisation.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features). Dimensions will be
    ///   padded to the next multiple of LINE_SIZE if necessary.
    /// * `metric` - Distance metric
    /// * `k` - Final neighbours per node (default 30)
    /// * `build_k` - Internal NNDescent degree before CAGRA pruning.
    ///   Defaults to `1.5 * k`. Must be >= `k`.
    /// * `max_iters` - Maximum NNDescent iterations (default 15)
    /// * `n_trees` - Number of Annoy trees for graph initialisation.
    ///   Defaults to `5 + n^0.25`, capped at 32.
    /// * `delta` - Convergence threshold as fraction of n*k (default `0.001`)
    /// * `rho` - Sampling rate for the local join (default `0.5`)
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    /// * `device` - CubeCL runtime device
    ///
    /// ### Returns
    ///
    /// Initialised struct with the completed kNN and CAGRA navigational graphs
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: MatRef<T>,
        metric: Dist,
        k: Option<usize>,
        build_k: Option<usize>,
        max_iters: Option<usize>,
        n_trees: Option<usize>,
        delta: Option<f32>,
        rho: Option<f32>,
        refine_knn: Option<usize>,
        seed: usize,
        verbose: bool,
        retain_gpu: bool,
        device: R::Device,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (vectors_flat, n, dim) = matrix_to_flat(data);
        let k = k.unwrap_or(30);
        let build_k = build_k.unwrap_or((1.5 * k as f32) as usize).max(k);
        let max_iters = max_iters.unwrap_or(DEFAULT_MAX_ITERS);
        let delta = delta.unwrap_or(DEFAULT_DELTA);
        let rho = rho.unwrap_or(DEFAULT_RHO);
        let rho_thresh = (rho * 65535.0) as u32;
        let refine_knn = refine_knn.unwrap_or(0);

        let medoid = compute_medoid(&vectors_flat, n, dim, metric);

        // pad dim to next multiple of LINE_SIZE
        let line = LINE_SIZE;
        let dim_padded = dim.next_multiple_of(line);
        let dim_vec = dim_padded / line;

        let vectors_padded = if dim_padded != dim {
            pad_vectors(&vectors_flat, n, dim, dim_padded)
        } else {
            vectors_flat.clone()
        };

        let norms = if metric == Dist::Cosine {
            (0..n)
                .into_par_iter()
                .map(|i| T::calculate_l2_norm(&vectors_flat[i * dim..(i + 1) * dim]))
                .collect()
        } else {
            Vec::new()
        };

        if verbose {
            println!(
                "NNDescent-GPU: {} vectors, dim={} (padded to {}), k={}, build_k={}",
                n.separate_with_underscores(),
                dim,
                dim_padded,
                k,
                build_k,
            );
        }

        let start = Instant::now();

        // ---- 0: GPU setup ----

        let n_trees_forest = n_trees.unwrap_or_else(|| {
            let calculated = 5 + ((n as f64).powf(0.25)).round() as usize;
            calculated.min(20)
        });

        let client = R::client(&device);
        let use_cosine = metric == Dist::Cosine;

        // upload vectors (stays resident for the entire build)
        let vectors_gpu =
            GpuTensor::<R, T>::from_slice(&vectors_padded, vec![n, dim_padded], &client);

        // norms tensor (dummy scalar if Euclidean to avoid Option in kernel args)
        let norms_gpu = if use_cosine {
            GpuTensor::<R, T>::from_slice(&norms, vec![n], &client)
        } else {
            GpuTensor::<R, T>::from_slice(&[T::zero()], vec![1], &client)
        };

        // Pre-allocate graph with sentinels
        let graph_idx_gpu = GpuTensor::<R, u32>::from_slice(
            &vec![0x7FFFFFFFu32; n * build_k],
            vec![n, build_k],
            &client,
        );
        let graph_dist_gpu = GpuTensor::<R, T>::from_slice(
            &vec![<T as num_traits::Float>::max_value(); n * build_k],
            vec![n, build_k],
            &client,
        );

        // Proposal buffers (shared between forest init and NNDescent iterations)
        let max_prop = MAX_PROPOSALS;
        let prop_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, max_prop], &client);
        let prop_dist_gpu = GpuTensor::<R, T>::empty(vec![n, max_prop], &client);
        let prop_count_gpu = GpuTensor::<R, u32>::empty(vec![n], &client);
        let update_counter_gpu = GpuTensor::<R, u32>::empty(vec![1], &client);

        let (grid_n_x, grid_n_y) = grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X));

        // 1: random graph initialisation (baseline for NNDescent)
        if verbose {
            println!("  Random graph initialisation...");
        }

        unsafe {
            init_random_graph::launch_unchecked::<T, R>(
                &client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.clone().into_tensor_arg(),
                norms_gpu.clone().into_tensor_arg(),
                graph_idx_gpu.clone().into_tensor_arg(),
                graph_dist_gpu.clone().into_tensor_arg(),
                n as u32,
                seed as u32,
                use_cosine,
                dim_vec,
            );
        }

        // 1b: GPU forest graph initialisation
        let router = gpu_forest_init(
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
            n_trees_forest,
            seed,
            use_cosine,
            verbose,
            &client,
        )?;

        // 1c: Mark all graph entries as new for NNDescent
        let total_entries = (n * build_k) as u32;
        let mark_grid_flat = total_entries.div_ceil(WORKGROUP_SIZE_X);
        let mark_cubes_x = mark_grid_flat.min(65535);
        let mark_cubes_y = mark_grid_flat.div_ceil(mark_cubes_x);
        unsafe {
            mark_all_new::launch_unchecked::<R>(
                &client,
                CubeCount::Static(mark_cubes_x, mark_cubes_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                graph_idx_gpu.clone().into_tensor_arg(),
                total_entries,
            );
        }

        // 2: NNDescent iterations on the GPU

        let iter_start = Instant::now();
        let mut converged = false;

        let reverse_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, build_k], &client);
        let reverse_count_gpu = GpuTensor::<R, u32>::empty(vec![n], &client);

        for iter in 0..max_iters {
            let cubes_x = 65535u32;
            let cubes_y = (n as u32).div_ceil(cubes_x);

            // 1. Reset proposal counts, reverse counts, and update counter
            unsafe {
                reset_proposals::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static(grid_n_x, grid_n_y, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    prop_count_gpu.clone().into_tensor_arg(),
                    update_counter_gpu.clone().into_tensor_arg(),
                    n as u32,
                );

                reset_proposals::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static(grid_n_x, grid_n_y, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    reverse_count_gpu.clone().into_tensor_arg(),
                    update_counter_gpu.clone().into_tensor_arg(),
                    n as u32,
                );
            }

            // 2. Build reverse edges
            unsafe {
                build_reverse_candidates::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static(grid_n_x, grid_n_y, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    graph_idx_gpu.clone().into_tensor_arg(),
                    reverse_idx_gpu.clone().into_tensor_arg(),
                    reverse_count_gpu.clone().into_tensor_arg(),
                    n as u32,
                    build_k as u32,
                );
            }

            let iter_seed = seed as u32 ^ (iter as u32).wrapping_mul(0x9E3779B9u32);

            // 3. Local join
            unsafe {
                local_join_shared::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(cubes_x, cubes_y, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    line,
                    vectors_gpu.clone().into_tensor_arg(),
                    norms_gpu.clone().into_tensor_arg(),
                    graph_idx_gpu.clone().into_tensor_arg(),
                    graph_dist_gpu.clone().into_tensor_arg(),
                    reverse_idx_gpu.clone().into_tensor_arg(),
                    reverse_count_gpu.clone().into_tensor_arg(),
                    prop_idx_gpu.clone().into_tensor_arg(),
                    prop_dist_gpu.clone().into_tensor_arg(),
                    prop_count_gpu.clone().into_tensor_arg(),
                    n as u32,
                    rho_thresh,
                    iter_seed,
                    MAX_PROPOSALS as u32,
                    use_cosine,
                    dim_vec,
                    build_k,
                );
            }

            // 4. Merge proposals into the graph
            unsafe {
                merge_proposals::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(grid_n_x, grid_n_y, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    graph_idx_gpu.clone().into_tensor_arg(),
                    graph_dist_gpu.clone().into_tensor_arg(),
                    prop_idx_gpu.clone().into_tensor_arg(),
                    prop_dist_gpu.clone().into_tensor_arg(),
                    prop_count_gpu.clone().into_tensor_arg(),
                    update_counter_gpu.clone().into_tensor_arg(),
                    n as u32,
                    MAX_PROPOSALS as u32,
                );
            }

            // 5. Download single u32 to check convergence
            let counter_data = update_counter_gpu.clone().read(&client)?;
            let updates = counter_data[0] as f64;
            let rate = updates / (n * build_k) as f64;

            if verbose {
                println!(
                    "   Iter {}: {} updates (rate={:.6})",
                    iter + 1,
                    (updates as usize).separate_with_underscores(),
                    rate
                );
            }

            if rate < delta as f64 {
                if verbose {
                    println!("  Converged after {} iterations", iter + 1);
                }
                converged = true;
                break;
            }
        }

        if verbose {
            println!("  NNDescent iterations: {:.2?}", iter_start.elapsed());
        }

        // ---- 3: 2-Hop Refinement ----

        if verbose && refine_knn > 0 {
            println!("  Running 2-Hop Refinement Sweep...");
        }

        let refinement_start = Instant::now();

        let cubes_x = 65535u32;
        let cubes_y = (n as u32).div_ceil(cubes_x);

        for sweep in 0..refine_knn {
            unsafe {
                reset_proposals::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static(grid_n_x, grid_n_y, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    prop_count_gpu.clone().into_tensor_arg(),
                    update_counter_gpu.clone().into_tensor_arg(),
                    n as u32,
                );
            }

            unsafe {
                two_hop_refinement::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(cubes_x, cubes_y, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    line,
                    vectors_gpu.clone().into_tensor_arg(),
                    norms_gpu.clone().into_tensor_arg(),
                    graph_idx_gpu.clone().into_tensor_arg(),
                    graph_dist_gpu.clone().into_tensor_arg(),
                    prop_idx_gpu.clone().into_tensor_arg(),
                    prop_dist_gpu.clone().into_tensor_arg(),
                    prop_count_gpu.clone().into_tensor_arg(),
                    n as u32,
                    MAX_PROPOSALS as u32,
                    use_cosine,
                    dim_vec,
                );
            }

            unsafe {
                merge_proposals::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(grid_n_x, grid_n_y, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    graph_idx_gpu.clone().into_tensor_arg(),
                    graph_dist_gpu.clone().into_tensor_arg(),
                    prop_idx_gpu.clone().into_tensor_arg(),
                    prop_dist_gpu.clone().into_tensor_arg(),
                    prop_count_gpu.clone().into_tensor_arg(),
                    update_counter_gpu.clone().into_tensor_arg(),
                    n as u32,
                    MAX_PROPOSALS as u32,
                );
            }

            if verbose {
                let counter_data = update_counter_gpu.clone().read(&client)?;
                println!(
                    "    2-Hop sweep {}: {} updates",
                    sweep + 1,
                    counter_data[0].separate_with_underscores()
                );
            }

            let refinement_stop = refinement_start.elapsed();

            if verbose {
                println!("  NNDescent refinement done in: {:.2?}", refinement_stop);
            }
        }

        // ---- 4: Extract kNN graph from NNDescent result ----

        let nndescent_idx = graph_idx_gpu.clone().read(&client)?;
        let nndescent_dist = graph_dist_gpu.clone().read(&client)?;
        let pid_mask = 0x7FFFFFFFu32;
        let sentinel = 0x7FFFFFFFusize;

        let mut knn_graph = vec![(sentinel, <T as num_traits::Float>::max_value()); n * k];

        knn_graph
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(i, slot)| {
                let mut written = 0;
                for j in 0..build_k {
                    if written >= k {
                        break;
                    }
                    let raw = nndescent_idx[i * build_k + j];
                    let pid = (raw & pid_mask) as usize;
                    if pid < n && pid != i && pid != sentinel {
                        let dist = nndescent_dist[i * build_k + j];
                        slot[written] = (pid, dist);
                        written += 1;
                    }
                }
            });

        // ---- 5: CAGRA graph optimisation: prune from build_k -> k ----

        let cagra_start = Instant::now();

        let pruned_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, k], &client);
        let reverse_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, k], &client);
        let reverse_counts_gpu = GpuTensor::<R, u32>::from_slice(&vec![0u32; n], vec![n], &client);
        let final_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, k], &client);

        let cubes_x = 65535u32;
        let cubes_y = (n as u32).div_ceil(cubes_x);

        unsafe {
            cagra_rank_prune_shared::launch_unchecked::<R>(
                &client,
                CubeCount::Static(cubes_x, cubes_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                graph_idx_gpu.into_tensor_arg(),
                pruned_idx_gpu.clone().into_tensor_arg(),
                n as u32,
                build_k,
                k,
            );

            cagra_build_reverse::launch_unchecked::<R>(
                &client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                pruned_idx_gpu.clone().into_tensor_arg(),
                reverse_idx_gpu.clone().into_tensor_arg(),
                reverse_counts_gpu.clone().into_tensor_arg(),
                n as u32,
                k,
            );

            cagra_merge_graphs::launch_unchecked::<R>(
                &client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                pruned_idx_gpu.into_tensor_arg(),
                reverse_idx_gpu.into_tensor_arg(),
                reverse_counts_gpu.into_tensor_arg(),
                final_idx_gpu.clone().into_tensor_arg(),
                n as u32,
                k,
            );
        }

        if verbose {
            println!("  CAGRA optimisation: {:.2?}", cagra_start.elapsed());
        }

        // ---- 6: Download CAGRA graph and compute CPU distances ----

        let final_idx = final_idx_gpu.clone().read(&client)?;
        let pid_mask = 0x7FFFFFFFu32;
        let sentinel = 0x7FFFFFFFusize;

        let mut cagra_graph = vec![(sentinel, <T as num_traits::Float>::max_value()); n * k];

        cagra_graph
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(i, slot)| {
                for j in 0..k {
                    let raw = final_idx[i * k + j];
                    let pid = (raw & pid_mask) as usize;

                    if pid < n && pid != sentinel {
                        let a = &vectors_flat[i * dim..(i + 1) * dim];
                        let b = &vectors_flat[pid * dim..(pid + 1) * dim];
                        let dist = match metric {
                            Dist::SquaredEuclidean => T::euclidean_simd(a, b),
                            Dist::Cosine => {
                                let dot = T::dot_simd(a, b);
                                T::one() - dot / (norms[i] * norms[pid])
                            }
                            Dist::Manhattan => unreachable!(),
                        };
                        slot[j] = (pid, dist);
                    }
                }

                slot.sort_unstable_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            });

        if verbose {
            println!("  Total build time: {:.2?}", start.elapsed());
        }

        let (nav_graph_gpu, vectors_gpu, norms_gpu) = if retain_gpu {
            (Some(final_idx_gpu), Some(vectors_gpu), Some(norms_gpu))
        } else {
            (None, None, None)
        };

        Ok(Self {
            vectors_flat,
            dim,
            dim_padded,
            n,
            k,
            medoid,
            norms,
            metric,
            router,
            knn_graph,
            nav_graph: cagra_graph,
            converged,
            nav_graph_gpu,
            vectors_gpu,
            norms_gpu,
            _device: device,
        })
    }

    ///////////
    // Query //
    ///////////

    /// Query for k nearest neighbours using beam search.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Beam width. Higher values improve recall at the
    ///   cost of latency. Defaults to `max(2*k, 50)` clamped to 200.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted by distance ascending
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let k = k.min(self.n);
        let ef = ef_search.unwrap_or_else(|| (k * 2).clamp(50, 200)).max(k);

        let query_norm = if self.metric == Dist::Cosine {
            num_traits::Float::sqrt(query_vec.iter().map(|x| *x * *x).sum::<T>())
        } else {
            T::one()
        };

        self.query_internal(query_vec, query_norm, k, ef)
    }

    /// Query using a matrix row reference.
    ///
    /// Uses a zero-copy path when stride is 1, otherwise copies to a
    /// temporary vector.
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference into a faer matrix
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Beam width; see `query` for details
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted by distance ascending
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, ef_search);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, ef_search)
    }

    /// Batch query via GPU beam search on the CAGRA navigational graph.
    ///
    /// Re-uploads tensors if they were not retained during build.
    /// For small batch sizes (< ~32), the CPU path may be faster due
    /// to kernel launch overhead.
    ///
    /// ### Params
    ///
    /// * `queries_flat` - Flattened query vectors, row-major [n_queries, dim]
    /// * `n_queries` - Number of query vectors
    /// * `query_params` - Optional Cagra beam search parameters if you want
    ///   to specify width, iterations.
    /// * `k` - Number of neighbours to return per query
    /// * `seed` - Random seed for entry point generation
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` per query, sorted by distance ascending
    pub fn query_batch_gpu(
        &mut self,
        queries_flat: &[T],
        n_queries: usize,
        query_params: Option<CagraGpuSearchParams>,
        k: usize,
        seed: usize,
    ) -> KnnResult<T>
    where
        T: AnnSearchGpuFloat + num_traits::Float,
    {
        let dim_query = queries_flat.len() / n_queries;

        self.check_dim(dim_query)?;

        let query_params =
            query_params.unwrap_or_else(|| CagraGpuSearchParams::from_graph(k, self.k));
        let n_entry = query_params.get_n_entry();
        self.ensure_gpu_tensors();
        let client = R::client(&self._device);
        let use_cosine = self.metric == Dist::Cosine;

        let medoid = self.medoid;
        let entry_flat: Vec<u32> = (0..n_queries)
            .into_par_iter()
            .flat_map_iter(|i| {
                let query = &queries_flat[i * self.dim..(i + 1) * self.dim];
                let mut candidates = self.router.find_entry_points(query, n_entry * 4);

                candidates.sort_unstable_by(|&a, &b| {
                    let dist_a = match self.metric {
                        Dist::SquaredEuclidean => {
                            let va = &self.vectors_flat[a * self.dim..(a + 1) * self.dim];
                            T::euclidean_simd(query, va)
                        }
                        Dist::Cosine => {
                            let va = &self.vectors_flat[a * self.dim..(a + 1) * self.dim];
                            let dot = T::dot_simd(query, va);
                            let q_norm = T::calculate_l2_norm(query);
                            T::one() - dot / (q_norm * self.norms[a])
                        }
                        Dist::Manhattan => unreachable!(),
                    };
                    let dist_b = match self.metric {
                        Dist::SquaredEuclidean => {
                            let vb = &self.vectors_flat[b * self.dim..(b + 1) * self.dim];
                            T::euclidean_simd(query, vb)
                        }
                        Dist::Cosine => {
                            let vb = &self.vectors_flat[b * self.dim..(b + 1) * self.dim];
                            let dot = T::dot_simd(query, vb);
                            let q_norm = T::calculate_l2_norm(query);
                            T::one() - dot / (q_norm * self.norms[b])
                        }
                        Dist::Manhattan => unreachable!(),
                    };
                    dist_a
                        .partial_cmp(&dist_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // deduplicate against medoid, take the closest ones
                candidates.retain(|&e| e != medoid as usize);
                candidates.truncate(n_entry - 1);

                let mut final_entries = Vec::with_capacity(n_entry);
                final_entries.push(medoid);
                final_entries.extend(candidates.into_iter().map(|idx| idx as u32));
                final_entries.resize(n_entry, 0);
                final_entries.into_iter()
            })
            .collect();

        let result = cagra_search_batch_gpu(
            queries_flat,
            n_queries,
            self.dim,
            self.vectors_gpu.as_ref().unwrap(),
            self.norms_gpu.as_ref().unwrap(),
            self.nav_graph_gpu.as_ref().unwrap(),
            self.n,
            self.k,
            k,
            use_cosine,
            seed,
            &query_params,
            Some(&entry_flat),
            &client,
        )?;

        Ok(result)
    }

    ///////////
    // Utils //
    ///////////

    /// Return the CAGRA navigational neighbours of node `idx`.
    ///
    /// ### Params
    ///
    /// * `idx` - Node index
    ///
    /// ### Returns
    ///
    /// Slice of `(neighbour_index, distance)` pairs, length `k`
    fn graph_neighbours(&self, idx: usize) -> &[(usize, T)] {
        &self.nav_graph[idx * self.k..(idx + 1) * self.k]
    }

    /// Whether NNDescent reached the convergence threshold during construction.
    ///
    /// ### Returns
    ///
    /// `true` if the update rate fell below `delta` before `max_iters` was
    /// exhausted, `false` otherwise.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Returns the CPU-side memory footprint of the index in bytes.
    ///
    /// Does not account for any GPU-resident tensors. Use the VRAM figures
    /// from `GpuTensor::vram_bytes` separately if needed.
    ///
    /// ### Returns
    ///
    /// Total bytes allocated on the CPU for this struct and its owned Vecs.
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
            + self.nav_graph.capacity() * std::mem::size_of::<(usize, T)>()
            + self.knn_graph.capacity() * std::mem::size_of::<(usize, T)>()
    }

    /// Extract the kNN graph as index/distance vectors.
    ///
    /// This is a zero-copy reshape of the internal graph -- no search
    /// or distance computation is performed.
    ///
    /// ### Params
    ///
    /// * `return_dist` - Whether to include distances in the output
    ///
    /// ### Returns
    ///
    /// `(knn_indices, optional distances)` where each inner Vec has
    /// length `k`, sorted by distance ascending. Sentinel entries
    /// (unfilled slots) are excluded.
    pub fn extract_knn(&self, return_dist: bool) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        let sentinel = 0x7FFFFFFFusize;

        let indices: Vec<Vec<usize>> = (0..self.n)
            .map(|i| {
                self.knn_graph[i * self.k..(i + 1) * self.k]
                    .iter()
                    .filter(|&&(pid, _)| pid != sentinel)
                    .map(|&(pid, _)| pid)
                    .collect()
            })
            .collect();

        let distances = if return_dist {
            Some(
                (0..self.n)
                    .map(|i| {
                        self.knn_graph[i * self.k..(i + 1) * self.k]
                            .iter()
                            .filter(|&&(pid, _)| pid != sentinel)
                            .map(|&(_, dist)| dist)
                            .collect()
                    })
                    .collect(),
            )
        } else {
            None
        };

        (indices, distances)
    }

    /// Self-query: run GPU beam search for every vector in the index.
    ///
    /// Searches the CAGRA navigational graph, so results differ from
    /// `extract_knn` (which returns the raw NNDescent graph).
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `seed` - Random seed for entry points
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` per vector, sorted by distance ascending
    pub fn self_query_gpu(
        &mut self,
        k: usize,
        query_params: Option<CagraGpuSearchParams>,
        seed: usize,
    ) -> KnnResult<T>
    where
        T: AnnSearchGpuFloat + AnnSearchFloat,
    {
        self.ensure_gpu_tensors();

        let query_params =
            query_params.unwrap_or_else(|| CagraGpuSearchParams::from_graph(k, self.k));
        let n_entry = query_params.get_n_entry();

        let client = R::client(&self._device);
        let use_cosine = self.metric == Dist::Cosine;

        let entry_flat: Vec<u32> = (0..self.n)
            .flat_map(|i| {
                let row = &self.knn_graph[i * self.k..(i + 1) * self.k];
                let valid: Vec<u32> = row
                    .iter()
                    .filter(|&&(pid, _)| pid != SENTINEL_PID)
                    .map(|&(pid, _)| pid as u32)
                    .collect();
                // Slot 0 is the node itself: guarantees the beam lands on it
                // (distance 0) and expands its own adjacency list immediately.
                let remaining = n_entry - 1;
                let stride = (valid.len() / remaining.max(1)).max(1);
                let mut entries: Vec<u32> = vec![i as u32];
                entries.extend((0..remaining).filter_map(|j| valid.get(j * stride).copied()));
                let mut rng_val = (i as u32) ^ (seed as u32);
                while entries.len() < n_entry {
                    rng_val = rng_val.wrapping_mul(1664525).wrapping_add(1013904223);
                    entries.push(rng_val % self.n as u32);
                }
                entries
            })
            .collect();

        let queries_flat = self.vectors_flat.clone();

        cagra_search_batch_gpu(
            &queries_flat,
            self.n,
            self.dim,
            self.vectors_gpu.as_ref().unwrap(),
            self.norms_gpu.as_ref().unwrap(),
            self.nav_graph_gpu.as_ref().unwrap(),
            self.n,
            self.k,
            k,
            use_cosine,
            seed,
            &query_params,
            Some(&entry_flat),
            &client,
        )
    }

    /// Ensure GPU tensors are resident, re-uploading from CPU data if needed.
    ///
    /// If `retain_gpu` was `false` during `build`, the GPU tensors are `None`.
    /// This method reconstructs them from `vectors_flat`, `norms`, and
    /// `nav_graph` so that `query_batch_gpu` and `self_query_gpu` can proceed.
    /// No-ops if tensors are already present.
    ///
    /// ### Params
    ///
    /// * `&mut self` - Mutates `vectors_gpu`, `norms_gpu`, and `nav_graph_gpu`
    ///   in place if they are `None`
    fn ensure_gpu_tensors(&mut self) {
        if self.nav_graph_gpu.is_some() {
            return;
        }

        let client = R::client(&self._device);
        let dim_padded = self.dim_padded;

        let vectors_padded = if dim_padded != self.dim {
            pad_vectors(&self.vectors_flat, self.n, self.dim, dim_padded)
        } else {
            self.vectors_flat.clone()
        };
        self.vectors_gpu = Some(GpuTensor::<R, T>::from_slice(
            &vectors_padded,
            vec![self.n, dim_padded],
            &client,
        ));

        self.norms_gpu = Some(if self.metric == Dist::Cosine {
            GpuTensor::<R, T>::from_slice(&self.norms, vec![self.n], &client)
        } else {
            GpuTensor::<R, T>::from_slice(&[T::zero()], vec![1], &client)
        });

        let nav_flat: Vec<u32> = self.nav_graph.iter().map(|&(pid, _)| pid as u32).collect();
        self.nav_graph_gpu = Some(GpuTensor::<R, u32>::from_slice(
            &nav_flat,
            vec![self.n, self.k],
            &client,
        ));
    }
}

/////////////////
// KnnGraphGpu //
/////////////////

/// Raw kNN graph built on the GPU, without CAGRA optimisation or query
/// support.
///
/// A slim counterpart to [`NNDescentGpu`] aimed at consumers that only need a
/// true k-nearest-neighbour graph (NSG feeding, downstream MRNG pruning, plain
/// kNN extraction) and do not want to pay for CAGRA's rank-prune +
/// reverse-merge kernels, the CPU distance recomputation pass, or the second
/// `nav_graph` copy in memory.
///
/// The graph is exactly `n * k` `(pid, dist)` pairs. Rows are sorted ascending
/// by distance and sentinel-padded when NNDescent returned fewer than `k` valid
/// non-self neighbours for a node.
pub struct KnnGraphGpu<T> {
    /// Original (unpadded) vector data, flattened row-major
    pub vectors_flat: Vec<T>,
    /// Original embedding dimensionality
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Neighbours per node
    pub k: usize,
    /// Pre-computed L2 norms (Cosine only; empty for Euclidean)
    pub norms: Vec<T>,
    /// Distance metric
    pub metric: Dist,
    /// Flat kNN graph of size `n * k`, sorted per row ascending by distance
    pub knn_graph: Vec<(usize, T)>,
    /// Whether NNDescent hit the delta convergence threshold
    pub converged: bool,
}

/// Build a raw kNN graph on the GPU without touching the CAGRA path.
///
/// Reuses every kernel that [`NNDescentGpu::build`] uses for the NNDescent
/// phase but skips the CAGRA rank-prune, reverse-merge, and the CPU distance
/// recomputation that follows.
///
/// Query methods are deliberately absent: `KnnGraphGpu` is a data
/// handoff, not a queryable index. Feed it to [`NsgIndex::build_from_gpu_knn`]
/// for graph-based query, or unpack `knn_graph` directly for raw kNN consumers.
///
/// ### Params
///
/// * `data` - Row-major sample matrix
/// * `metric` - Distance metric (Manhattan is rejected)
/// * `k` - Neighbours per node in the returned graph. Defaults to 30
/// * `build_k` - Wider working degree during NNDescent iterations.
///   Defaults to `max(k, 1.5 * k)`. Larger `build_k` improves final
///   graph quality at a linear iteration-cost hit
/// * `max_iters` - Maximum NNDescent iterations. Defaults to
///   [`DEFAULT_MAX_ITERS`]
/// * `n_trees` - Trees for GPU forest init. Defaults to a `n^0.25` rule
/// * `delta` - Convergence threshold (fraction of `n*build_k` edges
///   updated). Defaults to [`DEFAULT_DELTA`]
/// * `rho` - Local-join sampling rate. Defaults to [`DEFAULT_RHO`]
/// * `refine_knn` - Number of 2-hop refinement sweeps after the main
///   NNDescent loop. Defaults to `0`
/// * `seed` - RNG seed for reproducibility
/// * `verbose` - Print per-phase progress
/// * `device` - CubeCL runtime device
///
/// ### Returns
///
/// Populated [`KnnGraphGpu`].
#[allow(clippy::too_many_arguments)]
pub fn build_knn_graph_gpu<T, R>(
    data: MatRef<T>,
    metric: Dist,
    k: Option<usize>,
    build_k: Option<usize>,
    max_iters: Option<usize>,
    n_trees: Option<usize>,
    delta: Option<f32>,
    rho: Option<f32>,
    refine_knn: Option<usize>,
    seed: usize,
    verbose: bool,
    device: R::Device,
) -> Result<KnnGraphGpu<T>, AnnSearchErrors>
where
    T: AnnSearchFloat + AnnSearchGpuFloat,
    R: Runtime,
{
    if metric == Dist::Manhattan {
        return Err(AnnSearchErrors::DistanceNotSupported(metric));
    }

    let (vectors_flat, n, dim) = matrix_to_flat(data);
    let k = k.unwrap_or(30);
    let build_k = build_k.unwrap_or((1.5 * k as f32) as usize).max(k);
    let max_iters = max_iters.unwrap_or(DEFAULT_MAX_ITERS);
    let delta = delta.unwrap_or(DEFAULT_DELTA);
    let rho = rho.unwrap_or(DEFAULT_RHO);
    let rho_thresh = (rho * 65535.0) as u32;
    let refine_knn = refine_knn.unwrap_or(0);

    let line = LINE_SIZE;
    let dim_padded = dim.next_multiple_of(line);
    let dim_vec = dim_padded / line;

    let vectors_padded = if dim_padded != dim {
        pad_vectors(&vectors_flat, n, dim, dim_padded)
    } else {
        vectors_flat.clone()
    };

    let norms = if metric == Dist::Cosine {
        (0..n)
            .into_par_iter()
            .map(|i| T::calculate_l2_norm(&vectors_flat[i * dim..(i + 1) * dim]))
            .collect()
    } else {
        Vec::new()
    };

    if verbose {
        println!(
            "kNN-Graph-GPU: {} vectors, dim={} (padded to {}), k={}, build_k={}",
            n.separate_with_underscores(),
            dim,
            dim_padded,
            k,
            build_k,
        );
    }

    let start = Instant::now();

    // ---- GPU setup ----

    let n_trees_forest = n_trees.unwrap_or_else(|| {
        let calculated = 5 + ((n as f64).powf(0.25)).round() as usize;
        calculated.min(20)
    });

    let client = R::client(&device);
    let use_cosine = metric == Dist::Cosine;

    let vectors_gpu = GpuTensor::<R, T>::from_slice(&vectors_padded, vec![n, dim_padded], &client);

    let norms_gpu = if use_cosine {
        GpuTensor::<R, T>::from_slice(&norms, vec![n], &client)
    } else {
        GpuTensor::<R, T>::from_slice(&[T::zero()], vec![1], &client)
    };

    let graph_idx_gpu = GpuTensor::<R, u32>::from_slice(
        &vec![0x7FFFFFFFu32; n * build_k],
        vec![n, build_k],
        &client,
    );
    let graph_dist_gpu = GpuTensor::<R, T>::from_slice(
        &vec![<T as num_traits::Float>::max_value(); n * build_k],
        vec![n, build_k],
        &client,
    );

    let max_prop = MAX_PROPOSALS;
    let prop_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, max_prop], &client);
    let prop_dist_gpu = GpuTensor::<R, T>::empty(vec![n, max_prop], &client);
    let prop_count_gpu = GpuTensor::<R, u32>::empty(vec![n], &client);
    let update_counter_gpu = GpuTensor::<R, u32>::empty(vec![1], &client);

    let (grid_n_x, grid_n_y) = grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X));

    // ---- Random graph initialisation ----

    if verbose {
        println!("  Random graph initialisation...");
    }
    unsafe {
        init_random_graph::launch_unchecked::<T, R>(
            &client,
            CubeCount::Static(grid_n_x, grid_n_y, 1),
            CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
            line,
            vectors_gpu.clone().into_tensor_arg(),
            norms_gpu.clone().into_tensor_arg(),
            graph_idx_gpu.clone().into_tensor_arg(),
            graph_dist_gpu.clone().into_tensor_arg(),
            n as u32,
            seed as u32,
            use_cosine,
            dim_vec,
        );
    }

    // ---- Forest graph initialisation ----

    let _router = gpu_forest_init(
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
        n_trees_forest,
        seed,
        use_cosine,
        verbose,
        &client,
    )?;

    // ---- Mark all graph entries as new ----

    let total_entries = (n * build_k) as u32;
    let mark_grid_flat = total_entries.div_ceil(WORKGROUP_SIZE_X);
    let mark_cubes_x = mark_grid_flat.min(65535);
    let mark_cubes_y = mark_grid_flat.div_ceil(mark_cubes_x);
    unsafe {
        mark_all_new::launch_unchecked::<R>(
            &client,
            CubeCount::Static(mark_cubes_x, mark_cubes_y, 1),
            CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
            graph_idx_gpu.clone().into_tensor_arg(),
            total_entries,
        );
    }

    // ---- NNDescent iterations ----

    let iter_start = Instant::now();
    let mut converged = false;

    let reverse_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, build_k], &client);
    let reverse_count_gpu = GpuTensor::<R, u32>::empty(vec![n], &client);

    for iter in 0..max_iters {
        let cubes_x = 65535u32;
        let cubes_y = (n as u32).div_ceil(cubes_x);

        unsafe {
            reset_proposals::launch_unchecked::<R>(
                &client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                prop_count_gpu.clone().into_tensor_arg(),
                update_counter_gpu.clone().into_tensor_arg(),
                n as u32,
            );

            reset_proposals::launch_unchecked::<R>(
                &client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                reverse_count_gpu.clone().into_tensor_arg(),
                update_counter_gpu.clone().into_tensor_arg(),
                n as u32,
            );
        }

        unsafe {
            build_reverse_candidates::launch_unchecked::<R>(
                &client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                graph_idx_gpu.clone().into_tensor_arg(),
                reverse_idx_gpu.clone().into_tensor_arg(),
                reverse_count_gpu.clone().into_tensor_arg(),
                n as u32,
                build_k as u32,
            );
        }

        let iter_seed = seed as u32 ^ (iter as u32).wrapping_mul(0x9E3779B9u32);

        unsafe {
            local_join_shared::launch_unchecked::<T, R>(
                &client,
                CubeCount::Static(cubes_x, cubes_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.clone().into_tensor_arg(),
                norms_gpu.clone().into_tensor_arg(),
                graph_idx_gpu.clone().into_tensor_arg(),
                graph_dist_gpu.clone().into_tensor_arg(),
                reverse_idx_gpu.clone().into_tensor_arg(),
                reverse_count_gpu.clone().into_tensor_arg(),
                prop_idx_gpu.clone().into_tensor_arg(),
                prop_dist_gpu.clone().into_tensor_arg(),
                prop_count_gpu.clone().into_tensor_arg(),
                n as u32,
                rho_thresh,
                iter_seed,
                MAX_PROPOSALS as u32,
                use_cosine,
                dim_vec,
                build_k,
            );
        }

        unsafe {
            merge_proposals::launch_unchecked::<T, R>(
                &client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                graph_idx_gpu.clone().into_tensor_arg(),
                graph_dist_gpu.clone().into_tensor_arg(),
                prop_idx_gpu.clone().into_tensor_arg(),
                prop_dist_gpu.clone().into_tensor_arg(),
                prop_count_gpu.clone().into_tensor_arg(),
                update_counter_gpu.clone().into_tensor_arg(),
                n as u32,
                MAX_PROPOSALS as u32,
            );
        }

        let counter_data = update_counter_gpu.clone().read(&client)?;
        let updates = counter_data[0] as f64;
        let rate = updates / (n * build_k) as f64;

        if verbose {
            println!(
                "   Iter {}: {} updates (rate={:.6})",
                iter + 1,
                (updates as usize).separate_with_underscores(),
                rate
            );
        }

        if rate < delta as f64 {
            if verbose {
                println!("  Converged after {} iterations", iter + 1);
            }
            converged = true;
            break;
        }
    }

    if verbose {
        println!("  NNDescent iterations: {:.2?}", iter_start.elapsed());
    }

    // ---- Optional 2-hop refinement ----

    if verbose && refine_knn > 0 {
        println!("  Running 2-Hop Refinement Sweep...");
    }

    let refinement_start = Instant::now();
    let cubes_x = 65535u32;
    let cubes_y = (n as u32).div_ceil(cubes_x);

    for sweep in 0..refine_knn {
        unsafe {
            reset_proposals::launch_unchecked::<R>(
                &client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                prop_count_gpu.clone().into_tensor_arg(),
                update_counter_gpu.clone().into_tensor_arg(),
                n as u32,
            );

            two_hop_refinement::launch_unchecked::<T, R>(
                &client,
                CubeCount::Static(cubes_x, cubes_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.clone().into_tensor_arg(),
                norms_gpu.clone().into_tensor_arg(),
                graph_idx_gpu.clone().into_tensor_arg(),
                graph_dist_gpu.clone().into_tensor_arg(),
                prop_idx_gpu.clone().into_tensor_arg(),
                prop_dist_gpu.clone().into_tensor_arg(),
                prop_count_gpu.clone().into_tensor_arg(),
                n as u32,
                MAX_PROPOSALS as u32,
                use_cosine,
                dim_vec,
            );

            merge_proposals::launch_unchecked::<T, R>(
                &client,
                CubeCount::Static(grid_n_x, grid_n_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                graph_idx_gpu.clone().into_tensor_arg(),
                graph_dist_gpu.clone().into_tensor_arg(),
                prop_idx_gpu.clone().into_tensor_arg(),
                prop_dist_gpu.clone().into_tensor_arg(),
                prop_count_gpu.clone().into_tensor_arg(),
                update_counter_gpu.clone().into_tensor_arg(),
                n as u32,
                MAX_PROPOSALS as u32,
            );
        }

        if verbose {
            let counter_data = update_counter_gpu.clone().read(&client)?;
            println!(
                "    2-Hop sweep {}: {} updates",
                sweep + 1,
                counter_data[0].separate_with_underscores()
            );
        }
    }

    if verbose && refine_knn > 0 {
        println!("  Refinement done in: {:.2?}", refinement_start.elapsed());
    }

    // ---- Download graph and extract top-k per row ----

    let nndescent_idx = graph_idx_gpu.read(&client)?;
    let nndescent_dist = graph_dist_gpu.read(&client)?;
    let pid_mask = 0x7FFFFFFFu32;
    let sentinel = SENTINEL_PID;

    let mut knn_graph = vec![(sentinel, <T as num_traits::Float>::max_value()); n * k];

    knn_graph
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(i, slot)| {
            let mut written = 0;
            for j in 0..build_k {
                if written >= k {
                    break;
                }
                let raw = nndescent_idx[i * build_k + j];
                let pid = (raw & pid_mask) as usize;
                if pid < n && pid != i && pid != sentinel {
                    let dist = nndescent_dist[i * build_k + j];
                    slot[written] = (pid, dist);
                    written += 1;
                }
            }
            slot.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        });

    if verbose {
        println!("  Total build time: {:.2?}", start.elapsed());
    }

    Ok(KnnGraphGpu {
        vectors_flat,
        dim,
        n,
        k,
        norms,
        metric,
        knn_graph,
        converged,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::WgpuDevice;
    use cubecl::wgpu::WgpuRuntime;
    use faer::Mat;

    /// Try to create a wgpu device. Returns None if no GPU backend is
    /// available (e.g. headless CI runners).
    fn try_device() -> Option<WgpuDevice> {
        // WgpuDevice::DefaultDevice will panic during kernel launch if
        // no adapter is found. We catch this by attempting a minimal
        // client creation first.
        let device = WgpuDevice::DefaultDevice;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cubecl::wgpu::WgpuRuntime::client(&device);
        }));
        result.ok().map(|_| device)
    }

    #[test]
    fn test_nndescent_gpu_basic() {
        let Some(device) = try_device() else {
            eprintln!("Skipping test: no wgpu backend available");
            return;
        };

        let data = Mat::from_fn(20, 4, |i, j| ((i * 3 + j) as f32) / 10.0);

        let index = NNDescentGpu::<f32, WgpuRuntime>::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(5),
            None,
            Some(10),
            None,
            Some(0.001),
            Some(0.5),
            None,
            42,
            false,
            false,
            device,
        )
        .unwrap();

        assert_eq!(index.nav_graph.len(), 20 * 5);
        for i in 0..20 {
            let nbrs = index.graph_neighbours(i);
            assert_eq!(nbrs.len(), 5);
            for w in nbrs.windows(2) {
                assert!(w[1].1 >= w[0].1);
            }
        }
    }

    #[test]
    fn test_nndescent_gpu_cosine() {
        let Some(device) = try_device() else {
            eprintln!("Skipping test: no wgpu backend available");
            return;
        };

        let data = Mat::from_fn(16, 4, |i, _| (i as f32) + 1.0);

        let index = NNDescentGpu::<f32, WgpuRuntime>::build(
            data.as_ref(),
            Dist::Cosine,
            Some(3),
            None,
            Some(10),
            None,
            Some(0.001),
            Some(0.5),
            None,
            42,
            false,
            false,
            device,
        )
        .unwrap();

        assert_eq!(index.nav_graph.len(), 16 * 3);
        assert!(!index.norms.is_empty());
    }

    #[test]
    fn test_nndescent_gpu_padded_dim() {
        let Some(device) = try_device() else {
            eprintln!("Skipping test: no wgpu backend available");
            return;
        };

        let data = Mat::from_fn(12, 3, |i, j| (i + j) as f32);

        let index = NNDescentGpu::<f32, WgpuRuntime>::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(3),
            None,
            Some(10),
            None,
            Some(0.001),
            Some(0.5),
            None,
            42,
            false,
            false,
            device,
        )
        .unwrap();

        assert_eq!(index.dim, 3);
        assert_eq!(index.nav_graph.len(), 12 * 3);
    }

    #[test]
    fn test_extract_knn() {
        let Some(device) = try_device() else {
            eprintln!("Skipping test: no wgpu backend available");
            return;
        };

        let data = Mat::from_fn(20, 4, |i, j| ((i * 3 + j) as f32) / 10.0);

        let index = NNDescentGpu::<f32, WgpuRuntime>::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(5),
            None,
            Some(10),
            None,
            Some(0.001),
            Some(0.5),
            None,
            42,
            false,
            false,
            device,
        )
        .unwrap();

        let (indices, Some(distances)) = index.extract_knn(true) else {
            panic!("Expected distances");
        };

        assert_eq!(indices.len(), 20);
        assert_eq!(distances.len(), 20);
        for i in 0..20 {
            assert_eq!(indices[i].len(), 5);
            assert_eq!(distances[i].len(), 5);
            // No self-loops
            assert!(!indices[i].contains(&i));
        }

        // Without distances
        let (indices, dists) = index.extract_knn(false);
        assert_eq!(indices.len(), 20);
        assert!(dists.is_none());
    }
}

/// Kernel-level tests targeting individual GPU operations.
///
/// Each test creates its own wgpu device and skips gracefully if no
/// backend is available.
#[cfg(test)]
#[cfg(feature = "gpu-tests")]
mod kernel_tests {
    use super::*;
    use cubecl::wgpu::WgpuDevice;
    use cubecl::wgpu::WgpuRuntime;
    use faer::Mat;

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cubecl::wgpu::WgpuRuntime::client(&device);
        }));
        result.ok().map(|_| device)
    }

    #[cube(launch_unchecked)]
    fn probe_stride<F: Float, N: Size>(vectors: &Tensor<Vector<F, N>>, out: &mut Tensor<u32>) {
        if ABSOLUTE_POS_X == 0u32 {
            out[0usize] = vectors.stride(0) as u32;
            out[1usize] = vectors.shape(1) as u32;
            out[2usize] = vectors.stride(1) as u32;
            out[3usize] = vectors.shape(0) as u32;
        }
    }

    #[test]
    fn test_stride_probe() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);
        let line: usize = LINE_SIZE;

        // 8 vectors of dim 32 -> dim_padded=32, dim_vec=8
        let n = 8usize;
        let dim_padded = 32usize;
        let dim_vec = dim_padded / line;
        let data: Vec<f32> = (0..n * dim_padded).map(|i| i as f32).collect();

        let vectors_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim_padded], &client);
        let out_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32; 4], vec![4], &client);

        unsafe {
            probe_stride::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(1, 1),
                line,
                vectors_gpu.into_tensor_arg(),
                out_gpu.clone().into_tensor_arg(),
            );
        }

        let result = out_gpu.read(&client).unwrap();
        let stride_0 = result[0];
        let shape_1 = result[1];
        let stride_1 = result[2];
        let shape_0 = result[3];

        println!("Tensor [n={n}, dim_padded={dim_padded}] with line_size={line}:");
        println!("  stride(0) = {stride_0}  (expected {dim_vec} in Line units, or {dim_padded} in f32 units)");
        println!("  shape(1)  = {shape_1}  (expected {dim_vec} in Line units)");
        println!("  stride(1) = {stride_1}  (expected 1)");
        println!("  shape(0)  = {shape_0}  (expected {n})");

        // CubeCL reports stride(0) and shape(1) in the *element type* units
        // (f32), not in Line<F> units. This means kernels must use a comptime
        // `dim_lines` parameter for row offsets, not `vectors.stride(0)`.
        assert_eq!(shape_0, n as u32, "shape(0) should be n");
        assert_eq!(
            stride_0, dim_padded as u32,
            "stride(0) should be dim_padded (f32 units)"
        );
    }

    #[cube(launch_unchecked)]
    fn read_vector_via_stride<F: Float, N: Size>(
        vectors: &Tensor<Vector<F, N>>,
        row_idx: u32,
        out: &mut Tensor<F>,
        #[comptime] dim_lines: usize,
    ) {
        if ABSOLUTE_POS_X == 0u32 {
            let off = row_idx as usize * dim_lines;

            let mut d = 0usize;
            while d < dim_lines {
                let line_val = vectors[off + d];
                out[d * 4usize] = line_val[0usize];
                out[d * 4usize + 1usize] = line_val[1usize];
                out[d * 4usize + 2usize] = line_val[2usize];
                out[d * 4usize + 3usize] = line_val[3usize];
                d += 1usize;
            }
        }
    }

    #[test]
    fn test_vector_roundtrip_line() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);
        let line: usize = LINE_SIZE;
        let n = 4usize;
        let dim = 8usize; // 2 lines per row
        let dim_vec = dim / line;

        // Each vector has recognisable values: row i has values i*100+0, i*100+1, ...
        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                data[i * dim + j] = (i * 100 + j) as f32;
            }
        }

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);

        // Read each row and verify
        for row in 0..n {
            // Reset output
            let out_gpu =
                GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![-1.0f32; dim], vec![dim], &client);

            unsafe {
                read_vector_via_stride::launch_unchecked::<f32, WgpuRuntime>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_2d(1, 1),
                    line,
                    vectors_gpu.clone().into_tensor_arg(),
                    row as u32,
                    out_gpu.clone().into_tensor_arg(),
                    dim_vec,
                );
            }

            let result = out_gpu.read(&client).unwrap();
            let expected: Vec<f32> = (0..dim).map(|j| (row * 100 + j) as f32).collect();

            println!("Row {row}: got {:?}", &result[..dim]);
            println!("         exp {:?}", &expected);

            for j in 0..dim {
                if (result[j] - expected[j]).abs() > 1e-6 {
                    eprintln!(
                        "*** MISMATCH at row={row}, col={j}: got {}, expected {} ***",
                        result[j], expected[j]
                    );
                }
            }
            assert_eq!(&result[..dim], &expected[..], "Row {row} data mismatch");
        }
    }

    #[cube(launch_unchecked)]
    fn compute_pairwise_dist<F: Float, N: Size>(
        vectors: &Tensor<Vector<F, N>>,
        norms: &Tensor<F>,
        out_sq_euclid: &mut Tensor<F>,
        out_cosine: &mut Tensor<F>,
        n_pts: u32,
        #[comptime] use_cosine: bool,
        #[comptime] dim_lines: usize,
    ) {
        let idx = ABSOLUTE_POS_X;
        let n_pairs = n_pts * (n_pts - 1u32) / 2u32;
        if idx >= n_pairs {
            terminate!();
        }

        let mut rem = idx;
        let mut i = 0u32;
        let mut step = n_pts - 1u32;
        while rem >= step {
            rem -= step;
            i += 1u32;
            step = n_pts - 1u32 - i;
        }
        let j = i + 1u32 + rem;

        out_sq_euclid[idx as usize] = dist_sq_euclidean(vectors, i, j, dim_lines);
        if use_cosine {
            out_cosine[idx as usize] = dist_cosine(vectors, norms, i, j, dim_lines);
        }
    }

    #[test]
    fn test_gpu_distances_euclidean() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);
        let line: usize = LINE_SIZE;
        let n = 4usize;
        let dim = 8usize;
        let dim_vec = dim / line;

        // Known vectors
        let mut data = vec![0.0f32; n * dim];
        data[0..dim].copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // v0
        data[dim..2 * dim].copy_from_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // v1
        data[2 * dim..3 * dim].copy_from_slice(&[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]); // v2
        data[3 * dim..4 * dim].copy_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]); // v3

        let n_pairs = n * (n - 1) / 2; // 6

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32], vec![1], &client);
        let out_euclid = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; n_pairs],
            vec![n_pairs],
            &client,
        );
        let out_cosine = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; n_pairs],
            vec![n_pairs],
            &client,
        );

        unsafe {
            compute_pairwise_dist::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.into_tensor_arg(),
                norms_gpu.into_tensor_arg(),
                out_euclid.clone().into_tensor_arg(),
                out_cosine.clone().into_tensor_arg(),
                n as u32,
                false,
                dim_vec,
            );
        }

        let euclid = out_euclid.read(&client).unwrap();

        // Expected squared Euclidean distances:
        // (0,1): |v0-v1|^2 = 1+1 = 2
        // (0,2): |v0-v2|^2 = 0+1 = 1
        // (0,3): |v0-v3|^2 = 1+1 = 2
        // (1,2): |v1-v2|^2 = 1+0 = 1
        // (1,3): |v1-v3|^2 = 1+1 = 2
        // (2,3): |v2-v3|^2 = 1+1+1 = 3  (wait: [1,1,0,...,0] vs [0,0,...,0,1])
        let expected = [2.0f32, 1.0, 2.0, 1.0, 2.0, 3.0];

        println!("Squared Euclidean distances:");
        let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        for (k, &(i, j)) in pairs.iter().enumerate() {
            println!(
                "  ({i},{j}): gpu={:.4}  expected={:.4}  match={}",
                euclid[k],
                expected[k],
                (euclid[k] - expected[k]).abs() < 1e-4
            );
        }

        for (k, &exp) in expected.iter().enumerate() {
            assert!(
                (euclid[k] - exp).abs() < 1e-4,
                "Pair {:?}: gpu={}, expected={}",
                pairs[k],
                euclid[k],
                exp
            );
        }
    }

    #[test]
    fn test_gpu_distances_cosine() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);
        let line: usize = LINE_SIZE;
        let n = 4usize;
        let dim = 8usize;
        let dim_vec = dim / line;
        let mut data = vec![0.0f32; n * dim];
        data[0..dim].copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        data[dim..2 * dim].copy_from_slice(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        data[2 * dim..3 * dim].copy_from_slice(&[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        data[3 * dim..4 * dim].copy_from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);

        let norms: Vec<f32> = (0..n)
            .map(|i| {
                let row = &data[i * dim..(i + 1) * dim];
                row.iter().map(|x| x * x).sum::<f32>().sqrt()
            })
            .collect();
        // norms = [1.0, 1.0, sqrt(2), 1.0]

        let n_pairs = n * (n - 1) / 2;
        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&norms, vec![n], &client);
        let out_euclid = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; n_pairs],
            vec![n_pairs],
            &client,
        );
        let out_cosine = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; n_pairs],
            vec![n_pairs],
            &client,
        );

        unsafe {
            compute_pairwise_dist::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.into_tensor_arg(),
                norms_gpu.into_tensor_arg(),
                out_euclid.clone().into_tensor_arg(),
                out_cosine.clone().into_tensor_arg(),
                n as u32,
                true,
                dim_vec,
            );
        }

        let cosine = out_cosine.read(&client).unwrap();

        // Expected cosine distances: 1 - dot/(norm_a * norm_b)
        // (0,1): 1 - 0/(1*1) = 1.0
        // (0,2): 1 - 1/(1*sqrt(2)) = 1 - 0.7071 = 0.2929
        // (0,3): 1 - 0/(1*1) = 1.0
        // (1,2): 1 - 1/(1*sqrt(2)) = 0.2929
        // (1,3): 1 - 0/(1*1) = 1.0
        // (2,3): 1 - 0/(sqrt(2)*1) = 1.0
        let sqrt2 = 2.0f32.sqrt();
        let expected = [1.0, 1.0 - 1.0 / sqrt2, 1.0, 1.0 - 1.0 / sqrt2, 1.0, 1.0];

        println!("Cosine distances:");
        let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        for (k, &(i, j)) in pairs.iter().enumerate() {
            println!(
                "  ({i},{j}): gpu={:.6}  expected={:.6}  match={}",
                cosine[k],
                expected[k],
                (cosine[k] - expected[k]).abs() < 1e-4
            );
            // Check for negative values -- physically impossible
            if cosine[k] < -1e-6 {
                eprintln!("  *** NEGATIVE cosine distance: {} ***", cosine[k]);
            }
        }

        for (k, &exp) in expected.iter().enumerate() {
            assert!(
                (cosine[k] - exp).abs() < 1e-3,
                "Pair {:?}: gpu={}, expected={}",
                pairs[k],
                cosine[k],
                exp
            );
        }
    }

    #[test]
    fn test_local_join_distances() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);
        let line: usize = LINE_SIZE;

        let n = 8usize;
        let dim = 8usize;
        let dim_vec = dim / line;
        let build_k = 4usize;

        // Create vectors with known distances
        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            // Unit vector along dimension i (mod dim)
            data[i * dim + (i % dim)] = 1.0;
        }

        // Norms (all 1.0 for unit vectors)
        let norms = vec![1.0f32; n];

        // Build a simple graph: each node's neighbours are the next build_k nodes (wrap)
        let is_new_bit = 1u32 << 31;
        let mut graph_idx = vec![0u32; n * build_k];
        let mut graph_dist = vec![0.0f32; n * build_k];

        for i in 0..n {
            for j in 0..build_k {
                let nbr = (i + j + 1) % n;
                graph_idx[i * build_k + j] = (nbr as u32) | is_new_bit;
                let a = &data[i * dim..(i + 1) * dim];
                let b = &data[nbr * dim..(nbr + 1) * dim];
                let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                graph_dist[i * build_k + j] = 1.0 - dot; // / (1.0 * 1.0)
            }
            // Sort by distance
            let base = i * build_k;
            let mut pairs: Vec<(u32, f32)> = (0..build_k)
                .map(|j| (graph_idx[base + j], graph_dist[base + j]))
                .collect();
            pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for (j, (idx, dist)) in pairs.into_iter().enumerate() {
                graph_idx[base + j] = idx;
                graph_dist[base + j] = dist;
            }
        }

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&norms, vec![n], &client);
        let graph_idx_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&graph_idx, vec![n, build_k], &client);
        let graph_dist_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&graph_dist, vec![n, build_k], &client);

        // Empty reverse edges (no reverse pass for this test)
        let reverse_idx_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
            &vec![0u32; n * build_k],
            vec![n, build_k],
            &client,
        );
        let reverse_count_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client);

        let max_prop = MAX_PROPOSALS;
        let prop_idx_gpu = GpuTensor::<WgpuRuntime, u32>::empty(vec![n, max_prop], &client);
        let prop_dist_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, max_prop], &client);
        let prop_count_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client);

        let rho_thresh = 65535u32; // rho=1.0, accept all pairs

        unsafe {
            local_join_shared::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(n as u32, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.into_tensor_arg(),
                norms_gpu.into_tensor_arg(),
                graph_idx_gpu.into_tensor_arg(),
                graph_dist_gpu.into_tensor_arg(),
                reverse_idx_gpu.into_tensor_arg(),
                reverse_count_gpu.into_tensor_arg(),
                prop_idx_gpu.clone().into_tensor_arg(),
                prop_dist_gpu.clone().into_tensor_arg(),
                prop_count_gpu.clone().into_tensor_arg(),
                n as u32,
                rho_thresh,
                42u32,
                MAX_PROPOSALS as u32,
                true, // use_cosine
                dim_vec,
                build_k,
            );
        }

        let p_idx = prop_idx_gpu.read(&client).unwrap();
        let p_dist = prop_dist_gpu.read(&client).unwrap();
        let p_count = prop_count_gpu.read(&client).unwrap();

        println!("Local join proposals (n={n}, build_k={build_k}, cosine):");
        let mut any_negative = false;
        let mut any_mismatch = false;

        for node in 0..n {
            let count = (p_count[node] as usize).min(max_prop);
            if count == 0 {
                continue;
            }

            println!("  node {node}: {count} proposals");
            for p in 0..count.min(5) {
                let cand = p_idx[node * max_prop + p] as usize;
                let gpu_dist = p_dist[node * max_prop + p];

                // Recompute on CPU
                let a = &data[node * dim..(node + 1) * dim];
                let b = &data[cand * dim..(cand + 1) * dim];
                let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let cpu_dist = 1.0 - dot / (norms[node] * norms[cand]);

                let ok = (gpu_dist - cpu_dist).abs() < 1e-4;
                println!(
                    "    -> cand {cand}: gpu={:.6e}  cpu={:.6e}  match={ok}",
                    gpu_dist, cpu_dist
                );

                if gpu_dist < -1e-6 {
                    any_negative = true;
                    eprintln!("    *** NEGATIVE distance: {gpu_dist} ***");
                }
                if !ok {
                    any_mismatch = true;
                }
            }
        }

        assert!(
            !any_negative,
            "Negative cosine distances found in local_join proposals"
        );
        assert!(
            !any_mismatch,
            "Distance mismatches found in local_join proposals"
        );
    }

    #[test]
    fn test_merge_proposals() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);
        let n = 4usize;
        let k = 3usize;
        let pid_mask = 0x7FFFFFFFu32;

        // Initial graph: each node has 3 neighbours with known distances
        // Node 0: neighbours [1, 2, 3] with distances [0.1, 0.5, 0.9]
        let graph_idx_data: Vec<u32> = vec![
            1, 2, 3, // node 0
            0, 2, 3, // node 1
            0, 1, 3, // node 2
            0, 1, 2, // node 3
        ];
        let graph_dist_data: Vec<f32> = vec![
            0.1, 0.5, 0.9, // node 0
            0.1, 0.3, 0.8, // node 1
            0.2, 0.3, 0.7, // node 2
            0.2, 0.4, 0.6, // node 3
        ];

        // Proposals for node 0: candidate 2 with dist 0.05 (better than current best!)
        // and candidate 1 with dist 0.08 (duplicate, should be skipped or replaced)
        let mut prop_idx = vec![0u32; n * MAX_PROPOSALS];
        let mut prop_dist = vec![0.0f32; n * MAX_PROPOSALS];
        let mut prop_count = vec![0u32; n];

        // Node 0 gets 2 proposals
        prop_idx[0] = 2; // candidate 2
        prop_dist[0] = 0.05; // very close
        prop_idx[1] = 1; // duplicate of existing
        prop_dist[1] = 0.08;
        prop_count[0] = 2;

        let graph_idx_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&graph_idx_data, vec![n, k], &client);
        let graph_dist_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&graph_dist_data, vec![n, k], &client);
        let prop_idx_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&prop_idx, vec![n, MAX_PROPOSALS], &client);
        let prop_dist_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&prop_dist, vec![n, MAX_PROPOSALS], &client);
        let prop_count_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&prop_count, vec![n], &client);
        let update_counter = GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32], vec![1], &client);

        let grid_n = (n as u32).div_ceil(WORKGROUP_SIZE_X);

        unsafe {
            merge_proposals::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(grid_n, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                graph_idx_gpu.clone().into_tensor_arg(),
                graph_dist_gpu.clone().into_tensor_arg(),
                prop_idx_gpu.into_tensor_arg(),
                prop_dist_gpu.into_tensor_arg(),
                prop_count_gpu.into_tensor_arg(),
                update_counter.clone().into_tensor_arg(),
                n as u32,
                MAX_PROPOSALS as u32,
            );
        }

        let result_idx = graph_idx_gpu.read(&client).unwrap();
        let result_dist = graph_dist_gpu.read(&client).unwrap();
        let updates = update_counter.read(&client).unwrap();

        println!("Merge proposals result:");
        println!("  Total updates: {}", updates[0]);

        for node in 0..n {
            let base = node * k;
            print!("  Node {node}:");
            for j in 0..k {
                let pid = result_idx[base + j] & pid_mask;
                let is_new = result_idx[base + j] & (1u32 << 31) != 0;
                let dist = result_dist[base + j];
                print!("  ({pid}, {dist:.4}{}) ", if is_new { "*" } else { "" });
            }
            println!();
        }

        // Node 0 checks:
        // - Candidate 2 at dist 0.05 should be inserted (better than worst 0.9)
        // - Candidate 1 at dist 0.08 is duplicate (1 already in graph) -> skipped
        // After merge: [2@0.05, 1@0.1, 2@0.5] -- wait, 2 is already in graph at 0.5!
        // So candidate 2 at 0.05 is a duplicate of existing pid=2. Should be skipped!

        // Actually, the existing graph has pid=2 at dist=0.5. The proposal is pid=2
        // at dist=0.05. merge_proposals checks for duplicate PIDs. So 2 is already
        // there, the proposal is skipped.
        // Proposal 1 is pid=1, already at dist=0.1. Also skipped.
        // Result should be unchanged: [1@0.1, 2@0.5, 3@0.9] with all flags cleared.

        let base = 0;
        assert_eq!(
            result_idx[base] & pid_mask,
            1,
            "Node 0, slot 0 should be pid=1"
        );
        assert_eq!(
            result_idx[base + 1] & pid_mask,
            2,
            "Node 0, slot 1 should be pid=2"
        );
        assert_eq!(
            result_idx[base + 2] & pid_mask,
            3,
            "Node 0, slot 2 should be pid=3"
        );

        // Now test with a genuinely new proposal
        #[allow(unused_assignments)]
        let mut prop_idx2 = vec![0u32; n * MAX_PROPOSALS];
        #[allow(unused_assignments)]
        let mut prop_dist2 = vec![0.0f32; n * MAX_PROPOSALS];
        #[allow(unused_assignments)]
        let mut prop_count2 = vec![0u32; n];

        let n2 = 5usize;
        let k2 = 3usize;
        let graph_idx_data2: Vec<u32> = vec![
            1, 2, 3, // node 0
            0, 2, 3, // node 1
            0, 1, 3, // node 2
            0, 1, 2, // node 3
            0, 1, 2, // node 4
        ];
        let graph_dist_data2: Vec<f32> = vec![
            0.1, 0.5, 0.9, // node 0
            0.1, 0.3, 0.8, // node 1
            0.2, 0.3, 0.7, // node 2
            0.2, 0.4, 0.6, // node 3
            0.1, 0.2, 0.3, // node 4
        ];

        prop_idx2 = vec![0u32; n2 * MAX_PROPOSALS];
        prop_dist2 = vec![0.0f32; n2 * MAX_PROPOSALS];
        prop_count2 = vec![0u32; n2];

        prop_idx2[0] = 4; // truly new for node 0
        prop_dist2[0] = 0.3;
        prop_count2[0] = 1;

        let graph_idx_gpu2 =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&graph_idx_data2, vec![n2, k2], &client);
        let graph_dist_gpu2 =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&graph_dist_data2, vec![n2, k2], &client);
        let prop_idx_gpu2 =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&prop_idx2, vec![n2, MAX_PROPOSALS], &client);
        let prop_dist_gpu2 = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &prop_dist2,
            vec![n2, MAX_PROPOSALS],
            &client,
        );
        let prop_count_gpu2 =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&prop_count2, vec![n2], &client);
        let update_counter2 = GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32], vec![1], &client);

        let grid_n2 = (n2 as u32).div_ceil(WORKGROUP_SIZE_X);

        unsafe {
            merge_proposals::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(grid_n2, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                graph_idx_gpu2.clone().into_tensor_arg(),
                graph_dist_gpu2.clone().into_tensor_arg(),
                prop_idx_gpu2.into_tensor_arg(),
                prop_dist_gpu2.into_tensor_arg(),
                prop_count_gpu2.into_tensor_arg(),
                update_counter2.clone().into_tensor_arg(),
                n2 as u32,
                MAX_PROPOSALS as u32,
            );
        }

        let r_idx = graph_idx_gpu2.read(&client).unwrap();
        let r_dist = graph_dist_gpu2.read(&client).unwrap();
        let r_updates = update_counter2.read(&client).unwrap();

        println!("\nMerge with new candidate:");
        println!("  Updates: {}", r_updates[0]);
        let base = 0;
        for j in 0..k2 {
            let pid = r_idx[base + j] & pid_mask;
            let is_new = r_idx[base + j] & (1u32 << 31) != 0;
            let dist = r_dist[base + j];
            println!("  Node 0 slot {j}: pid={pid} dist={dist:.4} new={is_new}");
        }

        assert_eq!(r_updates[0], 1, "Should have exactly 1 update");
        assert_eq!(r_idx[base] & pid_mask, 1, "Slot 0: pid=1 (unchanged)");
        assert_eq!(r_idx[base + 1] & pid_mask, 4, "Slot 1: pid=4 (new)");
        assert!(
            r_idx[base + 1] & (1u32 << 31) != 0,
            "Slot 1 should be flagged new"
        );
        assert_eq!(r_idx[base + 2] & pid_mask, 2, "Slot 2: pid=2 (shifted)");
        assert!((r_dist[base] - 0.1).abs() < 1e-6);
        assert!((r_dist[base + 1] - 0.3).abs() < 1e-6);
        assert!((r_dist[base + 2] - 0.5).abs() < 1e-6);

        // Verify node 0's slot 2 (pid=3 at dist=0.9) was evicted
        for j in 0..k2 {
            assert_ne!(
                r_idx[base + j] & pid_mask,
                3,
                "pid=3 should have been evicted"
            );
        }
    }

    #[test]
    fn test_end_to_end_quality() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let n = 100;
        let dim = 8;
        let k = 5;

        let data_flat: Vec<f32> = (0..n * dim)
            .map(|idx| {
                let i = idx / dim;
                let j = idx % dim;
                let cluster = (i / 10) as f32 * 5.0;
                cluster + (i % 10) as f32 * 0.1 + j as f32 * 0.01
            })
            .collect();

        let data = Mat::from_fn(n, dim, |i, j| data_flat[i * dim + j]);

        // Brute-force ground truth (squared Euclidean)
        let mut ground_truth: Vec<Vec<usize>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let a = &data_flat[i * dim..(i + 1) * dim];
                    let b = &data_flat[j * dim..(j + 1) * dim];
                    let d: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            ground_truth.push(dists.iter().take(k).map(|&(j, _)| j).collect());
        }

        let index = NNDescentGpu::<f32, WgpuRuntime>::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(k),
            None,
            Some(15),
            None,
            Some(0.001),
            Some(0.5),
            None,
            42,
            true,
            false,
            device,
        )
        .unwrap();

        let (knn_indices, _) = index.extract_knn(false);

        let mut total_hits = 0;
        let total_possible = n * k;
        for i in 0..n {
            let gt_set: std::collections::HashSet<usize> =
                ground_truth[i].iter().copied().collect();
            for &idx in &knn_indices[i] {
                if gt_set.contains(&idx) {
                    total_hits += 1;
                }
            }
        }

        let recall = total_hits as f64 / total_possible as f64;
        println!("End-to-end extract recall@{k}: {recall:.4} ({total_hits}/{total_possible})");

        // With proper distance computation, should be > 0.8 at minimum
        assert!(recall > 0.7, "End-to-end recall too low: {recall:.4}");
    }

    #[test]
    fn test_distances_dim32() {
        let Some(device) = try_device() else {
            return;
        };
        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE;
        let n = 16usize;
        let dim = 32usize;
        let dim_vec = dim / line; // 8

        // Vectors where each row has a distinct pattern
        let data: Vec<f32> = (0..n * dim)
            .map(|i| ((i % 7) as f32) * 0.1 + (i / dim) as f32)
            .collect();

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32], vec![1], &client);

        // Compute dist(0, 1) on GPU via dist_sq_euclidean
        // We need a tiny wrapper kernel:
        // (reuse compute_pairwise_dist with n=2 subset, or write a one-off)

        let n_pairs = n * (n - 1) / 2;
        let out_euclid = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; n_pairs],
            vec![n_pairs],
            &client,
        );
        let out_cos = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; n_pairs],
            vec![n_pairs],
            &client,
        );

        unsafe {
            compute_pairwise_dist::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.into_tensor_arg(),
                norms_gpu.into_tensor_arg(),
                out_euclid.clone().into_tensor_arg(),
                out_cos.into_tensor_arg(),
                n as u32,
                false,
                dim_vec,
            );
        }

        let euclid = out_euclid.read(&client).unwrap();

        // Check first pair: dist(0, 1)
        let a = &data[0..dim];
        let b = &data[dim..2 * dim];
        let expected: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();

        println!("dim=32 dist(0,1): gpu={:.6} cpu={:.6}", euclid[0], expected);
        assert!(
            (euclid[0] - expected).abs() < 1e-3,
            "dim=32 distance mismatch: gpu={}, cpu={}",
            euclid[0],
            expected
        );
    }

    /// Minimal reproduction of local_join's shared memory pattern.
    /// Loads two vectors into shared memory using the same indexing
    /// as local_join_shared, computes their distance, and writes
    /// the result plus the raw shared memory contents to output.
    #[cube(launch_unchecked)]
    fn debug_shared_mem_dist<F: Float, N: Size>(
        vectors: &Tensor<Vector<F, N>>,
        norms: &Tensor<F>,
        pid_a: u32,
        pid_b: u32,
        out_dist: &mut Tensor<F>,
        out_raw: &mut Tensor<F>,
        #[comptime] _max_proposals: u32,
        #[comptime] use_cosine: bool,
        #[comptime] dim_lines: usize,
        #[comptime] build_k: usize,
    ) {
        let tx = UNIT_POS_X;
        let max_cands_comp = build_k * 2usize;
        let dim_scalars = dim_lines * 4usize;

        let mut shared_vecs = SharedMemory::<F>::new(max_cands_comp * dim_scalars);
        let mut shared_pids = SharedMemory::<u32>::new(max_cands_comp);

        let mut shared_norms = SharedMemory::<F>::new(max_cands_comp);

        if tx == 0u32 {
            shared_pids[0usize] = pid_a;
            shared_pids[1usize] = pid_b;
            if use_cosine {
                shared_norms[0usize] = norms[pid_a as usize];
                shared_norms[1usize] = norms[pid_b as usize];
            }
        }
        sync_cube();

        let total_scalars = 2usize * dim_scalars;
        let mut idx_load = tx as usize;
        while idx_load < total_scalars {
            let n_idx = idx_load / dim_scalars;
            let s_idx = idx_load % dim_scalars;
            let line_idx = s_idx / 4usize;
            let lane = s_idx % 4usize;
            let pid = shared_pids[n_idx];
            let vec_offset = pid as usize * dim_lines + line_idx;
            let line_val = vectors[vec_offset];
            shared_vecs[idx_load] = line_val[lane];
            idx_load += WORKGROUP_SIZE_X as usize;
        }
        sync_cube();

        if tx == 0u32 {
            let mut sum = F::new(0.0_f32);
            let mut s = 0usize;
            while s < dim_scalars {
                let va = shared_vecs[s];
                let vb = shared_vecs[dim_scalars + s];
                if use_cosine {
                    sum += va * vb;
                } else {
                    let diff = va - vb;
                    sum += diff * diff;
                }
                s += 1usize;
            }

            let dist = if use_cosine {
                F::new(1.0_f32) - (sum / (shared_norms[0usize] * shared_norms[1usize]))
            } else {
                sum
            };

            out_dist[0usize] = dist;
            out_dist[1usize] = sum;
            if use_cosine {
                out_dist[2usize] = shared_norms[0usize];
                out_dist[3usize] = shared_norms[1usize];
            }

            let mut i = 0usize;
            while i < total_scalars {
                out_raw[i] = shared_vecs[i];
                i += 1usize;
            }
        }
    }

    #[test]
    fn test_shared_mem_local_join_pattern() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE;

        // Production dimensions: dim=32 (dim_lines=8), build_k=30
        let n = 100usize;
        let dim = 32usize;
        let dim_vec = dim / line; // 8
        let build_k = 30usize;

        // Create recognisable data
        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                data[i * dim + j] = (i * 1000 + j) as f32;
            }
        }

        let norms: Vec<f32> = (0..n)
            .map(|i| {
                let row = &data[i * dim..(i + 1) * dim];
                row.iter().map(|x| x * x).sum::<f32>().sqrt()
            })
            .collect();

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&norms, vec![n], &client);
        let out_dist = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32; 4], vec![4], &client);
        let out_raw = GpuTensor::<WgpuRuntime, f32>::from_slice(
            &vec![0.0f32; 2 * dim],
            vec![2 * dim],
            &client,
        );

        // Test distance between rows 0 and 1
        let pid_a = 0u32;
        let pid_b = 1u32;

        unsafe {
            debug_shared_mem_dist::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                line,
                vectors_gpu.clone().into_tensor_arg(),
                norms_gpu.clone().into_tensor_arg(),
                pid_a,
                pid_b,
                out_dist.clone().into_tensor_arg(),
                out_raw.clone().into_tensor_arg(),
                MAX_PROPOSALS as u32, // same comptime order as local_join
                false,                // euclidean first
                dim_vec,              // dim_lines
                build_k,              // build_k
            );
        }

        let dist_result = out_dist.read(&client).unwrap();
        let raw = out_raw.read(&client).unwrap();

        // Check raw shared memory contents
        let expected_a: Vec<f32> = (0..dim).map(|j| j as f32).collect();
        let expected_b: Vec<f32> = (0..dim).map(|j| (1000 + j) as f32).collect();

        println!("Shared mem vec A (first 8): {:?}", &raw[..8]);
        println!("Expected vec A  (first 8): {:?}", &expected_a[..8]);
        println!("Shared mem vec B (first 8): {:?}", &raw[dim..dim + 8]);
        println!("Expected vec B  (first 8): {:?}", &expected_b[..8]);

        let vec_a_ok = (0..dim).all(|j| (raw[j] - expected_a[j]).abs() < 1e-4);
        let vec_b_ok = (0..dim).all(|j| (raw[dim + j] - expected_b[j]).abs() < 1e-4);

        println!("Vec A correct: {vec_a_ok}");
        println!("Vec B correct: {vec_b_ok}");

        // Check distance
        let cpu_dist: f32 = expected_a
            .iter()
            .zip(&expected_b)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let gpu_dist = dist_result[0];

        println!(
            "GPU dist: {gpu_dist:.4}  CPU dist: {cpu_dist:.4}  match: {}",
            (gpu_dist - cpu_dist).abs() < 1e-2
        );

        assert!(vec_a_ok, "Vector A in shared memory is wrong");
        assert!(vec_b_ok, "Vector B in shared memory is wrong");
        assert!(
            (gpu_dist - cpu_dist).abs() < 1e-2,
            "Distance mismatch: gpu={gpu_dist}, cpu={cpu_dist}"
        );
    }

    ///////////////
    // Debugging //
    ///////////////

    // Note to self, there has been some very weird bugs since cubecl 0.10.0
    // with comparisons on u32 -> Claude and I are debugged this stuff...
    // Need to raise with the authors

    // #[test]
    // fn test_nndescent_iteration_produces_updates() {
    //     let Some(device) = try_device() else {
    //         eprintln!("Skipping test: no wgpu backend available");
    //         return;
    //     };
    //     let client = WgpuRuntime::client(&device);

    //     // Mirror the production failure: dim=32, build_k=45, rho=0.5, euclidean.
    //     let n = 512usize;
    //     let dim = 32usize;
    //     let build_k = 45usize;
    //     let line = LINE_SIZE;
    //     let dim_padded = dim.next_multiple_of(line);
    //     let dim_vec = dim_padded / line;
    //     let seed = 42u32;
    //     let use_cosine = false;
    //     let rho_thresh = 65535u32;

    //     let data: Vec<f32> = (0..n * dim_padded)
    //         .map(|i| {
    //             let x = (i as u32).wrapping_mul(2654435761) ^ 0x9E3779B9;
    //             ((x % 1000) as f32) / 1000.0
    //         })
    //         .collect();

    //     let vectors_gpu =
    //         GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim_padded], &client);
    //     let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32], vec![1], &client);

    //     let graph_idx_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
    //         &vec![0x7FFFFFFFu32; n * build_k],
    //         vec![n, build_k],
    //         &client,
    //     );
    //     let graph_dist_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
    //         &vec![f32::MAX; n * build_k],
    //         vec![n, build_k],
    //         &client,
    //     );

    //     let max_prop = MAX_PROPOSALS;
    //     let prop_idx_gpu = GpuTensor::<WgpuRuntime, u32>::empty(vec![n, max_prop], &client);
    //     let prop_dist_gpu = GpuTensor::<WgpuRuntime, f32>::empty(vec![n, max_prop], &client);
    //     let prop_count_gpu =
    //         GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client);
    //     let update_counter_gpu =
    //         GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32], vec![1], &client);
    //     let reverse_idx_gpu = GpuTensor::<WgpuRuntime, u32>::empty(vec![n, build_k], &client);
    //     let reverse_count_gpu =
    //         GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client);

    //     let (grid_n_x, grid_n_y) = grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X));

    //     // 1: random init
    //     unsafe {
    //         init_random_graph::launch_unchecked::<f32, WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(grid_n_x, grid_n_y, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             line,
    //             vectors_gpu.clone().into_tensor_arg(),
    //             norms_gpu.clone().into_tensor_arg(),
    //             graph_idx_gpu.clone().into_tensor_arg(),
    //             graph_dist_gpu.clone().into_tensor_arg(),
    //             n as u32,
    //             seed,
    //             use_cosine,
    //             dim_vec,
    //         );
    //     }

    //     // 1c: mark all new (as build() does)
    //     let total_entries = (n * build_k) as u32;
    //     let (mark_x, mark_y) = grid_2d(total_entries.div_ceil(WORKGROUP_SIZE_X));
    //     unsafe {
    //         mark_all_new::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(mark_x, mark_y, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             graph_idx_gpu.clone().into_tensor_arg(),
    //             total_entries,
    //         );
    //     }

    //     {
    //         let g = graph_idx_gpu.clone().read(&client).unwrap();
    //         let marked = g.iter().filter(|&&e| e >= (1u32 << 31)).count();
    //         println!("entries with IS_NEW set: {marked} / {}", g.len());
    //         assert!(marked > 0, "mark_all_new did not set the new flag");
    //     }

    //     // 2: one iteration -- reset, reverse, local join
    //     unsafe {
    //         reset_proposals::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(grid_n_x, grid_n_y, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             prop_count_gpu.clone().into_tensor_arg(),
    //             update_counter_gpu.clone().into_tensor_arg(),
    //             n as u32,
    //         );
    //         reset_proposals::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(grid_n_x, grid_n_y, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             reverse_count_gpu.clone().into_tensor_arg(),
    //             update_counter_gpu.clone().into_tensor_arg(),
    //             n as u32,
    //         );
    //         build_reverse_candidates::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(grid_n_x, grid_n_y, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             graph_idx_gpu.clone().into_tensor_arg(),
    //             reverse_idx_gpu.clone().into_tensor_arg(),
    //             reverse_count_gpu.clone().into_tensor_arg(),
    //             n as u32,
    //             build_k as u32,
    //         );
    //     }

    //     let cubes_x = (n as u32).min(65535);
    //     let cubes_y = (n as u32).div_ceil(cubes_x);
    //     unsafe {
    //         local_join_shared::launch_unchecked::<f32, WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(cubes_x, cubes_y, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             line,
    //             vectors_gpu.clone().into_tensor_arg(),
    //             norms_gpu.clone().into_tensor_arg(),
    //             graph_idx_gpu.clone().into_tensor_arg(),
    //             graph_dist_gpu.clone().into_tensor_arg(),
    //             reverse_idx_gpu.clone().into_tensor_arg(),
    //             reverse_count_gpu.clone().into_tensor_arg(),
    //             prop_idx_gpu.clone().into_tensor_arg(),
    //             prop_dist_gpu.clone().into_tensor_arg(),
    //             prop_count_gpu.clone().into_tensor_arg(),
    //             n as u32,
    //             rho_thresh,
    //             seed,
    //             MAX_PROPOSALS as u32,
    //             use_cosine,
    //             dim_vec,
    //             build_k,
    //         );
    //     }

    //     // Stage diagnostics (merge does not touch these, safe to read now)
    //     let rev_total: u64 = reverse_count_gpu
    //         .clone()
    //         .read(&client)
    //         .unwrap()
    //         .iter()
    //         .map(|&c| c as u64)
    //         .sum();
    //     let prop_total: u64 = prop_count_gpu
    //         .clone()
    //         .read(&client)
    //         .unwrap()
    //         .iter()
    //         .map(|&c| c as u64)
    //         .sum();

    //     unsafe {
    //         merge_proposals::launch_unchecked::<f32, WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(grid_n_x, grid_n_y, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             graph_idx_gpu.clone().into_tensor_arg(),
    //             graph_dist_gpu.clone().into_tensor_arg(),
    //             prop_idx_gpu.clone().into_tensor_arg(),
    //             prop_dist_gpu.clone().into_tensor_arg(),
    //             prop_count_gpu.clone().into_tensor_arg(),
    //             update_counter_gpu.clone().into_tensor_arg(),
    //             n as u32,
    //             MAX_PROPOSALS as u32,
    //         );
    //     }

    //     let updates = update_counter_gpu.clone().read(&client).unwrap()[0];

    //     println!("reverse edges total : {rev_total}");
    //     println!("proposals total     : {prop_total}");
    //     println!("merge updates       : {updates}");

    //     assert!(
    //         rev_total > 0,
    //         "build_reverse_candidates produced no reverse edges"
    //     );
    //     assert!(prop_total > 0, "local_join_shared produced no proposals");
    //     assert!(
    //         updates > 0,
    //         "merge_proposals applied no updates on iteration 1"
    //     );
    // }

    // #[cube(launch_unchecked)]
    // fn debug_local_join_node<F: Float, N: Size>(
    //     vectors: &Tensor<Vector<F, N>>,
    //     graph_idx: &Tensor<u32>,
    //     graph_dist: &Tensor<F>,
    //     reverse_idx: &Tensor<u32>,
    //     reverse_count: &Tensor<u32>,
    //     out_u32: &mut Tensor<u32>,
    //     out_f32: &mut Tensor<F>,
    //     node: u32,
    //     rho_thresh: u32,
    //     iter_seed: u32,
    //     #[comptime] dim_lines: usize,
    //     #[comptime] build_k: usize,
    // ) {
    //     if UNIT_POS_X != 0u32 {
    //         terminate!();
    //     }

    //     let k = graph_idx.shape(1usize) as u32;
    //     let pid_mask = 0x7FFFFFFFu32;
    //     let is_new_bit = 1u32 << 31;
    //     let max_cands = build_k * 2usize;

    //     let mut pids = SharedMemory::<u32>::new(max_cands);
    //     let mut isnew = SharedMemory::<u32>::new(max_cands);

    //     let rc = reverse_count[node as usize];
    //     let rev_k = if rc > k { k } else { rc };
    //     let raw_total = k + rev_k;

    //     let mut t = 0u32;
    //     while t < raw_total {
    //         let entry = if t < k {
    //             graph_idx[(node * k + t) as usize]
    //         } else {
    //             reverse_idx[(node * k + t - k) as usize]
    //         };
    //         pids[t as usize] = entry & pid_mask;
    //         isnew[t as usize] = if entry >= is_new_bit { 1u32 } else { 0u32 };
    //         t += 1u32;
    //     }

    //     let mut write = 0u32;
    //     let mut has_new = 0u32;
    //     let mut read = 0u32;
    //     while read < raw_total {
    //         let hash = entry_hash(node, read, iter_seed);
    //         if (hash & 0xFFFFu32) < rho_thresh {
    //             pids[write as usize] = pids[read as usize];
    //             isnew[write as usize] = isnew[read as usize];
    //             if isnew[read as usize] != 0u32 {
    //                 has_new = 1u32;
    //             }
    //             write += 1u32;
    //         }
    //         read += 1u32;
    //     }
    //     let total_cands = write;

    //     let mut gate_pass = 0u32;
    //     let mut below = 0u32;
    //     let mut min_dist = F::new(3.0e38);
    //     let mut first_pid_i = 0u32;
    //     let mut first_pid_j = 0u32;
    //     let mut first_dist = F::new(-1.0);
    //     let mut first_ti = F::new(-1.0);
    //     let mut recorded: u32 = 0u32;

    //     let mut a = 0u32;
    //     while a < total_cands {
    //         let mut b = a + 1u32;
    //         while b < total_cands {
    //             let pid_i = pids[a as usize];
    //             let pid_j = pids[b as usize];
    //             if pid_i != pid_j {
    //                 gate_pass += 1u32;
    //                 let off_i = pid_i as usize * dim_lines;
    //                 let off_j = pid_j as usize * dim_lines;
    //                 let mut sum = F::new(0.0_f32);
    //                 let mut l = 0usize;
    //                 while l < dim_lines {
    //                     let d = vectors[off_i + l] - vectors[off_j + l];
    //                     let sq = d * d;
    //                     #[unroll]
    //                     for lane in 0..LINE_SIZE {
    //                         sum += sq[lane];
    //                     }
    //                     l += 1usize;
    //                 }
    //                 let ti = graph_dist[pid_i as usize * k as usize + k as usize - 1usize];
    //                 if sum < min_dist {
    //                     min_dist = sum;
    //                 }
    //                 if sum < ti {
    //                     below += 1u32;
    //                 }
    //                 if recorded == 0u32 {
    //                     first_pid_i = pid_i;
    //                     first_pid_j = pid_j;
    //                     first_dist = sum;
    //                     first_ti = ti;
    //                     recorded = 1u32;
    //                 }
    //             }
    //             b += 1u32;
    //         }
    //         a += 1u32;
    //     }

    //     out_u32[0usize] = total_cands;
    //     out_u32[1usize] = raw_total;
    //     out_u32[2usize] = has_new;
    //     out_u32[3usize] = gate_pass;
    //     out_u32[4usize] = below;
    //     out_u32[5usize] = first_pid_i;
    //     out_u32[6usize] = first_pid_j;
    //     out_f32[0usize] = first_dist;
    //     out_f32[1usize] = first_ti;
    //     out_f32[2usize] = min_dist;
    // }

    // #[test]
    // fn test_debug_local_join_node() {
    //     let Some(device) = try_device() else {
    //         eprintln!("Skipping test: no wgpu backend available");
    //         return;
    //     };
    //     let client = WgpuRuntime::client(&device);

    //     let n = 512usize;
    //     let dim = 32usize;
    //     let build_k = 45usize;
    //     let line = LINE_SIZE;
    //     let dim_padded = dim.next_multiple_of(line);
    //     let dim_vec = dim_padded / line;
    //     let seed = 42u32;

    //     let data: Vec<f32> = (0..n * dim_padded)
    //         .map(|i| {
    //             let x = (i as u32).wrapping_mul(2654435761) ^ 0x9E3779B9;
    //             ((x % 1000) as f32) / 1000.0
    //         })
    //         .collect();

    //     let vectors_gpu =
    //         GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim_padded], &client);
    //     let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32], vec![1], &client);
    //     let graph_idx_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
    //         &vec![0x7FFFFFFFu32; n * build_k],
    //         vec![n, build_k],
    //         &client,
    //     );
    //     let graph_dist_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
    //         &vec![f32::MAX; n * build_k],
    //         vec![n, build_k],
    //         &client,
    //     );
    //     let reverse_idx_gpu = GpuTensor::<WgpuRuntime, u32>::empty(vec![n, build_k], &client);
    //     let reverse_count_gpu =
    //         GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![0u32; n], vec![n], &client);
    //     let update_counter_gpu =
    //         GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32], vec![1], &client);

    //     let (gx, gy) = grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X));

    //     unsafe {
    //         init_random_graph::launch_unchecked::<f32, WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(gx, gy, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             line,
    //             vectors_gpu.clone().into_tensor_arg(),
    //             norms_gpu.clone().into_tensor_arg(),
    //             graph_idx_gpu.clone().into_tensor_arg(),
    //             graph_dist_gpu.clone().into_tensor_arg(),
    //             n as u32,
    //             seed,
    //             false,
    //             dim_vec,
    //         );
    //         let total_entries = (n * build_k) as u32;
    //         let (mx, my) = grid_2d(total_entries.div_ceil(WORKGROUP_SIZE_X));
    //         mark_all_new::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(mx, my, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             graph_idx_gpu.clone().into_tensor_arg(),
    //             total_entries,
    //         );
    //         reset_proposals::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(gx, gy, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             reverse_count_gpu.clone().into_tensor_arg(),
    //             update_counter_gpu.clone().into_tensor_arg(),
    //             n as u32,
    //         );
    //         build_reverse_candidates::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(gx, gy, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             graph_idx_gpu.clone().into_tensor_arg(),
    //             reverse_idx_gpu.clone().into_tensor_arg(),
    //             reverse_count_gpu.clone().into_tensor_arg(),
    //             n as u32,
    //             build_k as u32,
    //         );
    //     }

    //     for node in [0u32, 1, 7, 100] {
    //         let out_u32 = GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32; 7], vec![7], &client);
    //         let out_f32 = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32; 3], vec![3], &client);
    //         unsafe {
    //             debug_local_join_node::launch_unchecked::<f32, WgpuRuntime>(
    //                 &client,
    //                 CubeCount::Static(1, 1, 1),
    //                 CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //                 line,
    //                 vectors_gpu.clone().into_tensor_arg(),
    //                 graph_idx_gpu.clone().into_tensor_arg(),
    //                 graph_dist_gpu.clone().into_tensor_arg(),
    //                 reverse_idx_gpu.clone().into_tensor_arg(),
    //                 reverse_count_gpu.clone().into_tensor_arg(),
    //                 out_u32.clone().into_tensor_arg(),
    //                 out_f32.clone().into_tensor_arg(),
    //                 node,
    //                 65535u32,
    //                 seed,
    //                 dim_vec,
    //                 build_k,
    //             );
    //         }
    //         let u = out_u32.read(&client).unwrap();
    //         let f = out_f32.read(&client).unwrap();
    //         println!(
    //             "node {node}: total_cands={} raw_total={} has_new={} gate_pass={} below_thresh={} | first pair {}->{} dist={:.4} thresh_i={:.4} min_dist={:.4}",
    //             u[0], u[1], u[2], u[3], u[4], u[5], u[6], f[0], f[1], f[2]
    //         );
    //     }
    // }

    // #[cube(launch_unchecked)]
    // fn debug_msb_probe(graph_idx: &Tensor<u32>, out: &mut Tensor<u32>) {
    //     if UNIT_POS_X != 0u32 {
    //         terminate!();
    //     }
    //     let bit = 1u32 << 31;
    //     let e0 = graph_idx[0usize];
    //     let e1 = graph_idx[1usize];
    //     out[0usize] = bit;
    //     out[1usize] = e0;
    //     out[2usize] = e1;
    //     out[3usize] = e0 & bit;
    //     out[4usize] = if e0 >= bit { 1u32 } else { 0u32 };
    //     out[5usize] = if (e0 & bit) != 0u32 { 1u32 } else { 0u32 };
    // }

    // #[test]
    // fn test_debug_msb_probe() {
    //     let Some(device) = try_device() else {
    //         eprintln!("Skipping: no wgpu backend");
    //         return;
    //     };
    //     let client = WgpuRuntime::client(&device);

    //     let n = 512usize;
    //     let dim = 32usize;
    //     let build_k = 45usize;
    //     let line = LINE_SIZE;
    //     let dim_padded = dim.next_multiple_of(line);
    //     let dim_vec = dim_padded / line;

    //     let data: Vec<f32> = (0..n * dim_padded)
    //         .map(|i| {
    //             let x = (i as u32).wrapping_mul(2654435761) ^ 0x9E3779B9;
    //             ((x % 1000) as f32) / 1000.0
    //         })
    //         .collect();
    //     let vectors_gpu =
    //         GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim_padded], &client);
    //     let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32], vec![1], &client);
    //     let graph_idx_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
    //         &vec![0x7FFFFFFFu32; n * build_k],
    //         vec![n, build_k],
    //         &client,
    //     );
    //     let graph_dist_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
    //         &vec![f32::MAX; n * build_k],
    //         vec![n, build_k],
    //         &client,
    //     );

    //     let (gx, gy) = grid_2d((n as u32).div_ceil(WORKGROUP_SIZE_X));
    //     unsafe {
    //         init_random_graph::launch_unchecked::<f32, WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(gx, gy, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             line,
    //             vectors_gpu.clone().into_tensor_arg(),
    //             norms_gpu.clone().into_tensor_arg(),
    //             graph_idx_gpu.clone().into_tensor_arg(),
    //             graph_dist_gpu.clone().into_tensor_arg(),
    //             n as u32,
    //             42u32,
    //             false,
    //             dim_vec,
    //         );
    //         let total = (n * build_k) as u32;
    //         let (mx, my) = grid_2d(total.div_ceil(WORKGROUP_SIZE_X));
    //         mark_all_new::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(mx, my, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             graph_idx_gpu.clone().into_tensor_arg(),
    //             total,
    //         );
    //     }

    //     let host = graph_idx_gpu.clone().read(&client).unwrap();
    //     let out = GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32; 6], vec![6], &client);
    //     unsafe {
    //         debug_msb_probe::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(1, 1, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             graph_idx_gpu.clone().into_tensor_arg(),
    //             out.clone().into_tensor_arg(),
    //         );
    //     }
    //     let o = out.read(&client).unwrap();
    //     println!("host    graph_idx[0] = 0x{:08X}", host[0]);
    //     println!("kernel  is_new_bit   = 0x{:08X}", o[0]);
    //     println!("kernel  graph_idx[0] = 0x{:08X}", o[1]);
    //     println!("kernel  graph_idx[1] = 0x{:08X}", o[2]);
    //     println!("kernel  e0 & bit     = 0x{:08X}", o[3]);
    //     println!("kernel  e0 >= bit    = {}", o[4]);
    //     println!("kernel  (e0&bit)!=0  = {}", o[5]);
    // }

    // #[cube(launch_unchecked)]
    // fn debug_shift_probe(graph_idx: &Tensor<u32>, out: &mut Tensor<u32>) {
    //     if UNIT_POS_X != 0u32 {
    //         terminate!();
    //     }
    //     let e0 = graph_idx[0usize]; // 0x80000082, MSB set
    //     let sentinel = graph_idx[1usize]; // 0x7FFFFFFF, MSB clear
    //     out[0usize] = e0 >> 31;
    //     out[1usize] = sentinel >> 31;
    //     out[2usize] = if (e0 >> 31) != 0u32 { 1u32 } else { 0u32 };
    //     out[3usize] = if (sentinel >> 31) != 0u32 { 1u32 } else { 0u32 };
    //     out[4usize] = if 1u32 != 0u32 { 1u32 } else { 0u32 };
    //     out[5usize] = if e0 > 0x7FFFFFFFu32 { 1u32 } else { 0u32 };
    // }

    // #[test]
    // fn test_debug_shift_probe() {
    //     let Some(device) = try_device() else {
    //         eprintln!("Skipping: no wgpu backend");
    //         return;
    //     };
    //     let client = WgpuRuntime::client(&device);
    //     let graph_idx_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
    //         &[0x80000082u32, 0x7FFFFFFFu32],
    //         vec![2],
    //         &client,
    //     );
    //     let out = GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32; 6], vec![6], &client);
    //     unsafe {
    //         debug_shift_probe::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(1, 1, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             graph_idx_gpu.clone().into_tensor_arg(),
    //             out.clone().into_tensor_arg(),
    //         );
    //     }
    //     let o = out.read(&client).unwrap();
    //     println!("e0>>31           = {}  (expect 1)", o[0]);
    //     println!("sentinel>>31     = {}  (expect 0)", o[1]);
    //     println!("(e0>>31)!=0      = {}  (expect 1)", o[2]);
    //     println!("(sentinel>>31)!=0= {}  (expect 0)", o[3]);
    //     println!("1!=0             = {}  (expect 1)", o[4]);
    //     println!("e0 > 0x7FFFFFFF  = {}  (expect 1)", o[5]);
    // }

    // #[cube(launch_unchecked)]
    // fn debug_flag_idioms(vals: &Tensor<u32>, out: &mut Tensor<u32>) {
    //     if UNIT_POS_X != 0u32 {
    //         terminate!();
    //     }
    //     let e0 = vals[0usize]; // 0x80000082, new
    //     let sentinel = vals[1usize]; // 0x7FFFFFFF, old

    //     let mut sh = SharedMemory::<u32>::new(2usize);
    //     sh[0usize] = e0 >> 31;
    //     sh[1usize] = sentinel >> 31;
    //     sync_cube();
    //     let r0 = sh[0usize];
    //     let r1 = sh[1usize];

    //     out[0usize] = if r0 != 0u32 { 1u32 } else { 0u32 }; // expect 1
    //     out[1usize] = if r1 != 0u32 { 1u32 } else { 0u32 }; // expect 0
    //     out[2usize] = if r0 == 1u32 { 1u32 } else { 0u32 }; // expect 1
    //     out[3usize] = if r1 == 1u32 { 1u32 } else { 0u32 }; // expect 0
    //     out[4usize] = r0; // expect 1
    //     out[5usize] = r1; // expect 0
    // }

    // #[test]
    // fn test_debug_flag_idioms() {
    //     let Some(device) = try_device() else {
    //         eprintln!("Skipping: no wgpu backend");
    //         return;
    //     };
    //     let client = WgpuRuntime::client(&device);
    //     let vals = GpuTensor::<WgpuRuntime, u32>::from_slice(
    //         &[0x80000082u32, 0x7FFFFFFFu32],
    //         vec![2],
    //         &client,
    //     );
    //     let out = GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32; 6], vec![6], &client);
    //     unsafe {
    //         debug_flag_idioms::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(1, 1, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             vals.clone().into_tensor_arg(),
    //             out.clone().into_tensor_arg(),
    //         );
    //     }
    //     let o = out.read(&client).unwrap();
    //     println!("r0!=0  = {}  (expect 1)", o[0]);
    //     println!("r1!=0  = {}  (expect 0)", o[1]);
    //     println!("r0==1  = {}  (expect 1)", o[2]);
    //     println!("r1==1  = {}  (expect 0)", o[3]);
    //     println!("r0     = {}  (expect 1)", o[4]);
    //     println!("r1     = {}  (expect 0)", o[5]);
    // }

    // #[cube(launch_unchecked)]
    // fn debug_launder(input: &Tensor<u32>, out: &mut Tensor<u32>) {
    //     let tx = UNIT_POS_X;
    //     let mut sh = SharedMemory::<u32>::new(32usize);
    //     sh[tx as usize] = input[tx as usize] >> 31; // shift-store, no comparison
    //     sync_cube(); // legal: every thread reaches it
    //     if tx == 0u32 {
    //         let r0 = sh[0usize]; // fresh load, from input[0]=0x80000082 -> 1
    //         let r1 = sh[1usize]; // fresh load, from input[1]=0x7FFFFFFF -> 0
    //         out[0usize] = r0; // expect 1
    //         out[1usize] = r1; // expect 0
    //         out[2usize] = if r0 != 0u32 { 1u32 } else { 0u32 }; // expect 1
    //         out[3usize] = if r1 != 0u32 { 1u32 } else { 0u32 }; // expect 0
    //     }
    // }

    // #[test]
    // fn test_debug_launder() {
    //     let Some(device) = try_device() else {
    //         eprintln!("Skipping: no wgpu backend");
    //         return;
    //     };
    //     let client = WgpuRuntime::client(&device);
    //     let mut input = vec![0u32; 32];
    //     input[0] = 0x80000082;
    //     input[1] = 0x7FFFFFFF;
    //     let input_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&input, vec![32], &client);
    //     let out = GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32; 4], vec![4], &client);
    //     unsafe {
    //         debug_launder::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(1, 1, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             input_gpu.clone().into_tensor_arg(),
    //             out.clone().into_tensor_arg(),
    //         );
    //     }
    //     let o = out.read(&client).unwrap();
    //     println!("r0      = {}  (expect 1)", o[0]);
    //     println!("r1      = {}  (expect 0)", o[1]);
    //     println!("r0!=0   = {}  (expect 1)", o[2]);
    //     println!("r1!=0   = {}  (expect 0)", o[3]);
    // }

    // #[cube(launch_unchecked)]
    // fn debug_arith_gate(input: &Tensor<u32>, out: &mut Tensor<u32>) {
    //     let tx = UNIT_POS_X;
    //     let mut sh = SharedMemory::<u32>::new(32usize);
    //     sh[tx as usize] = input[tx as usize] >> 31;
    //     sync_cube();
    //     if tx == 0u32 {
    //         let f_new = sh[0usize]; // from 0x80000082 -> 1
    //         let f_old = sh[1usize]; // from 0x7FFFFFFF -> 0

    //         // OR accumulation (has_new pattern)
    //         let mut acc: u32 = 0u32;
    //         acc = acc | f_new;
    //         out[0usize] = acc; // expect 1
    //         out[1usize] = 0u32 | f_old; // expect 0

    //         // gate as OR of two flags, then used as a count
    //         let gate_nn = f_new | f_new; // expect 1
    //         let gate_no = f_new | f_old; // expect 1
    //         let gate_oo = f_old | f_old; // expect 0
    //         out[2usize] = gate_nn;
    //         out[3usize] = gate_no;
    //         out[4usize] = gate_oo;

    //         // arithmetic selection instead of an if: dist masked by gate
    //         // (does multiplying a normal int by a tainted 0/1 work?)
    //         let masked = 7u32 * gate_no; // expect 7
    //         out[5usize] = masked;
    //         let masked0 = 7u32 * gate_oo; // expect 0
    //         out[6usize] = masked0;
    //     }
    // }

    // #[test]
    // fn test_debug_arith_gate() {
    //     let Some(device) = try_device() else {
    //         eprintln!("Skipping: no wgpu backend");
    //         return;
    //     };
    //     let client = WgpuRuntime::client(&device);
    //     let mut input = vec![0u32; 32];
    //     input[0] = 0x80000082;
    //     input[1] = 0x7FFFFFFF;
    //     let input_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&input, vec![32], &client);
    //     let out = GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32; 7], vec![7], &client);
    //     unsafe {
    //         debug_arith_gate::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(1, 1, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             input_gpu.clone().into_tensor_arg(),
    //             out.clone().into_tensor_arg(),
    //         );
    //     }
    //     let o = out.read(&client).unwrap();
    //     println!("acc(0|f_new)     = {}  (expect 1)", o[0]);
    //     println!("0|f_old          = {}  (expect 0)", o[1]);
    //     println!("gate new|new     = {}  (expect 1)", o[2]);
    //     println!("gate new|old     = {}  (expect 1)", o[3]);
    //     println!("gate old|old     = {}  (expect 0)", o[4]);
    //     println!("7 * (new|old)    = {}  (expect 7)", o[5]);
    //     println!("7 * (old|old)    = {}  (expect 0)", o[6]);
    // }

    // #[cube(launch_unchecked)]
    // fn debug_fix_constructs(input: &Tensor<u32>, out: &mut Tensor<u32>) {
    //     let tx = UNIT_POS_X;
    //     let pid_mask = 0x7FFFFFFFu32;
    //     let mut sh_pid = SharedMemory::<u32>::new(32usize);
    //     let mut sh_flag = SharedMemory::<u32>::new(32usize);
    //     sh_pid[tx as usize] = input[tx as usize] & pid_mask;
    //     sh_flag[tx as usize] = input[tx as usize] >> 31;
    //     sync_cube();
    //     if tx == 0u32 {
    //         let p0 = sh_pid[0usize]; // 130
    //         let p1 = sh_pid[1usize]; // 494
    //         let p2 = sh_pid[2usize]; // 130
    //         let f0 = sh_flag[0usize]; // 1
    //         let f3 = sh_flag[3usize]; // 0

    //         // masked-pid comparisons (the gate the fix relies on)
    //         out[0usize] = if p0 != p1 { 1u32 } else { 0u32 }; // expect 1
    //         out[1usize] = if p0 != p2 { 1u32 } else { 0u32 }; // expect 0

    //         // branch on OR of flags (optional optimisation, pair gate)
    //         out[2usize] = if (f0 | f3) != 0u32 { 1u32 } else { 0u32 }; // expect 1
    //         out[3usize] = if (f3 | f3) != 0u32 { 1u32 } else { 0u32 }; // expect 0

    //         // has_new via OR-accumulation then compare (optional, early exit)
    //         let mut hn = 0u32;
    //         hn = hn | f0;
    //         out[4usize] = if hn == 0u32 { 0u32 } else { 1u32 }; // expect 1
    //         let mut ho = 0u32;
    //         ho = ho | f3;
    //         out[5usize] = if ho == 0u32 { 0u32 } else { 1u32 }; // expect 0
    //     }
    // }

    // #[test]
    // fn test_debug_fix_constructs() {
    //     let Some(device) = try_device() else {
    //         eprintln!("Skipping: no wgpu backend");
    //         return;
    //     };
    //     let client = WgpuRuntime::client(&device);
    //     let mut input = vec![0u32; 32];
    //     input[0] = 0x80000082; // pid 130, new
    //     input[1] = 0x800001EE; // pid 494, new
    //     input[2] = 0x80000082; // pid 130, new
    //     input[3] = 0x7FFFFFFF; // sentinel, old
    //     let input_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(&input, vec![32], &client);
    //     let out = GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32; 6], vec![6], &client);
    //     unsafe {
    //         debug_fix_constructs::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(1, 1, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             input_gpu.clone().into_tensor_arg(),
    //             out.clone().into_tensor_arg(),
    //         );
    //     }
    //     let o = out.read(&client).unwrap();
    //     println!("p0!=p1   = {}  (expect 1)", o[0]);
    //     println!("p0!=p2   = {}  (expect 0)", o[1]);
    //     println!("(f0|f3)!=0 = {}  (expect 1)", o[2]);
    //     println!("(f3|f3)!=0 = {}  (expect 0)", o[3]);
    //     println!("hn(OR)!=0  = {}  (expect 1)", o[4]);
    //     println!("ho(OR)==0  = {}  (expect 0)", o[5]);
    // }

    // #[cube(launch_unchecked)]
    // fn repro_ne(input: &Tensor<u32>, out: &mut Tensor<u32>) {
    //     if UNIT_POS_X != 0u32 {
    //         terminate!();
    //     }
    //     let a = input[0usize]; // 130
    //     let b = input[1usize]; // 494
    //     out[0usize] = if a != b { 1u32 } else { 0u32 }; // expect 1
    //     out[1usize] = if a == b { 1u32 } else { 0u32 }; // expect 0
    //     out[2usize] = a;
    //     out[3usize] = b;
    // }

    // #[test]
    // fn test_repro_ne() {
    //     let Some(device) = try_device() else {
    //         eprintln!("Skipping: no wgpu backend");
    //         return;
    //     };
    //     let client = WgpuRuntime::client(&device);
    //     let input = GpuTensor::<WgpuRuntime, u32>::from_slice(&[130u32, 494u32], vec![2], &client);
    //     let out = GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32; 4], vec![4], &client);
    //     unsafe {
    //         repro_ne::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(1, 1, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             input.clone().into_tensor_arg(),
    //             out.clone().into_tensor_arg(),
    //         );
    //     }
    //     let o = out.read(&client).unwrap();
    //     println!("a!=b = {}  (expect 1)", o[0]);
    //     println!("a==b = {}  (expect 0)", o[1]);
    //     println!("a    = {}  b = {}", o[2], o[3]);
    // }

    // #[cube(launch_unchecked)]
    // fn repro_if_kinds(input: &Tensor<u32>, out: &mut Tensor<u32>) {
    //     if UNIT_POS_X != 0u32 {
    //         terminate!();
    //     }
    //     let a = input[0usize]; // 130
    //     let b = input[1usize]; // 494

    //     // statement-if, no else: body should run (a != b is true)
    //     out[0usize] = 7u32;
    //     if a != b {
    //         out[0usize] = 1u32;
    //     }

    //     // statement-if, condition false: body should NOT run (a == b is false)
    //     out[1usize] = 7u32;
    //     if a == b {
    //         out[1usize] = 1u32;
    //     }

    //     // expression-if for reference
    //     out[2usize] = if a != b { 1u32 } else { 0u32 };

    //     // while with runtime bound (expected to work)
    //     let mut c = 0u32;
    //     let mut i = 0u32;
    //     while i < b {
    //         c += 1u32;
    //         i += 1u32;
    //     }
    //     out[3usize] = c;
    // }

    // #[test]
    // fn test_repro_if_kinds() {
    //     let Some(device) = try_device() else {
    //         eprintln!("Skipping: no wgpu backend");
    //         return;
    //     };
    //     let client = WgpuRuntime::client(&device);
    //     let input = GpuTensor::<WgpuRuntime, u32>::from_slice(&[130u32, 494u32], vec![2], &client);
    //     let out = GpuTensor::<WgpuRuntime, u32>::from_slice(&[0u32; 4], vec![4], &client);
    //     unsafe {
    //         repro_if_kinds::launch_unchecked::<WgpuRuntime>(
    //             &client,
    //             CubeCount::Static(1, 1, 1),
    //             CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
    //             input.clone().into_tensor_arg(),
    //             out.clone().into_tensor_arg(),
    //         );
    //     }
    //     let o = out.read(&client).unwrap();
    //     println!("stmt-if true  -> {}  (expect 1, broken if 7)", o[0]);
    //     println!("stmt-if false -> {}  (expect 7, broken if 1)", o[1]);
    //     println!("expr-if       -> {}  (expect 1, regression if 0)", o[2]);
    //     println!("while count   -> {}  (expect 494)", o[3]);
    // }
}
