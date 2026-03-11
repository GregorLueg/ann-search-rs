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

use crate::annoy::*;
use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::nndescent::*;
use crate::prelude::*;
use crate::utils::*;

///////////
// Const //
///////////

/// Max proposals per node per iteration. Overflow is silently dropped.
const MAX_PROPOSALS: usize = 128;
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
/// * `vectors` - Row-major vector matrix, line-vectorised along the feature
///   dimension
/// * `a` - Row index of the first vector
/// * `b` - Row index of the second vector
///
/// ### Returns
///
/// Squared Euclidean distance between the two vectors
#[cube]
fn dist_sq_euclidean<F: Float + CubePrimitive>(vectors: &Tensor<Line<F>>, a: u32, b: u32) -> F {
    let dim_lines = vectors.shape(1);
    let stride = vectors.stride(0);
    let off_a = a as usize * stride;
    let off_b = b as usize * stride;
    let mut sum = F::new(0.0);

    for i in 0..dim_lines {
        let va = vectors[off_a + i];
        let vb = vectors[off_b + i];
        let diff = va - vb;
        let sq = diff * diff;
        sum += sq[0];
        sum += sq[1];
        sum += sq[2];
        sum += sq[3];
    }
    sum
}

/// Cosine distance (1 - cosine similarity) between vectors at `a` and `b`.
///
/// Requires pre-computed L2 norms.
///
/// ### Params
///
/// * `vectors` - Row-major vector matrix, line-vectorised along the feature
///   dimension
/// * `norms` - Pre-computed L2 norms, one per row
/// * `a` - Row index of the first vector
/// * `b` - Row index of the second vector
///
/// ### Returns
///
/// Cosine distance in the range [0, 2]
#[cube]
fn dist_cosine<F: Float>(vectors: &Tensor<Line<F>>, norms: &Tensor<F>, a: u32, b: u32) -> F {
    let dim_lines = vectors.shape(1);
    let stride = vectors.stride(0);
    let off_a = a as usize * stride;
    let off_b = b as usize * stride;
    let mut dot = F::new(0.0);

    for i in 0..dim_lines {
        let va = vectors[off_a + i];
        let vb = vectors[off_b + i];
        let prod = va * vb;
        dot += prod[0];
        dot += prod[1];
        dot += prod[2];
        dot += prod[3];
    }
    F::new(1.0) - dot / (norms[a as usize] * norms[b as usize])
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
/// * `n` - Number of vectors
/// * `seed` - Random seed for neighbour generation
/// * `use_cosine` - Whether to use cosine distance instead of squared Euclidean
///
/// Writes an initialised sorted kNN graph into `graph_idx` and `graph_dist`.
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> node index
#[cube(launch_unchecked)]
fn init_random_graph<F: Float>(
    vectors: &Tensor<Line<F>>,
    norms: &Tensor<F>,
    graph_idx: &mut Tensor<u32>,
    graph_dist: &mut Tensor<F>,
    n: u32,
    seed: u32,
    #[comptime] use_cosine: bool,
) {
    let node = ABSOLUTE_POS_X;
    if node >= n {
        terminate!();
    }

    let k = graph_idx.shape(1);
    let is_new_bit = 1u32 << 31;
    let base = node as usize * k;

    let mut rng = xorshift(node ^ seed ^ 0xDEADBEEFu32);

    for slot in 0..k {
        // generate random PID, reject self-loops
        rng = xorshift(rng);
        let mut pid = rng % n;
        if pid == node {
            pid = (pid + 1u32) % n;
        }

        // compute distance
        let dist = if use_cosine {
            dist_cosine(vectors, norms, node, pid)
        } else {
            dist_sq_euclidean(vectors, node, pid)
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
/// One thread per node. Thread 0 additionally resets the update counter.
///
/// ### Params
///
/// * `n` - Number of nodes
///
/// Zeroes all entries in `prop_count` and resets `update_counter[0]` to zero.
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> node index
#[cube(launch_unchecked)]
fn reset_proposals(prop_count: &mut Tensor<u32>, update_counter: &mut Tensor<u32>, n: u32) {
    let idx = ABSOLUTE_POS_X;
    if idx < n {
        prop_count[idx as usize] = 0u32;
    }
    if idx == 0u32 {
        update_counter[0usize] = 0u32;
    }
}

/// Scatters forward edges into a reverse edge buffer to ensure symmetric
/// information flow during the local join phase.
#[cube(launch_unchecked)]
pub fn build_reverse_candidates(
    graph_idx: &Tensor<u32>,
    reverse_idx: &mut Tensor<u32>,
    reverse_count: &Tensor<Atomic<u32>>,
    n: u32,
    #[comptime] build_k: u32,
) {
    let node = ABSOLUTE_POS_X;
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
#[cube(launch_unchecked)]
pub fn local_join_shared<F: Float>(
    vectors: &Tensor<Line<F>>,
    norms: &Tensor<F>,
    graph_idx: &Tensor<u32>,
    graph_dist: &Tensor<F>, // <--- NEW: Need this to threshold proposals!
    reverse_idx: &Tensor<u32>,
    reverse_count: &Tensor<u32>,
    prop_idx: &mut Tensor<u32>,
    prop_dist: &mut Tensor<F>,
    prop_count: &Tensor<Atomic<u32>>,
    n: u32,
    rho_thresh: u32,
    iter_seed: u32,
    #[comptime] max_proposals: u32,
    #[comptime] use_cosine: bool,
    #[comptime] dim_lines: usize,
    #[comptime] build_k: usize,
) {
    let node = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if node >= n {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let k = graph_idx.shape(1usize) as u32;
    let pid_mask = 0x7FFFFFFFu32;
    let is_new_bit = 1u32 << 31;

    let max_cands_comp = build_k * 2usize;

    let mut shared_vecs = SharedMemory::<Line<F>>::new(max_cands_comp * dim_lines);
    let mut shared_pids = SharedMemory::<u32>::new(max_cands_comp);
    let mut shared_is_new = SharedMemory::<bool>::new(max_cands_comp);
    let mut shared_norms = SharedMemory::<F>::new(max_cands_comp);

    let mut shared_rev_count = SharedMemory::<u32>::new(1usize);
    if tx == 0u32 {
        let rc = reverse_count[node as usize];
        shared_rev_count[0usize] = if rc > k { k } else { rc };
    }
    sync_cube();

    let rev_k = shared_rev_count[0usize];
    let total_cands = k + rev_k;

    let mut i_load = tx;
    while i_load < total_cands {
        let entry = if i_load < k {
            graph_idx[(node * k + i_load) as usize]
        } else {
            reverse_idx[(node * k + i_load - k) as usize]
        };

        shared_pids[i_load as usize] = entry & pid_mask;
        shared_is_new[i_load as usize] = entry >= is_new_bit;
        if use_cosine {
            shared_norms[i_load as usize] = norms[shared_pids[i_load as usize] as usize];
        }
        i_load += WORKGROUP_SIZE_X;
    }
    sync_cube();

    let total_loads = total_cands as usize * dim_lines;
    let mut idx_load = tx as usize;
    while idx_load < total_loads {
        let n_idx = idx_load / dim_lines;
        let d_idx = idx_load % dim_lines;
        let pid = shared_pids[n_idx];

        if pid < n {
            let vec_offset = pid as usize * vectors.stride(0usize) as usize + d_idx;
            shared_vecs[idx_load] = vectors[vec_offset];
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

        let hash_i = entry_hash(node, i as u32, iter_seed);
        if (hash_i & 0xFFFFu32) < rho_thresh {
            let hash_j = entry_hash(node, j as u32, iter_seed);

            if (hash_j & 0xFFFFu32) < rho_thresh {
                let is_new_i = shared_is_new[i];
                let is_new_j = shared_is_new[j];
                let pid_i = shared_pids[i];
                let pid_j = shared_pids[j];

                if (is_new_i || is_new_j) && pid_i != pid_j {
                    let mut sum = F::new(0.0);
                    let mut d = 0usize;
                    while d < dim_lines {
                        let va = shared_vecs[i * dim_lines + d];
                        let vb = shared_vecs[j * dim_lines + d];

                        if use_cosine {
                            let prod = va * vb;
                            sum += prod[0usize];
                            sum += prod[1usize];
                            sum += prod[2usize];
                            sum += prod[3usize];
                        } else {
                            let diff = va - vb;
                            let sq = diff * diff;
                            sum += sq[0usize];
                            sum += sq[1usize];
                            sum += sq[2usize];
                            sum += sq[3usize];
                        }
                        d += 1usize;
                    }

                    let dist = if use_cosine {
                        F::new(1.0) - (sum / (shared_norms[i] * shared_norms[j]))
                    } else {
                        sum
                    };

                    // --- THE FIX: THRESHOLD BEFORE PROPOSING ---
                    // Read the worst distance for both targets
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
            }
        }
        pair_idx += WORKGROUP_SIZE_X as usize;
    }
}

/// Merge proposals into the sorted kNN graph.
///
/// One thread per node. For each node:
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
/// Updates `graph_idx` and `graph_dist` in place with any improvements, flags
/// inserted entries as new, and accumulates the total improvement count into
/// `update_counter[0]`.
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> node index
#[cube(launch_unchecked)]
fn merge_proposals<F: Float>(
    graph_idx: &mut Tensor<u32>,
    graph_dist: &mut Tensor<F>,
    prop_idx: &Tensor<u32>,
    prop_dist: &Tensor<F>,
    prop_count: &Tensor<u32>,
    update_counter: &Tensor<Atomic<u32>>,
    n: u32,
    #[comptime] max_proposals: u32,
) {
    let node = ABSOLUTE_POS_X;
    if node >= n {
        terminate!();
    }

    let k = graph_idx.shape(1);
    let pid_mask = 0x7FFFFFFFu32;
    let is_new_bit = 1u32 << 31;
    let base = node as usize * k;

    // Clear new flags on all existing entries
    for j in 0..k {
        graph_idx[base + j] = graph_idx[base + j] & pid_mask;
    }

    // Read how many proposals this node received (capped at max_proposals)
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

/// 2-Hop Refinement Kernel
///
/// Runs after NNDescent convergence. For each node, evaluates the K^2
/// second-degree neighbors. Filters aggressively against duplicates and the
/// current worst distance before pushing to the proposal buffer.
#[cube(launch_unchecked)]
pub fn two_hop_refinement<F: Float>(
    vectors: &Tensor<Line<F>>,
    norms: &Tensor<F>,
    graph_idx: &Tensor<u32>,
    graph_dist: &Tensor<F>,
    prop_idx: &mut Tensor<u32>,
    prop_dist: &mut Tensor<F>,
    prop_count: &Tensor<Atomic<u32>>,
    n: u32,
    #[comptime] max_proposals: u32,
    #[comptime] use_cosine: bool,
    #[comptime] dim_lines: usize,
) {
    let node = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if node >= n {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let k = graph_idx.shape(1usize);
    let pid_mask = 0x7FFFFFFFu32;
    let graph_base = node as usize * k as usize;

    let mut shared_source = SharedMemory::<Line<F>>::new(dim_lines);
    let mut shared_worst_dist = SharedMemory::<F>::new(1usize);

    let mut d_idx = tx as usize;
    while d_idx < dim_lines {
        let vec_offset = node as usize * vectors.stride(0usize) + d_idx;
        shared_source[d_idx] = vectors[vec_offset];
        d_idx += WORKGROUP_SIZE_X as usize;
    }

    if tx == 0u32 {
        shared_worst_dist[0usize] = graph_dist[graph_base + k - 1usize];
    }
    sync_cube();

    let worst_dist = shared_worst_dist[0usize];
    let num_candidates = k * k;
    let mut cand_idx = tx as usize;

    // Linearize the K^2 candidate evaluation
    while cand_idx < num_candidates {
        let n1_idx = cand_idx / k;
        let n2_idx = cand_idx % k;

        let n1_raw = graph_idx[graph_base + n1_idx];
        let n1_pid = n1_raw & pid_mask;

        if n1_pid < n {
            // Get the 2-hop neighbor
            let n2_raw = graph_idx[n1_pid as usize * k + n2_idx];
            let cand_pid = n2_raw & pid_mask;

            // Basic sanity checks: don't evaluate self
            if cand_pid < n && cand_pid != node {
                // --- THE FIX: DUPLICATE FILTERING ---
                let mut is_dup: bool = false;
                let mut scan_idx = 0usize;
                while scan_idx < k {
                    if (graph_idx[graph_base + scan_idx] & pid_mask) == cand_pid {
                        is_dup = true;
                    }
                    scan_idx += 1usize;
                }

                if !is_dup {
                    // Compute distance
                    let mut sum = F::new(0.0);
                    let mut d = 0usize;
                    while d < dim_lines {
                        let va = shared_source[d];
                        let vb = vectors[cand_pid as usize * vectors.stride(0usize) + d];
                        if use_cosine {
                            let prod = va * vb;
                            sum += prod[0usize];
                            sum += prod[1usize];
                            sum += prod[2usize];
                            sum += prod[3usize];
                        } else {
                            let diff = va - vb;
                            let sq = diff * diff;
                            sum += sq[0usize];
                            sum += sq[1usize];
                            sum += sq[2usize];
                            sum += sq[3usize];
                        }
                        d += 1usize;
                    }

                    let dist = if use_cosine {
                        F::new(1.0) - (sum / (norms[node as usize] * norms[cand_pid as usize]))
                    } else {
                        sum
                    };

                    // Only propose if it actually improves our graph
                    if dist < worst_dist {
                        let slot = prop_count[node as usize].fetch_add(1u32);
                        if slot < max_proposals {
                            let off = node as usize * max_proposals as usize + slot as usize;
                            prop_idx[off] = cand_pid;
                            prop_dist[off] = dist;
                        } else {
                            // Reservoir sample into the proposal buffer
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
    let node = ABSOLUTE_POS_X;
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
    let node = ABSOLUTE_POS_X;
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
fn pad_vectors<T: Float>(flat: &[T], n: usize, dim: usize, dim_padded: usize) -> Vec<T> {
    let mut padded = vec![T::zero(); n * dim_padded];
    for i in 0..n {
        let src = &flat[i * dim..(i + 1) * dim];
        let dst = &mut padded[i * dim_padded..i * dim_padded + dim];
        dst.copy_from_slice(src);
    }
    padded
}

/// Build Annoy-initialised graph arrays ready for GPU upload.
///
/// Queries the Annoy forest for each node to obtain `build_k` initial
/// neighbours, sorts them by distance, and encodes the IS_NEW flag into
/// the MSB of each index entry.
///
/// ### Returns
///
/// `(idx_flat, dist_flat)` -- flat row-major arrays of size `n * build_k`
fn annoy_init_graph<T>(
    vectors_flat: &[T],
    n: usize,
    dim: usize,
    build_k: usize,
    forest: &AnnoyIndex<T>,
) -> (Vec<u32>, Vec<T>)
where
    T: AnnSearchFloat,
{
    let is_new_bit = 1u32 << 31;
    let mut idx_flat = vec![0u32; n * build_k];
    let mut dist_flat = vec![T::zero(); n * build_k];

    let search_k = build_k * forest.n_trees * 2;

    // parallel per-node Annoy queries
    idx_flat
        .par_chunks_mut(build_k)
        .zip(dist_flat.par_chunks_mut(build_k))
        .enumerate()
        .for_each(|(i, (idx_slot, dist_slot))| {
            let query = &vectors_flat[i * dim..(i + 1) * dim];
            let (indices, distances) = forest.query(query, build_k + 1, Some(search_k));

            // Skip self (index 0 in results is typically the query itself),
            // take up to build_k neighbours, already sorted by distance from Annoy
            let mut written = 0;
            for (idx, dist) in indices.into_iter().zip(distances) {
                if idx == i || written >= build_k {
                    continue;
                }
                idx_slot[written] = (idx as u32) | is_new_bit;
                dist_slot[written] = dist;
                written += 1;
            }

            // Pad remaining slots with sentinels
            for s in written..build_k {
                idx_slot[s] = 0x7FFFFFFFu32 | is_new_bit;
                dist_slot[s] = T::max_value();
            }
        });

    (idx_flat, dist_flat)
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
            ) -> (Vec<usize>, Vec<$float>) {
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
                                Dist::Euclidean => self.query_euclidean(
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
            ) -> (Vec<usize>, Vec<$float>) {
                let init_candidates = (ef / 2).max(2 * k).min(self.n);
                let search_k = init_candidates * 3;
                let (init_indices, _) =
                    self.forest
                        .query(query_vec, init_candidates, Some(search_k));

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

                final_results
                    .into_iter()
                    .map(|(OrderedFloat(d), i)| (i, d))
                    .unzip()
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
            ) -> (Vec<usize>, Vec<$float>) {
                let init_candidates = (ef / 2).max(k).min(self.n);
                let search_k = init_candidates * 3;
                let (init_indices, _) =
                    self.forest
                        .query(query_vec, init_candidates, Some(search_k));

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

                final_results
                    .into_iter()
                    .map(|(OrderedFloat(d), i)| (i, d))
                    .unzip()
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
pub struct NNDescentGpu<T: Float, R: Runtime> {
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
    pub metric: Dist,
    /// True kNN graph of size n * k, sorted by distance per row.
    /// Extracted from NNDescent output before CAGRA pruning.
    knn_graph: Vec<(usize, T)>,
    /// CAGRA navigational graph of size n * k, used for beam search.
    /// NOT a faithful kNN graph -- edges are reordered for reachability.
    nav_graph: Vec<(usize, T)>,
    /// Whether NNDescent hit the delta threshold
    converged: bool,
    /// Annoy index initialisation and finding the entry points
    forest: AnnoyIndex<T>,
    /// CubeCL runtime device
    _device: R::Device,
}

/// VectorDistance implementation for NNDescentGPU
impl<T, R> VectorDistance<T> for NNDescentGpu<T, R>
where
    T: AnnSearchFloat + cubecl::frontend::Float + cubecl::CubeElement,
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
    /// * `k` - Final neighbours per node (default 30). This is the degree
    ///   of the output CAGRA graph.
    /// * `build_k` - Internal NNDescent degree before CAGRA pruning.
    ///   Defaults to `2 * k`. Must be >= `k`.
    /// * `max_iters` - Maximum NNDescent iterations (default 15)
    /// * `delta` - Convergence threshold as fraction of n*k (default 0.001)
    /// * `rho` - Sampling rate for the local join (default 0.5)
    /// * `use_annoy_init` - If true (default), use an Annoy forest for
    ///   graph initialisation instead of random neighbours. Produces a
    ///   better starting graph and converges in fewer iterations.
    /// * `n_trees` - Number of Annoy trees (only used when `use_annoy_init`
    ///   is true). Defaults to `5 + n^0.25`, capped at 32.
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    /// * `device` - CubeCL runtime device
    ///
    /// ### Returns
    ///
    /// Initialised struct with the completed kNN graph
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
        seed: usize,
        verbose: bool,
        device: R::Device,
    ) -> Self {
        let (vectors_flat, n, dim) = matrix_to_flat(data);
        let k = k.unwrap_or(30);
        let build_k = build_k.unwrap_or(2 * k).max(k);
        let max_iters = max_iters.unwrap_or(DEFAULT_MAX_ITERS);
        let delta = delta.unwrap_or(DEFAULT_DELTA);
        let rho = rho.unwrap_or(DEFAULT_RHO);
        let rho_thresh = (rho * 65535.0) as u32;

        // pad dim to next multiple of LINE_SIZE
        let line = LINE_SIZE as usize;
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

        // ---- 1: Annoy graph initialisation ----

        // Annoy-based initialisation (CPU)
        let n_trees = n_trees.unwrap_or_else(|| {
            let calculated = 5 + ((n as f64).powf(0.25)).round() as usize;
            calculated.min(32)
        });

        if verbose {
            println!("  Building Annoy forest ({} trees)...", n_trees);
        }

        let annoy_start = Instant::now();
        let forest = AnnoyIndex::new(data, n_trees, metric, seed);

        let (idx_flat, dist_flat) = annoy_init_graph(&vectors_flat, n, dim, build_k, &forest);

        if verbose {
            println!(
                "  Annoy forest built and queried for graph initialisation: {:.2?}",
                annoy_start.elapsed()
            );
        }

        // ---- 2: NNDescent iterations on the GPU ----

        let client = R::client(&device);
        let use_cosine = metric == Dist::Cosine;

        // upload vectors (stays resident for the entire build)
        let vectors_gpu = GpuTensor::<R, T>::from_slice(&vectors_padded, vec![n, dim_vec], &client);

        // norms tensor (dummy scalar if Euclidean to avoid Option in kernel args)
        let norms_gpu = if use_cosine {
            GpuTensor::<R, T>::from_slice(&norms, vec![n], &client)
        } else {
            GpuTensor::<R, T>::from_slice(&[T::zero()], vec![1], &client)
        };

        // upload pre-built graph to GPU
        let graph_idx_gpu = GpuTensor::<R, u32>::from_slice(&idx_flat, vec![n, build_k], &client);
        let graph_dist_gpu = GpuTensor::<R, T>::from_slice(&dist_flat, vec![n, build_k], &client);

        // Proposal buffers on GPU
        let max_prop = MAX_PROPOSALS;
        let prop_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, max_prop], &client);
        let prop_dist_gpu = GpuTensor::<R, T>::empty(vec![n, max_prop], &client);
        let prop_count_gpu = GpuTensor::<R, u32>::empty(vec![n], &client);

        // Convergence counter (single u32)
        let update_counter_gpu = GpuTensor::<R, u32>::empty(vec![1], &client);

        let grid_n = (n as u32).div_ceil(WORKGROUP_SIZE_X);

        // NNDescent iterations
        let iter_start = Instant::now();
        let mut converged = false;

        let reverse_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, build_k], &client);
        let reverse_count_gpu = GpuTensor::<R, u32>::empty(vec![n], &client);

        for iter in 0..max_iters {
            // Calculate 2D grid: max x-dimension is typically 65535
            let cubes_x = 65535u32;
            let cubes_y = (n as u32).div_ceil(cubes_x);

            // 1. Reset BOTH proposal counts and reverse counts (and the update counter)
            unsafe {
                // Reset proposals
                let _ = reset_proposals::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static(grid_n, 1, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    prop_count_gpu.clone().into_tensor_arg(1),
                    update_counter_gpu.clone().into_tensor_arg(1),
                    ScalarArg { elem: n as u32 },
                );

                // Reset reverse counts (safe to zero update_counter a second time)
                let _ = reset_proposals::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static(grid_n, 1, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    reverse_count_gpu.clone().into_tensor_arg(1),
                    update_counter_gpu.clone().into_tensor_arg(1),
                    ScalarArg { elem: n as u32 },
                );
            }

            // 2. Build reverse edges
            unsafe {
                let _ = build_reverse_candidates::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static(grid_n, 1, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    graph_idx_gpu.clone().into_tensor_arg(1),
                    reverse_idx_gpu.clone().into_tensor_arg(1),
                    reverse_count_gpu.clone().into_tensor_arg(1),
                    ScalarArg { elem: n as u32 },
                    build_k as u32,
                );
            }

            let iter_seed = seed as u32 ^ (iter as u32).wrapping_mul(0x9E3779B9u32);

            // 3. Local join (Now receiving the reverse edges!)
            unsafe {
                let _ = local_join_shared::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(cubes_x, cubes_y, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    vectors_gpu.clone().into_tensor_arg(line),
                    norms_gpu.clone().into_tensor_arg(1),
                    graph_idx_gpu.clone().into_tensor_arg(1),
                    graph_dist_gpu.clone().into_tensor_arg(1),
                    reverse_idx_gpu.clone().into_tensor_arg(1),
                    reverse_count_gpu.clone().into_tensor_arg(1),
                    prop_idx_gpu.clone().into_tensor_arg(1),
                    prop_dist_gpu.clone().into_tensor_arg(1),
                    prop_count_gpu.clone().into_tensor_arg(1),
                    ScalarArg { elem: n as u32 },
                    ScalarArg { elem: rho_thresh },
                    ScalarArg { elem: iter_seed },
                    MAX_PROPOSALS as u32,
                    use_cosine,
                    dim_vec,
                    build_k, // <--- ADD THIS BACK HERE
                );
            }

            // 4. Merge proposals into the graph
            unsafe {
                let _ = merge_proposals::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(grid_n, 1, 1),
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

            // 5. Download single u32 to check convergence
            let counter_data = update_counter_gpu.clone().read(&client);
            let updates = counter_data[0] as f64;
            let rate = updates / (n * build_k) as f64;

            if verbose {
                println!(
                    "  Iter {}: {} updates (rate={:.6})",
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

        if verbose {
            println!("  Running 2-Hop Refinement Sweep...");
        }

        let refinement_start = Instant::now();

        let cubes_x = 65535u32;
        let cubes_y = (n as u32).div_ceil(cubes_x);

        // Zero out proposal counts
        unsafe {
            let _ = reset_proposals::launch_unchecked::<R>(
                &client,
                CubeCount::Static(grid_n, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                prop_count_gpu.clone().into_tensor_arg(1),
                update_counter_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
            );
        }

        // Run the 2-hop sweep
        unsafe {
            let _ = two_hop_refinement::launch_unchecked::<T, R>(
                &client,
                CubeCount::Static(cubes_x, cubes_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                vectors_gpu.clone().into_tensor_arg(line),
                norms_gpu.clone().into_tensor_arg(1),
                graph_idx_gpu.clone().into_tensor_arg(1),
                graph_dist_gpu.clone().into_tensor_arg(1),
                prop_idx_gpu.clone().into_tensor_arg(1),
                prop_dist_gpu.clone().into_tensor_arg(1),
                prop_count_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                MAX_PROPOSALS as u32,
                use_cosine,
                dim_vec, // Note: build_k was removed here
            );
        }

        // Merge the refined proposals
        unsafe {
            let _ = merge_proposals::launch_unchecked::<T, R>(
                &client,
                CubeCount::Static(grid_n, 1, 1),
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

        if verbose {
            println!(
                "  2-Hop Sweep finished in {:.2?}",
                refinement_start.elapsed()
            );
        }

        let nndescent_idx = graph_idx_gpu.clone().read(&client);
        let nndescent_dist = graph_dist_gpu.clone().read(&client);
        let pid_mask = 0x7FFFFFFFu32;
        let sentinel = 0x7FFFFFFFusize;

        let mut knn_graph = vec![(sentinel, <T as num_traits::Float>::max_value()); n * k];

        knn_graph
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(i, slot)| {
                // NNDescent graph is sorted by distance, so first k entries
                // of the build_k row are the k nearest neighbours.
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

        // ---- CAGRA graph optimisation: prune from build_k -> k ----

        let cagra_start = Instant::now();

        // allocate buffers for CAGRA steps
        let pruned_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, k], &client);
        let reverse_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, k], &client);
        let reverse_counts_gpu = GpuTensor::<R, u32>::from_slice(&vec![0u32; n], vec![n], &client);
        let final_idx_gpu = GpuTensor::<R, u32>::empty(vec![n, k], &client);

        // grid configs
        let cubes_x = 65535u32;
        let cubes_y = (n as u32).div_ceil(cubes_x);

        unsafe {
            // Step 1: Rank-based pruning (1 Cube per Node)
            let _ = cagra_rank_prune_shared::launch_unchecked::<R>(
                &client,
                CubeCount::Static(cubes_x, cubes_y, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                graph_idx_gpu.into_tensor_arg(1),
                pruned_idx_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                build_k,
                k,
            );

            // Step 2: Reverse edges (1 Thread per Node)
            let _ = cagra_build_reverse::launch_unchecked::<R>(
                &client,
                CubeCount::Static(grid_n, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                pruned_idx_gpu.clone().into_tensor_arg(1),
                reverse_idx_gpu.clone().into_tensor_arg(1),
                reverse_counts_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                k,
            );

            // Step 3: Merge graphs (1 Thread per Node)
            let _ = cagra_merge_graphs::launch_unchecked::<R>(
                &client,
                CubeCount::Static(grid_n, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                pruned_idx_gpu.into_tensor_arg(1),
                reverse_idx_gpu.into_tensor_arg(1),
                reverse_counts_gpu.into_tensor_arg(1),
                final_idx_gpu.clone().into_tensor_arg(1),
                ScalarArg { elem: n as u32 },
                k,
            );
        }

        if verbose {
            println!("  CAGRA optimisation: {:.2?}", cagra_start.elapsed());
        }

        // download and compute distances quickly on CPU

        let final_idx = final_idx_gpu.read(&client);
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
                            Dist::Euclidean => T::euclidean_simd(a, b),
                            Dist::Cosine => {
                                let dot = T::dot_simd(a, b);
                                T::one() - dot / (norms[i] * norms[pid])
                            }
                        };
                        slot[j] = (pid, dist);
                    }
                }

                // Sort each node's neighbours by distance
                slot.sort_unstable_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            });

        if verbose {
            println!("  Total build time: {:.2?}", start.elapsed());
        }

        Self {
            vectors_flat,
            dim,
            n,
            k,
            norms,
            metric,
            forest,
            knn_graph,
            nav_graph: cagra_graph,
            converged,
            _device: device,
        }
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
    ) -> (Vec<usize>, Vec<T>) {
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
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        ef_search: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, ef_search);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, ef_search)
    }

    /// Returns the neighbours of node `i` as a slice of `(index, distance)`.
    fn graph_neighbours(&self, idx: usize) -> &[(usize, T)] {
        &self.nav_graph[idx * self.k..(idx + 1) * self.k]
    }

    /// Whether the algorithm converged during construction.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Returns the size of the struct in bytes (CPU side only).
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
            Dist::Euclidean,
            Some(5),
            None,
            Some(10),
            None,
            Some(0.001),
            Some(0.5),
            42,
            false,
            device,
        );

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
            42,
            false,
            device,
        );

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
            Dist::Euclidean,
            Some(3),
            None,
            Some(10),
            None,
            Some(0.001),
            Some(0.5),
            42,
            false,
            device,
        );

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
            Dist::Euclidean,
            Some(5),
            None,
            Some(10),
            None,
            Some(0.001),
            Some(0.5),
            42,
            false,
            device,
        );

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

///////////////
// Dead code //
///////////////

// Keeping this for potential future iterations

// // NOTE: cagra_search_shared is shelved pending CubeCL warp-level primitive
// // support. The kernel has known correctness issues (no shared-memory
// // broadcasting of thread-0 state, no cooperative distance reduction, broken
// // visited-node tracking). Revisit when CubeCL exposes warp shuffles.
// #[allow(dead_code)]
// #[cube(launch_unchecked)]
// pub fn cagra_search_shared<F: Float>(
//     vectors: &Tensor<Line<F>>,
//     queries: &Tensor<Line<F>>,
//     graph: &Tensor<u32>,
//     out_indices: &mut Tensor<u32>,
//     out_dists: &mut Tensor<F>,
//     n_nodes: u32,
//     #[comptime] d: u32,
//     #[comptime] m: u32,
//     #[comptime] dim_lines: u32,
//     #[comptime] max_iters: u32,
//     #[comptime] hash_size: u32,
// ) {
//     let q_idx = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
//     if q_idx >= queries.shape(0) as u32 {
//         terminate!();
//     }

//     let tx = UNIT_POS_X;
//     let pid_mask = 0x7FFFFFFFu32;
//     let parent_flag = 1u32 << 31;

//     let mut sq_vec = SharedMemory::<Line<F>>::new(dim_lines as usize);
//     let mut s_top_m_dist = SharedMemory::<F>::new(m as usize);
//     let mut s_top_m_idx = SharedMemory::<u32>::new(m as usize);
//     let mut s_hash = SharedMemory::<u32>::new(hash_size as usize);

//     let mut i = tx;
//     while i < m {
//         s_top_m_dist[i as usize] = F::new(999999.0);
//         s_top_m_idx[i as usize] = 0xFFFFFFFFu32;
//         i += WORKGROUP_SIZE_X;
//     }

//     let mut j = tx;
//     while j < hash_size {
//         s_hash[j as usize] = 0xFFFFFFFFu32;
//         j += WORKGROUP_SIZE_X;
//     }

//     let mut d_idx = tx;
//     while d_idx < dim_lines {
//         sq_vec[d_idx as usize] = queries[(q_idx * dim_lines + d_idx) as usize];
//         d_idx += WORKGROUP_SIZE_X;
//     }

//     sync_cube();

//     if tx == 0u32 {
//         let mut seed = q_idx ^ 0xDEADBEEFu32;
//         seed ^= seed << 13u32;
//         seed ^= seed >> 17u32;
//         seed ^= seed << 5u32;

//         let entry_node = seed % n_nodes;
//         s_top_m_idx[0] = entry_node;

//         let hash_idx = entry_node % hash_size;
//         s_hash[hash_idx as usize] = entry_node;
//     }
//     sync_cube();

//     // TODO: This kernel is shelved. See note above.
//     // Known issues:
//     // - parent_node / active / is_new are thread-0-only, not broadcast via
//     //   shared memory
//     // - No cooperative reduction for distance partial sums
//     // - is_new check commented out, defeating the hash table
//     // - Single entry point instead of p*d random init

//     let k_out = out_indices.shape(1) as u32;
//     let mut write_idx = tx;
//     while write_idx < k_out {
//         if write_idx < m {
//             out_indices[(q_idx * k_out + write_idx) as usize] =
//                 s_top_m_idx[write_idx as usize] & pid_mask;
//             out_dists[(q_idx * k_out + write_idx) as usize] = s_top_m_dist[write_idx as usize];
//         }
//         write_idx += WORKGROUP_SIZE_X;
//     }
// }
