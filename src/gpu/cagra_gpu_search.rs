//! GPU-accelerated CAGRA beam search for query-time nearest neighbour retrieval.
//!
//! One workgroup per query. The query vector is loaded into scalar shared memory,
//! then beam search expands candidates from the CAGRA navigational graph using
//! a linear-probing hash table for visited-node tracking.

#![allow(missing_docs)]

use cubecl::frontend::{Float, SharedMemory};
use cubecl::prelude::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::prelude::*;

///////////
// Const //
///////////

/// Beam width (number of active candidates maintained during search)
const BEAM_WIDTH: usize = 16;
/// Maximum beam search iterations before forced termination
const MAX_BEAM_ITERS: usize = 48;
/// Hash table size for visited-node tracking (must be power of 2)
const HASH_SIZE: usize = 2048;
/// Expansion per given iteration. 1 -> one additional neighbour is explored,
/// 2 -> two, etc. pp. Usually something between 1 to 4.
const EXPAND_PER_ITER: usize = 3;

/// Number of random entry points per query
pub const N_ENTRY_POINTS: usize = 8;

////////////
// Params //
////////////

/// Parameters for the CAGRA style GPU beam search
pub struct CagraGpuSearchParams {
    /// Optional width of the beam. If not provided, will default to
    /// `BEAM_WIDTH`
    pub beam_width: Option<usize>,
    /// Optional maximum iterations for the beam search. Good rule of thumb is
    /// 3x beam_width. Will default to `MAX_BEAM_ITERS` if not provided.
    pub max_beam_iters: Option<usize>,
    /// Optional number of entry points. If not provided, will default to
    /// `N_ENTRY_POINTS`.
    pub n_entry_points: Option<usize>,
    /// Number of neighbours to explore per iteration
    pub expand_per_iter: Option<usize>,
}

impl CagraGpuSearchParams {
    /// Generates a new instance of the search
    ///
    /// ### Params
    ///
    /// * `beam_width` - Beam width for the kNN search
    /// * `max_beam_iters` - Maximum numbers of iterations to do. Rule of thumb
    ///   to be 2 to 3x beam width
    /// * `n_entry_points` - Number of entry points to use in the CAGRA graph.
    /// * `expand_per_iter` - Number of additional neighbours to explore per
    ///   iteration. Usually something between 1 to 4.
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(
        beam_width: Option<usize>,
        max_beam_iters: Option<usize>,
        n_entry_points: Option<usize>,
        expand_per_iter: Option<usize>,
    ) -> Self {
        Self {
            beam_width,
            max_beam_iters,
            n_entry_points,
            expand_per_iter,
        }
    }

    /// Pull out the needed values for the beam search
    ///
    /// ### Returns
    ///
    /// Tuple of `(width, iters, n_entry, expand)`
    pub fn get_vals(&self) -> (usize, usize, usize, usize) {
        let width = self.beam_width.unwrap_or(BEAM_WIDTH);
        let iters = self.max_beam_iters.unwrap_or(MAX_BEAM_ITERS);
        let n_entry = self.n_entry_points.unwrap_or(N_ENTRY_POINTS);
        let expand = self.expand_per_iter.unwrap_or(EXPAND_PER_ITER);

        (width, iters, n_entry, expand)
    }

    /// Get the number of entry points
    ///
    /// ### Returns
    ///
    /// n_entry
    pub fn get_n_entry(&self) -> usize {
        self.n_entry_points.unwrap_or(N_ENTRY_POINTS)
    }

    /// Create params with defaults scaled to the graph and query parameters.
    ///
    /// ### Params
    ///
    /// * `k_out` - Number of neighbours to return per query
    /// * `k_graph` - Degree of the navigational graph
    ///
    /// ### Returns
    ///
    /// Params with beam width and iterations scaled appropriately
    pub fn from_graph(k_out: usize, k_graph: usize) -> Self {
        let beam_width = k_out.max(k_graph).max(16) * 2;
        let max_beam_iters = beam_width * 3;
        Self {
            beam_width: Some(beam_width),
            max_beam_iters: Some(max_beam_iters),
            n_entry_points: None,
            expand_per_iter: None,
        }
    }
}

/// Default implementation for CagraGpuSearchParams
impl Default for CagraGpuSearchParams {
    fn default() -> Self {
        Self::new(None, None, None, None)
    }
}

////////////////////
// Kernel helpers //
////////////////////

/// Single xorshift step used to generate random node offsets.
///
/// ### Params
///
/// * `state` - Current RNG state (must be non-zero)
///
/// ### Returns
///
/// Next RNG state
#[cube]
fn xorshift_search(state: u32) -> u32 {
    let mut x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

///////////////////////////
// Distance probe kernel //
///////////////////////////

/// Probe kernel: loads a query vector into scalar shared memory,
/// computes distance to a single database vector from global memory.
///
/// ### Params
///
/// * `vectors` - Database vectors `[n_nodes, dim/LINE_SIZE]` as `Line<F>`
/// * `query` - Query vector `[dim/LINE_SIZE]` as `Line<F>`
/// * `out_dist` - Output scalar distance `[1]`
/// * `out_shared` - Output copy of the query shared memory `[dim_scalars]`
/// * `target_node` - Index of the database vector to compare against
/// * `use_cosine` - If true, computes dot product instead of squared L2 (comptime)
/// * `dim_lines` - Number of `Line<F>` elements per vector row (comptime)
///
/// ### Returns
///
/// Writes computed distance to `out_dist[0]` and shared memory contents to
/// `out_shared`. Both written by thread 0 only.
///
/// ### Grid mapping
///
/// * Single workgroup `(1,1,1)`, result written by thread 0
#[cube(launch_unchecked)]
fn probe_query_distance<F: Float>(
    vectors: &Tensor<Line<F>>,
    query: &Tensor<Line<F>>,
    out_dist: &mut Tensor<F>,
    out_shared: &mut Tensor<F>,
    target_node: u32,
    #[comptime] use_cosine: bool,
    #[comptime] dim_lines: usize,
) {
    let tx = UNIT_POS_X;
    let dim_scalars = dim_lines * 4usize;

    // Load query vector into scalar shared memory (all threads cooperate)
    let mut sq_vec = SharedMemory::<F>::new(dim_scalars);

    let mut idx_load = tx as usize;
    while idx_load < dim_scalars {
        let line_idx = idx_load / 4usize;
        let lane = idx_load % 4usize;
        let line_val = query[line_idx];
        sq_vec[idx_load] = line_val[lane];
        idx_load += WORKGROUP_SIZE_X as usize;
    }
    sync_cube();

    // Thread 0: compute distance from shared-memory query to global-memory target
    if tx == 0u32 {
        // Dump shared memory contents for verification
        let mut s = 0usize;
        while s < dim_scalars {
            out_shared[s] = sq_vec[s];
            s += 1usize;
        }

        // Compute distance: query (shared) vs target (global)
        let mut sum = F::new(0.0);
        let mut s2 = 0usize;
        while s2 < dim_scalars {
            let line_idx = s2 / 4usize;
            let lane = s2 % 4usize;
            let line_val = vectors[target_node as usize * dim_lines + line_idx];
            let vb = line_val[lane];
            let va = sq_vec[s2];

            if use_cosine {
                sum += va * vb;
            } else {
                let diff = va - vb;
                sum += diff * diff;
            }
            s2 += 1usize;
        }

        out_dist[0usize] = sum;
    }
}

/// Hash table probe: insert node IDs, then query for presence.
///
/// Thread 0 inserts `n_insert` IDs from `insert_ids` into the hash table,
/// then probes for `n_probe` IDs from `probe_ids`, writing 1 (found) or
/// 0 (not found) into `probe_results`.
///
/// ### Params
///
/// * `insert_ids` - Node IDs to insert `[n_insert]`
/// * `probe_ids` - Node IDs to probe for `[n_probe]`
/// * `probe_results` - Output presence flags `[n_probe]`; 1 = found, 0 = absent
/// * `n_insert` - Number of IDs to insert
/// * `n_probe` - Number of IDs to probe
/// * `hash_size` - Hash table capacity (comptime, must be a power of 2)
///
/// ### Returns
///
/// Writes per-probe presence flags into `probe_results`.
///
/// ### Grid mapping
///
/// * Single workgroup `(1,1,1)`, all work done by thread 0
#[cube(launch_unchecked)]
fn probe_hash_table(
    insert_ids: &Tensor<u32>,
    probe_ids: &Tensor<u32>,
    probe_results: &mut Tensor<u32>,
    n_insert: u32,
    n_probe: u32,
    #[comptime] hash_size: usize,
) {
    let tx = UNIT_POS_X;
    let hash_mask = hash_size as u32 - 1u32;
    let sentinel = 0x7FFFFFFFu32;

    let mut s_hash = SharedMemory::<u32>::new(hash_size);

    // All threads cooperate to clear the hash table
    let mut i = tx as usize;
    while i < hash_size {
        s_hash[i] = sentinel;
        i += WORKGROUP_SIZE_X as usize;
    }
    sync_cube();

    // Thread 0: insert all IDs
    if tx == 0u32 {
        let mut ins = 0u32;
        while ins < n_insert {
            let node_id = insert_ids[ins as usize];
            let mut slot = node_id & hash_mask;
            let mut attempts = 0u32;

            // Linear probe until we find an empty slot or the ID already present
            let mut done = false;
            while !done && attempts < hash_size as u32 {
                let existing = s_hash[slot as usize];
                if existing == sentinel {
                    // Empty slot -- insert
                    s_hash[slot as usize] = node_id;
                    done = true;
                } else if existing == node_id {
                    // Already present -- skip
                    done = true;
                } else {
                    // Collision -- linear probe
                    slot = (slot + 1u32) & hash_mask;
                    attempts += 1u32;
                }
            }
            ins += 1u32;
        }
    }
    sync_cube();

    // Thread 0: probe for each ID
    if tx == 0u32 {
        let mut p = 0u32;
        while p < n_probe {
            let node_id = probe_ids[p as usize];
            let mut slot = node_id & hash_mask;
            let mut attempts = 0u32;
            let mut found = 0u32;

            let mut done = false;
            while !done && attempts < hash_size as u32 {
                let existing = s_hash[slot as usize];
                if existing == node_id {
                    found = 1u32;
                    done = true;
                } else if existing == sentinel {
                    // Empty slot -- not present
                    done = true;
                } else {
                    slot = (slot + 1u32) & hash_mask;
                    attempts += 1u32;
                }
            }

            probe_results[p as usize] = found;
            p += 1u32;
        }
    }
}

/////////////////
// Beam search //
/////////////////

/// CAGRA beam search kernel. One workgroup per query.
///
/// Thread 0 manages the sorted candidate queue and the linear-probing hash
/// table for visited-node deduplication. All threads cooperate on loading
/// the query vector into scalar shared memory and computing distances to
/// the candidate's graph neighbours.
///
/// ### Params
///
/// * `vectors` - Database vectors `[n_nodes, dim/LINE_SIZE]` as `Line<F>`
/// * `norms` - Pre-computed L2 norms `[n_nodes]` (ignored when `use_cosine` is
///   false)
/// * `graph` - CAGRA navigational graph `[n_nodes, k_graph]` of neighbour IDs
/// * `queries` - Query vectors `[n_queries, dim/LINE_SIZE]` as `Line<F>`
/// * `entry_points` - Initial seed nodes `[n_queries, n_entry]`
/// * `out_indices` - Output neighbour indices `[n_queries, k_out]`
/// * `out_dists` - Output neighbour distances `[n_queries, k_out]`
/// * `out_iters` - Number of beam iterations actually used per query
///   `[n_queries]`
/// * `n_nodes` - Total number of nodes in the graph
/// * `k_out` - Number of neighbours to return per query
/// * `k_graph` - Degree of the navigational graph (comptime)
/// * `use_cosine` - Whether to compute cosine distance (comptime)
/// * `dim_lines` - Number of `Line<F>` elements per vector row (comptime)
/// * `beam_width` - Number of active candidates maintained during search
///   (comptime)
/// * `hash_size` - Hash table capacity for visited tracking (comptime, must be
///   power of 2)
/// * `max_iters` - Maximum beam iterations before forced termination (comptime)
/// * `n_entry` - Number of entry points per query (comptime)
///
/// ### Grid mapping
///
/// * One Cube per query: `q_idx = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X`
#[cube(launch_unchecked)]
pub fn cagra_beam_search<F: Float>(
    vectors: &Tensor<Line<F>>,
    norms: &Tensor<F>,
    graph: &Tensor<u32>,
    queries: &Tensor<Line<F>>,
    entry_points: &Tensor<u32>,
    out_indices: &mut Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_iters: &mut Tensor<u32>,
    n_nodes: u32,
    k_out: u32,
    #[comptime] k_graph: usize,
    #[comptime] use_cosine: bool,
    #[comptime] dim_lines: usize,
    #[comptime] beam_width: usize,
    #[comptime] hash_size: usize,
    #[comptime] max_iters: usize,
    #[comptime] n_entry: usize,
    #[comptime] expand_per_iter: usize,
) {
    let q_idx = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let n_queries = out_indices.shape(0usize) as u32;
    if q_idx >= n_queries {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let dim_scalars = dim_lines * 4usize;
    let hash_mask = hash_size as u32 - 1u32;
    let sentinel = 0x7FFFFFFFu32;
    let f_max = F::new(999999999.0);
    let bw = beam_width as u32;
    let bw_last = beam_width - 1usize;
    let total_slots = k_graph * expand_per_iter;
    let expand_u32 = expand_per_iter as u32;

    // shared memory set up
    let mut sq_vec = SharedMemory::<F>::new(dim_scalars);
    let mut s_cand_dist = SharedMemory::<F>::new(beam_width);
    let mut s_cand_idx = SharedMemory::<u32>::new(beam_width);
    let mut s_cand_expanded = SharedMemory::<u32>::new(beam_width);
    let mut s_hash = SharedMemory::<u32>::new(hash_size);
    let mut s_nbr_idx = SharedMemory::<u32>::new(total_slots);
    let mut s_nbr_dist = SharedMemory::<F>::new(total_slots);
    let mut s_active_flag = SharedMemory::<u32>::new(1usize);
    let mut s_num_cands = SharedMemory::<u32>::new(1usize);
    let mut s_query_norm = SharedMemory::<F>::new(1usize);

    let q_line_offset = q_idx as usize * dim_lines;
    let mut il = tx as usize;
    while il < dim_scalars {
        let line_idx = il / 4usize;
        let lane = il % 4usize;
        let line_val = queries[q_line_offset + line_idx];
        sq_vec[il] = line_val[lane];
        il += WORKGROUP_SIZE_X as usize;
    }

    let mut ih = tx as usize;
    while ih < hash_size {
        s_hash[ih] = sentinel;
        ih += WORKGROUP_SIZE_X as usize;
    }
    let mut ic = tx as usize;
    while ic < beam_width {
        s_cand_dist[ic] = f_max;
        s_cand_idx[ic] = sentinel;
        s_cand_expanded[ic] = 0u32;
        ic += WORKGROUP_SIZE_X as usize;
    }

    sync_cube();

    if tx == 0u32 {
        if use_cosine {
            let mut norm_sq = F::new(0.0);
            let mut s = 0usize;
            while s < dim_scalars {
                let v = sq_vec[s];
                norm_sq += v * v;
                s += 1usize;
            }
            s_query_norm[0usize] = F::sqrt(norm_sq);
        }

        let entry_base = q_idx as usize * n_entry;
        let mut num_cands = 0u32;

        let mut e = 0usize;
        while e < n_entry {
            let node_id = entry_points[entry_base + e];
            if node_id < n_nodes {
                let mut hs = node_id & hash_mask;
                let mut ha = 0u32;
                let mut hd = false;
                let mut is_new: bool = false;
                while !hd && ha < hash_size as u32 {
                    let ex = s_hash[hs as usize];
                    if ex == sentinel {
                        s_hash[hs as usize] = node_id;
                        is_new = true;
                        hd = true;
                    } else if ex == node_id {
                        hd = true;
                    } else {
                        hs = (hs + 1u32) & hash_mask;
                        ha += 1u32;
                    }
                }

                if is_new {
                    let mut sum = F::new(0.0);
                    for li in 0..dim_lines {
                        let lv = vectors[node_id as usize * dim_lines + li];
                        let s_off = li * 4usize;
                        if use_cosine {
                            sum += sq_vec[s_off] * lv[0]
                                + sq_vec[s_off + 1usize] * lv[1]
                                + sq_vec[s_off + 2usize] * lv[2]
                                + sq_vec[s_off + 3usize] * lv[3];
                        } else {
                            let d0 = sq_vec[s_off] - lv[0];
                            let d1 = sq_vec[s_off + 1usize] - lv[1];
                            let d2 = sq_vec[s_off + 2usize] - lv[2];
                            let d3 = sq_vec[s_off + 3usize] - lv[3];
                            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                        }
                    }
                    let dist = if use_cosine {
                        F::new(1.0) - sum / (s_query_norm[0usize] * norms[node_id as usize])
                    } else {
                        sum
                    };

                    let mut insert_pos = num_cands;
                    let mut ip = 0u32;
                    while ip < num_cands {
                        if dist < s_cand_dist[ip as usize] && insert_pos == num_cands {
                            insert_pos = ip;
                        }
                        ip += 1u32;
                    }
                    if insert_pos < num_cands {
                        let mut sh = num_cands;
                        while sh > insert_pos {
                            s_cand_dist[sh as usize] = s_cand_dist[(sh - 1u32) as usize];
                            s_cand_idx[sh as usize] = s_cand_idx[(sh - 1u32) as usize];
                            s_cand_expanded[sh as usize] = 0u32;
                            sh -= 1u32;
                        }
                    }
                    s_cand_dist[insert_pos as usize] = dist;
                    s_cand_idx[insert_pos as usize] = node_id;
                    s_cand_expanded[insert_pos as usize] = 0u32;
                    num_cands += 1u32;
                }
            }
            e += 1usize;
        }
        s_num_cands[0usize] = num_cands;
    }

    sync_cube();

    let max_iter_u32 = max_iters as u32;
    let mut iter: u32 = 0u32;
    while iter < max_iter_u32 {
        if tx == 0u32 {
            out_iters[q_idx as usize] = iter;
            s_active_flag[0usize] = sentinel;
            let nc = s_num_cands[0usize];
            let mut active_count: u32 = 0u32;

            // Claim up to P unexpanded candidates in beam order (ascending dist).
            let mut fc = 0u32;
            while fc < nc && active_count < expand_u32 {
                if s_cand_expanded[fc as usize] == 0u32 {
                    let active = s_cand_idx[fc as usize];
                    s_cand_expanded[fc as usize] = 1u32;

                    let gb = active as usize * k_graph;
                    let slot_base = active_count as usize * k_graph;
                    let mut j = 0usize;
                    while j < k_graph {
                        let nbr = graph[gb + j];
                        if nbr < n_nodes && nbr != sentinel {
                            let mut hs = nbr & hash_mask;
                            let mut ha = 0u32;
                            let mut hd = false;
                            let mut is_new: bool = false;
                            while !hd && ha < hash_size as u32 {
                                let ex = s_hash[hs as usize];
                                if ex == sentinel {
                                    s_hash[hs as usize] = nbr;
                                    is_new = true;
                                    hd = true;
                                } else if ex == nbr {
                                    hd = true;
                                } else {
                                    hs = (hs + 1u32) & hash_mask;
                                    ha += 1u32;
                                }
                            }
                            if is_new {
                                s_nbr_idx[slot_base + j] = nbr;
                            } else {
                                s_nbr_idx[slot_base + j] = sentinel;
                            }
                        } else {
                            s_nbr_idx[slot_base + j] = sentinel;
                        }
                        j += 1usize;
                    }
                    active_count += 1u32;
                }
                fc += 1u32;
            }

            // Pad remaining expansion slots with sentinels.
            while active_count < expand_u32 {
                let slot_base = active_count as usize * k_graph;
                let mut j = 0usize;
                while j < k_graph {
                    s_nbr_idx[slot_base + j] = sentinel;
                    j += 1usize;
                }
                active_count += 1u32;
            }

            // Signal termination only if nothing was claimed.
            if fc > 0u32 || nc > 0u32 {
                // at least one iteration of the scan ran; flag based on whether
                // we found an unexpanded candidate
                let mut any_unexpanded: bool = false;
                let mut sc = 0u32;
                while sc < nc {
                    if s_cand_expanded[sc as usize] == 0u32 {
                        any_unexpanded = true;
                    }
                    sc += 1u32;
                }
                // We expanded `expand_u32 - (expand_u32 - actually_expanded)` this iter.
                // If we expanded at least one, keep going; the termination check
                // simply reflects whether we produced any work this iter.
                if any_unexpanded {
                    s_active_flag[0usize] = 0u32;
                } else {
                    // flag stays sentinel, loop will terminate
                }
            }

            // Simpler, correct rule: if we expanded nothing this iter, terminate.
            // Overwrite based on whether any non-sentinel was written into s_nbr_idx.
            let mut any_real: bool = false;
            let mut ck = 0usize;
            while ck < total_slots {
                if s_nbr_idx[ck] != sentinel {
                    any_real = true;
                }
                ck += 1usize;
            }
            if any_real {
                s_active_flag[0usize] = 0u32;
            } else {
                s_active_flag[0usize] = sentinel;
            }
        }

        sync_cube();

        let flag = s_active_flag[0usize];
        if flag == sentinel {
            iter = max_iter_u32;
        }

        if flag != sentinel {
            let mut ms = tx as usize;
            while ms < total_slots {
                let nbr = s_nbr_idx[ms];
                if nbr != sentinel {
                    let mut sum = F::new(0.0);
                    for li in 0..dim_lines {
                        let lv = vectors[nbr as usize * dim_lines + li];
                        let s_off = li * 4usize;
                        if use_cosine {
                            sum += sq_vec[s_off] * lv[0]
                                + sq_vec[s_off + 1usize] * lv[1]
                                + sq_vec[s_off + 2usize] * lv[2]
                                + sq_vec[s_off + 3usize] * lv[3];
                        } else {
                            let d0 = sq_vec[s_off] - lv[0];
                            let d1 = sq_vec[s_off + 1usize] - lv[1];
                            let d2 = sq_vec[s_off + 2usize] - lv[2];
                            let d3 = sq_vec[s_off + 3usize] - lv[3];
                            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                        }
                    }
                    let dist = if use_cosine {
                        F::new(1.0) - sum / (s_query_norm[0usize] * norms[nbr as usize])
                    } else {
                        sum
                    };
                    s_nbr_dist[ms] = dist;
                } else {
                    s_nbr_dist[ms] = f_max;
                }
                ms += WORKGROUP_SIZE_X as usize;
            }
        }

        sync_cube();

        if tx == 0u32 && flag != sentinel {
            let mut nc = s_num_cands[0usize];

            let mut j: usize = 0usize;
            while j < total_slots {
                if s_nbr_idx[j] != sentinel {
                    let dist = s_nbr_dist[j];
                    let nbr = s_nbr_idx[j];

                    let worst = s_cand_dist[bw_last];
                    let mut skip: bool = false;
                    if nc >= bw && dist >= worst {
                        skip = true;
                    }

                    if !skip {
                        let mut slen = nc;
                        if slen > bw {
                            slen = bw;
                        }

                        let mut insert_pos = slen;
                        let mut ip = 0u32;
                        while ip < slen {
                            if dist < s_cand_dist[ip as usize] && insert_pos == slen {
                                insert_pos = ip;
                            }
                            ip += 1u32;
                        }

                        let mut do_insert: bool = true;
                        if insert_pos >= bw {
                            do_insert = false;
                        }

                        if do_insert {
                            let mut shift_end = nc;
                            if nc >= bw {
                                shift_end = bw;
                                shift_end -= 1u32;
                            }
                            if shift_end > insert_pos {
                                let mut sh = shift_end;
                                while sh > insert_pos {
                                    s_cand_dist[sh as usize] = s_cand_dist[(sh - 1u32) as usize];
                                    s_cand_idx[sh as usize] = s_cand_idx[(sh - 1u32) as usize];
                                    s_cand_expanded[sh as usize] =
                                        s_cand_expanded[(sh - 1u32) as usize];
                                    sh -= 1u32;
                                }
                            }

                            s_cand_dist[insert_pos as usize] = dist;
                            s_cand_idx[insert_pos as usize] = nbr;
                            s_cand_expanded[insert_pos as usize] = 0u32;

                            if nc < bw {
                                nc += 1u32;
                            }
                        }
                    }
                }
                j += 1usize;
            }
            s_num_cands[0usize] = nc;
        }

        iter += 1u32;
    }

    sync_cube();

    let num_cands = s_num_cands[0usize];
    let out_base = q_idx * k_out;
    let mut wr = tx;
    while wr < k_out {
        if wr < num_cands {
            out_indices[(out_base + wr) as usize] = s_cand_idx[wr as usize];
            out_dists[(out_base + wr) as usize] = s_cand_dist[wr as usize];
        } else {
            out_indices[(out_base + wr) as usize] = sentinel;
            out_dists[(out_base + wr) as usize] = f_max;
        }
        wr += WORKGROUP_SIZE_X;
    }
}

//////////////
// Dispatch //
//////////////

/// Batch CAGRA beam search on GPU.
///
/// Pads query vectors to the next multiple of LINE_SIZE, uploads them,
/// generates or uses provided entry points, launches one workgroup per
/// query, and downloads results.
///
/// ### Params
///
/// * `queries_flat` - Flattened query vectors [n_queries * dim]
/// * `n_queries` - Number of queries
/// * `dim` - Original (unpadded) query dimensionality
/// * `vectors_gpu` - GPU-resident database vectors [n, dim_padded/LINE_SIZE]
/// * `norms_gpu` - GPU-resident L2 norms `[n]` (Cosine) or a dummy scalar
///   (Euclidean)
/// * `graph_gpu` - GPU-resident CAGRA navigational graph [n, k_graph]
/// * `n` - Number of vectors in the database
/// * `k_graph` - Degree of the navigational graph
/// * `k_out` - Number of neighbours to return per query
/// * `use_cosine` - Whether to use cosine distance
/// * `seed` - Random seed used when `entry_points` is `None`
/// * `query_params` - Beam search parameters (beam width, max iterations,
///   number of entry points); defaults applied where fields are `None`
/// * `entry_points` - Optional pre-computed entry point IDs
///   `[n_queries * query_params.get_n_entry()]`.
///   If `None`, random entry points are sampled from `[0, n)`.
/// * `client` - GPU compute client
///
/// ### Returns
///
/// `(indices, distances)` per query, sorted by distance ascending.
/// Sentinel entries (unfilled slots) are filtered out.
#[allow(clippy::too_many_arguments)]
pub fn cagra_search_batch_gpu<T, R>(
    queries_flat: &[T],
    n_queries: usize,
    dim: usize,
    vectors_gpu: &GpuTensor<R, T>,
    norms_gpu: &GpuTensor<R, T>,
    graph_gpu: &GpuTensor<R, u32>,
    n: usize,
    k_graph: usize,
    k_out: usize,
    use_cosine: bool,
    seed: usize,
    query_params: &CagraGpuSearchParams,
    entry_points: Option<&[u32]>,
    client: &ComputeClient<R>,
) -> (Vec<Vec<usize>>, Vec<Vec<T>>)
where
    R: Runtime,
    T: AnnSearchGpuFloat + num_traits::Float,
{
    let line = LINE_SIZE as usize;
    let dim_padded = dim.next_multiple_of(line);
    let dim_vec = dim_padded / line;

    let (width, iters, n_entry, expand) = query_params.get_vals();

    // Pad queries
    let queries_padded = if dim_padded != dim {
        let mut padded = vec![T::zero(); n_queries * dim_padded];
        for i in 0..n_queries {
            for j in 0..dim {
                padded[i * dim_padded + j] = queries_flat[i * dim + j];
            }
        }
        padded
    } else {
        queries_flat.to_vec()
    };

    let queries_gpu =
        GpuTensor::<R, T>::from_slice(&queries_padded, vec![n_queries, dim_padded], client);

    // Entry points: use provided or fall back to random
    let entry_flat = match entry_points {
        Some(pts) => {
            assert_eq!(pts.len(), n_queries * n_entry);
            pts.to_vec()
        }
        None => {
            let mut rng = SmallRng::seed_from_u64(seed as u64);
            (0..n_queries * n_entry)
                .map(|_| rng.random_range(0..n as u32))
                .collect()
        }
    };
    let entry_gpu = GpuTensor::<R, u32>::from_slice(&entry_flat, vec![n_queries, n_entry], client);

    // Output tensors
    let out_idx_gpu = GpuTensor::<R, u32>::empty(vec![n_queries, k_out], client);
    let out_dist_gpu = GpuTensor::<R, T>::empty(vec![n_queries, k_out], client);
    let out_iters_gpu = GpuTensor::<R, u32>::empty(vec![n_queries], client);

    // 2D grid for large query counts
    let cubes_x = (n_queries as u32).min(65535);
    let cubes_y = (n_queries as u32).div_ceil(cubes_x);

    unsafe {
        let _ = cagra_beam_search::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(cubes_x, cubes_y, 1),
            CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
            vectors_gpu.clone().into_tensor_arg(line),
            norms_gpu.clone().into_tensor_arg(1),
            graph_gpu.clone().into_tensor_arg(1),
            queries_gpu.into_tensor_arg(line),
            entry_gpu.into_tensor_arg(1),
            out_idx_gpu.clone().into_tensor_arg(1),
            out_dist_gpu.clone().into_tensor_arg(1),
            out_iters_gpu.clone().into_tensor_arg(1),
            ScalarArg { elem: n as u32 },
            ScalarArg { elem: k_out as u32 },
            k_graph,
            use_cosine,
            dim_vec,
            width,
            HASH_SIZE,
            iters,
            n_entry,
            expand,
        );
    }

    // Download
    let idx_flat = out_idx_gpu.read(client);
    let dist_flat = out_dist_gpu.read(client);
    let sentinel_usize = 0x7FFFFFFFusize;

    let indices: Vec<Vec<usize>> = (0..n_queries)
        .map(|i| {
            (0..k_out)
                .map(|j| (idx_flat[i * k_out + j] & 0x7FFFFFFFu32) as usize)
                .filter(|&pid| pid < n && pid != sentinel_usize)
                .collect()
        })
        .collect();

    let distances: Vec<Vec<T>> = (0..n_queries)
        .map(|i| {
            (0..k_out)
                .filter(|&j| {
                    let pid = (idx_flat[i * k_out + j] & 0x7FFFFFFFu32) as usize;
                    pid < n && pid != sentinel_usize
                })
                .map(|j| dist_flat[i * k_out + j])
                .collect()
        })
        .collect();

    (indices, distances)
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
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cubecl::wgpu::WgpuRuntime::client(&device);
        }));
        result.ok().map(|_| device)
    }

    #[test]
    fn test_probe_query_distance_euclidean_dim32() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE as usize;
        let n = 50usize;
        let dim = 32usize;
        let dim_vec = dim / line;

        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                data[i * dim + j] = (i * 100 + j) as f32;
            }
        }

        let query: Vec<f32> = (0..dim).map(|j| j as f32 + 0.5).collect();

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let query_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&query, vec![dim], &client);
        let out_dist = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32; 1], vec![1], &client);
        let out_shared =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; dim], vec![dim], &client);

        let target_node = 3u32;

        unsafe {
            let _ = probe_query_distance::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                vectors_gpu.into_tensor_arg(line),
                query_gpu.into_tensor_arg(line),
                out_dist.clone().into_tensor_arg(1),
                out_shared.clone().into_tensor_arg(1),
                ScalarArg { elem: target_node },
                false,
                dim_vec,
            );
        }

        let shared_result = out_shared.read(&client);
        let dist_result = out_dist.read(&client);

        for j in 0..dim {
            let expected = j as f32 + 0.5;
            assert!(
                (shared_result[j] - expected).abs() < 1e-4,
                "Shared memory mismatch at [{}]: got {} expected {}",
                j,
                shared_result[j],
                expected
            );
        }

        let cpu_dist: f32 = (0..dim)
            .map(|j| {
                let diff = query[j] - data[target_node as usize * dim + j];
                diff * diff
            })
            .sum();
        let gpu_dist = dist_result[0];

        println!("GPU dist: {gpu_dist:.4}  CPU dist: {cpu_dist:.4}");
        assert!(
            (gpu_dist - cpu_dist).abs() < 1e-1,
            "Distance mismatch: gpu={gpu_dist}, cpu={cpu_dist}"
        );
    }

    #[test]
    fn test_probe_query_distance_cosine_dim32() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);
        let line = LINE_SIZE as usize;
        let n = 50usize;
        let dim = 32usize;
        let dim_vec = dim / line;

        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                data[i * dim + j] = (i * 100 + j) as f32;
            }
        }

        let query: Vec<f32> = (0..dim).map(|j| j as f32 + 0.5).collect();

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let query_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&query, vec![dim], &client);
        let out_dist = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32; 1], vec![1], &client);
        let out_shared =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![0.0f32; dim], vec![dim], &client);

        let target_node = 5u32;

        unsafe {
            let _ = probe_query_distance::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                vectors_gpu.into_tensor_arg(line),
                query_gpu.into_tensor_arg(line),
                out_dist.clone().into_tensor_arg(1),
                out_shared.clone().into_tensor_arg(1),
                ScalarArg { elem: target_node },
                true,
                dim_vec,
            );
        }

        let dist_result = out_dist.read(&client);

        let cpu_dot: f32 = (0..dim)
            .map(|j| query[j] * data[target_node as usize * dim + j])
            .sum();
        let gpu_dot = dist_result[0];

        println!("GPU dot: {gpu_dot:.4}  CPU dot: {cpu_dot:.4}");
        assert!(
            (gpu_dot - cpu_dot).abs() / cpu_dot.abs().max(1e-6) < 1e-3,
            "Dot product mismatch: gpu={gpu_dot}, cpu={cpu_dot}"
        );
    }

    #[test]
    fn test_hash_table_basic() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);

        let insert_ids = vec![10u32, 20, 30, 500, 1023];
        let probe_ids = vec![10u32, 20, 31, 500, 999, 1023, 0];
        let expected = [1u32, 1, 0, 1, 0, 1, 0];

        let insert_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&insert_ids, vec![insert_ids.len()], &client);
        let probe_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&probe_ids, vec![probe_ids.len()], &client);
        let results_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
            &vec![0u32; probe_ids.len()],
            vec![probe_ids.len()],
            &client,
        );

        unsafe {
            let _ = probe_hash_table::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                insert_gpu.into_tensor_arg(1),
                probe_gpu.into_tensor_arg(1),
                results_gpu.clone().into_tensor_arg(1),
                ScalarArg {
                    elem: insert_ids.len() as u32,
                },
                ScalarArg {
                    elem: probe_ids.len() as u32,
                },
                HASH_SIZE,
            );
        }

        let results = results_gpu.read(&client);
        for (i, (&got, &exp)) in results.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got, exp,
                "Probe [{}] for ID {}: got {} expected {}",
                i, probe_ids[i], got, exp
            );
        }
    }

    #[test]
    fn test_hash_table_collisions() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);

        let hash_size = HASH_SIZE as u32;
        let insert_ids = vec![0u32, hash_size, hash_size * 2, hash_size * 3];
        let probe_ids = vec![0u32, hash_size, hash_size * 2, hash_size * 3, hash_size * 4];
        let expected = [1u32, 1, 1, 1, 0];

        let insert_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&insert_ids, vec![insert_ids.len()], &client);
        let probe_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&probe_ids, vec![probe_ids.len()], &client);
        let results_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
            &vec![0u32; probe_ids.len()],
            vec![probe_ids.len()],
            &client,
        );

        unsafe {
            let _ = probe_hash_table::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                insert_gpu.into_tensor_arg(1),
                probe_gpu.into_tensor_arg(1),
                results_gpu.clone().into_tensor_arg(1),
                ScalarArg {
                    elem: insert_ids.len() as u32,
                },
                ScalarArg {
                    elem: probe_ids.len() as u32,
                },
                HASH_SIZE,
            );
        }

        let results = results_gpu.read(&client);
        for (i, (&got, &exp)) in results.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got, exp,
                "Collision probe [{}] for ID {}: got {} expected {}",
                i, probe_ids[i], got, exp
            );
        }
    }

    #[test]
    fn test_hash_table_duplicates() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);

        let insert_ids = vec![42u32, 42, 42, 99, 99, 42];
        let probe_ids = vec![42u32, 99, 100];
        let expected = [1u32, 1, 0];

        let insert_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&insert_ids, vec![insert_ids.len()], &client);
        let probe_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&probe_ids, vec![probe_ids.len()], &client);
        let results_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
            &vec![0u32; probe_ids.len()],
            vec![probe_ids.len()],
            &client,
        );

        unsafe {
            let _ = probe_hash_table::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                insert_gpu.into_tensor_arg(1),
                probe_gpu.into_tensor_arg(1),
                results_gpu.clone().into_tensor_arg(1),
                ScalarArg {
                    elem: insert_ids.len() as u32,
                },
                ScalarArg {
                    elem: probe_ids.len() as u32,
                },
                HASH_SIZE,
            );
        }

        let results = results_gpu.read(&client);
        for (i, (&got, &exp)) in results.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got, exp,
                "Duplicate probe [{}] for ID {}: got {} expected {}",
                i, probe_ids[i], got, exp
            );
        }
    }

    #[test]
    fn test_beam_search_star_graph() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);
        let n = 50usize;
        let dim = 32usize;
        let k_graph = 10usize;
        let k_out = 5usize;

        // Node 0 at origin, nodes 1..49 at increasing distance
        let mut data = vec![0.0f32; n * dim];
        for i in 1..n {
            for j in 0..dim {
                data[i * dim + j] = (i as f32) * 0.1 + (j as f32) * 0.001;
            }
        }

        let graph_flat = build_brute_force_graph(&data, n, dim, k_graph);

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32], vec![1], &client);
        let graph_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&graph_flat, vec![n, k_graph], &client);

        let query = vec![0.0f32; dim];

        let (indices, distances) = cagra_search_batch_gpu(
            &query,
            1,
            dim,
            &vectors_gpu,
            &norms_gpu,
            &graph_gpu,
            n,
            k_graph,
            k_out,
            false,
            42,
            &CagraGpuSearchParams::default(),
            None,
            &client,
        );

        println!("Star graph results: {:?}", indices[0]);
        println!("Star graph dists:   {:?}", distances[0]);

        let gt: std::collections::HashSet<usize> = (1..=k_out).collect();
        let found: std::collections::HashSet<usize> = indices[0].iter().copied().collect();
        let hits = gt.intersection(&found).count();
        println!("Star graph recall: {}/{}", hits, k_out);
        assert!(
            hits >= k_out - 1,
            "Star graph: expected at least {} of top-{} neighbours, got {}",
            k_out - 1,
            k_out,
            hits
        );
    }

    #[test]
    fn test_beam_search_recall_euclidean() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);
        let n = 500usize;
        let dim = 32usize;
        let k_graph = 15usize;
        let k_out = 10usize;
        let n_queries = 20usize;

        // Uniform random data -- ensures brute-force kNN graph has good
        // connectivity (no isolated clusters that random entry points can't reach)
        let mut rng = SmallRng::seed_from_u64(123);
        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                data[i * dim + j] = rng.random_range(-10.0..10.0f32);
            }
        }

        let graph_flat = build_brute_force_graph(&data, n, dim, k_graph);

        // Queries: perturbed copies of data points
        let mut queries = vec![0.0f32; n_queries * dim];
        for qi in 0..n_queries {
            let src = (qi * 25) % n;
            for j in 0..dim {
                queries[qi * dim + j] = data[src * dim + j] + rng.random_range(-0.5..0.5f32);
            }
        }

        let vectors_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client);
        let norms_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(&[0.0f32], vec![1], &client);
        let graph_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&graph_flat, vec![n, k_graph], &client);

        let (gpu_indices, _) = cagra_search_batch_gpu(
            &queries,
            n_queries,
            dim,
            &vectors_gpu,
            &norms_gpu,
            &graph_gpu,
            n,
            k_graph,
            k_out,
            false,
            42,
            &CagraGpuSearchParams::default(),
            None,
            &client,
        );

        let gt = brute_force_knn(&queries, &data, n_queries, n, dim, k_out);

        let mut total_hits = 0;
        let total_possible = n_queries * k_out;
        for qi in 0..n_queries {
            let gt_set: std::collections::HashSet<usize> = gt[qi].iter().copied().collect();
            let found_set: std::collections::HashSet<usize> =
                gpu_indices[qi].iter().copied().collect();
            total_hits += gt_set.intersection(&found_set).count();
        }

        let recall = total_hits as f64 / total_possible as f64;
        println!(
            "Beam search recall@{}: {:.4} ({}/{})",
            k_out, recall, total_hits, total_possible
        );
        assert!(
            recall > 0.85,
            "Recall too low: {recall:.4} (expected > 0.85 with brute-force graph)"
        );
    }

    fn build_brute_force_graph(data: &[f32], n: usize, dim: usize, k: usize) -> Vec<u32> {
        let sentinel = 0x7FFFFFFFu32;
        let mut graph = vec![sentinel; n * k];
        for i in 0..n {
            let mut dists: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f32 = (0..dim)
                        .map(|d| {
                            let diff = data[i * dim + d] - data[j * dim + d];
                            diff * diff
                        })
                        .sum();
                    (d, j)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for slot in 0..k.min(dists.len()) {
                graph[i * k + slot] = dists[slot].1 as u32;
            }
        }
        graph
    }

    fn brute_force_knn(
        queries: &[f32],
        data: &[f32],
        n_queries: usize,
        n: usize,
        dim: usize,
        k: usize,
    ) -> Vec<Vec<usize>> {
        (0..n_queries)
            .map(|qi| {
                let mut dists: Vec<(f32, usize)> = (0..n)
                    .map(|j| {
                        let d: f32 = (0..dim)
                            .map(|d| {
                                let diff = queries[qi * dim + d] - data[j * dim + d];
                                diff * diff
                            })
                            .sum();
                        (d, j)
                    })
                    .collect();
                dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                dists.iter().take(k).map(|&(_, j)| j).collect()
            })
            .collect()
    }
}
