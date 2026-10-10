//! GPU-accelerated CAGRA beam search for query-time nearest neighbour retrieval.
//!
//! One workgroup per query. The query vector is loaded into scalar shared memory,
//! then beam search expands candidates from the CAGRA navigational graph using
//! a linear-probing hash table for visited-node tracking.

#![allow(missing_docs)]

use cubecl::frontend::{Float, SharedMemory};
use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};

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

    /// Create params with the beam scaled to the requested `k`.
    ///
    /// ### Params
    ///
    /// * `k_out` - Number of neighbours to return per query
    ///
    /// ### Returns
    ///
    /// Params with beam width and iterations scaled appropriately
    pub fn from_k(k_out: usize) -> Self {
        let beam_width = k_out.max(BEAM_WIDTH) * 2;
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
/// * `vectors` - Database vectors `[n_nodes, dim/N]` as `Vector<F, N>`
/// * `graph` - CAGRA navigational graph `[n_nodes, k_graph]` of neighbour IDs
/// * `queries` - Query vectors `[n_queries, dim/N]` as `Vector<F, N>`
/// * `entry_points` - Initial seed nodes `[n_queries, n_entry]`
/// * `out_indices` - Output neighbour indices `[n_queries, k_out]`
/// * `out_dists` - Output neighbour distances `[n_queries, k_out]`
/// * `out_iters` - Number of beam iterations actually used per query
///   `[n_queries]`
/// * `n_nodes` - Total number of nodes in the graph
/// * `k_out` - Number of neighbours to return per query
/// * `k_graph` - Degree of the navigational graph (comptime)
/// * `use_cosine` - Whether to compute cosine distance, `1 - dot` on
///   unit-normalised vectors and queries (comptime)
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime)
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
pub fn cagra_beam_search<F: Float, N: Size>(
    vectors: &Tensor<Vector<F, N>>,
    graph: &Tensor<u32>,
    queries: &Tensor<Vector<F, N>>,
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

    let lanes = LINE_SIZE;
    let tx = UNIT_POS_X;
    let dim_scalars = dim_lines * lanes;
    let hash_mask = hash_size as u32 - 1u32;
    let sentinel = 0x7FFFFFFFu32;
    let f_max = F::new(999999999.0_f32);
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
    let mut s_hash_count = SharedMemory::<u32>::new(1usize);

    let q_line_offset = q_idx as usize * dim_lines;
    let mut il = tx as usize;
    while il < dim_scalars {
        let line_idx = il / lanes;
        let lane = il % lanes;
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
                    let mut sum = F::new(0.0_f32);
                    for li in 0..dim_lines {
                        let lv = vectors[node_id as usize * dim_lines + li];
                        let s_off = li * lanes;
                        if use_cosine {
                            #[unroll]
                            for lane in 0..lanes {
                                sum += sq_vec[s_off + lane] * lv[lane];
                            }
                        } else {
                            #[unroll]
                            for lane in 0..lanes {
                                let d = sq_vec[s_off + lane] - lv[lane];
                                sum += d * d;
                            }
                        }
                    }
                    let dist = if use_cosine {
                        F::new(1.0_f32) - sum
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
        // Every entry that went in was new, so the beam and the table agree.
        s_hash_count[0usize] = num_cands;
    }

    sync_cube();

    // Forget the table before this iteration's inserts could push it past
    // three quarters full. Linear probing degrades sharply above that, and a
    // full table drops every new neighbour, which caps the search at
    // `hash_size` visited nodes whatever the beam width. Re-inserting the beam
    // keeps its members deduplicated; any other node seen again is worse than
    // the beam's worst (the worst only ever improves), so the merge skips it.
    let reset_at = (hash_size * 3 / 4) as u32;
    let total_slots_u32 = total_slots as u32;

    let max_iter_u32 = max_iters as u32;
    let mut iter: u32 = 0u32;
    let mut last_iter: u32 = 0u32;
    while iter < max_iter_u32 {
        if tx == 0u32 {
            last_iter = iter;
            s_active_flag[0usize] = sentinel;
            let nc = s_num_cands[0usize];
            let mut active_count: u32 = 0u32;
            let mut hash_count = s_hash_count[0usize];

            if hash_count + total_slots_u32 > reset_at {
                let mut hr = 0usize;
                while hr < hash_size {
                    s_hash[hr] = sentinel;
                    hr += 1usize;
                }
                hash_count = 0u32;
                let mut bi = 0u32;
                while bi < nc {
                    let node = s_cand_idx[bi as usize];
                    let mut hs = node & hash_mask;
                    let mut ha = 0u32;
                    let mut placed = false;
                    // Bounded like every other probe: a planner-shrunk table
                    // can be smaller than the beam on a small device.
                    while !placed && ha < hash_size as u32 {
                        if s_hash[hs as usize] == sentinel {
                            s_hash[hs as usize] = node;
                            placed = true;
                        } else {
                            hs = (hs + 1u32) & hash_mask;
                            ha += 1u32;
                        }
                    }
                    hash_count += 1u32;
                    bi += 1u32;
                }
            }

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
                                hash_count += 1u32;
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

            s_hash_count[0usize] = hash_count;
            let claimed = active_count;

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

            // Stop once every beam entry has been expanded. A claimed node
            // whose neighbours were all seen already is not a reason to stop:
            // unexpanded entries further down the beam may still lead
            // somewhere.
            if claimed > 0u32 {
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
                    let mut sum = F::new(0.0_f32);
                    for li in 0..dim_lines {
                        let lv = vectors[nbr as usize * dim_lines + li];
                        let s_off = li * lanes;
                        if use_cosine {
                            #[unroll]
                            for lane in 0..lanes {
                                sum += sq_vec[s_off + lane] * lv[lane];
                            }
                        } else {
                            #[unroll]
                            for lane in 0..lanes {
                                let d = sq_vec[s_off + lane] - lv[lane];
                                sum += d * d;
                            }
                        }
                    }
                    let dist = if use_cosine {
                        F::new(1.0_f32) - sum
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

    if tx == 0u32 {
        out_iters[q_idx as usize] = last_iter;
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
/// * `vectors_gpu` - GPU-resident database vectors [n, dim_padded/LINE_SIZE],
///   unit-normalised under cosine
/// * `graph_gpu` - GPU-resident CAGRA navigational graph [n, k_graph]
/// * `n` - Number of vectors in the database
/// * `k_graph` - Degree of the navigational graph
/// * `k_out` - Number of neighbours to return per query
/// * `use_cosine` - Whether to use cosine distance; the queries are
///   unit-normalised here
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
    graph_gpu: &GpuTensor<R, u32>,
    n: usize,
    k_graph: usize,
    k_out: usize,
    use_cosine: bool,
    seed: usize,
    query_params: &CagraGpuSearchParams,
    entry_points: Option<&[u32]>,
    client: &ComputeClient<R>,
) -> KnnResult<T>
where
    R: Runtime,
    T: CubeclFloat + AnnSearchFloat,
{
    let limits = GpuLimits::from_client(client);
    let line = LINE_SIZE;
    let dim_padded = dim.next_multiple_of(line);
    let dim_vec = dim_padded / line;

    let (width, iters, n_entry, expand) = query_params.get_vals();

    // Pad queries
    let mut queries_padded = if dim_padded != dim {
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
    if use_cosine {
        normalise_rows(&mut queries_padded, dim_padded);
    }

    let queries_gpu =
        GpuTensor::<R, T>::from_slice(&queries_padded, vec![n_queries, dim_padded], client)?;

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
    let entry_gpu = GpuTensor::<R, u32>::from_slice(&entry_flat, vec![n_queries, n_entry], client)?;

    // Output tensors
    let out_idx_gpu = GpuTensor::<R, u32>::empty(vec![n_queries, k_out], client)?;
    let out_dist_gpu = GpuTensor::<R, T>::empty(vec![n_queries, k_out], client)?;
    let out_iters_gpu = GpuTensor::<R, u32>::empty(vec![n_queries], client)?;

    // 2D grid for large query counts
    let (cubes_x, cubes_y) = grid_2d(n_queries as u32, &limits)?;

    // The hash table is the only elastic term in the kernel's shared-memory
    // footprint; it shrinks on a device that cannot hold the preferred size.
    let staging = plan_beam_search_staging(
        dim_padded,
        k_graph,
        width,
        expand,
        HASH_SIZE,
        size_of::<T>(),
        &limits,
    )?;

    unsafe {
        cagra_beam_search::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(cubes_x, cubes_y, 1),
            CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
            line,
            vectors_gpu.clone().into_tensor_arg(),
            graph_gpu.clone().into_tensor_arg(),
            queries_gpu.into_tensor_arg(),
            entry_gpu.into_tensor_arg(),
            out_idx_gpu.clone().into_tensor_arg(),
            out_dist_gpu.clone().into_tensor_arg(),
            out_iters_gpu.clone().into_tensor_arg(),
            n as u32,
            k_out as u32,
            k_graph,
            use_cosine,
            dim_vec,
            width,
            staging.hash_size,
            iters,
            n_entry,
            expand,
        );
    }

    // Download
    let idx_flat = out_idx_gpu.read(client)?;
    let dist_flat = out_dist_gpu.read(client)?;
    let sentinel_usize = 0x7FFFFFFFusize;

    let indices: Vec<Vec<usize>> = (0..n_queries)
        .map(|i| {
            (0..k_out)
                .map(|j| (idx_flat[i * k_out + j] & 0x7FFFFFFFu32) as usize)
                .filter(|&pid| pid < n && pid != sentinel_usize)
                .collect()
        })
        .collect();

    // Cosine can round a self-distance to just under zero.
    let distances: Vec<Vec<T>> = (0..n_queries)
        .map(|i| {
            (0..k_out)
                .filter(|&j| {
                    let pid = (idx_flat[i * k_out + j] & 0x7FFFFFFFu32) as usize;
                    pid < n && pid != sentinel_usize
                })
                .map(|j| {
                    let d = dist_flat[i * k_out + j];
                    if d < T::zero() {
                        T::zero()
                    } else {
                        d
                    }
                })
                .collect()
        })
        .collect();

    Ok((indices, distances))
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

        let vectors_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client).unwrap();
        let graph_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&graph_flat, vec![n, k_graph], &client)
                .unwrap();

        let query = vec![0.0f32; dim];

        let (indices, distances) = cagra_search_batch_gpu(
            &query,
            1,
            dim,
            &vectors_gpu,
            &graph_gpu,
            n,
            k_graph,
            k_out,
            false,
            42,
            &CagraGpuSearchParams::default(),
            None,
            &client,
        )
        .unwrap();

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

        let vectors_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client).unwrap();
        let graph_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&graph_flat, vec![n, k_graph], &client)
                .unwrap();

        let (gpu_indices, _) = cagra_search_batch_gpu(
            &queries,
            n_queries,
            dim,
            &vectors_gpu,
            &graph_gpu,
            n,
            k_graph,
            k_out,
            false,
            42,
            &CagraGpuSearchParams::default(),
            None,
            &client,
        )
        .unwrap();

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

    /// A beam that visits far more nodes than the visited table holds. Before
    /// the table was reset, every insert past `hash_size` was dropped and the
    /// search stopped early, so recall collapsed once the table filled.
    #[test]
    fn test_beam_search_outgrows_the_hash_table() {
        let Some(device) = try_device() else {
            eprintln!("Skipping: no wgpu backend");
            return;
        };

        let client = WgpuRuntime::client(&device);
        let n = 2000usize;
        let dim = 32usize;
        let k_graph = 15usize;
        let k_out = 10usize;
        let n_queries = 20usize;
        let (beam_width, max_iters, n_entry, expand) = (64usize, 192usize, 8usize, 3usize);
        let hash_size = 128usize;

        let mut rng = SmallRng::seed_from_u64(7);
        let data: Vec<f32> = (0..n * dim)
            .map(|_| rng.random_range(-10.0..10.0f32))
            .collect();
        let graph_flat = build_brute_force_graph(&data, n, dim, k_graph);
        let queries: Vec<f32> = (0..n_queries * dim)
            .map(|_| rng.random_range(-10.0..10.0f32))
            .collect();
        let entries: Vec<u32> = (0..n_queries * n_entry)
            .map(|_| rng.random_range(0..n as u32))
            .collect();

        let vectors_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&data, vec![n, dim], &client).unwrap();
        let graph_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&graph_flat, vec![n, k_graph], &client)
                .unwrap();
        let queries_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(&queries, vec![n_queries, dim], &client)
                .unwrap();
        let entry_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&entries, vec![n_queries, n_entry], &client)
                .unwrap();
        let out_idx =
            GpuTensor::<WgpuRuntime, u32>::empty(vec![n_queries, k_out], &client).unwrap();
        let out_dist =
            GpuTensor::<WgpuRuntime, f32>::empty(vec![n_queries, k_out], &client).unwrap();
        let out_iters = GpuTensor::<WgpuRuntime, u32>::empty(vec![n_queries], &client).unwrap();

        unsafe {
            cagra_beam_search::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(n_queries as u32, 1, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                LINE_SIZE,
                vectors_gpu.into_tensor_arg(),
                graph_gpu.into_tensor_arg(),
                queries_gpu.into_tensor_arg(),
                entry_gpu.into_tensor_arg(),
                out_idx.clone().into_tensor_arg(),
                out_dist.into_tensor_arg(),
                out_iters.into_tensor_arg(),
                n as u32,
                k_out as u32,
                k_graph,
                false,
                dim / LINE_SIZE,
                beam_width,
                hash_size,
                max_iters,
                n_entry,
                expand,
            );
        }

        let idx = out_idx.read(&client).unwrap();
        let gt = brute_force_knn(&queries, &data, n_queries, n, dim, k_out);
        let hits: usize = (0..n_queries)
            .map(|qi| {
                let found = &idx[qi * k_out..(qi + 1) * k_out];
                gt[qi]
                    .iter()
                    .filter(|&&g| found.contains(&(g as u32)))
                    .count()
            })
            .sum();
        let recall = hits as f64 / (n_queries * k_out) as f64;
        println!("Recall with a {hash_size}-slot table and beam {beam_width}: {recall:.4}");
        assert!(recall > 0.85, "Recall too low: {recall:.4}");
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
