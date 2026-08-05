//! GPU-accelerated distance calculations and top-k selection for GPU-based
//! indices. Contains kernels for both the exhaustive search pipeline and the
//! IVF fire-and-forget pipeline.

#![allow(missing_docs)]

use cubecl::prelude::*;
use std::iter::Sum;
use cubecl_utils_rs::prelude::*;

use crate::gpu::*;
use crate::prelude::KnnResult;
use crate::utils::dist::Dist;

/////////////
// Helpers //
/////////////

/// Container for batch query/DB data passed to `query_batch_gpu`
pub struct BatchData<'a, T> {
    /// Flattened vector data (n * dim elements)
    pub data: &'a [T],
    /// Pre-computed L2 norms (n elements, empty if not cosine)
    pub norm: &'a [T],
    /// Number of vectors
    pub n: usize,
}

impl<'a, T> BatchData<'a, T> {
    /// Create a new BatchData instance
    ///
    /// ### Params
    ///
    /// * `data` - Flattened vector data `[n * dim]`
    /// * `norm` - Pre-computed L2 norms `[n]`, empty slice if not using cosine
    /// * `n` - Number of vectors
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(data: &'a [T], norm: &'a [T], n: usize) -> Self {
        Self { data, norm, n }
    }
}

///////////////////////
// Exhaustive search //
///////////////////////

/// Tiled squared Euclidean distance kernel with shared-memory query caching
///
/// All threads in a workgroup cooperatively load the query tile into scalar
/// shared memory, eliminating redundant global reads across threads sharing
/// the same query row. DB vectors are read directly from global memory via
/// the `db_start` offset into a pre-uploaded full DB tensor.
///
/// Shared memory usage: `WORKGROUP_SIZE_Y * dim_lines * N` scalars.
///
/// ### Params
///
/// * `query_vectors` - Query vectors `[n_queries, dim / N]` as `Vector<F, N>`
/// * `db_vectors` - Database vectors `[n_db, dim / N]` as `Vector<F, N>`
/// * `distances` - Output distance matrix `[n_queries, dist_stride]`
/// * `db_start` - Global offset into `db_vectors` for this chunk
/// * `n_db_chunk` - Number of DB vectors in this chunk
/// * `n_queries` - Total number of query vectors
/// * `dist_stride` - Column stride of the output distance matrix
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime)
/// * `size_y` - Safe workgroup size Y for the given dimensionality
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> DB vector index within chunk
/// * `ABSOLUTE_POS_Y` -> query vector index
#[cube(launch_unchecked)]
pub fn euclidean_tiled<F: Float, N: Size>(
    query_vectors: &Tensor<Vector<F, N>>,
    db_vectors: &Tensor<Vector<F, N>>,
    distances: &mut Tensor<F>,
    db_start: u32,
    n_db_chunk: u32,
    n_queries: u32,
    dist_stride: u32,
    #[comptime] dim_lines: usize,
    #[comptime] size_y: u32,
) {
    let lanes = LINE_SIZE;
    let db_idx = ABSOLUTE_POS_X as usize;
    let query_idx = ((CUBE_POS_Z * CUBE_COUNT_Y + CUBE_POS_Y) * size_y + UNIT_POS_Y) as usize;
    let local_y = UNIT_POS_Y as usize;
    let local_x = UNIT_POS_X as usize;
    let dim_scalars = dim_lines * lanes;
    let wg_y = size_y as usize;
    // Scalar shared memory only (vectorised shared mem silently broadcasts lane 0)
    let mut s_query = SharedMemory::<F>::new(wg_y * dim_scalars);
    // Cooperative load: all threads in the workgroup fill the query tile
    let thread_id = local_y * WORKGROUP_SIZE_X as usize + local_x;
    let total_threads = WORKGROUP_SIZE_X as usize * wg_y;
    let total_elems = wg_y * dim_scalars;
    let q_base = query_idx - local_y;
    let mut load_idx = thread_id;
    while load_idx < total_elems {
        let q_local = load_idx / dim_scalars;
        let elem = load_idx % dim_scalars;
        let q_global = q_base + q_local;
        if q_global < n_queries as usize {
            let line_idx = elem / lanes;
            let lane = elem % lanes;
            let line_val = query_vectors[q_global * dim_lines + line_idx];
            s_query[load_idx] = line_val[lane];
        } else {
            s_query[load_idx] = F::new(0.0_f32);
        }
        load_idx += total_threads;
    }
    sync_cube();
    if query_idx >= n_queries as usize || db_idx >= n_db_chunk as usize {
        terminate!();
    }
    let global_db_idx = db_start as usize + db_idx;
    let q_shared_base = local_y * dim_scalars;
    let mut sum = F::new(0.0_f32);
    for i in 0..dim_lines {
        let d_line = db_vectors[global_db_idx * dim_lines + i];
        let s_off = q_shared_base + i * lanes;
        #[unroll]
        for lane in 0..lanes {
            let diff = s_query[s_off + lane] - d_line[lane];
            sum += diff * diff;
        }
    }
    distances[query_idx * dist_stride as usize + db_idx] = sum;
}

/// Tiled cosine distance kernel with shared-memory query caching
///
/// Same tiling strategy as `euclidean_tiled` but computes
/// `1 - dot(q, d) / (||q|| * ||d||)`.
///
/// ### Params
///
/// * `query_vectors` - Query vectors `[n_queries, dim / N]` as `Vector<F, N>`
/// * `db_vectors` - Database vectors `[n_db, dim / N]` as `Vector<F, N>`
/// * `query_norms` - Pre-computed L2 norms `[n_queries]`
/// * `db_norms` - Pre-computed L2 norms `[n_db]`
/// * `distances` - Output distance matrix `[n_queries, dist_stride]`
/// * `db_start` - Global offset into `db_vectors` for this chunk
/// * `n_db_chunk` - Number of DB vectors in this chunk
/// * `n_queries` - Total number of query vectors
/// * `dist_stride` - Column stride of the output distance matrix
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime)
/// * `size_y` - Safe workgroup size Y for the given dimensionality
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> DB vector index within chunk
/// * `ABSOLUTE_POS_Y` -> query vector index
#[cube(launch_unchecked)]
pub fn cosine_tiled<F: Float, N: Size>(
    query_vectors: &Tensor<Vector<F, N>>,
    db_vectors: &Tensor<Vector<F, N>>,
    query_norms: &Tensor<F>,
    db_norms: &Tensor<F>,
    distances: &mut Tensor<F>,
    db_start: u32,
    n_db_chunk: u32,
    n_queries: u32,
    dist_stride: u32,
    #[comptime] dim_lines: usize,
    #[comptime] size_y: u32,
) {
    let lanes = LINE_SIZE;
    let db_idx = ABSOLUTE_POS_X as usize;
    let query_idx = ((CUBE_POS_Z * CUBE_COUNT_Y + CUBE_POS_Y) * size_y + UNIT_POS_Y) as usize;
    let local_y = UNIT_POS_Y as usize;
    let local_x = UNIT_POS_X as usize;
    let dim_scalars = dim_lines * lanes;
    let wg_y = size_y as usize;
    // Scalar shared memory only (vectorised shared mem silently broadcasts lane 0)
    let mut s_query = SharedMemory::<F>::new(wg_y * dim_scalars);
    // Cooperative load: all threads in the workgroup fill the query tile
    let thread_id = local_y * WORKGROUP_SIZE_X as usize + local_x;
    let total_threads = WORKGROUP_SIZE_X as usize * wg_y;
    let total_elems = wg_y * dim_scalars;
    let q_base = query_idx - local_y;
    let mut load_idx = thread_id;
    while load_idx < total_elems {
        let q_local = load_idx / dim_scalars;
        let elem = load_idx % dim_scalars;
        let q_global = q_base + q_local;
        if q_global < n_queries as usize {
            let line_idx = elem / lanes;
            let lane = elem % lanes;
            let line_val = query_vectors[q_global * dim_lines + line_idx];
            s_query[load_idx] = line_val[lane];
        } else {
            s_query[load_idx] = F::new(0.0_f32);
        }
        load_idx += total_threads;
    }
    sync_cube();
    if query_idx >= n_queries as usize || db_idx >= n_db_chunk as usize {
        terminate!();
    }
    let global_db_idx = db_start as usize + db_idx;
    let q_shared_base = local_y * dim_scalars;
    let mut dot = F::new(0.0_f32);
    for i in 0..dim_lines {
        let d_line = db_vectors[global_db_idx * dim_lines + i];
        let s_off = q_shared_base + i * lanes;
        #[unroll]
        for lane in 0..lanes {
            dot += s_query[s_off + lane] * d_line[lane];
        }
    }
    let q_norm = query_norms[query_idx];
    let d_norm = db_norms[global_db_idx];
    distances[query_idx * dist_stride as usize + db_idx] =
        F::new(1.0_f32) - (dot / (q_norm * d_norm));
}

/// Register-tiled Euclidean distance kernel
///
/// Each thread computes a `tile_q x tile_d` block of the distance matrix
/// instead of a single entry, so each loaded value feeds several FMAs.
///
/// ### Params
///
/// * `query_vectors` - Query vectors `[n_queries, dim / N]` as `Vector<F, N>`
/// * `db_vectors` - Database vectors `[n_db, dim / N]` as `Vector<F, N>`
/// * `distances` - Output distance matrix `[n_queries, dist_stride]`
/// * `db_start` - Global offset into `db_vectors` for this chunk
/// * `n_db_chunk` - Number of DB vectors in this chunk
/// * `n_queries` - Total number of query vectors
/// * `dist_stride` - Column stride of the output distance matrix
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime)
/// * `size_y` - Number of query rows staged in shared memory (comptime).
///   Must be divisible by `tile_q`
/// * `tile_d` - DB vectors per thread (comptime)
/// * `tile_q` - Query vectors per thread (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> block of `WORKGROUP_SIZE_X * tile_d` DB vectors
/// * `UNIT_POS_X` -> lane within that block
/// * `UNIT_POS_Y` -> block of `tile_q` query rows within the shared tile
#[cube(launch_unchecked)]
pub fn euclidean_tiled_reg<F: Float, N: Size>(
    query_vectors: &Tensor<Vector<F, N>>,
    db_vectors: &Tensor<Vector<F, N>>,
    distances: &mut Tensor<F>,
    db_start: u32,
    n_db_chunk: u32,
    n_queries: u32,
    dist_stride: u32,
    #[comptime] dim_lines: usize,
    #[comptime] size_y: u32,
    #[comptime] tile_d: usize,
    #[comptime] tile_q: usize,
) {
    let lanes = LINE_SIZE;
    let dim_scalars = dim_lines * lanes;
    let wg_y = size_y as usize;
    let local_x = UNIT_POS_X as usize;
    let local_y = UNIT_POS_Y as usize;

    // Scalar shared memory only (vectorised shared mem silently broadcasts lane 0)
    let mut s_query = SharedMemory::<F>::new(wg_y * dim_scalars);

    // Locals at kernel scope, never inside a branch or loop.
    let mut acc = Array::<F>::new(tile_q * tile_d);
    let mut d_scalars = Array::<F>::new(tile_d * lanes);

    let threads_y = wg_y / tile_q;
    let thread_id = local_y * WORKGROUP_SIZE_X as usize + local_x;
    let total_threads = WORKGROUP_SIZE_X as usize * threads_y;
    let total_elems = wg_y * dim_scalars;

    // First query row owned by this cube.
    let q_tile_base = ((CUBE_POS_Z * CUBE_COUNT_Y + CUBE_POS_Y) as usize) * wg_y;

    let mut load_idx = thread_id;
    while load_idx < total_elems {
        let q_local = load_idx / dim_scalars;
        let elem = load_idx % dim_scalars;
        let q_global = q_tile_base + q_local;
        if q_global < n_queries as usize {
            let line_idx = elem / lanes;
            let lane = elem % lanes;
            let line_val = query_vectors[q_global * dim_lines + line_idx];
            s_query[load_idx] = line_val[lane];
        } else {
            s_query[load_idx] = F::new(0.0_f32);
        }
        load_idx += total_threads;
    }
    sync_cube();

    #[unroll]
    for a in 0..tile_q * tile_d {
        acc[a] = F::new(0.0_f32);
    }

    let db_tile_base = (CUBE_POS_X as usize) * (WORKGROUP_SIZE_X as usize) * tile_d;
    let q_row_base = local_y * tile_q;

    for i in 0..dim_lines {
        // Stage this thread's tile_d DB lines as scalars.
        #[unroll]
        for r in 0..tile_d {
            let db_local = db_tile_base + local_x + r * WORKGROUP_SIZE_X as usize;
            // Clamp out-of-range rows to row 0; the result is masked on write.
            let mut idx = 0usize;
            if db_local < n_db_chunk as usize {
                idx = (db_start as usize + db_local) * dim_lines + i;
            }
            let line_val = db_vectors[idx];
            #[unroll]
            for lane in 0..lanes {
                d_scalars[r * lanes + lane] = line_val[lane];
            }
        }

        #[unroll]
        for t in 0..tile_q {
            let s_off = (q_row_base + t) * dim_scalars + i * lanes;
            #[unroll]
            for lane in 0..lanes {
                let qv = s_query[s_off + lane];
                #[unroll]
                for r in 0..tile_d {
                    let diff = qv - d_scalars[r * lanes + lane];
                    acc[t * tile_d + r] += diff * diff;
                }
            }
        }
    }

    #[unroll]
    for t in 0..tile_q {
        let q_global = q_tile_base + q_row_base + t;
        #[unroll]
        for r in 0..tile_d {
            let db_local = db_tile_base + local_x + r * WORKGROUP_SIZE_X as usize;
            if q_global < n_queries as usize && db_local < n_db_chunk as usize {
                distances[q_global * dist_stride as usize + db_local] = acc[t * tile_d + r];
            }
        }
    }
}

/// Register-tiled cosine distance kernel
///
/// Same tiling strategy as `euclidean_tiled_reg` but accumulates dot products
/// and normalises on writeback, computing `1 - dot(q, d) / (||q|| * ||d||)`.
///
/// ### Params
///
/// * `query_vectors` - Query vectors `[n_queries, dim / N]` as `Vector<F, N>`
/// * `db_vectors` - Database vectors `[n_db, dim / N]` as `Vector<F, N>`
/// * `query_norms` - Pre-computed L2 norms `[n_queries]`
/// * `db_norms` - Pre-computed L2 norms `[n_db]`
/// * `distances` - Output distance matrix `[n_queries, dist_stride]`
/// * `db_start` - Global offset into `db_vectors` for this chunk
/// * `n_db_chunk` - Number of DB vectors in this chunk
/// * `n_queries` - Total number of query vectors
/// * `dist_stride` - Column stride of the output distance matrix
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime)
/// * `size_y` - Number of query rows staged in shared memory (comptime).
///   Must be divisible by `tile_q`
/// * `tile_d` - DB vectors per thread (comptime)
/// * `tile_q` - Query vectors per thread (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> block of `WORKGROUP_SIZE_X * tile_d` DB vectors
/// * `UNIT_POS_X` -> lane within that block
/// * `UNIT_POS_Y` -> block of `tile_q` query rows within the shared tile
#[cube(launch_unchecked)]
pub fn cosine_tiled_reg<F: Float, N: Size>(
    query_vectors: &Tensor<Vector<F, N>>,
    db_vectors: &Tensor<Vector<F, N>>,
    query_norms: &Tensor<F>,
    db_norms: &Tensor<F>,
    distances: &mut Tensor<F>,
    db_start: u32,
    n_db_chunk: u32,
    n_queries: u32,
    dist_stride: u32,
    #[comptime] dim_lines: usize,
    #[comptime] size_y: u32,
    #[comptime] tile_d: usize,
    #[comptime] tile_q: usize,
) {
    let lanes = LINE_SIZE;
    let dim_scalars = dim_lines * lanes;
    let wg_y = size_y as usize;
    let local_x = UNIT_POS_X as usize;
    let local_y = UNIT_POS_Y as usize;

    // Scalar shared memory only (vectorised shared mem silently broadcasts lane 0)
    let mut s_query = SharedMemory::<F>::new(wg_y * dim_scalars);

    // Locals at kernel scope, never inside a branch or loop.
    let mut acc = Array::<F>::new(tile_q * tile_d);
    let mut d_scalars = Array::<F>::new(tile_d * lanes);

    let threads_y = wg_y / tile_q;
    let thread_id = local_y * WORKGROUP_SIZE_X as usize + local_x;
    let total_threads = WORKGROUP_SIZE_X as usize * threads_y;
    let total_elems = wg_y * dim_scalars;

    let q_tile_base = ((CUBE_POS_Z * CUBE_COUNT_Y + CUBE_POS_Y) as usize) * wg_y;

    let mut load_idx = thread_id;
    while load_idx < total_elems {
        let q_local = load_idx / dim_scalars;
        let elem = load_idx % dim_scalars;
        let q_global = q_tile_base + q_local;
        if q_global < n_queries as usize {
            let line_idx = elem / lanes;
            let lane = elem % lanes;
            let line_val = query_vectors[q_global * dim_lines + line_idx];
            s_query[load_idx] = line_val[lane];
        } else {
            s_query[load_idx] = F::new(0.0_f32);
        }
        load_idx += total_threads;
    }
    sync_cube();

    #[unroll]
    for a in 0..tile_q * tile_d {
        acc[a] = F::new(0.0_f32);
    }

    let db_tile_base = (CUBE_POS_X as usize) * (WORKGROUP_SIZE_X as usize) * tile_d;
    let q_row_base = local_y * tile_q;

    for i in 0..dim_lines {
        #[unroll]
        for r in 0..tile_d {
            let db_local = db_tile_base + local_x + r * WORKGROUP_SIZE_X as usize;
            // Clamp out-of-range rows to row 0; the result is masked on write.
            let mut idx = 0usize;
            if db_local < n_db_chunk as usize {
                idx = (db_start as usize + db_local) * dim_lines + i;
            }
            let line_val = db_vectors[idx];
            #[unroll]
            for lane in 0..lanes {
                d_scalars[r * lanes + lane] = line_val[lane];
            }
        }

        #[unroll]
        for t in 0..tile_q {
            let s_off = (q_row_base + t) * dim_scalars + i * lanes;
            #[unroll]
            for lane in 0..lanes {
                let qv = s_query[s_off + lane];
                #[unroll]
                for r in 0..tile_d {
                    acc[t * tile_d + r] += qv * d_scalars[r * lanes + lane];
                }
            }
        }
    }

    #[unroll]
    for t in 0..tile_q {
        let q_global = q_tile_base + q_row_base + t;
        #[unroll]
        for r in 0..tile_d {
            let db_local = db_tile_base + local_x + r * WORKGROUP_SIZE_X as usize;
            if q_global < n_queries as usize && db_local < n_db_chunk as usize {
                let q_norm = query_norms[q_global];
                let d_norm = db_norms[db_start as usize + db_local];
                distances[q_global * dist_stride as usize + db_local] =
                    F::new(1.0_f32) - (acc[t * tile_d + r] / (q_norm * d_norm));
            }
        }
    }
}

/////////////////////
// Top-k selection //
/////////////////////

/// Returns true if the runtime benefits from coalesced top-k extraction.
/// Discrete GPUs (CUDA, Vulkan) have strict coalescing requirements.
/// Unified memory architectures (Metal/Apple) are largely indifferent.
///
/// ### Returns
///
/// True if not metal
#[allow(dead_code)]
fn prefer_coalesced_topk<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let name = R::name(client).to_lowercase();
    // Metal on Apple Silicon handles both patterns equally well.
    // CUDA, Vulkan, and OpenCL benefit significantly from coalescing.
    !name.contains("metal")
}

/// Initialise top-k buffers to sentinel values (`f32::MAX` / `0`)
///
/// ### Params
///
/// * `dists` - Distance buffer `[n_queries, k]` to fill with `f32::MAX`
/// * `indices` - Index buffer `[n_queries, k]` to fill with `0`
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> k slot index
/// * `ABSOLUTE_POS_Y` -> query index
/// * `size_y` - Safe workgroup size Y for the given dimensionality
#[cube(launch_unchecked)]
pub fn init_topk<F: Float>(
    dists: &mut Tensor<F>,
    indices: &mut Tensor<u32>,
    #[comptime] size_y: u32,
) {
    let query_idx = ((CUBE_POS_Z * CUBE_COUNT_Y + CUBE_POS_Y) * size_y + UNIT_POS_Y) as usize;
    let k_idx = ABSOLUTE_POS_X as usize;
    let k = dists.shape(1);

    if query_idx >= dists.shape(0) || k_idx >= k {
        terminate!();
    }

    let offset = query_idx * dists.stride(0) + k_idx;
    dists[offset] = F::new(f32::MAX);
    indices[offset] = 0u32;
}

/// Extract top-k smallest distances per query via insertion sort
///
/// One thread per query, serial scan of the distance row. The running top-k
/// is held in registers for the duration of the scan and flushed back once at
/// the end, so the per-candidate cost is a single global read of the distance
/// matrix. The buffer must be pre-initialised with `init_topk`.
///
/// ### Params
///
/// * `distances` - Full distance matrix for this chunk
///   `[n_queries, dist_stride]`
/// * `out_dists` - Running top-k distance buffer `[n_queries, k]`
/// * `out_indices` - Running top-k index buffer `[n_queries, k]`
/// * `chunk_offset` - Global DB index corresponding to column 0 of this chunk
/// * `actual_chunk_size` - Number of valid columns in this chunk
/// * `k_param` - Runtime value of k (must equal comptime `k`)
/// * `k` - Comptime top-k count; must match `k_param` at launch (comptime)
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> query index
#[cube(launch_unchecked)]
pub fn extract_topk<F: Float>(
    distances: &Tensor<F>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
    chunk_offset: u32,
    actual_chunk_size: u32,
    k_param: u32,
    #[comptime] k: usize,
) {
    let query_idx =
        ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X) as usize;

    if query_idx >= distances.shape(0) {
        terminate!();
    }

    let kr = k_param as usize;
    let dist_offset = query_idx * distances.stride(0);
    let out_offset = query_idx * out_dists.stride(0);

    // Stage the running top-k in registers. The previous version re-read
    // `out_dists[out_offset + k - 1]` from global on every column of the chunk,
    // and re-read the whole k-row on every accepted candidate.
    let mut local_dists = Array::<F>::new(k);
    let mut local_indices = Array::<u32>::new(k);
    for i in 0..k {
        local_dists[i] = out_dists[out_offset + i];
        local_indices[i] = out_indices[out_offset + i];
    }

    for i in 0..actual_chunk_size {
        let dist = distances[dist_offset + i as usize];

        if dist < local_dists[kr - 1] {
            // First-match guard rather than a `bool` sentinel: see the codegen
            // rules in the `nndescent_gpu` module header.
            let mut pos = kr - 1;
            for j in 0..k {
                if dist < local_dists[j] && pos == kr - 1 {
                    pos = j;
                }
            }

            let mut s = kr - 1;
            while s > pos {
                local_dists[s] = local_dists[s - 1];
                local_indices[s] = local_indices[s - 1];
                s -= 1usize;
            }

            local_dists[pos] = dist;
            local_indices[pos] = chunk_offset + i;
        }
    }

    for i in 0..k {
        out_dists[out_offset + i] = local_dists[i];
        out_indices[out_offset + i] = local_indices[i];
    }
}

/// Coalesced top-k extraction. One workgroup per query, threads cooperate by
/// striding through consecutive columns of the distance row. Memory accesses
/// are coalesced: adjacent threads read adjacent elements.
///
/// Per-thread top-k kept in local Arrays (registers) during the scan,
/// copied to shared memory only for the final merge.
///
/// ### Params
///
/// * `distances` - Full distance matrix for this chunk
///   `[n_queries, dist_stride]`
/// * `out_dists` - Running top-k distance buffer `[n_queries, k]`
/// * `out_indices` - Running top-k index buffer `[n_queries, k]`
/// * `chunk_offset` - Global DB index corresponding to column 0 of this chunk
/// * `actual_chunk_size` - Number of valid columns in this chunk
/// * `dist_stride` - Column stride of the distance matrix
/// * `k_param` - Runtime value of k (must equal comptime `k`)
/// * `k` - Comptime top-k count; must match `k_param` at launch (comptime)
/// * `group` - Lanes staged per merge round (comptime)
/// * `single_round` - Whether all lanes fit in one round (comptime)
/// * `slots` - Number of `k`-element lists in shared memory (comptime)
///
/// ### Shared memory
///
/// The merge staging is chunked to the device budget by `plan_topk_merge`;
/// see [`reduce_ivf_topk_coalesced`] for the reasoning, the two kernels share
/// the scheme.
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> query index
/// * `UNIT_POS_X` -> thread within workgroup (32 threads)
#[cube(launch_unchecked)]
pub fn extract_topk_coalesced<F: Float>(
    distances: &Tensor<F>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
    chunk_offset: u32,
    actual_chunk_size: u32,
    dist_stride: u32,
    k_param: u32,
    #[comptime] k: usize,
    #[comptime] group: usize,
    #[comptime] single_round: bool,
    #[comptime] slots: usize,
) {
    let query_idx = CUBE_POS_X as usize;
    let tx = UNIT_POS_X as usize;
    let wg = WORKGROUP_SIZE_X as usize;
    let kr = k_param as usize;

    if query_idx >= out_dists.shape(0) {
        terminate!();
    }

    // Per-thread top-k in registers
    let mut local_dists = Array::<F>::new(k);
    let mut local_indices = Array::<u32>::new(k);
    for i in 0..k {
        local_dists[i] = F::new(f32::MAX);
        local_indices[i] = 0u32;
    }

    // Coalesced strided scan: adjacent threads read adjacent columns
    let dist_base = query_idx * dist_stride as usize;
    let mut col = tx as u32;
    while col < actual_chunk_size {
        let dist = distances[dist_base + col as usize];

        if dist < local_dists[kr - 1] {
            let mut pos = kr - 1;
            for j in 0..k {
                if dist < local_dists[j] && pos == kr - 1 {
                    pos = j;
                }
            }

            let mut s = kr - 1;
            while s > pos {
                local_dists[s] = local_dists[s - 1];
                local_indices[s] = local_indices[s - 1];
                s -= 1usize;
            }

            local_dists[pos] = dist;
            local_indices[pos] = chunk_offset + col;
        }

        col += wg as u32;
    }

    let mut s_dist = SharedMemory::<F>::new(slots * k);
    let mut s_idx = SharedMemory::<u32>::new(slots * k);

    if single_round {
        // -- Unchunked fast path --
        //
        // Every lane's list fits at once, so this is the original kernel
        // verbatim: `slots == WORKGROUP_SIZE_X` and thread 0's own slot doubles
        // as the accumulator, giving exactly the pre-chunking footprint.
        let s_base = tx * kr;
        for i in 0..k {
            s_dist[s_base + i] = local_dists[i];
            s_idx[s_base + i] = local_indices[i];
        }

        sync_cube();

        if tx == 0usize {
            // Merge threads 1..31 into thread 0's list
            let mut t = 1usize;
            while t < wg {
                let t_base = t * kr;
                let mut done: u32 = 0u32;
                for i in 0..k {
                    if done == 0u32 {
                        let cd = s_dist[t_base + i];
                        let ci = s_idx[t_base + i];

                        if cd >= s_dist[kr - 1] {
                            done = 1u32;
                        } else {
                            let mut pos = kr - 1;
                            for j in 0..k {
                                if cd < s_dist[j] && pos == kr - 1 {
                                    pos = j;
                                }
                            }

                            let mut s_i = kr - 1;
                            while s_i > pos {
                                s_dist[s_i] = s_dist[s_i - 1];
                                s_idx[s_i] = s_idx[s_i - 1];
                                s_i -= 1usize;
                            }

                            s_dist[pos] = cd;
                            s_idx[pos] = ci;
                        }
                    }
                }
                t += 1usize;
            }

            // Merge with running top-k from previous chunks
            let out_base = query_idx * kr;
            for i in 0..k {
                let running_dist = out_dists[out_base + i];
                let running_idx = out_indices[out_base + i];

                if running_dist < s_dist[kr - 1] {
                    let mut pos = kr - 1;
                    for j in 0..k {
                        if running_dist < s_dist[j] && pos == kr - 1 {
                            pos = j;
                        }
                    }

                    let mut s_i = kr - 1;
                    while s_i > pos {
                        s_dist[s_i] = s_dist[s_i - 1];
                        s_idx[s_i] = s_idx[s_i - 1];
                        s_i -= 1usize;
                    }

                    s_dist[pos] = running_dist;
                    s_idx[pos] = running_idx;
                }
            }

            // Write final result
            for i in 0..k {
                out_dists[out_base + i] = s_dist[i];
                out_indices[out_base + i] = s_idx[i];
            }
        }
    } else {
        // -- Chunked merge --
        //
        // `group` lanes stage their lists per round; thread 0 folds each round
        // into the accumulator list held at slot `group`.
        let acc = group * kr;

        if tx == 0usize {
            for i in 0..k {
                s_dist[acc + i] = F::new(f32::MAX);
                s_idx[acc + i] = 0u32;
            }
        }
        sync_cube();

        let n_rounds = wg / group;
        let mut round = 0usize;
        while round < n_rounds {
            if tx / group == round {
                let s_base = (tx % group) * kr;
                for i in 0..k {
                    s_dist[s_base + i] = local_dists[i];
                    s_idx[s_base + i] = local_indices[i];
                }
            }
            sync_cube();

            if tx == 0usize {
                let mut t = 0usize;
                while t < group {
                    let t_base = t * kr;
                    let mut done: u32 = 0u32;
                    for i in 0..k {
                        if done == 0u32 {
                            let cd = s_dist[t_base + i];
                            let ci = s_idx[t_base + i];

                            if cd >= s_dist[acc + kr - 1] {
                                done = 1u32;
                            } else {
                                let mut pos = kr - 1;
                                for j in 0..k {
                                    if cd < s_dist[acc + j] && pos == kr - 1 {
                                        pos = j;
                                    }
                                }

                                let mut s_i = kr - 1;
                                while s_i > pos {
                                    s_dist[acc + s_i] = s_dist[acc + s_i - 1];
                                    s_idx[acc + s_i] = s_idx[acc + s_i - 1];
                                    s_i -= 1usize;
                                }

                                s_dist[acc + pos] = cd;
                                s_idx[acc + pos] = ci;
                            }
                        }
                    }
                    t += 1usize;
                }
            }
            // Thread 0 must finish reading the staged slots before the next
            // round's lanes overwrite them.
            sync_cube();
            round += 1usize;
        }

        if tx == 0usize {
            // Merge with running top-k from previous chunks
            let out_base = query_idx * kr;
            for i in 0..k {
                let running_dist = out_dists[out_base + i];
                let running_idx = out_indices[out_base + i];

                if running_dist < s_dist[acc + kr - 1] {
                    let mut pos = kr - 1;
                    for j in 0..k {
                        if running_dist < s_dist[acc + j] && pos == kr - 1 {
                            pos = j;
                        }
                    }

                    let mut s_i = kr - 1;
                    while s_i > pos {
                        s_dist[acc + s_i] = s_dist[acc + s_i - 1];
                        s_idx[acc + s_i] = s_idx[acc + s_i - 1];
                        s_i -= 1usize;
                    }

                    s_dist[acc + pos] = running_dist;
                    s_idx[acc + pos] = running_idx;
                }
            }

            // Write final result
            for i in 0..k {
                out_dists[out_base + i] = s_dist[acc + i];
                out_indices[out_base + i] = s_idx[acc + i];
            }
        }
    }
}

/// Run batch kNN queries on the GPU
///
/// Uses tiled distance kernels with shared-memory query caching, a single
/// DB upload, and serial insertion-sort top-k extraction directly into a
/// running buffer (no ping-pong or merge step).
///
/// ### Params
///
/// * `k` - Number of neighbours to return
/// * `query_data` - Query vectors as `BatchData`
/// * `db_data` - Database vectors as `BatchData`
/// * `dim` - Embedding dimensionality (must be divisible by LINE_SIZE)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `device` - CubeCL runtime device
/// * `verbose` - Print progress for large batches
///
/// ### Returns
///
/// Tuple of `(indices, distances)` where each inner Vec has k elements
pub fn query_batch_gpu<T, R>(
    k: usize,
    query_data: &BatchData<T>,
    db_data: &BatchData<T>,
    dim: usize,
    metric: &Dist,
    device: R::Device,
    verbose: bool,
) -> KnnResult<T>
where
    R: Runtime,
    T: Float + Sum + cubecl::CubeElement + num_traits::Float + num_traits::FromPrimitive,
{
    let client = R::client(&device);
    let limits = GpuLimits::from_client(&client);
    let vec_size = LINE_SIZE;
    let dim_lines = dim / vec_size;
    let safe_worksize_y = pick_wg_y(dim, size_of::<T>(), &limits)?;

    let n_query_chunks = query_data.n.div_ceil(QUERY_CHUNK_SIZE);
    // The DB chunk shrinks when the distance transient would not fit one
    // binding on this device. On a 4 GiB binding limit it is the full chunk.
    let db_chunk = plan_db_chunk(QUERY_CHUNK_SIZE.min(query_data.n), size_of::<T>(), &limits);
    let n_db_chunks = db_data.n.div_ceil(db_chunk);

    // Single DB upload for the entire query
    let db_gpu = GpuTensor::<R, T>::from_slice(db_data.data, vec![db_data.n, dim], &client)?;

    let db_norms_gpu = if *metric == Dist::Cosine {
        Some(GpuTensor::<R, T>::from_slice(
            db_data.norm,
            vec![db_data.n],
            &client,
        )?)
    } else {
        None
    };

    let mut all_indices = Vec::with_capacity(query_data.n);
    let mut all_distances = Vec::with_capacity(query_data.n);

    let max_db_chunk = db_chunk.min(db_data.n);

    for query_chunk_idx in 0..n_query_chunks {
        if verbose && query_chunk_idx % 10 == 0 {
            println!(
                "Processed {} query chunks out of {}",
                query_chunk_idx, n_query_chunks
            );
        }

        let query_start = query_chunk_idx * QUERY_CHUNK_SIZE;
        let query_end = (query_start + QUERY_CHUNK_SIZE).min(query_data.n);
        let n_q = query_end - query_start;

        let query_gpu = GpuTensor::<R, T>::from_slice(
            &query_data.data[query_start * dim..query_end * dim],
            vec![n_q, dim],
            &client,
        )?;

        let query_norms_gpu = if *metric == Dist::Cosine {
            Some(GpuTensor::<R, T>::from_slice(
                &query_data.norm[query_start..query_end],
                vec![n_q],
                &client,
            )?)
        } else {
            None
        };

        // Running top-k buffer (no ping-pong needed)
        let topk_dists = GpuTensor::<R, T>::empty(vec![n_q, k], &client)?;
        let topk_indices = GpuTensor::<R, u32>::empty(vec![n_q, k], &client)?;

        // The x axis is the neighbour count, with y and z already carrying the
        // query axis, so there is nothing to flatten into. Check it instead.
        let init_gx = (k as u32).div_ceil(WORKGROUP_SIZE_X);
        let (init_gy, init_gz) = grid_2d((n_q as u32).div_ceil(safe_worksize_y), &limits)?;
        let init_count = checked_cube_count("init_topk", init_gx, init_gy, init_gz, &limits)?;
        unsafe {
            init_topk::launch_unchecked::<T, R>(
                &client,
                init_count,
                CubeDim::new_2d(WORKGROUP_SIZE_X, safe_worksize_y),
                topk_dists.clone().into_tensor_arg(),
                topk_indices.clone().into_tensor_arg(),
                safe_worksize_y,
            );
        }

        // Reusable distance buffer sized for the largest possible chunk
        let distances_gpu = GpuTensor::<R, T>::empty(vec![n_q, max_db_chunk], &client)?;

        for db_chunk_idx in 0..n_db_chunks {
            let db_start = db_chunk_idx * db_chunk;
            let db_end = (db_start + db_chunk).min(db_data.n);
            let n_db = db_end - db_start;

            let grid_x = (n_db as u32).div_ceil(WORKGROUP_SIZE_X);
            let (grid_y, grid_z) = grid_2d((n_q as u32).div_ceil(safe_worksize_y), &limits)?;

            match *metric {
                // Register-tiled path where the tile divides the query tile
                // height; roughly 1.4x to 2.2x over the untiled kernel and
                // bit-exact against it. See `TILE_D` for the measurements.
                Dist::SquaredEuclidean if tile_fits(safe_worksize_y) => unsafe {
                    let reg_grid_x = (n_db as u32).div_ceil(WORKGROUP_SIZE_X * TILE_D as u32);
                    euclidean_tiled_reg::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(reg_grid_x, grid_y, grid_z),
                        CubeDim::new_2d(WORKGROUP_SIZE_X, safe_worksize_y / TILE_Q as u32),
                        vec_size,
                        query_gpu.clone().into_tensor_arg(),
                        db_gpu.clone().into_tensor_arg(),
                        distances_gpu.clone().into_tensor_arg(),
                        db_start as u32,
                        n_db as u32,
                        n_q as u32,
                        max_db_chunk as u32,
                        dim_lines,
                        safe_worksize_y,
                        TILE_D,
                        TILE_Q,
                    );
                },
                Dist::SquaredEuclidean => unsafe {
                    euclidean_tiled::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(grid_x, grid_y, grid_z),
                        CubeDim::new_2d(WORKGROUP_SIZE_X, safe_worksize_y),
                        vec_size,
                        query_gpu.clone().into_tensor_arg(),
                        db_gpu.clone().into_tensor_arg(),
                        distances_gpu.clone().into_tensor_arg(),
                        db_start as u32,
                        n_db as u32,
                        n_q as u32,
                        max_db_chunk as u32,
                        dim_lines,
                        safe_worksize_y,
                    );
                },
                Dist::Cosine if tile_fits(safe_worksize_y) => unsafe {
                    let reg_grid_x = (n_db as u32).div_ceil(WORKGROUP_SIZE_X * TILE_D as u32);
                    cosine_tiled_reg::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(reg_grid_x, grid_y, grid_z),
                        CubeDim::new_2d(WORKGROUP_SIZE_X, safe_worksize_y / TILE_Q as u32),
                        vec_size,
                        query_gpu.clone().into_tensor_arg(),
                        db_gpu.clone().into_tensor_arg(),
                        query_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(),
                        db_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(),
                        distances_gpu.clone().into_tensor_arg(),
                        db_start as u32,
                        n_db as u32,
                        n_q as u32,
                        max_db_chunk as u32,
                        dim_lines,
                        safe_worksize_y,
                        TILE_D,
                        TILE_Q,
                    );
                },
                Dist::Cosine => unsafe {
                    cosine_tiled::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(grid_x, grid_y, grid_z),
                        CubeDim::new_2d(WORKGROUP_SIZE_X, safe_worksize_y),
                        vec_size,
                        query_gpu.clone().into_tensor_arg(),
                        db_gpu.clone().into_tensor_arg(),
                        query_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(),
                        db_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(),
                        distances_gpu.clone().into_tensor_arg(),
                        db_start as u32,
                        n_db as u32,
                        n_q as u32,
                        max_db_chunk as u32,
                        dim_lines,
                        safe_worksize_y,
                    );
                },
                Dist::Manhattan => unreachable!(),
            }

            // Extract directly into the running top-k buffer
            let (extract_grid_x, extract_grid_y) =
                grid_2d((n_q as u32).div_ceil(WORKGROUP_SIZE_X), &limits)?;
            unsafe {
                extract_topk::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(extract_grid_x, extract_grid_y, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    distances_gpu.clone().into_tensor_arg(),
                    topk_dists.clone().into_tensor_arg(),
                    topk_indices.clone().into_tensor_arg(),
                    db_start as u32,
                    n_db as u32,
                    k as u32,
                    k,
                );
            }
        }

        // Single GPU->CPU read per query chunk
        let final_dists = topk_dists.read(&client)?;
        let final_indices = topk_indices.read(&client)?;

        for q in 0..n_q {
            let start = q * k;
            let end = start + k;
            all_distances.push(final_dists[start..end].to_vec());
            all_indices.push(
                final_indices[start..end]
                    .iter()
                    .map(|&i| i as usize)
                    .collect(),
            );
        }
    }

    Ok((all_indices, all_distances))
}

/////////////////////////////////
// Fire-and-Forget IVF kernels //
/////////////////////////////////

/// Coalesced top-k reduction for the IVF variable-length candidate buffer.
///
/// One workgroup (32 threads) per query. Threads cooperatively stride through
/// the candidate slice, each maintaining a thread-local sorted top-k in
/// registers. After the scan, per-thread results are written to shared memory
/// and thread 0 merges all 32 sorted lists into the final top-k output.
///
/// ### Params
///
/// * `candidate_dists` - Candidate distances `[n_queries, max_candidates]`.
///   Each row contains the distances produced by the mega distance kernel,
///   of which only the first `candidates_per_query[q]` entries are valid.
/// * `candidate_indices` - Candidate DB indices `[n_queries, max_candidates]`.
///   Layout mirrors `candidate_dists`.
/// * `candidates_per_query` - Number of valid candidates per query
///   `[n_queries]`. Entries beyond this count in each row are ignored.
/// * `out_dists` - Output top-k distances `[n_queries, k]`. Written only by
///   thread 0 of each workgroup after the merge phase.
/// * `out_indices` - Output top-k DB indices `[n_queries, k]`. Written only
///   by thread 0 of each workgroup after the merge phase.
/// * `k_param` - Runtime value of k (must equal comptime `k`). Required
///   because CubeCL comptime values cannot be used in runtime assignments.
/// * `k` - Number of nearest neighbours to extract per query (comptime).
///   Used only for `Array::new` and `SharedMemory::new` sizing.
/// * `group` - Lanes whose lists are staged per merge round (comptime)
/// * `single_round` - Whether all lanes fit in one round (comptime)
/// * `slots` - Number of `k`-element lists in shared memory (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> query index (one cube per
///   query)
/// * `UNIT_POS_X` -> thread within workgroup (`0..WORKGROUP_SIZE_X`), used
///   for strided candidate scanning and cooperative merge
#[cube(launch_unchecked)]
pub fn reduce_ivf_topk_coalesced<F: Float>(
    candidate_dists: &Tensor<F>,
    candidate_indices: &Tensor<u32>,
    candidates_per_query: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
    k_param: u32,
    #[comptime] k: usize,
    #[comptime] group: usize,
    #[comptime] single_round: bool,
    #[comptime] slots: usize,
) {
    let q_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    let tx = UNIT_POS_X as usize;
    let wg = WORKGROUP_SIZE_X as usize;
    let kr = k_param as usize;

    if q_idx >= candidate_dists.shape(0) {
        terminate!();
    }

    let count = candidates_per_query[q_idx] as usize;
    let in_base = q_idx * candidate_dists.stride(0);

    // ── Phase 1: per-thread top-k in registers ──
    //
    // Structural pattern mirrors `extract_topk_coalesced`: `usize` counters
    // and comptime-unrolled `for i in 0..k` loops. cubecl 0.10 has miscompiled
    // `while init_i < k_param` init loops and short-circuited `while a && !b`
    // insertion searches with u32 counters on lavapipe/Vulkan, so both are
    // avoided here even though the algorithm is unchanged.

    let mut local_dists = Array::<F>::new(k);
    let mut local_indices = Array::<u32>::new(k);
    for i in 0..k {
        local_dists[i] = F::new(f32::MAX);
        local_indices[i] = 0u32;
    }

    // Coalesced strided scan: adjacent threads read adjacent columns
    let mut col = tx;
    while col < count {
        let dist = candidate_dists[in_base + col];

        if dist < local_dists[kr - 1] {
            // Full comptime-unrolled scan; guarded by `pos == kr - 1` so
            // only the first match wins. Matches extract_topk_coalesced.
            let mut pos = kr - 1;
            for j in 0..k {
                if dist < local_dists[j] && pos == kr - 1 {
                    pos = j;
                }
            }

            let mut s = kr - 1;
            while s > pos {
                local_dists[s] = local_dists[s - 1];
                local_indices[s] = local_indices[s - 1];
                s -= 1usize;
            }

            local_dists[pos] = dist;
            local_indices[pos] = candidate_indices[in_base + col];
        }

        col += wg;
    }

    // ── Phase 2: shared memory merge ──

    let mut s_dist = SharedMemory::<F>::new(slots * k);
    let mut s_idx = SharedMemory::<u32>::new(slots * k);

    if single_round {
        // ---- Unchunked fast path -------------------------------------------
        // Every lane's list fits at once, so this is the original kernel
        // verbatim: `slots == WORKGROUP_SIZE_X` and thread 0's own slot doubles
        // as the accumulator, giving exactly the pre-chunking footprint.
        let s_base = tx * kr;
        for i in 0..k {
            s_dist[s_base + i] = local_dists[i];
            s_idx[s_base + i] = local_indices[i];
        }

        sync_cube();

        // Thread 0 merges all 32 sorted thread-local lists
        if tx == 0usize {
            let mut t = 1usize;
            while t < wg {
                let t_base = t * kr;
                let mut done: u32 = 0u32;

                // Sorted per-thread list: once cd >= current worst, all
                // remaining are worse. `done: u32` sentinel matches extract's
                // pattern; a `bool` here miscompiles on cubecl 0.10 / lavapipe.
                for i in 0..k {
                    if done == 0u32 {
                        let cd = s_dist[t_base + i];
                        let ci = s_idx[t_base + i];

                        if cd >= s_dist[kr - 1] {
                            done = 1u32;
                        } else {
                            let mut pos = kr - 1;
                            for j in 0..k {
                                if cd < s_dist[j] && pos == kr - 1 {
                                    pos = j;
                                }
                            }

                            let mut s_i = kr - 1;
                            while s_i > pos {
                                s_dist[s_i] = s_dist[s_i - 1];
                                s_idx[s_i] = s_idx[s_i - 1];
                                s_i -= 1usize;
                            }

                            s_dist[pos] = cd;
                            s_idx[pos] = ci;
                        }
                    }
                }
                t += 1usize;
            }

            // Write final result
            let out_base = q_idx * kr;
            for i in 0..k {
                out_dists[out_base + i] = s_dist[i];
                out_indices[out_base + i] = s_idx[i];
            }
        }
    } else {
        // ---- Chunked merge --------------------------------------------------
        // `group` lanes stage their lists per round; thread 0 folds each round
        // into the accumulator list held at slot `group`. Lanes are visited in
        // ascending order exactly as on the fast path, so ties resolve the same
        // way.
        let acc = group * kr;

        if tx == 0usize {
            for i in 0..k {
                s_dist[acc + i] = F::new(f32::MAX);
                s_idx[acc + i] = 0u32;
            }
        }
        sync_cube();

        let n_rounds = wg / group;
        let mut round = 0usize;
        while round < n_rounds {
            if tx / group == round {
                let s_base = (tx % group) * kr;
                for i in 0..k {
                    s_dist[s_base + i] = local_dists[i];
                    s_idx[s_base + i] = local_indices[i];
                }
            }
            sync_cube();

            if tx == 0usize {
                let mut t = 0usize;
                while t < group {
                    let t_base = t * kr;
                    let mut done: u32 = 0u32;

                    for i in 0..k {
                        if done == 0u32 {
                            let cd = s_dist[t_base + i];
                            let ci = s_idx[t_base + i];

                            if cd >= s_dist[acc + kr - 1] {
                                done = 1u32;
                            } else {
                                let mut pos = kr - 1;
                                for j in 0..k {
                                    if cd < s_dist[acc + j] && pos == kr - 1 {
                                        pos = j;
                                    }
                                }

                                let mut s_i = kr - 1;
                                while s_i > pos {
                                    s_dist[acc + s_i] = s_dist[acc + s_i - 1];
                                    s_idx[acc + s_i] = s_idx[acc + s_i - 1];
                                    s_i -= 1usize;
                                }

                                s_dist[acc + pos] = cd;
                                s_idx[acc + pos] = ci;
                            }
                        }
                    }
                    t += 1usize;
                }
            }
            // Thread 0 must finish reading the staged slots before the next
            // round's lanes overwrite them.
            sync_cube();
            round += 1usize;
        }

        // Write final result
        if tx == 0usize {
            let out_base = q_idx * kr;
            for i in 0..k {
                out_dists[out_base + i] = s_dist[acc + i];
                out_indices[out_base + i] = s_idx[acc + i];
            }
        }
    }
}

//////////////////////////////
// IVF mega kernel variants //
//////////////////////////////

/// Compute Euclidean distances using a flattened IVF task list
///
/// Each task represents one (query, cluster) pair. The grid maps threads
/// directly to `(db_element, task)` pairs, avoiding a per-cluster kernel
/// launch.
///
/// ### Params
///
/// * `query_vectors` - Query vectors `[n_queries, dim / N]` as `Vector<F, N>`
/// * `db_vectors` - Full database vectors `[n_db, dim / N]` as `Vector<F, N>`
/// * `task_q_idx` - Query index for each task `[n_tasks]`
/// * `task_db_start` - Global DB start index for each task `[n_tasks]`
/// * `task_write_offset` - Write offset into the candidate row for each task
///   `[n_tasks]`
/// * `task_db_count` - Number of DB vectors in each task's cluster `[n_tasks]`
/// * `out_dists` - Output candidate distances `[n_queries, max_candidates]`
/// * `out_indices` - Output candidate DB indices `[n_queries, max_candidates]`
/// * `size_y` - Safe workgroup size Y for the given dimensionality
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> vector index within the task's cluster (`0..db_count`)
/// * `ABSOLUTE_POS_Y` -> task index (`0..n_tasks`)
#[cube(launch_unchecked)]
pub fn compute_ivf_mega_euclidean<F: Float, N: Size>(
    query_vectors: &Tensor<Vector<F, N>>,
    db_vectors: &Tensor<Vector<F, N>>,
    task_q_idx: &Tensor<u32>,
    task_db_start: &Tensor<u32>,
    task_write_offset: &Tensor<u32>,
    task_db_count: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
    #[comptime] size_y: u32,
) {
    let lanes = LINE_SIZE;
    let local_db_idx = ABSOLUTE_POS_X;
    let task_idx = (CUBE_POS_Z * CUBE_COUNT_Y + CUBE_POS_Y) * size_y + UNIT_POS_Y;

    if task_idx >= task_q_idx.len() as u32 {
        terminate!();
    }

    let db_count = task_db_count[task_idx as usize];
    if local_db_idx >= db_count {
        terminate!();
    }

    let q_idx = task_q_idx[task_idx as usize];
    let db_start = task_db_start[task_idx as usize];
    let write_offset = task_write_offset[task_idx as usize];

    let real_db_idx = db_start + local_db_idx;
    let write_pos = write_offset + local_db_idx;

    let mut sum = F::new(0.0_f32);

    let dim_lines = query_vectors.shape(1) / lanes;
    let q_offset = q_idx as usize * dim_lines;
    let d_offset = real_db_idx as usize * dim_lines;

    for i in 0..dim_lines {
        let q_line = query_vectors[q_offset + i];
        let d_line = db_vectors[d_offset + i];
        let diff = q_line - d_line;
        let sq = diff * diff;

        #[unroll]
        for lane in 0..lanes {
            sum += sq[lane];
        }
    }

    let out_offset = q_idx as usize * out_dists.stride(0) + write_pos as usize;
    out_dists[out_offset] = sum;
    out_indices[out_offset] = real_db_idx;
}

/// In-place top-k reduction for the IVF variable-length candidate buffer
///
/// One thread per query. Performs insertion-sort over the variable-length
/// candidate slice produced by the distance kernels and writes the k
/// smallest results into the output buffers.
///
/// ### Params
///
/// * `candidate_dists` - Candidate distances `[n_queries, max_candidates]`
/// * `candidate_indices` - Candidate DB indices `[n_queries, max_candidates]`
/// * `candidates_per_query` - Number of valid candidates for each query
///   `[n_queries]`
/// * `out_dists` - Output top-k distances `[n_queries, k]`
/// * `out_indices` - Output top-k DB indices `[n_queries, k]`
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> query index
#[cube(launch_unchecked)]
pub fn reduce_ivf_topk<F: Float>(
    candidate_dists: &Tensor<F>,
    candidate_indices: &Tensor<u32>,
    candidates_per_query: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
) {
    let q_idx = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X) as usize;

    if q_idx >= candidate_dists.shape(0) {
        terminate!();
    }

    let k = out_dists.shape(1);
    let count = candidates_per_query[q_idx];

    let in_offset = q_idx * candidate_dists.stride(0);
    let out_offset = q_idx * out_dists.stride(0);

    for i in 0..count {
        let dist = candidate_dists[in_offset + i as usize];
        let idx = candidate_indices[in_offset + i as usize];

        if dist < out_dists[out_offset + k - 1] {
            let mut insert_pos: usize = k - 1;
            for j in 0..k {
                if dist < out_dists[out_offset + j] && insert_pos == k - 1 {
                    insert_pos = j;
                }
            }

            for j in 0..k - 1 {
                let src = k - 2 - j;
                let dst = k - 1 - j;
                if src >= insert_pos {
                    out_dists[out_offset + dst] = out_dists[out_offset + src];
                    out_indices[out_offset + dst] = out_indices[out_offset + src];
                }
            }

            out_dists[out_offset + insert_pos] = dist;
            out_indices[out_offset + insert_pos] = idx;
        }
    }
}

/// Euclidean mega kernel with shared-memory query caching.
///
/// Same task-based architecture as `compute_ivf_mega_euclidean`, but
/// cooperatively loads query vectors into scalar shared memory so that all
/// X-threads in a workgroup row (and Y-threads sharing the same query) read
/// from shared memory instead of global memory.
///
/// Also caches per-task metadata (q_idx, db_start, write_offset, db_count) in
/// shared memory to avoid redundant global reads across the 32 X-threads in
/// each row.
///
/// ### Params
///
/// * `query_vectors` - Query vectors `[n_queries, dim]` as `Vector<F, N>`.
///   Shape must be in element units (not vector units). Accessed via
///   the comptime `dim_lines` parameter, never via tensor strides.
/// * `db_vectors` - Full database vectors `[n_db, dim]` as `Vector<F, N>`.
///   Same element-unit shape convention as `query_vectors`.
/// * `task_q_idx` - Query index for each task `[n_tasks]`. Tasks must
///   be sorted by this value for optimal shared-memory reuse.
/// * `task_db_start` - Global DB start index for each task `[n_tasks]`.
///   Points into the cluster-reorganised `db_vectors`.
/// * `task_write_offset` - Write offset into the candidate row for each
///   task `[n_tasks]`. Determines where this task's distances are written
///   within the query's candidate buffer row.
/// * `task_db_count` - Number of DB vectors in each task's cluster
///   `[n_tasks]`. Threads with `ABSOLUTE_POS_X >= db_count` terminate.
/// * `out_dists` - Output candidate distances `[n_queries, max_candidates]`.
///   Written at position `[q_idx, write_offset + local_db_idx]`.
/// * `out_indices` - Output candidate DB indices `[n_queries, max_candidates]`.
///   Written at the same position as `out_dists`.
/// * `n_tasks` - Total number of tasks. Used for bounds checking since
///   `task_q_idx.len()` is unreliable when vectorised tensors are present
///   in the same kernel.
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime).
///   Equal to `dim / N`. Used for all indexing into vector tensors and for
///   shared memory sizing. Passed as comptime to avoid reliance on tensor
///   metadata.
/// * `size_y` - Safe workgroup size Y for the given dimensionality
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> vector index within the task's cluster
///   (`0..db_count`)
/// * `(CUBE_POS_Z * CUBE_COUNT_Y + CUBE_POS_Y) * WORKGROUP_SIZE_Y
///   + UNIT_POS_Y` -> task index (`0..n_tasks`)
///
/// ### Shared memory layout
///
/// * `s_q_idx[32]` - query index per Y-slot
/// * `s_db_start[32]` - DB start per Y-slot
/// * `s_write_offset[32]` - write offset per Y-slot
/// * `s_db_count[32]` - DB count per Y-slot
/// * `s_query[32 * dim_scalars]` - query vectors in scalar form, where
///   `dim_scalars = dim_lines * N`
#[cube(launch_unchecked)]
pub fn compute_ivf_mega_euclidean_cached<F: Float, N: Size>(
    query_vectors: &Tensor<Vector<F, N>>,
    db_vectors: &Tensor<Vector<F, N>>,
    task_q_idx: &Tensor<u32>,
    task_db_start: &Tensor<u32>,
    task_write_offset: &Tensor<u32>,
    task_db_count: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
    n_tasks: u32,
    #[comptime] dim_lines: usize,
    #[comptime] size_y: u32,
) {
    let lanes = LINE_SIZE;
    let local_db_idx = ABSOLUTE_POS_X;
    let task_idx = (CUBE_POS_Z * CUBE_COUNT_Y + CUBE_POS_Y) * size_y + UNIT_POS_Y;
    let local_y = UNIT_POS_Y as usize;
    let local_x = UNIT_POS_X as usize;

    let dim_scalars = dim_lines * lanes;
    let wg_y = size_y as usize;

    let share_mem_size = size_y as usize;

    let mut s_q_idx = SharedMemory::<u32>::new(share_mem_size);
    let mut s_db_start = SharedMemory::<u32>::new(share_mem_size);
    let mut s_write_offset = SharedMemory::<u32>::new(share_mem_size);
    let mut s_db_count = SharedMemory::<u32>::new(share_mem_size);

    // WORKAROUND: cubecl 0.10 miscompiles expression-position `if/else`
    // with a runtime condition on wgpu/Metal and lavapipe — it always
    // returns the else branch. Writing to shared memory in both arms of
    // a statement-if lets the optimiser fuse the pair into per-assignment
    // expression-ifs, which then always take the else branch (all zeros).
    // With `s_db_count` stuck at 0 the whole workgroup terminates below
    // and the kernel silently becomes a no-op, leaving the candidate
    // tensors as uninitialised VRAM that the reducer treats as valid
    // distances/indices — hence the ~1e9 OOB reorg_idx values. Compute
    // values into locals first, then write shared memory once.
    let mut q_val = 0u32;
    let mut ds_val = 0u32;
    let mut wo_val = 0u32;
    let mut dc_val = 0u32;
    if task_idx < n_tasks {
        q_val = task_q_idx[task_idx as usize];
        ds_val = task_db_start[task_idx as usize];
        wo_val = task_write_offset[task_idx as usize];
        dc_val = task_db_count[task_idx as usize];
    }
    if local_x == 0usize {
        s_q_idx[local_y] = q_val;
        s_db_start[local_y] = ds_val;
        s_write_offset[local_y] = wo_val;
        s_db_count[local_y] = dc_val;
    }

    sync_cube();

    let mut s_query = SharedMemory::<F>::new(share_mem_size * dim_scalars);

    let thread_id = local_y * WORKGROUP_SIZE_X as usize + local_x;
    let total_threads = WORKGROUP_SIZE_X as usize * wg_y;
    let total_elems = wg_y * dim_scalars;

    let mut load_idx = thread_id;
    while load_idx < total_elems {
        let q_local = load_idx / dim_scalars;
        let elem = load_idx % dim_scalars;
        let q_global = s_q_idx[q_local];

        let line_idx = elem / lanes;
        let lane = elem % lanes;
        let line_val = query_vectors[q_global as usize * dim_lines + line_idx];
        s_query[load_idx] = line_val[lane];

        load_idx += total_threads;
    }

    sync_cube();

    // ── Phase 3: Bounds check and compute distance ──

    if task_idx >= n_tasks {
        terminate!();
    }

    let db_count = s_db_count[local_y];
    if local_db_idx >= db_count {
        terminate!();
    }

    let real_db_idx = s_db_start[local_y] + local_db_idx;
    let write_pos = s_write_offset[local_y] + local_db_idx;
    let q_shared_base = local_y * dim_scalars;
    let d_offset = real_db_idx as usize * dim_lines;

    let mut sum = F::new(0.0_f32);
    for i in 0..dim_lines {
        let d_line = db_vectors[d_offset + i];
        let s_off = q_shared_base + i * lanes;
        #[unroll]
        for lane in 0..lanes {
            let diff = s_query[s_off + lane] - d_line[lane];
            sum += diff * diff;
        }
    }

    let q_idx = s_q_idx[local_y];
    let out_offset = q_idx as usize * out_dists.stride(0) + write_pos as usize;
    out_dists[out_offset] = sum;
    out_indices[out_offset] = real_db_idx;
}

/// Cosine mega kernel with shared-memory query caching.
///
/// Same shared-memory caching strategy as
/// `compute_ivf_mega_euclidean_cached`, but computes
/// `1 - dot(q, d) / (||q|| * ||d||)`.
///
/// Additionally caches per-query L2 norms in a small shared memory
/// array `s_query_norms[32]` to avoid redundant global reads.
///
/// ### Params
///
/// * `query_vectors` - Query vectors `[n_queries, dim]` as `Vector<F, N>`.
///   Shape in element units.
/// * `db_vectors` - Full database vectors `[n_db, dim]` as `Vector<F, N>`.
///   Shape in element units.
/// * `query_norms` - Pre-computed L2 norms for queries `[n_queries]`.
///   Scalar tensor, one norm per query.
/// * `db_norms` - Pre-computed L2 norms for DB vectors `[n_db]`.
///   Scalar tensor, one norm per DB vector.
/// * `task_q_idx` - Query index for each task `[n_tasks]`. Sorted.
/// * `task_db_start` - Global DB start index for each task `[n_tasks]`.
/// * `task_write_offset` - Write offset into the candidate row for each
///   task `[n_tasks]`.
/// * `task_db_count` - Number of DB vectors per task `[n_tasks]`.
/// * `out_dists` - Output candidate distances `[n_queries, max_candidates]`.
/// * `out_indices` - Output candidate DB indices `[n_queries, max_candidates]`.
/// * `n_tasks` - Total number of tasks for bounds checking.
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime).
/// * `size_y` - Safe workgroup size Y for the given dimensionality
///
/// ### Grid mapping
///
/// Same as `compute_ivf_mega_euclidean_cached`.
///
/// ### Shared memory layout
///
/// Same as Euclidean variant, plus:
/// * `s_query_norms[32]` - L2 norm per Y-slot query
#[cube(launch_unchecked)]
pub fn compute_ivf_mega_cosine_cached<F: Float, N: Size>(
    query_vectors: &Tensor<Vector<F, N>>,
    db_vectors: &Tensor<Vector<F, N>>,
    query_norms: &Tensor<F>,
    db_norms: &Tensor<F>,
    task_q_idx: &Tensor<u32>,
    task_db_start: &Tensor<u32>,
    task_write_offset: &Tensor<u32>,
    task_db_count: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
    n_tasks: u32,
    #[comptime] dim_lines: usize,
    #[comptime] size_y: u32,
) {
    let lanes = LINE_SIZE;
    let local_db_idx = ABSOLUTE_POS_X;
    let task_idx = (CUBE_POS_Z * CUBE_COUNT_Y + CUBE_POS_Y) * size_y + UNIT_POS_Y;
    let local_y = UNIT_POS_Y as usize;
    let local_x = UNIT_POS_X as usize;

    let share_mem_size = size_y as usize;

    let dim_scalars = dim_lines * lanes;
    let wg_y = size_y as usize;

    let mut s_q_idx = SharedMemory::<u32>::new(share_mem_size);
    let mut s_db_start = SharedMemory::<u32>::new(share_mem_size);
    let mut s_write_offset = SharedMemory::<u32>::new(share_mem_size);
    let mut s_db_count = SharedMemory::<u32>::new(share_mem_size);
    let mut s_query_norms = SharedMemory::<F>::new(share_mem_size);

    // Same workaround as the Euclidean variant: avoid symmetric writes
    // in an if/else block that cubecl 0.10 can fuse into per-assignment
    // expression-ifs (which then always take the else branch on
    // wgpu/Metal + lavapipe).
    let mut q_val = 0u32;
    let mut ds_val = 0u32;
    let mut wo_val = 0u32;
    let mut dc_val = 0u32;
    let mut qn_val = F::new(1.0_f32);
    if task_idx < n_tasks {
        let q = task_q_idx[task_idx as usize];
        q_val = q;
        ds_val = task_db_start[task_idx as usize];
        wo_val = task_write_offset[task_idx as usize];
        dc_val = task_db_count[task_idx as usize];
        qn_val = query_norms[q as usize];
    }
    if local_x == 0usize {
        s_q_idx[local_y] = q_val;
        s_db_start[local_y] = ds_val;
        s_write_offset[local_y] = wo_val;
        s_db_count[local_y] = dc_val;
        s_query_norms[local_y] = qn_val;
    }

    sync_cube();

    let mut s_query = SharedMemory::<F>::new(share_mem_size * dim_scalars);

    let thread_id = local_y * WORKGROUP_SIZE_X as usize + local_x;
    let total_threads = WORKGROUP_SIZE_X as usize * wg_y;
    let total_elems = wg_y * dim_scalars;

    let mut load_idx = thread_id;
    while load_idx < total_elems {
        let q_local = load_idx / dim_scalars;
        let elem = load_idx % dim_scalars;
        let q_global = s_q_idx[q_local];

        let line_idx = elem / lanes;
        let lane = elem % lanes;
        let line_val = query_vectors[q_global as usize * dim_lines + line_idx];
        s_query[load_idx] = line_val[lane];

        load_idx += total_threads;
    }

    sync_cube();

    if task_idx >= n_tasks {
        terminate!();
    }

    let db_count = s_db_count[local_y];
    if local_db_idx >= db_count {
        terminate!();
    }

    let real_db_idx = s_db_start[local_y] + local_db_idx;
    let write_pos = s_write_offset[local_y] + local_db_idx;
    let q_shared_base = local_y * dim_scalars;
    let d_offset = real_db_idx as usize * dim_lines;

    let mut dot = F::new(0.0_f32);
    for i in 0..dim_lines {
        let d_line = db_vectors[d_offset + i];
        let s_off = q_shared_base + i * lanes;
        #[unroll]
        for lane in 0..lanes {
            dot += s_query[s_off + lane] * d_line[lane];
        }
    }

    let q_norm = s_query_norms[local_y];
    let d_norm = db_norms[real_db_idx as usize];

    let q_idx = s_q_idx[local_y];
    let out_offset = q_idx as usize * out_dists.stride(0) + write_pos as usize;
    out_dists[out_offset] = F::new(1.0_f32) - (dot / (q_norm * d_norm));
    out_indices[out_offset] = real_db_idx;
}

///////////
// Tests //
///////////

#[cfg(test)]
#[cfg(feature = "gpu-tests")]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    fn try_device() -> Option<WgpuDevice> {
        Some(WgpuDevice::default())
    }

    fn cpu_euclidean_dists(
        queries: &[f32],
        db: &[f32],
        nq: usize,
        ndb: usize,
        dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; nq * ndb];
        for q in 0..nq {
            for d in 0..ndb {
                let mut sum = 0.0f32;
                for j in 0..dim {
                    let diff = queries[q * dim + j] - db[d * dim + j];
                    sum += diff * diff;
                }
                out[q * ndb + d] = sum;
            }
        }
        out
    }

    fn cpu_cosine_dists(
        queries: &[f32],
        db: &[f32],
        q_norms: &[f32],
        d_norms: &[f32],
        nq: usize,
        ndb: usize,
        dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; nq * ndb];
        for q in 0..nq {
            for d in 0..ndb {
                let mut dot = 0.0f32;
                for j in 0..dim {
                    dot += queries[q * dim + j] * db[d * dim + j];
                }
                out[q * ndb + d] = 1.0 - dot / (q_norms[q] * d_norms[d]);
            }
        }
        out
    }

    fn cpu_topk(
        distances: &[f32],
        nq: usize,
        ndb: usize,
        k: usize,
    ) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let mut indices = Vec::with_capacity(nq);
        let mut dists = Vec::with_capacity(nq);
        for q in 0..nq {
            let row = &distances[q * ndb..(q + 1) * ndb];
            let mut pairs: Vec<(f32, usize)> = row
                .iter()
                .copied()
                .enumerate()
                .map(|(i, d)| (d, i))
                .collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            indices.push(pairs.iter().take(k).map(|p| p.1).collect());
            dists.push(pairs.iter().take(k).map(|p| p.0).collect());
        }
        (indices, dists)
    }

    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    // Pipeline: Euclidean at dim=8

    #[test]
    fn test_pipeline_euclidean_dim8() {
        let Some(device) = try_device() else { return };
        let nq = 10usize;
        let ndb = 50usize;
        let dim = 8usize;
        let k = 5usize;

        let queries: Vec<f32> = (0..nq * dim)
            .map(|i| ((i * 13 + 7) % 29) as f32 * 0.1)
            .collect();
        let db: Vec<f32> = (0..ndb * dim)
            .map(|i| ((i * 17 + 3) % 31) as f32 * 0.1)
            .collect();

        let qb = BatchData::new(&queries, &[], nq);
        let dbb = BatchData::new(&db, &[], ndb);

        let (_, gpu_dist) = query_batch_gpu::<f32, WgpuRuntime>(
            k,
            &qb,
            &dbb,
            dim,
            &Dist::SquaredEuclidean,
            device,
            false,
        )
        .unwrap();

        let cpu_d = cpu_euclidean_dists(&queries, &db, nq, ndb, dim);
        let (_, cpu_dist) = cpu_topk(&cpu_d, nq, ndb, k);

        for q in 0..nq {
            for i in 0..k {
                assert!(
                    (gpu_dist[q][i] - cpu_dist[q][i]).abs() < 1e-3,
                    "Query {} rank {}: gpu dist {} != cpu dist {}",
                    q,
                    i,
                    gpu_dist[q][i],
                    cpu_dist[q][i]
                );
            }
        }
    }

    // Pipeline: Euclidean at dim=32 (production dimension)

    #[test]
    fn test_pipeline_euclidean_dim32() {
        let Some(device) = try_device() else { return };
        let nq = 8usize;
        let ndb = 40usize;
        let dim = 32usize;
        let k = 5usize;

        let queries: Vec<f32> = (0..nq * dim)
            .map(|i| ((i * 13 + 7) % 29) as f32 * 0.1)
            .collect();
        let db: Vec<f32> = (0..ndb * dim)
            .map(|i| ((i * 17 + 3) % 31) as f32 * 0.1)
            .collect();

        let qb = BatchData::new(&queries, &[], nq);
        let dbb = BatchData::new(&db, &[], ndb);

        let (_, gpu_dist) = query_batch_gpu::<f32, WgpuRuntime>(
            k,
            &qb,
            &dbb,
            dim,
            &Dist::SquaredEuclidean,
            device,
            false,
        )
        .unwrap();

        let cpu_d = cpu_euclidean_dists(&queries, &db, nq, ndb, dim);
        let (_, cpu_dist) = cpu_topk(&cpu_d, nq, ndb, k);

        for q in 0..nq {
            for i in 0..k {
                assert!(
                    (gpu_dist[q][i] - cpu_dist[q][i]).abs() < 1e-2,
                    "dim=32 query {} rank {}: gpu dist {} != cpu dist {}",
                    q,
                    i,
                    gpu_dist[q][i],
                    cpu_dist[q][i]
                );
            }
        }
    }

    // Pipeline: Cosine at dim=32

    #[test]
    fn test_pipeline_cosine_dim32() {
        let Some(device) = try_device() else { return };
        let nq = 4usize;
        let ndb = 20usize;
        let dim = 32usize;
        let k = 3usize;

        let queries: Vec<f32> = (0..nq * dim)
            .map(|i| ((i * 7 + 1) % 11) as f32 + 0.5)
            .collect();
        let db: Vec<f32> = (0..ndb * dim)
            .map(|i| ((i * 13 + 3) % 17) as f32 + 0.5)
            .collect();

        let q_norms: Vec<f32> = (0..nq)
            .map(|q| l2_norm(&queries[q * dim..(q + 1) * dim]))
            .collect();
        let d_norms: Vec<f32> = (0..ndb)
            .map(|d| l2_norm(&db[d * dim..(d + 1) * dim]))
            .collect();

        let qb = BatchData::new(&queries, &q_norms, nq);
        let dbb = BatchData::new(&db, &d_norms, ndb);

        let (_, gpu_dist) =
            query_batch_gpu::<f32, WgpuRuntime>(k, &qb, &dbb, dim, &Dist::Cosine, device, false)
                .unwrap();

        let cpu_d = cpu_cosine_dists(&queries, &db, &q_norms, &d_norms, nq, ndb, dim);
        let (_, cpu_dist) = cpu_topk(&cpu_d, nq, ndb, k);

        for q in 0..nq {
            for i in 0..k {
                assert!(
                    (gpu_dist[q][i] - cpu_dist[q][i]).abs() < 1e-3,
                    "Cosine query {} rank {}: gpu dist {} != cpu dist {}",
                    q,
                    i,
                    gpu_dist[q][i],
                    cpu_dist[q][i]
                );
            }
        }
    }

    // Self-query: each vector should find itself as nearest

    #[test]
    fn test_self_query_finds_self() {
        let Some(device) = try_device() else { return };
        let n = 64usize;
        let dim = 32usize;

        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let batch = BatchData::new(&data, &[], n);

        let (indices, distances) = query_batch_gpu::<f32, WgpuRuntime>(
            3,
            &batch,
            &batch,
            dim,
            &Dist::SquaredEuclidean,
            device,
            false,
        )
        .unwrap();

        for q in 0..n {
            assert_eq!(
                indices[q][0], q,
                "Query {} nearest should be itself, got {}",
                q, indices[q][0]
            );
            assert!(
                distances[q][0] < 1e-4,
                "Self-distance for query {}: got {}",
                q,
                distances[q][0]
            );
        }
    }

    // Output must be sorted by distance with no duplicate indices

    #[test]
    fn test_output_is_sorted() {
        let Some(device) = try_device() else { return };
        let nq = 16usize;
        let ndb = 64usize;
        let dim = 32usize;
        let k = 5usize;

        let queries: Vec<f32> = (0..nq * dim).map(|i| ((i * 7 + 3) % 13) as f32).collect();
        let db: Vec<f32> = (0..ndb * dim).map(|i| ((i * 11 + 5) % 17) as f32).collect();

        let qb = BatchData::new(&queries, &[], nq);
        let dbb = BatchData::new(&db, &[], ndb);

        let (indices, distances) = query_batch_gpu::<f32, WgpuRuntime>(
            k,
            &qb,
            &dbb,
            dim,
            &Dist::SquaredEuclidean,
            device,
            false,
        )
        .unwrap();

        for q in 0..nq {
            for i in 1..k {
                assert!(
                    distances[q][i] >= distances[q][i - 1],
                    "Query {}: not sorted at {}: {} < {}",
                    q,
                    i,
                    distances[q][i],
                    distances[q][i - 1]
                );
            }
            let unique: std::collections::HashSet<usize> = indices[q].iter().copied().collect();
            assert_eq!(unique.len(), k, "Query {}: duplicate indices", q);
        }
    }

    // Edge case: k=1

    #[test]
    fn test_k_equals_one() {
        let Some(device) = try_device() else { return };
        let data: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0,
        ];
        let query: Vec<f32> = vec![0.9, 0.0, 0.0, 0.0];

        let qb = BatchData::new(&query, &[], 1);
        let dbb = BatchData::new(&data, &[], 4);

        let (idx, dist) = query_batch_gpu::<f32, WgpuRuntime>(
            1,
            &qb,
            &dbb,
            4,
            &Dist::SquaredEuclidean,
            device,
            false,
        )
        .unwrap();

        assert_eq!(idx[0][0], 1);
        assert!((dist[0][0] - 0.01).abs() < 1e-3);
    }

    // Planted nearest neighbour at a known index

    #[test]
    fn test_planted_nearest() {
        let Some(device) = try_device() else { return };
        let dim = 32usize;
        let k = 3usize;
        let nq = 2usize;
        let ndb = 200usize;

        let mut db: Vec<f32> = (0..ndb * dim).map(|i| ((i * 17 + 3) % 31) as f32).collect();

        let target = vec![100.0f32; dim];
        db[73 * dim..74 * dim].copy_from_slice(&target);

        let mut queries = target.clone();
        queries[0] += 0.001;
        queries.extend_from_slice(&vec![0.0f32; dim]);

        let qb = BatchData::new(&queries, &[], nq);
        let dbb = BatchData::new(&db, &[], ndb);

        let (idx, dist) = query_batch_gpu::<f32, WgpuRuntime>(
            k,
            &qb,
            &dbb,
            dim,
            &Dist::SquaredEuclidean,
            device,
            false,
        )
        .unwrap();

        assert_eq!(idx[0][0], 73, "Should find planted nearest at index 73");
        assert!(dist[0][0] < 0.01);

        let cpu_d = cpu_euclidean_dists(&queries, &db, nq, ndb, dim);
        let (cpu_idx, _) = cpu_topk(&cpu_d, nq, ndb, k);
        for q in 0..nq {
            assert_eq!(idx[q], cpu_idx[q], "Query {} mismatch vs CPU", q);
        }
    }

    // Edge case: single query, single DB vector

    #[test]
    fn test_single_query_single_db() {
        let Some(device) = try_device() else { return };
        let dim = 4usize;
        let query = vec![1.0f32, 2.0, 3.0, 4.0];
        let db = vec![5.0f32, 6.0, 7.0, 8.0];

        let qb = BatchData::new(&query, &[], 1);
        let dbb = BatchData::new(&db, &[], 1);

        let (idx, dist) = query_batch_gpu::<f32, WgpuRuntime>(
            1,
            &qb,
            &dbb,
            dim,
            &Dist::SquaredEuclidean,
            device,
            false,
        )
        .unwrap();

        assert_eq!(idx[0][0], 0);
        assert!((dist[0][0] - 64.0).abs() < 1e-3);
    }

    // ── Diagnostic tests for reduce_ivf_topk_coalesced ──

    /// Helper: run the OLD serial reduce_ivf_topk on known data and return
    /// (dists, indices). Provides the ground truth to compare against.
    fn run_serial_reduce(
        candidate_dists: &[f32],
        candidate_indices: &[u32],
        candidates_per_query: &[u32],
        n_queries: usize,
        max_candidates: usize,
        k: usize,
        device: &WgpuDevice,
    ) -> (Vec<f32>, Vec<u32>) {
        let client = WgpuRuntime::client(device);
        let limits = GpuLimits::from_client(&client);

        let cd_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            candidate_dists,
            vec![n_queries, max_candidates],
            &client,
        )
        .unwrap();
        let ci_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
            candidate_indices,
            vec![n_queries, max_candidates],
            &client,
        )
        .unwrap();
        let cpq_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
            candidates_per_query,
            vec![n_queries],
            &client,
        )
        .unwrap();

        let topk_d = GpuTensor::<WgpuRuntime, f32>::empty(vec![n_queries, k], &client).unwrap();
        let topk_i = GpuTensor::<WgpuRuntime, u32>::empty(vec![n_queries, k], &client).unwrap();

        // init_topk
        let init_gx = (k as u32).div_ceil(WORKGROUP_SIZE_X);
        let (init_gy, init_gz) = grid_2d((n_queries as u32).div_ceil(4), &limits).unwrap();
        unsafe {
            init_topk::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(init_gx, init_gy, init_gz),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 4),
                topk_d.clone().into_tensor_arg(),
                topk_i.clone().into_tensor_arg(),
                4,
            );
        }

        // serial reduce
        let (rgx, rgy) = grid_2d((n_queries as u32).div_ceil(WORKGROUP_SIZE_X), &limits).unwrap();
        unsafe {
            reduce_ivf_topk::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(rgx, rgy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                cd_gpu.into_tensor_arg(),
                ci_gpu.into_tensor_arg(),
                cpq_gpu.into_tensor_arg(),
                topk_d.clone().into_tensor_arg(),
                topk_i.clone().into_tensor_arg(),
            );
        }

        (topk_d.read(&client).unwrap(), topk_i.read(&client).unwrap())
    }

    /// Helper: run the NEW coalesced reduce on the same data.
    fn run_coalesced_reduce(
        candidate_dists: &[f32],
        candidate_indices: &[u32],
        candidates_per_query: &[u32],
        n_queries: usize,
        max_candidates: usize,
        k: usize,
        device: &WgpuDevice,
    ) -> (Vec<f32>, Vec<u32>) {
        let client = WgpuRuntime::client(device);
        let limits = GpuLimits::from_client(&client);

        let cd_gpu = GpuTensor::<WgpuRuntime, f32>::from_slice(
            candidate_dists,
            vec![n_queries, max_candidates],
            &client,
        )
        .unwrap();
        let ci_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
            candidate_indices,
            vec![n_queries, max_candidates],
            &client,
        )
        .unwrap();
        let cpq_gpu = GpuTensor::<WgpuRuntime, u32>::from_slice(
            candidates_per_query,
            vec![n_queries],
            &client,
        )
        .unwrap();

        let topk_d = GpuTensor::<WgpuRuntime, f32>::empty(vec![n_queries, k], &client).unwrap();
        let topk_i = GpuTensor::<WgpuRuntime, u32>::empty(vec![n_queries, k], &client).unwrap();

        let (rgx, rgy) = grid_2d(n_queries as u32, &limits).unwrap();
        let merge = plan_topk_merge(k, size_of::<f32>(), &limits).unwrap();
        unsafe {
            reduce_ivf_topk_coalesced::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(rgx, rgy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                cd_gpu.into_tensor_arg(),
                ci_gpu.into_tensor_arg(),
                cpq_gpu.into_tensor_arg(),
                topk_d.clone().into_tensor_arg(),
                topk_i.clone().into_tensor_arg(),
                k as u32,
                k,
                merge.group,
                merge.single_round,
                merge.slots,
            );
        }

        (topk_d.read(&client).unwrap(), topk_i.read(&client).unwrap())
    }

    // Test 1: Trivial case. 1 query, 5 candidates, k=3.
    // Candidates are in reverse order so we know sorting matters.
    #[test]
    fn test_coalesced_reduce_trivial() {
        let Some(device) = try_device() else { return };

        let k = 3usize;
        let n_queries = 1usize;
        let max_candidates = 5usize;

        // Distances: 50, 40, 30, 20, 10 -- indices: 100, 101, 102, 103, 104
        // Expected top-3: dist=[10, 20, 30], idx=[104, 103, 102]
        let candidate_dists = vec![50.0f32, 40.0, 30.0, 20.0, 10.0];
        let candidate_indices = vec![100u32, 101, 102, 103, 104];
        let candidates_per_query = vec![5u32];

        let (dists, indices) = run_coalesced_reduce(
            &candidate_dists,
            &candidate_indices,
            &candidates_per_query,
            n_queries,
            max_candidates,
            k,
            &device,
        );

        println!("Trivial: dists={:?}, indices={:?}", dists, indices);
        assert_eq!(indices, vec![104, 103, 102], "Wrong indices");
        assert_eq!(dists, vec![10.0, 20.0, 30.0], "Wrong distances");
    }

    // Test 2: Verify thread 0 alone finds the right answer when
    // candidates <= 32 (single thread's stride covers everything).
    #[test]
    fn test_coalesced_reduce_single_stride() {
        let Some(device) = try_device() else { return };

        let k = 3usize;
        let n_queries = 1usize;
        let max_candidates = 20usize;

        // 20 candidates, distances = 20, 19, ..., 1
        let candidate_dists: Vec<f32> = (0..20).rev().map(|i| (i + 1) as f32).collect();
        let candidate_indices: Vec<u32> = (0..20).map(|i| i as u32 + 200).collect();
        let candidates_per_query = vec![20u32];

        let (dists, indices) = run_coalesced_reduce(
            &candidate_dists,
            &candidate_indices,
            &candidates_per_query,
            n_queries,
            max_candidates,
            k,
            &device,
        );

        println!("Single stride: dists={:?}, indices={:?}", dists, indices);
        // Smallest dists are 1.0, 2.0, 3.0 at positions 19, 18, 17
        assert_eq!(dists, vec![1.0, 2.0, 3.0]);
        assert_eq!(indices, vec![219, 218, 217]);
    }

    // Test 3: Multi-thread merge required. 200 candidates forces multiple
    // threads to each hold partial top-k that must be merged correctly.
    #[test]
    fn test_coalesced_reduce_needs_merge() {
        let Some(device) = try_device() else { return };

        let k = 5usize;
        let n_queries = 1usize;
        let max_candidates = 200usize;

        // Plant known winners at positions that land on DIFFERENT threads.
        // Thread assignment = position % 32.
        // Best 5: pos 3 (thread 3), pos 35 (thread 3), pos 64 (thread 0),
        //         pos 100 (thread 4), pos 129 (thread 1)
        let mut candidate_dists = vec![999.0f32; max_candidates];
        let candidate_indices: Vec<u32> = (0..max_candidates as u32).collect();

        candidate_dists[3] = 1.0; // thread 3
        candidate_dists[35] = 2.0; // thread 3
        candidate_dists[64] = 3.0; // thread 0
        candidate_dists[100] = 4.0; // thread 4
        candidate_dists[129] = 5.0; // thread 1

        let candidates_per_query = vec![max_candidates as u32];

        let (dists, indices) = run_coalesced_reduce(
            &candidate_dists,
            &candidate_indices,
            &candidates_per_query,
            n_queries,
            max_candidates,
            k,
            &device,
        );

        println!("Merge test: dists={:?}, indices={:?}", dists, indices);
        assert_eq!(dists, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(indices, vec![3, 35, 64, 100, 129]);
    }

    // Large k forces the chunked merge arm, where the 32 per-lane lists no
    // longer fit in shared memory at once. Cross-checked against the serial
    // reducer, which has no shared-memory merge at all. Without this the
    // chunked path has no unit coverage: every other reducer test uses a small
    // k and so only ever exercises the single-round arm.
    #[test]
    fn test_coalesced_reduce_chunked_matches_serial() {
        let Some(device) = try_device() else { return };

        // k = 200 needs 32 * 200 * 8 = 51 200 bytes for a single-round merge,
        // over the 32 KiB budget, so this must take the chunked path.
        let k = 200usize;
        let n_queries = 3usize;
        let max_candidates = 1024usize;

        let mut candidate_dists = vec![0.0f32; n_queries * max_candidates];
        let mut candidate_indices = vec![0u32; n_queries * max_candidates];
        for q in 0..n_queries {
            for c in 0..max_candidates {
                let i = q * max_candidates + c;
                // Deterministic, non-monotonic, with distinct values so the
                // ordering is unambiguous.
                candidate_dists[i] = (((c * 7919 + q * 104_729) % 10_007) as f32) * 0.01;
                candidate_indices[i] = c as u32;
            }
        }
        let candidates_per_query = vec![max_candidates as u32; n_queries];

        let (coal_d, coal_i) = run_coalesced_reduce(
            &candidate_dists,
            &candidate_indices,
            &candidates_per_query,
            n_queries,
            max_candidates,
            k,
            &device,
        );
        let (ser_d, ser_i) = run_serial_reduce(
            &candidate_dists,
            &candidate_indices,
            &candidates_per_query,
            n_queries,
            max_candidates,
            k,
            &device,
        );

        // A no-op dispatch leaves the output untouched, so guard against the
        // silent-failure mode this kernel was fixed for.
        assert!(
            coal_d.iter().any(|d| *d != 0.0),
            "chunked reduce produced all zeros: the dispatch almost certainly did no work"
        );

        for q in 0..n_queries {
            for i in 0..k {
                let idx = q * k + i;
                assert!(
                    (coal_d[idx] - ser_d[idx]).abs() <= 1e-5,
                    "query {q} slot {i}: chunked {} vs serial {}",
                    coal_d[idx],
                    ser_d[idx]
                );
                assert_eq!(
                    coal_i[idx], ser_i[idx],
                    "query {q} slot {i}: index {} vs {}",
                    coal_i[idx], ser_i[idx]
                );
            }
            // Each row must come back sorted ascending.
            for i in 1..k {
                assert!(
                    coal_d[q * k + i - 1] <= coal_d[q * k + i],
                    "query {q} not sorted at slot {i}"
                );
            }
        }
    }

    // Test 4: A/B comparison with serial kernel on random-ish data.
    // If this fails but tests 1-3 pass, the bug is scale-dependent.
    #[test]
    fn test_coalesced_vs_serial_random() {
        let Some(device) = try_device() else { return };

        let k = 10usize;
        let n_queries = 8usize;
        let max_candidates = 500usize;

        // Deterministic pseudo-random distances
        let mut candidate_dists = vec![0.0f32; n_queries * max_candidates];
        let mut candidate_indices = vec![0u32; n_queries * max_candidates];
        for q in 0..n_queries {
            for c in 0..max_candidates {
                let idx = q * max_candidates + c;
                candidate_dists[idx] = ((idx * 17 + 31) % 9973) as f32 * 0.1;
                candidate_indices[idx] = (q * 10000 + c) as u32;
            }
        }
        let candidates_per_query = vec![max_candidates as u32; n_queries];

        let (serial_d, serial_i) = run_serial_reduce(
            &candidate_dists,
            &candidate_indices,
            &candidates_per_query,
            n_queries,
            max_candidates,
            k,
            &device,
        );

        let (coal_d, coal_i) = run_coalesced_reduce(
            &candidate_dists,
            &candidate_indices,
            &candidates_per_query,
            n_queries,
            max_candidates,
            k,
            &device,
        );

        for q in 0..n_queries {
            let s = q * k;
            let e = s + k;
            let sd = &serial_d[s..e];
            let cd = &coal_d[s..e];
            let si = &serial_i[s..e];
            let ci = &coal_i[s..e];

            for i in 0..k {
                assert!(
                    (sd[i] - cd[i]).abs() < 1e-4,
                    "Query {} rank {}: serial dist {} != coalesced dist {}",
                    q,
                    i,
                    sd[i],
                    cd[i]
                );
                assert_eq!(
                    si[i], ci[i],
                    "Query {} rank {}: serial idx {} != coalesced idx {}",
                    q, i, si[i], ci[i]
                );
            }
        }
    }

    // Test 5: Multiple queries to verify q_idx indexing is correct
    // (catches stride/offset bugs in the output write)
    #[test]
    fn test_coalesced_reduce_multi_query() {
        let Some(device) = try_device() else { return };

        let k = 3usize;
        let n_queries = 4usize;
        let max_candidates = 10usize;

        let mut candidate_dists = vec![f32::MAX; n_queries * max_candidates];
        let mut candidate_indices = vec![0u32; n_queries * max_candidates];

        // Each query has a unique planted winner
        for q in 0..n_queries {
            let base = q * max_candidates;
            for c in 0..max_candidates {
                candidate_dists[base + c] = 100.0 + c as f32;
                candidate_indices[base + c] = (q * 1000 + c) as u32;
            }
            // Plant winner at different position per query
            candidate_dists[base + q + 1] = 0.5 + q as f32 * 0.1;
        }

        let candidates_per_query = vec![max_candidates as u32; n_queries];

        let (dists, indices) = run_coalesced_reduce(
            &candidate_dists,
            &candidate_indices,
            &candidates_per_query,
            n_queries,
            max_candidates,
            k,
            &device,
        );

        for q in 0..n_queries {
            let s = q * k;
            let best_dist = dists[s];
            let best_idx = indices[s];
            let expected_dist = 0.5 + q as f32 * 0.1;
            let expected_idx = (q * 1000 + q + 1) as u32;

            assert!(
                (best_dist - expected_dist).abs() < 1e-4,
                "Query {}: best dist {} != expected {}",
                q,
                best_dist,
                expected_dist
            );
            assert_eq!(
                best_idx, expected_idx,
                "Query {}: best idx {} != expected {}",
                q, best_idx, expected_idx
            );
        }
    }

    // Test 6: Candidates fewer than k -- output should contain only
    // valid entries, rest should remain f32::MAX / 0
    #[test]
    fn test_coalesced_reduce_fewer_than_k() {
        let Some(device) = try_device() else { return };

        let k = 5usize;
        let n_queries = 1usize;
        let max_candidates = 10usize;

        let mut candidate_dists = vec![f32::MAX; max_candidates];
        let mut candidate_indices = vec![0u32; max_candidates];

        // Only 2 valid candidates
        candidate_dists[0] = 5.0;
        candidate_dists[1] = 3.0;
        candidate_indices[0] = 42;
        candidate_indices[1] = 99;

        let candidates_per_query = vec![2u32];

        let (dists, indices) = run_coalesced_reduce(
            &candidate_dists,
            &candidate_indices,
            &candidates_per_query,
            n_queries,
            max_candidates,
            k,
            &device,
        );

        println!("Fewer than k: dists={:?}, indices={:?}", dists, indices);
        assert!((dists[0] - 3.0).abs() < 1e-4, "First should be 3.0");
        assert!((dists[1] - 5.0).abs() < 1e-4, "Second should be 5.0");
        assert_eq!(indices[0], 99);
        assert_eq!(indices[1], 42);
        // Remaining should be sentinel
        assert!(dists[2] >= f32::MAX / 2.0, "Slot 2 should be sentinel");
    }

    /// Helper: build synthetic IVF task data and run a mega kernel,
    /// returning the output (dists, indices) flat buffers.
    ///
    /// * `queries` - flat query data `[n_queries * dim]`
    /// * `db` - flat DB data `[n_db * dim]`
    /// * `tasks` - Vec of `(q_idx, db_start, write_offset, db_count)`
    /// * `n_queries`, `n_db`, `dim`, `max_candidates` - dimensions
    /// * `device` - GPU device
    /// * `use_cached` - if true, runs the new cached kernel; otherwise
    ///   the old one
    #[allow(clippy::too_many_arguments)]
    fn run_mega_euclidean(
        queries: &[f32],
        db: &[f32],
        tasks: &[(u32, u32, u32, u32)],
        n_queries: usize,
        n_db: usize,
        dim: usize,
        max_candidates: usize,
        device: &WgpuDevice,
        use_cached: bool,
    ) -> (Vec<f32>, Vec<u32>) {
        let client = WgpuRuntime::client(device);
        let limits = GpuLimits::from_client(&client);
        let vec_size = LINE_SIZE;
        let dim_lines = dim / vec_size;
        let n_tasks = tasks.len();
        let q_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(queries, vec![n_queries, dim], &client)
                .unwrap();
        let db_gpu =
            GpuTensor::<WgpuRuntime, f32>::from_slice(db, vec![n_db, dim], &client).unwrap();
        let task_q: Vec<u32> = tasks.iter().map(|t| t.0).collect();
        let task_db_s: Vec<u32> = tasks.iter().map(|t| t.1).collect();
        let task_wo: Vec<u32> = tasks.iter().map(|t| t.2).collect();
        let task_dc: Vec<u32> = tasks.iter().map(|t| t.3).collect();
        let tq_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&task_q, vec![n_tasks], &client).unwrap();
        let tds_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&task_db_s, vec![n_tasks], &client).unwrap();
        let two_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&task_wo, vec![n_tasks], &client).unwrap();
        let tdc_gpu =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&task_dc, vec![n_tasks], &client).unwrap();
        let out_d =
            GpuTensor::<WgpuRuntime, f32>::empty(vec![n_queries, max_candidates], &client).unwrap();
        let out_i =
            GpuTensor::<WgpuRuntime, u32>::empty(vec![n_queries, max_candidates], &client).unwrap();
        let max_db_count = tasks.iter().map(|t| t.3).max().unwrap_or(0);
        let gx = max_db_count.div_ceil(WORKGROUP_SIZE_X).max(1);
        let (gy, gz) = grid_2d((n_tasks as u32).div_ceil(4), &limits).unwrap();
        if use_cached {
            unsafe {
                compute_ivf_mega_euclidean_cached::launch_unchecked::<f32, WgpuRuntime>(
                    &client,
                    CubeCount::Static(gx, gy, gz),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 4),
                    vec_size,
                    q_gpu.into_tensor_arg(),
                    db_gpu.into_tensor_arg(),
                    tq_gpu.into_tensor_arg(),
                    tds_gpu.into_tensor_arg(),
                    two_gpu.into_tensor_arg(),
                    tdc_gpu.into_tensor_arg(),
                    out_d.clone().into_tensor_arg(),
                    out_i.clone().into_tensor_arg(),
                    n_tasks as u32,
                    dim_lines,
                    4,
                );
            }
        } else {
            unsafe {
                compute_ivf_mega_euclidean::launch_unchecked::<f32, WgpuRuntime>(
                    &client,
                    CubeCount::Static(gx, gy, gz),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 4),
                    vec_size,
                    q_gpu.into_tensor_arg(),
                    db_gpu.into_tensor_arg(),
                    tq_gpu.into_tensor_arg(),
                    tds_gpu.into_tensor_arg(),
                    two_gpu.into_tensor_arg(),
                    tdc_gpu.into_tensor_arg(),
                    out_d.clone().into_tensor_arg(),
                    out_i.clone().into_tensor_arg(),
                    4,
                );
            }
        }
        (out_d.read(&client).unwrap(), out_i.read(&client).unwrap())
    }

    /// CPU reference: squared Euclidean distance between query q and DB
    /// vector d.
    fn cpu_sq_euclidean(queries: &[f32], db: &[f32], q: usize, d: usize, dim: usize) -> f32 {
        let mut sum = 0.0f32;
        for j in 0..dim {
            let diff = queries[q * dim + j] - db[d * dim + j];
            sum += diff * diff;
        }
        sum
    }

    // Test 1: Known-answer. 2 queries, 3 tasks, dim=4.
    // Verify exact distances against CPU computation.
    #[test]
    fn test_mega_cached_known_answer() {
        let Some(device) = try_device() else { return };

        let dim = 4usize;
        let n_queries = 2usize;
        let n_db = 5usize;
        let max_candidates = 5usize;

        // q0 = [1, 0, 0, 0], q1 = [0, 1, 0, 0]
        let queries = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        // db0=[1,1,0,0] db1=[2,0,0,0] db2=[0,0,1,1] db3=[0,2,0,0] db4=[0,0,0,3]
        let db = vec![
            1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 3.0,
        ];

        // Tasks (sorted by q_idx):
        //   q0 probes cluster [0..3) and [3..5)
        //   q1 probes cluster [0..3)
        let tasks = vec![
            (0u32, 0u32, 0u32, 3u32), // q0, db[0..3], write at 0
            (0u32, 3u32, 3u32, 2u32), // q0, db[3..5], write at 3
            (1u32, 0u32, 0u32, 3u32), // q1, db[0..3], write at 0
        ];

        let (dists, indices) = run_mega_euclidean(
            &queries,
            &db,
            &tasks,
            n_queries,
            n_db,
            dim,
            max_candidates,
            &device,
            true,
        );

        // Verify q0's candidates
        for c in 0..3 {
            let expected = cpu_sq_euclidean(&queries, &db, 0, c, dim);
            let got = dists[c];
            assert!(
                (got - expected).abs() < 1e-4,
                "q0 vs db{}: got {} expected {}",
                c,
                got,
                expected,
            );
            assert_eq!(indices[c], c as u32);
        }
        for c in 0..2 {
            let db_idx = 3 + c;
            let expected = cpu_sq_euclidean(&queries, &db, 0, db_idx, dim);
            let got = dists[3 + c];
            assert!(
                (got - expected).abs() < 1e-4,
                "q0 vs db{}: got {} expected {}",
                db_idx,
                got,
                expected,
            );
            assert_eq!(indices[3 + c], db_idx as u32);
        }
        // Verify q1's candidates
        for c in 0..3 {
            let expected = cpu_sq_euclidean(&queries, &db, 1, c, dim);
            let got = dists[max_candidates + c];
            assert!(
                (got - expected).abs() < 1e-4,
                "q1 vs db{}: got {} expected {}",
                c,
                got,
                expected,
            );
        }

        println!("Known-answer: PASSED");
    }

    // Test 1: Known-answer. 2 queries, 3 tasks, dim=4.
    // Verify exact distances against CPU computation.
    #[test]
    fn test_mega_cached_known_answer_2() {
        let Some(device) = try_device() else { return };

        let dim = 4usize;
        let n_queries = 2usize;
        let n_db = 5usize;
        let max_candidates = 5usize;

        // q0 = [1, 0, 0, 0], q1 = [0, 1, 0, 0]
        let queries = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        // db0=[1,1,0,0] db1=[2,0,0,0] db2=[0,0,1,1] db3=[0,2,0,0] db4=[0,0,0,3]
        let db = vec![
            1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 3.0,
        ];

        // Tasks (sorted by q_idx):
        //   q0 probes cluster [0..3) and [3..5)
        //   q1 probes cluster [0..3)
        let tasks = vec![
            (0u32, 0u32, 0u32, 3u32), // q0, db[0..3], write at 0
            (0u32, 3u32, 3u32, 2u32), // q0, db[3..5], write at 3
            (1u32, 0u32, 0u32, 3u32), // q1, db[0..3], write at 0
        ];

        let (dists, indices) = run_mega_euclidean(
            &queries,
            &db,
            &tasks,
            n_queries,
            n_db,
            dim,
            max_candidates,
            &device,
            true,
        );

        // Verify q0's candidates
        for c in 0..3 {
            let expected = cpu_sq_euclidean(&queries, &db, 0, c, dim);
            let got = dists[c];
            assert!(
                (got - expected).abs() < 1e-4,
                "q0 vs db{}: got {} expected {}",
                c,
                got,
                expected,
            );
            assert_eq!(indices[c], c as u32);
        }
        for c in 0..2 {
            let db_idx = 3 + c;
            let expected = cpu_sq_euclidean(&queries, &db, 0, db_idx, dim);
            let got = dists[3 + c];
            assert!(
                (got - expected).abs() < 1e-4,
                "q0 vs db{}: got {} expected {}",
                db_idx,
                got,
                expected,
            );
            assert_eq!(indices[3 + c], db_idx as u32);
        }
        // Verify q1's candidates
        for c in 0..3 {
            let expected = cpu_sq_euclidean(&queries, &db, 1, c, dim);
            let got = dists[max_candidates + c];
            assert!(
                (got - expected).abs() < 1e-4,
                "q1 vs db{}: got {} expected {}",
                c,
                got,
                expected,
            );
        }

        println!("Known-answer: PASSED");
    }
}
