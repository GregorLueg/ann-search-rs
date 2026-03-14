//! GPU-accelerated distance calculations and top-k selection for GPU-based
//! indices. Contains kernels for both the exhaustive search pipeline and the
//! IVF fire-and-forget pipeline.

#![allow(missing_docs)]

use cubecl::prelude::*;
use std::iter::Sum;

use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::utils::dist::Dist;

/////////////////////////////////////
// Exhaustive distance computation //
/////////////////////////////////////

/// Tiled squared Euclidean distance kernel with shared-memory query caching
///
/// All threads in a workgroup cooperatively load the query tile into scalar
/// shared memory, eliminating redundant global reads across threads sharing
/// the same query row. DB vectors are read directly from global memory via
/// the `db_start` offset into a pre-uploaded full DB tensor.
///
/// Shared memory usage: WORKGROUP_SIZE_Y * dim_lines * 4 scalars.
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> DB vector index within chunk
/// * `ABSOLUTE_POS_Y` -> query vector index
#[cube(launch_unchecked)]
pub fn euclidean_tiled<F: Float>(
    query_vectors: &Tensor<Line<F>>,
    db_vectors: &Tensor<Line<F>>,
    distances: &mut Tensor<F>,
    db_start: u32,
    n_db_chunk: u32,
    n_queries: u32,
    dist_stride: u32,
    #[comptime] dim_lines: usize,
) {
    let db_idx = ABSOLUTE_POS_X as usize;
    let query_idx = ABSOLUTE_POS_Y as usize;
    let local_y = UNIT_POS_Y as usize;
    let local_x = UNIT_POS_X as usize;

    let dim_scalars = dim_lines * 4usize;
    let wg_y = WORKGROUP_SIZE_Y as usize;

    // Scalar shared memory only (Line<F> shared mem silently broadcasts lane 0)
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
            let line_idx = elem / 4usize;
            let lane = elem % 4usize;
            // Manual offset: tensor.stride(0) reports element units, not Line units
            let line_val = query_vectors[q_global * dim_lines + line_idx];
            s_query[load_idx] = line_val[lane];
        } else {
            s_query[load_idx] = F::new(0.0);
        }
        load_idx += total_threads;
    }

    sync_cube();

    if query_idx >= n_queries as usize || db_idx >= n_db_chunk as usize {
        terminate!();
    }

    let global_db_idx = db_start as usize + db_idx;
    let q_shared_base = local_y * dim_scalars;

    let mut sum = F::new(0.0);
    for i in 0..dim_lines {
        let d_line = db_vectors[global_db_idx * dim_lines + i];
        let s_off = q_shared_base + i * 4usize;

        let diff0 = s_query[s_off] - d_line[0];
        let diff1 = s_query[s_off + 1usize] - d_line[1];
        let diff2 = s_query[s_off + 2usize] - d_line[2];
        let diff3 = s_query[s_off + 3usize] - d_line[3];

        sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
    }

    distances[query_idx * dist_stride as usize + db_idx] = sum;
}

/// Tiled cosine distance kernel with shared-memory query caching
///
/// Same tiling strategy as `euclidean_tiled` but computes
/// `1 - dot(q, d) / (||q|| * ||d||)`.
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> DB vector index within chunk
/// * `ABSOLUTE_POS_Y` -> query vector index
#[cube(launch_unchecked)]
pub fn cosine_tiled<F: Float>(
    query_vectors: &Tensor<Line<F>>,
    db_vectors: &Tensor<Line<F>>,
    query_norms: &Tensor<F>,
    db_norms: &Tensor<F>,
    distances: &mut Tensor<F>,
    db_start: u32,
    n_db_chunk: u32,
    n_queries: u32,
    dist_stride: u32,
    #[comptime] dim_lines: usize,
) {
    let db_idx = ABSOLUTE_POS_X as usize;
    let query_idx = ABSOLUTE_POS_Y as usize;
    let local_y = UNIT_POS_Y as usize;
    let local_x = UNIT_POS_X as usize;

    let dim_scalars = dim_lines * 4usize;
    let wg_y = WORKGROUP_SIZE_Y as usize;

    let mut s_query = SharedMemory::<F>::new(wg_y * dim_scalars);

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
            let line_idx = elem / 4usize;
            let lane = elem % 4usize;
            let line_val = query_vectors[q_global * dim_lines + line_idx];
            s_query[load_idx] = line_val[lane];
        } else {
            s_query[load_idx] = F::new(0.0);
        }
        load_idx += total_threads;
    }

    sync_cube();

    if query_idx >= n_queries as usize || db_idx >= n_db_chunk as usize {
        terminate!();
    }

    let global_db_idx = db_start as usize + db_idx;
    let q_shared_base = local_y * dim_scalars;

    let mut dot = F::new(0.0);
    for i in 0..dim_lines {
        let d_line = db_vectors[global_db_idx * dim_lines + i];
        let s_off = q_shared_base + i * 4usize;

        dot += s_query[s_off] * d_line[0]
            + s_query[s_off + 1usize] * d_line[1]
            + s_query[s_off + 2usize] * d_line[2]
            + s_query[s_off + 3usize] * d_line[3];
    }

    let q_norm = query_norms[query_idx];
    let d_norm = db_norms[global_db_idx];

    distances[query_idx * dist_stride as usize + db_idx] = F::new(1.0) - (dot / (q_norm * d_norm));
}

/////////////////////
// Top-k selection //
/////////////////////

/// Extract top-k smallest distances per query via insertion sort
///
/// One thread per query, serial scan of the distance row. Writes directly
/// into the running top-k buffer, so no separate merge step is needed.
/// The buffer must be pre-initialised with `init_topk`.
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
) {
    let query_idx = ABSOLUTE_POS_X as usize;

    if query_idx >= distances.shape(0) {
        terminate!();
    }

    let k = out_dists.shape(1);
    let dist_offset = query_idx * distances.stride(0);
    let out_offset = query_idx * out_dists.stride(0);

    for i in 0..actual_chunk_size {
        let dist = distances[dist_offset + i as usize];

        if dist < out_dists[out_offset + k - 1] {
            // Find insertion point
            let mut insert_pos: usize = k - 1;
            for j in 0..k {
                if dist < out_dists[out_offset + j] && insert_pos == k - 1 {
                    insert_pos = j;
                }
            }

            // Shift right
            for j in 0..k - 1 {
                let src = k - 2 - j;
                let dst = k - 1 - j;
                if src >= insert_pos {
                    out_dists[out_offset + dst] = out_dists[out_offset + src];
                    out_indices[out_offset + dst] = out_indices[out_offset + src];
                }
            }

            out_dists[out_offset + insert_pos] = dist;
            out_indices[out_offset + insert_pos] = chunk_offset + i;
        }
    }
}

/// Initialise top-k buffers to sentinel values (`f32::MAX` / `0`)
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> k slot index
/// * `ABSOLUTE_POS_Y` -> query index
#[cube(launch_unchecked)]
pub fn init_topk<F: Float>(dists: &mut Tensor<F>, indices: &mut Tensor<u32>) {
    let query_idx = ABSOLUTE_POS_Y as usize;
    let k_idx = ABSOLUTE_POS_X as usize;
    let k = dists.shape(1);

    if query_idx >= dists.shape(0) || k_idx >= k {
        terminate!();
    }

    let offset = query_idx * dists.stride(0) + k_idx;
    dists[offset] = F::new(f32::MAX);
    indices[offset] = 0u32;
}

/////////////////////////////////
// Fire-and-Forget IVF kernels //
/////////////////////////////////

/// Compute squared Euclidean distances on GPU
///
/// Each GPU thread computes the distance between one query vector and one
/// database vector. Vectors are processed in LINE_SIZE chunks for vectorised
/// memory access.
///
/// ### Params
///
/// * `query_vectors` - Query vectors [n_queries, dim / LINE_SIZE] as Line<F>
/// * `db_chunk` - Database vectors [n_db, dim / LINE_SIZE] as Line<F>
/// * `distances` - Output matrix [n_queries, n_db] of squared distances
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` → database vector index
/// * `ABSOLUTE_POS_Y` → query vector index
#[cube(launch_unchecked)]
pub fn euclidean_distances_gpu_chunk<F: Float>(
    query_vectors: &Tensor<Line<F>>,
    db_chunk: &Tensor<Line<F>>,
    distances: &mut Tensor<F>,
) {
    let query_idx = ABSOLUTE_POS_Y as usize;
    let db_idx = ABSOLUTE_POS_X as usize;

    if query_idx < query_vectors.shape(0) && db_idx < db_chunk.shape(0) {
        let dim_lines = query_vectors.shape(1);
        let mut sum = F::new(0.0);
        for i in 0..dim_lines {
            let q_line = query_vectors[query_idx * query_vectors.stride(0) + i];
            let d_line = db_chunk[db_idx * db_chunk.stride(0) + i];
            let diff = q_line - d_line;
            let sq = diff * diff;
            sum += sq[0];
            sum += sq[1];
            sum += sq[2];
            sum += sq[3];
        }
        distances[query_idx * distances.stride(0) + db_idx] = sum;
    }
}

/// Compute cosine distances on GPU
///
/// Each GPU thread computes the cosine distance (1 - cosine similarity)
/// between one query vector and one database vector. Requires pre-computed
/// L2 norms for both query and database vectors.
///
/// ### Params
///
/// * `query_vectors` - Query vectors [n_queries, dim / LINE_SIZE] as Line<F>
/// * `db_chunk` - Database vectors [n_db, dim / LINE_SIZE] as Line<F>
/// * `query_norms` - Pre-computed L2 norms [n_queries]
/// * `db_norms` - Pre-computed L2 norms [n_db]
/// * `distances` - Output matrix [n_queries, n_db] of cosine distances
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` → database vector index
/// * `ABSOLUTE_POS_Y` → query vector index
#[cube(launch_unchecked)]
pub fn cosine_distances_gpu_chunk<F: Float>(
    query_vectors: &Tensor<Line<F>>,
    db_chunk: &Tensor<Line<F>>,
    query_norms: &Tensor<F>,
    db_norms: &Tensor<F>,
    distances: &mut Tensor<F>,
) {
    let query_idx = ABSOLUTE_POS_Y as usize;
    let db_idx = ABSOLUTE_POS_X as usize;

    if query_idx < query_vectors.shape(0) && db_idx < db_chunk.shape(0) {
        let dim_lines = query_vectors.shape(1);

        let mut dot = F::new(0.0);

        for i in 0..dim_lines {
            let q_line = query_vectors[query_idx * query_vectors.stride(0) + i];
            let d_line = db_chunk[db_idx * db_chunk.stride(0) + i];
            let prod = q_line * d_line;

            // Manual unroll for LINE_SIZE = 4
            dot += prod[0];
            dot += prod[1];
            dot += prod[2];
            dot += prod[3];
        }

        let q_norm = query_norms[query_idx];
        let d_norm = db_norms[db_idx];

        distances[query_idx * distances.stride(0) + db_idx] =
            F::new(1.0) - (dot / (q_norm * d_norm));
    }
}

/// Compute Euclidean distances for a single IVF cluster and write to a
/// pre-allocated global candidate buffer at per-query offsets
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> vector index within the cluster (0..db_count)
/// * `ABSOLUTE_POS_Y` -> index within the active query list (0..n_active)
#[cube(launch_unchecked)]
pub fn compute_candidates_euclidean<F: Float>(
    query_vectors: &Tensor<Line<F>>,
    db_vectors: &Tensor<Line<F>>,
    active_indices: &Tensor<u32>,
    write_offsets: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
    db_start: u32,
    db_count: u32,
) {
    let local_db_idx = ABSOLUTE_POS_X;
    let active_q_idx = ABSOLUTE_POS_Y;

    if active_q_idx >= active_indices.len() as u32 || local_db_idx >= db_count {
        terminate!();
    }

    let real_q_idx = active_indices[active_q_idx as usize];
    let write_pos = write_offsets[active_q_idx as usize] + local_db_idx;
    let db_idx = db_start + local_db_idx;

    let dim_lines = query_vectors.shape(1);
    let mut sum = F::new(0.0);

    let q_offset = real_q_idx as usize * query_vectors.stride(0);
    let d_offset = db_idx as usize * db_vectors.stride(0);

    for i in 0..dim_lines {
        let q_line = query_vectors[q_offset + i];
        let d_line = db_vectors[d_offset + i];
        let diff = q_line - d_line;
        let sq = diff * diff;

        sum += sq[0];
        sum += sq[1];
        sum += sq[2];
        sum += sq[3];
    }

    let out_offset = real_q_idx as usize * out_dists.stride(0) + write_pos as usize;
    out_dists[out_offset] = sum;
    out_indices[out_offset] = db_idx;
}

/// Compute cosine distances for a single IVF cluster and write to a
/// pre-allocated global candidate buffer at per-query offsets
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> vector index within the cluster (0..db_count)
/// * `ABSOLUTE_POS_Y` -> index within the active query list (0..n_active)
#[cube(launch_unchecked)]
pub fn compute_candidates_cosine<F: Float>(
    query_vectors: &Tensor<Line<F>>,
    db_vectors: &Tensor<Line<F>>,
    query_norms: &Tensor<F>,
    db_norms: &Tensor<F>,
    active_indices: &Tensor<u32>,
    write_offsets: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
    db_start: u32,
    db_count: u32,
) {
    let local_db_idx = ABSOLUTE_POS_X;
    let active_q_idx = ABSOLUTE_POS_Y;

    if active_q_idx >= active_indices.len() as u32 || local_db_idx >= db_count {
        terminate!();
    }

    let real_q_idx = active_indices[active_q_idx as usize];
    let write_pos = write_offsets[active_q_idx as usize] + local_db_idx;
    let db_idx = db_start + local_db_idx;

    let dim_lines = query_vectors.shape(1);
    let mut dot = F::new(0.0);

    let q_offset = real_q_idx as usize * query_vectors.stride(0);
    let d_offset = db_idx as usize * db_vectors.stride(0);

    for i in 0..dim_lines {
        let q_line = query_vectors[q_offset + i];
        let d_line = db_vectors[d_offset + i];
        let prod = q_line * d_line;

        dot += prod[0];
        dot += prod[1];
        dot += prod[2];
        dot += prod[3];
    }

    let q_norm = query_norms[real_q_idx as usize];
    let d_norm = db_norms[db_idx as usize];

    let out_offset = real_q_idx as usize * out_dists.stride(0) + write_pos as usize;
    out_dists[out_offset] = F::new(1.0) - (dot / (q_norm * d_norm));
    out_indices[out_offset] = db_idx;
}

//////////////////////////////
// IVF mega kernel variants //
//////////////////////////////

/// Compute Euclidean distances using a flattened IVF task list
#[cube(launch_unchecked)]
pub fn compute_ivf_mega_euclidean<F: Float>(
    query_vectors: &Tensor<Line<F>>,
    db_vectors: &Tensor<Line<F>>,
    task_q_idx: &Tensor<u32>,
    task_db_start: &Tensor<u32>,
    task_write_offset: &Tensor<u32>,
    task_db_count: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
) {
    let local_db_idx = ABSOLUTE_POS_X;
    let task_idx = ABSOLUTE_POS_Y;

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

    let dim_lines = query_vectors.shape(1);
    let mut sum = F::new(0.0);

    let q_offset = q_idx as usize * query_vectors.stride(0);
    let d_offset = real_db_idx as usize * db_vectors.stride(0);

    for i in 0..dim_lines {
        let q_line = query_vectors[q_offset + i];
        let d_line = db_vectors[d_offset + i];
        let diff = q_line - d_line;
        let sq = diff * diff;

        sum += sq[0];
        sum += sq[1];
        sum += sq[2];
        sum += sq[3];
    }

    let out_offset = q_idx as usize * out_dists.stride(0) + write_pos as usize;
    out_dists[out_offset] = sum;
    out_indices[out_offset] = real_db_idx;
}

/// Compute cosine distances using a flattened IVF task list
#[cube(launch_unchecked)]
pub fn compute_ivf_mega_cosine<F: Float>(
    query_vectors: &Tensor<Line<F>>,
    db_vectors: &Tensor<Line<F>>,
    query_norms: &Tensor<F>,
    db_norms: &Tensor<F>,
    task_q_idx: &Tensor<u32>,
    task_db_start: &Tensor<u32>,
    task_write_offset: &Tensor<u32>,
    task_db_count: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
) {
    let local_db_idx = ABSOLUTE_POS_X;
    let task_idx = ABSOLUTE_POS_Y;

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

    let dim_lines = query_vectors.shape(1);
    let mut dot = F::new(0.0);

    let q_offset = q_idx as usize * query_vectors.stride(0);
    let d_offset = real_db_idx as usize * db_vectors.stride(0);

    for i in 0..dim_lines {
        let q_line = query_vectors[q_offset + i];
        let d_line = db_vectors[d_offset + i];
        let prod = q_line * d_line;

        dot += prod[0];
        dot += prod[1];
        dot += prod[2];
        dot += prod[3];
    }

    let q_norm = query_norms[q_idx as usize];
    let d_norm = db_norms[real_db_idx as usize];

    let out_offset = q_idx as usize * out_dists.stride(0) + write_pos as usize;
    out_dists[out_offset] = F::new(1.0) - (dot / (q_norm * d_norm));
    out_indices[out_offset] = real_db_idx;
}

/// In-place top-k reduction for the IVF variable-length candidate buffer
#[cube(launch_unchecked)]
pub fn reduce_ivf_topk<F: Float>(
    candidate_dists: &Tensor<F>,
    candidate_indices: &Tensor<u32>,
    candidates_per_query: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
) {
    let q_idx = ABSOLUTE_POS_X as usize;

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

////////////////////
// Main functions //
////////////////////

/// Container for batch query/DB data passed to `query_batch_gpu`
///
/// ### Fields
///
/// * `data` - Flattened vector data (n * dim elements)
/// * `norm` - Pre-computed L2 norms (n elements, empty if not cosine)
/// * `n` - Number of vectors
pub struct BatchData<'a, T> {
    pub data: &'a [T],
    pub norm: &'a [T],
    pub n: usize,
}

impl<'a, T> BatchData<'a, T> {
    /// Create a new BatchData instance
    pub fn new(data: &'a [T], norm: &'a [T], n: usize) -> Self {
        Self { data, norm, n }
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
) -> (Vec<Vec<usize>>, Vec<Vec<T>>)
where
    R: Runtime,
    T: Float + Sum + cubecl::CubeElement + num_traits::Float + num_traits::FromPrimitive,
{
    let client = R::client(&device);
    let vec_size = LINE_SIZE as usize;
    let dim_lines = dim / vec_size;

    let n_query_chunks = query_data.n.div_ceil(QUERY_CHUNK_SIZE);
    let n_db_chunks = db_data.n.div_ceil(DB_CHUNK_SIZE);

    // Single DB upload for the entire query
    let db_gpu = GpuTensor::<R, T>::from_slice(db_data.data, vec![db_data.n, dim], &client);

    let db_norms_gpu = if *metric == Dist::Cosine {
        Some(GpuTensor::<R, T>::from_slice(
            db_data.norm,
            vec![db_data.n],
            &client,
        ))
    } else {
        None
    };

    let mut all_indices = Vec::with_capacity(query_data.n);
    let mut all_distances = Vec::with_capacity(query_data.n);

    let max_db_chunk = DB_CHUNK_SIZE.min(db_data.n);

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
        );

        let query_norms_gpu = if *metric == Dist::Cosine {
            Some(GpuTensor::<R, T>::from_slice(
                &query_data.norm[query_start..query_end],
                vec![n_q],
                &client,
            ))
        } else {
            None
        };

        // Running top-k buffer (no ping-pong needed)
        let topk_dists = GpuTensor::<R, T>::empty(vec![n_q, k], &client);
        let topk_indices = GpuTensor::<R, u32>::empty(vec![n_q, k], &client);

        let init_gx = (k as u32).div_ceil(WORKGROUP_SIZE_X);
        let init_gy = (n_q as u32).div_ceil(WORKGROUP_SIZE_Y);
        unsafe {
            let _ = init_topk::launch_unchecked::<T, R>(
                &client,
                CubeCount::Static(init_gx, init_gy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y),
                topk_dists.clone().into_tensor_arg(1),
                topk_indices.clone().into_tensor_arg(1),
            );
        }

        // Reusable distance buffer sized for the largest possible chunk
        let distances_gpu = GpuTensor::<R, T>::empty(vec![n_q, max_db_chunk], &client);

        for db_chunk_idx in 0..n_db_chunks {
            let db_start = db_chunk_idx * DB_CHUNK_SIZE;
            let db_end = (db_start + DB_CHUNK_SIZE).min(db_data.n);
            let n_db = db_end - db_start;

            let grid_x = (n_db as u32).div_ceil(WORKGROUP_SIZE_X);
            let grid_y = (n_q as u32).div_ceil(WORKGROUP_SIZE_Y);

            match *metric {
                Dist::Euclidean => unsafe {
                    let _ = euclidean_tiled::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(grid_x, grid_y, 1),
                        CubeDim::new_2d(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y),
                        query_gpu.clone().into_tensor_arg(vec_size),
                        db_gpu.clone().into_tensor_arg(vec_size),
                        distances_gpu.clone().into_tensor_arg(1),
                        ScalarArg {
                            elem: db_start as u32,
                        },
                        ScalarArg { elem: n_db as u32 },
                        ScalarArg { elem: n_q as u32 },
                        ScalarArg {
                            elem: max_db_chunk as u32,
                        },
                        dim_lines,
                    );
                },
                Dist::Cosine => unsafe {
                    let _ = cosine_tiled::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(grid_x, grid_y, 1),
                        CubeDim::new_2d(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y),
                        query_gpu.clone().into_tensor_arg(vec_size),
                        db_gpu.clone().into_tensor_arg(vec_size),
                        query_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(1),
                        db_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(1),
                        distances_gpu.clone().into_tensor_arg(1),
                        ScalarArg {
                            elem: db_start as u32,
                        },
                        ScalarArg { elem: n_db as u32 },
                        ScalarArg { elem: n_q as u32 },
                        ScalarArg {
                            elem: max_db_chunk as u32,
                        },
                        dim_lines,
                    );
                },
            }

            // Extract directly into the running top-k buffer
            let extract_grid = (n_q as u32).div_ceil(WORKGROUP_SIZE_X);
            unsafe {
                let _ = extract_topk::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(extract_grid, 1, 1),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                    distances_gpu.clone().into_tensor_arg(1),
                    topk_dists.clone().into_tensor_arg(1),
                    topk_indices.clone().into_tensor_arg(1),
                    ScalarArg {
                        elem: db_start as u32,
                    },
                    ScalarArg { elem: n_db as u32 },
                );
            }
        }

        // Single GPU->CPU read per query chunk
        let final_dists = topk_dists.read(&client);
        let final_indices = topk_indices.read(&client);

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

    (all_indices, all_distances)
}

///////////
// Tests //
///////////

#[cfg(test)]
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

        let (_, gpu_dist) =
            query_batch_gpu::<f32, WgpuRuntime>(k, &qb, &dbb, dim, &Dist::Euclidean, device, false);

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

        let (_, gpu_dist) =
            query_batch_gpu::<f32, WgpuRuntime>(k, &qb, &dbb, dim, &Dist::Euclidean, device, false);

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
            query_batch_gpu::<f32, WgpuRuntime>(k, &qb, &dbb, dim, &Dist::Cosine, device, false);

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
            &Dist::Euclidean,
            device,
            false,
        );

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

        let (indices, distances) =
            query_batch_gpu::<f32, WgpuRuntime>(k, &qb, &dbb, dim, &Dist::Euclidean, device, false);

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

        let (idx, dist) =
            query_batch_gpu::<f32, WgpuRuntime>(1, &qb, &dbb, 4, &Dist::Euclidean, device, false);

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

        let (idx, dist) =
            query_batch_gpu::<f32, WgpuRuntime>(k, &qb, &dbb, dim, &Dist::Euclidean, device, false);

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

        let (idx, dist) =
            query_batch_gpu::<f32, WgpuRuntime>(1, &qb, &dbb, dim, &Dist::Euclidean, device, false);

        assert_eq!(idx[0][0], 0);
        assert!((dist[0][0] - 64.0).abs() < 1e-3);
    }
}
