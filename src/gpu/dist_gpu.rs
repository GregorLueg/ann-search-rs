//! GPU-accelerated distance calculations and top-k selection for GPU-based
//! indices. Contains kernels for both the exhaustive search pipeline and the
//! IVF fire-and-forget pipeline.
//!
//! The insertion-sort reducers here are the fallback arms; the radix-select
//! reducers that serve the common case live in [`crate::gpu::topk_gpu`].

#![allow(missing_docs)]

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use std::iter::Sum;

use crate::gpu::topk_gpu::{radix_select_topk, radix_select_usable, RADIX_SELECT_MIN_K};
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

/// Tiled distance kernel with shared-memory query caching
///
/// All threads in a workgroup cooperatively load the query tile into scalar
/// shared memory, eliminating redundant global reads across threads sharing
/// the same query row. DB vectors are read directly from global memory via
/// the `db_start` offset into a pre-uploaded full DB tensor.
///
/// Computes squared Euclidean, or `1 - dot(q, d) / (||q|| * ||d||)` under
/// cosine.
///
/// Shared memory usage: `WORKGROUP_SIZE_Y * dim_lines * N` scalars.
///
/// ### Params
///
/// * `query_vectors` - Query vectors `[n_queries, dim / N]` as `Vector<F, N>`
/// * `db_vectors` - Database vectors `[n_db, dim / N]` as `Vector<F, N>`
/// * `query_norms` - Pre-computed L2 norms `[n_queries]`; read only under
///   cosine
/// * `db_norms` - Pre-computed L2 norms `[n_db]`; read only under cosine
/// * `distances` - Output distance matrix `[n_queries, dist_stride]`
/// * `db_start` - Global offset into `db_vectors` for this chunk
/// * `n_db_chunk` - Number of DB vectors in this chunk
/// * `n_queries` - Total number of query vectors
/// * `dist_stride` - Column stride of the output distance matrix
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime)
/// * `size_y` - Safe workgroup size Y for the given dimensionality
/// * `use_cosine` - Cosine instead of squared Euclidean (comptime)
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` -> DB vector index within chunk
/// * `ABSOLUTE_POS_Y` -> query vector index
#[cube(launch_unchecked)]
pub fn dist_tiled<F: Float, N: Size>(
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
    #[comptime] use_cosine: bool,
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
            if use_cosine {
                sum += s_query[s_off + lane] * d_line[lane];
            } else {
                let diff = s_query[s_off + lane] - d_line[lane];
                sum += diff * diff;
            }
        }
    }
    if use_cosine {
        sum = F::new(1.0_f32) - (sum / (query_norms[query_idx] * db_norms[global_db_idx]));
    }
    distances[query_idx * dist_stride as usize + db_idx] = sum;
}

/// Register-tiled distance kernel
///
/// Each thread computes a `tile_q x tile_d` block of the distance matrix
/// instead of a single entry, so each loaded value feeds several FMAs.
///
/// The reduction axis is blocked: only `kb_lines` lines of the query tile are
/// staged at a time, so the shared-memory footprint is independent of `dim` and
/// `size_y` can stay at its knee at any dimensionality.
///
/// Computes squared Euclidean, or `1 - dot(q, d) / (||q|| * ||d||)` under
/// cosine, normalising on writeback.
///
/// ### Params
///
/// * `query_vectors` - Query vectors `[n_queries, dim / N]` as `Vector<F, N>`
/// * `db_vectors` - Database vectors `[n_db, dim / N]` as `Vector<F, N>`
/// * `query_norms` - Pre-computed L2 norms `[n_queries]`; read only under
///   cosine
/// * `db_norms` - Pre-computed L2 norms `[n_db]`; read only under cosine
/// * `distances` - Output distance matrix `[n_queries, dist_stride]`
/// * `q_start` - Global offset into `query_vectors` (and `query_norms`) of
///   row 0 of this launch. Non-zero when the roles are swapped so the output
///   is written candidate-major, see `query_batch_gpu`
/// * `db_start` - Global offset into `db_vectors` for this chunk
/// * `n_db_chunk` - Number of DB vectors in this chunk
/// * `n_queries` - Total number of query vectors
/// * `dist_stride` - Column stride of the output distance matrix
/// * `dim_lines` - Number of `Vector<F, N>` elements per vector row (comptime)
/// * `size_y` - Number of query rows staged in shared memory (comptime).
///   Must be divisible by `tile_q`
/// * `tile_d` - DB vectors per thread (comptime)
/// * `tile_q` - Query vectors per thread (comptime)
/// * `kb_lines` - Reduction lines staged per block (comptime). Must divide
///   `dim_lines` exactly
/// * `use_cosine` - Cosine instead of squared Euclidean (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_X` -> block of `WORKGROUP_SIZE_X * tile_d` DB vectors
/// * `UNIT_POS_X` -> lane within that block
/// * `UNIT_POS_Y` -> block of `tile_q` query rows within the shared tile
#[cube(launch_unchecked)]
pub fn dist_tiled_reg<F: Float, N: Size>(
    query_vectors: &Tensor<Vector<F, N>>,
    db_vectors: &Tensor<Vector<F, N>>,
    query_norms: &Tensor<F>,
    db_norms: &Tensor<F>,
    distances: &mut Tensor<F>,
    q_start: u32,
    db_start: u32,
    n_db_chunk: u32,
    n_queries: u32,
    dist_stride: u32,
    #[comptime] dim_lines: usize,
    #[comptime] size_y: u32,
    #[comptime] tile_d: usize,
    #[comptime] tile_q: usize,
    #[comptime] kb_lines: usize,
    #[comptime] use_cosine: bool,
) {
    let lanes = LINE_SIZE;
    let kb_scalars = kb_lines * lanes;
    let n_blocks = dim_lines / kb_lines;
    let wg_y = size_y as usize;
    let local_x = UNIT_POS_X as usize;
    let local_y = UNIT_POS_Y as usize;

    // Scalar shared memory only (vectorised shared mem silently broadcasts lane 0)
    let mut s_query = SharedMemory::<F>::new(wg_y * kb_scalars);

    // Locals at kernel scope, never inside a branch or loop.
    let mut acc = Array::<F>::new(tile_q * tile_d);
    let mut d_scalars = Array::<F>::new(tile_d * lanes);

    let threads_y = wg_y / tile_q;
    let thread_id = local_y * WORKGROUP_SIZE_X as usize + local_x;
    let total_threads = WORKGROUP_SIZE_X as usize * threads_y;
    let total_elems = wg_y * kb_scalars;

    // First query row owned by this cube.
    let q_tile_base = ((CUBE_POS_Z * CUBE_COUNT_Y + CUBE_POS_Y) as usize) * wg_y;

    #[unroll]
    for a in 0..tile_q * tile_d {
        acc[a] = F::new(0.0_f32);
    }

    let db_tile_base = (CUBE_POS_X as usize) * (WORKGROUP_SIZE_X as usize) * tile_d;
    let q_row_base = local_y * tile_q;

    for b in 0..n_blocks {
        let kb_base = b * kb_lines;

        let mut load_idx = thread_id;
        while load_idx < total_elems {
            let q_local = load_idx / kb_scalars;
            let elem = load_idx % kb_scalars;
            let q_global = q_tile_base + q_local;
            if q_global < n_queries as usize {
                let line_idx = kb_base + elem / lanes;
                let lane = elem % lanes;
                let line_val = query_vectors[(q_start as usize + q_global) * dim_lines + line_idx];
                s_query[load_idx] = line_val[lane];
            } else {
                s_query[load_idx] = F::new(0.0_f32);
            }
            load_idx += total_threads;
        }
        sync_cube();

        for j in 0..kb_lines {
            let i = kb_base + j;
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
                let s_off = (q_row_base + t) * kb_scalars + j * lanes;
                #[unroll]
                for lane in 0..lanes {
                    let qv = s_query[s_off + lane];
                    #[unroll]
                    for r in 0..tile_d {
                        if use_cosine {
                            acc[t * tile_d + r] += qv * d_scalars[r * lanes + lane];
                        } else {
                            let diff = qv - d_scalars[r * lanes + lane];
                            acc[t * tile_d + r] += diff * diff;
                        }
                    }
                }
            }
        }
        // The next block overwrites the tile, so every thread must be done
        // reading it first. Comptime because with one block there is no next
        // one, and the barrier is not free.
        if comptime!(n_blocks > 1) {
            sync_cube();
        }
    }

    #[unroll]
    for t in 0..tile_q {
        let q_global = q_tile_base + q_row_base + t;
        #[unroll]
        for r in 0..tile_d {
            let db_local = db_tile_base + local_x + r * WORKGROUP_SIZE_X as usize;
            if q_global < n_queries as usize && db_local < n_db_chunk as usize {
                let mut dist = acc[t * tile_d + r];
                if use_cosine {
                    let q_norm = query_norms[q_start as usize + q_global];
                    let d_norm = db_norms[db_start as usize + db_local];
                    dist = F::new(1.0_f32) - (dist / (q_norm * d_norm));
                }
                distances[q_global * dist_stride as usize + db_local] = dist;
            }
        }
    }
}

/////////////////////
// Top-k selection //
/////////////////////

/// Distance loads `extract_topk` issues per thread before consuming the first.
const EXTRACT_UNROLL: usize = 8;

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
/// The scan is latency bound, so loads are issued `EXTRACT_UNROLL` at a time
/// ahead of their use, and the candidate-major layout makes adjacent lanes read
/// adjacent words.
///
/// This is the low-`k` arm of the exhaustive path: `query_batch_gpu` dispatches
/// here below [`RADIX_SELECT_MIN_K`], and also whenever [`radix_select_usable`]
/// says the radix reducer cannot serve the configuration, i.e. non-f32 elements
/// or a runtime without `u32` atomics. Its acceptance test is a strict `<` while
/// scanning columns upwards, so among bit-identical distances the earliest
/// candidate wins; the radix reducer selects under the same total order.
///
/// ### Params
///
/// * `distances` - Distance matrix for this chunk, in either orientation;
///   entry `(q, i)` sits at `q * q_stride + i * i_stride`
/// * `out_dists` - Running top-k distance buffer `[n_queries, k]`
/// * `out_indices` - Running top-k index buffer `[n_queries, k]`
/// * `chunk_offset` - Global DB index corresponding to column 0 of this chunk
/// * `actual_chunk_size` - Number of valid candidates in this chunk
/// * `q_stride` - Stride between queries in `distances`
/// * `i_stride` - Stride between candidates in `distances`. Set to 1 with
///   `q_stride = dist_stride` for the query-major layout, or the reverse for
///   the candidate-major layout, where adjacent lanes read adjacent words
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
    q_stride: u32,
    i_stride: u32,
    k_param: u32,
    #[comptime] k: usize,
) {
    let query_idx =
        ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * WORKGROUP_SIZE_X + UNIT_POS_X) as usize;

    if query_idx >= out_dists.shape(0) {
        terminate!();
    }

    let kr = k_param as usize;
    let dist_offset = query_idx * q_stride as usize;
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

    // Running k-th distance, refreshed only on an insertion.
    let mut thresh = local_dists[kr - 1];
    // `EXTRACT_UNROLL` loads are issued before the first is consumed, so they
    // are in flight together. Out-of-range slots read as `MAX` and never pass
    // the threshold.
    let mut batch = Array::<F>::new(EXTRACT_UNROLL);
    let mut base = 0u32;
    while base < actual_chunk_size {
        #[unroll]
        for u in 0..EXTRACT_UNROLL {
            let ii = base + u as u32;
            let mut v = F::new(f32::MAX);
            if ii < actual_chunk_size {
                v = distances[dist_offset + ii as usize * i_stride as usize];
            }
            batch[u] = v;
        }

        #[unroll]
        for u in 0..EXTRACT_UNROLL {
            let dist = batch[u];
            if dist < thresh {
                // First-match guard rather than a `bool` sentinel: see the
                // codegen rules in the `nndescent_gpu` module header.
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
                local_indices[pos] = chunk_offset + base + u as u32;
                thresh = local_dists[kr - 1];
            }
        }
        base += EXTRACT_UNROLL as u32;
    }

    for i in 0..k {
        out_dists[out_offset + i] = local_dists[i];
        out_indices[out_offset + i] = local_indices[i];
    }
}

/// Run batch kNN queries on the GPU
///
/// Uses tiled distance kernels with shared-memory query caching, a single
/// DB upload, and radix-select top-k extraction directly into a running buffer
/// (no ping-pong or merge step). Falls back to the serial insertion-sort
/// `extract_topk` for element types whose bits the radix key cannot
/// reinterpret.
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
    query_batch_gpu_resident(
        k,
        query_data,
        &db_gpu,
        db_norms_gpu.as_ref(),
        db_data.n,
        dim,
        metric,
        &client,
        verbose,
    )
}

/// Run batch kNN queries on the GPU against an already-uploaded database
///
/// [`query_batch_gpu`] without the upload: the caller owns the database
/// tensors and can reuse them across calls. On a large database the upload and
/// its synchronisation dominate a small batch's wall time.
///
/// ### Params
///
/// * `k` - Number of neighbours to return
/// * `query_data` - Query vectors as `BatchData`
/// * `db_gpu` - Database vectors on the device, `[n_db, dim]`
/// * `db_norms_gpu` - Database L2 norms on the device, `[n_db]`; required
///   for Cosine, ignored otherwise
/// * `n_db` - Number of database vectors
/// * `dim` - Embedding dimensionality (must be divisible by LINE_SIZE)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `client` - Compute client the tensors live on
/// * `verbose` - Print progress for large batches
///
/// ### Returns
///
/// Tuple of `(indices, distances)` where each inner Vec has k elements
#[allow(clippy::too_many_arguments)]
pub fn query_batch_gpu_resident<T, R>(
    k: usize,
    query_data: &BatchData<T>,
    db_gpu: &GpuTensor<R, T>,
    db_norms_gpu: Option<&GpuTensor<R, T>>,
    n_db: usize,
    dim: usize,
    metric: &Dist,
    client: &ComputeClient<R>,
    verbose: bool,
) -> KnnResult<T>
where
    R: Runtime,
    T: Float + Sum + cubecl::CubeElement + num_traits::Float + num_traits::FromPrimitive,
{
    // Handles only: cloning a `GpuTensor` shares the device buffer.
    let client = client.clone();
    let db_gpu = db_gpu.clone();
    let db_norms_gpu = db_norms_gpu.cloned();
    let use_cosine = match *metric {
        Dist::SquaredEuclidean => false,
        Dist::Cosine => true,
        Dist::Manhattan => unreachable!(),
    };
    // Stands in for the norm buffers the Euclidean arm never reads.
    let norm_dummy = GpuTensor::<R, T>::from_slice(&[T::zero()], vec![1], &client)?;
    let limits = GpuLimits::from_client(&client);
    let vec_size = LINE_SIZE;
    let dim_lines = dim / vec_size;
    let safe_worksize_y = pick_wg_y(dim, size_of::<T>(), &limits)?;
    // Blocking the reduction axis frees the query tile from `dim`, so the
    // register-tiled path runs at `EXH_WG_Y` where `pick_wg_y` would have to
    // shrink. `None` means the divisors of `dim_lines` rule it out; the
    // whole-row staging below is then the arm that runs.
    let staging = plan_exhaustive_staging(dim, size_of::<T>(), &limits);

    let n_query_chunks = query_data.n.div_ceil(QUERY_CHUNK_SIZE);
    // The DB chunk shrinks when the distance transient would not fit one
    // binding on this device. On a 4 GiB binding limit it is the full chunk.
    let db_chunk = plan_db_chunk(QUERY_CHUNK_SIZE.min(query_data.n), size_of::<T>(), &limits);
    let n_db_chunks = n_db.div_ceil(db_chunk);

    let mut all_indices = Vec::with_capacity(query_data.n);
    let mut all_distances = Vec::with_capacity(query_data.n);

    let max_db_chunk = db_chunk.min(n_db);

    let wg = WORKGROUP_SIZE_X as usize;
    let use_radix =
        k >= RADIX_SELECT_MIN_K && radix_select_usable(&client, k, size_of::<T>(), wg, &limits);
    // Radix select runs one cube per query along a contiguous row, so it keeps
    // the query-major layout.
    let candidate_major = staging.is_some() && !use_radix;

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
            let db_end = (db_start + db_chunk).min(n_db);
            let n_db = db_end - db_start;

            let grid_x = (n_db as u32).div_ceil(WORKGROUP_SIZE_X);
            let (grid_y, grid_z) = grid_2d((n_q as u32).div_ceil(safe_worksize_y), &limits)?;

            // The tiled kernel stages one operand in shared memory and spreads
            // the other across lanes, writing `[staged, lane]`. Candidate-major
            // stages the DB chunk and spreads the queries, so `extract_topk`'s
            // one-thread-per-query scan reads adjacent words across lanes
            // instead of rows a whole chunk apart.
            let (stage_start, n_stage, lane_start, n_lane, out_stride) = if candidate_major {
                (db_start, n_db, 0, n_q, n_q)
            } else {
                (0, n_q, db_start, n_db, max_db_chunk)
            };

            let q_norms = query_norms_gpu.as_ref().unwrap_or(&norm_dummy);
            let d_norms = db_norms_gpu.as_ref().unwrap_or(&norm_dummy);
            if let Some(plan) = staging {
                // Register-tiled path. Bit-exact against the untiled kernel.
                let (stage_vecs, lane_vecs, stage_norms, lane_norms) = if candidate_major {
                    (&db_gpu, &query_gpu, d_norms, q_norms)
                } else {
                    (&query_gpu, &db_gpu, q_norms, d_norms)
                };
                let reg_grid_x = (n_lane as u32).div_ceil(WORKGROUP_SIZE_X * TILE_D as u32);
                let (reg_y, reg_z) = grid_2d((n_stage as u32).div_ceil(plan.wg_y), &limits)?;
                unsafe {
                    dist_tiled_reg::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(reg_grid_x, reg_y, reg_z),
                        CubeDim::new_2d(WORKGROUP_SIZE_X, plan.wg_y / TILE_Q as u32),
                        vec_size,
                        stage_vecs.clone().into_tensor_arg(),
                        lane_vecs.clone().into_tensor_arg(),
                        stage_norms.clone().into_tensor_arg(),
                        lane_norms.clone().into_tensor_arg(),
                        distances_gpu.clone().into_tensor_arg(),
                        stage_start as u32,
                        lane_start as u32,
                        n_lane as u32,
                        n_stage as u32,
                        out_stride as u32,
                        dim_lines,
                        plan.wg_y,
                        TILE_D,
                        TILE_Q,
                        plan.kb_lines,
                        use_cosine,
                    );
                }
            } else {
                unsafe {
                    dist_tiled::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(grid_x, grid_y, grid_z),
                        CubeDim::new_2d(WORKGROUP_SIZE_X, safe_worksize_y),
                        vec_size,
                        query_gpu.clone().into_tensor_arg(),
                        db_gpu.clone().into_tensor_arg(),
                        q_norms.clone().into_tensor_arg(),
                        d_norms.clone().into_tensor_arg(),
                        distances_gpu.clone().into_tensor_arg(),
                        db_start as u32,
                        n_db as u32,
                        n_q as u32,
                        max_db_chunk as u32,
                        dim_lines,
                        safe_worksize_y,
                        use_cosine,
                    );
                }
            }

            // Extract directly into the running top-k buffer.
            //
            // Radix select above `RADIX_SELECT_MIN_K`, the serial insertion
            // sort below it.
            //
            // `extract_topk` is also the fallback for element types whose bits
            // the key transform cannot reinterpret, and for runtimes without
            // `u32` atomics.
            let (q_stride, i_stride) = if candidate_major {
                (1, n_q)
            } else {
                (max_db_chunk, 1)
            };
            if use_radix {
                let (rx, ry) = grid_2d(n_q as u32, &limits)?;
                unsafe {
                    radix_select_topk::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(rx, ry, 1),
                        CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                        distances_gpu.clone().into_tensor_arg(),
                        topk_dists.clone().into_tensor_arg(),
                        topk_indices.clone().into_tensor_arg(),
                        db_start as u32,
                        n_db as u32,
                        k as u32,
                        k,
                        wg,
                    );
                }
            } else {
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
                        q_stride as u32,
                        i_stride as u32,
                        k as u32,
                        k,
                    );
                }
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

    // Cosine can round a self-distance to just under zero; clamp once here
    // rather than in the kernel, since this runs once per query, not per
    // candidate.
    for row in all_distances.iter_mut() {
        for d in row.iter_mut() {
            if *d < T::zero() {
                *d = T::zero();
            }
        }
    }

    Ok((all_indices, all_distances))
}

//////////////////////////////
// IVF mega kernel variants //
//////////////////////////////

/// In-place top-k reduction for the IVF variable-length candidate buffer
///
/// One thread per query. Performs insertion-sort over the variable-length
/// candidate slice produced by the distance kernels and writes the k
/// smallest results into the output buffers.
///
/// The IVF fallback arm, taken whenever [`radix_select_usable`] says the radix
/// reducer cannot serve the configuration, i.e. non-f32 elements or a runtime
/// without `u32` atomics. Unlike the radix reducer it inserts straight into the
/// output buffers rather than writing every slot, so the call site must seed
/// them with `init_topk` first; omitting that produces garbage rather than an
/// error.
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

/// IVF mega kernel with shared-memory query caching.
///
/// Each task is one (query, cluster) pair, and the grid maps threads directly
/// to `(db_element, task)` pairs, avoiding a per-cluster kernel launch. Query
/// vectors are loaded cooperatively into scalar shared memory so that all
/// X-threads in a workgroup row (and Y-threads sharing the same query) read
/// from shared memory instead of global memory.
///
/// Also caches per-task metadata (q_idx, db_start, write_offset, db_count and,
/// under cosine, the query norm) in shared memory to avoid redundant global
/// reads across the 32 X-threads in each row.
///
/// Computes squared Euclidean, or `1 - dot(q, d) / (||q|| * ||d||)` under
/// cosine.
///
/// ### Params
///
/// * `query_vectors` - Query vectors `[n_queries, dim]` as `Vector<F, N>`.
///   Shape must be in element units (not vector units). Accessed via
///   the comptime `dim_lines` parameter, never via tensor strides.
/// * `db_vectors` - Full database vectors `[n_db, dim]` as `Vector<F, N>`.
///   Same element-unit shape convention as `query_vectors`.
/// * `query_norms` - Pre-computed L2 norms `[n_queries]`; read only under
///   cosine
/// * `db_norms` - Pre-computed L2 norms `[n_db]`; read only under cosine
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
/// * `use_cosine` - Cosine instead of squared Euclidean (comptime)
///
/// ### Note
///
/// Register tiling this over `TILE_D` DB vectors, which pays on the exhaustive
/// kernel, measured *slower* here. The x extent is ragged: the grid is sized by
/// the largest cluster, so for an average task most of it is out of range, and
/// 1x1 lets those threads `terminate!()` at once where a 4-wide tile keeps a
/// thread alive for its whole row as soon as one of its four DB vectors is in
/// range. Do not retry it without first bucketing tasks by cluster size.
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
/// * `s_query_norms[32]` - query L2 norm per Y-slot; allocated under both
///   metrics because shared memory must be declared at kernel scope
/// * `s_query[32 * dim_scalars]` - query vectors in scalar form, where
///   `dim_scalars = dim_lines * N`
#[cube(launch_unchecked)]
pub fn compute_ivf_mega_cached<F: Float, N: Size>(
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
    #[comptime] use_cosine: bool,
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
    let mut s_query_norms = SharedMemory::<F>::new(share_mem_size);

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
        if use_cosine {
            qn_val = query_norms[q as usize];
        }
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

    let mut sum = F::new(0.0_f32);
    for i in 0..dim_lines {
        let d_line = db_vectors[d_offset + i];
        let s_off = q_shared_base + i * lanes;
        #[unroll]
        for lane in 0..lanes {
            if use_cosine {
                sum += s_query[s_off + lane] * d_line[lane];
            } else {
                let diff = s_query[s_off + lane] - d_line[lane];
                sum += diff * diff;
            }
        }
    }
    if use_cosine {
        let q_norm = s_query_norms[local_y];
        let d_norm = db_norms[real_db_idx as usize];
        sum = F::new(1.0_f32) - (sum / (q_norm * d_norm));
    }

    let q_idx = s_q_idx[local_y];
    let out_offset = q_idx as usize * out_dists.stride(0) + write_pos as usize;
    out_dists[out_offset] = sum;
    out_indices[out_offset] = real_db_idx;
}

/// Fields per entry of the IVF tile list: first task, task count, cluster
/// start in the reorganised DB, offset of the tile within the cluster, and the
/// tile's DB count.
pub const IVF_TILE_FIELDS: usize = 5;

/// Register-tiled IVF distance kernel over a cluster-major tile list
///
/// Each cube owns one tile: up to `size_y` tasks that all probe the same
/// cluster, against up to `WORKGROUP_SIZE_X * tile_d` of that cluster's
/// points. The tasks' query rows are staged a reduction block at a time, as in
/// `dist_tiled_reg`, and every thread computes a `tile_q x tile_d` block.
/// The tile list is built on the host from the task list's cluster grouping,
/// so the grid is sized by the real work rather than by the largest cluster.
///
/// Accumulation runs over lines then lanes in ascending order, the same order
/// as the per-task mega kernels, so the distances are bit-identical to theirs.
///
/// ### Params
///
/// * `query_vectors` - Query vectors `[n_queries, dim / N]` as `Vector<F, N>`
/// * `db_vectors` - Reorganised DB vectors `[n_db, dim / N]` as `Vector<F, N>`
/// * `query_norms` - Query L2 norms `[n_queries]`; read only under cosine
/// * `db_norms` - DB L2 norms `[n_db]`; read only under cosine
/// * `task_q_idx` - Query index per task `[n_tasks]`, grouped by cluster
/// * `task_write_offset` - Start of the task's cluster in its query's candidate
///   row `[n_tasks]`
/// * `tiles` - Tile list `[n_tiles * IVF_TILE_FIELDS]`
/// * `out_dists` - Candidate distances `[n_queries, out_stride]`
/// * `out_indices` - Candidate DB indices `[n_queries, out_stride]`
/// * `out_stride` - Row stride of the candidate buffers
/// * `n_tiles` - Number of tiles
/// * `dim_lines` - `Vector<F, N>` elements per row (comptime)
/// * `size_y` - Tasks per tile (comptime). Must be divisible by `tile_q`
/// * `tile_d` - DB vectors per thread (comptime)
/// * `tile_q` - Tasks per thread (comptime)
/// * `kb_lines` - Reduction lines staged per block (comptime). Must divide
///   `dim_lines` exactly
/// * `use_cosine` - Cosine instead of squared Euclidean (comptime)
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> tile index
/// * `UNIT_POS_X` -> lane over the tile's DB vectors
/// * `UNIT_POS_Y` -> block of `tile_q` tasks within the tile
#[cube(launch_unchecked)]
pub fn ivf_tiled<F: Float, N: Size>(
    query_vectors: &Tensor<Vector<F, N>>,
    db_vectors: &Tensor<Vector<F, N>>,
    query_norms: &Tensor<F>,
    db_norms: &Tensor<F>,
    task_q_idx: &Tensor<u32>,
    task_write_offset: &Tensor<u32>,
    tiles: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
    out_stride: u32,
    n_tiles: u32,
    #[comptime] dim_lines: usize,
    #[comptime] size_y: u32,
    #[comptime] tile_d: usize,
    #[comptime] tile_q: usize,
    #[comptime] kb_lines: usize,
    #[comptime] use_cosine: bool,
) {
    let tile = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    if tile >= n_tiles {
        terminate!();
    }

    let lanes = LINE_SIZE;
    let kb_scalars = kb_lines * lanes;
    let n_blocks = dim_lines / kb_lines;
    let wg_y = size_y as usize;
    let local_x = UNIT_POS_X as usize;
    let local_y = UNIT_POS_Y as usize;

    let tb = tile as usize * IVF_TILE_FIELDS;
    let task_start = tiles[tb] as usize;
    let n_t = tiles[tb + 1] as usize;
    let cluster_start = tiles[tb + 2] as usize;
    let db_off = tiles[tb + 3] as usize;
    let n_db = tiles[tb + 4] as usize;

    // Scalar shared memory only (vectorised shared mem silently broadcasts lane 0)
    let mut s_query = SharedMemory::<F>::new(wg_y * kb_scalars);
    let mut acc = Array::<F>::new(tile_q * tile_d);
    let mut d_scalars = Array::<F>::new(tile_d * lanes);

    let threads_y = wg_y / tile_q;
    let thread_id = local_y * WORKGROUP_SIZE_X as usize + local_x;
    let total_threads = WORKGROUP_SIZE_X as usize * threads_y;
    let total_elems = wg_y * kb_scalars;

    #[unroll]
    for a in 0..tile_q * tile_d {
        acc[a] = F::new(0.0_f32);
    }

    let q_row_base = local_y * tile_q;

    for b in 0..n_blocks {
        let kb_base = b * kb_lines;

        let mut load_idx = thread_id;
        while load_idx < total_elems {
            let q_local = load_idx / kb_scalars;
            let elem = load_idx % kb_scalars;
            if q_local < n_t {
                let q = task_q_idx[task_start + q_local] as usize;
                let line_val = query_vectors[q * dim_lines + kb_base + elem / lanes];
                s_query[load_idx] = line_val[elem % lanes];
            } else {
                s_query[load_idx] = F::new(0.0_f32);
            }
            load_idx += total_threads;
        }
        sync_cube();

        for j in 0..kb_lines {
            let i = kb_base + j;
            #[unroll]
            for r in 0..tile_d {
                let d_local = local_x + r * WORKGROUP_SIZE_X as usize;
                // Clamp out-of-range rows to the tile's first row; the result
                // is masked on write.
                let mut idx = (cluster_start + db_off) * dim_lines + i;
                if d_local < n_db {
                    idx = (cluster_start + db_off + d_local) * dim_lines + i;
                }
                let line_val = db_vectors[idx];
                #[unroll]
                for lane in 0..lanes {
                    d_scalars[r * lanes + lane] = line_val[lane];
                }
            }

            #[unroll]
            for t in 0..tile_q {
                let s_off = (q_row_base + t) * kb_scalars + j * lanes;
                #[unroll]
                for lane in 0..lanes {
                    let qv = s_query[s_off + lane];
                    #[unroll]
                    for r in 0..tile_d {
                        if use_cosine {
                            acc[t * tile_d + r] += qv * d_scalars[r * lanes + lane];
                        } else {
                            let diff = qv - d_scalars[r * lanes + lane];
                            acc[t * tile_d + r] += diff * diff;
                        }
                    }
                }
            }
        }
        if comptime!(n_blocks > 1) {
            sync_cube();
        }
    }

    #[unroll]
    for t in 0..tile_q {
        let q_local = q_row_base + t;
        #[unroll]
        for r in 0..tile_d {
            let d_local = local_x + r * WORKGROUP_SIZE_X as usize;
            // Nested rather than `&&`: compound conditions miscompile, see the
            // codegen rules in the `nndescent_gpu` module header. The `&&`
            // form dropped single lanes' writes on the CPU runtime.
            #[allow(clippy::collapsible_if)]
            if q_local < n_t {
                if d_local < n_db {
                    let task = task_start + q_local;
                    let q = task_q_idx[task] as usize;
                    let db_idx = cluster_start + db_off + d_local;
                    let pos = q * out_stride as usize
                        + task_write_offset[task] as usize
                        + db_off
                        + d_local;
                    let mut dist = acc[t * tile_d + r];
                    if use_cosine {
                        dist = F::new(1.0_f32) - (dist / (query_norms[q] * db_norms[db_idx]));
                    }
                    out_dists[pos] = dist;
                    out_indices[pos] = db_idx as u32;
                }
            }
        }
    }
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

    /// Helper: build synthetic IVF task data and run a mega kernel,
    /// returning the output (dists, indices) flat buffers.
    ///
    /// * `queries` - flat query data `[n_queries * dim]`
    /// * `db` - flat DB data `[n_db * dim]`
    /// * `tasks` - Vec of `(q_idx, db_start, write_offset, db_count)`
    /// * `n_queries`, `n_db`, `dim`, `max_candidates` - dimensions
    /// * `device` - GPU device
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
        let no_norms = GpuTensor::<WgpuRuntime, f32>::from_slice(&[1.0], vec![1], &client).unwrap();
        let max_db_count = tasks.iter().map(|t| t.3).max().unwrap_or(0);
        let gx = max_db_count.div_ceil(WORKGROUP_SIZE_X).max(1);
        let (gy, gz) = grid_2d((n_tasks as u32).div_ceil(4), &limits).unwrap();
        unsafe {
            compute_ivf_mega_cached::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(gx, gy, gz),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 4),
                vec_size,
                q_gpu.into_tensor_arg(),
                db_gpu.into_tensor_arg(),
                no_norms.clone().into_tensor_arg(),
                no_norms.into_tensor_arg(),
                tq_gpu.into_tensor_arg(),
                tds_gpu.into_tensor_arg(),
                two_gpu.into_tensor_arg(),
                tdc_gpu.into_tensor_arg(),
                out_d.clone().into_tensor_arg(),
                out_i.clone().into_tensor_arg(),
                n_tasks as u32,
                dim_lines,
                4,
                false,
            );
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
