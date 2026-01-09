use cubecl::prelude::*;
use std::iter::Sum;

use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::utils::dist::Dist;

//////////////////////
// Distance kernels //
//////////////////////

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
    let query_idx = ABSOLUTE_POS_Y;
    let db_idx = ABSOLUTE_POS_X;

    if query_idx < query_vectors.shape(0) && db_idx < db_chunk.shape(0) {
        let dim_lines = query_vectors.shape(1);

        let mut sum = F::new(0.0);

        for i in 0..dim_lines {
            let q_line = query_vectors[query_idx * query_vectors.stride(0) + i];
            let d_line = db_chunk[db_idx * db_chunk.stride(0) + i];
            let diff = q_line - d_line;
            let sq = diff * diff;

            // Manual unroll for LINE_SIZE = 4
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
    let query_idx = ABSOLUTE_POS_Y;
    let db_idx = ABSOLUTE_POS_X;

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

///////////////////
// Top N kernels //
///////////////////

/// Extract top-k smallest distances per query from a distance chunk
///
/// Simple insertion-based approach - one thread per query.
/// O(chunk_size × k) per query, but saves massive CPU transfer.
///
/// ### Params
/// * `distances` - [n_queries, chunk_size] distance matrix
/// * `out_dists` - [n_queries, k] output distances (must be pre-init to MAX)
/// * `out_indices` - [n_queries, k] output indices
/// * `chunk_offset` - Global index offset for this DB chunk
#[cube(launch_unchecked)]
pub fn extract_topk<F: Float>(
    distances: &Tensor<F>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
    chunk_offset: u32,
) {
    let query_idx = ABSOLUTE_POS_X;

    if query_idx >= distances.shape(0) {
        terminate!();
    }

    let chunk_size = distances.shape(1);
    let k = out_dists.shape(1);
    let dist_offset = query_idx * distances.stride(0);
    let out_offset = query_idx * out_dists.stride(0);

    for i in 0..chunk_size {
        let dist = distances[dist_offset + i];

        // Only process if better than current worst
        if dist < out_dists[out_offset + k - 1] {
            // Find insertion point (first position where dist < current)
            let mut insert_pos: u32 = k - 1;
            for j in 0..k {
                if dist < out_dists[out_offset + j] && insert_pos == k - 1 {
                    insert_pos = j;
                }
            }

            // Shift elements right from insert_pos to k-2
            for j in 0..k - 1 {
                let src = k - 2 - j;
                let dst = k - 1 - j;
                if src >= insert_pos {
                    out_dists[out_offset + dst] = out_dists[out_offset + src];
                    out_indices[out_offset + dst] = out_indices[out_offset + src];
                }
            }

            // Insert
            out_dists[out_offset + insert_pos] = dist;
            out_indices[out_offset + insert_pos] = chunk_offset + i;
        }
    }
}

/// Merge two sorted top-k lists into one
///
/// Standard sorted merge, taking first k elements.
///
/// ### Params
/// * `dists_a`, `indices_a` - First top-k list [n_queries, k]
/// * `dists_b`, `indices_b` - Second top-k list [n_queries, k]
/// * `out_dists`, `out_indices` - Merged result [n_queries, k]
#[cube(launch_unchecked)]
pub fn merge_topk<F: Float>(
    dists_a: &Tensor<F>,
    indices_a: &Tensor<u32>,
    dists_b: &Tensor<F>,
    indices_b: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
) {
    let query_idx = ABSOLUTE_POS_X;
    let k = dists_a.shape(1);

    if query_idx >= dists_a.shape(0) {
        terminate!();
    }

    let offset_a = query_idx * dists_a.stride(0);
    let offset_b = query_idx * dists_b.stride(0);
    let offset_out = query_idx * out_dists.stride(0);

    let mut ptr_a: u32 = 0;
    let mut ptr_b: u32 = 0;

    for out_idx in 0..k {
        let a_valid = ptr_a < k;
        let b_valid = ptr_b < k;

        // Take from A if: A valid AND (B exhausted OR A <= B)
        let dist_a = dists_a[offset_a + ptr_a];
        let dist_b = dists_b[offset_b + ptr_b];
        let take_a = a_valid && (!b_valid || dist_a <= dist_b);

        if take_a {
            out_dists[offset_out + out_idx] = dist_a;
            out_indices[offset_out + out_idx] = indices_a[offset_a + ptr_a];
            ptr_a += 1;
        } else {
            out_dists[offset_out + out_idx] = dist_b;
            out_indices[offset_out + out_idx] = indices_b[offset_b + ptr_b];
            ptr_b += 1;
        }
    }
}

/// Initialise top-k tensors with MAX distance values
#[cube(launch_unchecked)]
pub fn init_topk<F: Float>(dists: &mut Tensor<F>, indices: &mut Tensor<u32>) {
    let query_idx = ABSOLUTE_POS_Y;
    let k_idx = ABSOLUTE_POS_X;
    let k = dists.shape(1);

    if query_idx >= dists.shape(0) || k_idx >= k {
        terminate!();
    }

    let offset = query_idx * dists.stride(0) + k_idx;
    dists[offset] = F::new(f32::MAX);
    indices[offset] = 0u32;
}

/////////////////////////////////////////
// Offset-based kernels for IVF index  //
/////////////////////////////////////////

/// Compute squared Euclidean distances with DB offset
///
/// For IVF: DB tensor contains all vectors reorganised by cluster.
/// Offset and count define the cluster slice to process.
///
/// ### Params
///
/// * `query_vectors` - Query vectors [n_queries, dim / LINE_SIZE] as Line<F>
/// * `db_vectors` - All DB vectors [n_total, dim / LINE_SIZE] as Line<F>
/// * `distances` - Output matrix [n_queries, db_count] of squared distances
/// * `db_offset` - Start index in db_vectors (cluster_offsets[cluster_idx])
/// * `db_count` - Number of vectors in this cluster
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` → local database vector index (0..db_count)
/// * `ABSOLUTE_POS_Y` → query vector index
#[cube(launch_unchecked)]
pub fn euclidean_distances_gpu_offset<F: Float>(
    query_vectors: &Tensor<Line<F>>,
    db_vectors: &Tensor<Line<F>>,
    distances: &mut Tensor<F>,
    db_offset: u32,
    db_count: u32,
) {
    let query_idx = ABSOLUTE_POS_Y;
    let local_db_idx = ABSOLUTE_POS_X;

    if query_idx < query_vectors.shape(0) && local_db_idx < db_count {
        let db_idx = db_offset + local_db_idx;
        let dim_lines = query_vectors.shape(1);

        let mut sum = F::new(0.0);

        for i in 0..dim_lines {
            let q_line = query_vectors[query_idx * query_vectors.stride(0) + i];
            let d_line = db_vectors[db_idx * db_vectors.stride(0) + i];
            let diff = q_line - d_line;
            let sq = diff * diff;

            sum += sq[0];
            sum += sq[1];
            sum += sq[2];
            sum += sq[3];
        }

        distances[query_idx * distances.stride(0) + local_db_idx] = sum;
    }
}

/// Compute cosine distances with DB offset
///
/// For IVF: DB tensor and norms contain all vectors reorganised by cluster.
/// Offset and count define the cluster slice to process.
///
/// ### Params
///
/// * `query_vectors` - Query vectors [n_queries, dim / LINE_SIZE] as Line<F>
/// * `db_vectors` - All DB vectors [n_total, dim / LINE_SIZE] as Line<F>
/// * `query_norms` - Pre-computed L2 norms [n_queries]
/// * `db_norms` - All DB norms [n_total]
/// * `distances` - Output matrix [n_queries, db_count] of cosine distances
/// * `db_offset` - Start index in db_vectors
/// * `db_count` - Number of vectors in this cluster
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` → local database vector index (0..db_count)
/// * `ABSOLUTE_POS_Y` → query vector index
#[cube(launch_unchecked)]
pub fn cosine_distances_gpu_offset<F: Float>(
    query_vectors: &Tensor<Line<F>>,
    db_vectors: &Tensor<Line<F>>,
    query_norms: &Tensor<F>,
    db_norms: &Tensor<F>,
    distances: &mut Tensor<F>,
    db_offset: u32,
    db_count: u32,
) {
    let query_idx = ABSOLUTE_POS_Y;
    let local_db_idx = ABSOLUTE_POS_X;

    if query_idx < query_vectors.shape(0) && local_db_idx < db_count {
        let db_idx = db_offset + local_db_idx;
        let dim_lines = query_vectors.shape(1);

        let mut dot = F::new(0.0);

        for i in 0..dim_lines {
            let q_line = query_vectors[query_idx * query_vectors.stride(0) + i];
            let d_line = db_vectors[db_idx * db_vectors.stride(0) + i];
            let prod = q_line * d_line;

            dot += prod[0];
            dot += prod[1];
            dot += prod[2];
            dot += prod[3];
        }

        let q_norm = query_norms[query_idx];
        let d_norm = db_norms[db_offset + local_db_idx];

        distances[query_idx * distances.stride(0) + local_db_idx] =
            F::new(1.0) - (dot / (q_norm * d_norm));
    }
}

////////////////////
// Main functions //
////////////////////

/// Structure to store given BatchData
///
/// ### Fields
///
/// * `data` - Slice of the flattened data
/// * `norm` - Slice of the normalised data
/// * `n` - Number of data points
pub struct BatchData<'a, T> {
    pub data: &'a [T],
    pub norm: &'a [T],
    pub n: usize,
}

impl<'a, T> BatchData<'a, T> {
    /// Generate a new instance of BatchData
    ///
    /// ### Params
    ///
    /// * `data` - Slice of the flattened data
    /// * `norm` - Normalised values of the flattened data
    /// * `n` - Number of samples in the data
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn new(data: &'a [T], norm: &'a [T], n: usize) -> Self {
        Self { data, norm, n }
    }
}

/// Run batch queries on the GPU
///
/// ### Params
///
/// * `k` - Number of neighbours to return
/// * `query_data` - The `BatchData` structure for the query data.
/// * `db_data` - The `BatchData` structuref for the DB data
/// * `dim` - The dimensions of the data
/// * `metric` - The chosen distance metric
/// * `device` - The runtime device
///
/// ### Returns
///
/// The (Vec<indices>, Vec<dist>) for the batch.
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
    let vec_size = LINE_SIZE as u8;
    let dim_vectorized = dim / LINE_SIZE as usize;

    let n_query_chunks = query_data.n.div_ceil(QUERY_CHUNK_SIZE);
    let n_db_chunks = db_data.n.div_ceil(DB_CHUNK_SIZE);

    let mut all_indices = Vec::with_capacity(query_data.n);
    let mut all_distances = Vec::with_capacity(query_data.n);

    for query_chunk_idx in 0..n_query_chunks {
        if verbose && query_chunk_idx % 10 == 0 {
            println!(
                "Processed {} query chunks out of {}",
                query_chunk_idx, n_query_chunks
            );
        }

        let query_start = query_chunk_idx * QUERY_CHUNK_SIZE;
        let query_end = (query_start + QUERY_CHUNK_SIZE).min(query_data.n);
        let current_query_chunk_size = query_end - query_start;

        let query_chunk_data = &query_data.data[query_start * dim..query_end * dim];
        let query_gpu = GpuTensor::<R, T>::from_slice(
            query_chunk_data,
            vec![current_query_chunk_size, dim_vectorized],
            &client,
        );

        let query_norms_gpu = if *metric == Dist::Cosine {
            Some(GpuTensor::<R, T>::from_slice(
                &query_data.norm[query_start..query_end],
                vec![current_query_chunk_size],
                &client,
            ))
        } else {
            None
        };

        // Allocate running top-k on GPU
        let mut topk_dists = GpuTensor::<R, T>::empty(vec![current_query_chunk_size, k], &client);
        let mut topk_indices =
            GpuTensor::<R, u32>::empty(vec![current_query_chunk_size, k], &client);

        // Initialise to MAX
        let init_grid_x = (k as u32).div_ceil(WORKGROUP_SIZE_X);
        let init_grid_y = (current_query_chunk_size as u32).div_ceil(WORKGROUP_SIZE_Y);
        unsafe {
            init_topk::launch_unchecked::<T, R>(
                &client,
                CubeCount::Static(init_grid_x, init_grid_y, 1),
                CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                topk_dists.clone().into_tensor_arg(1),
                topk_indices.clone().into_tensor_arg(1),
            );
        }

        // Pre-allocate distance buffer for largest possible chunk
        let max_db_chunk = DB_CHUNK_SIZE.min(db_data.n);
        let distances_gpu =
            GpuTensor::<R, T>::empty(vec![current_query_chunk_size, max_db_chunk], &client);

        // Temp buffers for merge (double-buffer pattern)
        let chunk_topk_dists = GpuTensor::<R, T>::empty(vec![current_query_chunk_size, k], &client);
        let chunk_topk_indices =
            GpuTensor::<R, u32>::empty(vec![current_query_chunk_size, k], &client);
        let merged_dists = GpuTensor::<R, T>::empty(vec![current_query_chunk_size, k], &client);
        let merged_indices = GpuTensor::<R, u32>::empty(vec![current_query_chunk_size, k], &client);

        for db_chunk_idx in 0..n_db_chunks {
            let db_start = db_chunk_idx * DB_CHUNK_SIZE;
            let db_end = (db_start + DB_CHUNK_SIZE).min(db_data.n);
            let current_db_chunk_size = db_end - db_start;

            let grid_x = (current_db_chunk_size as u32).div_ceil(WORKGROUP_SIZE_X);
            let grid_y = (current_query_chunk_size as u32).div_ceil(WORKGROUP_SIZE_Y);

            let db_chunk_data = &db_data.data[db_start * dim..db_end * dim];
            let db_gpu = GpuTensor::<R, T>::from_slice(
                db_chunk_data,
                vec![current_db_chunk_size, dim_vectorized],
                &client,
            );

            // 1. Compute distances
            match *metric {
                Dist::Euclidean => unsafe {
                    euclidean_distances_gpu_chunk::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(grid_x, grid_y, 1),
                        CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                        query_gpu.clone().into_tensor_arg(vec_size),
                        db_gpu.into_tensor_arg(vec_size),
                        distances_gpu.clone().into_tensor_arg(1),
                    );
                },
                Dist::Cosine => {
                    let db_norms_gpu = GpuTensor::<R, T>::from_slice(
                        &db_data.norm[db_start..db_end],
                        vec![current_db_chunk_size],
                        &client,
                    );
                    unsafe {
                        cosine_distances_gpu_chunk::launch_unchecked::<T, R>(
                            &client,
                            CubeCount::Static(grid_x, grid_y, 1),
                            CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                            query_gpu.clone().into_tensor_arg(vec_size),
                            db_gpu.into_tensor_arg(vec_size),
                            query_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(1),
                            db_norms_gpu.into_tensor_arg(1),
                            distances_gpu.clone().into_tensor_arg(1),
                        );
                    }
                }
            }

            // 2. Init chunk top-k
            unsafe {
                init_topk::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(init_grid_x, init_grid_y, 1),
                    CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                    chunk_topk_dists.clone().into_tensor_arg(1),
                    chunk_topk_indices.clone().into_tensor_arg(1),
                );
            }

            // 3. Extract top-k from this chunk's distances
            let extract_grid = (current_query_chunk_size as u32).div_ceil(WORKGROUP_SIZE_X);
            unsafe {
                extract_topk::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(extract_grid, 1, 1),
                    CubeDim::new(WORKGROUP_SIZE_X, 1, 1),
                    distances_gpu.clone().into_tensor_arg(1),
                    chunk_topk_dists.clone().into_tensor_arg(1),
                    chunk_topk_indices.clone().into_tensor_arg(1),
                    cubecl::frontend::ScalarArg {
                        elem: db_start as u32,
                    },
                );
            }

            // 4. Merge with running top-k
            unsafe {
                merge_topk::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(extract_grid, 1, 1),
                    CubeDim::new(WORKGROUP_SIZE_X, 1, 1),
                    topk_dists.clone().into_tensor_arg(1),
                    topk_indices.clone().into_tensor_arg(1),
                    chunk_topk_dists.clone().into_tensor_arg(1),
                    chunk_topk_indices.clone().into_tensor_arg(1),
                    merged_dists.clone().into_tensor_arg(1),
                    merged_indices.clone().into_tensor_arg(1),
                );
            }

            // Swap: merged becomes new running top-k
            topk_dists = merged_dists.clone();
            topk_indices = merged_indices.clone();
        }

        // Read final results - only k items per query!
        let final_dists = topk_dists.read(&client);
        let final_indices = topk_indices.read(&client);

        for q in 0..current_query_chunk_size {
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
    use cubecl::cpu::CpuDevice;
    use cubecl::cpu::CpuRuntime;

    #[test]
    fn test_euclidean_batch_query() {
        let device = CpuDevice;

        // 4 query vectors, 8 db vectors, 4 dimensions (divisible by LINE_SIZE=4)
        let query_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        let db_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // matches query 0
            0.0, 1.0, 0.0, 0.0, // matches query 1
            0.0, 0.0, 1.0, 0.0, // matches query 2
            0.0, 0.0, 0.0, 1.0, // matches query 3
            0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5,
        ];

        let query_batch = BatchData::new(&query_data, &[], 4);
        let db_batch = BatchData::new(&db_data, &[], 8);

        let (indices, distances) = query_batch_gpu::<f32, CpuRuntime>(
            3,
            &query_batch,
            &db_batch,
            4,
            &Dist::Euclidean,
            device,
            false,
        );

        assert_eq!(indices.len(), 4);
        assert_eq!(distances.len(), 4);

        // First query should find perfect match at index 0
        assert_eq!(indices[0][0], 0);
        assert!(distances[0][0] < 0.01);
    }

    #[test]
    fn test_cosine_batch_query() {
        let device = CpuDevice;

        let query_data: Vec<f32> = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        let db_data: Vec<f32> = vec![
            2.0, 2.0, 0.0, 0.0, // parallel to query 0
            0.0, 0.0, 3.0, 3.0, // parallel to query 1
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ];

        // Pre-compute norms
        let query_norms: Vec<f32> = vec![(2.0_f32).sqrt(), (2.0_f32).sqrt()];

        let db_norms: Vec<f32> = vec![(8.0_f32).sqrt(), (18.0_f32).sqrt(), 1.0, 1.0];

        let query_batch = BatchData::new(&query_data, &query_norms, 2);
        let db_batch = BatchData::new(&db_data, &db_norms, 4);

        let (indices, distances) = query_batch_gpu::<f32, CpuRuntime>(
            2,
            &query_batch,
            &db_batch,
            4,
            &Dist::Cosine,
            device,
            false,
        );

        assert_eq!(indices.len(), 2);

        // First query should find parallel vector at index 0 as closest
        assert_eq!(indices[0][0], 0);
        assert!(distances[0][0] < 0.01); // cosine distance ≈ 0 for parallel vectors
    }
}
