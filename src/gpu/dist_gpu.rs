use cubecl::prelude::*;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;

use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::utils::dist::Dist;
use crate::utils::heap_structs::*;

/////////////
// Helpers //
/////////////

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
                "Processed {} query chunks out of {} on the GPU.",
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

        let mut heaps: Vec<BinaryHeap<(OrderedFloat<T>, usize)>> =
            vec![BinaryHeap::with_capacity(k + 1); current_query_chunk_size];

        // track pending GPU work
        let mut pending_distances: Option<(GpuTensor<R, T>, usize, usize)> = None;

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

            let distances_gpu = GpuTensor::<R, T>::empty(
                vec![current_query_chunk_size, current_db_chunk_size],
                &client,
            );

            // launch GPU kernel
            match *metric {
                Dist::Euclidean => unsafe {
                    euclidean_distances_gpu_chunk::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(grid_x, grid_y, 1),
                        CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                        query_gpu.clone().into_tensor_arg(vec_size),
                        db_gpu.into_tensor_arg(vec_size),
                        distances_gpu.into_tensor_arg(1),
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
                            distances_gpu.into_tensor_arg(1),
                        );
                    }
                }
            }

            // process PREVIOUS chunk's results while GPU works on current chunk
            if let Some((prev_distances_gpu, prev_db_start, prev_db_chunk_size)) =
                pending_distances.take()
            {
                let chunk_distances = prev_distances_gpu.read(&client);

                heaps = heaps
                    .into_par_iter()
                    .enumerate()
                    .map(|(q, mut heap)| {
                        let row_start = q * prev_db_chunk_size;

                        for i in 0..prev_db_chunk_size {
                            let dist = chunk_distances[row_start + i];
                            let global_idx = prev_db_start + i;

                            if heap.len() < k {
                                heap.push((OrderedFloat(dist), global_idx));
                            } else if dist < heap.peek().unwrap().0 .0 {
                                heap.pop();
                                heap.push((OrderedFloat(dist), global_idx));
                            }
                        }
                        heap
                    })
                    .collect();
            }

            // store current chunk for next iteration
            pending_distances = Some((distances_gpu, db_start, current_db_chunk_size));
        }

        // process final chunk
        if let Some((prev_distances_gpu, prev_db_start, prev_db_chunk_size)) = pending_distances {
            let chunk_distances = prev_distances_gpu.read(&client);

            heaps = heaps
                .into_par_iter()
                .enumerate()
                .map(|(q, mut heap)| {
                    let row_start = q * prev_db_chunk_size;

                    for i in 0..prev_db_chunk_size {
                        let dist = chunk_distances[row_start + i];
                        let global_idx = prev_db_start + i;

                        if heap.len() < k {
                            heap.push((OrderedFloat(dist), global_idx));
                        } else if dist < heap.peek().unwrap().0 .0 {
                            heap.pop();
                            heap.push((OrderedFloat(dist), global_idx));
                        }
                    }
                    heap
                })
                .collect();
        }

        let chunk_results: Vec<_> = heaps
            .into_par_iter()
            .map(|heap| {
                let mut results: Vec<_> = heap.into_iter().collect();
                results.sort_unstable_by_key(|&(dist, _)| dist);

                let (distances, indices): (Vec<_>, Vec<_>) = results
                    .into_iter()
                    .map(|(OrderedFloat(dist), idx)| (dist, idx))
                    .unzip();

                (indices, distances)
            })
            .collect();

        for (indices, distances) in chunk_results {
            all_indices.push(indices);
            all_distances.push(distances);
        }
    }

    (all_indices, all_distances)
}
