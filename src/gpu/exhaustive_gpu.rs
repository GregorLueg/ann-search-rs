use cubecl::prelude::*;
use faer::MatRef;
use num_traits::Float;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;

use crate::gpu::dist_gpu::*;
use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::utils::dist::Dist;
use crate::utils::heap_structs::OrderedFloat;

////////////////////////
// ExhaustiveIndexGpu //
////////////////////////

/// Exhaustive (brute-force) nearest neighbour index (on GPU)
///
/// ### Fields
///
/// * `vectors_flat` - Original vector data for distance calculations. Flattened
///   for better cache locality
/// * `norms` - Normalised pre-calculated values per sample if distance is set to
///   Cosine
/// * `dim` - Embedding dimensions
/// * `n` - Number of samples
/// * `dist_metric` - The type of distance the index is designed for
/// * `device` - The cubecl runtime
pub struct ExhaustiveIndexGpu<T: Float, R: Runtime> {
    vectors_flat: Vec<T>,
    norms: Vec<T>,
    dim: usize,
    n: usize,
    metric: Dist,
    device: R::Device,
}

impl<T, R> ExhaustiveIndexGpu<T, R>
where
    R: Runtime,
    T: Float + Sum + cubecl::frontend::Float + cubecl::CubeElement,
{
    /// Generate a new exhaustive index (on the GPU)
    ///
    /// ### Params
    ///
    /// * `data` - The data for which to generate the index. Samples x features
    /// * `metric` - Which distance metric the index shall be generated for.
    /// * `device` - The runtime device for the cubecl
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive index (on GPU)
    pub fn new(data: MatRef<T>, metric: Dist, device: R::Device) -> Self {
        let n = data.nrows();
        let dim = data.ncols();

        // assertion for the manual unrolling
        assert!(
            dim.is_multiple_of(LINE_SIZE as usize),
            "Dimension {} must be divisible by LINE_SIZE {}",
            dim,
            LINE_SIZE
        );

        let mut vectors_flat = Vec::with_capacity(n * dim);
        for i in 0..n {
            vectors_flat.extend(data.row(i).iter().copied());
        }

        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    vectors_flat[start..start + dim]
                        .iter()
                        .map(|&x| x * x)
                        .sum::<T>()
                        .sqrt()
                })
                .collect()
        } else {
            Vec::new()
        };

        Self {
            vectors_flat,
            norms,
            dim,
            n,
            metric,
            device,
        }
    }

    /// Query the exhaustive index
    ///
    /// ### Params
    ///
    /// * `query_mat` - The samples x features matrix to query. n(features)
    ///   needs to be divisible by 4!
    /// * `k` - Number of neighbours to return
    ///
    /// ### Returns
    ///
    /// /// A tuple of `(Vec<indices>, Vec<distances>)`
    pub fn query_batch(&self, query_mat: MatRef<T>, k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        assert!(
            self.dim.is_multiple_of(LINE_SIZE as usize),
            "Dimension {} must be divisible by LINE_SIZE {}",
            self.dim,
            LINE_SIZE
        );

        let client = R::client(&self.device);
        let n_queries = query_mat.nrows();

        let mut query_vectors_flat = Vec::with_capacity(n_queries * self.dim);
        for i in 0..n_queries {
            query_vectors_flat.extend(query_mat.row(i).iter().copied());
        }

        let query_norms = if self.metric == Dist::Cosine {
            (0..n_queries)
                .into_par_iter()
                .map(|i| {
                    let start = i * self.dim;
                    query_vectors_flat[start..start + self.dim]
                        .iter()
                        .map(|&x| x * x)
                        .sum::<T>()
                        .sqrt()
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let vec_size = LINE_SIZE as u8;
        let dim_vectorized = self.dim / LINE_SIZE as usize;

        let n_query_chunks = n_queries.div_ceil(QUERY_CHUNK_SIZE);
        let n_db_chunks = self.n.div_ceil(DB_CHUNK_SIZE);

        let mut all_indices = Vec::with_capacity(n_queries);
        let mut all_distances = Vec::with_capacity(n_queries);

        for query_chunk_idx in 0..n_query_chunks {
            let query_start = query_chunk_idx * QUERY_CHUNK_SIZE;
            let query_end = (query_start + QUERY_CHUNK_SIZE).min(n_queries);
            let current_query_chunk_size = query_end - query_start;

            let query_chunk_data =
                &query_vectors_flat[query_start * self.dim..query_end * self.dim];
            let query_gpu = GpuTensor::<R, T>::from_slice(
                query_chunk_data,
                vec![current_query_chunk_size, dim_vectorized],
                &client,
            );

            let query_norms_gpu = if self.metric == Dist::Cosine {
                Some(GpuTensor::<R, T>::from_slice(
                    &query_norms[query_start..query_end],
                    vec![current_query_chunk_size],
                    &client,
                ))
            } else {
                None
            };

            let mut heaps: Vec<BinaryHeap<(OrderedFloat<f32>, usize)>> =
                vec![BinaryHeap::with_capacity(k + 1); current_query_chunk_size];

            // track pending GPU work
            let mut pending_distances: Option<(GpuTensor<R, f32>, usize, usize)> = None;

            for db_chunk_idx in 0..n_db_chunks {
                let db_start = db_chunk_idx * DB_CHUNK_SIZE;
                let db_end = (db_start + DB_CHUNK_SIZE).min(self.n);
                let current_db_chunk_size = db_end - db_start;
                let grid_x = (current_db_chunk_size as u32).div_ceil(WORKGROUP_SIZE_X);
                let grid_y = (current_query_chunk_size as u32).div_ceil(WORKGROUP_SIZE_Y);

                let db_chunk_data = &self.vectors_flat[db_start * self.dim..db_end * self.dim];
                let db_gpu = GpuTensor::<R, T>::from_slice(
                    db_chunk_data,
                    vec![current_db_chunk_size, dim_vectorized],
                    &client,
                );

                let distances_gpu = GpuTensor::<R, f32>::empty(
                    vec![current_query_chunk_size, current_db_chunk_size],
                    &client,
                );

                // launch GPU kernel
                match self.metric {
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
                            &self.norms[db_start..db_end],
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
            if let Some((prev_distances_gpu, prev_db_start, prev_db_chunk_size)) = pending_distances
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
}
