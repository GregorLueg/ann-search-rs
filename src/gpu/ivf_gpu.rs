use cubecl::prelude::*;
use faer::MatRef;
use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;
use thousands::*;

use crate::gpu::dist_gpu::*;
use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::prelude::*;
use crate::utils::dist::Dist;
use crate::utils::heap_structs::*;
use crate::utils::ivf_utils::*;
use crate::utils::*;

/// To not explode memory VRAM memory
const IVF_GPU_QUERY_BATCH_SIZE: usize = 100_000;
/// Target max size for the candidate buffer
const TARGET_BUFFER_MB: usize = 1500;

/// Batched IVF index with GPU acceleration
///
/// Designed for large-scale batch queries (100k-1M queries) against large
/// databases (1M-10M vectors). Minimises kernel launches by batching operations
/// and processing all queries against each cluster in a single kernel.
///
/// ### Architecture
///
/// - Database vectors reorganised by cluster for contiguous access
/// - All vectors and norms kept on GPU for fast access
/// - Centroids kept on GPU for fast probe selection
/// - Query pipeline:
///   1. Compute all query-centroid distances (1 kernel)
///   2. Select top nprobe clusters per query (CPU)
///   3. For each cluster: batch all queries probing it into one kernel
///
/// ### Type Parameters
///
/// * `T` - Float type (f32 or f64)
/// * `R` - CubeCL runtime
///
/// ### Fields
///
/// * `vectors_gpu` - All vectors reorganised by cluster, resident on GPU
/// * `norms_gpu` - All norms reorganised by cluster, resident on GPU (Cosine only)
/// * `original_indices` - Maps reorganised position -> original index
/// * `cluster_offsets` - CSR offsets
/// * `centroids_gpu` - Centroids kept on the GPU
/// * `centroid_norms_gpu` - Centroid norms kept on the GPU
/// * `dim` - Dimensionality of the index
/// * `n` - Number of samples in the index
/// * `nlist` - Number of lists in the index
/// * `metric` - Distance metric used
/// * `device` - Device runtime for the GPU work
pub struct IvfIndexGpu<T: Float + cubecl::frontend::Float + cubecl::CubeElement, R: Runtime> {
    vectors_gpu: GpuTensor<R, T>,
    norms_gpu: Option<GpuTensor<R, T>>,
    original_indices: Vec<usize>,
    cluster_offsets: Vec<usize>,
    centroids_gpu: GpuTensor<R, T>,
    centroid_norms_gpu: Option<GpuTensor<R, T>>,
    dim: usize,
    n: usize,
    nlist: usize,
    metric: Dist,
    device: R::Device,
}

impl<T, R> IvfIndexGpu<T, R>
where
    R: Runtime,
    T: Float
        + Sum
        + cubecl::frontend::Float
        + cubecl::CubeElement
        + FromPrimitive
        + Send
        + Sync
        + SimdDistance,
{
    /// Build a batched IVF index
    ///
    /// ### Params
    ///
    /// * `data` - Database vectors [n, dim]
    /// * `metric` - Distance metric
    /// * `nlist` - Number of clusters (defaults to √n)
    /// * `max_iters` - K-means iterations (defaults to 30)
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    /// * `device` - GPU device
    pub fn build(
        data: MatRef<T>,
        metric: Dist,
        nlist: Option<usize>,
        max_iters: Option<usize>,
        seed: usize,
        verbose: bool,
        device: R::Device,
    ) -> Self {
        let (vectors_flat, n, dim) = matrix_to_flat(data);

        assert!(
            dim.is_multiple_of(LINE_SIZE as usize),
            "Dimension {} must be divisible by LINE_SIZE {}",
            dim,
            LINE_SIZE
        );

        let dim_vectorized = dim / LINE_SIZE as usize;
        let max_iters = max_iters.unwrap_or(30);
        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        // Subsample for training if large
        let (training_data, n_train) = if n > 500_000 {
            if verbose {
                println!("  Sampling 250k vectors for training");
            }
            let (data, _) = sample_vectors(&vectors_flat, dim, n, 250_000, seed);
            (data, 250_000)
        } else {
            (vectors_flat.clone(), n)
        };

        if verbose {
            println!(
                "  Building IVF-GPU-Batched index with {} clusters for {} vectors",
                nlist,
                n.separate_with_underscores()
            );
        }

        // train centroids
        let centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            max_iters,
            seed,
            verbose,
        );

        // assign vectors to clusters
        let data_norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    vectors_flat[i * dim..(i + 1) * dim]
                        .iter()
                        .map(|&x| x * x)
                        .sum::<T>()
                        .sqrt()
                })
                .collect()
        } else {
            vec![T::one(); n]
        };

        let centroid_norms = if metric == Dist::Cosine {
            (0..nlist)
                .map(|i| {
                    centroids[i * dim..(i + 1) * dim]
                        .iter()
                        .map(|&x| x * x)
                        .sum::<T>()
                        .sqrt()
                })
                .collect()
        } else {
            vec![T::one(); nlist]
        };

        let assignments = assign_all_parallel(
            &vectors_flat,
            &data_norms,
            dim,
            n,
            &centroids,
            &centroid_norms,
            nlist,
            &metric,
        );

        if verbose {
            print_cluster_summary(&assignments, nlist);
        }

        // reorganise vectors by cluster
        let (vectors_by_cluster, original_indices, cluster_offsets, norms_by_cluster) =
            reorganise_by_cluster(&vectors_flat, dim, n, &assignments, nlist, &metric);

        if verbose {
            println!("  Uploading all vectors to GPU");
        }

        let client = R::client(&device);

        // Upload all vectors to GPU (the key optimisation)
        let vectors_gpu =
            GpuTensor::<R, T>::from_slice(&vectors_by_cluster, vec![n, dim_vectorized], &client);

        let norms_gpu = if metric == Dist::Cosine {
            Some(GpuTensor::<R, T>::from_slice(
                &norms_by_cluster,
                vec![n],
                &client,
            ))
        } else {
            None
        };

        let centroids_gpu =
            GpuTensor::<R, T>::from_slice(&centroids, vec![nlist, dim_vectorized], &client);

        let centroid_norms_gpu = if metric == Dist::Cosine {
            Some(GpuTensor::<R, T>::from_slice(
                &centroid_norms,
                vec![nlist],
                &client,
            ))
        } else {
            None
        };

        if verbose {
            println!("  Index ready");
        }

        Self {
            vectors_gpu,
            norms_gpu,
            original_indices,
            cluster_offsets,
            centroids_gpu,
            centroid_norms_gpu,
            dim,
            n,
            nlist,
            metric,
            device,
        }
    }

    /// Internal helper for querying
    ///
    /// ### Params
    ///
    /// * `queries_flat` - The query vector flattened
    /// * `n_queries` - The number of queries
    /// * `dim_query` - The dimensions
    /// * `k` - Number of neighbours per query
    /// * `nprobe` - Number of clusters to search (defaults to √nlist)
    /// * `nquery` - Number of vectors to load in one go into the GPU. If not
    ///   provided, it will default to `100_000`.
    /// * `verbose` - Controls the verbosity of the function
    ///
    /// ### Returns
    ///
    /// Tuple of `(Vec<indices>, Vec<dist>)` for the queries.
    #[allow(clippy::too_many_arguments)]
    fn query_internal(
        &self,
        queries_flat: &[T],
        n_queries: usize,
        dim_query: usize,
        k: usize,
        nprobe: Option<usize>,
        nquery: Option<usize>,
        client: &ComputeClient<<R as Runtime>::Server>,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Vec<Vec<T>>) {
        assert_eq!(
            dim_query, self.dim,
            "Query dimension {} != index dimension {}",
            dim_query, self.dim
        );

        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let nquery = nquery.unwrap_or(IVF_GPU_QUERY_BATCH_SIZE);
        if verbose {
            println!(
                "Using nquery batch size: {}",
                nquery.separate_with_underscores()
            );
        }

        let k = k.min(self.n);

        let n_batches = n_queries.div_ceil(nquery);

        if n_batches == 1 {
            // Small enough to process in one go
            return self.query_batch_internal(queries_flat, n_queries, k, nprobe, verbose, client);
        }

        let mut all_indices = Vec::with_capacity(n_queries);
        let mut all_distances = Vec::with_capacity(n_queries);

        for batch_idx in 0..n_batches {
            if verbose {
                println!(
                    "Processing query batch {}/{} ({} queries per batch)",
                    batch_idx + 1,
                    n_batches,
                    nquery.separate_with_underscores()
                );
            }

            let batch_start = batch_idx * nquery;
            let batch_end = (batch_start + nquery).min(n_queries);
            let batch_size = batch_end - batch_start;

            let batch_queries = &queries_flat[batch_start * self.dim..batch_end * self.dim];

            let (batch_indices, batch_dists) =
                self.query_batch_internal(batch_queries, batch_size, k, nprobe, verbose, client);

            all_indices.extend(batch_indices);
            all_distances.extend(batch_dists);
        }

        (all_indices, all_distances)
    }

    /// Query the index with a batch of vectors
    ///
    /// ### Params
    ///
    /// * `query_mat` - Query vectors [n_queries, dim]
    /// * `k` - Number of neighbours per query
    /// * `nprobe` - Number of clusters to search (defaults to √nlist)
    /// * `nquery` - Number of vectors to load in one go into the GPU. If not
    ///   provided, it will default to `100_000`.
    /// * `verbose` - Controls verbosity of the function.
    ///
    /// ### Returns
    ///
    /// Tuple of `(Vec<indices>, Vec<dist>)` for the queries.
    pub fn query_batch(
        &self,
        query_mat: MatRef<T>,
        k: usize,
        nprobe: Option<usize>,
        nquery: Option<usize>,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Vec<Vec<T>>) {
        let (queries_flat, n_queries, dim_query) = matrix_to_flat(query_mat);
        let client: ComputeClient<<R as Runtime>::Server> = R::client(&self.device);

        let nprobe_val = nprobe.unwrap_or(((self.nlist as f32).sqrt() as usize).max(1));

        let batch_size = nquery.unwrap_or_else(|| self.calculate_safe_batch_size(nprobe_val));

        let (indices, dist) = self.query_internal(
            &queries_flat,
            n_queries,
            dim_query,
            k,
            nprobe,
            Some(batch_size),
            &client,
            verbose,
        );

        client.memory_cleanup();
        (indices, dist)
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete
    /// kNN graph.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `return_dist` - Whether to return distances
    /// * `nprobe` - Number of centroids to check.
    /// * `nquery` - Number of queries to load into the GPU.
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)` where each row corresponds
    /// to a vector in the index
    pub fn generate_knn(
        &self,
        k: usize,
        nprobe: Option<usize>,
        nquery: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        let client: ComputeClient<<R as Runtime>::Server> = R::client(&self.device);

        // 1. Determine parameters
        let nprobe = nprobe.unwrap_or(((self.nlist as f32).sqrt() as usize).max(1));

        // 2. Auto-calculate batch size if not provided
        // This prevents the BufferTooBig panic
        let batch_size = nquery.unwrap_or_else(|| {
            let safe = self.calculate_safe_batch_size(nprobe);
            if verbose {
                println!("  Auto-tuned batch size to {} (based on density)", safe);
            }
            safe
        });

        // 3. Read vectors back from GPU (Needed for query input)
        if verbose {
            println!("  Reading vectors from GPU for self-query...");
        }
        let vectors_by_cluster = self.vectors_gpu.clone().read(&client);

        // 4. Run batched query
        let (indices_reorg, dist_reorg) = self.query_internal(
            &vectors_by_cluster,
            self.n,
            self.dim,
            k,
            Some(nprobe),
            Some(batch_size), // Pass the safe batch size
            &client,
            verbose,
        );

        client.memory_cleanup();

        // 5. Reorder results
        if verbose {
            println!("  Reordering results...");
        }

        let mut indices = vec![Vec::new(); self.n];
        let mut dist = if return_dist {
            vec![Vec::new(); self.n]
        } else {
            Vec::new()
        };

        for (reorg_idx, orig_idx) in self.original_indices.iter().enumerate() {
            indices[*orig_idx] = indices_reorg[reorg_idx].clone();
            if return_dist {
                dist[*orig_idx] = dist_reorg[reorg_idx].clone();
            }
        }

        if return_dist {
            (indices, Some(dist))
        } else {
            (indices, None)
        }
    }

    /// Returns memory usage
    ///
    /// Also returns the data stored on the GPU
    ///
    /// ### Returns
    ///
    /// `(RAM bytes, VRAM bytes)`
    pub fn memory_usage_bytes(&self) -> (usize, usize) {
        let ram = std::mem::size_of_val(self)
            + self.original_indices.capacity() * std::mem::size_of::<usize>()
            + self.cluster_offsets.capacity() * std::mem::size_of::<usize>();

        let vram = self.vectors_gpu.vram_bytes()
            + self.norms_gpu.as_ref().map_or(0, |t| t.vram_bytes())
            + self.centroids_gpu.vram_bytes()
            + self
                .centroid_norms_gpu
                .as_ref()
                .map_or(0, |t| t.vram_bytes());

        (ram, vram)
    }

    /// Process a single batch of queries against the index
    ///
    /// ### Params
    ///
    /// * `queries_flat` - The query vectors for this batch (flattened)
    /// * `n_queries` - Number of queries in this batch
    /// * `k` - Number of neighbours per query
    /// * `nprobe` - Number of clusters to search
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(Vec<indices>, Vec<dist>)` for this batch
    fn query_batch_internal(
        &self,
        queries_flat: &[T],
        n_queries: usize,
        k: usize,
        nprobe: usize,
        verbose: bool,
        client: &ComputeClient<<R as Runtime>::Server>,
    ) -> (Vec<Vec<usize>>, Vec<Vec<T>>) {
        let dim_vectorized = self.dim / LINE_SIZE as usize;
        let vec_size = LINE_SIZE as u8;

        let query_norms = if self.metric == Dist::Cosine {
            (0..n_queries)
                .into_par_iter()
                .map(|i| {
                    let start = i * self.dim;
                    queries_flat[start..start + self.dim]
                        .iter()
                        .map(|&x| x * x)
                        .sum::<T>()
                        .sqrt()
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        // upload ALL queries to GPU once
        let queries_gpu =
            GpuTensor::<R, T>::from_slice(queries_flat, vec![n_queries, dim_vectorized], client);

        let query_norms_gpu = if self.metric == Dist::Cosine {
            Some(GpuTensor::<R, T>::from_slice(
                &query_norms,
                vec![n_queries],
                client,
            ))
        } else {
            None
        };

        let centroid_dists_gpu = GpuTensor::<R, T>::empty(vec![n_queries, self.nlist], client);
        let grid_x = (self.nlist as u32).div_ceil(WORKGROUP_SIZE_X);
        let grid_y = (n_queries as u32).div_ceil(WORKGROUP_SIZE_Y);

        match self.metric {
            Dist::Euclidean => unsafe {
                euclidean_distances_gpu_chunk::launch_unchecked::<T, R>(
                    client,
                    CubeCount::Static(grid_x, grid_y, 1),
                    CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                    queries_gpu.clone().into_tensor_arg(vec_size),
                    self.centroids_gpu.clone().into_tensor_arg(vec_size),
                    centroid_dists_gpu.into_tensor_arg(1),
                );
            },
            Dist::Cosine => unsafe {
                cosine_distances_gpu_chunk::launch_unchecked::<T, R>(
                    client,
                    CubeCount::Static(grid_x, grid_y, 1),
                    CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                    queries_gpu.clone().into_tensor_arg(vec_size),
                    self.centroids_gpu.clone().into_tensor_arg(vec_size),
                    query_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(1),
                    self.centroid_norms_gpu
                        .as_ref()
                        .unwrap()
                        .clone()
                        .into_tensor_arg(1),
                    centroid_dists_gpu.into_tensor_arg(1),
                );
            },
        }

        // read centroid distances - one block here
        let centroid_dists = centroid_dists_gpu.read(client);

        // select top clusters (CPU)
        let probe_lists: Vec<Vec<usize>> = (0..n_queries)
            .into_par_iter()
            .map(|q| {
                let row_start = q * self.nlist;
                let mut cluster_dists: Vec<(T, usize)> = (0..self.nlist)
                    .map(|c| (centroid_dists[row_start + c], c))
                    .collect();
                // Unstable sort is faster and sufficient
                cluster_dists.sort_unstable_by(|a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                cluster_dists
                    .into_iter()
                    .take(nprobe)
                    .map(|(_, c)| c)
                    .collect()
            })
            .collect();

        // Invert: Cluster -> [Queries that probe it]
        let cluster_to_queries = invert_probe_lists(&probe_lists, self.nlist);

        let candidates_per_query: Vec<usize> = probe_lists
            .iter()
            .map(|clusters| {
                clusters
                    .iter()
                    .map(|&c| self.cluster_offsets[c + 1] - self.cluster_offsets[c])
                    .sum()
            })
            .collect();

        let max_candidates = *candidates_per_query.iter().max().unwrap_or(&0);

        if verbose {
            println!(
                "  Allocating candidate buffer: {} x {} floats",
                n_queries, max_candidates
            );
        }

        // Allocate the "Fire and Forget" buffers
        // We allocate based on the worst-case query, but most queries will just leave the tail empty
        let candidate_dists_gpu = GpuTensor::<R, T>::empty(vec![n_queries, max_candidates], client);
        let candidate_indices_gpu =
            GpuTensor::<R, u32>::empty(vec![n_queries, max_candidates], client);

        // track where each query is currently writing in the buffer
        let mut cpu_write_pointers = vec![0u32; n_queries];

        for cluster_idx in 0..self.nlist {
            let active_queries = &cluster_to_queries[cluster_idx];
            if active_queries.is_empty() {
                continue;
            }

            let cluster_start = self.cluster_offsets[cluster_idx];
            let cluster_count = self.cluster_offsets[cluster_idx + 1] - cluster_start;

            if cluster_count == 0 {
                continue;
            }

            // Prepare the "Schedule" for this cluster
            // We need to tell the GPU:
            // 1. Which queries are active? (active_indices)
            // 2. Where do they write? (write_offsets)

            let n_active = active_queries.len();
            let mut active_indices_vec = Vec::with_capacity(n_active);
            let mut write_offsets_vec = Vec::with_capacity(n_active);

            for &q_idx in active_queries {
                active_indices_vec.push(q_idx as u32);
                write_offsets_vec.push(cpu_write_pointers[q_idx]);

                // advance the pointer for next time
                cpu_write_pointers[q_idx] += cluster_count as u32;
            }

            // Upload the schedule
            let active_indices_gpu =
                GpuTensor::<R, u32>::from_slice(&active_indices_vec, vec![n_active], client);
            let write_offsets_gpu =
                GpuTensor::<R, u32>::from_slice(&write_offsets_vec, vec![n_active], client);

            let grid_x = (cluster_count as u32).div_ceil(WORKGROUP_SIZE_X);
            let grid_y = (n_active as u32).div_ceil(WORKGROUP_SIZE_Y);

            // Launch Kernel
            match self.metric {
                Dist::Euclidean => unsafe {
                    compute_candidates_euclidean::launch_unchecked::<T, R>(
                        client,
                        CubeCount::Static(grid_x, grid_y, 1),
                        CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                        queries_gpu.clone().into_tensor_arg(vec_size),
                        self.vectors_gpu.clone().into_tensor_arg(vec_size),
                        active_indices_gpu.into_tensor_arg(1),
                        write_offsets_gpu.into_tensor_arg(1),
                        candidate_dists_gpu.clone().into_tensor_arg(1),
                        candidate_indices_gpu.clone().into_tensor_arg(1),
                        ScalarArg {
                            elem: cluster_start as u32,
                        },
                        ScalarArg {
                            elem: cluster_count as u32,
                        },
                    );
                },
                Dist::Cosine => unsafe {
                    compute_candidates_cosine::launch_unchecked::<T, R>(
                        client,
                        CubeCount::Static(grid_x, grid_y, 1),
                        CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                        queries_gpu.clone().into_tensor_arg(vec_size),
                        self.vectors_gpu.clone().into_tensor_arg(vec_size),
                        query_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(1),
                        self.norms_gpu.as_ref().unwrap().clone().into_tensor_arg(1),
                        active_indices_gpu.into_tensor_arg(1),
                        write_offsets_gpu.into_tensor_arg(1),
                        candidate_dists_gpu.clone().into_tensor_arg(1),
                        candidate_indices_gpu.clone().into_tensor_arg(1),
                        ScalarArg {
                            elem: cluster_start as u32,
                        },
                        ScalarArg {
                            elem: cluster_count as u32,
                        },
                    );
                },
            }
        }

        if verbose {
            println!("  GPU work queued. Waiting for results...");
        }

        // ONE BLOCKING READ for all data
        let all_dists_flat = candidate_dists_gpu.read(client);
        let all_indices_flat = candidate_indices_gpu.read(client);

        if verbose {
            println!("  Reducing results on CPU...");
        }

        // Parallel Top-K selection on CPU
        let results: Vec<(Vec<usize>, Vec<T>)> = (0..n_queries)
            .into_par_iter()
            .map(|q_idx| {
                let start = q_idx * max_candidates;
                let count = candidates_per_query[q_idx];

                let dist_row = &all_dists_flat[start..start + count];
                let idx_row = &all_indices_flat[start..start + count];

                let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> =
                    BinaryHeap::with_capacity(k + 1);

                for i in 0..count {
                    let d = dist_row[i];
                    let idx = self.original_indices[idx_row[i] as usize];

                    if heap.len() == k {
                        if let Some((max_dist, _)) = heap.peek() {
                            if d >= max_dist.0 {
                                continue;
                            }
                        }
                    }

                    heap.push((OrderedFloat(d), idx));

                    if heap.len() > k {
                        heap.pop();
                    }
                }

                let mut final_indices = Vec::with_capacity(k);
                let mut final_dists = Vec::with_capacity(k);

                let mut items = heap.into_vec();
                items.sort_unstable();

                for (d_wrapper, idx) in items {
                    final_indices.push(idx);
                    final_dists.push(d_wrapper.0);
                }

                (final_indices, final_dists)
            })
            .collect();

        // Unzip results
        let (indices, dists) = results.into_iter().unzip();
        (indices, dists)
    }

    /// Calculate a memory-safe batch size for the Candidate Buffer strategy
    ///
    /// The new "Fire and Forget" strategy requires allocating a buffer of size:
    /// [batch_size * nprobe * avg_cluster_size].
    ///
    /// ### Params
    ///
    /// * `nprobe` - Number of probes to use
    ///
    /// ### Returns
    ///
    /// The batch size
    fn calculate_safe_batch_size(&self, nprobe: usize) -> usize {
        // f32 dist + u32 index
        const BYTES_PER_CANDIDATE: usize = 8;
        // To account for variable cluster sizes
        const SAFETY_MARGIN: f32 = 1.5;

        let avg_cluster_size = self.n as f32 / self.nlist as f32;
        let candidates_per_query = nprobe as f32 * avg_cluster_size * SAFETY_MARGIN;

        let bytes_per_query = candidates_per_query * BYTES_PER_CANDIDATE as f32;

        // calculate how many queries fit in the target memory
        let safe_batch = ((TARGET_BUFFER_MB * 1024 * 1024) as f32 / bytes_per_query) as usize;

        // clamp between 100 (sanity min) and 20k (sanity max)
        safe_batch.clamp(100, 20_000)
    }
}

/// Reorganise vectors by cluster for contiguous access
///
/// Helper function that re-organises the vectors by cluster, i.e.,
/// [cluster_0_vecs, cluster_1_vecs, ...]. This helps for subsequent GPU
/// launches.
///
/// ### Params
///
/// * `vectors_flat` - Original flat vectors
/// * `dim` - Dimensionality of the data set
/// * `n` - Number of samples in the index
/// * `assignments` - Cluster assignments
/// * `nlist` - Number of total lists
/// * `metric` - Distance metric
///
/// ### Returns
///
/// `(reordered flat vec, reordered indices, offsets, reordered norms)`
fn reorganise_by_cluster<T: Float + Copy + Send + Sync + Sum>(
    vectors_flat: &[T],
    dim: usize,
    n: usize,
    assignments: &[usize],
    nlist: usize,
    metric: &Dist,
) -> (Vec<T>, Vec<usize>, Vec<usize>, Vec<T>) {
    // Count vectors per cluster
    let mut counts = vec![0usize; nlist];
    for &cluster in assignments {
        counts[cluster] += 1;
    }

    // Build offsets
    let mut offsets = vec![0usize; nlist + 1];
    for i in 0..nlist {
        offsets[i + 1] = offsets[i] + counts[i];
    }

    // Place vectors and compute norms
    let mut vectors_reorg = vec![T::zero(); n * dim];
    let mut indices_reorg = vec![0usize; n];
    let mut norms_reorg = if *metric == Dist::Cosine {
        vec![T::zero(); n]
    } else {
        Vec::new()
    };
    let mut write_pos = offsets.clone();

    for vec_idx in 0..n {
        let cluster = assignments[vec_idx];
        let pos = write_pos[cluster];
        write_pos[cluster] += 1;

        indices_reorg[pos] = vec_idx;

        let src_start = vec_idx * dim;
        let dst_start = pos * dim;
        vectors_reorg[dst_start..dst_start + dim]
            .copy_from_slice(&vectors_flat[src_start..src_start + dim]);

        if *metric == Dist::Cosine {
            norms_reorg[pos] = vectors_flat[src_start..src_start + dim]
                .iter()
                .map(|&x| x * x)
                .sum::<T>()
                .sqrt();
        }
    }

    (vectors_reorg, indices_reorg, offsets, norms_reorg)
}

/// Invert probe lists to map clusters to queries
///
/// Transforms per-query cluster lists into per-cluster query lists.
/// Given probe_lists[q] = [c1, c2, ...] returns result[c] = [q1, q2, ...]
///
/// ### Params
///
/// * `probe_lists` - For each query, which clusters it probes
/// * `nlist` - Total number of clusters
///
/// ### Returns
///
/// For each cluster, which queries probe it
fn invert_probe_lists(probe_lists: &[Vec<usize>], nlist: usize) -> Vec<Vec<usize>> {
    let mut result = vec![Vec::new(); nlist];
    for (query_idx, probes) in probe_lists.iter().enumerate() {
        for &cluster_idx in probes {
            result[cluster_idx].push(query_idx);
        }
    }
    result
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::cpu::CpuDevice;
    use cubecl::cpu::CpuRuntime;
    use faer::Mat;

    #[test]
    fn test_ivf_index_build() {
        let device = CpuDevice;

        // 100 samples, 4 dimensions
        let data = Mat::from_fn(100, 4, |i, j| ((i + j) as f32) / 10.0);

        let index = IvfIndexGpu::<f32, CpuRuntime>::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(10),
            Some(5),
            42,
            false,
            device,
        );

        assert_eq!(index.dim, 4);
        assert_eq!(index.n, 100);
        assert_eq!(index.nlist, 10);
        assert_eq!(index.cluster_offsets.len(), 11);
    }

    #[test]
    fn test_ivf_index_query() {
        let device = CpuDevice;

        let data = Mat::from_fn(50, 4, |i, j| if i % 10 == j { 1.0_f32 } else { 0.1_f32 });

        let index = IvfIndexGpu::<f32, CpuRuntime>::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(5),
            Some(10),
            42,
            false,
            device,
        );

        let query = Mat::from_fn(3, 4, |i, j| if i == j { 1.0_f32 } else { 0.0_f32 });

        let (indices, distances) = index.query_batch(query.as_ref(), 5, Some(3), None, false);

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
        assert_eq!(indices[0].len(), 5);
    }

    #[test]
    fn test_ivf_index_cosine() {
        let device = CpuDevice;

        let data = Mat::from_fn(40, 4, |i, _j| (i as f32 + 1.0) / 10.0);

        let index = IvfIndexGpu::<f32, CpuRuntime>::build(
            data.as_ref(),
            Dist::Cosine,
            Some(5),
            Some(10),
            42,
            false,
            device,
        );

        let query = Mat::from_fn(2, 4, |_, _| 1.0_f32);
        let (indices, distances) = index.query_batch(query.as_ref(), 3, Some(2), None, false);

        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0].len(), 3);
        assert!(distances[0][0] >= 0.0);
    }

    #[test]
    fn test_ivf_generate_knn() {
        let device = CpuDevice;

        let data = Mat::from_fn(30, 4, |i, j| ((i * 3 + j) as f32) / 20.0);

        let index = IvfIndexGpu::<f32, CpuRuntime>::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(5),
            Some(5),
            42,
            false,
            device,
        );

        let (indices, distances) = index.generate_knn(4, Some(3), None, true, false);

        assert_eq!(indices.len(), 30);
        assert!(distances.is_some());
        assert_eq!(distances.unwrap().len(), 30);
    }

    #[test]
    fn test_reorganise_by_cluster() {
        let vectors: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let assignments = vec![0, 1, 0, 1];

        let (reorg, indices, offsets, _) =
            reorganise_by_cluster(&vectors, 4, 4, &assignments, 2, &Dist::Euclidean);

        assert_eq!(reorg.len(), 16);
        assert_eq!(indices.len(), 4);
        assert_eq!(offsets.len(), 3);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[2], 4);
    }

    #[test]
    fn test_invert_probe_lists() {
        let probe_lists = vec![vec![0, 1, 2], vec![1, 2], vec![0, 3]];

        let inverted = invert_probe_lists(&probe_lists, 4);

        assert_eq!(inverted.len(), 4);
        assert_eq!(inverted[0], vec![0, 2]);
        assert_eq!(inverted[1], vec![0, 1]);
        assert_eq!(inverted[2], vec![0, 1]);
        assert_eq!(inverted[3], vec![2]);
    }
}
