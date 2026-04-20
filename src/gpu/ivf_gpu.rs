//! Inverted file GPU-accelerated index. Keeps the data on GPU to avoid moving
//! data around.

use cubecl::prelude::*;
use faer::MatRef;
use num_traits::Float;
use rayon::prelude::*;
use std::iter::Sum;
use thousands::*;

use crate::gpu::dist_gpu::*;
use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::prelude::*;
use crate::utils::dist::Dist;
use crate::utils::k_means_utils::*;
use crate::utils::*;

/// Maximum number of queries processed in a single GPU batch to avoid
/// exhausting VRAM
const IVF_GPU_QUERY_BATCH_SIZE: usize = 100_000;
/// Target maximum size for the candidate buffer in megabytes
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
pub struct IvfIndexGpu<T: AnnSearchFloat + AnnSearchGpuFloat, R: Runtime> {
    /// All vectors reorganised by cluster, resident on GPU
    vectors_gpu: GpuTensor<R, T>,
    /// All norms reorganised by cluster, resident on GPU (Cosine only)
    norms_gpu: Option<GpuTensor<R, T>>,
    /// Reorganised vector data mirrored on CPU, used as query input for
    /// `generate_knn` without a GPU readback
    vectors_cpu: Vec<T>,
    /// Maps reorganised position -> original index
    original_indices: Vec<usize>,
    /// CSR-style offsets into `vectors_gpu` per cluster; length `nlist + 1`
    cluster_offsets: Vec<usize>,
    /// Centroids kept on the GPU
    centroids_gpu: GpuTensor<R, T>,
    ///  Centroid norms kept on the GPU
    centroid_norms_gpu: Option<GpuTensor<R, T>>,
    /// Dimensionality of the index
    dim: usize,
    /// Padded dimensionality of the index
    dim_padded: usize,
    /// Number of samples in the index
    n: usize,
    /// Number of lists in the index
    nlist: usize,
    /// Distance metric used
    metric: Dist,
    /// Device runtime for the GPU work
    device: R::Device,
}

impl<T, R> IvfIndexGpu<T, R>
where
    R: Runtime,
    T: AnnSearchFloat + AnnSearchGpuFloat,
{
    /// Build a batched IVF index
    ///
    /// ### Params
    ///
    /// * `data` - Database vectors [n, dim]
    /// * `metric` - Distance metric
    /// * `nlist` - Number of clusters (defaults to `sqrt(n)`)
    /// * `max_iters` - Optional k-means iterations (defaults to `50`)
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    /// * `device` - GPU device
    ///
    /// ### Returns
    ///
    /// Initialised `IvfIndexGpu` with all vectors and centroids resident on GPU
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

        let max_iters = max_iters.unwrap_or(50);
        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        let line = LINE_SIZE as usize;
        let dim_padded = dim.next_multiple_of(line);

        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        if verbose {
            println!("  Generating IVF index with {} Voronoi cells.", nlist);
        }

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

        // Norms on original (unpadded) data
        let data_norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| T::calculate_l2_norm(&vectors_flat[i * dim..(i + 1) * dim]))
                .collect()
        } else {
            vec![T::one(); n]
        };

        let centroid_norms = if metric == Dist::Cosine {
            (0..nlist)
                .map(|i| T::calculate_l2_norm(&centroids[i * dim..(i + 1) * dim]))
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

        let (vectors_by_cluster, original_indices, cluster_offsets, norms_by_cluster) =
            reorganise_by_cluster(&vectors_flat, dim, n, &assignments, nlist, &metric);

        if verbose {
            println!("  Uploading all vectors to GPU");
        }

        let client = R::client(&device);

        // Pad vectors and centroids for GPU
        let vectors_padded = if dim_padded != dim {
            pad_vectors(&vectors_by_cluster, n, dim, dim_padded)
        } else {
            vectors_by_cluster.clone()
        };

        let centroids_padded = if dim_padded != dim {
            pad_vectors(&centroids, nlist, dim, dim_padded)
        } else {
            centroids.clone()
        };

        let vectors_cpu = vectors_padded.clone();

        let vectors_gpu =
            GpuTensor::<R, T>::from_slice(&vectors_padded, vec![n, dim_padded], &client);

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
            GpuTensor::<R, T>::from_slice(&centroids_padded, vec![nlist, dim_padded], &client);

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
            vectors_cpu,
            original_indices,
            cluster_offsets,
            centroids_gpu,
            centroid_norms_gpu,
            dim,
            dim_padded,
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
        client: &ComputeClient<R>,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Vec<Vec<T>>) {
        assert_eq!(
            dim_query, self.dim_padded,
            "Query dimension {} != index padded dimension {}",
            dim_query, self.dim_padded
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
            return self.query_batch_internal(queries_flat, n_queries, k, nprobe, client);
        }

        let mut all_indices = Vec::with_capacity(n_queries);
        let mut all_distances = Vec::with_capacity(n_queries);

        for batch_idx in 0..n_batches {
            if verbose
                && (batch_idx == 0 || (batch_idx + 1) % 100 == 0 || batch_idx + 1 == n_batches)
            {
                println!("  Query batch {}/{}", batch_idx + 1, n_batches,);
            }

            let batch_start = batch_idx * nquery;
            let batch_end = (batch_start + nquery).min(n_queries);
            let batch_size = batch_end - batch_start;

            let batch_queries =
                &queries_flat[batch_start * self.dim_padded..batch_end * self.dim_padded];

            let (batch_indices, batch_dists) =
                self.query_batch_internal(batch_queries, batch_size, k, nprobe, client);

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
        assert_eq!(
            dim_query, self.dim,
            "Query dimension {} != index dimension {}",
            dim_query, self.dim
        );

        let client: ComputeClient<R> = R::client(&self.device);

        let nprobe_val = nprobe.unwrap_or(((self.nlist as f32).sqrt() as usize).max(1));
        let batch_size = nquery.unwrap_or_else(|| self.calculate_safe_batch_size(nprobe_val));

        let queries_padded = if self.dim_padded != self.dim {
            pad_vectors(&queries_flat, n_queries, self.dim, self.dim_padded)
        } else {
            queries_flat
        };

        let (indices, dist) = self.query_internal(
            &queries_padded,
            n_queries,
            self.dim_padded,
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
        let client: ComputeClient<R> = R::client(&self.device);

        let nprobe = nprobe.unwrap_or(((self.nlist as f32).sqrt() as usize).max(1));

        let batch_size = nquery.unwrap_or_else(|| {
            let safe = self.calculate_safe_batch_size(nprobe);
            if verbose {
                println!("  Auto-tuned batch size to {} (based on density)", safe);
            }
            safe
        });

        if verbose {
            println!("  Reading vectors from GPU for self-query...");
        }
        let vectors_by_cluster = &self.vectors_cpu;

        let (indices_reorg, dist_reorg) = self.query_internal(
            vectors_by_cluster,
            self.n,
            self.dim_padded,
            k,
            Some(nprobe),
            Some(batch_size),
            &client,
            verbose,
        );

        client.memory_cleanup();

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

    /// Returns the approximate memory footprint of the index.
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
        client: &ComputeClient<R>,
    ) -> (Vec<Vec<usize>>, Vec<Vec<T>>) {
        let vec_size = LINE_SIZE as usize;
        let dim_lines = self.dim_padded / vec_size;

        let query_norms = if self.metric == Dist::Cosine {
            (0..n_queries)
                .into_par_iter()
                .map(|i| {
                    let start = i * self.dim_padded;
                    T::calculate_l2_norm(&queries_flat[start..start + self.dim_padded])
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let queries_gpu =
            GpuTensor::<R, T>::from_slice(queries_flat, vec![n_queries, self.dim_padded], client);

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
        let (grid_y, grid_z) = grid_2d((n_queries as u32).div_ceil(WORKGROUP_SIZE_Y));

        match self.metric {
            Dist::Euclidean => unsafe {
                let _ = euclidean_tiled::launch_unchecked::<T, R>(
                    client,
                    CubeCount::Static(grid_x, grid_y, grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y),
                    queries_gpu.clone().into_tensor_arg(vec_size),
                    self.centroids_gpu.clone().into_tensor_arg(vec_size),
                    centroid_dists_gpu.into_tensor_arg(1),
                    ScalarArg { elem: 0u32 },
                    ScalarArg {
                        elem: self.nlist as u32,
                    },
                    ScalarArg {
                        elem: n_queries as u32,
                    },
                    ScalarArg {
                        elem: self.nlist as u32,
                    },
                    dim_lines,
                );
            },
            Dist::Cosine => unsafe {
                let _ = cosine_tiled::launch_unchecked::<T, R>(
                    client,
                    CubeCount::Static(grid_x, grid_y, grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y),
                    queries_gpu.clone().into_tensor_arg(vec_size),
                    self.centroids_gpu.clone().into_tensor_arg(vec_size),
                    query_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(1),
                    self.centroid_norms_gpu
                        .as_ref()
                        .unwrap()
                        .clone()
                        .into_tensor_arg(1),
                    centroid_dists_gpu.into_tensor_arg(1),
                    ScalarArg { elem: 0u32 },
                    ScalarArg {
                        elem: self.nlist as u32,
                    },
                    ScalarArg {
                        elem: n_queries as u32,
                    },
                    ScalarArg {
                        elem: self.nlist as u32,
                    },
                    dim_lines,
                );
            },
        }

        let centroid_dists = centroid_dists_gpu.read(client);

        let probe_lists: Vec<Vec<usize>> = (0..n_queries)
            .into_par_iter()
            .map(|q| {
                let row_start = q * self.nlist;
                let mut cluster_dists: Vec<(T, usize)> = (0..self.nlist)
                    .map(|c| (centroid_dists[row_start + c], c))
                    .collect();

                if nprobe < self.nlist {
                    cluster_dists.select_nth_unstable_by(nprobe, |a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    cluster_dists[..nprobe].sort_unstable_by(|a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                } else {
                    cluster_dists.sort_unstable_by(|a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                cluster_dists[..nprobe].iter().map(|&(_, c)| c).collect()
            })
            .collect();

        let mut cpu_write_pointers = vec![0u32; n_queries];
        let mut tasks: Vec<(u32, u32, u32, u32)> = Vec::new();
        let mut max_db_count = 0u32;

        for q_idx in 0..n_queries {
            for &c in &probe_lists[q_idx] {
                let start = self.cluster_offsets[c];
                let count = self.cluster_offsets[c + 1] - start;

                if count > 0 {
                    tasks.push((
                        q_idx as u32,
                        start as u32,
                        cpu_write_pointers[q_idx],
                        count as u32,
                    ));

                    cpu_write_pointers[q_idx] += count as u32;
                    if count as u32 > max_db_count {
                        max_db_count = count as u32;
                    }
                }
            }
        }

        tasks.sort_unstable_by_key(|t| t.0);

        let n_tasks = tasks.len();
        if n_tasks == 0 {
            return (vec![vec![]; n_queries], vec![vec![]; n_queries]);
        }

        let task_q_idx: Vec<u32> = tasks.iter().map(|t| t.0).collect();
        let task_db_start: Vec<u32> = tasks.iter().map(|t| t.1).collect();
        let task_write_offset: Vec<u32> = tasks.iter().map(|t| t.2).collect();
        let task_db_count: Vec<u32> = tasks.iter().map(|t| t.3).collect();

        let max_candidates: usize = cpu_write_pointers
            .iter()
            .fold(0, |acc, &x| acc.max(x as usize));

        let candidate_dists_gpu = GpuTensor::<R, T>::empty(vec![n_queries, max_candidates], client);
        let candidate_indices_gpu =
            GpuTensor::<R, u32>::empty(vec![n_queries, max_candidates], client);

        let task_q_idx_gpu = GpuTensor::<R, u32>::from_slice(&task_q_idx, vec![n_tasks], client);
        let task_db_start_gpu =
            GpuTensor::<R, u32>::from_slice(&task_db_start, vec![n_tasks], client);
        let task_write_offset_gpu =
            GpuTensor::<R, u32>::from_slice(&task_write_offset, vec![n_tasks], client);
        let task_db_count_gpu =
            GpuTensor::<R, u32>::from_slice(&task_db_count, vec![n_tasks], client);

        let mega_grid_x = max_db_count.div_ceil(WORKGROUP_SIZE_X).max(1);
        let (mega_grid_y, mega_grid_z) = grid_2d((n_tasks as u32).div_ceil(WORKGROUP_SIZE_Y));

        match self.metric {
            Dist::Euclidean => unsafe {
                let _ = compute_ivf_mega_euclidean_cached::launch_unchecked::<T, R>(
                    client,
                    CubeCount::Static(mega_grid_x, mega_grid_y, mega_grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y),
                    queries_gpu.clone().into_tensor_arg(vec_size),
                    self.vectors_gpu.clone().into_tensor_arg(vec_size),
                    task_q_idx_gpu.into_tensor_arg(1),
                    task_db_start_gpu.into_tensor_arg(1),
                    task_write_offset_gpu.into_tensor_arg(1),
                    task_db_count_gpu.into_tensor_arg(1),
                    candidate_dists_gpu.clone().into_tensor_arg(1),
                    candidate_indices_gpu.clone().into_tensor_arg(1),
                    ScalarArg {
                        elem: n_tasks as u32,
                    },
                    dim_lines,
                );
            },
            Dist::Cosine => unsafe {
                let _ = compute_ivf_mega_cosine_cached::launch_unchecked::<T, R>(
                    client,
                    CubeCount::Static(mega_grid_x, mega_grid_y, mega_grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y),
                    queries_gpu.clone().into_tensor_arg(vec_size),
                    self.vectors_gpu.clone().into_tensor_arg(vec_size),
                    query_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(1),
                    self.norms_gpu.as_ref().unwrap().clone().into_tensor_arg(1),
                    task_q_idx_gpu.into_tensor_arg(1),
                    task_db_start_gpu.into_tensor_arg(1),
                    task_write_offset_gpu.into_tensor_arg(1),
                    task_db_count_gpu.into_tensor_arg(1),
                    candidate_dists_gpu.clone().into_tensor_arg(1),
                    candidate_indices_gpu.clone().into_tensor_arg(1),
                    ScalarArg {
                        elem: n_tasks as u32,
                    },
                    dim_lines,
                );
            },
        }

        let topk_dists = GpuTensor::<R, T>::empty(vec![n_queries, k], client);
        let topk_indices = GpuTensor::<R, u32>::empty(vec![n_queries, k], client);

        let cpq = GpuTensor::<R, u32>::from_slice(&cpu_write_pointers, vec![n_queries], client);
        let (coal_gx, coal_gy) = grid_2d(n_queries as u32);
        unsafe {
            let _ = reduce_ivf_topk_coalesced::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(coal_gx, coal_gy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                candidate_dists_gpu.clone().into_tensor_arg(1),
                candidate_indices_gpu.clone().into_tensor_arg(1),
                cpq.into_tensor_arg(1),
                topk_dists.clone().into_tensor_arg(1),
                topk_indices.clone().into_tensor_arg(1),
                ScalarArg { elem: k as u32 },
                k,
            );
        }

        let final_dists = topk_dists.read(client);
        let final_indices = topk_indices.read(client);

        let mut results_indices = Vec::with_capacity(n_queries);
        let mut results_dists = Vec::with_capacity(n_queries);

        for q in 0..n_queries {
            let mut row_idx = Vec::with_capacity(k);
            let mut row_dist = Vec::with_capacity(k);
            let start = q * k;

            for i in 0..k {
                let d = final_dists[start + i];
                if d < T::from_f32(f32::MAX).unwrap() {
                    let reorg_idx = final_indices[start + i] as usize;
                    row_idx.push(self.original_indices[reorg_idx]);
                    row_dist.push(d);
                }
            }
            results_indices.push(row_idx);
            results_dists.push(row_dist);
        }

        (results_indices, results_dists)
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
}

#[cfg(test)]
#[cfg(feature = "gpu-tests")]
mod tests_wpgu {
    use super::*;
    use cubecl::wgpu::WgpuDevice;
    use cubecl::wgpu::WgpuRuntime;
    use faer::Mat;

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cubecl::wgpu::WgpuRuntime::client(&device);
        }));
        result.ok().map(|_| device)
    }

    #[test]
    fn test_ivf_generate_knn() {
        let Some(device) = try_device() else {
            eprintln!("Skipping test: no wgpu backend available");
            return;
        };

        let data = Mat::from_fn(30, 4, |i, j| ((i * 3 + j) as f32) / 20.0);

        let index = IvfIndexGpu::<f32, WgpuRuntime>::build(
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
}
