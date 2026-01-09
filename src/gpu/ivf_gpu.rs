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
use crate::utils::dist::Dist;
use crate::utils::dist::*;
use crate::utils::heap_structs::*;
use crate::utils::ivf_utils::*;
use crate::utils::*;

/// To not explode memory VRAM memory
const IVF_GPU_QUERY_BATCH_SIZE: usize = 100_000;

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

        let (indices, dist) = self.query_internal(
            &queries_flat,
            n_queries,
            dim_query,
            k,
            nprobe,
            nquery,
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

        // We need to read vectors back from GPU for the query
        // This is unavoidable since query_internal expects CPU data
        let vectors_by_cluster = self.vectors_gpu.clone().read(&client);

        let (indices_reorg, dist_reorg) = self.query_internal(
            &vectors_by_cluster,
            self.n,
            self.dim,
            k,
            nprobe,
            nquery,
            &client,
            verbose,
        );

        client.memory_cleanup();

        // reorder results to match original vector ordering
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

    // Generate kNN graph using cluster-pair iteration
    ///
    /// Optimised for self-kNN where queries = database. Instead of iterating
    /// over queries (O(n × nprobe) kernel launches), iterates over cluster pairs
    /// (O(nlist × nprobe) kernel launches).
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `nprobe` - Number of nearby clusters to compare (defaults to √nlist)
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`
    pub fn generate_knn_cluster_pairs(
        &self,
        k: usize,
        nprobe: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        let client: ComputeClient<<R as Runtime>::Server> = R::client(&self.device);
        let vec_size = LINE_SIZE as u8;

        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);

        if verbose {
            println!(
                "  Cluster-pair kNN: {} clusters, nprobe={}, {} vectors",
                self.nlist, nprobe, self.n
            );
        }

        // Step 1: Compute centroid-centroid distances to find nearby cluster pairs
        let centroid_dists_gpu = GpuTensor::<R, T>::empty(vec![self.nlist, self.nlist], &client);

        let grid_x = (self.nlist as u32).div_ceil(WORKGROUP_SIZE_X);
        let grid_y = (self.nlist as u32).div_ceil(WORKGROUP_SIZE_Y);

        match self.metric {
            Dist::Euclidean => unsafe {
                euclidean_distances_gpu_chunk::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(grid_x, grid_y, 1),
                    CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                    self.centroids_gpu.clone().into_tensor_arg(vec_size),
                    self.centroids_gpu.clone().into_tensor_arg(vec_size),
                    centroid_dists_gpu.into_tensor_arg(1),
                );
            },
            Dist::Cosine => unsafe {
                cosine_distances_gpu_chunk::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(grid_x, grid_y, 1),
                    CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                    self.centroids_gpu.clone().into_tensor_arg(vec_size),
                    self.centroids_gpu.clone().into_tensor_arg(vec_size),
                    self.centroid_norms_gpu
                        .as_ref()
                        .unwrap()
                        .clone()
                        .into_tensor_arg(1),
                    self.centroid_norms_gpu
                        .as_ref()
                        .unwrap()
                        .clone()
                        .into_tensor_arg(1),
                    centroid_dists_gpu.into_tensor_arg(1),
                );
            },
        }

        let centroid_dists = centroid_dists_gpu.read(&client);

        // Step 2: For each cluster, find its nprobe nearest clusters
        let nearby_clusters: Vec<Vec<usize>> = (0..self.nlist)
            .into_par_iter()
            .map(|c| {
                let row_start = c * self.nlist;
                let mut cluster_dists: Vec<(T, usize)> = (0..self.nlist)
                    .map(|other| (centroid_dists[row_start + other], other))
                    .collect();
                cluster_dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                // Include self (distance 0) and nprobe-1 others
                cluster_dists
                    .into_iter()
                    .take(nprobe)
                    .map(|(_, c)| c)
                    .collect()
            })
            .collect();

        // Step 3: Initialise heaps for all vectors
        let mut heaps: Vec<BinaryHeap<(OrderedFloat<T>, usize)>> =
            vec![BinaryHeap::with_capacity(k + 1); self.n];

        // Step 4: Process cluster pairs
        let total_pairs: usize = nearby_clusters.iter().map(|v| v.len()).sum();
        let mut pairs_processed = 0;

        for cluster_i in 0..self.nlist {
            let start_i = self.cluster_offsets[cluster_i];
            let end_i = self.cluster_offsets[cluster_i + 1];
            let size_i = end_i - start_i;

            if size_i == 0 {
                continue;
            }

            for &cluster_j in &nearby_clusters[cluster_i] {
                let start_j = self.cluster_offsets[cluster_j];
                let end_j = self.cluster_offsets[cluster_j + 1];
                let size_j = end_j - start_j;

                if size_j == 0 {
                    continue;
                }

                pairs_processed += 1;
                if verbose && pairs_processed % 1000 == 0 {
                    println!(
                        "  Processed {}/{} cluster pairs",
                        pairs_processed, total_pairs
                    );
                }

                // allocate distance buffer for this pair
                let dist_gpu = GpuTensor::<R, T>::empty(vec![size_i, size_j], &client);

                let grid_x = (size_j as u32).div_ceil(WORKGROUP_SIZE_X);
                let grid_y = (size_i as u32).div_ceil(WORKGROUP_SIZE_Y);

                // Compute all pairwise distances between cluster_i and cluster_j
                match self.metric {
                    Dist::Euclidean => unsafe {
                        euclidean_distances_gpu_offset_pair::launch_unchecked::<T, R>(
                            &client,
                            CubeCount::Static(grid_x, grid_y, 1),
                            CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                            self.vectors_gpu.clone().into_tensor_arg(vec_size),
                            dist_gpu.clone().into_tensor_arg(1),
                            ScalarArg {
                                elem: start_i as u32,
                            },
                            ScalarArg {
                                elem: size_i as u32,
                            },
                            ScalarArg {
                                elem: start_j as u32,
                            },
                            ScalarArg {
                                elem: size_j as u32,
                            },
                        );
                    },
                    Dist::Cosine => unsafe {
                        cosine_distances_gpu_offset_pair::launch_unchecked::<T, R>(
                            &client,
                            CubeCount::Static(grid_x, grid_y, 1),
                            CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                            self.vectors_gpu.clone().into_tensor_arg(vec_size),
                            self.norms_gpu.as_ref().unwrap().clone().into_tensor_arg(1),
                            dist_gpu.clone().into_tensor_arg(1),
                            ScalarArg {
                                elem: start_i as u32,
                            },
                            ScalarArg {
                                elem: size_i as u32,
                            },
                            ScalarArg {
                                elem: start_j as u32,
                            },
                            ScalarArg {
                                elem: size_j as u32,
                            },
                        );
                    },
                }

                // read distances and update heaps
                let dists = dist_gpu.read(&client);

                // update heaps for vectors in cluster_i
                for local_i in 0..size_i {
                    let global_i = self.original_indices[start_i + local_i];
                    let heap = &mut heaps[global_i];
                    let row_start = local_i * size_j;

                    for local_j in 0..size_j {
                        let global_j = self.original_indices[start_j + local_j];

                        // Skip self-matches
                        if global_i == global_j {
                            continue;
                        }

                        let dist = dists[row_start + local_j];

                        if heap.len() < k {
                            heap.push((OrderedFloat(dist), global_j));
                        } else if dist < heap.peek().unwrap().0 .0 {
                            heap.pop();
                            heap.push((OrderedFloat(dist), global_j));
                        }
                    }
                }
            }
        }

        client.memory_cleanup();

        // Extract results from heaps
        let results: Vec<_> = heaps
            .into_par_iter()
            .map(|heap| {
                let mut sorted: Vec<_> = heap.into_iter().collect();
                sorted.sort_unstable_by_key(|&(d, _)| d);

                let (distances, indices): (Vec<_>, Vec<_>) = sorted
                    .into_iter()
                    .map(|(OrderedFloat(d), idx)| (d, idx))
                    .unzip();

                (indices, distances)
            })
            .collect();

        let (indices, distances): (Vec<_>, Vec<_>) = results.into_iter().unzip();

        if return_dist {
            (indices, Some(distances))
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

        // Compute query norms for cosine
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

        // Upload queries to GPU
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

        // Phase 1: compute query-centroid distances
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

        let centroid_dists = centroid_dists_gpu.read(client);

        // Phase 2: select top nprobe clusters per query (CPU, parallel)
        let probe_lists: Vec<Vec<usize>> = (0..n_queries)
            .into_par_iter()
            .map(|q| {
                let row_start = q * self.nlist;
                let mut cluster_dists: Vec<(T, usize)> = (0..self.nlist)
                    .map(|c| (centroid_dists[row_start + c], c))
                    .collect();
                cluster_dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                cluster_dists
                    .into_iter()
                    .take(nprobe)
                    .map(|(_, c)| c)
                    .collect()
            })
            .collect();

        // Invert: for each cluster, which queries probe it?
        let cluster_to_queries = invert_probe_lists(&probe_lists, self.nlist);

        // Phase 3: per-cluster batched search with pre-allocated buffers
        let mut heaps: Vec<BinaryHeap<(OrderedFloat<T>, usize)>> =
            vec![BinaryHeap::with_capacity(k + 1); n_queries];

        for cluster_idx in 0..self.nlist {
            if verbose && cluster_idx % 100 == 0 {
                println!("  Cluster {}/{} for current batch", cluster_idx, self.nlist);
            }

            let query_indices = &cluster_to_queries[cluster_idx];
            if query_indices.is_empty() {
                continue;
            }

            let cluster_start = self.cluster_offsets[cluster_idx];
            let cluster_end = self.cluster_offsets[cluster_idx + 1];
            let cluster_size = cluster_end - cluster_start;

            if cluster_size == 0 {
                continue;
            }

            let n_probing = query_indices.len();

            // Gather queries that probe this cluster into contiguous CPU buffer
            let mut gathered_queries = Vec::with_capacity(n_probing * self.dim);
            let mut gathered_norms_cpu = Vec::with_capacity(n_probing);

            for &q_idx in query_indices {
                let start = q_idx * self.dim;
                gathered_queries.extend_from_slice(&queries_flat[start..start + self.dim]);
                if self.metric == Dist::Cosine {
                    gathered_norms_cpu.push(query_norms[q_idx]);
                }
            }

            // Upload gathered queries
            let gathered_gpu = GpuTensor::<R, T>::from_slice(
                &gathered_queries,
                vec![n_probing, dim_vectorized],
                client,
            );

            let gathered_norms_gpu = if self.metric == Dist::Cosine {
                Some(GpuTensor::<R, T>::from_slice(
                    &gathered_norms_cpu,
                    vec![n_probing],
                    client,
                ))
            } else {
                None
            };

            // Allocate distance buffer for this cluster's actual size
            let dist_gpu = GpuTensor::<R, T>::empty(vec![n_probing, cluster_size], client);

            let grid_x = (cluster_size as u32).div_ceil(WORKGROUP_SIZE_X);
            let grid_y = (n_probing as u32).div_ceil(WORKGROUP_SIZE_Y);

            // Use offset-based kernels - no cluster data upload needed!
            match self.metric {
                Dist::Euclidean => unsafe {
                    euclidean_distances_gpu_offset::launch_unchecked::<T, R>(
                        client,
                        CubeCount::Static(grid_x, grid_y, 1),
                        CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                        gathered_gpu.into_tensor_arg(vec_size),
                        self.vectors_gpu.clone().into_tensor_arg(vec_size),
                        dist_gpu.clone().into_tensor_arg(1),
                        ScalarArg {
                            elem: cluster_start as u32,
                        },
                        ScalarArg {
                            elem: cluster_size as u32,
                        },
                    );
                },
                Dist::Cosine => unsafe {
                    cosine_distances_gpu_offset::launch_unchecked::<T, R>(
                        client,
                        CubeCount::Static(grid_x, grid_y, 1),
                        CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                        gathered_gpu.into_tensor_arg(vec_size),
                        self.vectors_gpu.clone().into_tensor_arg(vec_size),
                        gathered_norms_gpu.unwrap().into_tensor_arg(1),
                        self.norms_gpu.as_ref().unwrap().clone().into_tensor_arg(1),
                        dist_gpu.clone().into_tensor_arg(1),
                        ScalarArg {
                            elem: cluster_start as u32,
                        },
                        ScalarArg {
                            elem: cluster_size as u32,
                        },
                    );
                },
            }

            // Read results and process
            let dists = dist_gpu.read(client);
            process_cluster_results(
                &dists,
                query_indices,
                cluster_idx,
                cluster_size,
                &self.original_indices,
                &self.cluster_offsets,
                k,
                &mut heaps,
            );
        }

        // Extract results from heaps (parallel)
        let results: Vec<_> = heaps
            .into_par_iter()
            .map(|heap| {
                let mut sorted: Vec<_> = heap.into_iter().collect();
                sorted.sort_unstable_by_key(|&(d, _)| d);

                let (distances, indices): (Vec<_>, Vec<_>) = sorted
                    .into_iter()
                    .map(|(OrderedFloat(d), idx)| (d, idx))
                    .unzip();

                (indices, distances)
            })
            .collect();

        let (all_indices, all_distances): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        (all_indices, all_distances)
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

/// Process cluster search results and update per-query heaps
///
/// Takes distance matrix from a batched cluster search and updates the
/// k-NN heaps for each query. Only retains the k closest neighbours seen
/// so far across all processed clusters.
///
/// ### Params
///
/// * `distances` - Distance matrix [n_queries, max_cluster_size] (only first cluster_size cols valid)
/// * `query_indices` - Global query indices for this batch
/// * `cluster_idx` - Index of the cluster being processed
/// * `cluster_size` - Actual size of this cluster
/// * `original_indices` - Maps reorganised position -> original database index
/// * `cluster_offsets` - CSR offsets for clusters
/// * `k` - Number of neighbours to retain
/// * `heaps` - Per-query max-heaps storing current k-NN candidates
#[allow(clippy::too_many_arguments)]
fn process_cluster_results<T: Float>(
    distances: &[T],
    query_indices: &[usize],
    cluster_idx: usize,
    cluster_size: usize,
    original_indices: &[usize],
    cluster_offsets: &[usize],
    k: usize,
    heaps: &mut [BinaryHeap<(OrderedFloat<T>, usize)>],
) {
    let cluster_start = cluster_offsets[cluster_idx];
    // Note: distances tensor may be larger than cluster_size (pre-allocated for max)
    // We use the max_cluster_size as stride since that's how the tensor was allocated
    let max_cluster_size = distances.len() / query_indices.len();

    for (local_q, &global_q) in query_indices.iter().enumerate() {
        let heap = &mut heaps[global_q];
        let row_start = local_q * max_cluster_size;

        for i in 0..cluster_size {
            let dist = distances[row_start + i];
            let original_idx = original_indices[cluster_start + i];

            if heap.len() < k {
                heap.push((OrderedFloat(dist), original_idx));
            } else if dist < heap.peek().unwrap().0 .0 {
                heap.pop();
                heap.push((OrderedFloat(dist), original_idx));
            }
        }
    }
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
