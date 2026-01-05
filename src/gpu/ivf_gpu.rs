use cubecl::prelude::*;
use faer::MatRef;
use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;

use crate::gpu::dist_gpu::*;
use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::utils::dist::Dist;
use crate::utils::heap_structs::*;
use crate::utils::ivf_utils::*;
use crate::utils::*;

/// Batched IVF index with GPU acceleration
///
/// Designed for large-scale batch queries (100k-1M queries) against large
/// databases (1M-10M vectors). Minimises kernel launches by batching operations
/// and processing all queries against each cluster in a single kernel.
///
/// ### Architecture
///
/// - Database vectors reorganised by cluster for contiguous access
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
/// * `vectors_by_cluster` - Vectors reorganised by cluster
/// * `norms_by_cluster` - Norms reorganised by cluster
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
    vectors_by_cluster: Vec<T>,
    norms_by_cluster: Vec<T>,
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
    T: Float + Sum + cubecl::frontend::Float + cubecl::CubeElement + FromPrimitive + Send + Sync,
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
                nlist, n
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

        // reorganise vectors by cluster
        let (vectors_by_cluster, original_indices, cluster_offsets, norms_by_cluster) =
            reorganise_by_cluster(&vectors_flat, dim, n, &assignments, nlist, &metric);

        if verbose {
            println!("  Uploading centroids to GPU");
        }

        let client = R::client(&device);

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
            vectors_by_cluster,
            norms_by_cluster,
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
    /// * `verbose` - Controls the verbosity of the function
    ///
    /// ### Returns
    ///
    /// Tuple of `(Vec<indices>, Vec<dist>)` for the queries.
    fn query_internal(
        &self,
        queries_flat: &[T],
        n_queries: usize,
        dim_query: usize,
        k: usize,
        nprobe: Option<usize>,
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
        let k = k.min(self.n);

        let dim_vectorized = self.dim / LINE_SIZE as usize;
        let vec_size = LINE_SIZE as u8;
        let client = R::client(&self.device);

        // compute query norms for cosine
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

        // upload queries to GPU
        let queries_gpu =
            GpuTensor::<R, T>::from_slice(queries_flat, vec![n_queries, dim_vectorized], &client);

        let query_norms_gpu = if self.metric == Dist::Cosine {
            Some(GpuTensor::<R, T>::from_slice(
                &query_norms,
                vec![n_queries],
                &client,
            ))
        } else {
            None
        };

        // phase 1: compute query-centroid distances (one kernel launch here)
        let centroid_dists_gpu = GpuTensor::<R, T>::empty(vec![n_queries, self.nlist], &client);

        let grid_x = (self.nlist as u32).div_ceil(WORKGROUP_SIZE_X);
        let grid_y = (n_queries as u32).div_ceil(WORKGROUP_SIZE_Y);

        match self.metric {
            Dist::Euclidean => unsafe {
                euclidean_distances_gpu_chunk::launch_unchecked::<T, R>(
                    &client,
                    CubeCount::Static(grid_x, grid_y, 1),
                    CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                    queries_gpu.clone().into_tensor_arg(vec_size),
                    self.centroids_gpu.clone().into_tensor_arg(vec_size),
                    centroid_dists_gpu.into_tensor_arg(1),
                );
            },
            Dist::Cosine => unsafe {
                cosine_distances_gpu_chunk::launch_unchecked::<T, R>(
                    &client,
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

        let centroid_dists = centroid_dists_gpu.read(&client);

        // phase 2: select top nprobe clusters per query
        // happens on CPU in parallel - even with larger data sets this should be fine...
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

        // invert: for each cluster, which queries probe it?
        let cluster_to_queries = invert_probe_lists(&probe_lists, self.nlist);

        // phase 3: per-cluster batched search with double buffering
        let mut heaps: Vec<BinaryHeap<(OrderedFloat<T>, usize)>> =
            vec![BinaryHeap::with_capacity(k + 1); n_queries];

        // pending work: (distances_gpu, cluster_idx, query_indices)
        let mut pending: Option<(GpuTensor<R, T>, usize, Vec<usize>)> = None;

        for cluster_idx in 0..self.nlist {
            if verbose && cluster_idx % 10 == 0 {
                println!("Processed {} clusters out of {}", cluster_idx, self.nlist);
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

            // Gather queries that probe this cluster
            let mut gathered_queries = Vec::with_capacity(n_probing * self.dim);
            let mut gathered_norms = Vec::with_capacity(n_probing);

            for &q_idx in query_indices {
                let start = q_idx * self.dim;
                gathered_queries.extend_from_slice(&queries_flat[start..start + self.dim]);
                if self.metric == Dist::Cosine {
                    gathered_norms.push(query_norms[q_idx]);
                }
            }

            let gathered_gpu = GpuTensor::<R, T>::from_slice(
                &gathered_queries,
                vec![n_probing, dim_vectorized],
                &client,
            );

            let gathered_norms_gpu = if self.metric == Dist::Cosine {
                Some(GpuTensor::<R, T>::from_slice(
                    &gathered_norms,
                    vec![n_probing],
                    &client,
                ))
            } else {
                None
            };

            // get cluster vectors (contiguous in vectors_by_cluster)
            let cluster_vec_start = cluster_start * self.dim;
            let cluster_vec_end = cluster_end * self.dim;
            let cluster_data = &self.vectors_by_cluster[cluster_vec_start..cluster_vec_end];

            let cluster_gpu = GpuTensor::<R, T>::from_slice(
                cluster_data,
                vec![cluster_size, dim_vectorized],
                &client,
            );

            let cluster_norms_gpu = if self.metric == Dist::Cosine {
                Some(GpuTensor::<R, T>::from_slice(
                    &self.norms_by_cluster[cluster_start..cluster_end],
                    vec![cluster_size],
                    &client,
                ))
            } else {
                None
            };

            let dist_gpu = GpuTensor::<R, T>::empty(vec![n_probing, cluster_size], &client);

            let grid_x = (cluster_size as u32).div_ceil(WORKGROUP_SIZE_X);
            let grid_y = (n_probing as u32).div_ceil(WORKGROUP_SIZE_Y);

            match self.metric {
                Dist::Euclidean => unsafe {
                    euclidean_distances_gpu_chunk::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(grid_x, grid_y, 1),
                        CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                        gathered_gpu.into_tensor_arg(vec_size),
                        cluster_gpu.into_tensor_arg(vec_size),
                        dist_gpu.into_tensor_arg(1),
                    );
                },
                Dist::Cosine => unsafe {
                    cosine_distances_gpu_chunk::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(grid_x, grid_y, 1),
                        CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                        gathered_gpu.into_tensor_arg(vec_size),
                        cluster_gpu.into_tensor_arg(vec_size),
                        gathered_norms_gpu.unwrap().into_tensor_arg(1),
                        cluster_norms_gpu.unwrap().into_tensor_arg(1),
                        dist_gpu.into_tensor_arg(1),
                    );
                },
            }

            // process previous cluster's results while GPU computes current
            if let Some((prev_dists_gpu, prev_cluster_idx, prev_query_indices)) = pending.take() {
                let dists = prev_dists_gpu.read(&client);
                process_cluster_results(
                    &dists,
                    &prev_query_indices,
                    prev_cluster_idx,
                    &self.original_indices,
                    &self.cluster_offsets,
                    k,
                    &mut heaps,
                );
            }

            pending = Some((dist_gpu, cluster_idx, query_indices.clone()));
        }

        // process final cluster
        if let Some((prev_dists_gpu, prev_cluster_idx, prev_query_indices)) = pending {
            let dists = prev_dists_gpu.read(&client);
            process_cluster_results(
                &dists,
                &prev_query_indices,
                prev_cluster_idx,
                &self.original_indices,
                &self.cluster_offsets,
                k,
                &mut heaps,
            );
        }

        // extract results from heaps (parallel)
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

    /// Query the index with a batch of vectors
    ///
    /// ### Params
    ///
    /// * `query_mat` - Query vectors [n_queries, dim]
    /// * `k` - Number of neighbours per query
    /// * `nprobe` - Number of clusters to search (defaults to √nlist)
    ///
    /// ### Returns
    ///
    /// Tuple of `(Vec<indices>, Vec<dist>)` for the queries.
    pub fn query_batch(
        &self,
        query_mat: MatRef<T>,
        k: usize,
        nprobe: Option<usize>,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Vec<Vec<T>>) {
        let (queries_flat, n_queries, dim_query) = matrix_to_flat(query_mat);

        self.query_internal(&queries_flat, n_queries, dim_query, k, nprobe, verbose)
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
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        let (indices_reorg, dist_reorg) = self.query_internal(
            &self.vectors_by_cluster,
            self.n,
            self.dim,
            k,
            nprobe,
            verbose,
        );

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
/// * `distances` - Distance matrix [n_queries, cluster_size]
/// * `query_indices` - Global query indices for this batch
/// * `cluster_idx` - Index of the cluster being processed
/// * `original_indices` - Maps reorganised position -> original database index
/// * `cluster_offsets` - CSR offsets for clusters
/// * `k` - Number of neighbours to retain
/// * `heaps` - Per-query max-heaps storing current k-NN candidates
fn process_cluster_results<T: Float>(
    distances: &[T],
    query_indices: &[usize],
    cluster_idx: usize,
    original_indices: &[usize],
    cluster_offsets: &[usize],
    k: usize,
    heaps: &mut [BinaryHeap<(OrderedFloat<T>, usize)>],
) {
    let cluster_start = cluster_offsets[cluster_idx];
    let cluster_end = cluster_offsets[cluster_idx + 1];
    let cluster_size = cluster_end - cluster_start;

    for (local_q, &global_q) in query_indices.iter().enumerate() {
        let heap = &mut heaps[global_q];
        let row_start = local_q * cluster_size;

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

        let (indices, distances) = index.query_batch(query.as_ref(), 5, Some(3), false);

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
        let (indices, distances) = index.query_batch(query.as_ref(), 3, Some(2), false);

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

        let (indices, distances) = index.generate_knn(4, Some(3), true, false);

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
