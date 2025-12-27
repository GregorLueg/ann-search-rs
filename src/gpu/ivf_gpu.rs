use cubecl::prelude::*;
use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;

use crate::gpu::dist_gpu::*;
use crate::gpu::tensor::*;
use crate::gpu::*;
use crate::utils::dist::Dist;
use crate::utils::heap_structs::OrderedFloat;
use crate::utils::ivf_utils::*;

/// IVF index with GPU-accelerated search
pub struct IvfIndexGpu<T: Float, R: Runtime> {
    vectors_flat: Vec<T>,
    dim: usize,
    n: usize,
    norms: Vec<T>,
    metric: Dist,
    centroids: Vec<T>,
    all_indices: Vec<usize>,
    offsets: Vec<usize>,
    nlist: usize,
    device: R::Device,
}

impl<T, R> IvfIndexGpu<T, R>
where
    R: Runtime,
    T: Float + Sum + cubecl::frontend::Float + cubecl::CubeElement + FromPrimitive,
{
    /// Build an IVF index with GPU support
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        vectors_flat: Vec<T>,
        dim: usize,
        n: usize,
        norms: Vec<T>,
        metric: Dist,
        nlist: Option<usize>,
        max_iters: Option<usize>,
        seed: usize,
        verbose: bool,
        device: R::Device,
    ) -> Self {
        assert!(
            dim.is_multiple_of(LINE_SIZE as usize),
            "Dimension {} must be divisible by LINE_SIZE {}",
            dim,
            LINE_SIZE
        );

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
            println!("  Generating IVF-GPU index with {} Voronoi cells.", nlist);
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

        let assignments = assign_all_parallel(&vectors_flat, dim, n, &centroids, nlist, &metric);
        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        Self {
            vectors_flat,
            dim,
            n,
            norms,
            metric,
            centroids,
            all_indices,
            offsets,
            nlist,
            device,
        }
    }

    /// Build a kNN graph using GPU-accelerated distance computation
    ///
    /// Processes cluster-by-cluster for optimal GPU utilisation.
    pub fn build_knn_graph(
        &self,
        k: usize,
        nprobe: Option<usize>,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);

        let client = R::client(&self.device);
        let vec_size = LINE_SIZE as u8;
        let dim_vectorized = self.dim / LINE_SIZE as usize;

        // 1. Precompute centroid neighbours
        if verbose {
            println!("  Precomputing centroid neighbours...");
        }
        let centroid_neighbours = self.compute_centroid_neighbours(nprobe);

        // 2. Prepare result storage
        let mut all_indices: Vec<Vec<usize>> = vec![Vec::new(); self.n];
        let mut all_distances: Vec<Vec<f32>> = vec![Vec::new(); self.n];

        // 3. Process each cluster
        for cluster_id in 0..self.nlist {
            let query_start = self.offsets[cluster_id];
            let query_end = self.offsets[cluster_id + 1];
            let query_indices: Vec<usize> = self.all_indices[query_start..query_end].to_vec();

            if query_indices.is_empty() {
                continue;
            }

            if verbose && cluster_id % 100 == 0 {
                println!("  Processing cluster {} / {}", cluster_id, self.nlist);
            }

            // Gather candidate clusters
            let candidate_clusters: Vec<usize> = std::iter::once(cluster_id)
                .chain(centroid_neighbours[cluster_id].iter().copied())
                .collect();

            // Gather all candidate vector indices
            let candidate_indices: Vec<usize> = candidate_clusters
                .iter()
                .flat_map(|&c| {
                    let start = self.offsets[c];
                    let end = self.offsets[c + 1];
                    self.all_indices[start..end].iter().copied()
                })
                .collect();

            if candidate_indices.is_empty() {
                continue;
            }

            // Gather query vectors
            let query_vectors: Vec<T> = query_indices
                .iter()
                .flat_map(|&idx| {
                    self.vectors_flat[idx * self.dim..(idx + 1) * self.dim]
                        .iter()
                        .copied()
                })
                .collect();

            // Gather candidate vectors
            let candidate_vectors: Vec<T> = candidate_indices
                .iter()
                .flat_map(|&idx| {
                    self.vectors_flat[idx * self.dim..(idx + 1) * self.dim]
                        .iter()
                        .copied()
                })
                .collect();

            let n_queries = query_indices.len();
            let n_candidates = candidate_indices.len();

            // Upload to GPU
            let query_gpu = GpuTensor::<R, T>::from_slice(
                &query_vectors,
                vec![n_queries, dim_vectorized],
                &client,
            );

            let candidate_gpu = GpuTensor::<R, T>::from_slice(
                &candidate_vectors,
                vec![n_candidates, dim_vectorized],
                &client,
            );

            let distances_gpu = GpuTensor::<R, f32>::empty(vec![n_queries, n_candidates], &client);

            let grid_x = (n_candidates as u32).div_ceil(WORKGROUP_SIZE_X);
            let grid_y = (n_queries as u32).div_ceil(WORKGROUP_SIZE_Y);

            // Launch kernel
            match self.metric {
                Dist::Euclidean => unsafe {
                    euclidean_distances_gpu_chunk::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(grid_x, grid_y, 1),
                        CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                        query_gpu.into_tensor_arg(vec_size),
                        candidate_gpu.into_tensor_arg(vec_size),
                        distances_gpu.into_tensor_arg(1),
                    );
                },
                Dist::Cosine => {
                    let query_norms: Vec<T> =
                        query_indices.iter().map(|&idx| self.norms[idx]).collect();
                    let candidate_norms: Vec<T> = candidate_indices
                        .iter()
                        .map(|&idx| self.norms[idx])
                        .collect();

                    let query_norms_gpu =
                        GpuTensor::<R, T>::from_slice(&query_norms, vec![n_queries], &client);
                    let candidate_norms_gpu = GpuTensor::<R, T>::from_slice(
                        &candidate_norms,
                        vec![n_candidates],
                        &client,
                    );

                    unsafe {
                        cosine_distances_gpu_chunk::launch_unchecked::<T, R>(
                            &client,
                            CubeCount::Static(grid_x, grid_y, 1),
                            CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                            query_gpu.into_tensor_arg(vec_size),
                            candidate_gpu.into_tensor_arg(vec_size),
                            query_norms_gpu.into_tensor_arg(1),
                            candidate_norms_gpu.into_tensor_arg(1),
                            distances_gpu.into_tensor_arg(1),
                        );
                    }
                }
            }

            // Read back and process results
            let distances = distances_gpu.read(&client);

            // Extract top-k per query (parallel over queries in this cluster)
            let cluster_results: Vec<(Vec<usize>, Vec<f32>)> = (0..n_queries)
                .into_par_iter()
                .map(|q| {
                    let query_global_idx = query_indices[q];
                    let row_start = q * n_candidates;

                    let mut heap: BinaryHeap<(OrderedFloat<f32>, usize)> =
                        BinaryHeap::with_capacity(k + 1);

                    for (c, &cand_idx) in candidate_indices.iter().enumerate() {
                        if cand_idx == query_global_idx {
                            continue; // skip self
                        }

                        let dist = distances[row_start + c];

                        if heap.len() < k {
                            heap.push((OrderedFloat(dist), cand_idx));
                        } else if dist < heap.peek().unwrap().0 .0 {
                            heap.pop();
                            heap.push((OrderedFloat(dist), cand_idx));
                        }
                    }

                    let mut results: Vec<_> = heap.into_iter().collect();
                    results.sort_unstable_by_key(|&(d, _)| d);

                    let (dists, idxs): (Vec<f32>, Vec<usize>) = results
                        .into_iter()
                        .map(|(OrderedFloat(d), i)| (d, i))
                        .unzip();

                    (idxs, dists)
                })
                .collect();

            // Store results
            for (q, (indices, dists)) in cluster_results.into_iter().enumerate() {
                let global_idx = query_indices[q];
                all_indices[global_idx] = indices;
                all_distances[global_idx] = dists;
            }
        }

        (all_indices, all_distances)
    }

    /// Query the index with a batch of external query vectors
    pub fn query_batch(
        &self,
        query_vectors_flat: &[T],
        n_queries: usize,
        k: usize,
        nprobe: Option<usize>,
    ) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);

        let client = R::client(&self.device);
        let vec_size = LINE_SIZE as u8;
        let dim_vectorized = self.dim / LINE_SIZE as usize;

        // 1. Find top nprobe centroids for each query (CPU)
        let query_clusters: Vec<Vec<usize>> = (0..n_queries)
            .into_par_iter()
            .map(|q| {
                let query = &query_vectors_flat[q * self.dim..(q + 1) * self.dim];
                self.find_nearest_centroids(query, nprobe)
            })
            .collect();

        // Compute query norms if needed
        let query_norms: Vec<T> = if self.metric == Dist::Cosine {
            (0..n_queries)
                .into_par_iter()
                .map(|q| {
                    let start = q * self.dim;
                    query_vectors_flat[start..start + self.dim]
                        .iter()
                        .map(|&x| x * x)
                        .sum::<T>()
                        .sqrt()
                })
                .collect()
        } else {
            Vec::new()
        };

        // 2. Process in batches to control union size
        const BATCH_SIZE: usize = 4;
        let mut all_indices: Vec<Vec<usize>> = Vec::with_capacity(n_queries);
        let mut all_distances: Vec<Vec<f32>> = Vec::with_capacity(n_queries);

        for batch_start in (0..n_queries).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(n_queries);
            let batch_size = batch_end - batch_start;

            // Union of candidate clusters for this batch
            let mut cluster_set = vec![false; self.nlist];
            for q in batch_start..batch_end {
                for &c in &query_clusters[q] {
                    cluster_set[c] = true;
                }
            }

            // Gather candidate indices
            let candidate_indices: Vec<usize> = cluster_set
                .iter()
                .enumerate()
                .filter(|(_, &in_set)| in_set)
                .flat_map(|(c, _)| {
                    let start = self.offsets[c];
                    let end = self.offsets[c + 1];
                    self.all_indices[start..end].iter().copied()
                })
                .collect();

            if candidate_indices.is_empty() {
                for _ in 0..batch_size {
                    all_indices.push(Vec::new());
                    all_distances.push(Vec::new());
                }
                continue;
            }

            // Build validity mask: which candidates are valid for which query
            let candidate_to_cluster: Vec<usize> = cluster_set
                .iter()
                .enumerate()
                .filter(|(_, &in_set)| in_set)
                .flat_map(|(c, _)| {
                    let start = self.offsets[c];
                    let end = self.offsets[c + 1];
                    std::iter::repeat_n(c, end - start)
                })
                .collect();

            let query_cluster_set: Vec<Vec<bool>> = (batch_start..batch_end)
                .map(|q| {
                    let mut set = vec![false; self.nlist];
                    for &c in &query_clusters[q] {
                        set[c] = true;
                    }
                    set
                })
                .collect();

            // Gather vectors
            let batch_queries: Vec<T> =
                query_vectors_flat[batch_start * self.dim..batch_end * self.dim].to_vec();

            let candidate_vectors: Vec<T> = candidate_indices
                .iter()
                .flat_map(|&idx| {
                    self.vectors_flat[idx * self.dim..(idx + 1) * self.dim]
                        .iter()
                        .copied()
                })
                .collect();

            let n_candidates = candidate_indices.len();

            // Upload to GPU
            let query_gpu = GpuTensor::<R, T>::from_slice(
                &batch_queries,
                vec![batch_size, dim_vectorized],
                &client,
            );

            let candidate_gpu = GpuTensor::<R, T>::from_slice(
                &candidate_vectors,
                vec![n_candidates, dim_vectorized],
                &client,
            );

            let distances_gpu = GpuTensor::<R, f32>::empty(vec![batch_size, n_candidates], &client);

            let grid_x = (n_candidates as u32).div_ceil(WORKGROUP_SIZE_X);
            let grid_y = (batch_size as u32).div_ceil(WORKGROUP_SIZE_Y);

            match self.metric {
                Dist::Euclidean => unsafe {
                    euclidean_distances_gpu_chunk::launch_unchecked::<T, R>(
                        &client,
                        CubeCount::Static(grid_x, grid_y, 1),
                        CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                        query_gpu.into_tensor_arg(vec_size),
                        candidate_gpu.into_tensor_arg(vec_size),
                        distances_gpu.into_tensor_arg(1),
                    );
                },
                Dist::Cosine => {
                    let batch_norms = &query_norms[batch_start..batch_end];
                    let candidate_norms: Vec<T> = candidate_indices
                        .iter()
                        .map(|&idx| self.norms[idx])
                        .collect();

                    let query_norms_gpu =
                        GpuTensor::<R, T>::from_slice(batch_norms, vec![batch_size], &client);
                    let candidate_norms_gpu = GpuTensor::<R, T>::from_slice(
                        &candidate_norms,
                        vec![n_candidates],
                        &client,
                    );

                    unsafe {
                        cosine_distances_gpu_chunk::launch_unchecked::<T, R>(
                            &client,
                            CubeCount::Static(grid_x, grid_y, 1),
                            CubeDim::new(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
                            query_gpu.into_tensor_arg(vec_size),
                            candidate_gpu.into_tensor_arg(vec_size),
                            query_norms_gpu.into_tensor_arg(1),
                            candidate_norms_gpu.into_tensor_arg(1),
                            distances_gpu.into_tensor_arg(1),
                        );
                    }
                }
            }

            let distances = distances_gpu.read(&client);

            // Extract top-k per query, filtering by validity
            let batch_results: Vec<(Vec<usize>, Vec<f32>)> = (0..batch_size)
                .into_par_iter()
                .map(|q| {
                    let row_start = q * n_candidates;
                    let valid_clusters = &query_cluster_set[q];

                    let mut heap: BinaryHeap<(OrderedFloat<f32>, usize)> =
                        BinaryHeap::with_capacity(k + 1);

                    for (c, &cand_idx) in candidate_indices.iter().enumerate() {
                        let cand_cluster = candidate_to_cluster[c];
                        if !valid_clusters[cand_cluster] {
                            continue;
                        }

                        let dist = distances[row_start + c];

                        if heap.len() < k {
                            heap.push((OrderedFloat(dist), cand_idx));
                        } else if dist < heap.peek().unwrap().0 .0 {
                            heap.pop();
                            heap.push((OrderedFloat(dist), cand_idx));
                        }
                    }

                    let mut results: Vec<_> = heap.into_iter().collect();
                    results.sort_unstable_by_key(|&(d, _)| d);

                    let (dists, idxs): (Vec<f32>, Vec<usize>) = results
                        .into_iter()
                        .map(|(OrderedFloat(d), i)| (d, i))
                        .unzip();

                    (idxs, dists)
                })
                .collect();

            for (indices, dists) in batch_results {
                all_indices.push(indices);
                all_distances.push(dists);
            }
        }

        (all_indices, all_distances)
    }

    /// Compute nprobe nearest neighbours for each centroid
    fn compute_centroid_neighbours(&self, nprobe: usize) -> Vec<Vec<usize>> {
        (0..self.nlist)
            .into_par_iter()
            .map(|c| {
                let cent = &self.centroids[c * self.dim..(c + 1) * self.dim];
                let mut dists: Vec<(T, usize)> = (0..self.nlist)
                    .filter(|&other| other != c)
                    .map(|other| {
                        let other_cent = &self.centroids[other * self.dim..(other + 1) * self.dim];
                        let dist = match self.metric {
                            Dist::Euclidean => cent
                                .iter()
                                .zip(other_cent.iter())
                                .map(|(&a, &b)| (a - b) * (a - b))
                                .sum::<T>(),
                            Dist::Cosine => {
                                T::one()
                                    - cent
                                        .iter()
                                        .zip(other_cent.iter())
                                        .map(|(&a, &b)| a * b)
                                        .sum::<T>()
                            }
                        };
                        (dist, other)
                    })
                    .collect();

                dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                dists
                    .into_iter()
                    .take(nprobe.saturating_sub(1))
                    .map(|(_, idx)| idx)
                    .collect()
            })
            .collect()
    }

    /// Find nprobe nearest centroids to a query vector
    fn find_nearest_centroids(&self, query: &[T], nprobe: usize) -> Vec<usize> {
        let mut dists: Vec<(T, usize)> = (0..self.nlist)
            .map(|c| {
                let cent = &self.centroids[c * self.dim..(c + 1) * self.dim];
                let dist = match self.metric {
                    Dist::Euclidean => query
                        .iter()
                        .zip(cent.iter())
                        .map(|(&a, &b)| (a - b) * (a - b))
                        .sum::<T>(),
                    Dist::Cosine => {
                        T::one()
                            - query
                                .iter()
                                .zip(cent.iter())
                                .map(|(&a, &b)| a * b)
                                .sum::<T>()
                    }
                };
                (dist, c)
            })
            .collect();

        if nprobe < self.nlist {
            dists.select_nth_unstable_by(nprobe, |a, b| a.0.partial_cmp(&b.0).unwrap());
        }

        dists.into_iter().take(nprobe).map(|(_, c)| c).collect()
    }
}
