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

/// Maximum number of queries processed in a single GPU batch to avoid
/// exhausting VRAM
const IVF_GPU_QUERY_BATCH_SIZE: usize = 100_000;
/// Target maximum size for the candidate buffer in megabytes
const TARGET_BUFFER_MB: usize = 1500;
/// Divisor setting the slack on the reused candidate scratch buffer.
///
/// `max_candidates` drifts by a few percent between query batches, so sizing
/// the buffer exactly to the first batch makes later ones reallocate and pay
/// the page-fault cost again. 4 gives 25% headroom, enough to absorb the drift
/// seen at 150k x 32D without a meaningful VRAM penalty.
const CANDIDATE_SCRATCH_HEADROOM_DIV: usize = 4;

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

/// Reusable GPU scratch for the IVF candidate buffers
///
/// The mega kernel's first write to a fresh allocation faults its pages in,
/// which measures ~39 ms per call at 15k queries and dominates the kernel's own
/// ~22 ms. Holding the buffers across batches confines that to the first batch.
struct CandidateScratch<R: Runtime, T: AnnSearchFloat + AnnSearchGpuFloat> {
    /// Candidate distances, flat; viewed as `[n_queries, max_candidates]`
    dists: GpuTensor<R, T>,
    /// Candidate indices, flat; viewed as `[n_queries, max_candidates]`
    indices: GpuTensor<R, u32>,
    /// Element capacity of each buffer
    capacity: usize,
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T, R> DimensionValidation for IvfIndexGpu<T, R>
where
    R: Runtime,
    T: AnnSearchGpuFloat + AnnSearchFloat,
{
    // needs to be allowed here, because dim_padded is the relevant dim for GPU
    // indices
    #[allow(clippy::misnamed_getters)]
    fn dim(&self) -> usize {
        self.dim_padded
    }
}

/////////////////////////
// Main implementation //
/////////////////////////

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
    /// * `k_means_params` - Optional k-means trainings parameters, see
    ///   [KMeansTrainingParams]. If not provided, will default to sensible
    ///   defaults.
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
        k_means_params: Option<KMeansTrainingParams>,
        seed: usize,
        verbose: bool,
        device: R::Device,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (vectors_flat, n, dim) = matrix_to_flat(data);

        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        let line = LINE_SIZE;
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
            k_means_params,
            seed,
            verbose,
        )?;

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

        Ok(Self {
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
        })
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
    ) -> KnnResult<T> {
        self.check_dim(dim_query)?;

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
            let mut scratch = None;
            let res =
                self.query_batch_internal(queries_flat, n_queries, k, nprobe, client, &mut scratch)?;

            return Ok(res);
        }

        let mut all_indices = Vec::with_capacity(n_queries);
        let mut all_distances = Vec::with_capacity(n_queries);
        let mut scratch: Option<CandidateScratch<R, T>> = None;

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

            let (batch_indices, batch_dists) = self.query_batch_internal(
                batch_queries,
                batch_size,
                k,
                nprobe,
                client,
                &mut scratch,
            )?;

            all_indices.extend(batch_indices);
            all_distances.extend(batch_dists);
        }

        Ok((all_indices, all_distances))
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
    ) -> KnnResult<T> {
        let (queries_flat, n_queries, dim_query) = matrix_to_flat(query_mat);
        self.check_dim(dim_query)?;

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
        )?;

        client.memory_cleanup();
        Ok((indices, dist))
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
    ) -> KnnOptionResult<T> {
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
        )?;

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
            Ok((indices, Some(dist)))
        } else {
            Ok((indices, None))
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
        scratch: &mut Option<CandidateScratch<R, T>>,
    ) -> KnnResult<T> {
        let vec_size = LINE_SIZE;
        let dim_lines = self.dim_padded / vec_size;

        let safe_worksize_y = pick_wg_y(self.dim_padded)?;

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
        let (grid_y, grid_z) = grid_2d((n_queries as u32).div_ceil(safe_worksize_y));
        let reg_grid_x = (self.nlist as u32).div_ceil(WORKGROUP_SIZE_X * TILE_D as u32);

        match self.metric {
            Dist::SquaredEuclidean if tile_fits(safe_worksize_y) => unsafe {
                euclidean_tiled_reg::launch_unchecked::<T, R>(
                    client,
                    CubeCount::Static(reg_grid_x, grid_y, grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, safe_worksize_y / TILE_Q as u32),
                    vec_size,
                    queries_gpu.clone().into_tensor_arg(),
                    self.centroids_gpu.clone().into_tensor_arg(),
                    centroid_dists_gpu.into_tensor_arg(),
                    0u32,
                    self.nlist as u32,
                    n_queries as u32,
                    self.nlist as u32,
                    dim_lines,
                    safe_worksize_y,
                    TILE_D,
                    TILE_Q,
                );
            },
            Dist::Cosine if tile_fits(safe_worksize_y) => unsafe {
                cosine_tiled_reg::launch_unchecked::<T, R>(
                    client,
                    CubeCount::Static(reg_grid_x, grid_y, grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, safe_worksize_y / TILE_Q as u32),
                    vec_size,
                    queries_gpu.clone().into_tensor_arg(),
                    self.centroids_gpu.clone().into_tensor_arg(),
                    query_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(),
                    self.centroid_norms_gpu
                        .as_ref()
                        .unwrap()
                        .clone()
                        .into_tensor_arg(),
                    centroid_dists_gpu.into_tensor_arg(),
                    0u32,
                    self.nlist as u32,
                    n_queries as u32,
                    self.nlist as u32,
                    dim_lines,
                    safe_worksize_y,
                    TILE_D,
                    TILE_Q,
                );
            },
            Dist::SquaredEuclidean => unsafe {
                euclidean_tiled::launch_unchecked::<T, R>(
                    client,
                    CubeCount::Static(grid_x, grid_y, grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, safe_worksize_y),
                    vec_size,
                    queries_gpu.clone().into_tensor_arg(),
                    self.centroids_gpu.clone().into_tensor_arg(),
                    centroid_dists_gpu.into_tensor_arg(),
                    0u32,
                    self.nlist as u32,
                    n_queries as u32,
                    self.nlist as u32,
                    dim_lines,
                    safe_worksize_y,
                );
            },
            Dist::Cosine => unsafe {
                cosine_tiled::launch_unchecked::<T, R>(
                    client,
                    CubeCount::Static(grid_x, grid_y, grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, safe_worksize_y),
                    vec_size,
                    queries_gpu.clone().into_tensor_arg(),
                    self.centroids_gpu.clone().into_tensor_arg(),
                    query_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(),
                    self.centroid_norms_gpu
                        .as_ref()
                        .unwrap()
                        .clone()
                        .into_tensor_arg(),
                    centroid_dists_gpu.into_tensor_arg(),
                    0u32,
                    self.nlist as u32,
                    n_queries as u32,
                    self.nlist as u32,
                    dim_lines,
                    safe_worksize_y,
                );
            },
            Dist::Manhattan => unreachable!(),
        }

        let centroid_dists = centroid_dists_gpu.read(client)?;

        // Per-query top-nprobe selection on CPU, expanded until reachable >= k.
        // The downstream mega-kernel already handles ragged per-query candidate
        // counts (see `cpu_write_pointers` / `max_candidates`), so variable
        // probe-list lengths cost us nothing extra there.
        let probe_lists: Vec<Vec<usize>> = (0..n_queries)
            .into_par_iter()
            .map(|q| {
                let row_start = q * self.nlist;
                let mut cluster_dists: Vec<(T, usize)> = (0..self.nlist)
                    .map(|c| (centroid_dists[row_start + c], c))
                    .collect();
                select_probed_clusters(
                    &mut cluster_dists,
                    &self.cluster_offsets,
                    nprobe,
                    k,
                )
            })
            .collect();

        let mut cpu_write_pointers = vec![0u32; n_queries];
        let mut max_db_count = 0u32;

        // Build the four device-bound arrays directly rather than a Vec of
        // tuples plus four map-collect passes. The outer loop runs q_idx
        // ascending and writes it as the task's query id, so the task list is
        // already grouped by query: the `sort_unstable_by_key(|t| t.0)` that
        // used to sit here was a no-op on already-ordered data, and being
        // unstable it could only permute tasks within a query. Each task
        // carries its own write offset, so that ordering never mattered.
        let n_tasks_upper: usize = probe_lists.iter().map(|p| p.len()).sum();
        let mut task_q_idx: Vec<u32> = Vec::with_capacity(n_tasks_upper);
        let mut task_db_start: Vec<u32> = Vec::with_capacity(n_tasks_upper);
        let mut task_write_offset: Vec<u32> = Vec::with_capacity(n_tasks_upper);
        let mut task_db_count: Vec<u32> = Vec::with_capacity(n_tasks_upper);

        for q_idx in 0..n_queries {
            for &c in &probe_lists[q_idx] {
                let start = self.cluster_offsets[c];
                let count = self.cluster_offsets[c + 1] - start;

                if count > 0 {
                    task_q_idx.push(q_idx as u32);
                    task_db_start.push(start as u32);
                    task_write_offset.push(cpu_write_pointers[q_idx]);
                    task_db_count.push(count as u32);

                    cpu_write_pointers[q_idx] += count as u32;
                    if count as u32 > max_db_count {
                        max_db_count = count as u32;
                    }
                }
            }
        }

        let n_tasks = task_q_idx.len();
        if n_tasks == 0 {
            return Ok((vec![vec![]; n_queries], vec![vec![]; n_queries]));
        }

        // Group tasks by cluster, not by query. `UNIT_POS_Y` in the mega kernel
        // binds one task per row, so a cube's rows read whichever DB regions
        // their tasks point at. Ordered by query, a cube is one query against
        // `wg_y` different clusters: every row reads a disjoint DB region and
        // there is no reuse. Ordered by cluster it is `wg_y` different queries
        // against one cluster, so the DB tile is read once and reused across
        // rows, which is what makes `euclidean_tiled` fast.
        //
        // Reordering is safe: each task carries its own `task_write_offset`, so
        // where its results land does not depend on task order.
        let mut order: Vec<u32> = (0..n_tasks as u32).collect();
        order.sort_unstable_by_key(|&i| task_db_start[i as usize]);
        let permute = |src: &[u32]| -> Vec<u32> {
            order.iter().map(|&i| src[i as usize]).collect()
        };
        let task_q_idx = permute(&task_q_idx);
        let task_db_start = permute(&task_db_start);
        let task_write_offset = permute(&task_write_offset);
        let task_db_count = permute(&task_db_count);

        let max_candidates: usize = cpu_write_pointers
            .iter()
            .fold(0, |acc, &x| acc.max(x as usize));

        // Reuse the candidate buffers across batches. A fresh allocation costs
        // ~39 ms per call here (measured: an identical second launch over the
        // same buffers runs in 22.3 ms against 61.1 ms for the first), because
        // the kernel's first write faults in ~1.4 GB. Only the first batch
        // should pay that.
        let needed = n_queries * max_candidates;
        let reuse = scratch.as_ref().is_some_and(|s| s.capacity >= needed);
        if !reuse {
            // Over-allocate so that the batch-to-batch drift in
            // `max_candidates` does not force a reallocation, which would
            // reintroduce the fault this buffer exists to avoid.
            let capacity = needed + needed / CANDIDATE_SCRATCH_HEADROOM_DIV;
            *scratch = Some(CandidateScratch {
                dists: GpuTensor::<R, T>::empty(vec![capacity], client),
                indices: GpuTensor::<R, u32>::empty(vec![capacity], client),
                capacity,
            });
        }
        let held = scratch.as_ref().expect("scratch was just populated");
        let candidate_dists_gpu = held.dists.reshaped_view(vec![n_queries, max_candidates]);
        let candidate_indices_gpu = held.indices.reshaped_view(vec![n_queries, max_candidates]);

        let task_q_idx_gpu = GpuTensor::<R, u32>::from_slice(&task_q_idx, vec![n_tasks], client);
        let task_db_start_gpu =
            GpuTensor::<R, u32>::from_slice(&task_db_start, vec![n_tasks], client);
        let task_write_offset_gpu =
            GpuTensor::<R, u32>::from_slice(&task_write_offset, vec![n_tasks], client);
        let task_db_count_gpu =
            GpuTensor::<R, u32>::from_slice(&task_db_count, vec![n_tasks], client);

        // NOT register-tiled, deliberately, and this has now been measured
        // twice. Register-tiled variants (TILE_D DB vectors per thread) were
        // written, measured and removed both times:
        //
        //   * before the task list was grouped by cluster: 1.8x SLOWER
        //     (72.97 -> 134.39 ms/launch, IVF bench)
        //   * after grouping, when the DB tile is reused across cube rows and
        //     the same tiling wins 1.4-2.2x on the exhaustive kernels:
        //     22.60 -> 22.91 ms/launch, i.e. no change (gridsearch, 150k x 32D)
        //
        // An early exit for threads past the cluster end made no difference
        // either, so it is not wasted tail work. Whatever bounds this kernel,
        // it is not memory-operation issue count. Do not retry without a new
        // hypothesis and a measurement.
        let mega_grid_x = max_db_count.div_ceil(WORKGROUP_SIZE_X).max(1);
        let (mega_grid_y, mega_grid_z) = grid_2d((n_tasks as u32).div_ceil(safe_worksize_y));

        match self.metric {
            Dist::SquaredEuclidean => unsafe {
                compute_ivf_mega_euclidean_cached::launch_unchecked::<T, R>(
                    client,
                    CubeCount::Static(mega_grid_x, mega_grid_y, mega_grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, safe_worksize_y),
                    vec_size,
                    queries_gpu.clone().into_tensor_arg(),
                    self.vectors_gpu.clone().into_tensor_arg(),
                    task_q_idx_gpu.into_tensor_arg(),
                    task_db_start_gpu.into_tensor_arg(),
                    task_write_offset_gpu.into_tensor_arg(),
                    task_db_count_gpu.into_tensor_arg(),
                    candidate_dists_gpu.clone().into_tensor_arg(),
                    candidate_indices_gpu.clone().into_tensor_arg(),
                    n_tasks as u32,
                    dim_lines,
                    safe_worksize_y,
                );
            },
            Dist::Cosine => unsafe {
                compute_ivf_mega_cosine_cached::launch_unchecked::<T, R>(
                    client,
                    CubeCount::Static(mega_grid_x, mega_grid_y, mega_grid_z),
                    CubeDim::new_2d(WORKGROUP_SIZE_X, safe_worksize_y),
                    vec_size,
                    queries_gpu.clone().into_tensor_arg(),
                    self.vectors_gpu.clone().into_tensor_arg(),
                    query_norms_gpu.as_ref().unwrap().clone().into_tensor_arg(),
                    self.norms_gpu.as_ref().unwrap().clone().into_tensor_arg(),
                    task_q_idx_gpu.into_tensor_arg(),
                    task_db_start_gpu.into_tensor_arg(),
                    task_write_offset_gpu.into_tensor_arg(),
                    task_db_count_gpu.into_tensor_arg(),
                    candidate_dists_gpu.clone().into_tensor_arg(),
                    candidate_indices_gpu.clone().into_tensor_arg(),
                    n_tasks as u32,
                    dim_lines,
                    safe_worksize_y,
                );
            },
            Dist::Manhattan => unreachable!(),
        }

        let topk_dists = GpuTensor::<R, T>::empty(vec![n_queries, k], client);
        let topk_indices = GpuTensor::<R, u32>::empty(vec![n_queries, k], client);

        let cpq = GpuTensor::<R, u32>::from_slice(&cpu_write_pointers, vec![n_queries], client);
        let (coal_gx, coal_gy) = grid_2d(n_queries as u32);
        let merge = plan_topk_merge(
            k,
            size_of::<T>(),
            client.properties().hardware.max_shared_memory_size,
        )?;
        unsafe {
            reduce_ivf_topk_coalesced::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(coal_gx, coal_gy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                candidate_dists_gpu.clone().into_tensor_arg(),
                candidate_indices_gpu.clone().into_tensor_arg(),
                cpq.into_tensor_arg(),
                topk_dists.clone().into_tensor_arg(),
                topk_indices.clone().into_tensor_arg(),
                k as u32,
                k,
                merge.group,
                merge.single_round,
                merge.slots,
            );
        }

        let final_dists = topk_dists.read(client)?;
        let final_indices = topk_indices.read(client)?;

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

        Ok((results_indices, results_dists))
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

    fn get_default_k_means() -> Option<KMeansTrainingParams> {
        Some(KMeansTrainingParams::new(10, None, None))
    }

    #[test]
    fn test_ivf_index_build() {
        let device = CpuDevice;

        // 100 samples, 4 dimensions
        let data = Mat::from_fn(100, 4, |i, j| ((i + j) as f32) / 10.0);

        let index = IvfIndexGpu::<f32, CpuRuntime>::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(10),
            get_default_k_means(),
            42,
            false,
            device,
        )
        .unwrap();

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
            Dist::SquaredEuclidean,
            Some(5),
            get_default_k_means(),
            42,
            false,
            device,
        )
        .unwrap();

        let query = Mat::from_fn(3, 4, |i, j| if i == j { 1.0_f32 } else { 0.0_f32 });

        let (indices, distances) = index
            .query_batch(query.as_ref(), 5, Some(3), None, false)
            .unwrap();

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
            get_default_k_means(),
            42,
            false,
            device,
        )
        .unwrap();

        let query = Mat::from_fn(2, 4, |_, _| 1.0_f32);
        let (indices, distances) = index
            .query_batch(query.as_ref(), 3, Some(2), None, false)
            .unwrap();

        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0].len(), 3);
        assert!(distances[0][0] >= 0.0);
    }

    #[test]
    fn test_ivf_gpu_expands_nprobe_when_probed_cells_underfill_k() {
        // 50 vectors across 20 cells -> avg ~2 per cell. nprobe=1 alone cannot
        // reach k=10; the per-query CPU-side selection must expand.
        let device = CpuDevice;

        let data = Mat::from_fn(50, 4, |i, j| ((i + j) as f32) / 10.0);
        let index = IvfIndexGpu::<f32, CpuRuntime>::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(20),
            get_default_k_means(),
            42,
            false,
            device,
        )
        .unwrap();

        let query = Mat::from_fn(1, 4, |_, _| 0.0_f32);
        let (indices, distances) = index
            .query_batch(query.as_ref(), 10, Some(1), None, false)
            .unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].len(), 10);
        assert_eq!(distances[0].len(), 10);
    }

    #[test]
    fn test_reorganise_by_cluster() {
        let vectors: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let assignments = vec![0, 1, 0, 1];

        let (reorg, indices, offsets, _) =
            reorganise_by_cluster(&vectors, 4, 4, &assignments, 2, &Dist::SquaredEuclidean);

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

    fn get_default_k_means() -> Option<KMeansTrainingParams> {
        Some(KMeansTrainingParams::new(10, None, None))
    }

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
            Dist::SquaredEuclidean,
            Some(5),
            get_default_k_means(),
            42,
            false,
            device,
        )
        .unwrap();

        let (indices, distances) = index.generate_knn(4, Some(3), None, true, false).unwrap();

        assert_eq!(indices.len(), 30);
        assert!(distances.is_some());
        assert_eq!(distances.unwrap().len(), 30);
    }

    /// Regression test for the tsne_gpu / IVF Linux CI crash.
    ///
    /// The original failure was `index out of bounds: the len is 15000
    /// but the index is 1109321353` inside `original_indices[reorg_idx]`.
    /// `0x4218E949 = 1109321353` is the f32 bit pattern of ~38.24, i.e.
    /// a plausible squared Euclidean distance — a distance value was
    /// leaking into an index slot.
    ///
    /// The trigger is the codegen-buggy reducer pattern (runtime `while`
    /// with u32 counters + `bool` flags on cubecl 0.10 / lavapipe). This
    /// test exercises the same shape as the production failure at
    /// CI-friendly scale.
    #[test]
    fn test_ivf_generate_knn_clustered() {
        let Some(device) = try_device() else {
            eprintln!("Skipping test: no wgpu backend available");
            return;
        };

        // Match failure shape: k=31 (perplexity 10 → k = 3 * 10 + 1),
        // dim=32, planted clusters, SquaredEuclidean, self-query.
        let n = 2000usize;
        let dim = 32usize;
        let n_clusters = 30usize;
        let k = 31usize;

        // Deterministic per-sample cluster + jitter. Avoid rand crate.
        let data = Mat::from_fn(n, dim, |i, j| {
            let cluster = (i % n_clusters) as f32;
            let jitter_bits = ((i.wrapping_mul(2654435761) ^ j.wrapping_mul(40503)) & 0xffff) as f32;
            let jitter = (jitter_bits / 65536.0 - 0.5) * 0.3;
            cluster + jitter
        });

        let index = IvfIndexGpu::<f32, WgpuRuntime>::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(40),
            get_default_k_means(),
            42,
            false,
            device,
        )
        .unwrap();

        // Any panic below (OOB, wrong indices) indicates the reducer
        // regression is back.
        let (indices, distances) = index
            .generate_knn(k, Some(6), None, true, false)
            .unwrap();

        assert_eq!(indices.len(), n);
        let distances = distances.unwrap();
        assert_eq!(distances.len(), n);

        for (q, row) in indices.iter().enumerate() {
            assert!(!row.is_empty(), "query {q} returned zero neighbours");
            for &idx in row {
                assert!(
                    idx < n,
                    "query {q}: OOB neighbour index {idx} (n = {n})"
                );
            }
        }

        // Sanity check: with 30 planted clusters and k=31, most neighbours
        // for a given query should share its cluster. Well below the ideal
        // to stay robust against IVF miss rate, but far above chance.
        let mut same_cluster = 0usize;
        let mut total = 0usize;
        for (q, row) in indices.iter().enumerate() {
            let q_cluster = q % n_clusters;
            for &idx in row {
                if idx % n_clusters == q_cluster {
                    same_cluster += 1;
                }
                total += 1;
            }
        }
        let frac = same_cluster as f32 / total as f32;
        assert!(
            frac > 0.5,
            "same-cluster fraction {frac} too low ({same_cluster}/{total}); reducer likely returning garbage indices"
        );
    }
}
