//! Inverted file index. Leverages k-means clustering to partition data into
//! Voronoi cells that are being searched during querying.

use faer::{Accum, MatMut, MatRef, Par, RowRef};
use rayon::prelude::*;
use thousands::Separable;

use crate::prelude::*;
use crate::utils::dist::cosine_from_dot;
use crate::utils::gemm::gemm;
use crate::utils::k_means_utils::*;
use crate::utils::verbosity::{BuildLog, QueryProgress};
use crate::utils::*;
use std::time::Instant;

/// Bounds on the queries per block of the batched search. Queries in a block
/// are ordered by nearest centroid, so they share most probed cells and each
/// cell is read once per block instead of once per query: larger blocks mean
/// larger GEMMs. The block is the batch split evenly over the threads,
/// clamped to these bounds, so a small batch still keeps every thread busy.
const MIN_QUERY_BLOCK: usize = 256;
/// See [`MIN_QUERY_BLOCK`]. Past this the gain flattens while the per-block
/// pair list and heaps keep growing.
const MAX_QUERY_BLOCK: usize = 4096;

/// Fewest queries sharing a cell in a block before that cell is scored with a
/// GEMM. Below it the gather and the GEMM set-up cost more than the four-row
/// SIMD scan saves.
const GEMM_MIN_GROUP: usize = 8;

/// Fewest queries for which [`IvfIndex::query_batch`] takes the batched path.
/// Below it the per-call `|x|^2` pass over the whole index and the probe
/// pre-pass are not amortised. Not tuned.
const BATCH_MIN_QUERIES: usize = 1024;

/// Extra candidates kept per query on the Euclidean batched path, rescored
/// exactly with the rest and then dropped. Covers neighbours that
/// `|q|^2 + |x|^2 - 2 q.x` misranks across the k-th distance in `f32`.
const RERANK_EXTRA: usize = 8;

/// Rows of a cell per GEMM tile, bounding the distance tile the heaps scan.
/// Flat across the sizes tried, so not critical.
const CELL_ROW_TILE: usize = 512;

/////////////
// IvfIndex//
/////////////

/// IVF (Inverted File) index for similarity search
///
/// Uses k-means clustering to partition vectors into nlist clusters. Each
/// cluster maintains an inverted list of vector indices assigned to it.
/// Queries search only the nprobe nearest clusters, trading perfect recall
/// for speed.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct IvfIndex<T> {
    /// Original vector data, flattened for cache locality
    pub vectors_flat: Vec<T>,
    /// Embedding dimensions
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Pre-computed norms for Cosine distance (empty for Euclidean)
    pub norms: Vec<T>,
    /// The type of distance the index is designed for
    metric: Dist,
    /// Cluster centres (nlist * dim elements), i.e., the centroids
    centroids: Vec<T>,
    /// The norms of the centroids for Cosine distance calculations
    centroids_norm: Vec<T>,
    /// Vector indices for each cluster (in a flat structure)
    all_indices: Vec<u32>,
    /// Offsets of the elements of each inverted list
    offsets: Vec<u32>,
    /// Number of clusters in the index
    nlist: usize,
    /// Original indices
    original_ids: Vec<u32>,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for IvfIndex<T>
where
    T: AnnSearchFloat,
{
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn norms(&self) -> &[T] {
        &self.norms
    }
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for IvfIndex<T>
where
    T: AnnSearchFloat,
{
    fn centroids(&self) -> &[T] {
        &self.centroids
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> Dist {
        self.metric
    }

    fn nlist(&self) -> usize {
        self.nlist
    }

    fn centroids_norm(&self) -> &[T] {
        &self.centroids_norm
    }
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for IvfIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////////
// Index build //
/////////////////

impl<T> IvfIndex<T>
where
    T: AnnSearchFloat,
{
    /// Build an IVF index with optimised memory layout and parallel training.
    ///
    /// Constructs an inverted file index by clustering vectors with k-means,
    /// then assigns each vector to its nearest centroid. Uses a CSR (Compressed
    /// Sparse Row) layout for cache-efficient cluster traversal during search.
    ///
    /// ### Workflow
    ///
    /// 1. Subsamples 250k vectors for training if dataset exceeds 500k
    /// 2. Runs k-means clustering to find nlist centroids
    /// 3. Assigns all vectors to their nearest centroid in parallel
    /// 4. Builds CSR layout grouping vectors by cluster for locality
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `nlist` - Optional number of clusters. Defaults to `sqrt(n)`.
    /// * `k_means_params` - Optional k-means trainings parameters, see
    ///   [KMeansTrainingParams]. If not provided, will default to sensible
    ///   defaults.
    /// * `seed` - Random seed for reproducibility
    /// * `verbosity` - `Normal` prints start and end, `Detailed` adds timed
    ///   stages
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    pub fn build(
        data: impl AnnMatrix<T>,
        metric: Dist,
        nlist: Option<usize>,
        k_means_params: Option<KMeansTrainingParams>,
        seed: usize,
        verbosity: Verbosity,
    ) -> Result<Self, AnnSearchErrors> {
        let log = BuildLog::new("IVF index", verbosity);
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (vectors_flat, n, dim) = data.into_row_major();

        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    T::calculate_l2_norm(&vectors_flat[start..end])
                })
                .collect()
        } else {
            Vec::new()
        };

        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        // 2. train the centroids
        let start = Instant::now();
        let centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            k_means_params,
            seed,
            Verbosity::Quiet,
        )?;
        if verbosity.detailed_verbosity() {
            println!(
                " Trained k-means ({} centroids, {} samples) in {:.2?}",
                nlist.separate_with_underscores(),
                n_train.separate_with_underscores(),
                start.elapsed()
            );
        }

        let centroids_norm = if metric == Dist::Cosine {
            (0..nlist)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    centroids[start..end]
                        .iter()
                        .map(|x| *x * *x)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt()
                })
                .collect()
        } else {
            Vec::new()
        };

        // 3. assign the Rest
        let data_norms_for_assignment = if metric == Dist::Cosine {
            norms.clone()
        } else {
            vec![T::one(); n]
        };

        let start = Instant::now();
        let assignments = assign_all_parallel(
            &vectors_flat,
            &data_norms_for_assignment,
            dim,
            n,
            &centroids,
            &centroids_norm,
            nlist,
            &metric,
        );
        if verbosity.detailed_verbosity() {
            println!(
                " Assigned {} vectors in {:.2?}",
                n.separate_with_underscores(),
                start.elapsed()
            );
        }

        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        // 5. optimise memory layout for access pattern
        let mut idx = Self {
            vectors_flat,
            dim,
            n,
            norms,
            metric,
            centroids,
            centroids_norm,
            all_indices,
            offsets,
            nlist,
            original_ids: Vec::new(),
        };

        let new_to_old = idx.optimise_memory_layout();
        idx.original_ids = new_to_old;

        log.finish();

        Ok(idx)
    }

    /// Function will optimise memory layout and put vectors sorted by cluster
    /// id
    ///
    /// ### Returns
    ///
    /// New to old mapping
    fn optimise_memory_layout(&mut self) -> Vec<u32> {
        let mut new_to_old = Vec::with_capacity(self.n);

        for cluster in 0..self.nlist {
            let start = self.offsets[cluster] as usize;
            let end = self.offsets[cluster + 1] as usize;
            new_to_old.extend_from_slice(&self.all_indices[start..end]);
        }

        let mut new_vectors_flat = Vec::with_capacity(self.vectors_flat.len());
        let mut new_norms = if self.norms.is_empty() {
            Vec::new()
        } else {
            Vec::with_capacity(self.n)
        };

        for &old_id in &new_to_old {
            let old_id = old_id as usize;
            let start = old_id * self.dim;
            new_vectors_flat.extend_from_slice(&self.vectors_flat[start..start + self.dim]);
            if !self.norms.is_empty() {
                new_norms.push(self.norms[old_id]);
            }
        }

        self.vectors_flat = new_vectors_flat;
        self.norms = new_norms;

        // Offsets index `vectors_flat` directly; `all_indices` is no longer needed
        self.all_indices.clear();
        self.all_indices.shrink_to_fit();

        new_to_old
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Index size `in n bytes`
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
            + self.centroids.capacity() * std::mem::size_of::<T>()
            + self.centroids_norm.capacity() * std::mem::size_of::<T>()
            + self.all_indices.capacity() * std::mem::size_of::<u32>()
            + self.offsets.capacity() * std::mem::size_of::<u32>()
    }
}

///////////
// Query //
///////////

impl<T> IvfIndex<T>
where
    T: AnnSearchFloat,
{
    /// Query the index for approximate nearest neighbours
    ///
    /// Performs two-stage search: first finds nprobe nearest centroids to the
    /// query, then exhaustively searches all vectors in those clusters. Uses
    /// a max-heap to track top-k candidates.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search. A good default here is
    ///   `sqrt(nlist)`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
    #[inline]
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        nprobe: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k: usize = k.min(self.n);
        let query_norm = if matches!(self.metric, Dist::Cosine) {
            query_vec
                .iter()
                .map(|&v| v * v)
                .fold(T::zero(), |a, b| a + b)
                .sqrt()
        } else {
            T::one()
        };

        // 1. find the top `nprobe` centroids, expanding to guarantee >= k
        //    reachable vectors so short-returning is impossible whenever the
        //    index itself holds >= k vectors.
        let mut cluster_dists: Vec<(T, usize)> =
            self.get_centroids_dist(query_vec, query_norm, nprobe);
        let probed = select_probed_clusters(&mut cluster_dists, &self.offsets, nprobe, k);

        // 2. search only those clusters in the CSR layout.
        //
        // `optimise_memory_layout` sorted the rows by cell, so a cell is one
        // contiguous slab and the scan can take four rows at a time with no
        // gather. The metric is resolved once per query rather than per row.
        let mut heap = BoundedMaxHeap::new(k);

        for cluster_idx in probed {
            let start = self.offsets[cluster_idx] as usize;
            let end = self.offsets[cluster_idx + 1] as usize;
            self.scan_cell(start, end, query_vec, query_norm, &mut heap);
        }

        heap.sort();

        let distances = heap.dists().to_vec();
        let indices = heap
            .ids()
            .iter()
            .map(|i| self.original_ids[*i] as usize)
            .collect();

        Ok((indices, distances))
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies
    /// to a temporary vector. Uses `self.query()` under the hood.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search. A good default here is
    ///   `sqrt(nlist)`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        nprobe: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, nprobe);
        }

        // fallback
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, nprobe)
    }

    /// Query the index with a batch of vectors
    ///
    /// From [`BATCH_MIN_QUERIES`] queries up, runs the cell-grouped batched
    /// search, otherwise one [`IvfIndex::query`] per query in parallel.
    ///
    /// ### Params
    ///
    /// * `queries` - Row-major queries, `nq * query_dim`
    /// * `query_dim` - Query dimensionality, must match the index
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search (defaults to sqrt(nlist))
    /// * `return_dist` - Whether to return distances
    /// * `verbosity` - Prints start, progress and end at `Normal`
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`, one row per query
    pub fn query_batch(
        &self,
        queries: &[T],
        query_dim: usize,
        k: usize,
        nprobe: Option<usize>,
        return_dist: bool,
        verbosity: Verbosity,
    ) -> KnnOptionResult<T> {
        self.check_dim(query_dim)?;
        let nq = queries.len() / query_dim;
        let progress = QueryProgress::new("IVF index", nq, false, verbosity);
        let results = if nq >= BATCH_MIN_QUERIES {
            self.search_batched(queries, nq, k, nprobe, &progress)
        } else {
            queries
                .par_chunks_exact(query_dim)
                .map(|q| {
                    progress.tick();
                    self.query(q, k, nprobe)
                })
                .collect::<Result<Vec<_>, AnnSearchErrors>>()?
        };
        progress.finish();
        Ok(pack_knn_results(results, return_dist))
    }

    /// Batched search for many queries at once
    ///
    /// Probes every query as [`IvfIndex::query`] does, then orders the queries
    /// by nearest centroid and works through them in blocks of
    /// `nq / threads`, clamped to [`MIN_QUERY_BLOCK`] and [`MAX_QUERY_BLOCK`].
    /// Within a block the `(cell, query)` pairs are sorted by
    /// cell, so each probed cell is visited once: when at least
    /// [`GEMM_MIN_GROUP`] queries want it, their dot products against the
    /// cell's contiguous rows come from one GEMM, otherwise from the four-row
    /// SIMD scan. Euclidean scores with `|q|^2 + |x|^2 - 2 q.x`, keeps
    /// [`RERANK_EXTRA`] candidates past `k`, and recomputes those exactly
    /// before cutting to `k`, so returned distances and their order carry no
    /// cancellation error.
    ///
    /// ### Params
    ///
    /// * `queries` - Row-major queries, `nq * dim`
    /// * `nq` - Number of queries
    /// * `k` - Number of neighbours
    /// * `nprobe` - Number of clusters to search (defaults to sqrt(nlist))
    /// * `progress` - Ticked once per query
    ///
    /// ### Returns
    ///
    /// Per query, in input order, `(original ids, distances)` sorted nearest
    /// first.
    fn search_batched(
        &self,
        queries: &[T],
        nq: usize,
        k: usize,
        nprobe: Option<usize>,
        progress: &QueryProgress,
    ) -> Vec<(Vec<usize>, Vec<T>)> {
        let dim = self.dim;
        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k = k.min(self.n);
        let cosine = matches!(self.metric, Dist::Cosine);
        let two = T::one() + T::one();
        // Euclidean keeps a margin past `k` for the exact rerank, so expansion
        // error near the k-th distance cannot drop a true neighbour
        let k_heap = if cosine {
            k
        } else {
            (k + RERANK_EXTRA).min(self.n)
        };

        // per query: |q|^2 for Euclidean, |q| for Cosine
        let q_norm: Vec<T> = queries[..nq * dim]
            .par_chunks_exact(dim)
            .map(|q| {
                let sq = T::dot_simd(q, q);
                if cosine {
                    sq.sqrt()
                } else {
                    sq
                }
            })
            .collect();
        // |x|^2 per row for the Euclidean expansion; Cosine has `norms`
        let x_sq: Vec<T> = if cosine {
            Vec::new()
        } else {
            self.vectors_flat
                .par_chunks_exact(dim)
                .map(|x| T::dot_simd(x, x))
                .collect()
        };

        let probes: Vec<Vec<u32>> = (0..nq)
            .into_par_iter()
            .map(|i| {
                let q = &queries[i * dim..(i + 1) * dim];
                let qn = if cosine { q_norm[i] } else { T::one() };
                let mut cd = self.get_centroids_dist(q, qn, nprobe);
                select_probed_clusters(&mut cd, &self.offsets, nprobe, k)
                    .into_iter()
                    .map(|c| c as u32)
                    .collect()
            })
            .collect();

        let block_size =
            (nq / rayon::current_num_threads().max(1)).clamp(MIN_QUERY_BLOCK, MAX_QUERY_BLOCK);
        let mut order: Vec<u32> = (0..nq as u32).collect();
        order.par_sort_by_key(|&i| probes[i as usize][0]);

        let block_results: Vec<Vec<(u32, Vec<usize>, Vec<T>)>> = order
            .par_chunks(block_size)
            .map_init(
                || (Vec::new(), Vec::new(), Vec::new()),
                |(pairs, q_tile, d_tile): &mut (Vec<u64>, Vec<T>, Vec<T>), block| {
                    let mut heaps: Vec<BoundedMaxHeap<T>> = (0..block.len())
                        .map(|_| BoundedMaxHeap::new(k_heap))
                        .collect();

                    pairs.clear();
                    for (bi, &qi) in block.iter().enumerate() {
                        for &c in &probes[qi as usize] {
                            pairs.push(((c as u64) << 32) | bi as u64);
                        }
                    }
                    pairs.sort_unstable();

                    let mut g0 = 0;
                    while g0 < pairs.len() {
                        let cell = (pairs[g0] >> 32) as usize;
                        let mut g1 = g0 + 1;
                        while g1 < pairs.len() && (pairs[g1] >> 32) as usize == cell {
                            g1 += 1;
                        }
                        let group = &pairs[g0..g1];
                        let start = self.offsets[cell] as usize;
                        let end = self.offsets[cell + 1] as usize;

                        if group.len() < GEMM_MIN_GROUP {
                            for &pk in group {
                                let bi = (pk & 0xFFFF_FFFF) as usize;
                                let qi = block[bi] as usize;
                                let q = &queries[qi * dim..(qi + 1) * dim];
                                self.scan_cell(start, end, q, q_norm[qi], &mut heaps[bi]);
                            }
                        } else {
                            let g = group.len();
                            q_tile.clear();
                            for &pk in group {
                                let qi = block[(pk & 0xFFFF_FFFF) as usize] as usize;
                                q_tile.extend_from_slice(&queries[qi * dim..(qi + 1) * dim]);
                            }
                            let q_mat = MatRef::from_row_major_slice(q_tile, g, dim);
                            for r0 in (start..end).step_by(CELL_ROW_TILE) {
                                let r1 = (r0 + CELL_ROW_TILE).min(end);
                                let rows = r1 - r0;
                                d_tile.resize(g * rows, T::zero());
                                gemm(
                                    MatMut::from_row_major_slice_mut(
                                        &mut d_tile[..g * rows],
                                        g,
                                        rows,
                                    ),
                                    Accum::Replace,
                                    q_mat,
                                    MatRef::from_row_major_slice(
                                        &self.vectors_flat[r0 * dim..r1 * dim],
                                        rows,
                                        dim,
                                    )
                                    .transpose(),
                                    T::one(),
                                    Par::Seq,
                                );
                                for (gi, &pk) in group.iter().enumerate() {
                                    let bi = (pk & 0xFFFF_FFFF) as usize;
                                    let qn = q_norm[block[bi] as usize];
                                    let heap = &mut heaps[bi];
                                    let dots = &d_tile[gi * rows..(gi + 1) * rows];
                                    if cosine {
                                        for (j, &d) in dots.iter().enumerate() {
                                            let dist = cosine_from_dot(d, qn * self.norms[r0 + j]);
                                            if dist < heap.threshold() {
                                                heap.push(dist, r0 + j);
                                            }
                                        }
                                    } else {
                                        for (j, &d) in dots.iter().enumerate() {
                                            let dist = qn + x_sq[r0 + j] - two * d;
                                            if dist < heap.threshold() {
                                                heap.push(dist, r0 + j);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        g0 = g1;
                    }

                    block
                        .iter()
                        .zip(heaps)
                        .map(|(&qi, mut heap)| {
                            progress.tick();
                            let q = &queries[qi as usize * dim..(qi as usize + 1) * dim];
                            let mut out: Vec<(T, usize)> = if cosine {
                                heap.sort();
                                heap.dists()
                                    .iter()
                                    .copied()
                                    .zip(heap.ids().iter().copied())
                                    .collect()
                            } else {
                                // exact distances for the survivors, then reorder
                                let mut e: Vec<(T, usize)> = heap
                                    .ids()
                                    .iter()
                                    .map(|&id| (self.euclidean_distance_to_query(id, q), id))
                                    .collect();
                                e.sort_unstable_by(|a, b| {
                                    a.0.partial_cmp(&b.0)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                        .then(a.1.cmp(&b.1))
                                });
                                e.truncate(k);
                                e
                            };
                            let ids = out
                                .iter()
                                .map(|&(_, r)| self.original_ids[r] as usize)
                                .collect();
                            let dists = out.drain(..).map(|(d, _)| d).collect();
                            (qi, ids, dists)
                        })
                        .collect()
                },
            )
            .collect();

        let mut results: Vec<(Vec<usize>, Vec<T>)> = vec![(Vec::new(), Vec::new()); nq];
        for block in block_results {
            for (qi, ids, dists) in block {
                results[qi as usize] = (ids, dists);
            }
        }
        results
    }

    /// Scan one cell's contiguous rows into a heap, four rows at a time
    ///
    /// ### Params
    ///
    /// * `start` - First row of the cell
    /// * `end` - One past the last row
    /// * `query_vec` - The query
    /// * `query_norm` - `|q|` for Cosine; ignored for Euclidean
    /// * `heap` - Receives `(distance, internal row)`
    #[inline]
    fn scan_cell(
        &self,
        start: usize,
        end: usize,
        query_vec: &[T],
        query_norm: T,
        heap: &mut BoundedMaxHeap<T>,
    ) {
        let tail = start + (end - start) / 4 * 4;
        match self.metric {
            Dist::SquaredEuclidean => {
                for base in (start..tail).step_by(4) {
                    let dists = self.euclidean_distance_to_query_4(base, query_vec);
                    for k_off in 0..4 {
                        heap.push(dists[k_off], base + k_off);
                    }
                }
                for vec_idx in tail..end {
                    heap.push(
                        self.euclidean_distance_to_query(vec_idx, query_vec),
                        vec_idx,
                    );
                }
            }
            Dist::Cosine => {
                for base in (start..tail).step_by(4) {
                    let dists = self.cosine_distance_to_query_4(base, query_vec, query_norm);
                    for k_off in 0..4 {
                        heap.push(dists[k_off], base + k_off);
                    }
                }
                for vec_idx in tail..end {
                    heap.push(
                        self.cosine_distance_to_query(vec_idx, query_vec, query_norm),
                        vec_idx,
                    );
                }
            }
            Dist::Manhattan => unreachable!(),
        }
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete
    /// kNN graph.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `nprobe` - Number of clusters to search (defaults to sqrt(nlist) if None)
    /// * `return_dist` - Whether to return distances
    /// * `verbosity` - Prints start, progress and end at `Normal`
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
        verbosity: Verbosity,
    ) -> KnnOptionResult<T> {
        let progress = QueryProgress::new("IVF index", self.n, true, verbosity);
        let results = self.search_batched(&self.vectors_flat, self.n, k, nprobe, &progress);
        progress.finish();

        let mut final_indices = vec![Vec::new(); self.n];
        let mut final_dists = if return_dist {
            Some(vec![Vec::new(); self.n])
        } else {
            None
        };

        // row `i` of the results is internal row `i`
        for (i, (indices, dists)) in results.into_iter().enumerate() {
            let orig_id = self.original_ids[i] as usize;
            final_indices[orig_id] = indices;
            if let Some(ref mut fd) = final_dists {
                fd[orig_id] = dists;
            }
        }

        fix_neg_dist(&mut final_dists);
        Ok((final_indices, final_dists))
    }
}

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for IvfIndex<T>
where
    T: AnnSearchFloat,
{
    fn query_for_validation(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.query(query_vec, k, None)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn n(&self) -> usize {
        self.n
    }

    fn metric(&self) -> Dist {
        self.metric
    }

    fn original_ids(&self) -> &[u32] {
        &self.original_ids
    }
}

///////////
// Tests //
///////////

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for IvfIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "ivf";
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::Mat;

    fn create_simple_matrix() -> Mat<f32> {
        // 5 points in 3D space
        let data = [
            1.0, 0.0, 0.0, // Point 0
            0.0, 1.0, 0.0, // Point 1
            0.0, 0.0, 1.0, // Point 2
            1.0, 1.0, 0.0, // Point 3
            1.0, 0.0, 1.0, // Point 4
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
    }

    #[test]
    fn test_ivf_index_creation() {
        let data = create_simple_matrix();
        let _ = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2), // nlist
            None,
            42,
            Verbosity::Quiet,
        );
    }

    #[test]
    fn test_ivf_query_finds_self() {
        let data = create_simple_matrix();
        let index = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1, None).unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ivf_query_euclidean() {
        let data = create_simple_matrix();
        let index = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None).unwrap();

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_ivf_query_cosine() {
        let data = create_simple_matrix();

        let index = IvfIndex::build(
            data.as_ref(),
            Dist::Cosine,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None).unwrap();

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ivf_query_k_larger_than_dataset() {
        let data = create_simple_matrix();
        let index = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 10, None).unwrap();

        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_ivf_query_nprobe() {
        let data = create_simple_matrix();
        let index = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(3),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices1, _) = index.query(&query, 3, Some(1)).unwrap();
        let (indices2, _) = index.query(&query, 3, Some(2)).unwrap();

        assert!(!indices1.is_empty());
        assert!(!indices2.is_empty());
    }

    #[test]
    fn test_ivf_reproducibility() {
        let data = create_simple_matrix();

        let index1 = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();
        let index2 = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![0.5, 0.5, 0.0];
        let (indices1, _) = index1.query(&query, 3, None).unwrap();
        let (indices2, _) = index2.query(&query, 3, None).unwrap();

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_ivf_different_seeds() {
        let data = create_simple_matrix();

        let index1 = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();
        let index2 = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            123,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![0.5, 0.5, 0.0];
        let (indices1, _) = index1.query(&query, 3, Some(2)).unwrap();
        let (indices2, _) = index2.query(&query, 3, Some(2)).unwrap();

        assert!(!indices1.is_empty());
        assert!(!indices2.is_empty());
    }

    #[test]
    fn test_ivf_larger_dataset() {
        let n = 100;
        let dim = 10;
        let data = Mat::from_fn(n, dim, |i, j| (i * j) as f32 / 10.0);

        let index = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(10), // sqrt(100)
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5, None).unwrap();

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_ivf_orthogonal_vectors() {
        let data = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });

        let index = IvfIndex::build(
            data.as_ref(),
            Dist::Cosine,
            Some(3),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None).unwrap();

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        if indices.len() >= 2 {
            assert_relative_eq!(distances[1], 1.0, epsilon = 1e-5);
        }
        if indices.len() >= 3 {
            assert_relative_eq!(distances[2], 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_ivf_query_returns_full_k_when_nprobe_underfills() {
        // 5 distinct points, one cluster each. nprobe=1 lands on a single
        // singleton cluster. Without the auto-expansion the query would only
        // return 1 result even though k=3 was requested.
        let data = create_simple_matrix();
        let index = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(5),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, Some(1)).unwrap();

        assert_eq!(indices.len(), 3, "expected auto-expanded nprobe to reach k");
        assert_eq!(distances.len(), 3);
        // Nearest is the query point itself.
        assert_eq!(indices[0], 0);
        // Distances must remain sorted ascending after expansion.
        for w in distances.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn test_ivf_more_clusters() {
        let data = create_simple_matrix();

        let index_few = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();
        let index_many = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(4),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![0.9, 0.1, 0.0];
        let (indices1, _) = index_few.query(&query, 3, Some(2)).unwrap();
        let (indices2, _) = index_many.query(&query, 3, Some(4)).unwrap();

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    /// Deterministic pseudo-random rows with no exact distance ties
    fn noisy_matrix(n: usize, dim: usize, seed: u64) -> Mat<f32> {
        let mut state = seed;
        Mat::from_fn(n, dim, |_, _| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
    }

    /// The batched path against one `query` per row
    fn assert_batch_matches_query(metric: Dist) {
        let dim = 24;
        let data = noisy_matrix(4_000, dim, 7);
        let queries = noisy_matrix(BATCH_MIN_QUERIES + 300, dim, 11);
        let index =
            IvfIndex::build(data.as_ref(), metric, Some(30), None, 42, Verbosity::Quiet).unwrap();

        let flat: Vec<f32> = (0..queries.nrows())
            .flat_map(|i| (0..dim).map(move |j| (i, j)))
            .map(|(i, j)| queries[(i, j)])
            .collect();
        let (ids, dists) = index
            .query_batch(&flat, dim, 10, Some(5), true, Verbosity::Quiet)
            .unwrap();
        let dists = dists.unwrap();

        for (i, q) in flat.chunks_exact(dim).enumerate() {
            let (want_ids, want_d) = index.query(q, 10, Some(5)).unwrap();
            assert_eq!(ids[i], want_ids, "row {i}");
            for (a, b) in dists[i].iter().zip(&want_d) {
                assert!(
                    (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                    "row {i}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn test_ivf_query_batch_matches_query_euclidean() {
        assert_batch_matches_query(Dist::SquaredEuclidean);
    }

    #[test]
    fn test_ivf_query_batch_matches_query_cosine() {
        assert_batch_matches_query(Dist::Cosine);
    }

    #[test]
    fn test_ivf_generate_knn_matches_query() {
        let dim = 24;
        let data = noisy_matrix(3_000, dim, 3);
        let index = IvfIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(25),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();
        let (graph, _) = index
            .generate_knn(8, Some(4), false, Verbosity::Quiet)
            .unwrap();
        for i in (0..3_000).step_by(37) {
            let row: Vec<f32> = (0..dim).map(|j| data[(i, j)]).collect();
            let (want, _) = index.query(&row, 8, Some(4)).unwrap();
            assert_eq!(graph[i], want, "row {i}");
        }
    }
}
