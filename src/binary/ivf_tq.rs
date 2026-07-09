//! Inverted-file TurboQuant index.
//!
//! Two-stage search: float-centroid IVF routing to pick the `nprobe` nearest
//! cells, then TurboQuant distance estimation within the probed cells.
//!
//! Encoding is global and data-oblivious: one shared random rotation and one
//! shared Lloyd-Max codebook for every vector, with no per-cluster residuals.
//! IVF here is routing only. All vectors are encoded once into bit-plane codes,
//! then bucketed into a CSR-by-cluster layout. Each cluster owns an independent
//! `repack` of its codes (its own `BLOCK` padding), so a cluster's blocked
//! codes are self-contained and the fused SIMD kernel can scan them directly
//! via `score_into_heaps` with the cluster's CSR offset as `base_index`.
//!
//! 2-bit and 4-bit cells use the fused kernel; 3-bit cells have no SIMD kernel
//! and fall back to the scalar `turboquant_dist` scan over the global storage.
//!
//! Batch queries union the probed cells of each group of four queries and scan
//! every cell in the union against all four (keeping the kernel's per-block
//! load shared). A query is scored against cells outside its own probe set as a
//! harmless side effect: extra candidates only ever improve its recall, and
//! ranking is by key, so they compete fairly. Consequently a batched query may
//! scan a few more cells than the single-query path, and `query_batch` results
//! are not guaranteed identical to per-row `query` results.

use bytemuck::Pod;
use faer::{MatRef, RowRef};
use rayon::prelude::*;
use std::collections::{BTreeSet, BinaryHeap};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::*;

use crate::binary::turboquant::dists::{prepare_scalar_scoring, VectorDistanceTurboQuant};
use crate::binary::turboquant::pack::{repack, BlockedLayout};
use crate::binary::turboquant::quantiser::{
    TurboQuantEncoder, TurboQuantQuantiser, TurboQuantStorage,
};
use crate::binary::turboquant::search::{build_query_lut, score_into_heaps, QueryLut, TopK};
use crate::binary::vec_store::*;
use crate::prelude::*;
use crate::utils::k_means_utils::*;

/// One cluster's blocked SIMD layout (2-bit / 4-bit only).
struct ClusterBlocked {
    /// Packed nibble bytes in block-major order for this cluster's vectors.
    data: Vec<u8>,
    /// Number of `BLOCK`-sized blocks in `data`.
    n_blocks: usize,
}

/// IVF index with TurboQuant quantisation.
///
/// Build once, then query, re-rank, or generate a full kNN graph. Immutable
/// after construction.
pub struct IvfTurboQuant<T> {
    /// Encoder plus global bit-plane storage. The encoder is shared across all
    /// cells; the global storage is the scoring source for 3-bit cells and the
    /// source the per-cluster blocked layouts are built from.
    quantiser: TurboQuantQuantiser<T>,
    /// Flat row-major centroids, `nlist * dim`. Normalised for Cosine.
    centroids: Vec<T>,
    /// Per-centroid L2 norms, length `nlist`.
    centroids_norm: Vec<T>,
    /// Number of IVF cells.
    nlist: usize,
    /// Dimensionality.
    dim: usize,
    /// Bits per coordinate.
    bits: usize,
    /// Distance metric.
    metric: Dist,
    /// CSR cell boundaries, length `nlist + 1`.
    cluster_offsets: Vec<usize>,
    /// CSR slot to original vector index, length `n`.
    vector_indices: Vec<usize>,
    /// Per-cluster blocked codes for 2/4-bit, one entry per cell. `None` for
    /// 3-bit, which scores through the scalar path instead.
    blocked: Option<Vec<ClusterBlocked>>,
    /// Per-vector raw L2 norms in CSR order, pre-cast to f32 for the SIMD kernels.
    norms_f32_csr: Vec<f32>,
    /// Per-vector dot-correction scalars `1 / <u, x_hat>` in CSR order, pre-cast to f32 for the SIMD kernels.
    corrections_f32_csr: Vec<f32>,
    /// Number of stored vectors.
    n: usize,
    /// Optional on-disk original vectors for exact re-ranking.
    vector_store: Option<MmapVectorStore<T>>,
}

//////////////////////////////
// VectorDistanceTurboQuant //
//////////////////////////////

impl<T> VectorDistanceTurboQuant<T> for IvfTurboQuant<T>
where
    T: AnnSearchFloat,
{
    fn storage(&self) -> &TurboQuantStorage<T> {
        &self.quantiser.storage
    }

    fn encoder(&self) -> &TurboQuantEncoder<T> {
        &self.quantiser.encoder
    }
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for IvfTurboQuant<T>
where
    T: AnnSearchFloat,
{
    fn centroids(&self) -> &[T] {
        &self.centroids
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn nlist(&self) -> usize {
        self.nlist
    }

    fn metric(&self) -> Dist {
        self.metric
    }

    fn centroids_norm(&self) -> &[T] {
        &self.centroids_norm
    }
}

impl<T> IvfTurboQuant<T>
where
    T: AnnSearchFloat + Pod,
{
    /// Build an IVF-TurboQuant index.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (n × dim); `dim` must be a multiple of 8
    /// * `metric` - Distance metric (`Cosine` or `SquaredEuclidean`)
    /// * `nlist` - Number of IVF cells (defaults to `sqrt(n)`)
    /// * `k_means_params` - Optional k-means training parameters
    /// * `bits` - Bits per coordinate (2, 3, or 4)
    /// * `seed` - Random seed for clustering and the rotation
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Initialised index, or `DistanceNotSupported` for `Manhattan`.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: MatRef<T>,
        metric: Dist,
        nlist: Option<usize>,
        k_means_params: Option<KMeansTrainingParams>,
        bits: usize,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (vectors_flat, n, dim) = matrix_to_flat(data);

        // original L2 norms; used for cluster assignment (Cosine) and routing.
        let norms: Vec<T> = (0..n)
            .map(|i| compute_l2_norm(&vectors_flat[i * dim..(i + 1) * dim]))
            .collect();

        // vectors used for clustering: unit-normalised for Cosine, raw for
        // squared euclidean.
        let vectors_for_storage: Vec<T> = if metric == Dist::Cosine {
            vectors_flat
                .chunks(dim)
                .zip(norms.iter())
                .flat_map(|(v, &norm)| {
                    if norm > T::epsilon() {
                        v.iter().map(|&x| x / norm).collect::<Vec<_>>()
                    } else {
                        v.to_vec()
                    }
                })
                .collect()
        } else {
            vectors_flat.clone()
        };

        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        if verbose {
            println!("  Building IVF-TurboQuant index with {} cells.", nlist);
        }

        let cluster_norms = if metric == Dist::Cosine {
            vec![T::one(); n]
        } else {
            norms.clone()
        };

        // subsample for centroid training when the data set is large.
        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        let mut centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            k_means_params,
            seed,
            verbose,
        )?;

        // normalise centroids for Cosine so routing dots are cosine similarities.
        if metric == Dist::Cosine {
            for c in 0..nlist {
                let start = c * dim;
                let end = start + dim;
                let norm = compute_l2_norm(&centroids[start..end]);
                if norm > T::epsilon() {
                    for x in &mut centroids[start..end] {
                        *x = *x / norm;
                    }
                }
            }
        }

        let centroids_norm: Vec<T> = (0..nlist)
            .map(|i| compute_l2_norm(&centroids[i * dim..(i + 1) * dim]))
            .collect();

        let assignments = assign_all_parallel(
            &vectors_for_storage,
            &cluster_norms,
            dim,
            n,
            &centroids,
            &centroids_norm,
            nlist,
            &metric,
        );

        let (vector_indices, cluster_offsets) = build_csr_layout(assignments, n, nlist);

        // global, data-oblivious TurboQuant encoding of every vector.
        let quantiser = TurboQuantQuantiser::new(data, &metric, bits, seed as u64)?;
        let bits = quantiser.storage.bits;

        // per-vector raw norms and dot-corrections reordered into CSR slot order for the kernels.
        let norms_f32_csr: Vec<f32> = (0..n)
            .map(|s| quantiser.storage.norms[vector_indices[s]].to_f32().unwrap())
            .collect();
        let corrections_f32_csr: Vec<f32> = (0..n)
            .map(|s| {
                quantiser.storage.corrections[vector_indices[s]]
                    .to_f32()
                    .unwrap()
            })
            .collect();

        // Per-cluster blocked layouts for 2/4-bit. 3-bit scores via the scalar
        // path over the global storage and needs no blocked layout.
        let blocked = if bits == 3 {
            None
        } else {
            let bytes_per_vec = quantiser.storage.bytes_per_vec;
            let mut per_cluster = Vec::with_capacity(nlist);
            for c in 0..nlist {
                let off = cluster_offsets[c];
                let end = cluster_offsets[c + 1];
                let csize = end - off;

                // Gather this cell's bit-plane codes in CSR slot order.
                let mut packed = vec![0u8; csize * bytes_per_vec];
                for (local, s) in (off..end).enumerate() {
                    let src = quantiser.storage.vector_packed(vector_indices[s]);
                    packed[local * bytes_per_vec..(local + 1) * bytes_per_vec].copy_from_slice(src);
                }

                // Throwaway storage purely to drive `repack`; only packed_codes,
                // n, bits, and dim are read, so norms and corrections are left empty.

                let norms: Vec<T> = Vec::new();
                let corrections: Vec<T> = Vec::new();

                let cell = TurboQuantStorage {
                    packed_codes: packed,
                    norms,
                    corrections,
                    dim,
                    bits,
                    bytes_per_vec,
                    n: csize,
                };
                match repack(&cell)? {
                    BlockedLayout::Standard(b) => per_cluster.push(ClusterBlocked {
                        data: b.data,
                        n_blocks: b.n_blocks,
                    }),
                    BlockedLayout::ThreeBit(_) => {
                        unreachable!("repack returned ThreeBit layout for non-3-bit codes")
                    }
                }
            }
            Some(per_cluster)
        };

        Ok(Self {
            quantiser,
            centroids,
            centroids_norm,
            nlist,
            dim,
            bits,
            metric,
            cluster_offsets,
            vector_indices,
            blocked,
            norms_f32_csr,
            corrections_f32_csr,
            n,
            vector_store: None,
        })
    }

    /// Build an IVF-TurboQuant index with an on-disk vector store for exact
    /// re-ranking.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (n × dim); `dim` must be a multiple of 8
    /// * `metric` - Distance metric (`Cosine` or `SquaredEuclidean`)
    /// * `nlist` - Number of IVF cells (defaults to `sqrt(n)`)
    /// * `k_means_params` - Optional k-means training parameters
    /// * `bits` - Bits per coordinate (2, 3, or 4)
    /// * `seed` - Random seed for clustering and the rotation
    /// * `verbose` - Print progress
    /// * `save_path` - Directory for the persisted vectors
    ///
    /// ### Returns
    ///
    /// Initialised index with re-ranking enabled.
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_vector_store(
        data: MatRef<T>,
        metric: Dist,
        nlist: Option<usize>,
        k_means_params: Option<KMeansTrainingParams>,
        bits: usize,
        seed: usize,
        verbose: bool,
        save_path: impl AsRef<Path>,
    ) -> Result<Self, AnnSearchErrors> {
        let mut index = Self::build(data, metric, nlist, k_means_params, bits, seed, verbose)?;

        let (vectors_flat, n, dim) = matrix_to_flat(data);
        let norms: Vec<T> = (0..n)
            .map(|i| compute_l2_norm(&vectors_flat[i * dim..(i + 1) * dim]))
            .collect();

        std::fs::create_dir_all(&save_path)?;
        let vectors_path = save_path.as_ref().join("vectors_flat.bin");
        let norms_path = save_path.as_ref().join("norms.bin");

        MmapVectorStore::save(&vectors_flat, &norms, dim, n, &vectors_path, &norms_path)?;
        index.vector_store = Some(MmapVectorStore::new(vectors_path, norms_path, dim, n)?);

        Ok(index)
    }

    /// Return the cell ids to probe, ascending-distance ordered.
    ///
    /// Routing uses the float centroids: the query is unit-normalised for
    /// Cosine (raw for Euclidean) to match the stored centroids, then scored
    /// via [`CentroidDistance::get_centroids_prenorm`]. `nprobe` is the floor
    /// and the walk keeps going until `>= k` reachable vectors are covered,
    /// so the query never short-returns due to small cells.
    fn probed_clusters(&self, query_vec: &[T], nprobe: usize, k: usize) -> Vec<usize> {
        let query_normalised: Vec<T> = match self.metric {
            Dist::Cosine => {
                let norm = compute_l2_norm(query_vec);
                if norm > T::epsilon() {
                    query_vec.iter().map(|&x| x / norm).collect()
                } else {
                    query_vec.to_vec()
                }
            }
            Dist::SquaredEuclidean => query_vec.to_vec(),
            Dist::Manhattan => unreachable!("Manhattan rejected at construction"),
        };

        let mut cluster_dists = self.get_centroids_prenorm(&query_normalised, nprobe);
        select_probed_clusters(&mut cluster_dists, &self.cluster_offsets, nprobe, k)
    }

    /// Default `nprobe` is `sqrt(nlist)`, clamped to `nlist`.
    ///
    /// ### Params
    ///
    /// * `nprobe` - Optional number of n_probes
    ///
    /// ### Returns
    ///
    /// The number of clusters to probe
    #[inline]
    fn resolve_nprobe(&self, nprobe: Option<usize>) -> usize {
        nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist)
    }

    /// Approximate k nearest neighbours for a query.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours
    /// * `nprobe` - Number of cells to probe (defaults to `sqrt(nlist)`)
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted nearest-first, distances being the
    /// TurboQuant-reconstructed estimates.
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        nprobe: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let nprobe = self.resolve_nprobe(nprobe);
        let k = k.min(self.n);
        if k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let probed = self.probed_clusters(query_vec, nprobe, k);

        let eq = self.quantiser.encode_query(query_vec)?;
        let (q_rot_f32, levels_f32) = prepare_scalar_scoring(&eq, &self.quantiser.encoder);

        if self.bits == 3 {
            // Scalar bit-plane scan over the probed cells' vectors.
            let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);
            for &c in &probed {
                for s in self.cluster_offsets[c]..self.cluster_offsets[c + 1] {
                    let orig = self.vector_indices[s];
                    let dist = self.turboquant_dist(&eq, &q_rot_f32, &levels_f32, orig);
                    if heap.len() < k {
                        heap.push((OrderedFloat(dist), orig));
                    } else if dist < heap.peek().unwrap().0 .0 {
                        heap.pop();
                        heap.push((OrderedFloat(dist), orig));
                    }
                }
            }
            let mut results: Vec<_> = heap.into_iter().collect();
            results.sort_unstable();
            let (distances, indices): (Vec<T>, Vec<usize>) =
                results.into_iter().map(|(d, i)| (d.0, i)).unzip();
            return Ok((indices, distances));
        }

        // 2/4-bit: fused scan into one persistent heap, accumulating across the
        // probed cells via per-cell `base_index`.
        let lut = build_query_lut(&q_rot_f32, &levels_f32, self.bits, self.dim)?;
        let lut_refs: [&QueryLut; 4] = [&lut, &lut, &lut, &lut];
        let qn = eq.query_norm.to_f32().unwrap();
        let blocked = self.blocked.as_ref().expect("blocked present for 2/4-bit");

        let mut heaps = vec![TopK::new(k.max(1))];
        for &c in &probed {
            let off = self.cluster_offsets[c];
            let end = self.cluster_offsets[c + 1];
            let csize = end - off;
            if csize == 0 {
                continue;
            }
            let cb = &blocked[c];
            score_into_heaps(
                &lut_refs,
                &cb.data,
                csize,
                cb.n_blocks,
                &self.norms_f32_csr[off..end],
                &self.corrections_f32_csr[off..end],
                [qn; 4],
                self.metric,
                1,
                off as u32,
                &mut heaps,
            );
        }

        let (csr_idx, dists_f32) = heaps.pop().unwrap().into_sorted(qn, self.metric);
        let indices = csr_idx
            .iter()
            .map(|&s| self.vector_indices[s as usize])
            .collect();
        let distances = dists_f32.iter().map(|&d| T::from_f32(d).unwrap()).collect();
        Ok((indices, distances))
    }

    /// Approximate k nearest neighbours for a query row.
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of neighbours
    /// * `nprobe` - Number of cells to probe (defaults to `sqrt(nlist)`)
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted nearest-first.
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
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, nprobe)
    }

    /// Re-rank approximate candidates with exact distances from the store.
    ///
    /// ### Params
    ///
    /// * `vector_store` - The internal vector store
    /// * `query_vec` - The query vector
    /// * `candidates` - Indices of the candidates
    /// * `k` - Number of neighbours to return
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` of the re-ranked data.
    fn rerank_candidates(
        &self,
        vector_store: &MmapVectorStore<T>,
        query_vec: &[T],
        candidates: &[usize],
        k: usize,
    ) -> (Vec<usize>, Vec<T>) {
        let query_norm = match self.metric {
            Dist::Cosine => compute_l2_norm(query_vec),
            _ => T::one(),
        };

        let mut scored: Vec<(T, usize)> = candidates
            .iter()
            .map(|&idx| {
                let dist = match self.metric {
                    Dist::Cosine => {
                        vector_store.cosine_distance_to_query(idx, query_vec, query_norm)
                    }
                    Dist::SquaredEuclidean => {
                        vector_store.euclidean_distance_to_query(idx, query_vec)
                    }
                    Dist::Manhattan => unreachable!("Manhattan rejected at construction"),
                };
                (dist, idx)
            })
            .collect();

        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(k);

        let indices = scored.iter().map(|p| p.1).collect();
        let dists = scored.iter().map(|p| p.0).collect();
        (indices, dists)
    }

    /// k nearest neighbours with exact-distance re-ranking.
    ///
    /// Pulls `k * rerank_factor` approximate candidates, re-scores them with
    /// exact distances from the vector store, and keeps the top `k`.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours
    /// * `nprobe` - Number of cells to probe (defaults to `sqrt(nlist)`)
    /// * `rerank_factor` - Candidate multiplier (defaults to 20)
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted nearest-first by exact distance.
    pub fn query_reranking(
        &self,
        query_vec: &[T],
        k: usize,
        nprobe: Option<usize>,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let rerank_factor = rerank_factor.unwrap_or(20);
        let vector_store = self
            .vector_store
            .as_ref()
            .ok_or(AnnSearchErrors::VectorStoreNotAvailable)?;

        let (candidates, _) = self.query(query_vec, k * rerank_factor, nprobe)?;
        Ok(self.rerank_candidates(vector_store, query_vec, &candidates, k))
    }

    /// k nearest neighbours with re-ranking, from a query row.
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of neighbours
    /// * `nprobe` - Number of cells to probe (defaults to `sqrt(nlist)`)
    /// * `rerank_factor` - Candidate multiplier (defaults to 20)
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted nearest-first by exact distance.
    pub fn query_row_reranking(
        &self,
        query_row: RowRef<T>,
        k: usize,
        nprobe: Option<usize>,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query_reranking(slice, k, nprobe, rerank_factor);
        }
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query_reranking(&query_vec, k, nprobe, rerank_factor)
    }

    /// k nearest neighbours for a batch of queries, with 4-query fusion.
    ///
    /// For 2/4-bit, queries are grouped into fours. Each group takes the union
    /// of its members' probed cells and scans every cell in the union against
    /// all four queries at once, so the kernel's per-block code load is shared.
    /// A query may therefore be scored against cells outside its own probe set;
    /// this only adds candidates and never removes correct ones, so results may
    /// differ slightly from the single-query path. 3-bit has no fused kernel and
    /// runs per query.
    ///
    /// ### Params
    ///
    /// * `queries` - Query matrix (n_queries × dim)
    /// * `k` - Number of neighbours per query
    /// * `nprobe` - Number of cells to probe (defaults to `sqrt(nlist)`)
    /// * `rerank` - Whether to re-rank with exact distances
    /// * `rerank_factor` - Candidate multiplier when re-ranking (defaults to 20)
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// `(indices, optional distances)`, one row per query, nearest-first.
    #[allow(clippy::too_many_arguments)]
    pub fn query_batch(
        &self,
        queries: MatRef<T>,
        k: usize,
        nprobe: Option<usize>,
        rerank: bool,
        rerank_factor: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<T> {
        let nq = queries.nrows();

        if queries.ncols() != self.dim {
            return Err(AnnSearchErrors::DimensionMismatch {
                index_dim: self.dim,
                query_dim: queries.ncols(),
            });
        }

        let vector_store = if rerank {
            Some(
                self.vector_store
                    .as_ref()
                    .ok_or(AnnSearchErrors::VectorStoreNotAvailable)?,
            )
        } else {
            None
        };

        let nprobe = self.resolve_nprobe(nprobe);
        let k_eff = k.min(self.n);
        let k_search = if rerank {
            (k_eff * rerank_factor.unwrap_or(20)).max(1)
        } else {
            k_eff
        };

        let counter = Arc::new(AtomicUsize::new(0));
        let report = |counter: &Arc<AtomicUsize>, added: usize| {
            if verbose {
                let count = counter.fetch_add(added, Ordering::Relaxed) + added;
                if count % 100_000 < added {
                    println!(
                        "  Processed {} / {} queries.",
                        count.separate_with_underscores(),
                        nq.separate_with_underscores()
                    );
                }
            }
        };

        let results: Vec<(Vec<usize>, Vec<T>)> = if self.bits == 3 {
            // No fused kernel for 3-bit: score each query independently.
            // This is sloooooow...
            (0..nq)
                .into_par_iter()
                .map(|i| {
                    let qvec: Vec<T> = queries.row(i).iter().cloned().collect();
                    let out = if rerank {
                        self.query_reranking(&qvec, k, Some(nprobe), rerank_factor)?
                    } else {
                        self.query(&qvec, k, Some(nprobe))?
                    };
                    report(&counter, 1);
                    Ok(out)
                })
                .collect::<Result<Vec<_>, AnnSearchErrors>>()?
        } else {
            let blocked = self.blocked.as_ref().expect("blocked present for 2/4-bit");

            let batch_starts: Vec<usize> = (0..nq).step_by(4).collect();
            let nested: Vec<Vec<(Vec<usize>, Vec<T>)>> = batch_starts
                .into_par_iter()
                .map(|start| {
                    let batch_nq = (nq - start).min(4);

                    let qrows: Vec<Vec<T>> = (0..batch_nq)
                        .map(|qi| queries.row(start + qi).iter().cloned().collect())
                        .collect();

                    // Encode each query, build its LUT, and union its probe set.
                    let mut luts_owned: Vec<QueryLut> = Vec::with_capacity(batch_nq);
                    let mut qnorms = [0.0f32; 4];
                    let mut probed: BTreeSet<usize> = BTreeSet::new();
                    for qi in 0..batch_nq {
                        let eq = self.quantiser.encode_query(&qrows[qi])?;
                        let (q_rot_f32, levels_f32) =
                            prepare_scalar_scoring(&eq, &self.quantiser.encoder);
                        luts_owned.push(build_query_lut(
                            &q_rot_f32,
                            &levels_f32,
                            self.bits,
                            self.dim,
                        )?);
                        qnorms[qi] = eq.query_norm.to_f32().unwrap();
                        probed.extend(self.probed_clusters(&qrows[qi], nprobe, k_search));
                    }

                    let last = batch_nq - 1;
                    let lut_refs: [&QueryLut; 4] = [
                        &luts_owned[0],
                        &luts_owned[1.min(last)],
                        &luts_owned[2.min(last)],
                        &luts_owned[3.min(last)],
                    ];

                    let mut heaps: Vec<TopK> =
                        (0..batch_nq).map(|_| TopK::new(k_search.max(1))).collect();
                    for &c in &probed {
                        let off = self.cluster_offsets[c];
                        let end = self.cluster_offsets[c + 1];
                        let csize = end - off;
                        if csize == 0 {
                            continue;
                        }
                        let cb = &blocked[c];
                        score_into_heaps(
                            &lut_refs,
                            &cb.data,
                            csize,
                            cb.n_blocks,
                            &self.norms_f32_csr[off..end],
                            &self.corrections_f32_csr[off..end],
                            qnorms,
                            self.metric,
                            batch_nq,
                            off as u32,
                            &mut heaps,
                        );
                    }

                    let mut out = Vec::with_capacity(batch_nq);
                    for (qi, heap) in heaps.into_iter().enumerate().take(batch_nq) {
                        let (csr_idx, dists_f32) = heap.into_sorted(qnorms[qi], self.metric);
                        let cands: Vec<usize> = csr_idx
                            .iter()
                            .map(|&s| self.vector_indices[s as usize])
                            .collect();
                        if let Some(vs) = vector_store {
                            out.push(self.rerank_candidates(vs, &qrows[qi], &cands, k));
                        } else {
                            let dists =
                                dists_f32.iter().map(|&d| T::from_f32(d).unwrap()).collect();
                            out.push((cands, dists));
                        }
                    }
                    report(&counter, batch_nq);
                    Ok(out)
                })
                .collect::<Result<Vec<_>, AnnSearchErrors>>()?;

            nested.into_iter().flatten().collect()
        };

        if return_dist {
            let (indices, distances) = results.into_iter().unzip();
            Ok((indices, Some(distances)))
        } else {
            let indices = results.into_iter().map(|(idx, _)| idx).collect();
            Ok((indices, None))
        }
    }

    /// Build the full kNN graph over all stored vectors via re-ranked queries.
    ///
    /// Requires a vector store. Self-queries run in parallel through
    /// [`query_reranking`](Self::query_reranking).
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per vector
    /// * `nprobe` - Number of cells to probe (defaults to `sqrt(nlist)`)
    /// * `rerank_factor` - Candidate multiplier (defaults to 20)
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// `(knn_indices, optional knn_distances)`.
    pub fn generate_knn(
        &self,
        k: usize,
        nprobe: Option<usize>,
        rerank_factor: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<T> {
        let vector_store = self
            .vector_store
            .as_ref()
            .ok_or(AnnSearchErrors::VectorStoreNotAvailable)?;

        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let vec = vector_store.load_vector(i);

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            self.n.separate_with_underscores()
                        );
                    }
                }

                self.query_reranking(vec, k, nprobe, rerank_factor)
            })
            .collect::<Result<Vec<_>, _>>()?;

        if return_dist {
            let (indices, distances) = results.into_iter().unzip();
            Ok((indices, Some(distances)))
        } else {
            let indices = results.into_iter().map(|(idx, _)| idx).collect();
            Ok((indices, None))
        }
    }

    /// Number of stored vectors.
    ///
    /// ### Returns
    ///
    /// The number of stored vectors
    pub fn n_vectors(&self) -> usize {
        self.n
    }

    /// Memory usage in bytes (excludes the on-disk vector store).
    pub fn memory_usage_bytes(&self) -> usize {
        let blocked = self
            .blocked
            .as_ref()
            .map_or(0, |cells| cells.iter().map(|c| c.data.capacity()).sum());

        std::mem::size_of_val(self)
            + self.quantiser.memory_usage_bytes()
            + self.centroids.capacity() * std::mem::size_of::<T>()
            + self.centroids_norm.capacity() * std::mem::size_of::<T>()
            + self.cluster_offsets.capacity() * std::mem::size_of::<usize>()
            + self.vector_indices.capacity() * std::mem::size_of::<usize>()
            + self.norms_f32_csr.capacity() * std::mem::size_of::<f32>()
            + self.corrections_f32_csr.capacity() * std::mem::size_of::<f32>()
            + blocked
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use std::collections::HashSet;
    use tempfile::TempDir;

    /// Decorrelated pseudo-random data, so the nearest neighbour to a stored
    /// vector is unambiguously itself.
    fn test_data(n: usize, dim: usize) -> Mat<f32> {
        Mat::from_fn(n, dim, |i, j| {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                ^ 0x1234_5678_9ABC_DEF0;
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            (x as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
    }

    fn row(data: &Mat<f32>, i: usize) -> Vec<f32> {
        (0..data.ncols()).map(|j| data[(i, j)]).collect()
    }

    fn params() -> Option<KMeansTrainingParams> {
        Some(KMeansTrainingParams::new(10, None, None))
    }

    fn exact_knn(data: &Mat<f32>, query: &[f32], k: usize, metric: Dist) -> Vec<usize> {
        let n = data.nrows();
        let dim = data.ncols();
        let qn: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut scored: Vec<(f32, usize)> = (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|j| data[(i, j)]).collect();
                let dist = match metric {
                    Dist::Cosine => {
                        let dot: f32 = v.iter().zip(query).map(|(a, b)| a * b).sum();
                        let vn: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        1.0 - dot / (qn * vn)
                    }
                    Dist::SquaredEuclidean => {
                        v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum()
                    }
                    Dist::Manhattan => unreachable!(),
                };
                (dist, i)
            })
            .collect();
        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(k);
        scored.into_iter().map(|(_, i)| i).collect()
    }

    #[test]
    fn test_construction() {
        let data = test_data(200, 64);
        let index = IvfTurboQuant::build(
            data.as_ref(),
            Dist::Cosine,
            Some(16),
            params(),
            4,
            42,
            false,
        )
        .unwrap();
        assert_eq!(index.n_vectors(), 200);
        assert_eq!(index.nlist, 16);
        assert_eq!(index.cluster_offsets.len(), 17);
        assert_eq!(*index.cluster_offsets.last().unwrap(), 200);
        assert_eq!(index.vector_indices.len(), 200);
    }

    #[test]
    fn test_manhattan_rejected() {
        let data = test_data(40, 64);
        let res = IvfTurboQuant::build(
            data.as_ref(),
            Dist::Manhattan,
            Some(4),
            params(),
            4,
            42,
            false,
        );
        assert!(matches!(
            res,
            Err(AnnSearchErrors::DistanceNotSupported(Dist::Manhattan))
        ));
    }

    #[test]
    fn test_default_nlist() {
        let data = test_data(100, 64);
        let index = IvfTurboQuant::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            None,
            params(),
            4,
            42,
            false,
        )
        .unwrap();
        assert_eq!(index.nlist, 10);
    }

    #[test]
    fn test_query_returns_k_sorted() {
        let data = test_data(200, 64);
        let index = IvfTurboQuant::build(
            data.as_ref(),
            Dist::Cosine,
            Some(16),
            params(),
            4,
            42,
            false,
        )
        .unwrap();
        let (indices, distances) = index.query(&row(&data, 3), 10, Some(16)).unwrap();
        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
        for w in distances.windows(2) {
            assert!(w[0] <= w[1], "distances not ascending");
        }
    }

    #[test]
    fn test_query_k_exceeds_n() {
        let data = test_data(50, 64);
        let index = IvfTurboQuant::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(8),
            params(),
            4,
            42,
            false,
        )
        .unwrap();
        let (indices, _) = index.query(&row(&data, 0), 100, Some(8)).unwrap();
        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_query_row_matches_query() {
        let data = test_data(150, 64);
        let index = IvfTurboQuant::build(
            data.as_ref(),
            Dist::Cosine,
            Some(12),
            params(),
            4,
            42,
            false,
        )
        .unwrap();
        let (i_row, _) = index.query_row(data.as_ref().row(5), 10, Some(12)).unwrap();
        let (i_vec, _) = index.query(&row(&data, 5), 10, Some(12)).unwrap();
        assert_eq!(i_row, i_vec);
    }

    #[test]
    fn test_recall_4bit_full_probe() {
        // Probing every cell with re-ranking should recover exact neighbours
        // on decorrelated data at 4-bit.
        let n = 200;
        let dim = 128;
        let k = 10;
        let nlist = 16;
        let data = test_data(n, dim);
        let dir = TempDir::new().unwrap();
        let index = IvfTurboQuant::build_with_vector_store(
            data.as_ref(),
            Dist::Cosine,
            Some(nlist),
            params(),
            4,
            42,
            false,
            dir.path(),
        )
        .unwrap();

        let mut hits = 0usize;
        for t in 0..n {
            let exact: HashSet<usize> = exact_knn(&data, &row(&data, t), k, Dist::Cosine)
                .into_iter()
                .collect();
            let (got, _) = index
                .query_reranking(&row(&data, t), k, Some(nlist), Some(20))
                .unwrap();
            hits += got.iter().filter(|i| exact.contains(i)).count();
        }
        let recall = hits as f32 / (n * k) as f32;
        assert!(recall > 0.9, "recall@{k} = {recall} below 0.9");
    }

    #[test]
    fn test_3bit_query_returns_k_sorted() {
        let data = test_data(120, 128);
        let index =
            IvfTurboQuant::build(data.as_ref(), Dist::Cosine, Some(8), params(), 3, 42, false)
                .unwrap();
        let (indices, distances) = index.query(&row(&data, 1), 8, Some(8)).unwrap();
        assert_eq!(indices.len(), 8);
        for w in distances.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn test_2bit_query_shape() {
        let data = test_data(160, 256);
        let index = IvfTurboQuant::build(
            data.as_ref(),
            Dist::Cosine,
            Some(12),
            params(),
            2,
            42,
            false,
        )
        .unwrap();
        let (indices, _) = index.query(&row(&data, 0), 10, Some(12)).unwrap();
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_query_reranking_without_store_errors() {
        let data = test_data(60, 64);
        let index =
            IvfTurboQuant::build(data.as_ref(), Dist::Cosine, Some(8), params(), 4, 42, false)
                .unwrap();
        let res = index.query_reranking(&row(&data, 0), 5, Some(8), Some(5));
        assert!(matches!(res, Err(AnnSearchErrors::VectorStoreNotAvailable)));
    }

    #[test]
    fn test_generate_knn_without_store_errors() {
        let data = test_data(60, 64);
        let index = IvfTurboQuant::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(8),
            params(),
            4,
            42,
            false,
        )
        .unwrap();
        let res = index.generate_knn(5, Some(8), Some(5), false, false);
        assert!(matches!(res, Err(AnnSearchErrors::VectorStoreNotAvailable)));
    }

    #[test]
    fn test_query_expands_nprobe_when_probed_cells_underfill_k() {
        // 120 vectors across 30 cells -> avg 4 per cell. nprobe=1 cannot cover
        // k=20; expansion must walk more cells so the query fills k.
        let data = test_data(120, 128);
        let index =
            IvfTurboQuant::build(data.as_ref(), Dist::Cosine, Some(30), params(), 4, 42, false)
                .unwrap();
        let (indices, distances) = index.query(&row(&data, 0), 20, Some(1)).unwrap();
        assert_eq!(indices.len(), 20);
        assert_eq!(distances.len(), 20);
    }

    #[test]
    fn test_generate_knn_shape() {
        let n = 120;
        let dim = 128;
        let k = 5;
        let data = test_data(n, dim);
        let dir = TempDir::new().unwrap();
        let index = IvfTurboQuant::build_with_vector_store(
            data.as_ref(),
            Dist::Cosine,
            Some(12),
            params(),
            4,
            42,
            false,
            dir.path(),
        )
        .unwrap();

        let (knn, dists) = index
            .generate_knn(k, Some(12), Some(20), true, false)
            .unwrap();
        assert_eq!(knn.len(), n);
        assert!(dists.is_some());
        for neigh in knn.iter() {
            assert_eq!(neigh.len(), k);
        }
    }

    #[test]
    fn test_query_batch_shape_and_recall() {
        // Batched recall should match the single-query path closely; it need not
        // be identical because the union scans extra cells.
        let n = 200;
        let dim = 128;
        let k = 10;
        let nlist = 16;
        let data = test_data(n, dim);
        let queries = test_data(7, dim); // a full batch of 4 plus 3
        let index = IvfTurboQuant::build(
            data.as_ref(),
            Dist::Cosine,
            Some(nlist),
            params(),
            4,
            42,
            false,
        )
        .unwrap();

        let (batch_idx, batch_dist) = index
            .query_batch(queries.as_ref(), k, Some(nlist), false, None, true, false)
            .unwrap();
        let batch_dist = batch_dist.unwrap();

        assert_eq!(batch_idx.len(), 7);
        for i in 0..7 {
            assert_eq!(batch_idx[i].len(), k);
            for w in batch_dist[i].windows(2) {
                assert!(w[0] <= w[1]);
            }
            let exact: HashSet<usize> = exact_knn(&data, &row(&queries, i), k, Dist::Cosine)
                .into_iter()
                .collect();
            let hits = batch_idx[i].iter().filter(|x| exact.contains(x)).count();
            assert!(hits as f32 / k as f32 > 0.8, "batch query {i} recall low");
        }
    }

    #[test]
    fn test_query_batch_rerank_without_store_errors() {
        let data = test_data(60, 64);
        let queries = test_data(4, 64);
        let index =
            IvfTurboQuant::build(data.as_ref(), Dist::Cosine, Some(8), params(), 4, 42, false)
                .unwrap();
        let res = index.query_batch(queries.as_ref(), 5, Some(8), true, Some(5), false, false);
        assert!(matches!(res, Err(AnnSearchErrors::VectorStoreNotAvailable)));
    }

    #[test]
    fn test_query_batch_dimension_mismatch() {
        let data = test_data(60, 64);
        let queries = test_data(4, 32);
        let index =
            IvfTurboQuant::build(data.as_ref(), Dist::Cosine, Some(8), params(), 4, 42, false)
                .unwrap();
        let res = index.query_batch(queries.as_ref(), 5, Some(8), false, None, false, false);
        assert!(matches!(
            res,
            Err(AnnSearchErrors::DimensionMismatch {
                index_dim: 64,
                query_dim: 32
            })
        ));
    }
}
