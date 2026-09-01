//! Inverted file binary index. Compresses the vectors via binarisation and
//! leverages Voronoi cells to reduce the search space.

use bytemuck::Pod;
use faer::RowRef;
use faer_traits::ComplexField;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::Separable;

use crate::binary::binariser::*;
use crate::binary::dist_binary::*;
use crate::binary::vec_store::*;
use crate::prelude::*;
use crate::utils::k_means_utils::*;

////////////////////
// IvfIndexBinary //
////////////////////

/// IVF index with binary quantisation
// `bound` is pinned because the skipped `vector_store` field makes serde
// infer a spurious `T: Default`
#[cfg_attr(
    feature = "serialise",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "T: AnnSearchFloat")
)]
pub struct IvfIndexBinary<T> {
    /// Binary codes, flattened (n * n_bytes)
    pub vectors_flat_binarised: Vec<u8>,
    /// Bytes per vector, taken from the binariser. Equals `n_bits / 8` for the
    /// projection-based methods; sign-based ignores `n_bits` and uses `dim`.
    pub n_bytes: usize,
    /// Number of samples in the index
    pub n: usize,
    /// Original vector dimensionality
    pub dim: usize,
    /// Distance metric
    metric: Dist,
    /// Binarisation type to use
    binarisation_type: BinarisationInit,
    /// Binariser
    binariser: Binariser<T>,
    /// Flat representation of the centroids
    centroids_float: Vec<T>,
    /// Flat representations of the L2 norms of the centroids for Cosine
    /// distances
    centroids_norm: Vec<T>,
    /// Vector indices for each cluster (CSR format)
    all_indices: Vec<usize>,
    /// Offsets for CSR access
    offsets: Vec<usize>,
    /// Number of clusters/lists in this index
    nlist: usize,
    /// Optional vector store that is saved in binary on disk
    #[cfg_attr(feature = "serialise", serde(skip))]
    vector_store: Option<MmapVectorStore<T>>,
    /// Shape of `vector_store`, so it can be re-opened after a load. Only
    /// read by the `serialise` feature; the field is kept unconditionally so
    /// the constructors stay free of `cfg`.
    #[cfg_attr(not(feature = "serialise"), allow(dead_code))]
    store_meta: Option<StoreMeta>,
    /// New to old mapping
    original_ids: Vec<usize>,
    /// Old to new mapping. Read by the asymmetric query path, which gets
    /// original ids back from [`IvfIndexBinary::query`] and has to reach the
    /// physical code slot they live in.
    old_to_new: Vec<usize>,
}

//////////////////////////
// VectorDistanceBinary //
//////////////////////////

impl<T> VectorDistanceBinary for IvfIndexBinary<T> {
    fn vectors_flat_binarised(&self) -> &[u8] {
        &self.vectors_flat_binarised
    }

    fn n_bytes(&self) -> usize {
        self.n_bytes
    }
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for IvfIndexBinary<T>
where
    T: AnnSearchFloat,
{
    fn centroids(&self) -> &[T] {
        &self.centroids_float
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

impl<T> IvfIndexBinary<T>
where
    T: AnnSearchFloat + ComplexField + Pod,
{
    /// Build an IVF index with binary quantisation
    ///
    /// Uses hybrid approach: float centroids for routing, binary codes for
    /// storage. Clustering happens in float space to preserve semantic
    /// structure.
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `binarisation_init` - Initialisation method (`"random"`, `"pca"`, or
    ///   `"sign"`)
    /// * `n_bits` - Number of bits per binary code (must be multiple of 8).
    ///   Ignored by sign-based binarisation, which always emits `dim` bits.
    /// * `metric` - Distance metric for centroid routing
    /// * `nlist` - Optional number of clusters (defaults to sqrt(n))
    /// * `k_means_params` - Optional k-means trainings parameters, see
    ///   [KMeansTrainingParams]. If not provided, will default to sensible
    ///   defaults.
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Constructed index
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: impl AnnMatrix<T>,
        binarisation_init: &str,
        n_bits: usize,
        metric: Dist,
        nlist: Option<usize>,
        k_means_params: Option<KMeansTrainingParams>,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        if !n_bits.is_multiple_of(8) {
            return Err(AnnSearchErrors::NBitsMustBe8Multiple { n_bits });
        }
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (vectors_flat, n, dim) = data.into_row_major();

        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        if verbose {
            println!(
                "  Generating IVF-Binary index with {} Voronoi cells.",
                nlist
            );
        }

        // 1. subsample for training if needed
        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        // 2. train float centroids
        let centroids_float = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            k_means_params,
            seed,
            verbose,
        )?;

        let centroids_norm = if metric == Dist::Cosine {
            (0..nlist)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    T::calculate_l2_norm(&centroids_float[start..end])
                })
                .collect()
        } else {
            Vec::new()
        };

        // 3. assign all vectors to centroids in float space
        let data_norms = if metric == Dist::Cosine {
            vectors_flat
                .par_chunks_exact(dim)
                .map(T::calculate_l2_norm)
                .collect()
        } else {
            vec![T::one(); n]
        };

        let assignments = assign_all_parallel(
            &vectors_flat,
            &data_norms,
            dim,
            n,
            &centroids_float,
            &centroids_norm,
            nlist,
            &metric,
        );

        // 4. build CSR layout
        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        // 5. initialise binariser and encode all vectors
        let init = parse_binarisation_init(binarisation_init).unwrap_or_else(|| {
            println!("[WARNING] Unknown binarisation string provided. Using the default");
            BinarisationInit::default()
        });

        let binariser = match init {
            BinarisationInit::PcaHashing => {
                Binariser::new_pca_hashing(&vectors_flat, n, dim, n_bits, seed)?
            }
            BinarisationInit::RandomProjections => {
                Binariser::new_simhash(&vectors_flat, n, dim, n_bits, seed)?
            }
            BinarisationInit::SignBased => Binariser::new_sign_based(dim),
        };

        // Ask the binariser, do not derive from `n_bits`: sign-based ignores
        // that argument and emits `dim` bits
        let n_bytes = binariser.n_bytes();

        let mut vectors_flat_binarised = vec![0u8; n * n_bytes];
        binariser.encode_all(&vectors_flat, n, &mut vectors_flat_binarised)?;

        let mut idx = Self {
            vectors_flat_binarised,
            n_bytes,
            n,
            dim,
            binariser,
            binarisation_type: init,
            centroids_float,
            centroids_norm,
            metric,
            all_indices,
            offsets,
            nlist,
            vector_store: None,
            store_meta: None,
            original_ids: Vec::new(),
            old_to_new: Vec::new(),
        };

        let new_to_old = idx.optimise_memory_layout();
        idx.original_ids = new_to_old;

        Ok(idx)
    }

    /// Build an IVF index with binary quantisation and vector store for reranking
    ///
    /// Creates IVF binary index and saves/loads vector store for exact distance
    /// reranking.
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `binarisation_init` - Initialisation method (`"random"`, `"pca"`, or
    ///   `"sign"`)
    /// * `n_bits` - Number of bits per binary code (must be multiple of 8).
    ///   Ignored by sign-based binarisation, which always emits `dim` bits.
    /// * `metric` - Distance metric for centroid routing and reranking
    /// * `nlist` - Optional number of clusters (defaults to sqrt(n))
    /// * `k_means_params` - Optional k-means trainings parameters, see
    ///   [KMeansTrainingParams]. If not provided, will default to sensible
    ///   defaults.
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    /// * `save_path` - Directory to save vector store files
    ///
    /// ### Returns
    ///
    /// Constructed index with vector store
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_vector_store(
        data: impl AnnMatrix<T>,
        binarisation_init: &str,
        n_bits: usize,
        metric: Dist,
        nlist: Option<usize>,
        k_means_params: Option<KMeansTrainingParams>,
        seed: usize,
        verbose: bool,
        save_path: impl AsRef<Path>,
    ) -> Result<Self, AnnSearchErrors> {
        if !n_bits.is_multiple_of(8) {
            return Err(AnnSearchErrors::NBitsMustBe8Multiple { n_bits });
        }
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (vectors_flat, n, dim) = data.into_row_major();

        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        if verbose {
            println!(
                "  Generating IVF-Binary index with {} Voronoi cells.",
                nlist
            );
        }

        // 1. subsample for training if needed
        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        // 2. train float centroids
        let centroids_float = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            k_means_params,
            seed,
            verbose,
        )?;

        let centroids_norm = if metric == Dist::Cosine {
            centroids_float
                .chunks_exact(dim)
                .map(T::calculate_l2_norm)
                .collect()
        } else {
            Vec::new()
        };

        // 3. assign all vectors to centroids in float space
        let data_norms: Vec<T> = if metric == Dist::Cosine {
            vectors_flat
                .par_chunks_exact(dim)
                .map(T::calculate_l2_norm)
                .collect()
        } else {
            vec![T::one(); n]
        };

        let assignments = assign_all_parallel(
            &vectors_flat,
            &data_norms,
            dim,
            n,
            &centroids_float,
            &centroids_norm,
            nlist,
            &metric,
        );

        // 4. build CSR layout
        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        // 5. initialise binariser and encode all vectors
        let init = parse_binarisation_init(binarisation_init).unwrap_or_else(|| {
            println!("[WARNING] Unknown binarisation string provided. Using the default");
            BinarisationInit::default()
        });

        let binariser = match init {
            BinarisationInit::PcaHashing => {
                Binariser::new_pca_hashing(&vectors_flat, n, dim, n_bits, seed)?
            }
            BinarisationInit::RandomProjections => {
                Binariser::new_simhash(&vectors_flat, n, dim, n_bits, seed)?
            }
            BinarisationInit::SignBased => Binariser::new_sign_based(dim),
        };

        // Ask the binariser, do not derive from `n_bits`: sign-based ignores
        // that argument and emits `dim` bits
        let n_bytes = binariser.n_bytes();

        let mut vectors_flat_binarised = vec![0u8; n * n_bytes];
        binariser.encode_all(&vectors_flat, n, &mut vectors_flat_binarised)?;

        // Save vector store
        std::fs::create_dir_all(&save_path)?;

        // Cosine already computed exactly these for the assignment step; the
        // other metrics filled `data_norms` with ones and need the real thing.
        let norms: Vec<T> = if metric == Dist::Cosine {
            data_norms
        } else {
            vectors_flat
                .par_chunks_exact(dim)
                .map(T::calculate_l2_norm)
                .collect()
        };

        let (vectors_path, norms_path) = MmapVectorStore::<T>::paths_in(&save_path);

        MmapVectorStore::save(&vectors_flat, &norms, dim, n, &vectors_path, &norms_path)?;

        let vector_store = MmapVectorStore::new(vectors_path, norms_path, dim, n)?;

        let mut idx = Self {
            vectors_flat_binarised,
            n_bytes,
            n,
            dim,
            binariser,
            binarisation_type: init,
            centroids_float,
            centroids_norm,
            metric,
            all_indices,
            offsets,
            nlist,
            vector_store: Some(vector_store),
            store_meta: Some(StoreMeta { dim, n }),
            original_ids: Vec::new(),
            old_to_new: Vec::new(),
        };

        let new_to_old = idx.optimise_memory_layout();
        idx.original_ids = new_to_old;

        Ok(idx)
    }

    ///////////
    // Query //
    ///////////

    /// Query the index for approximate nearest neighbours
    ///
    /// Two-stage search: finds nprobe nearest centroids using float distance,
    /// then searches those clusters using Hamming distance on binary codes.
    ///
    /// Codes live in one global frame, so the query is binarised once and the
    /// Hamming distances compare across cells. A code is a coarse ordering
    /// though; [`Self::query_asymmetric`] and [`Self::query_reranking`] widen
    /// the pool and re-sort it on finer arithmetic.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector in float space
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search (defaults to `sqrt(nlist)`)
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` where distances are Hamming distances
    #[inline]
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        nprobe: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<u32>), AnnSearchErrors> {
        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k = k.min(self.n);

        let query_norm = if matches!(self.metric, Dist::Cosine) {
            T::calculate_l2_norm(query_vec)
        } else {
            T::one()
        };

        let query_binary = self.binariser.encode(query_vec)?;

        // 1. Find nprobe nearest centroids using float distance, expanding to
        //    cover >= k reachable vectors so the query never short-returns.
        let mut cluster_dists = self.get_centroids_dist(query_vec, query_norm, nprobe);
        let probed = select_probed_clusters(&mut cluster_dists, &self.offsets, nprobe, k);

        // 2. Search clusters using Hamming distance
        let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(k + 1);
        let mut block = [0u32; HAMMING_BLOCK];

        for cluster_idx in probed {
            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];

            let mut vec_idx = start;
            while vec_idx < end {
                let take = HAMMING_BLOCK.min(end - vec_idx);
                let codes = &self.vectors_flat_binarised
                    [vec_idx * self.n_bytes..(vec_idx + take) * self.n_bytes];
                let block_min =
                    hamming_block(&query_binary, codes, self.n_bytes, &mut block[..take]);

                if heap.len() < k || block_min < heap.peek().unwrap().0 {
                    for (j, &dist) in block[..take].iter().enumerate() {
                        if heap.len() < k {
                            heap.push((dist, vec_idx + j));
                        } else if dist < heap.peek().unwrap().0 {
                            heap.pop();
                            heap.push((dist, vec_idx + j));
                        }
                    }
                }

                vec_idx += take;
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);

        let (distances, indices): (Vec<_>, Vec<_>) = results
            .into_iter()
            .map(|(d, i)| (d, self.original_ids[i]))
            .unzip();

        Ok((indices, distances))
    }

    /// Query using row reference
    ///
    /// Uses optimised paths if the query row is contiguous in memory.
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search (defaults to `sqrt(nlist)`)
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        nprobe: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<u32>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, nprobe);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, nprobe)
    }

    /// Query function for asymmetric querying
    ///
    /// This function will first calculate the Hamming distance between the
    /// query vector and the binary codes (leveraging the Voronoi cells) and
    /// then do an additional asymmetric dot product distance calculation
    /// between the query vector and the binary codes of the candidates.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (will be binarised internally)
    /// * `k` - Number of nearest neighbours to return
    /// * `nprobe` - Number of clusters to search (defaults to `sqrt(nlist)`)
    /// * `rerank_factor` - Multiplier for candidate set size (searches k *
    ///   rerank_factor candidates). If not supplied, defaults to `20`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, scores)` where the score is the asymmetric dot
    /// product, sorted descending: higher means more similar. Errors when the
    /// binarisation type is not sign-based.
    #[inline]
    pub fn query_asymmetric(
        &self,
        query_vec: &[T],
        k: usize,
        nprobe: Option<usize>,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if !self.use_asymmetric() {
            return Err(AnnSearchErrors::AsymmetricQueryMisMatch);
        }
        let rerank_factor = rerank_factor.unwrap_or(20);

        let (candidates, _) = self.query(query_vec, k * rerank_factor, nprobe)?;

        // One pass over the query, not one per candidate
        let query_sum = query_vec.iter().fold(T::zero(), |acc, &x| acc + x);

        // `query` hands back original ids; the codes are stored in cell order,
        // so the physical slot has to come back through `old_to_new`
        let mut scored: Vec<(usize, T)> = candidates
            .iter()
            .map(|&idx| {
                let start_i = self.old_to_new[idx] * self.n_bytes;
                let vec_i = unsafe {
                    self.vectors_flat_binarised
                        .get_unchecked(start_i..start_i + self.n_bytes)
                };

                (
                    idx,
                    asymmetric_binary_dot_presummed(query_vec, query_sum, vec_i, self.dim),
                )
            })
            .collect();

        // `asymmetric_binary_dot` is a similarity, not a distance: the query's
        // own code maximises it. Descending, or the funnel keeps the k
        // *least* similar candidates
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(k);

        let mut indices: Vec<usize> = Vec::with_capacity(k);
        let mut dists: Vec<T> = Vec::with_capacity(k);

        for (idx, dist) in scored {
            indices.push(idx);
            dists.push(dist);
        }

        Ok((indices, dists))
    }

    /// Query using row reference (asymmetric)
    ///
    /// Uses optimised paths if the query row is contiguous in memory.
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search (defaults to `sqrt(nlist)`)
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    #[inline]
    pub fn query_row_asymmetric(
        &self,
        query_row: RowRef<T>,
        k: usize,
        nprobe: Option<usize>,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query_asymmetric(slice, k, nprobe, rerank_factor);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query_asymmetric(&query_vec, k, nprobe, rerank_factor)
    }

    /// Query with reranking using exact distances
    ///
    /// Three-stage search: IVF routing to clusters, Hamming distance to find
    /// candidates, then exact distance for final ranking. Requires
    /// vector_store to be available. If the binarisation type is sign-based,
    /// asymmetric dot product calculations are used between the Hamming
    /// distance and the exact distance step.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of nearest neighbours to return
    /// * `nprobe` - Number of clusters to search (defaults to sqrt(nlist))
    /// * `rerank_factor` - Multiplier for candidate set size (searches k *
    ///   rerank_factor candidates). If not supplied, defaults to `20`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` where distances are exact (Euclidean or Cosine)
    #[inline]
    pub fn query_reranking(
        &self,
        query_vec: &[T],
        k: usize,
        nprobe: Option<usize>,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let vector_store = self
            .vector_store
            .as_ref()
            .ok_or(AnnSearchErrors::VectorStoreNotAvailable)?;
        let rerank_factor = rerank_factor.unwrap_or(20);

        // `query_asymmetric` truncates to its own `k`, so it has to be asked for
        // `k * rerank_factor` candidates, not `k`. Passing `k` collapses the
        // funnel to `k -> k` and the exact stage can only reorder what the
        // binary stages already picked, never recover a dropped neighbour.
        let candidates = if matches!(self.binarisation_type, BinarisationInit::SignBased) {
            let (idx, _) = self.query_asymmetric(query_vec, k * rerank_factor, nprobe, Some(2))?;
            idx
        } else {
            let (idx, _) = self.query(query_vec, k * rerank_factor, nprobe)?;
            idx
        };

        let query_norm = match self.metric {
            Dist::Cosine => T::calculate_l2_norm(query_vec),
            Dist::SquaredEuclidean => T::one(),
            Dist::Manhattan => unreachable!(),
        };

        let mut scored: Vec<_> = candidates
            .iter()
            .map(|&idx| {
                let dist = match self.metric {
                    Dist::Cosine => {
                        vector_store.cosine_distance_to_query(idx, query_vec, query_norm)
                    }
                    Dist::SquaredEuclidean => {
                        vector_store.euclidean_distance_to_query(idx, query_vec)
                    }
                    Dist::Manhattan => unreachable!(),
                };
                (dist, idx)
            })
            .collect();

        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(k);

        let mut indices: Vec<usize> = Vec::with_capacity(k);
        let mut dists: Vec<T> = Vec::with_capacity(k);

        for (dist, idx) in scored {
            indices.push(idx);
            dists.push(dist);
        }

        Ok((indices, dists))
    }

    /// Query with reranking for row references
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search
    /// * `rerank_factor` - Multiplier for candidate set size
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    #[inline]
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

    /// Generate kNN graph from vectors stored in the index
    ///
    /// If vector_store is available, uses it for exact distance reranking.
    /// Otherwise, uses Hamming distances only.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `nprobe` - Number of clusters to search (defaults to sqrt(nlist))
    /// * `rerank_factor` - Multiplier for candidate set (only used if vector_store available)
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`
    pub fn generate_knn(
        &self,
        k: usize,
        nprobe: Option<usize>,
        rerank_factor: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<T> {
        let counter = Arc::new(AtomicUsize::new(0));

        if let Some(vector_store) = &self.vector_store {
            let unordered_results: Vec<(usize, Vec<usize>, Vec<T>)> = (0..self.n)
                .into_par_iter()
                .map(|i| {
                    let orig_id = self.original_ids[i];
                    let vec = vector_store.load_vector(orig_id);

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

                    let (indices, dists) = self.query_reranking(vec, k, nprobe, rerank_factor)?;
                    Ok((orig_id, indices, dists))
                })
                .collect::<Result<Vec<_>, AnnSearchErrors>>()?;

            let mut final_indices = vec![Vec::new(); self.n];
            let mut final_dists = if return_dist {
                Some(vec![Vec::new(); self.n])
            } else {
                None
            };

            for (orig_id, indices, dists) in unordered_results {
                final_indices[orig_id] = indices;
                if let Some(ref mut fd) = final_dists {
                    fd[orig_id] = dists;
                }
            }

            fix_neg_dist(&mut final_dists);
            Ok((final_indices, final_dists))
        } else {
            let unordered_results: Vec<(usize, Vec<usize>, Vec<u32>)> = (0..self.n)
                .into_par_iter()
                .map(|i| {
                    let orig_id = self.original_ids[i];
                    let start = i * self.n_bytes;
                    let query_binary = &self.vectors_flat_binarised[start..start + self.n_bytes];
                    let k = k.min(self.n);

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

                    let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(k + 1);
                    let mut block = [0u32; HAMMING_BLOCK];

                    for cluster_idx in 0..self.nlist {
                        let start_idx = self.offsets[cluster_idx];
                        let end_idx = self.offsets[cluster_idx + 1];

                        let mut vec_idx = start_idx;
                        while vec_idx < end_idx {
                            let take = HAMMING_BLOCK.min(end_idx - vec_idx);
                            let codes = &self.vectors_flat_binarised
                                [vec_idx * self.n_bytes..(vec_idx + take) * self.n_bytes];
                            let block_min = hamming_block(
                                query_binary,
                                codes,
                                self.n_bytes,
                                &mut block[..take],
                            );

                            if heap.len() < k || block_min < heap.peek().unwrap().0 {
                                for (j, &dist) in block[..take].iter().enumerate() {
                                    if heap.len() < k {
                                        heap.push((dist, vec_idx + j));
                                    } else if dist < heap.peek().unwrap().0 {
                                        heap.pop();
                                        heap.push((dist, vec_idx + j));
                                    }
                                }
                            }

                            vec_idx += take;
                        }
                    }

                    let mut results: Vec<_> = heap.into_iter().collect();
                    results.sort_unstable_by_key(|&(dist, _)| dist);

                    let (distances, indices): (Vec<_>, Vec<_>) = results
                        .into_iter()
                        .map(|(d, i)| (d, self.original_ids[i]))
                        .unzip();

                    (orig_id, indices, distances)
                })
                .collect();

            let mut final_indices = vec![Vec::new(); self.n];
            let mut final_dists = if return_dist {
                Some(vec![Vec::new(); self.n])
            } else {
                None
            };

            for (orig_id, indices, dists) in unordered_results {
                final_indices[orig_id] = indices.clone();
                if let Some(ref mut fd) = final_dists {
                    fd[orig_id] = dists.into_iter().map(|d| T::from_u32(d).unwrap()).collect();
                }
            }

            Ok((final_indices, final_dists))
        }
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// The memory finger print in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat_binarised.capacity()
            + self.binariser.memory_usage_bytes()
            + self.centroids_float.capacity() * std::mem::size_of::<T>()
            + self.centroids_norm.capacity() * std::mem::size_of::<T>()
            + self.all_indices.capacity() * std::mem::size_of::<usize>()
            + self.offsets.capacity() * std::mem::size_of::<usize>()
    }

    /// Returns whether the index supports asymmetric queries
    ///
    /// ### Returns
    ///
    /// True if yes
    pub fn use_asymmetric(&self) -> bool {
        matches!(self.binarisation_type, BinarisationInit::SignBased)
    }

    /// Function will optimise memory layout and put vectors sorted by cluster
    /// id
    ///
    /// ### Returns
    ///
    /// New to old mapping
    fn optimise_memory_layout(&mut self) -> Vec<usize> {
        let mut new_to_old = Vec::with_capacity(self.n);
        let mut old_to_new = vec![0usize; self.n];

        for cluster in 0..self.nlist {
            let start = self.offsets[cluster];
            let end = self.offsets[cluster + 1];
            for &old_id in &self.all_indices[start..end] {
                old_to_new[old_id] = new_to_old.len();
                new_to_old.push(old_id);
            }
        }

        let n_bytes = self.n_bytes;
        let mut new_binarised = vec![0u8; self.vectors_flat_binarised.len()];
        new_binarised
            .par_chunks_mut(n_bytes)
            .zip(new_to_old.par_iter())
            .for_each(|(slot, &old_id)| {
                let start = old_id * n_bytes;
                slot.copy_from_slice(&self.vectors_flat_binarised[start..start + n_bytes]);
            });

        self.vectors_flat_binarised = new_binarised;
        self.old_to_new = old_to_new;
        self.all_indices.clear();
        self.all_indices.shrink_to_fit();

        new_to_old
    }
}

///////////
// Tests //
///////////

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
use crate::utils::staging::StagedFiles;

#[cfg(feature = "serialise")]
impl<T> IndexIo for IvfIndexBinary<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "ivf_binary";

    fn stage_aux(&self, dir: &Path, staged: &mut StagedFiles) -> Result<(), AnnSearchErrors> {
        match &self.vector_store {
            Some(store) => store.stage_copy_into(dir, staged),
            None => Ok(()),
        }
    }

    fn load_aux(&mut self, dir: &Path) -> Result<(), AnnSearchErrors> {
        if let Some(meta) = self.store_meta {
            meta.check(self.n, self.dim)?;
        }
        self.vector_store = MmapVectorStore::open_in_dir(dir, self.store_meta)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use num_traits::{Float, FromPrimitive};
    use tempfile::TempDir;

    fn create_test_data<T: Float + FromPrimitive + ComplexField>(n: usize, dim: usize) -> Mat<T> {
        let mut data = Mat::zeros(n, dim);
        for i in 0..n {
            for j in 0..dim {
                data[(i, j)] = T::from_f64((i * dim + j) as f64 * 0.1).unwrap();
            }
        }
        data
    }

    /// Deterministic data with both signs present
    ///
    /// `create_test_data` ramps upwards from zero, so every sign-based code
    /// comes out as all ones, every Hamming distance is 0 and any assertion on
    /// neighbour identity is vacuous. Anything testing the sign path needs
    /// this instead.
    ///
    /// ### Params
    ///
    /// * `n` - Rows
    /// * `dim` - Columns
    /// * `seed` - Seed for reproducibility
    ///
    /// ### Returns
    ///
    /// An `n` x `dim` matrix of uniform values in `[-1, 1)`.
    fn create_signed_test_data(n: usize, dim: usize, seed: u64) -> Mat<f32> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed);

        Mat::from_fn(n, dim, |_, _| rng.random::<f32>() * 2.0 - 1.0)
    }

    fn get_default_k_means() -> Option<KMeansTrainingParams> {
        Some(KMeansTrainingParams::new(10, None, None))
    }

    /// Sign-based binarisation ignores `n_bits` and emits `dim` bits, so the
    /// code stride must come from the binariser. Taking it from `n_bits`
    /// instead used to walk `optimise_memory_layout` off the end of
    /// `vectors_flat_binarised` whenever `n_bits != dim`.
    #[test]
    fn test_sign_based_stride_ignores_n_bits() {
        let (n, dim) = (128, 32);
        let data = create_signed_test_data(n, dim, 7);

        for n_bits in [8, 32, 64, 128] {
            let index = IvfIndexBinary::build(
                data.as_ref(),
                "sign",
                n_bits,
                Dist::Cosine,
                Some(4),
                get_default_k_means(),
                42,
                false,
            )
            .unwrap();

            assert_eq!(index.n_bytes, dim / 8, "stride tracked n_bits = {}", n_bits);
            assert_eq!(index.vectors_flat_binarised.len(), index.n * index.n_bytes);

            // With mixed signs every code is distinct, so a stored vector must
            // find itself at Hamming distance 0 before anything else. A wrong
            // but in-bounds stride reads a neighbour's bytes and fails here
            for row in [0, 17, 63, 127] {
                let query: Vec<f32> = data.row(row).iter().cloned().collect();
                let (indices, dists) = index.query(&query, 5, Some(4)).unwrap();

                assert_eq!(indices.len(), 5);
                assert_eq!(indices[0], row, "self-retrieval failed, n_bits = {n_bits}");
                assert_eq!(dists[0], 0);
            }
        }
    }

    /// Sign-based codes now live in one global frame, so a full kNN graph no
    /// longer needs the float vectors. This used to be a hard error.
    #[test]
    fn test_knn_graph_without_store_works_for_sign() {
        let (n, dim, k) = (256, 32, 5);
        let data = create_signed_test_data(n, dim, 3);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "sign",
            dim,
            Dist::SquaredEuclidean,
            Some(8),
            get_default_k_means(),
            42,
            false,
        )
        .unwrap();

        let (indices, _) = index.generate_knn(k, Some(4), None, false, false).unwrap();

        assert_eq!(indices.len(), n);
        for (row, neighbours) in indices.iter().enumerate() {
            assert_eq!(neighbours.len(), k, "row {row} came back short");
            assert_eq!(neighbours[0], row, "row {row} did not find itself first");
        }
    }

    /// Widening `nprobe` must not *cost* recall. Under the old residual frame
    /// it did: codes from a distant cell were taken against that cell's own
    /// centroid, so they scored spuriously well and crowded out true
    /// neighbours. In a global frame the extra cells can only add candidates.
    #[test]
    fn test_sign_recall_does_not_drop_as_nprobe_widens() {
        let (n, dim, k) = (2000, 32, 10);
        let data = create_clustered_test_data(n, dim, 8, 21);
        let temp_dir = TempDir::new().unwrap();

        let index = IvfIndexBinary::build_with_vector_store(
            data.as_ref(),
            "sign",
            dim,
            Dist::SquaredEuclidean,
            Some(16),
            get_default_k_means(),
            42,
            false,
            temp_dir.path(),
        )
        .unwrap();

        let recall_at = |nprobe: usize| {
            let (mut hits, mut total) = (0usize, 0usize);
            for row in (0..n).step_by(53) {
                let query: Vec<f32> = data.row(row).iter().cloned().collect();
                let truth = brute_force(&data, &query, k);
                let (got, _) = index
                    .query_reranking(&query, k, Some(nprobe), Some(10))
                    .unwrap();
                hits += got.iter().filter(|i| truth.contains(i)).count();
                total += k;
            }
            hits as f64 / total as f64
        };

        let narrow = recall_at(4);
        let wide = recall_at(16);

        assert!(
            wide >= narrow - 0.02,
            "recall fell from {narrow:.3} at nprobe 4 to {wide:.3} at nprobe 16"
        );
    }

    /// Brute-force top-k by squared Euclidean distance
    ///
    /// Ground truth for the recall assertions below.
    ///
    /// ### Params
    ///
    /// * `data` - Index matrix
    /// * `query` - Query vector
    /// * `k` - Number of neighbours
    ///
    /// ### Returns
    ///
    /// The `k` nearest row indices, nearest first.
    fn brute_force(data: &Mat<f32>, query: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = (0..data.nrows())
            .map(|i| {
                let d: f32 = (0..data.ncols())
                    .map(|j| (data[(i, j)] - query[j]).powi(2))
                    .sum();
                (i, d)
            })
            .collect();

        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);

        scored.into_iter().map(|(i, _)| i).collect()
    }

    /// Gaussian blobs whose centres sit far from the origin
    ///
    /// The regime that defeats raw sign binarisation: the cluster offset
    /// dominates the within-cluster spread, so every member of a cluster shares
    /// almost every sign bit and the code identifies the cluster rather than
    /// the point.
    ///
    /// ### Params
    ///
    /// * `n` - Rows
    /// * `dim` - Columns
    /// * `n_clusters` - Number of blobs
    /// * `seed` - Seed for reproducibility
    ///
    /// ### Returns
    ///
    /// An `n` x `dim` matrix.
    fn create_clustered_test_data(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Mat<f32> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed);

        let centres: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.random_range(-7.5..7.5)).collect())
            .collect();

        Mat::from_fn(n, dim, |i, j| {
            centres[i % n_clusters][j] + (rng.random::<f32>() * 2.0 - 1.0)
        })
    }

    /// Recall on clustered data, the hardest regime for sign codes
    ///
    /// Blobs sit far from the origin, so the sign bits carry the cluster label
    /// loudly and within-cluster structure quietly. The Hamming stage is a
    /// coarse ordering there and recall has to be bought with pool width.
    ///
    /// Measured on this fixture, 8 blobs at dim 32 with 32-bit codes:
    ///
    /// | rerank_factor | nprobe | recall@10 |
    /// |---------------|--------|-----------|
    /// | 5             | 8      | 0.345     |
    /// | 10            | 8      | 0.579     |
    /// | 25            | 8      | 1.000     |
    /// | 10            | 16     | 0.579     |
    /// | 50            | 16     | 1.000     |
    ///
    /// Recall is bought with pool width, because 32 sign bits are a coarse
    /// ordering. Widening nprobe at fixed pool width changes nothing, which is
    /// the point: codes share one frame, so the extra cells can only add
    /// candidates. The residual-coded version of this index scored 0.739 at
    /// `rf = 10, nprobe = 8` but *fell* to 0.542 at nprobe 16, and cost a
    /// per-cell re-encode on every query to do it.
    #[test]
    fn test_reranking_recall_on_clustered_data() {
        let (n, dim, k) = (2000, 32, 10);
        let data = create_clustered_test_data(n, dim, 8, 21);
        let temp_dir = TempDir::new().unwrap();

        let index = IvfIndexBinary::build_with_vector_store(
            data.as_ref(),
            "sign",
            dim,
            Dist::SquaredEuclidean,
            Some(16),
            get_default_k_means(),
            42,
            false,
            temp_dir.path(),
        )
        .unwrap();

        let mut hits = 0;
        let mut total = 0;

        for row in (0..n).step_by(53) {
            let query: Vec<f32> = data.row(row).iter().cloned().collect();
            let truth = brute_force(&data, &query, k);

            // rerank_factor 10, not 25: at 25 the Hamming stage already returns
            // an eighth of the corpus and pool geometry alone forces 1.0, so
            // the assertion would pass with the codes disabled entirely
            let (got, _) = index.query_reranking(&query, k, Some(8), Some(10)).unwrap();

            hits += got.iter().filter(|i| truth.contains(i)).count();
            total += k;
        }

        let recall = hits as f64 / total as f64;

        // Measures 0.579. A pool of 100 candidates drawn at random out of 2000
        // would score 0.05, so the floor sits well above chance and well below
        // the measurement
        assert!(
            recall > 0.50,
            "recall@{k} on clustered data collapsed to {recall:.3}"
        );
    }

    /// `query_reranking` used to pass `k` as the *k* argument to
    /// `query_asymmetric`, which truncates to that argument internally. The
    /// exact stage was then handed exactly `k` candidates and could only
    /// reorder them, never recover a neighbour the binary stages had dropped,
    /// so re-ranking was a no-op for recall. It must see `k * rerank_factor`.
    #[test]
    fn test_reranking_widens_the_candidate_pool() {
        let (n, dim, k, rerank_factor) = (512, 32, 10, 8);
        let data = create_signed_test_data(n, dim, 3);
        let temp_dir = TempDir::new().unwrap();

        let index = IvfIndexBinary::build_with_vector_store(
            data.as_ref(),
            "sign",
            dim,
            Dist::SquaredEuclidean,
            Some(8),
            get_default_k_means(),
            42,
            false,
            temp_dir.path(),
        )
        .unwrap();

        let mut differs = 0;
        let mut hits_rerank = 0;
        let mut hits_asym = 0;

        for row in (0..n).step_by(37) {
            let query: Vec<f32> = data.row(row).iter().cloned().collect();
            let truth = brute_force(&data, &query, k);

            // Exactly what the exact stage used to be handed
            let (asym, _) = index
                .query_asymmetric(&query, k, Some(8), Some(2 * rerank_factor))
                .unwrap();
            let (rerank, _) = index
                .query_reranking(&query, k, Some(8), Some(rerank_factor))
                .unwrap();

            if rerank.iter().any(|i| !asym.contains(i)) {
                differs += 1;
            }
            hits_asym += asym.iter().filter(|i| truth.contains(i)).count();
            hits_rerank += rerank.iter().filter(|i| truth.contains(i)).count();
        }

        assert!(
            differs > 0,
            "re-ranking returned a permutation of the asymmetric top-k on every \
             query, so the exact stage never saw a wider pool"
        );
        assert!(
            hits_rerank > hits_asym,
            "re-ranking did not improve recall: {hits_rerank} vs {hits_asym} hits"
        );
    }

    /// A `dim` that is not a multiple of 8 leaves a partial last byte. The
    /// padding bits are zero-filled on both sides, so they XOR away, but
    /// nothing pinned that.
    #[test]
    fn test_sign_based_handles_partial_last_byte() {
        for dim in [30, 31, 32] {
            let data = create_signed_test_data(96, dim, 7);
            let index = IvfIndexBinary::build(
                data.as_ref(),
                "sign",
                32,
                Dist::Cosine,
                Some(4),
                get_default_k_means(),
                42,
                false,
            )
            .unwrap();

            assert_eq!(index.n_bytes, dim.div_ceil(8));
            assert_eq!(index.vectors_flat_binarised.len(), index.n * index.n_bytes);

            let query: Vec<f32> = data.row(5).iter().cloned().collect();
            let (indices, dists) = index.query(&query, 3, Some(4)).unwrap();

            assert_eq!(indices[0], 5, "self-retrieval failed at dim = {dim}");
            assert_eq!(dists[0], 0);
        }
    }

    /// The projection-based methods still take their stride from `n_bits`.
    #[test]
    fn test_projection_stride_follows_n_bits() {
        let data = create_test_data::<f32>(128, 32);

        for n_bits in [16, 64] {
            let index = IvfIndexBinary::build(
                data.as_ref(),
                "random",
                n_bits,
                Dist::Cosine,
                Some(4),
                get_default_k_means(),
                42,
                false,
            )
            .unwrap();

            assert_eq!(index.n_bytes, n_bits / 8);
            assert_eq!(index.vectors_flat_binarised.len(), index.n * index.n_bytes);
        }
    }

    #[test]
    fn test_ivf_binary_construction() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(10),
            get_default_k_means(),
            42,
            false,
        )
        .unwrap();

        assert_eq!(index.n, 200);
        assert_eq!(index.n_bytes, 8);
        assert_eq!(index.dim, 32);
        assert_eq!(index.nlist, 10);
        assert_eq!(index.vectors_flat_binarised.len(), 200 * 8);
    }

    #[test]
    fn test_ivf_binary_query_returns_k_results() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(10),
            get_default_k_means(),
            42,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10, Some(10)).unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_ivf_binary_query_sorted() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(10),
            get_default_k_means(),
            42,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (_, distances) = index.query(&query, 10, Some(10)).unwrap();

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_ivf_binary_nprobe_affects_results() {
        let data = create_test_data::<f32>(500, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(20),
            get_default_k_means(),
            42,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices1, _) = index.query(&query, 10, Some(2)).unwrap();
        let (indices2, _) = index.query(&query, 10, Some(10)).unwrap();

        assert!(indices1.len() <= 10);
        assert!(indices2.len() <= 10);
    }

    #[test]
    fn test_ivf_binary_query_row() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(10),
            get_default_k_means(),
            42,
            false,
        )
        .unwrap();

        let (indices, distances) = index.query_row(data.as_ref().row(0), 10, Some(10)).unwrap();

        assert!(indices.len() <= 10);
        assert!(distances.len() <= 10);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_ivf_binary_knn_graph_no_vector_store() {
        let data = create_test_data::<f32>(100, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(10),
            get_default_k_means(),
            42,
            false,
        )
        .unwrap();

        let (knn_indices, knn_distances) =
            index.generate_knn(5, Some(10), None, true, false).unwrap();

        assert_eq!(knn_indices.len(), 100);
        assert!(knn_distances.is_some());

        for neighbours in knn_indices.iter() {
            assert!(neighbours.len() <= 5);
        }
    }

    #[test]
    fn test_build_with_vector_store() {
        let data = create_test_data::<f32>(100, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = IvfIndexBinary::build_with_vector_store(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(10),
            get_default_k_means(),
            42,
            false,
            temp_dir.path(),
        )
        .unwrap();

        assert_eq!(index.n, 100);
        assert_eq!(index.n_bytes, 8);
        assert!(index.vector_store.is_some());
        assert_eq!(index.metric, Dist::Cosine);
    }

    #[test]
    fn test_query_reranking() {
        let data = create_test_data::<f32>(200, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = IvfIndexBinary::build_with_vector_store(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(10),
            get_default_k_means(),
            42,
            false,
            temp_dir.path(),
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index
            .query_reranking(&query, 10, Some(10), Some(5))
            .unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_query_row_reranking() {
        let data = create_test_data::<f32>(200, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = IvfIndexBinary::build_with_vector_store(
            data.as_ref(),
            "random",
            64,
            Dist::SquaredEuclidean,
            Some(10),
            get_default_k_means(),
            42,
            false,
            temp_dir.path(),
        )
        .unwrap();

        let (indices, distances) = index
            .query_row_reranking(data.as_ref().row(0), 10, Some(10), Some(5))
            .unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
        assert!(distances[0] < 1e-5);
    }

    #[test]
    fn test_knn_graph_with_vector_store() {
        let data = create_test_data::<f32>(100, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = IvfIndexBinary::build_with_vector_store(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(10),
            get_default_k_means(),
            42,
            false,
            temp_dir.path(),
        )
        .unwrap();

        let (knn_indices, knn_distances) = index
            .generate_knn(5, Some(10), Some(10), true, false)
            .unwrap();

        assert_eq!(knn_indices.len(), 100);
        assert!(knn_distances.is_some());

        for neighbours in knn_indices.iter() {
            assert_eq!(neighbours.len(), 5);
        }
    }

    #[test]
    fn test_query_reranking_without_vector_store() {
        let data = create_test_data::<f32>(100, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(10),
            get_default_k_means(),
            42,
            false,
        )
        .unwrap();
        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let result = index.query_reranking(&query, 10, Some(10), Some(5));
        assert!(matches!(
            result,
            Err(AnnSearchErrors::VectorStoreNotAvailable)
        ));
    }

    #[test]
    fn test_nprobe_default() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(16),
            get_default_k_means(),
            42,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, _) = index.query(&query, 10, None).unwrap();

        assert!(indices.len() <= 10);
    }

    #[test]
    fn test_query_expands_nprobe_when_probed_cells_underfill_k() {
        // 200 vectors across 40 cells - avg 5 per cell. nprobe=1 alone cannot
        // reach k=25; expansion must walk further cells.
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(40),
            get_default_k_means(),
            42,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 25, Some(1)).unwrap();

        assert_eq!(indices.len(), 25);
        assert_eq!(distances.len(), 25);
    }
}
