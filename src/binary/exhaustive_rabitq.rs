//! Exhaustive RaBitQ index. Compresses the vectors via RaBitQ and does
//! exhaustive searches against query vectors.

use bytemuck::Pod;
use faer::RowRef;
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use std::path::Path;

use crate::binary::dist_binary::*;
use crate::binary::rabitq::rotator::RotatorKind;
use crate::binary::rabitq::*;
use crate::binary::vec_store::*;
use crate::prelude::*;
use crate::utils::k_means_utils::CentroidDistance;
use crate::utils::pack_knn_results;
use crate::utils::verbosity::{BuildLog, QueryProgress};

/// Exhaustive RaBitQ index with multi-centroid support
///
/// Uses IVF-style partitioning with RaBitQ encoding per cluster.
/// At query time, probes the nearest clusters and searches exhaustively
/// within each.
// `bound` is pinned because the skipped `vector_store` field makes serde
// infer a spurious `T: Default`
#[cfg_attr(
    feature = "serialise",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "T: AnnSearchFloat")
)]
pub struct ExhaustiveIndexRaBitQ<T> {
    /// The RaBitQQuantiser
    quantiser: RaBitQQuantiser<T>,
    /// Number of vectors
    n: usize,
    /// Optional on-disk vector storage
    #[cfg_attr(feature = "serialise", serde(skip))]
    vector_store: Option<VectorStore<T>>,
    /// Fingerprint of `vector_store`, checked when it is re-opened after a
    /// load. Only
    /// read by the `serialise` feature; the field is kept unconditionally so
    /// the constructors stay free of `cfg`.
    #[cfg_attr(not(feature = "serialise"), allow(dead_code))]
    store_fingerprint: Option<u64>,
}

//////////////////////////
// VectorDistanceRaBitQ //
//////////////////////////

impl<T> VectorDistanceRaBitQ<T> for ExhaustiveIndexRaBitQ<T>
where
    T: Float + FromPrimitive,
{
    fn storage(&self) -> &RaBitQStorage<T> {
        &self.quantiser.storage
    }

    fn encoder(&self) -> &RaBitQEncoder<T> {
        &self.quantiser.encoder
    }
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for ExhaustiveIndexRaBitQ<T> {
    fn dim(&self) -> usize {
        self.quantiser.encoder.dim
    }
}

impl<T> ExhaustiveIndexRaBitQ<T>
where
    T: AnnSearchFloat + ComplexField + SimdDistance + Pod,
{
    /// Create a new exhaustive RaBitQ index
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (n_samples × dim)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `n_clusters` - Number of clusters. If None, uses 0.5 * sqrt(n)
    /// * `ex_bits` - Magnitude bits per coordinate on top of the sign,
    ///   `0..=8`. Above zero, candidates the one-bit bound cannot rule out
    ///   are re-scored on the wider code
    /// * `seed` - Random seed
    /// * `verbosity` - Prints start and end at `Normal`
    ///
    /// ### Returns
    ///
    /// Initialised index
    pub fn new(
        data: impl AnnMatrix<T>,
        metric: &Dist,
        n_clusters: Option<usize>,
        ex_bits: usize,
        seed: usize,
        verbosity: Verbosity,
    ) -> Result<Self, AnnSearchErrors> {
        let log = BuildLog::new("Exhaustive-RaBitQ index", verbosity);
        let index = Self::with_rotator_kind(data, metric, n_clusters, ex_bits, None, seed);
        log.finish();
        index
    }

    /// Create a new exhaustive RaBitQ index with an explicitly chosen rotation
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (n_samples × dim)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `n_clusters` - Number of clusters. If None, uses 0.5 * sqrt(n)
    /// * `rotator_kind` - Which rotation to encode with, or `None` to let the
    ///   dimensionality decide
    /// * `ex_bits` - Magnitude bits per coordinate on top of the sign,
    ///   `0..=8`. Above zero, candidates the one-bit bound cannot rule out
    ///   are re-scored on the wider code
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Initialised index
    pub fn with_rotator_kind(
        data: impl AnnMatrix<T>,
        metric: &Dist,
        n_clusters: Option<usize>,
        ex_bits: usize,
        rotator_kind: Option<RotatorKind>,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        if *metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(*metric));
        }

        let (vectors_flat, n, dim) = data.into_row_major();
        let quantiser = RaBitQQuantiser::with_rotator_kind(
            (&vectors_flat[..], n, dim),
            metric,
            n_clusters,
            ex_bits,
            rotator_kind,
            seed,
        )?;
        Ok(Self {
            quantiser,
            n,
            vector_store: None,
            store_fingerprint: None,
        })
    }

    /// Create a new exhaustive RaBitQ index with vector store for reranking
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (n_samples × dim)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `n_clusters` - Number of clusters. If None, uses 0.5 * sqrt(n)
    /// * `ex_bits` - Magnitude bits per coordinate on top of the sign,
    ///   `0..=8`. Above zero, candidates the one-bit bound cannot rule out
    ///   are re-scored on the wider code
    /// * `seed` - Random seed
    /// * `verbosity` - Prints start and end at `Normal`
    /// * `save_path` - Where to save the vector storage
    /// * `store_encoding` - Precision the store keeps the rows at
    ///
    /// ### Returns
    ///
    /// Initialised index
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_vector_store(
        data: impl AnnMatrix<T>,
        metric: &Dist,
        n_clusters: Option<usize>,
        ex_bits: usize,
        seed: usize,
        verbosity: Verbosity,
        save_path: impl AsRef<Path>,
        store_encoding: StoreEncoding,
    ) -> Result<Self, AnnSearchErrors> {
        let log = BuildLog::new("Exhaustive-RaBitQ index", verbosity);
        if *metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(*metric));
        }

        // One walk of the caller's matrix. The quantiser normalises its own
        // copy for cosine; the store needs the raw vectors for exact
        // re-ranking, so `vectors_flat` must stay untouched here.
        let (vectors_flat, n, dim) = data.into_row_major();
        let quantiser = RaBitQQuantiser::new(
            (&vectors_flat[..], n, dim),
            metric,
            n_clusters,
            ex_bits,
            seed,
        )?;

        let norms: Vec<T> = vectors_flat
            .chunks_exact(dim)
            .map(compute_l2_norm)
            .collect();

        let vector_store =
            VectorStore::create(&save_path, &vectors_flat, &norms, n, dim, store_encoding)?;
        let store_fingerprint = Some(vector_store.fingerprint());

        log.finish();

        Ok(Self {
            quantiser,
            n,
            vector_store: Some(vector_store),
            store_fingerprint,
        })
    }

    /// Query for k nearest neighbours
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `n_probe` - Number of clusters to search. If None, searches 25% of the
    ///   centroids.
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
    #[inline]
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        n_probe: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let n_probe = n_probe
            .unwrap_or((self.quantiser.n_clusters() as f32 * 0.2) as usize)
            .max(1);
        let k = k.min(self.n);

        let query_normalised = self.quantiser.encoder.normalise_query(query_vec);

        let probed: Vec<usize> = self
            .quantiser
            .get_centroids_prenorm(&query_normalised, n_probe)
            .into_iter()
            .take(n_probe)
            .map(|(_, c_idx)| c_idx)
            .collect();

        let query = self
            .quantiser
            .encoder
            .encode_query(&query_normalised, self.storage().ex_bits)?;

        // The heap ranks on *squared* distance: the square root is monotone, so
        // it only needs applying to the k survivors.
        Ok(self
            .rabitq_scan(
                &query,
                &probed,
                k,
                Refine::without_store(self.storage().ex_bits),
            )
            .into_iter()
            .map(|(d, i)| (i, d.0.sqrt()))
            .unzip())
    }

    /// Query using a row reference
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of neighbours to return
    /// * `n_probe` - Number of clusters to search
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        n_probe: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, n_probe);
        }
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, n_probe)
    }

    /// Query with reranking using exact distances
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `n_probe` - Number of clusters to search. If None, searches 25% of the
    ///   centroids.
    /// * `rerank_factor` - `Some(f)` ranks the `f * k` best estimates exactly.
    ///   `None` lets the estimate's error bound pick which candidates need an
    ///   exact distance, see
    ///   [`query_reranking_bounded`](Self::query_reranking_bounded).
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`.
    #[inline]
    pub fn query_reranking(
        &self,
        query_vec: &[T],
        k: usize,
        n_probe: Option<usize>,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let Some(rerank_factor) = rerank_factor else {
            return self.query_reranking_bounded(query_vec, k, n_probe);
        };
        let vector_store = self
            .vector_store
            .as_ref()
            .ok_or(AnnSearchErrors::VectorStoreNotAvailable)?;

        let (candidates, _) = self.query(query_vec, k * rerank_factor, n_probe)?;

        let query_norm = match self.quantiser.encoder.metric {
            Dist::Cosine => compute_l2_norm(query_vec),
            Dist::SquaredEuclidean => T::one(),
            Dist::Manhattan => unreachable!(),
        };

        let mut scored: Vec<_> = candidates
            .iter()
            .map(|&idx| {
                let dist = match self.quantiser.encoder.metric {
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

    /// Query with exact distances where the estimate's bound requires them
    ///
    /// One pass: every candidate whose lower bound beats the current `k`-th
    /// exact distance is scored exactly from the vector store, see
    /// [`VectorDistanceRaBitQ::rabitq_scan`]. No shortlist size to pick.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `n_probe` - Number of clusters to search. If None, searches 20% of the
    ///   centroids.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`.
    #[inline]
    pub fn query_reranking_bounded(
        &self,
        query_vec: &[T],
        k: usize,
        n_probe: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let vector_store = self
            .vector_store
            .as_ref()
            .ok_or(AnnSearchErrors::VectorStoreNotAvailable)?;

        let n_probe = n_probe
            .unwrap_or((self.quantiser.n_clusters() as f32 * 0.2) as usize)
            .max(1);
        let k = k.min(self.n);

        let query_normalised = self.quantiser.encoder.normalise_query(query_vec);
        let probed: Vec<usize> = self
            .quantiser
            .get_centroids_prenorm(&query_normalised, n_probe)
            .into_iter()
            .take(n_probe)
            .map(|(_, c_idx)| c_idx)
            .collect();
        let query = self
            .quantiser
            .encoder
            .encode_query(&query_normalised, self.storage().ex_bits)?;

        // The estimate lives in the normalised space for cosine, where the
        // squared distance is twice the cosine distance.
        let metric = self.quantiser.encoder.metric;
        let query_norm = compute_l2_norm(query_vec);
        let two = T::one() + T::one();
        let exact = |idx: usize| match metric {
            Dist::Cosine => two * vector_store.cosine_distance_to_query(idx, query_vec, query_norm),
            Dist::SquaredEuclidean => vector_store.euclidean_distance_to_query(idx, query_vec),
            Dist::Manhattan => unreachable!(),
        };

        Ok(self
            .rabitq_scan(&query, &probed, k, Refine::Exact(exact))
            .into_iter()
            .map(|(d, i)| match metric {
                Dist::Cosine => (i, d.0 / two),
                _ => (i, d.0),
            })
            .unzip())
    }

    /// Query row with reranking using exact distances
    ///
    /// Function with optimised path for distances
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `n_probe` - Number of clusters to search. If None, searches 25% of the
    ///   centroids.
    /// * `rerank_factor` - `Some(f)` ranks the `f * k` best estimates exactly.
    ///   `None` lets the estimate's error bound pick which candidates need an
    ///   exact distance, see
    ///   [`query_reranking_bounded`](Self::query_reranking_bounded).
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`.
    #[inline]
    pub fn query_row_reranking(
        &self,
        query_row: RowRef<T>,
        k: usize,
        n_probe: Option<usize>,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query_reranking(slice, k, n_probe, rerank_factor);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query_reranking(&query_vec, k, n_probe, rerank_factor)
    }

    /// Generate kNN graph
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `n_probe` - Number of clusters to search per query
    /// * `rerank_factor` - Shortlist multiplier for exact distances, `None` for
    ///   the bound-gated rerank
    /// * `return_dist` - Whether to return distances
    /// * `verbosity` - Prints start, progress and end at `Normal`
    ///
    /// ### Returns
    ///
    /// Tuple of (knn_indices, optional distances)
    pub fn generate_knn(
        &self,
        k: usize,
        n_probe: Option<usize>,
        rerank_factor: Option<usize>,
        return_dist: bool,
        verbosity: Verbosity,
    ) -> KnnOptionResult<T> {
        let vector_store = self
            .vector_store
            .as_ref()
            .ok_or(AnnSearchErrors::VectorStoreNotAvailable)?;

        let progress = QueryProgress::new("Exhaustive-RaBitQ index", self.n, true, verbosity);

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let vec = vector_store.row(i);

                progress.tick();

                self.query_reranking(&vec, k, n_probe, rerank_factor)
            })
            .collect::<Result<Vec<_>, _>>()?;
        progress.finish();

        Ok(pack_knn_results(results, return_dist))
    }

    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.quantiser.memory_usage_bytes()
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
impl<T> IndexIo for ExhaustiveIndexRaBitQ<T>
where
    T: AnnSearchFloat + Pod,
{
    type Elem = T;

    const KIND: &'static str = "exhaustive_rabitq";

    fn stage_aux(&self, dir: &Path, staged: &mut StagedFiles) -> Result<(), AnnSearchErrors> {
        match &self.vector_store {
            Some(store) => store.stage_copy_into(dir, staged),
            None => Ok(()),
        }
    }

    fn load_aux(&mut self, dir: &Path) -> Result<(), AnnSearchErrors> {
        self.vector_store = VectorStore::open_for(
            dir,
            self.store_fingerprint,
            self.n,
            self.quantiser.encoder.dim,
        )?;
        self.quantiser.storage.reblock_for_this_arch();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
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

    /// Clustered Gaussian-ish data at a dimensionality where the encoder picks
    /// the Hadamard rotation, so the padded code path is actually exercised.
    fn clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Mat<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / (1u64 << 53) as f32 - 0.5
        };

        let centres: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| next() * 8.0).collect())
            .collect();

        Mat::from_fn(n, dim, |i, j| centres[i % n_clusters][j] + next())
    }

    fn brute_force(data: &Mat<f32>, query: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = (0..data.nrows())
            .map(|i| {
                let d = (0..data.ncols())
                    .map(|j| (data[(i, j)] - query[j]).powi(2))
                    .sum::<f32>();
                (d, i)
            })
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    #[test]
    fn test_exhaustive_rabitq_recall_at_the_padded_dimension() {
        let (n, dim, k) = (2000usize, 100usize, 10usize);
        let data = clustered(n, dim, 20, 7);
        let index = ExhaustiveIndexRaBitQ::with_rotator_kind(
            data.as_ref(),
            &Dist::SquaredEuclidean,
            Some(40),
            0,
            Some(RotatorKind::FhtKac),
            42,
        )
        .unwrap();

        // 100 dimensions rounds up to 128 under the Hadamard rotation, so the
        // codes carry 28 bits that no input coordinate maps to directly.
        assert_eq!(index.quantiser.encoder.padded_dim, 128);
        assert_eq!(index.quantiser.encoder.n_bytes, 16);

        let mut hits = 0usize;
        let n_queries = 40;
        for q in 0..n_queries {
            let query: Vec<f32> = (0..dim).map(|j| data[(q * 7 % n, j)]).collect();
            let truth = brute_force(&data, &query, k);
            let (got, _) = index.query(&query, k, Some(40)).unwrap();
            hits += got.iter().filter(|i| truth.contains(i)).count();
        }

        let recall = hits as f64 / (n_queries * k) as f64;
        assert!(recall > 0.6, "recall@{k} was {recall}");
    }

    #[test]
    fn test_exhaustive_rabitq_construction() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(
            data.as_ref(),
            &Dist::SquaredEuclidean,
            Some(10),
            0,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        assert_eq!(index.n, 100);
        assert_eq!(index.quantiser.n_clusters(), 10);
        assert_eq!(index.quantiser.n_vectors(), 100);
    }

    #[test]
    fn test_exhaustive_rabitq_query_returns_k_results() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(
            data.as_ref(),
            &Dist::SquaredEuclidean,
            Some(10),
            0,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10, Some(5)).unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_rabitq_query_sorted() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(
            data.as_ref(),
            &Dist::SquaredEuclidean,
            Some(10),
            0,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (_, distances) = index.query(&query, 10, Some(5)).unwrap();

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_exhaustive_rabitq_query_k_exceeds_n() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexRaBitQ::new(
            data.as_ref(),
            &Dist::SquaredEuclidean,
            Some(5),
            0,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, _) = index.query(&query, 100, Some(5)).unwrap();

        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_exhaustive_rabitq_query_row() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(
            data.as_ref(),
            &Dist::SquaredEuclidean,
            Some(10),
            0,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let (indices, distances) = index.query_row(data.as_ref().row(0), 10, Some(5)).unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_rabitq_cosine() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(
            data.as_ref(),
            &Dist::Cosine,
            Some(10),
            0,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10, Some(10)).unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_rabitq_default_nprobe() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(
            data.as_ref(),
            &Dist::SquaredEuclidean,
            Some(10),
            0,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, _) = index.query(&query, 10, None).unwrap();

        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_new_with_vector_store() {
        let data = create_test_data::<f32>(50, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = ExhaustiveIndexRaBitQ::new_with_vector_store(
            data.as_ref(),
            &Dist::SquaredEuclidean,
            Some(5),
            0,
            42,
            Verbosity::Quiet,
            temp_dir.path(),
            StoreEncoding::Full,
        )
        .unwrap();

        assert_eq!(index.n, 50);
        assert!(index.vector_store.is_some());
    }

    #[test]
    fn test_query_reranking() {
        let data = create_test_data::<f32>(100, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = ExhaustiveIndexRaBitQ::new_with_vector_store(
            data.as_ref(),
            &Dist::Cosine,
            Some(10),
            0,
            42,
            Verbosity::Quiet,
            temp_dir.path(),
            StoreEncoding::Full,
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
        let data = create_test_data::<f32>(100, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = ExhaustiveIndexRaBitQ::new_with_vector_store(
            data.as_ref(),
            &Dist::SquaredEuclidean,
            Some(10),
            0,
            42,
            Verbosity::Quiet,
            temp_dir.path(),
            StoreEncoding::Full,
        )
        .unwrap();

        let (indices, distances) = index
            .query_row_reranking(data.as_ref().row(0), 10, Some(5), Some(5))
            .unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_knn_graph_with_vector_store() {
        let data = create_test_data::<f32>(50, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = ExhaustiveIndexRaBitQ::new_with_vector_store(
            data.as_ref(),
            &Dist::Cosine,
            Some(5),
            0,
            42,
            Verbosity::Quiet,
            temp_dir.path(),
            StoreEncoding::Full,
        )
        .unwrap();

        let (knn_indices, knn_distances) = index
            .generate_knn(5, Some(5), Some(10), true, Verbosity::Quiet)
            .unwrap();

        assert_eq!(knn_indices.len(), 50);
        assert!(knn_distances.is_some());
        assert_eq!(knn_distances.as_ref().unwrap().len(), 50);

        for neighbours in knn_indices.iter() {
            assert_eq!(neighbours.len(), 5);
        }
    }

    #[test]
    fn test_knn_without_vector_store_panics() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexRaBitQ::new(
            data.as_ref(),
            &Dist::SquaredEuclidean,
            Some(5),
            0,
            42,
            Verbosity::Quiet,
        )
        .unwrap();
        let result = index.generate_knn(5, Some(5), Some(10), false, Verbosity::Quiet);

        assert!(matches!(
            result,
            Err(AnnSearchErrors::VectorStoreNotAvailable)
        ));
    }

    #[test]
    fn test_query_reranking_without_vector_store_panics() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexRaBitQ::new(
            data.as_ref(),
            &Dist::SquaredEuclidean,
            Some(5),
            0,
            42,
            Verbosity::Quiet,
        )
        .unwrap();
        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let result = index.query_reranking(&query, 10, Some(5), Some(5));
        assert!(matches!(
            result,
            Err(AnnSearchErrors::VectorStoreNotAvailable)
        ));
    }
    #[test]
    fn test_bounded_reranking_matches_a_full_rerank() {
        // Probing every cluster, a shortlist covering the whole dataset is the
        // exact answer. The bound is probabilistic, so allow a near-tie or two
        // to differ, but the distances must agree almost everywhere.
        let (n, dim, k) = (400usize, 64usize, 10usize);
        let data = Mat::from_fn(n, dim, |i, j| {
            let x = ((i * 7919 + j * 104_729) % 1009) as f32 / 1009.0;
            x + (i % 5) as f32 * 0.5
        });

        for metric in [Dist::SquaredEuclidean, Dist::Cosine] {
            let temp_dir = TempDir::new().unwrap();
            let index = ExhaustiveIndexRaBitQ::new_with_vector_store(
                data.as_ref(),
                &metric,
                Some(8),
                0,
                42,
                Verbosity::Quiet,
                temp_dir.path(),
                StoreEncoding::Full,
            )
            .unwrap();

            let mut agree = 0;
            for qi in 0..20 {
                let query: Vec<f32> = data.row(qi * 13).iter().map(|x| x + 0.01).collect();
                let (_, full) = index
                    .query_reranking(&query, k, Some(8), Some(n / k))
                    .unwrap();
                let (_, bounded) = index.query_reranking_bounded(&query, k, Some(8)).unwrap();

                assert_eq!(bounded.len(), k);
                assert!(bounded.windows(2).all(|w| w[0] <= w[1]));
                agree += full
                    .iter()
                    .zip(&bounded)
                    .filter(|(a, b)| (*a - *b).abs() <= 1e-4 * a.abs().max(1.0))
                    .count();
            }
            assert!(
                agree >= 20 * k * 95 / 100,
                "{metric:?}: {agree} / {}",
                20 * k
            );
        }
    }
}
