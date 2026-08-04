//! Exhaustive binary index. Compresses the vectors via binarisation and does
//! exhaustive searches against query vectors.

use bytemuck::Pod;
use faer::{MatRef, RowRef};
use faer_traits::ComplexField;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::path::Path;
use thousands::*;

use crate::binary::binariser::*;
use crate::binary::dist_binary::*;
use crate::binary::vec_store::*;
use crate::prelude::*;

///////////////////////////
// ExhaustiveIndexBinary //
///////////////////////////

/// Exhaustive (brute-force) binary nearest neighbour index
///
/// Stores vectors as binary codes and uses Hamming distance for queries.
// `bound` is pinned because the skipped `vector_store` field makes serde
// infer a spurious `T: Default`
#[cfg_attr(
    feature = "serialise",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "T: AnnSearchFloat")
)]
pub struct ExhaustiveIndexBinary<T> {
    /// Binary codes, flattened (n * n_bytes)
    pub vectors_flat_binarised: Vec<u8>,
    /// Bytes per vector, taken from the binariser. Equals `n_bits / 8` for the
    /// projection-based methods; sign-based ignores `n_bits` and uses `dim`.
    pub n_bytes: usize,
    /// Number of samples in the index
    pub n: usize,
    /// Original dimensionality
    dim: usize,
    /// Distance metric to use
    metric: Dist,
    /// Binarisation type to use
    binarisation_type: BinarisationInit,
    /// Binariser
    binariser: Binariser<T>,
    /// Optional vector store that is saved in binary on disk
    #[cfg_attr(feature = "serialise", serde(skip))]
    vector_store: Option<MmapVectorStore<T>>,
    /// Shape of `vector_store`, so it can be re-opened after a load. Only
    /// read by the `serialise` feature; the field is kept unconditionally so
    /// the constructors stay free of `cfg`.
    #[cfg_attr(not(feature = "serialise"), allow(dead_code))]
    store_meta: Option<StoreMeta>,
}

//////////////////////////
// VectorDistanceBinary //
//////////////////////////

impl<T> VectorDistanceBinary for ExhaustiveIndexBinary<T> {
    fn vectors_flat_binarised(&self) -> &[u8] {
        &self.vectors_flat_binarised
    }

    fn n_bytes(&self) -> usize {
        self.n_bytes
    }
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for ExhaustiveIndexBinary<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

///////////////////////////
// ExhaustiveIndexBinary //
///////////////////////////

impl<T> ExhaustiveIndexBinary<T>
where
    T: AnnSearchFloat + ComplexField + Pod,
{
    /// Generate a new exhaustive binary index
    ///
    /// Binarises all vectors using the specified hash function and stores them
    /// as compact binary codes. This works solely for Cosine distance!
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features)
    /// * `binarisation_init` - Initialisation method (`"random"`, `"pca"`, or
    ///   `"sign"`)
    /// * `n_bits` - Number of bits per binary code (must be multiple of 8).
    ///   Ignored by sign-based binarisation, which always emits `dim` bits.
    /// * `seed` - Random seed for binariser
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive binary index
    pub fn new(
        data: MatRef<T>,
        binarisation_init: &str,
        n_bits: usize,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        if !n_bits.is_multiple_of(8) {
            return Err(AnnSearchErrors::NBitsMustBe8Multiple { n_bits });
        }

        let init = parse_binarisation_init(binarisation_init).unwrap_or_else(|| {
            println!("[WARNING] Unknown binarisation string provided. Using the default");
            BinarisationInit::default()
        });

        let n = data.nrows();
        let dim = data.ncols();

        let binariser = match init {
            BinarisationInit::PcaHashing => Binariser::new_pca_hashing(data, dim, n_bits, seed)?,
            BinarisationInit::RandomProjections => Binariser::new_simhash(data, dim, n_bits, seed)?,
            BinarisationInit::SignBased => Binariser::new_sign_based(dim),
        };

        // Ask the binariser, do not derive from `n_bits`: sign-based ignores
        // that argument and emits `dim` bits
        let n_bytes = binariser.n_bytes();

        let mut vectors_flat_binarised: Vec<u8> = Vec::with_capacity(n * n_bytes);

        for i in 0..n {
            let original: Vec<T> = data.row(i).iter().cloned().collect();
            vectors_flat_binarised.extend(binariser.encode(&original)?);
        }

        Ok(Self {
            vectors_flat_binarised,
            n_bytes,
            n,
            dim,
            binariser,
            binarisation_type: init,
            vector_store: None,
            store_meta: None,
            metric: Dist::Cosine,
        })
    }

    /// Generate a new exhaustive binary index with vector store for reranking
    ///
    /// Creates binary index and saves/loads vector store for exact distance reranking.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features)
    /// * `binarisation_init` - Initialisation method (`"random"`, `"pca"`, or
    ///   `"sign"`)
    /// * `n_bits` - Number of bits per binary code (must be multiple of 8).
    ///   Ignored by sign-based binarisation, which always emits `dim` bits.
    /// * `metric` - Distance metric for reranking
    /// * `seed` - Random seed for binariser
    /// * `save_path` - Directory to save vector store files
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive binary index with vector store
    pub fn new_with_vector_store(
        data: MatRef<T>,
        binarisation_init: &str,
        n_bits: usize,
        metric: Dist,
        seed: usize,
        save_path: impl AsRef<Path>,
    ) -> Result<Self, AnnSearchErrors> {
        if !n_bits.is_multiple_of(8) {
            return Err(AnnSearchErrors::NBitsMustBe8Multiple { n_bits });
        }

        let init = parse_binarisation_init(binarisation_init).unwrap_or_default();

        let n = data.nrows();
        let dim = data.ncols();

        let binariser = match init {
            BinarisationInit::PcaHashing => Binariser::new_pca_hashing(data, dim, n_bits, seed)?,
            BinarisationInit::RandomProjections => Binariser::new_simhash(data, dim, n_bits, seed)?,
            BinarisationInit::SignBased => Binariser::new_sign_based(dim),
        };

        // Ask the binariser, do not derive from `n_bits`: sign-based ignores
        // that argument and emits `dim` bits
        let n_bytes = binariser.n_bytes();

        let mut vectors_flat_binarised: Vec<u8> = Vec::with_capacity(n * n_bytes);

        for i in 0..n {
            let original: Vec<T> = data.row(i).iter().cloned().collect();
            vectors_flat_binarised.extend(binariser.encode(&original)?);
        }

        // Save vector store
        std::fs::create_dir_all(&save_path)?;

        let vectors_flat: Vec<T> = (0..n).flat_map(|i| data.row(i).iter().cloned()).collect();

        let norms: Vec<T> = (0..n)
            .map(|i| {
                data.row(i)
                    .iter()
                    .map(|&x| x * x)
                    .fold(T::zero(), |a, b| a + b)
                    .sqrt()
            })
            .collect();

        let (vectors_path, norms_path) = MmapVectorStore::<T>::paths_in(&save_path);

        MmapVectorStore::save(&vectors_flat, &norms, dim, n, &vectors_path, &norms_path)?;

        let vector_store = MmapVectorStore::new(vectors_path, norms_path, dim, n)?;

        Ok(Self {
            vectors_flat_binarised,
            n_bytes,
            n,
            dim,
            binariser,
            binarisation_type: init,
            vector_store: Some(vector_store),
            store_meta: Some(StoreMeta { dim, n }),
            metric,
        })
    }

    ///////////
    // Query //
    ///////////

    /// Query function
    ///
    /// Exhaustive search over all binary codes using Hamming distance.
    /// Binary codes are generated via the trained binariser during
    /// initialisation.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (will be binarised internally)
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` where distances are Hamming distances
    #[inline]
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<u32>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        let query_binary = self.binariser.encode(query_vec)?;
        let k = k.min(self.n);

        let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(k + 1);

        for idx in 0..self.n {
            let dist = self.hamming_distance_query(&query_binary, idx);

            if heap.len() < k {
                heap.push((dist, idx));
            } else if dist < heap.peek().unwrap().0 {
                heap.pop();
                heap.push((dist, idx));
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);

        let (distances, indices): (Vec<_>, Vec<_>) = results.into_iter().unzip();

        Ok((indices, distances))
    }

    /// Query function for asymmetric querying
    ///
    /// This function will first calculate the Hamming distance between the
    /// query vector and the binary codes and then do an additional asymmetric
    /// dot product distance calculation between the query vector and the binary
    /// code of the candidates.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (will be binarised internally)
    /// * `k` - Number of nearest neighbours to return
    /// * `rerank_factor` - Multiplier for candidate set size (searches k *
    ///   rerank_factor candidates). If not supplied, defaults to `20`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, scores)` where the score is the asymmetric dot
    /// product, sorted descending: higher means more similar. Errors when the
    /// binarisation type is not sign-based.
    pub fn query_asymmetric(
        &self,
        query_vec: &[T],
        k: usize,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if !self.use_asymmetric() {
            return Err(AnnSearchErrors::AsymmetricQueryMisMatch);
        }

        let rerank_factor = rerank_factor.unwrap_or(20);
        let k = k.min(self.n);

        let (candidates, _) = self.query(query_vec, k * rerank_factor)?;

        let mut scored: Vec<(usize, T)> = candidates
            .iter()
            .map(|&idx| {
                let start_i = idx * self.n_bytes;
                let vec_i = unsafe {
                    self.vectors_flat_binarised
                        .get_unchecked(start_i..start_i + self.n_bytes)
                };

                let dist_i = asymmetric_binary_dot(query_vec, vec_i, self.dim);
                (idx, dist_i)
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

    /// Query function for row references
    ///
    /// Exhaustive search using Hamming distance on binarised query. Leverages
    /// optimised unsafe paths if possible.
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
    ) -> Result<(Vec<usize>, Vec<u32>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k)
    }

    /// Query function for row references (asymmetric)
    ///
    /// Exhaustive search using Hamming distance on binarised query. Leverages
    /// optimised unsafe paths if possible.
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, scores)`, see [`ExhaustiveIndexBinary::query_asymmetric`]
    pub fn query_row_asymmetric(
        &self,
        query_row: RowRef<T>,
        k: usize,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query_asymmetric(slice, k, rerank_factor);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query_asymmetric(&query_vec, k, rerank_factor)
    }

    /// Query with reranking using exact distances
    ///
    /// Two-stage search: Hamming distance to find candidates, then exact
    /// distance for final ranking. Requires vector_store to be available.
    /// If the binarisation type is sign-based, an intermediate reranking
    /// via dot product calculations will be done.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of nearest neighbours to return
    /// * `rerank_factor` - Multiplier for candidate set size (searches k *
    ///   rerank_factor candidates). If not supplied, defaults to `20`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` where distances are exact (Euclidean or
    /// Cosine)
    #[inline]
    pub fn query_reranking(
        &self,
        query_vec: &[T],
        k: usize,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let rerank_factor = rerank_factor.unwrap_or(20);
        let vector_store = self
            .vector_store
            .as_ref()
            .ok_or(AnnSearchErrors::VectorStoreNotAvailable)?;

        // `query_asymmetric` truncates to its own `k`, so it has to be asked for
        // `k * rerank_factor` candidates, not `k`. Passing `k` collapses the
        // funnel to `k -> k` and the exact stage can only reorder what the
        // binary stages already picked, never recover a dropped neighbour.
        let candidates = if matches!(self.binarisation_type, BinarisationInit::SignBased) {
            let (idx, _) = self.query_asymmetric(query_vec, k * rerank_factor, Some(2))?;
            idx
        } else {
            let (idx, _) = self.query(query_vec, k * rerank_factor)?;
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
                (idx, dist)
            })
            .collect();

        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);

        let mut indices: Vec<usize> = Vec::with_capacity(k);
        let mut dists: Vec<T> = Vec::with_capacity(k);

        for (idx, dist) in scored {
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
    /// * `k` - Number of nearest neighbours to return
    /// * `rerank_factor` - Multiplier for candidate set size (searches k *
    ///   rerank_factor candidates). If not supplied, defaults to `20`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    #[inline]
    pub fn query_row_reranking(
        &self,
        query_row: RowRef<T>,
        k: usize,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query_reranking(slice, k, rerank_factor);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query_reranking(&query_vec, k, rerank_factor)
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// If vector_store is available, uses it for exact distance reranking.
    /// Otherwise, uses Hamming distances only.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `rerank_factor` - Multiplier for candidate set (only used if
    ///   vector_store available)
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`
    pub fn generate_knn(
        &self,
        k: usize,
        rerank_factor: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<T> {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let counter = Arc::new(AtomicUsize::new(0));

        if let Some(vector_store) = &self.vector_store {
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

                    self.query_reranking(vec, k, rerank_factor)
                })
                .collect::<Result<Vec<_>, _>>()?;

            if return_dist {
                let (indices, distances) = results.into_iter().unzip();
                Ok((indices, Some(distances)))
            } else {
                let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _)| idx).collect();
                Ok((indices, None))
            }
        } else {
            // Fallback to binary-only search
            let results: Vec<(Vec<usize>, Vec<u32>)> = (0..self.n)
                .into_par_iter()
                .map(|i| {
                    let start = i * self.n_bytes;
                    let query_binary = &self.vectors_flat_binarised[start..start + self.n_bytes];

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

                    let k = k.min(self.n);
                    let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(k + 1);

                    for idx in 0..self.n {
                        let dist = self.hamming_distance_query(query_binary, idx);

                        if heap.len() < k {
                            heap.push((dist, idx));
                        } else if dist < heap.peek().unwrap().0 {
                            heap.pop();
                            heap.push((dist, idx));
                        }
                    }

                    let mut results: Vec<(u32, usize)> = heap.into_iter().collect();
                    results.sort_unstable_by_key(|&(dist, _)| dist);

                    let (distances, indices): (Vec<_>, Vec<_>) = results.into_iter().unzip();

                    (indices, distances)
                })
                .collect();

            if return_dist {
                let (indices, distances): (Vec<Vec<usize>>, Vec<Vec<u32>>) =
                    results.into_iter().unzip();
                let distances_converted: Vec<Vec<T>> = distances
                    .into_iter()
                    .map(|v| v.into_iter().map(|d| T::from_u32(d).unwrap()).collect())
                    .collect();
                Ok((indices, Some(distances_converted)))
            } else {
                let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _)| idx).collect();
                Ok((indices, None))
            }
        }
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat_binarised.capacity()
            + self.binariser.memory_usage_bytes()
    }

    /// Returns whether the index supports asymmetric queries
    ///
    /// ### Returns
    ///
    /// True if yes
    pub fn use_asymmetric(&self) -> bool {
        matches!(self.binarisation_type, BinarisationInit::SignBased)
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
impl<T> IndexIo for ExhaustiveIndexBinary<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "exhaustive_binary";

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
    /// comes out as all ones and every Hamming distance is 0. Anything testing
    /// the sign path needs this instead.
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

    /// Sign-based binarisation ignores `n_bits` and emits `dim` bits, so the
    /// code stride must come from the binariser. Taking it from `n_bits`
    /// instead built out-of-bounds slices in the Hamming kernel: an abort with
    /// debug assertions on, plausible-looking garbage without.
    #[test]
    fn test_sign_based_stride_ignores_n_bits() {
        let (n, dim) = (128, 32);
        let data = create_signed_test_data(n, dim, 7);

        for n_bits in [8, 32, 64, 128] {
            let index = ExhaustiveIndexBinary::new(data.as_ref(), "sign", n_bits, 42).unwrap();

            assert_eq!(index.n_bytes, dim / 8, "stride tracked n_bits = {}", n_bits);
            assert_eq!(index.vectors_flat_binarised.len(), n * index.n_bytes);

            // Mixed signs make every code distinct, so a stored vector must
            // find itself at Hamming distance 0 before anything else
            for row in [0, 17, 63, 127] {
                let query: Vec<f32> = data.row(row).iter().cloned().collect();
                let (indices, dists) = index.query(&query, 5).unwrap();

                assert_eq!(indices.len(), 5);
                assert_eq!(indices[0], row, "self-retrieval failed, n_bits = {n_bits}");
                assert_eq!(dists[0], 0);
            }
        }
    }

    /// The store-backed constructor takes the same path and used to have the
    /// same defect.
    #[test]
    fn test_sign_based_stride_with_vector_store() {
        let (n, dim) = (64, 32);
        let data = create_signed_test_data(n, dim, 11);
        let temp_dir = TempDir::new().unwrap();

        let index = ExhaustiveIndexBinary::new_with_vector_store(
            data.as_ref(),
            "sign",
            128,
            Dist::Cosine,
            42,
            temp_dir.path(),
        )
        .unwrap();

        assert_eq!(index.n_bytes, dim / 8);
        assert_eq!(index.vectors_flat_binarised.len(), n * index.n_bytes);

        // Re-ranking runs through `query_asymmetric` on the sign path, which is
        // where the out-of-bounds slice used to be built
        for row in [0, 31, 63] {
            let query: Vec<f32> = data.row(row).iter().cloned().collect();
            let (indices, dists) = index.query_reranking(&query, 5, Some(4)).unwrap();

            assert_eq!(indices[0], row);
            assert!(dists[0] < 1e-5);
        }
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

        let index = ExhaustiveIndexBinary::new_with_vector_store(
            data.as_ref(),
            "sign",
            dim,
            Dist::SquaredEuclidean,
            42,
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
                .query_asymmetric(&query, k, Some(2 * rerank_factor))
                .unwrap();
            let (rerank, _) = index
                .query_reranking(&query, k, Some(rerank_factor))
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

    /// Recall floor for the sign path on origin-centred data
    ///
    /// This is the regime sign binarisation is designed for: coordinates are
    /// roughly zero-mean, so each bit is balanced and Hamming distance tracks
    /// angle. There is no cell structure to take a residual against here, so
    /// this is as good as the exhaustive binary index gets.
    #[test]
    fn test_reranking_recall_on_centred_data() {
        let (n, dim, k) = (2000, 32, 10);
        let data = create_signed_test_data(n, dim, 21);
        let temp_dir = TempDir::new().unwrap();

        let index = ExhaustiveIndexBinary::new_with_vector_store(
            data.as_ref(),
            "sign",
            dim,
            Dist::SquaredEuclidean,
            42,
            temp_dir.path(),
        )
        .unwrap();

        let mut hits = 0;
        let mut total = 0;

        for row in (0..n).step_by(53) {
            let query: Vec<f32> = data.row(row).iter().cloned().collect();
            let truth = brute_force(&data, &query, k);

            let (got, _) = index.query_reranking(&query, k, Some(25)).unwrap();

            hits += got.iter().filter(|i| truth.contains(i)).count();
            total += k;
        }

        let recall = hits as f64 / total as f64;

        assert!(recall > 0.9, "recall@{k} collapsed to {recall:.3}");
    }

    /// A `dim` that is not a multiple of 8 leaves a partial last byte. The
    /// padding bits are zero-filled on both sides, so they XOR away.
    #[test]
    fn test_sign_based_handles_partial_last_byte() {
        for dim in [30, 31, 32] {
            let data = create_signed_test_data(64, dim, 7);
            let index = ExhaustiveIndexBinary::new(data.as_ref(), "sign", 32, 42).unwrap();

            assert_eq!(index.n_bytes, dim.div_ceil(8));
            assert_eq!(index.vectors_flat_binarised.len(), 64 * index.n_bytes);

            let query: Vec<f32> = data.row(5).iter().cloned().collect();
            let (indices, dists) = index.query(&query, 3).unwrap();

            assert_eq!(indices[0], 5, "self-retrieval failed at dim = {dim}");
            assert_eq!(dists[0], 0);
        }
    }

    /// The projection-based methods still take their stride from `n_bits`.
    #[test]
    fn test_projection_stride_follows_n_bits() {
        let data = create_signed_test_data(64, 32, 7);

        for n_bits in [16, 64] {
            let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", n_bits, 42).unwrap();

            assert_eq!(index.n_bytes, n_bits / 8);
            assert_eq!(index.vectors_flat_binarised.len(), 64 * index.n_bytes);
        }
    }

    #[test]
    fn test_exhaustive_binary_construction() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42).unwrap();

        assert_eq!(index.n, 100);
        assert_eq!(index.n_bytes, 8);
        assert_eq!(index.vectors_flat_binarised.len(), 100 * 8);
    }

    #[test]
    fn test_exhaustive_binary_query_returns_k_results() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42).unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10).unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_binary_query_sorted() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42).unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (_, distances) = index.query(&query, 10).unwrap();

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_exhaustive_binary_query_k_exceeds_n() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42).unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, _) = index.query(&query, 100).unwrap();

        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_exhaustive_binary_query_row() {
        // `create_test_data` ramps along a straight line, so once SimHash
        // centres on the training mean a whole block of rows shares one code
        // and self-retrieval is a coin flip between ties at Hamming 0
        let data = create_signed_test_data(100, 32, 5);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42).unwrap();

        let (indices1, distances1) = index.query_row(data.as_ref().row(0), 10).unwrap();

        assert_eq!(indices1.len(), 10);
        assert_eq!(distances1.len(), 10);
        assert_eq!(indices1[0], 0);
    }

    #[test]
    fn test_exhaustive_binary_knn_graph_no_vector_store() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42).unwrap();

        let (knn_indices, knn_distances) = index.generate_knn(5, None, true, false).unwrap();

        assert_eq!(knn_indices.len(), 50);
        assert!(knn_distances.is_some());
        assert_eq!(knn_distances.as_ref().unwrap().len(), 50);

        for neighbours in knn_indices.iter() {
            assert_eq!(neighbours.len(), 5);
        }
    }

    #[test]
    fn test_hamming_distances_in_valid_range() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42).unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (_, distances) = index.query(&query, 20).unwrap();

        for &dist in &distances {
            assert!(dist <= 64);
        }
    }

    #[test]
    fn test_new_with_vector_store() {
        let data = create_test_data::<f32>(50, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = ExhaustiveIndexBinary::new_with_vector_store(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            42,
            temp_dir.path(),
        )
        .unwrap();

        assert_eq!(index.n, 50);
        assert_eq!(index.n_bytes, 8);
        assert!(index.vector_store.is_some());
        assert_eq!(index.metric, Dist::Cosine);
    }

    #[test]
    fn test_query_reranking() {
        let data = create_test_data::<f32>(100, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = ExhaustiveIndexBinary::new_with_vector_store(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            42,
            temp_dir.path(),
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query_reranking(&query, 10, Some(5)).unwrap();

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

        let index = ExhaustiveIndexBinary::new_with_vector_store(
            data.as_ref(),
            "random",
            64,
            Dist::SquaredEuclidean,
            42,
            temp_dir.path(),
        )
        .unwrap();

        let (indices, distances) = index
            .query_row_reranking(data.as_ref().row(0), 10, Some(5))
            .unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
        assert_eq!(indices[0], 0);
        assert!(distances[0] < 1e-5);
    }

    #[test]
    fn test_knn_graph_with_vector_store() {
        let data = create_test_data::<f32>(50, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = ExhaustiveIndexBinary::new_with_vector_store(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            42,
            temp_dir.path(),
        )
        .unwrap();

        let (knn_indices, knn_distances) = index.generate_knn(5, Some(10), true, false).unwrap();

        assert_eq!(knn_indices.len(), 50);
        assert!(knn_distances.is_some());
        assert_eq!(knn_distances.as_ref().unwrap().len(), 50);

        for neighbours in knn_indices.iter() {
            assert_eq!(neighbours.len(), 5);
        }
    }

    #[test]
    fn test_query_reranking_without_vector_store() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42).unwrap();
        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let result = index.query_reranking(&query, 10, Some(5));
        assert!(matches!(
            result,
            Err(AnnSearchErrors::VectorStoreNotAvailable)
        ));
    }
}
