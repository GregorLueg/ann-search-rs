use faer::{MatRef, RowRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;
use std::path::Path;
use thousands::*;

use crate::binary::binariser::*;
use crate::binary::dist_binary::*;
use crate::binary::vec_store::*;
use crate::utils::dist::*;

///////////////////////////
// ExhaustiveIndexBinary //
///////////////////////////

/// Exhaustive (brute-force) binary nearest neighbour index
///
/// Stores vectors as binary codes and uses Hamming distance for queries.
///
/// ### Fields
///
/// * `vectors_flat_binarised` - Binary codes, flattened (n * n_bytes)
/// * `n_bytes` - Bytes per vector (n_bits / 8)
/// * `n` - Number of samples
/// * `dim` - Vector dimensionality
/// * `metric` - The distance metric
/// * `binariser` - Binariser for encoding query vectors
/// * `vector_store` - Optional on-disk vector storage
pub struct ExhaustiveIndexBinary<T> {
    pub vectors_flat_binarised: Vec<u8>,
    pub n_bytes: usize,
    pub n: usize,
    pub dim: usize,
    metric: Dist,
    binariser: Binariser<T>,
    vector_store: Option<MmapVectorStore<T>>,
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

impl<T> ExhaustiveIndexBinary<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField,
{
    /// Generate a new exhaustive binary index
    ///
    /// Binarises all vectors using the specified hash function and stores them
    /// as compact binary codes. This works solely for Cosine distance!
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features)
    /// * `binarisation_init` - Initialisation method ("itq" or "random")
    /// * `n_bits` - Number of bits per binary code (must be multiple of 8)
    /// * `seed` - Random seed for binariser
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive binary index
    pub fn new(data: MatRef<T>, binarisation_init: &str, n_bits: usize, seed: usize) -> Self {
        assert!(n_bits.is_multiple_of(8), "n_bits must be multiple of 8");

        let init = parse_binarisation_init(binarisation_init).unwrap_or_default();

        let n_bytes = n_bits / 8;
        let n = data.nrows();
        let dim = data.ncols();

        let binariser = match init {
            BinarisationInit::ITQ => Binariser::initialise_with_pca(data, dim, n_bits, seed),
            BinarisationInit::RandomProjections => Binariser::new(dim, n_bits, seed),
        };

        let mut vectors_flat_binarised: Vec<u8> = Vec::with_capacity(n * n_bytes);

        for i in 0..n {
            let original: Vec<T> = data.row(i).iter().cloned().collect();
            vectors_flat_binarised.extend(binariser.encode(&original));
        }

        Self {
            vectors_flat_binarised,
            n_bytes,
            n,
            dim,
            binariser,
            vector_store: None,
            metric: Dist::Cosine,
        }
    }

    /// Generate a new exhaustive binary index with vector store for reranking
    ///
    /// Creates binary index and saves/loads vector store for exact distance reranking.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features)
    /// * `binarisation_init` - Initialisation method ("itq" or "random")
    /// * `n_bits` - Number of bits per binary code (must be multiple of 8)
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
    ) -> std::io::Result<Self> {
        assert!(n_bits.is_multiple_of(8), "n_bits must be multiple of 8");

        let init = parse_binarisation_init(binarisation_init).unwrap_or_default();

        let n_bytes = n_bits / 8;
        let n = data.nrows();
        let dim = data.ncols();

        let binariser = match init {
            BinarisationInit::ITQ => Binariser::initialise_with_pca(data, dim, n_bits, seed),
            BinarisationInit::RandomProjections => Binariser::new(dim, n_bits, seed),
        };

        let mut vectors_flat_binarised: Vec<u8> = Vec::with_capacity(n * n_bytes);

        for i in 0..n {
            let original: Vec<T> = data.row(i).iter().cloned().collect();
            vectors_flat_binarised.extend(binariser.encode(&original));
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

        let vectors_path = save_path.as_ref().join("vectors_flat.bin");
        let norms_path = save_path.as_ref().join("norms.bin");

        MmapVectorStore::save(&vectors_flat, &norms, dim, n, &vectors_path, &norms_path)?;

        let vector_store = MmapVectorStore::new(vectors_path, norms_path, dim, n)?;

        Ok(Self {
            vectors_flat_binarised,
            n_bytes,
            n,
            dim,
            binariser,
            vector_store: Some(vector_store),
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
    pub fn query(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<u32>) {
        let query_binary = self.binariser.encode(query_vec);
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

        (indices, distances)
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
    pub fn query_row(&self, query_row: RowRef<T>, k: usize) -> (Vec<usize>, Vec<u32>) {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k)
    }

    /// Query with reranking using exact distances
    ///
    /// Two-stage search: Hamming distance to find candidates, then exact
    /// distance for final ranking. Requires vector_store to be available.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of nearest neighbours to return
    /// * `rerank_factor` - Multiplier for candidate set size (searches k *
    ///   rerank_factor candidates)
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
    ) -> (Vec<usize>, Vec<T>) {
        let rerank_factor = rerank_factor.unwrap_or(20);
        let vector_store = self
            .vector_store
            .as_ref()
            .expect("Vector store required for reranking - use new_with_vector_store()");

        let (candidates, _) = self.query(query_vec, k * rerank_factor);

        let query_norm = match self.metric {
            Dist::Cosine => query_vec
                .iter()
                .map(|&x| x * x)
                .fold(T::zero(), |a, b| a + b)
                .sqrt(),
            Dist::Euclidean => T::one(),
        };

        let mut scored: Vec<_> = candidates
            .iter()
            .map(|&idx| {
                let dist = match self.metric {
                    Dist::Cosine => {
                        vector_store.cosine_distance_to_query(idx, query_vec, query_norm)
                    }
                    Dist::Euclidean => vector_store.euclidean_distance_to_query(idx, query_vec),
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

        (indices, dists)
    }

    /// Query with reranking for row references
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of nearest neighbours to return
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
        rerank_factor: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
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
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
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
                .collect();

            if return_dist {
                let (indices, distances) = results.into_iter().unzip();
                (indices, Some(distances))
            } else {
                let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _)| idx).collect();
                (indices, None)
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
                (indices, Some(distances_converted))
            } else {
                let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _)| idx).collect();
                (indices, None)
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
}

///////////
// Tests //
///////////

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

    #[test]
    fn test_exhaustive_binary_construction() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        assert_eq!(index.n, 100);
        assert_eq!(index.n_bytes, 8);
        assert_eq!(index.vectors_flat_binarised.len(), 100 * 8);
    }

    #[test]
    fn test_exhaustive_binary_query_returns_k_results() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10);

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_binary_query_sorted() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (_, distances) = index.query(&query, 10);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_exhaustive_binary_query_k_exceeds_n() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, _) = index.query(&query, 100);

        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_exhaustive_binary_query_row() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let (indices1, distances1) = index.query_row(data.as_ref().row(0), 10);

        assert_eq!(indices1.len(), 10);
        assert_eq!(distances1.len(), 10);
        assert_eq!(indices1[0], 0);
    }

    #[test]
    fn test_exhaustive_binary_knn_graph_no_vector_store() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let (knn_indices, knn_distances) = index.generate_knn(5, None, true, false);

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
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (_, distances) = index.query(&query, 20);

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
        let (indices, distances) = index.query_reranking(&query, 10, Some(5));

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
            Dist::Euclidean,
            42,
            temp_dir.path(),
        )
        .unwrap();

        let (indices, distances) = index.query_row_reranking(data.as_ref().row(0), 10, Some(5));

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

        let (knn_indices, knn_distances) = index.generate_knn(5, Some(10), true, false);

        assert_eq!(knn_indices.len(), 50);
        assert!(knn_distances.is_some());
        assert_eq!(knn_distances.as_ref().unwrap().len(), 50);

        for neighbours in knn_indices.iter() {
            assert_eq!(neighbours.len(), 5);
        }
    }

    #[test]
    #[should_panic(expected = "Vector store required for reranking")]
    fn test_query_reranking_without_vector_store_panics() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let _ = index.query_reranking(&query, 10, Some(5));
    }
}
