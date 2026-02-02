use bytemuck::Pod;
use faer::{MatRef, RowRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use thousands::*;

use crate::binary::dist_binary::*;
use crate::binary::rabitq::*;
use crate::binary::vec_store::*;
use crate::prelude::*;
use crate::utils::ivf_utils::CentroidDistance;
use crate::utils::*;

/// Exhaustive RaBitQ index with multi-centroid support
///
/// Uses IVF-style partitioning with RaBitQ encoding per cluster.
/// At query time, probes the nearest clusters and searches exhaustively
/// within each.
///
/// ### Fields
///
/// * `quantiser` - The RaBitQQuantiser
/// * `n` - Number of vectors
/// * `vector_store` - Optional on-disk vector storage
pub struct ExhaustiveIndexRaBitQ<T> {
    quantiser: RaBitQQuantiser<T>,
    n: usize,
    vector_store: Option<MmapVectorStore<T>>,
}

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

impl<T> ExhaustiveIndexRaBitQ<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    /// Create a new exhaustive RaBitQ index
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (n_samples × dim)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `n_clusters` - Number of clusters. If None, uses 0.5 * sqrt(n)
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Initialised index
    pub fn new(data: MatRef<T>, metric: &Dist, n_clusters: Option<usize>, seed: usize) -> Self {
        let n = data.nrows();
        let quantiser = RaBitQQuantiser::new(data, metric, n_clusters, seed);
        Self {
            quantiser,
            n,
            vector_store: None,
        }
    }

    /// Create a new exhaustive RaBitQ index with vector store for reranking
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (n_samples × dim)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `n_clusters` - Number of clusters. If None, uses 0.5 * sqrt(n)
    /// * `seed` - Random seed
    /// * `save_path` - Where to save the vector storage
    ///
    /// ### Returns
    ///
    /// Initialised index
    pub fn new_with_vector_store(
        data: MatRef<T>,
        metric: &Dist,
        n_clusters: Option<usize>,
        seed: usize,
        save_path: impl AsRef<Path>,
    ) -> std::io::Result<Self> {
        let n = data.nrows();
        let dim = data.ncols();
        let quantiser = RaBitQQuantiser::new(data, metric, n_clusters, seed);

        std::fs::create_dir_all(&save_path)?;

        let (vectors_flat, _, _) = matrix_to_flat(data);
        let norms: Vec<T> = (0..n)
            .map(|i| compute_l2_norm(&vectors_flat[i * dim..(i + 1) * dim]))
            .collect();

        let vectors_path = save_path.as_ref().join("vectors_flat.bin");
        let norms_path = save_path.as_ref().join("norms.bin");

        MmapVectorStore::save(&vectors_flat, &norms, dim, n, &vectors_path, &norms_path)?;
        let vector_store = MmapVectorStore::new(vectors_path, norms_path, dim, n)?;

        Ok(Self {
            quantiser,
            n,
            vector_store: Some(vector_store),
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
    pub fn query(&self, query_vec: &[T], k: usize, n_probe: Option<usize>) -> (Vec<usize>, Vec<T>) {
        let n_probe = n_probe
            .unwrap_or((self.quantiser.n_clusters() as f32 * 0.2) as usize)
            .max(1);
        let k = k.min(self.n);

        // Normalise for cosine
        let query_normalised: Vec<T> = match self.quantiser.encoder.metric {
            Dist::Cosine => {
                let norm = compute_l2_norm(query_vec);
                if norm > T::epsilon() {
                    query_vec.iter().map(|&x| x / norm).collect()
                } else {
                    query_vec.to_vec()
                }
            }
            Dist::Euclidean => query_vec.to_vec(),
        };

        let cluster_dists = self
            .quantiser
            .get_centroids_prenorm(&query_normalised, n_probe);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for &(_, c_idx) in cluster_dists.iter().take(n_probe) {
            let query_encoded = self.quantiser.encode_query(&query_normalised, c_idx);
            let cluster_size = self.storage().cluster_size(c_idx);

            for local_idx in 0..cluster_size {
                let dist = self.rabitq_dist(&query_encoded, c_idx, local_idx);
                let global_idx = self.storage().cluster_vector_indices(c_idx)[local_idx];

                if heap.len() < k {
                    heap.push((OrderedFloat(dist), global_idx));
                } else if dist < heap.peek().unwrap().0 .0 {
                    heap.pop();
                    heap.push((OrderedFloat(dist), global_idx));
                }
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable();

        let (distances, indices): (Vec<T>, Vec<usize>) =
            results.into_iter().map(|(d, i)| (d.0, i)).unzip();

        (indices, distances)
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
    ) -> (Vec<usize>, Vec<T>) {
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
    /// * `rerank_factor` - How many more neighbours to rank exactly. Defaults
    ///   to `20`, i.e., `20 * k` neighbours get re-ranked.
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
    ) -> (Vec<usize>, Vec<T>) {
        let rerank_factor = rerank_factor.unwrap_or(20);
        let vector_store = self
            .vector_store
            .as_ref()
            .expect("Vector store required for reranking");

        let (candidates, _) = self.query(query_vec, k * rerank_factor, n_probe);

        let query_norm = match self.quantiser.encoder.metric {
            Dist::Cosine => compute_l2_norm(query_vec),
            Dist::Euclidean => T::one(),
        };

        let mut scored: Vec<_> = candidates
            .iter()
            .map(|&idx| {
                let dist = match self.quantiser.encoder.metric {
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
    /// * `rerank_factor` - How many more neighbours to rank exactly. Defaults
    ///   to `20`, i.e., `20 * k` neighbours get re-ranked.
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
    ) -> (Vec<usize>, Vec<T>) {
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
    /// * `rerank_factor` - Reranking factor for exact distances
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Print progress
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
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        let vector_store = self
            .vector_store
            .as_ref()
            .expect("generate_knn requires vector_store");

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

                self.query_reranking(vec, k, n_probe, rerank_factor)
            })
            .collect();

        if return_dist {
            let (indices, distances) = results.into_iter().unzip();
            (indices, Some(distances))
        } else {
            let indices = results.into_iter().map(|(idx, _)| idx).collect();
            (indices, None)
        }
    }

    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.quantiser.memory_usage_bytes()
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
    fn test_exhaustive_rabitq_construction() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::Euclidean, Some(10), 42);

        assert_eq!(index.n, 100);
        assert_eq!(index.quantiser.n_clusters(), 10);
        assert_eq!(index.quantiser.n_vectors(), 100);
    }

    #[test]
    fn test_exhaustive_rabitq_query_returns_k_results() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::Euclidean, Some(10), 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10, Some(5));

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_rabitq_query_sorted() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::Euclidean, Some(10), 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (_, distances) = index.query(&query, 10, Some(5));

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_exhaustive_rabitq_query_k_exceeds_n() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::Euclidean, Some(5), 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, _) = index.query(&query, 100, Some(5));

        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_exhaustive_rabitq_query_row() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::Euclidean, Some(10), 42);

        let (indices, distances) = index.query_row(data.as_ref().row(0), 10, Some(5));

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_rabitq_cosine() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::Cosine, Some(10), 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10, Some(10));

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_rabitq_default_nprobe() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::Euclidean, Some(10), 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, _) = index.query(&query, 10, None);

        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_new_with_vector_store() {
        let data = create_test_data::<f32>(50, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = ExhaustiveIndexRaBitQ::new_with_vector_store(
            data.as_ref(),
            &Dist::Euclidean,
            Some(5),
            42,
            temp_dir.path(),
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
            42,
            temp_dir.path(),
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query_reranking(&query, 10, Some(10), Some(5));

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
            &Dist::Euclidean,
            Some(10),
            42,
            temp_dir.path(),
        )
        .unwrap();

        let (indices, distances) =
            index.query_row_reranking(data.as_ref().row(0), 10, Some(5), Some(5));

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
            42,
            temp_dir.path(),
        )
        .unwrap();

        let (knn_indices, knn_distances) = index.generate_knn(5, Some(5), Some(10), true, false);

        assert_eq!(knn_indices.len(), 50);
        assert!(knn_distances.is_some());
        assert_eq!(knn_distances.as_ref().unwrap().len(), 50);

        for neighbours in knn_indices.iter() {
            assert_eq!(neighbours.len(), 5);
        }
    }

    #[test]
    #[should_panic]
    fn test_knn_without_vector_store_panics() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::Euclidean, Some(5), 42);

        let _ = index.generate_knn(5, Some(5), Some(10), false, false);
    }

    #[test]
    #[should_panic]
    fn test_query_reranking_without_vector_store_panics() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::Euclidean, Some(5), 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let _ = index.query_reranking(&query, 10, Some(5), Some(5));
    }
}
