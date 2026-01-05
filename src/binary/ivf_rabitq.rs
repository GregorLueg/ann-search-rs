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
use crate::utils::dist::*;
use crate::utils::heap_structs::*;
use crate::utils::ivf_utils::*;
use crate::utils::*;

/// IVF index with RaBitQ quantisation
///
/// Two-stage search: IVF routing using float centroids, then RaBitQ
/// distance estimation within probed clusters.
///
/// ### Fields
///
/// * `encoder` - The RabitQ encoder
/// * `storage` - The RabitQ storage
/// * `n` - Number of vectors in the structure
/// * `vector_store` - Optional on-disk vector storage
pub struct IvfIndexRaBitQ<T> {
    encoder: RaBitQEncoder<T>,
    storage: RaBitQStorage<T>,
    n: usize,
    vector_store: Option<MmapVectorStore<T>>,
}

//////////////////////////
// VectorDistanceRaBitQ //
//////////////////////////

/// Trait implementation for the distance calculations
impl<T> VectorDistanceRaBitQ<T> for IvfIndexRaBitQ<T>
where
    T: Float + FromPrimitive,
{
    fn storage(&self) -> &RaBitQStorage<T> {
        &self.storage
    }

    fn encoder(&self) -> &RaBitQEncoder<T> {
        &self.encoder
    }
}

//////////////////////////
// VectorDistanceRaBitQ //
//////////////////////////

/// Trait implementation for the CentroidDistances
impl<T> CentroidDistance<T> for IvfIndexRaBitQ<T>
where
    T: Float + FromPrimitive + Sum,
{
    fn centroids(&self) -> &[T] {
        &self.storage.centroids
    }

    fn dim(&self) -> usize {
        self.storage.dim
    }

    fn nlist(&self) -> usize {
        self.storage.nlist
    }

    fn metric(&self) -> Dist {
        self.encoder.metric
    }

    fn centroids_norm(&self) -> &[T] {
        &self.storage.centroids_norm
    }
}

impl<T> IvfIndexRaBitQ<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField,
{
    /// Build IVF-RaBitQ index
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (n × dim)
    /// * `metric` - Distance metric
    /// * `nlist` - Number of IVF cells (defaults to sqrt(n))
    /// * `max_iters` - K-means iterations (defaults to 30)
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Initialised self
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: MatRef<T>,
        metric: Dist,
        nlist: Option<usize>,
        max_iters: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let (vectors_flat, n, dim) = matrix_to_flat(data);

        // compute norms for Cosine distance
        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    vectors_flat[start..end]
                        .iter()
                        .map(|x| *x * *x)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt()
                })
                .collect()
        } else {
            Vec::new()
        };

        let vectors_for_storage = if metric == Dist::Cosine {
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
        let max_iters = max_iters.unwrap_or(30);

        if verbose {
            println!("  Building IVF-RaBitQ index with {} cells.", nlist);
        }

        let cluster_norms = if matches!(metric, Dist::Cosine) {
            vec![T::one(); n]
        } else {
            norms
        };

        // subsample for training if large
        let (training_data, n_train) = if n > 500_000 {
            if verbose {
                println!("  Sampling 250k vectors for centroid training.");
            }
            let (sampled, _) = sample_vectors(&vectors_flat, dim, n, 250_000, seed);
            (sampled, 250_000)
        } else {
            (vectors_flat.clone(), n)
        };

        // train centroids
        let mut centroids_flat = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            max_iters,
            seed,
            verbose,
        );

        // normalise centroids for Cosine
        if metric == Dist::Cosine {
            for c in 0..nlist {
                let start = c * dim;
                let end = start + dim;
                let norm = centroids_flat[start..end]
                    .iter()
                    .map(|x| *x * *x)
                    .fold(T::zero(), |a, b| a + b)
                    .sqrt();
                if norm > T::epsilon() {
                    for i in start..end {
                        centroids_flat[i] = centroids_flat[i] / norm;
                    }
                }
            }
        }

        let centroids_norm: Vec<T> = (0..nlist)
            .map(|i| compute_norm(&centroids_flat[i * dim..(i + 1) * dim]))
            .collect();

        // Assign all vectors
        let assignments = assign_all_parallel(
            &vectors_for_storage,
            &cluster_norms,
            dim,
            n,
            &centroids_flat,
            &centroids_norm,
            nlist,
            &metric,
        );

        // create encoder with shared rotation
        let encoder = RaBitQEncoder::new(dim, metric, seed as u64);

        // build CSR storage
        let storage = build_rabitq_storage(
            &vectors_for_storage,
            dim,
            n,
            &centroids_flat,
            nlist,
            &assignments,
            &encoder,
        );

        Self {
            encoder,
            storage,
            n,
            vector_store: None,
        }
    }

    /// Build IVF-RaBitQ index with vector store
    ///
    /// This version also generates a vector store for re-ranking
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (n × dim)
    /// * `metric` - Distance metric
    /// * `nlist` - Number of IVF cells (defaults to sqrt(n))
    /// * `max_iters` - K-means iterations (defaults to 30)
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    /// * `save_path` - Path to save vector store
    ///
    /// ### Returns
    ///
    /// Initialised self
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_vector_store(
        data: MatRef<T>,
        metric: Dist,
        nlist: Option<usize>,
        max_iters: Option<usize>,
        seed: usize,
        verbose: bool,
        save_path: impl AsRef<Path>,
    ) -> std::io::Result<Self> {
        let (vectors_flat, n, dim) = matrix_to_flat(data);

        // compute norms for Cosine distance
        let norms: Vec<T> = (0..n)
            .map(|i| {
                let start = i * dim;
                let end = start + dim;
                vectors_flat[start..end]
                    .iter()
                    .map(|x| *x * *x)
                    .fold(T::zero(), |a, b| a + b)
                    .sqrt()
            })
            .collect();

        let vectors_for_storage = if metric == Dist::Cosine {
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
        let max_iters = max_iters.unwrap_or(30);

        if verbose {
            println!("  Building IVF-RaBitQ index with {} cells.", nlist);
        }

        let cluster_norms = if matches!(metric, Dist::Cosine) {
            vec![T::one(); n]
        } else {
            norms.clone()
        };

        let (training_data, n_train) = if n > 500_000 {
            if verbose {
                println!("  Sampling 250k vectors for centroid training.");
            }
            let (sampled, _) = sample_vectors(&vectors_flat, dim, n, 250_000, seed);
            (sampled, 250_000)
        } else {
            (vectors_flat.clone(), n)
        };

        let mut centroids_flat = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            max_iters,
            seed,
            verbose,
        );

        // Normalise centroids for Cosine
        if metric == Dist::Cosine {
            for c in 0..nlist {
                let start = c * dim;
                let end = start + dim;
                let norm = centroids_flat[start..end]
                    .iter()
                    .map(|x| *x * *x)
                    .fold(T::zero(), |a, b| a + b)
                    .sqrt();
                if norm > T::epsilon() {
                    for i in start..end {
                        centroids_flat[i] = centroids_flat[i] / norm;
                    }
                }
            }
        }

        let centroids_norm: Vec<T> = (0..nlist)
            .map(|i| compute_norm(&centroids_flat[i * dim..(i + 1) * dim]))
            .collect();

        let assignments = assign_all_parallel(
            &vectors_for_storage,
            &cluster_norms,
            dim,
            n,
            &centroids_flat,
            &centroids_norm,
            nlist,
            &metric,
        );

        let encoder = RaBitQEncoder::new(dim, metric, seed as u64);

        let storage = build_rabitq_storage(
            &vectors_for_storage,
            dim,
            n,
            &centroids_flat,
            nlist,
            &assignments,
            &encoder,
        );

        // Save vector store
        std::fs::create_dir_all(&save_path)?;

        let vectors_path = save_path.as_ref().join("vectors_flat.bin");
        let norms_path = save_path.as_ref().join("norms.bin");

        MmapVectorStore::save(&vectors_flat, &norms, dim, n, &vectors_path, &norms_path)?;
        let vector_store = MmapVectorStore::new(vectors_path, norms_path, dim, n)?;

        Ok(Self {
            encoder,
            storage,
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
    /// * `nprobe` - Number of IVF cells to probe. If None, uses sqrt(nlist)
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize, nprobe: Option<usize>) -> (Vec<usize>, Vec<T>) {
        let nprobe = nprobe
            .unwrap_or_else(|| ((self.storage.nlist as f64).sqrt() as usize).max(1))
            .min(self.storage.nlist);
        let k = k.min(self.n);

        // Normalise query for cosine
        let (query_normalised, _): (Vec<T>, T) = match self.encoder.metric {
            Dist::Cosine => {
                let norm = compute_norm(query_vec);
                if norm > T::epsilon() {
                    (query_vec.iter().map(|&x| x / norm).collect(), norm)
                } else {
                    (query_vec.to_vec(), T::one())
                }
            }
            Dist::Euclidean => (query_vec.to_vec(), T::one()),
        };

        // Use trait method - prenorm version since we've already normalised
        let cluster_dists = self.get_centroids_prenorm(&query_normalised, nprobe);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for &(_, c_idx) in cluster_dists.iter().take(nprobe) {
            let centroid = self.storage.centroid(c_idx);
            let query_encoded = self.encoder.encode_query(&query_normalised, centroid);
            let cluster_size = self.storage.cluster_size(c_idx);

            for local_idx in 0..cluster_size {
                let dist = self.rabitq_dist(&query_encoded, c_idx, local_idx);
                let global_idx = self.storage.cluster_vector_indices(c_idx)[local_idx];

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

        results.into_iter().map(|(d, i)| (i, d.0)).unzip()
    }

    /// Query using a row reference
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of IVF cells to probe. If None, uses sqrt(nlist)
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        nprobe: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, nprobe);
        }
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, nprobe)
    }

    /// Query with exact distance reranking
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of IVF cells to probe. If None, uses sqrt(nlist)
    /// * `rerank_factor` - How many more neighbours to rank exactly. Defaults
    ///   to `20`, i.e., `20 * k` neighbours get re-ranked.
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
    #[inline]
    pub fn query_reranking(
        &self,
        query_vec: &[T],
        k: usize,
        nprobe: Option<usize>,
        rerank_factor: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        let rerank_factor = rerank_factor.unwrap_or(20);
        let vector_store = self
            .vector_store
            .as_ref()
            .expect("Vector store required for reranking");

        let (candidates, _) = self.query(query_vec, k * rerank_factor, nprobe);

        let query_norm = match self.encoder.metric {
            Dist::Cosine => compute_norm(query_vec),
            Dist::Euclidean => T::one(),
        };

        let mut scored: Vec<_> = candidates
            .iter()
            .map(|&idx| {
                let dist = match self.encoder.metric {
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
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of IVF cells to probe. If None, uses sqrt(nlist)
    /// * `rerank_factor` - How many more neighbours to rank exactly. Defaults
    ///   to `20`, i.e., `20 * k` neighbours get re-ranked.
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
    #[inline]
    pub fn query_row_reranking(
        &self,
        query_row: RowRef<T>,
        k: usize,
        nprobe: Option<usize>,
        rerank_factor: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query_reranking(slice, k, nprobe, rerank_factor);
        }
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query_reranking(&query_vec, k, nprobe, rerank_factor)
    }

    /// Generate kNN graph
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `nprobe` - Number of IVF cells to probe per query
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
        nprobe: Option<usize>,
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
                    if count % 100_000 == 0 {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            self.n.separate_with_underscores()
                        );
                    }
                }

                self.query_reranking(vec, k, nprobe, rerank_factor)
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

    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.encoder.memory_usage_bytes()
            + self.storage.memory_usage_bytes()
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
    fn test_ivf_rabitq_construction() {
        let data = create_test_data::<f32>(100, 32);
        let index = IvfIndexRaBitQ::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(10),
            Some(10),
            42,
            false,
        );

        assert_eq!(index.n, 100);
        assert_eq!(index.storage.nlist, 10);
        assert_eq!(index.storage.n_vectors(), 100);
    }

    #[test]
    fn test_ivf_rabitq_query_returns_k_results() {
        let data = create_test_data::<f32>(100, 32);
        let index = IvfIndexRaBitQ::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(10),
            Some(10),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10, Some(10));

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_ivf_rabitq_query_sorted() {
        let data = create_test_data::<f32>(100, 32);
        let index = IvfIndexRaBitQ::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(10),
            Some(10),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (_, distances) = index.query(&query, 10, Some(10));

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_ivf_rabitq_query_k_exceeds_n() {
        let data = create_test_data::<f32>(50, 32);
        let index =
            IvfIndexRaBitQ::build(data.as_ref(), Dist::Euclidean, Some(5), Some(10), 42, false);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, _) = index.query(&query, 100, Some(5));

        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_ivf_rabitq_query_row() {
        let data = create_test_data::<f32>(100, 32);
        let index = IvfIndexRaBitQ::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(10),
            Some(10),
            42,
            false,
        );

        let (indices, distances) = index.query_row(data.as_ref().row(0), 10, Some(10));

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_ivf_rabitq_cosine() {
        let data = create_test_data::<f32>(100, 32);
        let index =
            IvfIndexRaBitQ::build(data.as_ref(), Dist::Cosine, Some(10), Some(10), 42, false);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10, Some(10));

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_ivf_rabitq_default_nprobe() {
        let data = create_test_data::<f32>(100, 32);
        let index = IvfIndexRaBitQ::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(10),
            Some(10),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, _) = index.query(&query, 5, None);

        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_build_with_vector_store() {
        let data = create_test_data::<f32>(50, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = IvfIndexRaBitQ::build_with_vector_store(
            data.as_ref(),
            Dist::Euclidean,
            Some(5),
            Some(10),
            42,
            false,
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

        let index = IvfIndexRaBitQ::build_with_vector_store(
            data.as_ref(),
            Dist::Cosine,
            Some(10),
            Some(10),
            42,
            false,
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

        let index = IvfIndexRaBitQ::build_with_vector_store(
            data.as_ref(),
            Dist::Euclidean,
            Some(10),
            Some(10),
            42,
            false,
            temp_dir.path(),
        )
        .unwrap();

        let (indices, distances) =
            index.query_row_reranking(data.as_ref().row(0), 10, Some(10), Some(5));

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_knn_graph_with_vector_store() {
        let data = create_test_data::<f32>(50, 32);
        let temp_dir = TempDir::new().unwrap();

        let index = IvfIndexRaBitQ::build_with_vector_store(
            data.as_ref(),
            Dist::Cosine,
            Some(5),
            Some(10),
            42,
            false,
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
        let index =
            IvfIndexRaBitQ::build(data.as_ref(), Dist::Euclidean, Some(5), Some(10), 42, false);

        let _ = index.generate_knn(5, Some(5), Some(10), false, false);
    }

    #[test]
    #[should_panic]
    fn test_query_reranking_without_vector_store_panics() {
        let data = create_test_data::<f32>(50, 32);
        let index =
            IvfIndexRaBitQ::build(data.as_ref(), Dist::Euclidean, Some(5), Some(10), 42, false);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let _ = index.query_reranking(&query, 10, Some(5), Some(5));
    }

    #[test]
    fn test_default_nlist() {
        let data = create_test_data::<f32>(100, 32);
        let index =
            IvfIndexRaBitQ::build(data.as_ref(), Dist::Euclidean, None, Some(10), 42, false);

        assert_eq!(index.storage.nlist, 10);
    }
}
