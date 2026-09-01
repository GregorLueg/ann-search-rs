//! Exhaustive RaBitQ index. Compresses the vectors via RaBitQ and does
//! exhaustive searches against query vectors.

use bytemuck::Pod;
use faer::RowRef;
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use thousands::*;

use crate::binary::dist_binary::*;
use crate::binary::rabitq::*;
use crate::binary::vec_store::*;
use crate::prelude::*;
use crate::utils::k_means_utils::CentroidDistance;
use crate::utils::pack_knn_results;

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
    vector_store: Option<MmapVectorStore<T>>,
    /// Shape of `vector_store`, so it can be re-opened after a load. Only
    /// read by the `serialise` feature; the field is kept unconditionally so
    /// the constructors stay free of `cfg`.
    #[cfg_attr(not(feature = "serialise"), allow(dead_code))]
    store_meta: Option<StoreMeta>,
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
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Initialised index
    pub fn new(
        data: impl AnnMatrix<T>,
        metric: &Dist,
        n_clusters: Option<usize>,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        if *metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(*metric));
        }

        let (vectors_flat, n, dim) = data.into_row_major();
        let quantiser =
            RaBitQQuantiser::new((&vectors_flat[..], n, dim), metric, n_clusters, seed)?;
        Ok(Self {
            quantiser,
            n,
            vector_store: None,
            store_meta: None,
        })
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
        data: impl AnnMatrix<T>,
        metric: &Dist,
        n_clusters: Option<usize>,
        seed: usize,
        save_path: impl AsRef<Path>,
    ) -> Result<Self, AnnSearchErrors> {
        if *metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(*metric));
        }

        // One walk of the caller's matrix. The quantiser normalises its own
        // copy for cosine; the store needs the raw vectors for exact
        // re-ranking, so `vectors_flat` must stay untouched here.
        let (vectors_flat, n, dim) = data.into_row_major();
        let quantiser =
            RaBitQQuantiser::new((&vectors_flat[..], n, dim), metric, n_clusters, seed)?;

        std::fs::create_dir_all(&save_path)?;

        let norms: Vec<T> = vectors_flat
            .chunks_exact(dim)
            .map(compute_l2_norm)
            .collect();

        let (vectors_path, norms_path) = MmapVectorStore::<T>::paths_in(&save_path);

        MmapVectorStore::save(&vectors_flat, &norms, dim, n, &vectors_path, &norms_path)?;
        let vector_store = MmapVectorStore::new(vectors_path, norms_path, dim, n)?;

        Ok(Self {
            quantiser,
            n,
            vector_store: Some(vector_store),
            store_meta: Some(StoreMeta { dim, n }),
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

        let cluster_dists = self
            .quantiser
            .get_centroids_prenorm(&query_normalised, n_probe);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        // The heap ranks on *squared* distance: the square root is monotone, so
        // it only needs applying to the k survivors.
        let mut block = [T::zero(); RABITQ_BLOCK];

        // One rotation per query, not one per probed cluster
        let q_rot = self.quantiser.encoder.apply_rotation(&query_normalised);

        for &(_, c_idx) in cluster_dists.iter().take(n_probe) {
            let query_encoded = self.quantiser.encode_query_prerotated(&q_rot, c_idx);
            let cluster_size = self.storage().cluster_size(c_idx);
            let indices = self.storage().cluster_vector_indices(c_idx);

            let mut local_idx = 0;
            while local_idx < cluster_size {
                let take = RABITQ_BLOCK.min(cluster_size - local_idx);
                let block_min =
                    self.rabitq_block_sq(&query_encoded, c_idx, local_idx, &mut block[..take]);

                if heap.len() < k || block_min < heap.peek().unwrap().0 .0 {
                    for (j, &dist) in block[..take].iter().enumerate() {
                        let global_idx = indices[local_idx + j];

                        if heap.len() < k {
                            heap.push((OrderedFloat(dist), global_idx));
                        } else if dist < heap.peek().unwrap().0 .0 {
                            heap.pop();
                            heap.push((OrderedFloat(dist), global_idx));
                        }
                    }
                }

                local_idx += take;
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable();

        let (distances, indices): (Vec<T>, Vec<usize>) =
            results.into_iter().map(|(d, i)| (d.0.sqrt(), i)).unzip();

        Ok((indices, distances))
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
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let rerank_factor = rerank_factor.unwrap_or(20);
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

                self.query_reranking(vec, k, n_probe, rerank_factor)
            })
            .collect::<Result<Vec<_>, _>>()?;

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
    T: AnnSearchFloat,
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
        if let Some(meta) = self.store_meta {
            meta.check(self.n, self.quantiser.encoder.dim)?;
        }
        self.vector_store = MmapVectorStore::open_in_dir(dir, self.store_meta)?;

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

    #[test]
    fn test_exhaustive_rabitq_construction() {
        let data = create_test_data::<f32>(100, 32);
        let index =
            ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::SquaredEuclidean, Some(10), 42)
                .unwrap();

        assert_eq!(index.n, 100);
        assert_eq!(index.quantiser.n_clusters(), 10);
        assert_eq!(index.quantiser.n_vectors(), 100);
    }

    #[test]
    fn test_exhaustive_rabitq_query_returns_k_results() {
        let data = create_test_data::<f32>(100, 32);
        let index =
            ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::SquaredEuclidean, Some(10), 42)
                .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10, Some(5)).unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_rabitq_query_sorted() {
        let data = create_test_data::<f32>(100, 32);
        let index =
            ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::SquaredEuclidean, Some(10), 42)
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
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::SquaredEuclidean, Some(5), 42)
            .unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, _) = index.query(&query, 100, Some(5)).unwrap();

        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_exhaustive_rabitq_query_row() {
        let data = create_test_data::<f32>(100, 32);
        let index =
            ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::SquaredEuclidean, Some(10), 42)
                .unwrap();

        let (indices, distances) = index.query_row(data.as_ref().row(0), 10, Some(5)).unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_rabitq_cosine() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::Cosine, Some(10), 42).unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10, Some(10)).unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_rabitq_default_nprobe() {
        let data = create_test_data::<f32>(100, 32);
        let index =
            ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::SquaredEuclidean, Some(10), 42)
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
            42,
            temp_dir.path(),
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
            42,
            temp_dir.path(),
        )
        .unwrap();

        let (knn_indices, knn_distances) = index
            .generate_knn(5, Some(5), Some(10), true, false)
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
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::SquaredEuclidean, Some(5), 42)
            .unwrap();
        let result = index.generate_knn(5, Some(5), Some(10), false, false);

        println!("What is this {:?}", result);

        assert!(matches!(
            result,
            Err(AnnSearchErrors::VectorStoreNotAvailable)
        ));
    }

    #[test]
    fn test_query_reranking_without_vector_store_panics() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexRaBitQ::new(data.as_ref(), &Dist::SquaredEuclidean, Some(5), 42)
            .unwrap();
        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let result = index.query_reranking(&query, 10, Some(5), Some(5));
        assert!(matches!(
            result,
            Err(AnnSearchErrors::VectorStoreNotAvailable)
        ));
    }
}
