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

use crate::binary::binariser::*;
use crate::binary::dist_binary::*;
use crate::binary::vec_store::*;
use crate::utils::dist::*;
use crate::utils::heap_structs::*;
use crate::utils::*;

/// Exhaustive RaBitQ index with multi-centroid support
///
/// Uses IVF-style partitioning with RaBitQ encoding per cluster.
/// At query time, probes the nearest clusters and searches exhaustively
/// within each.
pub struct ExhaustiveIndexRaBitQ<T> {
    quantiser: RaBitQQuantiser<T>,
    n: usize,
    vector_store: Option<MmapVectorStore<T>>,
}

impl<T> VectorDistanceRaBitQ<T> for ExhaustiveIndexRaBitQ<T>
where
    T: Float + FromPrimitive,
{
    fn clusters(&self) -> &[RaBitQCluster<T>] {
        &self.quantiser.clusters
    }

    fn n_bytes(&self) -> usize {
        self.quantiser.n_bytes
    }

    fn dim(&self) -> usize {
        self.quantiser.dim
    }
}

impl<T> ExhaustiveIndexRaBitQ<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField,
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
        let (vectors_flat, n, dim) = matrix_to_flat(data);
        let quantiser = RaBitQQuantiser::new(data, metric, n_clusters, seed);

        std::fs::create_dir_all(&save_path)?;

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
        let n_probe = n_probe.unwrap_or((self.quantiser.n_clusters() as f32 * 0.2) as usize);
        let k = k.min(self.n);

        let cluster_indices = self.quantiser.find_nearest_clusters(query_vec, n_probe);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for &c_idx in &cluster_indices {
            let query_encoded = self.quantiser.encode_query(query_vec, c_idx);
            let cluster_size = self.cluster_size(c_idx);

            for local_idx in 0..cluster_size {
                let dist = self.rabitq_dist(&query_encoded, c_idx, local_idx);
                let global_idx = self.cluster_vector_indices(c_idx)[local_idx];

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
            .expect("Vector store required for reranking - use new_with_vector_store()");

        let (candidates, _) = self.query(query_vec, k * rerank_factor, n_probe);

        let query_norm = match self.quantiser.metric {
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
                let dist = match self.quantiser.metric {
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
    /// * `data` - Original data matrix
    /// * `k` - Number of neighbours per vector
    /// * `n_probe` - Number of clusters to search per query
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
        let counter = Arc::new(AtomicUsize::new(0));

        if let Some(vector_store) = &self.vector_store {
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

                    self.query_reranking(vec, k, n_probe, rerank_factor)
                })
                .collect();

            if return_dist {
                let (indices, distances): (Vec<_>, Vec<_>) = results.into_iter().unzip();
                (indices, Some(distances))
            } else {
                let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _)| idx).collect();
                (indices, None)
            }
        } else {
            let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
                .into_par_iter()
                .map(|_i| {
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

                    // Can't query ourselves without original vectors
                    // This is a limitation - needs the vector store
                    panic!("generate_knn requires vector_store - use new_with_vector_store()");
                })
                .collect();

            if return_dist {
                let (indices, distances): (Vec<_>, Vec<_>) = results.into_iter().unzip();
                (indices, Some(distances))
            } else {
                let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _)| idx).collect();
                (indices, None)
            }
        }
    }

    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.quantiser.memory_usage_bytes()
    }
}
