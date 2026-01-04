use faer::{MatRef, RowRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use thousands::*;

use crate::binary::binariser::*;
use crate::binary::dist_binary::*;
use crate::utils::dist::*;
use crate::utils::heap_structs::*;

/// Exhaustive RaBitQ index with multi-centroid support
///
/// Uses IVF-style partitioning with RaBitQ encoding per cluster.
/// At query time, probes the nearest clusters and searches exhaustively
/// within each.
pub struct ExhaustiveIndexRaBitQ<T> {
    quantiser: RaBitQQuantiser<T>,
    n: usize,
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

        Self { quantiser, n }
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
        data: MatRef<T>,
        k: usize,
        n_probe: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        assert_eq!(data.nrows(), self.n, "Data row count mismatch");

        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let vec: Vec<T> = data.row(i).iter().cloned().collect();

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

                self.query(&vec, k, n_probe)
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

    /// Number of vectors in the index
    pub fn len(&self) -> usize {
        self.n
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Number of clusters
    pub fn n_clusters(&self) -> usize {
        self.quantiser.n_clusters()
    }

    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.quantiser.memory_usage_bytes()
    }
}
