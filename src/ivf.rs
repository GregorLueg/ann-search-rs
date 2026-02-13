use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::iter::Sum;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::Separable;

use crate::prelude::*;
use crate::utils::ivf_utils::*;
use crate::utils::*;

/// IVF (Inverted File) index for similarity search
///
/// Uses k-means clustering to partition vectors into nlist clusters. Each
/// cluster maintains an inverted list of vector indices assigned to it.
/// Queries search only the nprobe nearest clusters, trading perfect recall
/// for speed.
///
/// ### Fields
///
/// * `vectors_flat` - Original vector data, flattened for cache locality
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `norms` - Pre-computed norms for Cosine distance (empty for Euclidean)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `centroids` - Cluster centres (nlist * dim elements)
/// * `all_indices` - Vector indices for each cluster (in a flat structure)
/// * `offsets` - Offsets of the elements of each inverted list.
/// * `nlist` - Number of clusters in the index
pub struct IvfIndex<T> {
    /// shared ones
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    pub norms: Vec<T>,
    metric: Dist,
    // index specific ones
    centroids: Vec<T>,
    centroids_norm: Vec<T>,
    all_indices: Vec<usize>,
    offsets: Vec<usize>,
    nlist: usize,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn norms(&self) -> &[T] {
        &self.norms
    }
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    fn centroids(&self) -> &[T] {
        &self.centroids
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

////////////////
// Main index //
////////////////

impl<T> IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Build an IVF index with optimised memory layout and parallel training.
    ///
    /// Constructs an inverted file index by clustering vectors with k-means,
    /// then assigns each vector to its nearest centroid. Uses a CSR (Compressed
    /// Sparse Row) layout for cache-efficient cluster traversal during search.
    ///
    /// ### Workflow
    ///
    /// 1. Subsamples 250k vectors for training if dataset exceeds 500k
    /// 2. Runs k-means clustering to find nlist centroids
    /// 3. Assigns all vectors to their nearest centroid in parallel
    /// 4. Builds CSR layout grouping vectors by cluster for locality
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `nlist` - Optional number of clusters. Defaults to `sqrt(n)`.
    /// * `max_iters` - Optional maximum k-means iterations (defaults to `30`).
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print training progress
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    pub fn build(
        data: MatRef<T>,
        metric: Dist,
        nlist: Option<usize>,
        max_iters: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let (vectors_flat, n, dim) = matrix_to_flat(data);

        // Compute norms for Cosine distance
        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    T::calculate_l2_norm(&vectors_flat[start..end])
                })
                .collect()
        } else {
            Vec::new()
        };

        let max_iters = max_iters.unwrap_or(30);
        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        // 1. subsample training data
        let (training_data, n_train) = if n > 500_000 {
            if verbose {
                println!("  Sampling 250k vectors for training");
            }
            let (data, _) = sample_vectors(&vectors_flat, dim, n, 250_000, seed);
            (data, 250_000)
        } else {
            (vectors_flat.clone(), n)
        };

        if verbose {
            println!("  Generating IVF index with {} Voronoi cells.", nlist);
        }

        // 2. train the centroids
        let centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            max_iters,
            seed,
            verbose,
        );

        let centroids_norm = if metric == Dist::Cosine {
            (0..nlist)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    centroids[start..end]
                        .iter()
                        .map(|x| *x * *x)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt()
                })
                .collect()
        } else {
            Vec::new()
        };

        // 3. assign the Rest
        let data_norms_for_assignment = if metric == Dist::Cosine {
            norms.clone()
        } else {
            vec![T::one(); n]
        };

        let assignments = assign_all_parallel(
            &vectors_flat,
            &data_norms_for_assignment,
            dim,
            n,
            &centroids,
            &centroids_norm,
            nlist,
            &metric,
        );

        if verbose {
            print_cluster_summary(&assignments, nlist);
        }

        // 4. generate a flat version for better cache locality
        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        Self {
            vectors_flat,
            dim,
            n,
            norms,
            metric,
            centroids,
            centroids_norm,
            all_indices,
            offsets,
            nlist,
        }
    }

    ///////////
    // Query //
    ///////////

    /// Query the index for approximate nearest neighbours
    ///
    /// Performs two-stage search: first finds nprobe nearest centroids to the
    /// query, then exhaustively searches all vectors in those clusters. Uses
    /// a max-heap to track top-k candidates.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search. A good default here is
    ///   `sqrt(nlist)`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize, nprobe: Option<usize>) -> (Vec<usize>, Vec<T>) {
        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k: usize = k.min(self.n);
        let query_norm = if matches!(self.metric, Dist::Cosine) {
            query_vec
                .iter()
                .map(|&v| v * v)
                .fold(T::zero(), |a, b| a + b)
                .sqrt()
        } else {
            T::one()
        };

        // 1. find the top `nprobe` centroids
        let cluster_dists: Vec<(T, usize)> = self.get_centroids_dist(query_vec, query_norm, nprobe);

        // 2. search only those clusters in the CSR layout
        let mut buffer = SortedBuffer::with_capacity(k);

        for &(_, cluster_idx) in cluster_dists.iter().take(nprobe) {
            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];

            for &vec_idx in &self.all_indices[start..end] {
                let dist = match self.metric {
                    Dist::Euclidean => self.euclidean_distance_to_query(vec_idx, query_vec),
                    Dist::Cosine => self.cosine_distance_to_query(vec_idx, query_vec, query_norm),
                };

                buffer.insert((OrderedFloat(dist), vec_idx), k);
            }
        }

        let (distances, indices) = buffer.data().iter().map(|(d, i)| (d.0, *i)).unzip();
        (indices, distances)
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies
    /// to a temporary vector. Uses `self.query()` under the hood.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search. A good default here is
    ///   `sqrt(nlist)`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
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

        // fallback
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, nprobe)
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete
    /// kNN graph.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `nprobe` - Number of clusters to search (defaults to sqrt(nlist) if None)
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)` where each row corresponds
    /// to a vector in the index
    pub fn generate_knn(
        &self,
        k: usize,
        nprobe: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dim;
                let end = start + self.dim;
                let vec = &self.vectors_flat[start..end];

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

                self.query(vec, k, nprobe)
            })
            .collect();

        if return_dist {
            let (indices, distances) = results.into_iter().unzip();
            (indices, Some(distances))
        } else {
            let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _)| idx).collect();
            (indices, None)
        }
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
            + self.centroids.capacity() * std::mem::size_of::<T>()
            + self.centroids_norm.capacity() * std::mem::size_of::<T>()
            + self.all_indices.capacity() * std::mem::size_of::<usize>()
            + self.offsets.capacity() * std::mem::size_of::<usize>()
    }
}

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        self.query(query_vec, k, None)
    }

    fn n(&self) -> usize {
        self.n
    }

    fn metric(&self) -> Dist {
        self.metric
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::Mat;

    fn create_simple_matrix() -> Mat<f32> {
        // 5 points in 3D space
        let data = [
            1.0, 0.0, 0.0, // Point 0
            0.0, 1.0, 0.0, // Point 1
            0.0, 0.0, 1.0, // Point 2
            1.0, 1.0, 0.0, // Point 3
            1.0, 0.0, 1.0, // Point 4
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
    }

    #[test]
    fn test_ivf_index_creation() {
        let data = create_simple_matrix();
        let _ = IvfIndex::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(2), // nlist
            None,
            42,
            false,
        );
    }

    #[test]
    fn test_ivf_query_finds_self() {
        let data = create_simple_matrix();
        let index = IvfIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1, None);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ivf_query_euclidean() {
        let data = create_simple_matrix();
        let index = IvfIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_ivf_query_cosine() {
        let data = create_simple_matrix();

        let index = IvfIndex::build(data.as_ref(), Dist::Cosine, Some(2), None, 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ivf_query_k_larger_than_dataset() {
        let data = create_simple_matrix();
        let index = IvfIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 10, None);

        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_ivf_query_nprobe() {
        let data = create_simple_matrix();
        let index = IvfIndex::build(data.as_ref(), Dist::Euclidean, Some(3), None, 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices1, _) = index.query(&query, 3, Some(1));
        let (indices2, _) = index.query(&query, 3, Some(2));

        assert!(!indices1.is_empty());
        assert!(!indices2.is_empty());
    }

    #[test]
    fn test_ivf_reproducibility() {
        let data = create_simple_matrix();

        let index1 = IvfIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);
        let index2 = IvfIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);

        let query = vec![0.5, 0.5, 0.0];
        let (indices1, _) = index1.query(&query, 3, None);
        let (indices2, _) = index2.query(&query, 3, None);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_ivf_different_seeds() {
        let data = create_simple_matrix();

        let index1 = IvfIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);
        let index2 = IvfIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 123, false);

        let query = vec![0.5, 0.5, 0.0];
        let (indices1, _) = index1.query(&query, 3, Some(2));
        let (indices2, _) = index2.query(&query, 3, Some(2));

        assert!(!indices1.is_empty());
        assert!(!indices2.is_empty());
    }

    #[test]
    fn test_ivf_larger_dataset() {
        let n = 100;
        let dim = 10;
        let data = Mat::from_fn(n, dim, |i, j| (i * j) as f32 / 10.0);

        let index = IvfIndex::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(10), // sqrt(100)
            None,
            42,
            false,
        );

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5, None);

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_ivf_orthogonal_vectors() {
        let data = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });

        let index = IvfIndex::build(data.as_ref(), Dist::Cosine, Some(3), None, 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        if indices.len() >= 2 {
            assert_relative_eq!(distances[1], 1.0, epsilon = 1e-5);
        }
        if indices.len() >= 3 {
            assert_relative_eq!(distances[2], 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_ivf_more_clusters() {
        let data = create_simple_matrix();

        let index_few = IvfIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);
        let index_many = IvfIndex::build(data.as_ref(), Dist::Euclidean, Some(4), None, 42, false);

        let query = vec![0.9, 0.1, 0.0];
        let (indices1, _) = index_few.query(&query, 3, Some(2));
        let (indices2, _) = index_many.query(&query, 3, Some(4));

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }
}
