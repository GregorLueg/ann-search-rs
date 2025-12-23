use faer::RowRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::{collections::BinaryHeap, iter::Sum};

use crate::utils::dist::*;
use crate::utils::heap_structs::*;
use crate::utils::k_means::*;
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
    all_indices: Vec<usize>,
    offsets: Vec<usize>,
    nlist: usize,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    /// Return the flat vectors
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    /// Return the original dimensions
    fn dim(&self) -> usize {
        self.dim
    }

    /// Return the normalised values for the Cosine calculation
    fn norms(&self) -> &[T] {
        &self.norms
    }
}

impl<T> IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
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
    /// * `vectors_flat` - Flattened vector data (length = n * dim)
    /// * `dim` - Embedding dimensions
    /// * `n` - Number of vectors
    /// * `norms` - Pre-computed norms for Cosine distance (empty for Euclidean)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `nlist` - Number of clusters (more = faster search, lower recall)
    /// * `max_iters` - Maximum k-means iterations (defaults to 30)
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print training progress
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        vectors_flat: Vec<T>,
        dim: usize,
        n: usize,
        norms: Vec<T>,
        metric: Dist,
        nlist: usize,
        max_iters: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let max_iters = max_iters.unwrap_or(30);

        // 1. Subsample training data
        let (training_data, n_train) = if n > 500_000 {
            if verbose {
                println!("  Sampling 250k vectors for training");
            }
            let (data, _) = sample_vectors(&vectors_flat, dim, n, 250_000, seed);
            (data, 250_000)
        } else {
            (vectors_flat.clone(), n)
        };

        // 2. Train the centroids
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

        // 3. Assign the Rest
        let assignments = assign_all_parallel(&vectors_flat, dim, n, &centroids, nlist, &metric);

        // 4. Generate a flat version for better cache locality
        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        Self {
            vectors_flat,
            dim,
            n,
            norms,
            metric,
            centroids,
            all_indices,
            offsets,
            nlist,
        }
    }

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
            .unwrap_or_else(|| (((self.nlist as f64) * 0.15) as usize).max(1))
            .min(self.nlist);
        let k = k.min(self.n);

        // 1. Find the top `nprobe` centroids
        let mut cluster_dists: Vec<(T, usize)> = (0..self.nlist)
            .map(|c| {
                let cent = &self.centroids[c * self.dim..(c + 1) * self.dim];
                let dist = match self.metric {
                    Dist::Euclidean => euclidean_distance_static(query_vec, cent),
                    Dist::Cosine => cosine_distance_static(query_vec, cent),
                };
                (dist, c)
            })
            .collect();

        if nprobe < self.nlist {
            cluster_dists.select_nth_unstable_by(nprobe, |a, b| a.0.partial_cmp(&b.0).unwrap());
        }

        // 2. Search only those clusters in the CSR layout
        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);
        let query_norm = if matches!(self.metric, Dist::Cosine) {
            query_vec
                .iter()
                .map(|&v| v * v)
                .fold(T::zero(), |a, b| a + b)
                .sqrt()
        } else {
            T::one()
        };

        for &(_, cluster_idx) in cluster_dists.iter().take(nprobe) {
            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];

            for &vec_idx in &self.all_indices[start..end] {
                let dist = match self.metric {
                    Dist::Euclidean => self.euclidean_distance_to_query(vec_idx, query_vec),
                    Dist::Cosine => self.cosine_distance_to_query(vec_idx, query_vec, query_norm),
                };

                if heap.len() < k {
                    heap.push((OrderedFloat(dist), vec_idx));
                } else if dist < heap.peek().unwrap().0 .0 {
                    heap.pop();
                    heap.push((OrderedFloat(dist), vec_idx));
                }
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);
        let (distances, indices) = results.into_iter().map(|(d, i)| (d.0, i)).unzip();
        (indices, distances)
    }

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
}

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    /// Internal querying function
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        self.query(query_vec, k, None)
    }

    /// Returns n
    fn n(&self) -> usize {
        self.n
    }

    /// Returns the distance metric
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

    fn create_simple_vectors() -> (Vec<f32>, usize, usize, Vec<f32>) {
        // 5 points in 3D space
        let vectors_flat = vec![
            1.0, 0.0, 0.0, // Point 0
            0.0, 1.0, 0.0, // Point 1
            0.0, 0.0, 1.0, // Point 2
            1.0, 1.0, 0.0, // Point 3
            1.0, 0.0, 1.0, // Point 4
        ];
        let dim = 3;
        let n = 5;
        let norms = vec![]; // Empty for Euclidean
        (vectors_flat, dim, n, norms)
    }

    #[test]
    fn test_ivf_index_creation() {
        let (vectors_flat, dim, n, norms) = create_simple_vectors();
        let _ = IvfIndex::build(
            vectors_flat,
            dim,
            n,
            norms,
            Dist::Euclidean,
            2, // nlist
            None,
            42,
            false,
        );
    }

    #[test]
    fn test_ivf_query_finds_self() {
        let (vectors_flat, dim, n, norms) = create_simple_vectors();
        let index = IvfIndex::build(
            vectors_flat,
            dim,
            n,
            norms,
            Dist::Euclidean,
            2,
            None,
            42,
            false,
        );

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1, None);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ivf_query_euclidean() {
        let (vectors_flat, dim, n, norms) = create_simple_vectors();
        let index = IvfIndex::build(
            vectors_flat,
            dim,
            n,
            norms,
            Dist::Euclidean,
            2,
            None,
            42,
            false,
        );

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
        let (vectors_flat, dim, n, _) = create_simple_vectors();

        // Compute norms for cosine
        let norms: Vec<f32> = (0..n)
            .map(|i| {
                let start = i * dim;
                vectors_flat[start..start + dim]
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();

        let index = IvfIndex::build(
            vectors_flat,
            dim,
            n,
            norms,
            Dist::Cosine,
            2,
            None,
            42,
            false,
        );

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ivf_query_k_larger_than_dataset() {
        let (vectors_flat, dim, n, norms) = create_simple_vectors();
        let index = IvfIndex::build(
            vectors_flat,
            dim,
            n,
            norms,
            Dist::Euclidean,
            2,
            None,
            42,
            false,
        );

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 10, None);

        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_ivf_query_nprobe() {
        let (vectors_flat, dim, n, norms) = create_simple_vectors();
        let index = IvfIndex::build(
            vectors_flat,
            dim,
            n,
            norms,
            Dist::Euclidean,
            3,
            None,
            42,
            false,
        );

        let query = vec![1.0, 0.0, 0.0];
        let (indices1, _) = index.query(&query, 3, Some(1));
        let (indices2, _) = index.query(&query, 3, Some(2));

        assert!(!indices1.is_empty());
        assert!(!indices2.is_empty());
    }

    #[test]
    fn test_ivf_reproducibility() {
        let (vectors_flat, dim, n, norms) = create_simple_vectors();

        let index1 = IvfIndex::build(
            vectors_flat.clone(),
            dim,
            n,
            norms.clone(),
            Dist::Euclidean,
            2,
            None,
            42,
            false,
        );
        let index2 = IvfIndex::build(
            vectors_flat,
            dim,
            n,
            norms,
            Dist::Euclidean,
            2,
            None,
            42,
            false,
        );

        let query = vec![0.5, 0.5, 0.0];
        let (indices1, _) = index1.query(&query, 3, None);
        let (indices2, _) = index2.query(&query, 3, None);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_ivf_different_seeds() {
        let (vectors_flat, dim, n, norms) = create_simple_vectors();

        let index1 = IvfIndex::build(
            vectors_flat.clone(),
            dim,
            n,
            norms.clone(),
            Dist::Euclidean,
            2,
            None,
            42,
            false,
        );
        let index2 = IvfIndex::build(
            vectors_flat,
            dim,
            n,
            norms,
            Dist::Euclidean,
            2,
            None,
            123,
            false,
        );

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
        let mut vectors_flat = Vec::with_capacity(n * dim);

        for i in 0..n {
            for j in 0..dim {
                vectors_flat.push((i * j) as f32 / 10.0);
            }
        }

        let index = IvfIndex::build(
            vectors_flat,
            dim,
            n,
            vec![],
            Dist::Euclidean,
            10, // sqrt(100)
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
        let vectors_flat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let dim = 3;
        let n = 3;

        let norms: Vec<f32> = (0..n)
            .map(|i| {
                let start = i * dim;
                vectors_flat[start..start + dim]
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();

        let index = IvfIndex::build(
            vectors_flat,
            dim,
            n,
            norms,
            Dist::Cosine,
            3,
            None,
            42,
            false,
        );

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Check remaining results if found
        if indices.len() >= 2 {
            assert_relative_eq!(distances[1], 1.0, epsilon = 1e-5);
        }
        if indices.len() >= 3 {
            assert_relative_eq!(distances[2], 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_ivf_more_clusters() {
        let (vectors_flat, dim, n, norms) = create_simple_vectors();

        let index_few = IvfIndex::build(
            vectors_flat.clone(),
            dim,
            n,
            norms.clone(),
            Dist::Euclidean,
            2,
            None,
            42,
            false,
        );
        let index_many = IvfIndex::build(
            vectors_flat,
            dim,
            n,
            norms,
            Dist::Euclidean,
            4,
            None,
            42,
            false,
        );

        let query = vec![0.9, 0.1, 0.0];
        let (indices1, _) = index_few.query(&query, 3, Some(2)); // Force enough clusters
        let (indices2, _) = index_many.query(&query, 3, Some(4));

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }
}
