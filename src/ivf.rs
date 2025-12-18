use faer::RowRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::BinaryHeap;

use crate::dist::*;
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
/// * `inverted_lists` - Vector indices for each cluster (nlist lists)
/// * `nlist` - Number of clusters in the index
pub struct IvfIndex<T> {
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    pub norms: Vec<T>,
    pub metric: Dist,
    pub centroids: Vec<T>,
    pub all_indices: Vec<usize>,
    pub offsets: Vec<usize>,
    pub nlist: usize,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    /// Build an IVF index with optimized memory layout and parallel training.
    ///
    /// ### Workflow:
    /// 1. **Subsampling**: Samples training data if the dataset is > 500k points.
    /// 2. **Fast Init**: Uses random selection for large cluster counts (nlist) to avoid
    ///    the $O(N \cdot k)$ bottleneck of k-means||.
    /// 3. **Parallel Lloyd's**: Recomputes centroids using a parallel reduction pattern.
    /// 4. **CSR Finalization**: Flattens assignments into a contiguous memory block.
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
            let (data, _) = Self::sample_vectors(&vectors_flat, dim, n, 250_000, seed);
            (data, 250_000)
        } else {
            (vectors_flat.clone(), n)
        };

        // 2. Fast Initialisation
        // If nlist is high (> 200), k-means|| becomes extremely slow.
        // We use random unique selection as a fast-start heuristic.
        let mut centroids = if nlist > 200 {
            if verbose {
                println!("  Initialising centroids via fast random selection");
            }
            Self::fast_random_init(&training_data, dim, n_train, nlist, seed)
        } else {
            if verbose {
                println!("  Initialising centroids via k-means||");
            }
            Self::kmeans_parallel_init(&training_data, dim, n_train, nlist, &metric, seed)
        };

        // 3. Parallel Lloyd's Iterations
        if verbose {
            println!("  Running parallel Lloyd's iterations");
        }
        Self::parallel_lloyd(
            &training_data,
            dim,
            n_train,
            &mut centroids,
            nlist,
            &metric,
            max_iters,
            verbose,
        );

        // 4. Final Assignment & CSR Conversion
        if verbose {
            println!("  Finalising CSR inverted lists");
        }
        let assignments =
            Self::assign_all_parallel(&vectors_flat, dim, n, &centroids, nlist, &metric);
        let (all_indices, offsets) = Self::build_csr_layout(assignments, n, nlist);

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
        let nprobe = nprobe.unwrap_or_else(|| (((self.nlist as f64) * 0.2) as usize).max(1));
        let k = k.min(self.n);

        // 1. Find the top `nprobe` centroids
        let mut cluster_dists: Vec<(T, usize)> = (0..self.nlist)
            .map(|c| {
                let cent = &self.centroids[c * self.dim..(c + 1) * self.dim];
                let dist = match self.metric {
                    Dist::Euclidean => Self::euclidean_distance_static(query_vec, cent),
                    Dist::Cosine => Self::cosine_distance_static(query_vec, cent),
                };
                (dist, c)
            })
            .collect();

        cluster_dists.select_nth_unstable_by(nprobe, |a, b| a.0.partial_cmp(&b.0).unwrap());

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

    /// Sample random vectors from dataset
    ///
    /// Randomly shuffles indices and selects first n_sample vectors for
    /// k-means training. Used when dataset is large to reduce clustering time.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Flattened vector data
    /// * `dim` - Embedding dimensions
    /// * `n` - Total number of vectors
    /// * `n_sample` - Number of vectors to sample
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Tuple of (sampled vector data, sampled indices)
    fn sample_vectors(
        vectors_flat: &[T],
        dim: usize,
        n: usize,
        n_sample: usize,
        seed: usize,
    ) -> (Vec<T>, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        indices.truncate(n_sample);

        let mut sampled = Vec::with_capacity(n_sample * dim);
        for &idx in &indices {
            let start = idx * dim;
            sampled.extend_from_slice(&vectors_flat[start..start + dim]);
        }

        (sampled, indices)
    }

    /// k-means|| initialisation
    ///
    /// Parallel variant of k-means++ that oversamples centres in multiple
    /// rounds, then clusters them down to k using weighted k-means++. Much
    /// faster than sequential k-means++ with comparable quality.
    ///
    /// ### Algorithm
    ///
    /// 1. Pick first centroid uniformly at random
    /// 2. For log(k) rounds: sample k*2 new centres proportional to D²
    /// 3. Cluster oversampled candidates down to k using weighted k-means++
    ///
    /// ### Params
    ///
    /// * `data` - Training vectors (flattened)
    /// * `dim` - Embedding dimensions
    /// * `n` - Number of training vectors
    /// * `k` - Number of clusters to create
    /// * `metric` - Distance metric
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Initial centroids (k * dim elements)
    fn kmeans_parallel_init(
        data: &[T],
        dim: usize,
        n: usize,
        k: usize,
        metric: &Dist,
        seed: usize,
    ) -> Vec<T> {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let oversampling_factor = 2;
        let n_rounds = ((k as f64).ln() + 1.0) as usize;

        let first_idx = rng.random_range(0..n);
        let mut candidates = Vec::with_capacity(k * oversampling_factor * dim);
        candidates.extend_from_slice(&data[first_idx * dim..(first_idx + 1) * dim]);

        for _ in 0..n_rounds {
            let n_candidates = candidates.len() / dim;

            let distances: Vec<T> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let vec = &data[i * dim..(i + 1) * dim];
                    Self::min_distance_to_centroids(vec, &candidates, dim, n_candidates, metric)
                })
                .collect();

            let total_dist: f64 = distances.iter().map(|&d| d.to_f64().unwrap()).sum();

            for _ in 0..k * oversampling_factor {
                let threshold = rng.random::<f64>() * total_dist;
                let mut cumsum = 0.0;

                for (idx, &dist) in distances.iter().enumerate() {
                    cumsum += dist.to_f64().unwrap();
                    if cumsum >= threshold {
                        candidates.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
                        break;
                    }
                }
            }
        }

        Self::weighted_kmeans_plus_plus(&candidates, dim, k, metric, seed + 1)
    }

    /// Weighted k-means++ on oversampled candidates
    ///
    /// Final stage of k-means|| initialisation. Clusters the oversampled
    /// candidate centres down to exactly k centres using D² weighting.
    ///
    /// ### Params
    ///
    /// * `data` - Candidate centres (flattened)
    /// * `dim` - Embedding dimensions
    /// * `k` - Target number of clusters
    /// * `metric` - Distance metric
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Final k centroids (k * dim elements)
    fn weighted_kmeans_plus_plus(
        data: &[T],
        dim: usize,
        k: usize,
        metric: &Dist,
        seed: usize,
    ) -> Vec<T> {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let n = data.len() / dim;

        if n <= k {
            return data.to_vec();
        }

        let mut centroids = Vec::with_capacity(k * dim);

        let first = rng.random_range(0..n);
        centroids.extend_from_slice(&data[first * dim..(first + 1) * dim]);

        for _ in 1..k {
            let n_cents = centroids.len() / dim;

            let distances: Vec<T> = (0..n)
                .map(|i| {
                    let vec = &data[i * dim..(i + 1) * dim];
                    Self::min_distance_to_centroids(vec, &centroids, dim, n_cents, metric)
                })
                .collect();

            let total: f64 = distances.iter().map(|&d| d.to_f64().unwrap()).sum();
            let threshold = rng.random::<f64>() * total;
            let mut cumsum = 0.0;

            for (idx, &dist) in distances.iter().enumerate() {
                cumsum += dist.to_f64().unwrap();
                if cumsum >= threshold {
                    centroids.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
                    break;
                }
            }
        }

        centroids
    }

    /// Find minimum distance from vector to any centroid
    ///
    /// Computes distance to all centroids and returns the smallest. Used
    /// during k-means initialisation for D² weighting.
    ///
    /// ### Params
    ///
    /// * `vec` - Query vector
    /// * `centroids` - Current centroids (flattened)
    /// * `dim` - Embedding dimensions
    /// * `n_centroids` - Number of centroids
    /// * `metric` - Distance metric
    ///
    /// ### Returns
    ///
    /// Minimum distance to any centroid
    fn min_distance_to_centroids(
        vec: &[T],
        centroids: &[T],
        dim: usize,
        n_centroids: usize,
        metric: &Dist,
    ) -> T {
        let mut min_dist = T::infinity();

        for c in 0..n_centroids {
            let cent = &centroids[c * dim..(c + 1) * dim];
            let dist = match metric {
                Dist::Euclidean => Self::euclidean_distance_static(vec, cent),
                Dist::Cosine => Self::cosine_distance_static(vec, cent),
            };
            if dist < min_dist {
                min_dist = dist;
            }
        }

        min_dist
    }

    /// Fast centroid initialisation via random unique selection
    ///
    /// Randomly selects k unique vectors as initial centroids. Trades
    /// initialisation quality for speed when nlist is large (>200).
    ///
    /// ### Params
    ///
    /// * `data` - Training vectors (flattened)
    /// * `dim` - Embedding dimensions
    /// * `n` - Number of training vectors
    /// * `k` - Number of clusters to create
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Initial centroids (k * dim elements)
    fn fast_random_init(data: &[T], dim: usize, n: usize, k: usize, seed: usize) -> Vec<T> {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let mut centroids = Vec::with_capacity(k * dim);
        for i in 0..k {
            let start = indices[i] * dim;
            centroids.extend_from_slice(&data[start..start + dim]);
        }
        centroids
    }

    /// Parallel Lloyd's k-means iterations
    ///
    /// Iteratively assigns vectors to nearest centroids and recomputes
    /// centroid positions. Uses Rayon for parallel assignment and
    /// fold-reduce for centroid updates.
    ///
    /// ### Params
    ///
    /// * `data` - Training vectors (flattened)
    /// * `dim` - Embedding dimensions
    /// * `n` - Number of training vectors
    /// * `centroids` - Current centroids (modified in-place)
    /// * `k` - Number of clusters
    /// * `metric` - Distance metric
    /// * `max_iters` - Maximum iterations
    /// * `verbose` - Print iteration progress
    #[allow(clippy::too_many_arguments)]
    fn parallel_lloyd(
        data: &[T],
        dim: usize,
        n: usize,
        centroids: &mut [T],
        k: usize,
        metric: &Dist,
        max_iters: usize,
        verbose: bool,
    ) {
        for iter in 0..max_iters {
            let assignments = Self::assign_all_parallel(data, dim, n, centroids, k, metric);

            let (new_sums, counts) = (0..n)
                .into_par_iter()
                .fold(
                    || (vec![T::zero(); k * dim], vec![0usize; k]),
                    |(mut sums, mut counts), i| {
                        let cluster = assignments[i];
                        counts[cluster] += 1;
                        let vec = &data[i * dim..(i + 1) * dim];
                        for d in 0..dim {
                            sums[cluster * dim + d] = sums[cluster * dim + d] + vec[d];
                        }
                        (sums, counts)
                    },
                )
                .reduce(
                    || (vec![T::zero(); k * dim], vec![0usize; k]),
                    |(mut sums1, mut counts1), (sums2, counts2)| {
                        for i in 0..sums1.len() {
                            sums1[i] = sums1[i] + sums2[i];
                        }
                        for i in 0..counts1.len() {
                            counts1[i] += counts2[i];
                        }
                        (sums1, counts1)
                    },
                );

            for c in 0..k {
                if counts[c] > 0 {
                    let count_t = T::from(counts[c]).unwrap();
                    for d in 0..dim {
                        centroids[c * dim + d] = new_sums[c * dim + d] / count_t;
                    }
                }
            }

            if verbose {
                println!("    Iteration {} complete", iter + 1);
            }
        }
    }

    /// Assign all vectors to their nearest centroids in parallel
    ///
    /// ### Params
    ///
    /// * `data` - Vectors to assign (flattened)
    /// * `dim` - Embedding dimensions
    /// * `n` - Number of vectors
    /// * `centroids` - Current centroids
    /// * `k` - Number of clusters
    /// * `metric` - Distance metric
    ///
    /// ### Returns
    ///
    /// Vector of cluster assignments (one per input vector)
    fn assign_all_parallel(
        data: &[T],
        dim: usize,
        n: usize,
        centroids: &[T],
        k: usize,
        metric: &Dist,
    ) -> Vec<usize> {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let vec = &data[i * dim..(i + 1) * dim];
                let mut best_cluster = 0;
                let mut best_dist = T::infinity();
                for c in 0..k {
                    let cent = &centroids[c * dim..(c + 1) * dim];
                    let dist = match metric {
                        Dist::Euclidean => Self::euclidean_distance_static(vec, cent),
                        Dist::Cosine => Self::cosine_distance_static(vec, cent),
                    };
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }
                best_cluster
            })
            .collect()
    }

    /// Convert flat assignments to CSR (Compressed Sparse Row) layout
    ///
    /// Transforms a vector of cluster assignments into an inverted index
    /// structure with contiguous storage. The CSR format uses two arrays:
    /// `all_indices` (vector IDs) and `offsets` (cluster boundaries).
    ///
    /// ### Params
    ///
    /// * `assignments` - Cluster ID for each vector
    /// * `n` - Number of vectors
    /// * `nlist` - Number of clusters
    ///
    /// ### Returns
    ///
    /// Tuple of (all_indices, offsets) for CSR access
    fn build_csr_layout(
        assignments: Vec<usize>,
        n: usize,
        nlist: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut offsets = vec![0usize; nlist + 1];
        for &cluster in &assignments {
            offsets[cluster + 1] += 1;
        }

        // Prefix sum to find starting positions
        for i in 1..=nlist {
            offsets[i] += offsets[i - 1];
        }

        let mut all_indices = vec![0usize; n];
        let mut current_pos = offsets.clone();

        for (vec_idx, &cluster) in assignments.iter().enumerate() {
            let pos = current_pos[cluster];
            all_indices[pos] = vec_idx;
            current_pos[cluster] += 1;
        }

        (all_indices, offsets)
    }
}

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for IvfIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
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
    use crate::utils::Dist;
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
            2,
            None,
            42,
            false,
        );

        let query = vec![1.0, 0.0, 0.0];

        let (indices1, _) = index.query(&query, 3, Some(1));
        let (indices2, _) = index.query(&query, 3, Some(2));

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
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
            2,
            None,
            42,
            false,
        );

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, Some(2)); // Search all clusters

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
        let (indices1, _) = index_few.query(&query, 3, None);
        let (indices2, _) = index_many.query(&query, 3, None);

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }
}
