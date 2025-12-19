use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::collections::BinaryHeap;

use crate::dist::*;

///////////////////
// Float on heap //
///////////////////

/// Wrapper for f32 that implements Ord for use in BinaryHeap
///
/// Faster than the sorts on full vectors and allows to keep data on heap
#[derive(Clone, Copy, Debug)]
pub struct OrderedFloat<T>(pub T);

/// Partial equality trait
impl<T: Float> PartialEq for OrderedFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

/// Equality trait
impl<T: Float> Eq for OrderedFloat<T> {}

/// Partial ordering trait
impl<T: Float> PartialOrd for OrderedFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Comparing one to the other
impl<T: Float> Ord for OrderedFloat<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

////////////////
// Validation //
////////////////

pub trait KnnValidation<T>: VectorDistance<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    /// Query for validation purposes
    ///
    /// * `query_vec` - The query Vec for which to do the exhaustive search
    ///   for.
    /// * `k` - Number of neighbours to return
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>);

    /// Returns number of samples
    ///
    /// ### Returns
    ///
    /// The number of samples stored in the index
    fn n(&self) -> usize;

    /// Returns the Distance metric
    ///
    /// ### Returns
    ///
    /// The Dist metric.
    fn metric(&self) -> Dist;

    /// Exhaustive search for ground truth
    ///
    /// ### Params
    ///
    /// * `query_vec` - The query Vec for which to do the exhaustive search
    ///   for.
    /// * `k` - Number of neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, dist)`
    fn exhaustive_query(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        let n_vectors = self.n();
        let k = k.min(n_vectors);
        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        match self.metric() {
            Dist::Euclidean => {
                for idx in 0..n_vectors {
                    let dist = self.euclidean_distance_to_query(idx, query_vec);

                    if heap.len() < k {
                        heap.push((OrderedFloat(dist), idx));
                    } else if dist < heap.peek().unwrap().0 .0 {
                        heap.pop();
                        heap.push((OrderedFloat(dist), idx));
                    }
                }
            }
            Dist::Cosine => {
                let query_norm = query_vec
                    .iter()
                    .map(|v| *v * *v)
                    .fold(T::zero(), |a, b| a + b)
                    .sqrt();

                for idx in 0..n_vectors {
                    let dist = self.cosine_distance_to_query(idx, query_vec, query_norm);

                    if heap.len() < k {
                        heap.push((OrderedFloat(dist), idx));
                    } else if dist < heap.peek().unwrap().0 .0 {
                        heap.pop();
                        heap.push((OrderedFloat(dist), idx));
                    }
                }
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);

        let (distances, indices): (Vec<_>, Vec<_>) = results
            .into_iter()
            .map(|(OrderedFloat(dist), idx)| (dist, idx))
            .unzip();

        (indices, distances)
    }

    /// Validation function for the index
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours to return
    /// * `seed` - Seed for reproducibility
    /// * `no_samples` - Optional number of samples to. Otherwise defaults to
    ///   `1000` or n, whichever is smaller.
    ///
    /// ### Returns
    ///
    /// Recall@k for a subset of queried samples.
    fn validate_index(&self, k: usize, seed: usize, no_samples: Option<usize>) -> f64 {
        let no_samples = no_samples.unwrap_or(1000).min(self.n());
        let mut rng = StdRng::seed_from_u64(seed as u64);

        let query_indices: Vec<usize> = (0..no_samples)
            .map(|_| rng.random_range(0..self.n()))
            .collect();

        let mut total_recall = 0.0;

        for &query_idx in &query_indices {
            let start = query_idx * self.dim();
            let query_vec = &self.vectors_flat()[start..start + self.dim()];

            let (approx_indices, _) = self.query_for_validation(query_vec, k);
            let (true_indices, _) = self.exhaustive_query(query_vec, k);

            let approx_set: FxHashSet<_> = approx_indices.into_iter().collect();
            let matches = true_indices
                .iter()
                .filter(|idx| approx_set.contains(idx))
                .count();

            total_recall += matches as f64 / k as f64;
        }

        total_recall / no_samples as f64
    }
}

////////////////////////
// k-means clustering //
////////////////////////

/////////////
// Helpers //
/////////////

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
fn min_distance_to_centroids<T>(
    vec: &[T],
    centroids: &[T],
    dim: usize,
    n_centroids: usize,
    metric: &Dist,
) -> T
where
    T: Float,
{
    let mut min_dist = T::infinity();

    for c in 0..n_centroids {
        let cent = &centroids[c * dim..(c + 1) * dim];
        let dist = match metric {
            Dist::Euclidean => euclidean_distance_static(vec, cent),
            Dist::Cosine => cosine_distance_static(vec, cent),
        };
        if dist < min_dist {
            min_dist = dist;
        }
    }

    min_dist
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
fn weighted_kmeans_plus_plus<T>(
    data: &[T],
    dim: usize,
    k: usize,
    metric: &Dist,
    seed: usize,
) -> Vec<T>
where
    T: Float,
{
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
                min_distance_to_centroids(vec, &centroids, dim, n_cents, metric)
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
fn kmeans_parallel_init<T>(
    data: &[T],
    dim: usize,
    n: usize,
    k: usize,
    metric: &Dist,
    seed: usize,
) -> Vec<T>
where
    T: Float + Send + Sync,
{
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
                min_distance_to_centroids(vec, &candidates, dim, n_candidates, metric)
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

    weighted_kmeans_plus_plus(&candidates, dim, k, metric, seed + 1)
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
fn fast_random_init<T>(data: &[T], dim: usize, n: usize, k: usize, seed: usize) -> Vec<T>
where
    T: Float,
{
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
fn parallel_lloyd<T>(
    data: &[T],
    dim: usize,
    n: usize,
    centroids: &mut [T],
    k: usize,
    metric: &Dist,
    max_iters: usize,
    verbose: bool,
) where
    T: Float + Send + Sync,
{
    for iter in 0..max_iters {
        let assignments = assign_all_parallel(data, dim, n, centroids, k, metric);

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

//////////
// Main //
//////////

/// Train k-means centroids
#[allow(clippy::too_many_arguments)]
pub fn train_centroids<T>(
    data: &[T],
    dim: usize,
    n: usize,
    nlist: usize,
    metric: &Dist,
    max_iters: usize,
    seed: usize,
    verbose: bool,
) -> Vec<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    let mut centroids = if nlist > 200 {
        if verbose {
            println!("  Initialising centroids via fast random selection");
        }
        fast_random_init(data, dim, n, nlist, seed)
    } else {
        if verbose {
            println!("  Initialising centroids via k-means||");
        }
        kmeans_parallel_init(data, dim, n, nlist, metric, seed)
    };

    if verbose {
        println!("  Running parallel Lloyd's iterations");
    }
    parallel_lloyd(
        data,
        dim,
        n,
        &mut centroids,
        nlist,
        metric,
        max_iters,
        verbose,
    );

    centroids
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
pub fn build_csr_layout(
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
pub fn assign_all_parallel<T>(
    data: &[T],
    dim: usize,
    n: usize,
    centroids: &[T],
    k: usize,
    metric: &Dist,
) -> Vec<usize>
where
    T: Float + Send + Sync,
{
    (0..n)
        .into_par_iter()
        .map(|i| {
            let vec = &data[i * dim..(i + 1) * dim];
            let mut best_cluster = 0;
            let mut best_dist = T::infinity();
            for c in 0..k {
                let cent = &centroids[c * dim..(c + 1) * dim];
                let dist = match metric {
                    Dist::Euclidean => euclidean_distance_static(vec, cent),
                    Dist::Cosine => cosine_distance_static(vec, cent),
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
pub fn sample_vectors<T>(
    vectors_flat: &[T],
    dim: usize,
    n: usize,
    n_sample: usize,
    seed: usize,
) -> (Vec<T>, Vec<usize>)
where
    T: Float,
{
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::{Ordering, Reverse};
    use std::collections::BinaryHeap;

    #[test]
    fn test_ordered_float_f32_equality() {
        let a = OrderedFloat(1.0_f32);
        let b = OrderedFloat(1.0_f32);
        let c = OrderedFloat(2.0_f32);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_ordered_float_f64_equality() {
        let a = OrderedFloat(1.0_f64);
        let b = OrderedFloat(1.0_f64);
        let c = OrderedFloat(2.0_f64);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_ordered_float_ordering() {
        let a = OrderedFloat(1.0_f32);
        let b = OrderedFloat(2.0_f32);
        let c = OrderedFloat(1.0_f32);

        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&a), Ordering::Greater);
        assert_eq!(a.cmp(&c), Ordering::Equal);
    }

    #[test]
    fn test_ordered_float_partial_ord() {
        let a = OrderedFloat(1.0_f32);
        let b = OrderedFloat(2.0_f32);

        assert!(a < b);
        assert!(b > a);
        assert!(a <= b);
        assert!(b >= a);
    }

    #[test]
    fn test_ordered_float_in_binary_heap() {
        let mut heap = BinaryHeap::new();
        heap.push(OrderedFloat(3.0_f32));
        heap.push(OrderedFloat(1.0_f32));
        heap.push(OrderedFloat(2.0_f32));

        // binaryHeap is a max-heap, so should pop in descending order
        assert_eq!(heap.pop(), Some(OrderedFloat(3.0)));
        assert_eq!(heap.pop(), Some(OrderedFloat(2.0)));
        assert_eq!(heap.pop(), Some(OrderedFloat(1.0)));
    }

    #[test]
    fn test_ordered_float_in_reverse_binary_heap() {
        let mut heap = BinaryHeap::new();
        heap.push(Reverse(OrderedFloat(3.0_f32)));
        heap.push(Reverse(OrderedFloat(1.0_f32)));
        heap.push(Reverse(OrderedFloat(2.0_f32)));

        // reverse makes it a min-heap, should pop in ascending order
        assert_eq!(heap.pop(), Some(Reverse(OrderedFloat(1.0))));
        assert_eq!(heap.pop(), Some(Reverse(OrderedFloat(2.0))));
        assert_eq!(heap.pop(), Some(Reverse(OrderedFloat(3.0))));
    }

    #[test]
    fn test_ordered_float_nan_handling() {
        let a = OrderedFloat(1.0_f32);
        let nan = OrderedFloat(f32::NAN);

        // NaN should be treated as equal to itself (via unwrap_or)
        assert_eq!(nan.cmp(&nan), Ordering::Equal);

        // Ordering with NaN should default to Equal
        assert_eq!(a.cmp(&nan), Ordering::Equal);
        assert_eq!(nan.cmp(&a), Ordering::Equal);
    }

    #[test]
    fn test_ordered_float_negative_values() {
        let a = OrderedFloat(-1.0_f32);
        let b = OrderedFloat(-2.0_f32);
        let c = OrderedFloat(0.0_f32);

        assert!(b < a); // -2 < -1
        assert!(a < c); // -1 < 0
    }

    #[test]
    fn test_ordered_float_zero_comparison() {
        let pos_zero = OrderedFloat(0.0_f32);
        let neg_zero = OrderedFloat(-0.0_f32);

        // IEEE 754: +0 == -0
        assert_eq!(pos_zero, neg_zero);
    }

    #[test]
    fn test_ordered_float_infinity() {
        let inf = OrderedFloat(f32::INFINITY);
        let neg_inf = OrderedFloat(f32::NEG_INFINITY);
        let finite = OrderedFloat(1.0_f32);

        assert!(neg_inf < finite);
        assert!(finite < inf);
        assert!(neg_inf < inf);
    }

    #[test]
    fn test_ordered_float_clone() {
        let a = OrderedFloat(3.15);
        let b = a;

        assert_eq!(a, b);
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn test_dist_clone() {
        let d1 = Dist::Euclidean;
        let d2 = d1;

        assert_eq!(d1, d2);
    }
}
