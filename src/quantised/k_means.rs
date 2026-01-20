use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;

use crate::utils::dist::SimdDistance;
use crate::utils::ivf_utils::*;

/////////////
// Helpers //
/////////////

/// Batch size for mini-batch k-means
const MINI_BATCH_SIZE: usize = 10_000;

/// Threshold below which we use scalar distance
const SCALAR_DIM_THRESHOLD: usize = 8;

/// Scalar Euclidean distance for small dimensions
///
/// When SIMD overhead is likely not worth it.
///
/// ### Params
///
/// * `a`: The first vector.
/// * `b`: The second vector.
///
/// ### Returns
///
/// Euclidean distance between `a` and `b`.
#[inline(always)]
fn euclidean_scalar<T: Float>(a: &[T], b: &[T]) -> T {
    let mut sum = T::zero();
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum = sum + diff * diff;
    }
    sum
}

/// Find nearest centroid - scalar version for small dim
///
/// ### Params
///
/// * `vec`: The vector to find the nearest centroid for.
/// * `centroids`: The centroids to compare against.
/// * `dim`: The dimensionality of the vectors.
/// * `k`: The number of centroids.
///
/// ### Returns
///
/// The index of the nearest centroid.
#[inline(always)]
fn find_nearest_scalar<T: Float>(vec: &[T], centroids: &[T], dim: usize, k: usize) -> usize {
    let mut best = 0;
    let mut best_dist = T::infinity();
    for c in 0..k {
        let cent = &centroids[c * dim..(c + 1) * dim];
        let dist = euclidean_scalar(vec, cent);
        if dist < best_dist {
            best_dist = dist;
            best = c;
        }
    }
    best
}

/// Find nearest centroid - SIMD version
///
/// ### Params
///
/// * `vec`: The vector to find the nearest centroid for.
/// * `centroids`: The centroids to compare against.
/// * `dim`: The dimensionality of the vectors.
/// * `k`: The number of centroids.
///
/// ### Returns
///
/// The index of the nearest centroid.
#[inline(always)]
fn find_nearest_simd<T: Float + SimdDistance>(
    vec: &[T],
    centroids: &[T],
    dim: usize,
    k: usize,
) -> usize {
    let mut best = 0;
    let mut best_dist = T::infinity();
    for c in 0..k {
        let cent = &centroids[c * dim..(c + 1) * dim];
        let dist = T::euclidean_simd(vec, cent);
        if dist < best_dist {
            best_dist = dist;
            best = c;
        }
    }
    best
}

/// Simple LCG for fast, deterministic sampling indices
struct FastRng {
    state: u64,
}

impl FastRng {
    /// Create a new FastRng instance with the given seed.
    ///
    /// ### Params
    ///
    /// * `seed`: The seed for the random number generator.
    fn new(seed: usize) -> Self {
        Self {
            state: seed as u64 ^ 0x5DEECE66D,
        }
    }

    /// Generate a random usize within the given bound.
    ///
    /// ### Params
    ///
    /// * `bound`: The upper bound for the random number.
    ///
    /// ### Returns
    ///
    /// A random usize within the given bound.
    #[inline(always)]
    fn next_usize(&mut self, bound: usize) -> usize {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.state >> 33) as usize) % bound
    }
}

/// Mini-batch k-means for PQ
///
/// Instead of assigning all vectors each iteration, samples a mini-batch.
/// Converges faster in wall-clock time with minimal accuracy loss.
///
/// ### Params
///
/// * `data` - Slice of the original data
/// * `dim` - Dimensionality of the original data
/// * `n` - Number of samples
/// * `centroids` - Initial centroids to update
/// * `k` - Number of centroids
/// * `max_iters` - Maximum iterations
/// * `seed` - Random seed for sampling
fn mini_batch_lloyd_pq<T>(
    data: &[T],
    dim: usize,
    n: usize,
    centroids: &mut [T],
    k: usize,
    max_iters: usize,
    seed: usize,
) where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + SimdDistance,
{
    let batch_size = MINI_BATCH_SIZE.min(n);
    let use_scalar = dim <= SCALAR_DIM_THRESHOLD;

    // Per-centroid counts for learning rate
    let mut centroid_counts = vec![0usize; k];
    let mut rng = FastRng::new(seed);

    // Pre-generate all batch indices for determinism with parallel iteration
    let mut all_batch_indices = Vec::with_capacity(max_iters * batch_size);
    for _ in 0..max_iters {
        for _ in 0..batch_size {
            all_batch_indices.push(rng.next_usize(n));
        }
    }

    for iter in 0..max_iters {
        let batch_start = iter * batch_size;
        let batch_indices = &all_batch_indices[batch_start..batch_start + batch_size];

        // parallel assignment of batch
        let assignments: Vec<usize> = batch_indices
            .par_iter()
            .map(|&i| {
                let vec = &data[i * dim..(i + 1) * dim];
                if use_scalar {
                    find_nearest_scalar(vec, centroids, dim, k)
                } else {
                    find_nearest_simd(vec, centroids, dim, k)
                }
            })
            .collect();

        // sequential centroid update with per-centroid learning rate
        // this is the mini-batch k-means update rule from Sculley (2010)
        for (batch_idx, &data_idx) in batch_indices.iter().enumerate() {
            let cluster = assignments[batch_idx];
            centroid_counts[cluster] += 1;

            // LR: 1 / count gives diminishing updates
            let eta = T::one() / T::from_usize(centroid_counts[cluster]).unwrap();
            let vec = &data[data_idx * dim..(data_idx + 1) * dim];
            let cent_start = cluster * dim;

            for d in 0..dim {
                let c = centroids[cent_start + d];
                centroids[cent_start + d] = c + eta * (vec[d] - c);
            }
        }
    }
}

/// Full Lloyd's algorithm for small datasets
///
/// Used when n <= MINI_BATCH_SIZE, as mini-batch overhead isn't worth it.
///
/// ### Params
///
/// * `data` - Slice of the original data
/// * `dim` - Dimensionality of the original data
/// * `n` - Number of samples
/// * `centroids` - Initial centroids to update
/// * `k` - Number of centroids
/// * `max_iters` - Maximum iterations
fn full_lloyd_pq<T>(
    data: &[T],
    dim: usize,
    n: usize,
    centroids: &mut [T],
    k: usize,
    max_iters: usize,
) where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + SimdDistance,
{
    let use_scalar = dim <= SCALAR_DIM_THRESHOLD;
    let mut prev_assignments = vec![usize::MAX; n];

    for _ in 0..max_iters {
        // parallel assignment
        let assignments: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| {
                let vec = &data[i * dim..(i + 1) * dim];
                if use_scalar {
                    find_nearest_scalar(vec, centroids, dim, k)
                } else {
                    find_nearest_simd(vec, centroids, dim, k)
                }
            })
            .collect();

        // early termination check
        if assignments == prev_assignments {
            break;
        }
        prev_assignments.clone_from(&assignments);

        // parallel accumulation for centroid update
        let (new_sums, counts) = (0..n)
            .into_par_iter()
            .fold(
                || (vec![T::zero(); k * dim], vec![0usize; k]),
                |(mut sums, mut counts), i| {
                    let cluster = assignments[i];
                    counts[cluster] += 1;
                    let vec = &data[i * dim..(i + 1) * dim];
                    let offset = cluster * dim;
                    for d in 0..dim {
                        sums[offset + d] = sums[offset + d] + vec[d];
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

        // update centroids
        for c in 0..k {
            if counts[c] > 0 {
                let count_t = T::from_usize(counts[c]).unwrap();
                let offset = c * dim;
                for d in 0..dim {
                    centroids[offset + d] = new_sums[offset + d] / count_t;
                }
            }
        }
    }
}

/// Train centroids for product quantisation
///
/// Automatically selects mini-batch or full Lloyd's based on dataset size.
///
/// ### Params
///
/// * `data` - Slice of the original data
/// * `dim` - Dimensionality of the original data
/// * `n` - Number of samples
/// * `n_centroids` - Number of centroids to identify
/// * `max_iters` - Maximum iterations
/// * `seed` - Random seed
///
/// ### Returns
///
/// Centroids in a flat structure
pub fn train_centroids_pq<T>(
    data: &[T],
    dim: usize,
    n: usize,
    n_centroids: usize,
    max_iters: usize,
    seed: usize,
) -> Vec<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + SimdDistance,
{
    let mut centroids = fast_random_init(data, dim, n, n_centroids, seed);

    if n <= MINI_BATCH_SIZE {
        full_lloyd_pq(data, dim, n, &mut centroids, n_centroids, max_iters);
    } else {
        mini_batch_lloyd_pq(data, dim, n, &mut centroids, n_centroids, max_iters, seed);
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_scalar() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        let dist = euclidean_scalar(&a, &b);
        // (4^2 + 4^2 + 4^2 + 4^2) = 64
        assert!((dist - 64.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_nearest_scalar() {
        let vec = vec![1.0_f32, 1.0];
        let centroids = vec![
            0.0_f32, 0.0, // centroid 0
            1.0, 1.0, // centroid 1
            5.0, 5.0, // centroid 2
        ];
        let nearest = find_nearest_scalar(&vec, &centroids, 2, 3);
        assert_eq!(nearest, 1);
    }

    #[test]
    fn test_fast_rng_deterministic() {
        let mut rng1 = FastRng::new(42);
        let mut rng2 = FastRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_usize(1000), rng2.next_usize(1000));
        }
    }

    #[test]
    fn test_fast_rng_bounded() {
        let mut rng = FastRng::new(123);
        for _ in 0..1000 {
            let val = rng.next_usize(256);
            assert!(val < 256);
        }
    }

    #[test]
    fn test_train_centroids_pq_small() {
        // Small dataset - should use full Lloyd's
        let data: Vec<f32> = (0..320).map(|x| x as f32).collect();
        let centroids = train_centroids_pq(&data, 32, 10, 4, 10, 42);

        assert_eq!(centroids.len(), 4 * 32);
    }

    #[test]
    fn test_train_centroids_pq_deterministic() {
        let data: Vec<f32> = (0..3200).map(|x| (x % 100) as f32).collect();

        let c1 = train_centroids_pq(&data, 32, 100, 4, 10, 42);
        let c2 = train_centroids_pq(&data, 32, 100, 4, 10, 42);

        assert_eq!(c1, c2);
    }

    #[test]
    fn test_train_centroids_small_dim() {
        // dim=4, should use scalar path
        let data: Vec<f32> = (0..400).map(|x| x as f32).collect();
        let centroids = train_centroids_pq(&data, 4, 100, 8, 10, 42);

        assert_eq!(centroids.len(), 8 * 4);
    }
}
