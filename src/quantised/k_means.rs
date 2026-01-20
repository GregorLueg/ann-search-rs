use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;

use crate::utils::dist::SimdDistance;
use crate::utils::ivf_utils::*;

/// Parallel Lloyds for PQ
///
/// Fast version with in-lining and only Euclidean distance. Modifies the
/// centroids.
///
/// ### Params
///
/// * `data` - Slice of the original data
/// * `dim` - Dimensionsality of the original data
/// * `n` - Number of samples
/// * `centroids` - Initial centroids to update
/// * `n_centroids` - Number of centroids to identify
/// * `max_iters` - Maximum iterations for the Lloyd algorithm
fn parallel_lloyd_pq<T>(
    data: &[T],
    dim: usize,
    n: usize,
    centroids: &mut [T],
    k: usize,
    max_iters: usize,
) where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + SimdDistance,
{
    for _ in 0..max_iters {
        // Parallel assignment with inline distance
        let assignments: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| {
                let vec = &data[i * dim..(i + 1) * dim];
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
            })
            .collect();

        // Update centroids
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
    }
}

/// Special implementation for PQ of k-means clustering
///
/// ### Params
///
/// * `data` - Slice of the original data
/// * `dim` - Dimensionsality of the original data
/// * `n` - Number of samples
/// * `n_centroids` - Number of centroids to identify
/// * `max_iters` - Maximum iterations for the Lloyd algorithm
/// * `seed` - For reproducibility
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
    // Fast init is fine for 256 centroids
    let mut centroids = fast_random_init(data, dim, n, n_centroids, seed);

    parallel_lloyd_pq(data, dim, n, &mut centroids, n_centroids, max_iters);

    centroids
}
