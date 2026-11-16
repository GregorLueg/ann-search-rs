//! K-means clustering specifically designed for the product quantisers in
//! this crate. Optimised specifically for the use case of product quantisation.

use num_traits::Float;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::k_means_utils::*;

/////////////
// Helpers //
/////////////

/// Batch size for mini-batch k-means
const MINI_BATCH_SIZE: usize = 10_000;

/// Independent running minima in [`argmin_lanes`]. Eight covers two NEON or
/// one AVX2 register of `f32`; not tuned.
const ARGMIN_LANES: usize = 8;

/// Transpose a row-major `k * dim` centroid buffer to dimension-major.
///
/// ### Params
///
/// * `centroids` - Row-major centroids
/// * `dim` - Dimensionality
/// * `k` - Number of centroids
/// * `out` - `dim * k` output; `out[d * k + c]` is dimension `d` of centroid `c`
pub(crate) fn transpose_centroids<T: Float>(centroids: &[T], dim: usize, k: usize, out: &mut [T]) {
    for c in 0..k {
        for d in 0..dim {
            out[d * k + c] = centroids[c * dim + d];
        }
    }
}

/// Squared Euclidean distance from one vector to every dimension-major centroid
///
/// The inner loop runs over centroids with independent accumulators, so it
/// vectorises across centroids at any `dim`. Each centroid still sums its
/// squared differences in dimension order, so the result is bit-identical to
/// a plain row-by-row scalar scan.
///
/// ### Params
///
/// * `vec` - The vector, length `dim`
/// * `centroids_t` - Dimension-major centroids from [`transpose_centroids`]
/// * `k` - Number of centroids
/// * `dist` - Output of length `k`
#[inline(always)]
pub(crate) fn sq_dists_transposed<T: Float>(
    vec: &[T],
    centroids_t: &[T],
    k: usize,
    dist: &mut [T],
) {
    // seed from dimension 0 rather than zero-filling: `0 + x == x` exactly
    let q0 = vec[0];
    for (acc, &c) in dist.iter_mut().zip(&centroids_t[..k]) {
        let diff = q0 - c;
        *acc = diff * diff;
    }
    for (d, &q) in vec.iter().enumerate().skip(1) {
        let row = &centroids_t[d * k..(d + 1) * k];
        for (acc, &c) in dist.iter_mut().zip(row) {
            let diff = q - c;
            *acc = *acc + diff * diff;
        }
    }
}

/// Scaled dot product of one vector with every dimension-major centroid
///
/// Same loop shape as [`sq_dists_transposed`], so it vectorises across
/// centroids. `out[c] = factor * <vec, centroid_c>`; the factor is folded into
/// the vector entries so it costs `dim` multiplies, not `k`.
///
/// ### Params
///
/// * `vec` - The vector, length `dim`
/// * `centroids_t` - Dimension-major centroids from [`transpose_centroids`]
/// * `k` - Number of centroids
/// * `factor` - Scale applied to every dot product
/// * `out` - Output of length `k`
#[inline(always)]
pub(crate) fn dots_transposed<T: Float>(
    vec: &[T],
    centroids_t: &[T],
    k: usize,
    factor: T,
    out: &mut [T],
) {
    let q0 = factor * vec[0];
    for (acc, &c) in out.iter_mut().zip(&centroids_t[..k]) {
        *acc = q0 * c;
    }
    for (d, &q) in vec.iter().enumerate().skip(1) {
        let q = factor * q;
        let row = &centroids_t[d * k..(d + 1) * k];
        for (acc, &c) in out.iter_mut().zip(row) {
            *acc = *acc + q * c;
        }
    }
}

/// Find nearest centroid against dimension-major centroids.
///
/// [`sq_dists_transposed`] followed by a lane-wise argmin.
///
/// ### Params
///
/// * `vec` - The vector to find the nearest centroid for
/// * `centroids_t` - Dimension-major centroids from [`transpose_centroids`]
/// * `k` - Number of centroids
/// * `dist` - Scratch of length `k`
///
/// ### Returns
///
/// The index of the nearest centroid, the lowest index on ties.
#[inline(always)]
pub(crate) fn find_nearest_transposed<T: Float>(
    vec: &[T],
    centroids_t: &[T],
    k: usize,
    dist: &mut [T],
) -> usize {
    sq_dists_transposed(vec, centroids_t, k, dist);
    argmin_lanes(dist)
}

/// Index of the smallest value, lowest index on ties.
///
/// Keeps [`ARGMIN_LANES`] independent running minima so the comparisons
/// vectorise into compare-and-blend, then merges the lanes. A strict `<` per
/// lane keeps each lane's first minimum and the merge prefers the lower index
/// on equal values, so the result matches a plain sequential scan.
///
/// ### Params
///
/// * `dist` - Values to scan, non-empty
///
/// ### Returns
///
/// The index of the first minimum
#[inline(always)]
fn argmin_lanes<T: Float>(dist: &[T]) -> usize {
    let mut best_v = [T::infinity(); ARGMIN_LANES];
    // u32 keeps the index blend at the width of the f32 compare
    let mut best_i = [0u32; ARGMIN_LANES];
    let (chunks, rem) = dist.as_chunks::<ARGMIN_LANES>();
    let rem_start = dist.len() - rem.len();
    for (ci, chunk) in chunks.iter().enumerate() {
        for l in 0..ARGMIN_LANES {
            let v = chunk[l];
            let take = v < best_v[l];
            best_v[l] = if take { v } else { best_v[l] };
            best_i[l] = if take {
                (ci * ARGMIN_LANES + l) as u32
            } else {
                best_i[l]
            };
        }
    }
    let mut best = 0;
    let mut best_dist = T::infinity();
    for l in 0..ARGMIN_LANES {
        let i = best_i[l] as usize;
        if best_v[l] < best_dist || (best_v[l] == best_dist && i < best) {
            best_dist = best_v[l];
            best = i;
        }
    }
    for (c, &v) in dist.iter().enumerate().skip(rem_start) {
        if v < best_dist {
            best_dist = v;
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
    T: AnnSearchFloat,
{
    let batch_size = MINI_BATCH_SIZE.min(n);

    let mut centroids_t = vec![T::zero(); k * dim];

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

        transpose_centroids(centroids, dim, k, &mut centroids_t);

        // parallel assignment of batch
        let assignments: Vec<usize> = batch_indices
            .par_iter()
            .map_init(
                || vec![T::zero(); k],
                |dist, &i| {
                    let vec = &data[i * dim..(i + 1) * dim];
                    find_nearest_transposed(vec, &centroids_t, k, dist)
                },
            )
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
    T: AnnSearchFloat,
{
    let mut prev_assignments = vec![usize::MAX; n];
    let mut centroids_t = vec![T::zero(); k * dim];

    for _ in 0..max_iters {
        transpose_centroids(centroids, dim, k, &mut centroids_t);

        // parallel assignment
        let assignments: Vec<usize> = (0..n)
            .into_par_iter()
            .map_init(
                || vec![T::zero(); k],
                |dist, i| {
                    let vec = &data[i * dim..(i + 1) * dim];
                    find_nearest_transposed(vec, &centroids_t, k, dist)
                },
            )
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
                        sums[offset + d] += vec[d];
                    }
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![T::zero(); k * dim], vec![0usize; k]),
                |(mut sums1, mut counts1), (sums2, counts2)| {
                    for i in 0..sums1.len() {
                        sums1[i] += sums2[i];
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
    T: AnnSearchFloat,
{
    let mut centroids = fast_random_init(data, dim, n, n_centroids, seed);

    if n <= MINI_BATCH_SIZE {
        full_lloyd_pq(data, dim, n, &mut centroids, n_centroids, max_iters);
    } else {
        mini_batch_lloyd_pq(data, dim, n, &mut centroids, n_centroids, max_iters, seed);
    }

    centroids
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_nearest_transposed_matches_row_scan() {
        let dim = 4;
        let k = 37;
        let centroids: Vec<f32> = (0..k * dim)
            .map(|x| ((x * 7919) % 101) as f32 * 0.1)
            .collect();
        let mut centroids_t = vec![0.0f32; k * dim];
        transpose_centroids(&centroids, dim, k, &mut centroids_t);
        let mut dist = vec![0.0f32; k];
        for p in 0..50 {
            let vec: Vec<f32> = (0..dim)
                .map(|d| ((p * 31 + d * 17) % 53) as f32 * 0.2)
                .collect();
            let mut best = 0;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let mut s = 0.0f32;
                for d in 0..dim {
                    let diff = vec[d] - centroids[c * dim + d];
                    s += diff * diff;
                }
                if s < best_d {
                    best_d = s;
                    best = c;
                }
            }
            assert_eq!(
                find_nearest_transposed(&vec, &centroids_t, k, &mut dist),
                best
            );
        }
    }

    #[test]
    fn test_argmin_lanes_ties_pick_lowest_index() {
        let mut v = vec![5.0f32; 21];
        v[3] = 1.0;
        v[11] = 1.0;
        v[19] = 1.0;
        assert_eq!(argmin_lanes(&v), 3);
        v[3] = 2.0;
        assert_eq!(argmin_lanes(&v), 11);
        let mut w = vec![5.0f32; 21];
        w[18] = 0.5;
        w[20] = 0.5;
        assert_eq!(argmin_lanes(&w), 18);
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
