use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::utils::dist::*;

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
///
/// ### Params
///
/// * `data` - The original data flattened
/// * `dim` - The dimensions of the data
/// * `n` - Number of samples in the data
/// * `n_centroids` - Number of centroids to identify
/// * `metric` - Distance metric to use
/// * `max_iters` - Maximum iterations for the k-means clustering.
/// * `seed` - Seed for reproducibility
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// Centroid assignment
#[allow(clippy::too_many_arguments)]
pub fn train_centroids<T>(
    data: &[T],
    dim: usize,
    n: usize,
    n_centroids: usize,
    metric: &Dist,
    max_iters: usize,
    seed: usize,
    verbose: bool,
) -> Vec<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    let mut centroids = if n_centroids > 200 {
        if verbose {
            println!("  Initialising centroids via fast random selection");
        }
        fast_random_init(data, dim, n, n_centroids, seed)
    } else {
        if verbose {
            println!("  Initialising centroids via k-means||");
        }
        kmeans_parallel_init(data, dim, n, n_centroids, metric, seed)
    };

    if verbose {
        println!("  Running parallel Lloyd's iterations");
    }
    parallel_lloyd(
        data,
        dim,
        n,
        &mut centroids,
        n_centroids,
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
