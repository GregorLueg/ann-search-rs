use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::iter::Sum;

use crate::utils::dist::*;
use crate::utils::Dist;

//////////////////////
// CentroidDistance //
//////////////////////

/// Trait for computing distances between Floats
pub trait CentroidDistance<T>
where
    T: Float + Sum + SimdDistance,
{
    /// Get the internal flat centroids representation
    fn centroids(&self) -> &[T];

    /// Get the internal dimensions
    fn dim(&self) -> usize;

    /// Get the number of internal dimensions
    fn nlist(&self) -> usize;

    /// Get the internal distance metric
    fn metric(&self) -> Dist;

    /// Get the centroids normalisation
    fn centroids_norm(&self) -> &[T];

    /// Calculate the distance to the centroids
    ///
    /// ### Params
    ///
    /// * `query_vec` - The slice of the query
    /// * `query_norm` - The norm of the query. Relevant for fast Cosine dist
    ///   calculations.
    /// * `nprobe` - Number of probes
    ///
    /// ### Returns
    ///
    /// The distance to the different clusters
    fn get_centroids_dist(&self, query_vec: &[T], query_norm: T, nprobe: usize) -> Vec<(T, usize)> {
        let mut cluster_dists: Vec<(T, usize)> = (0..self.nlist())
            .map(|c| {
                let cent = &self.centroids()[c * self.dim()..(c + 1) * self.dim()];
                let dist = match self.metric() {
                    Dist::Euclidean => euclidean_distance_static(query_vec, cent),
                    Dist::Cosine => {
                        let c_norm = &self.centroids_norm()[c];
                        cosine_distance_static_norm(query_vec, cent, &query_norm, c_norm)
                    }
                };
                (dist, c)
            })
            .collect();

        if nprobe < self.nlist() {
            cluster_dists.select_nth_unstable_by(nprobe, |a, b| a.0.partial_cmp(&b.0).unwrap());
        }

        cluster_dists
    }

    /// Special version that assumes pre-normalised vectors for Cosine
    ///
    /// ### Params
    ///
    /// * `query_vec` - The slice of the query
    /// * `nprobe` - Number of probes
    ///
    /// ### Returns
    ///
    /// The distance to the different clusters
    fn get_centroids_prenorm(&self, query_vec: &[T], nprobe: usize) -> Vec<(T, usize)> {
        // find top nprobe centroids
        let mut cluster_dists: Vec<(T, usize)> = (0..self.nlist())
            .map(|c| {
                let cent = &self.centroids()[c * self.dim()..(c + 1) * self.dim()];
                let dist = match self.metric() {
                    Dist::Cosine => T::one() - T::dot_simd(query_vec, cent),
                    Dist::Euclidean => T::euclidean_simd(query_vec, cent),
                };
                (dist, c)
            })
            .collect();

        let nprobe = nprobe.min(self.nlist());
        if nprobe < self.nlist() {
            cluster_dists.select_nth_unstable_by(nprobe, |a, b| a.0.partial_cmp(&b.0).unwrap());
        }

        cluster_dists
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
/// * `vec_norm` - The norm of the vector
/// * `centroids` - Current centroids (flattened)
/// * `centroid_norms` - The norms of the centroids
/// * `dim` - Embedding dimensions
/// * `n_centroids` - Number of centroids
/// * `metric` - Distance metric
///
/// ### Returns
///
/// Minimum distance to any centroid
fn min_distance_to_centroids<T>(
    vec: &[T],
    vec_norm: T,
    centroids: &[T],
    centroid_norms: &[T],
    dim: usize,
    n_centroids: usize,
    metric: &Dist,
) -> T
where
    T: Float + SimdDistance,
{
    let mut min_dist = T::infinity();

    for c in 0..n_centroids {
        let cent = &centroids[c * dim..(c + 1) * dim];
        let dist = match metric {
            Dist::Euclidean => euclidean_distance_static(vec, cent),
            Dist::Cosine => cosine_distance_static_norm(vec, cent, &vec_norm, &centroid_norms[c]),
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
/// * `data_norms` - The precomputed norms of the candidates.
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
    data_norms: &[T],
    dim: usize,
    k: usize,
    metric: &Dist,
    seed: usize,
) -> Vec<T>
where
    T: Float + SimdDistance,
{
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let n = data.len() / dim;

    if n <= k {
        return data.to_vec();
    }

    let mut centroids = Vec::with_capacity(k * dim);
    let mut centroid_norms = Vec::with_capacity(k);

    let first = rng.random_range(0..n);
    centroids.extend_from_slice(&data[first * dim..(first + 1) * dim]);
    centroid_norms.push(data_norms[first]);

    for _ in 1..k {
        let distances: Vec<T> = (0..n)
            .map(|i| {
                let vec = &data[i * dim..(i + 1) * dim];
                min_distance_to_centroids(
                    vec,
                    data_norms[i],
                    &centroids,
                    &centroid_norms,
                    dim,
                    centroid_norms.len(),
                    metric,
                )
            })
            .collect();

        let total: f64 = distances.iter().map(|&d| d.to_f64().unwrap()).sum();
        let threshold = rng.random::<f64>() * total;
        let mut cumsum = 0.0;

        for (idx, &dist) in distances.iter().enumerate() {
            cumsum += dist.to_f64().unwrap();
            if cumsum >= threshold {
                centroids.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
                centroid_norms.push(data_norms[idx]);
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
/// * `data_norms` - The norms of the trainint vectors.
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
    data_norms: &[T],
    dim: usize,
    n: usize,
    k: usize,
    metric: &Dist,
    seed: usize,
) -> Vec<T>
where
    T: Float + Send + Sync + SimdDistance,
{
    let mut rng = StdRng::seed_from_u64(seed as u64);
    let oversampling_factor = 2;
    let n_rounds = ((k as f64).ln() + 1.0) as usize;

    let first_idx = rng.random_range(0..n);
    let mut candidates = Vec::with_capacity(k * oversampling_factor * dim);
    let mut candidate_norms = Vec::with_capacity(k * oversampling_factor);

    candidates.extend_from_slice(&data[first_idx * dim..(first_idx + 1) * dim]);
    candidate_norms.push(data_norms[first_idx]);

    for _ in 0..n_rounds {
        let distances: Vec<T> = (0..n)
            .into_par_iter()
            .map(|i| {
                let vec = &data[i * dim..(i + 1) * dim];
                min_distance_to_centroids(
                    vec,
                    data_norms[i],
                    &candidates,
                    &candidate_norms,
                    dim,
                    candidate_norms.len(),
                    metric,
                )
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
                    candidate_norms.push(data_norms[idx]);
                    break;
                }
            }
        }
    }

    weighted_kmeans_plus_plus(&candidates, &candidate_norms, dim, k, metric, seed + 1)
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
pub fn fast_random_init<T>(data: &[T], dim: usize, n: usize, k: usize, seed: usize) -> Vec<T>
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
/// * `data_norms` - The precomputed norms of the data
/// * `dim` - Embedding dimensions
/// * `n` - Number of training vectors
/// * `centroids` - Current centroids (modified in-place)
/// * `centroid_norms` - Current centroid norms (modified in-place)
/// * `k` - Number of clusters
/// * `metric` - Distance metric
/// * `max_iters` - Maximum iterations
/// * `verbose` - Print iteration progress
#[allow(clippy::too_many_arguments)]
fn parallel_lloyd<T>(
    data: &[T],
    data_norms: &[T],
    dim: usize,
    n: usize,
    centroids: &mut [T],
    centroid_norms: &mut [T],
    k: usize,
    metric: &Dist,
    max_iters: usize,
    verbose: bool,
) where
    T: Float + Send + Sync + SimdDistance,
{
    let mut prev_assignments: Vec<usize> = vec![usize::MAX; n];
    let use_simd = dim >= 64;

    for iter in 0..max_iters {
        let mut assignments = assign_all_parallel(
            data,
            data_norms,
            dim,
            n,
            centroids,
            centroid_norms,
            k,
            metric,
        );

        let changed: usize = assignments
            .iter()
            .zip(prev_assignments.iter())
            .filter(|(a, b)| a != b)
            .count();

        if changed == 0 {
            if verbose {
                println!("    Converged at iteration {}", iter + 1);
            }
            break;
        }

        let (new_sums, counts) = if use_simd {
            (0..n)
                .into_par_iter()
                .fold(
                    || (vec![T::zero(); k * dim], vec![0usize; k]),
                    |(mut sums, mut counts), i| {
                        let cluster = assignments[i];
                        counts[cluster] += 1;
                        let vec = &data[i * dim..(i + 1) * dim];
                        T::add_assign_simd(&mut sums[cluster * dim..(cluster + 1) * dim], vec);
                        (sums, counts)
                    },
                )
                .reduce(
                    || (vec![T::zero(); k * dim], vec![0usize; k]),
                    |(mut sums1, mut counts1), (sums2, counts2)| {
                        T::add_assign_simd(&mut sums1, &sums2);
                        for i in 0..counts1.len() {
                            counts1[i] += counts2[i];
                        }
                        (sums1, counts1)
                    },
                )
        } else {
            (0..n)
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
                )
        };

        for c in 0..k {
            if counts[c] > 0 {
                let count_t = T::from(counts[c]).unwrap();
                for d in 0..dim {
                    centroids[c * dim + d] = new_sums[c * dim + d] / count_t;
                }
                if matches!(metric, Dist::Cosine) {
                    centroid_norms[c] = T::calculate_norm(&centroids[c * dim..(c + 1) * dim]);
                }
            }
        }

        std::mem::swap(&mut prev_assignments, &mut assignments);

        if verbose && (iter + 1) % 10 == 0 {
            println!(
                "    Iteration {} complete ({} assignments changed)",
                iter + 1,
                changed
            );
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + SimdDistance,
{
    let data_norms = if matches!(metric, Dist::Cosine) {
        (0..n)
            .map(|i| T::calculate_norm(&data[i * dim..(i + 1) * dim]))
            .collect()
    } else {
        vec![T::one(); n]
    };

    let mut centroids = if n_centroids > 200 {
        if verbose {
            println!("  Initialising centroids via fast random selection");
        }
        fast_random_init(data, dim, n, n_centroids, seed)
    } else {
        if verbose {
            println!("  Initialising centroids via k-means||");
        }
        kmeans_parallel_init(data, &data_norms, dim, n, n_centroids, metric, seed)
    };

    let mut centroid_norms = if matches!(metric, Dist::Cosine) {
        (0..n_centroids)
            .map(|i| T::calculate_norm(&centroids[i * dim..(i + 1) * dim]))
            .collect()
    } else {
        vec![T::one(); n_centroids]
    };

    if verbose {
        println!("  Running parallel Lloyd's iterations");
    }
    parallel_lloyd(
        data,
        &data_norms,
        dim,
        n,
        &mut centroids,
        &mut centroid_norms,
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
#[allow(clippy::too_many_arguments)]
pub fn assign_all_parallel<T>(
    data: &[T],
    data_norms: &[T],
    dim: usize,
    n: usize,
    centroids: &[T],
    centroid_norms: &[T],
    k: usize,
    metric: &Dist,
) -> Vec<usize>
where
    T: Float + Send + Sync + SimdDistance,
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
                    Dist::Cosine => {
                        cosine_distance_static_norm(vec, cent, &data_norms[i], &centroid_norms[c])
                    }
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

/// Print summary statistics of cluster assignments
///
/// ### Params
///
/// * `assignments` - Cluster assignment for each vector
/// * `nlist` - Number of clusters
pub fn print_cluster_summary(assignments: &[usize], nlist: usize) {
    let mut counts = vec![0usize; nlist];
    for &cluster in assignments {
        counts[cluster] += 1;
    }

    counts.sort_unstable();

    let n = assignments.len();
    let min = counts[0];
    let max = counts[nlist - 1];
    let p25 = counts[nlist / 4];
    let p50 = counts[nlist / 2];
    let p75 = counts[3 * nlist / 4];
    let mean = n / nlist;

    println!("Cluster size distribution:");
    println!("  Min:    {}", min);
    println!("  P25:    {}", p25);
    println!("  Median: {}", p50);
    println!("  P75:    {}", p75);
    println!("  Max:    {}", max);
    println!("  Mean:   {}", mean);
    println!("  Imbalance ratio: {:.2}", max as f64 / mean as f64);
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_build_csr_layout() {
        let assignments = vec![0, 1, 0, 2, 1, 0];
        let (indices, offsets) = build_csr_layout(assignments, 6, 3);

        // Cluster 0: vectors 0, 2, 5
        // Cluster 1: vectors 1, 4
        // Cluster 2: vector 3
        assert_eq!(offsets, vec![0, 3, 5, 6]);

        let cluster_0: Vec<_> = indices[offsets[0]..offsets[1]].to_vec();
        let cluster_1: Vec<_> = indices[offsets[1]..offsets[2]].to_vec();
        let cluster_2: Vec<_> = indices[offsets[2]..offsets[3]].to_vec();

        assert_eq!(cluster_0.len(), 3);
        assert!(cluster_0.contains(&0) && cluster_0.contains(&2) && cluster_0.contains(&5));
        assert_eq!(cluster_1.len(), 2);
        assert!(cluster_1.contains(&1) && cluster_1.contains(&4));
        assert_eq!(cluster_2, vec![3]);
    }

    #[test]
    fn test_build_csr_layout_single_cluster() {
        let assignments = vec![0, 0, 0];
        let (indices, offsets) = build_csr_layout(assignments, 3, 1);

        assert_eq!(offsets, vec![0, 3]);
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_build_csr_layout_empty_clusters() {
        let assignments = vec![0, 2, 0];
        let (_, offsets) = build_csr_layout(assignments, 3, 3);

        assert_eq!(offsets, vec![0, 2, 2, 3]);
        // Cluster 1 is empty
        assert_eq!(offsets[2] - offsets[1], 0);
    }

    #[test]
    fn test_assign_all_parallel_euclidean() {
        let data = vec![
            0.0, 0.0, // Near centroid 0
            0.1, 0.1, // Near centroid 0
            10.0, 10.0, // Near centroid 1
            9.9, 10.1, // Near centroid 1
        ];

        let centroids = vec![0.0, 0.0, 10.0, 10.0];

        let data_norms = vec![1.0; 4];
        let centroid_norms = vec![1.0; 2];

        let assignments = assign_all_parallel(
            &data,
            &data_norms,
            2,
            4,
            &centroids,
            &centroid_norms,
            2,
            &Dist::Euclidean,
        );

        assert_eq!(assignments, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_assign_all_parallel_cosine() {
        let data = vec![
            1.0, 0.0, // Aligned with centroid 0
            0.0, 1.0, // Aligned with centroid 1
            0.7, 0.1, // Closer to centroid 0
        ];

        let centroids = vec![1.0, 0.0, 0.0, 1.0];

        let data_norms: Vec<f64> = (0..3)
            .map(|i| {
                data[i * 2..(i + 1) * 2]
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        let centroid_norms: Vec<f64> = (0..2)
            .map(|i| {
                centroids[i * 2..(i + 1) * 2]
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        let assignments = assign_all_parallel(
            &data,
            &data_norms,
            2,
            3,
            &centroids,
            &centroid_norms,
            2,
            &Dist::Cosine,
        );

        assert_eq!(assignments[0], 0);
        assert_eq!(assignments[1], 1);
        assert_eq!(assignments[2], 0);
    }

    #[test]
    fn test_sample_vectors() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let (sampled, indices) = sample_vectors(&data, 2, 4, 2, 42);

        assert_eq!(sampled.len(), 4); // 2 samples * 2 dims
        assert_eq!(indices.len(), 2);

        // Verify sampled data matches indices
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(sampled[i * 2], data[idx * 2]);
            assert_eq!(sampled[i * 2 + 1], data[idx * 2 + 1]);
        }
    }

    #[test]
    fn test_sample_vectors_deterministic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (sample1, indices1) = sample_vectors(&data, 2, 3, 2, 42);
        let (sample2, indices2) = sample_vectors(&data, 2, 3, 2, 42);

        assert_eq!(indices1, indices2);
        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_fast_random_init() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let centroids = fast_random_init(&data, 2, 4, 2, 42);

        assert_eq!(centroids.len(), 4); // 2 centroids * 2 dims

        // Check centroids are from original data
        let mut found = 0;
        for i in 0..2 {
            let cent = &centroids[i * 2..(i + 1) * 2];
            for j in 0..4 {
                let vec = &data[j * 2..(j + 1) * 2];
                if cent[0] == vec[0] && cent[1] == vec[1] {
                    found += 1;
                    break;
                }
            }
        }
        assert_eq!(found, 2);
    }

    #[test]
    fn test_train_centroids_small() {
        let data = vec![0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];

        let centroids = train_centroids(&data, 2, 4, 2, &Dist::Euclidean, 10, 42, false);

        assert_eq!(centroids.len(), 4);

        // Check centroids are roughly at the two clusters
        let cent0 = (centroids[0], centroids[1]);
        let cent1 = (centroids[2], centroids[3]);

        let dist_00 = (cent0.0 - 0.05).powi(2) + (cent0.1 - 0.05).powi(2);
        let dist_01 = (cent0.0 - 10.05).powi(2) + (cent0.1 - 10.05).powi(2);
        let dist_10 = (cent1.0 - 0.05).powi(2) + (cent1.1 - 0.05).powi(2);
        let dist_11 = (cent1.0 - 10.05).powi(2) + (cent1.1 - 10.05).powi(2);

        // One centroid near (0,0), one near (10,10)
        assert!(
            (dist_00 < dist_01 && dist_11 < dist_10) || (dist_01 < dist_00 && dist_10 < dist_11)
        );
    }

    #[test]
    fn test_min_distance_to_centroids() {
        let vec = vec![5.0, 5.0];
        let vec_norm = (vec[0] * vec[0] + vec[1] * vec[1]).sqrt();
        let centroids = vec![0.0, 0.0, 10.0, 10.0];
        let centroid_norms = vec![0.0, (10.0f64 * 10.0 + 10.0 * 10.0).sqrt()];

        let dist = min_distance_to_centroids(
            &vec,
            vec_norm,
            &centroids,
            &centroid_norms,
            2,
            2,
            &Dist::Euclidean,
        );

        // Distance to (0,0) is 50, to (10,10) is 50, so min is 50
        assert_relative_eq!(dist, 50.0, epsilon = 1e-5);
    }

    #[test]
    fn test_weighted_kmeans_plus_plus() {
        let data = vec![0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 10.0, 10.0, 10.1, 10.1];
        let data_norms: Vec<f64> = (0..5)
            .map(|i| {
                data[i * 2..(i + 1) * 2]
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();

        let centroids = weighted_kmeans_plus_plus(&data, &data_norms, 2, 2, &Dist::Euclidean, 42);

        assert_eq!(centroids.len(), 4);

        // Should pick one from each cluster
        let cent0 = (centroids[0], centroids[1]);
        let cent1 = (centroids[2], centroids[3]);

        let near_zero_0 = cent0.0.abs() < 1.0 && cent0.1.abs() < 1.0;
        let near_ten_0 = (cent0.0 - 10.0).abs() < 1.0 && (cent0.1 - 10.0).abs() < 1.0;
        let near_zero_1 = cent1.0.abs() < 1.0 && cent1.1.abs() < 1.0;
        let near_ten_1 = (cent1.0 - 10.0).abs() < 1.0 && (cent1.1 - 10.0).abs() < 1.0;

        assert!((near_zero_0 && near_ten_1) || (near_ten_0 && near_zero_1));
    }
}
