//! Implementations of fast k-means clustering, leveraging SIMD or GEMM during
//! fitting, pending the data set sizes.

use faer::{linalg::matmul::matmul, Accum, Mat, MatRef, Par};
use faer_traits::ComplexField;
use num_traits::Float;
use num_traits::FromPrimitive;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::iter::Sum;
use std::num::NonZero;

use crate::prelude::AnnSearchFloat;
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

/// Tile size for GEMM-based assignment. Limits the intermediate dot-product
/// matrix to TILE_SIZE * k elements. 4096 is a reasonable default; tune to
/// your L2 cache size if needed.
const GEMM_TILE_SIZE: usize = 4096;

/// Below this number of dirty points, skip GEMM gather/scatter overhead
/// and compute distances directly via SIMD loops.
const GEMM_DIRTY_THRESHOLD: usize = 128;

/// Minimum dimension at which GEMM assignment outperforms direct SIMD loops.
/// Below this, the GEMM kernel setup and tile-scanning overhead exceeds the
/// cache-blocking benefit. This needs to be quite high for GEMM to actually
/// be better.
const GEMM_DIM_THRESHOLD: usize = 96;

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

    match metric {
        Dist::Euclidean => {
            for cent in centroids.chunks_exact(dim).take(n_centroids) {
                let dist = euclidean_distance_static(vec, cent);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
        Dist::Cosine => {
            let cent_iter = centroids.chunks_exact(dim);
            let norm_iter = centroid_norms.iter();

            for (cent, &c_norm) in cent_iter.zip(norm_iter).take(n_centroids) {
                let dist = cosine_distance_static_norm(vec, cent, &vec_norm, &c_norm);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
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

    let mut distances = vec![T::infinity(); n];

    for _ in 1..k {
        let latest_centroid = &centroids[(centroids.len() - dim)..];
        let latest_norm = *centroid_norms.last().unwrap();

        match metric {
            Dist::Euclidean => {
                for (i, dist) in distances.iter_mut().enumerate() {
                    let vec = &data[i * dim..(i + 1) * dim];
                    let d = euclidean_distance_static(vec, latest_centroid);
                    if d < *dist {
                        *dist = d;
                    }
                }
            }
            Dist::Cosine => {
                for (i, dist) in distances.iter_mut().enumerate() {
                    let vec = &data[i * dim..(i + 1) * dim];
                    let d = cosine_distance_static_norm(
                        vec,
                        latest_centroid,
                        &data_norms[i],
                        &latest_norm,
                    );
                    if d < *dist {
                        *dist = d;
                    }
                }
            }
        }

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

    let mut distances = vec![T::zero(); n];

    for _ in 0..n_rounds {
        distances.par_iter_mut().enumerate().for_each(|(i, dist)| {
            let vec = &data[i * dim..(i + 1) * dim];
            *dist = min_distance_to_centroids(
                vec,
                data_norms[i],
                &candidates,
                &candidate_norms,
                dim,
                candidate_norms.len(),
                metric,
            );
        });

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

///////////////////////////
// GEMM-based assignment //
///////////////////////////

/// Compute dot product tile: dots[i,c] = dot(data_block[i], centroids[c])
///
/// Uses faer GEMM to compute dots = data_mat * centroids^T. The output
/// matrix is reused across tiles and resized only when dimensions change.
///
/// ### Params
///
/// * `data_block` - Tile of input vectors, row-major (tile_n * dim elements)
/// * `tile_n` - Number of vectors in this tile
/// * `centroids` - All centroids, row-major (k * dim elements)
/// * `dim` - Embedding dimensions
/// * `k` - Number of centroids
/// * `dots` - Output matrix (tile_n x k), overwritten in place
#[inline]
fn gemm_dot_tile<T>(
    data_block: &[T],
    tile_n: usize,
    centroids: &[T],
    dim: usize,
    k: usize,
    dots: &mut Mat<T>,
) where
    T: Float + SimdDistance + faer_traits::ComplexField,
{
    let data_mat = MatRef::from_row_major_slice(data_block, tile_n, dim);
    let cent_mat = MatRef::from_row_major_slice(centroids, k, dim);

    if dots.nrows() != tile_n || dots.ncols() != k {
        *dots = Mat::<T>::zeros(tile_n, k);
    }

    // dots = 1.0 * data_mat * cent_mat^T, overwriting
    matmul(
        dots.as_mut(),
        Accum::Replace,
        data_mat,
        cent_mat.transpose(),
        T::one(),
        Par::Seq,
    );
}

/// Full GEMM-based nearest centroid assignment over all n vectors
///
/// Processes vectors in tiles of GEMM_TILE_SIZE. For each vector, finds
/// the closest and second-closest centroid using the dot-product trick
/// to avoid explicit distance computation.
///
/// For Euclidean: dist^2 = ||x||^2 - 2*dot(x,c) + ||c||^2, so maximising
/// 2*dot - ||c||^2 minimises squared distance.
///
/// For Cosine: similarity = dot(x,c) / (||x|| * ||c||), so maximising
/// dot / ||c|| (for fixed ||x||) minimises cosine distance.
///
/// ### Params
///
/// * `data` - All vectors, flattened row-major
/// * `data_norms_sq` - Per-vector norms: ||x||^2 for Euclidean, ||x|| for
///   Cosine
/// * `dim` - Embedding dimensions
/// * `centroids` - All centroids, flattened row-major
/// * `centroid_norms` - Per-centroid norms: ||c||^2 for Euclidean, ||c|| for
///   Cosine
/// * `k` - Number of centroids
/// * `metric` - Distance metric
/// * `assignments` - Output: nearest centroid index per vector
/// * `upper_bounds` - Output: distance to nearest centroid per vector
/// * `lower_bounds` - Output: distance to second-nearest centroid per vector
#[allow(clippy::too_many_arguments)]
fn gemm_assign_full<T>(
    data: &[T],
    data_norms_sq: &[T], // ||x||^2 for Euclidean; ||x|| for Cosine
    dim: usize,
    centroids: &[T],
    centroid_norms: &[T], // ||c||^2 for Euclidean; ||c|| for Cosine
    k: usize,
    metric: &Dist,
    assignments: &mut [usize],
    upper_bounds: &mut [T],
    lower_bounds: &mut [T],
) where
    T: Float + SimdDistance + faer_traits::ComplexField,
{
    let two = T::one() + T::one();

    data.par_chunks(GEMM_TILE_SIZE * dim)
        .zip(data_norms_sq.par_chunks(GEMM_TILE_SIZE))
        .zip(assignments.par_chunks_mut(GEMM_TILE_SIZE))
        .zip(upper_bounds.par_chunks_mut(GEMM_TILE_SIZE))
        .zip(lower_bounds.par_chunks_mut(GEMM_TILE_SIZE))
        .for_each_init(
            || Mat::<T>::new(), // Thread-local matrix buffer
            |dots, ((((data_block, norm_block), assign_block), upper_block), lower_block)| {
                let tile_n = norm_block.len();

                // Compute dots sequentially *within* this Rayon thread
                gemm_dot_tile(data_block, tile_n, centroids, dim, k, dots);

                // Sequential argmax reduction over the tile
                for local_i in 0..tile_n {
                    let mut best_c = 0;
                    let mut best_score = T::neg_infinity();
                    let mut second_score = T::neg_infinity();

                    match metric {
                        Dist::Euclidean => {
                            for c in 0..k {
                                let score = two * dots[(local_i, c)] - centroid_norms[c];
                                if score > best_score {
                                    second_score = best_score;
                                    best_score = score;
                                    best_c = c;
                                } else if score > second_score {
                                    second_score = score;
                                }
                            }
                            assign_block[local_i] = best_c;
                            upper_block[local_i] =
                                (norm_block[local_i] - best_score).max(T::zero()).sqrt();
                            lower_block[local_i] =
                                (norm_block[local_i] - second_score).max(T::zero()).sqrt();
                        }
                        Dist::Cosine => {
                            for c in 0..k {
                                let cn = centroid_norms[c];
                                let inv_cn = if cn > T::zero() {
                                    T::one() / cn
                                } else {
                                    T::zero()
                                };
                                let score = dots[(local_i, c)] * inv_cn;
                                if score > best_score {
                                    second_score = best_score;
                                    best_score = score;
                                    best_c = c;
                                } else if score > second_score {
                                    second_score = score;
                                }
                            }
                            let xn = norm_block[local_i];
                            let inv_xn = if xn > T::zero() {
                                T::one() / xn
                            } else {
                                T::zero()
                            };
                            assign_block[local_i] = best_c;
                            upper_block[local_i] = T::one() - best_score * inv_xn;
                            lower_block[local_i] = T::one() - second_score * inv_xn;
                        }
                    }
                }
            },
        );
}

/// Reassign a subset of "dirty" points whose bounds are no longer tight.
///
/// For small dirty sets (< GEMM_DIRTY_THRESHOLD), computes distances
/// directly via SIMD dot products to avoid gather/scatter overhead.
/// For larger sets, gathers dirty vectors into a contiguous buffer,
/// runs full GEMM assignment, and scatters results back.
///
/// ### Params
///
/// * `data` - All vectors, flattened row-major
/// * `data_norms_sq` - Per-vector norms: `||x||^2` for Euclidean, `||x||` for
///   Cosine
/// * `dim` - Embedding dimension
/// * `centroids` - All centroids, flattened row-major
/// * `centroid_norms` - Per-centroid norms: `||c||^2` for Euclidean, `||c||`
///   for Cosine
/// * `k` - Number of centroids
/// * `metric` - Distance metric
/// * `dirty` - Indices of vectors requiring reassignment
/// * `assignments` - In/out: nearest centroid index per vector
/// * `upper_bounds` - In/out: distance to nearest centroid per vector
/// * `lower_bounds` - In/out: distance to second-nearest centroid per vector
/// * `gathered_data` - Scratch buffer for gathering dirty vectors into a
///   contiguous block
/// * `gathered_norms` - Scratch buffer for norms corresponding to gathered
///   vectors
/// * `tmp_assign` - Scratch buffer for centroid assignments of gathered vectors
/// * `tmp_upper` - Scratch buffer for upper bounds of gathered vectors
/// * `tmp_lower` - Scratch buffer for lower bounds of gathered vectors
#[allow(clippy::too_many_arguments)]
fn gemm_reassign_dirty<T>(
    data: &[T],
    data_norms_sq: &[T],
    dim: usize,
    centroids: &[T],
    centroid_norms: &[T],
    k: usize,
    metric: &Dist,
    dirty: &[usize],
    assignments: &mut [usize],
    upper_bounds: &mut [T],
    lower_bounds: &mut [T],
    // scratch spaces
    gathered_data: &mut Vec<T>,
    gathered_norms: &mut Vec<T>,
    tmp_assign: &mut [usize],
    tmp_upper: &mut [T],
    tmp_lower: &mut [T],
) where
    T: Float + SimdDistance + faer_traits::ComplexField,
{
    let nd = dirty.len();

    if nd < GEMM_DIRTY_THRESHOLD {
        let two = T::one() + T::one();
        for &i in dirty {
            let vec = &data[i * dim..(i + 1) * dim];
            let mut best_c = 0;
            let mut best_score = T::neg_infinity();
            let mut second_score = T::neg_infinity();

            match metric {
                Dist::Euclidean => {
                    for c in 0..k {
                        let cent = &centroids[c * dim..(c + 1) * dim];
                        let dot = T::dot_simd(vec, cent);
                        let score = two * dot - centroid_norms[c];
                        if score > best_score {
                            second_score = best_score;
                            best_score = score;
                            best_c = c;
                        } else if score > second_score {
                            second_score = score;
                        }
                    }
                    assignments[i] = best_c;
                    upper_bounds[i] = (data_norms_sq[i] - best_score).max(T::zero()).sqrt();
                    lower_bounds[i] = (data_norms_sq[i] - second_score).max(T::zero()).sqrt();
                }
                Dist::Cosine => {
                    for c in 0..k {
                        let cent = &centroids[c * dim..(c + 1) * dim];
                        let dot = T::dot_simd(vec, cent);
                        let cn = centroid_norms[c];
                        let inv_cn = if cn > T::zero() {
                            T::one() / cn
                        } else {
                            T::zero()
                        };
                        let score = dot * inv_cn;
                        if score > best_score {
                            second_score = best_score;
                            best_score = score;
                            best_c = c;
                        } else if score > second_score {
                            second_score = score;
                        }
                    }
                    let xn = data_norms_sq[i];
                    let inv_xn = if xn > T::zero() {
                        T::one() / xn
                    } else {
                        T::zero()
                    };
                    assignments[i] = best_c;
                    upper_bounds[i] = T::one() - best_score * inv_xn;
                    lower_bounds[i] = T::one() - second_score * inv_xn;
                }
            }
        }
        return;
    }

    // gather dirty vectors into contiguous buffer
    gathered_data.clear();
    gathered_norms.clear();
    for &i in dirty {
        gathered_data.extend_from_slice(&data[i * dim..(i + 1) * dim]);
        gathered_norms.push(data_norms_sq[i]);
    }

    gemm_assign_full(
        gathered_data,
        gathered_norms,
        dim,
        centroids,
        centroid_norms,
        k,
        metric,
        &mut tmp_assign[..nd],
        &mut tmp_upper[..nd],
        &mut tmp_lower[..nd],
    );

    for (local, &global) in dirty.iter().enumerate() {
        assignments[global] = tmp_assign[local];
        upper_bounds[global] = tmp_upper[local];
        lower_bounds[global] = tmp_lower[local];
    }
}

///////////////////////////////
// Centroid update utilities //
///////////////////////////////

/// Recompute centroids as the mean of their assigned vectors
///
/// Uses parallel reduction with per-thread accumulators to sum vectors
/// and counts per cluster, then divides. Also recomputes centroid norms
/// in the format expected by the GEMM assignment path.
///
/// ### Params
///
/// * `data` - All vectors, flattened row-major
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `assignments` - Cluster assignment per vector
/// * `centroids` - In/out: centroids to update, flattened row-major
/// * `centroid_norms` - In/out: ||c||^2 for Euclidean, ||c|| for Cosine
/// * `k` - Number of centroids
/// * `metric` - Distance metric
#[allow(clippy::too_many_arguments)]
fn update_centroids<T>(
    data: &[T],
    dim: usize,
    n: usize,
    assignments: &[usize],
    centroids: &mut [T],
    centroid_norms: &mut [T],
    k: usize,
    metric: &Dist,
) where
    T: Float + Send + Sync + SimdDistance,
{
    let num_threads = rayon::current_num_threads();
    let chunk_size = (n + num_threads - 1) / num_threads.max(1);

    let (new_sums, counts) = assignments
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, assignment_chunk)| {
            let mut local_sums = vec![T::zero(); k * dim];
            let mut local_counts = vec![0usize; k];

            let start_idx = chunk_idx * chunk_size;
            let data_chunk = &data[start_idx * dim..(start_idx + assignment_chunk.len()) * dim];

            for (i, &cluster) in assignment_chunk.iter().enumerate() {
                local_counts[cluster] += 1;
                let vec = &data_chunk[i * dim..(i + 1) * dim];
                let offset = cluster * dim;
                T::add_assign_simd(&mut local_sums[offset..offset + dim], vec);
            }
            (local_sums, local_counts)
        })
        .reduce(
            || (vec![T::zero(); k * dim], vec![0usize; k]),
            |(mut s1, mut c1), (s2, c2)| {
                T::add_assign_simd(&mut s1, &s2);
                for i in 0..c1.len() {
                    c1[i] += c2[i];
                }
                (s1, c1)
            },
        );

    for c in 0..k {
        if counts[c] > 0 {
            let count_t = T::from(counts[c]).unwrap();
            let offset = c * dim;
            for d in 0..dim {
                centroids[offset + d] = new_sums[offset + d] / count_t;
            }
        }
        let cent = &centroids[c * dim..(c + 1) * dim];
        centroid_norms[c] = match metric {
            Dist::Euclidean => T::dot_simd(cent, cent), // ||c||^2
            Dist::Cosine => T::calculate_l2_norm(cent), // ||c||
        };
    }
}

/// Compute per-centroid drift after an update step
///
/// ### Params
///
/// * `old_centroids` - Centroids before the update, flattened row-major
/// * `new_centroids` - Centroids after the update, flattened row-major
/// * `dim` - Embedding dimensions
/// * `k` - Number of centroids
/// * `deltas` - Output: Euclidean distance each centroid moved
fn compute_centroid_drift<T>(
    old_centroids: &[T],
    new_centroids: &[T],
    dim: usize,
    k: usize,
    deltas: &mut [T],
) where
    T: Float + SimdDistance,
{
    for c in 0..k {
        let old = &old_centroids[c * dim..(c + 1) * dim];
        let new = &new_centroids[c * dim..(c + 1) * dim];
        deltas[c] = euclidean_distance_static(old, new);
    }
}

/// Compute s[c] = 0.5 * min_{c' != c} dist(c, c') for all centroids
///
/// Uses GEMM to compute the full centroid-centroid dot product matrix,
/// then derives pairwise Euclidean distances. Used in Hamerly's algorithm
/// to tighten lower bounds.
///
/// ### Params
///
/// * `centroids` - All centroids, flattened row-major
/// * `centroid_norms_sq` - Per-centroid ||c||^2
/// * `dim` - Embedding dimensions
/// * `k` - Number of centroids
///
/// ### Returns
///
/// Vector of length k with half-minimum inter-centroid distances
fn compute_half_min_centroid_dists<T>(
    centroids: &[T],
    centroid_norms_sq: &[T],
    dim: usize,
    k: usize,
) -> Vec<T>
where
    T: Float + SimdDistance + faer_traits::ComplexField,
{
    let cent_mat = MatRef::from_row_major_slice(centroids, k, dim);
    let mut cent_dots = Mat::<T>::zeros(k, k);

    matmul(
        cent_dots.as_mut(),
        Accum::Replace,
        cent_mat,
        cent_mat.transpose(),
        T::one(),
        Par::Rayon(NonZero::new(rayon::current_num_threads()).unwrap()),
    );

    let half = T::one() / (T::one() + T::one());
    let two = T::one() + T::one();
    let mut s = vec![T::infinity(); k];

    for i in 0..k {
        for j in 0..k {
            if i == j {
                continue;
            }
            let dist_sq = centroid_norms_sq[i] - two * cent_dots[(i, j)] + centroid_norms_sq[j];
            let dist = dist_sq.max(T::zero()).sqrt();
            if dist < s[i] {
                s[i] = dist;
            }
        }
        s[i] = s[i] * half;
    }

    s
}

/// Find the two largest centroid drifts
///
/// ### Params
///
/// * `deltas` - Per-centroid drift values
///
/// ### Returns
///
/// Tuple of (largest drift, second largest drift, index of largest)
fn top_two_deltas<T: Float>(deltas: &[T]) -> (T, T, usize) {
    let mut max1 = T::neg_infinity();
    let mut max2 = T::neg_infinity();
    let mut max1_idx = 0;

    for (c, &d) in deltas.iter().enumerate() {
        if d > max1 {
            max2 = max1;
            max1 = d;
            max1_idx = c;
        } else if d > max2 {
            max2 = d;
        }
    }

    (max1, max2, max1_idx)
}

/// Compute exact Euclidean distance between a single point and a centroid
///
/// Uses the identity dist = sqrt(||x||^2 - 2*dot(x,c) + ||c||^2) with
/// a SIMD dot product. Used to tighten upper bounds in Hamerly's algorithm.
///
/// ### Params
///
/// * `data` - All vectors, flattened row-major
/// * `data_norms_sq` - Per-vector ||x||^2
/// * `dim` - Embedding dimensions
/// * `i` - Index of the vector
/// * `centroids` - All centroids, flattened row-major
/// * `centroid_norms_sq` - Per-centroid ||c||^2
/// * `c` - Index of the centroid
///
/// ### Returns
///
/// Euclidean distance between vector i and centroid c
#[inline]
fn exact_point_centroid_dist<T>(
    data: &[T],
    data_norms_sq: &[T],
    dim: usize,
    i: usize,
    centroids: &[T],
    centroid_norms_sq: &[T],
    c: usize,
) -> T
where
    T: Float + SimdDistance,
{
    let vec = &data[i * dim..(i + 1) * dim];
    let cent = &centroids[c * dim..(c + 1) * dim];
    let dot = T::dot_simd(vec, cent);
    let two = T::one() + T::one();
    let dist_sq = data_norms_sq[i] - two * dot + centroid_norms_sq[c];
    dist_sq.max(T::zero()).sqrt()
}

//////////////////////////////
// Hamerly's Lloyd's (Eucl) //
//////////////////////////////

/// Hamerly's accelerated k-means for Euclidean distance
///
/// Maintains per-point upper and lower distance bounds to skip redundant
/// distance computations. Points are only reassigned when their bounds
/// become loose enough that a cluster change is possible. Uses GEMM for
/// both initial full assignment and dirty-point reassignment.
///
/// ### Params
///
/// * `data` - All vectors, flattened row-major
/// * `data_norms_sq` - Per-vector ||x||^2
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `centroids` - In/out: centroids, flattened row-major
/// * `centroid_norms_sq` - In/out: per-centroid ||c||^2
/// * `k` - Number of centroids
/// * `max_iters` - Maximum number of Lloyd's iterations
/// * `verbose` - Print convergence diagnostics
#[allow(clippy::too_many_arguments)]
fn hamerly_lloyd<T>(
    data: &[T],
    data_norms_sq: &[T],
    dim: usize,
    n: usize,
    centroids: &mut [T],
    centroid_norms_sq: &mut [T],
    k: usize,
    max_iters: usize,
    verbose: bool,
) where
    T: Float + Send + Sync + SimdDistance + faer_traits::ComplexField + FromPrimitive,
{
    let mut assignments = vec![0usize; n];
    let mut upper = vec![T::infinity(); n];
    let mut lower = vec![T::zero(); n];
    let mut old_centroids = vec![T::zero(); k * dim];
    let mut deltas = vec![T::zero(); k];
    let mut dirty = Vec::with_capacity(n);

    let mut ws_gathered_data = Vec::with_capacity(n * dim);
    let mut ws_gathered_norms = Vec::with_capacity(n);
    let mut ws_tmp_assign = vec![0usize; n];
    let mut ws_tmp_upper = vec![T::zero(); n];
    let mut ws_tmp_lower = vec![T::zero(); n];

    gemm_assign_full(
        data,
        data_norms_sq,
        dim,
        centroids,
        centroid_norms_sq,
        k,
        &Dist::Euclidean,
        &mut assignments,
        &mut upper,
        &mut lower,
    );

    for iter in 0..max_iters {
        old_centroids.copy_from_slice(centroids);

        update_centroids(
            data,
            dim,
            n,
            &assignments,
            centroids,
            centroid_norms_sq,
            k,
            &Dist::Euclidean,
        );

        compute_centroid_drift(&old_centroids, centroids, dim, k, &mut deltas);
        let (max_delta, second_max_delta, max_delta_idx) = top_two_deltas(&deltas);

        if max_delta <= T::from_f64(1e-5).unwrap() {
            if verbose {
                println!("    Converged at iteration {}", iter + 1);
            }
            break;
        }

        for i in 0..n {
            upper[i] = upper[i] + deltas[assignments[i]];
            let other_max = if assignments[i] == max_delta_idx {
                second_max_delta
            } else {
                max_delta
            };
            lower[i] = (lower[i] - other_max).max(T::zero());
        }

        let s = compute_half_min_centroid_dists(centroids, centroid_norms_sq, dim, k);

        dirty.clear();
        for i in 0..n {
            let m = if s[assignments[i]] > lower[i] {
                s[assignments[i]]
            } else {
                lower[i]
            };
            if upper[i] > m {
                upper[i] = exact_point_centroid_dist(
                    data,
                    data_norms_sq,
                    dim,
                    i,
                    centroids,
                    centroid_norms_sq,
                    assignments[i],
                );
                if upper[i] > m {
                    dirty.push(i);
                }
            }
        }

        if dirty.is_empty() {
            if verbose {
                println!("    Converged at iteration {} (bounds tight)", iter + 1);
            }
            break;
        }

        gemm_reassign_dirty(
            data,
            data_norms_sq,
            dim,
            centroids,
            centroid_norms_sq,
            k,
            &Dist::Euclidean,
            &dirty,
            &mut assignments,
            &mut upper,
            &mut lower,
            &mut ws_gathered_data,
            &mut ws_gathered_norms,
            &mut ws_tmp_assign,
            &mut ws_tmp_upper,
            &mut ws_tmp_lower,
        );

        if verbose && (iter + 1) % 10 == 0 {
            println!(
                "    Iteration {} ({} / {} points reassessed, {:.1}% pruned)",
                iter + 1,
                dirty.len(),
                n,
                (1.0 - dirty.len() as f64 / n as f64) * 100.0,
            );
        }
    }
}

////////////////////////////////
// GEMM-only Lloyd's (Cosine) //
////////////////////////////////

/// Plain Lloyd's k-means for cosine distance using GEMM assignment
///
/// Cosine distance does not satisfy the triangle inequality, so
/// Hamerly's bound-based pruning is not applicable. Instead, runs
/// full GEMM reassignment every iteration and converges when no
/// assignments change.
///
/// ### Params
///
/// * `data` - All vectors, flattened row-major
/// * `data_norms` - Per-vector ||x|| (L2 norms)
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `centroids` - In/out: centroids, flattened row-major
/// * `centroid_norms` - In/out: per-centroid ||c|| (L2 norms)
/// * `k` - Number of centroids
/// * `max_iters` - Maximum number of Lloyd's iterations
/// * `verbose` - Print convergence diagnostics
#[allow(clippy::too_many_arguments)]
fn gemm_lloyd_cosine<T>(
    data: &[T],
    data_norms: &[T],
    dim: usize,
    n: usize,
    centroids: &mut [T],
    centroid_norms: &mut [T],
    k: usize,
    max_iters: usize,
    verbose: bool,
) where
    T: Float + Send + Sync + SimdDistance + faer_traits::ComplexField,
{
    let mut assignments = vec![0usize; n];
    let mut prev_assignments = vec![usize::MAX; n];
    let mut upper = vec![T::zero(); n];
    let mut lower = vec![T::zero(); n];

    for iter in 0..max_iters {
        gemm_assign_full(
            data,
            data_norms,
            dim,
            centroids,
            centroid_norms,
            k,
            &Dist::Cosine,
            &mut assignments,
            &mut upper,
            &mut lower,
        );

        let changed: usize = assignments
            .par_iter()
            .zip(prev_assignments.par_iter())
            .filter(|(a, b)| a != b)
            .count();

        if changed == 0 {
            if verbose {
                println!("    Converged at iteration {}", iter + 1);
            }
            break;
        }

        update_centroids(
            data,
            dim,
            n,
            &assignments,
            centroids,
            centroid_norms,
            k,
            &Dist::Cosine,
        );

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

////////////////////
// Lloyd's (SIMD) //
////////////////////

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
    T: Float + Send + Sync + SimdDistance + ComplexField,
{
    let mut prev_assignments: Vec<usize> = vec![usize::MAX; n];
    let num_threads = rayon::current_num_threads();
    let chunk_size = (n + num_threads - 1) / num_threads.max(1);

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
            .par_iter()
            .zip(prev_assignments.par_iter())
            .filter(|(a, b)| a != b)
            .count();

        if changed == 0 {
            if verbose {
                println!("    Converged at iteration {}", iter + 1);
            }
            break;
        }

        let (new_sums, counts) = assignments
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, assignment_chunk)| {
                let mut local_sums = vec![T::zero(); k * dim];
                let mut local_counts = vec![0usize; k];

                let start_idx = chunk_idx * chunk_size;
                let data_chunk = &data[start_idx * dim..(start_idx + assignment_chunk.len()) * dim];

                for (i, &cluster) in assignment_chunk.iter().enumerate() {
                    local_counts[cluster] += 1;
                    let vec = &data_chunk[i * dim..(i + 1) * dim];
                    let cluster_offset = cluster * dim;

                    T::add_assign_simd(&mut local_sums[cluster_offset..cluster_offset + dim], vec);
                }
                (local_sums, local_counts)
            })
            .reduce(
                || (vec![T::zero(); k * dim], vec![0usize; k]),
                |(mut sums1, mut counts1), (sums2, counts2)| {
                    T::add_assign_simd(&mut sums1, &sums2);
                    for i in 0..counts1.len() {
                        counts1[i] += counts2[i];
                    }
                    (sums1, counts1)
                },
            );

        // Update centroids and compute STANDARD norms
        for c in 0..k {
            if counts[c] > 0 {
                let count_t = T::from(counts[c]).unwrap();
                let cluster_offset = c * dim;

                for d in 0..dim {
                    centroids[cluster_offset + d] = new_sums[cluster_offset + d] / count_t;
                }

                if matches!(metric, Dist::Cosine) {
                    let cent = &centroids[cluster_offset..cluster_offset + dim];
                    centroid_norms[c] = T::calculate_l2_norm(cent);
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

////////////////
// Assignment //
////////////////

/// Assign vectors to nearest centroids using GEMM-based distance computation
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
fn gemm_assign<T>(
    data: &[T],
    dim: usize,
    n: usize,
    centroids: &[T],
    k: usize,
    metric: &Dist,
) -> Vec<usize>
where
    T: Float + Send + Sync + SimdDistance + ComplexField,
{
    let data_norms: Vec<T> = match metric {
        Dist::Euclidean => (0..n)
            .map(|i| {
                let v = &data[i * dim..(i + 1) * dim];
                T::dot_simd(v, v)
            })
            .collect(),
        Dist::Cosine => (0..n)
            .map(|i| T::calculate_l2_norm(&data[i * dim..(i + 1) * dim]))
            .collect(),
    };
    let centroid_norms: Vec<T> = match metric {
        Dist::Euclidean => (0..k)
            .map(|c| {
                let cent = &centroids[c * dim..(c + 1) * dim];
                T::dot_simd(cent, cent)
            })
            .collect(),
        Dist::Cosine => (0..k)
            .map(|c| T::calculate_l2_norm(&centroids[c * dim..(c + 1) * dim]))
            .collect(),
    };

    let mut assignments = vec![0usize; n];
    let mut upper = vec![T::zero(); n];
    let mut lower = vec![T::zero(); n];

    gemm_assign_full(
        data,
        &data_norms,
        dim,
        centroids,
        &centroid_norms,
        k,
        metric,
        &mut assignments,
        &mut upper,
        &mut lower,
    );

    assignments
}

/// Assign vectors to nearest centroids via direct dot product comparisons
///
/// ### Params
///
/// * `data` - Vectors to assign (flattened)
/// * `_data_norms` - Norms of the vectors (unused)
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `centroids` - Current centroids
/// * `centroid_norms` - Norms of the centroids
/// * `k` - Number of clusters
/// * `metric` - Distance metric
///
/// ### Returns
///
/// Vector of cluster assignments (one per input vector)
#[allow(clippy::too_many_arguments)]
fn direct_assign<T>(
    data: &[T],
    _data_norms: &[T],
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
    let two = T::one() + T::one();

    let shortcut_norms: Vec<T> = match metric {
        Dist::Euclidean => (0..k)
            .map(|c| {
                let cent = &centroids[c * dim..(c + 1) * dim];
                T::dot_simd(cent, cent)
            })
            .collect(),
        Dist::Cosine => (0..k)
            .map(|c| {
                let norm = centroid_norms[c];
                if norm > T::zero() {
                    T::one() / norm
                } else {
                    T::zero()
                }
            })
            .collect(),
    };

    match metric {
        Dist::Euclidean => (0..n)
            .into_par_iter()
            .map(|i| {
                let vec = &data[i * dim..(i + 1) * dim];
                let mut best = 0;
                let mut max_score = T::neg_infinity();
                for c in 0..k {
                    let cent = &centroids[c * dim..(c + 1) * dim];
                    let score = two * T::dot_simd(vec, cent) - shortcut_norms[c];
                    if score > max_score {
                        max_score = score;
                        best = c;
                    }
                }
                best
            })
            .collect(),
        Dist::Cosine => (0..n)
            .into_par_iter()
            .map(|i| {
                let vec = &data[i * dim..(i + 1) * dim];
                let mut best = 0;
                let mut max_score = T::neg_infinity();
                for c in 0..k {
                    let cent = &centroids[c * dim..(c + 1) * dim];
                    let score = T::dot_simd(vec, cent) * shortcut_norms[c];
                    if score > max_score {
                        max_score = score;
                        best = c;
                    }
                }
                best
            })
            .collect(),
    }
}

/// Assign all vectors to their nearest centroids in parallel
///
/// ### Params
///
/// * `data` - Vectors to assign (flattened)
/// * `data_norms` - Norms of the vector
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `centroids` - Current centroids
/// * `centroid_norms` - Norms of the centroid
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
    T: Float + Send + Sync + SimdDistance + ComplexField,
{
    if dim >= GEMM_DIM_THRESHOLD {
        gemm_assign(data, dim, n, centroids, k, metric)
    } else {
        direct_assign(
            data,
            data_norms,
            dim,
            n,
            centroids,
            centroid_norms,
            k,
            metric,
        )
    }
}

//////////
// Main //
//////////

/// Train k-means centroids
///
/// Pending on the dimensionality of the data, it will use either
/// SIMD-accelerated k-means clustering via Lloyd's (n_dim ≤ 64) or use
/// a GEMM-accelerated version for larger data sets.
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
    T: AnnSearchFloat,
{
    let data_norms: Vec<T> = match metric {
        Dist::Euclidean => (0..n)
            .into_par_iter()
            .map(|i| {
                let v = &data[i * dim..(i + 1) * dim];
                T::dot_simd(v, v)
            })
            .collect(),
        Dist::Cosine => (0..n)
            .into_par_iter()
            .map(|i| T::calculate_l2_norm(&data[i * dim..(i + 1) * dim]))
            .collect(),
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
        let init_norms: Vec<T> = match metric {
            Dist::Euclidean => (0..n)
                .map(|i| T::calculate_l2_norm(&data[i * dim..(i + 1) * dim]))
                .collect(),
            Dist::Cosine => data_norms.clone(),
        };
        kmeans_parallel_init(data, &init_norms, dim, n, n_centroids, metric, seed)
    };

    let mut centroid_norms: Vec<T> = match metric {
        Dist::Euclidean => (0..n_centroids)
            .map(|i| {
                let c = &centroids[i * dim..(i + 1) * dim];
                T::dot_simd(c, c)
            })
            .collect(),
        Dist::Cosine => (0..n_centroids)
            .map(|i| T::calculate_l2_norm(&centroids[i * dim..(i + 1) * dim]))
            .collect(),
    };

    if verbose {
        println!("  Running Lloyd's iterations");
    }

    match metric {
        _ if dim < GEMM_DIM_THRESHOLD => {
            if verbose {
                println!(
                    "    (direct SIMD assignment, dim={} below GEMM threshold)",
                    dim
                );
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
        }
        Dist::Euclidean => {
            if verbose {
                println!("    (Hamerly's bounds + GEMM assignment)");
            }
            hamerly_lloyd(
                data,
                &data_norms,
                dim,
                n,
                &mut centroids,
                &mut centroid_norms,
                n_centroids,
                max_iters,
                verbose,
            );
        }
        Dist::Cosine => {
            if verbose {
                println!("    (GEMM assignment, no Hamerly -- cosine lacks triangle inequality)");
            }
            gemm_lloyd_cosine(
                data,
                &data_norms,
                dim,
                n,
                &mut centroids,
                &mut centroid_norms,
                n_centroids,
                max_iters,
                verbose,
            );
        }
    }

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
