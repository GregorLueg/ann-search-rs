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

use crate::errors::AnnSearchErrors;
use crate::prelude::AnnSearchFloat;
use crate::utils::dist::*;
use crate::utils::Dist;

////////////
// Consts //
////////////

/// Default shift for [`SoarRule::Shifted`] under squared Euclidean.
///
/// The derived shift is `eps * gamma`, with `eps` the local kNN radius and
/// `gamma` the mean alignment over the failure cap, neither of which is known
/// at build time. Expressed as a multiple of the primary residual it lands in
/// the low fractions, and 0.5 is the middle of the 0.2-1.0 band the gridsearch
/// sweeps.
pub const DEFAULT_SHIFT_MU: f64 = 0.5;

/// Default weight for [`SoarRule::Orthogonal`] under cosine.
///
/// The SOAR paper reports 1.0 on Glove-1M and 1.5 at billion scale. Datasets
/// here sit far closer to the former.
pub const DEFAULT_ORTHOGONAL_LAMBDA: f64 = 1.0;

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
                    Dist::SquaredEuclidean => euclidean_distance_static(query_vec, cent),
                    Dist::Cosine => {
                        let c_norm = &self.centroids_norm()[c];
                        cosine_distance_static_norm(query_vec, cent, &query_norm, c_norm)
                    }
                    Dist::Manhattan => {
                        unreachable!()
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
                    Dist::SquaredEuclidean => T::euclidean_simd(query_vec, cent),
                    Dist::Manhattan => {
                        unreachable!()
                    }
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

/// Minimum number of centroids at which Hamerly's pruning beats plain Lloyd's
/// on the SIMD path. Below this, per-iteration overhead of computing s[c] and
/// updating bounds outweighs the saved distance work.
const SIMD_HAMERLY_K_THRESHOLD: usize = 100;

/// Minimum number of dimensions at which Hamerly's pruning beats plain Lloyd's
/// on the SIMD path. Below this, per-iteration overhead of computing s[c] and
/// updating bounds outweighs the saved distance work.
const SIMD_HAMERLY_DIM_MIN: usize = 64;

/// Fraction of the average cluster size below which a centroid is reseeded by
/// [`adjust_centers`].
///
/// Matches the `balancing_threshold` default of RAFT's balanced k-means: only
/// clusters that have collapsed to under a quarter of their fair share get
/// touched, so a merely uneven partition is left alone. Raising it balances
/// harder at the cost of dragging centroids out of genuinely dense regions.
///
/// Shared with the GPU k-means, which runs the same policy on device.
pub(crate) const BALANCE_THRESHOLD: f64 = 0.25;

/// Weight the existing centroid keeps when pulled toward a donor point.
///
/// The reseeded centroid is `(c * w + x) / (w + 1)` with `w = min(count, this)`,
/// so an empty cluster (`count == 0`) jumps onto the donor outright while a
/// merely small one edges toward it over successive iterations. Matches RAFT's
/// `balancing_pullback`.
pub(crate) const BALANCE_PULLBACK: usize = 5;

/// Stride used to walk the dataset when hunting for a donor point.
///
/// A prime, so the walk visits every index before repeating for any `n` that is
/// not a multiple of it. Striding rather than scanning sequentially stops
/// several starved centroids from all landing in the same neighbourhood.
pub(crate) const BALANCE_DONOR_STRIDE: usize = 715_827_883;

///////////////////
// Enums, Params //
///////////////////

/// Initialisation of initial k-means centroids
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMeansInit {
    /// Random path -> useful on very large data sets
    Random,
    /// Uses the KMeansParallel path, better initial centroids but slower
    /// initialisation
    KMeansParallel,
}

/// Computation path for the k-means clustering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LloydPath {
    /// Uses Hamerly's bounds to reduce computations and GEMM acceleration.
    /// Ideal on large N and large dimensionality data sets with Euclidean
    /// distance.
    HamerlyGemm,
    /// Uses Hamerly's bounds to reduce computations and SIMD acceleration.
    /// Ideal on large N and moderate dimensionality data sets with Euclidean
    /// distance.
    HamerlySimd,
    /// Uses GEMM acceleration and standard Lloyd's
    GemmLloyd,
    /// ParallelLloyd uses SIMD
    ParallelLloyd,
}

/// Resolve init strategy: if `None`, pick based on `n_centroids`.
///
/// ### Params
///
/// * `init` - The Option of the [KMeansInit]
/// * `n_centroids` - The number of centroids. If `init` is None, it will use
///   this to generate a good default based on heuristics.
///
/// ### Returns
///
/// The [KMeansInit]
fn resolve_init(init: Option<KMeansInit>, n_centroids: usize) -> KMeansInit {
    init.unwrap_or({
        if n_centroids > 200 {
            KMeansInit::Random
        } else {
            KMeansInit::KMeansParallel
        }
    })
}

/// Resolve Lloyd's path
///
/// If a Hamerly variant is requested with Cosine (no triangle inequality), fall
/// back to a compatible non-Hamerly path: `HamerlyGemm` to `GemmLloyd`,
/// `HamerlySimd` to `ParallelLloyd`.
///
/// ### Params
///
/// * `path` - The Option of the [LloydPath].
/// * `dim` - The dimensionality of the data set. Will be used if `None` is
///   provided for `path`.
/// * `n_centroids` - The number of requested centroids. Will be used if `None`
///   is provided for `path`.
/// * `metric` - The distance metric, see [Dist].
///
/// ### Returns
///
/// The [LloydPath].
fn resolve_path(
    path: Option<LloydPath>,
    dim: usize,
    n_centroids: usize,
    metric: &Dist,
) -> LloydPath {
    let chosen = path.unwrap_or_else(|| match metric {
        Dist::SquaredEuclidean if dim >= GEMM_DIM_THRESHOLD => LloydPath::HamerlyGemm,
        Dist::SquaredEuclidean
            if (SIMD_HAMERLY_DIM_MIN..GEMM_DIM_THRESHOLD).contains(&dim)
                && n_centroids >= SIMD_HAMERLY_K_THRESHOLD =>
        {
            LloydPath::HamerlySimd
        }
        Dist::Cosine if dim >= GEMM_DIM_THRESHOLD => LloydPath::GemmLloyd,
        _ => LloydPath::ParallelLloyd,
    });

    match (chosen, metric) {
        (LloydPath::HamerlyGemm, Dist::Cosine) => LloydPath::GemmLloyd,
        (LloydPath::HamerlySimd, Dist::Cosine) => LloydPath::ParallelLloyd,
        _ => chosen,
    }
}

#[derive(Debug, Clone, Copy)]
/// K-means clustering parameters to enable tighter control over the method
pub struct KMeansTrainingParams {
    /// Number of iterations to run the clustering algorithm for
    pub iters: usize,
    /// Optional [KMeansInit] for tighter control over the initialisation
    pub init: Option<KMeansInit>,
    /// Optional [LloydPath] to control
    pub path: Option<LloydPath>,
    /// Reseed starved centroids each iteration via [`adjust_centers`].
    ///
    /// Off by default: it changes the partition, so turning it on shifts the
    /// results of every index built on top of this k-means.
    pub balanced: bool,
}

impl KMeansTrainingParams {
    /// Generate a new instance of self
    ///
    /// Balancing is off. Chain [`KMeansTrainingParams::with_balancing`] to turn
    /// it on.
    ///
    /// ### Params
    ///
    /// * `iters` - Number of iterations to run the clustering algorithm for
    /// * `init` - Optional specification of the [KMeansInit] for better
    ///   control
    /// * `path` - The Lloyd's path you want to use, see [LloydPath] for better
    ///   control
    ///
    /// ### Returns
    ///
    /// [KMeansTrainingParams]
    pub fn new(iters: usize, init: Option<KMeansInit>, path: Option<LloydPath>) -> Self {
        Self {
            iters,
            init,
            path,
            balanced: false,
        }
    }

    /// Turn size balancing on or off
    ///
    /// See [`adjust_centers`] for what balancing does and what it does not
    /// promise. Worth setting when the cluster sizes drive a downstream memory
    /// footprint rather than only quantisation error.
    ///
    /// ### Params
    ///
    /// * `balanced` - Whether to reseed starved centroids each iteration
    ///
    /// ### Returns
    ///
    /// `self` with the flag set, for chaining off `new` or `default`.
    pub fn with_balancing(mut self, balanced: bool) -> Self {
        self.balanced = balanced;
        self
    }
}

/// Default implementation for CagraGpuSearchParams
impl Default for KMeansTrainingParams {
    fn default() -> Self {
        Self::new(30, None, None)
    }
}

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
        Dist::SquaredEuclidean => {
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
        Dist::Manhattan => {
            unreachable!()
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
            Dist::SquaredEuclidean => {
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
            Dist::Manhattan => {
                unreachable!()
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
pub fn kmeans_parallel_init<T>(
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
                        Dist::SquaredEuclidean => {
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
                        Dist::Manhattan => {
                            unreachable!()
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
                Dist::SquaredEuclidean => {
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
                Dist::Manhattan => {
                    unreachable!()
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

/// Reseed starved centroids from points in over-full clusters
///
/// The balancing step of RAFT's balanced k-means. Plain Lloyd's has no notion
/// of cluster size, so on skewed data it happily produces one cluster holding
/// most of the dataset and several holding almost nothing. That is fine for
/// pure quantisation error but ruinous when the cluster sizes decide a memory
/// footprint, which is exactly the case for the batched NN-Descent build where
/// a cluster has to fit one GPU binding.
///
/// Each centroid whose cluster has fallen below `BALANCE_THRESHOLD` of the
/// average size is pulled toward a point drawn from a cluster that is above
/// average, weighted by `BALANCE_PULLBACK`. An empty cluster has weight zero
/// and therefore jumps onto the donor outright, which also gives empty-cluster
/// reseeding for free -- otherwise `update_centroids` leaves a dead centroid
/// parked where it was forever.
///
/// This is a soft guarantee: it steers the partition toward balance over
/// successive iterations, it does not cap any cluster's size. Callers that need
/// a hard bound must still enforce one.
///
/// ### Params
///
/// * `centroids` - In/out: centroids, flattened row-major `k * dim`
/// * `dim` - Embedding dimensions
/// * `k` - Number of centroids
/// * `data` - All vectors, flattened row-major `n * dim`
/// * `n` - Number of vectors
/// * `assignments` - Cluster assignment per vector, length `n`
/// * `counts` - Points per cluster, length `k`
/// * `seed` - Seeds the offset the donor walk starts from
///
/// ### Returns
///
/// Number of centroids that were reseeded. Zero means the partition was
/// already balanced enough to leave alone.
///
/// ### References
///
/// Adapted from `adjust_centers` in RAFT's `kmeans_balanced`
/// (NVIDIA RAPIDS), as used by cuVS IVF training.
#[allow(clippy::too_many_arguments)]
pub fn adjust_centers<T>(
    centroids: &mut [T],
    dim: usize,
    k: usize,
    data: &[T],
    n: usize,
    assignments: &[usize],
    counts: &[usize],
    seed: usize,
) -> usize
where
    T: Float,
{
    if k == 0 || n == 0 {
        return 0;
    }

    let average = n as f64 / k as f64;
    let floor = average * BALANCE_THRESHOLD;
    let mut adjusted = 0usize;
    // Carried across clusters so two starved centroids never start their walk
    // from the same index and pick the same donor.
    let mut cursor = seed % n;

    for c in 0..k {
        if (counts[c] as f64) > floor {
            continue;
        }

        let mut donor = None;
        for _ in 0..n {
            cursor = (cursor + BALANCE_DONOR_STRIDE) % n;
            let owner = assignments[cursor];
            if owner != c && (counts[owner] as f64) > average {
                donor = Some(cursor);
                break;
            }
        }

        let Some(j) = donor else { continue };

        let w = T::from(counts[c].min(BALANCE_PULLBACK)).unwrap();
        let denom = w + T::one();
        let (c_off, j_off) = (c * dim, j * dim);
        for d in 0..dim {
            centroids[c_off + d] = (centroids[c_off + d] * w + data[j_off + d]) / denom;
        }
        adjusted += 1;
    }

    adjusted
}

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
/// * `balanced` - Run [`adjust_centers`] before recomputing the norms
/// * `seed` - Seeds the donor walk when `balanced` is set
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
    balanced: bool,
    seed: usize,
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
    }

    // Balancing has to land between the means and the norms: it moves
    // centroids, so norms computed before it would be stale.
    if balanced {
        adjust_centers(centroids, dim, k, data, n, assignments, &counts, seed);
    }

    for c in 0..k {
        let cent = &centroids[c * dim..(c + 1) * dim];
        centroid_norms[c] = match metric {
            Dist::SquaredEuclidean => T::dot_simd(cent, cent), // ||c||^2
            Dist::Cosine => T::calculate_l2_norm(cent),        // ||c||
            Dist::Manhattan => {
                unreachable!()
            }
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
pub fn compute_centroid_drift<T>(
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
        // needs the square root here
        deltas[c] = euclidean_distance_static(old, new).sqrt();
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

///////////////////////////////////
// Hamerly's Lloyd's (Euclidean) //
///////////////////////////////////

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
/// * `balanced` - Reseed starved centroids each iteration via [`adjust_centers`]
/// * `seed` - Seeds the donor walk when `balanced` is set
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
    balanced: bool,
    seed: usize,
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
        &Dist::SquaredEuclidean,
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
            &Dist::SquaredEuclidean,
            balanced,
            seed.wrapping_add(iter),
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
            &Dist::SquaredEuclidean,
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

///////////////////////////////////
// GEMM-only Lloyd's (Euclidean) //
///////////////////////////////////

/// Plain Lloyd's k-means using GEMM assignment.
///
/// Runs full GEMM reassignment every iteration and converges when no
/// assignments change. Works for both Euclidean and Cosine; for Cosine this is
/// the GEMM path of choice because Hamerly's bound-based pruning is not
/// applicable (no triangle inequality).
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
/// * `balanced` - Reseed starved centroids each iteration via [`adjust_centers`]
/// * `seed` - Seeds the donor walk when `balanced` is set
/// * `verbose` - Print convergence diagnostics
#[allow(clippy::too_many_arguments)]
fn gemm_lloyd<T>(
    data: &[T],
    data_norms: &[T],
    dim: usize,
    n: usize,
    centroids: &mut [T],
    centroid_norms: &mut [T],
    k: usize,
    metric: &Dist,
    max_iters: usize,
    balanced: bool,
    seed: usize,
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
            metric,
            &mut assignments,
            &mut upper,
            &mut lower,
        );

        let changed: usize = assignments
            .par_iter()
            .zip(prev_assignments.par_iter())
            .filter(|(a, b)| a != b)
            .count();

        let change_floor: usize = (n / 10_000).max(1);

        if changed <= change_floor {
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
            metric,
            balanced,
            seed.wrapping_add(iter),
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
/// The only Lloyd path that inlines its own centroid update rather than
/// calling [`update_centroids`], so the balancing hook is repeated in the body
/// rather than inherited. It also tests convergence *before* the update, which
/// is why the test additionally requires that the previous iteration reseeded
/// nothing.
///
/// ### Params
///
/// * `data` - Training vectors (flattened)
/// * `data_norms` - The precomputed norms of the data
/// * `dim` - Embedding dimensions
/// * `n` - Number of training vectors
/// * `centroids` - Current centroids (modified in-place)
/// * `centroid_norms` - Current centroid norms (modified in-place); for cosine
///   these are refreshed for every centroid each iteration, including the ones
///   whose cluster came out empty
/// * `k` - Number of clusters
/// * `metric` - Distance metric
/// * `max_iters` - Maximum iterations
/// * `balanced` - Reseed starved centroids each iteration via [`adjust_centers`]
/// * `seed` - Seeds the donor walk when `balanced` is set
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
    balanced: bool,
    seed: usize,
    verbose: bool,
) where
    T: Float + Send + Sync + SimdDistance + ComplexField,
{
    let mut prev_assignments: Vec<usize> = vec![usize::MAX; n];
    let num_threads = rayon::current_num_threads();
    let chunk_size = (n + num_threads - 1) / num_threads.max(1);
    // Reseeds performed by the previous iteration. This path tests convergence
    // before the centroid update, so a starved cluster that only gets rescued
    // in the update half would otherwise never be rescued at all: assignments
    // settle, the loop breaks, and the balancing never runs. Staying in until
    // balancing has nothing left to do closes that. Pinned at zero whenever
    // `balanced` is off, so the unbalanced path keeps its exact old behaviour.
    let mut last_adjusted = 0usize;

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

        let change_floor: usize = (n / 10_000).max(1);

        if changed <= change_floor && last_adjusted == 0 {
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
            }
        }

        // This path inlines its own update rather than calling
        // `update_centroids`, so the balancing hook is repeated here. It must
        // land before the norms for the same reason: it moves centroids.
        if balanced {
            last_adjusted = adjust_centers(
                centroids,
                dim,
                k,
                data,
                n,
                &assignments,
                &counts,
                seed.wrapping_add(iter),
            );
        }

        if matches!(metric, Dist::Cosine) {
            for c in 0..k {
                let cent = &centroids[c * dim..(c + 1) * dim];
                centroid_norms[c] = T::calculate_l2_norm(cent);
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

////////////////////
// Hamerly's SIMD //
////////////////////

/// Find best and second-best centroid for a single vector via direct SIMD
///
/// Tracks squared distances in the inner loop and converts to actual
/// distances at the end. Used by both the full-assignment and dirty-point
/// paths in Hamerly's SIMD implementation.
///
/// ### Params
///
/// * `vec` - Query vector slice (dim elements)
/// * `centroids` - All centroids, flattened row-major
/// * `dim` - Embedding dimensions
/// * `k` - Number of centroids
///
/// ### Returns
///
/// Tuple of (best centroid index, distance to best, distance to second-best)
#[inline]
fn assign_one_with_bounds_simd<T>(vec: &[T], centroids: &[T], dim: usize, k: usize) -> (usize, T, T)
where
    T: Float + SimdDistance,
{
    let mut best_c = 0;
    let mut best_sq = T::infinity();
    let mut second_sq = T::infinity();

    for c in 0..k {
        let cent = &centroids[c * dim..(c + 1) * dim];
        let dist_sq = euclidean_distance_static(vec, cent);
        if dist_sq < best_sq {
            second_sq = best_sq;
            best_sq = dist_sq;
            best_c = c;
        } else if dist_sq < second_sq {
            second_sq = dist_sq;
        }
    }

    (best_c, best_sq.sqrt(), second_sq.sqrt())
}

/// Full nearest-centroid assignment via direct SIMD, with Hamerly bounds
///
/// SIMD analogue of `gemm_assign_full` for the Euclidean Hamerly path. Iterates
/// over all n vectors in parallel; for each, scans all k centroids and records
/// the closest and second-closest distances.
///
/// ### Params
///
/// * `data` - All vectors, flattened row-major
/// * `dim` - Embedding dimensions
/// * `centroids` - All centroids, flattened row-major
/// * `k` - Number of centroids
/// * `assignments` - Output: nearest centroid index per vector
/// * `upper_bounds` - Output: distance to nearest centroid per vector
/// * `lower_bounds` - Output: distance to second-nearest centroid per vector
fn simd_assign_full_with_bounds<T>(
    data: &[T],
    dim: usize,
    centroids: &[T],
    k: usize,
    assignments: &mut [usize],
    upper_bounds: &mut [T],
    lower_bounds: &mut [T],
) where
    T: Float + Send + Sync + SimdDistance,
{
    assignments
        .par_iter_mut()
        .zip(upper_bounds.par_iter_mut())
        .zip(lower_bounds.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((assign, upper), lower))| {
            let vec = &data[i * dim..(i + 1) * dim];
            let (best_c, best_dist, second_dist) =
                assign_one_with_bounds_simd(vec, centroids, dim, k);
            *assign = best_c;
            *upper = best_dist;
            *lower = second_dist;
        });
}

/// Reassign a subset of "dirty" points via direct SIMD
///
/// SIMD analogue of `gemm_reassign_dirty`. No gather/scatter is needed since
/// the distance kernel works directly on per-point slices. Distances are
/// computed in parallel and scattered back sequentially.
///
/// ### Params
///
/// * `data` - All vectors, flattened row-major
/// * `dim` - Embedding dimensions
/// * `centroids` - All centroids, flattened row-major
/// * `k` - Number of centroids
/// * `dirty` - Indices of vectors requiring reassignment
/// * `assignments` - In/out: nearest centroid index per vector
/// * `upper_bounds` - In/out: distance to nearest centroid per vector
/// * `lower_bounds` - In/out: distance to second-nearest centroid per vector
#[allow(clippy::too_many_arguments)]
fn simd_reassign_dirty<T>(
    data: &[T],
    dim: usize,
    centroids: &[T],
    k: usize,
    dirty: &[usize],
    assignments: &mut [usize],
    upper_bounds: &mut [T],
    lower_bounds: &mut [T],
) where
    T: Float + Send + Sync + SimdDistance,
{
    let updates: Vec<(usize, T, T)> = dirty
        .par_iter()
        .map(|&i| {
            let vec = &data[i * dim..(i + 1) * dim];
            assign_one_with_bounds_simd(vec, centroids, dim, k)
        })
        .collect();

    for (&i, (best_c, best_dist, second_dist)) in dirty.iter().zip(updates) {
        assignments[i] = best_c;
        upper_bounds[i] = best_dist;
        lower_bounds[i] = second_dist;
    }
}

/// Compute s[c] = 0.5 * min_{c' != c} dist(c, c') via direct SIMD
///
/// SIMD analogue of `compute_half_min_centroid_dists`. Computes k^2
/// pairwise centroid distances directly rather than via GEMM. Cheap
/// for moderate k (~10^4 evaluations at k=100).
///
/// ### Params
///
/// * `centroids` - All centroids, flattened row-major
/// * `dim` - Embedding dimensions
/// * `k` - Number of centroids
///
/// ### Returns
///
/// Vector of length k with half-minimum inter-centroid distances
fn compute_half_min_centroid_dists_simd<T>(centroids: &[T], dim: usize, k: usize) -> Vec<T>
where
    T: Float + Send + Sync + SimdDistance,
{
    let half = T::one() / (T::one() + T::one());

    (0..k)
        .into_par_iter()
        .map(|i| {
            let cent_i = &centroids[i * dim..(i + 1) * dim];
            let mut min_sq = T::infinity();
            for j in 0..k {
                if i == j {
                    continue;
                }
                let cent_j = &centroids[j * dim..(j + 1) * dim];
                let dist_sq = euclidean_distance_static(cent_i, cent_j);
                if dist_sq < min_sq {
                    min_sq = dist_sq;
                }
            }
            min_sq.sqrt() * half
        })
        .collect()
}

/// Hamerly's accelerated k-means for Euclidean distance via SIMD
///
/// SIMD-only variant of Hamerly's bound-based k-means. Used when dim is
/// below the GEMM threshold but k is large enough for bound pruning to
/// pay off. Initial full assignment runs via SIMD, then iteratively only
/// those points whose bounds have become loose are reassigned. Bound
/// updates and the bound-check / dirty-collection step run in parallel
/// via Rayon.
///
/// ### Params
///
/// * `data` - All vectors, flattened row-major
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `centroids` - In/out: centroids, flattened row-major
/// * `centroid_norms` - In/out: per-centroid norms (kept consistent for
///   `update_centroids`; not used for assignment on the SIMD path)
/// * `k` - Number of centroids
/// * `max_iters` - Maximum number of Lloyd's iterations
/// * `balanced` - Reseed starved centroids each iteration via [`adjust_centers`]
/// * `seed` - Seeds the donor walk when `balanced` is set
/// * `verbose` - Print convergence diagnostics
#[allow(clippy::too_many_arguments)]
fn hamerly_lloyd_simd<T>(
    data: &[T],
    dim: usize,
    n: usize,
    centroids: &mut [T],
    centroid_norms: &mut [T],
    k: usize,
    max_iters: usize,
    balanced: bool,
    seed: usize,
    verbose: bool,
) where
    T: Float + Send + Sync + SimdDistance + FromPrimitive,
{
    let mut assignments = vec![0usize; n];
    let mut upper = vec![T::infinity(); n];
    let mut lower = vec![T::zero(); n];
    let mut old_centroids = vec![T::zero(); k * dim];
    let mut deltas = vec![T::zero(); k];

    simd_assign_full_with_bounds(
        data,
        dim,
        centroids,
        k,
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
            centroid_norms,
            k,
            &Dist::SquaredEuclidean,
            balanced,
            seed.wrapping_add(iter),
        );

        compute_centroid_drift(&old_centroids, centroids, dim, k, &mut deltas);
        let (max_delta, second_max_delta, max_delta_idx) = top_two_deltas(&deltas);

        if max_delta <= T::from_f64(1e-5).unwrap() {
            if verbose {
                println!("    Converged at iteration {}", iter + 1);
            }
            break;
        }

        // Parallel bound update: shift upper by the drift of the assigned
        // centroid; loosen lower by the largest drift among other centroids
        upper
            .par_iter_mut()
            .zip(lower.par_iter_mut())
            .zip(assignments.par_iter())
            .for_each(|((u, l), &a)| {
                *u = *u + deltas[a];
                let other_max = if a == max_delta_idx {
                    second_max_delta
                } else {
                    max_delta
                };
                *l = (*l - other_max).max(T::zero());
            });

        let s = compute_half_min_centroid_dists_simd(centroids, dim, k);

        // Parallel bound check: tighten loose upper bounds against the
        // assigned centroid; collect surviving points into the dirty list
        let dirty: Vec<usize> = upper
            .par_iter_mut()
            .zip(lower.par_iter())
            .zip(assignments.par_iter())
            .enumerate()
            .filter_map(|(i, ((u, &l), &a))| {
                let s_a = s[a];
                let m = if s_a > l { s_a } else { l };
                if *u <= m {
                    return None;
                }
                let vec = &data[i * dim..(i + 1) * dim];
                let cent = &centroids[a * dim..(a + 1) * dim];
                *u = euclidean_distance_static(vec, cent).sqrt();
                if *u > m {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if dirty.is_empty() {
            if verbose {
                println!("    Converged at iteration {} (bounds tight)", iter + 1);
            }
            break;
        }

        simd_reassign_dirty(
            data,
            dim,
            centroids,
            k,
            &dirty,
            &mut assignments,
            &mut upper,
            &mut lower,
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
        Dist::SquaredEuclidean => (0..n)
            .map(|i| {
                let v = &data[i * dim..(i + 1) * dim];
                T::dot_simd(v, v)
            })
            .collect(),
        Dist::Cosine => (0..n)
            .map(|i| T::calculate_l2_norm(&data[i * dim..(i + 1) * dim]))
            .collect(),
        Dist::Manhattan => {
            unreachable!("Manhatten is not reachable for GEMM.")
        }
    };
    let centroid_norms: Vec<T> = match metric {
        Dist::SquaredEuclidean => (0..k)
            .map(|c| {
                let cent = &centroids[c * dim..(c + 1) * dim];
                T::dot_simd(cent, cent)
            })
            .collect(),
        Dist::Cosine => (0..k)
            .map(|c| T::calculate_l2_norm(&centroids[c * dim..(c + 1) * dim]))
            .collect(),
        Dist::Manhattan => {
            unreachable!("Manhatten is not reachable for GEMM.")
        }
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
        Dist::SquaredEuclidean => (0..k)
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
        Dist::Manhattan => {
            unreachable!("Manhatten is not reachable for direct assign.")
        }
    };

    match metric {
        Dist::SquaredEuclidean => (0..n)
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
        Dist::Manhattan => {
            unreachable!("Manhatten is not reachable for direct assign.")
        }
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

/// Assign every point to its `m` closest centroids
///
/// Multi-assignment counterpart to [`assign_all_parallel`], for partitioning
/// schemes that want overlapping cells. The batched NN-Descent build uses
/// `m = 2`: giving every point membership of its two nearest clusters means a
/// point near a boundary still has most of its true neighbours inside at least
/// one of the batches it belongs to, which is what keeps recall up when the
/// graph is built per batch and merged.
///
/// Deliberately a plain scan rather than dispatching to the GEMM path like
/// [`assign_all_parallel`] does: the caller here partitions into a handful of
/// clusters, not `sqrt(n)` of them, so the tiled GEMM setup costs more than the
/// `O(n * k * dim)` it saves. It is also deliberately host-side, because the
/// whole point of the batched build is that the dataset need not fit on the
/// device.
///
/// ### Params
///
/// * `data` - All vectors, flattened row-major `n * dim`
/// * `data_norms` - L2 norms per point; only read for cosine, so it may be
///   empty for any other metric
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `centroids` - Centroids, flattened row-major `k * dim`
/// * `centroid_norms` - L2 norms per centroid; only read for cosine
/// * `k` - Number of centroids
/// * `m` - Clusters each point joins; clamped into `1..=k`
/// * `metric` - Distance metric; `Manhattan` is not supported
///
/// ### Returns
///
/// `n * m` cluster ids, row `i` at `[i * m, (i + 1) * m)`, ascending by
/// distance so column 0 matches what [`assign_all_parallel`] would give.
#[allow(clippy::too_many_arguments)]
pub fn assign_all_parallel_top_m<T>(
    data: &[T],
    data_norms: &[T],
    dim: usize,
    n: usize,
    centroids: &[T],
    centroid_norms: &[T],
    k: usize,
    m: usize,
    metric: &Dist,
) -> Vec<u32>
where
    T: Float + Send + Sync + SimdDistance,
{
    let m = m.min(k).max(1);
    let mut out = vec![0u32; n * m];

    out.par_chunks_mut(m).enumerate().for_each(|(i, slot)| {
        let vec = &data[i * dim..(i + 1) * dim];
        let mut best_d = vec![T::infinity(); m];
        let mut best_c: Vec<u32> = (0..m as u32).collect();

        for c in 0..k {
            let cent = &centroids[c * dim..(c + 1) * dim];
            let d = match metric {
                Dist::Cosine => {
                    let denom = data_norms[i] * centroid_norms[c];
                    if denom > T::zero() {
                        T::one() - T::dot_simd(vec, cent) / denom
                    } else {
                        T::one()
                    }
                }
                _ => T::euclidean_simd(vec, cent),
            };

            if d >= best_d[m - 1] {
                continue;
            }
            let mut pos = m - 1;
            while pos > 0 && best_d[pos - 1] > d {
                best_d[pos] = best_d[pos - 1];
                best_c[pos] = best_c[pos - 1];
                pos -= 1;
            }
            best_d[pos] = d;
            best_c[pos] = c as u32;
        }

        slot.copy_from_slice(&best_c);
    });

    out
}

/// Invert a multi-assignment into per-cluster member lists
///
/// Counting-sort inverse of [`assign_all_parallel_top_m`], the multi-assignment
/// analogue of [`build_csr_layout`]. Each point appears once per cluster it
/// joined, so the member list holds `n * m` entries rather than `n`.
///
/// ### Params
///
/// * `assignments` - `n * m` cluster ids from [`assign_all_parallel_top_m`]
/// * `n` - Number of vectors
/// * `m` - Clusters each point joined
/// * `k` - Number of clusters
///
/// ### Returns
///
/// `(members, offsets)` where cluster `c` owns
/// `members[offsets[c]..offsets[c + 1]]` and `offsets` has `k + 1` entries.
/// Members within a cluster are in ascending point order.
///
/// ### Panics
///
/// If `assignments` is shorter than `n * m`. `m` must be the same value
/// [`assign_all_parallel_top_m`] clamped to, not the caller's requested one.
pub fn invert_assignments_csr(
    assignments: &[u32],
    n: usize,
    m: usize,
    k: usize,
) -> (Vec<u32>, Vec<usize>) {
    let mut offsets = vec![0usize; k + 1];
    for &c in &assignments[..n * m] {
        offsets[c as usize + 1] += 1;
    }
    for c in 0..k {
        offsets[c + 1] += offsets[c];
    }

    let mut cursor = offsets.clone();
    let mut members = vec![0u32; n * m];
    for i in 0..n {
        for j in 0..m {
            let c = assignments[i * m + j] as usize;
            members[cursor[c]] = i as u32;
            cursor[c] += 1;
        }
    }

    (members, offsets)
}

//////////
// SOAR //
//////////

/// Rule picking the second cluster a point joins under SOAR spilling
///
/// Spilling puts every point in two inverted lists so a query that misses the
/// first still has a chance at the second. Which second list is chosen is the
/// whole game: the useful one is the list that fails on *different* queries
/// than the first does, not simply the next-closest one.
///
/// The three arms differ only in the cost minimised over candidate centroids,
/// writing `r' = x - c` for the candidate residual and `r_1 = x - c_1` for the
/// primary one. For `Cosine` all of this happens on the unit sphere, since that
/// is the geometry the routing decision lives in.
///
/// ### References
///
/// Sun, Simcha, Simcha, Chern & Guo, arXiv:2404.00774, 2024 (SOAR)
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub enum SoarRule {
    /// Plain second-nearest centroid, minimising `||r'||^2`
    Nearest,
    /// Nearest centroid to the shifted point `x + mu * (x - c_1)`
    ///
    /// Derived for squared Euclidean. Conditioning on the queries that make the
    /// primary cell look bad gives a *signed* penalty
    /// `2*eps*gamma*<r_hat_1, r'>` rather than the published squared one, and
    /// completing the square turns the whole objective back into a plain
    /// nearest-centroid lookup against a point pushed away from its own
    /// centroid. `mu = 0` degenerates to [`SoarRule::Nearest`]. Negative values
    /// are clamped to zero.
    Shifted {
        /// Shift as a multiple of the primary residual. Sensible range 0.2-1.0.
        mu: f64,
    },
    /// Published SOAR loss, `||r'||^2 + lambda * (r' . r_hat_1)^2`
    ///
    /// Derived for MIPS under a uniform query distribution on the unit sphere,
    /// so it applies unchanged to cosine, where the ranking is the same
    /// problem.
    Orthogonal {
        /// Weight on the parallel component of the candidate residual.
        lambda: f64,
    },
}

/// Reciprocal that yields zero instead of infinity on a zero input
///
/// ### Params
///
/// * `x` - Value to invert
///
/// ### Returns
/// `1 / x`, or zero when `x` is not strictly positive.
#[inline(always)]
fn safe_recip<T: Float>(x: T) -> T {
    if x > T::zero() {
        x.recip()
    } else {
        T::zero()
    }
}

/// Nearest centroid to a point, skipping one cluster
///
/// Repair path for the rare case where a spilling rule lands back on the
/// primary cluster, which would waste the point's second list slot.
///
/// ### Params
///
/// * `point` - The point, `dim` elements
/// * `point_norm` - Its L2 norm; only read for cosine
/// * `centroids` - Centroids, flattened row-major `k * dim`
/// * `centroid_norms` - L2 norms per centroid; only read for cosine
/// * `dim` - Embedding dimensions
/// * `k` - Number of centroids
/// * `exclude` - Cluster to skip
/// * `metric` - Distance metric
///
/// ### Returns
///
/// Nearest cluster id other than `exclude`, or `exclude` itself when `k < 2`.
/// Seeded with a valid non-excluded id rather than with `exclude`, so a scan in
/// which every distance comes back non-finite still returns a distinct cluster.
#[allow(clippy::too_many_arguments)]
fn nearest_excluding<T>(
    point: &[T],
    point_norm: T,
    centroids: &[T],
    centroid_norms: &[T],
    dim: usize,
    k: usize,
    exclude: usize,
    metric: &Dist,
) -> u32
where
    T: AnnSearchFloat,
{
    if k < 2 {
        return exclude as u32;
    }
    let mut best = if exclude == 0 { 1u32 } else { 0u32 };
    let mut best_d = T::infinity();
    for c in 0..k {
        if c == exclude {
            continue;
        }
        let cent = &centroids[c * dim..(c + 1) * dim];
        let d = match metric {
            Dist::Cosine => {
                let denom = point_norm * centroid_norms[c];
                if denom > T::zero() {
                    T::one() - T::dot_simd(point, cent) / denom
                } else {
                    T::one()
                }
            }
            _ => T::euclidean_simd(point, cent),
        };
        if d < best_d {
            best_d = d;
            best = c as u32;
        }
    }
    best
}

/// Assign every point a second cluster under a SOAR spilling rule
///
/// Companion to [`assign_all_parallel`], which produces the primary assignment
/// this builds on. Returns one extra cluster per point, so the two together put
/// each point in exactly two inverted lists.
///
/// [`SoarRule::Shifted`] deliberately reuses [`assign_all_parallel`] on a
/// shifted copy of the data rather than introducing a scoring kernel of its
/// own, which means it inherits the tiled GEMM path for free. The other two
/// arms are plain `O(n * k * dim)` scans with no GEMM path.
///
/// ### Params
///
/// * `data` - All vectors, flattened row-major `n * dim`
/// * `data_norms` - L2 norms per point; only read for cosine
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `centroids` - Centroids, flattened row-major `k * dim`
/// * `centroid_norms` - L2 norms per centroid; only read for cosine
/// * `k` - Number of centroids
/// * `primary` - Primary assignment from [`assign_all_parallel`], `n` entries
/// * `rule` - Which secondary-assignment cost to minimise
/// * `metric` - Distance metric; `Manhattan` is not supported
///
/// ### Returns
///
/// `n` cluster ids, guaranteed different from `primary` whenever `k >= 2`. With
/// `k < 2` there is no second cluster to give, so `primary` is echoed back and
/// the caller ends up with a plain non-spilled index.
#[allow(clippy::too_many_arguments)]
pub fn assign_secondary_soar<T>(
    data: &[T],
    data_norms: &[T],
    dim: usize,
    n: usize,
    centroids: &[T],
    centroid_norms: &[T],
    k: usize,
    primary: &[usize],
    rule: &SoarRule,
    metric: &Dist,
) -> Vec<u32>
where
    T: AnnSearchFloat,
{
    if k < 2 {
        return primary.iter().map(|&c| c as u32).collect();
    }

    let cosine = matches!(metric, Dist::Cosine);

    match rule {
        SoarRule::Nearest => {
            let top = assign_all_parallel_top_m(
                data,
                data_norms,
                dim,
                n,
                centroids,
                centroid_norms,
                k,
                2,
                metric,
            );

            let mut out: Vec<u32> = (0..n)
                .map(|i| {
                    if top[i * 2] as usize != primary[i] {
                        top[i * 2]
                    } else {
                        top[i * 2 + 1]
                    }
                })
                .collect();

            out.par_iter_mut().enumerate().for_each(|(i, slot)| {
                if *slot as usize == primary[i] {
                    *slot = nearest_excluding(
                        &data[i * dim..(i + 1) * dim],
                        if cosine { data_norms[i] } else { T::one() },
                        centroids,
                        centroid_norms,
                        dim,
                        k,
                        primary[i],
                        metric,
                    );
                }
            });
            out
        }

        SoarRule::Shifted { mu } => {
            let mu = T::from_f64(mu.max(0.0)).unwrap_or_else(T::zero);
            let one_plus_mu = T::one() + mu;

            // x_tilde = (1 + mu) * x - mu * c_1, on the unit sphere for cosine.
            let mut shifted = vec![T::zero(); n * dim];
            shifted
                .par_chunks_mut(dim)
                .enumerate()
                .for_each(|(i, out)| {
                    let c1 = primary[i];
                    let vec = &data[i * dim..(i + 1) * dim];
                    let cent = &centroids[c1 * dim..(c1 + 1) * dim];
                    let (inv_v, inv_c) = if cosine {
                        (safe_recip(data_norms[i]), safe_recip(centroid_norms[c1]))
                    } else {
                        (T::one(), T::one())
                    };
                    for d in 0..dim {
                        out[d] = one_plus_mu * (vec[d] * inv_v) - mu * (cent[d] * inv_c);
                    }
                });

            let shifted_norms: Vec<T> = if cosine {
                shifted
                    .par_chunks(dim)
                    .map(|v| T::calculate_l2_norm(v))
                    .collect()
            } else {
                vec![T::one(); n]
            };

            let assigned = assign_all_parallel(
                &shifted,
                &shifted_norms,
                dim,
                n,
                centroids,
                centroid_norms,
                k,
                metric,
            );

            let mut out: Vec<u32> = assigned.into_iter().map(|c| c as u32).collect();
            // The shift moves away from `c_1`, so collisions thin out as `mu`
            // grows, but at small `mu` they are common. Repair against the
            // shifted point, which is what the rule actually ranks.
            out.par_iter_mut().enumerate().for_each(|(i, slot)| {
                if *slot as usize == primary[i] {
                    *slot = nearest_excluding(
                        &shifted[i * dim..(i + 1) * dim],
                        shifted_norms[i],
                        centroids,
                        centroid_norms,
                        dim,
                        k,
                        primary[i],
                        metric,
                    );
                }
            });
            out
        }

        SoarRule::Orthogonal { lambda } => {
            let lambda = T::from_f64(lambda.max(0.0)).unwrap_or_else(T::zero);
            let two = T::one() + T::one();
            let mut out = vec![0u32; n];

            out.par_iter_mut().enumerate().for_each_init(
                || vec![T::zero(); dim],
                |r1_hat, (i, slot)| {
                    let c1 = primary[i];
                    let vec = &data[i * dim..(i + 1) * dim];
                    let cent1 = &centroids[c1 * dim..(c1 + 1) * dim];
                    let (inv_v, inv_c1) = if cosine {
                        (safe_recip(data_norms[i]), safe_recip(centroid_norms[c1]))
                    } else {
                        (T::one(), T::one())
                    };

                    for d in 0..dim {
                        r1_hat[d] = vec[d] * inv_v - cent1[d] * inv_c1;
                    }
                    // A point sitting exactly on its centroid has no residual
                    // direction; `safe_recip` zeroes `r1_hat`, the penalty term
                    // vanishes and the arm falls back to nearest-excluding.
                    let inv_r1 = safe_recip(T::calculate_l2_norm(r1_hat.as_slice()));
                    for v in r1_hat.iter_mut() {
                        *v = *v * inv_r1;
                    }

                    // <x, r_hat_1> does not depend on the candidate, so hoist it
                    // and pay only one extra dot product per centroid.
                    let x_proj = T::dot_simd(vec, r1_hat.as_slice()) * inv_v;

                    // Seeded with a valid non-primary id, so an all-infinite
                    // scan still yields a distinct second cluster. `k >= 2` is
                    // guaranteed by the early return above.
                    let mut best = if c1 == 0 { 1u32 } else { 0u32 };
                    let mut best_cost = T::infinity();
                    for c in 0..k {
                        if c == c1 {
                            continue;
                        }
                        let cent = &centroids[c * dim..(c + 1) * dim];
                        let (base, proj) = if cosine {
                            let inv_c = safe_recip(centroid_norms[c]);
                            // ||x_hat - c_hat||^2 = 2 - 2 cos(x, c)
                            let cos = T::dot_simd(vec, cent) * inv_v * inv_c;
                            (
                                two - two * cos,
                                x_proj - T::dot_simd(cent, r1_hat.as_slice()) * inv_c,
                            )
                        } else {
                            (
                                T::euclidean_simd(vec, cent),
                                x_proj - T::dot_simd(cent, r1_hat.as_slice()),
                            )
                        };
                        let cost = base + lambda * proj * proj;
                        if cost < best_cost {
                            best_cost = cost;
                            best = c as u32;
                        }
                    }
                    *slot = best;
                },
            );
            out
        }
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
/// * `params_k_means` - An option for [KMeansTrainingParams]. This gives you
///   control over the maximum iterations, initialisation, path and size
///   balancing. If not provided, will default to sensible heuristics.
/// * `seed` - Seed for reproducibility
/// * `verbose` - Controls verbosity of the function
///
/// ### Returns
///
/// Flat `n_centroids * dim` row-major centroid buffer.
///
/// ### Errors
///
/// * `DistanceNotSupported` for `Manhattan`
/// * `TooFewSamplesForCentroids` if `n_centroids > n`. The random seeding path
///   draws distinct rows and would index past the end; k-means|| tolerates it
///   but only by emitting duplicate centroids.
#[allow(clippy::too_many_arguments)]
pub fn train_centroids<T>(
    data: &[T],
    dim: usize,
    n: usize,
    n_centroids: usize,
    metric: &Dist,
    params_k_means: Option<KMeansTrainingParams>,
    seed: usize,
    verbose: bool,
) -> Result<Vec<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    if *metric == Dist::Manhattan {
        return Err(AnnSearchErrors::DistanceNotSupported(*metric));
    }

    if n_centroids > n {
        return Err(AnnSearchErrors::TooFewSamplesForCentroids {
            n_centroids,
            n_samples: n,
        });
    }

    let params = params_k_means.unwrap_or_default();

    let data_norms: Vec<T> = match metric {
        Dist::SquaredEuclidean => (0..n)
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
        Dist::Manhattan => {
            unreachable!("Manhattan distance not reachable for train_centroids().")
        }
    };

    let init_method = resolve_init(params.init, n_centroids);

    let mut centroids = match init_method {
        KMeansInit::Random => {
            if verbose {
                println!("  Initialising centroids via fast random selection");
            }
            fast_random_init(data, dim, n, n_centroids, seed)
        }
        KMeansInit::KMeansParallel => {
            if verbose {
                println!("  Initialising centroids via k-means||");
            }
            let init_norms: Vec<T> = match metric {
                Dist::SquaredEuclidean => (0..n)
                    .map(|i| T::calculate_l2_norm(&data[i * dim..(i + 1) * dim]))
                    .collect(),
                Dist::Cosine => data_norms.clone(),
                Dist::Manhattan => {
                    unreachable!("Manhattan distance not reachable for train_centroids().")
                }
            };
            kmeans_parallel_init(data, &init_norms, dim, n, n_centroids, metric, seed)
        }
    };

    let mut centroid_norms: Vec<T> = match metric {
        Dist::SquaredEuclidean => (0..n_centroids)
            .map(|i| {
                let c = &centroids[i * dim..(i + 1) * dim];
                T::dot_simd(c, c)
            })
            .collect(),
        Dist::Cosine => (0..n_centroids)
            .map(|i| T::calculate_l2_norm(&centroids[i * dim..(i + 1) * dim]))
            .collect(),
        Dist::Manhattan => {
            unreachable!("Manhattan distance not reachable for train_centroids().")
        }
    };

    if verbose {
        println!("  Running Lloyd's iterations");
    }

    let lloyd_path = resolve_path(params.path, dim, n_centroids, metric);

    match lloyd_path {
        LloydPath::HamerlyGemm => {
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
                params.iters,
                params.balanced,
                seed,
                verbose,
            );
        }
        LloydPath::HamerlySimd => {
            if verbose {
                println!("    (Hamerly's bounds + SIMD assignment)");
            }
            hamerly_lloyd_simd(
                data,
                dim,
                n,
                &mut centroids,
                &mut centroid_norms,
                n_centroids,
                params.iters,
                params.balanced,
                seed,
                verbose,
            );
        }
        LloydPath::GemmLloyd => {
            if verbose {
                println!("    (GEMM assignment, no Hamerly)");
            }
            gemm_lloyd(
                data,
                &data_norms,
                dim,
                n,
                &mut centroids,
                &mut centroid_norms,
                n_centroids,
                metric,
                params.iters,
                params.balanced,
                seed,
                verbose,
            );
        }
        LloydPath::ParallelLloyd => {
            if verbose {
                println!("    (direct SIMD assignment)");
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
                params.iters,
                params.balanced,
                seed,
                verbose,
            );
        }
    }

    Ok(centroids)
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

/// Pick IVF cells to probe such that we reach at least `k` reachable vectors.
///
/// User-supplied `nprobe` is treated as a floor, never a ceiling: the returned
/// list always contains at least `nprobe` cells (clamped by `nlist`) AND enough
/// cells to cover `k` vectors when possible. Empty cells are included by rank
/// but naturally contribute nothing to the running total, so the walk keeps
/// going until enough non-empty candidates are reachable.
///
/// Without this, IVF queries silently return fewer than `k` neighbours when
/// the top-`nprobe` cells (or the top-1 landing on an empty cell) don't hold
/// enough vectors. Small datasets are the usual trigger.
///
/// Sorts `cluster_dists` ascending by distance in place.
///
/// ### Params
///
/// * `cluster_dists` - `(dist, cluster_id)` pairs for every cell in the index
/// * `offsets` - CSR offsets: cell `c` holds `offsets[c+1] - offsets[c]`
///   vectors
/// * `nprobe` - Requested number of cells (floor)
/// * `k` - Required number of reachable vectors
///
/// ### Returns
///
/// Cell ids in ascending-distance order, capped at `nlist`
pub fn select_probed_clusters<T>(
    cluster_dists: &mut [(T, usize)],
    offsets: &[usize],
    nprobe: usize,
    k: usize,
) -> Vec<usize>
where
    T: PartialOrd,
{
    cluster_dists
        .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut chosen = Vec::with_capacity(nprobe.max(1).min(cluster_dists.len()));
    let mut reachable = 0usize;
    for &(_, c) in cluster_dists.iter() {
        chosen.push(c);
        reachable += offsets[c + 1] - offsets[c];
        if chosen.len() >= nprobe && reachable >= k {
            break;
        }
    }
    chosen
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

    println!("   Cluster size distribution (diagnostics):");
    println!("     Min:    {}", min);
    println!("     P25:    {}", p25);
    println!("     Median: {}", p50);
    println!("     P75:    {}", p75);
    println!("     Max:    {}", max);
    println!("     Mean:   {}", mean);
    println!("     Imbalance ratio: {:.2}", max as f64 / mean as f64);
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_adjust_centers_leaves_balanced_partition_alone() {
        // Two clusters of two. Nothing is under a quarter of the average, so
        // the whole pass must be a no-op, centroids included.
        let data = vec![0.0_f32, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];
        let assignments = vec![0, 0, 1, 1];
        let counts = vec![2usize, 2];
        let mut centroids = vec![0.05_f32, 0.05, 10.05, 10.05];
        let before = centroids.clone();

        let adjusted = adjust_centers(&mut centroids, 2, 2, &data, 4, &assignments, &counts, 0);

        assert_eq!(adjusted, 0);
        assert_eq!(centroids, before);
    }

    #[test]
    fn test_adjust_centers_reseeds_empty_cluster_onto_donor() {
        // Cluster 1 is empty, so its weight is zero and the centroid must land
        // exactly on a donor point rather than merely drifting toward one.
        let data = vec![0.0_f32, 1.0, 2.0, 3.0];
        let assignments = vec![0, 0, 0, 0];
        let counts = vec![4usize, 0];
        let mut centroids = vec![1.5_f32, -99.0];

        let adjusted = adjust_centers(&mut centroids, 1, 2, &data, 4, &assignments, &counts, 0);

        assert_eq!(adjusted, 1);
        assert_eq!(centroids[0], 1.5, "the healthy centroid must not move");
        assert!(
            data.contains(&centroids[1]),
            "an empty cluster jumps onto the donor point, got {}",
            centroids[1]
        );
    }

    #[test]
    fn test_adjust_centers_pulls_starved_cluster_halfway() {
        // Nine points against one: cluster 1 is starved but not empty, so its
        // weight is 1 and the new centroid is the midpoint of the old one and
        // the donor.
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut assignments = vec![0usize; 10];
        assignments[9] = 1;
        let counts = vec![9usize, 1];
        let mut centroids = vec![4.0_f32, 100.0];

        let adjusted = adjust_centers(&mut centroids, 1, 2, &data, 10, &assignments, &counts, 0);

        assert_eq!(adjusted, 1);
        // (100 + donor) / 2 for some donor in 0..=8, so strictly between.
        assert!(
            centroids[1] > 50.0 && centroids[1] < 55.0,
            "expected a midpoint pull, got {}",
            centroids[1]
        );
        let donor = centroids[1] * 2.0 - 100.0;
        assert!(
            data[..9].contains(&donor),
            "recovered donor {} is not a point of the over-full cluster",
            donor
        );
    }

    #[test]
    fn test_adjust_centers_without_donors_is_a_noop() {
        // Every cluster sits exactly at the average, so nothing is above it and
        // there is no donor to draw from. Must bail rather than pick a victim.
        let data = vec![0.0_f32, 1.0, 2.0, 3.0];
        let assignments = vec![0, 1, 2, 3];
        let counts = vec![1usize, 1, 1, 1];
        let mut centroids = vec![0.0_f32, 1.0, 2.0, 3.0];
        let before = centroids.clone();

        let adjusted = adjust_centers(&mut centroids, 1, 4, &data, 4, &assignments, &counts, 0);

        assert_eq!(adjusted, 0);
        assert_eq!(centroids, before);
    }

    #[test]
    fn test_balanced_training_evens_out_a_skewed_partition() {
        // 200 points in one tight blob plus 3 far outliers. k-means|| is
        // D^2-weighted, so it seeds onto the outliers by construction, and
        // plain Lloyd's then parks three centroids on singletons and leaves the
        // fourth holding the entire blob. Balancing has to pull them back in.
        //
        // The outliers are what make this test able to fail: without them,
        // random init lands every centroid inside the blob, no cluster is ever
        // starved, and the assertion passes for the wrong reason.
        let (n, dim, k) = (203usize, 2usize, 4usize);
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..200 {
            data.push((i % 20) as f32 * 0.01);
            data.push((i / 20) as f32 * 0.01);
        }
        for i in 0..3 {
            data.push(1000.0 * (i + 1) as f32);
            data.push(-1000.0 * (i + 1) as f32);
        }

        let largest_cluster = |params: KMeansTrainingParams| {
            let centroids = train_centroids(
                &data,
                dim,
                n,
                k,
                &Dist::SquaredEuclidean,
                Some(params),
                7,
                false,
            )
            .unwrap();
            let norms: Vec<f32> = (0..k)
                .map(|c| {
                    let cent = &centroids[c * dim..(c + 1) * dim];
                    f32::dot_simd(cent, cent)
                })
                .collect();
            let assignments = assign_all_parallel(
                &data,
                &vec![0.0f32; n],
                dim,
                n,
                &centroids,
                &norms,
                k,
                &Dist::SquaredEuclidean,
            );
            let mut counts = vec![0usize; k];
            for &a in &assignments {
                counts[a] += 1;
            }
            *counts.iter().max().unwrap()
        };

        let params = KMeansTrainingParams::new(30, Some(KMeansInit::KMeansParallel), None);
        let plain = largest_cluster(params);
        let balanced = largest_cluster(params.with_balancing(true));

        assert_eq!(plain, 200, "the unbalanced baseline is not the skewed case");
        assert!(
            balanced < plain,
            "balancing did not shrink the largest cluster: {} vs {}",
            balanced,
            plain
        );
    }

    #[test]
    fn test_assign_top_m_stays_distinct_when_every_distance_is_infinite() {
        // A non-finite coordinate makes every candidate distance `+inf`, so the
        // insertion buffer never fires and the row is whatever it was seeded
        // with. That row has to be `m` *distinct* clusters: the batched
        // NN-Descent merge derives the soundness of an unsafe disjoint write
        // from it, so a repeat here is undefined behaviour downstream, not a
        // quality regression.
        let (n, dim, k, m) = (1usize, 2usize, 4usize, 2usize);
        let data = vec![f32::INFINITY, 0.0];
        let centroids: Vec<f32> = (0..k * dim).map(|i| i as f32).collect();
        let norms = vec![1.0f32; k.max(n)];

        let got = assign_all_parallel_top_m(
            &data,
            &norms,
            dim,
            n,
            &centroids,
            &norms,
            k,
            m,
            &Dist::SquaredEuclidean,
        );

        assert_eq!(got.len(), n * m);
        assert_ne!(
            got[0], got[1],
            "a point must not join the same cluster twice"
        );
        assert!(got.iter().all(|&c| (c as usize) < k));
    }

    #[test]
    fn test_invert_assignments_handles_the_clamped_width() {
        // `assign_all_parallel_top_m` clamps `m` to `k`. A caller that asks for
        // more assignments than there are clusters must feed the *clamped*
        // width here, or the inversion indexes past the end of the buffer.
        let (n, k) = (4usize, 2usize);
        let requested = 3usize;
        let data: Vec<f32> = (0..n * 2).map(|i| i as f32).collect();
        let centroids = vec![0.0f32, 0.0, 100.0, 100.0];
        let norms = vec![1.0f32; n.max(k)];

        let assignments = assign_all_parallel_top_m(
            &data,
            &norms,
            2,
            n,
            &centroids,
            &norms,
            k,
            requested,
            &Dist::SquaredEuclidean,
        );

        let effective = assignments.len() / n;
        assert_eq!(effective, k, "m must have been clamped to k");

        let (members, offsets) = invert_assignments_csr(&assignments, n, effective, k);
        assert_eq!(offsets[k], n * effective);
        assert_eq!(members.len(), n * effective);
    }

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
    fn test_select_probed_clusters_respects_nprobe_floor() {
        // 4 cells of size 5 each; nprobe=3, k=2 -> return 3 cells even though
        // 1 would already cover k.
        let mut dists = vec![(0.5f32, 0), (0.1, 1), (0.9, 2), (0.3, 3)];
        let offsets = vec![0, 5, 10, 15, 20];
        let chosen = select_probed_clusters(&mut dists, &offsets, 3, 2);
        assert_eq!(chosen.len(), 3);
        // Ascending distance order: 1 (0.1), 3 (0.3), 0 (0.5)
        assert_eq!(chosen, vec![1, 3, 0]);
    }

    #[test]
    fn test_select_probed_clusters_expands_to_reach_k() {
        // 4 tiny cells of size 2 each; nprobe=1, k=5 -> must expand to 3 cells.
        let mut dists = vec![(0.5f32, 0), (0.1, 1), (0.9, 2), (0.3, 3)];
        let offsets = vec![0, 2, 4, 6, 8];
        let chosen = select_probed_clusters(&mut dists, &offsets, 1, 5);
        assert_eq!(chosen.len(), 3);
        assert_eq!(chosen, vec![1, 3, 0]);
    }

    #[test]
    fn test_select_probed_clusters_skips_empty_cells() {
        // Nearest cell is empty; expansion should walk past it.
        let mut dists = vec![(0.1f32, 0), (0.5, 1), (0.9, 2)];
        let offsets = vec![0, 0, 3, 6]; // cell 0 empty, cell 1 size 3, cell 2 size 3
        let chosen = select_probed_clusters(&mut dists, &offsets, 1, 2);
        // nprobe=1 satisfied by cell 0, but reachable=0 forces expansion to
        // cell 1 (size 3) which pushes reachable to 3 >= k=2.
        assert_eq!(chosen, vec![0, 1]);
    }

    #[test]
    fn test_select_probed_clusters_caps_at_nlist() {
        // k larger than the whole index -> return every cell, in order.
        let mut dists = vec![(0.5f32, 0), (0.1, 1), (0.9, 2)];
        let offsets = vec![0, 2, 4, 6];
        let chosen = select_probed_clusters(&mut dists, &offsets, 1, 1000);
        assert_eq!(chosen.len(), 3);
        assert_eq!(chosen, vec![1, 0, 2]);
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
            &Dist::SquaredEuclidean,
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

        let centroids =
            train_centroids(&data, 2, 4, 2, &Dist::SquaredEuclidean, None, 42, false).unwrap();

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
            &Dist::SquaredEuclidean,
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

        let centroids =
            weighted_kmeans_plus_plus(&data, &data_norms, 2, 2, &Dist::SquaredEuclidean, 42);

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

    #[test]
    fn test_assign_one_with_bounds_simd() {
        // Three centroids at (0,0), (3,0), (10,0). Query at (1,0):
        // best = c0 (dist 1), second = c1 (dist 2)
        let centroids = vec![0.0_f64, 0.0, 3.0, 0.0, 10.0, 0.0];
        let vec = vec![1.0_f64, 0.0];

        let (best_c, best_d, second_d) = assign_one_with_bounds_simd(&vec, &centroids, 2, 3);

        assert_eq!(best_c, 0);
        assert_relative_eq!(best_d, 1.0, epsilon = 1e-10);
        assert_relative_eq!(second_d, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_assign_full_with_bounds() {
        let data = vec![
            0.0_f64, 0.0, // near c0
            0.1, 0.1, // near c0
            10.0, 10.0, // near c1
            9.9, 10.1, // near c1
        ];
        let centroids = vec![0.0_f64, 0.0, 10.0, 10.0];

        let mut assignments = vec![0usize; 4];
        let mut upper = vec![0.0_f64; 4];
        let mut lower = vec![0.0_f64; 4];

        simd_assign_full_with_bounds(
            &data,
            2,
            &centroids,
            2,
            &mut assignments,
            &mut upper,
            &mut lower,
        );

        assert_eq!(assignments, vec![0, 0, 1, 1]);
        // Vector 0 sits exactly on c0
        assert_relative_eq!(upper[0], 0.0, epsilon = 1e-10);
        // Lower bound for vector 0 is distance to c1
        assert_relative_eq!(lower[0], (200.0_f64).sqrt(), epsilon = 1e-10);
        // Upper < lower for every well-clustered point
        for i in 0..4 {
            assert!(upper[i] <= lower[i]);
        }
    }

    #[test]
    fn test_simd_reassign_dirty_updates_only_dirty() {
        let data = vec![
            0.0_f64, 0.0, //
            0.1, 0.1, //
            10.0, 10.0, //
            9.9, 10.1, //
        ];
        let centroids = vec![0.0_f64, 0.0, 10.0, 10.0];

        let mut assignments = vec![99usize; 4];
        let mut upper = vec![-1.0_f64; 4];
        let mut lower = vec![-1.0_f64; 4];

        // Only points 0 and 2 are dirty
        let dirty = vec![0usize, 2];

        simd_reassign_dirty(
            &data,
            2,
            &centroids,
            2,
            &dirty,
            &mut assignments,
            &mut upper,
            &mut lower,
        );

        // Dirty points overwritten
        assert_eq!(assignments[0], 0);
        assert_eq!(assignments[2], 1);
        assert!(upper[0] >= 0.0 && upper[2] >= 0.0);

        // Non-dirty points untouched
        assert_eq!(assignments[1], 99);
        assert_eq!(assignments[3], 99);
        assert_relative_eq!(upper[1], -1.0, epsilon = 1e-10);
        assert_relative_eq!(upper[3], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_half_min_centroid_dists_simd() {
        // Three centroids at (0,0), (3,0), (10,0).
        // dist(c0,c1)=3, dist(c0,c2)=10, dist(c1,c2)=7
        // s[0] = 3/2, s[1] = 3/2, s[2] = 7/2
        let centroids = vec![0.0_f64, 0.0, 3.0, 0.0, 10.0, 0.0];

        let s = compute_half_min_centroid_dists_simd(&centroids, 2, 3);

        assert_relative_eq!(s[0], 1.5, epsilon = 1e-10);
        assert_relative_eq!(s[1], 1.5, epsilon = 1e-10);
        assert_relative_eq!(s[2], 3.5, epsilon = 1e-10);
    }

    #[test]
    fn test_hamerly_lloyd_simd_two_clusters() {
        // Same shape as test_train_centroids_small but driving the SIMD
        // Hamerly path directly
        let data = vec![0.0_f64, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];
        let mut centroids = vec![0.0_f64, 0.0, 10.0, 10.0];
        let mut centroid_norms = vec![0.0_f64; 2];

        hamerly_lloyd_simd(
            &data,
            2,
            4,
            &mut centroids,
            &mut centroid_norms,
            2,
            20,
            false,
            0,
            false,
        );

        let cent0 = (centroids[0], centroids[1]);
        let cent1 = (centroids[2], centroids[3]);

        // Centroids should sit at the cluster means
        let near_low = |c: (f64, f64)| (c.0 - 0.05).abs() < 1e-6 && (c.1 - 0.05).abs() < 1e-6;
        let near_high = |c: (f64, f64)| (c.0 - 10.05).abs() < 1e-6 && (c.1 - 10.05).abs() < 1e-6;

        assert!((near_low(cent0) && near_high(cent1)) || (near_high(cent0) && near_low(cent1)));
    }

    #[test]
    fn test_train_centroids_dispatches_simd_hamerly() {
        // Synthetic 2D data with many tight clusters; n_centroids above
        // SIMD_HAMERLY_K_THRESHOLD forces the SIMD Hamerly path
        let n_clusters = 120;
        let pts_per_cluster = 5;
        let dim = 2;
        let n = n_clusters * pts_per_cluster;

        let mut data = Vec::with_capacity(n * dim);
        for c in 0..n_clusters {
            let cx = c as f64;
            let cy = (c * 3) as f64;
            for p in 0..pts_per_cluster {
                let jitter = p as f64 * 1e-3;
                data.push(cx + jitter);
                data.push(cy + jitter);
            }
        }

        let centroids = train_centroids(
            &data,
            dim,
            n,
            n_clusters,
            &Dist::SquaredEuclidean,
            None,
            42,
            false,
        )
        .unwrap();

        assert_eq!(centroids.len(), n_clusters * dim);

        // Every centroid should be finite
        assert!(centroids.iter().all(|x: &f64| x.is_finite()));
    }
}
