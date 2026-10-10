//! k-means on MLX: device-resident Lloyd's loop for the IVF MLX index.
//!
//! Assignment is a GEMM against the centroids plus a row argmin (the
//! exhaustive index's top-k kernel at K = 1), tiled over the points so the
//! `points x centroids` score matrix stays bounded. The
//! centroid update is two `scatter_add`s (coordinate sums and counts) and a
//! divide, so the whole loop stays lazy on the device and only the final
//! centroids come back. Initialisation reuses the host [`fast_random_init`] /
//! [`kmeans_parallel_init`], so the MLX and CPU k-means start from the same
//! centroids at the same seed.
//!
//! Every iteration runs: there is no convergence check, matching the default
//! fixed mode of the wgpu k-means and keeping the host out of the loop.
//! Balancing, when asked for, is the exception: [`adjust_centers`] runs on the
//! host and costs one sync per iteration.

use rayon::prelude::*;

use crate::mlx::exhaustive_mlx::RowTopK;
use crate::mlx::ffi::*;
use crate::prelude::*;
use crate::utils::k_means_utils::*;

////////////
// Consts //
////////////

/// Upper bound in bytes on one tile's `points x centroids` f32 score matrix.
const KMEANS_SCORE_TILE_BYTES: usize = 256 * 1024 * 1024;

/// Floor on a centroid's squared norm before the cosine rescale, so an
/// all-zero centroid does not turn into NaNs.
const MIN_CENTROID_SQ_NORM: f32 = 1e-24;

/////////////
// Helpers //
/////////////

/// Rows per score tile for a given centroid count.
///
/// ### Params
///
/// * `n_centroids` - Width of the score matrix
///
/// ### Returns
///
/// Rows per tile, at least 1
pub(crate) fn score_tile_rows(n_centroids: usize) -> usize {
    (KMEANS_SCORE_TILE_BYTES / (n_centroids.max(1) * size_of::<f32>())).max(1)
}

/// GEMM operands that turn `x @ ct` into a score whose row argmin is the
/// nearest centroid.
///
/// Euclidean ranks on `|c|^2 - 2 x.c` (`|x|^2` is constant per row). Cosine
/// ranks on `-x.c / |c|` (`|x|` likewise), so the centroids are rescaled to
/// unit length here and the points need not be.
pub(crate) struct CentroidOperands {
    /// Centroids, transposed: `[dim, n_centroids]`
    pub ct: Array,
    /// Additive GEMM term: `|c|^2` as `[1, n_centroids]` or the scalar 0
    pub add: Array,
    /// GEMM scale: -2 for Euclidean, -1 for Cosine
    pub alpha: f32,
}

impl CentroidOperands {
    /// Build the operands from a `[n_centroids, dim]` centroid array.
    ///
    /// ### Params
    ///
    /// * `c` - Centroids on the device
    /// * `n_centroids` - Number of centroids
    /// * `metric` - Euclidean or Cosine
    /// * `s` - Stream to run on
    ///
    /// ### Returns
    ///
    /// The (lazy) operands
    pub fn new(
        c: &Array,
        n_centroids: usize,
        metric: &Dist,
        s: &Stream,
    ) -> Result<Self, AnnSearchErrors> {
        let sq = c.square(s)?.sum_axis(1, true, s)?;
        match metric {
            Dist::Cosine => {
                let inv = sq
                    .maximum(&Array::scalar_f32(MIN_CENTROID_SQ_NORM), s)?
                    .rsqrt(s)?;
                Ok(Self {
                    ct: c.multiply(&inv, s)?.transpose(s)?,
                    add: Array::scalar_f32(0.0),
                    alpha: -1.0,
                })
            }
            _ => Ok(Self {
                ct: c.transpose(s)?,
                add: sq.reshape(&[1, n_centroids as i32], s)?,
                alpha: -2.0,
            }),
        }
    }

    /// Nearest centroid per row of a point tile.
    ///
    /// ### Params
    ///
    /// * `x` - Points, `[t, dim]`
    /// * `rows` - `t`
    /// * `n_centroids` - Number of centroids
    /// * `argmin` - Kernel from [`argmin_kernel`]
    /// * `s` - Stream to run on
    ///
    /// ### Returns
    ///
    /// The (lazy) u32 assignment, `[t]`
    pub fn assign(
        &self,
        x: &Array,
        rows: usize,
        n_centroids: usize,
        argmin: &RowTopK,
        s: &Stream,
    ) -> Result<Array, AnnSearchErrors> {
        let scores = Array::addmm(&self.add, x, &self.ct, self.alpha, 1.0, s)?;
        let (idx, _) = argmin.apply(&scores, rows, n_centroids, 1, s)?;
        idx.reshape(&[rows as i32], s)
    }
}

/// Row argmin as the exhaustive index's top-k at `K = 1`. MLX's own `argmin`
/// measured slower on these narrow rows.
///
/// ### Returns
///
/// The kernel pair
pub(crate) fn argmin_kernel() -> RowTopK {
    RowTopK::new("kmeans_argmin")
}

/// Upload row-major points as `[t, dim]` tiles.
///
/// ### Params
///
/// * `data` - Row-major points
/// * `dim` - Row length
/// * `tile` - Rows per tile
///
/// ### Returns
///
/// `(tile, rows in it)` pairs in order
fn upload_tiles(data: &[f32], dim: usize, tile: usize) -> Vec<(Array, usize)> {
    data.chunks(tile * dim)
        .map(|chunk| {
            let rows = chunk.len() / dim;
            (Array::from_f32(chunk, &[rows as i32, dim as i32]), rows)
        })
        .collect()
}

/// Sum of squared Euclidean distances from each point to its centroid.
///
/// ### Params
///
/// * `data` - Row-major points
/// * `dim` - Row length
/// * `centroids` - Row-major centroids
/// * `assignments` - Centroid index per point
///
/// ### Returns
///
/// The within-cluster sum of squares
#[cfg(test)]
pub(crate) fn inertia(data: &[f32], dim: usize, centroids: &[f32], assignments: &[usize]) -> f64 {
    data.par_chunks_exact(dim)
        .zip(assignments.par_iter())
        .map(|(x, &c)| {
            x.iter()
                .zip(&centroids[c * dim..(c + 1) * dim])
                .map(|(a, b)| ((a - b) * (a - b)) as f64)
                .sum::<f64>()
        })
        .sum()
}

//////////
// Main //
//////////

/// Train k-means centroids on MLX.
///
/// MLX counterpart of the CPU [`train_centroids`] and the wgpu
/// `train_centroids_gpu`, with the same contract: flat unpadded input, flat
/// unpadded `n_centroids * dim` centroids out.
///
/// ### Params
///
/// * `data` - Flattened row-major training data, `n * dim`
/// * `dim` - Embedding dimensionality
/// * `n` - Number of training points
/// * `n_centroids` - Number of centroids to train
/// * `metric` - Distance metric; `Manhattan` is rejected
/// * `params` - Optional [`KMeansTrainingParams`]. `iters`, `init` and
///   `balanced` are honoured; `path` has no MLX meaning and is ignored.
/// * `seed` - Seed for the initialisation and the balancing donor walk
/// * `s` - Stream to run on
/// * `verbose` - Controls verbosity
///
/// ### Returns
///
/// Flat `n_centroids * dim` row-major centroids
#[allow(clippy::too_many_arguments)]
pub fn train_centroids_mlx(
    data: &[f32],
    dim: usize,
    n: usize,
    n_centroids: usize,
    metric: &Dist,
    params: Option<KMeansTrainingParams>,
    seed: usize,
    s: &Stream,
    verbose: bool,
) -> Result<Vec<f32>, AnnSearchErrors> {
    if *metric == Dist::Manhattan {
        return Err(AnnSearchErrors::DistanceNotSupported(*metric));
    }
    if n_centroids > n {
        return Err(AnnSearchErrors::TooFewSamplesForCentroids {
            n_centroids,
            n_samples: n,
        });
    }
    let params = params.unwrap_or_default();

    let init = params.init.unwrap_or(if n_centroids > 200 {
        KMeansInit::Random
    } else {
        KMeansInit::KMeansParallel
    });
    let init_centroids = match init {
        KMeansInit::Random => fast_random_init(data, dim, n, n_centroids, seed),
        KMeansInit::KMeansParallel => {
            let norms: Vec<f32> = data
                .par_chunks_exact(dim)
                .map(f32::calculate_l2_norm)
                .collect();
            kmeans_parallel_init(data, &norms, dim, n, n_centroids, metric, seed)
        }
    };
    if verbose {
        println!(
            "  MLX k-means: {} centroids, {:?} init, {} iterations",
            n_centroids, init, params.iters
        );
    }

    let k = n_centroids as i32;
    let tiles: Vec<(Array, usize, Array, Array)> =
        upload_tiles(data, dim, score_tile_rows(n_centroids))
            .into_iter()
            .map(|(x, rows)| {
                let upd = x.reshape(&[rows as i32, 1, dim as i32], s)?;
                let ones = Array::ones(&[rows as i32, 1], MLX_FLOAT32, s)?;
                Ok((x, rows, upd, ones))
            })
            .collect::<Result<_, AnnSearchErrors>>()?;
    let argmin = argmin_kernel();

    let zero = Array::scalar_f32(0.0);
    let one = Array::scalar_f32(1.0);
    let mut c = Array::from_f32(&init_centroids, &[k, dim as i32]);

    for iter in 0..params.iters {
        let ops = CentroidOperands::new(&c, n_centroids, metric, s)?;
        let mut sums = Array::zeros(&[k, dim as i32], MLX_FLOAT32, s)?;
        let mut counts = Array::zeros(&[k], MLX_FLOAT32, s)?;
        let mut assigns = Vec::with_capacity(tiles.len());
        for (x, rows, upd, ones) in &tiles {
            let a = ops.assign(x, *rows, n_centroids, &argmin, s)?;
            sums = sums.scatter_add_rows(&a, upd, s)?;
            counts = counts.scatter_add_rows(&a, ones, s)?;
            assigns.push(a);
        }
        let counts = counts.reshape(&[k, 1], s)?;
        let mean = sums.divide(&counts.maximum(&one, s)?, s)?;
        // Empty clusters keep their previous centroid, as on the wgpu path.
        c = Array::select(&counts.greater(&zero, s)?, &mean, &c, s)?;

        if params.balanced {
            let mut wait: Vec<&Array> = vec![&c, &counts];
            wait.extend(assigns.iter());
            eval_all(&wait, false)?;
            let assignments: Vec<usize> = assigns
                .iter()
                .map(|a| Ok(a.as_u32()?.iter().map(|&v| v as usize)))
                .collect::<Result<Vec<_>, AnnSearchErrors>>()?
                .into_iter()
                .flatten()
                .collect();
            let cnt: Vec<usize> = counts.as_f32()?.iter().map(|&v| v as usize).collect();
            let mut host = c.as_f32()?.to_vec();
            adjust_centers(
                &mut host,
                dim,
                n_centroids,
                data,
                n,
                &assignments,
                &cnt,
                seed.wrapping_add(iter),
            );
            c = Array::from_f32(&host, &[k, dim as i32]);
        } else {
            // Bounds the lazy graph to one iteration without a host wait.
            eval_all(&[&c], true)?;
        }
    }

    eval_all(&[&c], false)?;
    Ok(c.as_f32()?.to_vec())
}

/// Assign every point to its nearest centroid on MLX.
///
/// MLX counterpart of the CPU [`assign_all_parallel`] and the wgpu
/// `assign_all_gpu`.
///
/// ### Params
///
/// * `data` - Flattened row-major points, `n * dim`
/// * `dim` - Embedding dimensionality
/// * `centroids` - Flattened row-major centroids, `n_centroids * dim`
/// * `n_centroids` - Number of centroids
/// * `metric` - Distance metric; `Manhattan` is rejected
/// * `s` - Stream to run on
///
/// ### Returns
///
/// Cluster index per point
pub fn assign_all_mlx(
    data: &[f32],
    dim: usize,
    centroids: &[f32],
    n_centroids: usize,
    metric: &Dist,
    s: &Stream,
) -> Result<Vec<usize>, AnnSearchErrors> {
    if *metric == Dist::Manhattan {
        return Err(AnnSearchErrors::DistanceNotSupported(*metric));
    }
    let c = Array::from_f32(centroids, &[n_centroids as i32, dim as i32]);
    let ops = CentroidOperands::new(&c, n_centroids, metric, s)?;
    let argmin = argmin_kernel();

    let pending = upload_tiles(data, dim, score_tile_rows(n_centroids))
        .into_iter()
        .map(|(x, rows)| {
            let a = ops.assign(&x, rows, n_centroids, &argmin, s)?;
            eval_all(&[&a], true)?;
            Ok(a)
        })
        .collect::<Result<Vec<_>, AnnSearchErrors>>()?;

    let mut out = Vec::with_capacity(data.len() / dim.max(1));
    for a in &pending {
        eval_all(&[a], false)?;
        out.extend(a.as_u32()?.iter().map(|&v| v as usize));
    }
    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Gaussian blobs, row-major.
    ///
    /// ### Params
    ///
    /// * `n` - Points
    /// * `dim` - Dimensions
    /// * `blobs` - Number of blob centres
    /// * `seed` - RNG seed
    ///
    /// ### Returns
    ///
    /// Flat `n * dim` data
    fn blobs(n: usize, dim: usize, blobs: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let centres: Vec<f32> = (0..blobs * dim)
            .map(|_| rng.random_range(-5.0..5.0))
            .collect();
        (0..n)
            .flat_map(|i| {
                let b = i % blobs;
                let row: Vec<f32> = (0..dim)
                    .map(|d| centres[b * dim + d] + rng.random_range(-1.0..1.0))
                    .collect();
                row
            })
            .collect()
    }

    /// MLX k-means against the CPU k-means at the same seed, init and
    /// iteration count; both start from the same host initialisation.
    ///
    /// ### Params
    ///
    /// * `metric` - Metric under test
    /// * `init` - Initialisation under test
    fn check_inertia(metric: Dist, init: KMeansInit) {
        let (n, dim, k) = (4_000, 16, 40);
        let data = blobs(n, dim, 25, 11);
        let params = KMeansTrainingParams::new(15, Some(init), None);
        let s = Stream::default_gpu();

        let mlx =
            train_centroids_mlx(&data, dim, n, k, &metric, Some(params), 7, &s, false).unwrap();
        let cpu = train_centroids(&data, dim, n, k, &metric, Some(params), 7, false).unwrap();

        let mlx_assign = assign_all_mlx(&data, dim, &mlx, k, &metric, &s).unwrap();
        let ones_n = vec![1.0f32; n];
        let ones_k = vec![1.0f32; k];
        let cpu_assign = assign_all_parallel(
            &data,
            &ones_n,
            dim,
            n,
            &cpu,
            &ones_k,
            k,
            &Dist::SquaredEuclidean,
        );

        let (a, b) = (
            inertia(&data, dim, &mlx, &mlx_assign),
            inertia(&data, dim, &cpu, &cpu_assign),
        );
        assert!(a.is_finite() && b.is_finite());
        assert!((a - b).abs() / b < 0.02, "mlx {a} vs cpu {b}");
    }

    #[test]
    fn test_mlx_kmeans_inertia_matches_cpu_euclidean() {
        check_inertia(Dist::SquaredEuclidean, KMeansInit::KMeansParallel);
        check_inertia(Dist::SquaredEuclidean, KMeansInit::Random);
    }

    #[test]
    fn test_mlx_kmeans_inertia_matches_cpu_cosine() {
        check_inertia(Dist::Cosine, KMeansInit::KMeansParallel);
    }

    #[test]
    fn test_mlx_assign_matches_cpu() {
        let (n, dim, k) = (3_000, 12, 30);
        let data = blobs(n, dim, 10, 3);
        let cents = fast_random_init(&data, dim, n, k, 1);
        let s = Stream::default_gpu();
        for metric in [Dist::SquaredEuclidean, Dist::Cosine] {
            let mlx = assign_all_mlx(&data, dim, &cents, k, &metric, &s).unwrap();
            let (dn, cn): (Vec<f32>, Vec<f32>) = if metric == Dist::Cosine {
                (
                    data.chunks(dim).map(f32::calculate_l2_norm).collect(),
                    cents.chunks(dim).map(f32::calculate_l2_norm).collect(),
                )
            } else {
                (vec![1.0; n], vec![1.0; k])
            };
            let cpu = assign_all_parallel(&data, &dn, dim, n, &cents, &cn, k, &metric);
            let agree = mlx.iter().zip(&cpu).filter(|(a, b)| a == b).count();
            // Near-ties may flip through the GEMM expansion.
            assert!(agree as f64 / n as f64 > 0.995, "{metric:?}: {agree}/{n}");
        }
    }

    #[test]
    fn test_mlx_kmeans_balanced_runs() {
        let (n, dim, k) = (2_000, 8, 20);
        let data = blobs(n, dim, 5, 5);
        let params = KMeansTrainingParams::new(5, None, None).with_balancing(true);
        let s = Stream::default_gpu();
        let c = train_centroids_mlx(
            &data,
            dim,
            n,
            k,
            &Dist::SquaredEuclidean,
            Some(params),
            3,
            &s,
            false,
        )
        .unwrap();
        assert_eq!(c.len(), k * dim);
        assert!(c.iter().all(|v| v.is_finite()));
    }
}
