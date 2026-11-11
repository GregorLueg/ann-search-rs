//! e-means: mini-batch k-means with an exponential moving average update and
//! fitted scaling laws for the batch size and the step.
//!
//! Everything here is host-side and backend-agnostic: the recipe, the EMA
//! state, cluster revival and the closing mean. A backend only supplies the
//! per-batch assignment plus per-cluster means and counts (see
//! [`run_e_means`]), so the CPU, wgpu and MLX trainers share all of the
//! statistics. Constants and laws follow the reference implementation in
//! `faiss/contrib/e_means/base.py`.

use num_traits::Float;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;
use thousands::*;

use crate::errors::AnnSearchErrors;
use crate::prelude::AnnSearchFloat;
use crate::utils::k_means_utils::{assign_all_parallel, fast_random_init};
use crate::utils::verbosity::Verbosity;
use crate::utils::Dist;

////////////
// Consts //
////////////

/// Floor of the cosine step schedule, as a fraction of the initial step.
const ALPHA_MIN_FACTOR: f64 = 0.1;

/// Batch updates between two revival checks.
const CHECK_UNUSED_EVERY: usize = 5;

/// Initial EMA mass of every cluster. Small enough that the first batch to
/// touch a cluster moves it almost all the way to the batch mean.
const INIT_USAGE: f64 = 1.0 / 65536.0;

/// Relative coordinate perturbation separating a split donor from its child.
const SPLIT_EPSILON: f64 = 1e-4;

/// Default revival threshold: clusters below this fraction of the mean mass
/// are reseeded.
const UNUSED_THRESHOLD: f64 = 0.1;

/// At or below this dimension revival is off by default. The reference found
/// it costs held-out error on 2-D data and makes no difference from 3-D up.
const NO_REVIVAL_MAX_D: usize = 2;

/// At or below this dimension [`EMeansBatchLaw::Auto`] picks the quality law,
/// whose extra updates matter more at low dimension.
const AUTO_QUALITY_MAX_D: usize = 16;

/// Throughput batch law `B = c * k^a * n^b * T^e`, as `(c, a, b, e)`.
const THROUGHPUT_LAW: (f64, f64, f64, f64) = (0.035134, 0.134749, 0.876473, 0.813344);

/// Quality batch law: one eighth of the throughput coefficient.
const QUALITY_LAW: (f64, f64, f64, f64) = (0.00439175, 0.134749, 0.876473, 0.813344);

/// Step law `alpha0 = c * k^a * B^b * n^e * T^f`, as `(c, a, b, e, f)`.
const STEP_LAW: (f64, f64, f64, f64, f64) = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057);

/// Lower clamp of the fitted initial step.
const ALPHA0_MIN: f64 = 1e-4;

///////////////////
// Enums, Params //
///////////////////

/// Which scaling law picks the e-means batch size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EMeansBatchLaw {
    /// Quality at `dim <= 16`, throughput above
    #[default]
    Auto,
    /// Larger batches, fewer updates, faster per epoch
    Throughput,
    /// Batches about eight times smaller; lower loss, slower on small data
    Quality,
}

/// e-means parameters. Every `None` resolves through the fitted laws, so
/// `epochs` is the only knob that normally needs touching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EMeansParams {
    /// Passes over the training data, may be fractional. 5 typically matches
    /// or beats Lloyd's loss at convergence.
    pub epochs: f64,
    /// Scaling law for the batch size; ignored when `batch_size` is set
    pub batch_law: EMeansBatchLaw,
    /// Batch size override. The step law still sets `alpha0` unless that is
    /// given as well.
    pub batch_size: Option<usize>,
    /// Initial EMA step override, in `(0, 1]`
    pub alpha0: Option<f64>,
    /// Revive clusters whose mass falls below this fraction of the mean.
    /// `None` is 0.1, or 0 (off) at `dim <= 2`.
    pub unused_threshold: Option<f64>,
}

impl EMeansParams {
    /// New parameters with every other field resolved from the laws.
    ///
    /// ### Params
    ///
    /// * `epochs` - Passes over the training data, may be fractional
    ///
    /// ### Returns
    ///
    /// [`EMeansParams`]
    pub fn new(epochs: f64) -> Self {
        Self {
            epochs,
            batch_law: EMeansBatchLaw::Auto,
            batch_size: None,
            alpha0: None,
            unused_threshold: None,
        }
    }
}

impl Default for EMeansParams {
    fn default() -> Self {
        Self::new(5.0)
    }
}

/// Which k-means algorithm trains the centroids.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum KMeansAlgorithm {
    /// Lloyd's algorithm: full passes until convergence or the iteration cap
    #[default]
    Lloyd,
    /// e-means: mini-batch EMA updates with fitted batch size and step
    ///
    /// ### References
    ///
    /// Bessa et al., e-means: outperforming k-means (in one-tenth the
    /// epochs), arXiv 2611.09501, 2026
    EMeans(EMeansParams),
}

////////////
// Recipe //
////////////

/// A resolved e-means training plan.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EMeansRecipe {
    /// Rows per batch update
    pub batch_size: usize,
    /// Initial EMA step
    pub alpha0: f64,
    /// Rows consumed per epoch; a fractional epoch is a prefix
    pub epoch_rows: Vec<usize>,
    /// Batch updates over the whole run
    pub total_steps: usize,
    /// Revival threshold, 0 for off
    pub unused_threshold: f64,
}

/// Round a positive value to the nearest power of two, ties going up.
///
/// ### Params
///
/// * `value` - Value to round
///
/// ### Returns
///
/// The nearest power of two, at least 1
fn round_power_of_two(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    let low = 2f64.powi(value.log2().floor() as i32);
    let high = 2.0 * low;
    if value / low < high / value {
        low as usize
    } else {
        high as usize
    }
}

/// Fitted initial EMA step for a batch size, clamped to `[1e-4, 1]`.
///
/// ### Params
///
/// * `n` - Training rows
/// * `k` - Number of centroids
/// * `epochs` - Passes over the data
/// * `batch_size` - Resolved batch size
///
/// ### Returns
///
/// The initial step
fn step_law(n: usize, k: usize, epochs: f64, batch_size: usize) -> f64 {
    let (c, ak, ab, an, at) = STEP_LAW;
    let alpha0 = c
        * (k as f64).powf(ak)
        * (batch_size as f64).powf(ab)
        * (n as f64).powf(an)
        * epochs.powf(at);
    alpha0.clamp(ALPHA0_MIN, 1.0)
}

/// Rows per epoch: whole passes, a rounded fractional prefix, then a closing
/// full pass. Below one pass only the prefix runs.
///
/// ### Params
///
/// * `n` - Training rows
/// * `epochs` - Passes over the data
///
/// ### Returns
///
/// Rows consumed in each epoch
fn epoch_rows(n: usize, epochs: f64) -> Vec<usize> {
    let whole = epochs.floor() as usize;
    let remainder = ((epochs - epochs.floor()) * n as f64).round_ties_even() as usize;
    if whole == 0 {
        return vec![remainder];
    }
    let mut rows = vec![n; whole - 1];
    if remainder > 0 {
        rows.push(remainder);
    }
    rows.push(n);
    rows
}

/// Cosine-annealed EMA step, decaying to `ALPHA_MIN_FACTOR * alpha0`.
///
/// ### Params
///
/// * `alpha0` - Initial step
/// * `step` - Updates done so far
/// * `total_steps` - Updates over the whole run
///
/// ### Returns
///
/// The step for this update
pub(crate) fn scheduled_alpha(alpha0: f64, step: usize, total_steps: usize) -> f64 {
    let fraction = step.min(total_steps) as f64 / total_steps.max(1) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

/// Resolve [`EMeansParams`] into a concrete plan for one run.
///
/// ### Params
///
/// * `n` - Training rows
/// * `k` - Number of centroids
/// * `dim` - Data dimensionality (unpadded); picks the auto batch law and the
///   revival default
/// * `params` - The user parameters
///
/// ### Returns
///
/// The [`EMeansRecipe`]
///
/// ### Errors
///
/// `InvalidEMeansParam` for non-positive or non-finite epochs, an epoch count
/// that rounds to zero rows, a zero batch, a step outside `(0, 1]` or a
/// negative revival threshold.
pub(crate) fn resolve_e_means_recipe(
    n: usize,
    k: usize,
    dim: usize,
    params: &EMeansParams,
) -> Result<EMeansRecipe, AnnSearchErrors> {
    let invalid = |name, value, reason| AnnSearchErrors::InvalidEMeansParam {
        name,
        value,
        reason,
    };
    let epochs = params.epochs;
    if !(epochs.is_finite() && epochs > 0.0) {
        return Err(invalid("epochs", epochs, "must be finite and positive"));
    }
    if let Some(a) = params.alpha0 {
        if !(a > 0.0 && a <= 1.0) {
            return Err(invalid("alpha0", a, "must lie in (0, 1]"));
        }
    }
    if let Some(t) = params.unused_threshold {
        if !(t.is_finite() && t >= 0.0) {
            return Err(invalid("unused_threshold", t, "must be finite and >= 0"));
        }
    }

    let batch_size = match params.batch_size {
        Some(0) => return Err(invalid("batch_size", 0.0, "must be at least 1")),
        Some(b) => b.min(n),
        None => {
            let quality = match params.batch_law {
                EMeansBatchLaw::Auto => dim <= AUTO_QUALITY_MAX_D,
                EMeansBatchLaw::Throughput => false,
                EMeansBatchLaw::Quality => true,
            };
            let (c, ak, an, at) = if quality { QUALITY_LAW } else { THROUGHPUT_LAW };
            let raw = c * (k as f64).powf(ak) * (n as f64).powf(an) * epochs.powf(at);
            let mut b = round_power_of_two(raw);
            // The two caps differ on purpose: they reproduce the fitted recipes.
            b = if quality {
                b.min(n / 2)
            } else {
                b.min(round_power_of_two(n as f64 / 2.0))
            };
            b.min(n).max(1)
        }
    };

    let alpha0 = params
        .alpha0
        .unwrap_or_else(|| step_law(n, k, epochs, batch_size));
    let epoch_rows = epoch_rows(n, epochs);
    if epoch_rows.iter().sum::<usize>() == 0 {
        return Err(invalid("epochs", epochs, "rounds to zero training rows"));
    }
    let total_steps = epoch_rows.iter().map(|r| r.div_ceil(batch_size)).sum();
    let unused_threshold = params
        .unused_threshold
        .unwrap_or(if dim <= NO_REVIVAL_MAX_D {
            0.0
        } else {
            UNUSED_THRESHOLD
        });

    Ok(EMeansRecipe {
        batch_size,
        alpha0,
        epoch_rows,
        total_steps,
        unused_threshold,
    })
}

////////////
// Driver //
////////////

/// What a backend reports for one batch.
pub(crate) struct BatchStats<T> {
    /// Per-cluster batch means, `k * dim`; only rows with a non-zero count
    /// are read
    pub means: Vec<T>,
    /// Batch rows per cluster, `k`
    pub counts: Vec<u32>,
    /// Nearest centroid per batch row, in batch order. Only filled when the
    /// driver asks for labels.
    pub labels: Vec<u32>,
}

/// Apply one batch's statistics through the EMA (Algorithm 1, line 8).
///
/// The sum-and-count EMA in position form: with mass `N_j` and centroid
/// `c_j = S_j / N_j`, a touched cluster gets `N_j <- (1 - a) N_j + a n_j` and
/// `c_j <- c_j + (a n_j / N_j)(mean_j - c_j)`. Untouched clusters keep their
/// position and only lose mass.
///
/// ### Params
///
/// * `centroids` - In/out: `k * dim` centroid positions
/// * `usage` - In/out: EMA mass per cluster
/// * `stats` - The batch statistics
/// * `alpha` - Step for this update
/// * `dim` - Row length
fn apply_ema<T>(
    centroids: &mut [T],
    usage: &mut [f64],
    stats: &BatchStats<T>,
    alpha: f64,
    dim: usize,
) where
    T: Float + Send + Sync,
{
    let decay = 1.0 - alpha;
    centroids
        .par_chunks_mut(dim)
        .zip(usage.par_iter_mut())
        .zip(stats.counts.par_iter())
        .zip(stats.means.par_chunks(dim))
        .for_each(|(((row, mass), &count), mean)| {
            *mass *= decay;
            if count == 0 {
                return;
            }
            let added = alpha * count as f64;
            *mass += added;
            let w = added / *mass;
            if w >= 1.0 {
                row.copy_from_slice(mean);
            } else {
                let wt = T::from(w).unwrap();
                for (c, &m) in row.iter_mut().zip(mean) {
                    *c = *c + wt * (m - *c);
                }
            }
        });
}

/// Reseed starved clusters by splitting busy ones (Algorithm 1, line 9).
///
/// Clusters with mass under `threshold * mean mass` are targets. Donors are
/// drawn without replacement in proportion to their mass via Gumbel top-k
/// keys. Each donor is split into two copies nudged by `+-SPLIT_EPSILON` in
/// alternating coordinates, the child taking the target's slot, and the two
/// share the donor's mass.
///
/// ### Params
///
/// * `centroids` - In/out: `k * dim` centroid positions
/// * `usage` - In/out: EMA mass per cluster
/// * `threshold` - Revival threshold
/// * `dim` - Row length
/// * `rng` - Random source for the donor draw
///
/// ### Returns
///
/// Number of clusters revived
fn revive_unused<T>(
    centroids: &mut [T],
    usage: &mut [f64],
    threshold: f64,
    dim: usize,
    rng: &mut StdRng,
) -> usize
where
    T: Float,
{
    let k = usage.len();
    let cut = usage.iter().sum::<f64>() * threshold / k as f64;
    let unused: Vec<usize> = (0..k).filter(|&j| usage[j] < cut).collect();
    let mut donors: Vec<(f64, usize)> = (0..k)
        .filter(|&j| usage[j] >= cut && usage[j] > 0.0)
        .map(|j| {
            let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
            (usage[j].ln() - (-u.ln()).ln(), j)
        })
        .collect();
    let n_split = unused.len().min(donors.len());
    if n_split == 0 {
        return 0;
    }
    donors.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let eps = T::from(SPLIT_EPSILON).unwrap();
    for (&(_, src), &dst) in donors.iter().zip(&unused).take(n_split) {
        let mut collapsed = true;
        for d in 0..dim {
            let v = centroids[src * dim + d];
            let off = if d % 2 == 0 { eps } else { -eps };
            let donor = v * (T::one() - off);
            let child = v * (T::one() + off);
            collapsed &= donor == child;
            centroids[src * dim + d] = donor;
            centroids[dst * dim + d] = child;
        }
        // A zero donor splits into two identical rows; nudge the child's first
        // coordinate so the pair is distinguishable.
        if collapsed && dim > 0 {
            let v = centroids[dst * dim];
            centroids[dst * dim] = if v == T::zero() {
                T::min_positive_value()
            } else {
                v + v.abs() * T::epsilon()
            };
        }
        let half = usage[src] / 2.0;
        usage[src] = half;
        usage[dst] = half;
    }
    n_split
}

/// Derive one epoch's shuffle seed from the run seed.
///
/// ### Params
///
/// * `seed` - Run seed
/// * `epoch` - Epoch index
///
/// ### Returns
///
/// The epoch seed
fn epoch_seed(seed: usize, epoch: usize) -> u64 {
    ((seed as u64)
        .wrapping_mul(1_000_003)
        .wrapping_add(epoch as u64))
        % (1u64 << 32)
}

/// Run e-means (Algorithm 1) around a backend's batch statistics.
///
/// Each epoch draws a fresh permutation of the rows and walks it in batches.
/// Per batch the backend assigns the rows against the current centroids and
/// reports means and counts; the host applies the EMA, revives starved
/// clusters every `CHECK_UNUSED_EVERY` updates, and in the last epoch keeps
/// exact per-cluster sums. Those give the closing means (line 12), so the
/// returned centroids are the means of the final pass's assignments at no
/// extra cost. Clusters the final pass never touched keep their EMA position.
///
/// ### Params
///
/// * `n` - Training rows
/// * `dim` - Row length of the centroids, padded if the backend pads
/// * `k` - Number of centroids
/// * `centroids` - Initial `k * dim` centroids
/// * `recipe` - Resolved plan from [`resolve_e_means_recipe`]
/// * `seed` - Seeds the epoch shuffles and the revival draws
/// * `capture_labels` - Whether to keep each row's last-pass label
/// * `verbosity` - Prints one line per epoch at `Detailed`
/// * `batch_fn` - Backend step: `(rows, centroids, want_labels, out)`. Must
///   fill `out.means` and `out.counts`, and `out.labels` in row order when
///   `want_labels` is set.
///
/// ### Returns
///
/// `(centroids, labels)`. `labels` is empty unless `capture_labels`, and
/// holds `u32::MAX` for rows the last epoch skipped (fewer than one epoch).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_e_means<T, F>(
    n: usize,
    dim: usize,
    k: usize,
    mut centroids: Vec<T>,
    recipe: &EMeansRecipe,
    seed: usize,
    capture_labels: bool,
    verbosity: Verbosity,
    mut batch_fn: F,
) -> Result<(Vec<T>, Vec<u32>), AnnSearchErrors>
where
    T: Float + Send + Sync,
    F: FnMut(&[u32], &[T], bool, &mut BatchStats<T>) -> Result<(), AnnSearchErrors>,
{
    let start = Instant::now();
    let mut usage = vec![INIT_USAGE; k];
    let mut split_rng = StdRng::seed_from_u64(seed as u64 ^ 0x9E37_79B9_7F4A_7C15);
    let mut perm: Vec<u32> = (0..n as u32).collect();
    let mut stats = BatchStats {
        means: vec![T::zero(); k * dim],
        counts: vec![0u32; k],
        labels: Vec::new(),
    };
    let mut labels = if capture_labels {
        vec![u32::MAX; n]
    } else {
        Vec::new()
    };
    let mut pass_sums = vec![0f64; k * dim];
    let mut pass_counts = vec![0u64; k];

    let n_epochs = recipe.epoch_rows.len();
    let mut step = 0usize;
    for (epoch, &rows) in recipe.epoch_rows.iter().enumerate() {
        perm.shuffle(&mut StdRng::seed_from_u64(epoch_seed(seed, epoch)));
        let last = epoch + 1 == n_epochs;
        let mut revived = 0usize;

        for b0 in (0..rows).step_by(recipe.batch_size) {
            let batch = &perm[b0..(b0 + recipe.batch_size).min(rows)];
            let want = last && capture_labels;
            batch_fn(batch, &centroids, want, &mut stats)?;

            if last {
                pass_sums
                    .par_chunks_mut(dim)
                    .zip(stats.means.par_chunks(dim))
                    .zip(stats.counts.par_iter())
                    .for_each(|((sum, mean), &count)| {
                        if count > 0 {
                            for (s, &m) in sum.iter_mut().zip(mean) {
                                *s += m.to_f64().unwrap() * count as f64;
                            }
                        }
                    });
                for (pc, &c) in pass_counts.iter_mut().zip(&stats.counts) {
                    *pc += c as u64;
                }
                if want {
                    for (&row, &l) in batch.iter().zip(&stats.labels) {
                        labels[row as usize] = l;
                    }
                }
            }

            let alpha = scheduled_alpha(recipe.alpha0, step, recipe.total_steps);
            apply_ema(&mut centroids, &mut usage, &stats, alpha, dim);
            step += 1;
            if step.is_multiple_of(CHECK_UNUSED_EVERY) && recipe.unused_threshold > 0.0 {
                revived += revive_unused(
                    &mut centroids,
                    &mut usage,
                    recipe.unused_threshold,
                    dim,
                    &mut split_rng,
                );
            }
        }

        if verbosity.detailed_verbosity() {
            println!(
                "  Epoch {}/{n_epochs}: {} clusters revived ({:.2?})",
                epoch + 1,
                revived.separate_with_underscores(),
                start.elapsed()
            );
        }
    }

    centroids
        .par_chunks_mut(dim)
        .zip(pass_sums.par_chunks(dim))
        .zip(pass_counts.par_iter())
        .for_each(|((row, sum), &count)| {
            if count > 0 {
                let inv = 1.0 / count as f64;
                for (c, &s) in row.iter_mut().zip(sum) {
                    *c = T::from(s * inv).unwrap();
                }
            }
        });

    Ok((centroids, labels))
}

/////////
// CPU //
/////////

/// e-means on the CPU.
///
/// Random initialisation from the data, then [`run_e_means`] with a batch
/// step that gathers the batch rows, assigns them through
/// [`assign_all_parallel`] (GEMM from 16 dimensions up), and reduces the
/// per-cluster means over a counting sort of the batch labels, one cluster
/// per rayon task, so no thread holds a `k * dim` accumulator.
///
/// ### Params
///
/// * `data` - Training data, flat row-major `n * dim`
/// * `dim` - Embedding dimensions
/// * `n` - Number of training rows
/// * `k` - Number of centroids
/// * `metric` - `SquaredEuclidean` or `Cosine`
/// * `params` - e-means parameters
/// * `seed` - Seed for the initialisation, shuffles and revival
/// * `verbosity` - Prints the recipe and per-epoch lines at `Detailed`
///
/// ### Returns
///
/// `(centroids, last-pass labels)`, flat `k * dim` and length `n`
#[allow(clippy::too_many_arguments)]
pub(crate) fn e_means_cpu<T>(
    data: &[T],
    dim: usize,
    n: usize,
    k: usize,
    metric: &Dist,
    params: &EMeansParams,
    seed: usize,
    capture_labels: bool,
    verbosity: Verbosity,
) -> Result<(Vec<T>, Vec<u32>), AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    let recipe = resolve_e_means_recipe(n, k, dim, params)?;
    if verbosity.detailed_verbosity() {
        print_recipe(&recipe);
    }
    let init = fast_random_init(data, dim, n, k, seed);

    let mut scratch: Vec<T> = Vec::with_capacity(recipe.batch_size * dim);
    let mut order: Vec<u32> = Vec::with_capacity(recipe.batch_size);
    let mut offsets = vec![0u32; k + 1];

    run_e_means(
        n,
        dim,
        k,
        init,
        &recipe,
        seed,
        capture_labels,
        verbosity,
        |rows, cents, want, out| {
            let b = rows.len();
            scratch.resize(b * dim, T::zero());
            scratch
                .par_chunks_mut(dim)
                .zip(rows.par_iter())
                .for_each(|(dst, &r)| {
                    let r = r as usize;
                    dst.copy_from_slice(&data[r * dim..(r + 1) * dim]);
                });

            let cnorms: Vec<T> = if *metric == Dist::Cosine {
                cents.par_chunks(dim).map(T::calculate_l2_norm).collect()
            } else {
                Vec::new()
            };
            let assign = assign_all_parallel(&scratch, &[], dim, b, cents, &cnorms, k, metric);

            // Counting sort of the batch by cluster.
            out.counts.fill(0);
            for &a in &assign {
                out.counts[a as usize] += 1;
            }
            offsets[0] = 0;
            for j in 0..k {
                offsets[j + 1] = offsets[j] + out.counts[j];
            }
            order.resize(b, 0);
            let mut cursor = offsets[..k].to_vec();
            for (i, &a) in assign.iter().enumerate() {
                order[cursor[a as usize] as usize] = i as u32;
                cursor[a as usize] += 1;
            }

            out.means
                .par_chunks_mut(dim)
                .enumerate()
                .for_each(|(j, mean)| {
                    let (s, e) = (offsets[j] as usize, offsets[j + 1] as usize);
                    if s == e {
                        return;
                    }
                    mean.fill(T::zero());
                    for &i in &order[s..e] {
                        let i = i as usize;
                        T::add_assign_simd(mean, &scratch[i * dim..(i + 1) * dim]);
                    }
                    let inv = T::one() / T::from(e - s).unwrap();
                    for m in mean.iter_mut() {
                        *m *= inv;
                    }
                });

            if want {
                out.labels = assign;
            }
            Ok(())
        },
    )
}

/// Print the resolved recipe as a `Detailed` stage line.
///
/// ### Params
///
/// * `recipe` - The resolved plan
pub(crate) fn print_recipe(recipe: &EMeansRecipe) {
    println!(
        " Running e-means ({} epochs, batch {}, alpha0 {:.3}, {} updates)",
        recipe.epoch_rows.len(),
        recipe.batch_size.separate_with_underscores(),
        recipe.alpha0,
        recipe.total_steps.separate_with_underscores()
    );
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Recipes pinned to the reference `faiss.contrib.e_means.base` output:
    /// `(n, k, dim, epochs, batch, alpha0, epoch_rows)`.
    #[test]
    fn test_e_means_recipe_matches_reference() {
        #[allow(clippy::type_complexity)]
        let cases: [(usize, usize, usize, f64, usize, f64, Vec<usize>); 8] = [
            (
                250_000,
                500,
                128,
                5.0,
                16384,
                0.917046289720882,
                vec![250_000; 5],
            ),
            (
                250_000,
                500,
                8,
                5.0,
                2048,
                0.1410390398649467,
                vec![250_000; 5],
            ),
            (1_000_000, 1024, 64, 5.0, 65536, 1.0, vec![1_000_000; 5]),
            (
                100_000,
                64,
                3,
                3.0,
                512,
                0.1667050355062096,
                vec![100_000; 3],
            ),
            (
                10_000,
                1000,
                32,
                2.5,
                512,
                0.13456479993966902,
                vec![10_000, 5_000, 10_000],
            ),
            (1000, 10, 2, 5.0, 8, 0.06985681308367728, vec![1000; 5]),
            (60_000, 256, 512, 10.0, 8192, 1.0, vec![60_000; 10]),
            (500, 100, 16, 1.0, 2, 0.011213949093399305, vec![500]),
        ];
        for (n, k, dim, epochs, batch, alpha0, rows) in cases {
            let r = resolve_e_means_recipe(n, k, dim, &EMeansParams::new(epochs)).unwrap();
            assert_eq!(r.batch_size, batch, "batch at n={n} k={k} d={dim}");
            assert_relative_eq!(r.alpha0, alpha0, max_relative = 1e-12);
            assert_eq!(r.epoch_rows, rows);
            let steps: usize = rows.iter().map(|x| x.div_ceil(batch)).sum();
            assert_eq!(r.total_steps, steps);
        }
    }

    #[test]
    fn test_e_means_schedule_matches_reference() {
        assert_relative_eq!(scheduled_alpha(0.3, 0, 100), 0.3, max_relative = 1e-15);
        assert_relative_eq!(
            scheduled_alpha(0.3, 37, 100),
            0.21861496523569537,
            max_relative = 1e-14
        );
        assert_relative_eq!(scheduled_alpha(0.3, 100, 100), 0.03, max_relative = 1e-14);
    }

    #[test]
    fn test_round_power_of_two() {
        assert_eq!(round_power_of_two(0.5), 1);
        assert_eq!(round_power_of_two(3.0), 4);
        assert_eq!(round_power_of_two(5.99), 8);
        assert_eq!(round_power_of_two(6.0), 8);
        assert_eq!(round_power_of_two(1024.0), 1024);
    }

    #[test]
    fn test_e_means_recipe_rejects_bad_params() {
        let bad = [
            EMeansParams::new(0.0),
            EMeansParams::new(f64::NAN),
            EMeansParams {
                alpha0: Some(1.5),
                ..Default::default()
            },
            EMeansParams {
                batch_size: Some(0),
                ..Default::default()
            },
            EMeansParams {
                unused_threshold: Some(-1.0),
                ..Default::default()
            },
            EMeansParams::new(1e-9),
        ];
        for p in bad {
            assert!(resolve_e_means_recipe(1000, 10, 8, &p).is_err(), "{p:?}");
        }
    }

    #[test]
    fn test_e_means_revival_defaults_off_at_low_dim() {
        let p = EMeansParams::default();
        assert_eq!(
            resolve_e_means_recipe(1000, 10, 2, &p)
                .unwrap()
                .unused_threshold,
            0.0
        );
        assert_eq!(
            resolve_e_means_recipe(1000, 10, 3, &p)
                .unwrap()
                .unused_threshold,
            0.1
        );
    }

    #[test]
    fn test_revive_splits_heaviest_into_dead() {
        let dim = 4;
        let mut cents = vec![
            1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0,
        ];
        let mut usage = vec![10.0, 0.0, 0.0];
        let mut rng = StdRng::seed_from_u64(1);
        let n = revive_unused(&mut cents, &mut usage, 0.1, dim, &mut rng);
        // Only cluster 0 is a donor, so only one of the two dead clusters
        // revives, the lowest index.
        assert_eq!(n, 1);
        assert_eq!(usage, vec![5.0, 5.0, 0.0]);
        assert!(cents[0] < 1.0 && cents[4] > 1.0);
        assert!(cents[1] > 2.0 && cents[5] < 2.0);
    }

    #[test]
    fn test_revive_separates_zero_donor() {
        let mut cents = vec![0.0f32; 4];
        let mut usage = vec![1.0, 0.0];
        let mut rng = StdRng::seed_from_u64(1);
        revive_unused(&mut cents, &mut usage, 0.1, 2, &mut rng);
        assert_ne!(&cents[0..2], &cents[2..4]);
    }

    #[test]
    fn test_apply_ema_alpha_one_is_lloyd_step() {
        let mut cents = vec![0.0f32, 0.0, 9.0, 9.0];
        let mut usage = vec![INIT_USAGE; 2];
        let stats = BatchStats {
            means: vec![1.0, 2.0, 0.0, 0.0],
            counts: vec![3, 0],
            labels: Vec::new(),
        };
        apply_ema(&mut cents, &mut usage, &stats, 1.0, 2);
        assert_eq!(cents, vec![1.0, 2.0, 9.0, 9.0]);
        assert_eq!(usage, vec![3.0, 0.0]);
    }

    fn blobs(n_per: usize, centres: &[[f32; 2]], seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = Vec::with_capacity(n_per * centres.len() * 2);
        for c in centres {
            for _ in 0..n_per {
                out.push(c[0] + rng.random::<f32>() * 0.1);
                out.push(c[1] + rng.random::<f32>() * 0.1);
            }
        }
        out
    }

    #[test]
    fn test_e_means_cpu_two_blobs() {
        let data = blobs(500, &[[0.0, 0.0], [10.0, 10.0]], 3);
        let (cents, labels) = e_means_cpu(
            &data,
            2,
            1000,
            2,
            &Dist::SquaredEuclidean,
            &EMeansParams::default(),
            42,
            true,
            Verbosity::Quiet,
        )
        .unwrap();
        let mut c: Vec<(f32, f32)> = cents.chunks(2).map(|r| (r[0], r[1])).collect();
        c.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert!((c[0].0 - 0.05).abs() < 0.02 && (c[1].0 - 10.05).abs() < 0.02);
        assert_eq!(labels.len(), 1000);
        assert!(labels[..500].iter().all(|&l| l == labels[0]));
        assert!(labels[500..].iter().all(|&l| l != labels[0]));
    }

    #[test]
    fn test_e_means_cpu_reproducible_and_k_one() {
        let data = blobs(200, &[[0.0, 0.0], [3.0, 1.0], [-2.0, 4.0]], 9);
        let p = EMeansParams::default();
        let run = |k| {
            e_means_cpu(
                &data,
                2,
                600,
                k,
                &Dist::SquaredEuclidean,
                &p,
                7,
                false,
                Verbosity::Quiet,
            )
            .unwrap()
            .0
        };
        assert_eq!(run(3), run(3));
        // k = 1 closes on the exact data mean.
        let c = run(1);
        let mean_x = data.chunks(2).map(|r| r[0] as f64).sum::<f64>() / 600.0;
        assert_relative_eq!(c[0] as f64, mean_x, max_relative = 1e-5);
    }

    #[test]
    fn test_e_means_cpu_k_equals_n() {
        let data = blobs(5, &[[0.0, 0.0], [1.0, 1.0]], 2);
        let (cents, _) = e_means_cpu(
            &data,
            2,
            10,
            10,
            &Dist::SquaredEuclidean,
            &EMeansParams::default(),
            1,
            false,
            Verbosity::Quiet,
        )
        .unwrap();
        assert_eq!(cents.len(), 20);
        assert!(cents.iter().all(|v| v.is_finite()));
    }
}
