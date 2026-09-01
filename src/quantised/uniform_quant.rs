//! Uniform scalar quantisation to 8-bit codes.
//!
//! Per-dimension offsets, one scale shared by every dimension. The shared scale
//! is the load-bearing part: with `x_j = s * c_j + b_j`, a difference is
//! `x_j - y_j = s * (c_j - d_j)`, so the offsets cancel and the scale factors
//! out. The integer distance between two codes therefore preserves the exact
//! ordering of the float distance, which is what lets one kernel serve both
//! graph construction and query. Per-dimension *scales* would break that; the
//! offsets are free.
//!
//! The offsets do not cancel for an inner product, so that path carries one
//! extra precomputed scalar per vector. See [`UniformQuantiser::inner_product`].
//!
//! Calibration trims a configurable fraction from each tail before fixing the
//! range. With a single shared scale the widest dimension sets the resolution
//! for all of them, so one heavy-tailed dimension would otherwise starve the
//! rest.

use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use rayon::prelude::*;
use std::cmp::Ordering;

use crate::prelude::*;
use crate::quantised::int_kernels::*;

////////////
// Consts //
////////////

/// Highest code value. Codes are whole bytes because the integer dot product
/// only vectorises at byte granularity; see the [`crate::quantised::int_kernels`]
/// module documentation.
pub const MAX_LEVEL: u8 = 255;

/// Rows sampled for calibration when the dataset is larger than this.
///
/// Quantiles of a tail are estimated from order statistics, and 100k rows pins
/// a 0.1% tail to roughly 100 observations. Sampling more moves the boundary by
/// less than one code level whilst the selection cost keeps growing.
const CALIBRATION_SAMPLE_ROWS: usize = 100_000;

/// Fraction trimmed from each tail by default.
///
/// Non-zero on purpose. Single-cell latent spaces routinely carry a handful of
/// extreme cells per dimension, and under a shared scale one of them would set
/// the range for the whole index.
const DEFAULT_DROP_RATIO: f64 = 1e-3;

////////////////////////
// UniformQuantParams //
////////////////////////

/// Calibration settings for [`UniformQuantiser`].
#[derive(Clone, Copy, Debug)]
pub struct UniformQuantParams {
    /// Fraction trimmed from *each* tail of every dimension before the range
    /// is fixed. Values outside the trimmed range clamp to the end codes.
    /// Must be in `[0, 0.5)`.
    pub drop_ratio: f64,
    /// Rows sampled for calibration. `None` auto-picks, capped at the dataset size.
    pub sample_rows: Option<usize>,
    /// Seed for the calibration row sample.
    pub seed: usize,
}

impl UniformQuantParams {
    /// Create calibration settings.
    ///
    /// ### Params
    ///
    /// * `drop_ratio` - Fraction trimmed from each tail, in `[0, 0.5)`
    /// * `sample_rows` - Rows sampled for calibration, `None` to auto-pick
    /// * `seed` - Seed for the row sample
    ///
    /// ### Returns
    ///
    /// The parameter struct
    pub fn new(drop_ratio: f64, sample_rows: Option<usize>, seed: usize) -> Self {
        Self {
            drop_ratio,
            sample_rows,
            seed,
        }
    }
}

impl Default for UniformQuantParams {
    fn default() -> Self {
        Self {
            drop_ratio: DEFAULT_DROP_RATIO,
            sample_rows: None,
            seed: 42,
        }
    }
}

//////////////////////
// UniformQuantiser //
//////////////////////

/// Uniform scalar quantiser with per-dimension offsets and a shared scale.
#[cfg_attr(
    feature = "serialise",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "")
)]
pub struct UniformQuantiser<T>
where
    T: AnnSearchFloat,
{
    /// Per-dimension offset, the trimmed minimum of each dimension.
    offsets: Vec<T>,
    /// Sum of squared offsets, the constant term of an inner product.
    offsets_norm_sq: T,
    /// Scale shared by every dimension: the widest trimmed range over
    /// [`MAX_LEVEL`].
    scale: T,
    /// Squared scale, the factor converting a code-space squared distance into
    /// a data-space one.
    scale_sq: T,
    /// Reciprocal of `scale`, so encoding multiplies instead of divides.
    inv_scale: T,
    /// Dimensionality of the vectors this quantiser was trained on.
    dim: usize,
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for UniformQuantiser<T>
where
    T: AnnSearchFloat,
{
    fn dim(&self) -> usize {
        self.dim
    }
}

impl<T> UniformQuantiser<T>
where
    T: AnnSearchFloat,
{
    /// Calibrate a quantiser against a dataset.
    ///
    /// Samples rows, takes the trimmed minimum and maximum of each dimension
    /// independently, then fixes one shared scale from the widest trimmed
    /// range. Dimensions narrower than the widest simply use fewer of the
    /// available code levels; that is the cost of the shared scale and the
    /// reason the tail trim matters.
    ///
    /// ### Params
    ///
    /// * `data` - Row-major flattened vectors of length `n * dim`
    /// * `n` - Number of vectors
    /// * `dim` - Dimensionality
    /// * `params` - Calibration settings, `None` for [`UniformQuantParams::default`]
    ///
    /// ### Returns
    ///
    /// The calibrated quantiser, or an error if `drop_ratio` is out of range
    /// or the inputs are empty.
    pub fn train(
        data: &[T],
        n: usize,
        dim: usize,
        params: Option<UniformQuantParams>,
    ) -> Result<Self, AnnSearchErrors> {
        let params = params.unwrap_or_default();

        if !(0.0..0.5).contains(&params.drop_ratio) || !params.drop_ratio.is_finite() {
            return Err(AnnSearchErrors::InvalidDropRatio {
                drop_ratio: params.drop_ratio,
            });
        }
        if n == 0 || dim == 0 {
            return Err(AnnSearchErrors::EmptyCalibrationSet { n, dim });
        }

        let sample = Self::sample_rows(data, n, dim, &params);
        let m = sample.len() / dim;

        // Order statistics of each tail. `m - 1 - lower` rather than a second
        // ratio computation keeps the two tails symmetric even when rounding
        // pushes them apart.
        let lower = ((m as f64) * params.drop_ratio).floor() as usize;
        let lower = lower.min(m.saturating_sub(1) / 2);
        let upper = m - 1 - lower;

        let bounds: Vec<(T, T)> = (0..dim)
            .into_par_iter()
            .map(|j| {
                let mut column: Vec<T> = (0..m).map(|i| sample[i * dim + j]).collect();
                let cmp = |a: &T, b: &T| a.partial_cmp(b).unwrap_or(Ordering::Equal);
                // Two selections rather than a sort: the tails are all that is
                // wanted and selection is linear.
                let (_, lo, rest) = column.select_nth_unstable_by(lower, cmp);
                let lo = *lo;
                // `rest` starts at `lower + 1`, so shift the target index.
                let hi = if upper > lower {
                    let (_, hi, _) = rest.select_nth_unstable_by(upper - lower - 1, cmp);
                    *hi
                } else {
                    lo
                };
                (lo, hi)
            })
            .collect();

        let max_level = T::from_u8(MAX_LEVEL).unwrap();
        let widest = bounds
            .iter()
            .map(|&(lo, hi)| hi - lo)
            .fold(T::zero(), |acc, r| if r > acc { r } else { acc });

        // A constant dataset has no range to divide by. Unit scale makes every
        // code zero, every distance zero, which is the correct answer.
        let scale = if widest > T::zero() {
            widest / max_level
        } else {
            T::one()
        };

        let offsets: Vec<T> = bounds.iter().map(|&(lo, _)| lo).collect();
        let offsets_norm_sq = offsets.iter().fold(T::zero(), |acc, &b| acc + b * b);

        Ok(Self {
            offsets,
            offsets_norm_sq,
            scale,
            scale_sq: scale * scale,
            inv_scale: T::one() / scale,
            dim,
        })
    }

    /// Draw the calibration row sample.
    ///
    /// Returns the whole dataset when it is small enough to use directly,
    /// otherwise a shuffled subset of rows copied out row-wise.
    ///
    /// ### Params
    ///
    /// * `data` - Row-major flattened vectors
    /// * `n` - Number of vectors
    /// * `dim` - Dimensionality
    /// * `params` - Calibration settings
    ///
    /// ### Returns
    ///
    /// Row-major sample of length `m * dim`
    fn sample_rows(data: &[T], n: usize, dim: usize, params: &UniformQuantParams) -> Vec<T> {
        let want = params
            .sample_rows
            .unwrap_or(CALIBRATION_SAMPLE_ROWS)
            .clamp(1, n);

        if want == n {
            return data[..n * dim].to_vec();
        }

        let mut rng = SmallRng::seed_from_u64(params.seed as u64);
        let mut ids: Vec<usize> = (0..n).collect();
        ids.shuffle(&mut rng);

        let mut out = Vec::with_capacity(want * dim);
        for &i in ids.iter().take(want) {
            out.extend_from_slice(&data[i * dim..(i + 1) * dim]);
        }
        out
    }

    /// Encode a vector into an existing code buffer.
    ///
    /// Values outside the calibrated range clamp to the end codes rather than
    /// wrapping. A non-finite input encodes to zero.
    ///
    /// ### Params
    ///
    /// * `vec` - Vector of length `dim`
    /// * `out` - Code buffer of length `dim`, overwritten in full
    #[inline]
    pub fn encode_into(&self, vec: &[T], out: &mut [u8]) {
        debug_assert_eq!(vec.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);

        let max_level = T::from_u8(MAX_LEVEL).unwrap();
        for j in 0..self.dim {
            let v = (vec[j] - self.offsets[j]) * self.inv_scale;
            // `Float::max` returns the non-NaN operand, so NaN lands on zero.
            let v = v.max(T::zero()).min(max_level);
            out[j] = v.round().to_u8().unwrap_or(0);
        }
    }

    /// Encode a vector into a freshly allocated code buffer.
    ///
    /// ### Params
    ///
    /// * `vec` - Vector of length `dim`
    ///
    /// ### Returns
    ///
    /// The code vector, or an error if `vec` has the wrong length
    #[inline]
    pub fn encode(&self, vec: &[T]) -> Result<Vec<u8>, AnnSearchErrors> {
        self.check_dim(vec.len())?;
        let mut out = vec![0u8; self.dim];
        self.encode_into(vec, &mut out);
        Ok(out)
    }

    /// Reconstruct an approximate vector from its code.
    ///
    /// ### Params
    ///
    /// * `code` - Code vector of length `dim`
    ///
    /// ### Returns
    ///
    /// The dequantised vector
    #[inline]
    pub fn decode(&self, code: &[u8]) -> Vec<T> {
        debug_assert_eq!(code.len(), self.dim);
        (0..self.dim)
            .map(|j| T::from_u8(code[j]).unwrap() * self.scale + self.offsets[j])
            .collect()
    }

    /// Squared Euclidean distance between two codes, in data space.
    ///
    /// The integer part preserves the exact ordering on its own; the scaling
    /// only restores the magnitude. Callers ranking candidates can skip it and
    /// compare [`sq_dist_from_dot`] directly.
    ///
    /// ### Params
    ///
    /// * `a` - First code vector
    /// * `b` - Second code vector
    /// * `norm_a` - Precomputed [`norm_sq_u8`] of `a`
    /// * `norm_b` - Precomputed [`norm_sq_u8`] of `b`
    ///
    /// ### Returns
    ///
    /// The approximate squared Euclidean distance
    #[inline]
    pub fn squared_euclidean(&self, a: &[u8], b: &[u8], norm_a: u32, norm_b: u32) -> T {
        let d = sq_dist_from_dot(a, b, norm_a, norm_b);
        T::from_i64(d).unwrap() * self.scale_sq
    }

    /// Per-vector correction term for the inner-product path.
    ///
    /// `sum_j offsets[j] * code[j]`. Unlike a difference, an inner product does
    /// not cancel the offsets, so each vector carries this scalar and the
    /// constant `sum_j offsets[j]^2` lives on the quantiser.
    ///
    /// ### Params
    ///
    /// * `code` - Code vector of length `dim`
    ///
    /// ### Returns
    ///
    /// The correction term
    #[inline]
    pub fn offset_dot(&self, code: &[u8]) -> T {
        let mut acc = T::zero();
        for j in 0..self.dim {
            acc = acc + self.offsets[j] * T::from_u8(code[j]).unwrap();
        }
        acc
    }

    /// Inner product between two codes, in data space.
    ///
    /// Expands `sum_j (s*a_j + b_j)(s*b_j + b_j)` into the integer dot product
    /// plus the two per-vector corrections and the shared constant.
    ///
    /// ### Params
    ///
    /// * `a` - First code vector
    /// * `b` - Second code vector
    /// * `offset_dot_a` - [`Self::offset_dot`] of `a`
    /// * `offset_dot_b` - [`Self::offset_dot`] of `b`
    ///
    /// ### Returns
    ///
    /// The approximate inner product
    #[inline]
    pub fn inner_product(&self, a: &[u8], b: &[u8], offset_dot_a: T, offset_dot_b: T) -> T {
        let dot = T::from_u32(dot_u8(a, b)).unwrap();
        self.scale_sq * dot + self.scale * (offset_dot_a + offset_dot_b) + self.offsets_norm_sq
    }

    /// The shared scale.
    ///
    /// ### Returns
    ///
    /// Data-space width of one code level
    #[inline]
    pub fn scale(&self) -> T {
        self.scale
    }

    /// The squared shared scale.
    ///
    /// Factor converting a code-space squared distance into a data-space one.
    /// Exposed so the search path can rank on the raw integer distance and pay
    /// the multiply once per returned neighbour rather than once per node
    /// visited.
    ///
    /// ### Returns
    ///
    /// `scale^2`
    #[inline]
    pub fn scale_sq(&self) -> T {
        self.scale_sq
    }

    /// The per-dimension offsets.
    ///
    /// ### Returns
    ///
    /// Slice of length `dim`
    #[inline]
    pub fn offsets(&self) -> &[T] {
        &self.offsets
    }

    /// Bytes held by the quantiser.
    ///
    /// ### Returns
    ///
    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.offsets.capacity() * std::mem::size_of::<T>()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Deterministic pseudo-random data whose per-dimension spread decays with
    /// the dimension index, mimicking PCA component variance decay. That decay
    /// is the awkward case for a shared scale, so it is what the tests use.
    fn pca_like<T: AnnSearchFloat>(n: usize, dim: usize) -> Vec<T> {
        let mut s = 0x2545F491u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        (0..n * dim)
            .map(|i| {
                let spread = 1.0 / (1.0 + (i % dim) as f64);
                T::from_f64(next() * spread).unwrap()
            })
            .collect()
    }

    fn exact_sq_l2<T: AnnSearchFloat>(a: &[T], b: &[T]) -> T {
        a.iter()
            .zip(b)
            .fold(T::zero(), |acc, (&x, &y)| acc + (x - y) * (x - y))
    }

    #[test]
    fn test_train_rejects_bad_drop_ratio() {
        let data = pca_like::<f32>(50, 4);
        for bad in [-0.1, 0.5, 0.9, f64::NAN] {
            let params = UniformQuantParams::new(bad, None, 1);
            assert!(UniformQuantiser::train(&data, 50, 4, Some(params)).is_err());
        }
    }

    #[test]
    fn test_train_rejects_empty_input() {
        let data: Vec<f32> = Vec::new();
        assert!(UniformQuantiser::<f32>::train(&data, 0, 4, None).is_err());
        assert!(UniformQuantiser::<f32>::train(&data, 4, 0, None).is_err());
    }

    #[test]
    fn test_encode_decode_round_trip_within_one_level() {
        let (n, dim) = (2000, 16);
        let data = pca_like::<f32>(n, dim);
        // No trim, so nothing clips and every value is inside the range.
        let params = UniformQuantParams::new(0.0, None, 7);
        let q = UniformQuantiser::train(&data, n, dim, Some(params)).unwrap();

        let half_level = q.scale() * 0.5;
        for i in 0..n {
            let row = &data[i * dim..(i + 1) * dim];
            let code = q.encode(row).unwrap();
            let back = q.decode(&code);
            for j in 0..dim {
                assert!(
                    (back[j] - row[j]).abs() <= half_level * 1.001,
                    "dim {j}: {} vs {}, half level {half_level}",
                    back[j],
                    row[j]
                );
            }
        }
    }

    #[test]
    fn test_constant_data_does_not_divide_by_zero() {
        let (n, dim) = (32, 8);
        let data = vec![3.5f32; n * dim];
        let q = UniformQuantiser::train(&data, n, dim, None).unwrap();
        let code = q.encode(&data[..dim]).unwrap();
        assert!(code.iter().all(|&c| c == 0));
        let back = q.decode(&code);
        assert!(back.iter().all(|&x| (x - 3.5).abs() < 1e-6));
    }

    #[test]
    fn test_per_dimension_offsets_cancel_out_of_code_distances() {
        // The invariant the shared scale buys: shifting each dimension by its
        // own constant moves the offsets and nothing else, so the codes and
        // every integer distance come out bit-identical. Per-dimension
        // *scales* would break this, which is why they are not on offer.
        // f64 on purpose. The property is algebraic, but adding a shift of
        // order 100 to data of order 1 costs f32 seven bits of mantissa, which
        // is enough to move the odd code by one level and would make this test
        // measure float precision rather than the invariant.
        let (n, dim) = (300, 12);
        let data = pca_like::<f64>(n, dim);
        let shifted: Vec<f64> = data
            .iter()
            .enumerate()
            .map(|(i, &x)| x + (i % dim) as f64 * 3.5 + 100.0)
            .collect();

        let params = UniformQuantParams::new(0.0, None, 5);
        let a = UniformQuantiser::train(&data, n, dim, Some(params)).unwrap();
        let b = UniformQuantiser::train(&shifted, n, dim, Some(params)).unwrap();

        assert_relative_eq!(a.scale(), b.scale(), max_relative = 1e-12);

        let codes_a: Vec<Vec<u8>> = (0..n)
            .map(|i| a.encode(&data[i * dim..(i + 1) * dim]).unwrap())
            .collect();
        let codes_b: Vec<Vec<u8>> = (0..n)
            .map(|i| b.encode(&shifted[i * dim..(i + 1) * dim]).unwrap())
            .collect();
        for i in 0..n {
            assert_eq!(
                codes_a[i], codes_b[i],
                "codes diverged at row {i} under a per-dimension shift"
            );
        }

        let norms_a: Vec<u32> = codes_a.iter().map(|c| norm_sq_u8(c)).collect();
        let norms_b: Vec<u32> = codes_b.iter().map(|c| norm_sq_u8(c)).collect();
        for i in 1..n {
            assert_eq!(
                sq_dist_from_dot(&codes_a[0], &codes_a[i], norms_a[0], norms_a[i]),
                sq_dist_from_dot(&codes_b[0], &codes_b[i], norms_b[0], norms_b[i]),
                "distance to {i} moved under a per-dimension shift"
            );
        }
    }

    #[test]
    fn test_integer_ranking_is_monotone_in_the_dequantised_distance() {
        // Ranking on the raw integer distance must never contradict the
        // dequantised one. Exact ties in the integer domain may come back in
        // either order once f32 rounding is involved, so the assertion is
        // monotonicity rather than an exact permutation match.
        let (n, dim) = (400, 12);
        let data = pca_like::<f32>(n, dim);
        let q = UniformQuantiser::train(&data, n, dim, None).unwrap();

        let codes: Vec<Vec<u8>> = (0..n)
            .map(|i| q.encode(&data[i * dim..(i + 1) * dim]).unwrap())
            .collect();
        let norms: Vec<u32> = codes.iter().map(|c| norm_sq_u8(c)).collect();
        let decoded_query = q.decode(&codes[0]);

        let mut ranked: Vec<(i64, f32)> = (1..n)
            .map(|i| {
                (
                    sq_dist_from_dot(&codes[0], &codes[i], norms[0], norms[i]),
                    exact_sq_l2(&decoded_query, &q.decode(&codes[i])),
                )
            })
            .collect();
        ranked.sort_by_key(|&(d, _)| d);

        // Relative, because the reference is an f32 sum of `dim` terms and
        // exact integer ties may come back either way round.
        for w in ranked.windows(2) {
            assert!(
                w[1].1 >= w[0].1 - w[0].1.abs() * 1e-4,
                "integer distances {} <= {} but float distances are {} > {}",
                w[0].0,
                w[1].0,
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn test_squared_euclidean_matches_dequantised_distance() {
        let (n, dim) = (200, 10);
        let data = pca_like::<f32>(n, dim);
        let q = UniformQuantiser::train(&data, n, dim, None).unwrap();

        let ca = q.encode(&data[0..dim]).unwrap();
        let cb = q.encode(&data[dim..2 * dim]).unwrap();
        let (na, nb) = (norm_sq_u8(&ca), norm_sq_u8(&cb));

        let got = q.squared_euclidean(&ca, &cb, na, nb);
        let want = exact_sq_l2(&q.decode(&ca), &q.decode(&cb));
        assert_relative_eq!(got, want, epsilon = 1e-4, max_relative = 1e-4);
    }

    #[test]
    fn test_inner_product_matches_dequantised_dot() {
        // The offsets do not cancel here, so this is the test that catches a
        // dropped correction term.
        let (n, dim) = (200, 10);
        let data = pca_like::<f32>(n, dim);
        let q = UniformQuantiser::train(&data, n, dim, None).unwrap();

        let ca = q.encode(&data[0..dim]).unwrap();
        let cb = q.encode(&data[dim..2 * dim]).unwrap();

        let got = q.inner_product(&ca, &cb, q.offset_dot(&ca), q.offset_dot(&cb));
        let (da, db) = (q.decode(&ca), q.decode(&cb));
        let want: f32 = da.iter().zip(&db).map(|(x, y)| x * y).sum();
        assert_relative_eq!(got, want, epsilon = 1e-4, max_relative = 1e-4);
    }

    #[test]
    fn test_drop_ratio_trims_outliers_and_clamps() {
        let (n, dim) = (1000, 4);
        let mut data = pca_like::<f32>(n, dim);
        // One cell with an extreme value in dimension 0.
        data[0] = 1000.0;

        let untrimmed =
            UniformQuantiser::train(&data, n, dim, Some(UniformQuantParams::new(0.0, None, 1)))
                .unwrap();
        let trimmed =
            UniformQuantiser::train(&data, n, dim, Some(UniformQuantParams::new(0.01, None, 1)))
                .unwrap();

        // Without the trim the outlier sets the shared scale for every
        // dimension; with it the scale collapses to the bulk of the data.
        assert!(
            trimmed.scale() < untrimmed.scale() * 0.01,
            "trimmed {} vs untrimmed {}",
            trimmed.scale(),
            untrimmed.scale()
        );

        // The outlier itself must clamp, not wrap.
        let code = trimmed.encode(&data[..dim]).unwrap();
        assert_eq!(code[0], MAX_LEVEL);
    }

    #[test]
    fn test_reproducible_across_seeds_when_sample_is_whole_dataset() {
        let (n, dim) = (500, 6);
        let data = pca_like::<f32>(n, dim);
        let a = UniformQuantiser::train(&data, n, dim, Some(UniformQuantParams::new(0.0, None, 1)))
            .unwrap();
        let b = UniformQuantiser::train(&data, n, dim, Some(UniformQuantParams::new(0.0, None, 2)))
            .unwrap();
        // The sample covers everything, so the seed cannot matter.
        assert_eq!(a.scale(), b.scale());
        assert_eq!(a.offsets(), b.offsets());
    }

    #[test]
    fn test_sampling_is_deterministic_for_a_seed() {
        let (n, dim) = (5000, 6);
        let data = pca_like::<f32>(n, dim);
        let p = UniformQuantParams::new(1e-3, Some(500), 99);
        let a = UniformQuantiser::train(&data, n, dim, Some(p)).unwrap();
        let b = UniformQuantiser::train(&data, n, dim, Some(p)).unwrap();
        assert_eq!(a.scale(), b.scale());
        assert_eq!(a.offsets(), b.offsets());
    }

    #[test]
    fn test_non_finite_input_encodes_to_zero() {
        let (n, dim) = (64, 4);
        let data = pca_like::<f32>(n, dim);
        let q = UniformQuantiser::train(&data, n, dim, None).unwrap();
        let code = q
            .encode(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0])
            .unwrap();
        assert_eq!(code[0], 0);
        assert_eq!(code[1], MAX_LEVEL);
        assert_eq!(code[2], 0);
    }

    #[test]
    fn test_encode_rejects_wrong_dimension() {
        let (n, dim) = (32, 8);
        let data = pca_like::<f32>(n, dim);
        let q = UniformQuantiser::train(&data, n, dim, None).unwrap();
        assert!(q.encode(&vec![0.0f32; dim + 1]).is_err());
    }
}
