//! Lloyd-Max scalar quantiser codebook for the Beta distribution arising from a
//! random orthogonal rotation.
//!
//! After a uniformly random orthogonal rotation, each coordinate of a unit
//! vector on the sphere `S^{d-1}` follows `Beta((d-1)/2, (d-1)/2)` shifted to
//! `[-1, 1]`. This module computes optimal scalar quantisation boundaries and
//! centroids for that distribution via Lloyd-Max iteration.
//!
//! See: "TurboQuant: Online Vector Quantization with Strong Theoretical
//! Guarantees" (Tepper, Carlson, et al., 2025).

use num_traits::{Float, FromPrimitive};
use statrs::distribution::{Beta, Continuous, ContinuousCDF};

use crate::errors::AnnSearchErrors;

////////////
// Consts //
////////////

/// Maximum iterations for Lloyds
const LLOYD_MAX_ITERS: usize = 200;
/// Tolerance for Lloyds
const LLOYD_MAX_TOL: f64 = 1e-12;
/// Tolerance for Simpsons
const SIMPSON_TOL: f64 = 1e-14;
/// Maximum depth for Simpsons
const SIMPSON_MAX_DEPTH: usize = 50;

/////////////
// Helpers //
/////////////

/// Recursive helper for adaptive Simpson's rule.
///
/// ### Params
///
/// * `f` - Integrand
/// * `a` - Left endpoint of the current subinterval
/// * `b` - Right endpoint of the current subinterval
/// * `fa` - `f(a)`
/// * `fb` - `f(b)`
/// * `fm` - `f((a + b) / 2)`
/// * `whole` - Simpson estimate over `[a, b]` from the parent call
/// * `tol` - Error tolerance for this subinterval
/// * `depth` - Remaining recursion depth
///
/// ### Returns
///
/// Refined integral estimate over `[a, b]` with Richardson extrapolation
/// applied.
#[allow(clippy::too_many_arguments)]
fn adaptive_simpson_rec<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
    tol: f64,
    depth: usize,
) -> f64 {
    let mid = (a + b) / 2.0;
    let m1 = (a + mid) / 2.0;
    let m2 = (mid + b) / 2.0;
    let fm1 = f(m1);
    let fm2 = f(m2);
    let left = (mid - a) / 6.0 * (fa + 4.0 * fm1 + fm);
    let right = (b - mid) / 6.0 * (fm + 4.0 * fm2 + fb);
    let refined = left + right;

    if depth == 0 || (refined - whole).abs() < 15.0 * tol {
        refined + (refined - whole) / 15.0
    } else {
        adaptive_simpson_rec(f, a, mid, fa, fm, fm1, left, tol / 2.0, depth - 1)
            + adaptive_simpson_rec(f, mid, b, fm, fb, fm2, right, tol / 2.0, depth - 1)
    }
}

/// Adaptive Simpson's rule over `[a, b]`.
///
/// ### Params
///
/// * `f` - Integrand
/// * `a` - Left endpoint
/// * `b` - Right endpoint
/// * `tol` - Absolute error tolerance
/// * `max_depth` - Maximum recursion depth
///
/// ### Returns
///
/// Integral estimate of `f` over `[a, b]`.
fn adaptive_simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64, max_depth: usize) -> f64 {
    let mid = (a + b) / 2.0;
    let fa = f(a);
    let fb = f(b);
    let fm = f(mid);
    let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    adaptive_simpson_rec(&f, a, b, fa, fb, fm, whole, tol, max_depth)
}

/// Lloyd-Max iteration for `Beta((dim-1)/2, (dim-1)/2)` shifted to `[-1, 1]`.
///
/// ### Params
///
/// * `bits` - Number of quantisation bits; yields `2^bits` levels
/// * `dim` - Dimensionality, controls the Beta shape parameter
/// * `max_iter` - Maximum number of Lloyd-Max iterations
/// * `tol` - Convergence tolerance on the maximum centroid shift
///
/// ### Returns
///
/// Tuple `(boundaries, centroids)` at convergence. There are `2^bits - 1`
/// boundaries and `2^bits` centroids, both sorted ascending.
fn lloyd_max(bits: usize, dim: usize, max_iter: usize, tol: f64) -> (Vec<f64>, Vec<f64>) {
    let a = (dim as f64 - 1.0) / 2.0;
    let beta = Beta::new(a, a).unwrap();

    let n_levels = 1usize << bits;

    // Variance of Beta(a, a) shifted to [-1, 1] is 1 / (2a + 1).
    // Initialise centroids spanning +/- 3 standard deviations.
    let std_dev = 1.0 / (2.0 * a + 1.0).sqrt();
    let spread = 3.0 * std_dev;
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| -spread + 2.0 * spread * i as f64 / (n_levels as f64 - 1.0))
        .collect();

    for _ in 0..max_iter {
        let boundaries: Vec<f64> = (0..n_levels - 1)
            .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
            .collect();

        let mut edges = Vec::with_capacity(n_levels + 1);
        edges.push(-1.0);
        edges.extend_from_slice(&boundaries);
        edges.push(1.0);

        let mut new_centroids = vec![0.0f64; n_levels];

        for i in 0..n_levels {
            let lo = edges[i];
            let hi = edges[i + 1];

            // cdf of the shifted Beta on [-1, 1] in terms of Beta on [0, 1].
            let cdf_lo = beta.cdf((lo + 1.0) / 2.0);
            let cdf_hi = beta.cdf((hi + 1.0) / 2.0);
            let prob = cdf_hi - cdf_lo;

            if prob < 1e-15 {
                new_centroids[i] = centroids[i];
            } else {
                // Conditional mean = integral(x * pdf(x)) / prob over [lo, hi].
                // Shifted PDF: pdf_shifted(x) = beta.pdf((x + 1) / 2) / 2.
                let mean = adaptive_simpson(
                    |x| {
                        let t = (x + 1.0) / 2.0;
                        x * beta.pdf(t) / 2.0
                    },
                    lo,
                    hi,
                    SIMPSON_TOL,
                    SIMPSON_MAX_DEPTH,
                );
                new_centroids[i] = mean / prob;
            }
        }

        let max_change = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        centroids = new_centroids;

        if max_change < tol {
            break;
        }
    }

    let boundaries: Vec<f64> = (0..n_levels - 1)
        .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
        .collect();

    (boundaries, centroids)
}

//////////////
// Codebook //
//////////////

/// Compute Lloyd-Max boundaries and centroids for the rotated-coordinate
/// distribution.
///
/// ### Params
///
/// * `bits` - Bits per coordinate (must be `2`, `3`, or `4`)
/// * `dim` - Dimensionality of the data
///
/// ### Returns
///
/// Tuple `(boundaries, centroids)`. There are `2^bits - 1` boundaries
/// and `2^bits` centroids, both sorted ascending. Centroids are symmetric
/// about zero.
pub fn codebook<T>(bits: usize, dim: usize) -> Result<(Vec<T>, Vec<T>), AnnSearchErrors>
where
    T: Float + FromPrimitive,
{
    if !(2..=4).contains(&bits) {
        return Err(AnnSearchErrors::TQInvalidBits { n_bits: bits });
    }

    let (boundaries_f64, centroids_f64) = lloyd_max(bits, dim, LLOYD_MAX_ITERS, LLOYD_MAX_TOL);

    let boundaries = boundaries_f64
        .into_iter()
        .map(|x| T::from_f64(x).unwrap())
        .collect();
    let centroids = centroids_f64
        .into_iter()
        .map(|x| T::from_f64(x).unwrap())
        .collect();

    Ok((boundaries, centroids))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_codebook_lengths() {
        for bits in 2..=4 {
            let (boundaries, centroids) = codebook::<f32>(bits, 128).unwrap();
            assert_eq!(centroids.len(), 1 << bits);
            assert_eq!(boundaries.len(), (1 << bits) - 1);
        }
    }

    #[test]
    fn test_centroids_sorted() {
        for bits in 2..=4 {
            let (_, centroids) = codebook::<f64>(bits, 128).unwrap();
            for i in 1..centroids.len() {
                assert!(centroids[i] > centroids[i - 1]);
            }
        }
    }

    #[test]
    fn test_boundaries_are_midpoints() {
        let (boundaries, centroids) = codebook::<f64>(4, 128).unwrap();
        for i in 0..boundaries.len() {
            let mid = (centroids[i] + centroids[i + 1]) / 2.0;
            assert_abs_diff_eq!(boundaries[i], mid, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_centroids_symmetric_about_zero() {
        for bits in 2..=4 {
            let (_, centroids) = codebook::<f64>(bits, 128).unwrap();
            let n = centroids.len();
            for i in 0..n / 2 {
                assert_abs_diff_eq!(centroids[i], -centroids[n - 1 - i], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_centroids_in_open_interval() {
        for bits in 2..=4 {
            let (_, centroids) = codebook::<f64>(bits, 128).unwrap();
            for &c in &centroids {
                assert!(c > -1.0 && c < 1.0);
            }
        }
    }

    #[test]
    fn test_deterministic() {
        let (b1, c1) = codebook::<f64>(4, 256).unwrap();
        let (b2, c2) = codebook::<f64>(4, 256).unwrap();
        assert_eq!(b1, b2);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_higher_dim_concentrates_distribution() {
        // Beta((d-1)/2, (d-1)/2) tightens around 1/2 as d grows, so the
        // shifted distribution concentrates at 0 and the extreme centroids
        // shrink in magnitude.
        let (_, c_low) = codebook::<f64>(4, 16).unwrap();
        let (_, c_high) = codebook::<f64>(4, 1024).unwrap();
        assert!(c_high[0].abs() < c_low[0].abs());
        assert!(c_high.last().unwrap() < c_low.last().unwrap());
    }

    #[test]
    fn test_f32_matches_f64() {
        let (b32, c32) = codebook::<f32>(4, 128).unwrap();
        let (b64, c64) = codebook::<f64>(4, 128).unwrap();
        for (a, b) in b32.iter().zip(b64.iter()) {
            assert_abs_diff_eq!(*a as f64, *b, epsilon = 1e-6);
        }
        for (a, b) in c32.iter().zip(c64.iter()) {
            assert_abs_diff_eq!(*a as f64, *b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_invalid_bits() {
        let result = codebook::<f32>(5, 128);
        assert!(matches!(
            result,
            Err(AnnSearchErrors::TQInvalidBits { n_bits: 5 })
        ));
    }

    #[test]
    fn test_invalid_dim() {
        let result = codebook::<f32>(4, 1);
        assert!(result.is_err());
    }
}
