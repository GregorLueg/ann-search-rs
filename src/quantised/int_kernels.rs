//! Integer distance kernels for uniformly quantised codes.
//!
//! ### Why these are plain scalar loops
//!
//! On `aarch64-apple-darwin` the default target CPU is `apple-m1`, which has
//! the ARMv8.2 dot-product extension, and LLVM autovectorises the loop below
//! into four independent `udot.4s` chains without any help. Hand-written
//! intrinsics are *worse*: the stable NEON surface has no `vdotq_u32`
//! (`stdarch_neon_dotprod` is unstable, rust#117224), so intrinsics fall back
//! to `umull` plus widening pairwise adds. Measured at dim 128, the intrinsic
//! version lost to this loop.
//!
//! Two things break the autovectorisation, so do not "tidy" them away:
//!
//! * Widening to `u32` inside the loop. `udot` accumulates u8 lanes into u32
//!   lanes directly; an explicit cast to a narrower accumulator makes LLVM
//!   choose `mul.16b` plus widening adds instead.
//! * Byte granularity. Masking sub-byte codes out of a packed word loses
//!   `udot` entirely, which is why this module quantises to whole bytes and
//!   does not offer a packed 4-bit or 2-bit variant.
//!
//! ### Why the distance is a dot product
//!
//! `||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b` lets the inner loop be a dot
//! product rather than a squared difference, and a squared difference cannot
//! use `udot` at all. The rearrangement is normally a cancellation footgun;
//! here every term is an exact integer, so it is free. Measured ~1.5x faster
//! than the squared-difference form in and out of cache.

/// Integer dot product of two code vectors.
///
/// The accumulator is `u32` because that is what `udot` accumulates into.
/// Codes are at most [`crate::quantised::uniform_quant::MAX_LEVEL`], so the sum is bounded
/// by `255 * 255 * dim`; `u32` covers dimensionality up to roughly 66 000.
///
/// ### Params
///
/// * `a` - First code vector
/// * `b` - Second code vector, same length as `a`
///
/// ### Returns
///
/// The exact integer dot product
#[inline]
pub fn dot_u8(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0u32;
    for i in 0..a.len() {
        acc += a[i] as u32 * b[i] as u32;
    }
    acc
}

/// Squared integer norm of a code vector.
///
/// Precomputed once per stored vector so the query path only pays for the dot
/// product.
///
/// ### Params
///
/// * `a` - Code vector
///
/// ### Returns
///
/// `sum(a[i]^2)` as an exact integer
#[inline]
pub fn norm_sq_u8(a: &[u8]) -> u32 {
    let mut acc = 0u32;
    for i in 0..a.len() {
        acc += a[i] as u32 * a[i] as u32;
    }
    acc
}

/// Squared code-space distance between two code vectors.
///
/// Returns `sum((a[i] - b[i])^2)` computed as `|a|^2 + |b|^2 - 2 a.b`. The
/// result is mathematically non-negative, but the intermediate is assembled in
/// `i64` because `norm_a + norm_b` can exceed `u32` at high dimensionality
/// (past roughly 33 000 dimensions at full code range).
///
/// This is the code-space distance, not the data-space one. Multiply by
/// `scale^2` to recover the latter; see
/// [`crate::quantised::uniform_quant::UniformQuantiser::squared_euclidean`].
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
/// The squared distance in code space
#[inline]
pub fn sq_dist_from_dot(a: &[u8], b: &[u8], norm_a: u32, norm_b: u32) -> i64 {
    let dot = dot_u8(a, b) as i64;
    let d = norm_a as i64 + norm_b as i64 - 2 * dot;
    // Exact integer arithmetic, so the only way this goes negative is a norm
    // that does not belong to its code vector.
    debug_assert!(d >= 0, "code norms do not match the code vectors");
    d
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference squared difference, the form these kernels replace.
    fn sq_diff_reference(a: &[u8], b: &[u8]) -> i64 {
        let mut acc = 0i64;
        for i in 0..a.len() {
            let d = a[i] as i64 - b[i] as i64;
            acc += d * d;
        }
        acc
    }

    fn codes(seed: u64, dim: usize) -> Vec<u8> {
        let mut s = seed | 1;
        (0..dim)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 256) as u8
            })
            .collect()
    }

    #[test]
    fn test_dot_u8_matches_scalar_reference() {
        let a = codes(11, 128);
        let b = codes(29, 128);
        let expected: u32 = a.iter().zip(&b).map(|(&x, &y)| x as u32 * y as u32).sum();
        assert_eq!(dot_u8(&a, &b), expected);
    }

    #[test]
    fn test_norm_sq_u8_matches_self_dot() {
        let a = codes(7, 96);
        assert_eq!(norm_sq_u8(&a), dot_u8(&a, &a));
    }

    #[test]
    fn test_sq_dist_matches_squared_difference() {
        // The whole point of the reformulation: it must be bit-identical to
        // the squared-difference form, not merely close.
        for dim in [1, 7, 16, 33, 128, 257] {
            let a = codes(3, dim);
            let b = codes(5, dim);
            let na = norm_sq_u8(&a);
            let nb = norm_sq_u8(&b);
            assert_eq!(
                sq_dist_from_dot(&a, &b, na, nb),
                sq_diff_reference(&a, &b),
                "mismatch at dim {dim}"
            );
        }
    }

    #[test]
    fn test_sq_dist_self_is_zero() {
        let a = codes(13, 64);
        let na = norm_sq_u8(&a);
        assert_eq!(sq_dist_from_dot(&a, &a, na, na), 0);
    }

    #[test]
    fn test_sq_dist_saturated_codes_do_not_overflow() {
        // Worst case for the accumulator: every code at the maximum level.
        let dim = 4096;
        let a = vec![255u8; dim];
        let b = vec![0u8; dim];
        let na = norm_sq_u8(&a);
        let nb = norm_sq_u8(&b);
        assert_eq!(na, 255u32 * 255 * dim as u32);
        assert_eq!(sq_dist_from_dot(&a, &b, na, nb), (255i64 * 255) * dim as i64);
    }

    #[test]
    fn test_dot_u8_empty() {
        assert_eq!(dot_u8(&[], &[]), 0);
        assert_eq!(norm_sq_u8(&[]), 0);
    }
}
