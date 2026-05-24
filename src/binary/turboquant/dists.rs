//! Distance calculations for TurboQuant-encoded vectors.
//!
//! The scalar scorer here is the reference implementation: it walks the
//! bit-plane codes, reconstructs each coordinate via the Lloyd-Max
//! levels, and accumulates the inner product in rotated space. Because
//! the rotation is orthogonal and both stored vectors and queries are
//! unit-normalised at encode time, that inner product approximates the
//! cosine similarity `cos(q, v)`. The final metric distance is
//! reconstructed from that similarity plus the retained L2 norms.
//!
//! The SIMD kernels (see `tq_search.rs`) are validated against
//! [`score_ip_scalar`]; any divergence beyond u8-LUT rounding is a bug.
//!
//! **3-bit codes have no SIMD kernel** and always route through the
//! scalar scorer. They are correct but markedly slower than 2-bit or
//! 4-bit; prefer 4-bit unless memory forces otherwise.

use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::binary::turboquant::quantiser::{
    TurboQuantEncoder, TurboQuantQuantiser, TurboQuantQuery, TurboQuantStorage,
};
use crate::prelude::*;

/////////////////////////////
// Scalar reference scorer //
/////////////////////////////

/// Inner product between a rotated query and one bit-plane-encoded vector.
///
/// Walks the `bits` planes, reconstructs each coordinate's code, maps it
/// through `levels_f32`, and accumulates `q_rot[d] * levels[code]`. This
/// is the reference the SIMD kernels must match.
///
/// ### Params
///
/// * `q_rot_f32` - Rotated, unit-normalised query (length `dim`), as f32
/// * `packed` - Bit-plane codes for a single vector (`bits * dim / 8` bytes)
/// * `levels_f32` - Lloyd-Max reconstruction levels (length `2^bits`), as f32
/// * `bits` - Bits per coordinate
/// * `dim` - Dimensionality
///
/// ### Returns
///
/// Approximate `cos(q, v)` in rotated space.
#[inline]
pub fn score_ip_scalar(
    q_rot_f32: &[f32],
    packed: &[u8],
    levels_f32: &[f32],
    bits: usize,
    dim: usize,
) -> f32 {
    let bytes_per_plane = dim / 8;
    let mut ip = 0.0f32;

    for d in 0..dim {
        let byte_pos = d / 8;
        let bit_mask = 1u8 << (7 - (d % 8));

        let mut code = 0usize;
        for p in 0..bits {
            let plane_byte = packed[p * bytes_per_plane + byte_pos];
            if plane_byte & bit_mask != 0 {
                code |= 1 << p;
            }
        }

        ip += q_rot_f32[d] * levels_f32[code];
    }

    ip
}

/////////////////////////////
// Distance reconstruction //
/////////////////////////////

/// Reconstruct a metric distance from the rotated-space inner product.
///
/// `ip` is the approximate `cos(q, v)`. With unit-normalised inputs:
/// - Cosine distance is `1 - ip`.
/// - Squared Euclidean is `‖q‖² + ‖v‖² - 2·‖q‖·‖v‖·ip`, clamped at zero.
///
/// Lower is nearer in both cases, matching the RaBitQ indices.
///
/// ### Params
///
/// * `ip` - Approximate cosine similarity from [`score_ip_scalar`]
/// * `query_norm` - L2 norm of the original query
/// * `vec_norm` - L2 norm of the original stored vector
/// * `metric` - Distance metric
///
/// ### Returns
///
/// Reconstructed distance.
#[inline]
pub fn reconstruct_distance<T>(ip: f32, query_norm: T, vec_norm: T, metric: Dist) -> T
where
    T: Float + FromPrimitive,
{
    let ip_t = T::from_f32(ip).unwrap();
    match metric {
        Dist::Cosine => T::one() - ip_t,
        Dist::SquaredEuclidean => {
            let two = T::one() + T::one();
            let d2 =
                query_norm * query_norm + vec_norm * vec_norm - two * query_norm * vec_norm * ip_t;
            d2.max(T::zero())
        }
        Dist::Manhattan => unreachable!("TurboQuant does not support Manhattan distance"),
    }
}

///////////////////////////////
// VectorDistanceTurboQuant //
///////////////////////////////

/// Distance computation over TurboQuant storage.
///
/// Implementors expose the storage and encoder; the default
/// [`turboquant_dist`](VectorDistanceTurboQuant::turboquant_dist) provides
/// the scalar scoring path. Indices use the SIMD kernels in `tq_search.rs`
/// for the hot path and fall back to this for 3-bit codes and tests.
pub trait VectorDistanceTurboQuant<T>
where
    T: Float + FromPrimitive + ToPrimitive,
{
    /// Reference to the encoded storage.
    fn storage(&self) -> &TurboQuantStorage<T>;

    /// Reference to the encoder (rotation, levels, metric).
    fn encoder(&self) -> &TurboQuantEncoder<T>;

    /// Dimensionality.
    #[inline]
    fn dim(&self) -> usize {
        self.storage().dim
    }

    /// Bits per coordinate.
    #[inline]
    fn bits(&self) -> usize {
        self.storage().bits
    }

    /// Scalar distance estimate between a query and the vector at `idx`.
    ///
    /// ### Params
    ///
    /// * `query` - Encoded query
    /// * `q_rot_f32` - The query's rotated coordinates pre-cast to f32
    ///   (caller-cached to avoid re-casting per vector)
    /// * `levels_f32` - Lloyd-Max levels pre-cast to f32
    /// * `idx` - Global slot index into `storage().packed_codes`
    ///
    /// ### Returns
    ///
    /// Reconstructed distance.
    #[inline]
    fn turboquant_dist(
        &self,
        query: &TurboQuantQuery<T>,
        q_rot_f32: &[f32],
        levels_f32: &[f32],
        idx: usize,
    ) -> T {
        let storage = self.storage();
        let packed = storage.vector_packed(idx);
        let ip = score_ip_scalar(q_rot_f32, packed, levels_f32, storage.bits, storage.dim);
        reconstruct_distance(
            ip,
            query.query_norm,
            storage.norms[idx],
            self.encoder().metric,
        )
    }
}

/// Cast a query's rotated coordinates and the encoder levels to f32 once,
/// for reuse across many [`VectorDistanceTurboQuant::turboquant_dist`] calls.
///
/// ### Returns
///
/// `(q_rot_f32, levels_f32)`.
#[inline]
pub fn prepare_scalar_scoring<T>(
    query: &TurboQuantQuery<T>,
    encoder: &TurboQuantEncoder<T>,
) -> (Vec<f32>, Vec<f32>)
where
    T: Float + ToPrimitive,
{
    let q_rot_f32: Vec<f32> = query.q_rot.iter().map(|x| x.to_f32().unwrap()).collect();
    let levels_f32: Vec<f32> = encoder.levels.iter().map(|x| x.to_f32().unwrap()).collect();
    (q_rot_f32, levels_f32)
}

impl<T> VectorDistanceTurboQuant<T> for TurboQuantQuantiser<T>
where
    T: Float + FromPrimitive + ToPrimitive,
{
    fn storage(&self) -> &TurboQuantStorage<T> {
        &self.storage
    }

    fn encoder(&self) -> &TurboQuantEncoder<T> {
        &self.encoder
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use faer::Mat;

    /// Decorrelated pseudo-random data via a splitmix64-style hash per
    /// (row, col). Unlike a structured sinusoid, this won't produce
    /// accidental near-parallel rows, so nearest-neighbour tests are
    /// unambiguous.
    fn test_data(n: usize, dim: usize) -> Mat<f32> {
        Mat::from_fn(n, dim, |i, j| {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                ^ 0x1234_5678_9ABC_DEF0;
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            (x as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
    }

    fn build(n: usize, dim: usize, bits: usize, metric: Dist) -> TurboQuantQuantiser<f32> {
        TurboQuantQuantiser::new(test_data(n, dim).as_ref(), &metric, bits, 42).unwrap()
    }

    #[test]
    fn test_scalar_ip_matches_independent_decode() {
        let dim = 64;
        let bits = 4;
        let q = build(8, dim, bits, Dist::Cosine);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.21).cos()).collect();
        let eq = q.encode_query(&query).unwrap();
        let (q_rot_f32, levels_f32) = prepare_scalar_scoring(&eq, &q.encoder);

        let bytes_per_plane = dim / 8;
        for idx in 0..q.n_vectors() {
            let packed = q.storage.vector_packed(idx);

            let mut expected = 0.0f32;
            for d in 0..dim {
                let mut code = 0usize;
                for p in 0..bits {
                    if packed[p * bytes_per_plane + d / 8] & (1u8 << (7 - (d % 8))) != 0 {
                        code |= 1 << p;
                    }
                }
                expected += q_rot_f32[d] * levels_f32[code];
            }

            let got = score_ip_scalar(&q_rot_f32, packed, &levels_f32, bits, dim);
            assert_abs_diff_eq!(got, expected, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_cosine_self_distance_near_zero() {
        // Self cosine distance is ~0, within quantisation noise. It can be
        // slightly NEGATIVE: the dequantised vector is not exactly unit
        // norm, so q_rot · v_decoded can exceed 1, making 1 - ip < 0. That
        // is expected, not a bug — hence the abs() bound rather than >= 0.
        let dim = 128;
        let q = build(16, dim, 4, Dist::Cosine);
        let data = test_data(16, dim);

        for t in 0..16 {
            let row: Vec<f32> = (0..dim).map(|j| data[(t, j)]).collect();
            let eq = q.encode_query(&row).unwrap();
            let (q_rot_f32, levels_f32) = prepare_scalar_scoring(&eq, &q.encoder);
            let dist = q.turboquant_dist(&eq, &q_rot_f32, &levels_f32, t);
            assert!(
                dist.abs() < 0.1,
                "self cosine distance {dist} not near zero"
            );
        }
    }

    #[test]
    fn test_squared_euclidean_self_distance_small() {
        let dim = 128;
        let q = build(16, dim, 4, Dist::SquaredEuclidean);
        let data = test_data(16, dim);

        for t in 0..16 {
            let row: Vec<f32> = (0..dim).map(|j| data[(t, j)]).collect();
            let vec_norm_sq: f32 = row.iter().map(|x| x * x).sum();
            let eq = q.encode_query(&row).unwrap();
            let (q_rot_f32, levels_f32) = prepare_scalar_scoring(&eq, &q.encoder);
            let dist = q.turboquant_dist(&eq, &q_rot_f32, &levels_f32, t);
            assert!(dist >= 0.0);
            assert!(
                dist < 0.5 * vec_norm_sq,
                "self sq-euclidean {dist} too large vs ‖v‖²={vec_norm_sq}"
            );
        }
    }

    #[test]
    fn test_approx_nearest_matches_exact_self_query_cosine() {
        // With decorrelated data the true nearest to a stored vector's own
        // value is unambiguously itself, so 4-bit quantisation recovers it.
        let n = 24;
        let dim = 128;
        let q = build(n, dim, 4, Dist::Cosine);
        let data = test_data(n, dim);

        for t in 0..n {
            let row: Vec<f32> = (0..dim).map(|j| data[(t, j)]).collect();
            let eq = q.encode_query(&row).unwrap();
            let (q_rot_f32, levels_f32) = prepare_scalar_scoring(&eq, &q.encoder);

            let mut best = (f32::INFINITY, usize::MAX);
            for idx in 0..n {
                let d = q.turboquant_dist(&eq, &q_rot_f32, &levels_f32, idx);
                if d < best.0 {
                    best = (d, idx);
                }
            }
            assert_eq!(
                best.1, t,
                "approx nearest for self-query {t} was {}",
                best.1
            );
        }
    }

    #[test]
    fn test_more_similar_vector_scores_nearer() {
        let dim = 128;
        let q = build(2, dim, 4, Dist::Cosine);
        let data = test_data(2, dim);

        let a: Vec<f32> = (0..dim).map(|j| data[(0, j)]).collect();
        let query: Vec<f32> = a
            .iter()
            .enumerate()
            .map(|(j, &x)| x + 0.01 * (j as f32).sin())
            .collect();
        let eq = q.encode_query(&query).unwrap();
        let (q_rot_f32, levels_f32) = prepare_scalar_scoring(&eq, &q.encoder);

        let d_a = q.turboquant_dist(&eq, &q_rot_f32, &levels_f32, 0);
        let d_b = q.turboquant_dist(&eq, &q_rot_f32, &levels_f32, 1);
        assert!(
            d_a < d_b,
            "near vector {d_a} should score below far vector {d_b}"
        );
    }

    #[test]
    fn test_reconstruct_cosine_identity() {
        assert_abs_diff_eq!(
            reconstruct_distance::<f32>(1.0, 1.0, 1.0, Dist::Cosine),
            0.0,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            reconstruct_distance::<f32>(0.0, 1.0, 1.0, Dist::Cosine),
            1.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_reconstruct_squared_euclidean_known() {
        assert_abs_diff_eq!(
            reconstruct_distance::<f32>(1.0, 1.0, 1.0, Dist::SquaredEuclidean),
            0.0,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            reconstruct_distance::<f32>(0.0, 1.0, 1.0, Dist::SquaredEuclidean),
            2.0,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            reconstruct_distance::<f32>(-1.0, 1.0, 1.0, Dist::SquaredEuclidean),
            4.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_reconstruct_squared_euclidean_clamped() {
        let d = reconstruct_distance::<f32>(2.0, 1.0, 1.0, Dist::SquaredEuclidean);
        assert_eq!(d, 0.0);
    }

    #[test]
    #[should_panic(expected = "Manhattan")]
    fn test_manhattan_unreachable() {
        let _ = reconstruct_distance::<f32>(0.5, 1.0, 1.0, Dist::Manhattan);
    }
}
