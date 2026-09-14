//! Random rotations for RaBitQ encoding.
//!
//! RaBitQ needs the residual spread evenly across coordinates before the sign
//! bits are taken, so a random orthogonal transform sits in front of every
//! encode and every query. Two of them live here.
//!
//! [`DenseRotator`] is a QR-derived orthogonal matrix applied as a matvec:
//! `O(dim^2)` per vector and `dim^2` floats of state. [`FhtKacRotator`] is four
//! rounds of random sign flips, a fast Hadamard transform and a Kac butterfly:
//! `O(dim log dim)` per vector, and its entire state is four sign bits per
//! coordinate. The Hadamard runs on the largest power of two that fits `dim`,
//! so below 64 coordinates it mixes too few of them to stand in for a real
//! rotation and [`DenseRotator`] is the only option.
//!
//! ### References
//!
//! Gao and Long, "RaBitQ: Quantizing High-Dimensional Vectors with a
//! Theoretical Error Bound for Approximate Nearest Neighbor Search",
//! SIGMOD 2024.

use faer::Mat;
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Rounds of flip-Hadamard-Kac applied by [`FhtKacRotator`].
///
/// Four is what the RaBitQ reference implementation uses. Each Kac butterfly
/// is unnormalised and scales the norm by `sqrt(2)`, so four of them need the
/// closing `1/4` in [`FhtKacRotator::rotate_into`].
const KAC_ROUNDS: usize = 4;

/// Smallest dimensionality [`FhtKacRotator`] will accept.
///
/// The Hadamard transform runs on `2^floor(log2(dim))` coordinates. At 64 that
/// is six butterfly stages, which is the point below which the reference
/// implementation stops wiring the transform up at all.
pub const FHT_MIN_DIM: usize = 64;

/// Dimensionality from which [`resolve_rotator_kind`] prefers the Hadamard.
///
/// The Hadamard is `O(dim log dim)` against the matvec's `O(dim^2)`, but it is
/// a scalar butterfly with a strided access pattern while the matvec is a run
/// of fully vectorised contiguous dot products. The asymptotics only win once
/// the matvec is large enough to stop fitting the cache well, which measures
/// out at around 200 coordinates; below that the Hadamard is the slower of the
/// two despite doing less arithmetic. Measured on aarch64, where the matvec
/// runs on 128-bit vectors; a machine with AVX-512 would push the crossover
/// higher still, so this is deliberately set at the first measured win rather
/// than the midpoint.
///
/// Between [`FHT_MIN_DIM`] and this the Hadamard still works and can be asked
/// for explicitly, since it also shrinks the stored rotation from `dim^2`
/// floats to four bits per coordinate.
pub const FHT_AUTO_MIN_DIM: usize = 192;

/// Coordinate multiple every [`FhtKacRotator`] pads up to.
///
/// The Kac butterfly halves the working length, and the byte-packed sign bits
/// want whole bytes, so a multiple of 64 keeps both aligned and leaves the
/// downstream 1-bit codes a whole number of 8-byte words.
pub const ROTATOR_PAD: usize = 64;

/// In-place fast Hadamard transform.
///
/// Plain butterfly loop rather than intrinsics: the stride pattern is exactly
/// what LLVM autovectorises, and the reference implementation's own `fht_avx`
/// header turns out to contain no intrinsics either.
///
/// ### Params
///
/// * `buf` - Buffer whose length is a power of two, transformed in place
fn fht<T>(buf: &mut [T])
where
    T: Float,
{
    let n = buf.len();
    debug_assert!(n.is_power_of_two());

    let mut len = 1;
    while len < n {
        let step = len << 1;
        for base in (0..n).step_by(step) {
            for j in base..base + len {
                let u = buf[j];
                let v = buf[j + len];
                buf[j] = u + v;
                buf[j + len] = u - v;
            }
        }
        len = step;
    }
}

/// Negate the coordinates whose sign bit is set.
///
/// Bits are read least-significant-first within each byte, so bit `i % 8` of
/// byte `i / 8` owns coordinate `i`.
///
/// ### Params
///
/// * `flip` - Packed sign bits, `data.len() / 8` bytes
/// * `data` - Buffer negated in place
fn flip_sign<T>(flip: &[u8], data: &mut [T])
where
    T: Float,
{
    for (i, x) in data.iter_mut().enumerate() {
        let set = flip[i / 8] & (1u8 << (i % 8)) != 0;
        *x = if set { -*x } else { *x };
    }
}

/// One unnormalised Kac butterfly across the two halves of a buffer.
///
/// ### Params
///
/// * `data` - Buffer of even length, transformed in place
fn kacs_walk<T>(data: &mut [T])
where
    T: Float,
{
    let half = data.len() / 2;
    let (lo, hi) = data.split_at_mut(half);
    for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
        let (u, v) = (*x, *y);
        *x = u + v;
        *y = u - v;
    }
}

/// Scale every element of a buffer.
///
/// ### Params
///
/// * `data` - Buffer scaled in place
/// * `factor` - Multiplier
#[inline]
fn rescale<T>(data: &mut [T], factor: T)
where
    T: Float,
{
    for x in data.iter_mut() {
        *x = *x * factor;
    }
}

//////////////////
// RotatorKind //
//////////////////

/// Which rotation a RaBitQ index should use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub enum RotatorKind {
    /// Dense orthogonal matrix. `O(dim^2)` per vector, `dim^2` floats of state.
    Dense,
    /// Fast Hadamard plus Kac walk. `O(dim log dim)` per vector, four sign bits
    /// per coordinate of state. Needs at least [`FHT_MIN_DIM`] coordinates.
    FhtKac,
}

/// Pick a rotation for a given dimensionality.
///
/// [`RotatorKind::FhtKac`] from [`FHT_AUTO_MIN_DIM`] up, where it is the faster
/// of the two as well as the smaller.
///
/// ### Params
///
/// * `dim` - Dimensionality of the data
///
/// ### Returns
///
/// The rotation to build
pub fn resolve_rotator_kind(dim: usize) -> RotatorKind {
    if dim >= FHT_AUTO_MIN_DIM {
        RotatorKind::FhtKac
    } else {
        RotatorKind::Dense
    }
}

/// Working dimensionality a rotation produces.
///
/// The dense rotation is square and pads nothing; the Hadamard path rounds up
/// to [`ROTATOR_PAD`].
///
/// ### Params
///
/// * `dim` - Dimensionality of the data
/// * `kind` - Which rotation is in use
///
/// ### Returns
///
/// Length of the rotated vector
pub fn padded_dim_for(dim: usize, kind: RotatorKind) -> usize {
    match kind {
        RotatorKind::Dense => dim,
        RotatorKind::FhtKac => dim.next_multiple_of(ROTATOR_PAD),
    }
}

///////////////////
// DenseRotator //
///////////////////

/// Orthogonal rotation held as a dense `dim * dim` matrix.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct DenseRotator<T> {
    /// Row-major rotation matrix, `dim * dim` entries
    rotation: Vec<T>,
    /// Dimensionality of the data
    dim: usize,
}

impl<T> DenseRotator<T>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField + SimdDistance,
{
    /// Build a random orthogonal rotation from the Q factor of a Gaussian matrix.
    ///
    /// ### Params
    ///
    /// * `dim` - Dimensionality of the data
    /// * `seed` - Random seed, for reproducibility
    ///
    /// ### Returns
    ///
    /// The rotation
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut mat = Mat::<T>::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                let val: f64 = rng.sample(StandardNormal);
                mat[(i, j)] = T::from_f64(val).unwrap();
            }
        }

        let q = mat.as_ref().qr().compute_Q();

        let mut rotation = Vec::with_capacity(dim * dim);
        for i in 0..dim {
            for j in 0..dim {
                rotation.push(q[(i, j)]);
            }
        }

        Self { rotation, dim }
    }

    /// Rotate one vector into a caller-supplied buffer.
    ///
    /// ### Params
    ///
    /// * `vec` - Vector of length `dim`
    /// * `out` - Output buffer of length `dim`
    #[inline]
    pub fn rotate_into(&self, vec: &[T], out: &mut [T]) {
        debug_assert_eq!(vec.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);

        for i in 0..self.dim {
            let row = &self.rotation[i * self.dim..(i + 1) * self.dim];
            out[i] = T::dot_simd(row, vec);
        }
    }

    /// Rotate one vector into a fresh buffer.
    ///
    /// ### Params
    ///
    /// * `vec` - Vector of length `dim`
    ///
    /// ### Returns
    ///
    /// The rotated vector, of length `dim`
    #[inline]
    pub fn rotate(&self, vec: &[T]) -> Vec<T> {
        let mut out = vec![T::zero(); self.dim];
        self.rotate_into(vec, &mut out);
        out
    }

    /// Bytes held by the rotation.
    ///
    /// ### Returns
    ///
    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.rotation.capacity() * std::mem::size_of::<T>()
    }
}

////////////////////
// FhtKacRotator //
////////////////////

/// Orthogonal rotation built from sign flips, Hadamard transforms and Kac walks.
///
/// Four rounds of {flip signs, Hadamard over the largest power of two that fits,
/// rescale, Kac butterfly}. When the padded length is itself a power of two the
/// Hadamard covers everything and the Kac step is unnecessary; otherwise the
/// transform window alternates between the start and the end of the buffer so
/// the coordinates the truncated Hadamard misses still get mixed.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct FhtKacRotator<T> {
    /// Packed sign bits, `KAC_ROUNDS * padded_dim / 8` bytes
    flip: Vec<u8>,
    /// Dimensionality of the data
    dim: usize,
    /// Length the rotation works at, a multiple of [`ROTATOR_PAD`]
    padded_dim: usize,
    /// Largest power of two not exceeding `dim`, the Hadamard window
    trunc_dim: usize,
    /// `1 / sqrt(trunc_dim)`, the Hadamard's normalisation
    fac: T,
}

impl<T> FhtKacRotator<T>
where
    T: Float + FromPrimitive,
{
    /// Build a rotation for `dim` coordinates.
    ///
    /// ### Params
    ///
    /// * `dim` - Dimensionality of the data, at least [`FHT_MIN_DIM`]
    /// * `seed` - Random seed, for reproducibility
    ///
    /// ### Returns
    ///
    /// The rotation, or an error when `dim` is below [`FHT_MIN_DIM`]
    pub fn new(dim: usize, seed: u64) -> Result<Self, AnnSearchErrors> {
        if dim < FHT_MIN_DIM {
            return Err(AnnSearchErrors::RotatorDimTooSmall {
                dim,
                min_dim: FHT_MIN_DIM,
            });
        }

        let padded_dim = padded_dim_for(dim, RotatorKind::FhtKac);
        let trunc_dim = 1usize << dim.ilog2();

        let mut rng = StdRng::seed_from_u64(seed);
        let mut flip = vec![0u8; KAC_ROUNDS * padded_dim / 8];
        rng.fill(&mut flip[..]);

        Ok(Self {
            flip,
            dim,
            padded_dim,
            trunc_dim,
            fac: T::from_f64(1.0 / (trunc_dim as f64).sqrt()).unwrap(),
        })
    }

    /// Rotate one vector into a caller-supplied buffer.
    ///
    /// The input is zero-padded to `padded_dim` before the first round, so
    /// `out` is longer than `vec` whenever `dim` is not a multiple of
    /// [`ROTATOR_PAD`].
    ///
    /// ### Params
    ///
    /// * `vec` - Vector of length `dim`
    /// * `out` - Output buffer of length `padded_dim`
    pub fn rotate_into(&self, vec: &[T], out: &mut [T]) {
        debug_assert_eq!(vec.len(), self.dim);
        debug_assert_eq!(out.len(), self.padded_dim);

        out[..self.dim].copy_from_slice(vec);
        out[self.dim..].fill(T::zero());

        let bytes: usize = self.padded_dim / 8;

        if self.trunc_dim == self.padded_dim {
            for round in 0..KAC_ROUNDS {
                flip_sign(&self.flip[round * bytes..(round + 1) * bytes], out);
                fht(out);
                rescale(out, self.fac);
            }
            return;
        }

        // The Hadamard covers only `trunc_dim` of the buffer, so alternate the
        // window between the two ends and let the Kac butterfly carry the rest.
        let tail = self.padded_dim - self.trunc_dim;
        for round in 0..KAC_ROUNDS {
            flip_sign(&self.flip[round * bytes..(round + 1) * bytes], out);
            let offset = if round % 2 == 0 { 0 } else { tail };
            let window = &mut out[offset..offset + self.trunc_dim];
            fht(window);
            rescale(window, self.fac);
            kacs_walk(out);
        }
        rescale(out, T::from_f64(0.25).unwrap());
    }

    /// Rotate one vector into a fresh buffer.
    ///
    /// ### Params
    ///
    /// * `vec` - Vector of length `dim`
    ///
    /// ### Returns
    ///
    /// The rotated vector, of length `padded_dim`
    #[inline]
    pub fn rotate(&self, vec: &[T]) -> Vec<T> {
        let mut out = vec![T::zero(); self.padded_dim];
        self.rotate_into(vec, &mut out);
        out
    }

    /// Bytes held by the rotation.
    ///
    /// ### Returns
    ///
    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.flip.capacity()
    }
}

///////////////////
// RaBitQRotator //
///////////////////

/// The rotation a RaBitQ encoder holds.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub enum RaBitQRotator<T> {
    /// Dense orthogonal matrix
    Dense(DenseRotator<T>),
    /// Fast Hadamard plus Kac walk
    FhtKac(FhtKacRotator<T>),
}

impl<T> RaBitQRotator<T>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField + SimdDistance,
{
    /// Build a rotation.
    ///
    /// ### Params
    ///
    /// * `dim` - Dimensionality of the data
    /// * `kind` - Which rotation to build, or `None` to let
    ///   [`resolve_rotator_kind`] decide
    /// * `seed` - Random seed, for reproducibility
    ///
    /// ### Returns
    ///
    /// The rotation, or an error when [`RotatorKind::FhtKac`] was asked for at
    /// a dimensionality it cannot serve
    pub fn new(dim: usize, kind: Option<RotatorKind>, seed: u64) -> Result<Self, AnnSearchErrors> {
        match kind.unwrap_or_else(|| resolve_rotator_kind(dim)) {
            RotatorKind::Dense => Ok(Self::Dense(DenseRotator::new(dim, seed))),
            RotatorKind::FhtKac => Ok(Self::FhtKac(FhtKacRotator::new(dim, seed)?)),
        }
    }

    /// Build the rotation [`resolve_rotator_kind`] picks for this dimensionality.
    ///
    /// Cannot fail: the resolver only returns [`RotatorKind::FhtKac`] where it
    /// is legal.
    ///
    /// ### Params
    ///
    /// * `dim` - Dimensionality of the data
    /// * `seed` - Random seed, for reproducibility
    ///
    /// ### Returns
    ///
    /// The rotation
    pub fn new_auto(dim: usize, seed: u64) -> Self {
        match resolve_rotator_kind(dim) {
            RotatorKind::Dense => Self::Dense(DenseRotator::new(dim, seed)),
            RotatorKind::FhtKac => Self::FhtKac(
                FhtKacRotator::new(dim, seed)
                    .expect("resolve_rotator_kind only picks FhtKac above its minimum dimension"),
            ),
        }
    }

    /// Which rotation this is.
    ///
    /// ### Returns
    ///
    /// The kind
    pub fn kind(&self) -> RotatorKind {
        match self {
            Self::Dense(_) => RotatorKind::Dense,
            Self::FhtKac(_) => RotatorKind::FhtKac,
        }
    }

    /// Length of a rotated vector.
    ///
    /// ### Returns
    ///
    /// The working dimensionality
    pub fn padded_dim(&self) -> usize {
        match self {
            Self::Dense(r) => r.dim,
            Self::FhtKac(r) => r.padded_dim,
        }
    }

    /// Rotate one vector into a caller-supplied buffer.
    ///
    /// ### Params
    ///
    /// * `vec` - Vector of length `dim`
    /// * `out` - Output buffer of length [`padded_dim`](Self::padded_dim)
    #[inline]
    pub fn rotate_into(&self, vec: &[T], out: &mut [T]) {
        match self {
            Self::Dense(r) => r.rotate_into(vec, out),
            Self::FhtKac(r) => r.rotate_into(vec, out),
        }
    }

    /// Rotate one vector into a fresh buffer.
    ///
    /// ### Params
    ///
    /// * `vec` - Vector of length `dim`
    ///
    /// ### Returns
    ///
    /// The rotated vector, of length [`padded_dim`](Self::padded_dim)
    #[inline]
    pub fn rotate(&self, vec: &[T]) -> Vec<T> {
        match self {
            Self::Dense(r) => r.rotate(vec),
            Self::FhtKac(r) => r.rotate(vec),
        }
    }

    /// Bytes held by the rotation.
    ///
    /// ### Returns
    ///
    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::Dense(r) => r.memory_usage_bytes(),
            Self::FhtKac(r) => r.memory_usage_bytes(),
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::StandardNormal;

    /// Gaussian test vectors, `n` rows of `dim`.
    fn gaussian(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * dim)
            .map(|_| rng.sample::<f64, _>(StandardNormal) as f32)
            .collect()
    }

    fn l2(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn test_fht_matches_naive_hadamard() {
        let n: usize = 8;
        let mut buf: Vec<f32> = (0..n).map(|i| i as f32 - 3.5).collect();
        let input = buf.clone();
        fht(&mut buf);

        // H[i][j] = (-1)^popcount(i & j)
        for i in 0..n {
            let expected: f32 = (0..n)
                .map(|j| {
                    let sign = if (i & j).count_ones() % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    sign * input[j]
                })
                .sum();
            approx::assert_relative_eq!(buf[i], expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_fht_is_involutive_up_to_scale() {
        let mut buf = gaussian(1, 64, 7);
        let input = buf.clone();
        fht(&mut buf);
        fht(&mut buf);
        for (got, want) in buf.iter().zip(&input) {
            approx::assert_relative_eq!(got / 64.0, want, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_fht_kac_preserves_norms() {
        for dim in [64usize, 100, 128, 512, 768] {
            let rot = FhtKacRotator::<f32>::new(dim, 42).unwrap();
            let data = gaussian(16, dim, dim as u64);
            let mut out = vec![0.0f32; rot.padded_dim];

            for row in data.chunks_exact(dim) {
                rot.rotate_into(row, &mut out);
                approx::assert_relative_eq!(l2(&out), l2(row), epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn test_fht_kac_preserves_pairwise_distances() {
        for dim in [64usize, 100, 128, 512, 768] {
            let rot = FhtKacRotator::<f32>::new(dim, 11).unwrap();
            let data = gaussian(8, dim, 3);
            let rotated: Vec<Vec<f32>> = data.chunks_exact(dim).map(|r| rot.rotate(r)).collect();
            let rows: Vec<&[f32]> = data.chunks_exact(dim).collect();

            for i in 0..rows.len() {
                for j in (i + 1)..rows.len() {
                    approx::assert_relative_eq!(
                        dist(&rotated[i], &rotated[j]),
                        dist(rows[i], rows[j]),
                        epsilon = 1e-3
                    );
                }
            }
        }
    }

    #[test]
    fn test_fht_kac_preserves_dot_products() {
        let dim = 256;
        let rot = FhtKacRotator::<f32>::new(dim, 5).unwrap();
        let data = gaussian(6, dim, 19);
        let rotated: Vec<Vec<f32>> = data.chunks_exact(dim).map(|r| rot.rotate(r)).collect();
        let rows: Vec<&[f32]> = data.chunks_exact(dim).collect();

        for i in 0..rows.len() {
            for j in 0..rows.len() {
                let got: f32 = rotated[i].iter().zip(&rotated[j]).map(|(a, b)| a * b).sum();
                let want: f32 = rows[i].iter().zip(rows[j]).map(|(a, b)| a * b).sum();
                approx::assert_relative_eq!(got, want, epsilon = 1e-2);
            }
        }
    }

    #[test]
    fn test_fht_kac_mixes_a_one_hot_vector() {
        // A single non-zero coordinate must spread across the whole buffer, or
        // the sign bits downstream carry no information.
        let dim = 128;
        let rot = FhtKacRotator::<f32>::new(dim, 2).unwrap();
        let mut v = vec![0.0f32; dim];
        v[0] = 1.0;
        let out = rot.rotate(&v);

        let max = out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(max < 0.5, "one-hot input stayed concentrated, max {max}");
    }

    #[test]
    fn test_fht_kac_is_deterministic_for_a_seed() {
        let dim = 128;
        let a = FhtKacRotator::<f32>::new(dim, 99).unwrap();
        let b = FhtKacRotator::<f32>::new(dim, 99).unwrap();
        let v = gaussian(1, dim, 1);
        assert_eq!(a.rotate(&v), b.rotate(&v));
    }

    #[test]
    fn test_fht_kac_differs_between_seeds() {
        let dim = 128;
        let a = FhtKacRotator::<f32>::new(dim, 1).unwrap();
        let b = FhtKacRotator::<f32>::new(dim, 2).unwrap();
        let v = gaussian(1, dim, 1);
        assert_ne!(a.rotate(&v), b.rotate(&v));
    }

    #[test]
    fn test_fht_kac_rejects_small_dimensions() {
        assert!(FhtKacRotator::<f32>::new(63, 0).is_err());
        assert!(FhtKacRotator::<f32>::new(64, 0).is_ok());
    }

    #[test]
    fn test_fht_kac_works_at_f64() {
        let dim = 128;
        let rot = FhtKacRotator::<f64>::new(dim, 8).unwrap();
        let v: Vec<f64> = gaussian(1, dim, 4).into_iter().map(|x| x as f64).collect();
        let mut out = vec![0.0f64; dim];
        rot.rotate_into(&v, &mut out);
        let norm_in = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_out = out.iter().map(|x| x * x).sum::<f64>().sqrt();
        approx::assert_relative_eq!(norm_out, norm_in, epsilon = 1e-9);
    }

    #[test]
    fn test_dense_rotator_preserves_norms() {
        let dim = 32;
        let rot = DenseRotator::<f32>::new(dim, 3);
        let data = gaussian(8, dim, 6);
        let mut out = vec![0.0f32; dim];
        for row in data.chunks_exact(dim) {
            rot.rotate_into(row, &mut out);
            approx::assert_relative_eq!(l2(&out), l2(row), epsilon = 1e-4);
        }
    }

    #[test]
    fn test_resolve_picks_fht_only_once_it_is_the_faster_one() {
        assert_eq!(resolve_rotator_kind(50), RotatorKind::Dense);
        assert_eq!(resolve_rotator_kind(64), RotatorKind::Dense);
        assert_eq!(resolve_rotator_kind(128), RotatorKind::Dense);
        assert_eq!(resolve_rotator_kind(FHT_AUTO_MIN_DIM), RotatorKind::FhtKac);
        assert_eq!(resolve_rotator_kind(768), RotatorKind::FhtKac);

        // Below the auto threshold it is still available on request.
        assert!(RaBitQRotator::<f32>::new(128, Some(RotatorKind::FhtKac), 0).is_ok());
    }

    #[test]
    fn test_padded_dim_rounds_to_the_pad_multiple() {
        assert_eq!(padded_dim_for(50, RotatorKind::Dense), 50);
        assert_eq!(padded_dim_for(64, RotatorKind::FhtKac), 64);
        assert_eq!(padded_dim_for(100, RotatorKind::FhtKac), 128);
        assert_eq!(padded_dim_for(768, RotatorKind::FhtKac), 768);
    }

    #[test]
    fn test_enum_rotator_dispatches_and_reports_padding() {
        let dense = RaBitQRotator::<f32>::new(50, None, 0).unwrap();
        assert_eq!(dense.kind(), RotatorKind::Dense);
        assert_eq!(dense.padded_dim(), 50);

        let fht = RaBitQRotator::<f32>::new(200, None, 0).unwrap();
        assert_eq!(fht.kind(), RotatorKind::FhtKac);
        assert_eq!(fht.padded_dim(), 256);
        assert_eq!(fht.rotate(&gaussian(1, 200, 0)).len(), 256);

        assert!(RaBitQRotator::<f32>::new(50, Some(RotatorKind::FhtKac), 0).is_err());
    }

    #[test]
    fn test_fht_kac_state_is_far_smaller_than_the_dense_matrix() {
        let dim = 512;
        let fht = RaBitQRotator::<f32>::new(dim, Some(RotatorKind::FhtKac), 0).unwrap();
        let dense = RaBitQRotator::<f32>::new(dim, Some(RotatorKind::Dense), 0).unwrap();
        assert!(fht.memory_usage_bytes() * 100 < dense.memory_usage_bytes());
    }
}
