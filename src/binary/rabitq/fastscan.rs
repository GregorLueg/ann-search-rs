//! Batched SIMD scoring for RaBitQ's one-bit codes.
//!
//! The RaBitQ estimator only ever needs one query-dependent quantity per stored
//! vector: `<sign(code), q_rot>`, the signed sum of the rotated query
//! coordinates picked out by the code's sign bits. The default path gets it by
//! quantising the query to int4 bit-planes and running an AND-popcount per
//! vector. This module gets it instead from the FAISS fast-scan trick already
//! implemented for TurboQuant: a nibble-indexed byte table, scanned 32 vectors
//! at a time with one `vpshufb` / `vqtbl1q_u8` per byte-group.
//!
//! Nothing new is needed on the kernel side. Taking four coordinates per nibble
//! with levels `[-1, +1]` makes each 16-entry sub-table the signed sum of those
//! four coordinates, which is exactly the shape
//! [`build_query_lut`](crate::binary::turboquant::search::build_query_lut)
//! already builds and the shape the scoring kernels already scan. The only
//! adaptation is bit order: RaBitQ packs coordinate `d` into bit `d % 8`, and
//! the blocked layout wants the lowest-indexed coordinate in the most
//! significant bit, so each code byte is reversed on the way in.

use num_traits::ToPrimitive;

use crate::binary::turboquant::pack::{BlockedCodes, BLOCK};
use crate::binary::turboquant::search::{build_query_lut, QueryLut};
use crate::prelude::*;

////////////
// Consts //
////////////

/// Code levels for the one-bit LUT.
///
/// A set bit means the rotated residual coordinate was non-negative, so the
/// sign vector the estimator wants is `-1` for a clear bit and `+1` for a set
/// one. Feeding these as the "levels" turns the generic nibble table into the
/// signed partial sums RaBitQ needs.
const SIGN_LEVELS: [f32; 2] = [-1.0, 1.0];

/// Bits per coordinate in a RaBitQ code.
const RABITQ_BITS: usize = 1;

/// Tag for the x86 blocked byte order.
const ARCH_X86: u8 = 1;

/// Tag for the natural-lane blocked byte order used everywhere else.
const ARCH_OTHER: u8 = 0;

/// Which blocked byte order this build produces.
///
/// The layout is not portable: x86 splits each code byte into nibbles and
/// interleaves lane pairs so the `vpshufb` cross-lane behaviour lines up, while
/// every other target keeps lanes in natural order. A saved index records the
/// tag it was written with so a load on a different machine can re-block rather
/// than read nonsense.
pub const BLOCKED_ARCH: u8 = if cfg!(target_arch = "x86_64") {
    ARCH_X86
} else {
    ARCH_OTHER
};

/// Lane order the x86 packing interleaves with.
const X86_PERM: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

/// Inverse of [`X86_PERM`].
const X86_PERM_INV: [usize; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15];

/////////////
// Packing //
/////////////

/// Re-pack a run of RaBitQ codes into the blocked fast-scan layout.
///
/// The run is a contiguous slice of one cluster's codes, `n_vectors` rows of
/// `n_bytes`. Each byte is bit-reversed so that coordinate `g * 8 + 0` lands in
/// the most significant bit, which is where the nibble sub-tables expect it,
/// and the rows are then interleaved into blocks of [`BLOCK`]. Rows past
/// `n_vectors` in the final block decode from zero bytes and are the caller's
/// to mask.
///
/// ### Params
///
/// * `codes` - Row-major RaBitQ codes, `n_vectors * n_bytes` long
/// * `n_vectors` - Number of codes in the run
/// * `n_bytes` - Bytes per code, `padded_dim / 8`
///
/// ### Returns
///
/// The blocked layout
pub fn pack_rabitq_blocked(codes: &[u8], n_vectors: usize, n_bytes: usize) -> BlockedCodes {
    pack_rabitq_blocked_for(codes, n_vectors, n_bytes, BLOCKED_ARCH)
}

/// Pack into a chosen architecture's blocked byte order.
///
/// Both orders are built here rather than behind `#[cfg]` so either can be
/// produced, and therefore tested, on any machine: a silent disagreement
/// between the packer and [`unpack_rabitq_blocked`] would give wrong distances
/// on the other architecture with nothing local to catch it.
///
/// ### Params
///
/// * `codes` - Row-major RaBitQ codes, `n_vectors * n_bytes` long
/// * `n_vectors` - Number of codes in the run
/// * `n_bytes` - Bytes per code
/// * `dst_arch` - The [`BLOCKED_ARCH`] tag to pack for
///
/// ### Returns
///
/// The blocked layout
pub fn pack_rabitq_blocked_for(
    codes: &[u8],
    n_vectors: usize,
    n_bytes: usize,
    dst_arch: u8,
) -> BlockedCodes {
    debug_assert_eq!(codes.len(), n_vectors * n_bytes);

    let n_blocks = n_vectors.div_ceil(BLOCK);
    let mut data = vec![0u8; n_blocks * n_bytes * BLOCK];

    // Coordinate `g * 8 + 0` has to end up in the most significant bit, which
    // is where the nibble sub-tables expect it, so every code byte is reversed
    // on the way in.
    let byte = |v: usize, g: usize| -> u8 {
        if v < n_vectors {
            codes[v * n_bytes + g].reverse_bits()
        } else {
            0
        }
    };

    for block_idx in 0..n_blocks {
        let base = block_idx * BLOCK;
        for g in 0..n_bytes {
            let out = &mut data[(block_idx * n_bytes + g) * BLOCK..][..BLOCK];
            if dst_arch == ARCH_X86 {
                // Split each byte into nibbles and interleave the lane pairs
                // `(X86_PERM[j], X86_PERM[j] + 16)`, which is what lines the
                // lane-crossing `vpshufb` reduction up.
                for j in 0..16 {
                    let a = byte(base + X86_PERM[j], g);
                    let b = byte(base + X86_PERM[j] + 16, g);
                    out[j] = (a >> 4) | ((b >> 4) << 4);
                    out[16 + j] = (a & 0x0F) | ((b & 0x0F) << 4);
                }
            } else {
                for (lane, slot) in out.iter_mut().enumerate() {
                    *slot = byte(base + lane, g);
                }
            }
        }
    }

    BlockedCodes { data, n_blocks }
}

/// Decode a blocked layout back to row-major RaBitQ codes.
///
/// The inverse of [`pack_rabitq_blocked`], and it can read either byte order so
/// an index saved on one architecture can be re-blocked on another. Padding
/// lanes in the final block are dropped.
///
/// ### Params
///
/// * `blocked` - Blocked bytes
/// * `n_vectors` - Number of codes the layout holds
/// * `n_bytes` - Bytes per code
/// * `src_arch` - The [`BLOCKED_ARCH`] tag the layout was written with
///
/// ### Returns
///
/// Row-major codes, `n_vectors * n_bytes` long
pub fn unpack_rabitq_blocked(
    blocked: &[u8],
    n_vectors: usize,
    n_bytes: usize,
    src_arch: u8,
) -> Vec<u8> {
    let n_blocks = n_vectors.div_ceil(BLOCK);
    let mut out = vec![0u8; n_vectors * n_bytes];

    for block_idx in 0..n_blocks {
        for g in 0..n_bytes {
            let group = &blocked[(block_idx * n_bytes + g) * BLOCK..][..BLOCK];
            for lane in 0..BLOCK {
                let v = block_idx * BLOCK + lane;
                if v >= n_vectors {
                    break;
                }
                let byte = if src_arch == ARCH_X86 {
                    // Lanes 0..16 hold the high nibbles of the pair, 16..32 the
                    // low ones, with the pair members at `j` and `j + 16`.
                    let (j, shift) = (X86_PERM_INV[lane % 16], 4 * (lane / 16) as u32);
                    let hi = (group[j] >> shift) & 0x0F;
                    let lo = (group[16 + j] >> shift) & 0x0F;
                    (hi << 4) | lo
                } else {
                    group[lane]
                };
                out[v * n_bytes + g] = byte.reverse_bits();
            }
        }
    }

    out
}

/////////
// LUT //
/////////

/// Build the per-query lookup table for one cluster.
///
/// ### Params
///
/// * `q_c_rotated` - The rotated, unit-length query residual, `padded_dim` long
///
/// ### Returns
///
/// The table, or an error if `padded_dim` is not a multiple of eight
pub fn build_sign_lut<T>(q_c_rotated: &[T]) -> Result<QueryLut, AnnSearchErrors>
where
    T: ToPrimitive,
{
    let q_f32: Vec<f32> = q_c_rotated
        .iter()
        .map(|x| x.to_f32().unwrap_or(0.0))
        .collect();

    build_query_lut(&q_f32, &SIGN_LEVELS, RABITQ_BITS, q_f32.len())
}

/// A query prepared for one cluster's fast-scan.
///
/// The table hands back the signed inner product directly, so there are no
/// bit-planes, quantisation bounds or quantised sum to carry: the counting
/// corrections those would exist for fall away with them.
pub struct SignScanQuery<T> {
    /// Nibble table over the rotated query residual
    pub lut: QueryLut,
    /// `||q - c||` in the rotated frame
    pub dist_to_centroid: T,
}

//////////////
// Scoring //
//////////////

/// Score one block of up to [`BLOCK`] codes against a query table.
///
/// Writes `<sign(code_lane), q_rot>` for every lane, padding lanes included.
///
/// ### Params
///
/// * `lut` - Table from [`build_sign_lut`]
/// * `blocked` - Blocked codes from [`pack_rabitq_blocked`]
/// * `block_idx` - Which block to score
/// * `out` - Per-lane output
#[inline]
pub fn score_sign_block(lut: &QueryLut, blocked: &[u8], block_idx: usize, out: &mut [f32; BLOCK]) {
    #[cfg(target_arch = "x86_64")]
    {
        use crate::binary::turboquant::search::score_block_avx2;
        if matches!(detect_simd_level(), SimdLevel::Avx2 | SimdLevel::Avx512)
            && is_x86_feature_detected!("fma")
        {
            // SAFETY: AVX2 and FMA just checked; the kernel's length
            // requirements are the same ones `score_block_scalar` asserts.
            unsafe { score_block_avx2(lut, blocked, block_idx, out) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use crate::binary::turboquant::search::score_block_neon;
        // SAFETY: NEON is in the aarch64 baseline.
        unsafe { score_block_neon(lut, blocked, block_idx, out) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    crate::binary::turboquant::search::score_block_scalar(lut, blocked, block_idx, out);
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary::turboquant::search::score_block_scalar;

    /// Deterministic pseudo-random floats in `[-0.5, 0.5)`.
    fn floats(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f32 / (1u64 << 53) as f32 - 0.5
            })
            .collect()
    }

    /// RaBitQ-style codes: bit `d % 8` of byte `d / 8` holds coordinate `d`.
    fn codes(n_vectors: usize, n_bytes: usize, seed: u64) -> Vec<u8> {
        let mut state = seed.wrapping_mul(0xD6E8_FEB8_6659_FD93) | 1;
        (0..n_vectors * n_bytes)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            })
            .collect()
    }

    /// The quantity the estimator wants, computed directly.
    fn exact_sign_dot(code: &[u8], q: &[f32]) -> f32 {
        q.iter()
            .enumerate()
            .map(|(d, &x)| {
                if code[d / 8] & (1 << (d % 8)) != 0 {
                    x
                } else {
                    -x
                }
            })
            .sum()
    }

    fn assert_block_tracks_exact(padded_dim: usize, n_vectors: usize) {
        let n_bytes = padded_dim / 8;
        let q = floats(padded_dim, padded_dim as u64);
        let raw = codes(n_vectors, n_bytes, 17);

        let lut = build_sign_lut(&q).unwrap();
        let blocked = pack_rabitq_blocked(&raw, n_vectors, n_bytes);

        // The table quantises to u8, so allow a slack proportional to the
        // number of byte-groups the error accumulates over.
        let tol = lut.scale * (n_bytes as f32) * 1.5;

        let mut out = [0.0f32; BLOCK];
        for block_idx in 0..blocked.n_blocks {
            score_sign_block(&lut, &blocked.data, block_idx, &mut out);
            for lane in 0..BLOCK {
                let v = block_idx * BLOCK + lane;
                if v >= n_vectors {
                    continue;
                }
                let want = exact_sign_dot(&raw[v * n_bytes..(v + 1) * n_bytes], &q);
                assert!(
                    (out[lane] - want).abs() <= tol,
                    "padded_dim {padded_dim} vector {v}: got {} want {want} tol {tol}",
                    out[lane]
                );
            }
        }
    }

    #[test]
    fn test_block_tracks_the_exact_sign_dot() {
        for padded_dim in [64usize, 128, 256, 512, 768, 1024] {
            assert_block_tracks_exact(padded_dim, 100);
        }
    }

    #[test]
    fn test_partial_final_block_is_handled() {
        // 70 vectors is two full blocks plus six lanes.
        assert_block_tracks_exact(128, 70);
        assert_block_tracks_exact(128, 1);
        assert_block_tracks_exact(128, 32);
        assert_block_tracks_exact(128, 33);
    }

    #[test]
    fn test_both_arch_layouts_round_trip() {
        // The x86 packer and the x86 decoder are cfg-free precisely so this
        // runs everywhere. A silent disagreement between them would give wrong
        // distances on the architecture this machine is not, and no local test
        // gated on `target_arch` could ever see it.
        for arch in [ARCH_OTHER, ARCH_X86] {
            for (padded_dim, n_vectors) in [
                (128usize, 100usize),
                (64, 1),
                (64, 31),
                (256, 32),
                (512, 33),
                (128, 70),
            ] {
                let n_bytes = padded_dim / 8;
                let raw = codes(n_vectors, n_bytes, 21);
                let blocked = pack_rabitq_blocked_for(&raw, n_vectors, n_bytes, arch);
                let back = unpack_rabitq_blocked(&blocked.data, n_vectors, n_bytes, arch);
                assert_eq!(
                    back, raw,
                    "arch {arch} padded_dim {padded_dim} n {n_vectors}"
                );
            }
        }
    }

    #[test]
    fn test_layout_agrees_with_the_kernel_packer() {
        // The scoring kernels are TurboQuant's, so this packing has to be
        // byte-identical to the one they were written against. Only the arm
        // for the machine running this can be checked, which is why CI on the
        // other architecture is what closes the loop.
        use crate::binary::turboquant::pack::pack_blocked;

        let (n_vectors, n_bytes) = (70usize, 16usize);
        let raw = codes(n_vectors, n_bytes, 91);
        let reversed: Vec<u8> = raw.iter().map(|b| b.reverse_bits()).collect();

        let mine = pack_rabitq_blocked(&raw, n_vectors, n_bytes);
        let theirs = pack_blocked(n_vectors, n_vectors.div_ceil(BLOCK), n_bytes, &reversed);

        assert_eq!(mine.data, theirs);
    }

    #[test]
    fn test_the_two_arch_layouts_are_actually_different() {
        // Guards the round-trip test above from passing vacuously, which is
        // what it would do if both branches produced the same bytes.
        let (n_vectors, n_bytes) = (70usize, 16usize);
        let raw = codes(n_vectors, n_bytes, 5);
        let a = pack_rabitq_blocked_for(&raw, n_vectors, n_bytes, ARCH_OTHER);
        let b = pack_rabitq_blocked_for(&raw, n_vectors, n_bytes, ARCH_X86);
        assert_eq!(a.data.len(), b.data.len());
        assert_ne!(a.data, b.data);
    }

    #[test]
    fn test_cross_arch_reblock_recovers_the_native_layout() {
        // What a load of a foreign index does: decode the other order, repack
        // in this one, and land on what a fresh encode here would have built.
        let (n_vectors, n_bytes) = (100usize, 32usize);
        let raw = codes(n_vectors, n_bytes, 77);

        let foreign_arch = if BLOCKED_ARCH == ARCH_X86 {
            ARCH_OTHER
        } else {
            ARCH_X86
        };
        let foreign = pack_rabitq_blocked_for(&raw, n_vectors, n_bytes, foreign_arch);

        let decoded = unpack_rabitq_blocked(&foreign.data, n_vectors, n_bytes, foreign_arch);
        let reblocked = pack_rabitq_blocked(&decoded, n_vectors, n_bytes);
        let native = pack_rabitq_blocked(&raw, n_vectors, n_bytes);

        assert_eq!(reblocked.data, native.data);
    }

    #[test]
    fn test_simd_block_matches_the_scalar_kernel() {
        let (padded_dim, n_vectors) = (256usize, 96usize);
        let n_bytes = padded_dim / 8;
        let q = floats(padded_dim, 5);
        let raw = codes(n_vectors, n_bytes, 9);

        let lut = build_sign_lut(&q).unwrap();
        let blocked = pack_rabitq_blocked(&raw, n_vectors, n_bytes);

        let mut simd = [0.0f32; BLOCK];
        let mut scalar = [0.0f32; BLOCK];
        for block_idx in 0..blocked.n_blocks {
            score_sign_block(&lut, &blocked.data, block_idx, &mut simd);
            score_block_scalar(&lut, &blocked.data, block_idx, &mut scalar);
            for lane in 0..BLOCK {
                approx::assert_relative_eq!(simd[lane], scalar[lane], epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn test_all_ones_and_all_zeros_codes() {
        let padded_dim = 128;
        let n_bytes = padded_dim / 8;
        let q = floats(padded_dim, 3);
        let sum: f32 = q.iter().sum();

        let mut raw = vec![0xFFu8; n_bytes];
        raw.extend(std::iter::repeat_n(0u8, n_bytes));

        let lut = build_sign_lut(&q).unwrap();
        let blocked = pack_rabitq_blocked(&raw, 2, n_bytes);

        let mut out = [0.0f32; BLOCK];
        score_sign_block(&lut, &blocked.data, 0, &mut out);

        let tol = lut.scale * (n_bytes as f32) * 1.5;
        assert!((out[0] - sum).abs() <= tol, "all-ones: {} vs {sum}", out[0]);
        assert!(
            (out[1] + sum).abs() <= tol,
            "all-zeros: {} vs {}",
            out[1],
            -sum
        );
    }
}
