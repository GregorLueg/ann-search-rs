//! SIMD-optimised distance functions for the binary indices

#![allow(dead_code)]

use num_traits::{Float, FromPrimitive};

use crate::binary::rabitq::*;
#[allow(unused_imports)]
use crate::prelude::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

////////////////////
// VectorDistance //
////////////////////

////////////
// Consts //
////////////

/// Vectors scored per call to [`hamming_block`]. The block exists so a caller
/// can reject a whole run of candidates against its current heap top with one
/// comparison instead of one per vector; 32 keeps the output array in
/// registers while still amortising that gate.
pub const HAMMING_BLOCK: usize = 32;

/// Vectors scored per call to [`VectorDistanceRaBitQ::rabitq_block_sq`]. Same
/// role as [`HAMMING_BLOCK`], but the output is `T` rather than `u32`.
pub const RABITQ_BLOCK: usize = 32;

/// Code length at or below which the scalar `u64` path is used on x86_64.
/// `POPCNT` is single-cycle there, so for short codes the horizontal reduction
/// the vector kernels need costs more than the popcounts themselves.
#[cfg(target_arch = "x86_64")]
const SCALAR_POPCNT_MAX_BYTES: usize = 32;

/// 16-byte chunks accumulated into u8 lanes before a widening reduction on
/// NEON. Each `vcntq_u8` lane is at most 8, so 31 chunks cannot overflow a u8
/// lane, and `vaddlvq_u8` then folds 16 lanes of at most 255 into a u16.
#[cfg(target_arch = "aarch64")]
const NEON_POPCNT_FLUSH: usize = 31;

//////////
// SIMD //
//////////

/// Cached `avx512vpopcntdq` availability.
///
/// Kept local rather than folded into `SimdLevel` because it only changes the
/// popcount kernels; every other dispatch in the crate is unaffected by it.
#[cfg(target_arch = "x86_64")]
static HAS_VPOPCNTDQ: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether the CPU has AVX-512 VPOPCNTDQ
///
/// ### Returns
///
/// `true` when `_mm512_popcnt_epi64` may be called
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_vpopcntdq() -> bool {
    *HAS_VPOPCNTDQ.get_or_init(|| {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vpopcntdq")
    })
}

/// Hamming distance over `u64` words using the scalar popcount instruction
///
/// Two accumulators so the popcounts issue independently instead of forming a
/// single dependency chain. This is the fastest path on x86_64 for short codes
/// and the portable fallback on targets with no vector kernel.
///
/// ### Params
///
/// * `a` - Slice of u8 to use
/// * `b` - Slice of u8 to use, same length as `a`
///
/// ### Returns
///
/// The Hamming distance between the two slices
#[inline(always)]
unsafe fn hamming_u64(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len();
    let n_words = len / 8;
    let pa = a.as_ptr() as *const u64;
    let pb = b.as_ptr() as *const u64;

    let (mut c0, mut c1) = (0u32, 0u32);
    let mut w = 0;
    while w + 1 < n_words {
        let x0 = pa.add(w).read_unaligned() ^ pb.add(w).read_unaligned();
        let x1 = pa.add(w + 1).read_unaligned() ^ pb.add(w + 1).read_unaligned();
        c0 += x0.count_ones();
        c1 += x1.count_ones();
        w += 2;
    }
    if w < n_words {
        c0 += (pa.add(w).read_unaligned() ^ pb.add(w).read_unaligned()).count_ones();
    }

    let mut count = c0 + c1;
    for i in (n_words * 8)..len {
        count += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones();
    }

    count
}

/// Hamming distance for AVX-512 with VPOPCNTDQ
///
/// One `vpopcntq` replaces the nibble lookup, the two masks and the two
/// shuffles that the plain AVX-512 path needs per chunk.
///
/// ### Params
///
/// * `a` - Slice of u8 to use
/// * `b` - Slice of u8 to use
///
/// ### Returns
///
/// The Hamming distance between the two slices
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vpopcntdq")]
unsafe fn hamming_avx512_vpopcnt(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len();
    let n_chunks = len / 64;
    let mut acc = _mm512_setzero_si512();

    for i in 0..n_chunks {
        let offset = i * 64;
        let va = _mm512_loadu_si512(a.as_ptr().add(offset) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(offset) as *const __m512i);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_xor_si512(va, vb)));
    }

    let lanes = std::mem::transmute::<__m512i, [u64; 8]>(acc);
    let mut count: u64 = lanes.iter().sum();

    for i in (n_chunks * 64)..len {
        count += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
    }

    count as u32
}

/// Hamming distance for AVX-512
///
/// ### Params
///
/// * `a` - Slice of u8 to use
/// * `b` - Slice of u8 to use
///
/// ### Returns
///
/// The Hamming distance between the two slices
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn hamming_avx512(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len();
    let n_chunks = len / 64;

    let nibble = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    let lookup = _mm512_broadcast_i32x4(nibble);
    let low_mask = _mm512_set1_epi8(0x0f);
    let zero = _mm512_setzero_si512();
    let mut acc = _mm512_setzero_si512();

    for i in 0..n_chunks {
        let offset = i * 64;
        let va = _mm512_loadu_si512(a.as_ptr().add(offset) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(offset) as *const __m512i);
        let v = _mm512_xor_si512(va, vb);

        let lo = _mm512_and_si512(v, low_mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), low_mask);
        let local = _mm512_add_epi8(
            _mm512_shuffle_epi8(lookup, lo),
            _mm512_shuffle_epi8(lookup, hi),
        );
        acc = _mm512_add_epi64(acc, _mm512_sad_epu8(local, zero));
    }

    let lanes = std::mem::transmute::<__m512i, [u64; 8]>(acc);
    let mut count: u64 = lanes.iter().sum();

    for i in (n_chunks * 64)..len {
        count += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
    }

    count as u32
}

/// Hamming distance for AVX-2
///
/// ### Params
///
/// * `a` - Slice of u8 to use
/// * `b` - Slice of u8 to use
///
/// ### Returns
///
/// The Hamming distance between the two slices
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hamming_avx2(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len();
    let n_chunks = len / 32;

    let nibble = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    let lookup = _mm256_broadcastsi128_si256(nibble);
    let low_mask = _mm256_set1_epi8(0x0f);
    let zero = _mm256_setzero_si256();
    let mut acc = _mm256_setzero_si256();

    for i in 0..n_chunks {
        let offset = i * 32;
        let va = _mm256_loadu_si256(a.as_ptr().add(offset) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);
        let v = _mm256_xor_si256(va, vb);

        let lo = _mm256_and_si256(v, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
        let local = _mm256_add_epi8(
            _mm256_shuffle_epi8(lookup, lo),
            _mm256_shuffle_epi8(lookup, hi),
        );
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, zero));
    }

    let lanes = std::mem::transmute::<__m256i, [u64; 4]>(acc);
    let mut count: u64 = lanes.iter().sum();

    for i in (n_chunks * 32)..len {
        count += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
    }

    count as u32
}

/// Hamming distance for SSE2
///
/// ### Params
///
/// * `a` - Slice of u8 to use
/// * `b` - Slice of u8 to use
///
/// ### Returns
///
/// The Hamming distance between the two slices
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn hamming_sse2(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len();
    let n_chunks = len / 16;

    let m1 = _mm_set1_epi8(0x55);
    let m2 = _mm_set1_epi8(0x33);
    let m4 = _mm_set1_epi8(0x0f);
    let zero = _mm_setzero_si128();
    let mut acc = _mm_setzero_si128();

    for i in 0..n_chunks {
        let offset = i * 16;
        let va = _mm_loadu_si128(a.as_ptr().add(offset) as *const __m128i);
        let vb = _mm_loadu_si128(b.as_ptr().add(offset) as *const __m128i);
        let mut v = _mm_xor_si128(va, vb);

        // SWAR per-byte popcount (16-bit shifts; masks clean cross-byte bits)
        v = _mm_sub_epi8(v, _mm_and_si128(_mm_srli_epi16(v, 1), m1));
        v = _mm_add_epi8(
            _mm_and_si128(v, m2),
            _mm_and_si128(_mm_srli_epi16(v, 2), m2),
        );
        v = _mm_and_si128(_mm_add_epi8(v, _mm_srli_epi16(v, 4)), m4);

        acc = _mm_add_epi64(acc, _mm_sad_epu8(v, zero));
    }

    let lanes = std::mem::transmute::<__m128i, [u64; 2]>(acc);
    let mut count = lanes[0] + lanes[1];

    for i in (n_chunks * 16)..len {
        count += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
    }

    count as u32
}

/// Hamming distance for NEON
///
/// The per-byte popcounts accumulate in u8 lanes and fold once per
/// [`NEON_POPCNT_FLUSH`] chunks with a single `uaddlv`. Widening every chunk
/// through `vpaddlq_u8`/`vpaddlq_u16`/`vpadalq_u32` instead puts a four-deep
/// reduction chain plus two lane extracts behind what is frequently a single
/// chunk of real work.
///
/// ### Params
///
/// * `a` - Slice of u8 to use
/// * `b` - Slice of u8 to use
///
/// ### Returns
///
/// The Hamming distance between the two slices
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hamming_neon(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len();
    let n_chunks = len / 16;
    let mut count = 0u32;

    let mut chunk = 0;
    while chunk < n_chunks {
        let batch_end = (chunk + NEON_POPCNT_FLUSH).min(n_chunks);
        let mut acc = vdupq_n_u8(0);

        while chunk < batch_end {
            let offset = chunk * 16;
            let va = vld1q_u8(a.as_ptr().add(offset));
            let vb = vld1q_u8(b.as_ptr().add(offset));
            acc = vaddq_u8(acc, vcntq_u8(veorq_u8(va, vb)));
            chunk += 1;
        }

        count += vaddlvq_u8(acc) as u32;
    }

    for i in (n_chunks * 16)..len {
        count += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones();
    }

    count
}

/// Hamming distance - scalar fall back
///
/// ### Params
///
/// * `a` - Slice of u8 to use
/// * `b` - Slice of u8 to use
///
/// ### Returns
///
/// The Hamming distance between the two slices
#[inline(always)]
unsafe fn hamming_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Hamming distance - SIMD dispatcher
///
/// ### Params
///
/// * `a` - Slice of u8 to use
/// * `b` - Slice of u8 to use
///
/// ### Returns
///
/// The Hamming distance between the two slices
#[inline(always)]
unsafe fn hamming_simd(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() <= SCALAR_POPCNT_MAX_BYTES {
            return hamming_u64(a, b);
        }
        if has_vpopcntdq() {
            return hamming_avx512_vpopcnt(a, b);
        }
        match detect_simd_level() {
            SimdLevel::Avx512 => hamming_avx512(a, b),
            SimdLevel::Avx2 => hamming_avx2(a, b),
            SimdLevel::Sse => hamming_sse2(a, b),
            SimdLevel::Scalar => hamming_scalar(a, b),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        hamming_neon(a, b)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        hamming_u64(a, b)
    }
}

/// Hamming distance between one query and a contiguous run of codes
///
/// Fills `out` with one distance per code and returns the smallest of them, so
/// the caller can reject the whole block against its current heap top with a
/// single comparison rather than one heap probe per candidate.
///
/// ### Params
///
/// * `query` - Binarised query, `n_bytes` long
/// * `codes` - Contiguous run of binarised vectors, `out.len() * n_bytes` long
/// * `n_bytes` - Bytes per vector
/// * `out` - Per-vector distance output; its length sets the block size
///
/// ### Returns
///
/// The minimum distance written into `out`, or `u32::MAX` when `out` is empty
#[inline]
pub fn hamming_block(query: &[u8], codes: &[u8], n_bytes: usize, out: &mut [u32]) -> u32 {
    debug_assert_eq!(query.len(), n_bytes);
    debug_assert!(codes.len() >= out.len() * n_bytes);

    let mut min = u32::MAX;
    for (j, slot) in out.iter_mut().enumerate() {
        let start = j * n_bytes;
        let dist = unsafe { hamming_simd(query, codes.get_unchecked(start..start + n_bytes)) };
        *slot = dist;
        min = min.min(dist);
    }

    min
}

/// Trait for computing distances between binarised vectors
pub trait VectorDistanceBinary {
    /// Get the internal flat vector representation (binarised to u8)
    ///
    /// ### Returns
    ///
    /// Reference to the flat binarised vector storage
    fn vectors_flat_binarised(&self) -> &[u8];

    /// Get the number of bytes(!) used binarisation
    ///
    /// ### Returns
    ///
    /// Number of bytes per vector
    fn n_bytes(&self) -> usize;

    /// Calculates the Hamming distance between two internal vectors
    ///
    /// ### Params
    ///
    /// * `i` - Position of i in the internal flat vec representation
    /// * `j` - Position of j in the internal flat vec representation
    ///
    /// ### Returns
    ///
    /// Hamming distance
    #[inline(always)]
    fn hamming_distance(&self, i: usize, j: usize) -> u32 {
        let start_i = i * self.n_bytes();
        let start_j = j * self.n_bytes();

        unsafe {
            let vec_i = self
                .vectors_flat_binarised()
                .get_unchecked(start_i..start_i + self.n_bytes());
            let vec_j = self
                .vectors_flat_binarised()
                .get_unchecked(start_j..start_j + self.n_bytes());

            hamming_simd(vec_i, vec_j)
        }
    }

    /// Calculates the Hamming distance between two internal vectors
    ///
    /// ### Params
    ///
    /// * `query` - The query projected into binary space
    /// * `i` - Position of j in the internal flat vec representation
    ///
    /// ### Returns
    ///
    /// Hamming distance between query and internal vector
    #[inline(always)]
    fn hamming_distance_query(&self, query: &[u8], i: usize) -> u32 {
        let start_i = i * self.n_bytes();

        unsafe {
            let vec_i = self
                .vectors_flat_binarised()
                .get_unchecked(start_i..start_i + self.n_bytes());

            hamming_simd(vec_i, query)
        }
    }
}

/// Sum of the query entries selected by the set bits of a binary code
///
/// Two accumulators and a conditional-move per bit, so nothing branches on the
/// (effectively random) code bits and nothing is allocated.
///
/// ### Params
///
/// * `query_vec` - Float query vector, `dim` long
/// * `binary_code` - Packed binary code (bit-packed u8 array)
/// * `dim` - Vector dimensionality (number of bits to read)
///
/// ### Returns
///
/// `sum over d where bit d is set of query_vec[d]`
#[inline]
pub fn masked_query_sum<T>(query_vec: &[T], binary_code: &[u8], dim: usize) -> T
where
    T: Float,
{
    let (mut a0, mut a1) = (T::zero(), T::zero());
    let full_bytes = dim / 8;

    for byte_idx in 0..full_bytes {
        let bits = unsafe { *binary_code.get_unchecked(byte_idx) };
        let base = byte_idx * 8;

        for pair in 0..4 {
            let d = base + pair * 2;
            let q0 = unsafe { *query_vec.get_unchecked(d) };
            let q1 = unsafe { *query_vec.get_unchecked(d + 1) };
            a0 = if (bits >> (pair * 2)) & 1 == 1 {
                a0 + q0
            } else {
                a0
            };
            a1 = if (bits >> (pair * 2 + 1)) & 1 == 1 {
                a1 + q1
            } else {
                a1
            };
        }
    }

    let remaining = dim % 8;
    if remaining > 0 {
        let bits = binary_code[full_bytes];
        let base = full_bytes * 8;
        for bit_pos in 0..remaining {
            let q = query_vec[base + bit_pos];
            a0 = if (bits >> bit_pos) & 1 == 1 { a0 + q } else { a0 };
        }
    }

    a0 + a1
}

/// Asymmetric dot product: query (float) vs binary vector, query sum supplied
///
/// Uses `dot(q, 2b - 1) = 2 * sum_{d: b_d = 1} q_d - sum_d q_d`, so the
/// `{-1, +1}` expansion never has to be materialised. Callers that score many
/// codes against one query hoist `query_sum` out of the loop.
///
/// ### Params
///
/// * `query_vec` - Float query vector
/// * `query_sum` - `sum_d query_vec[d]` over the first `dim` entries
/// * `binary_code` - Packed binary code (bit-packed u8 array)
/// * `dim` - Vector dimensionality (number of bits to unpack)
///
/// ### Returns
///
/// Dot product score (higher = more similar)
#[inline]
pub fn asymmetric_binary_dot_presummed<T>(
    query_vec: &[T],
    query_sum: T,
    binary_code: &[u8],
    dim: usize,
) -> T
where
    T: Float,
{
    // Hard assert, not debug: `masked_query_sum` indexes unchecked up to `dim`
    assert_eq!(query_vec.len(), dim);

    let two = T::one() + T::one();
    two * masked_query_sum(query_vec, binary_code, dim) - query_sum
}

/// Asymmetric dot product: query (float) vs binary vector
///
/// Computes dot(query_float, 2*binary-1) where binary is unpacked to `{-1, +1} `
/// from bit representation.
///
/// ### Params
///
/// * `query_vec` - Float query vector
/// * `binary_code` - Packed binary code (bit-packed u8 array)
/// * `dim` - Vector dimensionality (number of bits to unpack)
///
/// ### Returns
///
/// Dot product score (higher = more similar)
#[inline]
pub fn asymmetric_binary_dot<T>(query_vec: &[T], binary_code: &[u8], dim: usize) -> T
where
    T: Float,
{
    assert_eq!(query_vec.len(), dim);

    let query_sum = query_vec.iter().fold(T::zero(), |acc, &x| acc + x);
    asymmetric_binary_dot_presummed(query_vec, query_sum, binary_code, dim)
}

//////////////////////////
// VectorDistanceRaBitQ //
//////////////////////////

//////////
// SIMD //
//////////

/// Bit-planes in the int4 quantised RaBitQ query.
///
/// `encode_query` quantises to 0..=15, so four planes reproduce every query
/// coordinate exactly and the inner product becomes
/// `sum_j 2^j * popcount(plane_j AND code)`.
pub const RABITQ_QUERY_PLANES: usize = 4;

/// Bit-slice an int4 quantised query into [`RABITQ_QUERY_PLANES`] planes
///
/// Plane `j` occupies `[j * n_bytes, (j + 1) * n_bytes)` of the output and
/// holds bit `j` of every coordinate, coordinate `d` at bit `d % 8` of byte
/// `d / 8`. That is the same bit order `RaBitQEncoder::encode_vector` uses for
/// the stored sign bits, so the two AND together directly.
///
/// ### Params
///
/// * `quantised` - Int4 values, one per dimension
/// * `dim` - Vector dimensionality
/// * `n_bytes` - Bytes per plane, `dim.div_ceil(8)`
///
/// ### Returns
///
/// A `RABITQ_QUERY_PLANES * n_bytes` buffer of bit-planes
#[inline]
pub fn build_query_planes(quantised: &[u8], dim: usize, n_bytes: usize) -> Vec<u8> {
    let mut planes = vec![0u8; RABITQ_QUERY_PLANES * n_bytes];

    for d in 0..dim {
        let value = quantised[d];
        let byte = d / 8;
        let bit = 1u8 << (d % 8);
        for j in 0..RABITQ_QUERY_PLANES {
            if (value >> j) & 1 == 1 {
                planes[j * n_bytes + byte] |= bit;
            }
        }
    }

    planes
}

/// Reassemble the dense int4 query from its bit-planes
///
/// Inverse of [`build_query_planes`]. The scan path never needs this; it exists
/// so the plane layout can be checked against the dense reference kernel.
///
/// ### Params
///
/// * `planes` - Query bit-planes from [`build_query_planes`]
/// * `dim` - Vector dimensionality
/// * `n_bytes` - Bytes per plane
///
/// ### Returns
///
/// The int4 values, one per dimension
#[inline]
pub fn unpack_query_planes(planes: &[u8], dim: usize, n_bytes: usize) -> Vec<u8> {
    let mut quantised = vec![0u8; dim];

    for (d, slot) in quantised.iter_mut().enumerate() {
        let byte = d / 8;
        let bit = d % 8;
        for j in 0..RABITQ_QUERY_PLANES {
            *slot |= ((planes[j * n_bytes + byte] >> bit) & 1) << j;
        }
    }

    quantised
}

/// Bit-plane dot product over `u64` words
///
/// ### Params
///
/// * `planes` - Query bit-planes from [`build_query_planes`]
/// * `binary` - The binary code of one stored vector
/// * `n_bytes` - Bytes per plane and per code
///
/// ### Returns
///
/// The dot product of the quantised query and the binary vector
#[inline(always)]
unsafe fn dot_planes_u64(planes: &[u8], binary: &[u8], n_bytes: usize) -> u32 {
    let n_words = n_bytes / 8;
    let pb = binary.as_ptr() as *const u64;
    let mut acc = 0u32;

    for j in 0..RABITQ_QUERY_PLANES {
        let pp = planes.as_ptr().add(j * n_bytes) as *const u64;
        let mut c = 0u32;
        for w in 0..n_words {
            c += (pp.add(w).read_unaligned() & pb.add(w).read_unaligned()).count_ones();
        }
        for i in (n_words * 8)..n_bytes {
            c += (*planes.get_unchecked(j * n_bytes + i) & *binary.get_unchecked(i)).count_ones();
        }
        acc += c << j;
    }

    acc
}

/// Bit-plane dot product for NEON
///
/// One accumulator per plane so 16-byte chunks fold into u8 lanes for up to
/// [`NEON_POPCNT_FLUSH`] iterations before a single `uaddlv` per plane. The
/// four query planes stay in registers across a whole cluster scan.
///
/// ### Params
///
/// * `planes` - Query bit-planes from [`build_query_planes`]
/// * `binary` - The binary code of one stored vector
/// * `n_bytes` - Bytes per plane and per code
///
/// ### Returns
///
/// The dot product of the quantised query and the binary vector
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_planes_neon(planes: &[u8], binary: &[u8], n_bytes: usize) -> u32 {
    let n_chunks = n_bytes / 16;
    let p = planes.as_ptr();
    let b = binary.as_ptr();
    let mut acc = 0u32;

    let mut chunk = 0;
    while chunk < n_chunks {
        let batch_end = (chunk + NEON_POPCNT_FLUSH).min(n_chunks);
        let mut a0 = vdupq_n_u8(0);
        let mut a1 = vdupq_n_u8(0);
        let mut a2 = vdupq_n_u8(0);
        let mut a3 = vdupq_n_u8(0);

        while chunk < batch_end {
            let off = chunk * 16;
            let vb = vld1q_u8(b.add(off));
            a0 = vaddq_u8(a0, vcntq_u8(vandq_u8(vld1q_u8(p.add(off)), vb)));
            a1 = vaddq_u8(a1, vcntq_u8(vandq_u8(vld1q_u8(p.add(n_bytes + off)), vb)));
            a2 = vaddq_u8(a2, vcntq_u8(vandq_u8(vld1q_u8(p.add(2 * n_bytes + off)), vb)));
            a3 = vaddq_u8(a3, vcntq_u8(vandq_u8(vld1q_u8(p.add(3 * n_bytes + off)), vb)));
            chunk += 1;
        }

        acc += vaddlvq_u8(a0) as u32
            + ((vaddlvq_u8(a1) as u32) << 1)
            + ((vaddlvq_u8(a2) as u32) << 2)
            + ((vaddlvq_u8(a3) as u32) << 3);
    }

    for i in (n_chunks * 16)..n_bytes {
        let bb = *binary.get_unchecked(i);
        for j in 0..RABITQ_QUERY_PLANES {
            acc += (*planes.get_unchecked(j * n_bytes + i) & bb).count_ones() << j;
        }
    }

    acc
}

/// Dot product between a bit-planed query and a binary code
///
/// `sum_j 2^j * popcount(plane_j AND code)`, which replaces the per-dimension
/// masked add the dense-query kernel needs: at dim 128 that is four AND plus
/// four popcount instructions against sixteen bytes of code, with no
/// per-dimension work at all.
///
/// ### Params
///
/// * `planes` - Query bit-planes from [`build_query_planes`]
/// * `binary` - The binary code of one stored vector
/// * `n_bytes` - Bytes per plane and per code
///
/// ### Returns
///
/// The dot product of the quantised query and the binary vector
#[inline(always)]
pub fn dot_query_binary_planes(planes: &[u8], binary: &[u8], n_bytes: usize) -> u32 {
    debug_assert_eq!(planes.len(), RABITQ_QUERY_PLANES * n_bytes);
    debug_assert_eq!(binary.len(), n_bytes);

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_planes_neon(planes, binary, n_bytes) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        unsafe { dot_planes_u64(planes, binary, n_bytes) }
    }
}

/// Scalar fallback for dot product computation
///
/// ### Params
///
/// * `query` - The query vector
/// * `binary` - The binary vector
/// * `dim` - The dimension of the vectors
///
/// ### Returns
///
/// The dot product of the query and binary vectors
#[inline(always)]
pub fn dot_query_binary_scalar(query: &[u8], binary: &[u8], dim: usize) -> u32 {
    let mut sum = 0u32;
    let full_bytes = dim / 8;

    for byte_idx in 0..full_bytes {
        let bits = binary[byte_idx];
        let base = byte_idx * 8;
        sum += query[base] as u32 * (bits & 1) as u32;
        sum += query[base + 1] as u32 * ((bits >> 1) & 1) as u32;
        sum += query[base + 2] as u32 * ((bits >> 2) & 1) as u32;
        sum += query[base + 3] as u32 * ((bits >> 3) & 1) as u32;
        sum += query[base + 4] as u32 * ((bits >> 4) & 1) as u32;
        sum += query[base + 5] as u32 * ((bits >> 5) & 1) as u32;
        sum += query[base + 6] as u32 * ((bits >> 6) & 1) as u32;
        sum += query[base + 7] as u32 * ((bits >> 7) & 1) as u32;
    }

    let remaining = dim % 8;
    if remaining > 0 {
        let bits = binary[full_bytes];
        let base = full_bytes * 8;
        for bit_pos in 0..remaining {
            sum += query[base + bit_pos] as u32 * ((bits >> bit_pos) & 1) as u32;
        }
    }

    sum
}

/// Trait for RaBitQ distance computation over CSR storage
pub trait VectorDistanceRaBitQ<T>
where
    T: Float + FromPrimitive,
{
    /// Get the RaBitQ storage
    ///
    /// ### Returns
    ///
    /// Reference to the RaBitQ storage
    fn storage(&self) -> &RaBitQStorage<T>;

    /// Get the RaBitQ encoder
    ///
    /// ### Returns
    ///
    /// Reference to the RaBitQ encoder
    fn encoder(&self) -> &RaBitQEncoder<T>;

    /// Get the vector dimensionality
    ///
    /// ### Returns
    ///
    /// Number of dimensions
    #[inline]
    fn dim(&self) -> usize {
        self.storage().dim
    }

    /// Get the number of bytes per vector
    ///
    /// ### Returns
    ///
    /// Number of bytes per vector
    #[inline]
    fn n_bytes(&self) -> usize {
        self.storage().n_bytes
    }

    /// Popcount for vector at local index within cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` - Index of the cluster
    /// * `local_idx` - Local index of the vector within the cluster
    ///
    /// ### Returns
    ///
    /// Number of set bits in the binary vector
    #[inline]
    fn popcount(&self, cluster_idx: usize, local_idx: usize) -> u32 {
        self.storage()
            .get_vector_data(cluster_idx, local_idx)
            .popcount
    }

    /// Dot product between query and binary vector
    ///
    /// ### Params
    ///
    /// * `query` - The RaBitQ query
    /// * `cluster_idx` - Index of the cluster
    /// * `local_idx` - Local index of the vector within the cluster
    ///
    /// ### Returns
    ///
    /// Quantised dot product result
    #[inline(always)]
    fn dot_query_binary(
        &self,
        query: &RaBitQQuery<T>,
        cluster_idx: usize,
        local_idx: usize,
    ) -> u32 {
        let binary = self.storage().vector_binary(cluster_idx, local_idx);
        dot_query_binary_planes(&query.planes, binary, self.n_bytes())
    }

    /// Squared RaBitQ distance estimate
    ///
    /// The ranking quantity. `rabitq_dist` is this plus a square root, which is
    /// monotone, so scans rank on this and take the root only for the survivors.
    ///
    /// ### Params
    ///
    /// * `query` - The RaBitQ query
    /// * `cluster_idx` - Index of the cluster
    /// * `local_idx` - Local index of the vector within the cluster
    ///
    /// ### Returns
    ///
    /// Estimated squared Euclidean distance, clamped at zero
    #[inline]
    fn rabitq_dist_sq(&self, query: &RaBitQQuery<T>, cluster_idx: usize, local_idx: usize) -> T {
        let storage = self.storage();
        let packed = storage.get_vector_data(cluster_idx, local_idx); // Single cache line read

        let dim_f = T::from_usize(self.dim()).unwrap();
        let two = T::one() + T::one();

        let v_dist = packed.dist_to_centroid;
        let q_dist = query.dist_to_centroid;

        let qr = T::from_u32(self.dot_query_binary(query, cluster_idx, local_idx)).unwrap();
        let popcount = T::from_u32(packed.popcount).unwrap();
        let sum_q = T::from_u32(query.sum_quantised).unwrap();

        let inner_product_sgn = two * (query.width * qr + query.lower * popcount)
            - (query.width * sum_q + dim_f * query.lower);

        // `dot_correction_inv` is stored as zero when the L1 norm underflowed at
        // build time, which reproduces the old guarded divide without a branch.
        let q_dot_v =
            (inner_product_sgn * packed.dot_correction_inv).clamp(T::one().neg(), T::one());

        (v_dist * v_dist + q_dist * q_dist - two * v_dist * q_dist * q_dot_v).max(T::zero())
    }

    /// RaBitQ distance estimate
    ///
    /// ### Params
    ///
    /// * `query` - The RaBitQ query
    /// * `cluster_idx` - Index of the cluster
    /// * `local_idx` - Local index of the vector within the cluster
    ///
    /// ### Returns
    ///
    /// Estimated Euclidean distance (Cosine works due to normalisation)
    #[inline]
    fn rabitq_dist(&self, query: &RaBitQQuery<T>, cluster_idx: usize, local_idx: usize) -> T {
        self.rabitq_dist_sq(query, cluster_idx, local_idx).sqrt()
    }

    /// Squared RaBitQ distances for a contiguous run within one cluster
    ///
    /// Hoists everything that only depends on the query out of the per-vector
    /// work and returns the block minimum, so the caller can reject the whole
    /// run against its heap top with one comparison.
    ///
    /// ### Params
    ///
    /// * `query` - The RaBitQ query, already encoded against this cluster
    /// * `cluster_idx` - Index of the cluster
    /// * `local_start` - Local index of the first vector in the run
    /// * `out` - Per-vector output; its length sets the block size
    ///
    /// ### Returns
    ///
    /// The minimum written into `out`, or infinity when `out` is empty
    #[inline]
    fn rabitq_block_sq(
        &self,
        query: &RaBitQQuery<T>,
        cluster_idx: usize,
        local_start: usize,
        out: &mut [T],
    ) -> T {
        let storage = self.storage();
        let n_bytes = self.n_bytes();

        let one = T::one();
        let two = one + one;
        let dim_f = T::from_usize(self.dim()).unwrap();
        let q_dist = query.dist_to_centroid;
        let sum_q = T::from_u32(query.sum_quantised).unwrap();
        let q_term = query.width * sum_q + dim_f * query.lower;
        let q_dist_sq = q_dist * q_dist;

        let global_start = storage.offsets[cluster_idx] + local_start;
        let mut min = T::infinity();

        for (j, slot) in out.iter_mut().enumerate() {
            let g = global_start + j;
            let packed = unsafe { storage.packed_vectors.get_unchecked(g) };
            let binary =
                unsafe { storage.binary_codes.get_unchecked(g * n_bytes..(g + 1) * n_bytes) };

            let qr = T::from_u32(dot_query_binary_planes(&query.planes, binary, n_bytes)).unwrap();
            let popcount = T::from_u32(packed.popcount).unwrap();

            let inner_product_sgn = two * (query.width * qr + query.lower * popcount) - q_term;
            let q_dot_v = (inner_product_sgn * packed.dot_correction_inv).clamp(one.neg(), one);

            let v_dist = packed.dist_to_centroid;
            let dist =
                (v_dist * v_dist + q_dist_sq - two * v_dist * q_dist * q_dot_v).max(T::zero());

            *slot = dist;
            if dist < min {
                min = dist;
            }
        }

        min
    }
}

/////////////
// Helpers //
/////////////

/// Calculate the Hamming distance between two binary vectors
///
/// ### Params
///
/// * `a` - Slice of the first binary vector
/// * `b` - Slice of the second binary vector
///
/// ### Returns
///
/// The Hamming distance between the two vectors
#[inline(always)]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    unsafe { hamming_simd(a, b) }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::binary::rabitq::RaBitQQuantiser;
    use crate::utils::dist::Dist;
    use faer::Mat;
    use faer_traits::ComplexField;

    fn create_test_data<T: Float + FromPrimitive + ComplexField>(n: usize, dim: usize) -> Mat<T> {
        let mut data = Mat::zeros(n, dim);
        for i in 0..n {
            for j in 0..dim {
                data[(i, j)] = T::from_f64((i * dim + j) as f64 * 0.1).unwrap();
            }
        }
        data
    }

    struct TestBinaryVectors {
        data: Vec<u8>,
        n_bytes: usize,
    }

    impl TestBinaryVectors {
        fn new(vectors: Vec<Vec<u8>>) -> Self {
            assert!(!vectors.is_empty());
            let n_bytes = vectors[0].len();
            assert!(vectors.iter().all(|v| v.len() == n_bytes));

            let data: Vec<u8> = vectors.into_iter().flatten().collect();
            TestBinaryVectors { data, n_bytes }
        }
    }

    impl VectorDistanceBinary for TestBinaryVectors {
        fn vectors_flat_binarised(&self) -> &[u8] {
            &self.data
        }

        fn n_bytes(&self) -> usize {
            self.n_bytes
        }
    }

    #[test]
    fn test_hamming_distance_helper() {
        assert_eq!(hamming_distance(&[0b00000000], &[0b00000000]), 0);
        assert_eq!(hamming_distance(&[0b11111111], &[0b11111111]), 0);
        assert_eq!(hamming_distance(&[0b00000000], &[0b11111111]), 8);
        assert_eq!(hamming_distance(&[0b10101010], &[0b01010101]), 8);
        assert_eq!(hamming_distance(&[0b11110000], &[0b00001111]), 8);
        assert_eq!(hamming_distance(&[0b10000000], &[0b00000000]), 1);
    }

    #[test]
    fn test_hamming_distance_multi_byte() {
        let a = vec![0b11110000, 0b10101010];
        let b = vec![0b00001111, 0b01010101];
        assert_eq!(hamming_distance(&a, &b), 16);

        let c = vec![0b11111111, 0b11111111, 0b11111111];
        let d = vec![0b00000000, 0b00000000, 0b00000000];
        assert_eq!(hamming_distance(&c, &d), 24);
    }

    #[test]
    fn test_hamming_distance_symmetry() {
        let a = vec![0b10101010, 0b11001100];
        let b = vec![0b01010101, 0b00110011];
        assert_eq!(hamming_distance(&a, &b), hamming_distance(&b, &a));
    }

    #[test]
    fn test_trait_hamming_distance_basic() {
        let vectors = vec![vec![0b00000000], vec![0b11111111], vec![0b10101010]];
        let storage = TestBinaryVectors::new(vectors);

        assert_eq!(storage.hamming_distance(0, 0), 0);
        assert_eq!(storage.hamming_distance(1, 1), 0);
        assert_eq!(storage.hamming_distance(0, 1), 8);
        assert_eq!(storage.hamming_distance(1, 0), 8);
        assert_eq!(storage.hamming_distance(0, 2), 4);
        assert_eq!(storage.hamming_distance(2, 1), 4);
    }

    #[test]
    fn test_trait_hamming_distance_multi_byte() {
        let vectors = vec![
            vec![0b11110000, 0b10101010],
            vec![0b00001111, 0b01010101],
            vec![0b00000000, 0b00000000],
        ];
        let storage = TestBinaryVectors::new(vectors);

        assert_eq!(storage.hamming_distance(0, 1), 16);
        assert_eq!(storage.hamming_distance(0, 2), 8);
        assert_eq!(storage.hamming_distance(1, 2), 8);
    }

    #[test]
    fn test_trait_hamming_distance_query() {
        let vectors = vec![vec![0b00000000], vec![0b11111111], vec![0b10101010]];
        let storage = TestBinaryVectors::new(vectors);

        let query = vec![0b11001100];
        assert_eq!(storage.hamming_distance_query(&query, 0), 4);
        assert_eq!(storage.hamming_distance_query(&query, 1), 4);
        assert_eq!(storage.hamming_distance_query(&query, 2), 4);
    }

    #[test]
    fn test_trait_query_matches_internal() {
        let vectors = vec![
            vec![0b00000000, 0b11111111],
            vec![0b10101010, 0b01010101],
            vec![0b11110000, 0b00001111],
        ];
        let storage = TestBinaryVectors::new(vectors);

        for i in 0..3 {
            for j in 0..3 {
                let query_vector = if j == 0 {
                    vec![0b00000000, 0b11111111]
                } else if j == 1 {
                    vec![0b10101010, 0b01010101]
                } else {
                    vec![0b11110000, 0b00001111]
                };

                assert_eq!(
                    storage.hamming_distance(i, j),
                    storage.hamming_distance_query(&query_vector, i)
                );
            }
        }
    }

    #[test]
    fn test_all_zeros() {
        let vectors = vec![vec![0, 0, 0, 0], vec![0, 0, 0, 0]];
        let storage = TestBinaryVectors::new(vectors);
        assert_eq!(storage.hamming_distance(0, 1), 0);
    }

    #[test]
    fn test_all_ones() {
        let vectors = vec![vec![0xFF, 0xFF, 0xFF, 0xFF], vec![0xFF, 0xFF, 0xFF, 0xFF]];
        let storage = TestBinaryVectors::new(vectors);
        assert_eq!(storage.hamming_distance(0, 1), 0);
    }

    #[test]
    fn test_single_bit_differences() {
        let vectors = vec![
            vec![0b00000000],
            vec![0b00000001],
            vec![0b00000010],
            vec![0b00000100],
            vec![0b00001000],
        ];
        let storage = TestBinaryVectors::new(vectors);

        for i in 1..5 {
            assert_eq!(storage.hamming_distance(0, i), 1);
        }

        assert_eq!(storage.hamming_distance(1, 2), 2);
        assert_eq!(storage.hamming_distance(1, 3), 2);
        assert_eq!(storage.hamming_distance(2, 4), 2);
    }

    #[test]
    fn test_large_vectors() {
        let n_bytes = 32;
        let vec1: Vec<u8> = (0..n_bytes).map(|i| i as u8).collect();
        let vec2: Vec<u8> = (0..n_bytes).map(|i| (i as u8).wrapping_mul(2)).collect();

        let vectors = vec![vec1.clone(), vec2.clone()];
        let storage = TestBinaryVectors::new(vectors);

        let expected = hamming_distance(&vec1, &vec2);
        assert_eq!(storage.hamming_distance(0, 1), expected);
    }

    #[test]
    fn test_rabitq_trait_dim() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(5), 42).unwrap();

        assert_eq!(quantiser.dim(), 32);
    }

    #[test]
    fn test_rabitq_trait_n_bytes() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(5), 42).unwrap();

        assert_eq!(quantiser.n_bytes(), 4);
    }

    #[test]
    fn test_rabitq_popcount() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(5), 42).unwrap();

        let popcount = quantiser.popcount(0, 0);
        assert!(popcount <= 32);
    }

    #[test]
    fn test_rabitq_dot_query_binary() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(5), 42).unwrap();

        let query = vec![1.0f32; 32];
        let encoded_query = quantiser.encode_query(&query, 0).unwrap();

        let dot = quantiser.dot_query_binary(&encoded_query, 0, 0);
        assert!(dot <= 15 * 32);
    }

    #[test]
    fn test_rabitq_dist_positive() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(5), 42).unwrap();

        let query = vec![1.0f32; 32];
        let encoded_query = quantiser.encode_query(&query, 0).unwrap();

        let dist = quantiser.rabitq_dist(&encoded_query, 0, 0);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_rabitq_dist_consistency() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(5), 42).unwrap();

        let query = vec![1.0f32; 32];
        let encoded_query = quantiser.encode_query(&query, 0).unwrap();

        let dist1 = quantiser.rabitq_dist(&encoded_query, 0, 0);
        let dist2 = quantiser.rabitq_dist(&encoded_query, 0, 0);

        assert_eq!(dist1, dist2);
    }

    #[test]
    fn test_rabitq_dist_different_vectors() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(5), 42).unwrap();

        let query = vec![1.0f32; 32];
        let encoded_query = quantiser.encode_query(&query, 0).unwrap();

        let cluster_size = quantiser.storage().cluster_size(0);
        if cluster_size > 1 {
            let dist0 = quantiser.rabitq_dist(&encoded_query, 0, 0);
            let dist1 = quantiser.rabitq_dist(&encoded_query, 0, 1);

            assert!(dist0 >= 0.0 && dist1 >= 0.0);
        }
    }

    #[test]
    fn test_rabitq_dist_cosine() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser = RaBitQQuantiser::new(data.as_ref(), &Dist::Cosine, Some(5), 42).unwrap();

        let query = vec![1.0f32; 32];
        let encoded_query = quantiser.encode_query(&query, 0).unwrap();

        let dist = quantiser.rabitq_dist(&encoded_query, 0, 0);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_rabitq_multiple_clusters() {
        let data = create_test_data::<f32>(100, 32);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(10), 42).unwrap();

        let query = vec![1.0f32; 32];

        for cluster_idx in 0..quantiser.storage().nlist {
            let encoded_query = quantiser.encode_query(&query, cluster_idx).unwrap();
            let cluster_size = quantiser.storage().cluster_size(cluster_idx);

            for local_idx in 0..cluster_size {
                let dist = quantiser.rabitq_dist(&encoded_query, cluster_idx, local_idx);
                assert!(dist >= 0.0);
            }
        }
    }

    /////////////////////
    // Kernel oracles  //
    /////////////////////

    /// Splitmix64-style deterministic byte stream keyed by `(seed, i)`.
    fn splitmix_byte(seed: u64, i: usize) -> u8 {
        let mut x = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(seed.wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        (x & 0xFF) as u8
    }

    /// Dimensions that between them cross every boundary the kernels care
    /// about: sub-byte tails, sub-16-byte codes, the NEON chunk width, and
    /// `NEON_POPCNT_FLUSH` (31 chunks, i.e. 496 bytes or 3968 dimensions).
    const KERNEL_TEST_DIMS: [usize; 11] = [8, 15, 16, 17, 64, 100, 128, 129, 512, 1000, 4160];

    fn random_code(seed: u64, dim: usize, n_bytes: usize) -> Vec<u8> {
        let mut code = vec![0u8; n_bytes];
        for d in 0..dim {
            if splitmix_byte(seed, d) & 1 == 1 {
                code[d / 8] |= 1u8 << (d % 8);
            }
        }
        code
    }

    #[test]
    fn test_query_planes_round_trip() {
        for dim in KERNEL_TEST_DIMS {
            let n_bytes = dim.div_ceil(8);
            let quantised: Vec<u8> = (0..dim).map(|d| splitmix_byte(7, d) & 15).collect();

            let planes = build_query_planes(&quantised, dim, n_bytes);
            assert_eq!(planes.len(), RABITQ_QUERY_PLANES * n_bytes);
            assert_eq!(unpack_query_planes(&planes, dim, n_bytes), quantised);
        }
    }

    #[test]
    fn test_dot_planes_matches_dense_scalar() {
        for dim in KERNEL_TEST_DIMS {
            let n_bytes = dim.div_ceil(8);
            let quantised: Vec<u8> = (0..dim).map(|d| splitmix_byte(11, d) & 15).collect();
            let binary = random_code(13, dim, n_bytes);

            let planes = build_query_planes(&quantised, dim, n_bytes);

            assert_eq!(
                dot_query_binary_planes(&planes, &binary, n_bytes),
                dot_query_binary_scalar(&quantised, &binary, dim),
                "bit-plane dot disagrees with the dense reference at dim {dim}"
            );
        }
    }

    #[test]
    fn test_dot_planes_extremes() {
        let dim = 128;
        let n_bytes = dim / 8;

        // Every coordinate at the int4 maximum against an all-ones code is the
        // largest value the u8-lane accumulators ever have to carry.
        let planes = build_query_planes(&vec![15u8; dim], dim, n_bytes);
        assert_eq!(
            dot_query_binary_planes(&planes, &vec![0xFFu8; n_bytes], n_bytes),
            15 * dim as u32
        );
        assert_eq!(
            dot_query_binary_planes(&planes, &vec![0u8; n_bytes], n_bytes),
            0
        );

        let zero_planes = build_query_planes(&vec![0u8; dim], dim, n_bytes);
        assert_eq!(
            dot_query_binary_planes(&zero_planes, &vec![0xFFu8; n_bytes], n_bytes),
            0
        );
    }

    #[test]
    fn test_hamming_block_matches_per_vector() {
        for dim in KERNEL_TEST_DIMS {
            let n_bytes = dim.div_ceil(8);
            let n = 70; // spans two full blocks plus a short tail

            let query = random_code(3, dim, n_bytes);
            let codes: Vec<u8> = (0..n)
                .flat_map(|i| random_code(100 + i as u64, dim, n_bytes))
                .collect();

            let mut out = vec![0u32; n];
            let block_min = hamming_block(&query, &codes, n_bytes, &mut out);

            let expected: Vec<u32> = (0..n)
                .map(|i| hamming_distance(&query, &codes[i * n_bytes..(i + 1) * n_bytes]))
                .collect();

            assert_eq!(out, expected, "block Hamming disagrees at dim {dim}");
            assert_eq!(block_min, *expected.iter().min().unwrap());
        }
    }

    #[test]
    fn test_hamming_block_partial_and_empty() {
        let dim = 128;
        let n_bytes = dim / 8;
        let query = random_code(5, dim, n_bytes);
        let codes = random_code(6, dim, n_bytes);

        let mut one = [0u32; 1];
        assert_eq!(
            hamming_block(&query, &codes, n_bytes, &mut one),
            hamming_distance(&query, &codes)
        );

        let mut none: [u32; 0] = [];
        assert_eq!(hamming_block(&query, &codes, n_bytes, &mut none), u32::MAX);
    }

    #[test]
    fn test_rabitq_block_matches_per_vector() {
        let data = create_test_data::<f32>(200, 64);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(4), 42).unwrap();

        let query: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();

        for c_idx in 0..quantiser.storage().nlist {
            let encoded = quantiser.encode_query(&query, c_idx).unwrap();
            let cluster_size = quantiser.storage().cluster_size(c_idx);

            let mut block = vec![0.0f32; cluster_size];
            let block_min = quantiser.rabitq_block_sq(&encoded, c_idx, 0, &mut block);

            for local_idx in 0..cluster_size {
                let scalar = quantiser.rabitq_dist_sq(&encoded, c_idx, local_idx);
                assert_abs_diff_eq!(block[local_idx], scalar, epsilon = 1e-6);
                // The blocked scan ranks on the square, the API returns the root
                assert_abs_diff_eq!(
                    scalar.sqrt(),
                    quantiser.rabitq_dist(&encoded, c_idx, local_idx),
                    epsilon = 1e-6
                );
            }

            if cluster_size > 0 {
                assert_abs_diff_eq!(
                    block_min,
                    block.iter().cloned().fold(f32::INFINITY, f32::min),
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn test_rabitq_block_offset_run() {
        let data = create_test_data::<f32>(120, 32);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::Cosine, Some(3), 7).unwrap();

        let query = vec![0.5f32; 32];
        let c_idx = 0;
        let encoded = quantiser.encode_query(&query, c_idx).unwrap();
        let cluster_size = quantiser.storage().cluster_size(c_idx);

        // A run that does not start at the cluster's first vector, which is
        // what every block after the first looks like in the real scan.
        if cluster_size >= 3 {
            let start = 2;
            let take = cluster_size - start;
            let mut block = vec![0.0f32; take];
            quantiser.rabitq_block_sq(&encoded, c_idx, start, &mut block);

            for j in 0..take {
                assert_abs_diff_eq!(
                    block[j],
                    quantiser.rabitq_dist_sq(&encoded, c_idx, start + j),
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn test_asymmetric_dot_matches_unpacked_reference() {
        for dim in [8usize, 15, 64, 128, 129] {
            let n_bytes = dim.div_ceil(8);
            let query: Vec<f64> = (0..dim)
                .map(|d| (splitmix_byte(21, d) as f64 / 128.0) - 1.0)
                .collect();
            let code = random_code(23, dim, n_bytes);

            // The definition the old implementation materialised: expand the
            // code to {-1, +1} and take a plain dot product.
            let expected: f64 = (0..dim)
                .map(|d| {
                    let bit = (code[d / 8] >> (d % 8)) & 1;
                    query[d] * (2.0 * bit as f64 - 1.0)
                })
                .sum();

            let got = asymmetric_binary_dot(&query, &code, dim);
            assert_abs_diff_eq!(got, expected, epsilon = 1e-9);

            let query_sum: f64 = query.iter().sum();
            assert_abs_diff_eq!(
                asymmetric_binary_dot_presummed(&query, query_sum, &code, dim),
                expected,
                epsilon = 1e-9
            );
        }
    }
}
