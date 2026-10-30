//! SIMD-optimised distance functions for the binary indices

#![allow(dead_code)]

use num_traits::{Float, FromPrimitive};
use std::collections::BinaryHeap;

use crate::binary::rabitq::ex_bits::{ex_inner_product, excode_bytes};
#[cfg(test)]
use crate::binary::rabitq::fastscan::unpack_rabitq_blocked;
use crate::binary::rabitq::fastscan::{blocked_sign_ip, score_sign_block, SignScanQuery};
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

/// Vectors scored per call to
/// [`VectorDistanceRaBitQ::rabitq_block_sq_fastscan`]. Same role as
/// [`HAMMING_BLOCK`], but the output is `T` rather than `u32`.
pub const RABITQ_BLOCK: usize = 32;

/// Code length at or below which the scalar `u64` path is used on x86_64.
///
/// Only taken when the CPU actually has `POPCNT`, which is *not* in the
/// x86_64 baseline (`rustc --print cfg` gives fxsr, sse, sse2 and nothing
/// else). Without it `u64::count_ones` lowers to a SWAR sequence that loses to
/// the SSE2 kernel, which is why the dispatch gates on runtime detection rather
/// than length alone.
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

/// Cached scalar `POPCNT` availability.
#[cfg(target_arch = "x86_64")]
static HAS_POPCNT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Whether the CPU has the scalar `POPCNT` instruction
///
/// ### Returns
///
/// `true` when `u64::count_ones` lowers to a single instruction
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_popcnt() -> bool {
    *HAS_POPCNT.get_or_init(|| is_x86_feature_detected!("popcnt"))
}

/// [`hamming_u64`] compiled with `POPCNT` enabled
///
/// The generic body cannot carry `#[target_feature]` because it is also the
/// portable fallback on targets that have no such feature, so the x86 fast path
/// goes through this wrapper and inlines the body under the enabled feature.
///
/// ### Params
///
/// * `a` - Slice of u8 to use
/// * `b` - Slice of u8 to use, same length as `a`
///
/// ### Returns
///
/// The Hamming distance between the two slices
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn hamming_u64_popcnt(a: &[u8], b: &[u8]) -> u32 {
    hamming_u64(a, b)
}

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
        if a.len() <= SCALAR_POPCNT_MAX_BYTES && has_popcnt() {
            return hamming_u64_popcnt(a, b);
        }
        if has_vpopcntdq() {
            return hamming_avx512_vpopcnt(a, b);
        }
        match detect_simd_level() {
            SimdLevel::Avx512 => hamming_avx512(a, b),
            SimdLevel::Avx2 => hamming_avx2(a, b),
            SimdLevel::Sse => hamming_sse2(a, b),
            SimdLevel::Scalar => hamming_u64(a, b),
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
    // Hard asserts, not debug: this is a safe `pub fn` that indexes unchecked
    assert_eq!(query.len(), n_bytes);
    assert!(codes.len() >= out.len() * n_bytes);

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
/// Private: it indexes `query_vec` and `binary_code` unchecked up to `dim`, and
/// the length check that makes that sound lives in
/// [`asymmetric_binary_dot_presummed`].
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
fn masked_query_sum<T>(query_vec: &[T], binary_code: &[u8], dim: usize) -> T
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
            a0 = if (bits >> bit_pos) & 1 == 1 {
                a0 + q
            } else {
                a0
            };
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

/// RaBitQ distance estimation over a clustered code store.
///
/// Implemented by the IVF and exhaustive RaBitQ indices, which differ in how
/// they pick clusters but share the estimator. Scoring goes one fast-scan block
/// at a time: the query becomes a nibble table once per cluster, and each block
/// of 32 codes is scanned with a single byte-shuffle per byte-group.
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

    /// Get the dimensionality of the rotated frame
    ///
    /// This is the coordinate count the codes and the quantised query actually
    /// span, which is what the estimator's counting terms need. It only differs
    /// from [`dim`](Self::dim) when the encoder's rotation pads.
    ///
    /// ### Returns
    ///
    /// Number of coordinates after rotation
    #[inline]
    fn padded_dim(&self) -> usize {
        self.storage().padded_dim
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

    /// Squared RaBitQ estimates and lower bounds for one fast-scan block
    ///
    /// The signed inner product comes from a nibble table scanned 32 lanes at
    /// a time rather than a per-vector AND-popcount, so there is no int4 query
    /// quantisation and none of the popcount corrections it would need. See
    /// [`RaBitQPackedVector`] for the estimate and the bound.
    ///
    /// `local_start` must be a multiple of [`RABITQ_BLOCK`], which is what the
    /// scan loops step by.
    ///
    /// ### Params
    ///
    /// * `query` - The prepared query
    /// * `cluster_idx` - Index of the cluster
    /// * `local_start` - Local index of the first vector in the block
    /// * `g_add` - `||q - c||^2` for this cluster
    /// * `g_error` - `||q - c||` for this cluster
    /// * `est` - Per-vector estimates; its length sets how many lanes are kept
    /// * `low` - Per-vector lower bounds, same length as `est`
    ///
    /// ### Returns
    ///
    /// The minimum of `est` and the minimum of `low`, infinity when empty
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn rabitq_block_sq_fastscan(
        &self,
        query: &SignScanQuery<T>,
        cluster_idx: usize,
        local_start: usize,
        g_add: T,
        g_error: T,
        est: &mut [T],
        low: &mut [T],
    ) -> (T, T) {
        debug_assert_eq!(local_start % RABITQ_BLOCK, 0);
        debug_assert_eq!(est.len(), low.len());

        let storage = self.storage();
        let blocked = storage.cluster_blocked(cluster_idx);

        let mut lanes = [0.0f32; RABITQ_BLOCK];
        score_sign_block(
            &query.lut,
            &blocked.data,
            local_start / RABITQ_BLOCK,
            &mut lanes,
        );

        let global_start = storage.offsets[cluster_idx] as usize + local_start;
        let packed = &storage.packed_vectors[global_start..global_start + est.len()];
        let (mut min_est, mut min_low) = (T::infinity(), T::infinity());

        for (j, p) in packed.iter().enumerate() {
            let e = (p.f_add + g_add + p.f_rescale * T::from_f32(lanes[j]).unwrap()).max(T::zero());
            let l = e - p.f_error * g_error;
            est[j] = e;
            low[j] = l;
            min_est = min_est.min(e);
            min_low = min_low.min(l);
        }

        (min_est, min_low)
    }

    /// Full-width estimate for one vector, reusing its one-bit inner product
    ///
    /// The reference's `split_distance_boosting`, with one change: the sign
    /// product is recomputed exactly rather than reused from the fast-scan
    /// lane. It is scaled by `2^ex_bits` here, so the byte-rounded lane caps
    /// recall once the magnitude code is finer than the table (the reference
    /// switches to a 16-bit table for the same reason).
    ///
    /// ```text
    /// est = f_add + g_add + f_rescale * (2^ex * <q, s01> + <q, ex> + cb * sum(q))
    /// ```
    ///
    /// with `cb = -(2^ex - 1/2)`.
    ///
    /// ### Params
    ///
    /// * `query` - The prepared query, with its sign sums
    /// * `cluster_idx` - The vector's cluster
    /// * `local` - The vector's index within the cluster
    /// * `g_add` - `||q - c||^2` for its cluster
    ///
    /// ### Returns
    ///
    /// The estimated squared distance
    #[inline]
    fn rabitq_estimate_ex(
        &self,
        query: &SignScanQuery<T>,
        cluster_idx: usize,
        local: usize,
        g_add: T,
    ) -> T
    where
        T: AnnSearchFloat,
    {
        let storage = self.storage();
        let pos = storage.offsets[cluster_idx] as usize + local;
        let w = storage.ex_bits;
        let ex_bytes = excode_bytes(storage.padded_dim, w);
        let ip_ex = ex_inner_product(
            &query.rotated,
            &storage.ex_codes[pos * ex_bytes..(pos + 1) * ex_bytes],
            w,
        );

        let half = T::from_f64(0.5).unwrap();
        let step = T::from_u64(1u64 << w).unwrap();
        let ip_s01 = blocked_sign_ip(
            &query.sign_sums,
            &storage.cluster_blocked(cluster_idx).data,
            storage.n_bytes,
            local,
        );
        let cb = half - step;
        let (f_add, f_rescale) = (storage.ex_factors[2 * pos], storage.ex_factors[2 * pos + 1]);

        (f_add + g_add + f_rescale * (step * ip_s01 + ip_ex + cb * query.sum)).max(T::zero())
    }

    /// Scan the probed clusters and keep the `k` best
    ///
    /// The reference's one-stage search. Every lane gets a one-bit estimate
    /// and a lower bound. What happens next depends on `refine`:
    ///
    /// - [`Refine::Estimate`]: the heap ranks on the one-bit estimates.
    /// - [`Refine::ExBits`]: a lane whose bound beats the current `k`-th
    ///   distance is re-scored on the full-width code.
    /// - [`Refine::Exact`]: the same gate, but the lane gets an exact distance.
    ///
    /// The bound is probabilistic, so a gated scan can miss a neighbour a fixed
    /// shortlist would have caught, but it pays for refinement only where the
    /// one-bit estimate is ambiguous.
    ///
    /// ### Params
    ///
    /// * `query` - The prepared query
    /// * `probed` - Clusters to scan
    /// * `k` - Number of neighbours to keep
    /// * `refine` - What a lane that passes the bound gets
    ///
    /// ### Returns
    ///
    /// `(squared distance, original index)` ascending, at most `k` long
    fn rabitq_scan<F>(
        &self,
        query: &SignScanQuery<T>,
        probed: &[usize],
        k: usize,
        mut refine: Refine<F>,
    ) -> Vec<(OrderedFloat<T>, usize)>
    where
        T: AnnSearchFloat,
        F: FnMut(usize) -> T,
    {
        let storage = self.storage();
        debug_assert!(!matches!(refine, Refine::ExBits) || storage.ex_bits > 0);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);
        let mut est = [T::zero(); RABITQ_BLOCK];
        let mut low = [T::zero(); RABITQ_BLOCK];

        let top = |heap: &BinaryHeap<(OrderedFloat<T>, usize)>| {
            if heap.len() < k {
                T::infinity()
            } else {
                heap.peek().map_or(T::infinity(), |e| e.0 .0)
            }
        };

        for &c_idx in probed {
            let g_add = T::euclidean_simd(&query.rotated, storage.centroid_rotated(c_idx));
            let g_error = g_add.sqrt();
            let cluster_size = storage.cluster_size(c_idx);
            let indices = storage.cluster_vector_indices(c_idx);

            let mut local_idx = 0;
            while local_idx < cluster_size {
                let take = RABITQ_BLOCK.min(cluster_size - local_idx);
                let (min_est, min_low) = self.rabitq_block_sq_fastscan(
                    query,
                    c_idx,
                    local_idx,
                    g_add,
                    g_error,
                    &mut est[..take],
                    &mut low[..take],
                );

                let gate = if matches!(refine, Refine::Estimate) {
                    min_est
                } else {
                    min_low
                };
                if gate < top(&heap) {
                    for j in 0..take {
                        let idx = indices[local_idx + j] as usize;
                        let d = match &mut refine {
                            Refine::Estimate => est[j],
                            _ if low[j] >= top(&heap) => continue,
                            Refine::ExBits => {
                                self.rabitq_estimate_ex(query, c_idx, local_idx + j, g_add)
                            }
                            Refine::Exact(exact) => exact(idx),
                        };
                        if d < top(&heap) {
                            if heap.len() == k {
                                heap.pop();
                            }
                            heap.push((OrderedFloat(d), idx));
                        }
                    }
                }

                local_idx += take;
            }
        }

        heap.into_sorted_vec()
    }
}

/// What [`VectorDistanceRaBitQ::rabitq_scan`] does with a candidate whose
/// lower bound beats the current `k`-th distance.
pub enum Refine<F> {
    /// Nothing: rank every lane on its one-bit estimate
    Estimate,
    /// Re-score on the stored full-width code, see
    /// [`VectorDistanceRaBitQ::rabitq_estimate_ex`]
    ExBits,
    /// Exact squared distance (in the estimate's space) by original index
    Exact(F),
}

impl<T> Refine<fn(usize) -> T> {
    /// The refinement a store-free query gets
    ///
    /// ### Params
    ///
    /// * `ex_bits` - Magnitude bits the index was built with
    ///
    /// ### Returns
    ///
    /// [`Refine::ExBits`] when there is a wider code, [`Refine::Estimate`]
    /// otherwise
    pub fn without_store(ex_bits: usize) -> Self {
        if ex_bits > 0 {
            Refine::ExBits
        } else {
            Refine::Estimate
        }
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
    use crate::binary::rabitq::RaBitQQuantiser;
    use crate::utils::dist::Dist;
    use approx::assert_abs_diff_eq;
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
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(5), 0, 42).unwrap();

        assert_eq!(quantiser.dim(), 32);
    }

    #[test]
    fn test_rabitq_trait_n_bytes() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(5), 0, 42).unwrap();

        assert_eq!(quantiser.n_bytes(), 4);
    }

    /// The signed inner product `<sign, q_rot>` of one stored code, exact.
    fn oracle_sign_ip(
        quantiser: &RaBitQQuantiser<f32>,
        q_rot: &[f32],
        cluster_idx: usize,
        local_idx: usize,
    ) -> f32 {
        let storage = quantiser.storage();
        let n_bytes = storage.n_bytes;
        let size = storage.cluster_size(cluster_idx);

        let codes = unpack_rabitq_blocked(
            &storage.cluster_blocked(cluster_idx).data,
            size,
            n_bytes,
            storage.blocked_arch,
        );
        let code = &codes[local_idx * n_bytes..(local_idx + 1) * n_bytes];

        q_rot
            .iter()
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

    /// The estimate computed longhand from the stored factors and the exact
    /// signed inner product, with no table quantisation anywhere.
    fn oracle_dist_sq(
        quantiser: &RaBitQQuantiser<f32>,
        q_rot: &[f32],
        cluster_idx: usize,
        local_idx: usize,
    ) -> f32 {
        let storage = quantiser.storage();
        let c_rot = storage.centroid_rotated(cluster_idx);
        let g_add: f32 = q_rot
            .iter()
            .zip(c_rot)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let p = storage.get_vector_data(cluster_idx, local_idx);

        (p.f_add + g_add + p.f_rescale * oracle_sign_ip(quantiser, q_rot, cluster_idx, local_idx))
            .max(0.0)
    }

    /// Clustered pseudo-random data, not a ramp: near-collinear vectors drive
    /// the residual L1 norm towards zero, which leaves the estimate saying
    /// nothing about the kernel.
    fn clustered_data(n: usize, dim: usize) -> Mat<f32> {
        faer::Mat::from_fn(n, dim, |i, j| {
            let centre = (splitmix_byte(1, (i % 4) * dim + j) as f32 / 128.0) - 1.0;
            centre * 4.0 + (splitmix_byte(2, i * dim + j) as f32 / 128.0) - 1.0
        })
    }

    /// Scores every vector of one cluster through the block kernel.
    fn score_cluster(
        quantiser: &RaBitQQuantiser<f32>,
        query: &SignScanQuery<f32>,
        c_idx: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let storage = quantiser.storage();
        let size = storage.cluster_size(c_idx);
        let g_add: f32 = query
            .rotated
            .iter()
            .zip(storage.centroid_rotated(c_idx))
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        let (mut est, mut low) = (vec![0.0f32; size], vec![0.0f32; size]);
        let mut local = 0;
        while local < size {
            let take = RABITQ_BLOCK.min(size - local);
            let (min_est, min_low) = quantiser.rabitq_block_sq_fastscan(
                query,
                c_idx,
                local,
                g_add,
                g_add.sqrt(),
                &mut est[local..local + take],
                &mut low[local..local + take],
            );
            let fold = |v: &[f32]| v.iter().cloned().fold(f32::INFINITY, f32::min);
            assert_eq!(min_est, fold(&est[local..local + take]));
            assert_eq!(min_low, fold(&low[local..local + take]));
            local += take;
        }
        (est, low)
    }

    #[test]
    fn test_rabitq_block_is_non_negative_and_deterministic() {
        for metric in [Dist::SquaredEuclidean, Dist::Cosine] {
            let data = create_test_data::<f32>(100, 32);
            let quantiser = RaBitQQuantiser::new(data.as_ref(), &metric, Some(10), 0, 42).unwrap();
            let query = quantiser
                .encoder
                .encode_query(&quantiser.encoder.normalise_query(&[1.0f32; 32]), 0)
                .unwrap();

            for c_idx in 0..quantiser.storage().nlist {
                let (a, low) = score_cluster(&quantiser, &query, c_idx);
                let (b, _) = score_cluster(&quantiser, &query, c_idx);

                assert!(
                    a.iter().all(|&d| d >= 0.0),
                    "{metric:?} produced a negative"
                );
                assert!(a.iter().zip(&low).all(|(e, l)| l <= e));
                assert_eq!(a, b, "{metric:?} is not deterministic");
            }
        }
    }

    #[test]
    fn test_rabitq_block_tracks_the_oracle() {
        let (n, dim) = (200usize, 64usize);
        let data = clustered_data(n, dim);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(4), 0, 42).unwrap();

        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin()).collect();
        let prepared = quantiser.encoder.encode_query(&query, 0).unwrap();

        for c_idx in 0..quantiser.storage().nlist {
            let (est, _) = score_cluster(&quantiser, &prepared, c_idx);
            for (j, &e) in est.iter().enumerate() {
                // The table rounds to u8 and that error is carried through
                // `f_rescale`, so the bound is relative: the estimate tracks
                // the oracle, it does not reproduce it.
                approx::assert_relative_eq!(
                    e,
                    oracle_dist_sq(&quantiser, &prepared.rotated, c_idx, j),
                    max_relative = 0.01,
                    epsilon = 1e-3
                );
            }
        }
    }

    #[test]
    fn test_factors_reproduce_the_original_estimator() {
        // `f_add + g_add + f_rescale <s, q>` is the original
        // `||r||^2 + ||q - c||^2 - 2 ||r|| dci <s, q - c>` with the centroid
        // term moved into `f_add`. Check the algebra on raw encodings.
        let dim = 64;
        let encoder = RaBitQEncoder::<f32>::new(dim, Dist::SquaredEuclidean, 7);
        let data = clustered_data(20, dim);
        let centroid: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.11).cos()).collect();
        let c_rot = encoder.apply_rotation(&centroid);
        let query: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.29).sin() * 3.0).collect();
        let q_rot = encoder.apply_rotation(&query);

        for i in 0..20 {
            let v: Vec<f32> = data.row(i).iter().cloned().collect();
            let (code, dist, dci) = encoder.encode_vector(&v, &centroid).unwrap();
            let p = RaBitQPackedVector::from_code(&code, dist, dci, &c_rot);

            let sign = |x: &[f32]| -> f64 {
                x.iter()
                    .enumerate()
                    .map(|(d, &v)| {
                        if code[d / 8] >> (d % 8) & 1 == 1 {
                            v as f64
                        } else {
                            -(v as f64)
                        }
                    })
                    .sum()
            };
            let res: Vec<f32> = q_rot.iter().zip(&c_rot).map(|(a, b)| a - b).collect();
            let g_add: f64 = res.iter().map(|&x| (x as f64) * (x as f64)).sum();
            let (dist, dci) = (dist as f64, dci as f64);

            let original = dist * dist + g_add - 2.0 * dist * dci * sign(&res);
            let factored = p.f_add as f64 + g_add + p.f_rescale as f64 * sign(&q_rot);
            approx::assert_relative_eq!(factored, original, max_relative = 1e-4);
        }
    }

    #[test]
    fn test_lower_bound_holds_for_most_vectors() {
        // The bound is probabilistic: the reference's 1.9 epsilon is about a
        // 1.9 sigma one-sided interval, so roughly 3% of vectors fall below
        // it. A halved `f_error` would drop that to ~83%, a sign error
        // further still.
        let (n, dim) = (400usize, 128usize);
        let data = clustered_data(n, dim);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(4), 0, 42).unwrap();
        let storage = quantiser.storage();

        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() * 4.0).collect();
        let prepared = quantiser.encoder.encode_query(&query, 0).unwrap();

        let (mut held, mut total) = (0usize, 0usize);
        for c_idx in 0..storage.nlist {
            let (_, low) = score_cluster(&quantiser, &prepared, c_idx);
            for (j, &l) in low.iter().enumerate() {
                let idx = storage.cluster_vector_indices(c_idx)[j] as usize;
                let truth: f32 = data
                    .row(idx)
                    .iter()
                    .zip(&query)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                held += (l <= truth) as usize;
                total += 1;
            }
        }
        assert!(held as f64 >= 0.95 * total as f64, "{held} / {total}");
    }

    #[test]
    fn test_split_ex_estimate_matches_the_full_code_estimate() {
        // `rabitq_estimate_ex` reads the sign half out of the blocked layout
        // and the magnitude half from `ex_codes`; reassembled into one
        // `ExEncoding`, the reference-validated `estimate_ex` must agree.
        use crate::binary::rabitq::ex_bits::{estimate_ex, excode_bytes, ExEncoding};

        let (n, dim, ex_bits) = (150usize, 64usize, 4usize);
        let data = clustered_data(n, dim);
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(3), ex_bits, 42)
                .unwrap();
        let storage = quantiser.storage();
        let ex_bytes = excode_bytes(storage.padded_dim, ex_bits);

        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).cos() * 2.0).collect();
        let prepared = quantiser.encoder.encode_query(&query, ex_bits).unwrap();

        for c_idx in 0..storage.nlist {
            let c_rot = storage.centroid_rotated(c_idx);
            let g_add: f32 = prepared
                .rotated
                .iter()
                .zip(c_rot)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let size = storage.cluster_size(c_idx);
            let codes = unpack_rabitq_blocked(
                &storage.cluster_blocked(c_idx).data,
                size,
                storage.n_bytes,
                storage.blocked_arch,
            );

            for local in 0..size {
                let pos = storage.offsets[c_idx] as usize + local;
                let encoding = ExEncoding {
                    sign_code: codes[local * storage.n_bytes..(local + 1) * storage.n_bytes]
                        .to_vec(),
                    ex_code: storage.ex_codes[pos * ex_bytes..(pos + 1) * ex_bytes].to_vec(),
                    f_add: storage.ex_factors[2 * pos],
                    f_rescale: storage.ex_factors[2 * pos + 1],
                    f_error: 0.0,
                };
                approx::assert_relative_eq!(
                    quantiser.rabitq_estimate_ex(&prepared, c_idx, local, g_add),
                    estimate_ex(&encoding, &prepared.rotated, g_add, ex_bits).max(0.0),
                    max_relative = 1e-4,
                    epsilon = 1e-3
                );
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
