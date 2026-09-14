//! SIMD-optimised distance functions for the binary indices

#![allow(dead_code)]

use num_traits::{Float, FromPrimitive};

use crate::binary::rabitq::*;
#[cfg(test)]
use crate::binary::rabitq_fastscan::unpack_rabitq_blocked;
use crate::binary::rabitq_fastscan::{score_sign_block, SignScanQuery};
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

    /// Squared RaBitQ distances for one fast-scan block within a cluster
    ///
    /// Same estimate as [`rabitq_block_sq`](Self::rabitq_block_sq), sourcing
    /// the signed inner product from a nibble table scanned 32 lanes at a time
    /// instead of a per-vector AND-popcount. The int4 query quantisation and
    /// the popcount corrections it needed drop out with it, so only the
    /// per-vector norm and dot correction remain.
    ///
    /// `local_start` must be a multiple of [`RABITQ_BLOCK`], which is what the
    /// scan loops step by.
    ///
    /// ### Params
    ///
    /// * `query` - The query, already prepared against this cluster
    /// * `cluster_idx` - Index of the cluster
    /// * `local_start` - Local index of the first vector in the block
    /// * `out` - Per-vector output; its length sets how many lanes are kept
    ///
    /// ### Returns
    ///
    /// The minimum written into `out`, or infinity when `out` is empty
    #[inline]
    fn rabitq_block_sq_fastscan(
        &self,
        query: &SignScanQuery<T>,
        cluster_idx: usize,
        local_start: usize,
        out: &mut [T],
    ) -> T {
        debug_assert_eq!(local_start % RABITQ_BLOCK, 0);

        let storage = self.storage();
        let blocked = storage.cluster_blocked(cluster_idx);

        let mut lanes = [0.0f32; RABITQ_BLOCK];
        score_sign_block(
            &query.lut,
            &blocked.data,
            local_start / RABITQ_BLOCK,
            &mut lanes,
        );

        let one = T::one();
        let two = one + one;
        let q_dist = query.dist_to_centroid;
        let q_dist_sq = q_dist * q_dist;

        let global_start = storage.offsets[cluster_idx] + local_start;
        let mut min = T::infinity();

        for (j, slot) in out.iter_mut().enumerate() {
            let packed = unsafe { storage.packed_vectors.get_unchecked(global_start + j) };

            let sgn = T::from_f32(lanes[j]).unwrap();
            let q_dot_v = (sgn * packed.dot_correction_inv).clamp(one.neg(), one);

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

    /// The estimate computed longhand from the stored factors and the exact
    /// signed inner product, with no table quantisation anywhere.
    fn oracle_dist_sq(
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

        let c_rot = storage.centroid_rotated(cluster_idx);
        let res: Vec<f32> = q_rot.iter().zip(c_rot).map(|(a, b)| a - b).collect();
        let q_dist = res.iter().map(|x| x * x).sum::<f32>().sqrt();
        let unit: Vec<f32> = if q_dist > f32::EPSILON {
            res.iter().map(|x| x / q_dist).collect()
        } else {
            vec![0.0; res.len()]
        };

        let sgn: f32 = unit
            .iter()
            .enumerate()
            .map(|(d, &x)| {
                if code[d / 8] & (1 << (d % 8)) != 0 {
                    x
                } else {
                    -x
                }
            })
            .sum();

        let packed = storage.get_vector_data(cluster_idx, local_idx);
        let q_dot_v = (sgn * packed.dot_correction_inv).clamp(-1.0, 1.0);
        let v_dist = packed.dist_to_centroid;

        (v_dist * v_dist + q_dist * q_dist - 2.0 * v_dist * q_dist * q_dot_v).max(0.0)
    }

    #[test]
    fn test_rabitq_block_is_non_negative_and_deterministic() {
        for metric in [Dist::SquaredEuclidean, Dist::Cosine] {
            let data = create_test_data::<f32>(100, 32);
            let quantiser = RaBitQQuantiser::new(data.as_ref(), &metric, Some(10), 42).unwrap();
            let query = vec![1.0f32; 32];
            let q_rot = quantiser
                .encoder
                .apply_rotation(&quantiser.encoder.normalise_query(&query));

            for c_idx in 0..quantiser.storage().nlist {
                let encoded = quantiser.encode_query_prerotated(&q_rot, c_idx).unwrap();
                let size = quantiser.storage().cluster_size(c_idx);

                let mut a = vec![0.0f32; size.min(RABITQ_BLOCK)];
                let mut b = vec![0.0f32; size.min(RABITQ_BLOCK)];
                if a.is_empty() {
                    continue;
                }
                quantiser.rabitq_block_sq_fastscan(&encoded, c_idx, 0, &mut a);
                quantiser.rabitq_block_sq_fastscan(&encoded, c_idx, 0, &mut b);

                assert!(
                    a.iter().all(|&d| d >= 0.0),
                    "{metric:?} produced a negative"
                );
                assert_eq!(a, b, "{metric:?} is not deterministic");
            }
        }
    }

    #[test]
    fn test_rabitq_block_tracks_the_oracle() {
        // Clustered pseudo-random data, not the ramp: near-collinear vectors
        // drive the residual L1 norm towards zero, which sends
        // `dot_correction_inv` to infinity and leaves the estimate pinned at
        // the clamp, where it says nothing about the kernel.
        let (n, dim) = (200usize, 64usize);
        let data = faer::Mat::from_fn(n, dim, |i, j| {
            let centre = (splitmix_byte(1, (i % 4) * dim + j) as f32 / 128.0) - 1.0;
            centre * 4.0 + (splitmix_byte(2, i * dim + j) as f32 / 128.0) - 1.0
        });
        let quantiser =
            RaBitQQuantiser::new(data.as_ref(), &Dist::SquaredEuclidean, Some(4), 42).unwrap();

        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin()).collect();
        let q_rot = quantiser.encoder.apply_rotation(&query);

        for c_idx in 0..quantiser.storage().nlist {
            let encoded = quantiser.encode_query_prerotated(&q_rot, c_idx).unwrap();
            let size = quantiser.storage().cluster_size(c_idx);

            let mut local = 0;
            while local < size {
                let take = RABITQ_BLOCK.min(size - local);
                let mut block = vec![0.0f32; take];
                let block_min =
                    quantiser.rabitq_block_sq_fastscan(&encoded, c_idx, local, &mut block);

                for j in 0..take {
                    // The table rounds to u8 and that error is carried through
                    // `2 * v_dist * q_dist`, so the bound is relative: the
                    // estimate tracks the oracle, it does not reproduce it.
                    approx::assert_relative_eq!(
                        block[j],
                        oracle_dist_sq(&quantiser, &q_rot, c_idx, local + j),
                        max_relative = 0.01,
                        epsilon = 1e-4
                    );
                }

                assert_abs_diff_eq!(
                    block_min,
                    block.iter().cloned().fold(f32::INFINITY, f32::min),
                    epsilon = 1e-6
                );
                local += take;
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
