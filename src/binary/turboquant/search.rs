//! SIMD-accelerated scoring for TurboQuant.
//!
//! Scoring uses the FAISS PQ4 fast-scan trick: per query, a byte-indexed LUT
//! is built so that a single SIMD byte-shuffle scores many vectors at once. The
//! LUT holds, for each byte-group of the packed codes and each possible nibble
//! value, the partial inner product `sum_c q_rot[coord_c] * level[code_c]`,
//! quantised to u8.

use std::cmp::Ordering;

use crate::binary::turboquant::pack::BLOCK;
#[cfg(target_arch = "x86_64")]
use crate::binary::turboquant::pack::PERM0_INV;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Byte-groups accumulated into u16 lanes before each flush to f32. Tuned so
/// `FLUSH_EVERY * 2 * 127 < u16::MAX`: each group adds two u8 lookups (each ≤
/// 127 on NEON), so the u16 lane cannot overflow within a batch.
#[cfg(target_arch = "aarch64")]
const FLUSH_EVERY: usize = 256;

/////////
// LUT //
/////////

/// Per-query lookup table for fast-scan scoring.
///
/// `luts_u8` is laid out as `n_byte_groups` blocks of 32 bytes. Within a block,
/// bytes `[0, 16)` are the sub-table indexed by the *high* nibble of a code
/// byte, and bytes `[16, 32)` are indexed by the *low* nibble. The split is
/// dictated by the packing in `tq_pack`: for 4-bit codes the high nibble of a
/// byte holds the lower-indexed dimension, so the high-nibble sub-table is
/// built from `q_rot[dim_start + 0]`.
///
/// A score is recovered as `bias + scale * acc`, where `acc` is the u8 sum
/// of the looked-up entries over all byte-groups.
pub struct QueryLut {
    /// `n_byte_groups * 32` u8 entries.
    pub luts_u8: Vec<u8>,
    /// Shared dequantisation scale.
    pub scale: f32,
    /// Total decode bias (sum of per-sub-table minima), added once.
    pub bias: f32,
    /// Number of 32-byte groups (`dim / (8 / bits)`).
    pub n_byte_groups: usize,
}

/// Build a per-query LUT from rotated query coordinates and the levels.
///
/// Uses FAISS-style per-sub-table quantisation: each 16-entry nibble sub-table
/// subtracts its own minimum before u8 rounding, with one shared `scale`.
///
/// ### Params
///
/// * `q_rot_f32` - Rotated, unit-normalised query (length `dim`), f32
/// * `levels_f32` - Lloyd-Max levels (length `2^bits`), f32
/// * `bits` - Bits per coordinate (2 or 4; 3-bit has no LUT path)
/// * `dim` - Dimensionality
///
/// ### Returns
///
/// The query LUT.
pub fn build_query_lut(
    q_rot_f32: &[f32],
    levels_f32: &[f32],
    bits: usize,
    dim: usize,
) -> Result<QueryLut, AnnSearchErrors> {
    if bits != 2 && bits != 4 {
        return Err(AnnSearchErrors::TQLutError { bit: bits });
    }

    let codes_per_byte = 8 / bits;
    let codes_per_nibble = codes_per_byte / 2;
    let n_byte_groups = dim / codes_per_byte;
    let code_mask = (1u16 << bits) - 1;
    let n_subs = n_byte_groups * 2;

    let mut luts_u8 = vec![0u8; n_byte_groups * 32];
    let mut float_vals = vec![0.0f32; n_byte_groups * 32];
    let mut mins = vec![0.0f32; n_subs];
    let mut max_span = 0.0f32;
    let mut bias = 0.0f32;

    for g in 0..n_byte_groups {
        let dim_start = g * codes_per_byte;

        // high-nibble sub-table (stored at [0, 16)): covers the first
        // `codes_per_nibble` dims of this group.
        let mut lo_min = f32::MAX;
        let mut lo_max = f32::MIN;
        for nibble_val in 0u16..16 {
            let mut s = 0.0f32;
            for c in 0..codes_per_nibble {
                let shift = (codes_per_nibble - 1 - c) * bits;
                let code = (nibble_val >> shift) & code_mask;
                s += q_rot_f32[dim_start + c] * levels_f32[code as usize];
            }
            float_vals[g * 32 + nibble_val as usize] = s;
            if s < lo_min {
                lo_min = s;
            }
            if s > lo_max {
                lo_max = s;
            }
        }

        // Low-nibble sub-table (stored at [16, 32)): covers the second
        // `codes_per_nibble` dims of this group.
        let mut hi_min = f32::MAX;
        let mut hi_max = f32::MIN;
        for nibble_val in 0u16..16 {
            let mut s = 0.0f32;
            for c in 0..codes_per_nibble {
                let shift = (codes_per_nibble - 1 - c) * bits;
                let code = (nibble_val >> shift) & code_mask;
                s += q_rot_f32[dim_start + codes_per_nibble + c] * levels_f32[code as usize];
            }
            float_vals[g * 32 + 16 + nibble_val as usize] = s;
            if s < hi_min {
                hi_min = s;
            }
            if s > hi_max {
                hi_max = s;
            }
        }

        mins[g * 2] = lo_min;
        mins[g * 2 + 1] = hi_min;
        bias += lo_min + hi_min;

        let lo_span = lo_max - lo_min;
        let hi_span = hi_max - hi_min;
        if lo_span > max_span {
            max_span = lo_span;
        }
        if hi_span > max_span {
            max_span = hi_span;
        }
    }

    // x86 accumulates u8 lookups into u16 lanes, so cap per-entry value to keep
    // `2 * n_byte_groups` additions from overflowing. NEON flushes to f32
    // frequently enough to use the full u8 range.
    #[cfg(target_arch = "x86_64")]
    let max_lut = (65535.0 / (n_byte_groups as f64 * 2.0)).floor().min(127.0) as f32;
    #[cfg(not(target_arch = "x86_64"))]
    let max_lut = 127.0f32;

    let scale = if max_span > 1e-10 {
        max_span / max_lut
    } else {
        1.0
    };
    let inv_scale = 1.0 / scale;

    for g in 0..n_byte_groups {
        let lo_min = mins[g * 2];
        let hi_min = mins[g * 2 + 1];
        for i in 0..16 {
            let j_lo = g * 32 + i;
            let j_hi = g * 32 + 16 + i;
            luts_u8[j_lo] = ((float_vals[j_lo] - lo_min) * inv_scale)
                .round()
                .clamp(0.0, max_lut) as u8;
            luts_u8[j_hi] = ((float_vals[j_hi] - hi_min) * inv_scale)
                .round()
                .clamp(0.0, max_lut) as u8;
        }
    }

    Ok(QueryLut {
        luts_u8,
        scale,
        bias,
        n_byte_groups,
    })
}

/// Scalar reference: apply a [`QueryLut`] to one vector's bit-plane codes.
///
/// Reconstructs each byte-group's packed nibble byte exactly as `tq_pack` does,
/// then sums the two nibble-indexed LUT entries per group. This is the oracle
/// the SIMD kernels are validated against; it deliberately mirrors their
/// arithmetic (u8 LUT, integer accumulate, single final `bias + scale * acc`)
/// rather than the float math in `tq_dists::score_ip_scalar`.
///
/// ### Params
///
/// * `lut` - Query LUT
/// * `packed` - Bit-plane codes for one vector (`bits * dim / 8` bytes)
/// * `bits` - Bits per coordinate (2 or 4)
/// * `dim` - Dimensionality
///
/// ### Returns
///
/// Approximate inner product (same quantity as `score_ip_scalar`, up to
/// u8 rounding).
pub fn score_via_lut_bitplane(lut: &QueryLut, packed: &[u8], bits: usize, dim: usize) -> f32 {
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let bytes_per_plane = dim / 8;

    let mut acc: u32 = 0;
    for g in 0..n_byte_groups {
        let dim_start = g * codes_per_byte;
        let mut byte_val = 0u8;
        for c in 0..codes_per_byte {
            let j = dim_start + c;
            let byte_in_plane = j / 8;
            let mask = 1u8 << (7 - (j % 8));
            let mut code = 0u8;
            for p in 0..bits {
                if packed[p * bytes_per_plane + byte_in_plane] & mask != 0 {
                    code |= 1 << p;
                }
            }
            let shift = (codes_per_byte - 1 - c) * bits;
            byte_val |= code << shift;
        }
        let lo = (byte_val & 0x0F) as usize;
        let hi = (byte_val >> 4) as usize;
        // Low nibble -> [16, 32) sub-table; high nibble -> [0, 16).
        acc += lut.luts_u8[g * 32 + 16 + lo] as u32;
        acc += lut.luts_u8[g * 32 + hi] as u32;
    }

    lut.scale.mul_add(acc as f32, lut.bias)
}

//////////////////////////////////
// Scalar blocked-layout scorer //
//////////////////////////////////

/// Recover the full code byte for one lane of a blocked byte-group.
///
/// On x86_64 the group is perm0-interleaved: byte `j` (`0..16`) packs the high
/// nibbles of vectors `PERM0[j]` and `PERM0[j] + 16`, byte `16 + j` their low
/// nibbles. On other targets the layout is sequential and lane `i` is simply
/// `group[i]`, see function below.
///
/// ### Params
///
/// * `group` - The `BLOCK`-byte slice for one (block, byte-group)
/// * `lane` - Lane index within the block (`0..BLOCK`)
///
/// ### Returns
///
/// The reconstructed packed code byte for that lane.
#[cfg(target_arch = "x86_64")]
#[inline]
fn blocked_code_byte(group: &[u8], lane: usize) -> u8 {
    if lane < 16 {
        let j = PERM0_INV[lane];
        let hi = group[j] & 0x0F;
        let lo = group[16 + j] & 0x0F;
        (hi << 4) | lo
    } else {
        let j = PERM0_INV[lane - 16];
        let hi = group[j] >> 4;
        let lo = group[16 + j] >> 4;
        (hi << 4) | lo
    }
}
/// Recover the full code byte for one lane of a blocked byte-group.
///
/// ### Params
///
/// * `group` - The `BLOCK`-byte slice for one (block, byte-group)
/// * `lane` - Lane index within the block (`0..BLOCK`)
///
/// ### Returns
///
/// The reconstructed packed code byte for that lane.
#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn blocked_code_byte(group: &[u8], lane: usize) -> u8 {
    group[lane]
}

/// Score one block of up to `BLOCK` vectors against a query LUT, reading from
/// the blocked layout.
///
/// This is the scalar reference for the SIMD kernels: it traverses the same
/// blocked code layout they do and accumulates the same u8 LUT lookups, so its
/// per-lane output `bias + scale * acc` is bit-identical to
/// [`score_via_lut_bitplane`] on the corresponding vector. The SIMD kernels are
/// validated against this.
///
/// Padding lanes (where `block_idx * BLOCK + lane >= n_vectors`) decode from
/// zeroed bytes and are left for the caller to mask.
///
/// ### Params
///
/// * `lut` - Query LUT (2-bit or 4-bit)
/// * `blocked` - The blocked code layout (`BlockedCodes::data`)
/// * `block_idx` - Which block to score
/// * `out` - Per-lane inner-product output (length `BLOCK`)
pub fn score_block_scalar(
    lut: &QueryLut,
    blocked: &[u8],
    block_idx: usize,
    out: &mut [f32; BLOCK],
) {
    let n_byte_groups = lut.n_byte_groups;
    let mut acc = [0u32; BLOCK];

    for g in 0..n_byte_groups {
        let group_off = (block_idx * n_byte_groups + g) * BLOCK;
        let group = &blocked[group_off..group_off + BLOCK];
        let lut_g = &lut.luts_u8[g * 32..g * 32 + 32];
        for lane in 0..BLOCK {
            let code = blocked_code_byte(group, lane);
            let lo = (code & 0x0F) as usize;
            let hi = (code >> 4) as usize;
            // Low nibble -> [16, 32) sub-table, high nibble -> [0, 16),
            // matching build_query_lut and score_via_lut_bitplane.
            acc[lane] += lut_g[16 + lo] as u32 + lut_g[hi] as u32;
        }
    }

    for lane in 0..BLOCK {
        out[lane] = lut.scale.mul_add(acc[lane] as f32, lut.bias);
    }
}

/// Score one block of `BLOCK` vectors for a single query (NEON).
///
/// Outputs the raw inner product `bias + scale * acc` per lane into `out`.
/// Norm application and padding-lane masking are the caller's
/// responsibility — every lane (including padding) gets a defined score.
/// Bit-width-agnostic: driven entirely by `lut.n_byte_groups` and the
/// blocked code layout, so it serves both 2-bit and 4-bit.
///
/// ### Safety
///
/// `blocked` must hold at least `(block_idx + 1) * n_byte_groups * BLOCK`
/// bytes and `lut.luts_u8` at least `n_byte_groups * 32`.
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn score_block_neon(
    lut: &QueryLut,
    blocked: &[u8],
    block_idx: usize,
    out: &mut [f32; BLOCK],
) {
    // better to import here to avoid confusion...
    use std::arch::aarch64::*;

    let n_byte_groups = lut.n_byte_groups;
    let luts_base = lut.luts_u8.as_ptr();
    let codes_base = blocked.as_ptr().add(block_idx * n_byte_groups * BLOCK);

    let mask = vdupq_n_u8(0x0F);
    let v_scale = vdupq_n_f32(lut.scale);
    let n_batches = n_byte_groups.div_ceil(FLUSH_EVERY);

    // fa[i] holds lanes [i*4, i*4+4), seeded with the decode bias. Flushes
    // add scale * acc on top; bias is therefore applied exactly once.
    let mut fa = [vdupq_n_f32(lut.bias); 8];

    for batch in 0..n_batches {
        let g_start = batch * FLUSH_EVERY;
        let g_end = (g_start + FLUSH_EVERY).min(n_byte_groups);

        let mut accum = [vdupq_n_u16(0); 4];

        // 4-group unroll to hide vqtbl1q latency.
        let mut g = g_start;
        while g + 3 < g_end {
            for gg in 0..4 {
                let gi = g + gg;
                let lp = luts_base.add(gi * 32);
                let lut_hi = vld1q_u8(lp); // [0,16): high-nibble sub-table
                let lut_lo = vld1q_u8(lp.add(16)); // [16,32): low-nibble sub-table
                let cp = codes_base.add(gi * BLOCK);
                let c0 = vld1q_u8(cp);
                let c1 = vld1q_u8(cp.add(16));
                // low nibble -> lut_lo, high nibble -> lut_hi (matches scalar)
                let s0 = vaddq_u8(
                    vqtbl1q_u8(lut_lo, vandq_u8(c0, mask)),
                    vqtbl1q_u8(lut_hi, vshrq_n_u8(c0, 4)),
                );
                let s1 = vaddq_u8(
                    vqtbl1q_u8(lut_lo, vandq_u8(c1, mask)),
                    vqtbl1q_u8(lut_hi, vshrq_n_u8(c1, 4)),
                );
                accum[0] = vaddw_u8(accum[0], vget_low_u8(s0));
                accum[1] = vaddw_u8(accum[1], vget_high_u8(s0));
                accum[2] = vaddw_u8(accum[2], vget_low_u8(s1));
                accum[3] = vaddw_u8(accum[3], vget_high_u8(s1));
            }
            g += 4;
        }
        while g < g_end {
            let lp = luts_base.add(g * 32);
            let lut_hi = vld1q_u8(lp);
            let lut_lo = vld1q_u8(lp.add(16));
            let cp = codes_base.add(g * BLOCK);
            let c0 = vld1q_u8(cp);
            let c1 = vld1q_u8(cp.add(16));
            let s0 = vaddq_u8(
                vqtbl1q_u8(lut_lo, vandq_u8(c0, mask)),
                vqtbl1q_u8(lut_hi, vshrq_n_u8(c0, 4)),
            );
            let s1 = vaddq_u8(
                vqtbl1q_u8(lut_lo, vandq_u8(c1, mask)),
                vqtbl1q_u8(lut_hi, vshrq_n_u8(c1, 4)),
            );
            accum[0] = vaddw_u8(accum[0], vget_low_u8(s0));
            accum[1] = vaddw_u8(accum[1], vget_high_u8(s0));
            accum[2] = vaddw_u8(accum[2], vget_low_u8(s1));
            accum[3] = vaddw_u8(accum[3], vget_high_u8(s1));
            g += 1;
        }

        for i in 0..4 {
            let lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(accum[i])));
            let hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(accum[i])));
            fa[i * 2] = vfmaq_f32(fa[i * 2], v_scale, lo);
            fa[i * 2 + 1] = vfmaq_f32(fa[i * 2 + 1], v_scale, hi);
        }
    }

    let out_ptr = out.as_mut_ptr();
    for i in 0..8 {
        vst1q_f32(out_ptr.add(i * 4), fa[i]);
    }
}

/// Score one block of `BLOCK` vectors for a single query (AVX2 + FMA).
///
/// Outputs the raw inner product `bias + scale * acc` per lane into `out`,
/// in natural lane order (`out[lane]` is vector `block_idx * BLOCK + lane`).
/// Norm application and padding masking are the caller's responsibility.
/// Bit-width-agnostic via `lut.n_byte_groups`.
///
/// Accumulates every byte-group into u16 lanes in a single pass with no
/// intermediate flush. This relies on the x86 branch of `build_query_lut`,
/// which caps LUT entries so the recovered per-lane sum stays below
/// `u16::MAX`; the FAISS even/odd-byte split makes the transient u16
/// wrap-around cancel exactly in the epilogue.
///
/// ### Safety
///
/// `blocked` must hold ≥ `(block_idx + 1) * n_byte_groups * BLOCK` bytes
/// and `lut.luts_u8` ≥ `n_byte_groups * 32`. Requires AVX2 + FMA at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn score_block_avx2(
    lut: &QueryLut,
    blocked: &[u8],
    block_idx: usize,
    out: &mut [f32; BLOCK],
) {
    // same import part here...
    use std::arch::x86_64::*;

    let n_byte_groups = lut.n_byte_groups;
    let mask = _mm256_set1_epi8(0x0F);
    let codes_base = blocked.as_ptr().add(block_idx * n_byte_groups * BLOCK);
    let luts_base = lut.luts_u8.as_ptr();

    // Even/odd-byte split accumulators (FAISS fast-scan). acc0/acc2 hold
    // the low-nibble and high-nibble lookups with high-byte contamination;
    // acc1/acc3 hold the contaminating high bytes, subtracted out below.
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();

    for g in 0..n_byte_groups {
        let cp = codes_base.add(g * BLOCK);
        let codes_v = _mm256_loadu_si256(cp as *const __m256i);
        let clo = _mm256_and_si256(codes_v, mask);
        let chi = _mm256_and_si256(_mm256_srli_epi16(codes_v, 4), mask);

        // The 32-byte LUT broadcasts the same 16-entry sub-table into both
        // 128-bit lanes; shuffle_epi8 is lane-local, so each half scores
        // its own 16 vectors. [0,16) = high-nibble table, [16,32) = low.
        let lut_v = _mm256_loadu_si256(luts_base.add(g * 32) as *const __m256i);
        let res_lo = _mm256_shuffle_epi8(lut_v, clo);
        let res_hi = _mm256_shuffle_epi8(lut_v, chi);

        acc0 = _mm256_add_epi16(acc0, res_lo);
        acc1 = _mm256_add_epi16(acc1, _mm256_srli_epi16(res_lo, 8));
        acc2 = _mm256_add_epi16(acc2, res_hi);
        acc3 = _mm256_add_epi16(acc3, _mm256_srli_epi16(res_hi, 8));
    }

    // Recover the clean even-byte sums.
    acc0 = _mm256_sub_epi16(acc0, _mm256_slli_epi16(acc1, 8));
    acc2 = _mm256_sub_epi16(acc2, _mm256_slli_epi16(acc3, 8));

    // Recombine the interleaved u16 lanes into natural vector order.
    let dis0 = _mm256_add_epi16(
        _mm256_permute2x128_si256(acc0, acc1, 0x21),
        _mm256_blend_epi32(acc0, acc1, 0xF0),
    );
    let dis1 = _mm256_add_epi16(
        _mm256_permute2x128_si256(acc2, acc3, 0x21),
        _mm256_blend_epi32(acc2, acc3, 0xF0),
    );

    let v_scale = _mm256_set1_ps(lut.scale);
    let v_bias = _mm256_set1_ps(lut.bias);

    let f0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0)));
    let f1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0, 1)));
    let f2 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1)));
    let f3 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1, 1)));

    let bp = out.as_mut_ptr();
    _mm256_storeu_ps(bp, _mm256_fmadd_ps(v_scale, f0, v_bias));
    _mm256_storeu_ps(bp.add(8), _mm256_fmadd_ps(v_scale, f1, v_bias));
    _mm256_storeu_ps(bp.add(16), _mm256_fmadd_ps(v_scale, f2, v_bias));
    _mm256_storeu_ps(bp.add(24), _mm256_fmadd_ps(v_scale, f3, v_bias));
}

//////////////////////////
// Metric key transform //
//////////////////////////

/// Convert a raw inner product into a higher-is-better ranking key.
///
/// `ip` is the approximate `cos(q, v)` shrunk by the per-vector self-overlap
/// `<u, x̂>`; multiplying by `vec_correction = 1 / <u, x̂>` debiases it. The key
/// is monotone-decreasing in the final distance, so the top-k by key is the
/// nearest-k by distance.
///
/// ### Params
///
/// * `ip` - Approximate cosine similarity (raw kernel output)
/// * `query_norm` - L2 norm of the query
/// * `vec_norm` - L2 norm of the stored vector
/// * `vec_correction` - Per-vector debias factor `1 / <u, x̂>`
/// * `metric` - Distance metric
///
/// ### Returns
///
/// The distance (higher is better)
#[inline]
pub fn ip_to_key(
    ip: f32,
    query_norm: f32,
    vec_norm: f32,
    vec_correction: f32,
    metric: Dist,
) -> f32 {
    match metric {
        Dist::Cosine => vec_correction * ip,
        Dist::SquaredEuclidean => {
            (2.0 * query_norm * vec_norm * vec_correction).mul_add(ip, -(vec_norm * vec_norm))
        }
        Dist::Manhattan => unreachable!("TurboQuant does not support Manhattan distance"),
    }
}

/// Reconstruct the metric distance from a ranking key.
///
/// Inverse of [`ip_to_key`] composed with the distance definition; agrees with
/// [`crate::binary::tq_dists::reconstruct_distance()`] for the same inputs.
///
/// ### Params
///
/// * `key` - Ranking key from [`ip_to_key`]
/// * `query_norm` - L2 norm of the query
/// * `metric` - Distance metric
///
/// ### Returns
///
/// The distance (lower is better)
#[inline]
pub fn key_to_distance(key: f32, query_norm: f32, metric: Dist) -> f32 {
    match metric {
        Dist::Cosine => 1.0 - key,
        Dist::SquaredEuclidean => (query_norm * query_norm - key).max(0.0),
        Dist::Manhattan => unreachable!("TurboQuant does not support Manhattan distance"),
    }
}

///////////////////////
// Flat top-k buffer //
///////////////////////

/// Bounded top-k buffer keeping the `k` largest keys seen.
///
/// A flat array with linear min-tracking rather than a binary heap: for the
/// small `k` typical of ANN search this is more cache-friendly than a
/// pointer-chasing heap. Replacement is strict (`key > min`), so on exact ties
/// the earlier insertion is retained.
pub struct TopK {
    /// Ranking keys for the current top-k candidates, length `k`.
    keys: Vec<f32>,
    /// Vector indices corresponding to each key, length `k`.
    indices: Vec<u32>,
    /// Maximum number of candidates retained.
    k: usize,
    /// Number of candidates inserted so far (saturates at `k`).
    size: usize,
    /// Current minimum key among the retained candidates.
    min: f32,
    /// Position of the minimum key within `keys`.
    min_idx: usize,
}

impl TopK {
    /// Allocate a new buffer retaining the top `k` candidates.
    ///
    /// ### Params
    ///
    /// * `k` - Number of candidates to return
    ///
    /// ### Returns
    ///
    /// Self
    pub fn new(k: usize) -> Self {
        Self {
            keys: vec![f32::NEG_INFINITY; k],
            indices: vec![0u32; k],
            k,
            size: 0,
            min: f32::NEG_INFINITY,
            min_idx: 0,
        }
    }

    /// Offer a candidate for inclusion.
    ///
    /// Accepted unconditionally until the buffer is full, then only if `key`
    /// strictly exceeds the current minimum.
    ///
    /// ### Params
    ///
    /// * `key` - Ranking key (higher is better)
    /// * `idx` - Vector index
    #[inline]
    pub fn push(&mut self, key: f32, idx: u32) {
        if self.size < self.k {
            self.keys[self.size] = key;
            self.indices[self.size] = idx;
            self.size += 1;
            if self.size == self.k {
                self.recompute_min();
            }
        } else if key > self.min {
            self.keys[self.min_idx] = key;
            self.indices[self.min_idx] = idx;
            self.recompute_min();
        }
    }

    /// Recompute `min` and `min_idx` by linear scan over `keys[..k]`.
    ///
    /// Called after every insertion once the buffer is full.
    #[inline]
    fn recompute_min(&mut self) {
        self.min = self.keys[0];
        self.min_idx = 0;
        for h in 1..self.k {
            if self.keys[h] < self.min {
                self.min = self.keys[h];
                self.min_idx = h;
            }
        }
    }

    /// Drain into `(indices, distances)` sorted nearest-first.
    ///
    /// ### Params
    ///
    /// * `query_norm` - L2 norm of the query, used by [`key_to_distance`]
    /// * `metric` - Distance metric
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted descending by key, ties broken by
    /// ascending index for reproducibility.
    pub fn into_sorted(self, query_norm: f32, metric: Dist) -> (Vec<u32>, Vec<f32>) {
        let mut pairs: Vec<(f32, u32)> = self.keys[..self.size]
            .iter()
            .zip(self.indices[..self.size].iter())
            .map(|(&key, &idx)| (key, idx))
            .collect();

        pairs.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let indices = pairs.iter().map(|p| p.1).collect();
        let dists = pairs
            .iter()
            .map(|p| key_to_distance(p.0, query_norm, metric))
            .collect();
        (indices, dists)
    }

    /// Check if full
    ///
    /// ### Returns
    ///
    /// True if full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.size >= self.k
    }

    /// Current minimum key.
    ///
    /// Meaningful only once full; before that the kernel never prunes, so the
    /// sentinel `NEG_INFINITY` is never read.
    ///
    /// ### Returns
    ///
    /// Minimum key
    #[inline]
    pub fn min_key(&self) -> f32 {
        self.min
    }
}

////////////////////////////////
// Scalar single-query oracle //
////////////////////////////////

/// Single-query top-k over the blocked layout, scalar path.
///
/// Reference implementation for the SIMD fused driver: scores every block
/// with [`score_block_scalar`], converts to ranking keys, and keeps the
/// top-k. The SIMD kernels must reproduce its `(indices, distances)` for
/// each query (up to fmadd rounding in the score). 2-bit / 4-bit only —
/// 3-bit has no LUT and is served by `tq_dists::turboquant_dist`.
///
/// ### Params
///
/// * `lut` - Query LUT
/// * `blocked` - Blocked code layout
/// * `n_vectors` - Number of real (non-padding) vectors
/// * `n_blocks` - Number of blocks in `blocked`
/// * `norms_f32` - Per-vector L2 norms (length `n_vectors`), f32
/// * `corrections_f32` - Per-vector debias factors `1 / <u, x̂>` (length `n_vectors`), f32
/// * `query_norm` - L2 norm of the query
/// * `metric` - Distance metric
/// * `k` - Neighbours to return
///
/// ### Returns
///
/// `(indices, distances)` sorted nearest-first, length `min(k, n_vectors)`.
#[allow(clippy::too_many_arguments)]
pub fn score_query_topk_scalar(
    lut: &QueryLut,
    blocked: &[u8],
    n_vectors: usize,
    n_blocks: usize,
    norms_f32: &[f32],
    corrections_f32: &[f32],
    query_norm: f32,
    metric: Dist,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut heap = TopK::new(k.min(n_vectors).max(1));
    let mut out = [0.0f32; BLOCK];

    for block_idx in 0..n_blocks {
        score_block_scalar(lut, blocked, block_idx, &mut out);
        let base = block_idx * BLOCK;
        for lane in 0..BLOCK {
            let vi = base + lane;
            if vi >= n_vectors {
                break;
            }
            let key = ip_to_key(
                out[lane],
                query_norm,
                norms_f32[vi],
                corrections_f32[vi],
                metric,
            );
            heap.push(key, vi as u32);
        }
    }

    heap.into_sorted(query_norm, metric)
}

/// Score all blocks for up to 4 queries at once (AVX2 + FMA), returning
/// per-query top-k.
///
/// Loads each block's codes once and scores all `batch_nq` queries against
/// them, amortising the code load and nibble split across queries — the
/// core fusion win. Heaps are updated inline; once a heap is full, a block
/// whose every lane falls at or below the heap minimum is skipped entirely
/// via an in-register compare (the FAISS fast-scan prune).
///
/// Results are bit-identical to [`score_query_topk_scalar`] per query: the
/// integer accumulation matches exactly (the LUT cap keeps the recovered
/// u16 sums exact) and the f32 epilogue uses the same `mul_add` rounding.
///
/// `luts` must hold 4 entries; for `batch_nq < 4` the surplus slots are
/// scored but their results dropped, so callers pad with any valid LUT.
/// 2-bit / 4-bit only.
///
/// ### Safety
///
/// Requires AVX2 + FMA. `blocked`, `norms_f32`, and each `luts[qi]` must be
/// sized consistently with `n_vectors` / `n_byte_groups` as in the scalar path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fused4_avx2(
    luts: &[&QueryLut; 4],
    blocked: &[u8],
    n_vectors: usize,
    n_blocks: usize,
    norms_f32: &[f32],
    corrections_f32: &[f32],
    query_norms: [f32; 4],
    metric: Dist,
    k: usize,
    batch_nq: usize,
) -> Vec<(Vec<u32>, Vec<f32>)> {
    let mut heaps: Vec<TopK> = (0..4).map(|_| TopK::new(k.min(n_vectors).max(1))).collect();
    score_into_heaps_avx2(
        luts,
        blocked,
        n_vectors,
        n_blocks,
        norms_f32,
        corrections_f32,
        query_norms,
        metric,
        batch_nq,
        0,
        &mut heaps,
    );
    heaps
        .into_iter()
        .enumerate()
        .take(batch_nq)
        .map(|(qi, h)| h.into_sorted(query_norms[qi], metric))
        .collect()
}

/// Score `n_blocks` blocks for up to 4 queries into caller-owned heaps
/// (AVX2 + FMA).
///
/// The reusable scoring core behind both [`fused4_avx2`] (exhaustive, one
/// segment, `base_index = 0`) and the IVF path (one call per probed cluster,
/// `base_index` set to the cluster's global slot offset so pushed indices are
/// global). Heaps are persistent across calls, so the in-register prune sees
/// the running minimum across all segments scored so far.
///
/// ### Safety
///
/// Requires AVX2 + FMA. `heaps.len()` must be >= `batch_nq`; buffers sized as
/// in the scalar path.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn score_into_heaps_avx2(
    luts: &[&QueryLut; 4],
    blocked: &[u8],
    n_vectors: usize,
    n_blocks: usize,
    norms_f32: &[f32],
    corrections_f32: &[f32],
    query_norms: [f32; 4],
    metric: Dist,
    batch_nq: usize,
    base_index: u32,
    heaps: &mut [TopK],
) {
    use std::arch::x86_64::*;

    let n_byte_groups = luts[0].n_byte_groups;
    let scales = [luts[0].scale, luts[1].scale, luts[2].scale, luts[3].scale];
    let biases = [luts[0].bias, luts[1].bias, luts[2].bias, luts[3].bias];

    let mask = _mm256_set1_epi8(0x0F);
    let codes_base = blocked.as_ptr();

    for b in 0..n_blocks {
        let base_vec = b * BLOCK;
        let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;

        let mut accus = [[_mm256_setzero_si256(); 4]; 4];
        for g in 0..n_byte_groups {
            let cp = codes_base.add((b * n_byte_groups + g) * BLOCK);
            let codes_v = _mm256_loadu_si256(cp as *const __m256i);
            let clo = _mm256_and_si256(codes_v, mask);
            let chi = _mm256_and_si256(_mm256_srli_epi16(codes_v, 4), mask);
            for qi in 0..4 {
                let lut =
                    _mm256_loadu_si256(luts[qi].luts_u8.as_ptr().add(g * 32) as *const __m256i);
                let res_lo = _mm256_shuffle_epi8(lut, clo);
                let res_hi = _mm256_shuffle_epi8(lut, chi);
                accus[qi][0] = _mm256_add_epi16(accus[qi][0], res_lo);
                accus[qi][1] = _mm256_add_epi16(accus[qi][1], _mm256_srli_epi16(res_lo, 8));
                accus[qi][2] = _mm256_add_epi16(accus[qi][2], res_hi);
                accus[qi][3] = _mm256_add_epi16(accus[qi][3], _mm256_srli_epi16(res_hi, 8));
            }
        }

        for qi in 0..batch_nq {
            epilogue_one_query_avx2(
                &accus[qi],
                base_vec,
                end_lane,
                norms_f32,
                corrections_f32,
                scales[qi],
                biases[qi],
                query_norms[qi],
                metric,
                base_index,
                &mut heaps[qi],
            );
        }
    }
}

/// Per-query block epilogue (AVX2): combine accumulators, convert, apply
/// the metric key transform, prune, and update one query's heap.
///
/// Shared by [`fused4_avx2`] and the AVX-512BW driver (which extracts each
/// block's 256-bit accumulator half and calls this once per block).
/// `acc` holds the four interleaved u16 accumulators for a single query
/// over a single 32-vector block.
///
/// ### Safety
///
/// Requires AVX2 + FMA. `norms_f32` and `corrections_f32` must be valid for
/// `base_vec + end_lane`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn epilogue_one_query_avx2(
    acc: &[std::arch::x86_64::__m256i; 4],
    base_vec: usize,
    end_lane: usize,
    norms_f32: &[f32],
    corrections_f32: &[f32],
    scale: f32,
    bias: f32,
    query_norm: f32,
    metric: Dist,
    base_index: u32,
    heap: &mut TopK,
) {
    use std::arch::x86_64::*;

    let v_scale = _mm256_set1_ps(scale);
    let v_bias = _mm256_set1_ps(bias);

    let mut a0 = acc[0];
    let a1 = acc[1];
    let mut a2 = acc[2];
    let a3 = acc[3];
    a0 = _mm256_sub_epi16(a0, _mm256_slli_epi16(a1, 8));
    a2 = _mm256_sub_epi16(a2, _mm256_slli_epi16(a3, 8));
    let dis0 = _mm256_add_epi16(
        _mm256_permute2x128_si256(a0, a1, 0x21),
        _mm256_blend_epi32(a0, a1, 0xF0),
    );
    let dis1 = _mm256_add_epi16(
        _mm256_permute2x128_si256(a2, a3, 0x21),
        _mm256_blend_epi32(a2, a3, 0xF0),
    );

    let f0 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0)));
    let f1 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0, 1)));
    let f2 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1)));
    let f3 = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1, 1)));

    let ip0 = _mm256_fmadd_ps(v_scale, f0, v_bias);
    let ip1 = _mm256_fmadd_ps(v_scale, f1, v_bias);
    let ip2 = _mm256_fmadd_ps(v_scale, f2, v_bias);
    let ip3 = _mm256_fmadd_ps(v_scale, f3, v_bias);

    let norms_ptr = norms_f32.as_ptr().add(base_vec);
    let corrections_ptr = corrections_f32.as_ptr().add(base_vec);

    match metric {
        Dist::Cosine => {
            if end_lane == BLOCK {
                let c0 = _mm256_loadu_ps(corrections_ptr);
                let c1 = _mm256_loadu_ps(corrections_ptr.add(8));
                let c2 = _mm256_loadu_ps(corrections_ptr.add(16));
                let c3 = _mm256_loadu_ps(corrections_ptr.add(24));
                let k0 = _mm256_mul_ps(c0, ip0);
                let k1 = _mm256_mul_ps(c1, ip1);
                let k2 = _mm256_mul_ps(c2, ip2);
                let k3 = _mm256_mul_ps(c3, ip3);

                if heap.is_full() {
                    let thr = _mm256_set1_ps(heap.min_key());
                    let m = _mm256_movemask_ps(_mm256_cmp_ps(k0, thr, _CMP_GT_OQ))
                        | _mm256_movemask_ps(_mm256_cmp_ps(k1, thr, _CMP_GT_OQ))
                        | _mm256_movemask_ps(_mm256_cmp_ps(k2, thr, _CMP_GT_OQ))
                        | _mm256_movemask_ps(_mm256_cmp_ps(k3, thr, _CMP_GT_OQ));
                    if m == 0 {
                        return;
                    }
                }
                let mut buf = [0.0f32; BLOCK];
                let bp = buf.as_mut_ptr();
                _mm256_storeu_ps(bp, k0);
                _mm256_storeu_ps(bp.add(8), k1);
                _mm256_storeu_ps(bp.add(16), k2);
                _mm256_storeu_ps(bp.add(24), k3);
                for lane in 0..BLOCK {
                    heap.push(buf[lane], base_index + (base_vec + lane) as u32);
                }
            } else {
                let mut buf = [0.0f32; BLOCK];
                let bp = buf.as_mut_ptr();
                _mm256_storeu_ps(bp, ip0);
                _mm256_storeu_ps(bp.add(8), ip1);
                _mm256_storeu_ps(bp.add(16), ip2);
                _mm256_storeu_ps(bp.add(24), ip3);
                for lane in 0..end_lane {
                    let vi = base_vec + lane;
                    let key = ip_to_key(
                        buf[lane],
                        query_norm,
                        norms_f32[vi],
                        corrections_f32[vi],
                        metric,
                    );
                    heap.push(key, base_index + vi as u32);
                }
            }
        }
        Dist::SquaredEuclidean => {
            if end_lane == BLOCK {
                let v_two_qn = _mm256_set1_ps(2.0 * query_norm);
                let key_of = |ip: __m256, off: usize| {
                    let vn = _mm256_loadu_ps(norms_ptr.add(off));
                    let corr = _mm256_loadu_ps(corrections_ptr.add(off));
                    let coef = _mm256_mul_ps(_mm256_mul_ps(v_two_qn, vn), corr);
                    let neg_vn2 = _mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(vn, vn));
                    _mm256_fmadd_ps(coef, ip, neg_vn2)
                };
                let k0 = key_of(ip0, 0);
                let k1 = key_of(ip1, 8);
                let k2 = key_of(ip2, 16);
                let k3 = key_of(ip3, 24);

                if heap.is_full() {
                    let thr = _mm256_set1_ps(heap.min_key());
                    let m = _mm256_movemask_ps(_mm256_cmp_ps(k0, thr, _CMP_GT_OQ))
                        | _mm256_movemask_ps(_mm256_cmp_ps(k1, thr, _CMP_GT_OQ))
                        | _mm256_movemask_ps(_mm256_cmp_ps(k2, thr, _CMP_GT_OQ))
                        | _mm256_movemask_ps(_mm256_cmp_ps(k3, thr, _CMP_GT_OQ));
                    if m == 0 {
                        return;
                    }
                }
                let mut buf = [0.0f32; BLOCK];
                let bp = buf.as_mut_ptr();
                _mm256_storeu_ps(bp, k0);
                _mm256_storeu_ps(bp.add(8), k1);
                _mm256_storeu_ps(bp.add(16), k2);
                _mm256_storeu_ps(bp.add(24), k3);
                for lane in 0..BLOCK {
                    heap.push(buf[lane], base_index + (base_vec + lane) as u32);
                }
            } else {
                let mut buf = [0.0f32; BLOCK];
                let bp = buf.as_mut_ptr();
                _mm256_storeu_ps(bp, ip0);
                _mm256_storeu_ps(bp.add(8), ip1);
                _mm256_storeu_ps(bp.add(16), ip2);
                _mm256_storeu_ps(bp.add(24), ip3);
                for lane in 0..end_lane {
                    let vi = base_vec + lane;
                    let key = ip_to_key(
                        buf[lane],
                        query_norm,
                        norms_f32[vi],
                        corrections_f32[vi],
                        metric,
                    );
                    heap.push(key, base_index + vi as u32);
                }
            }
        }
        Dist::Manhattan => unreachable!("TurboQuant does not support Manhattan distance"),
    }
}

/// AVX-512BW counterpart to [`fused4_avx2`]: scores two 32-vector blocks
/// per inner iteration (64 vectors) via `_mm512_inserti64x4` + a broadcast
/// LUT shuffle, then runs [`epilogue_one_query_avx2`] on each block's
/// extracted 256-bit accumulator half. An odd final block is handled by an
/// inlined AVX2 accumulation pass. Results are identical to [`fused4_avx2`].
///
/// ### Safety
///
/// Requires AVX2 + FMA + AVX-512F + AVX-512BW. Same buffer-sizing contract
/// as [`fused4_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx2",
    enable = "fma",
    enable = "avx512f",
    enable = "avx512bw"
)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn fused4_avx512bw(
    luts: &[&QueryLut; 4],
    blocked: &[u8],
    n_vectors: usize,
    n_blocks: usize,
    norms_f32: &[f32],
    corrections_f32: &[f32],
    query_norms: [f32; 4],
    metric: Dist,
    k: usize,
    batch_nq: usize,
) -> Vec<(Vec<u32>, Vec<f32>)> {
    let mut heaps: Vec<TopK> = (0..4).map(|_| TopK::new(k.min(n_vectors).max(1))).collect();
    score_into_heaps_avx512bw(
        luts,
        blocked,
        n_vectors,
        n_blocks,
        norms_f32,
        corrections_f32,
        query_norms,
        metric,
        batch_nq,
        0,
        &mut heaps,
    );
    heaps
        .into_iter()
        .enumerate()
        .take(batch_nq)
        .map(|(qi, h)| h.into_sorted(query_norms[qi], metric))
        .collect()
}

/// AVX-512BW scoring core into caller-owned heaps. The block-pair counterpart
/// to [`score_into_heaps_avx2`]; see [`fused4_avx512bw`] for the algorithm and
/// [`score_into_heaps_avx2`] for the persistent-heap / `base_index` contract.
///
/// ### Safety
///
/// Requires AVX2 + FMA + AVX-512F + AVX-512BW. `heaps.len()` must be >=
/// `batch_nq`.
#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx2",
    enable = "fma",
    enable = "avx512f",
    enable = "avx512bw"
)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn score_into_heaps_avx512bw(
    luts: &[&QueryLut; 4],
    blocked: &[u8],
    n_vectors: usize,
    n_blocks: usize,
    norms_f32: &[f32],
    corrections_f32: &[f32],
    query_norms: [f32; 4],
    metric: Dist,
    batch_nq: usize,
    base_index: u32,
    heaps: &mut [TopK],
) {
    use std::arch::x86_64::*;

    let n_byte_groups = luts[0].n_byte_groups;
    let mask512 = _mm512_set1_epi8(0x0F);
    let mask256 = _mm256_set1_epi8(0x0F);
    let codes_base = blocked.as_ptr();
    let n_block_pairs = n_blocks / 2;

    for p in 0..n_block_pairs {
        let b0 = p * 2;
        let b1 = b0 + 1;

        let mut accus = [[_mm512_setzero_si512(); 4]; 4];

        for g in 0..n_byte_groups {
            let cp0 = codes_base.add((b0 * n_byte_groups + g) * BLOCK);
            let cp1 = codes_base.add((b1 * n_byte_groups + g) * BLOCK);
            let codes = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_loadu_si256(cp0 as *const __m256i)),
                _mm256_loadu_si256(cp1 as *const __m256i),
                1,
            );
            let clo = _mm512_and_si512(codes, mask512);
            let chi = _mm512_and_si512(_mm512_srli_epi16(codes, 4), mask512);

            for qi in 0..4 {
                let lut256 =
                    _mm256_loadu_si256(luts[qi].luts_u8.as_ptr().add(g * 32) as *const __m256i);
                let lut = _mm512_broadcast_i64x4(lut256);
                let res_lo = _mm512_shuffle_epi8(lut, clo);
                let res_hi = _mm512_shuffle_epi8(lut, chi);
                accus[qi][0] = _mm512_add_epi16(accus[qi][0], res_lo);
                accus[qi][1] = _mm512_add_epi16(accus[qi][1], _mm512_srli_epi16(res_lo, 8));
                accus[qi][2] = _mm512_add_epi16(accus[qi][2], res_hi);
                accus[qi][3] = _mm512_add_epi16(accus[qi][3], _mm512_srli_epi16(res_hi, 8));
            }
        }

        for which in 0..2 {
            let b = b0 + which;
            let base_vec = b * BLOCK;
            if base_vec >= n_vectors {
                break;
            }
            let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;
            for qi in 0..batch_nq {
                let half: [__m256i; 4] = if which == 0 {
                    [
                        _mm512_castsi512_si256(accus[qi][0]),
                        _mm512_castsi512_si256(accus[qi][1]),
                        _mm512_castsi512_si256(accus[qi][2]),
                        _mm512_castsi512_si256(accus[qi][3]),
                    ]
                } else {
                    [
                        _mm512_extracti64x4_epi64(accus[qi][0], 1),
                        _mm512_extracti64x4_epi64(accus[qi][1], 1),
                        _mm512_extracti64x4_epi64(accus[qi][2], 1),
                        _mm512_extracti64x4_epi64(accus[qi][3], 1),
                    ]
                };
                epilogue_one_query_avx2(
                    &half,
                    base_vec,
                    end_lane,
                    norms_f32,
                    corrections_f32,
                    luts[qi].scale,
                    luts[qi].bias,
                    query_norms[qi],
                    metric,
                    base_index,
                    &mut heaps[qi],
                );
            }
        }
    }

    if n_block_pairs * 2 < n_blocks {
        let b = n_block_pairs * 2;
        let base_vec = b * BLOCK;
        let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;

        let mut accus = [[_mm256_setzero_si256(); 4]; 4];
        for g in 0..n_byte_groups {
            let cp = codes_base.add((b * n_byte_groups + g) * BLOCK);
            let codes_v = _mm256_loadu_si256(cp as *const __m256i);
            let clo = _mm256_and_si256(codes_v, mask256);
            let chi = _mm256_and_si256(_mm256_srli_epi16(codes_v, 4), mask256);
            for qi in 0..4 {
                let lut =
                    _mm256_loadu_si256(luts[qi].luts_u8.as_ptr().add(g * 32) as *const __m256i);
                let res_lo = _mm256_shuffle_epi8(lut, clo);
                let res_hi = _mm256_shuffle_epi8(lut, chi);
                accus[qi][0] = _mm256_add_epi16(accus[qi][0], res_lo);
                accus[qi][1] = _mm256_add_epi16(accus[qi][1], _mm256_srli_epi16(res_lo, 8));
                accus[qi][2] = _mm256_add_epi16(accus[qi][2], res_hi);
                accus[qi][3] = _mm256_add_epi16(accus[qi][3], _mm256_srli_epi16(res_hi, 8));
            }
        }
        for qi in 0..batch_nq {
            epilogue_one_query_avx2(
                &accus[qi],
                base_vec,
                end_lane,
                norms_f32,
                corrections_f32,
                luts[qi].scale,
                luts[qi].bias,
                query_norms[qi],
                metric,
                base_index,
                &mut heaps[qi],
            );
        }
    }
}

///////////
// Tests //
///////////

/// NEON scoring core into caller-owned heaps (aarch64).
///
/// Per-query single-block kernel ([`score_block_neon`]) looped over the
/// `batch_nq` queries, then a scalar key transform + heap push. Not 4-query
/// fused (no NEON fused kernel exists), but still SIMD-scored. Same persistent
/// heap / `base_index` contract as [`score_into_heaps_avx2`].
///
/// ### Safety
///
/// `heaps.len()` must be >= `batch_nq`. Buffers sized as in the scalar path.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn score_into_heaps_neon(
    luts: &[&QueryLut; 4],
    blocked: &[u8],
    n_vectors: usize,
    n_blocks: usize,
    norms_f32: &[f32],
    corrections_f32: &[f32],
    query_norms: [f32; 4],
    metric: Dist,
    batch_nq: usize,
    base_index: u32,
    heaps: &mut [TopK],
) {
    let mut out = [0.0f32; BLOCK];
    for qi in 0..batch_nq {
        for b in 0..n_blocks {
            score_block_neon(luts[qi], blocked, b, &mut out);
            let base_vec = b * BLOCK;
            let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;
            for lane in 0..end_lane {
                let vi = base_vec + lane;
                let key = ip_to_key(
                    out[lane],
                    query_norms[qi],
                    norms_f32[vi],
                    corrections_f32[vi],
                    metric,
                );
                heaps[qi].push(key, base_index + vi as u32);
            }
        }
    }
}

/// Scalar scoring core into caller-owned heaps (portable fallback).
///
/// Used on x86 without AVX2/FMA and on architectures without a SIMD kernel.
/// 2-bit / 4-bit only (it consumes a [`QueryLut`]); 3-bit codes are served by
/// the bit-plane path in `tq_dists`. Same persistent heap / `base_index`
/// contract as [`score_into_heaps_avx2`].
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(crate) fn score_into_heaps_scalar(
    luts: &[&QueryLut; 4],
    blocked: &[u8],
    n_vectors: usize,
    n_blocks: usize,
    norms_f32: &[f32],
    corrections_f32: &[f32],
    query_norms: [f32; 4],
    metric: Dist,
    batch_nq: usize,
    base_index: u32,
    heaps: &mut [TopK],
) {
    let mut out = [0.0f32; BLOCK];
    for qi in 0..batch_nq {
        for b in 0..n_blocks {
            score_block_scalar(luts[qi], blocked, b, &mut out);
            let base_vec = b * BLOCK;
            let end_lane = (base_vec + BLOCK).min(n_vectors) - base_vec;
            for lane in 0..end_lane {
                let vi = base_vec + lane;
                let key = ip_to_key(
                    out[lane],
                    query_norms[qi],
                    norms_f32[vi],
                    corrections_f32[vi],
                    metric,
                );
                heaps[qi].push(key, base_index + vi as u32);
            }
        }
    }
}

/// Architecture-dispatched scoring of one blocked segment into caller-owned
/// heaps.
///
/// Picks AVX-512BW / AVX2 / NEON / scalar at runtime and scores `n_blocks`
/// blocks for `batch_nq` queries, pushing global indices `base_index + slot`
/// into `heaps`. Heaps persist across calls, so an IVF index can call this
/// once per probed cluster (with `base_index` set to the cluster's global slot
/// offset) and the running top-k accumulates across clusters. 2-bit / 4-bit
/// only.
///
/// `luts` must hold 4 entries; pad unused slots for `batch_nq < 4`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_into_heaps(
    luts: &[&QueryLut; 4],
    blocked: &[u8],
    n_vectors: usize,
    n_blocks: usize,
    norms_f32: &[f32],
    corrections_f32: &[f32],
    query_norms: [f32; 4],
    metric: Dist,
    batch_nq: usize,
    base_index: u32,
    heaps: &mut [TopK],
) {
    #[cfg(target_arch = "x86_64")]
    {
        let has_fma = is_x86_feature_detected!("fma");
        match detect_simd_level() {
            SimdLevel::Avx512 if has_fma && is_x86_feature_detected!("avx512bw") => unsafe {
                score_into_heaps_avx512bw(
                    luts,
                    blocked,
                    n_vectors,
                    n_blocks,
                    norms_f32,
                    corrections_f32,
                    query_norms,
                    metric,
                    batch_nq,
                    base_index,
                    heaps,
                );
            },
            SimdLevel::Avx512 | SimdLevel::Avx2 if has_fma => unsafe {
                score_into_heaps_avx2(
                    luts,
                    blocked,
                    n_vectors,
                    n_blocks,
                    norms_f32,
                    corrections_f32,
                    query_norms,
                    metric,
                    batch_nq,
                    base_index,
                    heaps,
                );
            },
            _ => score_into_heaps_scalar(
                luts,
                blocked,
                n_vectors,
                n_blocks,
                norms_f32,
                corrections_f32,
                query_norms,
                metric,
                batch_nq,
                base_index,
                heaps,
            ),
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always present on aarch64.
        unsafe {
            score_into_heaps_neon(
                luts,
                blocked,
                n_vectors,
                n_blocks,
                norms_f32,
                corrections_f32,
                query_norms,
                metric,
                batch_nq,
                base_index,
                heaps,
            );
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        score_into_heaps_scalar(
            luts,
            blocked,
            n_vectors,
            n_blocks,
            norms_f32,
            corrections_f32,
            query_norms,
            metric,
            batch_nq,
            base_index,
            heaps,
        );
    }
}

/// Top-k over a single blocked segment (exhaustive path).
///
/// Convenience wrapper over [`score_into_heaps`] with `base_index = 0`: builds
/// the heaps, scores, and returns per-query `(indices, distances)` sorted
/// nearest-first. The exhaustive index calls this; the IVF index drives
/// [`score_into_heaps`] directly with persistent heaps.
///
/// `luts` must hold 4 entries; pad unused slots for `batch_nq < 4`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn topk_blocked(
    luts: &[&QueryLut; 4],
    blocked: &[u8],
    n_vectors: usize,
    n_blocks: usize,
    norms_f32: &[f32],
    corrections_f32: &[f32],
    query_norms: [f32; 4],
    metric: Dist,
    k: usize,
    batch_nq: usize,
) -> Vec<(Vec<u32>, Vec<f32>)> {
    let mut heaps: Vec<TopK> = (0..batch_nq.max(1))
        .map(|_| TopK::new(k.min(n_vectors).max(1)))
        .collect();
    score_into_heaps(
        luts,
        blocked,
        n_vectors,
        n_blocks,
        norms_f32,
        corrections_f32,
        query_norms,
        metric,
        batch_nq,
        0,
        &mut heaps,
    );
    heaps
        .into_iter()
        .enumerate()
        .take(batch_nq)
        .map(|(qi, h)| h.into_sorted(query_norms[qi], metric))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::binary::turboquant::dists::{
        prepare_scalar_scoring, reconstruct_distance, score_ip_scalar,
    };
    use crate::binary::turboquant::pack::{repack, BlockedLayout, BLOCK};
    use crate::binary::turboquant::quantiser::TurboQuantQuantiser;
    use faer::Mat;

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

    fn build(n: usize, dim: usize, bits: usize) -> TurboQuantQuantiser<f32> {
        TurboQuantQuantiser::new(test_data(n, dim).as_ref(), &Dist::Cosine, bits, 42).unwrap()
    }

    fn blocked_data(q: &TurboQuantQuantiser<f32>) -> (Vec<u8>, usize) {
        match repack(&q.storage).unwrap() {
            BlockedLayout::Standard(b) => (b.data, b.n_blocks),
            BlockedLayout::ThreeBit(_) => panic!("3-bit has no blocked scorer"),
        }
    }

    ///////////////
    // LUT tests //
    ///////////////

    #[test]
    fn test_lut_dimensions() {
        for (bits, cpb) in [(4usize, 2usize), (2, 4)] {
            let dim = 128;
            let q = build(4, dim, bits);
            let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.13).sin()).collect();
            let eq = q.encode_query(&query).unwrap();
            let (q_rot, levels) = prepare_scalar_scoring(&eq, &q.encoder);
            let lut = build_query_lut(&q_rot, &levels, bits, dim).unwrap();
            assert_eq!(lut.n_byte_groups, dim / cpb);
            assert_eq!(lut.luts_u8.len(), (dim / cpb) * 32);
        }
    }

    /// The LUT scorer must track the float oracle within u8 rounding.
    /// Tolerance scales with the per-sub-table span (≈ scale * a few ULPs
    /// of accumulation), so we bound it by a small multiple of `scale`.
    fn assert_lut_tracks_oracle(bits: usize, dim: usize, n: usize) {
        let q = build(n, dim, bits);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.073 + 0.4).cos()).collect();
        let eq = q.encode_query(&query).unwrap();
        let (q_rot, levels) = prepare_scalar_scoring(&eq, &q.encoder);
        let lut = build_query_lut(&q_rot, &levels, bits, dim).unwrap();

        // Worst-case rounding: half a quantisation step per lookup, over
        // 2 * n_byte_groups lookups.
        let tol = 0.5 * lut.scale * (2 * lut.n_byte_groups) as f32 + 1e-4;

        for idx in 0..n {
            let packed = q.storage.vector_packed(idx);
            let oracle = score_ip_scalar(&q_rot, packed, &levels, bits, dim);
            let via_lut = score_via_lut_bitplane(&lut, packed, bits, dim);
            assert!(
                (oracle - via_lut).abs() <= tol,
                "idx {idx}: oracle {oracle} vs lut {via_lut}, tol {tol}"
            );
        }
    }

    #[test]
    fn test_lut_tracks_oracle_4bit() {
        assert_lut_tracks_oracle(4, 128, 32);
    }

    #[test]
    fn test_lut_tracks_oracle_2bit() {
        assert_lut_tracks_oracle(2, 128, 32);
    }

    #[test]
    fn test_lut_tracks_oracle_high_dim() {
        assert_lut_tracks_oracle(4, 768, 16);
    }

    /// Ranking is what search actually relies on: the LUT-scored argmax
    /// over a self-query should land on the query's own vector.
    fn assert_lut_ranking_recovers_self(bits: usize, dim: usize, n: usize) {
        let q = build(n, dim, bits);
        let data = test_data(n, dim);
        for t in 0..n {
            let row: Vec<f32> = (0..dim).map(|j| data[(t, j)]).collect();
            let eq = q.encode_query(&row).unwrap();
            let (q_rot, levels) = prepare_scalar_scoring(&eq, &q.encoder);
            let lut = build_query_lut(&q_rot, &levels, bits, dim).unwrap();

            let mut best = (f32::NEG_INFINITY, usize::MAX);
            for idx in 0..n {
                let ip = score_via_lut_bitplane(&lut, q.storage.vector_packed(idx), bits, dim);
                if ip > best.0 {
                    best = (ip, idx);
                }
            }
            assert_eq!(best.1, t, "self-query {t} ranked {} first", best.1);
        }
    }

    #[test]
    fn test_lut_ranking_recovers_self_4bit() {
        assert_lut_ranking_recovers_self(4, 256, 24);
    }

    #[test]
    fn test_lut_ranking_recovers_self_2bit() {
        // 2-bit is coarser; keep dim generous so quantisation noise stays
        // below the inter-vector gap on decorrelated data.
        assert_lut_ranking_recovers_self(2, 512, 16);
    }

    #[test]
    fn test_zero_query_constant_lut() {
        // A zero query rotates to (near) zero, so every product is ~0:
        // all sub-tables collapse, scale falls back to 1, score ≈ bias ≈ 0.
        let dim = 128;
        let q = build(4, dim, 4);
        let eq = q.encode_query(&vec![0.0f32; dim]).unwrap();
        let (q_rot, levels) = prepare_scalar_scoring(&eq, &q.encoder);
        let lut = build_query_lut(&q_rot, &levels, 4, dim).unwrap();
        let ip = score_via_lut_bitplane(&lut, q.storage.vector_packed(0), 4, dim);
        assert!(ip.abs() < 1e-3, "zero-query score {ip} not ≈ 0");
    }

    /////////////////
    // Block tests //
    /////////////////

    /// The blocked-layout scalar scorer must match the bit-plane LUT
    /// scorer exactly (same u8 LUT, same u32 accumulation, only the
    /// traversal differs).
    fn assert_block_matches_bitplane(bits: usize, dim: usize, n: usize) {
        let q = build(n, dim, bits);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.091 + 0.2).sin()).collect();
        let eq = q.encode_query(&query).unwrap();
        let (q_rot, levels) = prepare_scalar_scoring(&eq, &q.encoder);
        let lut = build_query_lut(&q_rot, &levels, bits, dim).unwrap();

        let (blocked, n_blocks) = blocked_data(&q);
        let mut out = [0.0f32; BLOCK];

        for block_idx in 0..n_blocks {
            score_block_scalar(&lut, &blocked, block_idx, &mut out);
            for lane in 0..BLOCK {
                let vec_idx = block_idx * BLOCK + lane;
                if vec_idx >= n {
                    break;
                }
                let expected =
                    score_via_lut_bitplane(&lut, q.storage.vector_packed(vec_idx), bits, dim);
                assert_eq!(
                    out[lane], expected,
                    "bits {bits} block {block_idx} lane {lane} (vec {vec_idx}): \
                         blocked {} != bitplane {expected}",
                    out[lane]
                );
            }
        }
    }

    #[test]
    fn test_block_matches_bitplane_4bit_full() {
        assert_block_matches_bitplane(4, 128, BLOCK);
    }

    #[test]
    fn test_block_matches_bitplane_4bit_partial() {
        assert_block_matches_bitplane(4, 128, BLOCK + 7);
    }

    #[test]
    fn test_block_matches_bitplane_4bit_multiblock() {
        assert_block_matches_bitplane(4, 256, BLOCK * 3 + 5);
    }

    #[test]
    fn test_block_matches_bitplane_2bit() {
        assert_block_matches_bitplane(2, 128, BLOCK * 2 + 3);
    }

    #[test]
    fn test_block_matches_bitplane_2bit_large_dim() {
        assert_block_matches_bitplane(2, 512, 40);
    }

    #[test]
    fn test_block_padding_lanes_decode_zero() {
        // First lane used, rest of the second block is padding -> code 0.
        // Each padded lane's score must equal the all-zero-code score:
        // bias + scale * sum_g (lut[g*32+16] + lut[g*32]).
        let dim = 128;
        let bits = 4;
        let q = build(BLOCK + 1, dim, bits);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.05).cos()).collect();
        let eq = q.encode_query(&query).unwrap();
        let (q_rot, levels) = prepare_scalar_scoring(&eq, &q.encoder);
        let lut = build_query_lut(&q_rot, &levels, bits, dim).unwrap();

        let zero_code_score = {
            let mut acc = 0u32;
            for g in 0..lut.n_byte_groups {
                acc += lut.luts_u8[g * 32 + 16] as u32 + lut.luts_u8[g * 32] as u32;
            }
            lut.scale.mul_add(acc as f32, lut.bias)
        };

        let (blocked, n_blocks) = blocked_data(&q);
        let mut out = [0.0f32; BLOCK];
        score_block_scalar(&lut, &blocked, n_blocks - 1, &mut out);
        for lane in 1..BLOCK {
            assert_eq!(out[lane], zero_code_score, "padded lane {lane} mismatch");
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn assert_neon_matches_scalar(bits: usize, dim: usize, n: usize) {
        let q = build(n, dim, bits);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.083 + 0.3).sin()).collect();
        let eq = q.encode_query(&query).unwrap();
        let (q_rot, levels) = prepare_scalar_scoring(&eq, &q.encoder);
        let lut = build_query_lut(&q_rot, &levels, bits, dim).unwrap();

        let (blocked, n_blocks) = blocked_data(&q);
        let mut scalar_out = [0.0f32; BLOCK];
        let mut neon_out = [0.0f32; BLOCK];

        for block_idx in 0..n_blocks {
            score_block_scalar(&lut, &blocked, block_idx, &mut scalar_out);
            unsafe { score_block_neon(&lut, &blocked, block_idx, &mut neon_out) };
            for lane in 0..BLOCK {
                let a = neon_out[lane];
                let b = scalar_out[lane];
                // Tolerance absorbs FMA-vs-mul/add rounding and per-batch
                // f32 flush accumulation; scales with magnitude. A real
                // lookup/ordering bug shifts a lane by O(0.01+), far above
                // this, so it stays a strong check.
                assert!(
                    (a - b).abs() <= 1e-4 * (1.0 + b.abs()),
                    "bits {bits} block {block_idx} lane {lane}: neon {a} vs scalar {b}"
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_matches_scalar_4bit_single_batch() {
        // n_byte_groups = 64 < FLUSH_EVERY -> one batch.
        assert_neon_matches_scalar(4, 128, BLOCK + 5);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_matches_scalar_2bit_single_batch() {
        assert_neon_matches_scalar(2, 128, BLOCK * 2 + 1);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_matches_scalar_4bit_flush_boundary() {
        // n_byte_groups = 256 == FLUSH_EVERY -> exactly one batch.
        assert_neon_matches_scalar(4, 512, BLOCK + 3);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_matches_scalar_4bit_multi_batch() {
        // n_byte_groups = 768 -> three batches, exercises flush accumulation.
        assert_neon_matches_scalar(4, 1536, BLOCK * 2 + 9);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_matches_scalar_2bit_multi_batch() {
        // 2-bit: n_byte_groups = dim/4 = 384 -> two batches.
        assert_neon_matches_scalar(2, 1536, 50);
    }

    #[cfg(target_arch = "x86_64")]
    fn assert_avx2_matches_scalar(bits: usize, dim: usize, n: usize) {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipping: AVX2/FMA not available");
            return;
        }

        let q = build(n, dim, bits);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.083 + 0.3).sin()).collect();
        let eq = q.encode_query(&query).unwrap();
        let (q_rot, levels) = prepare_scalar_scoring(&eq, &q.encoder);
        let lut = build_query_lut(&q_rot, &levels, bits, dim).unwrap();

        let (blocked, n_blocks) = blocked_data(&q);
        let mut scalar_out = [0.0f32; BLOCK];
        let mut avx_out = [0.0f32; BLOCK];

        for block_idx in 0..n_blocks {
            score_block_scalar(&lut, &blocked, block_idx, &mut scalar_out);
            unsafe { score_block_avx2(&lut, &blocked, block_idx, &mut avx_out) };
            for lane in 0..BLOCK {
                let a = avx_out[lane];
                let b = scalar_out[lane];
                // AVX2 integer-accumulates exactly like the scalar path, so
                // the only divergence is fmadd-vs-mul/add f32 rounding at
                // the final step. Tight relative tolerance; a real combine
                // or LUT bug shifts a lane far beyond this.
                assert!(
                    (a - b).abs() <= 1e-4 * (1.0 + b.abs()),
                    "bits {bits} block {block_idx} lane {lane}: avx2 {a} vs scalar {b}"
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matches_scalar_4bit() {
        assert_avx2_matches_scalar(4, 128, BLOCK + 5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matches_scalar_2bit() {
        assert_avx2_matches_scalar(2, 128, BLOCK * 2 + 1);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matches_scalar_4bit_multiblock() {
        assert_avx2_matches_scalar(4, 256, BLOCK * 3 + 7);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matches_scalar_high_dim() {
        // dim=1536, 4-bit -> n_byte_groups=768. Exercises the LUT cap that
        // keeps the no-flush u16 accumulation from overflowing.
        assert_avx2_matches_scalar(4, 1536, BLOCK + 3);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matches_scalar_2bit_high_dim() {
        // 2-bit, dim=1536 -> n_byte_groups=384.
        assert_avx2_matches_scalar(2, 1536, 50);
    }

    // ... //

    #[test]
    fn test_topk_keeps_largest() {
        let mut h = TopK::new(3);
        for (key, idx) in [
            (1.0, 0u32),
            (5.0, 1),
            (2.0, 2),
            (8.0, 3),
            (3.0, 4),
            (7.0, 5),
        ] {
            h.push(key, idx);
        }
        // Largest three keys 8,7,5 -> indices 3,5,1 (key descending).
        let (idx, _) = h.into_sorted(1.0, Dist::Cosine);
        assert_eq!(idx, vec![3, 5, 1]);
    }

    #[test]
    fn test_topk_fewer_than_k() {
        let mut h = TopK::new(5);
        h.push(2.0, 0);
        h.push(9.0, 1);
        h.push(4.0, 2);
        let (idx, _) = h.into_sorted(1.0, Dist::Cosine);
        assert_eq!(idx, vec![1, 2, 0]);
    }

    #[test]
    fn test_topk_strict_replacement_on_tie() {
        // Equal keys must not evict an incumbent (strict >).
        let mut h = TopK::new(2);
        h.push(5.0, 0);
        h.push(5.0, 1);
        h.push(5.0, 2); // tie with min, rejected
        let (idx, _) = h.into_sorted(1.0, Dist::Cosine);
        assert_eq!(idx, vec![0, 1]);
    }

    #[test]
    fn test_key_distance_roundtrip_matches_reconstruct() {
        // ip_to_key -> key_to_distance must equal tq_dists::reconstruct_distance.
        for &(qn, vn, corr) in &[(1.0f32, 1.0f32, 1.0f32), (2.0, 0.5, 0.95), (0.3, 1.7, 1.05)] {
            for &ip in &[-1.0f32, -0.3, 0.0, 0.6, 1.0] {
                for metric in [Dist::Cosine, Dist::SquaredEuclidean] {
                    let via_key = key_to_distance(ip_to_key(ip, qn, vn, corr, metric), qn, metric);
                    let direct: f32 = reconstruct_distance(ip, qn, vn, corr, metric);
                    assert!(
                        (via_key - direct).abs() <= 1e-5 * (1.0 + direct.abs()),
                        "{metric:?} qn {qn} vn {vn} corr {corr} ip {ip}: key path {via_key} vs direct {direct}"
                    );
                }
            }
        }
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn brute_force_topk(
        lut: &QueryLut,
        blocked: &[u8],
        n: usize,
        n_blocks: usize,
        norms: &[f32],
        corrections: &[f32],
        qnorm: f32,
        metric: Dist,
        k: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut all: Vec<(f32, u32)> = Vec::new();
        let mut out = [0.0f32; BLOCK];
        for b in 0..n_blocks {
            score_block_scalar(lut, blocked, b, &mut out);
            for lane in 0..BLOCK {
                let vi = b * BLOCK + lane;
                if vi >= n {
                    break;
                }
                all.push((
                    ip_to_key(out[lane], qnorm, norms[vi], corrections[vi], metric),
                    vi as u32,
                ));
            }
        }
        all.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1)) // tie-break: lower index first, matching TopK's first-seen
        });
        all.truncate(k.min(n));
        let idx = all.iter().map(|p| p.1).collect();
        let dist = all
            .iter()
            .map(|p| key_to_distance(p.0, qnorm, metric))
            .collect();
        (idx, dist)
    }

    fn assert_oracle_matches_brute(bits: usize, dim: usize, n: usize, k: usize, metric: Dist) {
        let q = TurboQuantQuantiser::new(test_data(n, dim).as_ref(), &metric, bits, 42).unwrap();
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.067 + 0.5).sin()).collect();
        let eq = q.encode_query(&query).unwrap();
        let (q_rot, levels) = prepare_scalar_scoring(&eq, &q.encoder);
        let lut = build_query_lut(&q_rot, &levels, bits, dim).unwrap();
        let (blocked, n_blocks) = blocked_data(&q);

        let (h_idx, h_dist) = score_query_topk_scalar(
            &lut,
            &blocked,
            n,
            n_blocks,
            &q.storage.norms,
            &q.storage.corrections,
            eq.query_norm,
            metric,
            k,
        );
        let (b_idx, b_dist) = brute_force_topk(
            &lut,
            &blocked,
            n,
            n_blocks,
            &q.storage.norms,
            &q.storage.corrections,
            eq.query_norm,
            metric,
            k,
        );

        assert_eq!(h_idx, b_idx, "{metric:?} bits {bits} indices diverge");
        assert_eq!(h_idx.len(), k.min(n));
        for (a, b) in h_dist.iter().zip(b_dist.iter()) {
            assert_eq!(a, b, "{metric:?} distance mismatch");
        }
        // Distances must be ascending.
        for w in h_dist.windows(2) {
            assert!(w[0] <= w[1], "{metric:?} distances not sorted");
        }
    }

    #[test]
    fn test_oracle_matches_brute_4bit_cosine() {
        assert_oracle_matches_brute(4, 128, BLOCK * 3 + 7, 10, Dist::Cosine);
    }

    #[test]
    fn test_oracle_matches_brute_4bit_sqeuclidean() {
        assert_oracle_matches_brute(4, 128, BLOCK * 3 + 7, 10, Dist::SquaredEuclidean);
    }

    #[test]
    fn test_oracle_matches_brute_2bit_cosine() {
        assert_oracle_matches_brute(2, 256, BLOCK * 2 + 5, 8, Dist::Cosine);
    }

    #[test]
    fn test_oracle_k_exceeds_n() {
        assert_oracle_matches_brute(4, 128, 50, 1000, Dist::Cosine);
    }

    #[test]
    fn test_oracle_k_one() {
        assert_oracle_matches_brute(4, 128, BLOCK * 2, 1, Dist::SquaredEuclidean);
    }

    #[cfg(target_arch = "x86_64")]
    type FusedKernel = unsafe fn(
        &[&QueryLut; 4],
        &[u8],
        usize,
        usize,
        &[f32],
        &[f32],
        [f32; 4],
        Dist,
        usize,
        usize,
    ) -> Vec<(Vec<u32>, Vec<f32>)>;

    #[cfg(target_arch = "x86_64")]
    fn assert_fused_matches_oracle(
        kernel: FusedKernel,
        needs_avx512: bool,
        bits: usize,
        dim: usize,
        n: usize,
        k: usize,
        metric: Dist,
        nq: usize,
    ) {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipping: AVX2/FMA not available");
            return;
        }
        if needs_avx512
            && !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw"))
        {
            eprintln!("skipping: AVX-512BW not available");
            return;
        }

        let q = TurboQuantQuantiser::new(test_data(n, dim).as_ref(), &metric, bits, 42).unwrap();
        let (blocked, n_blocks) = blocked_data(&q);

        let queries: Vec<Vec<f32>> = (0..nq)
            .map(|qi| {
                (0..dim)
                    .map(|i| ((i + 3 * qi) as f32 * 0.061 + qi as f32).sin())
                    .collect()
            })
            .collect();

        let mut expected = Vec::new();
        let mut encoded = Vec::new();
        for query in &queries {
            let eq = q.encode_query(query).unwrap();
            let (q_rot, levels) = prepare_scalar_scoring(&eq, &q.encoder);
            let lut = build_query_lut(&q_rot, &levels, bits, dim).unwrap();
            expected.push(score_query_topk_scalar(
                &lut,
                &blocked,
                n,
                n_blocks,
                &q.storage.norms,
                &q.storage.corrections,
                eq.query_norm,
                metric,
                k,
            ));
            encoded.push((lut, eq.query_norm));
        }

        let mut got = Vec::new();
        let mut qi = 0;
        while qi < nq {
            let batch_nq = (nq - qi).min(4);
            let pad = qi + batch_nq - 1;
            let luts: [&QueryLut; 4] = [
                &encoded[qi].0,
                &encoded[(qi + 1).min(pad)].0,
                &encoded[(qi + 2).min(pad)].0,
                &encoded[(qi + 3).min(pad)].0,
            ];
            let qnorms = [
                encoded[qi].1,
                encoded[(qi + 1).min(pad)].1,
                encoded[(qi + 2).min(pad)].1,
                encoded[(qi + 3).min(pad)].1,
            ];
            got.extend(unsafe {
                kernel(
                    &luts,
                    &blocked,
                    n,
                    n_blocks,
                    &q.storage.norms,
                    &q.storage.corrections,
                    qnorms,
                    metric,
                    k,
                    batch_nq,
                )
            });
            qi += batch_nq;
        }

        assert_eq!(got.len(), nq);
        for (i, ((g_idx, g_dist), (e_idx, e_dist))) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(g_idx, e_idx, "{metric:?} query {i} indices diverge");
            assert_eq!(g_dist, e_dist, "{metric:?} query {i} distances diverge");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_fused4_cosine_one_batch() {
        assert_fused_matches_oracle(
            fused4_avx2,
            false,
            4,
            256,
            BLOCK * 4 + 13,
            10,
            Dist::Cosine,
            4,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_fused4_sqeuclidean_one_batch() {
        assert_fused_matches_oracle(
            fused4_avx2,
            false,
            4,
            256,
            BLOCK * 4 + 13,
            10,
            Dist::SquaredEuclidean,
            4,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_fused4_cosine_tail_batch() {
        // nq = 7 -> one full batch of 4 + a padded batch of 3.
        assert_fused_matches_oracle(
            fused4_avx2,
            false,
            4,
            128,
            BLOCK * 3 + 5,
            8,
            Dist::Cosine,
            7,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_fused4_sqeuclidean_tail_batch() {
        assert_fused_matches_oracle(
            fused4_avx2,
            false,
            4,
            128,
            BLOCK * 3 + 5,
            8,
            Dist::SquaredEuclidean,
            7,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_fused4_2bit() {
        assert_fused_matches_oracle(
            fused4_avx2,
            false,
            2,
            512,
            BLOCK * 2 + 9,
            8,
            Dist::Cosine,
            4,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_fused4_prune_path() {
        assert_fused_matches_oracle(fused4_avx2, false, 4, 256, BLOCK * 20, 5, Dist::Cosine, 4);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_fused4_high_dim() {
        assert_fused_matches_oracle(
            fused4_avx2,
            false,
            4,
            1536,
            BLOCK * 5 + 1,
            10,
            Dist::SquaredEuclidean,
            4,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_cosine() {
        assert_fused_matches_oracle(
            fused4_avx512bw,
            true,
            4,
            256,
            BLOCK * 5 + 13,
            10,
            Dist::Cosine,
            4,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_sqeuclidean() {
        assert_fused_matches_oracle(
            fused4_avx512bw,
            true,
            4,
            256,
            BLOCK * 5 + 13,
            10,
            Dist::SquaredEuclidean,
            4,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_odd_blocks() {
        assert_fused_matches_oracle(fused4_avx512bw, true, 4, 128, BLOCK * 7, 8, Dist::Cosine, 4);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_even_blocks() {
        assert_fused_matches_oracle(
            fused4_avx512bw,
            true,
            4,
            128,
            BLOCK * 8,
            8,
            Dist::SquaredEuclidean,
            4,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_tail_batch() {
        assert_fused_matches_oracle(
            fused4_avx512bw,
            true,
            4,
            256,
            BLOCK * 4 + 7,
            8,
            Dist::Cosine,
            7,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx512_high_dim_prune() {
        assert_fused_matches_oracle(
            fused4_avx512bw,
            true,
            4,
            1536,
            BLOCK * 20,
            5,
            Dist::SquaredEuclidean,
            4,
        );
    }

    ////////////////////////////////
    // e4: score_into_heaps tests  //
    ////////////////////////////////

    /// The arch-dispatched `score_into_heaps` over a single segment
    /// (`base_index = 0`) must reproduce the scalar single-query oracle.
    fn assert_dispatch_matches_oracle(bits: usize, dim: usize, n: usize, k: usize, metric: Dist) {
        let q = TurboQuantQuantiser::new(test_data(n, dim).as_ref(), &metric, bits, 42).unwrap();
        let (blocked, n_blocks) = blocked_data(&q);

        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.071 + 0.2).sin()).collect();
        let eq = q.encode_query(&query).unwrap();
        let (q_rot, levels) = prepare_scalar_scoring(&eq, &q.encoder);
        let lut = build_query_lut(&q_rot, &levels, bits, dim).unwrap();

        let (e_idx, e_dist) = score_query_topk_scalar(
            &lut,
            &blocked,
            n,
            n_blocks,
            &q.storage.norms,
            &q.storage.corrections,
            eq.query_norm,
            metric,
            k,
        );

        let luts: [&QueryLut; 4] = [&lut, &lut, &lut, &lut];
        let qnorms = [eq.query_norm; 4];
        let mut heaps: Vec<TopK> = (0..1).map(|_| TopK::new(k.min(n).max(1))).collect();
        score_into_heaps(
            &luts,
            &blocked,
            n,
            n_blocks,
            &q.storage.norms,
            &q.storage.corrections,
            qnorms,
            metric,
            1,
            0,
            &mut heaps,
        );
        let (g_idx, g_dist) = heaps.pop().unwrap().into_sorted(eq.query_norm, metric);

        assert_eq!(
            g_idx, e_idx,
            "{metric:?} bits {bits} dispatch indices diverge"
        );
        for (a, b) in g_dist.iter().zip(e_dist.iter()) {
            assert_eq!(a, b, "{metric:?} dispatch distance mismatch");
        }
    }

    #[test]
    fn test_dispatch_matches_oracle_4bit_cosine() {
        assert_dispatch_matches_oracle(4, 256, BLOCK * 3 + 7, 10, Dist::Cosine);
    }

    #[test]
    fn test_dispatch_matches_oracle_4bit_sqeuclidean() {
        assert_dispatch_matches_oracle(4, 256, BLOCK * 3 + 7, 10, Dist::SquaredEuclidean);
    }

    #[test]
    fn test_dispatch_matches_oracle_2bit() {
        assert_dispatch_matches_oracle(2, 512, BLOCK * 2 + 5, 8, Dist::Cosine);
    }

    /// IVF emulation: two independently-encoded segments sharing one encoder
    /// (same dim/bits/seed), scored into ONE persistent heap with per-segment
    /// `base_index`, must equal a merged brute-force top-k over the global
    /// index space `[0, n0) ++ [n0, n0 + n1)`.
    fn assert_ivf_base_index_accumulates(
        bits: usize,
        dim: usize,
        n0: usize,
        n1: usize,
        k: usize,
        metric: Dist,
    ) {
        // Two distinct data sets, same encoder parameters -> identical rotation
        // and levels, so a single query LUT applies to both.
        let d0 = Mat::from_fn(n0, dim, |i, j| ((i * 31 + j * 7 + 1) as f32 * 0.013).sin());
        let d1 = Mat::from_fn(n1, dim, |i, j| {
            ((i * 17 + j * 5 + 9999) as f32 * 0.019).cos()
        });
        let q0 = TurboQuantQuantiser::new(d0.as_ref(), &metric, bits, 7).unwrap();
        let q1 = TurboQuantQuantiser::new(d1.as_ref(), &metric, bits, 7).unwrap();

        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.043 + 0.6).sin()).collect();
        let eq = q0.encode_query(&query).unwrap();
        let (q_rot, levels) = prepare_scalar_scoring(&eq, &q0.encoder);
        let lut = build_query_lut(&q_rot, &levels, bits, dim).unwrap();
        let qnorm = eq.query_norm;

        let (bl0, nb0) = blocked_data(&q0);
        let (bl1, nb1) = blocked_data(&q1);

        // Persistent heap across both segments.
        let total = n0 + n1;
        let luts: [&QueryLut; 4] = [&lut, &lut, &lut, &lut];
        let qnorms = [qnorm; 4];
        let mut heaps: Vec<TopK> = (0..1).map(|_| TopK::new(k.min(total).max(1))).collect();
        score_into_heaps(
            &luts,
            &bl0,
            n0,
            nb0,
            &q0.storage.norms,
            &q0.storage.corrections,
            qnorms,
            metric,
            1,
            0,
            &mut heaps,
        );
        score_into_heaps(
            &luts,
            &bl1,
            n1,
            nb1,
            &q1.storage.norms,
            &q1.storage.corrections,
            qnorms,
            metric,
            1,
            n0 as u32,
            &mut heaps,
        );
        let (g_idx, g_dist) = heaps.pop().unwrap().into_sorted(qnorm, metric);

        // Merged brute-force oracle over global indices via the bit-plane scorer.
        let mut all: Vec<(f32, u32)> = Vec::new();
        for i in 0..n0 {
            let ip = score_via_lut_bitplane(&lut, q0.storage.vector_packed(i), bits, dim);
            all.push((
                ip_to_key(
                    ip,
                    qnorm,
                    q0.storage.norms[i],
                    q0.storage.corrections[i],
                    metric,
                ),
                i as u32,
            ));
        }
        for j in 0..n1 {
            let ip = score_via_lut_bitplane(&lut, q1.storage.vector_packed(j), bits, dim);
            all.push((
                ip_to_key(
                    ip,
                    qnorm,
                    q1.storage.norms[j],
                    q1.storage.corrections[j],
                    metric,
                ),
                (n0 + j) as u32,
            ));
        }
        all.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        all.truncate(k.min(total));
        let e_idx: Vec<u32> = all.iter().map(|p| p.1).collect();
        let e_dist: Vec<f32> = all
            .iter()
            .map(|p| key_to_distance(p.0, qnorm, metric))
            .collect();

        assert_eq!(
            g_idx, e_idx,
            "{metric:?} bits {bits} IVF-accumulated indices diverge"
        );
        for (a, b) in g_dist.iter().zip(e_dist.iter()) {
            assert_eq!(a, b, "{metric:?} IVF-accumulated distance mismatch");
        }
    }

    #[test]
    fn test_ivf_base_index_cosine() {
        assert_ivf_base_index_accumulates(4, 256, BLOCK * 2 + 5, BLOCK * 3 + 11, 10, Dist::Cosine);
    }

    #[test]
    fn test_ivf_base_index_sqeuclidean() {
        assert_ivf_base_index_accumulates(
            4,
            256,
            BLOCK + 3,
            BLOCK * 2 + 7,
            8,
            Dist::SquaredEuclidean,
        );
    }

    #[test]
    fn test_ivf_base_index_2bit() {
        assert_ivf_base_index_accumulates(2, 512, BLOCK * 2, BLOCK + 9, 8, Dist::Cosine);
    }
}
