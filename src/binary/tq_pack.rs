//! Bit-plane to SIMD-blocked layout repacking for TurboQuant scoring.
//!
//! The flat storage in `TurboQuantStorage` keeps codes in bit-plane format (one
//! plane per bit, byte-packed). For SIMD scoring we need a layout that lets a
//! single `vpshufb` / `vqtbl1q_u8` lookup operate on one nibble across 16
//! vectors at a time.
//!
//! For 2-bit and 4-bit codes, this is a single byte array where each byte holds
//! two codes (or four, for 2-bit) packed as nibbles. For 3-bit codes, the lower
//! two bits go into the nibble array and bit 2 lives in a separate plane ->
//! this matches FAISS's PQ4 trick adapted to a non-power-of-two width.

use crate::binary::tq_quantiser::TurboQuantStorage;
use crate::prelude::*;
use num_traits::{Float, FromPrimitive};

/// Vectors per SIMD block. Matches the original TurboQuant crate; the
/// AVX2 / AVX-512BW / NEON kernels all assume this exact value.
pub const BLOCK: usize = 32;

#[cfg(target_arch = "x86_64")]
const PERM0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];

/// Blocked layout for 2-bit and 4-bit codes.
pub struct BlockedCodes {
    /// Packed nibble bytes in block-major order.
    pub data: Vec<u8>,
    /// Number of `BLOCK`-sized blocks (last may be partially padded).
    pub n_blocks: usize,
}

/// Blocked layout for 3-bit codes (two-array form).
pub struct BlockedCodes3Bit {
    /// Lower two bits as packed nibbles (same shape as `BlockedCodes`
    /// for 2-bit data).
    pub sub_codes: Vec<u8>,
    /// Bit-2 plane, blocked by 32 vectors. One byte per `BLOCK` lane
    /// covers 8 dimensions.
    pub plane2: Vec<u8>,
    /// Number of `BLOCK`-sized blocks.
    pub n_blocks: usize,
}

/// Re-pack a `TurboQuantStorage` for SIMD scoring.
pub enum BlockedLayout {
    /// Standard version
    Standard(BlockedCodes),
    /// Special case, 3 Bits
    ThreeBit(BlockedCodes3Bit),
}

impl BlockedLayout {
    /// Number of `BLOCK`-sized blocks.
    pub fn n_blocks(&self) -> usize {
        match self {
            BlockedLayout::Standard(b) => b.n_blocks,
            BlockedLayout::ThreeBit(b) => b.n_blocks,
        }
    }

    /// Memory usage in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            BlockedLayout::Standard(b) => b.data.capacity(),
            BlockedLayout::ThreeBit(b) => b.sub_codes.capacity() + b.plane2.capacity(),
        }
    }
}

/// Re-pack a `TurboQuantStorage` into the blocked SIMD layout.
///
/// Dispatches to [`repack_standard`] for 2-bit and 4-bit codes, and to
/// [`repack_3bit`] for 3-bit codes. The resulting layout is consumed by
/// the platform-specific SIMD scoring kernels.
///
/// ### Params
///
/// * `storage` - Encoded vector storage in bit-plane format
///
/// ### Returns
///
/// A [`BlockedLayout`] ready for SIMD scoring, or an error if the bit
/// width stored in `storage` is not 2, 3, or 4.
pub fn repack<T>(storage: &TurboQuantStorage<T>) -> Result<BlockedLayout, AnnSearchErrors>
where
    T: Float + FromPrimitive,
{
    match storage.bits {
        2 | 4 => Ok(BlockedLayout::Standard(repack_standard(
            &storage.packed_codes,
            storage.n,
            storage.bits,
            storage.dim,
        ))),
        3 => Ok(BlockedLayout::ThreeBit(repack_3bit(
            &storage.packed_codes,
            storage.n,
            storage.dim,
        ))),
        _ => Err(AnnSearchErrors::TQInvalidBits {
            n_bits: storage.bits,
        }),
    }
}

/// Re-pack 2-bit or 4-bit codes from bit-plane to blocked nibble layout.
///
/// Each output byte holds `8 / bits` codes packed most-significant-code
/// first. Vectors are grouped into blocks of [`BLOCK`] and arranged in the
/// platform-specific order expected by the SIMD scoring kernel.
///
/// ### Params
///
/// * `packed_codes` - Bit-plane storage from [`TurboQuantStorage`]
/// * `n_vectors` - Number of encoded vectors
/// * `bits` - Bits per coordinate (2 or 4)
/// * `dim` - Dimensionality of the vectors
///
/// ### Returns
///
/// A [`BlockedCodes`] with the repacked nibble data and block count.
fn repack_standard(packed_codes: &[u8], n_vectors: usize, bits: usize, dim: usize) -> BlockedCodes {
    let bytes_per_plane = dim / 8;
    let codes_per_byte = 8 / bits;
    let n_byte_groups = dim / codes_per_byte;
    let n_blocks = n_vectors.div_ceil(BLOCK);
    let bytes_per_row = bits * bytes_per_plane;

    // Step 1: walk the bit-plane storage and assemble one nibble byte per
    // (vector, byte_group). Each byte holds `codes_per_byte` codes packed
    // most-significant-code first to match what the SIMD kernel expects.
    let mut codes_flat = vec![0u8; n_vectors * n_byte_groups];
    for vec_idx in 0..n_vectors {
        for g in 0..n_byte_groups {
            let dim_start = g * codes_per_byte;
            let mut byte_val = 0u8;
            for c in 0..codes_per_byte {
                let j = dim_start + c;
                let byte_in_plane = j / 8;
                let bit_in_byte = 7 - (j % 8);
                let mask = 1u8 << bit_in_byte;

                let mut code = 0u8;
                for p in 0..bits {
                    let plane_byte =
                        packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane];
                    if plane_byte & mask != 0 {
                        code |= 1 << p;
                    }
                }

                let shift = (codes_per_byte - 1 - c) * bits;
                byte_val |= code << shift;
            }
            codes_flat[vec_idx * n_byte_groups + g] = byte_val;
        }
    }

    let data = pack_blocked(n_vectors, n_blocks, n_byte_groups, &codes_flat);
    BlockedCodes { data, n_blocks }
}

/// Re-pack 3-bit codes into nibble (lower 2 bits) + plane2 (bit 2) layout.
///
/// The lower two bits of each code are packed four-per-byte into `sub_codes`
/// using the same nibble layout as [`repack_standard`] for 2-bit data. Bit 2
/// is extracted into a separate byte-packed plane (`plane2`) blocked by
/// [`BLOCK`] vectors, one byte covering 8 dimensions. This matches the FAISS
/// PQ4 split-plane trick adapted for a non-power-of-two code width.
///
/// ### Params
///
/// * `packed_codes` - Bit-plane storage from [`TurboQuantStorage`]
/// * `n_vectors` - Number of encoded vectors
/// * `dim` - Dimensionality of the vectors
///
/// ### Returns
///
/// A [`BlockedCodes3Bit`] containing the split-plane blocked layout.
fn repack_3bit(packed_codes: &[u8], n_vectors: usize, dim: usize) -> BlockedCodes3Bit {
    let bytes_per_plane = dim / 8;
    let bytes_per_row = 3 * bytes_per_plane;
    let n_blocks = n_vectors.div_ceil(BLOCK);

    // Sub-codes: lower 2 bits of each 3-bit code, packed 4 codes per byte
    // (i.e. 2 codes per nibble). Same shape as 2-bit `repack_standard`
    // output, so we can reuse its scoring kernel.
    let sub_byte_groups = dim / 4;
    let mut sub_flat = vec![0u8; n_vectors * sub_byte_groups];
    for vec_idx in 0..n_vectors {
        for g in 0..sub_byte_groups {
            let dim_start = g * 4;
            let mut byte_val = 0u8;
            for c in 0..4 {
                let j = dim_start + c;
                let byte_in_plane = j / 8;
                let bit_in_byte = 7 - (j % 8);
                let mask = 1u8 << bit_in_byte;

                let mut code = 0u8;
                for p in 0..2 {
                    let plane_byte =
                        packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane];
                    if plane_byte & mask != 0 {
                        code |= 1 << p;
                    }
                }
                byte_val |= code << ((3 - c) * 2);
            }
            sub_flat[vec_idx * sub_byte_groups + g] = byte_val;
        }
    }
    let sub_codes = pack_blocked(n_vectors, n_blocks, sub_byte_groups, &sub_flat);

    // Plane2: just bit 2 of each code, byte-packed at 8 dims/byte. We
    // store one byte per (block, dim_group, lane) so the kernel can OR
    // it into the score after the nibble lookup.
    let plane2_byte_groups = bytes_per_plane;
    let mut plane2 = vec![0u8; n_blocks * plane2_byte_groups * BLOCK];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        for g in 0..plane2_byte_groups {
            let out_offset = (block_idx * plane2_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vec_idx = base_vec + lane;
                if vec_idx >= n_vectors {
                    continue;
                }
                plane2[out_offset + lane] =
                    packed_codes[vec_idx * bytes_per_row + 2 * bytes_per_plane + g];
            }
        }
    }

    BlockedCodes3Bit {
        sub_codes,
        plane2,
        n_blocks,
    }
}

/// Pack a flat `(n_vectors × n_byte_groups)` code array into the
/// platform-specific blocked layout consumed by the SIMD scoring kernels.
///
/// On x86-64 the FAISS `perm0` interleaving is applied so that AVX2 and
/// AVX-512BW cross-lane behaviour aligns with `vpshufb` lookups. On all
/// other targets a sequential per-lane layout is used, matching NEON
/// `vqtbl1q_u8` and the scalar fallback.
///
/// Partial blocks at the end of the vector array are zero-padded.
///
/// ### Params
///
/// * `n` - Number of vectors (may be less than `n_blocks * BLOCK`)
/// * `n_blocks` - Number of [`BLOCK`]-sized groups
/// * `n_byte_groups` - Number of code bytes per vector
/// * `codes_flat` - Row-major input with shape `n × n_byte_groups`
///
/// ### Returns
///
/// Blocked byte array with shape `n_blocks × n_byte_groups × BLOCK`.
#[cfg(target_arch = "x86_64")]
fn pack_blocked(n: usize, n_blocks: usize, n_byte_groups: usize, codes_flat: &[u8]) -> Vec<u8> {
    // FAISS perm0 layout: split each byte into hi/lo nibbles, interleave
    // pairs of vectors `(perm0[j], perm0[j] + 16)` so AVX2/AVX-512 cross-
    // lane behaviour aligns with the lookup.
    let blocked_size = n_blocks * n_byte_groups * BLOCK;
    let mut blocked = vec![0u8; blocked_size];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (block_idx * n_byte_groups + g) * BLOCK;
            for j in 0..16 {
                let va = base_vec + PERM0[j];
                let vb = base_vec + PERM0[j] + 16;
                let ba = if va < n {
                    codes_flat[va * n_byte_groups + g]
                } else {
                    0
                };
                let bb = if vb < n {
                    codes_flat[vb * n_byte_groups + g]
                } else {
                    0
                };
                blocked[out_offset + j] = (ba >> 4) | ((bb >> 4) << 4);
                blocked[out_offset + 16 + j] = (ba & 0x0F) | ((bb & 0x0F) << 4);
            }
        }
    }
    blocked
}

/// Pack a flat `(n_vectors × n_byte_groups)` code array into the
/// platform-specific blocked layout consumed by the SIMD scoring kernels.
///
/// On x86-64 the FAISS `perm0` interleaving is applied so that AVX2 and
/// AVX-512BW cross-lane behaviour aligns with `vpshufb` lookups. On all
/// other targets a sequential per-lane layout is used, matching NEON
/// `vqtbl1q_u8` and the scalar fallback.
///
/// Partial blocks at the end of the vector array are zero-padded.
///
/// ### Params
///
/// * `n` - Number of vectors (may be less than `n_blocks * BLOCK`)
/// * `n_blocks` - Number of [`BLOCK`]-sized groups
/// * `n_byte_groups` - Number of code bytes per vector
/// * `codes_flat` - Row-major input with shape `n × n_byte_groups`
///
/// ### Returns
///
/// Blocked byte array with shape `n_blocks × n_byte_groups × BLOCK`.
#[cfg(not(target_arch = "x86_64"))]
fn pack_blocked(n: usize, n_blocks: usize, n_byte_groups: usize, codes_flat: &[u8]) -> Vec<u8> {
    // Sequential layout: each code byte stored as-is, vectors in order.
    // Fits NEON's per-lane loads and the scalar fallback.
    let blocked_size = n_blocks * n_byte_groups * BLOCK;
    let mut blocked = vec![0u8; blocked_size];
    for block_idx in 0..n_blocks {
        let base_vec = block_idx * BLOCK;
        for g in 0..n_byte_groups {
            let out_offset = (block_idx * n_byte_groups + g) * BLOCK;
            for lane in 0..BLOCK {
                let vi = base_vec + lane;
                if vi < n {
                    blocked[out_offset + lane] = codes_flat[vi * n_byte_groups + g];
                }
            }
        }
    }
    blocked
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary::tq_quantiser::TurboQuantQuantiser;
    use faer::Mat;

    /// Decode a single coordinate's code from the bit-plane storage.
    fn decode_code_from_bitplane(
        packed_codes: &[u8],
        vec_idx: usize,
        dim_idx: usize,
        bits: usize,
        dim: usize,
    ) -> u8 {
        let bytes_per_plane = dim / 8;
        let bytes_per_row = bits * bytes_per_plane;
        let byte_in_plane = dim_idx / 8;
        let bit_in_byte = 7 - (dim_idx % 8);
        let mask = 1u8 << bit_in_byte;

        let mut code = 0u8;
        for p in 0..bits {
            let plane_byte =
                packed_codes[vec_idx * bytes_per_row + p * bytes_per_plane + byte_in_plane];
            if plane_byte & mask != 0 {
                code |= 1 << p;
            }
        }
        code
    }

    /// Decode a coordinate's code from the standard blocked layout.
    /// Inverts both the nibble packing and the platform-specific block layout.
    fn decode_code_from_blocked_standard(
        blocked: &[u8],
        vec_idx: usize,
        dim_idx: usize,
        bits: usize,
        dim: usize,
    ) -> u8 {
        let codes_per_byte = 8 / bits;
        let n_byte_groups = dim / codes_per_byte;
        let g = dim_idx / codes_per_byte;
        let c = dim_idx % codes_per_byte;
        let shift = (codes_per_byte - 1 - c) * bits;
        let code_mask = (1u8 << bits) - 1;

        let block_idx = vec_idx / BLOCK;
        let lane = vec_idx % BLOCK;
        let block_base = (block_idx * n_byte_groups + g) * BLOCK;

        let nibble = decode_nibble_at_lane(blocked, block_base, lane);
        (nibble >> shift) & code_mask
    }

    /// Pull the nibble for `lane` from a 32-byte block, inverting the
    /// platform-specific layout.
    #[cfg(target_arch = "x86_64")]
    fn decode_nibble_at_lane(blocked: &[u8], block_base: usize, lane: usize) -> u8 {
        // Find which `j` and which half (lo/hi 16 of the block) holds this lane.
        let (j, is_high) = if lane < 16 {
            (PERM0.iter().position(|&p| p == lane).unwrap(), false)
        } else {
            (PERM0.iter().position(|&p| p == lane - 16).unwrap(), true)
        };

        // Each block has two halves of 16 bytes each. The first 16 bytes
        // hold high nibbles (one for `lane = perm0[j]` in the low 4 bits,
        // one for `lane = perm0[j] + 16` in the high 4 bits). The second
        // 16 bytes hold low nibbles in the same arrangement.
        let hi_byte = blocked[block_base + j];
        let lo_byte = blocked[block_base + 16 + j];

        let hi = if is_high {
            (hi_byte >> 4) & 0x0F
        } else {
            hi_byte & 0x0F
        };
        let lo = if is_high {
            (lo_byte >> 4) & 0x0F
        } else {
            lo_byte & 0x0F
        };
        (hi << 4) | lo
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn decode_nibble_at_lane(blocked: &[u8], block_base: usize, lane: usize) -> u8 {
        blocked[block_base + lane]
    }

    fn build_quantiser(n: usize, dim: usize, bits: usize) -> TurboQuantQuantiser<f32> {
        let mut data = Mat::<f32>::zeros(n, dim);
        for i in 0..n {
            for j in 0..dim {
                data[(i, j)] = ((i * dim + j) as f32 * 0.137).sin();
            }
        }
        TurboQuantQuantiser::new(
            data.as_ref(),
            &crate::prelude::Dist::SquaredEuclidean,
            bits,
            42,
        )
        .unwrap()
    }

    fn assert_roundtrip_standard(n: usize, dim: usize, bits: usize) {
        let q = build_quantiser(n, dim, bits);
        let layout = repack(&q.storage).unwrap();
        let blocked = match layout {
            BlockedLayout::Standard(b) => b,
            _ => panic!("expected Standard layout for {bits}-bit"),
        };
        assert_eq!(blocked.n_blocks, n.div_ceil(BLOCK));

        for i in 0..n {
            for d in 0..dim {
                let from_bp = decode_code_from_bitplane(&q.storage.packed_codes, i, d, bits, dim);
                let from_bl = decode_code_from_blocked_standard(&blocked.data, i, d, bits, dim);
                assert_eq!(
                    from_bp, from_bl,
                    "vec {i} dim {d}: bit-plane {from_bp} != blocked {from_bl}"
                );
            }
        }
    }

    #[test]
    fn test_roundtrip_4bit_full_block() {
        assert_roundtrip_standard(BLOCK, 64, 4);
    }

    #[test]
    fn test_roundtrip_4bit_partial_block() {
        assert_roundtrip_standard(BLOCK + 5, 64, 4);
    }

    #[test]
    fn test_roundtrip_4bit_multi_block() {
        assert_roundtrip_standard(BLOCK * 3 + 7, 128, 4);
    }

    #[test]
    fn test_roundtrip_2bit() {
        assert_roundtrip_standard(BLOCK * 2 + 3, 64, 2);
    }

    #[test]
    fn test_roundtrip_2bit_large_dim() {
        assert_roundtrip_standard(50, 256, 2);
    }

    #[test]
    fn test_roundtrip_3bit() {
        let n = BLOCK + 11;
        let dim = 64;
        let bits = 3;
        let q = build_quantiser(n, dim, bits);
        let layout = repack(&q.storage).unwrap();
        let blocked = match layout {
            BlockedLayout::ThreeBit(b) => b,
            _ => panic!("expected ThreeBit layout"),
        };
        assert_eq!(blocked.n_blocks, n.div_ceil(BLOCK));

        let bytes_per_plane = dim / 8;
        let bytes_per_row = 3 * bytes_per_plane;

        for i in 0..n {
            for d in 0..dim {
                let from_bp = decode_code_from_bitplane(&q.storage.packed_codes, i, d, bits, dim);

                // Lower 2 bits via the standard nibble decoder applied
                // to a 2-bit-shaped layout.
                let lower2 = decode_code_from_blocked_standard(&blocked.sub_codes, i, d, 2, dim);

                // Bit 2 from plane2: blocked by (block, byte_group, lane).
                let block_idx = i / BLOCK;
                let lane = i % BLOCK;
                let g = d / 8;
                let bit_in_byte = 7 - (d % 8);
                let mask = 1u8 << bit_in_byte;
                let plane2_offset = (block_idx * bytes_per_plane + g) * BLOCK + lane;
                let bit2 = if blocked.plane2[plane2_offset] & mask != 0 {
                    1u8
                } else {
                    0
                };

                let reconstructed = (bit2 << 2) | lower2;

                // Sanity: bit-plane decoder should give the same answer
                // when we read 3 planes directly.
                let mut direct = 0u8;
                for p in 0..3 {
                    let pb =
                        q.storage.packed_codes[i * bytes_per_row + p * bytes_per_plane + d / 8];
                    if pb & (1u8 << (7 - (d % 8))) != 0 {
                        direct |= 1 << p;
                    }
                }
                assert_eq!(direct, from_bp);

                assert_eq!(
                    reconstructed, from_bp,
                    "vec {i} dim {d}: reconstructed {reconstructed} != bit-plane {from_bp}"
                );
            }
        }
    }

    #[test]
    fn test_blocked_size_4bit() {
        let n = 100;
        let dim = 64;
        let q = build_quantiser(n, dim, 4);
        let layout = repack(&q.storage).unwrap();
        if let BlockedLayout::Standard(b) = layout {
            let n_blocks = n.div_ceil(BLOCK);
            let n_byte_groups = dim / 2;
            assert_eq!(b.data.len(), n_blocks * n_byte_groups * BLOCK);
        } else {
            panic!("expected Standard");
        }
    }

    #[test]
    fn test_blocked_size_2bit() {
        let n = 50;
        let dim = 64;
        let q = build_quantiser(n, dim, 2);
        let layout = repack(&q.storage).unwrap();
        if let BlockedLayout::Standard(b) = layout {
            let n_blocks = n.div_ceil(BLOCK);
            let n_byte_groups = dim / 4;
            assert_eq!(b.data.len(), n_blocks * n_byte_groups * BLOCK);
        } else {
            panic!("expected Standard");
        }
    }

    #[test]
    fn test_padding_zeros() {
        // Last block has padding lanes; those must decode to code 0
        // (zeroed bytes), not garbage.
        let n = BLOCK + 1; // 1 used + 31 padded in second block
        let dim = 64;
        let q = build_quantiser(n, dim, 4);
        let layout = repack(&q.storage).unwrap();
        if let BlockedLayout::Standard(b) = layout {
            for lane in 1..BLOCK {
                for d in 0..dim {
                    let code = decode_code_from_blocked_standard(&b.data, BLOCK + lane, d, 4, dim);
                    assert_eq!(code, 0, "padded vec {} dim {d} should be 0", BLOCK + lane);
                }
            }
        }
    }
}
