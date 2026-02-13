#![allow(dead_code)]

use num_traits::{Float, FromPrimitive};
use wide::u8x16;

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

//////////
// SIMD //
//////////

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
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "popcnt")]
unsafe fn hamming_avx512(a: &[u8], b: &[u8]) -> u32 {
    let mut count = 0u64;
    let len = a.len();
    let n_chunks = len / 64;

    for i in 0..n_chunks {
        let offset = i * 64;
        let va = _mm512_loadu_si512(a.as_ptr().add(offset) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(offset) as *const __m512i);
        let xor = _mm512_xor_si512(va, vb);

        let xor_words = std::mem::transmute::<__m512i, [u64; 8]>(xor);
        for &word in &xor_words {
            count += _popcnt64(word as i64) as u64;
        }
    }

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
#[target_feature(enable = "avx2", enable = "popcnt")]
unsafe fn hamming_avx2(a: &[u8], b: &[u8]) -> u32 {
    let mut count = 0u64;
    let len = a.len();
    let n_chunks = len / 32;

    for i in 0..n_chunks {
        let offset = i * 32;
        let va = _mm256_loadu_si256(a.as_ptr().add(offset) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);
        let xor = _mm256_xor_si256(va, vb);

        let xor_words = std::mem::transmute::<__m256i, [u64; 4]>(xor);
        for &word in &xor_words {
            count += _popcnt64(word as i64) as u64;
        }
    }

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
    let mut count = 0u64;
    let len = a.len();
    let n_chunks = len / 16;

    for i in 0..n_chunks {
        let offset = i * 16;
        let va = _mm_loadu_si128(a.as_ptr().add(offset) as *const __m128i);
        let vb = _mm_loadu_si128(b.as_ptr().add(offset) as *const __m128i);
        let xor = _mm_xor_si128(va, vb);

        let xor_words = std::mem::transmute::<__m128i, [u64; 2]>(xor);
        for &word in &xor_words {
            count += _popcnt64(word as i64) as u64;
        }
    }

    for i in (n_chunks * 16)..len {
        count += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones() as u64;
    }

    count as u32
}

/// Hamming distance for NEON
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
    let mut count = 0u32;
    let len = a.len();
    let n_chunks = len / 16;

    let mut sum = vdupq_n_u64(0);

    for i in 0..n_chunks {
        let offset = i * 16;
        let va = vld1q_u8(a.as_ptr().add(offset));
        let vb = vld1q_u8(b.as_ptr().add(offset));
        let xor = veorq_u8(va, vb);
        let popcnt = vcntq_u8(xor);
        sum = vpadalq_u32(sum, vpaddlq_u16(vpaddlq_u8(popcnt)));
    }

    count += vgetq_lane_u64(sum, 0) as u32;
    count += vgetq_lane_u64(sum, 1) as u32;

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
unsafe fn hamming_simd(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
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
        hamming_scalar(a, b)
    }
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
    T: Float + FromPrimitive + SimdDistance,
{
    assert_eq!(query_vec.len(), dim);

    let mut unpacked = Vec::with_capacity(dim);

    for bit_idx in 0..dim {
        let byte_idx = bit_idx / 8;
        let bit_pos = bit_idx % 8;
        let bit = (binary_code[byte_idx] >> bit_pos) & 1;

        // Transform: bit ∈ {0,1} → value ∈ {-1,+1}
        let val = T::from_f64(2.0 * bit as f64 - 1.0).unwrap();
        unpacked.push(val);
    }

    T::dot_simd(query_vec, &unpacked)
}

//////////////////////////
// VectorDistanceRaBitQ //
//////////////////////////

//////////
// SIMD //
//////////

const BIT_MASKS_16: u8x16 = u8x16::new([1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128]);
const ONE_16: u8x16 = u8x16::new([1; 16]);

/// Horizontal sum of 16-bit vector
///
/// ### Params
///
/// * `v`: The vector to sum
///
/// ### Returns
///
/// The sum of the vector elements
#[inline(always)]
fn horizontal_sum_16(v: u8x16) -> u32 {
    let a: [u8; 16] = v.into();
    (a[0] as u32 + a[1] as u32 + a[2] as u32 + a[3] as u32)
        + (a[4] as u32 + a[5] as u32 + a[6] as u32 + a[7] as u32)
        + (a[8] as u32 + a[9] as u32 + a[10] as u32 + a[11] as u32)
        + (a[12] as u32 + a[13] as u32 + a[14] as u32 + a[15] as u32)
}

/// Dot product - 16-lane SIMD version
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
#[inline]
pub fn dot_query_binary_simd(query: &[u8], binary: &[u8], dim: usize) -> u32 {
    let full_bytes = dim / 8;
    let chunks = full_bytes / 2;
    let mut acc = u8x16::ZERO;

    for chunk_idx in 0..chunks {
        let byte_offset = chunk_idx * 2;
        let dim_offset = byte_offset * 8;

        let query_vals =
            u8x16::new(<[u8; 16]>::try_from(&query[dim_offset..dim_offset + 16]).unwrap());

        let b0 = binary[byte_offset];
        let b1 = binary[byte_offset + 1];
        let binary_broadcast = u8x16::new([
            b0, b0, b0, b0, b0, b0, b0, b0, b1, b1, b1, b1, b1, b1, b1, b1,
        ]);

        // anded is 0 or a power-of-2 (1,2,4,8,16,32,64,128)
        let anded = binary_broadcast & BIT_MASKS_16;

        // min(0, 1) = 0, min(any_power_of_2, 1) = 1
        let zero_or_one = anded.min(ONE_16);

        // 0 - 0 = 0x00, 0 - 1 = 0xFF (wrapping)
        let mask = u8x16::ZERO - zero_or_one;

        acc += query_vals & mask;
    }

    let mut sum = horizontal_sum_16(acc);

    // remaining full bytes
    for byte_idx in (chunks * 2)..full_bytes {
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

    // remaining bits
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
        let dim = self.dim();

        if dim >= 16 {
            dot_query_binary_simd(&query.quantised, binary, dim)
        } else {
            dot_query_binary_scalar(&query.quantised, binary, dim)
        }
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
        let storage = self.storage();
        let packed = storage.get_vector_data(cluster_idx, local_idx); // Single cache line read

        let dim_f = T::from_usize(self.dim()).unwrap();
        let two = T::one() + T::one();

        let v_dist = packed.dist_to_centroid;
        let q_dist = query.dist_to_centroid;
        let dot_corr = packed.dot_correction;

        let qr = T::from_u32(self.dot_query_binary(query, cluster_idx, local_idx)).unwrap();
        let popcount = T::from_u32(packed.popcount).unwrap();
        let sum_q = T::from_u32(query.sum_quantised).unwrap();

        let inner_product_sgn = two * (query.width * qr + query.lower * popcount)
            - (query.width * sum_q + dim_f * query.lower);

        let q_dot_v = if dot_corr > T::from_f32(1e-6).unwrap() {
            (inner_product_sgn / dot_corr).clamp(T::one().neg(), T::one())
        } else {
            T::zero()
        };

        let dist_sq = v_dist * v_dist + q_dist * q_dist - two * v_dist * q_dist * q_dot_v;
        dist_sq.max(T::zero()).sqrt()
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
        let quantiser = RaBitQQuantiser::new(data.as_ref(), &Dist::Euclidean, Some(5), 42);

        assert_eq!(quantiser.dim(), 32);
    }

    #[test]
    fn test_rabitq_trait_n_bytes() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser = RaBitQQuantiser::new(data.as_ref(), &Dist::Euclidean, Some(5), 42);

        assert_eq!(quantiser.n_bytes(), 4);
    }

    #[test]
    fn test_rabitq_popcount() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser = RaBitQQuantiser::new(data.as_ref(), &Dist::Euclidean, Some(5), 42);

        let popcount = quantiser.popcount(0, 0);
        assert!(popcount <= 32);
    }

    #[test]
    fn test_rabitq_dot_query_binary() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser = RaBitQQuantiser::new(data.as_ref(), &Dist::Euclidean, Some(5), 42);

        let query = vec![1.0f32; 32];
        let encoded_query = quantiser.encode_query(&query, 0);

        let dot = quantiser.dot_query_binary(&encoded_query, 0, 0);
        assert!(dot <= 15 * 32);
    }

    #[test]
    fn test_rabitq_dist_positive() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser = RaBitQQuantiser::new(data.as_ref(), &Dist::Euclidean, Some(5), 42);

        let query = vec![1.0f32; 32];
        let encoded_query = quantiser.encode_query(&query, 0);

        let dist = quantiser.rabitq_dist(&encoded_query, 0, 0);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_rabitq_dist_consistency() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser = RaBitQQuantiser::new(data.as_ref(), &Dist::Euclidean, Some(5), 42);

        let query = vec![1.0f32; 32];
        let encoded_query = quantiser.encode_query(&query, 0);

        let dist1 = quantiser.rabitq_dist(&encoded_query, 0, 0);
        let dist2 = quantiser.rabitq_dist(&encoded_query, 0, 0);

        assert_eq!(dist1, dist2);
    }

    #[test]
    fn test_rabitq_dist_different_vectors() {
        let data = create_test_data::<f32>(50, 32);
        let quantiser = RaBitQQuantiser::new(data.as_ref(), &Dist::Euclidean, Some(5), 42);

        let query = vec![1.0f32; 32];
        let encoded_query = quantiser.encode_query(&query, 0);

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
        let quantiser = RaBitQQuantiser::new(data.as_ref(), &Dist::Cosine, Some(5), 42);

        let query = vec![1.0f32; 32];
        let encoded_query = quantiser.encode_query(&query, 0);

        let dist = quantiser.rabitq_dist(&encoded_query, 0, 0);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_rabitq_multiple_clusters() {
        let data = create_test_data::<f32>(100, 32);
        let quantiser = RaBitQQuantiser::new(data.as_ref(), &Dist::Euclidean, Some(10), 42);

        let query = vec![1.0f32; 32];

        for cluster_idx in 0..quantiser.storage().nlist {
            let encoded_query = quantiser.encode_query(&query, cluster_idx);
            let cluster_size = quantiser.storage().cluster_size(cluster_idx);

            for local_idx in 0..cluster_size {
                let dist = quantiser.rabitq_dist(&encoded_query, cluster_idx, local_idx);
                assert!(dist >= 0.0);
            }
        }
    }
}
