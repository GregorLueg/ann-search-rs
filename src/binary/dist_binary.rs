#![allow(dead_code)]

use num_traits::{Float, FromPrimitive};

use crate::binary::rabitq::*;
use crate::utils::dist::*;

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

// AVX-512
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

// AVX2
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

// SSE2
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

// NEON
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

// Scalar fallback
#[inline(always)]
unsafe fn hamming_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

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

//////////////////////////
// VectorDistanceRaBitQ //
//////////////////////////

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
        self.storage().cluster_popcounts(cluster_idx)[local_idx]
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

        let mut sum = 0u32;
        let full_bytes = dim / 8;

        for byte_idx in 0..full_bytes {
            let bits = unsafe { *binary.get_unchecked(byte_idx) };
            let base = byte_idx * 8;

            unsafe {
                sum += *query.quantised.get_unchecked(base) as u32 * (bits & 1) as u32;
                sum += *query.quantised.get_unchecked(base + 1) as u32 * ((bits >> 1) & 1) as u32;
                sum += *query.quantised.get_unchecked(base + 2) as u32 * ((bits >> 2) & 1) as u32;
                sum += *query.quantised.get_unchecked(base + 3) as u32 * ((bits >> 3) & 1) as u32;
                sum += *query.quantised.get_unchecked(base + 4) as u32 * ((bits >> 4) & 1) as u32;
                sum += *query.quantised.get_unchecked(base + 5) as u32 * ((bits >> 5) & 1) as u32;
                sum += *query.quantised.get_unchecked(base + 6) as u32 * ((bits >> 6) & 1) as u32;
                sum += *query.quantised.get_unchecked(base + 7) as u32 * ((bits >> 7) & 1) as u32;
            }
        }

        let remaining = dim % 8;
        if remaining > 0 {
            let bits = unsafe { *binary.get_unchecked(full_bytes) };
            let base = full_bytes * 8;
            unsafe {
                for bit_pos in 0..remaining {
                    sum += *query.quantised.get_unchecked(base + bit_pos) as u32
                        * ((bits >> bit_pos) & 1) as u32;
                }
            }
        }

        sum
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
        let dim_f = self.dim() as f32;

        let v_dist = storage.cluster_dist_to_centroid(cluster_idx)[local_idx]
            .to_f32()
            .unwrap();
        let q_dist = query.dist_to_centroid.to_f32().unwrap();

        let dot_corr = storage.cluster_dot_corrections(cluster_idx)[local_idx]
            .to_f32()
            .unwrap();

        let qr = self.dot_query_binary(query, cluster_idx, local_idx) as f32;
        let popcount = self.popcount(cluster_idx, local_idx) as f32;
        let sum_q = query.sum_quantised as f32;

        let query_width = query.width.to_f32().unwrap();
        let query_lower = query.lower.to_f32().unwrap();

        let inner_product_sgn = 2.0 * (query_width * qr + query_lower * popcount)
            - (query_width * sum_q + dim_f * query_lower);

        let q_dot_v = if dot_corr > 1e-6 {
            (inner_product_sgn / dot_corr).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let dist_sq = v_dist * v_dist + q_dist * q_dist - 2.0 * v_dist * q_dist * q_dot_v;

        T::from_f32(dist_sq.max(0.0).sqrt()).unwrap()
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
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
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
