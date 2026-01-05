use num_traits::{Float, FromPrimitive};

use crate::binary::binariser::*;

////////////////////
// VectorDistance //
////////////////////

/// Trait for computing distances between binarised vectors
pub trait VectorDistanceBinary {
    /// Get the internal flat vector representation (binarised to u8)
    fn vectors_flat_binarised(&self) -> &[u8];

    /// Get the number of bytes(!) used binarisation
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
            vec_i
                .iter()
                .zip(vec_j.iter())
                .map(|(x, y)| (x ^ y).count_ones())
                .sum()
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
            query
                .iter()
                .zip(vec_i.iter())
                .map(|(x, y)| (x ^ y).count_ones())
                .sum()
        }
    }
}

//////////////////////////
// VectorDistanceRaBitQ //
//////////////////////////

/// Trait for RaBitQ distance estimation with multi-centroid support
pub trait VectorDistanceRaBitQ<T>
where
    T: Float + FromPrimitive,
{
    /// Get all clusters
    fn clusters(&self) -> &[RaBitQCluster<T>];

    /// Bytes per binary code
    fn n_bytes(&self) -> usize;

    /// Vector dimensionality
    fn dim(&self) -> usize;

    /// Get binary codes for a cluster
    #[inline]
    fn cluster_binary<'a>(&'a self, cluster_idx: usize) -> &'a [u8]
    where
        T: 'a,
    {
        &self.clusters()[cluster_idx].binary_codes
    }

    /// Get distances to centroid for a cluster
    #[inline]
    fn cluster_dist_to_centroid(&self, cluster_idx: usize) -> &[T] {
        &self.clusters()[cluster_idx].dist_to_centroid
    }

    /// Get dot corrections for a cluster
    #[inline]
    fn cluster_dot_corrections(&self, cluster_idx: usize) -> &[T] {
        &self.clusters()[cluster_idx].dot_corrections
    }

    /// Get original vector indices for a cluster
    #[inline]
    fn cluster_vector_indices<'a>(&'a self, cluster_idx: usize) -> &'a [usize]
    where
        T: 'a,
    {
        &self.clusters()[cluster_idx].vector_indices
    }

    /// Number of vectors in a cluster
    #[inline]
    fn cluster_size(&self, cluster_idx: usize) -> usize {
        self.clusters()[cluster_idx].vector_indices.len()
    }

    /// Count of 1-bits in binary code for vector at local index within cluster
    ///
    /// ### Params
    ///
    /// * `cluster_idx` - Cluster index
    /// * `local_idx` - Local index within cluster
    ///
    /// ### Returns
    ///
    /// Pop count for that vector
    #[inline]
    fn popcount(&self, cluster_idx: usize, local_idx: usize) -> u32 {
        let start = local_idx * self.n_bytes();
        let bytes = &self.cluster_binary(cluster_idx)[start..start + self.n_bytes()];
        bytes.iter().map(|b| b.count_ones()).sum()
    }

    /// Compute sum of query int4 values where binary bit is 1
    ///
    /// ### Params
    ///
    /// * `query` - The encoded query (must be encoded for same cluster)
    /// * `cluster_idx` - Cluster index
    /// * `local_idx` - Local index within cluster
    ///
    /// ### Returns
    ///
    /// Dot product as u32
    #[inline(always)]
    fn dot_query_binary(
        &self,
        query: &RaBitQQuery<T>,
        cluster_idx: usize,
        local_idx: usize,
    ) -> u32 {
        let start = local_idx * self.n_bytes();
        let binary = &self.cluster_binary(cluster_idx)[start..start + self.n_bytes()];
        let dim = self.dim();

        let mut sum = 0u32;
        let full_bytes = dim / 8;

        // Process full bytes - no branching, compiler can auto-vectorize
        for byte_idx in 0..full_bytes {
            let bits = unsafe { *binary.get_unchecked(byte_idx) };
            let base = byte_idx * 8;

            // Branchless: multiply by bit mask (0 or 1)
            // Compiler should vectorize this loop
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

        // Handle remaining bits (if dim % 8 != 0)
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

    /// Estimate distance using RaBitQ formula
    ///
    /// dist(v, q) ≈ sqrt(||v-c||² + ||q-c||² - 2·||v-c||·||q-c||·(q_c · v_c))
    ///
    /// ### Params
    ///
    /// * `query` - Encoded query (must be encoded for same cluster)
    /// * `cluster_idx` - Cluster index
    /// * `local_idx` - Local index within cluster
    ///
    /// ### Returns
    ///
    /// Approximated distance
    #[inline]
    fn rabitq_dist(&self, query: &RaBitQQuery<T>, cluster_idx: usize, local_idx: usize) -> T {
        let dim_f = self.dim() as f32;

        let v_dist = self.cluster_dist_to_centroid(cluster_idx)[local_idx]
            .to_f32()
            .unwrap();
        let q_dist = query.dist_to_centroid.to_f32().unwrap();

        let dot_corr = self.cluster_dot_corrections(cluster_idx)[local_idx]
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

    struct TestRaBitQVectors {
        clusters: Vec<RaBitQCluster<f64>>,
        n_bytes: usize,
        dim: usize,
    }

    impl TestRaBitQVectors {
        fn new(clusters: Vec<RaBitQCluster<f64>>, n_bytes: usize, dim: usize) -> Self {
            TestRaBitQVectors {
                clusters,
                n_bytes,
                dim,
            }
        }
    }

    impl VectorDistanceRaBitQ<f64> for TestRaBitQVectors {
        fn clusters(&self) -> &[RaBitQCluster<f64>] {
            &self.clusters
        }

        fn n_bytes(&self) -> usize {
            self.n_bytes
        }

        fn dim(&self) -> usize {
            self.dim
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

    // RaBitQ tests
    #[test]
    fn test_rabitq_popcount_basic() {
        let mut cluster = RaBitQCluster::new(vec![0.0; 8]);
        cluster.binary_codes = vec![0b00000000, 0b11111111, 0b10101010];
        cluster.vector_indices = vec![0, 1, 2];
        cluster.dist_to_centroid = vec![1.0, 1.0, 1.0];
        cluster.dot_corrections = vec![1.0, 1.0, 1.0];

        let storage = TestRaBitQVectors::new(vec![cluster], 1, 8);

        assert_eq!(storage.popcount(0, 0), 0);
        assert_eq!(storage.popcount(0, 1), 8);
        assert_eq!(storage.popcount(0, 2), 4);
    }

    #[test]
    fn test_rabitq_popcount_multi_byte() {
        let mut cluster = RaBitQCluster::new(vec![0.0; 16]);
        cluster.binary_codes = vec![
            0b11110000, 0b10101010, 0b00001111, 0b01010101, 0b11111111, 0b11111111,
        ];
        cluster.vector_indices = vec![0, 1, 2];
        cluster.dist_to_centroid = vec![1.0, 1.0, 1.0];
        cluster.dot_corrections = vec![1.0, 1.0, 1.0];

        let storage = TestRaBitQVectors::new(vec![cluster], 2, 16);

        assert_eq!(storage.popcount(0, 0), 8);
        assert_eq!(storage.popcount(0, 1), 8);
        assert_eq!(storage.popcount(0, 2), 16);
    }

    #[test]
    fn test_rabitq_dot_query_binary() {
        let dim = 8;
        let mut cluster = RaBitQCluster::new(vec![0.0; dim]);
        cluster.binary_codes = vec![0b11110000];
        cluster.vector_indices = vec![0];
        cluster.dist_to_centroid = vec![1.0];
        cluster.dot_corrections = vec![1.0];

        let storage = TestRaBitQVectors::new(vec![cluster], 1, dim);

        let query = RaBitQQuery {
            quantised: vec![1, 2, 3, 4, 5, 6, 7, 8],
            dist_to_centroid: 1.0,
            lower: 0.0,
            width: 1.0,
            sum_quantised: 36,
        };

        // bits set: positions 4,5,6,7
        // sum = 5 + 6 + 7 + 8 = 26
        let result = storage.dot_query_binary(&query, 0, 0);
        assert_eq!(result, 26);
    }

    #[test]
    fn test_rabitq_dot_query_all_ones() {
        let dim = 8;
        let mut cluster = RaBitQCluster::new(vec![0.0; dim]);
        cluster.binary_codes = vec![0b11111111];
        cluster.vector_indices = vec![0];
        cluster.dist_to_centroid = vec![1.0];
        cluster.dot_corrections = vec![1.0];

        let storage = TestRaBitQVectors::new(vec![cluster], 1, dim);

        let query = RaBitQQuery {
            quantised: vec![1, 2, 3, 4, 5, 6, 7, 8],
            dist_to_centroid: 1.0,
            lower: 0.0,
            width: 1.0,
            sum_quantised: 36,
        };

        let result = storage.dot_query_binary(&query, 0, 0);
        assert_eq!(result, 36);
    }

    #[test]
    fn test_rabitq_dot_query_all_zeros() {
        let dim = 8;
        let mut cluster = RaBitQCluster::new(vec![0.0; dim]);
        cluster.binary_codes = vec![0b00000000];
        cluster.vector_indices = vec![0];
        cluster.dist_to_centroid = vec![1.0];
        cluster.dot_corrections = vec![1.0];

        let storage = TestRaBitQVectors::new(vec![cluster], 1, dim);

        let query = RaBitQQuery {
            quantised: vec![1, 2, 3, 4, 5, 6, 7, 8],
            dist_to_centroid: 1.0,
            lower: 0.0,
            width: 1.0,
            sum_quantised: 36,
        };

        let result = storage.dot_query_binary(&query, 0, 0);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_rabitq_dot_query_non_aligned_dim() {
        let dim = 10;
        let mut cluster = RaBitQCluster::new(vec![0.0; dim]);
        cluster.binary_codes = vec![0b11111111, 0b00000011];
        cluster.vector_indices = vec![0];
        cluster.dist_to_centroid = vec![1.0];
        cluster.dot_corrections = vec![1.0];

        let storage = TestRaBitQVectors::new(vec![cluster], 2, dim);

        let query = RaBitQQuery {
            quantised: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            dist_to_centroid: 1.0,
            lower: 0.0,
            width: 1.0,
            sum_quantised: 55,
        };

        // First byte: all 8 bits set = 1+2+3+4+5+6+7+8 = 36
        // Second byte: bits 0,1 set = 9+10 = 19
        let result = storage.dot_query_binary(&query, 0, 0);
        assert_eq!(result, 55);
    }

    #[test]
    fn test_rabitq_dist_identical() {
        let dim = 8;
        let mut cluster = RaBitQCluster::new(vec![0.0; dim]);
        cluster.binary_codes = vec![0b11110000];
        cluster.vector_indices = vec![0];
        cluster.dist_to_centroid = vec![1.0];
        cluster.dot_corrections = vec![8.0];

        let storage = TestRaBitQVectors::new(vec![cluster], 1, dim);

        let query = RaBitQQuery {
            quantised: vec![8, 8, 8, 8, 8, 8, 8, 8],
            dist_to_centroid: 1.0,
            lower: 0.0,
            width: 1.0,
            sum_quantised: 64,
        };

        let dist = storage.rabitq_dist(&query, 0, 0);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_rabitq_dist_zero_correction() {
        let dim = 8;
        let mut cluster = RaBitQCluster::new(vec![0.0; dim]);
        cluster.binary_codes = vec![0b00000000];
        cluster.vector_indices = vec![0];
        cluster.dist_to_centroid = vec![2.0];
        cluster.dot_corrections = vec![0.0];

        let storage = TestRaBitQVectors::new(vec![cluster], 1, dim);

        let query = RaBitQQuery {
            quantised: vec![5; 8],
            dist_to_centroid: 3.0,
            lower: 0.0,
            width: 1.0,
            sum_quantised: 40,
        };

        let dist = storage.rabitq_dist(&query, 0, 0);
        // With zero dot_corr, q_dot_v should be 0
        // dist = sqrt(4 + 9 - 0) = sqrt(13)
        assert!((dist - 13.0_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_rabitq_cluster_accessors() {
        let dim = 8;
        let mut cluster = RaBitQCluster::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        cluster.binary_codes = vec![0b11110000, 0b00001111];
        cluster.vector_indices = vec![10, 20];
        cluster.dist_to_centroid = vec![1.5, 2.5];
        cluster.dot_corrections = vec![0.8, 0.9];

        let storage = TestRaBitQVectors::new(vec![cluster], 1, dim);

        assert_eq!(storage.cluster_size(0), 2);
        assert_eq!(storage.cluster_vector_indices(0), &[10, 20]);
        assert_eq!(storage.cluster_dist_to_centroid(0), &[1.5, 2.5]);
        assert_eq!(storage.cluster_dot_corrections(0), &[0.8, 0.9]);
        assert_eq!(storage.cluster_binary(0).len(), 2);
    }

    #[test]
    fn test_rabitq_multi_cluster() {
        let dim = 8;
        let mut cluster1 = RaBitQCluster::new(vec![0.0; dim]);
        cluster1.binary_codes = vec![0b11111111];
        cluster1.vector_indices = vec![0];
        cluster1.dist_to_centroid = vec![1.0];
        cluster1.dot_corrections = vec![1.0];

        let mut cluster2 = RaBitQCluster::new(vec![1.0; dim]);
        cluster2.binary_codes = vec![0b00000000];
        cluster2.vector_indices = vec![1];
        cluster2.dist_to_centroid = vec![2.0];
        cluster2.dot_corrections = vec![2.0];

        let storage = TestRaBitQVectors::new(vec![cluster1, cluster2], 1, dim);

        assert_eq!(storage.clusters().len(), 2);
        assert_eq!(storage.cluster_size(0), 1);
        assert_eq!(storage.cluster_size(1), 1);
        assert_eq!(storage.popcount(0, 0), 8);
        assert_eq!(storage.popcount(1, 0), 0);
    }
}
