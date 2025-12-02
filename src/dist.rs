use num_traits::Float;

/// Trait for types that can compute distances between vectors
pub trait VectorDistance<T: Float> {
    fn vectors_flat(&self) -> &[T];
    fn dim(&self) -> usize;
    fn norms(&self) -> &[T];

    /// Euclidean distance between two internal vectors (squared)
    ///
    /// ### Implementation note
    ///
    /// Uses iterator-based approach which allows LLVM to auto-vectorise
    /// optimally for the target CPU. Returns squared distance to avoid
    /// expensive sqrt - sufficient for comparison purposes.
    ///
    /// ### Params
    ///
    /// * `i` - Sample index i
    /// * `j` - Sample index j
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance between the two samples
    #[inline(always)]
    fn euclidean_distance(&self, i: usize, j: usize) -> T {
        let start_i = i * self.dim();
        let start_j = j * self.dim();
        let vec_i = &self.vectors_flat()[start_i..start_i + self.dim()];
        let vec_j = &self.vectors_flat()[start_j..start_j + self.dim()];

        vec_i
            .iter()
            .zip(vec_j.iter())
            .map(|(&a, &b)| {
                let diff = a - b;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x)
    }

    /// Euclidean distance between query vector and internal vector (squared)
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query` - Query vector slice
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance
    #[inline(always)]
    fn euclidean_distance_to_query(&self, internal_idx: usize, query: &[T]) -> T {
        let start = internal_idx * self.dim();
        let vec = &self.vectors_flat()[start..start + self.dim()];

        vec.iter()
            .zip(query.iter())
            .map(|(&a, &b)| {
                let diff = a - b;
                diff * diff
            })
            .fold(T::zero(), |acc, x| acc + x)
    }

    /// Cosine distance between two internal vectors
    ///
    /// ### Cosine Distance
    ///
    /// cosine_dist(u, v) = 1 - (u·v) / (||u|| ||v||)
    /// We pre-compute norms during initialisation to avoid repeated sqrt calls.
    ///
    /// ### Params
    ///
    /// * `i` - Sample index i
    /// * `j` - Sample index j
    ///
    /// ### Returns
    ///
    /// The Cosine distance between the two samples
    #[inline(always)]
    fn cosine_distance(&self, i: usize, j: usize) -> T {
        let start_i = i * self.dim();
        let start_j = j * self.dim();
        let vec_i = &self.vectors_flat()[start_i..start_i + self.dim()];
        let vec_j = &self.vectors_flat()[start_j..start_j + self.dim()];

        let dot = vec_i
            .iter()
            .zip(vec_j.iter())
            .map(|(&a, &b)| a * b)
            .fold(T::zero(), |acc, x| acc + x);

        T::one() - (dot / (self.norms()[i] * self.norms()[j]))
    }

    /// Cosine distance between query vector and internal vector
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query` - Query vector slice
    /// * `query_norm` - Pre-computed norm of query vector
    ///
    /// ### Returns
    ///
    /// The Cosine distance
    #[inline(always)]
    fn cosine_distance_to_query(&self, internal_idx: usize, query: &[T], query_norm: T) -> T {
        let start = internal_idx * self.dim();
        let vec = &self.vectors_flat()[start..start + self.dim()];

        let dot = vec
            .iter()
            .zip(query.iter())
            .map(|(&a, &b)| a * b)
            .fold(T::zero(), |acc, x| acc + x);

        T::one() - (dot / (query_norm * self.norms()[internal_idx]))
    }
}

// Specialised implementations //

#[inline(always)]
pub fn euclidean_dist_f32(vec_i: &[f32], vec_j: &[f32]) -> f32 {
    vec_i
        .iter()
        .zip(vec_j.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum()
}

#[inline(always)]
pub fn euclidean_dist_f64(vec_i: &[f64], vec_j: &[f64]) -> f64 {
    vec_i
        .iter()
        .zip(vec_j.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum()
}

#[inline(always)]
pub fn cosine_dist_f32(vec_i: &[f32], vec_j: &[f32], norm_i: f32, norm_j: f32) -> f32 {
    let dot: f32 = vec_i.iter().zip(vec_j.iter()).map(|(&a, &b)| a * b).sum();
    1.0 - (dot / (norm_i * norm_j))
}

#[inline(always)]
pub fn cosine_dist_f64(vec_i: &[f64], vec_j: &[f64], norm_i: f64, norm_j: f64) -> f64 {
    let dot: f64 = vec_i.iter().zip(vec_j.iter()).map(|(&a, &b)| a * b).sum();
    1.0 - (dot / (norm_i * norm_j))
}

#[inline(always)]
pub fn norm_f32(vec: &[f32]) -> f32 {
    vec.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

#[inline(always)]
pub fn norm_f64(vec: &[f64]) -> f64 {
    vec.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    struct TestVectors {
        data: Vec<f32>,
        dim: usize,
        norms: Vec<f32>,
    }

    impl VectorDistance<f32> for TestVectors {
        fn vectors_flat(&self) -> &[f32] {
            &self.data
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn norms(&self) -> &[f32] {
            &self.norms
        }
    }

    #[test]
    fn test_euclidean_distance_basic() {
        let data = vec![
            1.0, 0.0, 0.0, // Vector 0: [1, 0, 0]
            0.0, 1.0, 0.0, // Vector 1: [0, 1, 0]
            1.0, 1.0, 0.0, // Vector 2: [1, 1, 0]
        ];

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![],
        };

        // Distance between [1,0,0] and [0,1,0] should be sqrt(2) squared = 2
        let dist_01 = vecs.euclidean_distance(0, 1);
        assert_relative_eq!(dist_01, 2.0, epsilon = 1e-6);

        // Distance between [1,0,0] and [1,1,0] should be 1
        let dist_02 = vecs.euclidean_distance(0, 2);
        assert_relative_eq!(dist_02, 1.0, epsilon = 1e-6);

        // Distance to itself should be 0
        let dist_00 = vecs.euclidean_distance(0, 0);
        assert_relative_eq!(dist_00, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_euclidean_distance_symmetry() {
        let data = vec![2.0, 3.0, 5.0, 1.0, 4.0, 2.0];

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![],
        };

        let dist_01 = vecs.euclidean_distance(0, 1);
        let dist_10 = vecs.euclidean_distance(1, 0);

        assert_relative_eq!(dist_01, dist_10, epsilon = 1e-6);
    }

    #[test]
    fn test_euclidean_distance_unrolled() {
        // Test with dimension not divisible by 4 to test both unrolled and remainder loops
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, // 5 dimensions
            5.0, 4.0, 3.0, 2.0, 1.0,
        ];

        let vecs = TestVectors {
            data,
            dim: 5,
            norms: vec![],
        };

        let dist = vecs.euclidean_distance(0, 1);
        // Expected: (1-5)^2 + (2-4)^2 + (3-3)^2 + (4-2)^2 + (5-1)^2 = 16 + 4 + 0 + 4 + 16 = 40
        assert_relative_eq!(dist, 40.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance_basic() {
        let data = vec![
            1.0, 0.0, 0.0, // Vector 0
            0.0, 1.0, 0.0, // Vector 1
            1.0, 1.0, 0.0, // Vector 2 (45 degrees from both)
        ];

        // Pre-compute norms
        let norm0 = (1.0_f32 * 1.0 + 0.0 * 0.0 + 0.0 * 0.0).sqrt();
        let norm1 = (0.0_f32 * 0.0 + 1.0 * 1.0 + 0.0 * 0.0).sqrt();
        let norm2 = (1.0_f32 * 1.0 + 1.0 * 1.0 + 0.0 * 0.0).sqrt();

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![norm0, norm1, norm2],
        };

        // Orthogonal vectors: cosine similarity = 0, distance = 1
        let dist_01 = vecs.cosine_distance(0, 1);
        assert_relative_eq!(dist_01, 1.0, epsilon = 1e-6);

        // 45 degree angle: cosine similarity = 1/sqrt(2), distance = 1 - 1/sqrt(2)
        let dist_02 = vecs.cosine_distance(0, 2);
        assert_relative_eq!(dist_02, 1.0 - 1.0 / 2.0_f32.sqrt(), epsilon = 1e-5);

        // Same vector: cosine similarity = 1, distance = 0
        let dist_00 = vecs.cosine_distance(0, 0);
        assert_relative_eq!(dist_00, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance_symmetry() {
        let data = vec![2.0, 3.0, 5.0, 1.0, 4.0, 2.0];

        let norm0 = (2.0_f32 * 2.0 + 3.0 * 3.0 + 5.0 * 5.0).sqrt();
        let norm1 = (1.0_f32 * 1.0 + 4.0 * 4.0 + 2.0 * 2.0).sqrt();

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![norm0, norm1],
        };

        let dist_01 = vecs.cosine_distance(0, 1);
        let dist_10 = vecs.cosine_distance(1, 0);

        assert_relative_eq!(dist_01, dist_10, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance_unrolled() {
        // Test with dimension not divisible by 4
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let norm0 = (1.0_f32 + 4.0 + 9.0 + 16.0 + 25.0).sqrt();
        let norm1 = (25.0_f32 + 16.0 + 9.0 + 4.0 + 1.0).sqrt();

        let vecs = TestVectors {
            data,
            dim: 5,
            norms: vec![norm0, norm1],
        };

        // Dot product: 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5 + 8 + 9 + 8 + 5 = 35
        // Cosine similarity: 35 / (norm0 * norm1)
        let dist = vecs.cosine_distance(0, 1);
        let expected = 1.0 - (35.0 / (norm0 * norm1));
        assert_relative_eq!(dist, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_parallel_vectors() {
        let data = vec![
            1.0, 2.0, 3.0, 2.0, 4.0, 6.0, // Parallel to first (scaled by 2)
        ];

        let norm0 = (1.0_f32 + 4.0 + 9.0).sqrt();
        let norm1 = (4.0_f32 + 16.0 + 36.0).sqrt();

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![norm0, norm1],
        };

        // Parallel vectors should have cosine distance ≈ 0
        let dist = vecs.cosine_distance(0, 1);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_opposite_vectors() {
        let data = vec![
            1.0, 2.0, 3.0, -1.0, -2.0, -3.0, // Opposite direction
        ];

        let norm0 = (1.0_f32 + 4.0 + 9.0).sqrt();
        let norm1 = (1.0_f32 + 4.0 + 9.0).sqrt();

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![norm0, norm1],
        };

        // Opposite vectors should have cosine distance ≈ 2
        let dist = vecs.cosine_distance(0, 1);
        assert_relative_eq!(dist, 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_large_dimension() {
        // Test with larger dimension to stress the unrolling
        let dim = 100;
        let mut data = Vec::with_capacity(dim * 2);

        for i in 0..dim {
            data.push(i as f32);
        }
        for i in 0..dim {
            data.push((dim - i) as f32);
        }

        let norm0 = data[0..dim].iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm1 = data[dim..].iter().map(|x| x * x).sum::<f32>().sqrt();

        let vecs = TestVectors {
            data,
            dim,
            norms: vec![norm0, norm1],
        };

        // Just verify it computes without crashing and is symmetric
        let dist_01 = vecs.euclidean_distance(0, 1);
        let dist_10 = vecs.euclidean_distance(1, 0);
        assert_relative_eq!(dist_01, dist_10, epsilon = 1e-3);

        let cos_01 = vecs.cosine_distance(0, 1);
        let cos_10 = vecs.cosine_distance(1, 0);
        assert_relative_eq!(cos_01, cos_10, epsilon = 1e-5);
    }
}
