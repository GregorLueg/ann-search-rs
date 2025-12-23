use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::iter::Sum;

////////////
// Helper //
////////////

/// Enum for the Distance metric to use
#[derive(Clone, Debug, Copy, PartialEq, Default)]
pub enum Dist {
    /// Euclidean distance
    #[default]
    Euclidean,
    /// Cosine distance
    Cosine,
}

/// Parsing the approximate nearest neighbour distance
///
/// Currently, only Cosine and Euclidean are supported. Longer term, others
/// shall be implemented.
///
/// ### Params
///
/// * `s` - The string that defines the tied summarisation type
///
/// ### Results
///
/// The `Dist` defining the distance metric to use for the approximate
/// neighbour search.
pub fn parse_ann_dist(s: &str) -> Option<Dist> {
    match s.to_lowercase().as_str() {
        "euclidean" => Some(Dist::Euclidean),
        "cosine" => Some(Dist::Cosine),
        _ => None,
    }
}

////////////////////
// VectorDistance //
////////////////////

/// Trait for computing distances between Floats
pub trait VectorDistance<T>
where
    T: Float + Sum,
{
    /// Get the internal flat vector representation
    fn vectors_flat(&self) -> &[T];

    /// Get the internal dimensions
    fn dim(&self) -> usize;

    /// Get the normalised values
    fn norms(&self) -> &[T];

    ///////////////
    // Euclidean //
    ///////////////

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
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance between the two samples
    #[inline(always)]
    fn euclidean_distance(&self, i: usize, j: usize) -> T {
        let start_i = i * self.dim();
        let start_j = j * self.dim();
        unsafe {
            let vec_i = self
                .vectors_flat()
                .get_unchecked(start_i..start_i + self.dim());
            let vec_j = self
                .vectors_flat()
                .get_unchecked(start_j..start_j + self.dim());
            vec_i
                .iter()
                .zip(vec_j.iter())
                .map(|(&a, &b)| {
                    let diff = a - b;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x)
        }
    }

    /// Euclidean distance between query vector and internal vector (squared)
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query` - Query vector slice
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance
    #[inline(always)]
    fn euclidean_distance_to_query(&self, internal_idx: usize, query: &[T]) -> T {
        let start = internal_idx * self.dim();

        unsafe {
            let vec = &self.vectors_flat().get_unchecked(start..start + self.dim());
            vec.iter()
                .zip(query.iter())
                .map(|(&a, &b)| {
                    let diff = a - b;
                    diff * diff
                })
                .fold(T::zero(), |acc, x| acc + x)
        }
    }

    ////////////
    // Cosine //
    ////////////

    /// Cosine distance between two internal vectors
    ///
    /// Uses pre-computed norms.
    ///
    /// ### Params
    ///
    /// * `i` - Sample index i
    /// * `j` - Sample index j
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The Cosine distance between the two samples
    #[inline(always)]
    fn cosine_distance(&self, i: usize, j: usize) -> T {
        let start_i = i * self.dim();
        let start_j = j * self.dim();
        unsafe {
            let vec_i = &self
                .vectors_flat()
                .get_unchecked(start_i..start_i + self.dim());
            let vec_j = &self
                .vectors_flat()
                .get_unchecked(start_j..start_j + self.dim());

            let dot = vec_i
                .iter()
                .zip(vec_j.iter())
                .map(|(&a, &b)| a * b)
                .fold(T::zero(), |acc, x| acc + x);

            T::one() - (dot / (self.norms()[i] * self.norms()[j]))
        }
    }

    /// Cosine distance between query vector and internal vector
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query` - Query vector slice
    /// * `query_norm` - Pre-computed norm of query vector
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The Cosine distance
    #[inline(always)]
    fn cosine_distance_to_query(&self, internal_idx: usize, query: &[T], query_norm: T) -> T {
        let start = internal_idx * self.dim();

        unsafe {
            let vec = &self.vectors_flat().get_unchecked(start..start + self.dim());

            let dot = vec
                .iter()
                .zip(query.iter())
                .map(|(&a, &b)| a * b)
                .fold(T::zero(), |acc, x| acc + x);

            T::one() - (dot / (query_norm * self.norms()[internal_idx]))
        }
    }
}

///////////////////////
// VectorDistanceSq8 //
///////////////////////

/// Trait for computing distances between `i8`
pub trait VectorDistanceSq8<T>
where
    T: Float + FromPrimitive + ToPrimitive,
{
    /// Get the internal flat vector representation (quantised to i8)
    fn vectors_flat_quantised(&self) -> &[i8];

    /// Get the internal norms of the quantised vectors
    fn norms_quantised(&self) -> &[i32];

    /// Get the internal dimensions
    fn dim(&self) -> usize;

    ///////////////
    // Euclidean //
    ///////////////

    /// Calculate euclidean distance against quantised query
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query_i8` - Query vector slice quantised to i8
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance
    #[inline(always)]
    fn euclidean_distance_i8(&self, internal_idx: usize, query_i8: &[i8]) -> T {
        let start = internal_idx * self.dim();
        unsafe {
            let db_vec = &self
                .vectors_flat_quantised()
                .get_unchecked(start..start + self.dim());

            let sum: i32 = query_i8
                .iter()
                .zip(db_vec.iter())
                .map(|(&q, &d)| {
                    let diff = q as i32 - d as i32;
                    diff * diff
                })
                .sum();

            T::from_i32(sum).unwrap()
        }
    }

    /// Calculate cosine distance against quantised query
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query_i8` - Query vector slice quantised to i8
    /// * `query_norm_sq` - Squared norm of the query vector
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance
    #[inline(always)]
    fn cosine_distance_i8(&self, vec_idx: usize, query_i8: &[i8], query_norm_sq: i32) -> T {
        let start = vec_idx * self.dim();

        unsafe {
            let db_vec = &self
                .vectors_flat_quantised()
                .get_unchecked(start..start + self.dim());

            let dot: i32 = query_i8
                .iter()
                .zip(db_vec.iter())
                .map(|(&q, &d)| q as i32 * d as i32)
                .sum();

            let db_norm_sq: i32 = self.norms_quantised()[vec_idx];

            let query_norm = T::from_i32(query_norm_sq).unwrap().sqrt();
            let db_norm = T::from_i32(db_norm_sq).unwrap().sqrt();

            if query_norm > T::zero() && db_norm > T::zero() {
                T::one() - T::from_i32(dot).unwrap() / (query_norm * db_norm)
            } else {
                T::one()
            }
        }
    }
}

///////////////
// Functions //
///////////////

/// Static Euclidean distance between two arbitrary vectors (squared)
///
/// ### Params
///
/// * `a` - Slice of vector one
/// * `b` - Slice of vector two
///
/// ### Returns
///
/// Squared euclidean distance
#[inline(always)]
pub fn euclidean_distance_static<T>(a: &[T], b: &[T]) -> T
where
    T: Float,
{
    assert!(a.len() == b.len(), "Vectors a and b need to have same len!");

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .fold(T::zero(), |acc, x| acc + x)
}

/// Static Cosine distance between two arbitrary vectors
///
/// Computes norms on the fly
///
/// ### Params
///
/// * `a` - Slice of vector one
/// * `b` - Slice of vector two
///
/// ### Returns
///
/// Squared cosine distance
#[inline(always)]
pub fn cosine_distance_static<T>(a: &[T], b: &[T]) -> T
where
    T: Float,
{
    assert!(a.len() == b.len(), "Vectors a and b need to have same len!");

    let dot: T = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .fold(T::zero(), |acc, x| acc + x);

    let norm_a = a
        .iter()
        .map(|&x| x * x)
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt();

    let norm_b = b
        .iter()
        .map(|&x| x * x)
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt();

    T::one() - (dot / (norm_a * norm_b))
}

/// Helper to normalise vector in place
///
/// ### Params
///
/// * `vec` - The vector to normalise
#[inline]
pub fn normalise_vector<T: Float + Sum>(vec: &mut [T]) {
    let norm = vec
        .iter()
        .map(|&v| v * v)
        .fold(T::zero(), |acc, x| acc + x)
        .sqrt();
    if norm > T::zero() {
        vec.iter_mut().for_each(|v| *v = *v / norm);
    }
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
    fn test_parse_ann_dist_euclidean() {
        assert_eq!(parse_ann_dist("euclidean"), Some(Dist::Euclidean));
        assert_eq!(parse_ann_dist("Euclidean"), Some(Dist::Euclidean));
        assert_eq!(parse_ann_dist("EUCLIDEAN"), Some(Dist::Euclidean));
    }

    #[test]
    fn test_parse_ann_dist_cosine() {
        assert_eq!(parse_ann_dist("cosine"), Some(Dist::Cosine));
        assert_eq!(parse_ann_dist("Cosine"), Some(Dist::Cosine));
        assert_eq!(parse_ann_dist("COSINE"), Some(Dist::Cosine));
    }

    #[test]
    fn test_parse_ann_dist_invalid() {
        assert_eq!(parse_ann_dist("manhattan"), None);
        assert_eq!(parse_ann_dist(""), None);
        assert_eq!(parse_ann_dist("cosine "), None); // Trailing space
        assert_eq!(parse_ann_dist(" euclidean"), None); // Leading space
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
