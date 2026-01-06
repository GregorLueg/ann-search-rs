use faer::RowRef;
use num_traits::Float;
use std::iter::Sum;

#[cfg(feature = "quantised")]
use half::*;
#[cfg(feature = "quantised")]
use num_traits::{FromPrimitive, ToPrimitive};

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

#[cfg(feature = "quantised")]
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

////////////////////////
// VectorDistanceBf16 //
////////////////////////

#[cfg(feature = "quantised")]
/// Trait for computing distances between Floats
pub trait VectorDistanceBf16<T>
where
    T: Float + Sum + FromPrimitive + ToPrimitive,
{
    /// Get the internal flat vector representation
    fn vectors_flat(&self) -> &[bf16];

    /// Get the internal dimensions
    fn dim(&self) -> usize;

    /// Get the normalised values
    fn norms(&self) -> &[T];

    ///////////////
    // Euclidean //
    ///////////////

    /// Euclidean distance between two internal vectors (squared; bf16)
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
    fn euclidean_distance_bf16(&self, i: usize, j: usize) -> T {
        let start_i = i * self.dim();
        let start_j = j * self.dim();
        unsafe {
            let vec_i = self
                .vectors_flat()
                .get_unchecked(start_i..start_i + self.dim());
            let vec_j = self
                .vectors_flat()
                .get_unchecked(start_j..start_j + self.dim());
            let dist = vec_i
                .iter()
                .zip(vec_j.iter())
                .map(|(&a, &b)| {
                    let diff = a.to_f32() - b.to_f32();
                    diff * diff
                })
                .fold(0.0, |acc, x| acc + x);

            T::from_f32(dist).unwrap()
        }
    }

    /// Euclidean distance between query vector and internal vector
    /// (squared; bf16)
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
    fn euclidean_distance_to_query_bf16(&self, internal_idx: usize, query: &[T]) -> T {
        let start = internal_idx * self.dim();

        unsafe {
            let vec = &self.vectors_flat().get_unchecked(start..start + self.dim());
            let dist = vec
                .iter()
                .zip(query.iter())
                .map(|(&a, &b)| {
                    let diff = a.to_f32() - b.to_f32().unwrap();
                    diff * diff
                })
                .fold(0.0, |acc, x| acc + x);

            T::from_f32(dist).unwrap()
        }
    }

    /// Euclidean distance between query vector and internal vector
    /// (squared; bf16)
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
    fn euclidean_distance_to_query_dual_bf16(&self, internal_idx: usize, query: &[bf16]) -> T {
        let start = internal_idx * self.dim();

        unsafe {
            let vec = &self.vectors_flat().get_unchecked(start..start + self.dim());
            let dist = vec
                .iter()
                .zip(query.iter())
                .map(|(&a, &b)| {
                    let diff = a.to_f32() - b.to_f32();
                    diff * diff
                })
                .fold(0.0, |acc, x| acc + x);

            T::from_f32(dist).unwrap()
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
    fn cosine_distance_bf16(&self, i: usize, j: usize) -> T {
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
                .map(|(&a, &b)| a.to_f32() * b.to_f32())
                .fold(0.0, |acc, x| acc + x);

            let dist = 1.0
                - (dot / (self.norms()[i].to_f32().unwrap() * self.norms()[j].to_f32().unwrap()));

            T::from_f32(dist).unwrap()
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
    fn cosine_distance_to_query_bf16(&self, internal_idx: usize, query: &[T], query_norm: T) -> T {
        let start = internal_idx * self.dim();

        unsafe {
            let vec = &self.vectors_flat().get_unchecked(start..start + self.dim());

            let dot = vec
                .iter()
                .zip(query.iter())
                .map(|(&a, &b)| a.to_f32() * b.to_f32().unwrap())
                .fold(0.0, |acc, x| acc + x);

            let dist = 1.0
                - (dot
                    / (query_norm.to_f32().unwrap()
                        * self.norms()[internal_idx].to_f32().unwrap()));

            T::from_f32(dist).unwrap()
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
    fn cosine_distance_to_query_dual_bf16(
        &self,
        internal_idx: usize,
        query: &[bf16],
        query_norm: bf16,
    ) -> T {
        let start = internal_idx * self.dim();

        unsafe {
            let vec = &self.vectors_flat().get_unchecked(start..start + self.dim());

            let dot = vec
                .iter()
                .zip(query.iter())
                .map(|(&a, &b)| a.to_f32() * b.to_f32())
                .fold(0.0, |acc, x| acc + x);

            let dist =
                1.0 - (dot / (query_norm.to_f32() * self.norms()[internal_idx].to_f32().unwrap()));

            T::from_f32(dist).unwrap()
        }
    }
}

///////////////////////
// VectorDistanceAdc //
///////////////////////

#[cfg(feature = "quantised")]
pub trait VectorDistanceAdc<T>
where
    T: Float + FromPrimitive + ToPrimitive + Sum,
{
    /// Get the m value from the codebook
    fn codebook_m(&self) -> usize;

    /// Get the number of centroids from the codebook
    fn codebook_n_centroids(&self) -> usize;

    /// Get the subvector dimensions from the codebook
    fn codebook_subvec_dim(&self) -> usize;

    /// Get the internal flat centroids representation
    fn centroids(&self) -> &[T];

    /// Get the internal dimensions
    fn dim(&self) -> usize;

    /// Return the codebooks data
    fn codebooks(&self) -> &[Vec<T>];

    /// Get the quantised codes
    fn quantised_codes(&self) -> &[u8];

    /// Build ADC lookup tables for a specific cluster
    ///
    /// ### Params
    ///
    /// * `query` - The query vector
    /// * `cluster_idx`
    ///
    /// ### Returns
    ///
    /// Lookup table as flat Vec<T> of size M * n_centroids
    fn build_lookup_tables(&self, query_vec: &[T], cluster_idx: usize) -> Vec<T> {
        let m = self.codebook_m();
        let subvec_dim = self.codebook_subvec_dim();
        let n_cents = self.codebook_n_centroids();

        let centroid = &self.centroids()[cluster_idx * self.dim()..(cluster_idx + 1) * self.dim()];

        let query_residual: Vec<T> = query_vec
            .iter()
            .zip(centroid.iter())
            .map(|(&q, &c)| q - c)
            .collect();

        let mut table = vec![T::zero(); m * n_cents];

        for subspace in 0..m {
            let query_sub = &query_residual[subspace * subvec_dim..(subspace + 1) * subvec_dim];
            let table_offset = subspace * n_cents;

            for centroid_idx in 0..n_cents {
                let centroid_start = centroid_idx * subvec_dim;
                let pq_centroid =
                    &self.codebooks()[subspace][centroid_start..centroid_start + subvec_dim];

                // squared Euclidean distance for ADC
                let dist: T = query_sub
                    .iter()
                    .zip(pq_centroid.iter())
                    .map(|(&q, &c)| {
                        let diff = q - c;
                        diff * diff
                    })
                    .sum();

                table[table_offset + centroid_idx] = dist;
            }
        }

        table
    }

    /// Compute distance using ADC lookup tables
    ///
    /// Optimised with manual unrolling and unsafe indexing for small m
    ///
    /// ### Params
    ///
    /// * `vec_idx` - Index of database vector
    /// * `lookup_tables` - Precomputed distance table (flat layout)
    ///
    /// ### Returns
    ///
    /// Approximate distance
    #[inline(always)]
    fn compute_distance_adc(&self, vec_idx: usize, lookup_table: &[T]) -> T {
        let m = self.codebook_m();
        let n_cents = self.codebook_n_centroids();
        let codes_start = vec_idx * m;
        let codes = &self.quantised_codes()[codes_start..codes_start + m];

        // manual unrolling for common small m values with unsafe indexing
        match m {
            8 => {
                let mut sum = T::zero();
                for i in 0..8 {
                    let code = unsafe { *codes.get_unchecked(i) } as usize;
                    let offset = i * n_cents + code;
                    sum = sum + unsafe { *lookup_table.get_unchecked(offset) };
                }
                sum
            }
            16 => {
                let mut sum = T::zero();
                for i in 0..16 {
                    let code = unsafe { *codes.get_unchecked(i) } as usize;
                    let offset = i * n_cents + code;
                    sum = sum + unsafe { *lookup_table.get_unchecked(offset) };
                }
                sum
            }
            32 => {
                let mut sum = T::zero();
                for i in 0..32 {
                    let code = unsafe { *codes.get_unchecked(i) } as usize;
                    let offset = i * n_cents + code;
                    sum = sum + unsafe { *lookup_table.get_unchecked(offset) };
                }
                sum
            }
            _ => {
                // Generic fallback for other m values
                codes
                    .iter()
                    .enumerate()
                    .map(|(subspace, &code)| {
                        let offset = subspace * n_cents + (code as usize);
                        lookup_table[offset]
                    })
                    .fold(T::zero(), |acc, x| acc + x)
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

/// Static Cosine distance between two arbitrary vectors
///
/// This version accepts pre-calculated norms
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
pub fn cosine_distance_static_norm<T>(a: &[T], b: &[T], norm_a: &T, norm_b: &T) -> T
where
    T: Float,
{
    assert!(a.len() == b.len(), "Vectors a and b need to have same len!");

    let dot: T = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .fold(T::zero(), |acc, x| acc + x);

    T::one() - (dot / (*norm_a * *norm_b))
}

/// Helper to normalise vector in place
///
/// ### Params
///
/// * `vec` - The vector to normalise
#[inline(always)]
pub fn normalise_vector<T: Float + Sum>(vec: &mut [T]) {
    let norm = compute_norm(vec);
    if norm > T::zero() {
        vec.iter_mut().for_each(|v| *v = *v / norm);
    }
}

/// Compute the L2 norm of a slice
///
/// ### Params
///
/// * `vec` - Slice for which to calculate L2 norm
///
/// ### Returns
///
/// L2 norm
#[inline(always)]
pub fn compute_norm<T>(vec: &[T]) -> T
where
    T: Float,
{
    let mut sum = T::zero();
    for &x in vec {
        sum = sum + x * x;
    }
    sum.sqrt()
}

/// Compute the L2 norm of a row reference
///
/// ### Params
///
/// * `row` - Row for which to calculate L2 norm
///
/// ### Returns
///
/// L2 norm
#[inline(always)]
pub fn compute_norm_row<T>(row: RowRef<T>) -> T
where
    T: Float,
{
    let mut sum = T::zero();
    for i in 0..row.ncols() {
        sum = sum + row[i] * row[i];
    }
    sum.sqrt()
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

    #[test]
    fn test_euclidean_distance_to_query() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![],
        };

        let query = vec![1.0, 1.0, 0.0];

        // Distance from [1,0,0] to [1,1,0] should be 1
        let dist_0 = vecs.euclidean_distance_to_query(0, &query);
        assert_relative_eq!(dist_0, 1.0, epsilon = 1e-6);

        // Distance from [0,1,0] to [1,1,0] should be 1
        let dist_1 = vecs.euclidean_distance_to_query(1, &query);
        assert_relative_eq!(dist_1, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance_to_query() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];

        let norm0 = 1.0;
        let norm1 = 1.0;
        let norm2 = 2.0_f32.sqrt();

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![norm0, norm1, norm2],
        };

        let query = vec![1.0, 1.0, 0.0];
        let query_norm = 2.0_f32.sqrt();

        // Orthogonal: cosine distance should be 1
        let dist_0 = vecs.cosine_distance_to_query(0, &query, query_norm);
        assert_relative_eq!(dist_0, 1.0 - 1.0 / 2.0_f32.sqrt(), epsilon = 1e-6);

        // Same vector: cosine distance should be 0
        let dist_2 = vecs.cosine_distance_to_query(2, &query, query_norm);
        assert_relative_eq!(dist_2, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_euclidean_distance_static() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        let dist = euclidean_distance_static(&a, &b);
        assert_relative_eq!(dist, 27.0, epsilon = 1e-6);

        // Zero distance to self
        let dist_self = euclidean_distance_static(&a, &a);
        assert_relative_eq!(dist_self, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance_static() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        // Orthogonal vectors
        let dist = cosine_distance_static(&a, &b);
        assert_relative_eq!(dist, 1.0, epsilon = 1e-6);

        // Parallel vectors
        let c = vec![2.0, 0.0, 0.0];
        let dist_parallel = cosine_distance_static(&a, &c);
        assert_relative_eq!(dist_parallel, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_normalise_vector() {
        let mut vec = vec![3.0, 4.0, 0.0];
        normalise_vector(&mut vec);

        // Should be [0.6, 0.8, 0.0]
        assert_relative_eq!(vec[0], 0.6, epsilon = 1e-6);
        assert_relative_eq!(vec[1], 0.8, epsilon = 1e-6);
        assert_relative_eq!(vec[2], 0.0, epsilon = 1e-6);

        // Norm should be 1
        let norm = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_normalise_vector_zero() {
        let mut vec = vec![0.0, 0.0, 0.0];
        normalise_vector(&mut vec);

        // Zero vector should remain zero
        assert_relative_eq!(vec[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vec[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vec[2], 0.0, epsilon = 1e-6);
    }

    #[cfg(feature = "quantised")]
    mod quantised_tests {
        use super::*;

        struct TestVectorsSq8 {
            data: Vec<i8>,
            norms: Vec<i32>,
            dim: usize,
        }

        impl VectorDistanceSq8<f32> for TestVectorsSq8 {
            fn vectors_flat_quantised(&self) -> &[i8] {
                &self.data
            }

            fn norms_quantised(&self) -> &[i32] {
                &self.norms
            }

            fn dim(&self) -> usize {
                self.dim
            }
        }

        #[test]
        fn test_euclidean_distance_i8() {
            let data = vec![127, 0, 0, 0, 127, 0];

            let vecs = TestVectorsSq8 {
                data,
                norms: vec![],
                dim: 3,
            };

            let query = vec![127, 127, 0];

            // Distance from [127,0,0] to [127,127,0] should be 127^2
            let dist = vecs.euclidean_distance_i8(0, &query);
            assert_relative_eq!(dist, 16129.0, epsilon = 1e-3);
        }

        #[test]
        fn test_cosine_distance_i8() {
            let data = vec![127, 0, 0, 0, 127, 0, 127, 127, 0];

            let norm0 = 127 * 127;
            let norm1 = 127 * 127;
            let norm2 = 127 * 127 + 127 * 127;

            let vecs = TestVectorsSq8 {
                data,
                norms: vec![norm0, norm1, norm2],
                dim: 3,
            };

            let query = vec![127, 127, 0];
            let query_norm_sq = 127 * 127 + 127 * 127;

            // Orthogonal vectors
            let dist_0 = vecs.cosine_distance_i8(0, &query, query_norm_sq);
            assert_relative_eq!(dist_0, 1.0 - 1.0 / 2.0_f32.sqrt(), epsilon = 1e-5);

            // Same direction
            let dist_2 = vecs.cosine_distance_i8(2, &query, query_norm_sq);
            assert_relative_eq!(dist_2, 0.0, epsilon = 1e-5);
        }
    }
}
