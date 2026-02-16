use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::{collections::BinaryHeap, iter::Sum};
use thousands::*;

use crate::prelude::*;
use crate::utils::matrix_to_flat;

/////////////////////
// Index structure //
/////////////////////

/// Exhaustive (brute-force) nearest neighbour index
///
/// ### Fields
///
/// * `vectors_flat` - Original vector data for distance calculations. Flattened
///   for better cache locality
/// * `norms` - Normalised pre-calculated values per sample if distance is set
///   to Cosine
/// * `dim` - Embedding dimensions
/// * `n` - Number of samples
/// * `metric` - The type of distance the index is designed for
pub struct ExhaustiveIndex<T> {
    // shared ones
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    norms: Vec<T>,
    metric: Dist,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for ExhaustiveIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn norms(&self) -> &[T] {
        &self.norms
    }
}

/////////////////////
// ExhaustiveIndex //
/////////////////////

impl<T> ExhaustiveIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Generate a new exhaustive index
    ///
    /// ### Params
    ///
    /// * `data` - The data for which to generate the index. Samples x features
    /// * `metric` - Which distance metric the index shall be generated for.
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive index
    pub fn new(data: MatRef<T>, metric: Dist) -> Self {
        let (vectors_flat, n, dim) = matrix_to_flat(data);

        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    T::calculate_l2_norm(&vectors_flat[start..end])
                })
                .collect()
        } else {
            Vec::new()
        };

        Self {
            vectors_flat,
            norms,
            dim,
            metric,
            n,
        }
    }

    //////////////////
    // Query (dist) //
    //////////////////

    /// Query function
    ///
    /// This will do an exhaustive search over the full index (i.e., all samples)
    /// during querying. To note, this becomes prohibitively computationally
    /// expensive on large data sets!
    ///
    /// ### Params
    ///
    /// * `query_vec` - The query vector.
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// A tuple of `(indices, distances)`
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        assert!(
            query_vec.len() == self.dim,
            "The query vector has different dimensionality than the index"
        );

        let n_vectors = self.vectors_flat.len() / self.dim;
        let k = k.min(n_vectors);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        match self.metric {
            Dist::Euclidean => {
                for idx in 0..n_vectors {
                    let dist = self.euclidean_distance_to_query(idx, query_vec);

                    if heap.len() < k {
                        heap.push((OrderedFloat(dist), idx));
                    } else if dist < heap.peek().unwrap().0 .0 {
                        heap.pop();
                        heap.push((OrderedFloat(dist), idx));
                    }
                }
            }
            Dist::Cosine => {
                let query_norm = query_vec
                    .iter()
                    .map(|v| *v * *v)
                    .fold(T::zero(), |a, b| a + b)
                    .sqrt();

                for idx in 0..n_vectors {
                    let dist = self.cosine_distance_to_query(idx, query_vec, query_norm);

                    if heap.len() < k {
                        heap.push((OrderedFloat(dist), idx));
                    } else if dist < heap.peek().unwrap().0 .0 {
                        heap.pop();
                        heap.push((OrderedFloat(dist), idx));
                    }
                }
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);

        let (distances, indices): (Vec<_>, Vec<_>) = results
            .into_iter()
            .map(|(OrderedFloat(dist), idx)| (dist, idx))
            .unzip();

        (indices, distances)
    }

    /// Query function for row references
    ///
    /// This will do an exhaustive search over the full index (i.e., all samples)
    /// during querying. To note, this becomes prohibitively computationally
    /// expensive on large data sets!
    ///
    /// ### Params
    ///
    /// * `query_row` - The query row.
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// A tuple of `(indices, distances)`
    #[inline]
    pub fn query_row(&self, query_row: RowRef<T>, k: usize) -> (Vec<usize>, Vec<T>) {
        assert!(
            query_row.ncols() == self.dim,
            "The query row has different dimensionality than the index"
        );

        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k)
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete
    /// kNN graph.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)` where each row corresponds
    /// to a vector in the index
    pub fn generate_knn(
        &self,
        k: usize,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dim;
                let end = start + self.dim;
                let vec = &self.vectors_flat[start..end];

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            self.n.separate_with_underscores()
                        );
                    }
                }

                self.query(vec, k)
            })
            .collect();

        if return_dist {
            let (indices, distances) = results.into_iter().unzip();
            (indices, Some(distances))
        } else {
            let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _)| idx).collect();
            (indices, None)
        }
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::Mat;

    fn create_simple_matrix() -> Mat<f32> {
        // 5 points in 3D space
        let data = [
            1.0, 0.0, 0.0, // Point 0: [1, 0, 0]
            0.0, 1.0, 0.0, // Point 1: [0, 1, 0]
            0.0, 0.0, 1.0, // Point 2: [0, 0, 1]
            1.0, 1.0, 0.0, // Point 3: [1, 1, 0]
            1.0, 0.0, 1.0, // Point 4: [1, 0, 1]
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
    }

    #[test]
    fn test_exhaustive_index_creation_euclidean() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Euclidean);

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.vectors_flat.len(), 15);
        assert!(index.norms.is_empty()); // No norms for Euclidean
    }

    #[test]
    fn test_exhaustive_index_creation_cosine() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Cosine);

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.vectors_flat.len(), 15);
        assert_eq!(index.norms.len(), 5); // Norms computed for Cosine
    }

    #[test]
    fn test_exhaustive_query_finds_self_euclidean() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Euclidean);

        // Query with point 0, should find itself first
        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_query_finds_self_cosine() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Cosine);

        // Query with point 0, should find itself first
        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_query_euclidean_multiple() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Euclidean);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3);

        // Should find point 0 first (exact match)
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Results should be sorted by distance
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_exhaustive_query_cosine_orthogonal() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Cosine);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 5); // Get all 5

        // Should find point 0 first (identical direction)
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Points 3 and 4 are at 45° (closer than orthogonal)
        assert_relative_eq!(distances[1], 1.0 - 1.0 / 2.0_f32.sqrt(), epsilon = 1e-5);
        assert_relative_eq!(distances[2], 1.0 - 1.0 / 2.0_f32.sqrt(), epsilon = 1e-5);

        // Points 1 and 2 are orthogonal (furthest away)
        assert_relative_eq!(distances[3], 1.0, epsilon = 1e-5);
        assert_relative_eq!(distances[4], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_query_k_larger_than_dataset() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Euclidean);

        let query = vec![1.0, 0.0, 0.0];
        // Ask for 10 neighbours but only 5 points exist
        let (indices, _) = index.query(&query, 10);

        // Should return exactly 5 results
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_exhaustive_query_row() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Euclidean);

        // Query using a row from the matrix
        let (indices, distances) = index.query_row(mat.row(0), 1);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_euclidean_distances() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Euclidean);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 5);

        // Distance from [1,0,0] to [1,0,0] = 0
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Distance from [1,0,0] to [1,0,1] = 1
        // Distance from [1,0,0] to [1,1,0] = 1
        // Both should appear next (order might vary)
        assert!(distances[1] <= 1.01);
        assert!(distances[2] <= 1.01);

        // Distance from [1,0,0] to [0,1,0] = sqrt(2) ≈ 1.414
        // Distance from [1,0,0] to [0,0,1] = sqrt(2) ≈ 1.414
        assert_relative_eq!(distances[3], 2.0, epsilon = 0.1);
        assert_relative_eq!(distances[4], 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_exhaustive_all_points_found() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Euclidean);

        let query = vec![0.5, 0.5, 0.5];
        let (indices, _) = index.query(&query, 5);

        // All 5 points should be found
        assert_eq!(indices.len(), 5);

        // All unique indices
        let mut sorted_indices = indices.clone();
        sorted_indices.sort_unstable();
        assert_eq!(sorted_indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_exhaustive_larger_dataset() {
        // Create a larger dataset
        let n = 50;
        let dim = 10;
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 10.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Euclidean);

        // Query for point 0
        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5);

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0); // Should find exact match first
    }

    #[test]
    fn test_exhaustive_cosine_parallel_vectors() {
        let data = [
            1.0, 2.0, 3.0, // Vector 0
            2.0, 4.0, 6.0, // Vector 1 (parallel to 0, scaled by 2)
            -2.0, 1.0, 0.0, // Vector 2 (actually orthogonal to 0)
        ];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Cosine);

        let query = vec![1.0, 2.0, 3.0];
        let (indices, distances) = index.query(&query, 3);

        // Should find itself first
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Parallel vector should be second with distance ≈ 0
        assert_eq!(indices[1], 1);
        assert_relative_eq!(distances[1], 0.0, epsilon = 1e-5);

        // Orthogonal vector should be last with distance = 1
        assert_eq!(indices[2], 2);
        assert_relative_eq!(distances[2], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_implements_vector_distance() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Euclidean);

        // Test that we can call VectorDistance methods
        let dist = index.euclidean_distance(0, 1);
        assert!(dist > 0.0); // [1,0,0] vs [0,1,0] should have distance > 0

        let dist_self = index.euclidean_distance(0, 0);
        assert_relative_eq!(dist_self, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_cosine_implements_vector_distance() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Cosine);

        // Test that we can call VectorDistance methods
        let dist = index.cosine_distance(0, 1);
        assert_relative_eq!(dist, 1.0, epsilon = 1e-5); // Orthogonal vectors

        let dist_self = index.cosine_distance(0, 0);
        assert_relative_eq!(dist_self, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_query_consistency() {
        // Test that query and query_row give same results
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Euclidean);

        let query_vec = vec![1.0, 0.0, 0.0];
        let (indices1, distances1) = index.query(&query_vec, 3);
        let (indices2, distances2) = index.query_row(mat.row(0), 3);

        assert_eq!(indices1, indices2);
        for i in 0..distances1.len() {
            assert_relative_eq!(distances1[i], distances2[i], epsilon = 1e-5);
        }
    }
}
