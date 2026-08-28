//! Exhaustive SQ8 index: quantises the original data to scalar quantisation
//! to 8 bit (i8).

use faer::RowRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::{
    collections::BinaryHeap,
    iter::Sum,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use thousands::*;

use crate::prelude::*;
use crate::quantised::quantisers::*;

/////////////////////
// Index structure //
/////////////////////

/// Exhaustive (brute-force) nearest neighbour index with scalar 8-bit quantisation
///
/// ### Fields
///
/// * `quantised_vectors` - Original vector data quantised to `i8`. Flattened
///   for better cache locality
/// * `quantised_norms` - Pre-calculated quantised norms per sample if distance
///   is set to Cosine
/// * `dim` - Embedding dimensions
/// * `n` - Number of samples
/// * `metric` - The type of distance the index is designed for
/// * `codebook` - The codebook that contains the information of the quantisation
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct ExhaustiveSq8Index<T> {
    quantised_vectors: Vec<i8>,
    quantised_norms: Vec<i32>,
    dim: usize,
    n: usize,
    metric: Dist,
    codebook: ScalarQuantiser<T>,
}

//////////////////////
// VectorDistanceSq //
//////////////////////

impl<T> VectorDistanceSq8<T> for ExhaustiveSq8Index<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    fn vectors_flat_quantised(&self) -> &[i8] {
        &self.quantised_vectors
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn norms_quantised(&self) -> &[i32] {
        &self.quantised_norms
    }
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for ExhaustiveSq8Index<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////////////////
// ExhaustiveSq8Index //
/////////////////////////

impl<T> ExhaustiveSq8Index<T>
where
    T: AnnSearchFloat,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Generate a new exhaustive index with scalar 8-bit quantisation
    ///
    /// Constructs an exhaustive index with all vectors quantised to i8 using
    /// a global codebook. Reduces memory by 4x (for f32) whilst maintaining
    /// reasonable recall through learned quantisation bounds.
    ///
    /// ### Params
    ///
    /// * `data` - The data for which to generate the index. Samples x features
    /// * `metric` - Which distance metric the index shall be generated for
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive quantised index
    pub fn new(data: impl AnnMatrix<T>, metric: Dist) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (mut vectors_flat, n, dim) = data.into_row_major();

        // Normalise for cosine distance
        if metric == Dist::Cosine {
            vectors_flat
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        // Train codebook on all data
        let codebook = ScalarQuantiser::train(&vectors_flat, dim);

        // Quantise all vectors
        let mut quantised_vectors = vec![0i8; n * dim];
        quantised_vectors
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(vec_idx, chunk)| {
                let vec_start = vec_idx * dim;
                let vec = &vectors_flat[vec_start..vec_start + dim];
                let quantised = codebook.encode(vec);
                chunk.copy_from_slice(&quantised);
            });

        // Calculate quantised norms for cosine
        let quantised_norms = match metric {
            Dist::Cosine => quantised_vectors
                .par_chunks(dim)
                .map(|chunk| chunk.iter().map(|&v| v as i32 * v as i32).sum())
                .collect(),
            Dist::SquaredEuclidean => Vec::new(),
            Dist::Manhattan => unreachable!(),
        };

        Ok(Self {
            quantised_vectors,
            quantised_norms,
            dim,
            n,
            metric,
            codebook,
        })
    }

    //////////////////
    // Query (dist) //
    //////////////////

    /// Query function
    ///
    /// This will do an exhaustive search over the full index (i.e., all samples)
    /// during querying using quantised distance calculations. To note, this
    /// becomes prohibitively computationally expensive on large data sets!
    ///
    /// ### Params
    ///
    /// * `query_vec` - The query vector
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// A tuple of `(indices, distances)`
    #[inline]
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        let mut query_vec = query_vec.to_vec();
        let k = k.min(self.n);

        // Normalise query for cosine
        if self.metric == Dist::Cosine {
            normalise_vector(&mut query_vec);
        }

        // Encode query
        let query_i8 = self.codebook.encode(&query_vec);
        let query_norm_sq: i32 = query_i8.iter().map(|&q| q as i32 * q as i32).sum();

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        match self.metric {
            Dist::SquaredEuclidean => {
                for idx in 0..self.n {
                    let dist = self.euclidean_distance_i8(idx, &query_i8);

                    if heap.len() < k {
                        heap.push((OrderedFloat(dist), idx));
                    } else if dist < heap.peek().unwrap().0 .0 {
                        heap.pop();
                        heap.push((OrderedFloat(dist), idx));
                    }
                }
            }
            Dist::Cosine => {
                for idx in 0..self.n {
                    let dist = self.cosine_distance_i8(idx, &query_i8, query_norm_sq);

                    if heap.len() < k {
                        heap.push((OrderedFloat(dist), idx));
                    } else if dist < heap.peek().unwrap().0 .0 {
                        heap.pop();
                        heap.push((OrderedFloat(dist), idx));
                    }
                }
            }
            Dist::Manhattan => unreachable!(),
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);

        let (distances, indices): (Vec<_>, Vec<_>) = results
            .into_iter()
            .map(|(OrderedFloat(dist), idx)| (dist, idx))
            .unzip();

        Ok((indices, distances))
    }

    /// Query function for row references
    ///
    /// This will do an exhaustive search over the full index (i.e., all samples)
    /// during querying. To note, this becomes prohibitively computationally
    /// expensive on large data sets!
    ///
    /// ### Params
    ///
    /// * `query_row` - The query row
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// A tuple of `(indices, distances)`
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
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
    /// kNN graph. Uses pre-quantised vectors directly, avoiding encode overhead.
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
        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dim;
                let end = start + self.dim;
                let query_i8 = &self.quantised_vectors[start..end];
                let query_norm_sq = if self.metric == Dist::Cosine {
                    self.quantised_norms[i]
                } else {
                    0
                };

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

                let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> =
                    BinaryHeap::with_capacity(k + 1);

                for idx in 0..self.n {
                    let dist = match self.metric {
                        Dist::SquaredEuclidean => self.euclidean_distance_i8(idx, query_i8),
                        Dist::Cosine => self.cosine_distance_i8(idx, query_i8, query_norm_sq),
                        Dist::Manhattan => unreachable!(),
                    };

                    if heap.len() < k {
                        heap.push((OrderedFloat(dist), idx));
                    } else if dist < heap.peek().unwrap().0 .0 {
                        heap.pop();
                        heap.push((OrderedFloat(dist), idx));
                    }
                }

                let mut results: Vec<_> = heap.into_iter().collect();
                results.sort_unstable_by_key(|&(dist, _)| dist);
                let (distances, indices) = results.into_iter().map(|(d, i)| (d.0, i)).unzip();
                (indices, distances)
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
            + self.quantised_vectors.capacity() * std::mem::size_of::<i8>()
            + self.quantised_norms.capacity() * std::mem::size_of::<i32>()
            + self.codebook.memory_usage_bytes()
    }
}

///////////
// Tests //
///////////

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for ExhaustiveSq8Index<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "exhaustive_sq8";
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::Mat;

    fn create_simple_matrix() -> Mat<f32> {
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
    fn test_exhaustive_sq8_index_creation_euclidean() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean).unwrap();

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.quantised_vectors.len(), 15);
        assert!(index.quantised_norms.is_empty());
    }

    #[test]
    fn test_exhaustive_sq8_index_creation_cosine() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::Cosine).unwrap();

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.quantised_vectors.len(), 15);
        assert_eq!(index.quantised_norms.len(), 5);
    }

    #[test]
    fn test_exhaustive_sq8_query_finds_self_euclidean() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1).unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert!(distances[0] < 0.1); // Quantisation error
    }

    #[test]
    fn test_exhaustive_sq8_query_finds_self_cosine() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::Cosine).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _distances) = index.query(&query, 1).unwrap();

        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_exhaustive_sq8_query_euclidean_multiple() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3).unwrap();

        assert_eq!(indices[0], 0);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_exhaustive_sq8_query_cosine_orthogonal() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::Cosine).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 5).unwrap();

        assert_eq!(indices[0], 0);

        // Results should be sorted by distance
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_exhaustive_sq8_query_k_larger_than_dataset() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 10).unwrap();

        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_exhaustive_sq8_query_row() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean).unwrap();

        let (indices, distances) = index.query_row(mat.row(0), 1).unwrap();

        assert_eq!(indices[0], 0);
        assert!(distances[0] < 0.1);
    }

    #[test]
    fn test_exhaustive_sq8_euclidean_distances() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 5).unwrap();

        assert_eq!(indices[0], 0);
        assert!(distances[0] < 0.1);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_exhaustive_sq8_all_points_found() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean).unwrap();

        let query = vec![0.5, 0.5, 0.5];
        let (indices, _) = index.query(&query, 5).unwrap();

        assert_eq!(indices.len(), 5);

        let mut sorted_indices = indices.clone();
        sorted_indices.sort_unstable();
        assert_eq!(sorted_indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_exhaustive_sq8_larger_dataset() {
        let n = 50;
        let dim = 10;
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 10.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean).unwrap();

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5).unwrap();

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_exhaustive_sq8_cosine_parallel_vectors() {
        let data = [
            1.0, 2.0, 3.0, // Vector 0
            2.0, 4.0, 6.0, // Vector 1 (parallel to 0)
            -2.0, 1.0, 0.0, // Vector 2
        ];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::Cosine).unwrap();

        let query = vec![1.0, 2.0, 3.0];
        let (indices, distances) = index.query(&query, 3).unwrap();

        assert_eq!(indices[0], 0);
        assert!(distances[0] < 0.1);

        assert_eq!(indices[1], 1);
        assert!(distances[1] < 0.1);
    }

    #[test]
    fn test_exhaustive_sq8_implements_vector_distance() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let query_i8 = index.codebook.encode(&query);

        let dist = index.euclidean_distance_i8(0, &query_i8);
        assert!(dist < 1.0);

        let dist_other = index.euclidean_distance_i8(1, &query_i8);
        assert!(dist_other > dist);
    }

    #[test]
    fn test_exhaustive_sq8_cosine_implements_vector_distance() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::Cosine).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let query_i8 = index.codebook.encode(&query);
        let query_norm_sq: i32 = query_i8.iter().map(|&q| q as i32 * q as i32).sum();

        let dist = index.cosine_distance_i8(0, &query_i8, query_norm_sq);
        assert!(dist < 0.1);

        let dist_self = index.cosine_distance_i8(0, &query_i8, query_norm_sq);
        assert!(dist_self < 0.1);
    }

    #[test]
    fn test_exhaustive_sq8_query_consistency() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean).unwrap();

        let query_vec = vec![1.0, 0.0, 0.0];
        let (indices1, distances1) = index.query(&query_vec, 3).unwrap();
        let (indices2, distances2) = index.query_row(mat.row(0), 3).unwrap();

        assert_eq!(indices1, indices2);
        for i in 0..distances1.len() {
            assert_relative_eq!(distances1[i], distances2[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_exhaustive_sq8_generate_knn() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean).unwrap();

        let (knn_indices, knn_distances) = index.generate_knn(2, true, false);

        assert_eq!(knn_indices.len(), 5);
        assert!(knn_distances.is_some());

        let distances = knn_distances.unwrap();
        assert_eq!(distances.len(), 5);

        for i in 0..5 {
            assert_eq!(knn_indices[i].len(), 2);
            assert_eq!(distances[i].len(), 2);
            assert_eq!(knn_indices[i][0], i);
            assert!(distances[i][0] < 0.1);
        }
    }

    #[test]
    fn test_exhaustive_sq8_memory_usage() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean).unwrap();

        let memory = index.memory_usage_bytes();
        assert!(memory > 0);

        let expected_min = 5 * 3; // 5 vectors * 3 dims * 1 byte (i8)
        assert!(memory >= expected_min);
    }
}
