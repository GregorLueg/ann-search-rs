//! Exhaustive SQ8 index: stores the vectors as uniformly quantised 8-bit
//! codes and scans them with the integer kernel.

use faer::RowRef;
use rayon::prelude::*;
use std::{
    collections::BinaryHeap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use thousands::*;

use crate::prelude::*;
use crate::quantised::hnsw_quantised::codec::GraphCodec;
use crate::quantised::sq8u_codec::*;
use crate::quantised::uniform_quant::*;

/////////////////////
// Index structure //
/////////////////////

/// Exhaustive (brute-force) nearest neighbour index with scalar 8-bit
/// quantisation
///
/// ### Note
///
/// The quantisation shares one scale across every dimension. That is what
/// makes the integer distance between two codes preserve the ordering of the
/// float distance exactly, so the scan can stay in integer arithmetic. A
/// per-dimension scale would be finer-grained but would not cancel in a
/// code-to-code distance, which silently computes a reweighted metric instead
/// of the requested one.
///
/// ### Fields
///
/// * `codec` - Uniformly quantised storage and its distance arithmetic
/// * `dim` - Embedding dimensions
/// * `n` - Number of samples
/// * `metric` - The type of distance the index is designed for
#[cfg_attr(
    feature = "serialise",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "")
)]
pub struct ExhaustiveSq8Index<T>
where
    T: AnnSearchFloat,
{
    /// Uniformly quantised storage and its distance arithmetic.
    codec: Sq8uCodec<T>,
    /// Embedding dimensions.
    dim: usize,
    /// Number of samples.
    n: usize,
    /// The distance metric this index was built for.
    metric: Dist,
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for ExhaustiveSq8Index<T>
where
    T: AnnSearchFloat,
{
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
    /// Reduces memory by roughly 4x against `f32` whilst keeping the scan in
    /// integer arithmetic.
    ///
    /// ### Params
    ///
    /// * `data` - The data for which to generate the index. Samples x features
    /// * `metric` - Which distance metric the index shall be generated for
    /// * `quant_params` - Optional calibration settings, see
    ///   [`UniformQuantParams`]. Defaults trim 0.1% from each tail.
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive quantised index
    pub fn new(
        data: impl AnnMatrix<T>,
        metric: Dist,
        quant_params: Option<UniformQuantParams>,
    ) -> Result<Self, AnnSearchErrors> {
        let (flat, n, dim) = data.into_row_major();
        let codec = Sq8uCodec::new(&flat, n, dim, metric, quant_params)?;
        Ok(Self {
            codec,
            dim,
            n,
            metric,
        })
    }

    //////////////////
    // Query (dist) //
    //////////////////

    /// Scan every stored vector and keep the `k` nearest
    ///
    /// ### Params
    ///
    /// * `encoded` - The prepared query
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// A tuple of `(indices, distances)`, nearest first
    fn scan(&self, encoded: &Sq8uQuery<T>, k: usize) -> (Vec<usize>, Vec<T>) {
        let k = k.min(self.n);
        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for idx in 0..self.n {
            let score = self.codec.score(encoded, idx);
            if heap.len() < k {
                heap.push((OrderedFloat(score), idx));
            } else if score < heap.peek().unwrap().0 .0 {
                heap.pop();
                heap.push((OrderedFloat(score), idx));
            }
        }

        let mut out: Vec<(OrderedFloat<T>, usize)> = heap.into_vec();
        out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        out.into_iter()
            .map(|(OrderedFloat(s), i)| (i, self.codec.finalise(s)))
            .unzip()
    }

    /// Query function
    ///
    /// This will do an exhaustive search over the full index (i.e., all
    /// samples) during querying using quantised distance calculations. To
    /// note, this becomes prohibitively computationally expensive on large
    /// data sets!
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
        let encoded = self.codec.encode_query(query_vec)?;
        Ok(self.scan(&encoded, k))
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

    /// Generate the kNN graph over the stored vectors
    ///
    /// Every row queries through its own code, so no re-encoding happens and
    /// each point finds itself.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Print progress information
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
        let k = k.min(self.n);

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
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
                    let score = self.codec.score_sym(i, idx);
                    if heap.len() < k {
                        heap.push((OrderedFloat(score), idx));
                    } else if score < heap.peek().unwrap().0 .0 {
                        heap.pop();
                        heap.push((OrderedFloat(score), idx));
                    }
                }

                let mut out: Vec<(OrderedFloat<T>, usize)> = heap.into_vec();
                out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
                out.into_iter()
                    .map(|(OrderedFloat(s), idx)| (idx, self.codec.finalise(s)))
                    .unzip()
            })
            .collect();

        if return_dist {
            let (indices, distances) = results.into_iter().unzip();
            (indices, Some(distances))
        } else {
            (results.into_iter().map(|(idx, _)| idx).collect(), None)
        }
    }

    /// The distance metric this index was built for
    ///
    /// ### Returns
    ///
    /// The metric, see [`Dist`]
    pub fn metric(&self) -> Dist {
        self.metric
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self) + self.codec.memory_usage_bytes()
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
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean, None).unwrap();

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.metric(), Dist::SquaredEuclidean);
    }

    #[test]
    fn test_exhaustive_sq8_index_creation_cosine() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::Cosine, None).unwrap();

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.metric(), Dist::Cosine);
    }

    #[test]
    fn test_exhaustive_sq8_query_finds_self_euclidean() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean, None).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1).unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert!(distances[0] < 0.1); // Quantisation error
    }

    #[test]
    fn test_exhaustive_sq8_query_finds_self_cosine() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::Cosine, None).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _distances) = index.query(&query, 1).unwrap();

        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_exhaustive_sq8_query_euclidean_multiple() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean, None).unwrap();

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
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::Cosine, None).unwrap();

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
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean, None).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 10).unwrap();

        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_exhaustive_sq8_query_row() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean, None).unwrap();

        let (indices, distances) = index.query_row(mat.row(0), 1).unwrap();

        assert_eq!(indices[0], 0);
        assert!(distances[0] < 0.1);
    }

    #[test]
    fn test_exhaustive_sq8_euclidean_distances() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean, None).unwrap();

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
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean, None).unwrap();

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
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean, None).unwrap();

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
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::Cosine, None).unwrap();

        let query = vec![1.0, 2.0, 3.0];
        let (indices, distances) = index.query(&query, 3).unwrap();

        assert_eq!(indices[0], 0);
        assert!(distances[0] < 0.1);

        assert_eq!(indices[1], 1);
        assert!(distances[1] < 0.1);
    }

    #[test]
    fn test_exhaustive_sq8_scores_the_matching_vector_lowest() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean, None).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let encoded = index.codec.encode_query(&query).unwrap();

        let dist = index.codec.score(&encoded, 0);
        let dist_other = index.codec.score(&encoded, 1);
        assert!(dist_other > dist, "{dist_other} should exceed {dist}");
    }

    #[test]
    fn test_exhaustive_sq8_cosine_scores_the_matching_vector_lowest() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::Cosine, None).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let encoded = index.codec.encode_query(&query).unwrap();

        // Row 0 is the query direction; row 1 is orthogonal to it.
        let dist = index.codec.finalise(index.codec.score(&encoded, 0));
        let dist_other = index.codec.finalise(index.codec.score(&encoded, 1));
        assert!(dist < 0.1, "self distance {dist}");
        assert!(dist_other > dist);
    }

    #[test]
    fn test_exhaustive_sq8_symmetric_and_asymmetric_scores_agree() {
        // A stored vector used as a query must reproduce the symmetric score,
        // which is what lets `generate_knn` skip the re-encode.
        let mat = create_simple_matrix();
        for metric in [Dist::SquaredEuclidean, Dist::Cosine] {
            let index = ExhaustiveSq8Index::new(mat.as_ref(), metric, None).unwrap();
            let row: Vec<f32> = mat.row(0).iter().copied().collect();
            let encoded = index.codec.encode_query(&row).unwrap();
            for j in 0..5 {
                assert_relative_eq!(
                    index.codec.score(&encoded, j),
                    index.codec.score_sym(0, j),
                    max_relative = 1e-6
                );
            }
        }
    }

    #[test]
    fn test_exhaustive_sq8_query_consistency() {
        let mat = create_simple_matrix();
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean, None).unwrap();

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
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean, None).unwrap();

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
        let index = ExhaustiveSq8Index::new(mat.as_ref(), Dist::SquaredEuclidean, None).unwrap();

        let memory = index.memory_usage_bytes();
        assert!(memory > 0);

        let expected_min = 5 * 3; // 5 vectors * 3 dims * 1 byte (i8)
        assert!(memory >= expected_min);
    }
}
