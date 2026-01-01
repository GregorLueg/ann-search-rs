use faer::{MatRef, RowRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;
use thousands::*;

use crate::binary::binariser::*;
use crate::binary::dist_binary::*;

///////////////////////////
// ExhaustiveIndexBinary //
///////////////////////////

/// Exhaustive (brute-force) binary nearest neighbour index
///
/// Stores vectors as binary codes and uses Hamming distance for queries.
///
/// ### Fields
///
/// * `vectors_flat_binarised` - Binary codes, flattened (n * n_bytes)
/// * `n_bytes` - Bytes per vector (n_bits / 8)
/// * `n` - Number of samples
/// * `binariser` - Binariser for encoding query vectors
pub struct ExhaustiveIndexBinary<T> {
    // shared ones
    pub vectors_flat_binarised: Vec<u8>,
    pub n_bytes: usize,
    pub n: usize,
    binariser: Binariser<T>,
}

//////////////////////////
// VectorDistanceBinary //
//////////////////////////

impl<T> VectorDistanceBinary for ExhaustiveIndexBinary<T> {
    fn vectors_flat_binarised(&self) -> &[u8] {
        &self.vectors_flat_binarised
    }

    fn n_bytes(&self) -> usize {
        self.n_bytes
    }
}

impl<T> ExhaustiveIndexBinary<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField,
{
    /// Generate a new exhaustive binary index
    ///
    /// Binarises all vectors using the specified hash function and stores them
    /// as compact binary codes. This works solely for Cosine distance!
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features)
    /// * `n_bits` - Number of bits per binary code (must be multiple of 8)
    /// * `seed` - Random seed for binariser
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive binary index
    pub fn new(data: MatRef<T>, binarisation_init: &str, n_bits: usize, seed: usize) -> Self {
        assert!(n_bits % 8 == 0, "n_bits must be multiple of 8");

        let init = parse_binarisation_init(binarisation_init).unwrap_or_default();

        let n_bytes = n_bits / 8;
        let n = data.nrows();
        let dim = data.ncols();

        let binariser = match init {
            BinarisationInit::ITQ => Binariser::initialise_with_pca(data, dim, n_bits, seed),
            BinarisationInit::RandomProjections => Binariser::new(dim, n_bits, seed),
        };

        let mut vectors_flat_binarised: Vec<u8> = Vec::with_capacity(n * n_bytes);

        for i in 0..n {
            let original: Vec<T> = data.row(i).iter().cloned().collect();
            vectors_flat_binarised.extend(binariser.encode(&original));
        }

        Self {
            vectors_flat_binarised,
            n_bytes,
            n,
            binariser,
        }
    }

    ///////////
    // Query //
    ///////////

    /// Query function
    ///
    /// Exhaustive search over all binary codes using Hamming distance.
    /// Binary codes are generated via the trained binariser during
    /// initialisation.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (will be binarised internally)
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` where distances are Hamming distances
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<u32>) {
        let query_binary = self.binariser.encode(query_vec);
        let k = k.min(self.n);

        let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(k + 1);

        for idx in 0..self.n {
            let dist = self.hamming_distance_query(&query_binary, idx);

            if heap.len() < k {
                heap.push((dist, idx));
            } else if dist < heap.peek().unwrap().0 {
                heap.pop();
                heap.push((dist, idx));
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);

        let (distances, indices): (Vec<_>, Vec<_>) = results.into_iter().unzip();

        (indices, distances)
    }

    /// Query function for row references
    ///
    /// Exhaustive search using Hamming distance on binarised query.
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    #[inline]
    pub fn query_row(&self, query_row: RowRef<T>, k: usize) -> (Vec<usize>, Vec<u32>) {
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
    /// Queries each vector against all others to build complete kNN graph.
    /// Uses original float vectors which are binarised via the hash function,
    /// then Hamming distances computed between binary codes.
    ///
    /// ### Params
    ///
    /// * `data` - Original float data (needed to query each vector)
    /// * `k` - Number of neighbours per vector
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`
    pub fn generate_knn(
        &self,
        data: MatRef<T>,
        k: usize,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<u32>>>) {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        assert_eq!(data.nrows(), self.n, "Data row count mismatch");

        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<u32>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let vec: Vec<T> = data.row(i).iter().cloned().collect();

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

                self.query(&vec, k)
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
            + self.vectors_flat_binarised.capacity()
            + self.binariser.memory_usage_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn create_test_data<T: Float + FromPrimitive + ComplexField>(n: usize, dim: usize) -> Mat<T> {
        let mut data = Mat::zeros(n, dim);
        for i in 0..n {
            for j in 0..dim {
                data[(i, j)] = T::from_f64((i * dim + j) as f64 * 0.1).unwrap();
            }
        }
        data
    }

    #[test]
    fn test_exhaustive_binary_construction() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        assert_eq!(index.n, 100);
        assert_eq!(index.n_bytes, 8);
        assert_eq!(index.vectors_flat_binarised.len(), 100 * 8);
    }

    #[test]
    fn test_exhaustive_binary_query_returns_k_results() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10);

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_binary_query_sorted() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (_, distances) = index.query(&query, 10);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_exhaustive_binary_query_k_exceeds_n() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, _) = index.query(&query, 100);

        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_exhaustive_binary_query_row() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let (indices1, distances1) = index.query_row(data.as_ref().row(0), 10);

        assert_eq!(indices1.len(), 10);
        assert_eq!(distances1.len(), 10);
        assert_eq!(indices1[0], 0); // Should find itself first
    }

    #[test]
    fn test_exhaustive_binary_knn_graph() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let (knn_indices, knn_distances) = index.generate_knn(data.as_ref(), 5, true, false);

        assert_eq!(knn_indices.len(), 50);
        assert!(knn_distances.is_some());
        assert_eq!(knn_distances.as_ref().unwrap().len(), 50);

        for neighbours in knn_indices.iter() {
            assert_eq!(neighbours.len(), 5);
        }
    }

    #[test]
    fn test_hamming_distances_in_valid_range() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBinary::new(data.as_ref(), "random", 64, 42);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (_, distances) = index.query(&query, 20);

        // Hamming distance should be between 0 and n_bits
        for &dist in &distances {
            assert!(dist <= 64);
        }
    }
}
