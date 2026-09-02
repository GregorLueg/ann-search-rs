//! Exhaustive bf16 index: quantises the original data to bf16 (keeps the
//! norms).

use faer::RowRef;
use half::bf16;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use thousands::*;

use crate::prelude::*;
use crate::quantised::quantisers::*;
use crate::utils::pack_knn_results;

/////////////////////
// Index structure //
/////////////////////

/// Exhaustive (brute-force) nearest neighbour index (with bf16 quantisation)
///
/// Uses under the hood bf16 quantisation for reduced memory finger print and
/// increased query speed at cost of precision.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct ExhaustiveIndexBf16<T> {
    /// Original vector data for distance calculations. Flattened for better
    /// cache locality and quantised to bf16
    pub vectors_flat: Vec<bf16>,
    /// Embedding dimensions
    pub dim: usize,
    /// Number of samples
    pub n: usize,
    /// Normalised pre-calculated values per sample if distance is set to
    /// Cosine. Keep `T` here to avoid massive precision loss for Cosine.
    norms: Vec<T>,
    /// The type of distance the index is designed for
    metric: Dist,
    /// PhantomData
    _phantom: PhantomData<T>,
}

////////////////////////
// VectorDistanceBf16 //
////////////////////////

impl<T> VectorDistanceBf16<T> for ExhaustiveIndexBf16<T>
where
    T: AnnSearchFloat + Bf16Compatible,
{
    fn vectors_flat(&self) -> &[bf16] {
        &self.vectors_flat
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn norms(&self) -> &[T] {
        &self.norms
    }
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for ExhaustiveIndexBf16<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

///////////
// Shape //
///////////

impl<T> ExhaustiveIndexBf16<T> {
    /// Number of indexed samples.
    ///
    /// ### Returns
    ///
    /// The row count the index was built from. Pairs with
    /// [`DimensionValidation::dim`] so every index exposes its shape the same
    /// way, which is what the Python bindings' handle macro reads.
    pub fn n(&self) -> usize {
        self.n
    }
}

/////////////////////
// ExhaustiveIndex //
/////////////////////

impl<T> ExhaustiveIndexBf16<T>
where
    T: AnnSearchFloat + Bf16Compatible,
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
    pub fn new(data: impl AnnMatrix<T>, metric: Dist) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (vectors_flat, n, dim) = data.into_row_major();

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

        Ok(Self {
            vectors_flat: encode_bf16_quantisation(&vectors_flat),
            norms,
            dim,
            metric,
            n,
            _phantom: std::marker::PhantomData,
        })
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
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        let n_vectors = self.vectors_flat.len() / self.dim;
        let k = k.min(n_vectors);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        match self.metric {
            Dist::SquaredEuclidean => {
                for idx in 0..n_vectors {
                    let dist = self.euclidean_distance_to_query_bf16(idx, query_vec);

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
                    let dist = self.cosine_distance_to_query_bf16(idx, query_vec, query_norm);

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
    /// * `query_row` - The query row.
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

    /// Internal BF16 query function
    ///
    /// ### Params
    ///
    /// * `query_vec` - The query vec already quantised to `bf16`.
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// A tuple of `(indices, distances)`
    fn query_bf16(&self, query_vec: &[bf16], k: usize) -> (Vec<usize>, Vec<T>) {
        let n_vectors = self.vectors_flat.len() / self.dim;
        let k = k.min(n_vectors);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        match self.metric {
            Dist::SquaredEuclidean => {
                for idx in 0..n_vectors {
                    let dist = self.euclidean_distance_to_query_dual_bf16(idx, query_vec);

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
                    .map(|&v| v.to_f32() * v.to_f32())
                    .fold(0_f32, |a, b| a + b)
                    .sqrt();

                for idx in 0..n_vectors {
                    let dist = self.cosine_distance_to_query_dual_bf16(
                        idx,
                        query_vec,
                        bf16::from_f32(query_norm),
                    );

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

        (indices, distances)
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

                self.query_bf16(vec, k)
            })
            .collect();

        pack_knn_results(results, return_dist)
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<bf16>()
            + self.norms.capacity() * std::mem::size_of::<bf16>()
    }
}

///////////
// Tests //
///////////

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for ExhaustiveIndexBf16<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "exhaustive_bf16";
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use faer_traits::ComplexField;
    use num_traits::{Float, FromPrimitive};

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
    fn test_exhaustive_bf16_construction_euclidean() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBf16::new(data.as_ref(), Dist::SquaredEuclidean).unwrap();

        assert_eq!(index.n, 100);
        assert_eq!(index.dim, 32);
        assert_eq!(index.vectors_flat.len(), 100 * 32);
        assert!(index.norms.is_empty());
    }

    #[test]
    fn test_exhaustive_bf16_construction_cosine() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBf16::new(data.as_ref(), Dist::Cosine).unwrap();

        assert_eq!(index.n, 100);
        assert_eq!(index.dim, 32);
        assert_eq!(index.norms.len(), 100);
    }

    #[test]
    fn test_exhaustive_bf16_query_returns_k_results() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBf16::new(data.as_ref(), Dist::SquaredEuclidean).unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10).unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_exhaustive_bf16_query_sorted() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBf16::new(data.as_ref(), Dist::SquaredEuclidean).unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (_, distances) = index.query(&query, 10).unwrap();

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_exhaustive_bf16_query_k_exceeds_n() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexBf16::new(data.as_ref(), Dist::SquaredEuclidean).unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, _) = index.query(&query, 100).unwrap();

        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_exhaustive_bf16_query_row() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBf16::new(data.as_ref(), Dist::SquaredEuclidean).unwrap();

        let (indices, distances) = index.query_row(data.as_ref().row(0), 10).unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_exhaustive_bf16_query_cosine() {
        let data = create_test_data::<f32>(100, 32);
        let index = ExhaustiveIndexBf16::new(data.as_ref(), Dist::Cosine).unwrap();

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10).unwrap();

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_exhaustive_bf16_knn_graph() {
        let data = create_test_data::<f32>(50, 32);
        let index = ExhaustiveIndexBf16::new(data.as_ref(), Dist::SquaredEuclidean).unwrap();

        let (knn_indices, knn_distances) = index.generate_knn(5, true, false);

        assert_eq!(knn_indices.len(), 50);
        assert!(knn_distances.is_some());
        assert_eq!(knn_distances.as_ref().unwrap().len(), 50);

        for neighbours in knn_indices.iter() {
            assert_eq!(neighbours.len(), 5);
        }
    }
}
