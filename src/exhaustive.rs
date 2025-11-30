use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::utils::*;

/// Exhaustive (brute-force) nearest neighbour index
///
/// ### Fields
///
/// * `vectors_flat` - Original vector data for distance calculations. Flattened
///   for better cache locality
/// * `norm` - Normalised pre-calculated values per sample if distance is set to
///   Cosine
/// * `dim` - Embedding dimensions
/// * `dist` - The type of distance the index is designed for
pub struct ExhaustiveIndex<T> {
    vectors_flat: Vec<T>,
    norms: Vec<T>,
    dim: usize,
    dist: Dist,
}

impl<T> ExhaustiveIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    /// Generate a new exhaustive index
    ///
    /// ### Params
    ///
    /// * `data` - The data for which to generate the index. Samples x features
    /// * `dist` - Which distance metric the index shall be generated for.
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive index
    pub fn new(data: MatRef<T>, dist: Dist) -> Self {
        let n_vectors = data.nrows();
        let dim = data.ncols();

        let mut vectors_flat = Vec::with_capacity(n_vectors * dim);
        for i in 0..n_vectors {
            vectors_flat.extend(data.row(i).iter().cloned());
        }

        let norms = match dist {
            Dist::Cosine => (0..n_vectors)
                .map(|i| {
                    let vec_start = i * dim;
                    vectors_flat[vec_start..vec_start + dim]
                        .iter()
                        .map(|v| *v * *v)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt()
                })
                .collect(),
            Dist::Euclidean => Vec::new(),
        };

        Self {
            vectors_flat,
            norms,
            dim,
            dist,
        }
    }

    /// Query function
    ///
    /// This will do an exhaustive seach over the full index (i.e., all samples)
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
    ///
    /// ### Safety
    ///
    /// The function uses under the hood unsafe Rust with optimised layout for
    /// SIMD instructions. Be wary of this.
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        assert!(
            query_vec.len() == self.dim,
            "The query vector has different dimensionality than the index"
        );

        let n_vectors = self.vectors_flat.len() / self.dim;
        let k = k.min(n_vectors);

        let mut heap: BinaryHeap<(Reverse<OrderedFloat<T>>, usize)> =
            BinaryHeap::with_capacity(k + 1);

        match self.dist {
            Dist::Euclidean => {
                let query_ptr = query_vec.as_ptr();

                for idx in 0..n_vectors {
                    let vec_ptr = unsafe { self.vectors_flat.as_ptr().add(idx * self.dim) };
                    let mut sum = T::zero();
                    let mut i = 0;

                    while i + 4 <= self.dim {
                        let d0 = unsafe { *vec_ptr.add(i) - *query_ptr.add(i) };
                        let d1 = unsafe { *vec_ptr.add(i + 1) - *query_ptr.add(i + 1) };
                        let d2 = unsafe { *vec_ptr.add(i + 2) - *query_ptr.add(i + 2) };
                        let d3 = unsafe { *vec_ptr.add(i + 3) - *query_ptr.add(i + 3) };
                        sum = sum + d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
                        i += 4;
                    }

                    while i < self.dim {
                        let diff = unsafe { *vec_ptr.add(i) - *query_ptr.add(i) };
                        sum = sum + diff * diff;
                        i += 1;
                    }

                    if heap.len() < k {
                        heap.push((Reverse(OrderedFloat(sum)), idx));
                    } else if sum < heap.peek().unwrap().0 .0 .0 {
                        heap.pop();
                        heap.push((Reverse(OrderedFloat(sum)), idx));
                    }
                }
            }
            Dist::Cosine => {
                let norm_query: T = query_vec
                    .iter()
                    .map(|v| *v * *v)
                    .fold(T::zero(), |a, b| a + b)
                    .sqrt();

                let query_ptr = query_vec.as_ptr();

                for idx in 0..n_vectors {
                    let vec_ptr = unsafe { self.vectors_flat.as_ptr().add(idx * self.dim) };
                    let mut dot = T::zero();
                    let mut i = 0;

                    while i + 4 <= self.dim {
                        dot = dot
                            + unsafe { *vec_ptr.add(i) * *query_ptr.add(i) }
                            + unsafe { *vec_ptr.add(i + 1) * *query_ptr.add(i + 1) }
                            + unsafe { *vec_ptr.add(i + 2) * *query_ptr.add(i + 2) }
                            + unsafe { *vec_ptr.add(i + 3) * *query_ptr.add(i + 3) };
                        i += 4;
                    }

                    while i < self.dim {
                        dot = dot + unsafe { *vec_ptr.add(i) * *query_ptr.add(i) };
                        i += 1;
                    }

                    let dist =
                        T::one() - (dot / (norm_query * unsafe { *self.norms.get_unchecked(idx) }));

                    if heap.len() < k {
                        heap.push((Reverse(OrderedFloat(dist)), idx));
                    } else if dist < heap.peek().unwrap().0 .0 .0 {
                        heap.pop();
                        heap.push((Reverse(OrderedFloat(dist)), idx));
                    }
                }
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);

        let (distances, indices): (Vec<_>, Vec<_>) = results
            .into_iter()
            .map(|(Reverse(OrderedFloat(dist)), idx)| (dist, idx))
            .unzip();

        (indices, distances)
    }

    /// Query function for row references
    ///
    /// This will do an exhaustive seach over the full index (i.e., all samples)
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
    ///
    /// ### Safety
    ///
    /// The function uses under the hood unsafe Rust with optimised layout for
    /// SIMD instructions. Be wary of this.
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
}
