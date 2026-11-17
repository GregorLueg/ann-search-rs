//! All types of utilities of this crate. From SIMD-accelerated distance
//! calculations, to highly optimised k-means clustering to structure to be
//! kept on the heap, shared traits, etd.

pub mod dist;
pub mod e_means;
pub mod gemm;
pub mod graph_utils;
pub mod heap_structs;
pub mod hnsw_graph;
pub mod input;
pub mod k_means_utils;
pub mod nndescent_utils;
pub mod parallelism;
#[cfg(any(feature = "gpu", mlx_available))]
pub mod rp_forest;
pub mod traits;
pub mod tree_utils;
pub mod verbosity;

#[cfg(any(feature = "binary", feature = "serialise"))]
pub mod staging;

use faer::MatRef;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rustc_hash::FxHashSet;
use std::collections::BinaryHeap;
use std::iter::Sum;

use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Pack per-query results into the crate's `KnnOptionResult` shape
///
/// Applies [`fix_neg_dist`] on the way through, so every query path funnelling
/// through here returns non-negative distances. Paths that scatter results back
/// into original positions cannot use this and should call `fix_neg_dist`
/// directly.
///
/// ### Params
///
/// * `results` - Per-query `(indices, distances)` pairs
/// * `return_dist` - Whether the distances are kept
///
/// ### Returns
///
/// A tuple of `(knn_indices, optional distances)`
pub(crate) fn pack_knn_results<T>(
    results: Vec<(Vec<usize>, Vec<T>)>,
    return_dist: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>)
where
    T: num_traits::Zero + PartialOrd,
{
    if return_dist {
        let (indices, distances): (Vec<Vec<usize>>, Vec<Vec<T>>) = results.into_iter().unzip();
        let mut distances = Some(distances);
        fix_neg_dist(&mut distances);
        (indices, distances)
    } else {
        (results.into_iter().map(|(idx, _)| idx).collect(), None)
    }
}

/// Type alias for flattened structure
pub type FlattenData<T> = (Vec<T>, usize, usize);

/// Flatten a matrix to a vector
///
/// ### Params
///
/// * `mat` - Matrix reference to flatten
///
/// ### Returns
///
/// The flatten vector
pub fn matrix_to_flat<T>(mat: MatRef<T>) -> FlattenData<T>
where
    T: Float,
{
    let n = mat.nrows();
    let dim = mat.ncols();
    let len = n * dim;

    let mut out = Vec::<T>::with_capacity(len);
    let ptr = out.as_mut_ptr();

    for j in 0..dim {
        let col = mat.col(j);
        for i in 0..n {
            unsafe {
                ptr.add(i * dim + j).write(col[i]);
            }
        }
    }
    unsafe {
        out.set_len(len);
    }

    (out, n, dim)
}

/// Gather rows in place: afterwards row `i` holds what was row `order[i]`.
///
/// Follows the permutation's cycles with one row of scratch, so peak memory
/// stays at the buffer itself rather than doubling it as a gather into a new
/// buffer would. Sequential, since each step of a cycle depends on the last.
///
/// ### Params
///
/// * `data` - Flattened row-major rows, `order.len() * dim` elements
/// * `dim` - Elements per row
/// * `order` - Source row for each output row; must be a permutation
pub(crate) fn permute_rows_in_place<T: Copy>(data: &mut [T], dim: usize, order: &[u32]) {
    let n = order.len();
    debug_assert_eq!(data.len(), n * dim);
    if dim == 0 {
        return;
    }
    let mut done = vec![false; n];
    let mut scratch: Vec<T> = Vec::with_capacity(dim);
    for start in 0..n {
        if done[start] || order[start] as usize == start {
            continue;
        }
        scratch.clear();
        scratch.extend_from_slice(&data[start * dim..(start + 1) * dim]);
        let mut dst = start;
        loop {
            done[dst] = true;
            let src = order[dst] as usize;
            if src == start {
                data[dst * dim..(dst + 1) * dim].copy_from_slice(&scratch);
                break;
            }
            data.copy_within(src * dim..(src + 1) * dim, dst * dim);
            dst = src;
        }
    }
}

////////////////
// Validation //
////////////////

/// Trait for validation of the approximate nearest neighbour searches. This
/// will do an exhaustive search on a small subset and compare the `Recall@k`
/// against the internal query function of the index.
pub trait KnnValidation<T>: VectorDistance<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    /// Query for validation purposes
    ///
    /// * `query_vec` - The query Vec for which to do the exhaustive search
    ///   for.
    /// * `k` - Number of neighbours to return
    fn query_for_validation(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors>;

    /// Returns number of samples
    ///
    /// ### Returns
    ///
    /// The number of samples stored in the index
    fn n(&self) -> usize;

    /// Returns the dimensionality of index
    ///
    /// ### Returns
    ///
    /// The dimensionality of the index
    fn dim(&self) -> usize;

    /// Returns the Distance metric
    ///
    /// ### Returns
    ///
    /// The Dist metric.
    fn metric(&self) -> Dist;

    /// Return the original indices
    ///
    /// Needed for situations where the index remaps the data
    ///
    /// ### Returns
    ///
    /// The original indices of the index
    fn original_ids(&self) -> &[u32];

    /// Exhaustive search for ground truth
    ///
    /// ### Params
    ///
    /// * `query_vec` - The query Vec for which to do the exhaustive search
    ///   for.
    /// * `k` - Number of neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, dist)`
    fn exhaustive_query(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let n_vectors = self.n();
        let k = k.min(n_vectors);
        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        match self.metric() {
            Dist::SquaredEuclidean => {
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
            Dist::Manhattan => {
                for idx in 0..n_vectors {
                    let dist = self.manhattan_distance_to_query(idx, query_vec);

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
            .map(|(OrderedFloat(dist), idx)| (dist, self.original_ids()[idx] as usize))
            .unzip();

        Ok((indices, distances))
    }

    /// Validation function for the index
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours to return
    /// * `seed` - Seed for reproducibility
    /// * `no_samples` - Optional number of samples to. Otherwise defaults to
    ///   `1000` or n, whichever is smaller.
    ///
    /// ### Returns
    ///
    /// Recall@k for a subset of queried samples.
    fn validate_index(
        &self,
        k: usize,
        seed: usize,
        no_samples: Option<usize>,
    ) -> Result<f64, AnnSearchErrors> {
        let no_samples = no_samples.unwrap_or(1000).min(self.n());
        let mut rng = StdRng::seed_from_u64(seed as u64);

        let query_indices: Vec<usize> = (0..no_samples)
            .map(|_| rng.random_range(0..self.n()))
            .collect();

        let mut total_recall = 0.0;

        for &query_idx in &query_indices {
            let start = query_idx * KnnValidation::dim(self);
            let query_vec = &self.vectors_flat()[start..start + KnnValidation::dim(self)];

            let (approx_indices, _) = self.query_for_validation(query_vec, k)?;
            let (true_indices, _) = self.exhaustive_query(query_vec, k)?;

            let approx_set: FxHashSet<_> = approx_indices.into_iter().collect();
            let matches = true_indices
                .iter()
                .filter(|idx| approx_set.contains(idx))
                .count();

            total_recall += matches as f64 / k as f64;
        }

        Ok(total_recall / no_samples as f64)
    }
}

//////////////////////////
// Dimension validation //
//////////////////////////

/// Helper function to validate the dimensions
///
/// ### Params
///
/// * `query_dim` - Dimensionality of the queries
/// * `index_dim` - Dimensionality of the index
///
/// ### Returns
///
/// `Result<(), AnnSearchErrors::DimensionMismatch>`
pub fn validate_dim(query_dim: usize, index_dim: usize) -> Result<(), AnnSearchErrors> {
    if query_dim != index_dim {
        Err(AnnSearchErrors::DimensionMismatch {
            index_dim,
            query_dim,
        })
    } else {
        Ok(())
    }
}

/// Trait to check dimension mismatches
pub trait DimensionValidation {
    /// Returns the dimensionality of the index
    ///
    /// ### Returns
    ///
    /// Dimensionality of the index
    fn dim(&self) -> usize;

    /// Returns an error upon dimension mismatch
    ///
    /// ### Params
    ///
    /// * `query_dim` - The dimensionality of the error
    ///
    /// ### Returns
    ///
    /// An error upon dimension mismatch
    fn check_dim(&self, query_dim: usize) -> Result<(), AnnSearchErrors> {
        if query_dim != self.dim() {
            return Err(AnnSearchErrors::DimensionMismatch {
                index_dim: self.dim(),
                query_dim,
            });
        }

        Ok(())
    }
}

/////////////////////////
// Pre-fetching helper //
/////////////////////////

/// Helper function to pre-fetch next values
///
/// ### Params
///
/// * `ptr` - The memory address to pre-fetch
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // Cast the generic pointer to *const i8 as required by the x86 intrinsic
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        // The inline assembly just needs the 64-bit memory address in a register
        core::arch::asm!(
            "prfm pldl1keep, [{}]",
            in(reg) ptr,
            options(readonly, preserves_flags, nostack)
        );
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use rand::seq::SliceRandom;

    #[test]
    fn test_permute_rows_in_place_matches_gather() {
        let (n, dim) = (1_000, 3);
        let data: Vec<f32> = (0..n * dim).map(|v| v as f32).collect();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.shuffle(&mut StdRng::seed_from_u64(42));

        let expected: Vec<f32> = order
            .iter()
            .flat_map(|&src| data[src as usize * dim..(src as usize + 1) * dim].to_vec())
            .collect();
        let mut got = data.clone();
        permute_rows_in_place(&mut got, dim, &order);
        assert_eq!(got, expected);
    }
}
