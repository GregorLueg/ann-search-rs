//! All types of utilities of this crate. From SIMD-accelerated distance
//! calculations, to highly optimised k-means clustering to structure to be
//! kept on the heap, shared traits, etd.

pub mod dist;
pub mod graph_utils;
pub mod heap_structs;
pub mod k_means_utils;
pub mod parallelism;
pub mod traits;
pub mod tree_utils;

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

    let mut vectors_flat = Vec::with_capacity(n * dim);
    for i in 0..n {
        vectors_flat.extend(mat.row(i).iter().cloned());
    }

    (vectors_flat, n, dim)
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
    fn original_ids(&self) -> &[usize];

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
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);

        let (distances, indices): (Vec<_>, Vec<_>) = results
            .into_iter()
            .map(|(OrderedFloat(dist), idx)| (dist, self.original_ids()[idx]))
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
