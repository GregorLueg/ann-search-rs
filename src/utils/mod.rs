pub mod dist;
pub mod heap_structs;
pub mod k_means;

use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rustc_hash::FxHashSet;
use std::collections::BinaryHeap;
use std::iter::Sum;

use crate::utils::dist::*;
use crate::utils::heap_structs::*;

////////////////
// Validation //
////////////////

pub trait KnnValidation<T>: VectorDistance<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    /// Query for validation purposes
    ///
    /// * `query_vec` - The query Vec for which to do the exhaustive search
    ///   for.
    /// * `k` - Number of neighbours to return
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>);

    /// Returns number of samples
    ///
    /// ### Returns
    ///
    /// The number of samples stored in the index
    fn n(&self) -> usize;

    /// Returns the Distance metric
    ///
    /// ### Returns
    ///
    /// The Dist metric.
    fn metric(&self) -> Dist;

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
    fn exhaustive_query(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        let n_vectors = self.n();
        let k = k.min(n_vectors);
        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        match self.metric() {
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
    fn validate_index(&self, k: usize, seed: usize, no_samples: Option<usize>) -> f64 {
        let no_samples = no_samples.unwrap_or(1000).min(self.n());
        let mut rng = StdRng::seed_from_u64(seed as u64);

        let query_indices: Vec<usize> = (0..no_samples)
            .map(|_| rng.random_range(0..self.n()))
            .collect();

        let mut total_recall = 0.0;

        for &query_idx in &query_indices {
            let start = query_idx * self.dim();
            let query_vec = &self.vectors_flat()[start..start + self.dim()];

            let (approx_indices, _) = self.query_for_validation(query_vec, k);
            let (true_indices, _) = self.exhaustive_query(query_vec, k);

            let approx_set: FxHashSet<_> = approx_indices.into_iter().collect();
            let matches = true_indices
                .iter()
                .filter(|idx| approx_set.contains(idx))
                .count();

            total_recall += matches as f64 / k as f64;
        }

        total_recall / no_samples as f64
    }
}
