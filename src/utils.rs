use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rustc_hash::FxHashSet;
use std::collections::BinaryHeap;

use crate::dist::*;

///////////////
// Distances //
///////////////

/// Enum for the approximate nearest neighbour search
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

///////////////////
// Float on heap //
///////////////////

/// Wrapper for f32 that implements Ord for use in BinaryHeap
///
/// Faster than the sorts on full vectors and allows to keep data on heap
#[derive(Clone, Copy, Debug)]
pub struct OrderedFloat<T>(pub T);

/// Partial equality trait
impl<T: Float> PartialEq for OrderedFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

/// Equality trait
impl<T: Float> Eq for OrderedFloat<T> {}

/// Partial ordering trait
impl<T: Float> PartialOrd for OrderedFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Comparing one to the other
impl<T: Float> Ord for OrderedFloat<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

////////////////
// Validation //
////////////////

pub trait KnnValidation<T>: VectorDistance<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::{Ordering, Reverse};
    use std::collections::BinaryHeap;

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
    fn test_ordered_float_f32_equality() {
        let a = OrderedFloat(1.0_f32);
        let b = OrderedFloat(1.0_f32);
        let c = OrderedFloat(2.0_f32);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_ordered_float_f64_equality() {
        let a = OrderedFloat(1.0_f64);
        let b = OrderedFloat(1.0_f64);
        let c = OrderedFloat(2.0_f64);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_ordered_float_ordering() {
        let a = OrderedFloat(1.0_f32);
        let b = OrderedFloat(2.0_f32);
        let c = OrderedFloat(1.0_f32);

        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&a), Ordering::Greater);
        assert_eq!(a.cmp(&c), Ordering::Equal);
    }

    #[test]
    fn test_ordered_float_partial_ord() {
        let a = OrderedFloat(1.0_f32);
        let b = OrderedFloat(2.0_f32);

        assert!(a < b);
        assert!(b > a);
        assert!(a <= b);
        assert!(b >= a);
    }

    #[test]
    fn test_ordered_float_in_binary_heap() {
        let mut heap = BinaryHeap::new();
        heap.push(OrderedFloat(3.0_f32));
        heap.push(OrderedFloat(1.0_f32));
        heap.push(OrderedFloat(2.0_f32));

        // binaryHeap is a max-heap, so should pop in descending order
        assert_eq!(heap.pop(), Some(OrderedFloat(3.0)));
        assert_eq!(heap.pop(), Some(OrderedFloat(2.0)));
        assert_eq!(heap.pop(), Some(OrderedFloat(1.0)));
    }

    #[test]
    fn test_ordered_float_in_reverse_binary_heap() {
        let mut heap = BinaryHeap::new();
        heap.push(Reverse(OrderedFloat(3.0_f32)));
        heap.push(Reverse(OrderedFloat(1.0_f32)));
        heap.push(Reverse(OrderedFloat(2.0_f32)));

        // reverse makes it a min-heap, should pop in ascending order
        assert_eq!(heap.pop(), Some(Reverse(OrderedFloat(1.0))));
        assert_eq!(heap.pop(), Some(Reverse(OrderedFloat(2.0))));
        assert_eq!(heap.pop(), Some(Reverse(OrderedFloat(3.0))));
    }

    #[test]
    fn test_ordered_float_nan_handling() {
        let a = OrderedFloat(1.0_f32);
        let nan = OrderedFloat(f32::NAN);

        // NaN should be treated as equal to itself (via unwrap_or)
        assert_eq!(nan.cmp(&nan), Ordering::Equal);

        // Ordering with NaN should default to Equal
        assert_eq!(a.cmp(&nan), Ordering::Equal);
        assert_eq!(nan.cmp(&a), Ordering::Equal);
    }

    #[test]
    fn test_ordered_float_negative_values() {
        let a = OrderedFloat(-1.0_f32);
        let b = OrderedFloat(-2.0_f32);
        let c = OrderedFloat(0.0_f32);

        assert!(b < a); // -2 < -1
        assert!(a < c); // -1 < 0
    }

    #[test]
    fn test_ordered_float_zero_comparison() {
        let pos_zero = OrderedFloat(0.0_f32);
        let neg_zero = OrderedFloat(-0.0_f32);

        // IEEE 754: +0 == -0
        assert_eq!(pos_zero, neg_zero);
    }

    #[test]
    fn test_ordered_float_infinity() {
        let inf = OrderedFloat(f32::INFINITY);
        let neg_inf = OrderedFloat(f32::NEG_INFINITY);
        let finite = OrderedFloat(1.0_f32);

        assert!(neg_inf < finite);
        assert!(finite < inf);
        assert!(neg_inf < inf);
    }

    #[test]
    fn test_ordered_float_clone() {
        let a = OrderedFloat(3.15);
        let b = a;

        assert_eq!(a, b);
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn test_dist_clone() {
        let d1 = Dist::Euclidean;
        let d2 = d1;

        assert_eq!(d1, d2);
    }
}
