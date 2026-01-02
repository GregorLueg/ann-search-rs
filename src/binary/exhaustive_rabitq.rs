use faer::{MatRef, RowRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;
use thousands::*;

use crate::binary::binariser::*;
use crate::binary::dist_binary::*;
use crate::utils::dist::*;
use crate::utils::heap_structs::*;

///////////////////////////
// ExhaustiveIndexRaBitQ //
///////////////////////////

/// Exhaustive RaBitQ nearest neighbour index
///
/// ### Fields
///
/// * `vectors_binary` - Binary codes (n × n_bytes)
/// * `dist_to_centroid` - Distance to centroid per vector
/// * `dot_corrections` - Dot correction per vector (v_c · v~)
/// * `n_bytes` - Bytes per binary code
/// * `n` - Number of vectors
/// * `dim` - Dimensionality
/// * `quantiser` - RaBitQ quantiser
pub struct ExhaustiveIndexRaBitQ<T> {
    pub vectors_binary: Vec<u8>,
    pub dist_to_centroid: Vec<T>,
    pub dot_corrections: Vec<T>,
    pub n_bytes: usize,
    pub n: usize,
    pub dim: usize,
    quantiser: RaBitQQuantiser<T>,
}

//////////////////////////
// VectorDistanceRaBitQ //
//////////////////////////

impl<T> VectorDistanceRaBitQ<T> for ExhaustiveIndexRaBitQ<T>
where
    T: Float + FromPrimitive,
{
    fn vectors_binary(&self) -> &[u8] {
        &self.vectors_binary
    }

    fn dist_to_centroid(&self) -> &[T] {
        &self.dist_to_centroid
    }

    fn dot_corrections(&self) -> &[T] {
        &self.dot_corrections
    }

    fn n_bytes(&self) -> usize {
        self.n_bytes
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

impl<T> ExhaustiveIndexRaBitQ<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField,
{
    /// Create a new exhaustive RaBitQ index
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (n_samples × dim)
    /// * `metric` - Distance metric
    ///
    /// ### Returns
    ///
    /// Initialised index
    pub fn new(data: MatRef<T>, metric: &Dist, seed: usize) -> Self {
        let n = data.nrows();
        let dim = data.ncols();
        let n_bytes = dim.div_ceil(8);

        let quantiser = RaBitQQuantiser::new(data, metric, seed);

        let mut vectors_binary = Vec::with_capacity(n * n_bytes);
        let mut dist_to_centroid = Vec::with_capacity(n);
        let mut dot_corrections = Vec::with_capacity(n);

        for i in 0..n {
            let vec: Vec<T> = data.row(i).iter().cloned().collect();
            let (binary, dist, dot_corr) = quantiser.encode_vec(&vec);

            vectors_binary.extend(binary);
            dist_to_centroid.push(dist);
            dot_corrections.push(dot_corr);
        }

        Self {
            vectors_binary,
            dist_to_centroid,
            dot_corrections,
            n_bytes,
            n,
            dim,
            quantiser,
        }
    }

    /// Query for k nearest neighbours
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        let encoded_query = self.quantiser.encode_query(query_vec);
        let k = k.min(self.n);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for idx in 0..self.n {
            let dist = self.rabitq_dist(&encoded_query, idx);

            if heap.len() < k {
                heap.push((OrderedFloat(dist), idx));
            } else if dist < heap.peek().unwrap().0 .0 {
                heap.pop();
                heap.push((OrderedFloat(dist), idx));
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let indices: Vec<usize> = results.iter().map(|(_, idx)| *idx).collect();
        let distances: Vec<T> = results.iter().map(|(d, _)| d.0).collect();

        (indices, distances)
    }

    /// Query using a row reference
    #[inline]
    pub fn query_row(&self, query_row: RowRef<T>, k: usize) -> (Vec<usize>, Vec<T>) {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k)
    }

    /// Generate kNN graph
    ///
    /// ### Params
    ///
    /// * `data` - Original data matrix
    /// * `k` - Neighbours per vector
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Tuple of (knn_indices, optional distances)
    pub fn generate_knn(
        &self,
        data: MatRef<T>,
        k: usize,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        assert_eq!(data.nrows(), self.n, "Data row count mismatch");

        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
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

    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_binary.capacity()
            + self.dist_to_centroid.capacity() * std::mem::size_of::<f32>()
            + self.dot_corrections.capacity() * std::mem::size_of::<f32>()
            + self.quantiser.memory_usage_bytes()
    }

    pub fn debug_distance(&self, data: MatRef<T>, query_idx: usize, target_idx: usize) {
        let query_vec: Vec<T> = data.row(query_idx).iter().cloned().collect();
        let target_vec: Vec<T> = data.row(target_idx).iter().cloned().collect();

        // True Euclidean distance
        let true_dist: f64 = query_vec
            .iter()
            .zip(target_vec.iter())
            .map(|(&a, &b)| {
                let diff = a.to_f64().unwrap() - b.to_f64().unwrap();
                diff * diff
            })
            .sum::<f64>()
            .sqrt();

        let encoded_query = self.quantiser.encode_query(&query_vec);
        let est_dist = self.rabitq_dist(&encoded_query, target_idx);

        // Component breakdown
        let v_dist = self.dist_to_centroid[target_idx].to_f64().unwrap();
        let q_dist = encoded_query.dist_to_centroid.to_f64().unwrap();
        let dot_corr = self.dot_corrections[target_idx].to_f64().unwrap();
        let qr = self.dot_query_binary(&encoded_query, target_idx);
        let popcount = self.popcount(target_idx);

        println!("Query {} -> Target {}", query_idx, target_idx);
        println!("  True dist:        {:.4}", true_dist);
        println!("  Estimated dist:   {:.4}", est_dist.to_f64().unwrap());
        println!("  v_dist (||v-c||): {:.4}", v_dist);
        println!("  q_dist (||q-c||): {:.4}", q_dist);
        println!("  dot_correction:   {:.4}", dot_corr);
        println!("  qr (q̄·r):         {}", qr);
        println!("  popcount:         {}", popcount);
        println!("  sum_quantised:    {}", encoded_query.sum_quantised);
        println!(
            "  lower:            {:.4}",
            encoded_query.lower.to_f64().unwrap()
        );
        println!(
            "  width:            {:.4}",
            encoded_query.width.to_f64().unwrap()
        );
    }
}
