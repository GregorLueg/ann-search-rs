use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::ops::AddAssign;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::{collections::BinaryHeap, iter::Sum};
use thousands::*;

use crate::prelude::*;
use crate::quantised::quantisers::*;
use crate::utils::ivf_utils::sample_vectors;
use crate::utils::*;

/// Exhaustive PQ index
///
/// ### Fields
///
/// * `quantised_codes` - Encoded vectors (M u8 codes per vector)
/// * `dim` - Original dimensions
/// * `n` - Number of samples in the index
/// * `metric` - Distance metric
/// * `codebook` - Product quantiser with M codebooks
pub struct ExhaustiveOpqIndex<T> {
    quantised_codes: Vec<u8>,
    dim: usize,
    n: usize,
    metric: Dist,
    codebook: OptimisedProductQuantiser<T>,
}

///////////////////////
// VectorDistanceAdc //
///////////////////////

impl<T> VectorDistanceAdc<T> for ExhaustiveOpqIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + AddAssign + SimdDistance,
{
    fn codebook_m(&self) -> usize {
        self.codebook.m()
    }

    fn codebook_n_centroids(&self) -> usize {
        self.codebook.n_centroids()
    }

    fn codebook_subvec_dim(&self) -> usize {
        self.codebook.subvec_dim()
    }

    fn centroids(&self) -> &[T] {
        &[]
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn quantised_codes(&self) -> &[u8] {
        &self.quantised_codes
    }

    fn codebooks(&self) -> &[Vec<T>] {
        self.codebook.codebooks()
    }
}

/////////////////////////
// Main implementation //
/////////////////////////

impl<T> ExhaustiveOpqIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + AddAssign,
{
    /// Build an exhaustive product quantiser index
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `m` - Number of subspaces for PQ (dim must be divisible by m)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `max_iters` - Optional maximum k-means iterations (defaults to `30`)
    /// * `n_pq_centroids` - Number of centroids to use for the product
    ///   quantisation. If not provided, it uses the default `256`
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Index ready for querying
    pub fn build(
        data: MatRef<T>,
        m: usize,
        metric: Dist,
        max_iters: Option<usize>,
        n_pq_centroids: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let (mut vectors_flat, n, dim) = matrix_to_flat(data);
        let max_iters = max_iters.unwrap_or(30);

        if metric == Dist::Cosine {
            vectors_flat
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        let (training_data, _) = if n > 500_000 {
            let (data, _) = sample_vectors(&vectors_flat, dim, n, 250_000, seed);
            (data, 250_000)
        } else {
            (vectors_flat.clone(), n)
        };

        let codebook = OptimisedProductQuantiser::train(
            &training_data,
            dim,
            m,
            n_pq_centroids,
            None,
            max_iters,
            seed,
            verbose,
        );

        let mut quantised_codes = vec![0u8; n * m];
        quantised_codes
            .par_chunks_mut(m)
            .enumerate()
            .for_each(|(vec_idx, chunk)| {
                let vec_start = vec_idx * dim;
                let vec = &vectors_flat[vec_start..vec_start + dim];
                let codes = codebook.encode(vec);
                chunk.copy_from_slice(&codes);
            });

        Self {
            quantised_codes,
            codebook,
            dim,
            n,
            metric,
        }
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Uses asymmetric distance computation (ADC): builds lookup tables
    /// from query subvectors to all centroids, then computes distances
    /// via table lookups. Performs exhaustive search over all vectors.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances) sorted by distance
    pub fn query(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        let mut query_vec = query_vec.to_vec();
        if self.metric == Dist::Cosine {
            normalise_vector(&mut query_vec);
        }

        let lookup_tables = self.build_lookup_tables_direct(&query_vec);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for vec_idx in 0..self.n {
            let dist = self.compute_distance_adc(vec_idx, &lookup_tables);

            if heap.len() < k {
                heap.push((OrderedFloat(dist), vec_idx));
            } else if dist < heap.peek().unwrap().0 .0 {
                heap.pop();
                heap.push((OrderedFloat(dist), vec_idx));
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);
        let (distances, indices) = results.into_iter().map(|(d, i)| (d.0, i)).unzip();
        (indices, distances)
    }

    /// Query using a matrix row reference
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances) sorted by distance
    pub fn query_row(&self, query_row: RowRef<T>, k: usize) -> (Vec<usize>, Vec<T>) {
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
    /// Reconstructs each vector from PQ codes and performs exhaustive
    /// search to build a complete kNN graph.
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
        let m = self.codebook.m();
        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let codes = &self.quantised_codes[i * m..(i + 1) * m];

                // Reconstruct by decoding PQ codes
                let reconstructed = self.codebook.decode(codes);

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

                self.query(&reconstructed, k)
            })
            .collect();

        if return_dist {
            let (indices, distances) = results.into_iter().unzip();
            (indices, Some(distances))
        } else {
            let indices = results.into_iter().map(|(idx, _)| idx).collect();
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
            + self.quantised_codes.capacity() * std::mem::size_of::<u8>()
            + self.codebook.memory_usage_bytes()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn create_simple_dataset() -> Mat<f32> {
        let mut data = Vec::new();
        // Create 6 vectors of 32 dimensions
        // First 3 near origin
        for i in 0..3 {
            for j in 0..32 {
                data.push(i as f32 * 0.1 + j as f32 * 0.01);
            }
        }
        // Next 3 far from origin
        for i in 0..3 {
            for j in 0..32 {
                data.push(10.0 + i as f32 * 0.1 + j as f32 * 0.01);
            }
        }
        Mat::from_fn(6, 32, |i, j| data[i * 32 + j])
    }

    #[test]
    fn test_build_euclidean() {
        let data = create_simple_dataset();
        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(10),
            Some(4),
            42,
            false,
        );

        assert_eq!(index.dim, 32);
        assert_eq!(index.n, 6);
        assert_eq!(index.metric, Dist::Euclidean);
        assert_eq!(index.quantised_codes.len(), 48); // 6 vectors * 8 subspaces
    }

    #[test]
    fn test_build_cosine() {
        let data = create_simple_dataset();
        let index =
            ExhaustiveOpqIndex::build(data.as_ref(), 8, Dist::Cosine, Some(10), Some(4), 42, false);

        assert_eq!(index.metric, Dist::Cosine);
    }

    #[test]
    fn test_query_returns_k_results() {
        let data = create_simple_dataset();
        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(10),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, distances) = index.query(&query, 3);

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_query_k_exceeds_n() {
        let data = create_simple_dataset();
        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(10),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, _) = index.query(&query, 100);

        assert_eq!(indices.len(), 6);
    }

    #[test]
    fn test_query_distances_sorted() {
        let data = create_simple_dataset();
        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(10),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (_, distances) = index.query(&query, 3);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_query_cosine() {
        let data = create_simple_dataset();
        let index =
            ExhaustiveOpqIndex::build(data.as_ref(), 8, Dist::Cosine, Some(10), Some(4), 42, false);

        let query: Vec<f32> = (0..32).map(|x| if x < 16 { 1.0 } else { 0.0 }).collect();
        let (indices, distances) = index.query(&query, 3);

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_query_deterministic() {
        let data = create_simple_dataset();
        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(10),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| 0.5 + x as f32 * 0.01).collect();

        let (indices1, distances1) = index.query(&query, 3);
        let (indices2, distances2) = index.query(&query, 3);

        assert_eq!(indices1, indices2);
        assert_eq!(distances1, distances2);
    }

    #[test]
    fn test_query_row() {
        let data = create_simple_dataset();
        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(10),
            Some(4),
            42,
            false,
        );

        let query_mat = Mat::<f32>::from_fn(1, 32, |_, j| 0.5 + j as f32 * 0.01);
        let row = query_mat.row(0);

        let (indices, distances) = index.query_row(row, 3);

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_build_different_m() {
        let data = Mat::from_fn(20, 32, |i, j| (i + j) as f32);

        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(5),
            Some(4),
            42,
            false,
        );

        assert_eq!(index.codebook.m(), 8);
        assert_eq!(index.codebook.subvec_dim(), 4);
        assert_eq!(index.quantised_codes.len(), 160); // 20 vectors * 8 subspaces
    }

    #[test]
    fn test_build_lookup_tables() {
        let data = create_simple_dataset();
        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(10),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let table = index.build_lookup_tables_direct(&query);

        // M * n_centroids = 8 * 4
        assert_eq!(table.len(), 32);
    }

    #[test]
    fn test_compute_distance_adc() {
        let data = create_simple_dataset();
        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(10),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let table = index.build_lookup_tables_direct(&query);

        let dist = index.compute_distance_adc(0, &table);

        assert!(dist >= 0.0);
    }

    #[test]
    fn test_encoding_reconstruction() {
        let data = Mat::from_fn(50, 32, |i, j| (i + j) as f32);

        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(5),
            Some(4),
            42,
            false,
        );

        // Query with vector from dataset
        let query: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let (indices, _) = index.query(&query, 1);

        // Should find exact or very close match
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_memory_usage() {
        let data = create_simple_dataset();
        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(10),
            Some(4),
            42,
            false,
        );

        let memory = index.memory_usage_bytes();
        assert!(memory > 0);
    }

    #[test]
    fn test_generate_knn() {
        let data = Mat::from_fn(10, 32, |i, j| (i + j) as f32);

        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(5),
            Some(4),
            42,
            false,
        );

        let (knn_indices, knn_distances) = index.generate_knn(3, true, false);

        assert_eq!(knn_indices.len(), 10);
        assert!(knn_distances.is_some());

        let distances = knn_distances.unwrap();
        assert_eq!(distances.len(), 10);

        for i in 0..10 {
            assert_eq!(knn_indices[i].len(), 3);
            assert_eq!(distances[i].len(), 3);
        }
    }

    #[test]
    fn test_generate_knn_without_distances() {
        let data = Mat::from_fn(10, 32, |i, j| (i + j) as f32);

        let index = ExhaustiveOpqIndex::build(
            data.as_ref(),
            8,
            Dist::Euclidean,
            Some(5),
            Some(4),
            42,
            false,
        );

        let (knn_indices, knn_distances) = index.generate_knn(3, false, false);

        assert_eq!(knn_indices.len(), 10);
        assert!(knn_distances.is_none());

        for i in 0..10 {
            assert_eq!(knn_indices[i].len(), 3);
        }
    }

    #[test]
    fn test_different_m_values() {
        let data = Mat::from_fn(20, 64, |i, j| (i + j) as f32);

        // Test m=4
        let index_4 = ExhaustiveOpqIndex::build(
            data.as_ref(),
            4,
            Dist::Euclidean,
            Some(5),
            Some(4),
            42,
            false,
        );
        assert_eq!(index_4.codebook.m(), 4);
        assert_eq!(index_4.codebook.subvec_dim(), 16);

        // Test m=16
        let index_16 = ExhaustiveOpqIndex::build(
            data.as_ref(),
            16,
            Dist::Euclidean,
            Some(5),
            Some(4),
            42,
            false,
        );
        assert_eq!(index_16.codebook.m(), 16);
        assert_eq!(index_16.codebook.subvec_dim(), 4);
    }
}
