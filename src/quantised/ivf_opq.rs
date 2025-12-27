use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::ops::AddAssign;
use std::{collections::BinaryHeap, iter::Sum};

use crate::quantised::quantisers::*;
use crate::utils::dist::*;
use crate::utils::heap_structs::*;
use crate::utils::ivf_utils::*;
use crate::utils::*;

////////////////
// Main index //
////////////////

/// IVF index with optimised product quantisation
///
/// ### Fields
///
/// * `quantised_codes` - Encoded vectors (M u8 codes per vector)
/// * `dim` - Original dimensions
/// * `n` - Number of samples in the index
/// * `metric` - Distance metric
/// * `centroids` - K-means cluster centroids
/// * `centroids_norm` - Norms of the centroids - not relevant for this index.
/// * `all_indices` - Vector indices for each cluster (CSR format)
/// * `offsets` - Offsets for each inverted list
/// * `codebook` - Product quantiser with M codebooks
/// * `nlist` - Number of k-means clusters
pub struct IvfOpqIndex<T> {
    quantised_codes: Vec<u8>,
    dim: usize,
    n: usize,
    metric: Dist,
    centroids: Vec<T>,
    centroids_norm: Vec<T>,
    all_indices: Vec<usize>,
    offsets: Vec<usize>,
    codebook: OptimisedProductQuantiser<T>,
    nlist: usize,
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for IvfOpqIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    fn centroids(&self) -> &[T] {
        &self.centroids
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> Dist {
        self.metric
    }

    fn nlist(&self) -> usize {
        self.nlist
    }

    fn centroids_norm(&self) -> &[T] {
        &self.centroids_norm
    }
}

///////////////////////
// VectorDistanceAdc //
///////////////////////

impl<T> VectorDistanceAdc<T> for IvfOpqIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + AddAssign,
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
        &self.centroids
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

////////////////
// Main index //
////////////////

impl<T> IvfOpqIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + AddAssign,
{
    /// Build an IVF index with optimised product quantisation
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Flattened vector data (length = n * dim)
    /// * `dim` - Embedding dimensions
    /// * `n` - Number of vectors
    /// * `nlist` - Number of IVF clusters
    /// * `m` - Number of subspaces for PQ (dim must be divisible by m)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `max_iters` - Optional maximum k-means iterations. Defaults to `30`.
    /// * `opq_iter` - Optional number of iterations to get the rotation matrix.
    ///   Defaults to `3`.
    /// * `n_opq_centroids` - Number of centroids to use for the product
    ///   quantisation. If not provided, it uses the default `256`.
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Index ready for querying
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: MatRef<T>,
        nlist: Option<usize>,
        m: usize,
        metric: Dist,
        max_iters: Option<usize>,
        opq_iter: Option<usize>,
        n_opq_centroids: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let (mut vectors_flat, n, dim) = matrix_to_flat(data);

        let max_iters = max_iters.unwrap_or(30);
        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        // normalise for cosine distance
        if metric == Dist::Cosine {
            if verbose {
                println!("  Normalising vectors for cosine distance");
            }
            vectors_flat
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        // 1. Subsample training data
        let (training_data, n_train) = if n > 500_000 {
            if verbose {
                println!("  Sampling 250k vectors for training");
            }
            let (data, _) = sample_vectors(&vectors_flat, dim, n, 250_000, seed);
            (data, 250_000)
        } else {
            (vectors_flat.clone(), n)
        };

        if verbose {
            println!("  Generating IVF-OPQ index with {} Voronoi cells.", nlist);
        }

        // 2. Train IVF centroids
        let mut centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            max_iters,
            seed,
            verbose,
        );

        // Normalise centroids for cosine
        if metric == Dist::Cosine {
            if verbose {
                println!("  Normalising centroids");
            }
            centroids
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        // 3. Compute residuals for training data
        if verbose {
            println!("  Computing residuals for OPQ training");
        }
        let training_assignments =
            assign_all_parallel(&training_data, dim, n_train, &centroids, nlist, &metric);

        // For normalised vectors (cosine), ||q-v||² = 2(1 - <q,v>) so Euclidean distance
        // on residuals correctly approximates cosine distance
        let mut training_residuals = Vec::with_capacity(training_data.len());
        for (vec_idx, &cluster_id) in training_assignments.iter().enumerate() {
            let vec_start = vec_idx * dim;
            let vec = &training_data[vec_start..vec_start + dim];
            let centroid = &centroids[cluster_id * dim..(cluster_id + 1) * dim];

            for d in 0..dim {
                training_residuals.push(vec[d] - centroid[d]);
            }
        }

        // 4. Train OPQ on residuals
        if verbose {
            println!("  Training optimised product quantiser with m={}", m);
        }
        let codebook = OptimisedProductQuantiser::train(
            &training_residuals,
            dim,
            m,
            n_opq_centroids,
            opq_iter,
            max_iters,
            seed + 1000,
            verbose,
        );

        // 5. Assign all vectors to IVF clusters
        let assignments = assign_all_parallel(&vectors_flat, dim, n, &centroids, nlist, &metric);
        let (all_indices, offsets) = build_csr_layout(assignments.clone(), n, nlist);

        // 6. Encode all vectors with product quantiser
        if verbose {
            println!("  Encoding residuals");
        }
        let mut quantised_codes = vec![0u8; n * m];

        quantised_codes
            .par_chunks_mut(m)
            .zip(assignments.par_iter())
            .enumerate()
            .for_each(|(vec_idx, (chunk, &cluster_id))| {
                let vec_start = vec_idx * dim;
                let vec = &vectors_flat[vec_start..vec_start + dim];

                let centroid = &centroids[cluster_id * dim..(cluster_id + 1) * dim];
                let residual: Vec<T> = vec
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&v, &c)| v - c)
                    .collect();

                let codes = codebook.encode(&residual);
                chunk.copy_from_slice(&codes);
            });

        if verbose {
            println!("  (Optimised) Quantisation complete");
        }

        Self {
            quantised_codes,
            centroids,
            all_indices,
            offsets,
            codebook,
            dim,
            n,
            nlist,
            metric,
            centroids_norm: Vec::new(),
        }
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Uses asymmetric distance computation (ADC): builds lookup tables
    /// from query subvectors to all centroids, then computes distances
    /// via table lookups.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search (defaults to 15% of nlist)
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances) sorted by distance
    pub fn query(&self, query_vec: &[T], k: usize, nprobe: Option<usize>) -> (Vec<usize>, Vec<T>) {
        let mut query_vec = query_vec.to_vec();

        if self.metric == Dist::Cosine {
            normalise_vector(&mut query_vec);
        }

        let nprobe = nprobe.unwrap_or_else(|| (((self.nlist as f64) * 0.15) as usize).max(1));
        let k = k.min(self.n);

        // Find top nprobe centroids
        let cluster_scores: Vec<(T, usize)> = self.get_centroids_prenorm(&query_vec, nprobe);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for &(_, cluster_idx) in cluster_scores.iter().take(nprobe) {
            let lookup_tables = self.build_lookup_tables(&query_vec, cluster_idx);

            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];

            // early termination optimisation
            let mut worst_dist = if heap.len() >= k {
                heap.peek().unwrap().0 .0
            } else {
                T::infinity()
            };

            for &vec_idx in &self.all_indices[start..end] {
                let dist = self.compute_distance_adc(vec_idx, &lookup_tables);

                if dist >= worst_dist {
                    continue;
                }

                if heap.len() < k {
                    heap.push((OrderedFloat(dist), vec_idx));
                    if heap.len() == k {
                        worst_dist = heap.peek().unwrap().0 .0;
                    }
                } else {
                    heap.pop();
                    heap.push((OrderedFloat(dist), vec_idx));
                    worst_dist = heap.peek().unwrap().0 .0;
                }
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
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        nprobe: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, nprobe);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, nprobe)
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
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
            Some(1),
            Some(4),
            42,
            false,
        );

        assert_eq!(index.dim, 32);
        assert_eq!(index.n, 6);
        assert_eq!(index.nlist, 2);
        assert_eq!(index.metric, Dist::Euclidean);
        assert_eq!(index.quantised_codes.len(), 48);
        assert_eq!(index.centroids.len(), 64);
        assert_eq!(index.offsets.len(), 3);
    }

    #[test]
    fn test_build_cosine() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Cosine,
            Some(10),
            Some(1),
            Some(4),
            42,
            false,
        );

        assert_eq!(index.metric, Dist::Cosine);
    }

    #[test]
    fn test_query_returns_k_results() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
            Some(1),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, distances) = index.query(&query, 3, None);

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_query_k_exceeds_n() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
            Some(1),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, _) = index.query(&query, 100, None);

        assert!(indices.len() <= 6);
    }

    #[test]
    fn test_query_distances_sorted() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
            Some(1),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (_, distances) = index.query(&query, 3, Some(2));

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_query_cosine() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Cosine,
            Some(10),
            Some(1),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| if x < 16 { 1.0 } else { 0.0 }).collect();
        let (indices, distances) = index.query(&query, 3, None);

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_query_different_nprobe() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
            Some(1),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| 5.0 + x as f32 * 0.01).collect();

        let (indices1, _) = index.query(&query, 3, Some(1));
        let (indices2, _) = index.query(&query, 3, Some(2));

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_query_deterministic() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
            Some(1),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| 0.5 + x as f32 * 0.01).collect();

        let (indices1, distances1) = index.query(&query, 3, Some(2));
        let (indices2, distances2) = index.query(&query, 3, Some(2));

        assert_eq!(indices1, indices2);
        assert_eq!(distances1, distances2);
    }

    #[test]
    fn test_query_row() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
            Some(1),
            Some(4),
            42,
            false,
        );

        let query_mat = Mat::<f32>::from_fn(1, 32, |_, j| 0.5 + j as f32 * 0.01);
        let row = query_mat.row(0);

        let (indices, distances) = index.query_row(row, 3, None);

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_build_different_m() {
        let data = Mat::from_fn(20, 32, |i, j| (i + j) as f32);

        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(5),
            Some(1),
            Some(4),
            42,
            false,
        );

        assert_eq!(index.codebook.m(), 8);
        assert_eq!(index.codebook.subvec_dim(), 4);
        assert_eq!(index.quantised_codes.len(), 160);
    }

    #[test]
    fn test_build_lookup_tables() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
            Some(1),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let table = index.build_lookup_tables(&query, 0);

        assert_eq!(table.len(), 32);
    }

    #[test]
    fn test_compute_distance_adc() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
            Some(1),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let table = index.build_lookup_tables(&query, 0);

        let dist = index.compute_distance_adc(0, &table);

        assert!(dist >= 0.0);
    }

    #[test]
    fn test_opq_iterations() {
        let data = Mat::from_fn(50, 32, |i, j| (i + j) as f32);

        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(5),
            8,
            Dist::Euclidean,
            Some(5),
            Some(2),
            Some(4),
            42,
            false,
        );

        assert_eq!(index.codebook.m(), 8);
    }

    #[test]
    fn test_residual_encoding() {
        let data = Mat::from_fn(50, 32, |i, j| (i + j) as f32);

        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(5),
            8,
            Dist::Euclidean,
            Some(5),
            Some(1),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let (indices, _) = index.query(&query, 1, Some(5));

        assert_eq!(indices[0], 0);
    }
}
