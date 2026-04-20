//! Inverted file PQ index: quantises the original data via product
//! quantisation and uses Voronoi cells to identify the most interesting
//! candidates.

use faer::{MatRef, RowRef};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use thousands::*;

use crate::prelude::*;
use crate::quantised::quantisers::*;
use crate::utils::k_means_utils::*;
use crate::utils::*;

////////////////
// Main index //
////////////////

/// IVF index with product quantisation
pub struct IvfPqIndex<T> {
    /// Encoded vectors (M u8 codes per vector)
    quantised_codes: Vec<u8>,
    /// Original dimensions
    dim: usize,
    /// Number of samples in the index
    n: usize,
    /// Distance metric
    metric: Dist,
    /// K-means cluster centroids
    centroids: Vec<T>,
    /// Norms of the centroids - not relevant for this index.
    centroids_norm: Vec<T>,
    /// Vector indices for each cluster (CSR format)
    all_indices: Vec<usize>,
    /// Offsets for each inverted list
    offsets: Vec<usize>,
    /// Product quantiser with M codebooks
    codebook: ProductQuantiser<T>,
    /// Number of k-means clusters
    nlist: usize,
    /// Original indices
    original_ids: Vec<usize>,
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for IvfPqIndex<T>
where
    T: AnnSearchFloat,
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

impl<T> VectorDistanceAdc<T> for IvfPqIndex<T>
where
    T: AnnSearchFloat,
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

impl<T> IvfPqIndex<T>
where
    T: AnnSearchFloat,
{
    /// Build an IVF index with product quantisation
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `nlist` - Optional number of clusters. Defaults to `sqrt(n)`.
    /// * `m` - Number of subspaces for PQ (dim must be divisible by m)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `max_iters` - Optional maximum k-means iterations (defaults to `50`).
    /// * `n_pq_centroids` - Number of centroids to use for the product
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
        n_pq_centroids: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let (mut vectors_flat, n, dim) = matrix_to_flat(data);

        let max_iters = max_iters.unwrap_or(50);
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

        // 1. subsample training data
        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        if verbose {
            println!("  Generating IVF-PQ index with {} Voronoi cells.", nlist);
        }

        // 2. train IVF centroids
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

        // normalise centroids for cosine
        if metric == Dist::Cosine {
            if verbose {
                println!("  Normalising centroids");
            }
            centroids
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        // 3. compute residuals for training data
        if verbose {
            println!("  Computing residuals for PQ training");
        }
        let training_norms = vec![T::one(); n_train];
        let centroid_norms = vec![T::one(); nlist];
        let training_assignments = assign_all_parallel(
            &training_data,
            &training_norms,
            dim,
            n_train,
            &centroids,
            &centroid_norms,
            nlist,
            &metric,
        );

        let mut training_residuals = Vec::with_capacity(training_data.len());
        for (vec_idx, &cluster_id) in training_assignments.iter().enumerate() {
            let vec_start = vec_idx * dim;
            let vec = &training_data[vec_start..vec_start + dim];
            let centroid = &centroids[cluster_id * dim..(cluster_id + 1) * dim];
            let residuals = T::subtract_simd(vec, centroid);
            training_residuals.extend_from_slice(&residuals);
        }

        // 4. train PQ on residuals
        if verbose {
            println!("  Training product quantiser with m={}", m);
        }
        let codebook = ProductQuantiser::train(
            &training_residuals,
            dim,
            m,
            n_pq_centroids,
            &Dist::Euclidean,
            max_iters,
            seed + 1000,
            verbose,
        );

        // 5. assign all vectors to IVF clusters
        let data_norms = vec![T::one(); n];
        let assignments = assign_all_parallel(
            &vectors_flat,
            &data_norms,
            dim,
            n,
            &centroids,
            &centroid_norms,
            nlist,
            &metric,
        );

        if verbose {
            print_cluster_summary(&assignments, nlist);
        }

        let (all_indices, offsets) = build_csr_layout(assignments.clone(), n, nlist);

        // 6. encode all vectors with product quantiser
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

                let residual = T::subtract_simd(vec, centroid);

                let codes = codebook.encode(&residual);
                chunk.copy_from_slice(&codes);
            });

        if verbose {
            println!("  Quantisation complete");
        }

        let mut idx = Self {
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
            original_ids: Vec::new(),
        };

        let new_to_old = idx.optimise_memory_layout();
        idx.original_ids = new_to_old;
        idx
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
    /// Tuple of (indices, distances) sorted by
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize, nprobe: Option<usize>) -> (Vec<usize>, Vec<T>) {
        let mut query_vec = query_vec.to_vec();

        if self.metric == Dist::Cosine {
            normalise_vector(&mut query_vec);
        }

        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k = k.min(self.n);

        // Find top nprobe centroids
        let cluster_scores: Vec<(T, usize)> = self.get_centroids_prenorm(&query_vec, nprobe);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for &(_, cluster_idx) in cluster_scores.iter().take(nprobe) {
            let lookup_tables = self.build_lookup_tables_residual(&query_vec, cluster_idx);

            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];

            let mut worst_dist = if heap.len() >= k {
                heap.peek().unwrap().0 .0
            } else {
                T::infinity()
            };

            for vec_idx in start..end {
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

        let (distances, indices) = results
            .into_iter()
            .map(|(d, i)| (d.0, self.original_ids[i]))
            .unzip();

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
    #[inline]
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

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Reconstructs each vector (centroid + decoded residual) and uses ADC
    /// for accurate distance computation across clusters.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `nprobe` - Number of clusters to search (defaults to sqrt(nlist) if None)
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
        nprobe: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        let m = self.codebook.m();
        let counter = Arc::new(AtomicUsize::new(0));

        let mut cluster_assignments = vec![0usize; self.n];
        for cluster_idx in 0..self.nlist {
            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];
            for new_idx in start..end {
                cluster_assignments[new_idx] = cluster_idx;
            }
        }

        let unordered_results: Vec<(usize, Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let my_cluster = cluster_assignments[i];
                let codes = &self.quantised_codes[i * m..(i + 1) * m];

                let my_centroid =
                    &self.centroids[my_cluster * self.dim..(my_cluster + 1) * self.dim];
                let residual = self.codebook.decode(codes);
                let reconstructed: Vec<T> = T::add_simd(my_centroid, &residual);

                let orig_id = self.original_ids[i];

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

                let (indices, dists) = self.query(&reconstructed, k, nprobe);
                (orig_id, indices, dists)
            })
            .collect();

        let mut final_indices = vec![Vec::new(); self.n];
        let mut final_dists = if return_dist {
            Some(vec![Vec::new(); self.n])
        } else {
            None
        };

        for (orig_id, indices, dists) in unordered_results {
            final_indices[orig_id] = indices;
            if let Some(ref mut fd) = final_dists {
                fd[orig_id] = dists;
            }
        }

        (final_indices, final_dists)
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.quantised_codes.capacity() * std::mem::size_of::<u8>()
            + self.centroids.capacity() * std::mem::size_of::<T>()
            + self.centroids_norm.capacity() * std::mem::size_of::<T>()
            + self.all_indices.capacity() * std::mem::size_of::<usize>()
            + self.offsets.capacity() * std::mem::size_of::<usize>()
            + self.codebook.memory_usage_bytes()
    }

    /// Function will optimise memory layout and put vectors sorted by cluster
    /// id
    ///
    /// ### Returns
    ///
    /// New to old mapping
    fn optimise_memory_layout(&mut self) -> Vec<usize> {
        let m = self.codebook.m();
        let mut new_to_old = Vec::with_capacity(self.n);
        let mut old_to_new = vec![0usize; self.n];

        for cluster in 0..self.nlist {
            let start = self.offsets[cluster];
            let end = self.offsets[cluster + 1];
            for &old_id in &self.all_indices[start..end] {
                old_to_new[old_id] = new_to_old.len();
                new_to_old.push(old_id);
            }
        }

        let mut new_codes = Vec::with_capacity(self.quantised_codes.len());
        for &old_id in &new_to_old {
            let start = old_id * m;
            new_codes.extend_from_slice(&self.quantised_codes[start..start + m]);
        }

        self.quantised_codes = new_codes;
        self.all_indices.clear();
        self.all_indices.shrink_to_fit();

        new_to_old
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
        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
            Some(4),
            42,
            false,
        );

        assert_eq!(index.dim, 32);
        assert_eq!(index.n, 6);
        assert_eq!(index.nlist, 2);
        assert_eq!(index.metric, Dist::Euclidean);
        assert_eq!(index.quantised_codes.len(), 48); // 6 vectors * 8 subspaces
        assert_eq!(index.centroids.len(), 64); // 2 clusters * 32 dims
        assert_eq!(index.offsets.len(), 3);
    }

    #[test]
    fn test_build_cosine() {
        let data = create_simple_dataset();
        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Cosine,
            Some(10),
            Some(4),
            42,
            false,
        );

        assert_eq!(index.metric, Dist::Cosine);
    }

    #[test]
    fn test_query_returns_k_results() {
        let data = create_simple_dataset();
        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
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
        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
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
        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
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
        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Cosine,
            Some(10),
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
        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
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
        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
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
        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
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

        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(2),
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
        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let table = index.build_lookup_tables_residual(&query, 0);

        // M * n_centroids = 8 * 4
        assert_eq!(table.len(), 32);
    }

    #[test]
    fn test_compute_distance_adc() {
        let data = create_simple_dataset();
        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Euclidean,
            Some(10),
            Some(4),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let table = index.build_lookup_tables_residual(&query, 0);

        let dist = index.compute_distance_adc(0, &table);

        assert!(dist >= 0.0);
    }

    #[test]
    fn test_residual_encoding() {
        let data = Mat::from_fn(50, 32, |i, j| (i + j) as f32);

        let index = IvfPqIndex::build(
            data.as_ref(),
            Some(5),
            8,
            Dist::Euclidean,
            Some(5),
            Some(4),
            42,
            false,
        );

        // Query with vector from dataset
        let query: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let (indices, _) = index.query(&query, 1, Some(5));

        // Should find exact or very close match
        assert_eq!(indices[0], 0);
    }
}
