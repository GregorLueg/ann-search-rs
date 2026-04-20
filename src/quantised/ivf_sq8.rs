//! Inverted file SQ8 index: quantises the original data to scalar quantisation
//! to 8 bit (i8) and uses Voronoi cells to identify the most interesting
//! candidates.

use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::{
    collections::BinaryHeap,
    iter::Sum,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use thousands::*;

use crate::prelude::*;
use crate::quantised::quantisers::*;
use crate::utils::k_means_utils::*;
use crate::utils::*;

////////////////
// Main index //
////////////////

/// IVF index quantised to scalar 8 bits
pub struct IvfSq8Index<T> {
    /// The original vectors quantised to `i8`.
    quantised_vectors: Vec<i8>,
    /// The quantised norms (if metric is set to cosine).
    quantised_norms: Vec<i32>,
    /// The original dimensions
    dim: usize,
    /// Number of samples in the index
    n: usize,
    /// The chosen distance metric
    metric: Dist,
    /// The centrois of the each k-mean cluster
    centroids: Vec<T>,
    /// Norms of the centroids - not relevant for this index.
    centroids_norm: Vec<T>,
    /// Vector indices for each cluster (in a flat structure)
    all_indices: Vec<usize>,
    /// Offsets of the elements of each inverted list.
    offsets: Vec<usize>,
    /// The codebook that contains the information of the quantisation.
    codebook: ScalarQuantiser<T>,
    /// Number of k-means clusters.
    nlist: usize,
    /// Original indices
    original_ids: Vec<usize>,
}

//////////////////////
// VectorDistanceSq //
//////////////////////

impl<T> VectorDistanceSq8<T> for IvfSq8Index<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    /// Return the flat vectors (quantised)
    fn vectors_flat_quantised(&self) -> &[i8] {
        &self.quantised_vectors
    }

    /// Return the original dimensions
    fn dim(&self) -> usize {
        self.dim
    }

    /// Return the normalised values (quantised) for the Cosine calculation
    fn norms_quantised(&self) -> &[i32] {
        &self.quantised_norms
    }
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for IvfSq8Index<T>
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

////////////////
// Main index //
////////////////

impl<T> IvfSq8Index<T>
where
    T: AnnSearchFloat,
{
    /// Build an IVF index with scalar 8-bit quantisation.
    ///
    /// Constructs an inverted file index with all vectors quantised to i8 using
    /// a global codebook. Reduces memory by 4x (for f32) whilst maintaining
    /// reasonable recall through learned quantisation bounds. Also, enables
    /// fast querying via `i8` symmetric transformations (at the cost of
    /// Recall).
    ///
    /// ### Workflow
    ///
    /// 1. Normalises vectors if using Cosine distance
    /// 2. Subsamples 250k vectors for training if dataset exceeds 500k
    /// 3. Runs k-means clustering to find nlist centroids
    /// 4. Trains global scalar quantiser on training data
    /// 5. Assigns all vectors to nearest centroid in parallel
    /// 6. Quantises all vectors using the global codebook
    /// 7. Builds CSR layout grouping vectors by cluster
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `nlist` - Optional number of clusters. Defaults to `sqrt(n)`.
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `max_iters` - Optional maximum k-means iterations (defaults to `50`).
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print training progress
    ///
    /// ### Returns
    ///
    /// Constructed quantised index ready for querying
    pub fn build(
        data: MatRef<T>,
        nlist: Option<usize>,
        metric: Dist,
        max_iters: Option<usize>,
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
            println!("  Generating IVF-SQ8 index with {} Voronoi cells.", nlist);
        }

        // 2. train centroids
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

        // 3. train global codebook
        if verbose {
            println!("  Training global codebook");
        }
        let codebook = ScalarQuantiser::train(&training_data, dim);

        // 4. assign vectors to clusters
        let data_norms = vec![T::one(); n];
        let centroid_norms = vec![T::one(); nlist];
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

        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        // 5. quantise all vectors with global codebook
        if verbose {
            println!("  Quantising vectors");
        }
        let mut quantised_vectors = vec![0i8; n * dim];

        quantised_vectors
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(vec_idx, chunk)| {
                let vec_start = vec_idx * dim;
                let vec = &vectors_flat[vec_start..vec_start + dim];
                let quantised = codebook.encode(vec);
                chunk.copy_from_slice(&quantised);
            });

        let quantised_norms: Vec<i32> = if metric == Dist::Cosine {
            quantised_vectors
                .par_chunks(dim)
                .map(|chunk| chunk.iter().map(|&v| v as i32 * v as i32).sum())
                .collect()
        } else {
            Vec::new()
        };

        if verbose {
            println!("  Quantisation complete");
        }

        let mut idx = Self {
            quantised_vectors,
            centroids,
            all_indices,
            offsets,
            codebook,
            dim,
            quantised_norms,
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

    /// Query the index for approximate nearest neighbours.
    ///
    /// Performs two-stage search using quantised vectors: first finds nprobe
    /// nearest centroids, then computes distances in quantised space (`i8`
    /// arithmetic) for all vectors in those clusters. Normalises query if
    /// using Cosine distance.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search. Defaults to 20% of nlist
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize, nprobe: Option<usize>) -> (Vec<usize>, Vec<T>) {
        let mut query_vec = query_vec.to_vec();

        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k = k.min(self.n);

        if self.metric == Dist::Cosine {
            normalise_vector(&mut query_vec);
        }

        // Find top nprobe centroids
        let cluster_scores: Vec<(T, usize)> = self.get_centroids_prenorm(&query_vec, nprobe);

        let query_i8 = self.codebook.encode(&query_vec);
        let query_norm_sq: i32 = query_i8.iter().map(|&q| q as i32 * q as i32).sum();

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for &(_, cluster_idx) in cluster_scores.iter().take(nprobe) {
            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];

            for vec_idx in start..end {
                let dist = match self.metric {
                    Dist::Cosine => self.cosine_distance_i8(vec_idx, &query_i8, query_norm_sq),
                    Dist::Euclidean => self.euclidean_distance_i8(vec_idx, &query_i8),
                };

                if heap.len() < k {
                    heap.push((OrderedFloat(dist), vec_idx));
                } else if dist < heap.peek().unwrap().0 .0 {
                    heap.pop();
                    heap.push((OrderedFloat(dist), vec_idx));
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

    /// Query using a matrix row reference.
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies
    /// to a temporary vector. Uses `self.query()` under the hood.
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search. Defaults to 20% of nlist
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
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

    /// Query using an already-quantised internal vector
    ///
    /// Skips the encode step since the vector is already in i8 format.
    /// Only decodes for centroid search (which is O(nlist), small).
    #[inline]
    fn query_quantised(
        &self,
        query_i8: &[i8],
        query_norm_sq: i32,
        k: usize,
        nprobe: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k = k.min(self.n);

        // Decode only for centroid search (O(nlist) - cheap)
        let query_float = self.codebook.decode(query_i8);
        let cluster_scores: Vec<(T, usize)> = self.get_centroids_prenorm(&query_float, nprobe);

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for &(_, cluster_idx) in cluster_scores.iter().take(nprobe) {
            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];

            for vec_idx in start..end {
                let dist = match self.metric {
                    Dist::Cosine => self.cosine_distance_i8(vec_idx, query_i8, query_norm_sq),
                    Dist::Euclidean => self.euclidean_distance_i8(vec_idx, query_i8),
                };

                if heap.len() < k {
                    heap.push((OrderedFloat(dist), vec_idx));
                } else if dist < heap.peek().unwrap().0 .0 {
                    heap.pop();
                    heap.push((OrderedFloat(dist), vec_idx));
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

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete
    /// kNN graph. Uses pre-quantised vectors directly, avoiding encode overhead.
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
        let counter = Arc::new(AtomicUsize::new(0));

        let unordered_results: Vec<(usize, Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dim;
                let end = start + self.dim;
                let query_i8 = &self.quantised_vectors[start..end];
                let query_norm_sq = if self.metric == Dist::Cosine {
                    self.quantised_norms[i]
                } else {
                    0
                };
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

                let (indices, dists) = self.query_quantised(query_i8, query_norm_sq, k, nprobe);
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
            + self.quantised_vectors.capacity() * std::mem::size_of::<i8>()
            + self.quantised_norms.capacity() * std::mem::size_of::<i32>()
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

        let mut new_vectors = Vec::with_capacity(self.quantised_vectors.len());
        let mut new_norms = if self.quantised_norms.is_empty() {
            Vec::new()
        } else {
            Vec::with_capacity(self.n)
        };

        for &old_id in &new_to_old {
            let start = old_id * self.dim;
            new_vectors.extend_from_slice(&self.quantised_vectors[start..start + self.dim]);
            if !self.quantised_norms.is_empty() {
                new_norms.push(self.quantised_norms[old_id]);
            }
        }

        self.quantised_vectors = new_vectors;
        self.quantised_norms = new_norms;
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
        let index =
            IvfSq8Index::build(data.as_ref(), Some(2), Dist::Euclidean, Some(10), 42, false);

        assert_eq!(index.dim, 32);
        assert_eq!(index.n, 6);
        assert_eq!(index.nlist, 2);
        assert_eq!(index.metric, Dist::Euclidean);
        assert_eq!(index.quantised_vectors.len(), 192);
        assert_eq!(index.centroids.len(), 64);
        assert_eq!(index.offsets.len(), 3);
    }

    #[test]
    fn test_build_cosine() {
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(data.as_ref(), Some(2), Dist::Cosine, Some(10), 42, false);

        assert_eq!(index.metric, Dist::Cosine);
        assert_eq!(index.quantised_norms.len(), 6);
    }

    #[test]
    fn test_query_returns_k_results() {
        let data = create_simple_dataset();
        let index =
            IvfSq8Index::build(data.as_ref(), Some(2), Dist::Euclidean, Some(10), 42, false);

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, distances) = index.query(&query, 3, Some(2));

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_query_k_exceeds_n() {
        let data = create_simple_dataset();
        let index =
            IvfSq8Index::build(data.as_ref(), Some(2), Dist::Euclidean, Some(10), 42, false);

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, _) = index.query(&query, 100, None);

        assert!(indices.len() <= 6);
    }

    #[test]
    fn test_query_finds_nearest() {
        let data = create_simple_dataset();
        let index =
            IvfSq8Index::build(data.as_ref(), Some(2), Dist::Euclidean, Some(10), 42, false);

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, distances) = index.query(&query, 3, Some(2));

        assert_eq!(indices[0], 0);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_query_cosine() {
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(data.as_ref(), Some(2), Dist::Cosine, Some(10), 42, false);

        let query: Vec<f32> = (0..32).map(|x| if x < 16 { 1.0 } else { 0.0 }).collect();
        let (indices, distances) = index.query(&query, 3, Some(2));

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_query_different_nprobe() {
        let data = create_simple_dataset();
        let index =
            IvfSq8Index::build(data.as_ref(), Some(2), Dist::Euclidean, Some(10), 42, false);

        let query: Vec<f32> = (0..32).map(|x| 5.0 + x as f32 * 0.01).collect();

        let (indices1, _) = index.query(&query, 3, Some(1));
        let (indices2, _) = index.query(&query, 3, Some(2));

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_query_deterministic() {
        let data = create_simple_dataset();
        let index =
            IvfSq8Index::build(data.as_ref(), Some(2), Dist::Euclidean, Some(10), 42, false);

        let query: Vec<f32> = (0..32).map(|x| 0.5 + x as f32 * 0.01).collect();

        let (indices1, distances1) = index.query(&query, 3, Some(2));
        let (indices2, distances2) = index.query(&query, 3, Some(2));

        assert_eq!(indices1, indices2);
        assert_eq!(distances1, distances2);
    }

    #[test]
    fn test_query_row() {
        let data = create_simple_dataset();
        let index =
            IvfSq8Index::build(data.as_ref(), Some(2), Dist::Euclidean, Some(10), 42, false);

        let query_mat = Mat::<f32>::from_fn(1, 32, |_, j| 0.5 + j as f32 * 0.01);
        let row = query_mat.row(0);

        let (indices, distances) = index.query_row(row, 3, Some(2));

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_build_large_nlist() {
        let data = Mat::from_fn(100, 8, |i, j| (i + j) as f32);

        let index =
            IvfSq8Index::build(data.as_ref(), Some(10), Dist::Euclidean, Some(5), 42, false);

        assert_eq!(index.nlist, 10);
        assert_eq!(index.offsets.len(), 11);
    }

    #[test]
    fn test_quantisation_preserves_structure() {
        let data = create_simple_dataset();
        let index =
            IvfSq8Index::build(data.as_ref(), Some(2), Dist::Euclidean, Some(10), 42, false);

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, _) = index.query(&query, 1, Some(2));

        assert_eq!(indices[0], 0);
    }
}
