//! Inverted file index with bf16 quantisation: quantises the original data to
//! bf16 (keeps the norms) and uses Voronoi cells to identify the most
//! interesting candidates.

use faer::{MatRef, RowRef};
use half::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::Separable;

use crate::prelude::*;
use crate::quantised::quantisers::*;
use crate::utils::k_means_utils::*;
use crate::utils::*;

/// IVF index with bf16 quantisation
///
/// Uses k-means clustering to partition vectors into nlist clusters. Each
/// cluster maintains an inverted list of vector indices assigned to it.
/// Queries search only the nprobe nearest clusters, trading perfect recall
/// for speed. This version leverages `bf16` quantisation under the hood,
/// reducing memory fingerprint and query speed at cost of precision.
pub struct IvfIndexBf16<T> {
    /// Original vector data, flattened for cache locality and quantised
    /// to bf16
    pub vectors_flat: Vec<bf16>,
    /// Embedding dimensions
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Pre-computed norms for Cosine distance (empty for Euclidean). Keep
    /// `T` here to avoid massive precision loss for Cosine.
    pub norms: Vec<T>,
    /// Distance metric (Euclidean or Cosine)
    metric: Dist,
    /// Cluster centroids (nlist * dim elements)
    centroids: Vec<T>,
    /// Cluster centroids norms for Cosine distance
    centroids_norm: Vec<T>,
    /// Vector indices for each cluster (in a flat structure)
    all_indices: Vec<usize>,
    /// Offsets of the elements of each inverted list
    offsets: Vec<usize>,
    /// Number of clusters in the index
    nlist: usize,
    /// Original ids
    original_ids: Vec<usize>,
}

///////////////////////
// VectorDistanceB16 //
///////////////////////

impl<T> VectorDistanceBf16<T> for IvfIndexBf16<T>
where
    T: AnnSearchFloat + Bf16Compatible,
{
    fn vectors_flat(&self) -> &[bf16] {
        &self.vectors_flat
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn norms(&self) -> &[T] {
        &self.norms
    }
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for IvfIndexBf16<T>
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

impl<T> IvfIndexBf16<T>
where
    T: AnnSearchFloat + Bf16Compatible,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Build an IVF index with optimised memory layout and parallel training.
    ///
    /// Constructs an inverted file index by clustering vectors with k-means,
    /// then assigns each vector to its nearest centroid. Uses a CSR (Compressed
    /// Sparse Row) layout for cache-efficient cluster traversal during search.
    ///
    /// ### Workflow
    ///
    /// 1. Subsamples 250k vectors for training if dataset exceeds 500k
    /// 2. Runs k-means clustering to find nlist centroids
    /// 3. Assigns all vectors to their nearest centroid in parallel
    /// 4. Builds CSR layout grouping vectors by cluster for locality
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `nlist` - Optional number of clusters. Defaults to `sqrt(n)`.
    /// * `max_iters` - Optional maximum k-means iterations (defaults to `50`).
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print training progress
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    pub fn build(
        data: MatRef<T>,
        metric: Dist,
        nlist: Option<usize>,
        max_iters: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let (vectors_flat, n, dim) = matrix_to_flat(data);

        // Compute norms for Cosine distance
        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    T::calculate_l2_norm(&vectors_flat[start..end])
                })
                .collect()
        } else {
            Vec::new()
        };

        let max_iters = max_iters.unwrap_or(50);
        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        // 1. subsample training data
        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        if verbose {
            println!("  Generating IVF index with {} Voronoi cells.", nlist);
        }

        // 2. train the centroids
        let centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            max_iters,
            seed,
            verbose,
        );

        let centroids_norm = if metric == Dist::Cosine {
            (0..nlist)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    centroids[start..end]
                        .iter()
                        .map(|x| *x * *x)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt()
                })
                .collect()
        } else {
            Vec::new()
        };

        // 3. assign the Rest
        let data_norms_for_assignment = if metric == Dist::Cosine {
            norms.clone()
        } else {
            vec![T::one(); n]
        };

        let assignments = assign_all_parallel(
            &vectors_flat,
            &data_norms_for_assignment,
            dim,
            n,
            &centroids,
            &centroids_norm,
            nlist,
            &metric,
        );

        if verbose {
            print_cluster_summary(&assignments, nlist);
        }

        // 4. generate a flat version for better cache locality
        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        let mut idx = Self {
            vectors_flat: encode_bf16_quantisation(&vectors_flat),
            dim,
            n,
            norms,
            metric,
            centroids,
            centroids_norm,
            all_indices,
            offsets,
            nlist,
            original_ids: Vec::new(),
        };

        let new_to_old = idx.optimise_memory_layout();
        idx.original_ids = new_to_old;
        idx
    }

    ///////////
    // Query //
    ///////////

    /// Query the index for approximate nearest neighbours
    ///
    /// Performs two-stage search: first finds nprobe nearest centroids to the
    /// query, then exhaustively searches all vectors in those clusters. Uses
    /// a max-heap to track top-k candidates.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search. A good default here is
    ///   `sqrt(nlist)`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize, nprobe: Option<usize>) -> (Vec<usize>, Vec<T>) {
        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k: usize = k.min(self.n);
        let query_norm = if matches!(self.metric, Dist::Cosine) {
            query_vec
                .iter()
                .map(|&v| v * v)
                .fold(T::zero(), |a, b| a + b)
                .sqrt()
        } else {
            T::one()
        };

        // 1. find the top `nprobe` centroids
        let cluster_dists: Vec<(T, usize)> = self.get_centroids_dist(query_vec, query_norm, nprobe);

        // 2. search only those clusters in the CSR layout
        let mut buffer = SortedBuffer::with_capacity(k);

        for &(_, cluster_idx) in cluster_dists.iter().take(nprobe) {
            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];

            for vec_idx in start..end {
                let dist = match self.metric {
                    Dist::Euclidean => self.euclidean_distance_to_query_bf16(vec_idx, query_vec),
                    Dist::Cosine => {
                        self.cosine_distance_to_query_bf16(vec_idx, query_vec, query_norm)
                    }
                };
                buffer.insert((OrderedFloat(dist), vec_idx), k);
            }
        }

        let (distances, indices) = buffer
            .data()
            .iter()
            .map(|(d, i)| (d.0, self.original_ids[*i]))
            .unzip();
        (indices, distances)
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies
    /// to a temporary vector. Uses `self.query()` under the hood.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search. A good default here is
    ///   `sqrt(nlist)`.
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

        // fallback
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, nprobe)
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete
    /// kNN graph.
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
                let vec = &self.vectors_flat[start..end];
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

                let (indices, dists) = self.query_bf16(vec, k, nprobe);
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

    /// Query the index for approximate nearest neighbours
    ///
    /// Performs two-stage search: first finds nprobe nearest centroids to the
    /// query, then exhaustively searches all vectors in those clusters. Uses
    /// a max-heap to track top-k candidates.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search. A good default here is
    ///   `sqrt(nlist)`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
    #[inline]
    pub fn query_bf16(
        &self,
        query_vec: &[bf16],
        k: usize,
        nprobe: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        let query_vec_t: Vec<T> = decode_bf16_quantisation(query_vec);
        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k: usize = k.min(self.n);
        let query_norm: T = if matches!(self.metric, Dist::Cosine) {
            bf16_norm(query_vec)
        } else {
            T::from_f32(1.0).unwrap()
        };

        // 1. find the top `nprobe` centroids
        let cluster_dists: Vec<(_, usize)> =
            self.get_centroids_dist(&query_vec_t, query_norm, nprobe);

        // 2. search only those clusters in the CSR layout
        let mut buffer = SortedBuffer::with_capacity(k);

        for &(_, cluster_idx) in cluster_dists.iter().take(nprobe) {
            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];

            for vec_idx in start..end {
                let dist = match self.metric {
                    Dist::Euclidean => {
                        self.euclidean_distance_to_query_dual_bf16(vec_idx, query_vec)
                    }
                    Dist::Cosine => self.cosine_distance_to_query_dual_bf16(
                        vec_idx,
                        query_vec,
                        bf16::from_f32(query_norm.to_f32().unwrap()),
                    ),
                };
                buffer.insert((OrderedFloat(dist), vec_idx), k);
            }
        }

        let (distances, indices) = buffer
            .data()
            .iter()
            .map(|(d, i)| (d.0, self.original_ids[*i]))
            .unzip();
        (indices, distances)
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

        let mut new_vectors_flat = Vec::with_capacity(self.vectors_flat.len());
        let mut new_norms = if self.norms.is_empty() {
            Vec::new()
        } else {
            Vec::with_capacity(self.n)
        };

        for &old_id in &new_to_old {
            let start = old_id * self.dim;
            new_vectors_flat.extend_from_slice(&self.vectors_flat[start..start + self.dim]);
            if !self.norms.is_empty() {
                new_norms.push(self.norms[old_id]);
            }
        }

        self.vectors_flat = new_vectors_flat;
        self.norms = new_norms;
        self.all_indices.clear();
        self.all_indices.shrink_to_fit();

        new_to_old
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<bf16>()
            + self.norms.capacity() * std::mem::size_of::<bf16>()
            + self.centroids.capacity() * std::mem::size_of::<T>()
            + self.centroids_norm.capacity() * std::mem::size_of::<T>()
            + self.all_indices.capacity() * std::mem::size_of::<usize>()
            + self.offsets.capacity() * std::mem::size_of::<usize>()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use faer_traits::ComplexField;
    use num_traits::{Float, FromPrimitive};

    fn create_test_data<T: Float + FromPrimitive + ComplexField>(n: usize, dim: usize) -> Mat<T> {
        let mut data = Mat::zeros(n, dim);
        for i in 0..n {
            for j in 0..dim {
                data[(i, j)] = T::from_f64((i * dim + j) as f64 * 0.1).unwrap();
            }
        }
        data
    }

    #[test]
    fn test_ivf_bf16_construction_euclidean() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBf16::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(10),
            Some(10),
            42,
            false,
        );

        assert_eq!(index.n, 200);
        assert_eq!(index.dim, 32);
        assert_eq!(index.nlist, 10);
        assert_eq!(index.vectors_flat.len(), 200 * 32);
        assert!(index.norms.is_empty());
    }

    #[test]
    fn test_ivf_bf16_construction_cosine() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBf16::build(data.as_ref(), Dist::Cosine, Some(10), Some(10), 42, false);

        assert_eq!(index.n, 200);
        assert_eq!(index.dim, 32);
        assert_eq!(index.nlist, 10);
        assert_eq!(index.norms.len(), 200);
    }

    #[test]
    fn test_ivf_bf16_query_returns_k_results() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBf16::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(10),
            Some(10),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10, Some(10));

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_ivf_bf16_query_sorted() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBf16::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(10),
            Some(10),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (_, distances) = index.query(&query, 10, Some(10));

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_ivf_bf16_query_row() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBf16::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(10),
            Some(10),
            42,
            false,
        );

        let (indices, distances) = index.query_row(data.as_ref().row(0), 10, Some(10));

        assert!(indices.len() <= 10);
        assert!(distances.len() <= 10);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_ivf_bf16_query_cosine() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBf16::build(data.as_ref(), Dist::Cosine, Some(10), Some(10), 42, false);

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10, Some(10));

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_ivf_bf16_knn_graph() {
        let data = create_test_data::<f32>(100, 32);
        let index = IvfIndexBf16::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(10),
            Some(10),
            42,
            false,
        );

        let (knn_indices, knn_distances) = index.generate_knn(5, Some(10), true, false);

        assert_eq!(knn_indices.len(), 100);
        assert!(knn_distances.is_some());

        for neighbours in knn_indices.iter() {
            assert!(neighbours.len() <= 5);
        }
    }

    #[test]
    fn test_ivf_bf16_nprobe_variation() {
        let data = create_test_data::<f32>(500, 32);
        let index = IvfIndexBf16::build(
            data.as_ref(),
            Dist::Euclidean,
            Some(20),
            Some(10),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices1, _) = index.query(&query, 10, Some(2));
        let (indices2, _) = index.query(&query, 10, Some(10));

        assert!(indices1.len() <= 10);
        assert!(indices2.len() <= 10);
    }
}
