//! Inverted file SQ8 index: quantises the original data to uniform 8-bit codes
//! and uses Voronoi cells to identify the most interesting candidates.

use faer::RowRef;
use rayon::prelude::*;
use std::{
    collections::BinaryHeap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
use thousands::*;

use crate::prelude::*;
use crate::quantised::hnsw_quantised::codec::GraphCodec;
use crate::quantised::sq8u_codec::*;
use crate::quantised::uniform_quant::*;
use crate::utils::k_means_utils::*;

/////////////////
// IvfSq8Index //
/////////////////

/// IVF index quantised to scalar 8 bits
#[cfg_attr(
    feature = "serialise",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "")
)]
pub struct IvfSq8Index<T>
where
    T: AnnSearchFloat,
{
    /// Uniformly quantised storage and its distance arithmetic.
    codec: Sq8uCodec<T>,
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
    /// Number of k-means clusters.
    nlist: usize,
    /// Original indices
    original_ids: Vec<usize>,
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for IvfSq8Index<T>
where
    T: AnnSearchFloat,
{
    fn dim(&self) -> usize {
        self.dim
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
    /// Constructs an inverted file index with all vectors quantised to `u8`
    /// against a single shared scale, so the integer code distance preserves
    /// the ordering of the float one. Reduces memory by 4x (for f32) whilst
    /// maintaining reasonable recall, and the symmetric integer kernels make
    /// the scan faster than the `f32` one rather than slower.
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
    /// * `k_means_params` - Optional k-means trainings parameters, see
    ///   [KMeansTrainingParams]. If not provided, will default to sensible
    ///   defaults.
    /// * `seed` - Random seed for reproducibility
    /// * `quant_params` - Optional calibration settings, see
    ///   [`UniformQuantParams`]. Defaults trim 0.1% from each tail.
    /// * `verbose` - Print training progress
    ///
    /// ### Returns
    ///
    /// Constructed quantised index ready for querying
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: impl AnnMatrix<T>,
        nlist: Option<usize>,
        metric: Dist,
        k_means_params: Option<KMeansTrainingParams>,
        seed: usize,
        quant_params: Option<UniformQuantParams>,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (mut vectors_flat, n, dim) = data.into_row_major();

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
            k_means_params,
            seed,
            verbose,
        )?;

        // normalise centroids for cosine
        if metric == Dist::Cosine {
            if verbose {
                println!("  Normalising centroids");
            }
            centroids
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        // 3. assign vectors to clusters
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

        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        // 4. reorder into cluster order *before* encoding, so a probed cell is
        // one contiguous run of codes. Doing it afterwards would mean
        // permuting the codes and every precomputed per-vector term with them.
        let mut original_ids = Vec::with_capacity(n);
        for cluster in 0..nlist {
            original_ids.extend_from_slice(&all_indices[offsets[cluster]..offsets[cluster + 1]]);
        }

        let mut reordered = Vec::with_capacity(n * dim);
        for &old_id in &original_ids {
            reordered.extend_from_slice(&vectors_flat[old_id * dim..(old_id + 1) * dim]);
        }
        drop(vectors_flat);

        // 5. quantise
        if verbose {
            println!("  Quantising vectors");
        }
        let codec = Sq8uCodec::new(&reordered, n, dim, metric, quant_params)?;
        drop(reordered);

        if verbose {
            println!("  Quantisation complete");
        }

        Ok(Self {
            codec,
            centroids,
            all_indices: Vec::new(),
            offsets,
            dim,
            n,
            nlist,
            metric,
            centroids_norm: Vec::new(),
            original_ids,
        })
    }

    /// Query the index for approximate nearest neighbours.
    ///
    /// Performs two-stage search using quantised vectors: first finds nprobe
    /// nearest centroids, then computes distances in code space (`u8` integer
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
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        nprobe: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        let mut query_vec = query_vec.to_vec();

        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k = k.min(self.n);

        if self.metric == Dist::Cosine {
            normalise_vector(&mut query_vec);
        }

        // Find top nprobe centroids, expanding to cover >= k reachable vectors.
        let mut cluster_scores: Vec<(T, usize)> = self.get_centroids_prenorm(&query_vec, nprobe);
        let probed = select_probed_clusters(&mut cluster_scores, &self.offsets, nprobe, k);

        let encoded = self.codec.encode_query(&query_vec)?;
        Ok(self.scan_clusters(&probed, k, |idx| self.codec.score(&encoded, idx)))
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
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
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
    /// Skips the encode step since the vector is already in code space.
    /// Only decodes for centroid search (which is O(nlist), small).
    /// Scan the probed cells and keep the `k` best
    ///
    /// ### Params
    ///
    /// * `probed` - Cluster ids to scan
    /// * `k` - Number of neighbours to return
    /// * `score` - Scoring closure over an internal vector index
    ///
    /// ### Returns
    ///
    /// Tuple of `(original indices, distances)`, nearest first
    #[inline]
    fn scan_clusters<F>(&self, probed: &[usize], k: usize, score: F) -> (Vec<usize>, Vec<T>)
    where
        F: Fn(usize) -> T,
    {
        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

        for &cluster_idx in probed {
            for vec_idx in self.offsets[cluster_idx]..self.offsets[cluster_idx + 1] {
                let s = score(vec_idx);
                if heap.len() < k {
                    heap.push((OrderedFloat(s), vec_idx));
                } else if s < heap.peek().unwrap().0 .0 {
                    heap.pop();
                    heap.push((OrderedFloat(s), vec_idx));
                }
            }
        }

        let mut out: Vec<(OrderedFloat<T>, usize)> = heap.into_vec();
        out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        out.into_iter()
            .map(|(OrderedFloat(s), i)| (self.original_ids[i], self.codec.finalise(s)))
            .unzip()
    }

    /// Query using a vector already stored in the index
    ///
    /// Skips the encode: the stored code is already in the query's code space,
    /// so the symmetric score is exactly what the asymmetric one would give.
    /// The centroid search still needs floats, so the code is dequantised for
    /// that step alone, which is `O(nlist)` and off the hot path.
    ///
    /// ### Params
    ///
    /// * `id` - Internal index of the stored vector
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search
    ///
    /// ### Returns
    ///
    /// Tuple of `(original indices, distances)`, nearest first
    fn query_stored(&self, id: usize, k: usize, nprobe: Option<usize>) -> (Vec<usize>, Vec<T>) {
        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k = k.min(self.n);

        let as_float = self.codec.decode(id);
        let mut cluster_scores: Vec<(T, usize)> = self.get_centroids_prenorm(&as_float, nprobe);
        let probed = select_probed_clusters(&mut cluster_scores, &self.offsets, nprobe, k);

        self.scan_clusters(&probed, k, |idx| self.codec.score_sym(id, idx))
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete
    /// kNN graph. Uses pre-quantised vectors directly, avoiding encode
    /// overhead.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `nprobe` - Number of clusters to search (defaults to sqrt(nlist) if
    ///   None)
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

                let (indices, dists) = self.query_stored(i, k, nprobe);
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

        fix_neg_dist(&mut final_dists);
        (final_indices, final_dists)
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.codec.memory_usage_bytes()
            + self.centroids.capacity() * std::mem::size_of::<T>()
            + self.centroids_norm.capacity() * std::mem::size_of::<T>()
            + self.all_indices.capacity() * std::mem::size_of::<usize>()
            + self.offsets.capacity() * std::mem::size_of::<usize>()
            + self.original_ids.capacity() * std::mem::size_of::<usize>()
    }
}

///////////
// Tests //
///////////

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for IvfSq8Index<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "ivf_sq8";
}

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

    fn get_default_k_means() -> Option<KMeansTrainingParams> {
        Some(KMeansTrainingParams::new(10, None, None))
    }

    #[test]
    fn test_build_euclidean() {
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(
            data.as_ref(),
            Some(2),
            Dist::SquaredEuclidean,
            get_default_k_means(),
            42,
            None,
            false,
        )
        .unwrap();

        assert_eq!(index.dim, 32);
        assert_eq!(index.n, 6);
        assert_eq!(index.nlist, 2);
        assert_eq!(index.metric, Dist::SquaredEuclidean);
        assert_eq!(index.centroids.len(), 64);
        assert_eq!(index.offsets.len(), 3);
    }

    #[test]
    fn test_build_cosine() {
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(
            data.as_ref(),
            Some(2),
            Dist::Cosine,
            get_default_k_means(),
            42,
            None,
            false,
        )
        .unwrap();

        assert_eq!(index.metric, Dist::Cosine);
        assert_eq!(index.n, 6);
    }

    #[test]
    fn test_query_returns_k_results() {
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(
            data.as_ref(),
            Some(2),
            Dist::SquaredEuclidean,
            get_default_k_means(),
            42,
            None,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, distances) = index.query(&query, 3, Some(2)).unwrap();

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_query_k_exceeds_n() {
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(
            data.as_ref(),
            Some(2),
            Dist::SquaredEuclidean,
            get_default_k_means(),
            42,
            None,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, _) = index.query(&query, 100, None).unwrap();

        assert!(indices.len() <= 6);
    }

    #[test]
    fn test_query_finds_nearest() {
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(
            data.as_ref(),
            Some(2),
            Dist::SquaredEuclidean,
            get_default_k_means(),
            42,
            None,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, distances) = index.query(&query, 3, Some(2)).unwrap();

        assert_eq!(indices[0], 0);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_query_cosine() {
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(
            data.as_ref(),
            Some(2),
            Dist::Cosine,
            get_default_k_means(),
            42,
            None,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| if x < 16 { 1.0 } else { 0.0 }).collect();
        let (indices, distances) = index.query(&query, 3, Some(2)).unwrap();

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_query_different_nprobe() {
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(
            data.as_ref(),
            Some(2),
            Dist::SquaredEuclidean,
            get_default_k_means(),
            42,
            None,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| 5.0 + x as f32 * 0.01).collect();

        let (indices1, _) = index.query(&query, 3, Some(1)).unwrap();
        let (indices2, _) = index.query(&query, 3, Some(2)).unwrap();

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_query_deterministic() {
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(
            data.as_ref(),
            Some(2),
            Dist::SquaredEuclidean,
            get_default_k_means(),
            42,
            None,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| 0.5 + x as f32 * 0.01).collect();

        let (indices1, distances1) = index.query(&query, 3, Some(2)).unwrap();
        let (indices2, distances2) = index.query(&query, 3, Some(2)).unwrap();

        assert_eq!(indices1, indices2);
        assert_eq!(distances1, distances2);
    }

    #[test]
    fn test_query_expands_nprobe_when_probed_cells_underfill_k() {
        // 6 vectors split into 3 cells of ~2 each. nprobe=1 alone cannot cover
        // k=4; the fix must walk further cells until reachable >= k.
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(
            data.as_ref(),
            Some(3),
            Dist::SquaredEuclidean,
            get_default_k_means(),
            42,
            None,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, distances) = index.query(&query, 4, Some(1)).unwrap();

        assert_eq!(indices.len(), 4);
        assert_eq!(distances.len(), 4);
    }

    #[test]
    fn test_query_row() {
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(
            data.as_ref(),
            Some(2),
            Dist::SquaredEuclidean,
            get_default_k_means(),
            42,
            None,
            false,
        )
        .unwrap();

        let query_mat = Mat::<f32>::from_fn(1, 32, |_, j| 0.5 + j as f32 * 0.01);
        let row = query_mat.row(0);

        let (indices, distances) = index.query_row(row, 3, Some(2)).unwrap();

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_build_large_nlist() {
        let data = Mat::from_fn(100, 8, |i, j| (i + j) as f32);

        let index = IvfSq8Index::build(
            data.as_ref(),
            Some(10),
            Dist::SquaredEuclidean,
            get_default_k_means(),
            42,
            None,
            false,
        )
        .unwrap();

        assert_eq!(index.nlist, 10);
        assert_eq!(index.offsets.len(), 11);
    }

    #[test]
    fn test_quantisation_preserves_structure() {
        let data = create_simple_dataset();
        let index = IvfSq8Index::build(
            data.as_ref(),
            Some(2),
            Dist::SquaredEuclidean,
            get_default_k_means(),
            42,
            None,
            false,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, _) = index.query(&query, 1, Some(2)).unwrap();

        assert_eq!(indices[0], 0);
    }
}
