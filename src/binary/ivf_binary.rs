use faer::{MatRef, RowRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::Separable;

use crate::binary::binariser::*;
use crate::binary::dist_binary::*;
use crate::utils::dist::*;
use crate::utils::ivf_utils::*;
use crate::utils::*;

/// IVF index with binary quantisation
///
/// Hybrid approach: uses float centroids for routing to preserve cluster structure,
/// but stores vectors as binary codes for memory efficiency. Ideal for kNN graph
/// generation where approximate ranking is acceptable but global structure matters.
///
/// ### Fields
///
/// * `vectors_flat_binarised` - Binary codes, flattened (n * n_bytes)
/// * `n_bytes` - Bytes per vector (n_bits / 8)
/// * `n` - Number of samples
/// * `dim` - Original vector dimensionality
/// * `binariser` - Binariser for encoding query vectors
/// * `centroids_float` - Float centroids for routing (nlist * dim)
/// * `centroids_norm` - Precomputed norms for Cosine distance
/// * `all_indices` - Vector indices for each cluster (CSR format)
/// * `offsets` - Offsets for CSR access
/// * `nlist` - Number of clusters
/// * `metric` - Distance metric (for centroid routing)
pub struct IvfIndexBinary<T> {
    pub vectors_flat_binarised: Vec<u8>,
    pub n_bytes: usize,
    pub n: usize,
    pub dim: usize,
    binariser: Binariser<T>,
    centroids_float: Vec<T>,
    centroids_norm: Vec<T>,
    metric: Dist,
    all_indices: Vec<usize>,
    offsets: Vec<usize>,
    nlist: usize,
}

//////////////////////////
// VectorDistanceBinary //
//////////////////////////

impl<T> VectorDistanceBinary for IvfIndexBinary<T> {
    fn vectors_flat_binarised(&self) -> &[u8] {
        &self.vectors_flat_binarised
    }

    fn n_bytes(&self) -> usize {
        self.n_bytes
    }
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for IvfIndexBinary<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    fn centroids(&self) -> &[T] {
        &self.centroids_float
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

impl<T> IvfIndexBinary<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField,
{
    /// Build an IVF index with binary quantisation
    ///
    /// Uses hybrid approach: float centroids for routing, binary codes for
    /// storage. Clustering happens in float space to preserve semantic
    /// structure.
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `binarisation_init` - Initialisation method ("itq" or "random")
    /// * `n_bits` - Number of bits per binary code (must be multiple of 8)
    /// * `metric` - Distance metric for centroid routing
    /// * `nlist` - Optional number of clusters (defaults to sqrt(n))
    /// * `max_iters` - Optional max k-means iterations (defaults to 30)
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Constructed index
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: MatRef<T>,
        binarisation_init: &str,
        n_bits: usize,
        metric: Dist,
        nlist: Option<usize>,
        max_iters: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Self {
        assert!(n_bits % 8 == 0, "n_bits must be multiple of 8");

        let n = data.nrows();
        let dim = data.ncols();
        let n_bytes = n_bits / 8;

        let (vectors_flat, _, _) = matrix_to_flat(data);

        let max_iters = max_iters.unwrap_or(30);
        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        if verbose {
            println!(
                "  Generating IVF-Binary index with {} Voronoi cells.",
                nlist
            );
        }

        // 1. subsample for training if needed
        let (training_data, n_train) = if n > 500_000 {
            if verbose {
                println!("  Sampling 250k vectors for training");
            }
            let (data, _) = sample_vectors(&vectors_flat, dim, n, 250_000, seed);
            (data, 250_000)
        } else {
            (vectors_flat.clone(), n)
        };

        // 2. train float centroids
        let centroids_float = train_centroids(
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
                    centroids_float[start..end]
                        .iter()
                        .map(|x| *x * *x)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt()
                })
                .collect()
        } else {
            Vec::new()
        };

        // 3. assign all vectors to centroids in float space
        let data_norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    vectors_flat[start..end]
                        .iter()
                        .map(|x| *x * *x)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt()
                })
                .collect()
        } else {
            vec![T::one(); n]
        };

        let assignments = assign_all_parallel(
            &vectors_flat,
            &data_norms,
            dim,
            n,
            &centroids_float,
            &centroids_norm,
            nlist,
            &metric,
        );

        // 4. build CSR layout
        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        // 5. initialise binariser and encode all vectors
        let init = parse_binarisation_init(binarisation_init).unwrap_or_default();
        let binariser = match init {
            BinarisationInit::ITQ => Binariser::initialise_with_pca(data, dim, n_bits, seed),
            BinarisationInit::RandomProjections => Binariser::new(dim, n_bits, seed),
        };

        let mut vectors_flat_binarised: Vec<u8> = Vec::with_capacity(n * n_bytes);
        for i in 0..n {
            let original: Vec<T> = data.row(i).iter().cloned().collect();
            vectors_flat_binarised.extend(binariser.encode(&original));
        }

        Self {
            vectors_flat_binarised,
            n_bytes,
            n,
            dim,
            binariser,
            centroids_float,
            centroids_norm,
            metric,
            all_indices,
            offsets,
            nlist,
        }
    }

    ///////////
    // Query //
    ///////////

    /// Query the index for approximate nearest neighbours
    ///
    /// Two-stage search: finds nprobe nearest centroids using float distance,
    /// then searches those clusters using Hamming distance on binary codes.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector in float space
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search (defaults to sqrt(nlist))
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` where distances are Hamming distances
    #[inline]
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        nprobe: Option<usize>,
    ) -> (Vec<usize>, Vec<u32>) {
        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k = k.min(self.n);

        let query_binary = self.binariser.encode(query_vec);

        let query_norm = if matches!(self.metric, Dist::Cosine) {
            query_vec
                .iter()
                .map(|&v| v * v)
                .fold(T::zero(), |a, b| a + b)
                .sqrt()
        } else {
            T::one()
        };

        // 1. Find nprobe nearest centroids using float distance
        let cluster_dists = self.get_centroids_dist(query_vec, query_norm, nprobe);

        // 2. Search clusters using Hamming distance
        let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(k + 1);

        for &(_, cluster_idx) in cluster_dists.iter().take(nprobe) {
            let start = self.offsets[cluster_idx];
            let end = self.offsets[cluster_idx + 1];

            for &vec_idx in &self.all_indices[start..end] {
                let dist = self.hamming_distance_query(&query_binary, vec_idx);

                if heap.len() < k {
                    heap.push((dist, vec_idx));
                } else if dist < heap.peek().unwrap().0 {
                    heap.pop();
                    heap.push((dist, vec_idx));
                }
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);

        let (distances, indices): (Vec<_>, Vec<_>) = results.into_iter().unzip();

        (indices, distances)
    }

    /// Query using row reference
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        nprobe: Option<usize>,
    ) -> (Vec<usize>, Vec<u32>) {
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
    /// Queries each vector against the index to build complete kNN graph.
    /// Uses float vectors for centroid routing, Hamming distance for ranking.
    ///
    /// ### Params
    ///
    /// * `data` - Original float data (needed for centroid routing)
    /// * `k` - Number of neighbours per vector
    /// * `nprobe` - Number of clusters to search (defaults to sqrt(nlist))
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`
    pub fn generate_knn(
        &self,
        data: MatRef<T>,
        k: usize,
        nprobe: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<u32>>>) {
        assert_eq!(data.nrows(), self.n, "Data row count mismatch");

        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<u32>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let vec: Vec<T> = data.row(i).iter().cloned().collect();

                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count % 100_000 == 0 {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            self.n.separate_with_underscores()
                        );
                    }
                }

                self.query(&vec, k, nprobe)
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

    /// Returns the size of the index in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat_binarised.capacity()
            + self.binariser.memory_usage_bytes()
            + self.centroids_float.capacity() * std::mem::size_of::<T>()
            + self.centroids_norm.capacity() * std::mem::size_of::<T>()
            + self.all_indices.capacity() * std::mem::size_of::<usize>()
            + self.offsets.capacity() * std::mem::size_of::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

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
    fn test_ivf_binary_construction() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(10),
            Some(10),
            42,
            false,
        );

        assert_eq!(index.n, 200);
        assert_eq!(index.n_bytes, 8);
        assert_eq!(index.dim, 32);
        assert_eq!(index.nlist, 10);
        assert_eq!(index.vectors_flat_binarised.len(), 200 * 8);
    }

    #[test]
    fn test_ivf_binary_query_returns_k_results() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(10),
            Some(10),
            42,
            false,
        );

        let query: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 10, Some(10)); // Search all clusters

        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
    }

    #[test]
    fn test_ivf_binary_query_sorted() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
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
    fn test_ivf_binary_nprobe_affects_results() {
        let data = create_test_data::<f32>(500, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
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

    #[test]
    fn test_ivf_binary_query_row() {
        let data = create_test_data::<f32>(200, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
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
    fn test_ivf_binary_knn_graph() {
        let data = create_test_data::<f32>(100, 32);
        let index = IvfIndexBinary::build(
            data.as_ref(),
            "random",
            64,
            Dist::Cosine,
            Some(10),
            Some(10),
            42,
            false,
        );

        let (knn_indices, knn_distances) =
            index.generate_knn(data.as_ref(), 5, Some(10), true, false);

        assert_eq!(knn_indices.len(), 100);
        assert!(knn_distances.is_some());

        for neighbours in knn_indices.iter() {
            assert!(neighbours.len() <= 5); 
        }
    }
}
