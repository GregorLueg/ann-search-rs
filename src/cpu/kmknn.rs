//! k-means nearest neighbour (kMkNN) index. Exact nearest neighbour search
//! that leverages k-means clustering and the triangle inequality to prune
//! distance computations.

use faer::{MatRef, RowRef};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::Separable;

use crate::prelude::*;
use crate::utils::dist::*;
use crate::utils::k_means_utils::*;
use crate::utils::*;

/////////////////////
// Index structure //
/////////////////////

/// kMkNN (k-means nearest neighbour) index
///
/// An exact nearest neighbour index that partitions the data via k-means and
/// uses the triangle inequality for pruning. For Cosine distance, vectors are
/// L2-normalised at build time and the index operates in Euclidean space
/// internally (squared Euclidean on the unit sphere is monotone in Cosine
/// distance).
pub struct KmknnIndex<T> {
    /// Vector data in cluster-sorted order. For Cosine, L2-normalised.
    pub vectors_flat: Vec<T>,
    /// Embedding dimensions
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// The user's original distance metric (used for output conversion)
    metric: Dist,
    /// Cluster centres (nlist * dim elements)
    centroids: Vec<T>,
    /// CSR offsets into `vectors_flat` per cluster (length nlist + 1)
    offsets: Vec<usize>,
    /// Number of clusters
    nlist: usize,
    /// Mapping from internal position to original sample index
    original_ids: Vec<usize>,
    /// Squared distance from each point to its assigned centroid
    point_to_centroid_sq: Vec<T>,
    /// Per-cluster squared maximum point-to-centroid distance
    cluster_radii_sq: Vec<T>,
    /// Global max of `cluster_radii_sq` for early termination
    max_cluster_radius_sq: T,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for KmknnIndex<T>
where
    T: AnnSearchFloat,
{
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    fn dim(&self) -> usize {
        self.dim
    }

    /// Not populated: for Cosine the vectors are pre-normalised, and for
    /// Euclidean norms are unused. Callers relying on the cosine_distance
    /// default of `VectorDistance` should use `ExhaustiveIndex` instead.
    fn norms(&self) -> &[T] {
        &[]
    }
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for KmknnIndex<T>
where
    T: AnnSearchFloat,
{
    fn centroids(&self) -> &[T] {
        &self.centroids
    }

    fn dim(&self) -> usize {
        self.dim
    }

    /// Always Euclidean: Cosine is handled via normalisation at build time.
    fn metric(&self) -> Dist {
        Dist::Euclidean
    }

    fn nlist(&self) -> usize {
        self.nlist
    }

    fn centroids_norm(&self) -> &[T] {
        &[]
    }
}

////////////////
// Main index //
////////////////

impl<T> KmknnIndex<T>
where
    T: AnnSearchFloat,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Build a kMkNN index.
    ///
    /// For Cosine, vectors are L2-normalised and the index operates on
    /// Euclidean distances internally. All triangle-inequality bounds use
    /// true Euclidean distances.
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
        let (mut vectors_flat, n, dim) = matrix_to_flat(data);

        // Cosine: normalise in place and proceed in Euclidean space
        if metric == Dist::Cosine {
            for i in 0..n {
                let start = i * dim;
                normalise_vector(&mut vectors_flat[start..start + dim]);
            }
        }

        let max_iters = max_iters.unwrap_or(50);
        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        if verbose {
            println!("  Building kMkNN index with {} clusters.", nlist);
        }

        // always Euclidean internally
        let centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &Dist::Euclidean,
            max_iters,
            seed,
            verbose,
        );

        let dummy_data_norms = vec![T::one(); n];
        let dummy_centroid_norms = vec![T::one(); nlist];
        let assignments = assign_all_parallel(
            &vectors_flat,
            &dummy_data_norms,
            dim,
            n,
            &centroids,
            &dummy_centroid_norms,
            nlist,
            &Dist::Euclidean,
        );

        if verbose {
            print_cluster_summary(&assignments, nlist);
        }

        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        // Reorder vectors by cluster for cache locality
        let mut new_to_old = Vec::with_capacity(n);
        for c in 0..nlist {
            for &old_id in &all_indices[offsets[c]..offsets[c + 1]] {
                new_to_old.push(old_id);
            }
        }
        let mut new_vectors_flat = Vec::with_capacity(n * dim);
        for &old_id in &new_to_old {
            let start = old_id * dim;
            new_vectors_flat.extend_from_slice(&vectors_flat[start..start + dim]);
        }

        // precompute per-point and per-cluster squared bounds (lazy sqrt)
        let mut point_to_centroid_sq = vec![T::zero(); n];
        let mut cluster_radii_sq = vec![T::zero(); nlist];
        let mut max_cluster_radius_sq = T::zero();

        for c in 0..nlist {
            let cent = &centroids[c * dim..(c + 1) * dim];
            let mut max_d_sq = T::zero();
            for pos in offsets[c]..offsets[c + 1] {
                let v = &new_vectors_flat[pos * dim..(pos + 1) * dim];
                let d_sq = euclidean_distance_static(v, cent);
                point_to_centroid_sq[pos] = d_sq;
                if d_sq > max_d_sq {
                    max_d_sq = d_sq;
                }
            }
            cluster_radii_sq[c] = max_d_sq;
            if max_d_sq > max_cluster_radius_sq {
                max_cluster_radius_sq = max_d_sq;
            }
        }

        Self {
            vectors_flat: new_vectors_flat,
            dim,
            n,
            metric,
            centroids,
            offsets,
            nlist,
            original_ids: new_to_old,
            point_to_centroid_sq,
            cluster_radii_sq,
            max_cluster_radius_sq,
        }
    }

    ///////////
    // Query //
    ///////////

    /// Core search loop. Assumes `q` is already in the correct space (i.e.
    /// L2-normalised if the index is in Cosine mode). Output conversion
    /// (squared Euclidean → Cosine via `d²/2`) still applies.
    ///
    /// ### Params
    ///
    /// * `q` - The query vector/slice
    /// * `k` - Number of neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    #[inline]
    fn query_internal(&self, q: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        let k = k.min(self.n);

        let mut cluster_dists: Vec<(T, usize)> = (0..self.nlist)
            .map(|c| {
                let cent = &self.centroids[c * self.dim..(c + 1) * self.dim];
                (euclidean_distance_static(q, cent), c)
            })
            .collect();
        cluster_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);
        let mut r_k_sq = T::infinity();
        let mut r_k = T::infinity();

        let max_radius = self.max_cluster_radius_sq.sqrt();

        for &(d_qc_sq, c) in &cluster_dists {
            let d_qc = d_qc_sq.sqrt();

            if d_qc - max_radius >= r_k {
                break;
            }

            let r_c = self.cluster_radii_sq[c].sqrt();
            if d_qc - r_c >= r_k {
                continue;
            }

            let start = self.offsets[c];
            let end = self.offsets[c + 1];

            for pos in start..end {
                let d_pc = self.point_to_centroid_sq[pos].sqrt();
                if (d_qc - d_pc).abs() >= r_k {
                    continue;
                }

                let v = &self.vectors_flat[pos * self.dim..(pos + 1) * self.dim];
                let d_qp_sq = euclidean_distance_static(v, q);

                if heap.len() < k {
                    heap.push((OrderedFloat(d_qp_sq), pos));
                    if heap.len() == k {
                        r_k_sq = heap.peek().unwrap().0 .0;
                        r_k = r_k_sq.sqrt();
                    }
                } else if d_qp_sq < r_k_sq {
                    heap.pop();
                    heap.push((OrderedFloat(d_qp_sq), pos));
                    r_k_sq = heap.peek().unwrap().0 .0;
                    r_k = r_k_sq.sqrt();
                }
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(d, _)| d);

        let half = T::one() / (T::one() + T::one());
        let (distances, indices): (Vec<_>, Vec<_>) = results
            .into_iter()
            .map(|(OrderedFloat(d_sq), pos)| {
                let out = match self.metric {
                    Dist::Euclidean => d_sq,
                    Dist::Cosine => d_sq * half,
                };
                (out, self.original_ids[pos])
            })
            .unzip();

        (indices, distances)
    }

    /// Exact k-nearest neighbour query.
    ///
    /// Walks clusters in ascending query-to-centroid distance order, uses the
    /// cluster radius to skip entire clusters, and uses per-point centroid
    /// distances to skip individual candidates via the triangle inequality. For
    /// Cosine, normalises the query and converts the final squared Euclidean
    /// distances back to Cosine distance (`d² / 2`).
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted nearest-first. Euclidean mode
    /// returns squared distances; Cosine mode returns Cosine distance.
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        assert!(
            query_vec.len() == self.dim,
            "The query vector has different dimensionality than the index"
        );

        if self.metric == Dist::Cosine {
            let mut q = query_vec.to_vec();
            normalise_vector(&mut q);
            self.query_internal(&q, k)
        } else {
            self.query_internal(query_vec, k)
        }
    }

    /// Exact k-nearest neighbour query from a row reference.
    ///
    /// Fast path for contiguous rows (stride == 1); otherwise clones.
    #[inline]
    pub fn query_row(&self, query_row: RowRef<T>, k: usize) -> (Vec<usize>, Vec<T>) {
        assert!(
            query_row.ncols() == self.dim,
            "The query row has different dimensionality than the index"
        );

        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k)
    }

    /// Generate exact kNN graph from vectors stored in the index.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)` indexed by original id
    pub fn generate_knn(
        &self,
        k: usize,
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

                // Stored vectors are already normalised in Cosine mode, so
                // skip query() and hit the internal path directly.
                let (indices, dists) = self.query_internal(vec, k);
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
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.centroids.capacity() * std::mem::size_of::<T>()
            + self.offsets.capacity() * std::mem::size_of::<usize>()
            + self.original_ids.capacity() * std::mem::size_of::<usize>()
            + self.point_to_centroid_sq.capacity() * std::mem::size_of::<T>()
            + self.cluster_radii_sq.capacity() * std::mem::size_of::<T>()
    }
}

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for KmknnIndex<T>
where
    T: AnnSearchFloat,
{
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        self.query(query_vec, k)
    }

    fn n(&self) -> usize {
        self.n
    }

    fn metric(&self) -> Dist {
        self.metric
    }

    fn original_ids(&self) -> &[usize] {
        &self.original_ids
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::exhaustive::ExhaustiveIndex;
    use approx::assert_relative_eq;
    use faer::Mat;

    fn create_simple_matrix() -> Mat<f32> {
        let data = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
    }

    #[test]
    fn test_kmknn_index_creation() {
        let data = create_simple_matrix();
        let _ = KmknnIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);
    }

    #[test]
    fn test_kmknn_query_finds_self_euclidean() {
        let data = create_simple_matrix();
        let index = KmknnIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kmknn_query_finds_self_cosine() {
        let data = create_simple_matrix();
        let index = KmknnIndex::build(data.as_ref(), Dist::Cosine, Some(2), None, 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kmknn_query_sorted() {
        let data = create_simple_matrix();
        let index = KmknnIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (_, distances) = index.query(&query, 5);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_kmknn_query_k_larger_than_dataset() {
        let data = create_simple_matrix();
        let index = KmknnIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 10);

        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_kmknn_orthogonal_cosine() {
        let data = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let index = KmknnIndex::build(data.as_ref(), Dist::Cosine, Some(3), None, 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(distances[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(distances[2], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kmknn_reproducibility() {
        let data = create_simple_matrix();

        let idx1 = KmknnIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);
        let idx2 = KmknnIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);

        let query = vec![0.5, 0.5, 0.0];
        let (i1, _) = idx1.query(&query, 3);
        let (i2, _) = idx2.query(&query, 3);
        assert_eq!(i1, i2);
    }

    #[test]
    fn test_kmknn_query_row() {
        let data = create_simple_matrix();
        let index = KmknnIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);

        let (indices, distances) = index.query_row(data.row(0), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kmknn_larger_dataset() {
        let n = 100;
        let dim = 10;
        let data = Mat::from_fn(n, dim, |i, j| (i * j) as f32 / 10.0);

        let index = KmknnIndex::build(data.as_ref(), Dist::Euclidean, Some(10), None, 42, false);
        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5);

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_kmknn_generate_knn() {
        let data = create_simple_matrix();
        let index = KmknnIndex::build(data.as_ref(), Dist::Euclidean, Some(2), None, 42, false);

        let (indices, dists) = index.generate_knn(3, true, false);

        assert_eq!(indices.len(), 5);
        assert!(dists.is_some());
        // Each row's nearest neighbour is itself at distance 0
        for (i, row) in indices.iter().enumerate() {
            assert_eq!(row[0], i);
        }
        for row in dists.unwrap() {
            assert_relative_eq!(row[0], 0.0, epsilon = 1e-5);
        }
    }

    // Exactness check: kMkNN must return the same neighbours as the
    // brute-force index. This is the core correctness guarantee.
    #[test]
    fn test_kmknn_matches_exhaustive_euclidean() {
        let n = 200;
        let dim = 16;
        let data = Mat::from_fn(n, dim, |i, j| ((i * 7 + j * 13) % 97) as f32 / 10.0);

        let kmknn = KmknnIndex::build(data.as_ref(), Dist::Euclidean, Some(14), None, 42, false);
        let exhaustive = ExhaustiveIndex::new(data.as_ref(), Dist::Euclidean);

        let query: Vec<f32> = (0..dim).map(|j| (j * 3 % 17) as f32 / 5.0).collect();
        let k = 10;

        let (_, kmknn_dist) = kmknn.query(&query, k);
        let (_, exh_dist) = exhaustive.query(&query, k);

        // Distances must match exactly (up to float noise); indices may
        // differ only on ties
        for i in 0..k {
            assert_relative_eq!(kmknn_dist[i], exh_dist[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_kmknn_matches_exhaustive_cosine() {
        let n = 200;
        let dim = 16;
        let data = Mat::from_fn(n, dim, |i, j| ((i * 7 + j * 13) % 97) as f32 / 10.0 + 0.1);

        let kmknn = KmknnIndex::build(data.as_ref(), Dist::Cosine, Some(14), None, 42, false);
        let exhaustive = ExhaustiveIndex::new(data.as_ref(), Dist::Cosine);

        let query: Vec<f32> = (0..dim).map(|j| (j * 3 % 17) as f32 / 5.0 + 0.1).collect();
        let k = 10;

        let (_, kmknn_dist) = kmknn.query(&query, k);
        let (_, exh_dist) = exhaustive.query(&query, k);

        for i in 0..k {
            assert_relative_eq!(kmknn_dist[i], exh_dist[i], epsilon = 1e-4);
        }
    }
}
