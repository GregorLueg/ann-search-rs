//! IVF-PQ index with SOAR spilling. Every point is encoded twice, once as a
//! residual against each of the two cells it belongs to.
//!
//! Spilling loses to plain IVF on exact full-vector search: probing one more
//! cell costs nothing fixed there, so raising `nprobe` is simply a better use
//! of the same candidate budget. Product quantisation breaks that symmetry.
//! Codes are residuals against a cell's centroid, so each probed cell has to
//! rebuild the ADC lookup table (`build_lookup_tables_residual`), costing
//! `n_pq_centroids * dim` operations before a single candidate is scored. At
//! the sizes this crate targets that table build is several times the cost of
//! scanning the cell it serves, so twice the candidates out of *one* cell is
//! much cheaper than the same candidates out of two.
//!
//! ### References
//!
//! Sun, Simcha, Simcha, Chern & Guo, arXiv:2404.00774, 2024 (SOAR);
//! Guo, Sun, Lindgren, Geng, Simcha, Chern & Kumar, ICML, 2020 (ScaNN)

use faer::{MatRef, RowRef};
use fixedbitset::FixedBitSet;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::*;

use crate::prelude::*;
use crate::quantised::quantisers::*;
use crate::utils::k_means_utils::*;

//////////////////////////
// Thread-local buffers //
//////////////////////////

// Marks which cells the current query probes, so a spilled entry can tell
// whether its point's primary copy is also being scanned. `nlist` bits, reset
// in O(nprobe) by walking the probe list back.
thread_local! {
    static SOAR_PQ_PROBED: RefCell<FixedBitSet> = const { RefCell::new(FixedBitSet::new()) };
}

////////////////
// Main index //
////////////////

/// IVF-PQ index with SOAR spilling
///
/// Each point occupies two CSR positions with two independent residual codes.
/// Queries scan the `nprobe` nearest cells and always score a point from its
/// *primary* copy when that cell is probed, falling back to the spilled copy
/// only when it is not. The primary copy is encoded against the point's nearest
/// centroid, so it carries the smallest residual and the lowest ADC error.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct SoarPqIndex<T> {
    /// Encoded residuals, `m` codes per CSR position, `2 * n` positions
    quantised_codes: Vec<u8>,
    /// Original dimensions
    dim: usize,
    /// Number of distinct points in the index
    n: usize,
    /// Distance metric
    metric: Dist,
    /// K-means cluster centroids
    centroids: Vec<T>,
    /// Norms of the centroids. Empty: cosine is handled by normalising the
    /// vectors and centroids at build time, as in the plain IVF-PQ index
    centroids_norm: Vec<T>,
    /// Offsets over CSR positions, `nlist + 1` entries. Differences are list
    /// lengths *including* spilled entries
    offsets: Vec<usize>,
    /// Cumulative count of primary entries per cell, `nlist + 1` entries. Used
    /// only to tell `select_probed_clusters` how many *distinct* points a probe
    /// list reaches
    primary_offsets: Vec<usize>,
    /// CSR position holding each point's primary copy, `n` entries. Only needed
    /// to reconstruct a point for the self-query path
    primary_pos: Vec<u32>,
    /// Dedup tag per CSR position. `u32::MAX` for a point's primary copy, which
    /// is always scanned; otherwise the cell holding that point's primary copy,
    /// so a spilled entry can be skipped when the primary is also being probed.
    ///
    /// This is what keeps the *better* of the two copies. The primary is
    /// encoded against the point's nearest centroid, so it carries the smaller
    /// residual and the lower ADC error. Picking whichever copy a query happens
    /// to reach first would often take the spilled one, which is encoded
    /// against a deliberately not-nearest centroid, and that costs recall at
    /// the top of the curve
    dedup_key: Vec<u32>,
    /// Product quantiser with M codebooks
    codebook: ProductQuantiser<T>,
    /// Number of k-means clusters
    nlist: usize,
    /// Rule used for the secondary assignment
    rule: SoarRule,
    /// CSR position -> caller-facing id. Every id appears exactly twice. Held
    /// as `u32` rather than `usize` because there are `2 * n` of them
    original_ids: Vec<u32>,
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for SoarPqIndex<T>
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

impl<T> VectorDistanceAdc<T> for SoarPqIndex<T>
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

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for SoarPqIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////
// Helpers //
/////////////

/// Pick a spilling rule when the caller did not
///
/// Same metric split as the exact index, and for the same reason: the published
/// quadratic loss is derived for MIPS, which cosine ranking is a relabelling
/// of, and is wrong-signed under squared Euclidean.
///
/// ### Params
///
/// * `rule` - Caller's choice, `None` to auto-pick
/// * `metric` - Distance metric the index is being built for
///
/// ### Returns
///
/// The rule to build with.
fn resolve_soar_pq_rule(rule: Option<SoarRule>, metric: &Dist) -> SoarRule {
    rule.unwrap_or(match metric {
        Dist::Cosine => SoarRule::Orthogonal {
            lambda: DEFAULT_ORTHOGONAL_LAMBDA,
        },
        _ => SoarRule::Shifted {
            mu: DEFAULT_SHIFT_MU,
        },
    })
}

////////////////
// Main index //
////////////////

impl<T> SoarPqIndex<T>
where
    T: AnnSearchFloat,
{
    /// Build a SOAR-spilled IVF-PQ index
    ///
    /// ### Workflow
    ///
    /// 1. Trains `nlist` IVF centroids on a subsample
    /// 2. Assigns every vector a primary and a secondary cell
    /// 3. Trains the product quantiser on the *primary* residuals
    /// 4. Lays codes out in cell order, one row per CSR position
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `nlist` - Optional number of clusters. Defaults to `sqrt(n)`.
    /// * `m` - Number of PQ subspaces; must divide `dim`
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `rule` - Optional secondary-assignment rule, see [`SoarRule`]
    /// * `k_means_params` - Optional k-means trainings parameters, see
    ///   [KMeansTrainingParams]
    /// * `n_pq_centroids` - Optional codebook size, defaults to 256
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print training progress
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    ///
    /// ### Note
    ///
    /// With `nlist == 1` there is no second cell to hand out, so both copies of
    /// every point land in the same cell and the dedup pass discards one of
    /// them. The index still answers correctly, it just wastes half its codes.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: MatRef<T>,
        nlist: Option<usize>,
        m: usize,
        metric: Dist,
        rule: Option<SoarRule>,
        k_means_params: Option<KMeansTrainingParams>,
        n_pq_centroids: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (mut vectors_flat, n, dim) = matrix_to_flat(data);

        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);
        let rule = resolve_soar_pq_rule(rule, &metric);

        // Cosine is handled by normalising up front, exactly as IvfPqIndex does,
        // which is what lets every norm array below be all-ones.
        if metric == Dist::Cosine {
            vectors_flat
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        if verbose {
            println!(
                "  Generating SOAR-PQ index with {} Voronoi cells, rule {:?}.",
                nlist, rule
            );
        }

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

        if metric == Dist::Cosine {
            centroids
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        let training_norms = vec![T::one(); n_train];
        let centroid_norms = vec![T::one(); nlist];

        let training_primary = assign_all_parallel(
            &training_data,
            &training_norms,
            dim,
            n_train,
            &centroids,
            &centroid_norms,
            nlist,
            &metric,
        );

        if verbose {
            println!("  Computing primary residuals for PQ training");
        }
        let mut training_residuals = Vec::with_capacity(training_data.len());
        for (vec_idx, &cluster_id) in training_primary.iter().enumerate() {
            let vec = &training_data[vec_idx * dim..(vec_idx + 1) * dim];
            let centroid = &centroids[cluster_id * dim..(cluster_id + 1) * dim];
            training_residuals.extend_from_slice(&T::subtract_simd(vec, centroid));
        }

        if verbose {
            println!("  Training product quantiser with m={}", m);
        }
        let codebook = ProductQuantiser::train(
            &training_residuals,
            dim,
            m,
            n_pq_centroids,
            &Dist::SquaredEuclidean,
            30,
            seed + 1000,
            verbose,
        )?;

        // Full-dataset assignment
        let data_norms = vec![T::one(); n];
        let primary = assign_all_parallel(
            &vectors_flat,
            &data_norms,
            dim,
            n,
            &centroids,
            &centroid_norms,
            nlist,
            &metric,
        );
        let secondary = assign_secondary_soar(
            &vectors_flat,
            &data_norms,
            dim,
            n,
            &centroids,
            &centroid_norms,
            nlist,
            &primary,
            &rule,
            &metric,
        );

        // Interleave into the `n * 2` layout `invert_assignments_csr` expects.
        let mut both = vec![0u32; n * 2];
        for i in 0..n {
            both[i * 2] = primary[i] as u32;
            both[i * 2 + 1] = secondary[i];
        }
        let (members, offsets) = invert_assignments_csr(&both, n, 2, nlist);

        // Cell owning each CSR position, needed for the residual encode.
        let n_entries = members.len();
        let mut pos_cluster = vec![0u32; n_entries];
        for cluster in 0..nlist {
            for pos in offsets[cluster]..offsets[cluster + 1] {
                pos_cluster[pos] = cluster as u32;
            }
        }

        if verbose {
            println!(
                "  Encoding {} residuals",
                n_entries.separate_with_underscores()
            );
        }
        let mut quantised_codes = vec![0u8; n_entries * m];
        quantised_codes
            .par_chunks_mut(m)
            .enumerate()
            .for_each(|(pos, chunk)| {
                let point = members[pos] as usize;
                let cluster = pos_cluster[pos] as usize;
                let vec = &vectors_flat[point * dim..(point + 1) * dim];
                let centroid = &centroids[cluster * dim..(cluster + 1) * dim];
                let residual = T::subtract_simd(vec, centroid);
                chunk.copy_from_slice(&codebook.encode(&residual));
            });

        let mut primary_pos = vec![0u32; n];
        let mut dedup_key = vec![0u32; n_entries];
        let mut primary_counts = vec![0usize; nlist];
        let mut seen_primary = vec![false; n];
        for pos in 0..n_entries {
            let point = members[pos] as usize;
            let cluster = pos_cluster[pos] as usize;
            if cluster == primary[point] && !seen_primary[point] {
                seen_primary[point] = true;
                primary_pos[point] = pos as u32;
                primary_counts[cluster] += 1;
                dedup_key[pos] = u32::MAX;
            } else {
                dedup_key[pos] = primary[point] as u32;
            }
        }
        let mut primary_offsets = vec![0usize; nlist + 1];
        for c in 0..nlist {
            primary_offsets[c + 1] = primary_offsets[c] + primary_counts[c];
        }

        Ok(Self {
            quantised_codes,
            dim,
            n,
            metric,
            centroids,
            centroids_norm: Vec::new(),
            offsets,
            primary_offsets,
            primary_pos,
            dedup_key,
            codebook,
            nlist,
            rule,
            original_ids: members,
        })
    }

    /// Rule the index was built with
    ///
    /// ### Returns
    ///
    /// The resolved [`SoarRule`].
    pub fn rule(&self) -> SoarRule {
        self.rule
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
            + self.offsets.capacity() * std::mem::size_of::<usize>()
            + self.primary_offsets.capacity() * std::mem::size_of::<usize>()
            + self.primary_pos.capacity() * std::mem::size_of::<u32>()
            + self.dedup_key.capacity() * std::mem::size_of::<u32>()
            + self.original_ids.capacity() * std::mem::size_of::<u32>()
            + self.codebook.memory_usage_bytes()
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Builds one ADC lookup table per probed cell, then scans that cell's
    /// positions. A spilled entry is skipped when its point's primary cell is
    /// also being probed, so each point is scored exactly once and always from
    /// its primary copy, which carries the smaller residual and so the lower
    /// quantisation error. Note this is a structural test, not a visited set:
    /// picking whichever copy a query reaches first would often take the
    /// spilled one, since nearest *to the query* is not nearest *to the point*.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search. Defaults to `sqrt(nlist)`.
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
        if self.metric == Dist::Cosine {
            normalise_vector(&mut query_vec);
        }

        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k = k.min(self.n);

        let mut cluster_scores: Vec<(T, usize)> = self.get_centroids_prenorm(&query_vec, nprobe);
        let probed = select_probed_clusters(&mut cluster_scores, &self.primary_offsets, nprobe, k);

        SOAR_PQ_PROBED.with(|cell| {
            let mut probed_mask = cell.borrow_mut();
            if probed_mask.len() < self.nlist {
                probed_mask.grow(self.nlist);
            }
            for &c in &probed {
                probed_mask.insert(c);
            }

            let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

            for &cluster_idx in &probed {
                let lookup_tables = self.build_lookup_tables_residual(&query_vec, cluster_idx);

                let start = self.offsets[cluster_idx];
                let end = self.offsets[cluster_idx + 1];

                let mut worst_dist = if heap.len() >= k {
                    heap.peek().unwrap().0 .0
                } else {
                    T::infinity()
                };

                for pos in start..end {
                    // Spilled copy whose primary is also being probed: the
                    // primary carries the smaller residual, so let it be the
                    // one that gets scored and skip this one outright.
                    let key = self.dedup_key[pos];
                    if key != u32::MAX && probed_mask.contains(key as usize) {
                        continue;
                    }

                    let dist = self.compute_distance_adc(pos, &lookup_tables);
                    if dist >= worst_dist {
                        continue;
                    }

                    if heap.len() < k {
                        heap.push((OrderedFloat(dist), pos));
                        if heap.len() == k {
                            worst_dist = heap.peek().unwrap().0 .0;
                        }
                    } else {
                        heap.pop();
                        heap.push((OrderedFloat(dist), pos));
                        worst_dist = heap.peek().unwrap().0 .0;
                    }
                }
            }

            for &c in &probed {
                probed_mask.remove(c);
            }

            let mut results: Vec<_> = heap.into_iter().collect();
            results.sort_unstable_by_key(|&(dist, _)| dist);

            let (distances, indices) = results
                .into_iter()
                .map(|(d, pos)| (d.0, self.original_ids[pos] as usize))
                .unzip();

            Ok((indices, distances))
        })
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

    /// Generate a kNN graph from the vectors stored in the index
    ///
    /// Each point is reconstructed from its *primary* copy, which is encoded
    /// against its nearest centroid and so carries the smaller residual.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `nprobe` - Number of clusters to search
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`
    pub fn generate_knn(
        &self,
        k: usize,
        nprobe: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<T> {
        let m = self.codebook.m();
        let counter = Arc::new(AtomicUsize::new(0));

        let mut pos_cluster = vec![0u32; self.original_ids.len()];
        for cluster in 0..self.nlist {
            for pos in self.offsets[cluster]..self.offsets[cluster + 1] {
                pos_cluster[pos] = cluster as u32;
            }
        }

        let unordered_results: Vec<(usize, Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|point| {
                let pos = self.primary_pos[point] as usize;
                let cluster = pos_cluster[pos] as usize;
                let codes = &self.quantised_codes[pos * m..(pos + 1) * m];
                let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
                let residual = self.codebook.decode(codes);
                let reconstructed: Vec<T> = T::add_simd(centroid, &residual);

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

                let (indices, dists) = self.query(&reconstructed, k, nprobe)?;
                Ok((self.original_ids[pos] as usize, indices, dists))
            })
            .collect::<Result<Vec<_>, AnnSearchErrors>>()?;

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

        Ok((final_indices, final_dists))
    }
}

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for SoarPqIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "soar_pq";
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    /// Clustered test data, `dim` must be >= 32 for product quantisation
    fn clustered_data(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let centres: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| (0..dim).map(|_| rng.random_range(-20.0..20.0)).collect())
            .collect();
        (0..n)
            .flat_map(|i| {
                let c = &centres[i % n_clusters];
                (0..dim)
                    .map(|d| c[d] + rng.random_range(-1.0..1.0))
                    .collect::<Vec<f32>>()
            })
            .collect()
    }

    fn as_mat(data: &[f32], n: usize, dim: usize) -> Mat<f32> {
        Mat::from_fn(n, dim, |i, j| data[i * dim + j])
    }

    /// Sub-codebook size for the fixtures.
    ///
    /// The default of 256 exceeds the training-set size at these `n`, and
    /// `fast_random_init` samples `n_centroids` distinct training rows without
    /// replacement, so it would panic. Same workaround the serialise fixtures
    /// use.
    const N_CENTROIDS: usize = 16;

    fn build(n: usize, dim: usize, nlist: usize, metric: Dist) -> (SoarPqIndex<f32>, Vec<f32>) {
        let data = clustered_data(n, dim, 8, 11);
        let mat = as_mat(&data, n, dim);
        let idx = SoarPqIndex::build(
            mat.as_ref(),
            Some(nlist),
            8,
            metric,
            None,
            None,
            Some(N_CENTROIDS),
            1,
            false,
        )
        .unwrap();
        (idx, data)
    }

    #[test]
    fn test_soar_pq_every_point_encoded_twice() {
        let (idx, _) = build(400, 32, 10, Dist::SquaredEuclidean);
        assert_eq!(idx.original_ids.len(), 800);
        assert_eq!(idx.offsets[10], 800);
        assert_eq!(idx.quantised_codes.len(), 800 * 8);

        let mut counts = vec![0usize; 400];
        for &p in &idx.original_ids {
            counts[p as usize] += 1;
        }
        assert!(counts.iter().all(|&c| c == 2));

        // Exactly one position per point is tagged primary.
        let primaries = idx.dedup_key.iter().filter(|&&k| k == u32::MAX).count();
        assert_eq!(primaries, 400);
    }

    #[test]
    fn test_soar_pq_primary_offsets_count_distinct_points() {
        let (idx, _) = build(400, 32, 10, Dist::SquaredEuclidean);
        assert_eq!(idx.primary_offsets[10], 400);
    }

    #[test]
    fn test_soar_pq_no_duplicate_ids_in_results() {
        let (idx, data) = build(500, 32, 8, Dist::SquaredEuclidean);
        for i in (0..500).step_by(31) {
            let q = &data[i * 32..(i + 1) * 32];
            let (indices, _) = idx.query(q, 20, Some(8)).unwrap();
            let unique: HashSet<usize> = indices.iter().copied().collect();
            assert_eq!(unique.len(), indices.len(), "duplicates for query {}", i);
        }
    }

    /// Spilled entries must actually be scored when their primary cell is not
    /// probed.
    ///
    /// At `nprobe == nlist` every spill entry is skipped, so the tests that
    /// query at full probe never exercise the branch that makes spilling do
    /// anything. This compares the returned set against a direct scan over
    /// exactly the positions the dedup rule admits. An off-by-one in
    /// `dedup_key` that dropped or double-counted spilled entries would pass
    /// every other test.
    ///
    /// `nprobe = 1` is deliberate. A spill entry's primary cell is never the
    /// cell it sits in, so at a single probed cell *every* spill entry is
    /// admitted and the branch is guaranteed to fire. At larger `nprobe` on
    /// well-separated blobs it often does not: the probed cells are mutually
    /// adjacent, so a spilled entry's primary is usually probed as well.
    #[test]
    fn test_soar_pq_spilled_entries_are_scored_at_partial_probe() {
        let (idx, data) = build(500, 32, 12, Dist::SquaredEuclidean);
        let nprobe = 1;
        let k = 5;
        let mut total_spilled_scored = 0usize;

        for i in (0..500).step_by(7) {
            let q = &data[i * 32..(i + 1) * 32];
            let (indices, dists) = idx.query(q, k, Some(nprobe)).unwrap();

            // Reproduce the probe set the query would have chosen.
            let mut scores = idx.get_centroids_prenorm(q, nprobe);
            let probed = select_probed_clusters(&mut scores, &idx.primary_offsets, nprobe, k);
            let probed_set: HashSet<usize> = probed.iter().copied().collect();

            // Brute-force the positions the dedup rule admits.
            let mut expected: Vec<(f32, usize)> = Vec::new();
            let mut spilled_scored = 0usize;
            for &c in &probed {
                let table = idx.build_lookup_tables_residual(q, c);
                for pos in idx.offsets[c]..idx.offsets[c + 1] {
                    let key = idx.dedup_key[pos];
                    if key != u32::MAX && probed_set.contains(&(key as usize)) {
                        continue;
                    }
                    if key != u32::MAX {
                        spilled_scored += 1;
                    }
                    expected.push((
                        idx.compute_distance_adc(pos, &table),
                        idx.original_ids[pos] as usize,
                    ));
                }
            }
            expected.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Compare the distance profile, not the id order. PQ distances are
            // discrete sums over small tables, so exact ties are common and any
            // order among them is valid; asserting id order would test the
            // heap's tie-break rather than the dedup rule.
            let expected_d: Vec<f32> = expected.iter().take(k).map(|&(d, _)| d).collect();
            assert_eq!(dists.len(), k);
            for (rank, (got, want)) in dists.iter().zip(expected_d.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-5,
                    "query {} rank {}: got {} want {}",
                    i,
                    rank,
                    got,
                    want
                );
            }

            // Every returned id must be one the dedup rule actually admitted,
            // at the distance it was admitted with.
            let admitted: HashSet<usize> = expected.iter().map(|&(_, id)| id).collect();
            let unique: HashSet<usize> = indices.iter().copied().collect();
            assert_eq!(
                unique.len(),
                indices.len(),
                "query {} returned a duplicate",
                i
            );
            for id in &indices {
                assert!(
                    admitted.contains(id),
                    "query {} returned id {} which the dedup rule excludes",
                    i,
                    id
                );
            }
            total_spilled_scored += spilled_scored;
        }

        // Aggregated rather than per-query: spill assignments are heavily
        // skewed on well-separated blobs, so an individual probed cell may hold
        // none. Across the sweep the branch must fire, or the equality check
        // above is only testing the primary pass.
        assert!(
            total_spilled_scored > 0,
            "no spilled entry was scored across any query; the branch under test never ran"
        );
    }

    /// A point reachable through both of its cells must be scored from the
    /// *primary* one, which carries the smaller residual.
    ///
    /// This is the entire reason `dedup_key` exists rather than a visited set.
    /// Keeping whichever copy a query happens to reach first is a different,
    /// and worse, policy: nearest *to the query* is not nearest *to the point*.
    #[test]
    fn test_soar_pq_scores_the_primary_copy_when_both_cells_probed() {
        let (idx, data) = build(400, 32, 6, Dist::SquaredEuclidean);
        let nlist = 6;

        // Full probe, so every point's two cells are both in the probe set.
        let mut checked = 0usize;
        for i in (0..400).step_by(29) {
            let q = &data[i * 32..(i + 1) * 32];
            let (indices, dists) = idx.query(q, 15, Some(nlist)).unwrap();

            for (rank, &id) in indices.iter().enumerate() {
                let primary = idx.primary_pos[id] as usize;
                // Cell owning the primary position.
                let cell = (0..nlist)
                    .find(|&c| primary >= idx.offsets[c] && primary < idx.offsets[c + 1])
                    .unwrap();
                let table = idx.build_lookup_tables_residual(q, cell);
                let from_primary = idx.compute_distance_adc(primary, &table);
                assert!(
                    (dists[rank] - from_primary).abs() < 1e-5,
                    "id {} scored {} but its primary copy gives {}",
                    id,
                    dists[rank],
                    from_primary
                );
                checked += 1;
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn test_soar_pq_query_returns_k_results() {
        let (idx, data) = build(400, 32, 10, Dist::SquaredEuclidean);
        let (indices, dists) = idx.query(&data[0..32], 12, Some(3)).unwrap();
        assert_eq!(indices.len(), 12);
        assert_eq!(dists.len(), 12);
        assert!(dists.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_soar_pq_query_expands_nprobe_when_underfilling() {
        let (idx, data) = build(200, 32, 20, Dist::SquaredEuclidean);
        let (indices, _) = idx.query(&data[0..32], 50, Some(1)).unwrap();
        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_soar_pq_finds_self_in_top_k() {
        let (idx, data) = build(400, 32, 8, Dist::SquaredEuclidean);
        let mut hits = 0;
        for i in (0..400).step_by(17) {
            let q = &data[i * 32..(i + 1) * 32];
            let (indices, _) = idx.query(q, 10, Some(8)).unwrap();
            if indices.contains(&i) {
                hits += 1;
            }
        }
        // Quantisation blurs exact self-retrieval, but the point should be in
        // its own top-10 essentially always on well-separated blobs.
        assert!(hits >= 20, "only {} self-hits", hits);
    }

    #[test]
    fn test_soar_pq_cosine_builds_and_queries() {
        let (idx, data) = build(400, 32, 10, Dist::Cosine);
        assert!(matches!(idx.rule(), SoarRule::Orthogonal { .. }));
        let (indices, _) = idx.query(&data[0..32], 10, Some(10)).unwrap();
        let unique: HashSet<usize> = indices.iter().copied().collect();
        assert_eq!(unique.len(), indices.len());
    }

    #[test]
    fn test_soar_pq_single_cluster_degenerates() {
        let (idx, data) = build(200, 32, 1, Dist::SquaredEuclidean);
        let (indices, _) = idx.query(&data[0..32], 10, Some(1)).unwrap();
        let unique: HashSet<usize> = indices.iter().copied().collect();
        assert_eq!(unique.len(), indices.len());
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_soar_pq_manhattan_rejected() {
        let data = clustered_data(100, 32, 4, 1);
        let mat = as_mat(&data, 100, 32);
        assert!(SoarPqIndex::build(
            mat.as_ref(),
            None,
            8,
            Dist::Manhattan,
            None,
            None,
            Some(N_CENTROIDS),
            1,
            false
        )
        .is_err());
    }

    #[test]
    fn test_soar_pq_deterministic() {
        let (a, _) = build(300, 32, 8, Dist::SquaredEuclidean);
        let (b, _) = build(300, 32, 8, Dist::SquaredEuclidean);
        assert_eq!(a.original_ids, b.original_ids);
        assert_eq!(a.quantised_codes, b.quantised_codes);
    }

    #[test]
    fn test_soar_pq_generate_knn_shape() {
        let (idx, _) = build(300, 32, 8, Dist::SquaredEuclidean);
        let (indices, dists) = idx.generate_knn(7, Some(8), true, false).unwrap();
        assert_eq!(indices.len(), 300);
        assert!(indices.iter().all(|row| row.len() == 7));
        assert!(dists.unwrap().iter().all(|row| row.len() == 7));
    }
}
