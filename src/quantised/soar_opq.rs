//! IVF-OPQ index with SOAR spilling. The OPQ counterpart of
//! [`crate::quantised::soar_pq::SoarPqIndex`]: every point is encoded twice,
//! once as a rotated residual against each of the two cells it belongs to.
//!
//! The reason spilling pays here is the same as for plain PQ. Codes are
//! residuals against a cell centroid, so each probed cell must rebuild the ADC
//! lookup table at `n_pq_centroids * dim` operations before scoring a single
//! candidate. Drawing twice the candidates from one cell is therefore much
//! cheaper than drawing the same candidates from two. Exact full-vector search
//! has no such fixed cost, which is why spilling loses there.
//!
//! OPQ adds a learned rotation `R`. Codes are `PQ(R * r)`, so the codebooks
//! live in the rotated space and the lookup table has to be built there too:
//! `R` is orthogonal, so `||q_r - r|| == ||R q_r - R r||`, but only when both
//! sides are rotated. The query residual is rotated before the table is built.
//!
//! Costs `2 * n * m` code bytes rather than `n * m`, so the honest comparison
//! is against an [`crate::quantised::ivf_opq::IvfOpqIndex`] with twice the
//! subspaces.
//!
//! ### References
//!
//! Sun, Simcha, Simcha, Chern & Guo, arXiv:2404.00774, 2024 (SOAR);
//! Ge, He, Ke & Sun, CVPR, 2013 (optimised product quantisation)

use faer::RowRef;
use fixedbitset::FixedBitSet;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::ops::AddAssign;
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
    static SOAR_OPQ_PROBED: RefCell<FixedBitSet> = const { RefCell::new(FixedBitSet::new()) };
}

////////////////
// Main index //
////////////////

/// IVF-OPQ index with SOAR spilling
///
/// Each point occupies two CSR positions with two independent rotated-residual
/// codes. A point is always scored from its primary copy when that cell is
/// probed, since the primary residual is the smaller of the two and so carries
/// the lower quantisation error.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct SoarOpqIndex<T> {
    /// Encoded rotated residuals, `m` codes per CSR position, `2 * n` positions
    quantised_codes: Vec<u8>,
    /// Original dimensions
    dim: usize,
    /// Number of distinct points in the index
    n: usize,
    /// Distance metric
    metric: Dist,
    /// K-means cluster centroids
    centroids: Vec<T>,
    /// Centroids pre-rotated into the OPQ codebook space, `nlist * dim`.
    ///
    /// The ADC table has to be built from `R * (q - c)`. Rotating that residual
    /// per probed cell would cost `O(nprobe * dim^2)` scalar work per query,
    /// several times the table build it feeds at dim 512. `R` is linear, so
    /// `R(q - c) == R q - R c`: rotate the query once and subtract these.
    centroids_rotated: Vec<T>,
    /// Norms of the centroids. Empty: cosine is handled by normalising the
    /// vectors and centroids at build time
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
    /// so a spilled entry can be skipped when the primary is also being probed
    dedup_key: Vec<u32>,
    /// Optimised product quantiser: rotation plus M codebooks
    codebook: OptimisedProductQuantiser<T>,
    /// Number of k-means clusters
    nlist: usize,
    /// Rule used for the secondary assignment
    rule: SoarRule,
    /// CSR position -> caller-facing id. Every id appears exactly twice
    original_ids: Vec<u32>,
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for SoarOpqIndex<T>
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

impl<T> VectorDistanceAdc<T> for SoarOpqIndex<T>
where
    T: AnnSearchFloat + AddAssign,
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

    fn metric(&self) -> Dist {
        self.metric
    }

    fn codebooks(&self) -> &[Vec<T>] {
        self.codebook.codebooks()
    }
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for SoarOpqIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////
// Helpers //
/////////////

/// Pick a spilling rule when the caller did not
///
/// Same metric split as every other SOAR index: the published quadratic loss is
/// derived for MIPS, which cosine ranking is a relabelling of, and is
/// wrong-signed under squared Euclidean.
///
/// ### Params
///
/// * `rule` - Caller's choice, `None` to auto-pick
/// * `metric` - Distance metric the index is being built for
///
/// ### Returns
///
/// The rule to build with.
fn resolve_soar_opq_rule(rule: Option<SoarRule>, metric: &Dist) -> SoarRule {
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

impl<T> SoarOpqIndex<T>
where
    T: AnnSearchFloat + AddAssign,
{
    /// Build a SOAR-spilled IVF-OPQ index
    ///
    /// ### Workflow
    ///
    /// 1. Trains `nlist` IVF centroids on a subsample
    /// 2. Assigns every vector a primary and a secondary cell
    /// 3. Trains the optimised product quantiser on the *primary* residuals
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
    /// * `n_opq_centroids` - Optional codebook size, defaults to 256
    /// * `opq_iter` - Optional number of rotation-refinement iterations
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
    /// every point land in the same cell and the dedup pass discards one.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: impl AnnMatrix<T>,
        nlist: Option<usize>,
        m: usize,
        metric: Dist,
        rule: Option<SoarRule>,
        k_means_params: Option<KMeansTrainingParams>,
        n_opq_centroids: Option<usize>,
        opq_iter: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (mut vectors_flat, n, dim) = data.into_row_major();

        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);
        let rule = resolve_soar_opq_rule(rule, &metric);

        if metric == Dist::Cosine {
            vectors_flat
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        if verbose {
            println!(
                "  Generating SOAR-OPQ index with {} Voronoi cells, rule {:?}.",
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
            println!("  Computing primary residuals for OPQ training");
        }
        let mut training_residuals = Vec::with_capacity(training_data.len());
        for (vec_idx, &cluster_id) in training_primary.iter().enumerate() {
            let vec = &training_data[vec_idx * dim..(vec_idx + 1) * dim];
            let centroid = &centroids[cluster_id * dim..(cluster_id + 1) * dim];
            training_residuals.extend_from_slice(&T::subtract_simd(vec, centroid));
        }

        if verbose {
            println!("  Training optimised product quantiser with m={}", m);
        }
        let codebook = OptimisedProductQuantiser::train(
            &training_residuals,
            dim,
            m,
            n_opq_centroids,
            opq_iter,
            30,
            seed + 1000,
            verbose,
        )?;

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

        let mut both = vec![0u32; n * 2];
        for i in 0..n {
            both[i * 2] = primary[i] as u32;
            both[i * 2 + 1] = secondary[i];
        }
        let (members, offsets) = invert_assignments_csr(&both, n, 2, nlist);

        let n_entries = members.len();
        let mut pos_cluster = vec![0u32; n_entries];
        for cluster in 0..nlist {
            for pos in offsets[cluster]..offsets[cluster + 1] {
                pos_cluster[pos] = cluster as u32;
            }
        }

        if verbose {
            println!(
                "  Encoding {} rotated residuals",
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
                // `encode` rotates internally, so codes land in the rotated
                // space the codebooks were trained in.
                chunk.copy_from_slice(&codebook.encode(&residual));
            });

        // Tag exactly one position per point as its primary. `seen_primary`
        // matters when nlist == 1, where the secondary assignment degenerates
        // to the primary and both positions would otherwise claim the tag.
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

        // Pre-rotate once at build so the query path never pays `dim^2` per
        // probed cell.
        let centroids_rotated: Vec<T> = centroids
            .chunks_exact(dim)
            .flat_map(|c| codebook.rotate(c))
            .collect();

        Ok(Self {
            quantised_codes,
            dim,
            n,
            metric,
            centroids,
            centroids_rotated,
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
            + self.centroids_rotated.capacity() * std::mem::size_of::<T>()
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
    /// One ADC lookup table per probed cell, built from the *rotated* query
    /// residual so it matches the space the codebooks live in. Spilled entries
    /// whose point's primary cell is also probed are skipped, so each point is
    /// scored exactly once and always from its better-quantised copy.
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

        let query_rotated = self.codebook.rotate(&query_vec);

        SOAR_OPQ_PROBED.with(|cell| {
            let mut probed_mask = cell.borrow_mut();
            if probed_mask.len() < self.nlist {
                probed_mask.grow(self.nlist);
            }
            for &c in &probed {
                probed_mask.insert(c);
            }

            let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

            for &cluster_idx in &probed {
                // R is orthogonal, so ||q_r - r|| == ||R q_r - R r||, but only
                // when both sides are rotated. The query was rotated once above
                // and the centroids at build, since R(q - c) == R q - R c.
                let centroid_rot =
                    &self.centroids_rotated[cluster_idx * self.dim..(cluster_idx + 1) * self.dim];
                let residual_rot = T::subtract_simd(&query_rotated, centroid_rot);
                let lookup_tables = self.build_lookup_tables_direct(&residual_rot);

                let start = self.offsets[cluster_idx];
                let end = self.offsets[cluster_idx + 1];

                let mut worst_dist = if heap.len() >= k {
                    heap.peek().unwrap().0 .0
                } else {
                    T::infinity()
                };

                for pos in start..end {
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
    /// Each point is reconstructed from its *primary* copy, which carries the
    /// smaller residual. `decode` inverse-rotates, so the reconstruction lands
    /// back in the original space and can be fed straight to `query`.
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

        fix_neg_dist(&mut final_dists);
        Ok((final_indices, final_dists))
    }
}

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for SoarOpqIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "soar_opq";
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

    /// Sub-codebook size for the fixtures.
    ///
    /// The default of 256 exceeds the training-set size at these `n`, which is
    /// now a typed error rather than a panic.
    const N_CENTROIDS: usize = 16;

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

    fn build(n: usize, dim: usize, nlist: usize, metric: Dist) -> (SoarOpqIndex<f32>, Vec<f32>) {
        let data = clustered_data(n, dim, 8, 11);
        let mat = as_mat(&data, n, dim);
        let idx = SoarOpqIndex::build(
            mat.as_ref(),
            Some(nlist),
            8,
            metric,
            None,
            None,
            Some(N_CENTROIDS),
            Some(1),
            1,
            false,
        )
        .unwrap();
        (idx, data)
    }

    #[test]
    fn test_soar_opq_every_point_encoded_twice() {
        let (idx, _) = build(400, 32, 10, Dist::SquaredEuclidean);
        assert_eq!(idx.original_ids.len(), 800);
        assert_eq!(idx.offsets[10], 800);
        assert_eq!(idx.quantised_codes.len(), 800 * 8);

        let mut counts = vec![0usize; 400];
        for &p in &idx.original_ids {
            counts[p as usize] += 1;
        }
        assert!(counts.iter().all(|&c| c == 2));

        let primaries = idx.dedup_key.iter().filter(|&&k| k == u32::MAX).count();
        assert_eq!(primaries, 400);
    }

    #[test]
    fn test_soar_opq_primary_offsets_count_distinct_points() {
        let (idx, _) = build(400, 32, 10, Dist::SquaredEuclidean);
        assert_eq!(idx.primary_offsets[10], 400);
    }

    #[test]
    fn test_soar_opq_no_duplicate_ids_in_results() {
        let (idx, data) = build(500, 32, 8, Dist::SquaredEuclidean);
        for i in (0..500).step_by(31) {
            let q = &data[i * 32..(i + 1) * 32];
            let (indices, _) = idx.query(q, 20, Some(8)).unwrap();
            let unique: HashSet<usize> = indices.iter().copied().collect();
            assert_eq!(unique.len(), indices.len(), "duplicates for query {}", i);
        }
    }

    #[test]
    fn test_soar_opq_query_returns_k_results() {
        let (idx, data) = build(400, 32, 10, Dist::SquaredEuclidean);
        let (indices, dists) = idx.query(&data[0..32], 12, Some(3)).unwrap();
        assert_eq!(indices.len(), 12);
        assert_eq!(dists.len(), 12);
        assert!(dists.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn test_soar_opq_query_expands_nprobe_when_underfilling() {
        let (idx, data) = build(200, 32, 20, Dist::SquaredEuclidean);
        let (indices, _) = idx.query(&data[0..32], 50, Some(1)).unwrap();
        assert_eq!(indices.len(), 50);
    }

    #[test]
    fn test_soar_opq_finds_self_in_top_k() {
        let (idx, data) = build(400, 32, 8, Dist::SquaredEuclidean);
        let mut hits = 0;
        for i in (0..400).step_by(17) {
            let q = &data[i * 32..(i + 1) * 32];
            let (indices, _) = idx.query(q, 10, Some(8)).unwrap();
            if indices.contains(&i) {
                hits += 1;
            }
        }
        assert!(hits >= 20, "only {} self-hits", hits);
    }

    #[test]
    fn test_soar_opq_cosine_builds_and_queries() {
        let (idx, data) = build(400, 32, 10, Dist::Cosine);
        assert!(matches!(idx.rule(), SoarRule::Orthogonal { .. }));
        let (indices, _) = idx.query(&data[0..32], 10, Some(10)).unwrap();
        let unique: HashSet<usize> = indices.iter().copied().collect();
        assert_eq!(unique.len(), indices.len());
    }

    #[test]
    fn test_soar_opq_single_cluster_degenerates() {
        let (idx, data) = build(200, 32, 1, Dist::SquaredEuclidean);
        let (indices, _) = idx.query(&data[0..32], 10, Some(1)).unwrap();
        let unique: HashSet<usize> = indices.iter().copied().collect();
        assert_eq!(unique.len(), indices.len());
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_soar_opq_manhattan_rejected() {
        let data = clustered_data(100, 32, 4, 1);
        let mat = as_mat(&data, 100, 32);
        assert!(SoarOpqIndex::build(
            mat.as_ref(),
            None,
            8,
            Dist::Manhattan,
            None,
            None,
            Some(N_CENTROIDS),
            Some(1),
            1,
            false
        )
        .is_err());
    }

    #[test]
    fn test_soar_opq_deterministic() {
        let (a, _) = build(300, 32, 8, Dist::SquaredEuclidean);
        let (b, _) = build(300, 32, 8, Dist::SquaredEuclidean);
        assert_eq!(a.original_ids, b.original_ids);
        assert_eq!(a.quantised_codes, b.quantised_codes);
    }

    #[test]
    fn test_soar_opq_generate_knn_shape() {
        let (idx, _) = build(300, 32, 8, Dist::SquaredEuclidean);
        let (indices, dists) = idx.generate_knn(7, Some(8), true, false).unwrap();
        assert_eq!(indices.len(), 300);
        assert!(indices.iter().all(|row| row.len() == 7));
        assert!(dists.unwrap().iter().all(|row| row.len() == 7));
    }

    #[test]
    fn test_soar_opq_too_few_samples_for_codebook_errors() {
        // 100 rows cannot seed the default 256-centroid codebook. Used to panic
        // inside `fast_random_init`.
        let data = clustered_data(100, 32, 4, 1);
        let mat = as_mat(&data, 100, 32);
        let err = SoarOpqIndex::build(
            mat.as_ref(),
            Some(4),
            8,
            Dist::SquaredEuclidean,
            None,
            None,
            None,
            Some(1),
            1,
            false,
        );
        assert!(matches!(
            err,
            Err(AnnSearchErrors::TooFewSamplesForCentroids { .. })
        ));
    }
}
