//! Inverted file index with SOAR spilling. Every point lives in two Voronoi
//! cells rather than one, and the second cell is picked to fail on *different*
//! queries than the first does.
//!
//! Plain second-nearest spilling is close to useless: for a point at the edge
//! of its cell, the runner-up centroid usually sits just behind the winner, so
//! both copies are lost by the same queries. SOAR replaces the tie-break with a
//! cost that knows about the primary residual, see [`SoarRule`].
//!
//! The published loss is derived for MIPS under a uniform query distribution on
//! the sphere. That applies to cosine unchanged, since cosine ranking on
//! normalised vectors *is* MIPS ranking, but not to squared Euclidean, where
//! the Euclidean-to-MIPS lift pins a query coordinate and kills the premise.
//! [`SoarRule::Shifted`] is the rederivation for that case.
//!
//! Layout note: primary lists stay a contiguous row range exactly as in
//! [`crate::cpu::ivf::IvfIndex`], so the dominant scan loop pays no indirection
//! and the two indices are directly comparable. Only the spilled entries, a
//! minority of every scan, go through a member array.
//!
//! ### References
//!
//! Sun, Simcha, Simcha, Chern & Guo, arXiv:2404.00774, 2024 (SOAR)

use faer::RowRef;
use fixedbitset::FixedBitSet;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::Separable;

use crate::prelude::*;
use crate::utils::k_means_utils::*;
use crate::utils::*;

//////////////////////////
// Thread-local buffers //
//////////////////////////

// Marks which cells the current query probes. A spilled entry is a duplicate of
// a primary one exactly when the point's primary cell is also being probed, so
// this is all the dedup state needed: `nlist` bits rather than `n`, reset in
// O(nprobe) by walking the probe list back.
thread_local! {
    static SOAR_PROBED: RefCell<FixedBitSet> = const { RefCell::new(FixedBitSet::new()) };
}

///////////////
// SoarIndex //
///////////////

/// IVF index with SOAR spilling
///
/// Partitions vectors into `nlist` Voronoi cells with k-means, then gives every
/// point a second cell chosen by a [`SoarRule`]. Queries scan the `nprobe`
/// nearest cells, visiting both the primary residents and the spilled entries
/// whose own primary cell is not already being scanned.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct SoarIndex<T> {
    /// Original vector data, flattened and reordered into primary-cluster order
    pub vectors_flat: Vec<T>,
    /// Embedding dimensions
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Pre-computed norms for Cosine distance (empty for Euclidean), reordered
    pub norms: Vec<T>,
    /// The type of distance the index is designed for
    metric: Dist,
    /// Cluster centres (nlist * dim elements), i.e., the centroids
    centroids: Vec<T>,
    /// The norms of the centroids for Cosine distance calculations
    centroids_norm: Vec<T>,
    /// Primary lists. Cell `c` owns rows `offsets[c]..offsets[c + 1]` of
    /// `vectors_flat` directly, so no member array is needed for them
    offsets: Vec<usize>,
    /// Spilled entries per cell, as internal row ids. `n` entries total, one
    /// per point
    spill_members: Vec<u32>,
    /// Primary cell of each entry in `spill_members`, same length and order.
    /// Stored alongside rather than looked up so the dedup test is a sequential
    /// read instead of a random gather
    spill_primary: Vec<u32>,
    /// Offsets into `spill_members` / `spill_primary`, `nlist + 1` entries
    spill_offsets: Vec<usize>,
    /// Number of clusters in the index
    nlist: usize,
    /// Rule used for the secondary assignment, kept for introspection
    rule: SoarRule,
    /// Internal row -> caller-facing id
    original_ids: Vec<usize>,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for SoarIndex<T>
where
    T: AnnSearchFloat,
{
    fn vectors_flat(&self) -> &[T] {
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

impl<T> CentroidDistance<T> for SoarIndex<T>
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

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for SoarIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////
// Helpers //
/////////////

/// Pick a spilling rule when the caller did not
///
/// Metric-dependent, which is why there is no `Default` impl on [`SoarRule`]:
/// the published quadratic loss is correct for cosine and wrong-signed for
/// squared Euclidean, so a metric-blind default would silently pick badly for
/// half of all callers.
///
/// ### Params
///
/// * `rule` - Caller's choice, `None` to auto-pick
/// * `metric` - Distance metric the index is being built for
///
/// ### Returns
///
/// The rule to build with.
fn resolve_soar_rule(rule: Option<SoarRule>, metric: &Dist) -> SoarRule {
    rule.unwrap_or(match metric {
        Dist::Cosine => SoarRule::Orthogonal {
            lambda: DEFAULT_ORTHOGONAL_LAMBDA,
        },
        _ => SoarRule::Shifted {
            mu: DEFAULT_SHIFT_MU,
        },
    })
}

/////////////////
// Index build //
/////////////////

impl<T> SoarIndex<T>
where
    T: AnnSearchFloat,
{
    /// Build a SOAR-spilled IVF index
    ///
    /// ### Workflow
    ///
    /// 1. Subsamples up to `256 * nlist` vectors for k-means training
    /// 2. Trains `nlist` centroids and assigns every vector to its nearest
    /// 3. Assigns every vector a *second* cluster under `rule`
    /// 4. Reorders vectors into primary-cluster order for locality, and builds
    ///    the spill lists over the reordered rows
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `nlist` - Optional number of clusters. Defaults to `sqrt(n)`. Spilling
    ///   pays off in the fine-cell regime, so values above `sqrt(n)` are worth
    ///   trying here in a way they are not for a plain IVF index.
    /// * `rule` - Optional secondary-assignment rule, see [`SoarRule`]. Defaults
    ///   to the metric-appropriate choice.
    /// * `k_means_params` - Optional k-means trainings parameters, see
    ///   [KMeansTrainingParams]. If not provided, will default to sensible
    ///   defaults.
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print training progress
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    ///
    /// ### Note
    ///
    /// With `nlist == 1` there is no second cell to hand out, so the index
    /// degenerates to an unspilled one rather than erroring.
    pub fn build(
        data: impl AnnMatrix<T>,
        metric: Dist,
        nlist: Option<usize>,
        rule: Option<SoarRule>,
        k_means_params: Option<KMeansTrainingParams>,
        seed: usize,
        verbose: bool,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (vectors_flat, n, dim) = data.into_row_major();

        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    T::calculate_l2_norm(&vectors_flat[start..start + dim])
                })
                .collect()
        } else {
            Vec::new()
        };

        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);
        let rule = resolve_soar_rule(rule, &metric);

        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        if verbose {
            println!(
                "  Generating SOAR index with {} Voronoi cells, rule {:?}.",
                nlist, rule
            );
        }

        let centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
            k_means_params,
            seed,
            verbose,
        )?;

        let centroids_norm = if metric == Dist::Cosine {
            (0..nlist)
                .map(|i| {
                    let start = i * dim;
                    T::calculate_l2_norm(&centroids[start..start + dim])
                })
                .collect()
        } else {
            Vec::new()
        };

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

        // Secondary assignment reads the primary one, so it has to run before
        // `assignments` is consumed by the CSR build.
        let secondary = assign_secondary_soar(
            &vectors_flat,
            &data_norms_for_assignment,
            dim,
            n,
            &centroids,
            &centroids_norm,
            nlist,
            &assignments,
            &rule,
            &metric,
        );

        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        // Reorder vectors into primary-cluster order. `new_to_old` doubles as
        // `original_ids`; nothing needs the inverse map, because the spill
        // lists are built by walking the new order directly.
        let mut new_to_old: Vec<usize> = Vec::with_capacity(n);
        for cluster in 0..nlist {
            new_to_old.extend_from_slice(&all_indices[offsets[cluster]..offsets[cluster + 1]]);
        }

        let mut reordered = Vec::with_capacity(vectors_flat.len());
        let mut reordered_norms = if norms.is_empty() {
            Vec::new()
        } else {
            Vec::with_capacity(n)
        };
        for &old_id in &new_to_old {
            let start = old_id * dim;
            reordered.extend_from_slice(&vectors_flat[start..start + dim]);
            if !norms.is_empty() {
                reordered_norms.push(norms[old_id]);
            }
        }

        // Spill lists index the *reordered* rows, so translate the secondary
        // assignment out of caller-id space on the way in.
        let spill_assign: Vec<u32> = new_to_old.iter().map(|&old| secondary[old]).collect();
        let (spill_members, spill_offsets) = invert_assignments_csr(&spill_assign, n, 1, nlist);

        // Primary cell of every reordered row, from the contiguous ranges.
        let mut row_primary = vec![0u32; n];
        for cluster in 0..nlist {
            for row in offsets[cluster]..offsets[cluster + 1] {
                row_primary[row] = cluster as u32;
            }
        }
        let spill_primary: Vec<u32> = spill_members
            .iter()
            .map(|&row| row_primary[row as usize])
            .collect();

        Ok(Self {
            vectors_flat: reordered,
            dim,
            n,
            norms: reordered_norms,
            metric,
            centroids,
            centroids_norm,
            offsets,
            spill_members,
            spill_primary,
            spill_offsets,
            nlist,
            rule,
            original_ids: new_to_old,
        })
    }

    /// Rule the index was built with
    ///
    /// ### Returns
    ///
    /// The resolved [`SoarRule`], with `None` already turned into a concrete
    /// choice by the metric heuristic.
    pub fn rule(&self) -> SoarRule {
        self.rule
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Index size `in n bytes`
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
            + self.centroids.capacity() * std::mem::size_of::<T>()
            + self.centroids_norm.capacity() * std::mem::size_of::<T>()
            + self.offsets.capacity() * std::mem::size_of::<usize>()
            + self.spill_members.capacity() * std::mem::size_of::<u32>()
            + self.spill_primary.capacity() * std::mem::size_of::<u32>()
            + self.spill_offsets.capacity() * std::mem::size_of::<usize>()
    }
}

///////////
// Query //
///////////

impl<T> SoarIndex<T>
where
    T: AnnSearchFloat,
{
    /// Query the index for approximate nearest neighbours
    ///
    /// Two-stage as in a plain IVF search, but each probed cell is scanned
    /// twice: once over its primary residents, once over the entries spilled
    /// into it. A spilled entry whose own primary cell is also being probed is
    /// skipped, since it was already scored in the first pass. That is the only
    /// duplication possible, because each point has exactly one primary and one
    /// spilled home.
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
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        nprobe: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        let nprobe = nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist);
        let k: usize = k.min(self.n);
        let query_norm = if matches!(self.metric, Dist::Cosine) {
            T::calculate_l2_norm(query_vec)
        } else {
            T::one()
        };

        let mut cluster_dists: Vec<(T, usize)> =
            self.get_centroids_dist(query_vec, query_norm, nprobe);
        let probed = select_probed_clusters(&mut cluster_dists, &self.offsets, nprobe, k);

        let dist_to = |vec_idx: usize| match self.metric {
            Dist::SquaredEuclidean => self.euclidean_distance_to_query(vec_idx, query_vec),
            Dist::Cosine => self.cosine_distance_to_query(vec_idx, query_vec, query_norm),
            Dist::Manhattan => unreachable!(),
        };

        SOAR_PROBED.with(|cell| {
            let mut probed_mask = cell.borrow_mut();
            if probed_mask.len() < self.nlist {
                probed_mask.grow(self.nlist);
            }
            for &c in &probed {
                probed_mask.insert(c);
            }

            let mut buffer = SortedBuffer::with_capacity(k);

            // Primary pass: contiguous rows, identical to a plain IVF scan.
            for &cluster_idx in &probed {
                let start = self.offsets[cluster_idx];
                let end = self.offsets[cluster_idx + 1];
                for vec_idx in start..end {
                    buffer.insert((OrderedFloat(dist_to(vec_idx)), vec_idx), k);
                }
            }

            // Spill pass: gather, skipping anything the primary pass already saw.
            for &cluster_idx in &probed {
                let start = self.spill_offsets[cluster_idx];
                let end = self.spill_offsets[cluster_idx + 1];
                for pos in start..end {
                    if probed_mask.contains(self.spill_primary[pos] as usize) {
                        continue;
                    }
                    let vec_idx = self.spill_members[pos] as usize;
                    buffer.insert((OrderedFloat(dist_to(vec_idx)), vec_idx), k);
                }
            }

            for &c in &probed {
                probed_mask.remove(c);
            }

            let (distances, indices) = buffer
                .data()
                .iter()
                .map(|(d, i)| (d.0, self.original_ids[*i]))
                .unzip();

            Ok((indices, distances))
        })
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies
    /// to a temporary vector. Uses `self.query()` under the hood.
    ///
    /// ### Params
    ///
    /// * `query_row` - Query vector (must match index dimensionality)
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
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
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
    ) -> KnnOptionResult<T> {
        let counter = Arc::new(AtomicUsize::new(0));

        let unordered_results: Vec<(usize, Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dim;
                let vec = &self.vectors_flat[start..start + self.dim];
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

                let (indices, dists) = self.query(vec, k, nprobe)?;
                Ok((orig_id, indices, dists))
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

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for SoarIndex<T>
where
    T: AnnSearchFloat,
{
    fn query_for_validation(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.query(query_vec, k, None)
    }

    fn dim(&self) -> usize {
        self.dim
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

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for SoarIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "soar";
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

    /// Wrap a flat row-major slice as a matrix
    fn as_mat(data: &[f32], n: usize, dim: usize) -> Mat<f32> {
        Mat::from_fn(n, dim, |i, j| data[i * dim + j])
    }

    /// Clustered test data: `n_clusters` Gaussian-ish blobs in `dim` dimensions
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

    #[test]
    fn test_soar_shifted_prefers_far_side_centroid() {
        // Collinear layout on the x axis. The point sits at 10.0 and its primary
        // centroid at 9.5, so the primary residual points at +x.
        //
        //   cA(9.0)   c0(9.5)   x(10.0)          cB(11.2)
        //
        // cA is nearer to x than cB is, so plain second-nearest takes it, even
        // though cA fails on exactly the queries c0 fails on: both sit on the
        // -x side. The shifted rule ranks against x + mu * (x - c0) = 10.5 and
        // takes cB, which is what complementary spilling means.
        let dim = 2;
        let data: Vec<f32> = vec![10.0, 0.0];
        let centroids: Vec<f32> = vec![
            9.5, 0.0, // c0, the primary
            9.0, 0.0, // c1 = cA, aligned with the primary residual
            11.2, 0.0, // c2 = cB, anti-aligned
        ];
        let k = 3;
        let primary = vec![0usize];
        let norms = vec![1.0f32];
        let centroid_norms = vec![1.0f32; k];

        let nearest = assign_secondary_soar(
            &data,
            &norms,
            dim,
            1,
            &centroids,
            &centroid_norms,
            k,
            &primary,
            &SoarRule::Nearest,
            &Dist::SquaredEuclidean,
        );
        assert_eq!(nearest[0], 1, "plain second-nearest should take cA");

        let shifted = assign_secondary_soar(
            &data,
            &norms,
            dim,
            1,
            &centroids,
            &centroid_norms,
            k,
            &primary,
            &SoarRule::Shifted { mu: 1.0 },
            &Dist::SquaredEuclidean,
        );
        assert_eq!(shifted[0], 2, "shifted rule should take the far-side cB");

        // Everything here is collinear, so (r' . r_hat_1)^2 == ||r'||^2 and the
        // published quadratic loss is a pure rescale of the plain distance. It
        // cannot separate the far-side candidate, which is the concrete form the
        // MIPS-to-Euclidean mismatch takes.
        let orthogonal = assign_secondary_soar(
            &data,
            &norms,
            dim,
            1,
            &centroids,
            &centroid_norms,
            k,
            &primary,
            &SoarRule::Orthogonal { lambda: 1.0 },
            &Dist::SquaredEuclidean,
        );
        assert_eq!(
            orthogonal[0], 1,
            "quadratic loss cannot separate a collinear far-side candidate"
        );
    }

    #[test]
    fn test_soar_mu_zero_matches_nearest() {
        let (n, dim, nlist) = (400, 8, 12);
        let data = clustered_data(n, dim, nlist, 7);
        let norms = vec![1.0f32; n];
        let centroids = clustered_data(nlist, dim, nlist, 99);
        let centroid_norms = vec![1.0f32; nlist];
        let primary = assign_all_parallel(
            &data,
            &norms,
            dim,
            n,
            &centroids,
            &centroid_norms,
            nlist,
            &Dist::SquaredEuclidean,
        );

        let call = |rule: SoarRule| {
            assign_secondary_soar(
                &data,
                &norms,
                dim,
                n,
                &centroids,
                &centroid_norms,
                nlist,
                &primary,
                &rule,
                &Dist::SquaredEuclidean,
            )
        };

        assert_eq!(call(SoarRule::Shifted { mu: 0.0 }), call(SoarRule::Nearest));
    }

    #[test]
    fn test_soar_secondary_always_differs_from_primary() {
        let (n, dim, nlist) = (500, 6, 16);
        let data = clustered_data(n, dim, nlist, 3);
        let norms = vec![1.0f32; n];
        let centroids = clustered_data(nlist, dim, nlist, 11);
        let centroid_norms = vec![1.0f32; nlist];
        let primary = assign_all_parallel(
            &data,
            &norms,
            dim,
            n,
            &centroids,
            &centroid_norms,
            nlist,
            &Dist::SquaredEuclidean,
        );

        for rule in [
            SoarRule::Nearest,
            SoarRule::Shifted { mu: 0.5 },
            SoarRule::Orthogonal { lambda: 1.0 },
        ] {
            let sec = assign_secondary_soar(
                &data,
                &norms,
                dim,
                n,
                &centroids,
                &centroid_norms,
                nlist,
                &primary,
                &rule,
                &Dist::SquaredEuclidean,
            );
            for i in 0..n {
                assert_ne!(
                    sec[i] as usize, primary[i],
                    "rule {:?} collided at {}",
                    rule, i
                );
            }
        }
    }

    #[test]
    fn test_soar_no_duplicate_ids_in_results() {
        let (n, dim, nlist) = (600, 8, 10);
        let data = clustered_data(n, dim, nlist, 42);
        let mat = as_mat(&data, n, dim);
        let index = SoarIndex::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(nlist),
            None,
            None,
            1,
            false,
        )
        .unwrap();

        // Probe everything so primary and spill lists overlap as much as they
        // possibly can.
        for i in (0..n).step_by(37) {
            let q = &data[i * dim..(i + 1) * dim];
            let (indices, _) = index.query(q, 20, Some(nlist)).unwrap();
            let unique: HashSet<usize> = indices.iter().copied().collect();
            assert_eq!(unique.len(), indices.len(), "duplicate ids for query {}", i);
        }
    }

    #[test]
    fn test_soar_spill_lists_hold_every_point_once() {
        let (n, dim, nlist) = (300, 4, 9);
        let data = clustered_data(n, dim, nlist, 5);
        let mat = as_mat(&data, n, dim);
        let index = SoarIndex::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(nlist),
            None,
            None,
            1,
            false,
        )
        .unwrap();

        assert_eq!(index.spill_members.len(), n);
        assert_eq!(index.spill_primary.len(), n);
        assert_eq!(index.spill_offsets[nlist], n);
        let unique: HashSet<u32> = index.spill_members.iter().copied().collect();
        assert_eq!(
            unique.len(),
            n,
            "every point should be spilled exactly once"
        );
    }

    #[test]
    fn test_soar_query_returns_full_k_when_nprobe_underfills() {
        // nprobe = 1 cannot reach k on its own; select_probed_clusters has to
        // widen the probe list.
        let (n, dim, nlist) = (200, 4, 20);
        let data = clustered_data(n, dim, nlist, 13);
        let mat = as_mat(&data, n, dim);
        let index = SoarIndex::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(nlist),
            None,
            None,
            1,
            false,
        )
        .unwrap();

        let k = 50;
        let (indices, dists) = index.query(&data[0..dim], k, Some(1)).unwrap();
        assert_eq!(indices.len(), k);
        assert_eq!(dists.len(), k);
    }

    #[test]
    fn test_soar_query_finds_self() {
        let (n, dim, nlist) = (400, 8, 12);
        let data = clustered_data(n, dim, nlist, 21);
        let mat = as_mat(&data, n, dim);
        let index = SoarIndex::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(nlist),
            None,
            None,
            1,
            false,
        )
        .unwrap();

        for i in (0..n).step_by(23) {
            let q = &data[i * dim..(i + 1) * dim];
            let (indices, dists) = index.query(q, 5, Some(nlist)).unwrap();
            assert_eq!(indices[0], i);
            assert!(dists[0].abs() < 1e-4);
        }
    }

    #[test]
    fn test_soar_reproducible() {
        let (n, dim, nlist) = (300, 6, 10);
        let data = clustered_data(n, dim, nlist, 8);
        let mat = as_mat(&data, n, dim);

        let a = SoarIndex::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(nlist),
            None,
            None,
            4,
            false,
        )
        .unwrap();
        let b = SoarIndex::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(nlist),
            None,
            None,
            4,
            false,
        )
        .unwrap();

        assert_eq!(a.original_ids, b.original_ids);
        assert_eq!(a.spill_members, b.spill_members);
        assert_eq!(a.spill_offsets, b.spill_offsets);
    }

    #[test]
    fn test_soar_cosine_builds_and_queries() {
        let (n, dim, nlist) = (400, 8, 12);
        let data = clustered_data(n, dim, nlist, 17);
        let mat = as_mat(&data, n, dim);
        let index = SoarIndex::build(
            mat.as_ref(),
            Dist::Cosine,
            Some(nlist),
            None,
            None,
            1,
            false,
        )
        .unwrap();

        assert!(matches!(index.rule(), SoarRule::Orthogonal { .. }));
        for i in (0..n).step_by(41) {
            let q = &data[i * dim..(i + 1) * dim];
            let (indices, _) = index.query(q, 10, Some(nlist)).unwrap();
            assert_eq!(indices[0], i);
            let unique: HashSet<usize> = indices.iter().copied().collect();
            assert_eq!(unique.len(), indices.len());
        }
    }

    #[test]
    fn test_soar_single_cluster_degenerates() {
        let (n, dim) = (100, 4);
        let data = clustered_data(n, dim, 1, 2);
        let mat = as_mat(&data, n, dim);
        let index = SoarIndex::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(1),
            None,
            None,
            1,
            false,
        )
        .unwrap();

        // Only one cell exists, so every point spills into its own primary and
        // the dedup pass must drop all of it.
        let (indices, _) = index.query(&data[0..dim], 10, Some(1)).unwrap();
        let unique: HashSet<usize> = indices.iter().copied().collect();
        assert_eq!(unique.len(), indices.len());
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_soar_manhattan_rejected() {
        let (n, dim) = (50, 4);
        let data = clustered_data(n, dim, 2, 1);
        let mat = as_mat(&data, n, dim);
        assert!(
            SoarIndex::build(mat.as_ref(), Dist::Manhattan, None, None, None, 1, false).is_err()
        );
    }

    #[test]
    fn test_soar_recall_beats_threshold() {
        let (n, dim, nlist) = (2000, 16, 40);
        let data = clustered_data(n, dim, 20, 77);
        let mat = as_mat(&data, n, dim);
        let index = SoarIndex::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(nlist),
            None,
            None,
            1,
            false,
        )
        .unwrap();

        let recall = index.validate_index(10, 1, Some(200)).unwrap();
        assert!(recall > 0.8, "recall was {}", recall);
    }

    #[test]
    fn test_soar_generate_knn_shape() {
        let (n, dim, nlist) = (300, 6, 10);
        let data = clustered_data(n, dim, nlist, 31);
        let mat = as_mat(&data, n, dim);
        let index = SoarIndex::build(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(nlist),
            None,
            None,
            1,
            false,
        )
        .unwrap();

        let (indices, dists) = index.generate_knn(7, Some(nlist), true, false).unwrap();
        assert_eq!(indices.len(), n);
        assert!(indices.iter().all(|row| row.len() == 7));
        assert!(dists.unwrap().iter().all(|row| row.len() == 7));
    }
}
