//! Batched GPU NN-Descent for datasets that do not fit one device binding.
//!
//! [`build_knn_graph_gpu`](crate::gpu::nndescent_gpu::build_knn_graph_gpu)
//! uploads the whole dataset as a single `[n, dim_padded]` tensor and holds it
//! for the entire build, so it dies at the per-binding ceiling. The proposal
//! buffers `[n, MAX_PROPOSALS]` are usually what busts first, not the data: at
//! `f32` each is a flat 512 bytes per point whatever the dimensionality, which
//! matches the data itself at dim 128 and beats it below that.
//!
//! This module partitions instead. Balanced k-means on a subsample gives the
//! cluster centres, every point joins its two nearest clusters, NN-Descent runs
//! on one cluster at a time, and each subgraph is merged into a global graph on
//! the host. Only one cluster is device-resident at a time, so the binding
//! ceiling applies to the cluster rather than the dataset.

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use rayon::prelude::*;
use std::time::Instant;
use thousands::*;

use crate::gpu::k_means_gpu::{train_centroids_gpu, KMeansGpuParams};
use crate::gpu::nndescent_gpu::{
    default_forest_trees, nndescent_core, KnnGraphGpu, NnDescentCfg, DEFAULT_DELTA,
    DEFAULT_MAX_ITERS, DEFAULT_RHO, MAX_PROPOSALS,
};
use crate::prelude::*;
use crate::utils::k_means_utils::{
    assign_all_parallel_top_m, invert_assignments_csr, sample_vectors,
};
use crate::utils::nndescent_utils::SENTINEL_PID;

////////////
// Consts //
////////////

/// Headroom left on the per-cluster device footprint when choosing `C`.
///
/// `0.6` tolerates a largest cluster two thirds above the mean before anything
/// busts, which is well clear of what balanced k-means produces in practice.
const CLUSTER_BUDGET_FRACTION: f64 = 0.6;

/// Largest `C` the planner will propose.
const MAX_CLUSTERS: usize = 4096;

/// Default fraction of the dataset used to train the cluster centroids.
///
/// The cuML batching scheme uses 10% and notes that is normally enough to
/// recover a usable set of centres.
const DEFAULT_SAMPLE_FRAC: f32 = 0.1;

/// Clusters each point joins by default.
const DEFAULT_N_ASSIGN: usize = 2;

////////////
// Params //
////////////

/// Knobs specific to the batched build.
///
/// The NN-Descent knobs themselves are passed separately, unchanged from the
/// unbatched path.
#[derive(Clone, Copy, Debug)]
pub struct ClusteredBuildParams {
    /// Clusters to partition into. `None` derives it from the device limits
    /// via [`plan_cluster_count`].
    pub n_clusters: Option<usize>,
    /// Fraction of the dataset used to train the centroids.
    pub sample_frac: f32,
    /// Clusters each point joins. 1 disables the overlap.
    pub n_assign: usize,
    /// GPU k-means parameters. `None` takes the defaults with balancing on,
    /// which is what keeps the largest cluster inside the planned budget.
    pub kmeans: Option<KMeansGpuParams>,
}

impl ClusteredBuildParams {
    /// Generate a new instance of self
    ///
    /// ### Params
    ///
    /// * `n_clusters` - Clusters to partition into; `None` plans from the
    ///   device limits
    /// * `sample_frac` - Fraction of the dataset used to train the centroids
    /// * `n_assign` - Clusters each point joins
    /// * `kmeans` - Optional [`KMeansGpuParams`]
    ///
    /// ### Returns
    ///
    /// [`ClusteredBuildParams`]
    pub fn new(
        n_clusters: Option<usize>,
        sample_frac: f32,
        n_assign: usize,
        kmeans: Option<KMeansGpuParams>,
    ) -> Self {
        Self {
            n_clusters,
            sample_frac,
            n_assign,
            kmeans,
        }
    }
}

/// Default implementation for [ClusteredBuildParams]
impl Default for ClusteredBuildParams {
    fn default() -> Self {
        Self::new(None, DEFAULT_SAMPLE_FRAC, DEFAULT_N_ASSIGN, None)
    }
}

//////////////
// Planning //
//////////////

/// Bytes one binding of the NN-Descent working set needs for `m` points.
///
/// Returns the largest single binding, not the sum: the per-binding limit is
/// what rejects an allocation, and on unified memory the total is bounded by
/// host RAM rather than by anything queryable. The three candidates are the
/// vectors, one graph plane and one proposal plane.
///
/// The proposal plane is the one that surprises people. It is
/// `MAX_PROPOSALS * elem_bytes` per point regardless of `dim` and of
/// `build_k`, which at `f32` is 512 bytes: exactly the size of the data at
/// dim 128. Below that dimensionality the proposals, not the embedding, set
/// the ceiling.
///
/// ### Params
///
/// * `m` - Points in the cluster
/// * `dim_padded` - Padded embedding dimensionality
/// * `build_k` - NN-Descent working degree
/// * `elem_bytes` - Size of the float element type in bytes
///
/// ### Returns
///
/// Bytes of the largest single binding the build would allocate.
pub fn cluster_binding_bytes(
    m: usize,
    dim_padded: usize,
    build_k: usize,
    elem_bytes: usize,
) -> usize {
    let vectors = m * dim_padded * elem_bytes;
    let graph = m * build_k * elem_bytes.max(4);
    let proposals = m * MAX_PROPOSALS * elem_bytes.max(4);

    vectors.max(graph).max(proposals)
}

/// Smallest cluster count whose per-cluster working set fits one binding.
///
/// A pure function of [`GpuLimits`], so it is host-testable at synthetic
/// budgets with no device present, which is what the rest of this crate's
/// staging plans do.
///
/// Returns 1 when the dataset already fits, which callers take as the signal
/// to dispatch to the unbatched path rather than paying the overlap tax for
/// nothing.
///
/// ### Params
///
/// * `n` - Number of vectors in the dataset
/// * `dim_padded` - Padded embedding dimensionality
/// * `build_k` - NN-Descent working degree
/// * `elem_bytes` - Size of the float element type in bytes
/// * `overlap` - Clusters each point joins, so the partition holds
///   `overlap * n` memberships in total
/// * `limits` - Device limits from `GpuLimits::from_client`
///
/// ### Returns
///
/// Cluster count in `1..=MAX_CLUSTERS`. Saturates at the cap rather than
/// erroring: an over-budget cluster fails loudly at allocation, which is a
/// better failure than a silent success on a partition nobody asked for.
pub fn plan_cluster_count(
    n: usize,
    dim_padded: usize,
    build_k: usize,
    elem_bytes: usize,
    overlap: usize,
    limits: &GpuLimits,
) -> usize {
    if n == 0 {
        return 1;
    }

    let budget = (limits.max_binding_bytes as f64 * CLUSTER_BUDGET_FRACTION) as usize;

    // The unbatched path pays no overlap, so it is checked on `n` itself.
    if cluster_binding_bytes(n, dim_padded, build_k, elem_bytes) <= budget {
        return 1;
    }

    let overlap = overlap.max(1);
    let mut c = 2usize;
    while c < MAX_CLUSTERS {
        let m = (overlap * n).div_ceil(c);
        if cluster_binding_bytes(m, dim_padded, build_k, elem_bytes) <= budget {
            return c;
        }
        c += 1;
    }

    MAX_CLUSTERS
}

///////////
// Merge //
///////////

/// Raw pointer wrapper for lock-free writes into the global graph.
///
/// Sound because of how the caller drives it: clusters are merged one at a
/// time, and within one cluster every member is a distinct global node, so no
/// two rayon tasks ever touch the same `g * k` block. Same argument and same
/// shape as [`crate::utils::nndescent_utils::UnsafeGraphPtr`], which cannot be
/// reused directly here because the global graph is a `(usize, T)` pair buffer
/// rather than a `Neighbour<T>` one.
///
/// ### Fields
#[derive(Copy, Clone)]
struct UnsafeRowPtr<T>(
    /// Raw mutable pointer to the flat `(id, distance)` buffer
    *mut (usize, T),
);

unsafe impl<T> Send for UnsafeRowPtr<T> {}
unsafe impl<T> Sync for UnsafeRowPtr<T> {}

/// Merge one cluster's subgraph into the global kNN graph.
///
/// Local node ids are mapped back through `members`, then each affected global
/// row is rebuilt as the best `k` of its existing entries plus the cluster's
/// `build_k` candidates.
///
/// Deduplication is by sort rather than by membership scan: concatenating and
/// sorting by `(distance, id)` puts any repeat of the same neighbour adjacent,
/// so one linear pass removes it. A point that joined two clusters really does
/// produce repeats.
///
/// ### Params
///
/// * `global` - Global graph `n * k`, updated in place
/// * `k` - Neighbours kept per node
/// * `members` - Global ids of this cluster's points, `m` entries
/// * `graph_idx` - Raw packed local ids from the device, `m * build_k`
/// * `graph_dist` - Matching distances, `m * build_k`
/// * `build_k` - Working degree the cluster ran at
fn merge_cluster_into_global<T>(
    global: &mut [(usize, T)],
    k: usize,
    members: &[u32],
    graph_idx: &[u32],
    graph_dist: &[T],
    build_k: usize,
) where
    T: AnnSearchFloat,
{
    let pid_mask = 0x7FFFFFFFu32;
    let m = members.len();
    let base = UnsafeRowPtr(global.as_mut_ptr());

    (0..m).into_par_iter().for_each(|local| {
        let rows = base;
        let g = members[local] as usize;

        // Soundness guard, not an optimisation.
        if local > 0 && members[local - 1] == members[local] {
            return;
        }

        let mut merged: Vec<(usize, T)> = Vec::with_capacity(k + build_k);

        // Existing global row.
        for slot in 0..k {
            // SAFETY: `g` is unique within this cluster, so this row is not
            // touched by any other task in this loop.
            let entry = unsafe { *rows.0.add(g * k + slot) };
            if entry.0 != SENTINEL_PID {
                merged.push(entry);
            }
        }

        // This cluster's candidates, mapped to global ids.
        for j in 0..build_k {
            let local_pid = (graph_idx[local * build_k + j] & pid_mask) as usize;
            if local_pid >= m || local_pid == local {
                continue;
            }
            let cand = members[local_pid] as usize;
            if cand != g {
                merged.push((cand, graph_dist[local * build_k + j]));
            }
        }

        // Deduplicate by id, then order by distance.
        merged.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        merged.dedup_by_key(|entry| entry.0);
        merged.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        merged.truncate(k);

        for slot in 0..k {
            let value = merged
                .get(slot)
                .copied()
                .unwrap_or((SENTINEL_PID, <T as num_traits::Float>::max_value()));
            // SAFETY: as above, `g * k + slot` is owned by this task alone.
            unsafe { *rows.0.add(g * k + slot) = value };
        }
    });
}

//////////
// Main //
//////////

/// Build a kNN graph on the GPU in balanced batches.
///
/// Partitions the dataset with balanced k-means, runs NN-Descent on one cluster
/// at a time and merges the subgraphs. Only one cluster is device-resident at
/// a time, so the per-binding limit applies to `overlap * n / C` points rather
/// than to `n`. Falls through to
/// [`build_knn_graph_gpu`](crate::gpu::nndescent_gpu::build_knn_graph_gpu) when
/// the dataset already fits, since the batching is a pure cost in that case.
///
/// ### Params
///
/// * `data` - Row-major sample matrix
/// * `metric` - Distance metric (Manhattan is rejected)
/// * `k` - Neighbours per node in the returned graph. Defaults to 30
/// * `build_k` - Working degree during NN-Descent. Defaults to `1.5 * k`
/// * `max_iters` - Maximum NN-Descent iterations per cluster
/// * `n_trees` - Trees for the forest init; recomputed per cluster when `None`
/// * `delta` - Convergence threshold
/// * `rho` - Local-join sampling rate
/// * `refine_knn` - Two-hop refinement sweeps per cluster
/// * `cluster_params` - Optional [`ClusteredBuildParams`]
/// * `seed` - RNG seed for reproducibility
/// * `verbose` - Print per-cluster progress
/// * `device` - CubeCL runtime device
///
/// ### Returns
///
/// Populated [`KnnGraphGpu`], identical in shape to the unbatched path, so it
/// feeds `NsgIndex::build_from_gpu_knn` unchanged.
///
/// ### Note
///
/// CAGRA is deliberately not offered on this path. Both its rank-prune and its
/// beam search need the full vector set resident, so a dataset that needs
/// batching cannot use them anyway.
///
/// ### References
///
/// Park, Becker & Nolet, NVIDIA Technical Blog, 2024
#[allow(clippy::too_many_arguments)]
pub fn build_clustered_knn_graph_gpu<T, R>(
    data: impl AnnMatrix<T>,
    metric: Dist,
    k: Option<usize>,
    build_k: Option<usize>,
    max_iters: Option<usize>,
    n_trees: Option<usize>,
    delta: Option<f32>,
    rho: Option<f32>,
    refine_knn: Option<usize>,
    cluster_params: Option<ClusteredBuildParams>,
    seed: usize,
    verbose: bool,
    device: R::Device,
) -> Result<KnnGraphGpu<T>, AnnSearchErrors>
where
    T: AnnSearchFloat + CubeclFloat,
    R: Runtime,
{
    if metric == Dist::Manhattan {
        return Err(AnnSearchErrors::DistanceNotSupported(metric));
    }

    let params = cluster_params.unwrap_or_default();
    let k = k.unwrap_or(30);
    let build_k = build_k.unwrap_or((1.5 * k as f32) as usize).max(k);
    let n_assign = params.n_assign.max(1);

    let (vectors_flat, n, dim) = data.into_row_major();
    let dim_padded = dim.next_multiple_of(LINE_SIZE);

    let client = R::client(&device);
    let limits = GpuLimits::from_client(&client);

    let n_clusters = params
        .n_clusters
        .unwrap_or_else(|| {
            plan_cluster_count(n, dim_padded, build_k, size_of::<T>(), n_assign, &limits)
        })
        .clamp(1, n.max(1));

    // A point cannot join more clusters than exist, and
    // `assign_all_parallel_top_m` silently clamps to that itself. Clamping here
    // too keeps the width this function passes to `invert_assignments_csr` in
    // step with the width the assignment actually produced; without it, asking
    // for more assignments than clusters indexes past the end of the buffer.
    let n_assign = n_assign.min(n_clusters);

    // One cluster means the whole dataset fits, and the batching is then pure
    // overhead: the overlap alone doubles the work.
    if n_clusters == 1 {
        if verbose {
            println!("Clustered kNN-Graph-GPU: dataset fits one batch, building directly.");
        }
        // Hand over the buffer we already flattened. `FlattenData` converts by
        // identity, so this costs nothing.
        return crate::gpu::nndescent_gpu::build_knn_graph_gpu::<T, R>(
            (vectors_flat, n, dim),
            metric,
            Some(k),
            Some(build_k),
            max_iters,
            n_trees,
            delta,
            rho,
            refine_knn,
            seed,
            verbose,
            device,
        );
    }

    let start = Instant::now();
    let use_cosine = metric == Dist::Cosine;

    let norms: Vec<T> = if use_cosine {
        (0..n)
            .into_par_iter()
            .map(|i| T::calculate_l2_norm(&vectors_flat[i * dim..(i + 1) * dim]))
            .collect()
    } else {
        Vec::new()
    };

    if verbose {
        println!(
            "Clustered kNN-Graph-GPU: {} vectors, dim={}, k={}, build_k={}, {} clusters, {} \
             assignments per point",
            n.separate_with_underscores(),
            dim,
            k,
            build_k,
            n_clusters,
            n_assign,
        );
    }

    // ---- Centroids from a subsample ----

    // The training subsample goes up as one unchunked `[n_train, dim_padded]`
    // binding, so it needs the same ceiling the per-cluster working set gets.
    // Without this the batched path buys `1 / sample_frac` reach rather than
    // lifting the dataset ceiling, and the failure surfaces as an allocation
    // error inside the k-means that says nothing about `sample_frac`. Centroid
    // quality is barely affected: this is a subsample either way, and it only
    // decides the partition, not the graph.
    let train_cap =
        (limits.max_binding_bytes as usize / (dim_padded * size_of::<T>()).max(1)).max(n_clusters);

    let n_train = ((n as f32 * params.sample_frac) as usize)
        .max(n_clusters * 256)
        .min(n)
        .max(n_clusters)
        .min(train_cap);
    let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

    let kmeans_params = params
        .kmeans
        .unwrap_or_else(|| KMeansGpuParams::default().with_balancing(true));

    let centroids = train_centroids_gpu::<T, R>(
        &training_data,
        dim,
        n_train,
        n_clusters,
        &metric,
        Some(kmeans_params),
        seed,
        &client,
        verbose,
    )?;

    // ---- Partition ----

    let centroid_norms: Vec<T> = if use_cosine {
        (0..n_clusters)
            .map(|c| T::calculate_l2_norm(&centroids[c * dim..(c + 1) * dim]))
            .collect()
    } else {
        vec![T::one(); n_clusters]
    };

    // `norms` is empty on the Euclidean path, which is fine: only the cosine
    // arm of the assignment indexes it.
    let assignments = assign_all_parallel_top_m(
        &vectors_flat,
        &norms,
        dim,
        n,
        &centroids,
        &centroid_norms,
        n_clusters,
        n_assign,
        &metric,
    );

    let (members, offsets) = invert_assignments_csr(&assignments, n, n_assign, n_clusters);

    if verbose {
        let sizes: Vec<usize> = (0..n_clusters)
            .map(|c| offsets[c + 1] - offsets[c])
            .collect();
        println!(
            "  Partitioned in {:.2?}: cluster sizes {} .. {} (mean {})",
            start.elapsed(),
            sizes.iter().min().copied().unwrap_or(0),
            sizes.iter().max().copied().unwrap_or(0),
            (n * n_assign) / n_clusters,
        );
    }

    // ---- Per-cluster NN-Descent ----

    let mut global = vec![(SENTINEL_PID, <T as num_traits::Float>::max_value()); n * k];
    let mut converged = true;

    for c in 0..n_clusters {
        let member_ids = &members[offsets[c]..offsets[c + 1]];
        let m = member_ids.len();

        // A cluster too small to hold a graph contributes nothing. At the
        // default `n_assign = 2` its points are covered by the other cluster
        // they joined, so this is free. At `n_assign = 1` there is no other
        // cluster, and a singleton leaves that node's row entirely sentinel --
        // an isolated node in the returned graph rather than a crash, since the
        // layout is sentinel-padded by construction.
        if m <= 1 {
            continue;
        }

        let mut cluster_vectors = vec![T::zero(); m * dim_padded];
        cluster_vectors
            .par_chunks_mut(dim_padded)
            .enumerate()
            .for_each(|(local, row)| {
                let g = member_ids[local] as usize;
                row[..dim].copy_from_slice(&vectors_flat[g * dim..(g + 1) * dim]);
            });

        let cluster_norms: Vec<T> = if use_cosine {
            member_ids.iter().map(|&g| norms[g as usize]).collect()
        } else {
            Vec::new()
        };

        let cfg = NnDescentCfg {
            build_k: build_k.min(m - 1),
            max_iters: max_iters.unwrap_or(DEFAULT_MAX_ITERS),
            n_trees: n_trees.unwrap_or_else(|| default_forest_trees(m)),
            delta: delta.unwrap_or(DEFAULT_DELTA),
            rho_thresh: (rho.unwrap_or(DEFAULT_RHO) * 65535.0) as u32,
            refine_knn: refine_knn.unwrap_or(0),
            // Vary per cluster, otherwise every cluster draws the same random
            // graph and the same forest hyperplanes.
            seed: seed.wrapping_add(c.wrapping_mul(0x9E3779B9)),
            use_cosine,
        };

        let cluster_start = Instant::now();
        let (graph_idx, graph_dist, cluster_converged) = nndescent_core::<T, R>(
            &cluster_vectors,
            &cluster_norms,
            m,
            dim,
            dim_padded,
            &cfg,
            &client,
            &limits,
            false,
        )?;
        converged &= cluster_converged;

        merge_cluster_into_global(
            &mut global,
            k,
            member_ids,
            &graph_idx,
            &graph_dist,
            cfg.build_k,
        );

        if verbose {
            println!(
                "  Cluster {}/{}: {} points, {:.2?}{}",
                c + 1,
                n_clusters,
                m.separate_with_underscores(),
                cluster_start.elapsed(),
                if cluster_converged {
                    ""
                } else {
                    " (no converge)"
                },
            );
        }
    }

    if verbose {
        let filled = global.iter().filter(|e| e.0 != SENTINEL_PID).count();
        println!(
            "  Total build time: {:.2?} ({} of {} graph slots filled)",
            start.elapsed(),
            filled.separate_with_underscores(),
            (n * k).separate_with_underscores(),
        );
    }

    Ok(KnnGraphGpu {
        vectors_flat,
        dim,
        n,
        k,
        norms,
        metric,
        knn_graph: global,
        converged,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Apple Silicon: 32 KiB shared, 4 GiB binding.
    fn limits_with_binding(bytes: u64) -> GpuLimits {
        GpuLimits {
            max_shared_bytes: 32_768,
            max_cube_count: (65_535, 65_535, 65_535),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 1024),
            max_binding_bytes: bytes,
            plane_size_min: 32,
            plane_size_max: 32,
        }
    }

    #[test]
    fn test_proposal_buffer_dominates_below_dim_128() {
        // Not the obvious guess, and the planner leans on it: at f32 the
        // proposal plane is a flat MAX_PROPOSALS * 4 bytes per point whatever
        // `dim` is, so it outweighs the data below dim 128 and ties with it at
        // exactly 128.
        assert_eq!(cluster_binding_bytes(1, 64, 45, 4), MAX_PROPOSALS * 4);
        assert_eq!(cluster_binding_bytes(1, 128, 45, 4), MAX_PROPOSALS * 4);
        assert_eq!(cluster_binding_bytes(1, 128, 45, 4), 128 * 4);
        // Past the tie the vectors take over.
        assert_eq!(cluster_binding_bytes(1, 256, 45, 4), 256 * 4);
        assert_eq!(cluster_binding_bytes(1, 512, 45, 4), 512 * 4);
    }

    #[test]
    fn test_plan_cluster_count_returns_one_when_the_dataset_fits() {
        let limits = limits_with_binding(4 * 1024 * 1024 * 1024);
        assert_eq!(plan_cluster_count(100_000, 128, 45, 4, 2, &limits), 1);
    }

    #[test]
    fn test_plan_cluster_count_fits_at_c_and_not_at_c_minus_one() {
        // The boundary is the thing worth testing: a planner that merely
        // returns something large always "fits".
        let budgets = [128u64 << 20, 1 << 30, 4u64 << 30];
        let elems = [4usize, 8];

        for &bytes in &budgets {
            for &elem in &elems {
                for &(n, dim_padded, build_k) in
                    &[(5_000_000usize, 128usize, 45usize), (2_000_000, 768, 64)]
                {
                    let limits = limits_with_binding(bytes);
                    let c = plan_cluster_count(n, dim_padded, build_k, elem, 2, &limits);
                    assert!((1..=MAX_CLUSTERS).contains(&c));

                    let budget = (bytes as f64 * CLUSTER_BUDGET_FRACTION) as usize;
                    if c > 1 {
                        let m = (2 * n).div_ceil(c);
                        assert!(
                            cluster_binding_bytes(m, dim_padded, build_k, elem) <= budget,
                            "chosen C={c} does not fit at bytes={bytes}, elem={elem}"
                        );
                        if c > 2 {
                            let m_prev = (2 * n).div_ceil(c - 1);
                            assert!(
                                cluster_binding_bytes(m_prev, dim_padded, build_k, elem) > budget,
                                "C={c} is not minimal at bytes={bytes}, elem={elem}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_plan_cluster_count_grows_as_the_binding_shrinks() {
        let big = plan_cluster_count(5_000_000, 128, 45, 4, 2, &limits_with_binding(4u64 << 30));
        let small = plan_cluster_count(5_000_000, 128, 45, 4, 2, &limits_with_binding(128 << 20));
        assert!(
            small > big,
            "a tighter binding must need more clusters: {small} vs {big}"
        );
    }

    /// Points at 0, 1, 2, ... on a line, so every squared distance is
    /// `(i - j)^2` and the expected merge output can be written down.
    fn line_points(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn test_merge_into_empty_global_takes_the_cluster_rows() {
        let (k, build_k, n) = (2usize, 2usize, 4usize);
        let mut global = vec![(SENTINEL_PID, f32::MAX); n * k];
        // Cluster holds global nodes 1 and 3, two apart on the line.
        let members = vec![1u32, 3];
        let graph_idx = vec![1u32, 0x7FFFFFFF, 0u32, 0x7FFFFFFF];
        let graph_dist = vec![4.0f32, f32::MAX, 4.0, f32::MAX];

        merge_cluster_into_global(&mut global, k, &members, &graph_idx, &graph_dist, build_k);

        assert_eq!(global[k], (3, 4.0));
        assert_eq!(global[k + 1].0, SENTINEL_PID);
        assert_eq!(global[3 * k], (1, 4.0));
        // Untouched rows stay sentinel.
        assert_eq!(global[0].0, SENTINEL_PID);
        assert_eq!(global[2 * k].0, SENTINEL_PID);
    }

    #[test]
    fn test_merge_keeps_best_k_and_dedups_across_clusters() {
        let (k, build_k, n) = (2usize, 2usize, 3usize);
        let mut global = vec![(SENTINEL_PID, f32::MAX); n * k];
        // Node 0 already knows node 2, four away.
        global[0] = (2, 4.0);

        // This cluster proposes node 2 again, a genuine repeat, plus the
        // closer node 1.
        let members = vec![0u32, 1, 2];
        let graph_idx = vec![2u32, 1, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF];
        let graph_dist = vec![4.0f32, 1.0, f32::MAX, f32::MAX, f32::MAX, f32::MAX];

        merge_cluster_into_global(&mut global, k, &members, &graph_idx, &graph_dist, build_k);

        assert_eq!(global[0], (1, 1.0), "the closer neighbour must lead");
        assert_eq!(global[1], (2, 4.0), "the repeat must appear exactly once");
    }

    #[test]
    fn test_merge_orders_by_distance_not_slot_order() {
        // The merge ranks across two subgraphs, so it must sort rather than
        // take rows in the order the device happened to lay them out.
        let (k, build_k, n) = (2usize, 2usize, 4usize);
        let mut global = vec![(SENTINEL_PID, f32::MAX); n * k];
        let members: Vec<u32> = (0..n as u32).collect();
        // Row 0 offers node 3 (nine away) first and node 1 (one away) second.
        let graph_idx = vec![
            3u32, 1, //
            0x7FFFFFFF, 0x7FFFFFFF, //
            0x7FFFFFFF, 0x7FFFFFFF, //
            0x7FFFFFFF, 0x7FFFFFFF,
        ];
        let graph_dist = vec![
            9.0f32,
            1.0, //
            f32::MAX,
            f32::MAX, //
            f32::MAX,
            f32::MAX, //
            f32::MAX,
            f32::MAX,
        ];

        merge_cluster_into_global(&mut global, k, &members, &graph_idx, &graph_dist, build_k);

        assert_eq!(global[0], (1, 1.0));
        assert_eq!(global[1], (3, 9.0));
    }

    #[test]
    fn test_assign_top_m_first_column_matches_single_assignment() {
        // Column 0 has to agree with the single-assignment path, otherwise the
        // partition is not the one the rest of the crate would produce.
        let (n, dim, k) = (64usize, 4usize, 4usize);
        let data: Vec<f32> = (0..n * dim)
            .map(|idx| (idx / dim % k) as f32 * 10.0 + (idx % dim) as f32 * 0.01)
            .collect();
        let centroids: Vec<f32> = (0..k * dim)
            .map(|idx| (idx / dim) as f32 * 10.0 + (idx % dim) as f32 * 0.01)
            .collect();
        let norms = vec![1.0f32; n.max(k)];

        let top2 = assign_all_parallel_top_m(
            &data,
            &norms,
            dim,
            n,
            &centroids,
            &norms,
            k,
            2,
            &Dist::SquaredEuclidean,
        );
        let single = crate::utils::k_means_utils::assign_all_parallel(
            &data,
            &norms,
            dim,
            n,
            &centroids,
            &norms,
            k,
            &Dist::SquaredEuclidean,
        );

        for i in 0..n {
            assert_eq!(top2[i * 2] as usize, single[i], "point {i}");
            assert_ne!(top2[i * 2 + 1], top2[i * 2], "runner-up must differ");
        }
    }

    #[test]
    fn test_merge_with_identity_members_matches_compact_knn_rows() {
        // When a "cluster" is the whole dataset, merging its graph into an
        // empty global must be at least as good as what the unbatched path's
        // compaction produces from the same graph, and free of the repeats the
        // compaction leaves in (it takes the first `k` valid slots without
        // deduplicating, so a row offering the same neighbour twice wastes a
        // slot).
        use crate::gpu::nndescent_gpu::compact_knn_rows;

        let (n, k, build_k) = (16usize, 4usize, 8usize);
        let members: Vec<u32> = (0..n as u32).collect();
        let pts = line_points(n);

        // Deterministic pseudo-graph with some sentinels and self-edges. The
        // distances here are the *true* ones and each row is sorted by them,
        // which is the regime both paths assume: neither recomputes, so a
        // fabricated distance would make the comparison meaningless.
        let mut graph_idx = vec![0u32; n * build_k];
        let mut graph_dist = vec![0.0f32; n * build_k];
        for i in 0..n {
            let mut cands: Vec<u32> = (0..build_k)
                .map(|j| {
                    let raw = (i * 7 + j * 3) % (n + 2);
                    if raw >= n {
                        0x7FFFFFFF
                    } else {
                        raw as u32
                    }
                })
                .collect();
            cands.sort_by(|a, b| {
                let da = if *a == 0x7FFFFFFF {
                    f32::MAX
                } else {
                    (pts[i] - pts[*a as usize]).powi(2)
                };
                let db = if *b == 0x7FFFFFFF {
                    f32::MAX
                } else {
                    (pts[i] - pts[*b as usize]).powi(2)
                };
                da.partial_cmp(&db).unwrap()
            });
            for (j, &c) in cands.iter().enumerate() {
                graph_idx[i * build_k + j] = c;
                graph_dist[i * build_k + j] = if c == 0x7FFFFFFF {
                    f32::MAX
                } else {
                    (pts[i] - pts[c as usize]).powi(2)
                };
            }
        }

        let compacted = compact_knn_rows(&graph_idx, &graph_dist, n, k, build_k);

        let mut global = vec![(SENTINEL_PID, f32::MAX); n * k];
        merge_cluster_into_global(&mut global, k, &members, &graph_idx, &graph_dist, build_k);

        for i in 0..n {
            // The k nearest *distinct* candidates the row offers, worked out
            // independently of either implementation.
            let mut cands: Vec<(usize, f32)> = graph_idx[i * build_k..(i + 1) * build_k]
                .iter()
                .map(|raw| (raw & 0x7FFFFFFF) as usize)
                .filter(|&j| j < n && j != i)
                .map(|j| (j, (pts[i] - pts[j]).powi(2)))
                .collect();
            cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
            cands.dedup_by_key(|e| e.0);
            cands.truncate(k);

            let got: Vec<(usize, f32)> = global[i * k..(i + 1) * k]
                .iter()
                .filter(|e| e.0 != SENTINEL_PID)
                .copied()
                .collect();
            assert_eq!(got, cands, "row {i}");

            // And it is never worse than the compaction, which spends slots on
            // repeats because it does not deduplicate.
            let compact_ids: Vec<usize> = compacted[i * k..(i + 1) * k]
                .iter()
                .map(|e| e.0)
                .filter(|&j| j != SENTINEL_PID)
                .collect();
            let mut distinct = compact_ids.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert!(
                got.len() >= distinct.len(),
                "row {i}: merge kept {} distinct neighbours, compaction managed {}",
                got.len(),
                distinct.len()
            );
        }
    }

    #[test]
    fn test_invert_assignments_covers_every_membership() {
        let (n, m, k) = (5usize, 2usize, 3usize);
        let assignments = vec![0u32, 1, 1, 2, 0, 2, 2, 0, 1, 1];
        let (members, offsets) = invert_assignments_csr(&assignments, n, m, k);

        assert_eq!(offsets[k], n * m);
        for c in 0..k {
            for &g in &members[offsets[c]..offsets[c + 1]] {
                assert!(
                    assignments[g as usize * m..(g as usize + 1) * m].contains(&(c as u32)),
                    "point {g} landed in cluster {c} it never joined"
                );
            }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "gpu-tests")]
mod device_tests {
    use super::*;
    use crate::gpu::nndescent_gpu::build_knn_graph_gpu;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use faer::Mat;

    fn try_device() -> Option<WgpuDevice> {
        let device = WgpuDevice::DefaultDevice;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WgpuRuntime::client(&device);
        }))
        .ok()
        .map(|_| device)
    }

    /// Well-separated blobs, each an ordered filament, so every point has an
    /// unambiguous nearest neighbourhood and recall means something.
    ///
    /// Two earlier versions of this generator both produced references that
    /// could not measure what the test claims to measure, and both looked like
    /// clustered-build failures:
    ///
    /// - `(i * 31 + j * 17) % 13` jitter cycles every 13 rows for a fixed `j`,
    ///   so each point had several *exact* duplicates. Ground truth was then an
    ///   arbitrary tie-break between identical points.
    /// - Independent per-entry jitter across all `dim` axes makes each blob a
    ///   near-uniform cloud in 8-D, where the 5-NN structure barely exists.
    ///   The exhaustive baseline itself only reached 0.73 recall there.
    ///
    /// Marching the points along one direction fixes both: neighbours in index
    /// are neighbours in space, by a margin far larger than the tie-breaking
    /// jitter.
    fn blob_data(n_blobs: usize, per_blob: usize, dim: usize) -> (Mat<f32>, usize) {
        let n = n_blobs * per_blob;
        let mat = Mat::from_fn(n, dim, |i, j| {
            let blob = i / per_blob;
            let step = (i % per_blob) as f32;
            let centre = if j % n_blobs == blob { 40.0 } else { 0.0 };
            let along = step * 0.5 * ((j + 1) as f32 / dim as f32);
            let jitter = ((i * 7919 + j * 104_729) % 101) as f32 * 0.0005;
            centre + along + jitter
        });
        (mat, n)
    }

    /// Exhaustive ground truth, used to score the batched graph.
    fn brute_force_knn(mat: &Mat<f32>, k: usize) -> Vec<Vec<usize>> {
        let (n, dim) = (mat.nrows(), mat.ncols());
        (0..n)
            .map(|i| {
                let mut scored: Vec<(f32, usize)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let d: f32 = (0..dim)
                            .map(|c| {
                                let diff = mat[(i, c)] - mat[(j, c)];
                                diff * diff
                            })
                            .sum();
                        (d, j)
                    })
                    .collect();
                scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                scored.into_iter().take(k).map(|(_, j)| j).collect()
            })
            .collect()
    }

    /// Fraction of the true k neighbours a flat `n * k` graph recovered.
    fn recall_at_k(got: &[(usize, f32)], truth: &[Vec<usize>], n: usize, k: usize) -> f64 {
        let mut hits = 0usize;
        for i in 0..n {
            let row: Vec<usize> = got[i * k..(i + 1) * k].iter().map(|e| e.0).collect();
            hits += truth[i].iter().filter(|t| row.contains(t)).count();
        }
        hits as f64 / (n * k) as f64
    }

    #[test]
    fn test_clustered_matches_unbatched_when_one_cluster() {
        // Forcing a single cluster must take the direct path, so the two agree
        // on quality. Deliberately not asserting the graphs are bit-identical:
        // `emit_pair` claims proposal slots with an atomic, so which proposals
        // survive an overflowing buffer depends on thread arrival order and two
        // runs of the *same* build already differ.
        let Some(device) = try_device() else { return };
        let k = 5;
        let (mat, n) = blob_data(4, 40, 8);
        let truth = brute_force_knn(&mat, k);
        let params = ClusteredBuildParams::new(Some(1), 0.5, 2, None);

        let clustered = build_clustered_knn_graph_gpu::<f32, WgpuRuntime>(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(5),
            Some(10),
            Some(10),
            None,
            None,
            None,
            None,
            Some(params),
            7,
            false,
            device.clone(),
        )
        .unwrap();

        let direct = build_knn_graph_gpu::<f32, WgpuRuntime>(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(5),
            Some(10),
            Some(10),
            None,
            None,
            None,
            None,
            7,
            false,
            device,
        )
        .unwrap();

        // Structural equivalence first, and this part is deterministic.
        assert_eq!(clustered.n, direct.n);
        assert_eq!(clustered.k, direct.k);
        assert_eq!(clustered.dim, direct.dim);
        assert_eq!(clustered.metric, direct.metric);

        let clustered_recall = recall_at_k(&clustered.knn_graph, &truth, n, k);
        let direct_recall = recall_at_k(&direct.knn_graph, &truth, n, k);

        // Tolerance is deliberately wide. The GPU NN-Descent is not
        // deterministic -- proposal slots are claimed with an atomic -- and two
        // runs of the *same* build move by up to ~0.05 recall on this data, so
        // a tight bound here is a flaky test rather than a strict one. It stays
        // useful because the regression it guards against, the batched path
        // silently ranking on desynced device distances, was worth 0.19.
        //
        // No absolute floor either: `n_clusters = 1` dispatches to the direct
        // path verbatim, so this can only ever measure that path's own quality.
        assert!(
            (clustered_recall - direct_recall).abs() < 0.12,
            "single-cluster path diverged from direct: {clustered_recall:.3} vs {direct_recall:.3}"
        );
    }

    /// The batched machinery, driven by hand for a "cluster" that is the whole
    /// dataset at the same seed, must land on the same quality as the direct
    /// build. Isolates `nndescent_core` plus the merge from the partitioning
    /// around them.
    #[test]
    fn test_one_full_cluster_reproduces_the_direct_build() {
        let Some(device) = try_device() else { return };
        let (k, build_k, iters) = (5usize, 20usize, 30usize);
        let (mat, n) = blob_data(8, 60, 8);
        let dim = mat.ncols();
        let truth = brute_force_knn(&mat, k);

        let direct = build_knn_graph_gpu::<f32, WgpuRuntime>(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(k),
            Some(build_k),
            Some(iters),
            None,
            None,
            None,
            None,
            7,
            false,
            device.clone(),
        )
        .unwrap();
        let direct_recall = recall_at_k(&direct.knn_graph, &truth, n, k);

        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                flat.push(mat[(i, j)]);
            }
        }
        let client = WgpuRuntime::client(&device);
        let limits = GpuLimits::from_client(&client);

        // The driver trains centroids on this same client before any
        // NN-Descent runs. Doing it here too keeps the isolation honest: it is
        // the one thing the hand-driven path would otherwise skip.
        let _ = train_centroids_gpu::<f32, WgpuRuntime>(
            &flat,
            dim,
            n,
            2,
            &Dist::SquaredEuclidean,
            Some(KMeansGpuParams::default().with_balancing(true)),
            7,
            &client,
            false,
        )
        .unwrap();

        let cfg = NnDescentCfg {
            build_k,
            max_iters: iters,
            n_trees: default_forest_trees(n),
            delta: 0.001,
            rho_thresh: (0.5 * 65535.0) as u32,
            refine_knn: 0,
            seed: 7,
            use_cosine: false,
        };

        let (graph_idx, graph_dist, _) = nndescent_core::<f32, WgpuRuntime>(
            &flat,
            &[],
            n,
            dim,
            dim,
            &cfg,
            &client,
            &limits,
            false,
        )
        .unwrap();

        let members: Vec<u32> = (0..n as u32).collect();
        let mut global = vec![(SENTINEL_PID, f32::MAX); n * k];
        merge_cluster_into_global(&mut global, k, &members, &graph_idx, &graph_dist, build_k);
        let merged_recall = recall_at_k(&global, &truth, n, k);

        println!("  direct {direct_recall:.3} vs hand-merged {merged_recall:.3}");
        assert!(
            merged_recall >= direct_recall - 0.02,
            "hand-driven batched path {merged_recall:.3} fell below direct {direct_recall:.3}"
        );
    }

    #[test]
    fn test_clustered_recall_holds_across_cluster_counts() {
        let Some(device) = try_device() else { return };
        let (n_blobs, per_blob, dim, k) = (8usize, 60usize, 8usize, 5usize);
        let (mat, n) = blob_data(n_blobs, per_blob, dim);
        let truth = brute_force_knn(&mat, k);

        let baseline = build_knn_graph_gpu::<f32, WgpuRuntime>(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            Some(k),
            Some(4 * k),
            Some(30),
            None,
            None,
            None,
            None,
            7,
            false,
            device.clone(),
        )
        .unwrap();
        let base_recall = recall_at_k(&baseline.knn_graph, &truth, n, k);
        println!("  direct baseline recall {base_recall:.3}");

        for &c in &[2usize, 4, 8] {
            let params = ClusteredBuildParams::new(Some(c), 0.5, 2, None);
            let graph = build_clustered_knn_graph_gpu::<f32, WgpuRuntime>(
                mat.as_ref(),
                Dist::SquaredEuclidean,
                Some(k),
                Some(2 * k),
                Some(20),
                None,
                None,
                None,
                None,
                Some(params),
                7,
                false,
                device.clone(),
            )
            .unwrap();

            // Guard the silent-zero failure mode before trusting recall: every
            // launch here is `launch_unchecked`, so a cluster that busts a
            // limit returns zeros and reports nothing.
            let filled = graph
                .knn_graph
                .iter()
                .filter(|e| e.0 != SENTINEL_PID)
                .count();
            assert_eq!(
                filled,
                n * k,
                "graph has empty slots at C={c}; a cluster kernel probably did no work"
            );

            let dup_rows = (0..n)
                .filter(|&i| {
                    let mut ids: Vec<usize> = graph.knn_graph[i * k..(i + 1) * k]
                        .iter()
                        .map(|e| e.0)
                        .collect();
                    ids.sort_unstable();
                    let before = ids.len();
                    ids.dedup();
                    ids.len() != before
                })
                .count();
            assert_eq!(dup_rows, 0, "C={c} produced rows with repeated neighbours");

            // Every returned distance must belong to the pair it is filed
            // under. The merge ranks across two independently built subgraphs
            // and trusts the device's distances to do it, so a torn
            // (index, distance) pair promotes a wrong neighbour here rather
            // than merely mislabelling a right one.
            let bad_dist = (0..n)
                .map(|i| {
                    graph.knn_graph[i * k..(i + 1) * k]
                        .iter()
                        .filter(|(j, d)| {
                            let true_d: f32 = (0..dim)
                                .map(|c| {
                                    let diff = mat[(i, c)] - mat[(*j, c)];
                                    diff * diff
                                })
                                .sum();
                            (true_d - d).abs() > 1e-3 * true_d.max(1.0)
                        })
                        .count()
                })
                .sum::<usize>();
            assert_eq!(
                bad_dist, 0,
                "C={c} returned distances that do not match their pair"
            );

            let recall = recall_at_k(&graph.knn_graph, &truth, n, k);
            println!("  C={c} recall {recall:.3}");
            assert!(
                recall > 0.9,
                "recall {recall:.3} too low at C={c} on well-separated blobs"
            );
            assert!(
                recall >= base_recall,
                "C={c} recall {recall:.3} fell below the unbatched baseline {base_recall:.3}"
            );
        }
    }

    #[test]
    fn test_clustered_cosine_builds_a_full_graph() {
        let Some(device) = try_device() else { return };
        let (mat, n) = blob_data(6, 40, 12);
        let k = 5;
        let params = ClusteredBuildParams::new(Some(3), 0.5, 2, None);

        let graph = build_clustered_knn_graph_gpu::<f32, WgpuRuntime>(
            mat.as_ref(),
            Dist::Cosine,
            Some(k),
            Some(4 * k),
            Some(30),
            None,
            None,
            None,
            None,
            Some(params),
            7,
            false,
            device,
        )
        .unwrap();

        let filled = graph
            .knn_graph
            .iter()
            .filter(|e| e.0 != SENTINEL_PID)
            .count();
        assert_eq!(filled, n * k);
        // Rows stay sorted ascending, which the merge is responsible for.
        for i in 0..n {
            let row = &graph.knn_graph[i * k..(i + 1) * k];
            assert!(row.windows(2).all(|w| w[0].1 <= w[1].1), "row {i} unsorted");
            assert!(row.iter().all(|e| e.0 != i), "row {i} contains itself");
        }
    }
}
