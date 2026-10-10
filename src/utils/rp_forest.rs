//! Host-side pieces of the random-projection forest and the NN-Descent
//! graph handoff, shared by the wgpu ([`crate::gpu`]) and MLX
//! ([`crate::mlx`]) builds. The device computes the projections; everything
//! here runs on the host.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::prelude::*;
use crate::utils::heap_structs::OrderedFloat;
use crate::utils::nndescent_utils::SENTINEL_PID;

/// CSR leaves of one tree: `(leaf_points, leaf_offsets, n_leaves)`.
pub(crate) type LeafStructure = (Vec<u32>, Vec<u32>, usize);

/// Trees whose routing data the [`ForestRouter`] keeps.
pub(crate) const N_ROUTER_TREES: usize = 5;

/////////////
// Helpers //
/////////////

/// Default forest size for the NNDescent graph initialisation.
///
/// An `n^0.25` rule, capped at 20. Sits in its own function because the
/// clustered driver recomputes it per cluster rather than once for the whole
/// dataset: a cluster of `2n/C` points wants the forest its own size implies,
/// not the one the full dataset would.
///
/// ### Params
///
/// * `n` - Number of vectors the forest will index
///
/// ### Returns
///
/// Number of random-projection trees to build.
pub fn default_forest_trees(n: usize) -> usize {
    (5 + ((n as f64).powf(0.25)).round() as usize).min(20)
}

/// Compact the wide NNDescent working graph down to `k` neighbours per node.
///
/// The device keeps `build_k` slots per node so the descent has room to
/// manoeuvre; callers want `k`. Drops self-edges, sentinels and out-of-range
/// ids, keeps the first `k` survivors, and sorts each row ascending by
/// distance.
///
/// ### Params
///
/// * `graph_idx` - Raw packed ids from the device, `n * build_k`; the top bit
///   is the is-new flag and is masked off here
/// * `graph_dist` - Matching distances, `n * build_k`
/// * `n` - Number of nodes
/// * `k` - Neighbours to keep per node
/// * `build_k` - Working degree the device ran at
///
/// ### Returns
///
/// Flat `n * k` graph, row `i` at `[i * k, (i + 1) * k)`, unfilled slots left
/// as `(SENTINEL_PID, T::max_value())`.
pub fn compact_knn_rows<T>(
    graph_idx: &[u32],
    graph_dist: &[T],
    n: usize,
    k: usize,
    build_k: usize,
) -> Vec<(usize, T)>
where
    T: AnnSearchFloat,
{
    let pid_mask = SENTINEL_PID as u32;
    let sentinel = SENTINEL_PID;

    let mut knn_graph = vec![(sentinel, <T as num_traits::Float>::max_value()); n * k];

    knn_graph
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(i, slot)| {
            let mut written = 0;
            for j in 0..build_k {
                if written >= k {
                    break;
                }
                let pid = (graph_idx[i * build_k + j] & pid_mask) as usize;
                if pid < n && pid != i && pid != sentinel {
                    slot[written] = (pid, graph_dist[i * build_k + j]);
                    written += 1;
                }
            }
            slot.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        });

    knn_graph
}

/// Build leaf-point arrays from final partition IDs.
///
/// Groups points by partition, sorts them, and builds a CSR-style offset
/// array for subsequent per-leaf device kernels.
///
/// ### Params
///
/// * `partition_ids` - Partition ID per point, length `n`
/// * `n` - Number of points
///
/// ### Returns
///
/// `(leaf_points, leaf_offsets, n_leaves)` where `leaf_points` is the
/// global point IDs sorted by partition, `leaf_offsets` is the CSR offset
/// array of length `n_leaves + 1`, and `n_leaves` is the number of distinct
/// partitions.
pub(crate) fn build_leaf_structure(partition_ids: &[u32], n: usize) -> LeafStructure {
    let mut sorted: Vec<(u32, u32)> = partition_ids
        .iter()
        .enumerate()
        .map(|(i, &pid)| (pid, i as u32))
        .collect();
    sorted.par_sort_unstable_by_key(|&(pid, _)| pid);

    let leaf_points: Vec<u32> = sorted.iter().map(|&(_, idx)| idx).collect();

    let mut leaf_offsets = vec![0u32];
    for i in 1..n {
        if sorted[i].0 != sorted[i - 1].0 {
            leaf_offsets.push(i as u32);
        }
    }
    leaf_offsets.push(n as u32);

    let n_leaves = leaf_offsets.len() - 1;
    (leaf_points, leaf_offsets, n_leaves)
}

/// Compute per-partition median dot values on CPU.
///
/// Used after each random projection step to determine the split threshold
/// for bisecting each partition.
///
/// ### Params
///
/// * `partition_ids` - Current partition ID per point, length `n`
/// * `dot_values` - Dot product of each point with the projection vector,
///   length `n`
/// * `n_partitions` - Number of active partitions at the current tree level
///
/// ### Returns
///
/// Vector of length `n_partitions` containing the median dot value for each
/// partition. Empty partitions retain `T::zero()`.
pub(crate) fn compute_partition_medians<T: AnnSearchFloat>(
    partition_ids: &[u32],
    dot_values: &[T],
    n_partitions: usize,
) -> Vec<T> {
    // add 50% slack to the expected capacity to accommodate uneven splits
    // and drastically reduce reallocation thrashing.
    let expected_cap = dot_values.len() / n_partitions + (dot_values.len() / n_partitions / 2);
    let mut buckets: Vec<Vec<T>> = vec![Vec::with_capacity(expected_cap); n_partitions];

    for (&pid, &dot) in partition_ids.iter().zip(dot_values.iter()) {
        let p = pid as usize;
        if p < n_partitions {
            buckets[p].push(dot);
        }
    }

    buckets
        .into_par_iter()
        .map(|mut bucket| {
            if bucket.is_empty() {
                T::zero()
            } else {
                let mid = bucket.len() / 2;
                bucket.select_nth_unstable_by(mid, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                bucket[mid]
            }
        })
        .collect()
}

/////////////////
// Forest host //
/////////////////

/// Every tree's random projections, generated up front.
///
/// Each level's projection is a function of the tree seed and the level index
/// alone, so the device can compute all dot products in one pass.
///
/// ### Params
///
/// * `n_trees` - Number of trees
/// * `max_depth` - Levels per tree
/// * `dim` - Unpadded dimensionality
/// * `dim_padded` - Row stride of the flat output
/// * `seed` - Base seed
///
/// ### Returns
///
/// `(projections_flat, level_vecs)`: `[n_trees * max_depth, dim_padded]`
/// zero-padded and tree-major, plus the same unit vectors per tree and level
pub(crate) fn forest_projections<T: AnnSearchFloat>(
    n_trees: usize,
    max_depth: usize,
    dim: usize,
    dim_padded: usize,
    seed: usize,
) -> (Vec<T>, Vec<Vec<Vec<T>>>) {
    let mut projections_flat = vec![T::zero(); n_trees * max_depth * dim_padded];
    let mut tree_level_vecs: Vec<Vec<Vec<T>>> = Vec::with_capacity(n_trees);
    for tree_idx in 0..n_trees {
        let tree_seed =
            (seed as u64).wrapping_add((tree_idx as u64).wrapping_mul(0x9E3779B97F4A7C15u64));
        let mut level_vecs: Vec<Vec<T>> = Vec::with_capacity(max_depth);
        for level in 0..max_depth {
            let level_seed =
                tree_seed.wrapping_add((level as u64).wrapping_mul(0x517CC1B727220A95u64));
            let mut rng = SmallRng::seed_from_u64(level_seed);
            let mut random_vec = vec![T::zero(); dim];
            for v in random_vec.iter_mut() {
                *v = T::from_f64(rng.random_range(-1.0..1.0)).unwrap();
            }
            let norm_sq: T = random_vec.iter().map(|x| *x * *x).sum();
            let norm = num_traits::Float::sqrt(norm_sq);
            if norm > T::zero() {
                for x in random_vec.iter_mut() {
                    *x = *x / norm;
                }
            }
            let off = (tree_idx * max_depth + level) * dim_padded;
            projections_flat[off..off + dim].copy_from_slice(&random_vec);
            level_vecs.push(random_vec);
        }
        tree_level_vecs.push(level_vecs);
    }
    (projections_flat, tree_level_vecs)
}

/// Split every tree level by level against per-partition medians, then build
/// the leaves and the query router.
///
/// ### Params
///
/// * `all_dots` - `[n_trees * max_depth, n]` projections, level-major within
///   each tree
/// * `tree_level_vecs` - Projection vectors per tree and level, from
///   [`forest_projections`]
/// * `n` - Number of points
/// * `max_depth` - Levels per tree
/// * `dim` - Unpadded dimensionality
///
/// ### Returns
///
/// `(leaves per tree, router over the first few trees)`
pub(crate) fn partition_forest<T: AnnSearchFloat>(
    all_dots: &[T],
    tree_level_vecs: Vec<Vec<Vec<T>>>,
    n: usize,
    max_depth: usize,
    dim: usize,
) -> (Vec<LeafStructure>, ForestRouter<T>) {
    let n_trees = tree_level_vecs.len();
    let n_router_trees = n_trees.min(N_ROUTER_TREES);

    // (final partition per point, routing projections, routing medians);
    // the routing data only for the router trees.
    type TreeResult<T> = (Vec<u32>, Option<Vec<Vec<T>>>, Option<Vec<Vec<T>>>);
    let all_tree_results: Vec<TreeResult<T>> = tree_level_vecs
        .into_par_iter()
        .enumerate()
        .map(|(tree_idx, level_vecs)| {
            let save_routing = tree_idx < n_router_trees;
            let mut partition_ids = vec![0u32; n];
            let mut routing_vecs = Vec::new();
            let mut routing_medians = Vec::new();

            for (level, random_vec) in level_vecs.into_iter().enumerate() {
                // Level-major within the tree, so each level's block is contiguous.
                let off = (tree_idx * max_depth + level) * n;
                let dot_values = &all_dots[off..off + n];
                let n_partitions = 1usize << level;
                let medians = compute_partition_medians(&partition_ids, dot_values, n_partitions);
                partition_ids
                    .par_iter_mut()
                    .zip(dot_values.par_iter())
                    .for_each(|(pid, &dot)| {
                        let p = *pid as usize;
                        *pid = if dot <= medians[p] {
                            *pid * 2
                        } else {
                            *pid * 2 + 1
                        };
                    });
                if save_routing {
                    routing_vecs.push(random_vec);
                    routing_medians.push(medians);
                }
            }
            (
                partition_ids,
                save_routing.then_some(routing_vecs),
                save_routing.then_some(routing_medians),
            )
        })
        .collect();

    let leaf_structures: Vec<LeafStructure> = all_tree_results
        .par_iter()
        .map(|tree| build_leaf_structure(&tree.0, n))
        .collect();

    let mut router_rvecs = Vec::with_capacity(n_router_trees);
    let mut router_medians = Vec::with_capacity(n_router_trees);
    let mut router_leaves = Vec::with_capacity(n_router_trees);

    for (partition_ids, rvecs_opt, medians_opt) in all_tree_results.into_iter().take(n_router_trees)
    {
        router_rvecs.push(rvecs_opt.unwrap_or_default());
        router_medians.push(medians_opt.unwrap_or_default());

        let max_pid = partition_ids.iter().copied().max().unwrap_or(0) as usize;
        let mut leaves = vec![Vec::new(); max_pid + 1];
        for (i, &pid) in partition_ids.iter().enumerate() {
            leaves[pid as usize].push(i as u32);
        }
        router_leaves.push(leaves);
    }

    let router = ForestRouter {
        random_vecs: router_rvecs,
        medians: router_medians,
        leaves: router_leaves,
        max_depth,
        dim,
        n_trees: n_router_trees,
    };
    (leaf_structures, router)
}

//////////////////
// ForestRouter //
//////////////////

/// Lightweight query-time router reusing the forest's tree structure.
/// Replaces the Annoy index for beam search entry point selection.
pub struct ForestRouter<T: AnnSearchFloat> {
    /// Per tree, per level: random projection vector [n_trees][max_depth][dim]
    random_vecs: Vec<Vec<Vec<T>>>,
    /// Per tree, per level: median per partition [n_trees][max_depth][variable]
    medians: Vec<Vec<Vec<T>>>,
    /// Per tree: leaves[partition_id] -> point indices [n_trees][2^max_depth][]
    leaves: Vec<Vec<Vec<u32>>>,
    /// Tree depth
    max_depth: usize,
    /// Original (unpadded) dimensionality
    dim: usize,
    /// Number of trees stored for routing
    n_trees: usize,
}

impl<T: AnnSearchFloat> ForestRouter<T> {
    /// Route a query through every stored tree using priority-queue
    /// traversal (same strategy as Annoy) to find entry point candidates.
    ///
    /// Explores multiple leaves per tree by backtracking to the most
    /// promising unexplored branches, ranked by distance to the split
    /// hyperplane.
    ///
    /// ### Params
    ///
    /// * `query` - The query for which to identify the entry points
    /// * `max_candidates` - Candidate budget, split evenly across the trees
    ///
    /// ### Returns
    ///
    /// Leaf-co-members
    pub fn find_entry_points(&self, query: &[T], max_candidates: usize) -> Vec<usize> {
        let mut candidates = Vec::new();
        let q = &query[..self.dim];
        let per_tree = (max_candidates / self.n_trees).max(1);

        for t in 0..self.n_trees {
            // Priority queue: (margin to hyperplane, pid, level)
            // Smallest margin = most promising unexplored branch
            let mut pq: BinaryHeap<Reverse<(OrderedFloat<T>, u32, usize)>> = BinaryHeap::new();
            pq.push(Reverse((OrderedFloat(T::zero()), 0u32, 0usize)));

            let mut found = 0usize;

            while let Some(Reverse((_, pid, level))) = pq.pop() {
                if found >= per_tree {
                    break;
                }

                if level >= self.max_depth {
                    // Reached a leaf
                    if let Some(leaf) = self.leaves[t].get(pid as usize) {
                        candidates.extend(leaf.iter().map(|&p| p as usize));
                        found += leaf.len();
                    }
                    continue;
                }

                let dot = T::dot_simd(q, &self.random_vecs[t][level]);
                let median = self.medians[t][level]
                    .get(pid as usize)
                    .copied()
                    .unwrap_or_else(T::zero);
                let margin = if dot <= median {
                    median - dot
                } else {
                    dot - median
                };

                // Go to the preferred side first (margin = 0),
                // push the other side with its actual margin
                let (preferred, other) = if dot <= median {
                    (pid * 2, pid * 2 + 1)
                } else {
                    (pid * 2 + 1, pid * 2)
                };

                pq.push(Reverse((OrderedFloat(T::zero()), preferred, level + 1)));
                pq.push(Reverse((OrderedFloat(margin), other, level + 1)));
            }
        }

        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// The routing data, for a device-side copy.
    ///
    /// ### Returns
    ///
    /// `(projections [tree][level][dim], medians [tree][level][2^level],
    /// leaves [tree][partition][], max_depth)`. A tree's leaf list stops at its
    /// highest occupied partition.
    #[allow(clippy::type_complexity)]
    #[cfg_attr(not(mlx_available), allow(dead_code))]
    pub(crate) fn parts(&self) -> (&[Vec<Vec<T>>], &[Vec<Vec<T>>], &[Vec<Vec<u32>>], usize) {
        (
            &self.random_vecs,
            &self.medians,
            &self.leaves,
            self.max_depth,
        )
    }
}
