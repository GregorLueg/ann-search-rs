//! Kd-Tree forest implementation in ann-search-rs.
//!
//! Uses a forest of randomised Kd-trees for approximate nearest neighbour
//! search. Each tree splits along coordinate axes selected randomly from
//! the top highest-spread dimensions, providing diversity across trees.
//!
//! As with Annoy, the trees ([`KdTrees`]) are kept apart from the vectors and
//! each tree partitions one `u32` permutation in place.

use faer::RowRef;
use fixedbitset::FixedBitSet;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use thousands::*;

use crate::prelude::*;
use crate::utils::heap_structs::*;
use crate::utils::tree_utils::*;
use crate::utils::*;

/////////////
// Helpers //
/////////////

/// Node representation in the flattened Kd-tree structure
///
/// Uses tagged union pattern: `n_descendants == 1` indicates leaf node,
/// otherwise it's a split node. Split data (dimension index and scalar
/// threshold) is embedded directly in the node to avoid indirection
/// through a separate array.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct KdFlatNode<T> {
    /// `1` for leaf, `2` for split node
    n_descendants: u32,
    /// Split: left child index. Leaf: start within its tree's permutation
    child_a: u32,
    /// Split: right child index. Leaf: number of items
    child_b: u32,
    /// Split: dimension along which the split occurs. Leaf: the tree it
    /// belongs to, so the leaf start stays tree-local
    split_dim: u32,
    /// Threshold value for the split (split only)
    split_val: T,
}

/// Number of top-spread dimensions to randomly select from. Provides diversity
/// across trees in the forest without picking completely arbitrary axes.
const TOP_K_DIMS: usize = 3;

/// Maximum sample size for spread estimation and approximate median
/// computation. Keeps build cost per node O(1) rather than O(n).
const SUBSAMPLE_SIZE: usize = 128;

/// Per-tree build scratch, reused across every node of the tree.
struct KdScratch<T> {
    /// Subsampled ids for spread and median estimation
    sample: Vec<u32>,
    /// Sample values along the candidate dimension
    vals: Vec<T>,
    /// Staging for the right-hand items during the stable partition
    right: Vec<u32>,
}

/////////////
// KdTrees //
/////////////

/// The Kd forest of a [`KdTreeIndex`], without the vectors.
///
/// Tree `t` owns `leaf_indices[t * n..(t + 1) * n]`. For Cosine the trees are
/// built on normalised vectors, so [`Self::visit`] wants a normalised query.
#[derive(Clone)]
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct KdTrees<T> {
    /// Flattened tree structure containing all split and leaf nodes
    nodes: Vec<KdFlatNode<T>>,
    /// Starting node of each tree in the forest
    roots: Vec<u32>,
    /// Point ids stored in the leaves, `n` per tree
    leaf_indices: Vec<u32>,
    /// Number of points each tree partitions
    n: usize,
    /// Metric the margin lower bound is taken in
    metric: Dist,
}

impl<T> KdTrees<T>
where
    T: AnnSearchFloat,
{
    /// Build the forest in parallel, one tree per task.
    ///
    /// ### Params
    ///
    /// * `build_vectors` - Flattened vector data, normalised for Cosine
    /// * `n` - Number of vectors
    /// * `dim` - Dimensionality of vectors
    /// * `n_trees` - Number of trees to build
    /// * `metric` - Distance metric, used for the query-time bound
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// The forest, with ids in the order of `build_vectors`.
    pub(crate) fn build(
        build_vectors: &[T],
        n: usize,
        dim: usize,
        n_trees: usize,
        metric: Dist,
        seed: usize,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let seeds: Vec<u64> = (0..n_trees).map(|_| rng.random()).collect();

        let mut leaf_indices = vec![0u32; n_trees * n];
        let parts: Vec<Vec<KdFlatNode<T>>> = leaf_indices
            .par_chunks_mut(n.max(1))
            .zip(seeds.into_par_iter())
            .map(|(perm, tree_seed)| Self::build_tree(build_vectors, dim, tree_seed, perm))
            .collect();

        let total_nodes: usize = parts.iter().map(|t| t.len()).sum();
        let mut nodes = Vec::with_capacity(total_nodes);
        let mut roots = Vec::with_capacity(n_trees);

        for (tree, part) in parts.into_iter().enumerate() {
            let root_offset = nodes.len() as u32;
            roots.push(root_offset);
            nodes.extend(part.into_iter().map(|mut node| {
                if node.n_descendants == 1 {
                    node.split_dim = tree as u32;
                } else {
                    node.child_a += root_offset;
                    node.child_b += root_offset;
                }
                node
            }));
        }

        Self {
            nodes,
            roots,
            leaf_indices,
            n,
            metric,
        }
    }

    /// Build one tree by partitioning `perm` in place.
    ///
    /// ### Params
    ///
    /// * `build_vectors` - Flattened vector data, normalised for Cosine
    /// * `dim` - Dimensionality of vectors
    /// * `seed` - Seed for this tree
    /// * `perm` - Length `n`, overwritten with the tree's leaf order
    ///
    /// ### Returns
    ///
    /// The tree's nodes in pre-order, indexed from its root.
    pub(crate) fn build_tree(
        build_vectors: &[T],
        dim: usize,
        seed: u64,
        perm: &mut [u32],
    ) -> Vec<KdFlatNode<T>> {
        for (i, p) in perm.iter_mut().enumerate() {
            *p = i as u32;
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut scratch = KdScratch {
            sample: Vec::with_capacity(SUBSAMPLE_SIZE),
            vals: Vec::with_capacity(SUBSAMPLE_SIZE),
            right: Vec::with_capacity(perm.len()),
        };
        let mut nodes = Vec::new();
        Self::build_node(
            build_vectors,
            dim,
            perm,
            0,
            &mut nodes,
            &mut scratch,
            &mut rng,
        );
        nodes
    }

    /// Recursively build a node (split or leaf) over `items`.
    ///
    /// Selects a split dimension randomly from the top highest-spread
    /// dimensions (computed on a subsample for efficiency), then splits
    /// at the approximate median. The partition is stable.
    ///
    /// Falls back to a leaf if the node is small enough, the data is
    /// degenerate (zero spread on all dimensions), or no candidate
    /// dimension produces a split where both children are strictly
    /// smaller than the parent.
    ///
    /// ### Params
    ///
    /// * `build_vectors` - Flattened vector data (normalised for Cosine)
    /// * `dim` - Dimensionality of vectors
    /// * `items` - This node's slice of the tree permutation
    /// * `start` - Offset of `items` within the permutation
    /// * `nodes` - Accumulator for built nodes
    /// * `scratch` - Tree scratch
    /// * `rng` - Random number generator
    ///
    /// ### Returns
    ///
    /// Index of the created node in `nodes`
    fn build_node(
        build_vectors: &[T],
        dim: usize,
        items: &mut [u32],
        start: usize,
        nodes: &mut Vec<KdFlatNode<T>>,
        scratch: &mut KdScratch<T>,
        rng: &mut StdRng,
    ) -> usize {
        let leaf = |nodes: &mut Vec<KdFlatNode<T>>, len: usize| {
            nodes.push(KdFlatNode {
                n_descendants: 1,
                child_a: start as u32,
                child_b: len as u32,
                split_dim: 0,
                split_val: T::zero(),
            });
            nodes.len() - 1
        };

        if items.len() <= LEAF_MIN_MEMBERS {
            return leaf(nodes, items.len());
        }

        scratch.sample.clear();
        if items.len() > SUBSAMPLE_SIZE {
            for _ in 0..SUBSAMPLE_SIZE {
                scratch.sample.push(items[rng.random_range(0..items.len())]);
            }
        } else {
            scratch.sample.extend_from_slice(items);
        }

        let top_dims = Self::top_spread_dims(build_vectors, dim, &scratch.sample);

        if top_dims.is_empty() {
            return leaf(nodes, items.len());
        }

        // Shuffle top dims so we try them in random order
        let mut candidate_dims = top_dims;
        for i in (1..candidate_dims.len()).rev() {
            let j = rng.random_range(0..=i);
            candidate_dims.swap(i, j);
        }

        for &(split_dim, _) in &candidate_dims {
            // compute approximate median from the sample
            scratch.vals.clear();
            scratch.vals.extend(
                scratch
                    .sample
                    .iter()
                    .map(|&idx| build_vectors[idx as usize * dim + split_dim]),
            );
            scratch
                .vals
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let split_val = scratch.vals[scratch.vals.len() / 2];
            let is_left = |item: u32| build_vectors[item as usize * dim + split_dim] < split_val;

            let n_left = items.iter().filter(|&&item| is_left(item)).count();

            // both children must be non-empty
            if n_left == 0 || n_left == items.len() {
                continue;
            }

            // Stable partition: lefts compact forwards in place, rights wait
            // in scratch and land behind them.
            scratch.right.clear();
            let mut write = 0;
            for read in 0..items.len() {
                let item = items[read];
                if is_left(item) {
                    items[write] = item;
                    write += 1;
                } else {
                    scratch.right.push(item);
                }
            }
            items[n_left..].copy_from_slice(&scratch.right);

            let node_idx = nodes.len();
            nodes.push(KdFlatNode {
                n_descendants: 2,
                child_a: 0,
                child_b: 0,
                split_dim: split_dim as u32,
                split_val,
            });

            let (left_items, right_items) = items.split_at_mut(n_left);
            let left_idx =
                Self::build_node(build_vectors, dim, left_items, start, nodes, scratch, rng);
            let right_idx = Self::build_node(
                build_vectors,
                dim,
                right_items,
                start + n_left,
                nodes,
                scratch,
                rng,
            );

            nodes[node_idx].child_a = left_idx as u32;
            nodes[node_idx].child_b = right_idx as u32;
            return node_idx;
        }

        // No dimension produced a valid split -- fall back to leaf
        leaf(nodes, items.len())
    }

    /// Find the top-k dimensions by spread (max - min)
    ///
    /// Evaluates the range of values along each dimension for the given
    /// sample. Returns up to TOP_K_DIMS dimensions with the largest
    /// spread, excluding any with zero spread.
    ///
    /// ### Params
    ///
    /// * `build_vectors` - Flattened vector data
    /// * `dim` - Dimensionality
    /// * `sample` - Subsampled indices to evaluate
    ///
    /// ### Returns
    ///
    /// Up to TOP_K_DIMS `(dimension_index, spread)` pairs sorted descending by
    /// spread
    fn top_spread_dims(build_vectors: &[T], dim: usize, sample: &[u32]) -> Vec<(usize, T)> {
        let mut spreads: Vec<(usize, T)> = (0..dim)
            .map(|d| {
                let mut min_val = T::infinity();
                let mut max_val = T::neg_infinity();
                for &idx in sample {
                    let val = build_vectors[idx as usize * dim + d];
                    if val < min_val {
                        min_val = val;
                    }
                    if val > max_val {
                        max_val = val;
                    }
                }
                (d, max_val - min_val)
            })
            .collect();

        spreads.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let k = TOP_K_DIMS.min(dim);
        spreads.truncate(k);
        spreads.retain(|&(_, s)| s > T::zero());

        spreads
    }

    /// Number of trees in the forest.
    ///
    /// ### Returns
    ///
    /// The tree count.
    pub(crate) fn n_trees(&self) -> usize {
        self.roots.len()
    }

    /// Items of a leaf node.
    ///
    /// ### Params
    ///
    /// * `node` - A leaf node of this forest
    ///
    /// ### Returns
    ///
    /// The leaf's ids.
    #[inline(always)]
    fn leaf_items(&self, node: &KdFlatNode<T>) -> &[u32] {
        let start = node.split_dim as usize * self.n + node.child_a as usize;
        // SAFETY: leaf ranges are built inside their tree's chunk
        unsafe {
            self.leaf_indices
                .get_unchecked(start..start + node.child_b as usize)
        }
    }

    /// Leaves of one tree.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree index, below `n_trees`
    ///
    /// ### Returns
    ///
    /// Iterator over the tree's leaves as id slices.
    pub(crate) fn tree_leaves(&self, tree: usize) -> impl Iterator<Item = &[u32]> {
        let lo = self.roots[tree] as usize;
        let hi = self
            .roots
            .get(tree + 1)
            .map_or(self.nodes.len(), |&r| r as usize);
        self.nodes[lo..hi]
            .iter()
            .filter(|node| node.n_descendants == 1)
            .map(|node| self.leaf_items(node))
    }

    /// Best-first walk over all trees, handing each reached leaf to `on_leaf`.
    ///
    /// Explores the unexplored branch with the smallest margin first. Stops
    /// once `limit` leaf items have been handed out, or once the margin
    /// lower bound of the next branch exceeds the bound `on_leaf` returned.
    ///
    /// ### Params
    ///
    /// * `split_query` - Query in the build space (normalised for Cosine)
    /// * `limit` - Budget of leaf items to visit
    /// * `pq` - Reusable backtracking frontier, left empty on return
    /// * `on_leaf` - Called once per reached leaf with its ids. Returns the
    ///   current kth-best distance, or infinity while no bound exists yet
    #[inline]
    pub(crate) fn visit(
        &self,
        split_query: &[T],
        limit: usize,
        pq: &mut BinaryHeap<BacktrackEntry>,
        mut on_leaf: impl FnMut(&[u32]) -> f64,
    ) {
        let mut visited_count = 0;
        let mut bound = f64::INFINITY;
        pq.clear();
        pq.reserve(self.roots.len() * 2);
        for &root in &self.roots {
            pq.push(BacktrackEntry {
                margin: f64::MAX,
                node_idx: root,
            });
        }

        while visited_count < limit {
            let Some(entry) = pq.pop() else { break };

            // priority queue pruning: margin-based lower bound vs kth distance.
            if entry.margin != f64::MAX {
                let abs_margin = -entry.margin;
                // Squared Euclidean: one axis of the difference, so >= margin^2.
                // Cosine: splits live in normalised space where
                // ||a - b||^2 = 2 * cos_dist, so cos_dist >= margin^2 / 2.
                let lower_bound = match self.metric {
                    Dist::SquaredEuclidean => abs_margin * abs_margin,
                    Dist::Cosine => abs_margin * abs_margin * 0.5,
                    Dist::Manhattan => abs_margin,
                };
                if lower_bound > bound {
                    break;
                }
            }

            let mut current_idx = entry.node_idx;

            loop {
                // SAFETY: Tree structure is static and verified at build
                let node = unsafe { self.nodes.get_unchecked(current_idx as usize) };

                if node.n_descendants == 1 {
                    visited_count += node.child_b as usize;
                    bound = on_leaf(self.leaf_items(node));
                    break;
                }

                let margin = (split_query[node.split_dim as usize] - node.split_val)
                    .to_f64()
                    .unwrap();

                // Left child (child_a) holds items below split_val,
                // right child (child_b) holds items at or above
                let (closer, farther) = if margin < 0.0 {
                    (node.child_a, node.child_b)
                } else {
                    (node.child_b, node.child_a)
                };

                pq.push(BacktrackEntry {
                    margin: -margin.abs(),
                    node_idx: farther,
                });

                current_idx = closer;
            }
        }
        pq.clear();
    }

    /// Rewrite every leaf id through `old_to_new`.
    ///
    /// ### Params
    ///
    /// * `old_to_new` - New id for each old id
    fn remap(&mut self, old_to_new: &[u32]) {
        self.leaf_indices
            .par_iter_mut()
            .for_each(|id| *id = old_to_new[*id as usize]);
    }

    /// Returns the size of the forest in bytes
    ///
    /// ### Returns
    ///
    /// Forest size in bytes
    pub(crate) fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.nodes.capacity() * std::mem::size_of::<KdFlatNode<T>>()
            + self.roots.capacity() * std::mem::size_of::<u32>()
            + self.leaf_indices.capacity() * std::mem::size_of::<u32>()
    }
}

/////////////////
// KdTreeIndex //
/////////////////

/// Kd-Tree forest index for approximate nearest neighbour search
///
/// Uses a forest of randomised Kd-trees to partition the space. Each tree
/// recursively splits along coordinate axes, selecting randomly from the top
/// highest-spread dimensions for diversity across trees.
///
/// For Cosine metric, trees are built on normalised vectors so that
/// axis-aligned splits respect angular geometry. The normalised copy is
/// discarded after construction; only the raw vectors are stored.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct KdTreeIndex<T> {
    /// Original vector data, flattened for cache locality
    pub vectors_flat: Vec<T>,
    /// Embedding dimensions
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Pre-computed norms for Cosine distance (empty for Euclidean)
    norms: Vec<T>,
    /// Distance metric (Euclidean or Cosine)
    metric: Dist,
    /// The forest, in the reordered id space of `vectors_flat`
    trees: KdTrees<T>,
    /// Number of trees in the forest
    pub n_trees: usize,
    /// Original indices for remapping after memory layout optimisation
    original_ids: Vec<usize>,
}

////////////////////
// VectorDistance //
////////////////////

/// VectorDistance implementation for SIMD speed
impl<T> VectorDistance<T> for KdTreeIndex<T>
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

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for KdTreeIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////////
// Index build //
/////////////////

/// Unit-normalised copy of the vectors, the space Kd trees split for Cosine.
///
/// ### Params
///
/// * `vectors_flat` - Flattened vector data
/// * `norms` - L2 norm per row
/// * `dim` - Dimensionality
///
/// ### Returns
///
/// The normalised copy; zero rows stay zero.
pub(crate) fn normalised_copy<T: AnnSearchFloat>(
    vectors_flat: &[T],
    norms: &[T],
    dim: usize,
) -> Vec<T> {
    let mut nv = vectors_flat.to_vec();
    for (i, &norm) in norms.iter().enumerate() {
        if norm > T::zero() {
            let start = i * dim;
            for d in 0..dim {
                nv[start + d] /= norm;
            }
        }
    }
    nv
}

impl<T> KdTreeIndex<T>
where
    T: AnnSearchFloat,
{
    /// Construct a new Kd-Tree forest index
    ///
    /// Builds a forest of randomised Kd-trees in parallel, one tree per
    /// thread.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (rows = samples, columns = dimensions)
    /// * `n_trees` - Number of trees to build (more trees = better recall,
    ///   slower build)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    pub fn new(data: impl AnnMatrix<T>, n_trees: usize, metric: Dist, seed: usize) -> Self {
        let (vectors_flat, n, dim) = data.into_row_major();

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

        // for cosine, build trees on normalised vectors so that axis-aligned
        // splits approximate angular partitions.
        let trees = if metric == Dist::Cosine {
            let normalised = normalised_copy(&vectors_flat, &norms, dim);
            KdTrees::build(&normalised, n, dim, n_trees, metric, seed)
        } else {
            KdTrees::build(&vectors_flat, n, dim, n_trees, metric, seed)
        };

        let mut idx = KdTreeIndex {
            trees,
            vectors_flat,
            dim,
            n_trees,
            n,
            norms,
            metric,
            original_ids: (0..n).collect(),
        };

        let new_to_old = idx.optimise_memory_layout();
        idx.original_ids = new_to_old;

        idx
    }

    /// Reorders vectors in memory to match the DFS layout of Tree 0
    ///
    /// Places vectors that co-occur in leaves adjacent in memory, for cache
    /// locality during leaf evaluation.
    ///
    /// ### Returns
    ///
    /// New-to-old index mapping
    fn optimise_memory_layout(&mut self) -> Vec<usize> {
        if self.n_trees == 0 || self.n == 0 {
            return Vec::new();
        }

        // Tree 0's permutation is its leaves in DFS order and covers every
        // item exactly once.
        let new_to_old: Vec<usize> = self.trees.leaf_indices[..self.n]
            .iter()
            .map(|&i| i as usize)
            .collect();
        let mut old_to_new = vec![0u32; self.n];
        for (new_id, &old_id) in new_to_old.iter().enumerate() {
            old_to_new[old_id] = new_id as u32;
        }

        let mut new_vectors_flat = Vec::with_capacity(self.vectors_flat.len());
        let mut new_norms = if self.norms.is_empty() {
            Vec::new()
        } else {
            Vec::with_capacity(self.n)
        };

        for &old_id in &new_to_old {
            let start = old_id * self.dim;
            let end = start + self.dim;
            new_vectors_flat.extend_from_slice(&self.vectors_flat[start..end]);

            if !self.norms.is_empty() {
                new_norms.push(self.norms[old_id]);
            }
        }

        self.trees.remap(&old_to_new);
        self.vectors_flat = new_vectors_flat;
        self.norms = new_norms;

        new_to_old
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
            + self.trees.memory_usage_bytes()
            + self.original_ids.capacity() * std::mem::size_of::<usize>()
    }
}

///////////
// Query //
///////////

impl<T> KdTreeIndex<T>
where
    T: AnnSearchFloat,
{
    /// Query the index for approximate nearest neighbours
    ///
    /// Traverses all trees using priority-queue-ordered backtracking. Explores
    /// nodes with smallest margin (distance to split boundary) first. Prunes
    /// subtrees whose margin-based lower bound exceeds the current kth-best
    /// distance.
    ///
    /// For Cosine metric, split decisions use normalised coordinates (matching
    /// the normalised build space) while leaf distances are computed using the
    /// standard cosine formula on raw vectors.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (raw, not normalised)
    /// * `k` - Number of neighbours to return
    /// * `search_k` - Budget of items to examine (higher = better recall,
    ///   slower). Defaults to `k * n_trees * 20` if None
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
    #[inline]
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        search_k: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        let limit = search_k.unwrap_or(k * self.n_trees * 20);

        let mut visited = FixedBitSet::with_capacity(self.n);

        // for cosine, normalise query for split decisions (splits were built in
        // normalised space)
        let query_norm = if self.metric == Dist::Cosine {
            T::calculate_l2_norm(query_vec)
        } else {
            T::one()
        };

        let normalised_query: Option<Vec<T>> =
            if self.metric == Dist::Cosine && query_norm > T::zero() {
                Some(query_vec.iter().map(|&v| v / query_norm).collect())
            } else {
                None
            };
        let split_query = normalised_query.as_deref().unwrap_or(query_vec);

        let mut top_k = SortedBuffer::with_capacity(k + 1);
        let mut kth_dist = T::infinity();
        let mut pq = BinaryHeap::with_capacity(self.n_trees * 2);

        self.trees.visit(split_query, limit, &mut pq, |leaf_items| {
            for &item in leaf_items {
                let item = item as usize;
                if visited.contains(item) {
                    continue;
                }
                visited.insert(item);

                let dist = match self.metric {
                    Dist::SquaredEuclidean => self.euclidean_distance_to_query(item, query_vec),
                    Dist::Cosine => self.cosine_distance_to_query(item, query_vec, query_norm),
                    Dist::Manhattan => self.manhattan_distance_to_query(item, query_vec),
                };

                if dist < kth_dist || top_k.len() < k {
                    top_k.insert((OrderedFloat(dist), item), k);
                    if top_k.len() == k {
                        kth_dist = top_k.top().unwrap().0 .0;
                    }
                }
            }
            if top_k.len() == k {
                kth_dist.to_f64().unwrap()
            } else {
                f64::INFINITY
            }
        });

        let results: Vec<(usize, T)> = top_k
            .data()
            .iter()
            .map(|(OrderedFloat(dist), idx)| {
                let original_idx = self.original_ids[*idx];
                (original_idx, *dist)
            })
            .collect();

        Ok(results.into_iter().unzip())
    }

    /// Query using a matrix row reference
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies to
    /// a temporary vector. Uses `self.query()` under the hood.
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference
    /// * `k` - Number of neighbours to search
    /// * `search_k` - Budget of items to examine (higher = better recall,
    ///   slower). Defaults to `k * n_trees * 20` if None
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        search_k: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, search_k);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, search_k)
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete kNN
    /// graph. Results are returned in the original index order (before memory
    /// layout optimisation).
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `search_k` - Search budget (defaults to k * n_trees * 20 if
    ///   None)
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity of the function
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)` where each row
    /// corresponds to a vector in the original index order
    pub fn generate_knn(
        &self,
        k: usize,
        search_k: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<T> {
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

                let (indices, dists) = self.query(vec, k, search_k)?;
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

//////////////////////
// Validation trait //
//////////////////////

impl<T> KnnValidation<T> for KdTreeIndex<T>
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

    fn n(&self) -> usize {
        self.n
    }

    fn dim(&self) -> usize {
        self.dim
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

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for KdTreeIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "kd_forest";
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::Mat;

    fn create_simple_matrix() -> Mat<f32> {
        // 5 points in 3D space
        let data = [
            1.0, 0.0, 0.0, // Point 0
            0.0, 1.0, 0.0, // Point 1
            0.0, 0.0, 1.0, // Point 2
            1.0, 1.0, 0.0, // Point 3
            1.0, 0.0, 1.0, // Point 4
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
    }

    #[test]
    fn test_kd_tree_index_creation() {
        let mat = create_simple_matrix();
        let _ = KdTreeIndex::new(mat.as_ref(), 4, Dist::SquaredEuclidean, 42);

        // just verify it doesn't panic
    }

    #[test]
    fn test_kd_trees_leaves_partition_every_tree() {
        let (n, dim) = (3000, 8);
        let mut rng = StdRng::seed_from_u64(3);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.random::<f32>() - 0.5).collect();
        for metric in [Dist::SquaredEuclidean, Dist::Manhattan] {
            let trees = KdTrees::<f32>::build(&data, n, dim, 4, metric, 7);
            assert_eq!(trees.n_trees(), 4);
            for tree in 0..4 {
                let mut seen = vec![false; n];
                let mut total = 0;
                for leaf in trees.tree_leaves(tree) {
                    for &id in leaf {
                        assert!(!seen[id as usize], "id {id} in two leaves of tree {tree}");
                        seen[id as usize] = true;
                    }
                    total += leaf.len();
                }
                assert_eq!(total, n);
            }
        }
    }

    #[test]
    fn test_kd_tree_query_finds_self() {
        let mat = create_simple_matrix();
        let index = KdTreeIndex::new(mat.as_ref(), 4, Dist::SquaredEuclidean, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1, None).unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kd_tree_query_euclidean() {
        let mat = create_simple_matrix();
        let index = KdTreeIndex::new(mat.as_ref(), 8, Dist::SquaredEuclidean, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None).unwrap();

        // Should find point 0 first (exact match)
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Results should be sorted by distance
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_kd_tree_query_cosine() {
        let mat = create_simple_matrix();
        let index = KdTreeIndex::new(mat.as_ref(), 8, Dist::Cosine, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None).unwrap();

        // Should find point 0 first (identical direction)
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kd_tree_query_k_larger_than_dataset() {
        let mat = create_simple_matrix();
        let index = KdTreeIndex::new(mat.as_ref(), 4, Dist::SquaredEuclidean, 42);

        let query = vec![1.0, 0.0, 0.0];
        // ask for 10 neighbours but only 5 points exist
        let (indices, _) = index.query(&query, 10, None).unwrap();

        // Should return at most 5 results
        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_kd_tree_query_search_k() {
        let mat = create_simple_matrix();
        let index = KdTreeIndex::new(mat.as_ref(), 4, Dist::SquaredEuclidean, 42);

        let query = vec![1.0, 0.0, 0.0];

        // With higher search_k, should get same or better results
        let (indices1, _) = index.query(&query, 3, Some(10)).unwrap();
        let (indices2, _) = index.query(&query, 3, Some(50)).unwrap();

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_kd_tree_multiple_trees() {
        let mat = create_simple_matrix();

        // More trees should give better recall
        let index_few = KdTreeIndex::new(mat.as_ref(), 2, Dist::SquaredEuclidean, 42);
        let index_many = KdTreeIndex::new(mat.as_ref(), 16, Dist::SquaredEuclidean, 42);

        let query = vec![0.9, 0.1, 0.0];
        let (indices1, _) = index_few.query(&query, 3, None).unwrap();
        let (indices2, _) = index_many.query(&query, 3, None).unwrap();

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_kd_tree_query_row() {
        let mat = create_simple_matrix();
        let index = KdTreeIndex::new(mat.as_ref(), 8, Dist::SquaredEuclidean, 42);

        let (indices, distances) = index.query_row(mat.row(0), 1, None).unwrap();

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kd_tree_reproducibility() {
        let mat = create_simple_matrix();

        // Same seed should give same results
        let index1 = KdTreeIndex::new(mat.as_ref(), 8, Dist::SquaredEuclidean, 42);
        let index2 = KdTreeIndex::new(mat.as_ref(), 8, Dist::SquaredEuclidean, 42);

        let query = vec![0.5, 0.5, 0.0];
        let (indices1, _) = index1.query(&query, 3, None).unwrap();
        let (indices2, _) = index2.query(&query, 3, None).unwrap();

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_kd_tree_different_seeds() {
        let mat = create_simple_matrix();

        let index1 = KdTreeIndex::new(mat.as_ref(), 8, Dist::SquaredEuclidean, 42);
        let index2 = KdTreeIndex::new(mat.as_ref(), 8, Dist::SquaredEuclidean, 123);

        let query = vec![0.5, 0.5, 0.0];

        let (indices1, _) = index1.query(&query, 3, None).unwrap();
        let (indices2, _) = index2.query(&query, 3, None).unwrap();

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_kd_tree_larger_dataset() {
        let n = 100;
        let dim = 10;
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 10.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = KdTreeIndex::new(mat.as_ref(), 16, Dist::SquaredEuclidean, 42);

        // Query for point 0
        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5, None).unwrap();

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0); // Should find exact match
    }

    #[test]
    fn test_kd_tree_orthogonal_vectors() {
        let data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);
        let index = KdTreeIndex::new(mat.as_ref(), 4, Dist::Cosine, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None).unwrap();

        // First result should be the parallel vector
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Other orthogonal vectors should have cosine distance = 1
        assert_relative_eq!(distances[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(distances[2], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kd_tree_parallel_build() {
        let n = 50;
        let dim = 5;
        let data: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        let index = KdTreeIndex::new(mat.as_ref(), 32, Dist::SquaredEuclidean, 42);

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 3, None).unwrap();

        assert_eq!(indices.len(), 3);
    }
}
