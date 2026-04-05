//! Kd-Tree forest implementation in ann-search-rs.
//!
//! Uses a forest of randomised Kd-trees for approximate nearest neighbour
//! search. Each tree splits along coordinate axes selected randomly from
//! the top highest-spread dimensions, providing diversity across trees.
//! Supports optional spill-tree overlap at split boundaries for improved
//! recall: items within a fraction of the spread on each side of the
//! boundary are placed into both children, allowing the query to find
//! neighbours that a hard partition would miss.

use faer::{MatRef, RowRef};
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
struct KdFlatNode<T> {
    /// `1` for leaf, `2` for split node
    n_descendants: u32,
    /// For split: left child index; For leaf: start index in leaf_indices
    child_a: u32,
    /// For split: right child index; For leaf: count of items
    child_b: u32,
    /// Dimension along which the split occurs (split only)
    split_dim: u16,
    /// Explicit padding... Compiler likely does this anyways
    _pad: u16,
    /// Threshold value for the split (split only)
    split_val: T,
}

/// Build-time node representation
///
/// Temporary structure used during tree construction, later flattened
/// into KdFlatNode format for better cache performance during queries.
#[derive(Clone)]
enum BuildNode<T> {
    Split {
        /// Dimension index for the split
        split_dim: usize,
        /// Threshold value along that dimension
        split_val: T,
        /// Index of left child in build tree
        left: usize,
        /// Index of right child in build tree
        right: usize,
    },
    Leaf {
        /// Original data indices in this leaf
        items: Vec<usize>,
    },
}

/// Number of top-spread dimensions to randomly select from. Provides diversity
/// across trees in the forest without picking completely arbitrary axes.
const TOP_K_DIMS: usize = 3;

/// Maximum sample size for spread estimation and approximate median
/// computation. Keeps build cost per node O(1) rather than O(n).
const SUBSAMPLE_SIZE: usize = 128;

////////////////
// Main index //
////////////////

/// Kd-Tree forest index for approximate nearest neighbour search
///
/// Uses a forest of randomised Kd-trees to partition the space. Each tree
/// recursively splits along coordinate axes, selecting randomly from the top
/// highest-spread dimensions for diversity across trees.
///
/// For Cosine metric, trees are built on normalised vectors so that
/// axis-aligned splits respect angular geometry. The normalised copy is
/// discarded after construction; only the raw vectors are stored.
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
    /// Flattened tree structure containing all split and leaf nodes
    nodes: Vec<KdFlatNode<T>>,
    /// Starting indices for each tree in the forest
    roots: Vec<u32>,
    /// Actual data indices stored in leaf nodes
    leaf_indices: Vec<usize>,
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

impl<T> KdTreeIndex<T>
where
    T: AnnSearchFloat,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Construct a new Kd-Tree forest index with default overlap
    ///
    /// Builds a forest of randomised Kd-trees in parallel. Uses the
    /// default spill-tree overlap fraction of 5%.
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
    pub fn new(data: MatRef<T>, n_trees: usize, metric: Dist, seed: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let (vectors_flat, n, dim) = matrix_to_flat(data);

        // compute norms for Cosine distance
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
        let normalised = if metric == Dist::Cosine {
            let mut nv = vectors_flat.clone();
            for i in 0..n {
                let norm = norms[i];
                if norm > T::zero() {
                    let start = i * dim;
                    for d in 0..dim {
                        nv[start + d] = nv[start + d] / norm;
                    }
                }
            }
            Some(nv)
        } else {
            None
        };
        let build_vectors = normalised.as_deref().unwrap_or(&vectors_flat);

        let seeds: Vec<u64> = (0..n_trees).map(|_| rng.random()).collect();

        let forest: Vec<Vec<BuildNode<T>>> = seeds
            .into_par_iter()
            .map(|tree_seed| {
                let mut tree_rng = StdRng::seed_from_u64(tree_seed);
                Self::build_tree_recursive(build_vectors, dim, (0..n).collect(), &mut tree_rng)
            })
            .collect();

        // flatten all trees into contiguous storage
        let total_nodes: usize = forest.iter().map(|t| t.len()).sum();
        let mut nodes = Vec::with_capacity(total_nodes);
        let mut roots = Vec::with_capacity(n_trees);
        let mut leaf_indices = Vec::new();

        for tree in forest {
            let root_offset = nodes.len() as u32;
            roots.push(root_offset);

            for node in tree {
                match node {
                    BuildNode::Split {
                        split_dim,
                        split_val,
                        left,
                        right,
                    } => {
                        nodes.push(KdFlatNode {
                            n_descendants: 2,
                            child_a: root_offset + left as u32,
                            child_b: root_offset + right as u32,
                            split_dim: split_dim as u16,
                            _pad: 0,
                            split_val,
                        });
                    }
                    BuildNode::Leaf { items } => {
                        let start = leaf_indices.len() as u32;
                        let len = items.len() as u32;
                        leaf_indices.extend(items);

                        nodes.push(KdFlatNode {
                            n_descendants: 1,
                            child_a: start,
                            child_b: len,
                            split_dim: 0,
                            _pad: 0,
                            split_val: T::zero(),
                        });
                    }
                }
            }
        }

        let mut idx = KdTreeIndex {
            nodes,
            roots,
            leaf_indices,
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

    ////////////////
    // Tree build //
    ////////////////

    /// Build a single tree recursively
    ///
    /// Wrapper that initialises the node vector and starts the recursive
    /// splitting process.
    ///
    /// ### Params
    ///
    /// * `build_vectors` - Flattened vector data (normalised for Cosine)
    /// * `dim` - Dimensionality of vectors
    /// * `items` - Indices of items to partition
    /// * `rng` - Random number generator for this tree
    /// * `overlap` - Spill-tree overlap fraction
    ///
    /// ### Returns
    ///
    /// Vector of BuildNodes representing the complete tree
    fn build_tree_recursive(
        build_vectors: &[T],
        dim: usize,
        items: Vec<usize>,
        rng: &mut StdRng,
    ) -> Vec<BuildNode<T>> {
        let mut nodes = Vec::with_capacity(items.len());
        Self::build_node(build_vectors, dim, items, &mut nodes, rng);
        nodes
    }

    /// Recursively build a node (split or leaf)
    ///
    /// Selects a split dimension randomly from the top highest-spread
    /// dimensions (computed on a subsample for efficiency), then splits
    /// at the approximate median. Items within the spill-tree overlap
    /// zone are placed into both children.
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
    /// * `items` - Indices to partition in this node
    /// * `nodes` - Accumulator for built nodes
    /// * `rng` - Random number generator
    /// * `overlap` - Spill-tree overlap fraction
    ///
    /// ### Returns
    ///
    /// Index of the created node in `nodes`
    fn build_node(
        build_vectors: &[T],
        dim: usize,
        items: Vec<usize>,
        nodes: &mut Vec<BuildNode<T>>,
        rng: &mut StdRng,
    ) -> usize {
        if items.len() <= LEAF_MIN_MEMBERS {
            let node_idx = nodes.len();
            nodes.push(BuildNode::Leaf { items });
            return node_idx;
        }

        // Subsample for spread estimation and median computation
        let owned_sample: Vec<usize>;
        let sample: &[usize] = if items.len() > SUBSAMPLE_SIZE {
            owned_sample = (0..SUBSAMPLE_SIZE)
                .map(|_| items[rng.random_range(0..items.len())])
                .collect();
            &owned_sample
        } else {
            &items
        };

        // Find top-k dimensions by spread
        let top_dims = Self::top_spread_dims(build_vectors, dim, sample);

        if top_dims.is_empty() {
            let node_idx = nodes.len();
            nodes.push(BuildNode::Leaf { items });
            return node_idx;
        }

        // Shuffle top dims so we try them in random order
        let mut candidate_dims = top_dims.clone();
        for i in (1..candidate_dims.len()).rev() {
            let j = rng.random_range(0..=i);
            candidate_dims.swap(i, j);
        }

        for &(split_dim, _) in &candidate_dims {
            // compute approximate median from the sample
            let mut vals: Vec<T> = sample
                .iter()
                .map(|&idx| build_vectors[idx * dim + split_dim])
                .collect();
            vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let split_val = vals[vals.len() / 2];
            let mut left_items = Vec::new();
            let mut right_items = Vec::new();

            for &item in &items {
                let val = build_vectors[item * dim + split_dim];
                if val < split_val {
                    left_items.push(item);
                } else {
                    right_items.push(item);
                }
            }

            // both children must be non-empty
            if left_items.is_empty() || right_items.is_empty() {
                continue;
            }

            let node_idx = nodes.len();
            nodes.push(BuildNode::Split {
                split_dim,
                split_val,
                left: 0,
                right: 0,
            });

            let left_idx = Self::build_node(build_vectors, dim, left_items, nodes, rng);
            let right_idx = Self::build_node(build_vectors, dim, right_items, nodes, rng);

            if let BuildNode::Split {
                ref mut left,
                ref mut right,
                ..
            } = nodes[node_idx]
            {
                *left = left_idx;
                *right = right_idx;
            }

            return node_idx;
        }

        // No dimension produced a valid split -- fall back to leaf
        let node_idx = nodes.len();
        nodes.push(BuildNode::Leaf { items });
        node_idx
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
    fn top_spread_dims(build_vectors: &[T], dim: usize, sample: &[usize]) -> Vec<(usize, T)> {
        let mut spreads: Vec<(usize, T)> = (0..dim)
            .map(|d| {
                let mut min_val = T::infinity();
                let mut max_val = T::neg_infinity();
                for &idx in sample {
                    let val = build_vectors[idx * dim + d];
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

    /////////////////////////
    // Memory optimisation //
    /////////////////////////

    /// Reorders vectors in memory to match the DFS layout of Tree 0
    ///
    /// Places vectors that co-occur in leaves physically adjacent in memory.
    /// This drastically reduces L2/L3 cache misses during leaf evaluation and
    /// reduces memory bandwidth problems.
    ///
    /// ### Returns
    ///
    /// New-to-old index mapping
    fn optimise_memory_layout(&mut self) -> Vec<usize> {
        if self.roots.is_empty() || self.n == 0 {
            return Vec::new();
        }

        let mut new_to_old = Vec::with_capacity(self.n);
        let mut old_to_new = vec![usize::MAX; self.n];
        let mut visited = vec![false; self.n];

        // DFS traversal of Tree 0
        let mut stack = vec![self.roots[0]];
        while let Some(node_idx) = stack.pop() {
            let node = unsafe { self.nodes.get_unchecked(node_idx as usize) };

            if node.n_descendants == 1 {
                let start = node.child_a as usize;
                let len = node.child_b as usize;
                let leaf_items = unsafe { self.leaf_indices.get_unchecked(start..start + len) };

                for &old_id in leaf_items {
                    if !visited[old_id] {
                        visited[old_id] = true;
                        old_to_new[old_id] = new_to_old.len();
                        new_to_old.push(old_id);
                    }
                }
            } else {
                stack.push(node.child_b);
                stack.push(node.child_a);
            }
        }

        // catch any items not in Tree 0 (safety guard for items that
        // ended up only in other trees due to spill overlap patterns)
        for old_id in 0..self.n {
            if !visited[old_id] {
                old_to_new[old_id] = new_to_old.len();
                new_to_old.push(old_id);
            }
        }

        // shuffle vector data into new contiguous layout
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

        // rewrite all leaves in all trees to use new IDs
        for id_ref in self.leaf_indices.iter_mut() {
            *id_ref = old_to_new[*id_ref];
        }

        self.vectors_flat = new_vectors_flat;
        self.norms = new_norms;

        new_to_old
    }

    ///////////
    // Query //
    ///////////

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
    ) -> (Vec<usize>, Vec<T>) {
        let limit = search_k.unwrap_or(k * self.n_trees * 20);
        let mut visited_count = 0;

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

        for &root in &self.roots {
            pq.push(BacktrackEntry {
                margin: f64::MAX,
                node_idx: root,
            });
        }

        while visited_count < limit {
            let Some(entry) = pq.pop() else { break };

            // priority queue pruning: margin-based lower bound vs kth distance.
            if top_k.len() == k && entry.margin != f64::MAX {
                let abs_margin = -entry.margin;
                // for Euclidean (squared): lower bound = margin^2
                // for Cosine: unit vectors have ||a-b||^2 = 2(1-cos),
                // margin^2 is one component, so cos_dist >= margin^2/2
                // Claude says this is the right way to do this...
                let lower_bound = match self.metric {
                    Dist::Euclidean => abs_margin * abs_margin,
                    Dist::Cosine => abs_margin * abs_margin * 0.5,
                };
                if lower_bound > kth_dist.to_f64().unwrap() {
                    break;
                }
            }

            let mut current_idx = entry.node_idx;

            loop {
                // SAFETY: Tree structure is static and verified at build
                let node = unsafe { self.nodes.get_unchecked(current_idx as usize) };

                if node.n_descendants == 1 {
                    // leaf -> evaluate all items
                    let start = node.child_a as usize;
                    let len = node.child_b as usize;
                    visited_count += len;

                    let leaf_items = unsafe { self.leaf_indices.get_unchecked(start..start + len) };

                    for &item in leaf_items {
                        if visited.contains(item) {
                            continue;
                        }
                        visited.insert(item);

                        let dist = match self.metric {
                            Dist::Euclidean => self.euclidean_distance_to_query(item, query_vec),
                            Dist::Cosine => {
                                self.cosine_distance_to_query(item, query_vec, query_norm)
                            }
                        };

                        if dist < kth_dist || top_k.len() < k {
                            top_k.insert((OrderedFloat(dist), item), k);
                            if top_k.len() == k {
                                kth_dist = top_k.top().unwrap().0 .0;
                            }
                        }
                    }
                    break;
                } else {
                    // split node -> single coordinate comparison
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
        }

        // extract results and remap to original indices
        let results: Vec<(usize, T)> = top_k
            .data()
            .iter()
            .map(|(OrderedFloat(dist), idx)| {
                let original_idx = self.original_ids[*idx];
                (original_idx, *dist)
            })
            .collect();

        results.into_iter().unzip()
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
    ) -> (Vec<usize>, Vec<T>) {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, search_k);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, search_k)
    }

    ///////////////
    // kNN graph //
    ///////////////

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

                let (indices, dists) = self.query(vec, k, search_k);
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

    ////////////
    // Memory //
    ////////////

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
            + self.nodes.capacity() * std::mem::size_of::<KdFlatNode<T>>()
            + self.roots.capacity() * std::mem::size_of::<u32>()
            + self.leaf_indices.capacity() * std::mem::size_of::<usize>()
            + self.original_ids.capacity() * std::mem::size_of::<usize>()
    }
}

//////////////////////
// Validation trait //
//////////////////////

impl<T> KnnValidation<T> for KdTreeIndex<T>
where
    T: AnnSearchFloat,
{
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        self.query(query_vec, k, None)
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
        let _ = KdTreeIndex::new(mat.as_ref(), 4, Dist::Euclidean, 42);

        // just verify it doesn't panic
    }

    #[test]
    fn test_kd_tree_query_finds_self() {
        let mat = create_simple_matrix();
        let index = KdTreeIndex::new(mat.as_ref(), 4, Dist::Euclidean, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1, None);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kd_tree_query_euclidean() {
        let mat = create_simple_matrix();
        let index = KdTreeIndex::new(mat.as_ref(), 8, Dist::Euclidean, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None);

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
        let (indices, distances) = index.query(&query, 3, None);

        // Should find point 0 first (identical direction)
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kd_tree_query_k_larger_than_dataset() {
        let mat = create_simple_matrix();
        let index = KdTreeIndex::new(mat.as_ref(), 4, Dist::Euclidean, 42);

        let query = vec![1.0, 0.0, 0.0];
        // ask for 10 neighbours but only 5 points exist
        let (indices, _) = index.query(&query, 10, None);

        // Should return at most 5 results
        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_kd_tree_query_search_k() {
        let mat = create_simple_matrix();
        let index = KdTreeIndex::new(mat.as_ref(), 4, Dist::Euclidean, 42);

        let query = vec![1.0, 0.0, 0.0];

        // With higher search_k, should get same or better results
        let (indices1, _) = index.query(&query, 3, Some(10));
        let (indices2, _) = index.query(&query, 3, Some(50));

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_kd_tree_multiple_trees() {
        let mat = create_simple_matrix();

        // More trees should give better recall
        let index_few = KdTreeIndex::new(mat.as_ref(), 2, Dist::Euclidean, 42);
        let index_many = KdTreeIndex::new(mat.as_ref(), 16, Dist::Euclidean, 42);

        let query = vec![0.9, 0.1, 0.0];
        let (indices1, _) = index_few.query(&query, 3, None);
        let (indices2, _) = index_many.query(&query, 3, None);

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_kd_tree_query_row() {
        let mat = create_simple_matrix();
        let index = KdTreeIndex::new(mat.as_ref(), 8, Dist::Euclidean, 42);

        let (indices, distances) = index.query_row(mat.row(0), 1, None);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kd_tree_reproducibility() {
        let mat = create_simple_matrix();

        // Same seed should give same results
        let index1 = KdTreeIndex::new(mat.as_ref(), 8, Dist::Euclidean, 42);
        let index2 = KdTreeIndex::new(mat.as_ref(), 8, Dist::Euclidean, 42);

        let query = vec![0.5, 0.5, 0.0];
        let (indices1, _) = index1.query(&query, 3, None);
        let (indices2, _) = index2.query(&query, 3, None);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_kd_tree_different_seeds() {
        let mat = create_simple_matrix();

        let index1 = KdTreeIndex::new(mat.as_ref(), 8, Dist::Euclidean, 42);
        let index2 = KdTreeIndex::new(mat.as_ref(), 8, Dist::Euclidean, 123);

        let query = vec![0.5, 0.5, 0.0];

        let (indices1, _) = index1.query(&query, 3, None);
        let (indices2, _) = index2.query(&query, 3, None);

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
        let index = KdTreeIndex::new(mat.as_ref(), 16, Dist::Euclidean, 42);

        // Query for point 0
        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5, None);

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0); // Should find exact match
    }

    #[test]
    fn test_kd_tree_orthogonal_vectors() {
        let data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);
        let index = KdTreeIndex::new(mat.as_ref(), 4, Dist::Cosine, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None);

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

        let index = KdTreeIndex::new(mat.as_ref(), 32, Dist::Euclidean, 42);

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 3, None);

        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_kd_tree_no_overlap() {
        let n = 100;
        let dim = 10;
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 10.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        // Build with zero overlap (standard Kd-tree)
        let index = KdTreeIndex::new(mat.as_ref(), 8, Dist::Euclidean, 42);

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5, None);

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
    }
}
