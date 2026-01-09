use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::{cmp::Ordering, collections::BinaryHeap, iter::Sum};
use thousands::*;

use crate::utils::dist::*;
use crate::utils::heap_structs::*;
use crate::utils::*;

/////////////
// Helpers //
/////////////

/// Node representation in the flattened tree structure
///
/// Uses tagged union pattern: `n_descendants == 1` indicates leaf node,
/// otherwise it's a split node.
///
/// ### Fields
///
/// * `n_descendants` - 1 for leaf, 2 for split node
/// * `child_a` - For split: left child index; For leaf: start index in
///   leaf_indices
/// * `child_b` - For split: right child index; For leaf: count of items
/// * `split_idx` - Index into split_data (only used for split nodes)
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct FlatNode {
    n_descendants: u32,
    child_a: u32,
    child_b: u32,
    split_idx: u32,
}

/// Build-time node representation
///
/// Temporary structure used during tree construction, later flattened into
/// FlatNode format for better cache performance during queries.
#[derive(Clone)]
enum BuildNode<T> {
    Split {
        /// Hyperplane normal vector
        hyperplane: Vec<T>,
        /// Dot product threshold for split decision
        offset: T,
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

/// Priority queue entry for backtracking during search
///
/// Stores nodes to revisit, ordered by distance to query (further = lower
/// priority)
///
/// ### Fields
///
/// * `margin` - Negative absolute margin to hyperplane (for max-heap ->
///   min-distance)
/// * `node_idx` - Index of node to explore
struct BacktrackEntry {
    margin: f64,
    node_idx: u32,
}

impl PartialEq for BacktrackEntry {
    fn eq(&self, other: &Self) -> bool {
        self.margin == other.margin
    }
}

impl Eq for BacktrackEntry {}

impl PartialOrd for BacktrackEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BacktrackEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.margin
            .partial_cmp(&other.margin)
            .unwrap_or(Ordering::Equal)
    }
}

/// Bitset for tracking visited items during search
///
/// Uses one bit per item, packed into u64 chunks for cache efficiency
struct VisitedSet {
    data: Vec<u64>,
}

impl VisitedSet {
    /// Create a new bitset with capacity for n items
    ///
    /// Allocates enough u64 chunks to store one bit per item, rounding up
    /// to the nearest 64-item boundary.
    ///
    /// ### Params
    ///
    /// * `capacity` - Number of items to track
    ///
    /// ### Returns
    ///
    /// Initialised bitset with all bits set to 0 (unvisited)
    fn new(capacity: usize) -> Self {
        // Round up to nearest u64
        let size = capacity.div_ceil(64);
        Self {
            data: vec![0; size],
        }
    }

    /// Mark an item as visited
    ///
    /// Sets the bit corresponding to `index` to 1. Uses unsafe unchecked
    /// access for performance - caller must ensure index < capacity.
    ///
    /// ### Params
    ///
    /// * `index` - Item index to mark
    ///
    /// ### Returns
    ///
    /// `true` if item was already visited, `false` if newly marked
    #[inline]
    fn mark(&mut self, index: usize) -> bool {
        let chunk = index / 64;
        let bit = 1 << (index % 64);
        // Use get_unchecked for speed, safe if capacity is correct
        unsafe {
            let slot = self.data.get_unchecked_mut(chunk);
            if (*slot & bit) != 0 {
                return true;
            }
            *slot |= bit;
            false
        }
    }
}

////////////////
// Main index //
////////////////

const MIN_MEMBERS: usize = 64;

/// Annoy (Approximate Nearest Neighbours Oh Yeah) index for similarity search
///
/// Uses a forest of random projection trees to partition the space. Each tree
/// recursively splits the data using random hyperplanes until reaching leaves
/// of size ≤ 64 items.
///
/// ### Fields
///
/// * `vectors_flat` - Original vector data, flattened for cache locality
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `norms` - Pre-computed norms for Cosine distance (empty for Euclidean)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `nodes` - Flattened tree structure containing all split and leaf nodes
/// * `roots` - Starting indices for each tree in the forest
/// * `split_data` - Hyperplane coefficients and offsets for split nodes
/// * `leaf_indices` - Actual data indices stored in leaf nodes
/// * `n_trees` - Number of trees in the forest
pub struct AnnoyIndex<T> {
    // shared ones
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    norms: Vec<T>,
    metric: Dist,
    // index specific
    nodes: Vec<FlatNode>,
    roots: Vec<u32>,
    split_data: Vec<T>,
    leaf_indices: Vec<usize>,
    pub n_trees: usize,
}

impl<T> VectorDistance<T> for AnnoyIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
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

impl<T> AnnoyIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Construct a new Annoy index
    ///
    /// Builds a forest of random projection trees in parallel. Each tree
    /// recursively partitions the space using random hyperplanes chosen
    /// between random pairs of points.
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

        // Compute norms for Cosine distance
        let norms = if metric == Dist::Cosine {
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
            Vec::new()
        };

        let seeds: Vec<u64> = (0..n_trees).map(|_| rng.random()).collect();

        let forest: Vec<Vec<BuildNode<T>>> = seeds
            .into_par_iter()
            .map(|tree_seed| {
                let mut tree_rng = StdRng::seed_from_u64(tree_seed);
                Self::build_tree_recursive(
                    &vectors_flat,
                    dim,
                    (0..n).collect(),
                    &mut tree_rng,
                    metric,
                )
            })
            .collect();

        let total_nodes: usize = forest.iter().map(|t| t.len()).sum();
        let mut nodes = Vec::with_capacity(total_nodes);
        let mut roots = Vec::with_capacity(n_trees);
        let mut split_data = Vec::new();
        let mut leaf_indices = Vec::new();

        for tree in forest {
            let root_offset = nodes.len() as u32;
            roots.push(root_offset);

            for node in tree {
                match node {
                    BuildNode::Split {
                        hyperplane,
                        offset,
                        left,
                        right,
                    } => {
                        let split_idx = (split_data.len() / (dim + 1)) as u32;
                        split_data.extend(hyperplane);
                        split_data.push(offset);

                        nodes.push(FlatNode {
                            n_descendants: 2,
                            child_a: root_offset + left as u32,
                            child_b: root_offset + right as u32,
                            split_idx,
                        });
                    }
                    BuildNode::Leaf { items } => {
                        let start = leaf_indices.len() as u32;
                        let len = items.len() as u32;
                        leaf_indices.extend(items);

                        nodes.push(FlatNode {
                            n_descendants: 1,
                            child_a: start,
                            child_b: len,
                            split_idx: 0,
                        });
                    }
                }
            }
        }

        AnnoyIndex {
            nodes,
            roots,
            split_data,
            leaf_indices,
            vectors_flat,
            dim,
            n_trees,
            n,
            norms,
            metric,
        }
    }

    /// Build a single tree recursively
    ///
    /// Wrapper that initialises the node vector and starts the recursive
    /// splitting process.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Flattened vector data
    /// * `dim` - Dimensionality of vectors
    /// * `items` - Indices of items to partition
    /// * `rng` - Random number generator for this tree
    ///
    /// ### Returns
    ///
    /// Vector of BuildNodes representing the complete tree
    fn build_tree_recursive(
        vectors_flat: &[T],
        dim: usize,
        items: Vec<usize>,
        rng: &mut StdRng,
        metric: Dist,
    ) -> Vec<BuildNode<T>> {
        let mut nodes = Vec::with_capacity(items.len());
        Self::build_node(vectors_flat, dim, items, &mut nodes, rng, metric);
        nodes
    }

    /// Recursively build a node (split or leaf)
    ///
    /// Attempts up to 5 random hyperplane splits. For each split, picks two
    /// random points and uses their difference as the hyperplane normal.
    /// Accepts splits where both sides contain 5-95% of items. Falls back
    /// to a leaf if no good split is found or if items ≤ 64.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Flattened vector data
    /// * `dim` - Dimensionality of vectors
    /// * `items` - Indices to partition in this node
    /// * `nodes` - Accumulator for built nodes
    /// * `rng` - Random number generator
    ///
    /// ### Returns
    ///
    /// Index of the created node in `nodes`
    fn build_node(
        vectors_flat: &[T],
        dim: usize,
        items: Vec<usize>,
        nodes: &mut Vec<BuildNode<T>>,
        rng: &mut StdRng,
        metric: Dist,
    ) -> usize {
        if items.len() <= MIN_MEMBERS {
            let node_idx = nodes.len();
            nodes.push(BuildNode::Leaf { items });
            return node_idx;
        }

        for _ in 0..10 {
            let idx1 = items[rng.random_range(0..items.len())];
            let idx2 = items[rng.random_range(0..items.len())];
            if idx1 == idx2 {
                continue;
            }

            let v1_start = idx1 * dim;
            let v2_start = idx2 * dim;
            let v1 = &vectors_flat[v1_start..v1_start + dim];
            let v2 = &vectors_flat[v2_start..v2_start + dim];

            let (hyperplane, threshold) = match metric {
                Dist::Cosine => {
                    let norm1 = v1
                        .iter()
                        .map(|&x| x * x)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt();
                    let norm2 = v2
                        .iter()
                        .map(|&x| x * x)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt();

                    if norm1 == T::zero() || norm2 == T::zero() {
                        continue;
                    }

                    let hp: Vec<T> = (0..dim).map(|k| v1[k] / norm1 - v2[k] / norm2).collect();

                    (hp, T::zero())
                }
                Dist::Euclidean => {
                    let mut hp = Vec::with_capacity(dim);
                    let mut dot_v1 = T::zero();
                    let mut dot_v2 = T::zero();

                    for k in 0..dim {
                        let val1 = v1[k];
                        let val2 = v2[k];
                        hp.push(val1 - val2);
                        dot_v1 = dot_v1 + val1 * val1;
                        dot_v2 = dot_v2 + val2 * val2;
                    }

                    let thresh = (dot_v1 - dot_v2) / T::from_f64(2.0).unwrap();
                    (hp, thresh)
                }
            };

            let mut left_items = Vec::new();
            let mut right_items = Vec::new();

            for &item in &items {
                let vec_start = item * dim;
                let vec = &vectors_flat[vec_start..vec_start + dim];
                let mut dot = T::zero();
                for k in 0..dim {
                    dot = dot + vec[k] * hyperplane[k];
                }

                if dot > threshold {
                    left_items.push(item);
                } else {
                    right_items.push(item);
                }
            }

            if left_items.is_empty() || right_items.is_empty() {
                continue;
            }

            let ratio = left_items.len() as f64 / items.len() as f64;
            if (0.05..=0.95).contains(&ratio) {
                let node_idx = nodes.len();
                nodes.push(BuildNode::Split {
                    hyperplane,
                    offset: threshold,
                    left: 0,
                    right: 0,
                });

                let left_idx = Self::build_node(vectors_flat, dim, left_items, nodes, rng, metric);
                let right_idx =
                    Self::build_node(vectors_flat, dim, right_items, nodes, rng, metric);

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
        }

        let node_idx = nodes.len();
        nodes.push(BuildNode::Leaf { items });
        node_idx
    }

    /// Calculate signed distance (margin) from query to hyperplane
    ///
    /// Computes dot(query, normal) - offset. Positive means query is on
    /// the "left" side of the split, negative means "right".
    ///
    /// ### Params
    ///
    /// * `v1` - Query vector
    /// * `v2` - Hyperplane data (first dim elements = normal, last = offset)
    /// * `dim` - Dimensionality
    ///
    /// ### Returns
    ///
    /// Signed margin (distance to hyperplane along normal direction)
    #[inline(always)]
    fn get_margin(v1: &[T], v2: &[T], dim: usize) -> T {
        v1.iter()
            .zip(v2.iter())
            .map(|(&a, &b)| a * b)
            .fold(T::zero(), |acc, x| acc + x)
            - v2[dim]
    }

    ///////////
    // Query //
    ///////////

    /// Query the index for approximate nearest neighbours
    ///
    /// Performs best-first search across all trees, greedily descending to
    /// leaves and backtracking to promising unexplored branches. Stops when
    /// the search budget is exhausted.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `search_k` - Budget of items to examine (higher = better recall,
    ///   slower)
    ///   Defaults to `k * n_trees` if None
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

        let n_vectors = self.vectors_flat.len() / self.dim;
        let mut visited = VisitedSet::new(n_vectors);

        let query_norm = if self.metric == Dist::Cosine {
            query_vec
                .iter()
                .map(|&x| x * x)
                .fold(T::zero(), |acc, x| acc + x)
                .sqrt()
        } else {
            T::one()
        };

        let mut top_k: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);
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

            if top_k.len() == k && -entry.margin > kth_dist.to_f64().unwrap() {
                break;
            }

            let mut current_idx = entry.node_idx;

            loop {
                let node = unsafe { self.nodes.get_unchecked(current_idx as usize) };

                if node.n_descendants == 1 {
                    let start = node.child_a as usize;
                    let len = node.child_b as usize;
                    visited_count += len;

                    let leaf_items = unsafe { self.leaf_indices.get_unchecked(start..start + len) };

                    for &item in leaf_items {
                        if visited.mark(item) {
                            continue;
                        }

                        let vec_start = item * self.dim;
                        let vec = unsafe {
                            self.vectors_flat
                                .get_unchecked(vec_start..vec_start + self.dim)
                        };

                        let dist = match self.metric {
                            Dist::Euclidean => {
                                let mut dist_sq = T::zero();
                                for i in 0..self.dim {
                                    let diff = unsafe {
                                        *query_vec.get_unchecked(i) - *vec.get_unchecked(i)
                                    };
                                    dist_sq = dist_sq + diff * diff;
                                }
                                dist_sq.sqrt()
                            }
                            Dist::Cosine => {
                                let mut dot = T::zero();
                                for i in 0..self.dim {
                                    dot = dot
                                        + unsafe {
                                            *query_vec.get_unchecked(i) * *vec.get_unchecked(i)
                                        };
                                }
                                let norm = unsafe { *self.norms.get_unchecked(item) };
                                T::one() - dot / (query_norm * norm)
                            }
                        };

                        if dist < kth_dist || top_k.len() < k {
                            top_k.push((OrderedFloat(dist), item));
                            if top_k.len() > k {
                                top_k.pop();
                            }
                            if top_k.len() == k {
                                kth_dist = top_k.peek().unwrap().0 .0;
                            }
                        }
                    }
                    break;
                } else {
                    let split_offset = node.split_idx as usize * (self.dim + 1);
                    let plane = unsafe {
                        self.split_data
                            .get_unchecked(split_offset..split_offset + self.dim + 1)
                    };

                    let margin = Self::get_margin(query_vec, plane, self.dim)
                        .to_f64()
                        .unwrap();

                    let (closer, farther) = if margin > 0.0 {
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

        let mut results: Vec<(usize, T)> = top_k
            .into_iter()
            .map(|(OrderedFloat(dist), idx)| (idx, dist))
            .collect();

        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.into_iter().unzip()
    }

    /// Query using a matrix row reference
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies
    /// to a temporary vector. Uses `self.query()` under the hood.
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference
    /// * `k` - Number of neighbours to search
    /// * `search_k` - Budget of items to examine (higher = better recall,
    ///   slower) Defaults to `k * n_trees` if None
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

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete
    /// kNN graph.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `search_k` - Search budget (defaults to k * n_trees * 20 if None)
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity of the function
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)` where each row corresponds
    /// to a vector in the index
    pub fn generate_knn(
        &self,
        k: usize,
        search_k: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dim;
                let end = start + self.dim;
                let vec = &self.vectors_flat[start..end];

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

                self.query(vec, k, search_k)
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
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
            + self.nodes.capacity() * std::mem::size_of::<FlatNode>()
            + self.roots.capacity() * std::mem::size_of::<u32>()
            + self.split_data.capacity() * std::mem::size_of::<T>()
            + self.leaf_indices.capacity() * std::mem::size_of::<usize>()
    }
}

//////////////////////
// Validation trait //
//////////////////////

impl<T> KnnValidation<T> for AnnoyIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
{
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        // Use the default here
        self.query(query_vec, k, None)
    }

    fn n(&self) -> usize {
        self.n
    }

    fn metric(&self) -> Dist {
        self.metric
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
    fn test_annoy_index_creation() {
        let mat = create_simple_matrix();
        let _ = AnnoyIndex::new(mat.as_ref(), 4, Dist::Euclidean, 42);

        // just verify it doesn't panic
    }

    #[test]
    fn test_annoy_query_finds_self() {
        let mat = create_simple_matrix();
        let index = AnnoyIndex::new(mat.as_ref(), 4, Dist::Euclidean, 42);

        // Query with point 0, should find itself first
        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1, None);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_annoy_query_euclidean() {
        let mat = create_simple_matrix();
        let index = AnnoyIndex::new(mat.as_ref(), 8, Dist::Euclidean, 42);

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
    fn test_annoy_query_cosine() {
        use crate::utils::dist::*;

        let mat = create_simple_matrix();
        let index = AnnoyIndex::new(mat.as_ref(), 8, Dist::Cosine, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, None);

        // Should find point 0 first (identical direction)
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_annoy_query_k_larger_than_dataset() {
        use crate::utils::dist::*;

        let mat = create_simple_matrix();
        let index = AnnoyIndex::new(mat.as_ref(), 4, Dist::Euclidean, 42);

        let query = vec![1.0, 0.0, 0.0];
        // ask for 10 neighbours but only 5 points exist
        let (indices, _) = index.query(&query, 10, None);

        // Should return at most 5 results
        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_annoy_query_search_k() {
        use crate::utils::dist::*;

        let mat = create_simple_matrix();
        let index = AnnoyIndex::new(mat.as_ref(), 4, Dist::Euclidean, 42);

        let query = vec![1.0, 0.0, 0.0];

        // With higher search_k, should get same or better results
        let (indices1, _) = index.query(&query, 3, Some(10));
        let (indices2, _) = index.query(&query, 3, Some(50));

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_annoy_multiple_trees() {
        let mat = create_simple_matrix();

        // More trees should give better recall
        let index_few = AnnoyIndex::new(mat.as_ref(), 2, Dist::Euclidean, 42);
        let index_many = AnnoyIndex::new(mat.as_ref(), 16, Dist::Euclidean, 42);

        let query = vec![0.9, 0.1, 0.0];
        let (indices1, _) = index_few.query(&query, 3, None);
        let (indices2, _) = index_many.query(&query, 3, None);

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_annoy_query_row() {
        let mat = create_simple_matrix();
        let index = AnnoyIndex::new(mat.as_ref(), 8, Dist::Euclidean, 42);

        // Query using a row from the matrix
        let (indices, distances) = index.query_row(mat.row(0), 1, None);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_annoy_reproducibility() {
        let mat = create_simple_matrix();

        // Same seed should give same results
        let index1 = AnnoyIndex::new(mat.as_ref(), 8, Dist::Euclidean, 42);
        let index2 = AnnoyIndex::new(mat.as_ref(), 8, Dist::Euclidean, 42);

        let query = vec![0.5, 0.5, 0.0];
        let (indices1, _) = index1.query(&query, 3, None);
        let (indices2, _) = index2.query(&query, 3, None);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_annoy_different_seeds() {
        let mat = create_simple_matrix();

        // Different seeds might give different tree structures
        let index1 = AnnoyIndex::new(mat.as_ref(), 8, Dist::Euclidean, 42);
        let index2 = AnnoyIndex::new(mat.as_ref(), 8, Dist::Euclidean, 123);

        let query = vec![0.5, 0.5, 0.0];

        // Both should still find reasonable neighbours (though order might differ)
        let (indices1, _) = index1.query(&query, 3, None);
        let (indices2, _) = index2.query(&query, 3, None);

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_annoy_larger_dataset() {
        // Create a larger synthetic dataset
        let n = 100;
        let dim = 10;
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 10.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = AnnoyIndex::new(mat.as_ref(), 16, Dist::Euclidean, 42);

        // Query for point 0
        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5, None);

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0); // Should find exact match
    }

    #[test]
    fn test_annoy_orthogonal_vectors() {
        use crate::utils::dist::*;
        let data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);
        let index = AnnoyIndex::new(mat.as_ref(), 4, Dist::Cosine, 42);

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
    fn test_annoy_parallel_build() {
        let n = 50;
        let dim = 5;
        let data: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        let index = AnnoyIndex::new(mat.as_ref(), 32, Dist::Euclidean, 42);

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 3, None);

        assert_eq!(indices.len(), 3);
    }
}
