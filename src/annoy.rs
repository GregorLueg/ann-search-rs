use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::iter::Sum;

use crate::utils::Dist;

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
/// * `nodes` - Flattened tree structure containing all split and leaf nodes
/// * `roots` - Starting indices for each tree in the forest
/// * `split_data` - Hyperplane coefficients and offsets for split nodes
/// * `leaf_indices` - Actual data indices stored in leaf nodes
/// * `vectors_flat` - Original vector data, flattened for cache locality
/// * `dim` - Embedding dimensions
/// * `n_trees` - Number of trees in the forest
pub struct AnnoyIndex<T> {
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    nodes: Vec<FlatNode>,
    roots: Vec<u32>,
    split_data: Vec<T>,
    leaf_indices: Vec<usize>,
    pub n_trees: usize,
}

impl<T> AnnoyIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    /// Construct a new Annoy index
    ///
    /// Builds a forest of random projection trees in parallel. Each tree
    /// recursively partitions the space using random hyperplanes chosen
    /// between random pairs of points.
    ///
    /// ### Params
    ///
    /// * `mat` - Data matrix (rows = samples, columns = dimensions)
    /// * `n_trees` - Number of trees to build (more trees = better recall,
    ///   slower build)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    pub fn new(mat: MatRef<T>, n_trees: usize, seed: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let n_vectors = mat.nrows();
        let dim = mat.ncols();

        let mut vectors_flat = Vec::with_capacity(n_vectors * dim);
        for i in 0..n_vectors {
            vectors_flat.extend(mat.row(i).iter().cloned());
        }

        let seeds: Vec<u64> = (0..n_trees).map(|_| rng.random()).collect();

        let forest: Vec<Vec<BuildNode<T>>> = seeds
            .into_par_iter()
            .map(|tree_seed| {
                let mut tree_rng = StdRng::seed_from_u64(tree_seed);
                Self::build_tree_recursive(
                    &vectors_flat,
                    dim,
                    (0..n_vectors).collect(),
                    &mut tree_rng,
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
            n: n_vectors,
        }
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Performs best-first search across all trees, greedily descending to
    /// leaves and backtracking to promising unexplored branches. Stops when
    /// the search budget is exhausted.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `dist_metric` - Distance metric (Euclidean or Cosine)
    /// * `k` - Number of neighbours to return
    /// * `search_k` - Budget of items to examine (higher = better recall, slower)
    ///   Defaults to `k * n_trees` if None
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
    pub fn query(
        &self,
        query_vec: &[T],
        dist_metric: &Dist,
        k: usize,
        search_k: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        // if no search budget is provided, it will default to quite a decent
        // search budget
        let limit = search_k.unwrap_or(k * self.n_trees * 20);
        let mut visited_count = 0;

        let n_vectors = self.vectors_flat.len() / self.dim;
        let mut visited = VisitedSet::new(n_vectors);

        let mut candidates = Vec::with_capacity(limit);
        let mut pq = BinaryHeap::with_capacity(self.n_trees * 2);

        // 1. Initialise PQ with all roots
        for &root in &self.roots {
            pq.push(BacktrackEntry {
                margin: f64::MAX,
                node_idx: root,
            });
        }

        // 2. Tree Traversal
        while visited_count < limit {
            let Some(entry) = pq.pop() else { break };
            let mut current_idx = entry.node_idx;

            // Inner loop: Greedy descent to leaf
            loop {
                let node = unsafe { self.nodes.get_unchecked(current_idx as usize) };

                if node.n_descendants == 1 {
                    // LEAF NODE found
                    let start = node.child_a as usize;
                    let len = node.child_b as usize;

                    visited_count += len;

                    let leaf_items = unsafe { self.leaf_indices.get_unchecked(start..start + len) };

                    for &item in leaf_items {
                        if !visited.mark(item) {
                            candidates.push(item);
                        }
                    }
                    // Done with this path, go back to PQ
                    break;
                } else {
                    // SPLIT NODE
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

                    // push the "far" side to PQ for later
                    pq.push(BacktrackEntry {
                        margin: -margin.abs(),
                        node_idx: farther,
                    });

                    // continue down the "close" side immediately
                    current_idx = closer;
                }
            }
        }

        // 3. compute dist
        let mut scored: Vec<(usize, T)> = Vec::with_capacity(candidates.len());

        match dist_metric {
            Dist::Euclidean => {
                for &idx in &candidates {
                    let vec_start = idx * self.dim;
                    let vec = unsafe {
                        self.vectors_flat
                            .get_unchecked(vec_start..vec_start + self.dim)
                    };
                    let dist = Self::euclidean_dist_sq(query_vec, vec);
                    scored.push((idx, dist));
                }
            }
            Dist::Cosine => {
                let query_norm = query_vec
                    .iter()
                    .map(|&x| x * x)
                    .fold(T::zero(), |acc, x| acc + x)
                    .sqrt();

                for &idx in &candidates {
                    let vec_start = idx * self.dim;
                    let vec = unsafe {
                        self.vectors_flat
                            .get_unchecked(vec_start..vec_start + self.dim)
                    };
                    let dist = Self::cosine_dist(query_vec, vec, query_norm);
                    scored.push((idx, dist));
                }
            }
        }

        if k < scored.len() {
            scored
                .select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            scored.truncate(k);
        }
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        scored.into_iter().unzip()
    }

    /// Query using a matrix row reference
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies
    /// to a temporary vector.
    ///
    /// ### Params
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        dist_metric: &Dist,
        k: usize,
        search_k: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, dist_metric, k, search_k);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, dist_metric, k, search_k)
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
    ) -> Vec<BuildNode<T>> {
        let mut nodes = Vec::with_capacity(items.len());
        Self::build_node(vectors_flat, dim, items, &mut nodes, rng);
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
    ) -> usize {
        if items.len() <= MIN_MEMBERS {
            let node_idx = nodes.len();
            nodes.push(BuildNode::Leaf { items });
            return node_idx;
        }

        for _ in 0..5 {
            let idx1 = items[rng.random_range(0..items.len())];
            let idx2 = items[rng.random_range(0..items.len())];
            if idx1 == idx2 {
                continue;
            }

            let v1_start = idx1 * dim;
            let v2_start = idx2 * dim;
            let v1 = &vectors_flat[v1_start..v1_start + dim];
            let v2 = &vectors_flat[v2_start..v2_start + dim];

            let mut hyperplane = Vec::with_capacity(dim);
            let mut dot_v1 = T::zero();
            let mut dot_v2 = T::zero();

            for k in 0..dim {
                let val1 = v1[k];
                let val2 = v2[k];
                hyperplane.push(val1 - val2);
                dot_v1 = dot_v1 + val1 * val1;
                dot_v2 = dot_v2 + val2 * val2;
            }

            let threshold = (dot_v1 - dot_v2) / T::from_f64(2.0).unwrap();

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

                let left_idx = Self::build_node(vectors_flat, dim, left_items, nodes, rng);
                let right_idx = Self::build_node(vectors_flat, dim, right_items, nodes, rng);

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

    /// Calculate squared Euclidean distance between vectors
    ///
    /// Computes sum((v1[i] - v2[i])^2). Returns squared distance to avoid
    /// expensive sqrt - sufficient for comparison purposes.
    ///
    /// ### Params
    ///
    /// * `v1` - First vector
    /// * `v2` - Second vector
    ///
    /// ### Returns
    ///
    /// Squared Euclidean distance
    #[inline(always)]
    fn euclidean_dist_sq(v1: &[T], v2: &[T]) -> T {
        v1.iter()
            .zip(v2.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .fold(T::zero(), |acc, x| acc + x)
    }

    /// Calculate cosine distance between vectors
    ///
    /// Computes 1 - cos(θ) where θ is the angle between vectors.
    /// Handles zero-norm vectors by returning 1.0 (maximum distance).
    /// Clamps similarity to [0, 1] to prevent NaN from floating-point errors.
    ///
    /// ### Params
    ///
    /// * `v1` - First vector
    /// * `v2` - Second vector
    /// * `v1_norm` - Pre-computed L2 norm of v1 (for efficiency)
    ///
    /// ### Returns
    ///
    /// Cosine distance in range [0, 1]
    #[inline(always)]
    fn cosine_dist(v1: &[T], v2: &[T], v1_norm: T) -> T {
        let (dot, v2_norm_sq) = v1
            .iter()
            .zip(v2.iter())
            .fold((T::zero(), T::zero()), |(d, n), (&a, &b)| {
                (d + a * b, n + b * b)
            });

        let v2_norm = v2_norm_sq.sqrt();
        if v1_norm.is_zero() || v2_norm.is_zero() {
            return T::one();
        }
        // Clamp to prevent NaN due to precision errors
        let sim = dot / (v1_norm * v2_norm);
        if sim > T::one() {
            T::zero()
        } else {
            T::one() - sim
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::Dist;
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
        let _ = AnnoyIndex::new(mat.as_ref(), 4, 42);

        // just verify it doesn't panic
    }

    #[test]
    fn test_annoy_query_finds_self() {
        let mat = create_simple_matrix();
        let index = AnnoyIndex::new(mat.as_ref(), 4, 42);

        // Query with point 0, should find itself first
        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, &Dist::Euclidean, 1, None);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_annoy_query_euclidean() {
        let mat = create_simple_matrix();
        let index = AnnoyIndex::new(mat.as_ref(), 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, &Dist::Euclidean, 3, None);

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
        use crate::utils::Dist;

        let mat = create_simple_matrix();
        let index = AnnoyIndex::new(mat.as_ref(), 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, &Dist::Cosine, 3, None);

        // Should find point 0 first (identical direction)
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_annoy_query_k_larger_than_dataset() {
        use crate::utils::Dist;

        let mat = create_simple_matrix();
        let index = AnnoyIndex::new(mat.as_ref(), 4, 42);

        let query = vec![1.0, 0.0, 0.0];
        // ask for 10 neighbours but only 5 points exist
        let (indices, _) = index.query(&query, &Dist::Euclidean, 10, None);

        // Should return at most 5 results
        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_annoy_query_search_k() {
        use crate::utils::Dist;

        let mat = create_simple_matrix();
        let index = AnnoyIndex::new(mat.as_ref(), 4, 42);

        let query = vec![1.0, 0.0, 0.0];

        // With higher search_k, should get same or better results
        let (indices1, _) = index.query(&query, &Dist::Euclidean, 3, Some(10));
        let (indices2, _) = index.query(&query, &Dist::Euclidean, 3, Some(50));

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_annoy_multiple_trees() {
        let mat = create_simple_matrix();

        // More trees should give better recall
        let index_few = AnnoyIndex::new(mat.as_ref(), 2, 42);
        let index_many = AnnoyIndex::new(mat.as_ref(), 16, 42);

        let query = vec![0.9, 0.1, 0.0];
        let (indices1, _) = index_few.query(&query, &Dist::Euclidean, 3, None);
        let (indices2, _) = index_many.query(&query, &Dist::Euclidean, 3, None);

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_annoy_query_row() {
        let mat = create_simple_matrix();
        let index = AnnoyIndex::new(mat.as_ref(), 8, 42);

        // Query using a row from the matrix
        let (indices, distances) = index.query_row(mat.row(0), &Dist::Euclidean, 1, None);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_annoy_reproducibility() {
        let mat = create_simple_matrix();

        // Same seed should give same results
        let index1 = AnnoyIndex::new(mat.as_ref(), 8, 42);
        let index2 = AnnoyIndex::new(mat.as_ref(), 8, 42);

        let query = vec![0.5, 0.5, 0.0];
        let (indices1, _) = index1.query(&query, &Dist::Euclidean, 3, None);
        let (indices2, _) = index2.query(&query, &Dist::Euclidean, 3, None);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_annoy_different_seeds() {
        let mat = create_simple_matrix();

        // Different seeds might give different tree structures
        let index1 = AnnoyIndex::new(mat.as_ref(), 8, 42);
        let index2 = AnnoyIndex::new(mat.as_ref(), 8, 123);

        let query = vec![0.5, 0.5, 0.0];

        // Both should still find reasonable neighbours (though order might differ)
        let (indices1, _) = index1.query(&query, &Dist::Euclidean, 3, None);
        let (indices2, _) = index2.query(&query, &Dist::Euclidean, 3, None);

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
        let index = AnnoyIndex::new(mat.as_ref(), 16, 42);

        // Query for point 0
        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, &Dist::Euclidean, 5, None);

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0); // Should find exact match
    }

    #[test]
    fn test_annoy_orthogonal_vectors() {
        use crate::utils::Dist;
        let data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);
        let index = AnnoyIndex::new(mat.as_ref(), 4, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, &Dist::Cosine, 3, None);

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

        let index = AnnoyIndex::new(mat.as_ref(), 32, 42);

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, &Dist::Euclidean, 3, None);

        assert_eq!(indices.len(), 3);
    }
}
