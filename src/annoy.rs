use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::utils::Dist;

//////////////////
// Annoy search //
//////////////////

/// Tree node representation for binary space partitioning
///
/// Each node is either a split (with hyperplane and child pointers) or leaf
/// (with vector indices)
#[derive(Clone)]
enum AnnoyNode<T> {
    /// Internal node that splits space using a hyperplane
    Split {
        /// Random hyperplane normal vector for splitting
        hyperplane: Vec<T>,
        /// Threshold for hyperplane decision (median of projections)
        threshold: T,
        /// Index of left child node
        left: usize,
        /// Index of right child node
        right: usize,
    },
    /// Terminal node containing actual vector indices
    Leaf {
        /// Indices of vectors that ended up in this partition
        items: Vec<usize>,
    },
}

/// Approximate Nearest Neighbors index using random binary trees
///
/// Partitions vector space using multiple random hyperplanes to enable fast
/// approximate neighbour search. Trade-off between accuracy and speed
/// controlled by number of trees. (Reminds me of Random Forests...)
///
/// ### Fields
///
/// * `trees` - Collection of binary trees, each with different random
///   partitioning.
/// * `vectors_flat` - Original vector data for distance calculations. Flattened
///   for better cache locality
/// * `dim` - Number of dimensions in the vector
/// * `n_trees` - Number of trees built (more trees = better accuracy, slower
///   queries)
pub struct AnnoyIndex<T> {
    trees: Vec<Vec<AnnoyNode<T>>>,
    vectors_flat: Vec<T>,
    dim: usize,
    n_trees: usize,
}

impl<T> AnnoyIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    /// Creates a new Annoy index from an embedding-type matrix
    ///
    /// ### Params
    ///
    /// * `mat` - Matrix with rows = samples and columns = features.
    /// * `n_trees` - Number of random trees to build (50-100 recommended).
    /// * `seed` - Random seed for reproducible results.
    ///
    /// ### Returns
    ///
    /// Initialised AnnoyIndex ready for querying
    ///
    /// ### Algorithm Details
    ///
    /// **Tree Building Phase:**
    ///
    /// 1. For each tree, generate a random hyperplane (vector of uniform random
    ///    values in [-1,1]).
    /// 2. Project all vectors onto the hyperplane using dot products.
    /// 3. Sort projections and split at median: vectors ≤ median go left,
    ///    others go right.
    /// 4. Recursively apply steps 1-3 to each partition until ≤10 vectors
    ///    remain (leaf nodes).
    /// 5. Repeat process for n_trees independent trees using different random
    ///    seeds
    ///
    /// **Query Phase:**
    ///
    /// 1. For each tree, traverse from root following hyperplane decisions: if
    ///    query·hyperplane ≤ threshold go left, else right.
    /// 2. At boundary decisions (|query·hyperplane - threshold| < 0.5), explore
    ///    both subtrees if search budget allows.
    /// 3. Collect all vector indices from reached leaf nodes across all trees
    /// 4. Remove duplicates and compute exact Euclidean distances to all
    ///    candidates
    /// 5. Return k vectors with smallest distances
    pub fn new(mat: MatRef<T>, n_trees: usize, seed: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed as u64);

        let n_vectors = mat.nrows();
        let dim = mat.ncols();

        let mut vectors_flat = Vec::with_capacity(n_vectors * dim);
        for i in 0..n_vectors {
            vectors_flat.extend(mat.row(i).iter().cloned());
        }

        let seeds: Vec<u64> = (0..n_trees).map(|_| rng.random()).collect();

        let trees: Vec<Vec<AnnoyNode<T>>> = seeds
            .into_par_iter()
            .map(|tree_seed| {
                let mut tree_rng = StdRng::seed_from_u64(tree_seed);
                Self::build_tree(&vectors_flat, dim, (0..n_vectors).collect(), &mut tree_rng)
            })
            .collect();

        AnnoyIndex {
            trees,
            vectors_flat,
            dim,
            n_trees,
        }
    }

    /// Builds a single random tree from the given items
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - All vectors in the dataset. (As a flattened
    ///   structure.)
    /// * `dim` - The original dimensions (row numbers).
    /// * `items` - Subset of vector indices to include in this tree.
    /// * `rng` - Random number generator for this tree.
    ///
    /// ### Returns
    ///
    /// Vector of AnnoyNodes representing the tree structure (index 0 = root)
    fn build_tree(
        vectors_flat: &[T],
        dim: usize,
        items: Vec<usize>,
        rng: &mut StdRng,
    ) -> Vec<AnnoyNode<T>> {
        let mut nodes = Vec::with_capacity(items.len() * 2);
        Self::build_node(vectors_flat, dim, items, &mut nodes, rng);
        nodes
    }

    /// Recursively builds tree nodes using random hyperplanes
    ///
    /// ### Params
    ///  
    /// * `vectors` - All vectors in the dataset
    /// * `items` - Items to split at this node
    /// * `nodes` - Growing list of tree nodes
    /// * `rng` - Random number generator
    ///
    /// ### Returns
    ///
    /// Index of the created node in the nodes vector.
    ///
    /// ### Implementation Details
    ///
    /// - Creates leaf if ≤10 items remain (prevents over-partitioning)
    /// - Hyperplane: random vector with components element [-1,1]  
    /// - Threshold: median of dot products (balanced split)
    /// - Fallback: creates leaf if split fails (empty partitions)
    fn build_node(
        vectors_flat: &[T],
        dim: usize,
        items: Vec<usize>,
        nodes: &mut Vec<AnnoyNode<T>>,
        rng: &mut StdRng,
    ) -> usize {
        if items.len() <= 10 {
            let node_idx = nodes.len();
            nodes.push(AnnoyNode::Leaf { items });
            return node_idx;
        }

        let hyperplane: Vec<T> = (0..dim)
            .map(|_| T::from_f64(rng.random_range(-1.0..1.0)).unwrap())
            .collect();

        let mut item_dots: Vec<(usize, T)> = items
            .iter()
            .map(|&item| {
                let vec_start = item * dim;
                let vec_slice = &vectors_flat[vec_start..vec_start + dim];
                let dot = vec_slice
                    .iter()
                    .zip(&hyperplane)
                    .fold(T::zero(), |acc, (v, h)| acc + *v * *h);
                (item, dot)
            })
            .collect();

        item_dots.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let threshold = item_dots[item_dots.len() / 2].1;

        let mut left_items = Vec::new();
        let mut right_items = Vec::new();

        for (item, dot) in item_dots {
            if dot <= threshold {
                left_items.push(item);
            } else {
                right_items.push(item);
            }
        }

        if left_items.is_empty() || right_items.is_empty() {
            let node_idx = nodes.len();
            nodes.push(AnnoyNode::Leaf { items });
            return node_idx;
        }

        let node_idx = nodes.len();
        nodes.push(AnnoyNode::Split {
            hyperplane: hyperplane.clone(),
            threshold,
            left: 0,
            right: 0,
        });

        let left_idx = Self::build_node(vectors_flat, dim, left_items, nodes, rng);
        let right_idx = Self::build_node(vectors_flat, dim, right_items, nodes, rng);

        if let AnnoyNode::Split {
            ref mut left,
            ref mut right,
            ..
        } = nodes[node_idx]
        {
            *left = left_idx;
            *right = right_idx;
        }

        node_idx
    }

    /// Queries the index for k nearest neighbors with enhanced search
    ///
    /// ### Params
    ///
    /// * `query_vec` - Vector to find neighbors for.
    /// * `dist_metric` - Which distance metric to use.
    /// * `k` - Number of neighbors to return
    /// * `search_k` - Search budget (None = k * n_trees, higher = better
    ///   recall)
    ///
    /// ### Returns
    ///
    /// Tuple of `(nearest neighbour indices, distances)`
    #[inline]
    pub fn query(
        &self,
        query_vec: &[T],
        dist_metric: &Dist,
        k: usize,
        search_k: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        let search_k = search_k.unwrap_or(k * self.n_trees);
        let mut candidates = Vec::with_capacity(search_k * 2);
        let budget_per_tree = (search_k / self.n_trees).max(k);

        for tree in &self.trees {
            self.query_tree_enhanced(tree, query_vec, &mut candidates, budget_per_tree);
        }

        candidates.sort_unstable();
        candidates.dedup();

        let mut scored = Vec::with_capacity(candidates.len());

        match dist_metric {
            Dist::Euclidean => {
                for idx in candidates {
                    let vec_start = idx * self.dim;
                    let dist: T = unsafe {
                        let vec_ptr = self.vectors_flat.as_ptr().add(vec_start);
                        let query_ptr = query_vec.as_ptr();
                        (0..self.dim).fold(T::zero(), |acc, i| {
                            let diff = *vec_ptr.add(i) - *query_ptr.add(i);
                            acc + diff * diff
                        })
                    };
                    scored.push((idx, dist));
                }
            }
            Dist::Cosine => {
                let norm_query: T = query_vec
                    .iter()
                    .map(|v| *v * *v)
                    .fold(T::zero(), |a, b| a + b)
                    .sqrt();

                for idx in candidates {
                    let vec_start = idx * self.dim;
                    let (dot, norm_vec): (T, T) = unsafe {
                        let vec_ptr = self.vectors_flat.as_ptr().add(vec_start);
                        let query_ptr = query_vec.as_ptr();
                        (0..self.dim).fold((T::zero(), T::zero()), |(d, n), i| {
                            let v = *vec_ptr.add(i);
                            let q = *query_ptr.add(i);
                            (d + v * q, n + v * v)
                        })
                    };
                    let dist = T::one() - (dot / (norm_query * norm_vec.sqrt()));
                    scored.push((idx, dist));
                }
            }
        }

        let k = k.min(scored.len());
        if k < scored.len() {
            scored.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
            scored.truncate(k);
        }
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let (indices, distances): (Vec<_>, Vec<_>) = scored.into_iter().unzip();
        (indices, distances)
    }

    /// Queries the index for k nearest neighbors with enhanced search
    ///
    /// ### Params
    ///
    /// * `query_row` - RowRef from a matrix in which rows = samples and columns
    ///   features.
    /// * `dist_metric` - Which distance metric to use.
    /// * `k` - Number of neighbors to return
    /// * `search_k` - Search budget (None = k * n_trees, higher = better
    ///   recall)
    ///
    /// ### Returns
    ///
    /// Vector of k nearest neighbor indices, sorted by distance
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

    /// Enhanced tree query with multi-path traversal and search budget
    ///
    /// ### Params
    ///
    /// * `tree` - Array of tree nodes (index 0 = root)  
    /// * `query_vec` - Query vector for similarity matching
    /// * `candidates` - Accumulating list of potential neighbor indices
    /// * `budget` - Maximum candidates to collect from this tree
    #[inline]
    fn query_tree_enhanced(
        &self,
        tree: &[AnnoyNode<T>],
        query_vec: &[T],
        candidates: &mut Vec<usize>,
        budget: usize,
    ) {
        let mut remaining_budget = budget;
        Self::traverse_node_enhanced(tree, 0, query_vec, candidates, &mut remaining_budget);
    }

    /// Multi-path tree traversal with budget-controlled exploration
    ///
    /// Explores both sides of splits when query point is near the boundary,
    /// improving recall at the cost of some performance.
    ///
    /// ### Params
    ///
    /// * `tree` - Tree nodes
    /// * `node_idx` - Current node index
    /// * `query_vec` - Query vector
    /// * `candidates` - Growing candidate list
    /// * `remaining_budget` - Candidates left to collect
    fn traverse_node_enhanced(
        tree: &[AnnoyNode<T>],
        node_idx: usize,
        query_vec: &[T],
        candidates: &mut Vec<usize>,
        remaining_budget: &mut usize,
    ) {
        if *remaining_budget == 0 {
            return;
        }

        match &tree[node_idx] {
            AnnoyNode::Leaf { items } => {
                let items_to_add = items.len().min(*remaining_budget);
                candidates.extend(&items[..items_to_add]);
                *remaining_budget = remaining_budget.saturating_sub(items_to_add);
            }
            AnnoyNode::Split {
                hyperplane,
                threshold,
                left,
                right,
            } => {
                let dot = query_vec
                    .iter()
                    .zip(hyperplane)
                    .fold(T::zero(), |acc, (q, h)| acc + *q * *h);

                let margin = (dot - *threshold).abs();
                let half = T::from_f64(0.5).unwrap();

                // more sensible threshold now
                if margin < half && *remaining_budget > 5 {
                    let half_budget = *remaining_budget / 2;

                    let (first, second) = if dot <= *threshold {
                        (*left, *right)
                    } else {
                        (*right, *left)
                    };

                    let mut first_budget = half_budget + (*remaining_budget % 2);
                    Self::traverse_node_enhanced(
                        tree,
                        first,
                        query_vec,
                        candidates,
                        &mut first_budget,
                    );

                    let mut second_budget = half_budget;
                    Self::traverse_node_enhanced(
                        tree,
                        second,
                        query_vec,
                        candidates,
                        &mut second_budget,
                    );

                    *remaining_budget = 0;
                } else {
                    let next = if dot <= *threshold { *left } else { *right };
                    Self::traverse_node_enhanced(
                        tree,
                        next,
                        query_vec,
                        candidates,
                        remaining_budget,
                    );
                }
            }
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
