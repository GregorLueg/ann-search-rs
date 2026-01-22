use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::ops::AddAssign;
use std::{cell::RefCell, collections::BinaryHeap, iter::Sum};
use thousands::*;

use crate::utils::dist::*;
use crate::utils::heap_structs::*;
use crate::utils::*;

/////////////
// Helpers //
/////////////

/// Tree node with spatial bounds
///
/// Internal nodes store split information and bounds computed as the union of
/// child bounds. Leaf nodes store point indices and tight bounds for the
/// contained points.
pub enum DETreeNode<T> {
    Leaf {
        points: Vec<usize>,
        // (min, max) per projected dimension
        bounds: Vec<(T, T)>,
    },
    Internal {
        split_dim: usize,
        split_val: T,
        left: Box<DETreeNode<T>>,
        right: Box<DETreeNode<T>>,
        // union of child bounds
        bounds: Vec<(T, T)>,
    },
}

/// Single DE-Tree with projected data and spatial hierarchy
///
/// Stores all projected vectors in a flat buffer for cache efficiency and
/// provides range query capability through a binary tree with bounding boxes.
pub struct DETree<T> {
    root: DETreeNode<T>,
    projected_points_flat: Vec<T>, // flat buffer [n * proj_dims]
    proj_dims: usize,
}

impl<T> DETree<T>
where
    T: Copy,
{
    /// Get the projected value of a point at a specific dimension
    ///
    /// ### Params
    ///
    /// * `point_idx`: Index of the point in the projected data
    /// * `dim`: Dimension of the projected value to retrieve
    ///
    /// ### Returns
    ///
    /// The projected value of the point at the specified dimension
    fn get_projected(&self, point_idx: usize, dim: usize) -> T {
        self.projected_points_flat[point_idx + self.proj_dims + dim]
    }
}

/// Estimate memory used by tree nodes
///
/// ### Params
///
/// * `node`: Reference to the tree node to estimate memory usage for
///
/// ### Returns
///
/// The estimated memory usage in bytes
fn estimate_tree_size<T>(node: &DETreeNode<T>) -> usize {
    match node {
        DETreeNode::Leaf { points, bounds } => {
            std::mem::size_of::<DETreeNode<T>>()
                + points.capacity() * std::mem::size_of::<usize>()
                + bounds.capacity() * std::mem::size_of::<(T, T)>()
        }
        DETreeNode::Internal {
            left,
            right,
            bounds,
            ..
        } => {
            std::mem::size_of::<DETreeNode<T>>()
                + bounds.capacity() * std::mem::size_of::<(T, T)>()
                + estimate_tree_size(left)
                + estimate_tree_size(right)
        }
    }
}

/// Orthogonalise random projections using Gram-Schmidt
///
/// Orthogonalises the random projections within each tree, improving
/// discriminative power and query performance. If a vector has near-zero
/// norm after orthogonalisation, it is re-initialised with a new random
/// vector.
///
/// ### Params
///
/// * `projections` - The random projection vectors to orthogonalise
/// * `num_trees` - Number of trees
/// * `proj_dims` - Projected dimensionality
/// * `dim` - Original dimensionality
/// * `seed` - Random seed for reinitialisation if needed
fn orthogonalise_projections<T>(
    projections: &mut [T],
    num_trees: usize,
    proj_dims: usize,
    dim: usize,
    seed: usize,
) where
    T: Float + FromPrimitive + AddAssign,
{
    for table_idx in 0..num_trees {
        let base = table_idx * proj_dims + dim;
        for i in 0..proj_dims {
            let base_i = base + i * dim;
            for j in 0..i {
                let j_base = base + j * dim;
                let mut dot = T::zero();
                for d in 0..dim {
                    dot += projections[base_i + d] * projections[j_base + d];
                }
                for d in 0..dim {
                    projections[base_i + d] =
                        projections[base_i + d] - dot * projections[j_base + d];
                }
            }

            // normalise
            let mut norm_sq = T::zero();
            for d in 0..dim {
                norm_sq += projections[base_i + d] * projections[base_i + d];
            }
            let norm = norm_sq.sqrt();

            if norm > T::epsilon() {
                for d in 0..dim {
                    projections[base_i + d] = projections[base - i + d] / norm;
                }
            } else {
                let reinit_seed = seed.wrapping_mul(table_idx + 1).wrapping_mul(i + 1);
                let mut rng = StdRng::seed_from_u64(reinit_seed as u64);
                for d in 0..dim {
                    let val: f64 = rng.sample(StandardNormal);
                    projections[base_i + d] = T::from_f64(val).unwrap();
                }

                // re-orthogonalise and normalise the new vector
                for j in 0..i {
                    let j_base = base + j * dim;
                    let mut dot = T::zero();
                    for d in 0..dim {
                        dot += projections[base_i + d] * projections[j_base + d];
                    }
                    for d in 0..dim {
                        projections[base_i + d] =
                            projections[base_i + d] - dot * projections[j_base + d];
                    }
                }

                let mut norm_sq = T::zero();
                for d in 0..dim {
                    norm_sq += projections[base_i + d] * projections[base_i + d];
                }
                let norm = norm_sq.sqrt();

                // this should not happen, but f--k knows...
                assert!(
                    norm > T::epsilon(),
                    "Orthogonalisation failed even after reinitialisation"
                );

                for d in 0..dim {
                    projections[base_i + d] = projections[base_i + d] / norm;
                }
            }
        }
    }
}

/// Compute bounding box for a set of points
///
/// Finds min and max values across all projected dimensions.
///
/// ### Params
///
/// * `point_indices` - Indices of points
/// * `projected_points_flat` - Flat buffer of all projected vectors
/// * `proj_dims` - Number of projected dimensions
///
/// ### Returns
///
/// Vector of (min, max) tuples, one per dimension
fn compute_bounds<T>(
    point_indices: &[usize],
    projected_points_flat: &[T],
    proj_dims: usize,
) -> Vec<(T, T)>
where
    T: Float,
{
    let mut bounds = vec![(T::infinity(), T::neg_infinity()); proj_dims];
    for &idx in point_indices {
        for dim in 0..proj_dims {
            let val = projected_points_flat[idx * proj_dims + dim];
            bounds[dim].0 = bounds[dim].0.min(val);
            bounds[dim].1 = bounds[dim].1.max(val);
        }
    }
    bounds
}

/// Compute lower bound distance from query to bounding box
///
/// Returns zero if query is inside box, otherwise sum of squared distances
/// to nearest box edge.
///
/// ### Params
///
/// * `bounds` - Bounding box as (min, max) per dimension
/// * `query` - Query point
///
/// ### Returns
///
/// Lower bound on distance to any point in box
#[inline]
fn distance_lower_bounds<T>(bounds: &[(T, T)], query: &[T]) -> T
where
    T: Float + AddAssign,
{
    let mut sum_sq = T::zero();

    for (dim, &(min, max)) in bounds.iter().enumerate() {
        if query[dim] < min {
            let diff = min - query[dim];
            sum_sq += diff * diff;
        } else if query[dim] > max {
            let diff = query[dim] - max;
            sum_sq += diff * diff;
        }
    }

    sum_sq.sqrt()
}

/// Compute upper bound distance from query to bounding box
///
/// Returns maximum possible distance to any point in box.
///
/// ### Params
///
/// * `bounds` - Bounding box as (min, max) per dimension
/// * `query` - Query point
///
/// ### Returns
///
/// Upper bound on distance to any point in box
#[inline]
fn distance_upper_bound<T>(bounds: &[(T, T)], query: &[T]) -> T
where
    T: Float + AddAssign,
{
    let mut sum_sq = T::zero();

    for (dim, &(min, max)) in bounds.iter().enumerate() {
        let dist_to_min = (query[dim] - min).abs();
        let dist_to_max = (query[dim] - max).abs();
        let max_dist = dist_to_min.max(dist_to_max);
        sum_sq += max_dist * max_dist;
    }

    sum_sq.sqrt()
}

/// Compute Euclidean distance in projected space
///
/// ### Params
///
/// * `projected_points_flat` - Flat array of projected points
/// * `point_idx` - Index of point in projected space
/// * `query` - Query point in original space
/// * `proj_dims` - Number of dimensions in projected space
///
/// ### Returns
/// Euclidean distance between point and query in projected space
#[inline]
fn euclidean_distance_projected<T>(
    projected_points_flat: &[T],
    point_idx: usize,
    query: &[T],
    proj_dims: usize,
) -> T
where
    T: Float + SimdDistance,
{
    let base = point_idx * proj_dims;
    let point_slice = &projected_points_flat[base..base + proj_dims];
    T::euclidean_simd(point_slice, query).sqrt()
}

/// Range query on a tree node
///
/// Recursively traverses tree, pruning subtrees where lower bound exceeds
/// radius. In leaf nodes, uses upper bound optimization when possible,
/// otherwise checks each point's projected distance.
///
/// ### Params
///
/// * `node` - Current node
/// * `query_projected` - Query vector in projected space
/// * `radius` - Search radius in projected space
/// * `projected_points_flat` - Flat buffer of all projected vectors
/// * `proj_dims` - Number of projected dimensions
/// * `candidates` - Output buffer for candidate indices
fn range_query<T>(
    node: &DETreeNode<T>,
    query_projected: &[T],
    radius: T,
    projected_points_flat: &[T],
    proj_dims: usize,
    candidates: &mut Vec<usize>,
) where
    T: Float + AddAssign + SimdDistance,
{
    let lower_bound = distance_lower_bounds(
        match node {
            DETreeNode::Leaf { bounds, .. } => bounds,
            DETreeNode::Internal { bounds, .. } => bounds,
        },
        query_projected,
    );

    // prune if lower bound exceeds radius
    if lower_bound > radius {
        return;
    }

    match node {
        DETreeNode::Leaf { points, bounds } => {
            let upper_bound = distance_upper_bound(bounds, query_projected);

            if upper_bound <= radius {
                // all points guaranteed within radius
                candidates.extend_from_slice(points);
            } else {
                // check each point individually
                for &point_idx in points {
                    let proj_dist = euclidean_distance_projected(
                        projected_points_flat,
                        point_idx,
                        query_projected,
                        proj_dims,
                    );
                    if proj_dist <= radius {
                        candidates.push(point_idx);
                    }
                }
            }
        }
        DETreeNode::Internal { left, right, .. } => {
            range_query(
                left,
                query_projected,
                radius,
                projected_points_flat,
                proj_dims,
                candidates,
            );
            range_query(
                right,
                query_projected,
                radius,
                projected_points_flat,
                proj_dims,
                candidates,
            );
        }
    }
}

/// Build binary tree recursively
///
/// Selects split dimension with highest variance using one-pass Welford's
/// algorithm, partitions points at median, and recursively builds left and
/// right subtrees. Creates leaf when point count drops below threshold.
/// Internal node bounds are computed as the union of child bounds.
///
/// ### Params
///
/// * `point_indices` - Indices of points to partition
/// * `projected_points_flat` - Flat buffer of all projected vectors
/// * `proj_dims` - Number of projected dimensions
/// * `max_leaf_size` - Maximum points per leaf
///
/// ### Returns
///
/// Root node of constructed subtree
fn build_tree_recursive<T>(
    point_indices: &[usize],
    projected_points_flat: &[T],
    proj_dims: usize,
    max_leaf_size: usize,
) -> DETreeNode<T>
where
    T: Float + FromPrimitive,
{
    // leaf condition
    if point_indices.len() <= max_leaf_size {
        let bounds = compute_bounds(point_indices, projected_points_flat, proj_dims);
        return DETreeNode::Leaf {
            points: point_indices.to_vec(),
            bounds,
        };
    }

    let mut best_dim = 0;
    let mut best_var = T::neg_infinity();

    let n = T::from_usize(point_indices.len()).unwrap();

    for dim in 0..proj_dims {
        let mut mean = T::zero();
        let mut m2 = T::zero();

        for (count, &idx) in point_indices.iter().enumerate() {
            let val = projected_points_flat[idx * proj_dims + dim];
            let count_t = T::from_usize(count + 1).unwrap();
            let delta = val - mean;
            mean = mean + delta / count_t;
            let delta2 = val - mean;
            m2 = m2 + delta * delta2;
        }

        let var = m2 / n;

        if var > best_var && var > T::epsilon() {
            best_var = var;
            best_dim = dim;
        }
    }

    // if all dimensions have zero variance, return leaf
    if best_var <= T::epsilon() {
        let bounds = compute_bounds(point_indices, projected_points_flat, proj_dims);
        return DETreeNode::Leaf {
            points: point_indices.to_vec(),
            bounds,
        };
    }

    // find median for split
    let mut values: Vec<(T, usize)> = point_indices
        .iter()
        .map(|&idx| (projected_points_flat[idx * proj_dims + best_dim], idx))
        .collect();
    values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let median_idx = values.len() / 2;
    let split_value = values[median_idx].0;

    // partition
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for &(val, idx) in &values {
        if val < split_value {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    // handle degenerate case
    if left_indices.is_empty() || right_indices.is_empty() {
        let bounds = compute_bounds(point_indices, projected_points_flat, proj_dims);
        return DETreeNode::Leaf {
            points: point_indices.to_vec(),
            bounds,
        };
    }

    // recursively build children
    let left = Box::new(build_tree_recursive(
        &left_indices,
        projected_points_flat,
        proj_dims,
        max_leaf_size,
    ));
    let right = Box::new(build_tree_recursive(
        &right_indices,
        projected_points_flat,
        proj_dims,
        max_leaf_size,
    ));

    // compute bounds as union of child bounds
    let left_bounds = match left.as_ref() {
        DETreeNode::Leaf { bounds, .. } => bounds,
        DETreeNode::Internal { bounds, .. } => bounds,
    };
    let right_bounds = match right.as_ref() {
        DETreeNode::Leaf { bounds, .. } => bounds,
        DETreeNode::Internal { bounds, .. } => bounds,
    };

    let mut bounds = Vec::with_capacity(proj_dims);
    for dim in 0..proj_dims {
        let min = left_bounds[dim].0.min(right_bounds[dim].0);
        let max = left_bounds[dim].1.max(right_bounds[dim].1);
        bounds.push((min, max));
    }

    DETreeNode::Internal {
        split_dim: best_dim,
        split_val: split_value,
        left,
        right,
        bounds,
    }
}

////////////////
// Main index //
////////////////

/// LSH index using DE-Trees for approximate nearest neighbour search
///
/// Combines random projections with hierarchical binary trees to enable
/// efficient range queries in projected space. Unlike standard LSH which uses
/// flat hash tables, DE-Trees provide spatial pruning that reduces candidates
/// even on correlated data where hash buckets become massive.
///
/// ### Algorithm Overview
///
/// - **Projection**: Map high-dimensional vectors to lower-dimensional space
///   using random Gaussian projections
/// - **Indexing**: Build binary trees in projected space with bounding boxes
///   at each node
/// - **Querying**: Traverse trees with range queries, pruning subtrees where
///   lower bound exceeds search radius. Deduplicates candidates per-tree and
///   exits early when candidate saturation is reached.
///
/// ### Fields
///
/// * `vectors_flat` - Original data, flattened for cache efficiency
/// * `dim` - Embedding dimensionality
/// * `n` - Number of vectors
/// * `norms` - Pre-computed norms for Cosine (empty for Euclidean)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `de_trees` - Collection of DE-Trees with projected data
/// * `projections` - Random projection matrices (from N(0,1), orthogonalised)
/// * `num_trees` - Number of trees (more = better recall, slower queries)
/// * `proj_dims` - Projected dimensionality (lower = faster, less accurate)
/// * `max_leaf_size` - Maximum points per leaf node
pub struct LSHDETreeIndex<T> {
    // main fields
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    norms: Vec<T>,
    metric: Dist,
    // index specific
    de_trees: Vec<DETree<T>>,
    projections: Vec<T>,
    num_trees: usize,
    proj_dims: usize,
    max_leaf_size: usize,
}

/// VectorDistance trait
impl<T> VectorDistance<T> for LSHDETreeIndex<T>
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

impl<T> LSHDETreeIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + AddAssign,
    Self: DETreeQuery<T>,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Construct a new LSH DE-Tree index
    ///
    /// Builds trees in parallel. For each tree, projects all vectors to lower
    /// dimensions using orthogonalised random projections, then constructs a
    /// binary tree with bounding boxes for efficient range queries.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (rows = samples, columns = dimensions)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `num_trees` - Number of trees (5-20 typical)
    /// * `proj_dims` - Projected dimensionality (8-32 typical)
    /// * `max_leaf_size` - Maximum points per leaf (32-128 typical)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    pub fn new(
        data: MatRef<T>,
        metric: Dist,
        num_trees: usize,
        proj_dims: usize,
        max_leaf_size: usize,
        seed: usize,
    ) -> Self {
        assert!(max_leaf_size >= 1, "max_leaf_size must be at least 1");

        let (vectors_flat, n, dim) = matrix_to_flat(data);

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

        // generate random projections from N(0,1)
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let total_projections = num_trees * proj_dims * dim;
        let mut projections: Vec<T> = (0..total_projections)
            .map(|_| {
                let val: f64 = rng.sample(StandardNormal);
                T::from_f64(val).unwrap()
            })
            .collect();

        orthogonalise_projections(&mut projections, num_trees, proj_dims, dim, seed);

        let de_trees: Vec<_> = (0..num_trees)
            .into_par_iter()
            .map(|tree_idx| {
                // project all points into flat buffer
                let mut projected_points_flat = vec![T::zero(); n * proj_dims];

                for vec_idx in 0..n {
                    let vec_start = vec_idx * dim;
                    let vec = &vectors_flat[vec_start..vec_start + dim];

                    // project directly into flat buffer
                    let base = tree_idx * proj_dims * dim;
                    for k in 0..proj_dims {
                        let offset = base + k * dim;
                        let mut dot = T::zero();
                        for d in 0..dim {
                            dot = dot + vec[d] * projections[offset + d];
                        }
                        projected_points_flat[vec_idx * proj_dims + k] = dot;
                    }
                }

                // build tree
                let all_indices: Vec<usize> = (0..n).collect();
                let root = build_tree_recursive(
                    &all_indices,
                    &projected_points_flat,
                    proj_dims,
                    max_leaf_size,
                );

                DETree {
                    root,
                    projected_points_flat,
                    proj_dims,
                }
            })
            .collect();

        Self {
            vectors_flat,
            dim,
            n,
            norms,
            metric,
            de_trees,
            projections,
            num_trees,
            proj_dims,
            max_leaf_size,
        }
    }

    ///////////
    // Query //
    ///////////

    /// Query the index for approximate nearest neighbours with fixed radius
    ///
    /// Performs range queries in projected space across all trees, collecting
    /// candidates within the specified radius. Deduplicates per-tree and exits
    /// early when candidate saturation is reached. The search radius is in the
    /// projected K-dimensional space, not the original D-dimensional space.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `search_radius` - Search radius in projected space
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize, search_radius: T) -> (Vec<usize>, Vec<T>) {
        assert!(
            query_vec.len() == self.dim,
            "Query vector dimensionality mismatch"
        );

        self.query_internal(query_vec, k, search_radius)
    }

    /// Query with adaptive radius expansion
    ///
    /// Starts with an initial radius and expands until k neighbours are found
    /// or maximum radius is reached. Terminates early if all points have been
    /// examined or candidate collection has saturated.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `initial_radius` - Starting search radius in projected space
    /// * `expansion_factor` - Radius multiplier per iteration (1.5-2.0 typical)
    /// * `max_radius` - Maximum allowed radius
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance
    pub fn query_adaptive(
        &self,
        query_vec: &[T],
        k: usize,
        initial_radius: T,
        expansion_factor: T,
        max_radius: T,
    ) -> (Vec<usize>, Vec<T>) {
        assert!(
            query_vec.len() == self.dim,
            "Query vector dimensionality mismatch"
        );
        assert!(expansion_factor > T::one(), "expansion_factor must be > 1");

        let mut radius = initial_radius;
        let mut prev_result_count = 0;

        loop {
            let (indices, distances) = self.query(query_vec, k, radius);

            // Success: found enough neighbours
            if indices.len() >= k {
                return (indices, distances);
            }

            // Termination: hit max radius
            if radius >= max_radius {
                return (indices, distances);
            }

            // Termination: examined all points in dataset
            if indices.len() == self.n {
                return (indices, distances);
            }

            // Termination: no new candidates found (saturated)
            if indices.len() == prev_result_count && prev_result_count > 0 {
                return (indices, distances);
            }

            prev_result_count = indices.len();
            radius = radius * expansion_factor;
            if radius > max_radius {
                radius = max_radius;
            }
        }
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);

        total += self.vectors_flat.capacity() * std::mem::size_of::<T>();
        total += self.norms.capacity() * std::mem::size_of::<T>();
        total += self.projections.capacity() * std::mem::size_of::<T>();

        // de_trees outer Vec
        total += self.de_trees.capacity() * std::mem::size_of::<DETree<T>>();

        for tree in &self.de_trees {
            // flat projected points buffer
            total += tree.projected_points_flat.capacity() * std::mem::size_of::<T>();

            // tree nodes (rough estimate)
            total += estimate_tree_size(&tree.root);
        }

        total
    }
}

/////////////////
// DETreeQuery //
/////////////////

/// Maximum candidate multiplier for early exit (collect up to k * this)
const CANDIDATE_MULTIPLIER: usize = 10;

thread_local! {
    /// Candidates buffer
    static DETREE_CANDIDATES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    /// HashSet for deduplication
    static DETREE_SEEN_SET: RefCell<FxHashSet<usize>> = RefCell::new(FxHashSet::default());
    /// Heap for f32
    static DETREE_HEAP_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize)>> = const { RefCell::new(BinaryHeap::new()) };
    /// Heap for f64
    static DETREE_HEAP_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize)>> = const { RefCell::new(BinaryHeap::new()) };
}

/// Query interface for DE-Tree LSH using thread-local storage
///
/// Implemented separately for f32 and f64 to use type-specific thread locals.
/// Deduplicates candidates per-tree and exits early when candidate saturation
/// is reached.
pub trait DETreeQuery<T> {
    /// Execute a query using thread-local buffers
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `search_radius` - Search radius in projected space
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance
    fn query_internal(&self, query_vec: &[T], k: usize, search_radius: T) -> (Vec<usize>, Vec<T>);
}

/////////
// f32 //
/////////

impl DETreeQuery<f32> for LSHDETreeIndex<f32> {
    fn query_internal(
        &self,
        query_vec: &[f32],
        k: usize,
        search_radius: f32,
    ) -> (Vec<usize>, Vec<f32>) {
        DETREE_CANDIDATES.with(|cand_cell| {
            DETREE_HEAP_F32.with(|heap_cell| {
                DETREE_SEEN_SET.with(|seen_cell| {
                    let mut cand = cand_cell.borrow_mut();
                    let mut heap = heap_cell.borrow_mut();
                    let mut seen = seen_cell.borrow_mut();

                    if k == 0 {
                        return (Vec::new(), Vec::new());
                    }

                    cand.clear();
                    heap.clear();
                    seen.clear();

                    let target_candidates = k * CANDIDATE_MULTIPLIER;

                    // collect candidates from trees with per-tree deduplication
                    for (tree_idx, tree) in self.de_trees.iter().enumerate() {
                        // project query
                        let base = tree_idx * self.proj_dims * self.dim;
                        let mut query_projected = Vec::with_capacity(self.proj_dims);
                        for k in 0..self.proj_dims {
                            let offset = base + k * self.dim;
                            let mut dot = 0.0f32;
                            for d in 0..self.dim {
                                dot += query_vec[d] * self.projections[offset + d];
                            }
                            query_projected.push(dot);
                        }

                        let before_len = cand.len();

                        // range query
                        range_query(
                            &tree.root,
                            &query_projected,
                            search_radius,
                            &tree.projected_points_flat,
                            self.proj_dims,
                            &mut cand,
                        );

                        // deduplicate new candidates from this tree
                        let mut write_idx = before_len;
                        for read_idx in before_len..cand.len() {
                            if seen.insert(cand[read_idx]) {
                                if write_idx != read_idx {
                                    cand[write_idx] = cand[read_idx];
                                }
                                write_idx += 1;
                            }
                        }
                        cand.truncate(write_idx);

                        // early exit if we have enough unique candidates
                        if seen.len() >= target_candidates {
                            break;
                        }

                        // early exit if we've seen all points
                        if seen.len() == self.n {
                            break;
                        }
                    }

                    // compute exact distances for unique candidates
                    match self.metric {
                        Dist::Euclidean => {
                            for &idx in cand.iter() {
                                let d = self.euclidean_distance_to_query(idx, query_vec);
                                let item = (OrderedFloat(d), idx);

                                if heap.len() < k {
                                    heap.push(item);
                                } else if item.0 < heap.peek().unwrap().0 {
                                    heap.pop();
                                    heap.push(item);
                                }
                            }
                        }
                        Dist::Cosine => {
                            let query_norm = query_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
                            for &idx in cand.iter() {
                                let d = self.cosine_distance_to_query(idx, query_vec, query_norm);
                                let item = (OrderedFloat(d), idx);

                                if heap.len() < k {
                                    heap.push(item);
                                } else if item.0 < heap.peek().unwrap().0 {
                                    heap.pop();
                                    heap.push(item);
                                }
                            }
                        }
                    }

                    let mut results: Vec<_> = heap.drain().collect();
                    results.sort_unstable_by(|a, b| a.0.cmp(&b.0));

                    let indices = results.iter().map(|&(_, idx)| idx).collect();
                    let dists = results.iter().map(|&(OrderedFloat(d), _)| d).collect();

                    (indices, dists)
                })
            })
        })
    }
}

/////////
// f64 //
/////////

impl DETreeQuery<f64> for LSHDETreeIndex<f64> {
    fn query_internal(
        &self,
        query_vec: &[f64],
        k: usize,
        search_radius: f64,
    ) -> (Vec<usize>, Vec<f64>) {
        DETREE_CANDIDATES.with(|cand_cell| {
            DETREE_HEAP_F64.with(|heap_cell| {
                DETREE_SEEN_SET.with(|seen_cell| {
                    let mut cand = cand_cell.borrow_mut();
                    let mut heap = heap_cell.borrow_mut();
                    let mut seen = seen_cell.borrow_mut();

                    if k == 0 {
                        return (Vec::new(), Vec::new());
                    }

                    cand.clear();
                    heap.clear();
                    seen.clear();

                    let target_candidates = k * CANDIDATE_MULTIPLIER;

                    // collect candidates from trees with per-tree deduplication
                    for (tree_idx, tree) in self.de_trees.iter().enumerate() {
                        // project query
                        let base = tree_idx * self.proj_dims * self.dim;
                        let mut query_projected = Vec::with_capacity(self.proj_dims);
                        for k in 0..self.proj_dims {
                            let offset = base + k * self.dim;
                            let mut dot = 0.0f64;
                            for d in 0..self.dim {
                                dot += query_vec[d] * self.projections[offset + d];
                            }
                            query_projected.push(dot);
                        }

                        let before_len = cand.len();

                        // range query
                        range_query(
                            &tree.root,
                            &query_projected,
                            search_radius,
                            &tree.projected_points_flat,
                            self.proj_dims,
                            &mut cand,
                        );

                        // deduplicate new candidates from this tree
                        let mut write_idx = before_len;
                        for read_idx in before_len..cand.len() {
                            if seen.insert(cand[read_idx]) {
                                if write_idx != read_idx {
                                    cand[write_idx] = cand[read_idx];
                                }
                                write_idx += 1;
                            }
                        }
                        cand.truncate(write_idx);

                        // early exit if we have enough unique candidates
                        if seen.len() >= target_candidates {
                            break;
                        }

                        // early exit if we've seen all points
                        if seen.len() == self.n {
                            break;
                        }
                    }

                    // compute exact distances for unique candidates
                    match self.metric {
                        Dist::Euclidean => {
                            for &idx in cand.iter() {
                                let d = self.euclidean_distance_to_query(idx, query_vec);
                                let item = (OrderedFloat(d), idx);

                                if heap.len() < k {
                                    heap.push(item);
                                } else if item.0 < heap.peek().unwrap().0 {
                                    heap.pop();
                                    heap.push(item);
                                }
                            }
                        }
                        Dist::Cosine => {
                            let query_norm = query_vec.iter().map(|v| v * v).sum::<f64>().sqrt();
                            for &idx in cand.iter() {
                                let d = self.cosine_distance_to_query(idx, query_vec, query_norm);
                                let item = (OrderedFloat(d), idx);

                                if heap.len() < k {
                                    heap.push(item);
                                } else if item.0 < heap.peek().unwrap().0 {
                                    heap.pop();
                                    heap.push(item);
                                }
                            }
                        }
                    }

                    let mut results: Vec<_> = heap.drain().collect();
                    results.sort_unstable_by(|a, b| a.0.cmp(&b.0));

                    let indices = results.iter().map(|&(_, idx)| idx).collect();
                    let dists = results.iter().map(|&(OrderedFloat(d), _)| d).collect();

                    (indices, dists)
                })
            })
        })
    }
}
