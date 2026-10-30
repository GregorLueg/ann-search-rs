//! BallTree implementation in ann-search-rs.

use faer::RowRef;
use num_traits::Float;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::BinaryHeap;

use crate::prelude::*;
use crate::utils::dist::cosine_from_dot;
use crate::utils::tree_utils::*;
use crate::utils::verbosity::{BuildLog, QueryProgress};
use crate::utils::*;

////////////
// Consts //
////////////

/// Sequential Lloyd passes on a small node's two-way split (an in-place
/// 2-means). `train_centroids` is avoided because it fans out over rayon
/// internally, which is wasteful inside an already parallel recursion.
const BALL_SPLIT_ITERS: usize = 5;

/// Nodes above this size split on furthest-point pivots with a single Lloyd
/// pass; at or below it they get D²-sampled pivots and `BALL_SPLIT_ITERS`.
const BALL_SMALL_NODE: usize = 500;

/////////////
// Helpers //
/////////////

/// Find furthest points from pivot
///
/// ### Params
///
/// * `pivot` - Pivot point
/// * `data` - Slice of the data
/// * `indices` - The indices of the data
/// * `dim` - Dimensions of the data
///
/// ### Return
///
/// Furthest data point from the pivot
fn find_furthest_from<T>(pivot: &[T], data: &[T], indices: &[u32], dim: usize) -> u32
where
    T: Float + SimdDistance,
{
    let mut max_dist = T::zero();
    let mut furthest = indices[0];

    for &idx in indices {
        let i = idx as usize;
        let vec = &data[i * dim..(i + 1) * dim];
        let dist = euclidean_distance_static(pivot, vec);
        if dist > max_dist {
            max_dist = dist;
            furthest = idx;
        }
    }

    furthest
}

/// Sample a second pivot with probability proportional to its distance from
/// the first, i.e. k-means++ seeding for two centres.
///
/// ### Params
///
/// * `pivot` - First pivot
/// * `data` - Slice of the data
/// * `indices` - The indices of the data
/// * `dim` - Dimensions of the data
/// * `metric` - The distance metric
/// * `rng` - Random number generator
///
/// ### Return
///
/// The sampled data point; falls back to `indices[0]` if every distance is zero
fn sample_by_distance<T>(
    pivot: &[T],
    data: &[T],
    indices: &[u32],
    dim: usize,
    metric: &Dist,
    rng: &mut StdRng,
) -> u32
where
    T: Float + SimdDistance,
{
    let pivot_norm = T::calculate_l2_norm(pivot);
    let weights: Vec<f64> = indices
        .iter()
        .map(|&idx| {
            let i = idx as usize;
            let vec = &data[i * dim..(i + 1) * dim];
            let d = match metric {
                Dist::SquaredEuclidean => euclidean_distance_static(pivot, vec),
                Dist::Cosine => {
                    cosine_distance_static_norm(pivot, vec, &pivot_norm, &T::calculate_l2_norm(vec))
                }
                Dist::Manhattan => unreachable!(),
            };
            d.to_f64().unwrap_or(0.0).max(0.0)
        })
        .collect();

    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return indices[0];
    }

    let mut target = rng.random::<f64>() * total;
    for (i, &w) in weights.iter().enumerate() {
        target -= w;
        if target <= 0.0 {
            return indices[i];
        }
    }
    indices[indices.len() - 1]
}

/// Partion data into two sets via the pivots
///
/// ### Params
///
/// * `data` - Slice of the data
/// * `indices` - The indices of the data
/// * `dim` - Dimensions of the data
/// * `pivot_1` - The left pivot point
/// * `pivot_2` - The right pivot point
/// * `metric` - The distance metric
/// * `left` - Cleared, then filled with the indices nearer `pivot_1` (ties
///   included), in input order
/// * `right` - Cleared, then filled with the rest, in input order
#[allow(clippy::too_many_arguments)]
fn partition_by_nearest<T>(
    data: &[T],
    indices: &[u32],
    dim: usize,
    pivot_1: &[T],
    pivot_2: &[T],
    metric: &Dist,
    left: &mut Vec<u32>,
    right: &mut Vec<u32>,
) where
    T: Float + SimdDistance,
{
    left.clear();
    right.clear();

    let norm_pivot_1 = T::calculate_l2_norm(pivot_1);
    let norm_pivot_2 = T::calculate_l2_norm(pivot_2);

    for &idx in indices {
        let i = idx as usize;
        let vec = &data[i * dim..(i + 1) * dim];
        let vec_norm = match metric {
            Dist::SquaredEuclidean => T::zero(),
            Dist::Cosine => T::calculate_l2_norm(vec),
            Dist::Manhattan => unreachable!(),
        };

        let d1 = match metric {
            Dist::SquaredEuclidean => euclidean_distance_static(vec, pivot_1),
            Dist::Cosine => cosine_distance_static_norm(vec, pivot_1, &vec_norm, &norm_pivot_1),
            Dist::Manhattan => unreachable!(),
        };
        let d2 = match metric {
            Dist::SquaredEuclidean => euclidean_distance_static(vec, pivot_2),
            Dist::Cosine => cosine_distance_static_norm(vec, pivot_2, &vec_norm, &norm_pivot_2),
            Dist::Manhattan => unreachable!(),
        };

        if d1 <= d2 {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }
}

/// Compute the centroid
///
/// ### Params
///
/// * `data` - The data vector
/// * `indices` - The index vector
/// * `dim` - The dimensionality of the vector
///
/// ### Returns
///
/// The centroid of the data
fn compute_centroid<T>(data: &[T], indices: &[u32], dim: usize) -> Vec<T>
where
    T: Float,
{
    let mut centroid = vec![T::zero(); dim];
    let n = T::from(indices.len()).unwrap();

    for &idx in indices {
        let i = idx as usize;
        let vec = &data[i * dim..(i + 1) * dim];
        for d in 0..dim {
            centroid[d] = centroid[d] + vec[d];
        }
    }

    for d in 0..dim {
        centroid[d] = centroid[d] / n;
    }

    centroid
}

/// Calculate the radius of the ball
///
/// ### Params
///
/// * `center` - The center of the given ball
/// * `data` - Slice of the data
/// * `indices` - The indices of the data
/// * `dim` - Dimensions of the data
/// * `metric` - The distance metric
///
/// ### Return
///
/// The maximum distance of the data against the center
fn ball_radius<T>(center: &[T], data: &[T], indices: &[u32], dim: usize, metric: &Dist) -> T
where
    T: Float + SimdDistance,
{
    let mut max_dist = T::zero();
    let center_norm = T::calculate_l2_norm(center);

    for &idx in indices {
        let i = idx as usize;
        let vec = &data[i * dim..(i + 1) * dim];
        let dist = match metric {
            Dist::SquaredEuclidean => euclidean_distance_static(center, vec),
            Dist::Cosine => {
                let vec_norm = T::calculate_l2_norm(vec);
                cosine_distance_static_norm(center, vec, &center_norm, &vec_norm)
            }
            Dist::Manhattan => unreachable!(),
        };
        if dist > max_dist {
            max_dist = dist;
        }
    }

    max_dist
}

////////////////
// Main types //
////////////////

/// Node representation in the flattened ball tree structure
///
/// Uses tagged union pattern: `n_descendants == 1` indicates leaf node,
/// otherwise it's a split node.
///
/// ### Fields
///
/// * `n_descendants` - 1 for leaf, 2 for split node
/// * `child_a` - For split: left child index; For leaf: start row of the leaf
///   (rows are stored in leaf order)
/// * `child_b` - For split: right child index; For leaf: count of items
/// * `center_idx` - Index into split_data (only used for split nodes)
#[derive(Clone, Copy, Debug)]
#[repr(C)]
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
struct BallNode<T> {
    n_descendants: u32,
    child_a: u32,
    child_b: u32,
    center_idx: u32,
    radius: T,
}

/// Build-time node representation
///
/// Temporary structure used during tree construction, later flattened into
/// BallNode format for better cache performance during queries.
#[derive(Clone)]
enum BuildNode<T> {
    Split {
        /// Centroid of the ball
        center: Vec<T>,
        /// Radius of the ball
        radius: T,
        /// Index of left child in build tree
        left: usize,
        /// Index of right child in build tree
        right: usize,
    },
    Leaf {
        /// Start of this leaf's range in the build permutation
        start: usize,
        /// Number of points in this leaf
        len: usize,
    },
}

///////////////////
// BallTreeIndex //
///////////////////

/// BallTreeIndex
///
/// Structure that contains the BallTree index.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct BallTreeIndex<T> {
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
    nodes: Vec<BallNode<T>>,
    /// Starting index of the root
    root: u32,
    /// Data of the centers
    centers_data: Vec<T>,
    /// Norms of the centers
    centers_data_norm: Vec<T>,
    /// Original index of each stored row; rows are kept in leaf order
    original_ids: Vec<u32>,
}

////////////////////
// VectorDistance //
////////////////////

/// VectorDistance implementation
impl<T> VectorDistance<T> for BallTreeIndex<T>
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

impl<T> DimensionValidation for BallTreeIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////////
// Index build //
/////////////////

impl<T> BallTreeIndex<T>
where
    T: AnnSearchFloat,
{
    /// Generate a new BallTreeIndex
    ///
    /// Builds a BallTree
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (rows = samples, columns = dimensions)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `seed` - Random seed for reproducibility
    /// * `verbosity` - Prints start and end at `Normal`
    ///
    /// ### Returns
    ///
    /// Index ready for querying
    pub fn new(
        data: impl AnnMatrix<T>,
        metric: Dist,
        seed: usize,
        verbosity: Verbosity,
    ) -> Result<Self, AnnSearchErrors> {
        let log = BuildLog::new("BallTree index", verbosity);
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

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

        let max_parallel_depth = (rayon::current_num_threads() as f32).log2().ceil() as usize;
        let mut rng = StdRng::seed_from_u64(seed as u64);

        let mut perm: Vec<u32> = (0..n as u32).collect();
        let tree = Self::build_tree_parallel(
            &vectors_flat,
            dim,
            &mut perm,
            &mut rng,
            metric,
            max_parallel_depth,
        )?;

        let mut nodes = Vec::new();
        let mut centers_data = Vec::new();

        let root = Self::flatten_tree(
            tree,
            &mut nodes,
            &mut centers_data,
            &perm,
            &vectors_flat,
            &metric,
            dim,
        );

        // Store the vectors in leaf order so every leaf scan reads one
        // contiguous block. A single tree holds each item exactly once, so the
        // permutation is itself the new-to-old map.
        let mut reordered = Vec::with_capacity(vectors_flat.len());
        for &old in &perm {
            let old = old as usize;
            reordered.extend_from_slice(&vectors_flat[old * dim..(old + 1) * dim]);
        }
        let vectors_flat = reordered;
        let norms: Vec<T> = if norms.is_empty() {
            norms
        } else {
            perm.iter().map(|&old| norms[old as usize]).collect()
        };
        let original_ids = perm;

        let centers_data_norm = if metric == Dist::Cosine {
            (0..centers_data.len() / dim)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
                    T::calculate_l2_norm(&centers_data[start..end])
                })
                .collect()
        } else {
            Vec::new()
        };

        log.finish();

        Ok(BallTreeIndex {
            nodes,
            root,
            centers_data,
            centers_data_norm,
            vectors_flat,
            dim,
            n,
            norms,
            metric,
            original_ids,
        })
    }

    /// Build a local node
    ///
    /// ### Params
    ///
    /// * `vector_flat` - Flat data representation
    /// * `dim` - Dimensionality of the data
    /// * `perm` - Identity permutation of the points, partitioned in place into
    ///   leaf order
    /// * `rng` - Random number generator
    /// * `metric` - The distance metric
    /// * `max_parallel_depth` - Maximum depth until which to execute parallel
    ///   threading
    ///
    /// ### Returns
    ///
    /// The vec of `BuildNode`'s
    fn build_tree_parallel(
        vectors_flat: &[T],
        dim: usize,
        perm: &mut [u32],
        rng: &mut StdRng,
        metric: Dist,
        max_parallel_depth: usize,
    ) -> Result<Vec<BuildNode<T>>, AnnSearchErrors> {
        let mut nodes = Vec::new();
        Self::build_node_local(
            vectors_flat,
            dim,
            perm,
            0,
            &mut nodes,
            rng,
            metric,
            0,
            max_parallel_depth,
        )?;
        Ok(nodes)
    }

    /// Build a local node
    ///
    /// ### Params
    ///
    /// * `vector_flat` - Flat data representation
    /// * `dim` - Dimensionality of the data
    /// * `items` - This node's slice of the build permutation; reordered so the
    ///   left child's points come first
    /// * `base` - Offset of `items` within the whole permutation
    /// * `rng` - Random number generator
    /// * `metric` - The distance metric
    /// * `depth` - The current depth
    /// * `max_parallel_depth` - Maximum depth until which to execute parallel
    ///   threading
    #[allow(clippy::too_many_arguments)]
    fn build_node_local(
        vectors_flat: &[T],
        dim: usize,
        items: &mut [u32],
        base: usize,
        nodes: &mut Vec<BuildNode<T>>,
        rng: &mut StdRng,
        metric: Dist,
        depth: usize,
        max_parallel_depth: usize,
    ) -> Result<usize, AnnSearchErrors> {
        if items.len() <= LEAF_MIN_MEMBERS {
            let node_idx = nodes.len();
            nodes.push(BuildNode::Leaf {
                start: base,
                len: items.len(),
            });
            return Ok(node_idx);
        }

        let p1_idx = items[rng.random_range(0..items.len())] as usize;
        let p1 = &vectors_flat[p1_idx * dim..(p1_idx + 1) * dim];
        let p2_idx = if items.len() > BALL_SMALL_NODE {
            find_furthest_from(p1, vectors_flat, items, dim)
        } else {
            sample_by_distance(p1, vectors_flat, items, dim, &metric, rng)
        } as usize;
        let p2 = &vectors_flat[p2_idx * dim..(p2_idx + 1) * dim];

        let (mut left, mut right) = (Vec::new(), Vec::new());
        partition_by_nearest(
            vectors_flat,
            items,
            dim,
            p1,
            p2,
            &metric,
            &mut left,
            &mut right,
        );

        if left.is_empty() || right.is_empty() {
            let mid = items.len() / 2;
            left.clear();
            right.clear();
            left.extend_from_slice(&items[..mid]);
            right.extend_from_slice(&items[mid..]);
        }

        let iters = if items.len() > BALL_SMALL_NODE {
            1
        } else {
            BALL_SPLIT_ITERS
        };
        let (mut l, mut r) = (Vec::new(), Vec::new());
        for _ in 0..iters {
            let c1 = compute_centroid(vectors_flat, &left, dim);
            let c2 = compute_centroid(vectors_flat, &right, dim);
            partition_by_nearest(vectors_flat, items, dim, &c1, &c2, &metric, &mut l, &mut r);
            if l.is_empty() || r.is_empty() {
                break;
            }
            std::mem::swap(&mut left, &mut l);
            std::mem::swap(&mut right, &mut r);
        }

        // Centroid and radius sum in the pre-partition order, so they come
        // before the permutation is rewritten.
        let center = compute_centroid(vectors_flat, items, dim);

        let radius = ball_radius(&center, vectors_flat, items, dim, &metric);

        let n_left = left.len();
        items[..n_left].copy_from_slice(&left);
        items[n_left..].copy_from_slice(&right);
        drop((left, right, l, r));
        let (left_items, right_items) = items.split_at_mut(n_left);

        let node_idx = nodes.len();
        nodes.push(BuildNode::Split {
            center: center.clone(),
            radius,
            left: 0,
            right: 0,
        });

        // Child streams are drawn at every node, not only above the parallel
        // cutoff, so the tree does not depend on the thread count.
        let seed_left: u64 = rng.random();
        let seed_right: u64 = rng.random();

        let (left_idx, right_idx) = if depth < max_parallel_depth {
            let (left_result, right_result) = rayon::join(
                || {
                    let mut left_rng = rand::rngs::StdRng::seed_from_u64(seed_left);
                    Self::build_subtree(
                        vectors_flat,
                        dim,
                        left_items,
                        base,
                        &mut left_rng,
                        metric,
                        depth + 1,
                        max_parallel_depth,
                    )
                },
                || {
                    let mut right_rng = rand::rngs::StdRng::seed_from_u64(seed_right);
                    Self::build_subtree(
                        vectors_flat,
                        dim,
                        right_items,
                        base + n_left,
                        &mut right_rng,
                        metric,
                        depth + 1,
                        max_parallel_depth,
                    )
                },
            );

            let mut left_tree = left_result?;
            let mut right_tree = right_result?;

            let left_offset = nodes.len();
            Self::adjust_subtree_indices(&mut left_tree, left_offset);
            nodes.extend(left_tree);

            let right_offset = nodes.len();
            Self::adjust_subtree_indices(&mut right_tree, right_offset);
            nodes.extend(right_tree);

            (left_offset, right_offset)
        } else {
            let left_idx = Self::build_node_local(
                vectors_flat,
                dim,
                left_items,
                base,
                nodes,
                &mut StdRng::seed_from_u64(seed_left),
                metric,
                depth + 1,
                max_parallel_depth,
            )?;
            let right_idx = Self::build_node_local(
                vectors_flat,
                dim,
                right_items,
                base + n_left,
                nodes,
                &mut StdRng::seed_from_u64(seed_right),
                metric,
                depth + 1,
                max_parallel_depth,
            )?;
            (left_idx, right_idx)
        };

        if let BuildNode::Split {
            ref mut left,
            ref mut right,
            ..
        } = nodes[node_idx]
        {
            *left = left_idx;
            *right = right_idx;
        }

        Ok(node_idx)
    }

    /// Helper function to recursively build the sub trees
    ///
    /// ### Params
    ///
    /// * `vector_flat` - Flat data representation
    /// * `dim` - Dimensionality of the data
    /// * `items` - This subtree's slice of the build permutation
    /// * `base` - Offset of `items` within the whole permutation
    /// * `rng` - Random number generator
    /// * `metric` - The distance metric
    /// * `depth` - The current depth
    /// * `max_parallel_depth` - Maximum depth until which to execute parallel
    ///   threading
    #[allow(clippy::too_many_arguments)]
    fn build_subtree(
        vectors_flat: &[T],
        dim: usize,
        items: &mut [u32],
        base: usize,
        rng: &mut StdRng,
        metric: Dist,
        depth: usize,
        max_parallel_depth: usize,
    ) -> Result<Vec<BuildNode<T>>, AnnSearchErrors> {
        let mut nodes = Vec::new();
        Self::build_node_local(
            vectors_flat,
            dim,
            items,
            base,
            &mut nodes,
            rng,
            metric,
            depth,
            max_parallel_depth,
        )?;
        Ok(nodes)
    }

    /// Flatten the tree structures
    ///
    /// ### Params
    ///
    /// * `tree` - The vector of the BuildNodes
    /// * `nodes` - Mutable vector of BallNodes. It will be updated during the
    ///   call.
    /// * `centers_data` - Mutable vector of the centroids of each node in the
    ///   tree.
    /// * `radii_data` - Mutable vector of the radii of each node in the tree.
    /// * `perm` - The build permutation; leaves are ranges into it.
    #[allow(clippy::too_many_arguments)]
    fn flatten_tree(
        tree: Vec<BuildNode<T>>,
        nodes: &mut Vec<BallNode<T>>,
        centers_data: &mut Vec<T>,
        perm: &[u32],
        vectors_flat: &[T],
        metric: &Dist,
        dim: usize,
    ) -> u32 {
        if tree.is_empty() {
            return 0;
        }

        let base_offset = nodes.len() as u32;

        for node in tree {
            match node {
                BuildNode::Split {
                    center,
                    radius,
                    left,
                    right,
                } => {
                    let center_idx = (centers_data.len() / dim) as u32;
                    centers_data.extend(center);

                    nodes.push(BallNode {
                        n_descendants: 2,
                        child_a: base_offset + left as u32,
                        child_b: base_offset + right as u32,
                        center_idx,
                        radius,
                    });
                }
                BuildNode::Leaf { start, len } => {
                    let items = &perm[start..start + len];
                    let center = compute_centroid(vectors_flat, items, dim);
                    let radius = ball_radius(&center, vectors_flat, items, dim, metric);
                    let center_idx = (centers_data.len() / dim) as u32;
                    centers_data.extend(center);

                    nodes.push(BallNode {
                        n_descendants: 1,
                        child_a: start as u32,
                        child_b: len as u32,
                        center_idx,
                        radius,
                    });
                }
            }
        }

        base_offset
    }

    /// Offset the left/right child indices of each build node
    ///
    /// ### Params
    ///
    /// * `nodes` - Subtree nodes to adjust in place
    /// * `offset` - Amount to add to each child index
    fn adjust_subtree_indices(nodes: &mut [BuildNode<T>], offset: usize) {
        for node in nodes.iter_mut() {
            if let BuildNode::Split { left, right, .. } = node {
                *left += offset;
                *right += offset;
            }
        }
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
            + self.nodes.capacity() * std::mem::size_of::<BallNode<T>>()
            + self.centers_data.capacity() * std::mem::size_of::<T>()
            + self.original_ids.capacity() * std::mem::size_of::<u32>()
    }
}

///////////
// Query //
///////////

impl<T> BallTreeIndex<T>
where
    T: AnnSearchFloat,
{
    /// Query the index
    ///
    /// ### Params
    ///
    /// * `query_vec` - The vector to query aginst
    /// * `k` - Number of neighbours to return.
    /// * `search_k` - The budget. How many nodes to maximally visit per given
    ///   query.
    ///
    /// ### Returns
    ///
    /// `(indices, dist)`
    #[inline]
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        search_k: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        let limit = search_k.unwrap_or((self.n as f32 * 0.05) as usize);
        let mut visited_count = 0;

        let mut visited = VisitedSet::new(self.n);

        let query_norm = if self.metric == Dist::Cosine {
            T::calculate_l2_norm(query_vec)
        } else {
            T::one()
        };

        let mut top_k = SortedBuffer::with_capacity(k + 1);
        let mut kth_dist = T::infinity();
        let mut pq = BinaryHeap::with_capacity(64);

        pq.push(BacktrackEntry {
            margin: 0.0,
            node_idx: self.root,
        });

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

                    for item in start..start + len {
                        if visited.mark(item) {
                            continue;
                        }

                        let vec_start = item * self.dim;
                        let vec = unsafe {
                            self.vectors_flat
                                .get_unchecked(vec_start..vec_start + self.dim)
                        };

                        let dist = match self.metric {
                            Dist::SquaredEuclidean => T::euclidean_simd(query_vec, vec),
                            Dist::Cosine => {
                                let norm = unsafe { *self.norms.get_unchecked(item) };
                                cosine_from_dot(T::dot_simd(query_vec, vec), query_norm * norm)
                            }
                            Dist::Manhattan => unreachable!(),
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
                    let left_node = unsafe { self.nodes.get_unchecked(node.child_a as usize) };
                    let right_node = unsafe { self.nodes.get_unchecked(node.child_b as usize) };

                    let left_center_offset = left_node.center_idx as usize * self.dim;
                    let left_center = unsafe {
                        self.centers_data
                            .get_unchecked(left_center_offset..left_center_offset + self.dim)
                    };

                    let dist_to_left = match self.metric {
                        Dist::SquaredEuclidean => T::euclidean_simd(query_vec, left_center),
                        Dist::Cosine => {
                            let left_norm = unsafe {
                                *self
                                    .centers_data_norm
                                    .get_unchecked(left_node.center_idx as usize)
                            };
                            cosine_from_dot(
                                T::dot_simd(query_vec, left_center),
                                query_norm * left_norm,
                            )
                        }
                        Dist::Manhattan => unreachable!(),
                    };

                    let right_center_offset = right_node.center_idx as usize * self.dim;
                    let right_center = unsafe {
                        self.centers_data
                            .get_unchecked(right_center_offset..right_center_offset + self.dim)
                    };

                    let dist_to_right = match self.metric {
                        Dist::SquaredEuclidean => T::euclidean_simd(query_vec, right_center),
                        Dist::Cosine => {
                            let right_norm = unsafe {
                                *self
                                    .centers_data_norm
                                    .get_unchecked(right_node.center_idx as usize)
                            };
                            cosine_from_dot(
                                T::dot_simd(query_vec, right_center),
                                query_norm * right_norm,
                            )
                        }
                        Dist::Manhattan => unreachable!(),
                    };

                    let (closer, farther, farther_dist, farther_radius) =
                        if dist_to_left <= dist_to_right {
                            (node.child_a, node.child_b, dist_to_right, right_node.radius)
                        } else {
                            (node.child_b, node.child_a, dist_to_left, left_node.radius)
                        };

                    // Distances and radii are squared (Euclidean) or
                    // half-squared chords (Cosine), so the triangle inequality
                    // holds on their square roots: the ball can hold nothing
                    // nearer than (sqrt(d) - sqrt(r))^2.
                    let gap = (farther_dist.max(T::zero()).sqrt()
                        - farther_radius.max(T::zero()).sqrt())
                    .max(T::zero());
                    pq.push(BacktrackEntry {
                        margin: -(gap * gap).to_f64().unwrap(),
                        node_idx: farther,
                    });

                    current_idx = closer;
                }
            }
        }

        let results: Vec<(usize, T)> = top_k
            .data()
            .iter()
            .map(|(OrderedFloat(dist), idx)| (self.original_ids[*idx] as usize, *dist))
            .collect();

        Ok(results.into_iter().unzip())
    }

    /// Query the index with row references
    ///
    /// Uses an optimised (unsafe) path if possible; if not, creates deep copy
    ///
    /// ### Params
    ///
    /// * `query_row` - The row to query.
    /// * `k` - Number of neighbours to return.
    /// * `search_k` - The budget. How many nodes to maximally visit per given
    ///   query.
    ///
    /// ### Returns
    ///
    /// `(indices, dist)`
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

    /// Generate the kNN graph from self
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours to return.
    /// * `search_k` - The budget. How many nodes to maximally visit per given
    ///   query.
    /// * `return_dist` - Return the distances.
    /// * `verbosity` - Prints start, progress and end at `Normal`
    ///
    /// ### Returns
    ///
    /// `(Vec<indices>, Vec<optional dist>)`
    pub fn generate_knn(
        &self,
        k: usize,
        search_k: Option<usize>,
        return_dist: bool,
        verbosity: Verbosity,
    ) -> KnnOptionResult<T> {
        let progress = QueryProgress::new("BallTree index", self.n, true, verbosity);

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dim;
                let end = start + self.dim;
                let vec = &self.vectors_flat[start..end];

                progress.tick();

                self.query(vec, k, search_k)
            })
            .collect::<Result<Vec<_>, AnnSearchErrors>>()?;
        progress.finish();

        // Rows come out in storage order; put them back in original order.
        let mut ordered = vec![(Vec::new(), Vec::new()); self.n];
        for (internal, res) in results.into_iter().enumerate() {
            ordered[self.original_ids[internal] as usize] = res;
        }

        Ok(pack_knn_results(ordered, return_dist))
    }
}

///////////////////
// KnnValidation //
///////////////////

/// KnnValidation trait implementation for the BallTreeIndex
impl<T> KnnValidation<T> for BallTreeIndex<T>
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

    fn original_ids(&self) -> &[u32] {
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
impl<T> IndexIo for BallTreeIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "ball_tree";
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
    fn test_ball_tree_index_creation() {
        let mat = create_simple_matrix();
        let _ = BallTreeIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 42, Verbosity::Quiet);
    }

    #[test]
    fn test_ball_tree_query_finds_self() {
        let mat = create_simple_matrix();
        let index =
            BallTreeIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 42, Verbosity::Quiet).unwrap();

        // Query with point 0, should find itself first
        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1, Some(5)).unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ball_tree_query_euclidean() {
        let mat = create_simple_matrix();
        let index =
            BallTreeIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 42, Verbosity::Quiet).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, Some(5)).unwrap();

        // Should find point 0 first (exact match)
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Results should be sorted by distance
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_ball_tree_query_cosine() {
        use crate::prelude::*;

        let mat = create_simple_matrix();
        let index = BallTreeIndex::new(mat.as_ref(), Dist::Cosine, 42, Verbosity::Quiet).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, Some(5)).unwrap();

        // Should find point 0 first (identical direction)
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ball_tree_query_k_larger_than_dataset() {
        use crate::prelude::*;

        let mat = create_simple_matrix();
        let index =
            BallTreeIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 42, Verbosity::Quiet).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        // ask for 10 neighbours but only 5 points exist
        let (indices, _) = index.query(&query, 10, Some(5)).unwrap();

        // Should return at most 5 results
        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_ball_tree_query_search_k() {
        use crate::prelude::*;

        let mat = create_simple_matrix();
        let index =
            BallTreeIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 42, Verbosity::Quiet).unwrap();

        let query = vec![1.0, 0.0, 0.0];

        let (indices1, _) = index.query(&query, 3, Some(10)).unwrap();
        let (indices2, _) = index.query(&query, 3, Some(1)).unwrap();

        assert_eq!(indices1.len(), 3);
        assert!(!indices2.is_empty());
    }

    #[test]
    fn test_ball_tree_query_row() {
        let mat = create_simple_matrix();
        let index =
            BallTreeIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 42, Verbosity::Quiet).unwrap();

        // Query using a row from the matrix
        let (indices, distances) = index.query_row(mat.row(0), 1, Some(5)).unwrap();

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ball_tree_reproducibility() {
        let mat = create_simple_matrix();

        let index1 =
            BallTreeIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 42, Verbosity::Quiet).unwrap();
        let index2 =
            BallTreeIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 42, Verbosity::Quiet).unwrap();

        let query = vec![0.5, 0.5, 0.0];
        let (indices1, _) = index1.query(&query, 3, Some(5)).unwrap();
        let (indices2, _) = index2.query(&query, 3, Some(5)).unwrap();

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_ball_tree_different_seeds() {
        let mat = create_simple_matrix();

        let index1 =
            BallTreeIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 42, Verbosity::Quiet).unwrap();
        let index2 =
            BallTreeIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 123, Verbosity::Quiet)
                .unwrap();

        let query = vec![0.5, 0.5, 0.0];

        let (indices1, _) = index1.query(&query, 3, Some(5)).unwrap();
        let (indices2, _) = index2.query(&query, 3, Some(5)).unwrap();

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_ball_tree_larger_dataset() {
        let n = 100;
        let dim = 10;
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 10.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index =
            BallTreeIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 42, Verbosity::Quiet).unwrap();

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5, None).unwrap();

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_ball_tree_orthogonal_vectors() {
        use crate::prelude::*;
        let data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);
        let index = BallTreeIndex::new(mat.as_ref(), Dist::Cosine, 42, Verbosity::Quiet).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, Some(3)).unwrap();

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(distances[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(distances[2], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ball_tree_parallel_build() {
        let n = 50;
        let dim = 5;
        let data: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        let index =
            BallTreeIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 42, Verbosity::Quiet).unwrap();

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 3, None).unwrap(); // 5% of 50 is 2.5 -> 2. Safe-ish.

        assert_eq!(indices.len(), 3);
    }
}
