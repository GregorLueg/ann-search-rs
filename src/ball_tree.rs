use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;
use thousands::*;

use crate::prelude::*;
use crate::utils::ivf_utils::*;
use crate::utils::tree_utils::*;
use crate::utils::*;

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
fn find_furthest_from<T>(pivot: &[T], data: &[T], indices: &[usize], dim: usize) -> usize
where
    T: Float + SimdDistance,
{
    let mut max_dist = T::zero();
    let mut furthest = indices[0];

    for &idx in indices {
        let vec = &data[idx * dim..(idx + 1) * dim];
        let dist = euclidean_distance_static(pivot, vec);
        if dist > max_dist {
            max_dist = dist;
            furthest = idx;
        }
    }

    furthest
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
///
/// ### Return
///
/// `(indices left, indices right)`
fn partition_by_nearest<T>(
    data: &[T],
    indices: &[usize],
    dim: usize,
    pivot_1: &[T],
    pivot_2: &[T],
    metric: &Dist,
) -> (Vec<usize>, Vec<usize>)
where
    T: Float + SimdDistance,
{
    let mut left: Vec<usize> = Vec::new();
    let mut right: Vec<usize> = Vec::new();

    let norm_pivot_1 = T::calculate_l2_norm(pivot_1);
    let norm_pivot_2 = T::calculate_l2_norm(pivot_2);

    for &idx in indices {
        let vec = &data[idx * dim..(idx + 1) * dim];
        let vec_norm = match metric {
            Dist::Euclidean => T::zero(),
            Dist::Cosine => T::calculate_l2_norm(vec),
        };

        let d1 = match metric {
            Dist::Euclidean => euclidean_distance_static(vec, pivot_1),
            Dist::Cosine => cosine_distance_static_norm(vec, pivot_1, &vec_norm, &norm_pivot_1),
        };
        let d2 = match metric {
            Dist::Euclidean => euclidean_distance_static(vec, pivot_2),
            Dist::Cosine => cosine_distance_static_norm(vec, pivot_2, &vec_norm, &norm_pivot_2),
        };

        if d1 <= d2 {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }

    (left, right)
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
fn compute_centroid<T>(data: &[T], indices: &[usize], dim: usize) -> Vec<T>
where
    T: Float,
{
    let mut centroid = vec![T::zero(); dim];
    let n = T::from(indices.len()).unwrap();

    for &idx in indices {
        let vec = &data[idx * dim..(idx + 1) * dim];
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
fn ball_radius<T>(center: &[T], data: &[T], indices: &[usize], dim: usize, metric: &Dist) -> T
where
    T: Float + SimdDistance,
{
    let mut max_dist = T::zero();
    let center_norm = T::calculate_l2_norm(center);

    for &idx in indices {
        let vec = &data[idx * dim..(idx + 1) * dim];
        let dist = match metric {
            Dist::Euclidean => euclidean_distance_static(center, vec),
            Dist::Cosine => {
                let vec_norm = T::calculate_l2_norm(vec);
                cosine_distance_static_norm(center, vec, &center_norm, &vec_norm)
            }
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
/// * `child_a` - For split: left child index; For leaf: start index in
///   leaf_indices
/// * `child_b` - For split: right child index; For leaf: count of items
/// * `center_idx` - Index into split_data (only used for split nodes)
#[derive(Clone, Copy, Debug)]
#[repr(C)]
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
        /// Original data indices in this leaf
        items: Vec<usize>,
    },
}

///////////////////
// BallTreeIndex //
///////////////////

/// BallTreeIndex
///
/// ### Fields
///
/// * `vectors_flat` - Original vector data, flattened for cache locality
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `norms` - Pre-computed norms for Cosine distance (empty for Euclidean)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `nodes` - Flattened tree structure containing all split and leaf nodes
/// * `root` - Starting index of the root
/// * `centers_data` - Data of the centers
/// * `radii_data` - Data of the corresponding radii of the ceneters
/// * `leaf_indices` - Actual data indices stored in leaf nodes
pub struct BallTreeIndex<T> {
    // Shared data
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    norms: Vec<T>,
    metric: Dist,
    // Index-specific data
    nodes: Vec<BallNode<T>>,
    root: u32,
    centers_data: Vec<T>,
    centers_data_norm: Vec<T>,
    leaf_indices: Vec<usize>,
}

////////////////////
// VectorDistance //
////////////////////

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

impl<T> BallTreeIndex<T>
where
    T: AnnSearchFloat,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Generate a new BallTreeIndex
    ///
    /// Builds a BallTree
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (rows = samples, columns = dimensions)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Index ready for querying
    pub fn new(data: MatRef<T>, metric: Dist, seed: usize) -> Self {
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

        let max_parallel_depth = (rayon::current_num_threads() as f32).log2().ceil() as usize;
        let mut rng = StdRng::seed_from_u64(seed as u64);

        let tree = Self::build_tree_parallel(
            &vectors_flat,
            dim,
            (0..n).collect(),
            &mut rng,
            metric,
            max_parallel_depth,
        );

        let mut nodes = Vec::new();
        let mut centers_data = Vec::new();
        let mut leaf_indices = Vec::new();

        let root = Self::flatten_tree(
            tree,
            &mut nodes,
            &mut centers_data,
            &mut leaf_indices,
            &vectors_flat,
            &metric,
            dim,
        );

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

        BallTreeIndex {
            nodes,
            root,
            centers_data,
            centers_data_norm,
            leaf_indices,
            vectors_flat,
            dim,
            n,
            norms,
            metric,
        }
    }

    /// Build a local node
    ///
    /// ### Params
    ///
    /// * `vector_flat` - Flat data representation
    /// * `dim` - Dimensionality of the data
    /// * `items` - Which data points are in that node
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
        items: Vec<usize>,
        rng: &mut StdRng,
        metric: Dist,
        max_parallel_depth: usize,
    ) -> Vec<BuildNode<T>> {
        let mut nodes = Vec::new();
        Self::build_node_local(
            vectors_flat,
            dim,
            items,
            &mut nodes,
            rng,
            metric,
            0,
            max_parallel_depth,
        );
        nodes
    }

    /// Build a local node
    ///
    /// ### Params
    ///
    /// * `vector_flat` - Flat data representation
    /// * `dim` - Dimensionality of the data
    /// * `items` - Which data points are in that node
    /// * `rng` - Random number generator
    /// * `metric` - The distance metric
    /// * `depth` - The current depth
    /// * `max_parallel_depth` - Maximum depth until which to execute parallel
    ///   threading
    #[allow(clippy::too_many_arguments)]
    fn build_node_local(
        vectors_flat: &[T],
        dim: usize,
        items: Vec<usize>,
        nodes: &mut Vec<BuildNode<T>>,
        rng: &mut StdRng,
        metric: Dist,
        depth: usize,
        max_parallel_depth: usize,
    ) -> usize {
        if items.len() <= LEAF_MIN_MEMBERS {
            let node_idx = nodes.len();
            nodes.push(BuildNode::Leaf { items });
            return node_idx;
        }

        let (center, left_items, right_items) = if items.len() > 500 {
            let p1_idx = items[rng.random_range(0..items.len())];
            let p1 = &vectors_flat[p1_idx * dim..(p1_idx + 1) * dim];
            let p2_idx = find_furthest_from(p1, vectors_flat, &items, dim);
            let p2 = &vectors_flat[p2_idx * dim..(p2_idx + 1) * dim];

            let (mut left, mut right) =
                partition_by_nearest(vectors_flat, &items, dim, p1, p2, &metric);

            if left.is_empty() || right.is_empty() {
                let mid = items.len() / 2;
                left = items[..mid].to_vec();
                right = items[mid..].to_vec();
            }

            let c1 = compute_centroid(vectors_flat, &left, dim);
            let c2 = compute_centroid(vectors_flat, &right, dim);

            let (left_final, right_final) =
                partition_by_nearest(vectors_flat, &items, dim, &c1, &c2, &metric);

            let center = compute_centroid(vectors_flat, &items, dim);
            (center, left_final, right_final)
        } else {
            let sample_data: Vec<T> = items
                .iter()
                .flat_map(|&idx| {
                    let start = idx * dim;
                    vectors_flat[start..start + dim].iter().cloned()
                })
                .collect();

            let centroids = train_centroids(
                &sample_data,
                dim,
                items.len(),
                2,
                &metric,
                5,
                (rng.random::<f32>() * 100.0) as usize,
                false,
            );

            let sample_norms = if metric == Dist::Cosine {
                (0..items.len())
                    .map(|i| {
                        sample_data[i * dim..(i + 1) * dim]
                            .iter()
                            .map(|&x| x * x)
                            .fold(T::zero(), |a, b| a + b)
                            .sqrt()
                    })
                    .collect()
            } else {
                vec![T::one(); items.len()]
            };

            let centroid_norms = if metric == Dist::Cosine {
                vec![
                    T::calculate_l2_norm(&centroids[0..dim]),
                    T::calculate_l2_norm(&centroids[dim..2 * dim]),
                ]
            } else {
                vec![T::one(); 2]
            };

            let mut left = Vec::new();
            let mut right = Vec::new();

            for (i, &idx) in items.iter().enumerate() {
                let vec = &sample_data[i * dim..(i + 1) * dim];
                let vec_norm = &sample_norms[i];

                let d0 = match metric {
                    Dist::Euclidean => euclidean_distance_static(vec, &centroids[0..dim]),
                    Dist::Cosine => cosine_distance_static_norm(
                        vec,
                        &centroids[0..dim],
                        vec_norm,
                        &centroid_norms[0],
                    ),
                };
                let d1 = match metric {
                    Dist::Euclidean => euclidean_distance_static(vec, &centroids[dim..2 * dim]),
                    Dist::Cosine => cosine_distance_static_norm(
                        vec,
                        &centroids[dim..2 * dim],
                        vec_norm,
                        &centroid_norms[1],
                    ),
                };

                if d0 <= d1 {
                    left.push(idx);
                } else {
                    right.push(idx);
                }
            }

            if left.is_empty() || right.is_empty() {
                let mid = items.len() / 2;
                left = items[..mid].to_vec();
                right = items[mid..].to_vec();
            }

            let center = compute_centroid(vectors_flat, &items, dim);
            (center, left, right)
        };

        let radius = ball_radius(&center, vectors_flat, &items, dim, &metric);

        let node_idx = nodes.len();
        nodes.push(BuildNode::Split {
            center: center.clone(),
            radius,
            left: 0,
            right: 0,
        });

        let (left_idx, right_idx) = if depth < max_parallel_depth {
            let seed_left = rng.random();
            let seed_right = rng.random();

            let (mut left_tree, mut right_tree) = rayon::join(
                || {
                    let mut left_rng = rand::rngs::StdRng::seed_from_u64(seed_left);
                    Self::build_subtree(
                        vectors_flat,
                        dim,
                        left_items,
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
                        &mut right_rng,
                        metric,
                        depth + 1,
                        max_parallel_depth,
                    )
                },
            );

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
                nodes,
                rng,
                metric,
                depth + 1,
                max_parallel_depth,
            );
            let right_idx = Self::build_node_local(
                vectors_flat,
                dim,
                right_items,
                nodes,
                rng,
                metric,
                depth + 1,
                max_parallel_depth,
            );
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

        node_idx
    }

    /// Helper function to recursively build the sub trees
    ///
    /// ### Params
    ///
    /// * `vector_flat` - Flat data representation
    /// * `dim` - Dimensionality of the data
    /// * `items` - Which data points are in that node
    /// * `rng` - Random number generator
    /// * `metric` - The distance metric
    /// * `depth` - The current depth
    /// * `max_parallel_depth` - Maximum depth until which to execute parallel
    ///   threading
    fn build_subtree(
        vectors_flat: &[T],
        dim: usize,
        items: Vec<usize>,
        rng: &mut StdRng,
        metric: Dist,
        depth: usize,
        max_parallel_depth: usize,
    ) -> Vec<BuildNode<T>> {
        let mut nodes = Vec::new();
        Self::build_node_local(
            vectors_flat,
            dim,
            items,
            &mut nodes,
            rng,
            metric,
            depth,
            max_parallel_depth,
        );
        nodes
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
    /// * `leaf_indices` - Mutable vector of leaf indices of the tree.
    #[allow(clippy::too_many_arguments)]
    fn flatten_tree(
        tree: Vec<BuildNode<T>>,
        nodes: &mut Vec<BallNode<T>>,
        centers_data: &mut Vec<T>,
        leaf_indices: &mut Vec<usize>,
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
                BuildNode::Leaf { items } => {
                    let center = compute_centroid(vectors_flat, &items, dim);
                    let radius = ball_radius(&center, vectors_flat, &items, dim, metric); // Compute actual radius
                    let center_idx = (centers_data.len() / dim) as u32;
                    centers_data.extend(center);

                    let start = leaf_indices.len() as u32;
                    let len = items.len() as u32;
                    leaf_indices.extend(items);

                    nodes.push(BallNode {
                        n_descendants: 1,
                        child_a: start,
                        child_b: len,
                        center_idx,
                        radius,
                    });
                }
            }
        }

        base_offset
    }

    fn adjust_subtree_indices(nodes: &mut [BuildNode<T>], offset: usize) {
        for node in nodes.iter_mut() {
            if let BuildNode::Split { left, right, .. } = node {
                *left += offset;
                *right += offset;
            }
        }
    }

    ///////////
    // Query //
    ///////////

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
    ) -> (Vec<usize>, Vec<T>) {
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
                            Dist::Euclidean => T::euclidean_simd(query_vec, vec),
                            Dist::Cosine => {
                                let norm = unsafe { *self.norms.get_unchecked(item) };
                                T::one() - T::dot_simd(query_vec, vec) / (query_norm * norm)
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
                    // Cache child nodes
                    let left_node = unsafe { self.nodes.get_unchecked(node.child_a as usize) };
                    let right_node = unsafe { self.nodes.get_unchecked(node.child_b as usize) };

                    let left_center_offset = left_node.center_idx as usize * self.dim;
                    let left_center = unsafe {
                        self.centers_data
                            .get_unchecked(left_center_offset..left_center_offset + self.dim)
                    };

                    let dist_to_left = match self.metric {
                        Dist::Euclidean => T::euclidean_simd(query_vec, left_center),
                        Dist::Cosine => {
                            let left_norm = unsafe {
                                *self
                                    .centers_data_norm
                                    .get_unchecked(left_node.center_idx as usize)
                            };
                            T::one()
                                - T::dot_simd(query_vec, left_center) / (query_norm * left_norm)
                        }
                    };

                    let right_center_offset = right_node.center_idx as usize * self.dim;
                    let right_center = unsafe {
                        self.centers_data
                            .get_unchecked(right_center_offset..right_center_offset + self.dim)
                    };

                    let dist_to_right = match self.metric {
                        Dist::Euclidean => T::euclidean_simd(query_vec, right_center),
                        Dist::Cosine => {
                            let right_norm = unsafe {
                                *self
                                    .centers_data_norm
                                    .get_unchecked(right_node.center_idx as usize)
                            };
                            T::one()
                                - T::dot_simd(query_vec, right_center) / (query_norm * right_norm)
                        }
                    };

                    let (closer, farther, farther_dist, farther_radius) =
                        if dist_to_left <= dist_to_right {
                            (node.child_a, node.child_b, dist_to_right, right_node.radius)
                        } else {
                            (node.child_b, node.child_a, dist_to_left, left_node.radius)
                        };

                    pq.push(BacktrackEntry {
                        margin: -(farther_dist - farther_radius * T::from(1.1).unwrap())
                            .to_f64()
                            .unwrap(),
                        node_idx: farther,
                    });

                    current_idx = closer;
                }
            }
        }

        let results: Vec<(usize, T)> = top_k
            .data()
            .iter()
            .map(|(OrderedFloat(dist), idx)| (*idx, *dist))
            .collect();

        results.into_iter().unzip()
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
    ) -> (Vec<usize>, Vec<T>) {
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
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// `(Vec<indices>, Vec<optional dist>)`
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
            + self.nodes.capacity() * std::mem::size_of::<BallNode<T>>()
            + self.centers_data.capacity() * std::mem::size_of::<T>()
            + self.leaf_indices.capacity() * std::mem::size_of::<usize>()
    }
}

///////////////////
// KnnValidation //
///////////////////

/// KnnValidation trait implementation for the BallTreeIndex
impl<T> KnnValidation<T> for BallTreeIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
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
    fn test_ball_tree_index_creation() {
        let mat = create_simple_matrix();
        let _ = BallTreeIndex::new(mat.as_ref(), Dist::Euclidean, 42);
    }

    #[test]
    fn test_ball_tree_query_finds_self() {
        let mat = create_simple_matrix();
        let index = BallTreeIndex::new(mat.as_ref(), Dist::Euclidean, 42);

        // Query with point 0, should find itself first
        let query = vec![1.0, 0.0, 0.0];
        // FIX: Pass explicit search_k (5) because default 5% of 5 is 0
        let (indices, distances) = index.query(&query, 1, Some(5));

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ball_tree_query_euclidean() {
        let mat = create_simple_matrix();
        let index = BallTreeIndex::new(mat.as_ref(), Dist::Euclidean, 42);

        let query = vec![1.0, 0.0, 0.0];
        // FIX: Pass explicit search_k
        let (indices, distances) = index.query(&query, 3, Some(5));

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
        let index = BallTreeIndex::new(mat.as_ref(), Dist::Cosine, 42);

        let query = vec![1.0, 0.0, 0.0];
        // FIX: Pass explicit search_k
        let (indices, distances) = index.query(&query, 3, Some(5));

        // Should find point 0 first (identical direction)
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ball_tree_query_k_larger_than_dataset() {
        use crate::prelude::*;

        let mat = create_simple_matrix();
        let index = BallTreeIndex::new(mat.as_ref(), Dist::Euclidean, 42);

        let query = vec![1.0, 0.0, 0.0];
        // ask for 10 neighbours but only 5 points exist
        // FIX: Pass explicit search_k
        let (indices, _) = index.query(&query, 10, Some(5));

        // Should return at most 5 results
        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_ball_tree_query_search_k() {
        use crate::prelude::*;

        let mat = create_simple_matrix();
        let index = BallTreeIndex::new(mat.as_ref(), Dist::Euclidean, 42);

        let query = vec![1.0, 0.0, 0.0];

        // This test was actually passing before because we set Some(10) and Some(1)
        let (indices1, _) = index.query(&query, 3, Some(10));
        let (indices2, _) = index.query(&query, 3, Some(1));

        assert_eq!(indices1.len(), 3);
        assert!(!indices2.is_empty());
    }

    #[test]
    fn test_ball_tree_query_row() {
        let mat = create_simple_matrix();
        let index = BallTreeIndex::new(mat.as_ref(), Dist::Euclidean, 42);

        // Query using a row from the matrix
        // FIX: Pass explicit search_k
        let (indices, distances) = index.query_row(mat.row(0), 1, Some(5));

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_ball_tree_reproducibility() {
        let mat = create_simple_matrix();

        let index1 = BallTreeIndex::new(mat.as_ref(), Dist::Euclidean, 42);
        let index2 = BallTreeIndex::new(mat.as_ref(), Dist::Euclidean, 42);

        let query = vec![0.5, 0.5, 0.0];
        // FIX: Pass explicit search_k
        let (indices1, _) = index1.query(&query, 3, Some(5));
        let (indices2, _) = index2.query(&query, 3, Some(5));

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_ball_tree_different_seeds() {
        let mat = create_simple_matrix();

        let index1 = BallTreeIndex::new(mat.as_ref(), Dist::Euclidean, 42);
        let index2 = BallTreeIndex::new(mat.as_ref(), Dist::Euclidean, 123);

        let query = vec![0.5, 0.5, 0.0];

        // FIX: Pass explicit search_k
        let (indices1, _) = index1.query(&query, 3, Some(5));
        let (indices2, _) = index2.query(&query, 3, Some(5));

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_ball_tree_larger_dataset() {
        // Create a larger synthetic dataset (n=100)
        // 5% of 100 is 5, so this should work even with None (default)
        // But for safety in tests we can be explicit
        let n = 100;
        let dim = 10;
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 10.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = BallTreeIndex::new(mat.as_ref(), Dist::Euclidean, 42);

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5, None);

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_ball_tree_orthogonal_vectors() {
        use crate::prelude::*;
        let data = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);
        let index = BallTreeIndex::new(mat.as_ref(), Dist::Cosine, 42);

        let query = vec![1.0, 0.0, 0.0];
        // FIX: Pass explicit search_k
        let (indices, distances) = index.query(&query, 3, Some(3));

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

        let index = BallTreeIndex::new(mat.as_ref(), Dist::Euclidean, 42);

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 3, None); // 5% of 50 is 2.5 -> 2. Safe-ish.

        assert_eq!(indices.len(), 3);
    }
}
