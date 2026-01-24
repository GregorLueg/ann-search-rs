use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::{cell::RefCell, collections::BinaryHeap, iter::Sum};
use thousands::*;

use crate::utils::dist::*;
use crate::utils::heap_structs::*;
use crate::utils::*;

/////////////
// Helpers //
/////////////

///////////
// Enums //
///////////

/// Tree node with iSAX-style encoding
///
/// Root nodes have 2^K children representing all initial 1-bit encodings.
/// Internal nodes split one dimension by adding a bit.
/// Leaf nodes store point indices with bounds computed from symbolic encoding.
pub enum Node<T> {
    /// Root with 2^K children (all initial 1-bit combinations)
    Root { children: Vec<Box<Node<T>>> },
    /// Internal node with symbolic encoding
    Internal {
        encoding: Vec<Symbol>,
        split_dim: usize,
        left: Box<Node<T>>,
        right: Box<Node<T>>,
        bounds: Vec<(T, T)>, // computed from encoding
    },
    /// Leaf node with points and encoding
    Leaf {
        encoding: Vec<Symbol>,
        points: Vec<usize>,
        bounds: Vec<(T, T)>, // computed from encoding
    },
}

////////////////
// Structures //
////////////////

/// Symbolic representation of one dimension (iSAX-style)
///
/// Encodes a continuous value as a bit-wise symbol. For example:
///
/// - 1 bit: "0*" (value in [0.0, 0.5)) or "1*" (value in [0.5, 1.0))
/// - 2 bits: "00" [0.0, 0.25), "01" [0.25, 0.5), "10" [0.5, 0.75), "11" [0.75, 1.0)
/// - 3 bits: further subdivisions, etc.
///
/// ### Fields
///
/// * `bits`: bit pattern: [false, true] = "01"
/// * `cardinality`: number of bits (resolution)
#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    bits: Vec<bool>,
}

impl Symbol {
    /// Create a new symbol with specified bits
    ///
    /// ### Params
    ///
    /// * `bits`: bit pattern: [false, true] = "01"
    ///
    /// ### Returns
    ///
    /// `Symbol` - Self
    fn new(bits: Vec<bool>) -> Self {
        Self { bits }
    }

    /// Create initial 1-bit symbol (0* or 1*)
    ///
    /// ### Params
    ///
    /// * `bit`: initial bit value
    ///
    /// ### Returns
    ///
    /// `Symbol` - Self
    fn initial(bit: bool) -> Self {
        Self { bits: vec![bit] }
    }

    /// Cardinality of the symbol (number of bits)
    ///
    /// ### Returns
    ///
    /// `usize` - Cardinality
    fn cardinality(&self) -> usize {
        self.bits.len()
    }

    /// Refine by adding a bit (split into two regions)
    ///
    /// ### Params
    ///
    /// * `new_bit`: bit value to add
    ///
    /// ### Returns
    ///
    /// `Symbol` - Self
    fn refine(&self, new_bit: bool) -> Self {
        let mut new_bits = self.bits.clone();
        new_bits.push(new_bit);
        Self { bits: new_bits }
    }
}

/// Single DE-Tree with projected data and spatial hierarchy
///
/// Stores all projected vectors in a flat buffer and provides range query
/// capability through an iSAX-style encoding-based tree.
pub struct DETree<T> {
    root: Node<T>,
    projected_points_flat: Vec<T>, // flat buffer [n * proj_dims]
    proj_dims: usize,
}

impl<T: Copy> DETree<T> {
    /// Access projected point coordinate
    ///
    /// ### Params
    ///
    /// * `point_idx`: index of the point
    /// * `dim`: dimension of the point
    ///
    /// ### Returns
    ///
    /// `T` - Projected coordinate
    #[inline]
    #[allow(dead_code)]
    fn get_projected(&self, point_idx: usize, dim: usize) -> T {
        self.projected_points_flat[point_idx * self.proj_dims + dim]
    }
}

//////////////////////
// Helper functions //
//////////////////////

/// Estimate memory used by tree nodes
///
/// ### Params
///
/// * `node`: reference to the node
///
/// ### Returns
///
/// Memory used by the tree
fn estimate_tree_size<T>(node: &Node<T>) -> usize {
    match node {
        Node::Root { children } => {
            std::mem::size_of::<Node<T>>()
                + children.capacity() * std::mem::size_of::<Box<Node<T>>>()
                + children
                    .iter()
                    .map(|c| estimate_tree_size(c))
                    .sum::<usize>()
        }
        Node::Leaf {
            encoding,
            points,
            bounds,
        } => {
            std::mem::size_of::<Node<T>>()
                + encoding.iter().map(|s| s.bits.capacity()).sum::<usize>()
                + points.capacity() * std::mem::size_of::<usize>()
                + bounds.capacity() * std::mem::size_of::<(T, T)>()
        }
        Node::Internal {
            encoding,
            left,
            right,
            bounds,
            ..
        } => {
            std::mem::size_of::<Node<T>>()
                + encoding.iter().map(|s| s.bits.capacity()).sum::<usize>()
                + bounds.capacity() * std::mem::size_of::<(T, T)>()
                + estimate_tree_size(left)
                + estimate_tree_size(right)
        }
    }
}

/// Encode a point into iSAX symbolic representation
///
/// Converts continuous coordinates to bit-wise symbols based on region
/// boundaries. For example, with cardinality=2, dimension value 0.3 becomes
/// "01" (second of four regions: [0.25, 0.5)).
///
/// ### Params
///
/// * `point_flat` - Flat buffer containing all projected points
/// * `point_idx` - Index of the point to encode
/// * `proj_dims` - Number of projected dimensions
/// * `cardinalities` - Number of bits per dimension
///
/// ### Returns
///
/// Vector of symbols, one per dimension
fn encode_point<T>(
    point_flat: &[T],
    point_idx: usize,
    proj_dims: usize,
    cardinalities: &[usize],
) -> Vec<Symbol>
where
    T: Float + FromPrimitive,
{
    let mut encoding = Vec::with_capacity(proj_dims);
    let base = point_idx * proj_dims;

    for dim in 0..proj_dims {
        let value = point_flat[base + dim];
        let card = cardinalities[dim];
        let num_regions = 1usize << card; // 2^card

        // Normalise to [0, 1) range - assumes values are roughly in [-1, 1]
        // from projections. For robustness, we clamp to [0, 1)
        let normalised = (value + T::one()) / (T::one() + T::one());
        let clamped = normalised.max(T::zero()).min(T::from_f32(0.999).unwrap());

        // Determine which region
        let region = (clamped * T::from_usize(num_regions).unwrap())
            .floor()
            .to_usize()
            .unwrap();

        // Convert region to binary bits
        let mut bits = Vec::with_capacity(card);
        for i in (0..card).rev() {
            bits.push((region >> i) & 1 == 1);
        }

        encoding.push(Symbol::new(bits));
    }

    encoding
}

// /// Check if a point matches a given encoding
// ///
// /// ### Params
// ///
// /// * `point_flat` - Flat buffer containing all projected points
// /// * `point_idx` - Index of the point
// /// * `proj_dims` - Number of projected dimensions
// /// * `encoding` - Symbolic encoding to match against
// ///
// /// ### Returns
// ///
// /// True if point's encoding matches the given encoding
// fn matches_encoding<T>(
//     point_flat: &[T],
//     point_idx: usize,
//     proj_dims: usize,
//     encoding: &[Symbol],
// ) -> bool
// where
//     T: Float + FromPrimitive,
// {
//     let cardinalities: Vec<usize> = encoding.iter().map(|s| s.cardinality()).collect();
//     let point_encoding = encode_point(point_flat, point_idx, proj_dims, &cardinalities);

//     point_encoding == encoding
// }

/// Compute bounds from iSAX symbolic encoding
///
/// Converts bit patterns to continuous region boundaries. For example:
///
/// - Symbol "01" with cardinality 2: region [0.25, 0.5) in normalised space
/// - Symbol "1*" with cardinality 1: region [0.5, 1.0) in normalised space
///
/// ### Params
///
/// * `encoding` - Symbolic encoding
///
/// ### Returns
///
/// Vector of (min, max) bounds, one per dimension
fn compute_bounds_from_encoding<T>(encoding: &[Symbol]) -> Vec<(T, T)>
where
    T: Float + FromPrimitive,
{
    let mut bounds = Vec::with_capacity(encoding.len());

    for symbol in encoding {
        let card = symbol.cardinality();
        let num_regions = 1usize << card; // 2^card

        // Decode bit pattern to region number
        let mut region = 0usize;
        for (i, &bit) in symbol.bits.iter().enumerate() {
            if bit {
                region |= 1 << (card - 1 - i);
            }
        }

        // Convert region to normalised [0, 1) bounds
        let region_size = T::one() / T::from_usize(num_regions).unwrap();
        let min_normalized = T::from_usize(region).unwrap() * region_size;
        let max_normalized = min_normalized + region_size;

        // Convert back to [-1, 1) range (inverse of normalisation)
        let min = min_normalized * (T::one() + T::one()) - T::one();
        let max = max_normalized * (T::one() + T::one()) - T::one();

        bounds.push((min, max));
    }

    bounds
}

/// Split an iSAX node by adding a bit to one dimension
///
/// Chooses the dimension that produces the most even split, then increments
/// its cardinality by 1 (splits into two sub-regions).
///
/// ### Params
///
/// * `point_indices` - Points in this node
/// * `encoding` - Current symbolic encoding
/// * `projected_points_flat` - Flat buffer of projected coordinates
/// * `proj_dims` - Number of projected dimensions
/// * `max_leaf_size` - Maximum points per leaf
///
/// ### Returns
///
/// Internal or leaf node
fn split_isax_node<T>(
    point_indices: Vec<usize>,
    encoding: Vec<Symbol>,
    projected_points_flat: &[T],
    proj_dims: usize,
    max_leaf_size: usize,
) -> Node<T>
where
    T: Float + FromPrimitive,
{
    // find dimension with most even split
    let mut best_dim = 0;
    let mut best_balance = usize::MAX;

    for dim in 0..proj_dims {
        // count points that would go to left child (bit 0) vs right (bit 1)
        let mut left_encoding = encoding.clone();
        left_encoding[dim] = left_encoding[dim].refine(false);

        let left_cardinalities: Vec<usize> =
            left_encoding.iter().map(|s| s.cardinality()).collect();

        let left_count = point_indices
            .iter()
            .filter(|&&idx| {
                let point_enc =
                    encode_point(projected_points_flat, idx, proj_dims, &left_cardinalities);
                point_enc == left_encoding
            })
            .count();

        let right_count = point_indices.len() - left_count;
        let balance = left_count.abs_diff(right_count);

        if balance < best_balance {
            best_balance = balance;
            best_dim = dim;
        }
    }

    // split on best dimension
    let mut left_encoding = encoding.clone();
    let mut right_encoding = encoding.clone();

    left_encoding[best_dim] = left_encoding[best_dim].refine(false);
    right_encoding[best_dim] = right_encoding[best_dim].refine(true);

    // partition points
    let left_cardinalities: Vec<usize> = left_encoding.iter().map(|s| s.cardinality()).collect();

    let mut left_points = Vec::new();
    let mut right_points = Vec::new();

    for &idx in &point_indices {
        let point_enc = encode_point(projected_points_flat, idx, proj_dims, &left_cardinalities);
        if point_enc == left_encoding {
            left_points.push(idx);
        } else {
            right_points.push(idx);
        }
    }

    // handle degenerate case (all points to one side)
    if left_points.is_empty() || right_points.is_empty() {
        let bounds = compute_bounds_from_encoding(&encoding);
        return Node::Leaf {
            encoding,
            points: point_indices,
            bounds,
        };
    }

    // build children
    let left = if left_points.len() <= max_leaf_size {
        let bounds = compute_bounds_from_encoding(&left_encoding);
        Box::new(Node::Leaf {
            encoding: left_encoding.clone(),
            points: left_points,
            bounds,
        })
    } else {
        Box::new(split_isax_node(
            left_points,
            left_encoding.clone(),
            projected_points_flat,
            proj_dims,
            max_leaf_size,
        ))
    };

    let right = if right_points.len() <= max_leaf_size {
        let bounds = compute_bounds_from_encoding(&right_encoding);
        Box::new(Node::Leaf {
            encoding: right_encoding.clone(),
            points: right_points,
            bounds,
        })
    } else {
        Box::new(split_isax_node(
            right_points,
            right_encoding.clone(),
            projected_points_flat,
            proj_dims,
            max_leaf_size,
        ))
    };

    // compute bounds for internal node (union of children)
    let left_bounds = match left.as_ref() {
        Node::Leaf { bounds, .. } => bounds,
        Node::Internal { bounds, .. } => bounds,
        Node::Root { .. } => unreachable!("Root cannot be child"),
    };
    let right_bounds = match right.as_ref() {
        Node::Leaf { bounds, .. } => bounds,
        Node::Internal { bounds, .. } => bounds,
        Node::Root { .. } => unreachable!("Root cannot be child"),
    };

    let mut bounds = Vec::with_capacity(proj_dims);
    for dim in 0..proj_dims {
        let min = left_bounds[dim].0.min(right_bounds[dim].0);
        let max = left_bounds[dim].1.max(right_bounds[dim].1);
        bounds.push((min, max));
    }

    Node::Internal {
        encoding,
        split_dim: best_dim,
        left,
        right,
        bounds,
    }
}

/// Build iSAX-style DE-Tree
///
/// Creates root with 2^K children representing all initial 1-bit encodings,
/// then recursively refines by adding bits to dimensions with most even splits.
///
/// ### Params
///
/// * `point_indices` - Indices of all points
/// * `projected_points_flat` - Flat buffer of projected coordinates
/// * `proj_dims` - Number of projected dimensions (K)
/// * `max_leaf_size` - Maximum points per leaf
///
/// ### Returns
///
/// Root node of constructed tree
fn build_isax_tree<T>(
    point_indices: &[usize],
    projected_points_flat: &[T],
    proj_dims: usize,
    max_leaf_size: usize,
) -> Node<T>
where
    T: Float + FromPrimitive,
{
    // create root with 2^K children (all initial 1-bit encodings)
    let num_root_children = 1usize << proj_dims;
    let mut root_children = Vec::with_capacity(num_root_children);

    for i in 0..num_root_children {
        // generate encoding for this child
        let mut encoding = Vec::with_capacity(proj_dims);
        for dim in 0..proj_dims {
            let bit = (i >> (proj_dims - 1 - dim)) & 1 == 1;
            encoding.push(Symbol::initial(bit));
        }

        // collect points matching this encoding
        let cardinalities = vec![1; proj_dims];
        let mut child_points = Vec::new();
        for &idx in point_indices {
            let point_encoding =
                encode_point(projected_points_flat, idx, proj_dims, &cardinalities);
            if point_encoding == encoding {
                child_points.push(idx);
            }
        }

        // build child (leaf or internal)
        let child = if child_points.len() <= max_leaf_size {
            let bounds = compute_bounds_from_encoding(&encoding);
            Node::Leaf {
                encoding,
                points: child_points,
                bounds,
            }
        } else {
            split_isax_node(
                child_points,
                encoding,
                projected_points_flat,
                proj_dims,
                max_leaf_size,
            )
        };

        root_children.push(Box::new(child));
    }

    Node::Root {
        children: root_children,
    }
}

/// Orthogonalise random projections using Gram-Schmidt
///
/// Orthogonalises the random projections within each tree, improving
/// discriminative power and query performance. If a vector has near-zero
/// norm after orthogonalisation, it is reinitialised with a new random
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
    T: Float + FromPrimitive,
{
    for table_idx in 0..num_trees {
        let base = table_idx * proj_dims * dim;

        for i in 0..proj_dims {
            let i_base = base + i * dim;

            // orthogonalise against previous vectors
            for j in 0..i {
                let j_base = base + j * dim;
                let mut dot = T::zero();
                for d in 0..dim {
                    dot = dot + projections[i_base + d] * projections[j_base + d];
                }
                for d in 0..dim {
                    projections[i_base + d] =
                        projections[i_base + d] - dot * projections[j_base + d];
                }
            }

            // normalise
            let mut norm_sq = T::zero();
            for d in 0..dim {
                norm_sq = norm_sq + projections[i_base + d] * projections[i_base + d];
            }
            let norm = norm_sq.sqrt();

            if norm > T::epsilon() {
                for d in 0..dim {
                    projections[i_base + d] = projections[i_base + d] / norm;
                }
            } else {
                // re-initialise with new random vector if norm is too small
                let reinit_seed = seed.wrapping_mul(table_idx + 1).wrapping_mul(i + 1);
                let mut rng = StdRng::seed_from_u64(reinit_seed as u64);
                for d in 0..dim {
                    let val: f64 = rng.sample(StandardNormal);
                    projections[i_base + d] = T::from_f64(val).unwrap();
                }

                // re-orthogonalise and normalise the new vector
                for j in 0..i {
                    let j_base = base + j * dim;
                    let mut dot = T::zero();
                    for d in 0..dim {
                        dot = dot + projections[i_base + d] * projections[j_base + d];
                    }
                    for d in 0..dim {
                        projections[i_base + d] =
                            projections[i_base + d] - dot * projections[j_base + d];
                    }
                }

                let mut norm_sq = T::zero();
                for d in 0..dim {
                    norm_sq = norm_sq + projections[i_base + d] * projections[i_base + d];
                }
                let norm = norm_sq.sqrt();

                assert!(
                    norm > T::epsilon(),
                    "Orthogonalisation failed even after reinitialisation"
                );

                for d in 0..dim {
                    projections[i_base + d] = projections[i_base + d] / norm;
                }
            }
        }
    }
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
fn distance_lower_bound<T>(bounds: &[(T, T)], query: &[T]) -> T
where
    T: Float,
{
    let mut sum_sq = T::zero();

    for (dim, &(min, max)) in bounds.iter().enumerate() {
        if query[dim] < min {
            let diff = min - query[dim];
            sum_sq = sum_sq + diff * diff;
        } else if query[dim] > max {
            let diff = query[dim] - max;
            sum_sq = sum_sq + diff * diff;
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
    T: Float,
{
    let mut sum_sq = T::zero();

    for (dim, &(min, max)) in bounds.iter().enumerate() {
        let dist_to_min = (query[dim] - min).abs();
        let dist_to_max = (query[dim] - max).abs();
        let max_dist = dist_to_min.max(dist_to_max);
        sum_sq = sum_sq + max_dist * max_dist;
    }

    sum_sq.sqrt()
}

/// Compute Euclidean distance in projected space using SIMD
///
/// ###
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
/// radius. Handles Root nodes by traversing all 2^K children.
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
    node: &Node<T>,
    query_projected: &[T],
    radius: T,
    projected_points_flat: &[T],
    proj_dims: usize,
    candidates: &mut Vec<usize>,
) where
    T: Float + SimdDistance,
{
    match node {
        Node::Root { children } => {
            // Traverse all 2^K children
            for child in children {
                range_query(
                    child,
                    query_projected,
                    radius,
                    projected_points_flat,
                    proj_dims,
                    candidates,
                );
            }
        }
        Node::Leaf { points, bounds, .. } => {
            // Compute lower bound
            let lower_bound = distance_lower_bound(bounds, query_projected);

            // Prune if lower bound exceeds radius
            if lower_bound > radius {
                return;
            }

            let upper_bound = distance_upper_bound(bounds, query_projected);

            if upper_bound <= radius {
                // All points guaranteed within radius
                candidates.extend_from_slice(points);
            } else {
                // Check each point individually
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
        Node::Internal {
            left,
            right,
            bounds,
            ..
        } => {
            // Compute lower bound
            let lower_bound = distance_lower_bound(bounds, query_projected);

            // Prune if lower bound exceeds radius
            if lower_bound > radius {
                return;
            }

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

////////////////////
// LSHDETreeIndex //
////////////////////

/// LSH index using DE-Trees for approximate nearest neighbour search
///
/// Implements the DET-LSH algorithm using iSAX-style encoding-based trees.
/// Combines random projections with hierarchical symbolic encoding to enable
/// efficient range queries in projected space.
///
/// ### Algorithm Overview
///
/// - **Projection**: Map high-dimensional vectors to lower-dimensional space
///   using random Gaussian projections (LSH functions)
/// - **Encoding**: Convert projected coordinates to bit-wise symbolic
///   representations (iSAX-style)
/// - **Indexing**: Build trees with 2^K root children representing all initial
///   encodings, then refine by adding bits to dimensions with most even splits
/// - **Querying**: Traverse trees with range queries, pruning subtrees where
///   lower bound exceeds search radius
///
/// ### Key Differences from Standard kd-trees
///
/// - Root has 2^K children (all initial 1-bit encodings)
/// - Splits add bits to symbolic encoding rather than using continuous medians
/// - Bounds computed from symbolic regions, not actual point coordinates
/// - Split dimension chosen for most even division, not highest variance
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
    // index-specific ones
    de_trees: Vec<DETree<T>>,
    projections: Vec<T>,
    proj_dims: usize,
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
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
    Self: DETreeQuery<T>,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Construct a new LSH DE-Tree index
    ///
    /// Builds trees in parallel using iSAX-style encoding. For each tree:
    /// 1. Projects all vectors to K dimensions
    /// 2. Creates root with 2^K children (all initial 1-bit encodings)
    /// 3. Refines nodes by adding bits to dimensions with most even splits
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (rows = samples, columns = dimensions)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `num_trees` - Number of trees (5-20 typical)
    /// * `proj_dims` - Projected dimensionality (8-32 typical, but 2^K root children means memory grows exponentially)
    /// * `max_leaf_size` - Maximum points per leaf (32-128 typical)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    ///
    /// ### Note on proj_dims
    ///
    /// Root has 2^proj_dims children. For proj_dims=8, that's 256 children.
    /// For proj_dims=16, that's 65,536 children. Choose carefully based on
    /// memory constraints and dataset size.
    pub fn new(
        data: MatRef<T>,
        metric: Dist,
        num_trees: usize,
        proj_dims: usize,
        max_leaf_size: usize,
        seed: usize,
    ) -> Self {
        assert!(max_leaf_size >= 1, "max_leaf_size must be at least 1");
        assert!(
            proj_dims <= 16,
            "proj_dims > 16 would create 2^{} = {} root children (too many)",
            proj_dims,
            1usize << proj_dims
        );

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
                        let proj_vec = &projections[offset..offset + dim];
                        let dot = T::dot_simd(vec, proj_vec);
                        projected_points_flat[vec_idx * proj_dims + k] = dot;
                    }
                }

                // build tree with iSAX-style encoding
                let all_indices: Vec<usize> = (0..n).collect();
                let root = build_isax_tree(
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
            proj_dims,
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

    /// Query using a matrix row reference
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies
    /// to a temporary vector.
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference to query vector
    /// * `k` - Number of neighbours to return
    /// * `search_radius` - Search radius in projected space
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        search_radius: T,
    ) -> (Vec<usize>, Vec<T>) {
        assert!(
            query_row.ncols() == self.dim,
            "Query row dimensionality mismatch"
        );

        // Fast path for contiguous row data
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, search_radius);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, search_radius)
    }

    /// Query using a matrix row reference with adaptive radius
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference to query vector
    /// * `k` - Number of neighbours to return
    /// * `initial_radius` - Starting search radius in projected space
    /// * `expansion_factor` - Radius multiplier per iteration
    /// * `max_radius` - Maximum allowed radius
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    pub fn query_row_adaptive(
        &self,
        query_row: RowRef<T>,
        k: usize,
        initial_radius: T,
        expansion_factor: T,
        max_radius: T,
    ) -> (Vec<usize>, Vec<T>) {
        assert!(
            query_row.ncols() == self.dim,
            "Query row dimensionality mismatch"
        );

        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query_adaptive(slice, k, initial_radius, expansion_factor, max_radius);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query_adaptive(&query_vec, k, initial_radius, expansion_factor, max_radius)
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete
    /// kNN graph using adaptive radius search.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `initial_radius` - Starting search radius
    /// * `expansion_factor` - Radius multiplier per iteration
    /// * `max_radius` - Maximum allowed radius
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)` where each row corresponds
    /// to a vector in the index
    pub fn generate_knn(
        &self,
        k: usize,
        initial_radius: T,
        expansion_factor: T,
        max_radius: T,
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

                self.query_adaptive(vec, k, initial_radius, expansion_factor, max_radius)
            })
            .collect();

        if return_dist {
            let mut indices = Vec::with_capacity(results.len());
            let mut distances = Vec::with_capacity(results.len());

            for (idx, dist) in results {
                indices.push(idx);
                distances.push(dist);
            }
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

//////////////////
// DETreeQuery //
//////////////////

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

/// Maximum candidate multiplier for early exit (collect up to k * this)
const CANDIDATE_MULTIPLIER: usize = 10;

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
                        // project query using SIMD
                        let base = tree_idx * self.proj_dims * self.dim;
                        let mut query_projected = Vec::with_capacity(self.proj_dims);
                        for k in 0..self.proj_dims {
                            let offset = base + k * self.dim;
                            let proj_vec = &self.projections[offset..offset + self.dim];
                            let dot = f32::dot_simd(query_vec, proj_vec);
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
                            let query_norm = f32::calculate_l2_norm(query_vec);
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
                        // project query using SIMD
                        let base = tree_idx * self.proj_dims * self.dim;
                        let mut query_projected = Vec::with_capacity(self.proj_dims);
                        for k in 0..self.proj_dims {
                            let offset = base + k * self.dim;
                            let proj_vec = &self.projections[offset..offset + self.dim];
                            let dot = f64::dot_simd(query_vec, proj_vec);
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
                            let query_norm = f64::calculate_l2_norm(query_vec);
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn simple_test_data() -> Mat<f32> {
        // 5 vectors in 3 dimensions
        Mat::from_fn(5, 3, |i, j| match i {
            0 => [1.0, 0.0, 0.0][j],
            1 => [0.0, 1.0, 0.0][j],
            2 => [0.0, 0.0, 1.0][j],
            3 => [1.0, 1.0, 0.0][j],
            4 => [0.5, 0.5, 0.7][j],
            _ => 0.0,
        })
    }

    #[test]
    fn test_index_creation_euclidean() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.de_trees.len(), 4);
        assert_eq!(index.proj_dims, 2);
        assert_eq!(index.vectors_flat.len(), 15);
        assert_eq!(index.de_trees.len(), 4);

        // verify flat buffer structure
        for tree in &index.de_trees {
            assert_eq!(tree.projected_points_flat.len(), 5 * 2); // n * proj_dims
            assert_eq!(tree.proj_dims, 2);

            // verify root has 2^K children
            match &tree.root {
                Node::Root { children } => {
                    assert_eq!(children.len(), 1 << 2); // 2^2 = 4 children
                }
                _ => panic!("Root should be Node::Root"),
            }
        }
    }

    #[test]
    fn test_index_creation_cosine() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Cosine, 4, 2, 2, 42);

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.norms.len(), 5);
    }

    #[test]
    fn test_root_has_2k_children() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 2, 3, 2, 42);

        for tree in &index.de_trees {
            match &tree.root {
                Node::Root { children } => {
                    assert_eq!(children.len(), 1 << 3); // 2^3 = 8 children
                }
                _ => panic!("Root should be Node::Root with 2^K children"),
            }
        }
    }

    #[test]
    fn test_encoding_point() {
        let projected = vec![0.3f32, -0.7f32]; // 2 dimensions
        let cardinalities = vec![2, 2]; // 2 bits each

        let encoding = encode_point(&projected, 0, 2, &cardinalities);

        assert_eq!(encoding.len(), 2);
        assert_eq!(encoding[0].cardinality(), 2);
        assert_eq!(encoding[1].cardinality(), 2);

        // 0.3 normalized to [0,1): (0.3+1)/2 = 0.65
        // With 4 regions: 0.65 * 4 = 2.6 → region 2 → "10"
        assert_eq!(encoding[0].bits, vec![true, false]);

        // -0.7 normalized: (-0.7+1)/2 = 0.15
        // 0.15 * 4 = 0.6 → region 0 → "00"
        assert_eq!(encoding[1].bits, vec![false, false]);
    }

    #[test]
    fn test_symbol_refinement() {
        let sym = Symbol::initial(true); // "1*"
        assert_eq!(sym.bits, vec![true]);
        assert_eq!(sym.cardinality(), 1);

        let refined_left = sym.refine(false); // "10"
        assert_eq!(refined_left.bits, vec![true, false]);
        assert_eq!(refined_left.cardinality(), 2);

        let refined_right = sym.refine(true); // "11"
        assert_eq!(refined_right.bits, vec![true, true]);
        assert_eq!(refined_right.cardinality(), 2);
    }

    #[test]
    fn test_bounds_from_encoding() {
        let encoding = vec![
            Symbol::new(vec![false, true]), // "01" - region 1 of 4
            Symbol::new(vec![true]),        // "1*" - region 1 of 2
        ];

        let bounds: Vec<(f32, f32)> = compute_bounds_from_encoding(&encoding);

        // Dimension 0: "01" = region 1 of 4 = [0.25, 0.5) normalized
        // Converted back: [0.25*2-1, 0.5*2-1) = [-0.5, 0.0)
        assert!((bounds[0].0 - (-0.5)).abs() < 1e-5);
        assert!((bounds[0].1 - 0.0).abs() < 1e-5);

        // Dimension 1: "1*" = region 1 of 2 = [0.5, 1.0) normalized
        // Converted back: [0.5*2-1, 1.0*2-1) = [0.0, 1.0)
        assert!((bounds[1].0 - 0.0).abs() < 1e-5);
        assert!((bounds[1].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_basic_query() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, 5.0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());

        // distances should be sorted
        for i in 1..distances.len() {
            assert!(distances[i - 1] <= distances[i]);
        }
    }

    #[test]
    fn test_query_adaptive() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query_adaptive(&query, 3, 0.1, 2.0, 20.0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_query_adaptive_terminates_on_full_dataset() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];
        // ask for more neighbours than we have points
        let (indices, _) = index.query_adaptive(&query, 100, 0.1, 2.0, 1000.0);

        // should return all points and terminate, not loop forever
        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_query_adaptive_terminates_on_saturation() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];
        // with a very small max_radius, should terminate without finding enough
        let (indices, _) = index.query_adaptive(&query, 5, 0.01, 1.1, 0.05);

        // should terminate even if we don't have 5 neighbours
        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_query_cosine() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Cosine, 4, 2, 2, 42);

        let query = vec![2.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 2, 5.0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 2);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_query_row() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query_mat = Mat::from_fn(1, 3, |_, j| [1.0, 0.0, 0.0][j]);
        let (indices, distances) = index.query_row(query_mat.row(0), 3, 5.0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_query_row_adaptive() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query_mat = Mat::from_fn(1, 3, |_, j| [1.0, 0.0, 0.0][j]);
        let (indices, distances) = index.query_row_adaptive(query_mat.row(0), 3, 0.1, 2.0, 20.0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_k_larger_than_n() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 100, 10.0);

        assert!(indices.len() <= 5);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    #[should_panic(expected = "Query vector dimensionality mismatch")]
    fn test_dimension_mismatch() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0];
        index.query(&query, 3, 5.0);
    }

    #[test]
    #[should_panic(expected = "Query row dimensionality mismatch")]
    fn test_query_row_dimension_mismatch() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query_mat = Mat::from_fn(1, 2, |_, j| [1.0, 0.0][j]);
        index.query_row(query_mat.row(0), 3, 5.0);
    }

    #[test]
    #[should_panic(expected = "max_leaf_size must be at least 1")]
    fn test_invalid_max_leaf_size() {
        let mat = simple_test_data();
        LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 0, 42);
    }

    #[test]
    #[should_panic(expected = "expansion_factor must be > 1")]
    fn test_invalid_expansion_factor() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];
        index.query_adaptive(&query, 3, 1.0, 0.5, 10.0);
    }

    #[test]
    #[should_panic(expected = "proj_dims > 16")]
    fn test_proj_dims_too_large() {
        let mat = simple_test_data();
        LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 2, 17, 2, 42);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let mat = simple_test_data();

        let index1 = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);
        let index2 = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices1, _) = index1.query(&query, 3, 5.0);
        let (indices2, _) = index2.query(&query, 3, 5.0);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_f64_query() {
        let mat = Mat::from_fn(3, 3, |i, j| if i == j { 1.0f64 } else { 0.0f64 });
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 2, 5.0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 2);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_distances_sorted() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (_, distances) = index.query(&query, 5, 10.0);

        for i in 1..distances.len() {
            assert!(distances[i - 1] <= distances[i]);
        }
    }

    #[test]
    fn test_query_k_zero() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 0, 5.0);

        assert_eq!(indices.len(), 0);
        assert_eq!(distances.len(), 0);
    }

    #[test]
    fn test_query_returns_k_or_fewer() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];

        for k in 1..=5 {
            let (indices, distances) = index.query(&query, k, 10.0);
            assert!(indices.len() <= k);
            assert_eq!(indices.len(), distances.len());
        }
    }

    #[test]
    fn test_no_duplicate_results() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 5, 10.0);

        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();

        assert_eq!(indices.len(), sorted.len(), "Results contain duplicates");
    }

    #[test]
    fn test_larger_dataset() {
        let n = 1000;
        let dim = 50;
        let mat = Mat::from_fn(n, dim, |i, j| ((i * 7 + j * 13) % 100) as f32 / 100.0);

        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 10, 8, 32, 42);

        let query = vec![0.5; dim];
        let (indices, distances) = index.query(&query, 10, 5.0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 10);
        assert_eq!(indices.len(), distances.len());

        for &idx in &indices {
            assert!(idx < n);
        }

        // Verify root has 2^8 = 256 children
        for tree in &index.de_trees {
            match &tree.root {
                Node::Root { children } => {
                    assert_eq!(children.len(), 256);
                }
                _ => panic!("Root should be Node::Root"),
            }
        }
    }

    #[test]
    fn test_adaptive_radius_convergence() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];

        // should eventually find k neighbours or exhaust dataset
        let (indices, _) = index.query_adaptive(&query, 3, 0.01, 1.5, 50.0);
        assert!(indices.len() >= 3 || indices.len() == index.n.min(3));
    }

    #[test]
    fn test_projection_orthogonality() {
        let mut projections = vec![1.0f32; 4 * 3 * 5]; // 4 trees, 3 proj_dims, 5 dim
        orthogonalise_projections(&mut projections, 4, 3, 5, 42);

        // check one tree's projections are orthonormal
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0f32;
                for d in 0..5 {
                    dot += projections[i * 5 + d] * projections[j * 5 + d];
                }
                if i == j {
                    assert!((dot - 1.0).abs() < 1e-5, "Not normalised");
                } else {
                    assert!(dot.abs() < 1e-5, "Not orthogonal");
                }
            }
        }
    }

    #[test]
    fn test_tree_handles_identical_points() {
        // all points identical in projected space
        let mat = Mat::from_fn(5, 10, |_, _| 1.0f32);
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 2, 3, 2, 42);

        let query = vec![1.0; 10];
        let (indices, _) = index.query(&query, 3, 5.0);

        assert!(!indices.is_empty());
    }

    #[test]
    fn test_small_radius_returns_fewer() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, 2, 42);

        let query = vec![1.0, 0.0, 0.0];

        let (indices_small, _) = index.query(&query, 5, 0.1);
        let (indices_large, _) = index.query(&query, 5, 10.0);

        assert!(indices_small.len() <= indices_large.len());
    }

    #[test]
    fn test_flat_buffer_access() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 2, 3, 2, 42);

        // verify we can access projected points through the flat buffer
        for tree in &index.de_trees {
            for point_idx in 0..index.n {
                for dim in 0..tree.proj_dims {
                    let val = tree.get_projected(point_idx, dim);
                    // should not panic and should return a finite value
                    assert!(val.is_finite());
                }
            }
        }
    }

    #[test]
    fn test_early_exit_on_candidate_saturation() {
        let n = 1000;
        let dim = 50;
        let mat = Mat::from_fn(n, dim, |i, j| ((i * 7 + j * 13) % 100) as f32 / 100.0);

        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 20, 8, 32, 42);

        let query = vec![0.5; dim];
        // with a large radius, should collect many candidates but exit early
        let (indices, _) = index.query(&query, 10, 100.0);

        // should have found results without processing all 20 trees
        assert!(!indices.is_empty());
        assert!(indices.len() <= 10);
    }
}
