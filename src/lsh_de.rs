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

//////////////
// Constants //
//////////////

/// Number of regions per dimension (8-bit encoding = 256 regions)
const NUM_REGIONS: usize = 256;

/// Number of breakpoints per dimension (NUM_REGIONS + 1)
const NUM_BREAKPOINTS: usize = 257;

/// Default maximum leaf size
pub const MAX_LEAF_SIZE: usize = 64;

/// Sample ratio for breakpoint selection (10% as per paper)
const SAMPLE_RATIO: f64 = 0.1;

/// Minimum sample size for breakpoint selection
const MIN_SAMPLE_SIZE: usize = 1000;

///////////
// Enums //
///////////

/// Tree node for DE-Tree
///
/// Internal nodes split on one dimension's next bit.
/// Leaf nodes store point indices.
/// Bounds are computed from breakpoints during construction.
pub enum Node<T> {
    /// Internal node splitting on one dimension
    Internal {
        /// Dimension being split
        split_dim: usize,
        /// Bit position (0 = MSB, 7 = LSB)
        split_bit: usize,
        /// Threshold encoding value (points with bit=0 go left, bit=1 go right)
        left: Box<Node<T>>,
        right: Box<Node<T>>,
        /// Bounding box in projected space [dim] -> (min, max)
        bounds: Vec<(T, T)>,
    },
    /// Leaf node containing points
    Leaf {
        /// Indices of points in this leaf
        points: Vec<usize>,
        /// Bounding box in projected space
        bounds: Vec<(T, T)>,
    },
}

////////////////
// Structures //
////////////////

/// Single DE-Tree with breakpoints and projected data
pub struct DETree<T> {
    /// Root children indexed by first bit of each dimension (2^K children)
    /// Index is computed as: sum over dims of (first_bit[dim] << (K-1-dim))
    root_children: Vec<Option<Box<Node<T>>>>,
    /// Number of projected dimensions (K)
    proj_dims: usize,
    /// Breakpoints for each dimension: [dim][0..257]
    /// breakpoints[dim][i] is the i-th breakpoint for dimension dim
    breakpoints: Vec<Vec<T>>,
    /// Projected points flattened: [point_idx * proj_dims + dim]
    projected_points: Vec<T>,
    /// Pre-computed encodings: [point_idx * proj_dims + dim] -> 0..255
    encodings: Vec<u8>,
}

//////////////////////
// Helper Functions //
//////////////////////

/// Select breakpoints using QuickSelect algorithm (Algorithm 1 from paper)
///
/// Samples points and finds breakpoints that divide each dimension into
/// NUM_REGIONS regions with approximately equal point counts.
///
/// ### Params
///
/// * `projected_flat` - Flattened projected points [n * proj_dims]
/// * `n` - Number of points
/// * `proj_dims` - Number of projected dimensions
///
/// ### Returns
///
/// Breakpoints for each dimension: Vec<Vec<T>> where [dim][0..257]
fn select_breakpoints<T>(projected_flat: &[T], n: usize, proj_dims: usize) -> Vec<Vec<T>>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync,
{
    // Determine sample size (10% of data, minimum 1000, maximum n)
    let sample_size = ((n as f64 * SAMPLE_RATIO) as usize)
        .max(MIN_SAMPLE_SIZE)
        .min(n);

    let mut breakpoints = Vec::with_capacity(proj_dims);

    for dim in 0..proj_dims {
        // Extract values for this dimension from all points
        let mut values: Vec<T> = (0..n)
            .map(|i| projected_flat[i * proj_dims + dim])
            .collect();

        // If we're sampling, take a random subset
        if sample_size < n {
            let mut rng = StdRng::seed_from_u64((dim * 12345) as u64);
            values.shuffle(&mut rng);
            values.truncate(sample_size);
        }

        let ns = values.len();

        // Use divide-and-conquer with QuickSelect to find breakpoints
        // We need to find values at positions that divide into NUM_REGIONS equal parts
        let mut dim_breakpoints = vec![T::zero(); NUM_BREAKPOINTS];

        // Find breakpoints using iterative QuickSelect
        // Positions we need: 0, ns/256, 2*ns/256, ..., 255*ns/256, ns-1
        let region_size = ns / NUM_REGIONS;

        if region_size == 0 {
            // Too few points - use min/max and interpolate
            let min_val = values.iter().cloned().fold(T::infinity(), |a, b| a.min(b));
            let max_val = values
                .iter()
                .cloned()
                .fold(T::neg_infinity(), |a, b| a.max(b));
            let range = max_val - min_val;

            for i in 0..NUM_BREAKPOINTS {
                let frac = T::from_usize(i).unwrap() / T::from_usize(NUM_REGIONS).unwrap();
                dim_breakpoints[i] = min_val + range * frac;
            }
        } else {
            // Sort the sampled values (QuickSelect would be faster but sort is simpler
            // and we only do this once during index construction)
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // First breakpoint is minimum
            dim_breakpoints[0] = values[0];

            // Middle breakpoints at quantile positions
            for i in 1..NUM_REGIONS {
                let pos = (i * ns) / NUM_REGIONS;
                dim_breakpoints[i] = values[pos.min(ns - 1)];
            }

            // Last breakpoint is maximum (plus small epsilon for inclusive upper bound)
            dim_breakpoints[NUM_REGIONS] = values[ns - 1];
        }

        // Ensure breakpoints are strictly increasing (handle duplicates)
        for i in 1..NUM_BREAKPOINTS {
            if dim_breakpoints[i] <= dim_breakpoints[i - 1] {
                dim_breakpoints[i] = dim_breakpoints[i - 1] + T::epsilon();
            }
        }

        breakpoints.push(dim_breakpoints);
    }

    breakpoints
}

/// Encode a single value using binary search against breakpoints
///
/// Returns region index 0..255
#[inline]
fn encode_value<T: Float>(value: T, breakpoints: &[T]) -> u8 {
    // Binary search to find which region the value falls into
    // Region i contains values in [breakpoints[i], breakpoints[i+1])

    let mut lo = 0usize;
    let mut hi = NUM_REGIONS; // exclusive upper bound

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if value < breakpoints[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    // lo is now the first breakpoint > value, so region is lo - 1
    // Clamp to valid range [0, 255]
    lo.saturating_sub(1).min(NUM_REGIONS - 1) as u8
}

/// Encode all points for one tree (Algorithm 2 from paper)
///
/// ### Params
///
/// * `projected_flat` - Projected points [n * proj_dims]
/// * `n` - Number of points
/// * `proj_dims` - Number of dimensions
/// * `breakpoints` - Breakpoints per dimension [dim][0..257]
///
/// ### Returns
///
/// Encodings flattened: [point_idx * proj_dims + dim] -> 0..255
fn encode_points<T: Float>(
    projected_flat: &[T],
    n: usize,
    proj_dims: usize,
    breakpoints: &[Vec<T>],
) -> Vec<u8> {
    let mut encodings = vec![0u8; n * proj_dims];

    for point_idx in 0..n {
        let base = point_idx * proj_dims;
        for dim in 0..proj_dims {
            let value = projected_flat[base + dim];
            encodings[base + dim] = encode_value(value, &breakpoints[dim]);
        }
    }

    encodings
}

/// Get the first bit (MSB) of an 8-bit encoding
#[inline]
fn first_bit(encoding: u8) -> bool {
    (encoding & 0x80) != 0
}

/// Get a specific bit from an 8-bit encoding (bit 0 = MSB, bit 7 = LSB)
#[inline]
fn get_bit(encoding: u8, bit_pos: usize) -> bool {
    debug_assert!(bit_pos < 8);
    (encoding & (0x80 >> bit_pos)) != 0
}

/// Compute root child index from first bits of each dimension
///
/// Index = sum of first_bit[dim] * 2^(K-1-dim)
#[inline]
fn root_child_index(encodings: &[u8], proj_dims: usize) -> usize {
    let mut index = 0usize;
    for dim in 0..proj_dims {
        if first_bit(encodings[dim]) {
            index |= 1 << (proj_dims - 1 - dim);
        }
    }
    index
}

/// Compute bounds for a set of points from their actual projected coordinates
///
/// This gives tight bounds based on actual point positions.
fn compute_bounds_from_points<T: Float>(
    point_indices: &[usize],
    projected_flat: &[T],
    proj_dims: usize,
) -> Vec<(T, T)> {
    if point_indices.is_empty() {
        return vec![(T::zero(), T::zero()); proj_dims];
    }

    let mut bounds = vec![(T::infinity(), T::neg_infinity()); proj_dims];

    for &idx in point_indices {
        let base = idx * proj_dims;
        for dim in 0..proj_dims {
            let val = projected_flat[base + dim];
            bounds[dim].0 = bounds[dim].0.min(val);
            bounds[dim].1 = bounds[dim].1.max(val);
        }
    }

    bounds
}

/// Build a subtree from a set of points (Algorithm 3 from paper)
///
/// Recursively splits nodes by choosing the dimension that gives
/// the most even split, adding one more bit of precision.
fn build_subtree<T: Float>(
    point_indices: Vec<usize>,
    encodings: &[u8],
    projected_flat: &[T],
    proj_dims: usize,
    max_leaf_size: usize,
    current_bits: &[usize], // How many bits revealed per dimension (0-8)
) -> Node<T> {
    // Base case: small enough for leaf
    if point_indices.len() <= max_leaf_size {
        let bounds = compute_bounds_from_points(&point_indices, projected_flat, proj_dims);
        return Node::Leaf {
            points: point_indices,
            bounds,
        };
    }

    // Find best dimension to split (most even division)
    let mut best_dim = 0;
    let mut best_balance = usize::MAX;
    let mut best_bit = 0;

    for dim in 0..proj_dims {
        let bits_used = current_bits[dim];
        if bits_used >= 8 {
            continue; // Already at maximum precision
        }

        let next_bit = bits_used;

        // Count points that would go left (bit = 0) vs right (bit = 1)
        let mut left_count = 0;
        for &idx in &point_indices {
            let enc = encodings[idx * proj_dims + dim];
            if !get_bit(enc, next_bit) {
                left_count += 1;
            }
        }
        let right_count = point_indices.len() - left_count;

        // Skip if all points go one way (can't split)
        if left_count == 0 || right_count == 0 {
            continue;
        }

        let balance = left_count.abs_diff(right_count);
        if balance < best_balance {
            best_balance = balance;
            best_dim = dim;
            best_bit = next_bit;
        }
    }

    // If we can't split (all dimensions exhausted or all points identical), make leaf
    if best_balance == usize::MAX {
        let bounds = compute_bounds_from_points(&point_indices, projected_flat, proj_dims);
        return Node::Leaf {
            points: point_indices,
            bounds,
        };
    }

    // Partition points
    let mut left_points = Vec::new();
    let mut right_points = Vec::new();

    for idx in point_indices {
        let enc = encodings[idx * proj_dims + best_dim];
        if !get_bit(enc, best_bit) {
            left_points.push(idx);
        } else {
            right_points.push(idx);
        }
    }

    // Update bits used for recursive calls
    let mut left_bits = current_bits.to_vec();
    let mut right_bits = current_bits.to_vec();
    left_bits[best_dim] = best_bit + 1;
    right_bits[best_dim] = best_bit + 1;

    // Build children
    let left = Box::new(build_subtree(
        left_points,
        encodings,
        projected_flat,
        proj_dims,
        max_leaf_size,
        &left_bits,
    ));

    let right = Box::new(build_subtree(
        right_points,
        encodings,
        projected_flat,
        proj_dims,
        max_leaf_size,
        &right_bits,
    ));

    // Compute bounds as union of children
    let left_bounds = match left.as_ref() {
        Node::Internal { bounds, .. } => bounds,
        Node::Leaf { bounds, .. } => bounds,
    };
    let right_bounds = match right.as_ref() {
        Node::Internal { bounds, .. } => bounds,
        Node::Leaf { bounds, .. } => bounds,
    };

    let mut bounds = Vec::with_capacity(proj_dims);
    for dim in 0..proj_dims {
        bounds.push((
            left_bounds[dim].0.min(right_bounds[dim].0),
            left_bounds[dim].1.max(right_bounds[dim].1),
        ));
    }

    Node::Internal {
        split_dim: best_dim,
        split_bit: best_bit,
        left,
        right,
        bounds,
    }
}

/// Build a DE-Tree from encoded points (Algorithm 3 from paper)
fn build_tree<T: Float>(
    n: usize,
    encodings: &[u8],
    projected_flat: &[T],
    proj_dims: usize,
    max_leaf_size: usize,
    breakpoints: Vec<Vec<T>>,
) -> DETree<T> {
    let num_root_children = 1usize << proj_dims;
    let mut root_children: Vec<Option<Box<Node<T>>>> =
        (0..num_root_children).map(|_| None).collect();

    // Partition points by first bit of each dimension
    let mut child_points: Vec<Vec<usize>> = vec![Vec::new(); num_root_children];

    for idx in 0..n {
        let base = idx * proj_dims;
        let enc_slice = &encodings[base..base + proj_dims];
        let child_idx = root_child_index(enc_slice, proj_dims);
        child_points[child_idx].push(idx);
    }

    // Build each root child
    let initial_bits = vec![1usize; proj_dims]; // First bit already used for root

    for (child_idx, points) in child_points.into_iter().enumerate() {
        if points.is_empty() {
            continue;
        }

        let node = build_subtree(
            points,
            encodings,
            projected_flat,
            proj_dims,
            max_leaf_size,
            &initial_bits,
        );

        root_children[child_idx] = Some(Box::new(node));
    }

    DETree {
        root_children,
        proj_dims,
        breakpoints,
        projected_points: projected_flat.to_vec(),
        encodings: encodings.to_vec(),
    }
}

/// Orthogonalise random projections using Gram-Schmidt
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

            // Orthogonalise against previous vectors
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

            // Normalise
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
                // Re-initialise with new random vector
                let reinit_seed = seed.wrapping_mul(table_idx + 1).wrapping_mul(i + 1);
                let mut rng = StdRng::seed_from_u64(reinit_seed as u64);
                for d in 0..dim {
                    let val: f64 = rng.sample(StandardNormal);
                    projections[i_base + d] = T::from_f64(val).unwrap();
                }

                // Re-orthogonalise
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
                    "Orthogonalisation failed after reinitialisation"
                );

                for d in 0..dim {
                    projections[i_base + d] = projections[i_base + d] / norm;
                }
            }
        }
    }
}

/// Compute lower bound distance from query to bounding box
#[inline]
fn lower_bound_distance<T: Float>(bounds: &[(T, T)], query: &[T]) -> T {
    let mut sum_sq = T::zero();

    for (dim, &(min, max)) in bounds.iter().enumerate() {
        let q = query[dim];
        if q < min {
            let diff = min - q;
            sum_sq = sum_sq + diff * diff;
        } else if q > max {
            let diff = q - max;
            sum_sq = sum_sq + diff * diff;
        }
    }

    sum_sq.sqrt()
}

/// Compute upper bound distance from query to bounding box
#[inline]
fn upper_bound_distance<T: Float>(bounds: &[(T, T)], query: &[T]) -> T {
    let mut sum_sq = T::zero();

    for (dim, &(min, max)) in bounds.iter().enumerate() {
        let q = query[dim];
        let dist_to_min = (q - min).abs();
        let dist_to_max = (q - max).abs();
        let max_dist = dist_to_min.max(dist_to_max);
        sum_sq = sum_sq + max_dist * max_dist;
    }

    sum_sq.sqrt()
}

/// Euclidean distance in projected space
#[inline]
fn projected_distance<T>(projected_flat: &[T], point_idx: usize, query: &[T], proj_dims: usize) -> T
where
    T: Float + SimdDistance,
{
    let base = point_idx * proj_dims;
    let point = &projected_flat[base..base + proj_dims];
    T::euclidean_simd(point, query).sqrt()
}

/// Range query on a node (Algorithm 5 from paper)
fn range_query_node<T>(
    node: &Node<T>,
    query: &[T],
    radius: T,
    projected_flat: &[T],
    proj_dims: usize,
    candidates: &mut Vec<usize>,
) where
    T: Float + SimdDistance,
{
    match node {
        Node::Leaf { points, bounds } => {
            let lower = lower_bound_distance(bounds, query);

            // Prune if lower bound exceeds radius
            if lower > radius {
                return;
            }

            let upper = upper_bound_distance(bounds, query);

            if upper <= radius {
                // All points in this leaf are within radius
                candidates.extend_from_slice(points);
            } else {
                // Check each point individually
                for &idx in points {
                    let dist = projected_distance(projected_flat, idx, query, proj_dims);
                    if dist <= radius {
                        candidates.push(idx);
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
            let lower = lower_bound_distance(bounds, query);

            // Prune if lower bound exceeds radius
            if lower > radius {
                return;
            }

            range_query_node(left, query, radius, projected_flat, proj_dims, candidates);
            range_query_node(right, query, radius, projected_flat, proj_dims, candidates);
        }
    }
}

/// Range query on a DE-Tree (Algorithm 4 from paper)
fn range_query<T>(tree: &DETree<T>, query: &[T], radius: T, candidates: &mut Vec<usize>)
where
    T: Float + SimdDistance,
{
    for child_opt in &tree.root_children {
        if let Some(child) = child_opt {
            range_query_node(
                child,
                query,
                radius,
                &tree.projected_points,
                tree.proj_dims,
                candidates,
            );
        }
    }
}

/// Estimate memory used by a node
fn node_memory<T>(node: &Node<T>) -> usize {
    match node {
        Node::Leaf { points, bounds } => {
            std::mem::size_of::<Node<T>>()
                + points.capacity() * std::mem::size_of::<usize>()
                + bounds.capacity() * std::mem::size_of::<(T, T)>()
        }
        Node::Internal {
            left,
            right,
            bounds,
            ..
        } => {
            std::mem::size_of::<Node<T>>()
                + bounds.capacity() * std::mem::size_of::<(T, T)>()
                + node_memory(left)
                + node_memory(right)
        }
    }
}

////////////////////
// LSHDETreeIndex //
////////////////////

/// LSH index using DE-Trees for approximate nearest neighbour search
///
/// Implements the DET-LSH algorithm from the paper with:
/// - Dynamic breakpoint selection based on data distribution
/// - 8-bit iSAX-style encoding per dimension
/// - Efficient range queries with bound-based pruning
///
/// ### Algorithm Overview
///
/// 1. **Projection**: Map D-dimensional vectors to K dimensions using random
///    Gaussian projections
/// 2. **Breakpoint Selection**: Sample 10% of projected points, find quantiles
///    to divide each dimension into 256 equal-population regions
/// 3. **Encoding**: Convert each projected coordinate to 8-bit region index
/// 4. **Tree Construction**: Build trees with 2^K root children, split by
///    choosing dimension with most even split
/// 5. **Query**: Range query with lower/upper bound pruning, deduplicate
///    across trees
///
/// ### Parameters
///
/// * `num_trees` (L in paper): More trees = better recall, slower queries.
///   Typical: 5-20
/// * `proj_dims` (K in paper): Projected dimensionality. Lower = faster but
///   less accurate. Typical: 8-12. Note: 2^K root children per tree.
/// * `max_leaf_size`: Points per leaf. Typical: 32-128
/// * `search_radius`: Radius in projected space. Rule of thumb: start with
///   sqrt(proj_dims) and adjust based on recall requirements
pub struct LSHDETreeIndex<T> {
    /// Original vectors flattened [n * dim]
    pub vectors_flat: Vec<T>,
    /// Original dimensionality
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Pre-computed norms for cosine distance
    norms: Vec<T>,
    /// Distance metric
    metric: Dist,
    /// DE-Trees (one per hash table)
    trees: Vec<DETree<T>>,
    /// Random projections flattened [num_trees * proj_dims * dim]
    projections: Vec<T>,
    /// Number of trees
    num_trees: usize,
    /// Projected dimensionality
    proj_dims: usize,
}

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
{
    /// Construct a new LSH DE-Tree index
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (rows = samples, columns = dimensions)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `num_trees` - Number of trees (L). More = better recall, slower
    /// * `proj_dims` - Projected dimensionality (K). Max 16 due to 2^K root children
    /// * `max_leaf_size` - Maximum points per leaf node
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Constructed index
    pub fn new(
        data: MatRef<T>,
        metric: Dist,
        num_trees: usize,
        proj_dims: usize,
        max_leaf_size: Option<usize>,
        seed: usize,
    ) -> Self {
        let max_leaf_size = max_leaf_size.unwrap_or(MAX_LEAF_SIZE);

        assert!(max_leaf_size >= 1, "max_leaf_size must be >= 1");
        assert!(
            proj_dims <= 16,
            "proj_dims > 16 creates 2^{} = {} root children (too many)",
            proj_dims,
            1usize << proj_dims
        );
        assert!(num_trees >= 1, "num_trees must be >= 1");
        assert!(proj_dims >= 1, "proj_dims must be >= 1");

        let (vectors_flat, n, dim) = matrix_to_flat(data);

        // Pre-compute norms for cosine distance
        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    T::calculate_l2_norm(&vectors_flat[start..start + dim])
                })
                .collect()
        } else {
            Vec::new()
        };

        // Generate random projections
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let total_proj = num_trees * proj_dims * dim;
        let mut projections: Vec<T> = (0..total_proj)
            .map(|_| T::from_f64(rng.sample(StandardNormal)).unwrap())
            .collect();

        orthogonalise_projections(&mut projections, num_trees, proj_dims, dim, seed);

        // Build trees in parallel
        let trees: Vec<DETree<T>> = (0..num_trees)
            .into_par_iter()
            .map(|tree_idx| {
                // Project all points
                let mut projected = vec![T::zero(); n * proj_dims];
                let proj_base = tree_idx * proj_dims * dim;

                for vec_idx in 0..n {
                    let vec_start = vec_idx * dim;
                    let vec = &vectors_flat[vec_start..vec_start + dim];

                    for k in 0..proj_dims {
                        let proj_offset = proj_base + k * dim;
                        let proj_vec = &projections[proj_offset..proj_offset + dim];
                        let dot = T::dot_simd(vec, proj_vec);
                        projected[vec_idx * proj_dims + k] = dot;
                    }
                }

                // Select breakpoints from projected data
                let breakpoints = select_breakpoints(&projected, n, proj_dims);

                // Encode all points
                let encodings = encode_points(&projected, n, proj_dims, &breakpoints);

                // Build tree
                build_tree(
                    n,
                    &encodings,
                    &projected,
                    proj_dims,
                    max_leaf_size,
                    breakpoints,
                )
            })
            .collect();

        Self {
            vectors_flat,
            dim,
            n,
            norms,
            metric,
            trees,
            projections,
            num_trees,
            proj_dims,
        }
    }

    /// Query for k nearest neighbours with fixed search radius
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
        assert_eq!(
            query_vec.len(),
            self.dim,
            "Query dimensionality mismatch: expected {}, got {}",
            self.dim,
            query_vec.len()
        );

        if k == 0 {
            return (Vec::new(), Vec::new());
        }

        // Pre-compute all query projections (avoids redundant work in loop)
        let query_projections: Vec<Vec<T>> = (0..self.num_trees)
            .map(|tree_idx| {
                let proj_base = tree_idx * self.proj_dims * self.dim;
                (0..self.proj_dims)
                    .map(|k_dim| {
                        let offset = proj_base + k_dim * self.dim;
                        let proj_vec = &self.projections[offset..offset + self.dim];
                        T::dot_simd(query_vec, proj_vec)
                    })
                    .collect()
            })
            .collect();

        let mut candidates = Vec::new();
        let mut seen = FxHashSet::default();
        let target_candidates = k * 10;

        // Collect candidates from all trees
        for (tree_idx, tree) in self.trees.iter().enumerate() {
            let before_len = candidates.len();
            range_query(
                tree,
                &query_projections[tree_idx],
                search_radius,
                &mut candidates,
            );

            // Deduplicate new candidates
            let mut write_idx = before_len;
            for read_idx in before_len..candidates.len() {
                if seen.insert(candidates[read_idx]) {
                    if write_idx != read_idx {
                        candidates[write_idx] = candidates[read_idx];
                    }
                    write_idx += 1;
                }
            }
            candidates.truncate(write_idx);

            // Early exit if we have enough candidates
            if candidates.len() >= target_candidates {
                break;
            }
        }

        // Compute exact distances and find top-k
        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::new();

        match self.metric {
            Dist::Euclidean => {
                for &idx in &candidates {
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
                let query_norm = T::calculate_l2_norm(query_vec);
                for &idx in &candidates {
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

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let indices = results.iter().map(|&(_, idx)| idx).collect();
        let dists = results.iter().map(|&(OrderedFloat(d), _)| d).collect();

        (indices, dists)
    }

    /// Query with adaptive radius expansion (incremental)
    ///
    /// Processes trees incrementally, accumulating candidates across radius
    /// expansions. Much faster than naive approach of restarting each iteration.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours
    /// * `initial_radius` - Starting search radius
    /// * `expansion_factor` - Multiplier per iteration (must be > 1)
    /// * `max_radius` - Maximum radius
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    pub fn query_adaptive(
        &self,
        query_vec: &[T],
        k: usize,
        initial_radius: T,
        expansion_factor: T,
        max_radius: T,
    ) -> (Vec<usize>, Vec<T>) {
        assert_eq!(query_vec.len(), self.dim, "Query dimensionality mismatch");
        assert!(expansion_factor > T::one(), "expansion_factor must be > 1");

        if k == 0 {
            return (Vec::new(), Vec::new());
        }

        // Pre-project query for all trees (do this once)
        let mut query_projections: Vec<Vec<T>> = Vec::with_capacity(self.num_trees);
        for tree_idx in 0..self.num_trees {
            let proj_base = tree_idx * self.proj_dims * self.dim;
            let mut query_proj = Vec::with_capacity(self.proj_dims);
            for k_dim in 0..self.proj_dims {
                let offset = proj_base + k_dim * self.dim;
                let proj_vec = &self.projections[offset..offset + self.dim];
                let dot = T::dot_simd(query_vec, proj_vec);
                query_proj.push(dot);
            }
            query_projections.push(query_proj);
        }

        let mut seen = FxHashSet::default();
        let mut candidates_with_dist: Vec<(usize, T)> = Vec::new();
        let mut radius = initial_radius;
        let target_candidates = k * 10;

        // Pre-compute query norm for cosine distance
        let query_norm = match self.metric {
            Dist::Cosine => T::calculate_l2_norm(query_vec),
            Dist::Euclidean => T::zero(),
        };

        loop {
            // Query all trees at current radius, collecting new candidates
            for (tree_idx, tree) in self.trees.iter().enumerate() {
                let mut tree_candidates = Vec::new();
                range_query(
                    tree,
                    &query_projections[tree_idx],
                    radius,
                    &mut tree_candidates,
                );

                // Add only new candidates
                for idx in tree_candidates {
                    if seen.insert(idx) {
                        // Compute exact distance for new candidate
                        let dist = match self.metric {
                            Dist::Euclidean => self.euclidean_distance_to_query(idx, query_vec),
                            Dist::Cosine => {
                                self.cosine_distance_to_query(idx, query_vec, query_norm)
                            }
                        };
                        candidates_with_dist.push((idx, dist));
                    }
                }

                // Early exit if we have enough candidates
                if candidates_with_dist.len() >= target_candidates {
                    break;
                }
            }

            // Check termination conditions
            if candidates_with_dist.len() >= k || radius >= max_radius || seen.len() == self.n {
                break;
            }

            // Expand radius for next iteration
            radius = (radius * expansion_factor).min(max_radius);
        }

        // Sort by distance and take top-k
        candidates_with_dist
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let result_len = candidates_with_dist.len().min(k);
        let indices: Vec<usize> = candidates_with_dist
            .iter()
            .take(result_len)
            .map(|&(idx, _)| idx)
            .collect();
        let distances: Vec<T> = candidates_with_dist
            .iter()
            .take(result_len)
            .map(|&(_, d)| d)
            .collect();

        (indices, distances)
    }

    /// Query using a matrix row reference
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        search_radius: T,
    ) -> (Vec<usize>, Vec<T>) {
        assert_eq!(
            query_row.ncols(),
            self.dim,
            "Query row dimensionality mismatch"
        );

        if query_row.col_stride() == 1 {
            let slice = unsafe { std::slice::from_raw_parts(query_row.as_ptr(), self.dim) };
            return self.query(slice, k, search_radius);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, search_radius)
    }

    /// Query row with adaptive radius
    pub fn query_row_adaptive(
        &self,
        query_row: RowRef<T>,
        k: usize,
        initial_radius: T,
        expansion_factor: T,
        max_radius: T,
    ) -> (Vec<usize>, Vec<T>) {
        assert_eq!(
            query_row.ncols(),
            self.dim,
            "Query row dimensionality mismatch"
        );

        if query_row.col_stride() == 1 {
            let slice = unsafe { std::slice::from_raw_parts(query_row.as_ptr(), self.dim) };
            return self.query_adaptive(slice, k, initial_radius, expansion_factor, max_radius);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query_adaptive(&query_vec, k, initial_radius, expansion_factor, max_radius)
    }

    /// Generate kNN graph from indexed vectors (optimised)
    ///
    /// Uses pre-computed projections stored in trees, avoiding redundant
    /// projection computation during self-query.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per vector
    /// * `initial_radius` - Starting search radius
    /// * `expansion_factor` - Radius multiplier
    /// * `max_radius` - Maximum radius
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional_distances)`
    pub fn generate_knn(
        &self,
        k: usize,
        initial_radius: T,
        expansion_factor: T,
        max_radius: T,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|query_idx| {
                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count % 100_000 == 0 {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            self.n.separate_with_underscores()
                        );
                    }
                }

                // Use pre-computed projections from trees
                self.query_self_adaptive(query_idx, k, initial_radius, expansion_factor, max_radius)
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

    /// Self-query using pre-computed projections (internal)
    ///
    /// Optimised path for querying indexed vectors against themselves.
    fn query_self_adaptive(
        &self,
        query_idx: usize,
        k: usize,
        initial_radius: T,
        expansion_factor: T,
        max_radius: T,
    ) -> (Vec<usize>, Vec<T>) {
        if k == 0 {
            return (Vec::new(), Vec::new());
        }

        let query_vec_start = query_idx * self.dim;
        let query_vec = &self.vectors_flat[query_vec_start..query_vec_start + self.dim];

        // Pre-compute query norm for cosine distance
        let query_norm = match self.metric {
            Dist::Cosine => T::calculate_l2_norm(query_vec),
            Dist::Euclidean => T::zero(),
        };

        let mut seen = FxHashSet::default();
        let mut candidates_with_dist: Vec<(usize, T)> = Vec::new();
        let mut radius = initial_radius;
        let target_candidates = k * 10;

        loop {
            for tree in &self.trees {
                // Use pre-computed projection from tree
                let proj_start = query_idx * self.proj_dims;
                let query_proj = &tree.projected_points[proj_start..proj_start + self.proj_dims];

                let mut tree_candidates = Vec::new();
                range_query(tree, query_proj, radius, &mut tree_candidates);

                for idx in tree_candidates {
                    if idx != query_idx && seen.insert(idx) {
                        let dist = match self.metric {
                            Dist::Euclidean => self.euclidean_distance_to_query(idx, query_vec),
                            Dist::Cosine => {
                                self.cosine_distance_to_query(idx, query_vec, query_norm)
                            }
                        };
                        candidates_with_dist.push((idx, dist));
                    }
                }

                if candidates_with_dist.len() >= target_candidates {
                    break;
                }
            }

            if candidates_with_dist.len() >= k || radius >= max_radius || seen.len() >= self.n - 1 {
                break;
            }

            radius = (radius * expansion_factor).min(max_radius);
        }

        candidates_with_dist
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let result_len = candidates_with_dist.len().min(k);
        let indices: Vec<usize> = candidates_with_dist
            .iter()
            .take(result_len)
            .map(|&(idx, _)| idx)
            .collect();
        let distances: Vec<T> = candidates_with_dist
            .iter()
            .take(result_len)
            .map(|&(_, d)| d)
            .collect();

        (indices, distances)
    }

    /// Estimate search radius from original-space radius
    ///
    /// Uses the relationship: projected_radius ≈ original_radius * sqrt(K / D)
    /// with a safety factor for variance.
    ///
    /// ### Params
    ///
    /// * `original_radius` - Radius in original D-dimensional space
    /// * `safety_factor` - Multiplier to account for variance (1.5-2.0 typical)
    ///
    /// ### Returns
    ///
    /// Estimated radius for projected space
    pub fn estimate_projected_radius(&self, original_radius: T, safety_factor: T) -> T {
        let ratio = T::from_usize(self.proj_dims).unwrap() / T::from_usize(self.dim).unwrap();
        original_radius * ratio.sqrt() * safety_factor
    }

    /// Memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);

        total += self.vectors_flat.capacity() * std::mem::size_of::<T>();
        total += self.norms.capacity() * std::mem::size_of::<T>();
        total += self.projections.capacity() * std::mem::size_of::<T>();
        total += self.trees.capacity() * std::mem::size_of::<DETree<T>>();

        for tree in &self.trees {
            total += tree.projected_points.capacity() * std::mem::size_of::<T>();
            total += tree.encodings.capacity();
            total += tree.root_children.capacity() * std::mem::size_of::<Option<Box<Node<T>>>>();

            for bp in &tree.breakpoints {
                total += bp.capacity() * std::mem::size_of::<T>();
            }

            for child_opt in &tree.root_children {
                if let Some(child) = child_opt {
                    total += node_memory(child);
                }
            }
        }

        total
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
        Mat::from_fn(5, 3, |i, j| match i {
            0 => [1.0, 0.0, 0.0][j],
            1 => [0.0, 1.0, 0.0][j],
            2 => [0.0, 0.0, 1.0][j],
            3 => [1.0, 1.0, 0.0][j],
            4 => [0.5, 0.5, 0.7][j],
            _ => 0.0,
        })
    }

    fn larger_test_data(n: usize, dim: usize) -> Mat<f32> {
        Mat::from_fn(n, dim, |i, j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
    }

    #[test]
    fn test_encode_value() {
        // Breakpoints evenly spaced from 0 to 256
        let breakpoints: Vec<f32> = (0..=NUM_REGIONS).map(|i| i as f32).collect();

        assert_eq!(encode_value(0.5f32, &breakpoints), 0);
        assert_eq!(encode_value(1.5f32, &breakpoints), 1);
        assert_eq!(encode_value(127.5f32, &breakpoints), 127);
        assert_eq!(encode_value(255.5f32, &breakpoints), 255);

        // Edge cases
        assert_eq!(encode_value(-1.0f32, &breakpoints), 0);
        assert_eq!(encode_value(1000.0f32, &breakpoints), 255);
    }

    #[test]
    fn test_bit_operations() {
        assert!(first_bit(0b10000000));
        assert!(!first_bit(0b01111111));

        assert!(get_bit(0b10000000, 0));
        assert!(!get_bit(0b10000000, 1));
        assert!(get_bit(0b01000000, 1));
        assert!(get_bit(0b00000001, 7));
    }

    #[test]
    fn test_root_child_index() {
        // 2 dimensions
        let enc1 = [0b10000000u8, 0b00000000u8]; // first bits: 1, 0
        assert_eq!(root_child_index(&enc1, 2), 0b10);

        let enc2 = [0b00000000u8, 0b10000000u8]; // first bits: 0, 1
        assert_eq!(root_child_index(&enc2, 2), 0b01);

        let enc3 = [0b10000000u8, 0b10000000u8]; // first bits: 1, 1
        assert_eq!(root_child_index(&enc3, 2), 0b11);
    }

    #[test]
    fn test_select_breakpoints() {
        let projected: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let breakpoints = select_breakpoints(&projected, 1000, 1);

        assert_eq!(breakpoints.len(), 1);
        assert_eq!(breakpoints[0].len(), NUM_BREAKPOINTS);

        // Breakpoints should be increasing
        for i in 1..NUM_BREAKPOINTS {
            assert!(
                breakpoints[0][i] > breakpoints[0][i - 1],
                "Breakpoints not increasing at {}",
                i
            );
        }
    }

    #[test]
    fn test_index_creation_euclidean() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.num_trees, 4);
        assert_eq!(index.proj_dims, 2);
        assert_eq!(index.trees.len(), 4);

        // Check tree structure
        for tree in &index.trees {
            assert_eq!(tree.proj_dims, 2);
            assert_eq!(tree.root_children.len(), 4); // 2^2
            assert_eq!(tree.breakpoints.len(), 2);
            assert_eq!(tree.projected_points.len(), 5 * 2);
            assert_eq!(tree.encodings.len(), 5 * 2);
        }
    }

    #[test]
    fn test_index_creation_cosine() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Cosine, 4, 2, Some(2), 42);

        assert_eq!(index.n, 5);
        assert_eq!(index.norms.len(), 5);
    }

    #[test]
    fn test_basic_query() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);

        let query = vec![1.0f32, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, 5.0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());

        // Distances should be sorted
        for i in 1..distances.len() {
            assert!(
                distances[i - 1] <= distances[i],
                "Distances not sorted at {}",
                i
            );
        }
    }

    #[test]
    fn test_query_finds_exact_match() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 8, 2, Some(2), 42);

        // Query for a point that exists in the data
        let query = vec![1.0f32, 0.0, 0.0]; // Point 0
        let (indices, distances) = index.query(&query, 1, 10.0);

        assert!(!indices.is_empty());
        // First result should be very close (ideally exact match)
        assert!(
            distances[0] < 0.01,
            "Expected near-zero distance for exact match"
        );
    }

    #[test]
    fn test_query_adaptive() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);

        let query = vec![1.0f32, 0.0, 0.0];
        let (indices, distances) = index.query_adaptive(&query, 3, 0.1, 2.0, 20.0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_query_adaptive_terminates() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);

        let query = vec![1.0f32, 0.0, 0.0];
        // Ask for more than available
        let (indices, _) = index.query_adaptive(&query, 100, 0.1, 2.0, 1000.0);

        assert!(indices.len() <= 5);
    }

    #[test]
    fn test_query_cosine() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Cosine, 4, 2, Some(2), 42);

        let query = vec![2.0f32, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 2, 5.0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 2);
    }

    #[test]
    fn test_query_row() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);

        let query_mat = Mat::from_fn(1, 3, |_, j| [1.0f32, 0.0, 0.0][j]);
        let (indices, distances) = index.query_row(query_mat.row(0), 3, 5.0);

        assert!(!indices.is_empty());
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_k_larger_than_n() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);

        let query = vec![1.0f32, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 100, 10.0);

        assert!(indices.len() <= 5);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_k_zero() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);

        let query = vec![1.0f32, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 0, 5.0);

        assert!(indices.is_empty());
        assert!(distances.is_empty());
    }

    #[test]
    #[should_panic(expected = "Query dimensionality mismatch")]
    fn test_dimension_mismatch() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);

        let query = vec![1.0f32, 0.0];
        index.query(&query, 3, 5.0);
    }

    #[test]
    #[should_panic(expected = "max_leaf_size must be >= 1")]
    fn test_invalid_max_leaf_size() {
        let mat = simple_test_data();
        LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(0), 42);
    }

    #[test]
    #[should_panic(expected = "expansion_factor must be > 1")]
    fn test_invalid_expansion_factor() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);

        let query = vec![1.0f32, 0.0, 0.0];
        index.query_adaptive(&query, 3, 1.0, 0.5, 10.0);
    }

    #[test]
    #[should_panic(expected = "proj_dims > 16")]
    fn test_proj_dims_too_large() {
        let mat = simple_test_data();
        LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 2, 17, Some(2), 42);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let mat = simple_test_data();

        let index1 = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);
        let index2 = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);

        let query = vec![1.0f32, 0.0, 0.0];
        let (indices1, _) = index1.query(&query, 3, 5.0);
        let (indices2, _) = index2.query(&query, 3, 5.0);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_f64() {
        let mat = Mat::from_fn(3, 3, |i, j| if i == j { 1.0f64 } else { 0.0f64 });
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);

        let query = vec![1.0f64, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 2, 5.0);

        assert!(!indices.is_empty());
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_no_duplicates() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 8, 2, Some(2), 42);

        let query = vec![1.0f32, 0.0, 0.0];
        let (indices, _) = index.query(&query, 5, 10.0);

        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();

        assert_eq!(indices.len(), sorted.len(), "Duplicates found in results");
    }

    #[test]
    fn test_larger_dataset() {
        let mat = larger_test_data(1000, 50);
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 10, 8, Some(32), 42);

        let query = vec![0.5f32; 50];
        let (indices, distances) = index.query(&query, 10, 5.0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 10);

        for &idx in &indices {
            assert!(idx < 1000);
        }

        // Check root children count
        for tree in &index.trees {
            assert_eq!(tree.root_children.len(), 256); // 2^8
        }
    }

    #[test]
    fn test_distances_sorted() {
        let mat = larger_test_data(100, 20);
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 5, 4, Some(8), 42);

        let query = vec![0.5f32; 20];
        let (_, distances) = index.query(&query, 20, 10.0);

        for i in 1..distances.len() {
            assert!(
                distances[i - 1] <= distances[i],
                "Distances not sorted at {}",
                i
            );
        }
    }

    #[test]
    fn test_small_radius_returns_fewer() {
        let mat = larger_test_data(100, 20);
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 5, 4, Some(8), 42);

        let query = vec![0.5f32; 20];

        let (indices_small, _) = index.query(&query, 50, 0.1);
        let (indices_large, _) = index.query(&query, 50, 10.0);

        assert!(
            indices_small.len() <= indices_large.len(),
            "Smaller radius should return <= candidates"
        );
    }

    #[test]
    fn test_estimate_projected_radius() {
        let mat = simple_test_data();
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 4, 2, Some(2), 42);

        let original_radius = 1.0f32;
        let projected = index.estimate_projected_radius(original_radius, 1.5);

        // With proj_dims=2, dim=3: sqrt(2/3) * 1.5 ≈ 1.22
        assert!(projected > 0.0);
        assert!(projected < original_radius * 2.0);
    }

    #[test]
    fn test_identical_points() {
        // All points identical
        let mat = Mat::from_fn(5, 10, |_, _| 1.0f32);
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 2, 3, Some(2), 42);

        let query = vec![1.0f32; 10];
        let (indices, _) = index.query(&query, 3, 5.0);

        assert!(!indices.is_empty());
    }

    #[test]
    fn test_memory_usage() {
        let mat = larger_test_data(100, 20);
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 5, 4, Some(8), 42);

        let mem = index.memory_usage_bytes();
        assert!(mem > 0);
        // Should be reasonable - at least vectors + projections
        let min_expected = 100 * 20 * 4 + 5 * 4 * 20 * 4;
        assert!(mem >= min_expected);
    }

    #[test]
    fn test_recall_on_clustered_data() {
        // Create clustered data where neighbours should be easy to find
        let mut data = Vec::new();
        for cluster in 0..10 {
            let base = cluster as f32 * 10.0;
            for _ in 0..10 {
                let point: Vec<f32> = (0..20).map(|d| base + (d as f32 * 0.01)).collect();
                data.extend(point);
            }
        }

        let mat = Mat::from_fn(100, 20, |i, j| data[i * 20 + j]);
        let index = LSHDETreeIndex::new(mat.as_ref(), Dist::Euclidean, 10, 6, Some(8), 42);

        // Query with a point from cluster 5
        let query: Vec<f32> = (0..20).map(|d| 50.0 + (d as f32 * 0.01)).collect();
        let (indices, _) = index.query_adaptive(&query, 5, 1.0, 2.0, 100.0);

        // At least some results should be from cluster 5 (indices 50-59)
        let cluster_5_count = indices.iter().filter(|&&i| i >= 50 && i < 60).count();
        assert!(
            cluster_5_count >= 1,
            "Expected to find neighbours from same cluster, found {} from cluster 5",
            cluster_5_count
        );
    }
}
