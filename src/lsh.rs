use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::prelude::*;
use rand::rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use std::cell::RefCell;
use std::cmp::Ord;
use std::collections::BinaryHeap;
use std::iter::Sum;

use crate::utils::dist::*;
use crate::utils::heap_structs::*;
use crate::utils::*;

////////////////
// Main index //
////////////////

/// LSH index for approximate nearest neighbour search
///
/// Uses multiple hash tables to partition the space. Each table applies a set
/// of random projections to map vectors to bit strings (hashes). Vectors with
/// identical hashes are stored in the same bucket.
///
/// ### Hash Functions
///
/// - **Cosine (SimHash)**: Each bit is the sign of a random hyperplane
///   projection
/// - **Euclidean (E2LSH variant)**: Encodes which projection has maximum
///   magnitude and its sign, providing locality preservation for L2 distance
///
/// ### Fields
///
/// * `vectors_flat` - Original data, flattened for cache efficiency
/// * `dim` - Embedding dimensionality
/// * `n` - Number of vectors
/// * `norms` - Pre-computed norms for Cosine (empty for Euclidean)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `hash_tables` - Maps hash values to vector indices for each table
/// * `random_vecs` - Random projection vectors (from N(0,1))
/// * `num_tables` - Number of hash tables
/// * `bits_per_hash` - Bits in each hash code (higher = fewer collisions)
pub struct LSHIndex<T> {
    // main fields
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    norms: Vec<T>,
    metric: Dist,
    // index-specific ones
    hash_tables: Vec<FxHashMap<u64, Vec<usize>>>,
    random_vecs: Vec<T>,
    num_tables: usize,
    bits_per_hash: usize,
}

/// VectorDistance trait
impl<T> VectorDistance<T> for LSHIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    /// Return the flat vectors
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    /// Return the original dimensions
    fn dim(&self) -> usize {
        self.dim
    }

    /// Return the normalised values for the Cosine calculation
    fn norms(&self) -> &[T] {
        &self.norms
    }
}

impl<T> LSHIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
    Self: LSHQuery<T>,
{
    /// Construct a new LSH index
    ///
    /// Builds hash tables in parallel. For each table, hashes all vectors
    /// using a different set of random projections and groups them into buckets.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (rows = samples, columns = dimensions)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `num_tables` - Number of hash tables (more = better recall, slower build/query)
    /// * `bits_per_hash` - Bits per hash code (more = fewer collisions, smaller buckets)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    pub fn new(
        data: MatRef<T>,
        metric: Dist,
        num_tables: usize,
        bits_per_hash: usize,
        seed: usize,
    ) -> Self {
        let n = data.nrows();
        let dim = data.ncols();

        // Flatten vectors for cache-friendly distance calculations
        let mut vectors_flat = Vec::with_capacity(n * dim);
        for i in 0..n {
            vectors_flat.extend(data.row(i).iter().cloned());
        }

        let norms = match metric {
            Dist::Cosine => (0..n)
                .map(|i| {
                    let vec_start = i * dim;
                    vectors_flat[vec_start..vec_start + dim]
                        .iter()
                        .map(|v| *v * *v)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt()
                })
                .collect(),
            Dist::Euclidean => Vec::new(),
        };

        // generate random vectors from N(0,1) for hash functions
        // Cosine: random hyperplanes
        // Euclidean: random projections
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let total_random_vecs = num_tables * bits_per_hash * dim;
        let random_vecs: Vec<T> = (0..total_random_vecs)
            .map(|_| {
                let val: f64 = rng.sample(StandardNormal);
                T::from_f64(val).unwrap()
            })
            .collect();

        let hash_tables: Vec<_> = (0..num_tables)
            .into_par_iter()
            .map(|table_idx| {
                let mut table = FxHashMap::default();

                for vec_idx in 0..n {
                    let vec_start = vec_idx * dim;
                    let vec = &vectors_flat[vec_start..vec_start + dim];

                    let hash = Self::compute_hash(
                        vec,
                        table_idx,
                        bits_per_hash,
                        dim,
                        &random_vecs,
                        &metric,
                    );

                    table.entry(hash).or_insert_with(Vec::new).push(vec_idx);
                }

                table
            })
            .collect();

        Self {
            vectors_flat,
            dim,
            n,
            norms,
            metric,
            hash_tables,
            random_vecs,
            num_tables,
            bits_per_hash,
        }
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Hashes the query vector and retrieves candidates from matching buckets
    /// across all tables. If no candidates found, falls back to random
    /// sampling.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `max_cand` - Optional limit on candidates (stops early after this
    ///   many)
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances, fallback_triggered)` sorted by distance
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        max_cand: Option<usize>,
    ) -> (Vec<usize>, Vec<T>, bool) {
        assert!(
            query_vec.len() == self.dim,
            "Query vector dimensionality mismatch"
        );

        self.query_internal(query_vec, k, max_cand)
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
    /// * `max_cand` - Optional candidate limit
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances, fallback_triggered)`
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        max_cand: Option<usize>,
    ) -> (Vec<usize>, Vec<T>, bool) {
        assert!(
            query_row.ncols() == self.dim,
            "Query row dimensionality mismatch"
        );

        // Fast path for contiguous row data
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, max_cand);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, max_cand)
    }

    /// Compute hash code for a vector
    ///
    /// Uses different schemes depending on metric:
    /// - **Cosine**: SimHash - each bit is sign of random hyperplane projection
    /// - **Euclidean**: Modified E2LSH - encodes which projection has maximum
    ///   magnitude and its sign
    ///
    /// ### Params
    ///
    /// * `vec` - Vector to hash
    /// * `table_idx` - Which hash table (selects random projections)
    /// * `bits_per_hash` - Number of bits in output hash
    /// * `dim` - Dimensionality
    /// * `random_vecs` - Pool of random projection vectors
    /// * `metric` - Distance metric
    ///
    /// ### Returns
    ///
    /// Hash code as u64 (only lower `bits_per_hash` bits used)
    #[inline]
    fn compute_hash(
        vec: &[T],
        table_idx: usize,
        bits_per_hash: usize,
        dim: usize,
        random_vecs: &[T],
        metric: &Dist,
    ) -> u64 {
        let mut hash: u64 = 0;
        let random_base = table_idx * bits_per_hash * dim;

        match metric {
            // SimHash
            Dist::Cosine => {
                for bit_idx in 0..bits_per_hash {
                    let offset = random_base + bit_idx * dim;

                    let mut dot = T::zero();
                    for d in 0..dim {
                        dot = dot + vec[d] * random_vecs[offset + d];
                    }

                    // interesting syntax to do bit manipulations!
                    if dot >= T::zero() {
                        hash |= 1u64 << bit_idx;
                    }
                }
            }
            // E2LSH: hash = floor((a * x + b) / w) for each projection
            // Combine multiple projections with XOR for distribution
            Dist::Euclidean => {
                // For each group of dimensions, find the projection with max absolute value
                let group_size = bits_per_hash.div_ceil(2); // bits needed: index + sign

                for group_idx in 0..group_size {
                    let offset = random_base + group_idx * dim;

                    let mut max_idx = 0;
                    let mut max_abs_val = T::zero();
                    let mut max_sign_positive = true;

                    // Check a small set of random projections (e.g., 3-4)
                    for proj_idx in 0..3.min(bits_per_hash - group_idx * 2) {
                        let proj_offset = offset + proj_idx * dim / 3;

                        let mut dot = T::zero();
                        for d in 0..dim {
                            dot = dot
                                + vec[d]
                                    * random_vecs
                                        [(proj_offset + d) % (random_base + bits_per_hash * dim)];
                        }

                        let abs_val = dot.abs();
                        if abs_val > max_abs_val {
                            max_abs_val = abs_val;
                            max_idx = proj_idx;
                            max_sign_positive = dot >= T::zero();
                        }
                    }

                    // Encode: which projection was max (2 bits) + sign (1 bit)
                    hash |= (max_idx as u64) << (group_idx * 3);
                    if max_sign_positive {
                        hash |= 1u64 << (group_idx * 3 + 2);
                    }
                }
            }
        }

        hash
    }
}

//////////////
// LSHQuery //
//////////////

thread_local! {
    /// Candidates in a RefCell
    static LSH_CANDIDATES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    /// HashSet per thread
    static LSH_SEEN_SET: RefCell<FxHashSet<usize>> = RefCell::new(FxHashSet::default());
    /// Heap for f32
    static LSH_HEAP_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize)>> = const { RefCell::new(BinaryHeap::new()) };
    /// Heap for f64
    static LSH_HEAP_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize)>> = const { RefCell::new(BinaryHeap::new()) };
}

/// Query interface for LSH using thread-local storage
///
/// Implemented separately for f32 and f64 to use type-specific thread locals
pub trait LSHQuery<T> {
    /// Execute a query using thread-local buffers
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `max_cand` - Optional limit on candidates examined
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances, fallback_triggered)`. The fallback flag
    /// indicates whether random sampling was used due to empty buckets.
    fn query_internal(
        &self,
        query_vec: &[T],
        k: usize,
        max_cand: Option<usize>,
    ) -> (Vec<usize>, Vec<T>, bool);
}

/////////
// f32 //
/////////

impl LSHQuery<f32> for LSHIndex<f32> {
    fn query_internal(
        &self,
        query_vec: &[f32],
        k: usize,
        max_cand: Option<usize>,
    ) -> (Vec<usize>, Vec<f32>, bool) {
        LSH_CANDIDATES.with(|cand_cell| {
            LSH_HEAP_F32.with(|heap_cell| {
                LSH_SEEN_SET.with(|seen_cell| {
                    let mut cand = cand_cell.borrow_mut();
                    let mut heap = heap_cell.borrow_mut();
                    let mut seen = seen_cell.borrow_mut();

                    if k == 0 {
                        return (Vec::new(), Vec::new(), false);
                    }

                    cand.clear();
                    heap.clear();
                    seen.clear();

                    // pre-reserve
                    cand.reserve(10000);

                    for table_idx in 0..self.num_tables {
                        let hash = Self::compute_hash(
                            query_vec,
                            table_idx,
                            self.bits_per_hash,
                            self.dim,
                            &self.random_vecs,
                            &self.metric,
                        );

                        if let Some(bucket) = self.hash_tables[table_idx].get(&hash) {
                            cand.extend_from_slice(bucket);

                            if let Some(max) = max_cand {
                                if cand.len() >= max {
                                    break;
                                }
                            }
                        }
                    }

                    let fallback_triggered = cand.is_empty();
                    if fallback_triggered {
                        let mut rng = rng();
                        let sample_size = 1000.min(self.n);
                        cand.extend((0..self.n).choose_multiple(&mut rng, sample_size));
                    }

                    // Deduplicate during distance computation
                    match self.metric {
                        Dist::Euclidean => {
                            for &idx in cand.iter() {
                                if seen.insert(idx) {
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
                        }
                        Dist::Cosine => {
                            let query_norm = query_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
                            for &idx in cand.iter() {
                                if seen.insert(idx) {
                                    let d =
                                        self.cosine_distance_to_query(idx, query_vec, query_norm);
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
                    }

                    let mut results: Vec<_> = heap.drain().collect();
                    results.sort_unstable_by(|a, b| a.0.cmp(&b.0));

                    let indices = results.iter().map(|&(_, idx)| idx).collect();
                    let dists = results.iter().map(|&(OrderedFloat(d), _)| d).collect();

                    (indices, dists, fallback_triggered)
                })
            })
        })
    }
}

/////////
// f64 //
/////////

impl LSHQuery<f64> for LSHIndex<f64> {
    fn query_internal(
        &self,
        query_vec: &[f64],
        k: usize,
        max_cand: Option<usize>,
    ) -> (Vec<usize>, Vec<f64>, bool) {
        LSH_CANDIDATES.with(|cand_cell| {
            LSH_HEAP_F64.with(|heap_cell| {
                LSH_SEEN_SET.with(|seen_cell| {
                    let mut cand = cand_cell.borrow_mut();
                    let mut heap = heap_cell.borrow_mut();
                    let mut seen = seen_cell.borrow_mut();

                    if k == 0 {
                        return (Vec::new(), Vec::new(), false);
                    }

                    cand.clear();
                    heap.clear();
                    seen.clear();

                    // pre-reserve
                    cand.reserve(10000);

                    for table_idx in 0..self.num_tables {
                        let hash = Self::compute_hash(
                            query_vec,
                            table_idx,
                            self.bits_per_hash,
                            self.dim,
                            &self.random_vecs,
                            &self.metric,
                        );

                        if let Some(bucket) = self.hash_tables[table_idx].get(&hash) {
                            cand.extend_from_slice(bucket);

                            if let Some(max) = max_cand {
                                if cand.len() >= max {
                                    break;
                                }
                            }
                        }
                    }

                    let fallback_triggered = cand.is_empty();
                    if fallback_triggered {
                        let mut rng = rng();
                        let sample_size = 1000.min(self.n);
                        cand.extend((0..self.n).choose_multiple(&mut rng, sample_size));
                    }

                    // Deduplicate during distance computation
                    match self.metric {
                        Dist::Euclidean => {
                            for &idx in cand.iter() {
                                if seen.insert(idx) {
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
                        }
                        Dist::Cosine => {
                            let query_norm = query_vec.iter().map(|v| v * v).sum::<f64>().sqrt();
                            for &idx in cand.iter() {
                                if seen.insert(idx) {
                                    let d =
                                        self.cosine_distance_to_query(idx, query_vec, query_norm);
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
                    }

                    let mut results: Vec<_> = heap.drain().collect();
                    results.sort_unstable_by(|a, b| a.0.cmp(&b.0));

                    let indices = results.iter().map(|&(_, idx)| idx).collect();
                    let dists = results.iter().map(|&(OrderedFloat(d), _)| d).collect();

                    (indices, dists, fallback_triggered)
                })
            })
        })
    }
}

//////////////////////
// Validation trait //
//////////////////////

impl<T> KnnValidation<T> for LSHIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
    Self: LSHQuery<T>,
{
    /// Internal querying function
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        // No maximum candidates here...
        let (indices, dist, _) = self.query(query_vec, k, None);
        (indices, dist)
    }

    /// Returns n
    fn n(&self) -> usize {
        self.n
    }

    /// Returns the distance metric
    fn metric(&self) -> Dist {
        self.metric
    }
}

//////////////////////////
// HierarchicalLSHIndex //
//////////////////////////

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
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.num_tables, 4);
        assert_eq!(index.bits_per_hash, 8);
        assert_eq!(index.vectors_flat.len(), 15);
    }

    #[test]
    fn test_index_creation_cosine() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Cosine, 4, 8, 42);

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.norms.len(), 5);
    }

    #[test]
    fn test_basic_query() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 3, None);

        // LSH is approximate, so just check we get valid results
        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());

        // Should include the exact match
        assert!(indices.contains(&0));

        // Distances should be sorted
        for i in 1..distances.len() {
            assert!(distances[i - 1] <= distances[i]);
        }
    }

    #[test]
    fn test_query_cosine() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Cosine, 4, 8, 42);

        let query = vec![2.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 2, None);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 2);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_query_row() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query_mat = Mat::from_fn(1, 3, |_, j| [1.0, 0.0, 0.0][j]);
        let (indices, distances, _) = index.query_row(query_mat.row(0), 3, None);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_max_cand_limit() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 10, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _, _) = index.query(&query, 2, Some(3));

        assert!(indices.len() <= 2);
    }

    #[test]
    fn test_k_larger_than_n() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 100, None);

        // Should return at most n vectors
        assert!(indices.len() <= 5);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    #[should_panic(expected = "Query vector dimensionality mismatch")]
    fn test_dimension_mismatch() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0];
        index.query(&query, 3, None);
    }

    #[test]
    #[should_panic(expected = "Query row dimensionality mismatch")]
    fn test_query_row_dimension_mismatch() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query_mat = Mat::from_fn(1, 2, |_, j| [1.0, 0.0][j]);
        index.query_row(query_mat.row(0), 3, None);
    }

    #[test]
    fn test_fallback_mechanism() {
        // Create sparse high-dim vectors unlikely to hash together
        let mat = Mat::from_fn(10, 100, |i, j| if j == i * 10 { 1.0 } else { 0.0 });

        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 2, 16, 42);

        let query = vec![1.0; 100];
        let (indices, distances, _) = index.query(&query, 3, None);

        // Should get results via fallback or hash match
        assert!(!indices.is_empty());
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_deterministic_with_seed() {
        let mat = simple_test_data();

        let index1 = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);
        let index2 = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices1, _, _) = index1.query(&query, 3, None);
        let (indices2, _, _) = index2.query(&query, 3, None);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_f64_query() {
        let mat = Mat::from_fn(3, 3, |i, j| if i == j { 1.0f64 } else { 0.0f64 });
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 2, None);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 2);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_distances_sorted() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (_, distances, _) = index.query(&query, 5, None);

        // Distances should be in ascending order
        for i in 1..distances.len() {
            assert!(distances[i - 1] <= distances[i]);
        }
    }

    #[test]
    fn test_query_k_zero() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 0, None);

        assert_eq!(indices.len(), 0);
        assert_eq!(distances.len(), 0);
    }

    #[test]
    fn test_query_returns_k_or_fewer() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 8, 6, 42);

        let query = vec![1.0, 0.0, 0.0];

        for k in 1..=5 {
            let (indices, distances, _) = index.query(&query, k, None);
            assert!(indices.len() <= k);
            assert_eq!(indices.len(), distances.len());
        }
    }

    #[test]
    fn test_no_duplicate_results() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 8, 6, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _, _) = index.query(&query, 5, None);

        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();

        assert_eq!(indices.len(), sorted.len(), "Results contain duplicates");
    }

    #[test]
    fn test_larger_dataset() {
        // Test with more realistic dataset size
        let n = 1000;
        let dim = 50;
        let mat = Mat::from_fn(n, dim, |i, j| ((i * 7 + j * 13) % 100) as f32 / 100.0);

        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 10, 10, 42);

        let query = vec![0.5; dim];
        let (indices, distances, _) = index.query(&query, 10, None);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 10);
        assert_eq!(indices.len(), distances.len());

        // Check all indices are valid
        for &idx in &indices {
            assert!(idx < n);
        }
    }
}
