//! Exhaustive (flat) implementation for nearest neighbour searches in
//! ann-search-rs.

use faer::{linalg::matmul::matmul, Accum, Mat, MatRef, Par, RowRef};

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use thousands::*;

use crate::prelude::*;
use crate::utils::pack_knn_results;

////////////////////////
// Constants & tuning //
////////////////////////

/// Queries per thread below which the fused scan beats the blocked GEMM path.
///
/// The GEMM path pays an O(n_queries * n_samples) cost for materialising the
/// dot products and buys reuse of each database tile across the queries
/// sharing it. Batch size decides that trade, not dimension: the crossover
/// held between 96 and 128 queries across `dim` 32 and 128 and `n` from 10k to
/// 200k, a 20-fold range in database size.
///
/// It is stated per thread because the scan fans out one query per core while
/// the GEMM path can only fan out whole blocks, so a batch too small to fill
/// the machine with blocks loses however favourable the arithmetic is.
/// Measured crossovers were under 32 queries at 2 threads, 58 at 4 and 110 at
/// 10, so 11 to 16 queries per thread; the upper end is taken so the threshold
/// sits on a measured win rather than on the margin.
///
/// Note this is *not* the crossover `k_means_utils` measures for its own GEMM
/// assignment. There the database side is the centroid set, small enough to
/// stay cache-resident on its own, so blocking adds little and the crossover
/// lands in dimension instead.
const GEMM_MIN_QUERIES_PER_THREAD: usize = 16;

/// Largest number of queries per GEMM block.
///
/// With [`GEMM_DB_TILE`] this caps the dot-product tile at 256 * 1024 elements,
/// so roughly 1 MB at `f32`. Sized to sit in L2 alongside the two operand
/// blocks, since every thread holds its own tile.
const GEMM_QUERY_TILE: usize = 256;

/// Smallest number of queries per GEMM block.
///
/// The block loop is the only source of parallelism on the GEMM path, so the
/// tile shrinks below [`GEMM_QUERY_TILE`] to keep every thread fed on a small
/// batch. It stops here because a block narrower than this reuses a database
/// tile too few times to pay for materialising the dot products.
const GEMM_QUERY_TILE_MIN: usize = 32;

/// Blocks to aim for per thread when sizing the query tile.
///
/// One block per thread balances badly on an asymmetric machine: a block that
/// lands on an efficiency core holds up the whole batch. Cutting finer gives
/// rayon something to steal, at the cost of reusing each database tile across
/// fewer queries.
const GEMM_BLOCKS_PER_THREAD: usize = 4;

/// Database vectors per GEMM block.
///
/// The block is re-read by all [`GEMM_QUERY_TILE`] queries, so it wants to stay
/// resident: 1024 rows is 512 KB at `dim = 128` and `f32`.
const GEMM_DB_TILE: usize = 1024;

/////////////
// Helpers //
/////////////

/// Report scan progress every 100,000 queries.
///
/// Takes the batch size so the block-at-a-time GEMM path reports on the same
/// cadence as the one-at-a-time scan.
///
/// ### Params
///
/// * `counter` - Shared count of queries completed so far
/// * `delta` - Number of queries this call completed
/// * `total` - Total number of queries in the batch
fn report_progress(counter: &AtomicUsize, delta: usize, total: usize) {
    let before = counter.fetch_add(delta, Ordering::Relaxed);
    let after = before + delta;
    if before / 100_000 != after / 100_000 {
        println!(
            "  Processed {} / {} samples.",
            after.separate_with_underscores(),
            total.separate_with_underscores()
        );
    }
}

/// Scan every candidate once, retaining the `k` nearest in a bounded heap.
///
/// The distance function is monomorphised per call site and inlines, so the
/// metric branch is hoisted out of the scan rather than tested per candidate.
///
/// ### Params
///
/// * `n` - Number of candidates to scan
/// * `k` - Number of neighbours to retain
/// * `dist_fn` - Distance from the query to the candidate at a given index
///
/// ### Returns
///
/// A heap holding the `k` nearest candidates
#[inline(always)]
fn scan_into_heap<T, F>(n: usize, k: usize, dist_fn: F) -> BoundedMaxHeap<T>
where
    T: AnnSearchFloat,
    F: Fn(usize) -> T,
{
    let mut heap = BoundedMaxHeap::new(k);
    for idx in 0..n {
        heap.push(dist_fn(idx), idx);
    }
    heap
}

/////////////////////
// ExhaustiveIndex //
/////////////////////

/// Exhaustive (brute-force) nearest neighbour index
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct ExhaustiveIndex<T> {
    /// Original vector data for distance calculations. Flattened for better
    /// cache locality
    pub vectors_flat: Vec<T>,
    /// Embedding dimensions
    pub dim: usize,
    /// Number of samples
    pub n: usize,
    /// Normalised pre-calculated values per sample if distance is set to
    /// Cosine.
    norms: Vec<T>,
    /// The type of distance the index is designed for
    metric: Dist,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for ExhaustiveIndex<T>
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

impl<T> DimensionValidation for ExhaustiveIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////////
// Index build //
/////////////////

impl<T> ExhaustiveIndex<T>
where
    T: AnnSearchFloat,
{
    /// Generate a new exhaustive index
    ///
    /// ### Params
    ///
    /// * `data` - The data for which to generate the index. Samples x features
    /// * `metric` - Which distance metric the index shall be generated for.
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive index
    pub fn new(data: impl AnnMatrix<T>, metric: Dist) -> Self {
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

        Self {
            vectors_flat,
            norms,
            dim,
            metric,
            n,
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
    }
}

///////////
// Query //
///////////

impl<T> ExhaustiveIndex<T>
where
    T: AnnSearchFloat,
{
    /// Query function
    ///
    /// This will do an exhaustive search over the full index (i.e., all samples)
    /// during querying. To note, this becomes prohibitively computationally
    /// expensive on large data sets!
    ///
    /// ### Params
    ///
    /// * `query_vec` - The query vector.
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// A tuple of `(indices, distances)`
    #[inline]
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        let n_vectors = self.vectors_flat.len() / self.dim;
        let k = k.min(n_vectors);

        let heap = match self.metric {
            Dist::SquaredEuclidean => scan_into_heap(n_vectors, k, |idx| {
                self.euclidean_distance_to_query(idx, query_vec)
            }),
            Dist::Cosine => {
                let query_norm = T::calculate_l2_norm(query_vec);
                scan_into_heap(n_vectors, k, |idx| {
                    self.cosine_distance_to_query(idx, query_vec, query_norm)
                })
            }
            Dist::Manhattan => scan_into_heap(n_vectors, k, |idx| {
                self.manhattan_distance_to_query(idx, query_vec)
            }),
        };

        Ok(heap.into_sorted())
    }

    /// Query function for row references
    ///
    /// This will do an exhaustive search over the full index (i.e., all samples)
    /// during querying. To note, this becomes prohibitively computationally
    /// expensive on large data sets!
    ///
    /// ### Params
    ///
    /// * `query_row` - The query row.
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// A tuple of `(indices, distances)`
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k)
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete
    /// kNN graph. Every point is its own nearest neighbour, so the first
    /// column is the point itself.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)` where each row corresponds
    /// to a vector in the index
    pub fn generate_knn(&self, k: usize, return_dist: bool, verbose: bool) -> KnnOptionResult<T> {
        let results = self.query_batch(&self.vectors_flat, self.n, k, None, verbose)?;
        Ok(pack_knn_results(results, return_dist))
    }

    /// Query a batch of vectors against the index
    ///
    /// Dispatches between the fused per-query scan and a blocked GEMM path.
    /// The scan keeps each accumulation in registers but re-reads the whole
    /// database for every query; the GEMM path blocks both axes so a database
    /// tile is reused across a tile of queries, paying for that with a
    /// materialised dot-product tile.
    ///
    /// ### Params
    ///
    /// * `queries` - Query vectors, flattened row-major (`nq * dim` elements)
    /// * `nq` - Number of queries
    /// * `k` - Number of neighbours to return
    /// * `use_gemm` - Force the GEMM path on or off. `None` picks by batch
    ///   size relative to the thread count, see
    ///   [`GEMM_MIN_QUERIES_PER_THREAD`]. Both paths return exact distances,
    ///   so this is a performance knob only.
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Per-query tuples of `(indices, distances)`
    pub fn query_batch(
        &self,
        queries: &[T],
        nq: usize,
        k: usize,
        use_gemm: Option<bool>,
        verbose: bool,
    ) -> KnnBatchResult<T> {
        if nq == 0 {
            return Ok(Vec::new());
        }
        self.check_dim(queries.len() / nq)?;

        if self.gemm_applies(nq, use_gemm) {
            return self.query_batch_gemm(queries, nq, k.min(self.n), verbose);
        }

        let counter = AtomicUsize::new(0);
        (0..nq)
            .into_par_iter()
            .map(|i| {
                if verbose {
                    report_progress(&counter, 1, nq);
                }
                self.query(&queries[i * self.dim..(i + 1) * self.dim], k)
            })
            .collect()
    }

    /// Whether the blocked GEMM path applies to this batch
    ///
    /// ### Params
    ///
    /// * `nq` - Number of queries in the batch
    /// * `requested` - Explicit override, or `None` for the heuristic
    ///
    /// ### Returns
    ///
    /// `true` if the GEMM path should be taken
    fn gemm_applies(&self, nq: usize, requested: Option<bool>) -> bool {
        // The dot-product expansion has no analogue under Manhattan, so an
        // explicit request is refused rather than answering a different metric.
        if self.metric == Dist::Manhattan || self.n == 0 {
            return false;
        }

        requested.unwrap_or(nq >= rayon::current_num_threads().max(1) * GEMM_MIN_QUERIES_PER_THREAD)
    }

    /// Blocked GEMM nearest neighbour search over a batch of queries
    ///
    /// Both axes are tiled. For each tile pair a single GEMM produces the dot
    /// products, which become distances through the expansion
    /// `||x||^2 - 2<x,y> + ||y||^2` for Euclidean, or
    /// `1 - <x,y> / (||x|| ||y||)` for cosine. Each query keeps one heap
    /// across all database tiles.
    ///
    /// The Euclidean expansion cancels catastrophically once the distance is
    /// small relative to `||x||`, which is exactly the regime of the
    /// neighbours being returned, so the retained `k` are recomputed with the
    /// fused kernel before being handed back. Selection still happens on the
    /// expanded values, but every distance the caller sees is exact and the
    /// ordering within the `k` is correct.
    ///
    /// ### Params
    ///
    /// * `queries` - Query vectors, flattened row-major (`nq * dim` elements)
    /// * `nq` - Number of queries
    /// * `k` - Number of neighbours to return, already clamped to `n`
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Per-query tuples of `(indices, distances)`
    fn query_batch_gemm(
        &self,
        queries: &[T],
        nq: usize,
        k: usize,
        verbose: bool,
    ) -> KnnBatchResult<T> {
        let dim = self.dim;
        let n = self.n;
        let two = T::one() + T::one();
        let cosine = self.metric == Dist::Cosine;

        // Database norms are recomputed per batch rather than stored on the
        // index: O(n * dim) against the O(nq * n * dim) of the search itself,
        // and it leaves the serialised layout alone.
        let db_norms: Vec<T> = if cosine {
            self.norms.clone()
        } else {
            (0..n)
                .into_par_iter()
                .map(|j| {
                    let y = &self.vectors_flat[j * dim..(j + 1) * dim];
                    T::dot_simd(y, y)
                })
                .collect()
        };

        let q_norms: Vec<T> = (0..nq)
            .into_par_iter()
            .map(|i| {
                let x = &queries[i * dim..(i + 1) * dim];
                if cosine {
                    T::calculate_l2_norm(x)
                } else {
                    T::dot_simd(x, x)
                }
            })
            .collect();

        let counter = AtomicUsize::new(0);

        // The block loop carries all the parallelism, so a batch that would
        // fit in one block has to be split anyway or the whole search runs on
        // a single thread.
        let query_tile = nq
            .div_ceil((rayon::current_num_threads() * GEMM_BLOCKS_PER_THREAD).max(1))
            .clamp(GEMM_QUERY_TILE_MIN, GEMM_QUERY_TILE);

        let blocks: Vec<Vec<(Vec<usize>, Vec<T>)>> = (0..nq.div_ceil(query_tile))
            .into_par_iter()
            .map_init(Mat::<T>::new, |dots, block| {
                let i0 = block * query_tile;
                let i1 = (i0 + query_tile).min(nq);
                let bq = i1 - i0;

                let mut heaps: Vec<BoundedMaxHeap<T>> =
                    (0..bq).map(|_| BoundedMaxHeap::new(k)).collect();

                let x_block = MatRef::from_row_major_slice(&queries[i0 * dim..i1 * dim], bq, dim);

                for j0 in (0..n).step_by(GEMM_DB_TILE) {
                    let j1 = (j0 + GEMM_DB_TILE).min(n);
                    let bd = j1 - j0;

                    let y_block = MatRef::from_row_major_slice(
                        &self.vectors_flat[j0 * dim..j1 * dim],
                        bd,
                        dim,
                    );

                    if dots.nrows() != bq || dots.ncols() != bd {
                        *dots = Mat::<T>::zeros(bq, bd);
                    }

                    // Inner GEMM stays sequential: the outer rayon iterator
                    // already owns every core.
                    matmul(
                        dots.as_mut(),
                        Accum::Replace,
                        x_block,
                        y_block.transpose(),
                        T::one(),
                        Par::Seq,
                    );

                    // Metric branch is hoisted out of the inner scan.
                    if cosine {
                        for li in 0..bq {
                            let xn = q_norms[i0 + li];
                            let heap = &mut heaps[li];
                            for lj in 0..bd {
                                let denom = xn * db_norms[j0 + lj];
                                let dist = if denom > T::zero() {
                                    T::one() - dots[(li, lj)] / denom
                                } else {
                                    T::one()
                                };
                                heap.push(dist, j0 + lj);
                            }
                        }
                    } else {
                        for li in 0..bq {
                            let xn = q_norms[i0 + li];
                            let heap = &mut heaps[li];
                            for lj in 0..bd {
                                // Roundoff can push identical vectors below
                                // zero; clamp before it reaches the heap.
                                let dist =
                                    (xn + db_norms[j0 + lj] - two * dots[(li, lj)]).max(T::zero());
                                heap.push(dist, j0 + lj);
                            }
                        }
                    }
                }

                if verbose {
                    report_progress(&counter, bq, nq);
                }

                heaps
                    .into_iter()
                    .enumerate()
                    .map(|(li, heap)| {
                        let (ids, _) = heap.into_sorted();
                        let x = &queries[(i0 + li) * dim..(i0 + li + 1) * dim];

                        let mut exact: Vec<(OrderedFloat<T>, usize)> = ids
                            .into_iter()
                            .map(|j| {
                                let dist = if cosine {
                                    self.cosine_distance_to_query(j, x, q_norms[i0 + li])
                                } else {
                                    self.euclidean_distance_to_query(j, x)
                                };
                                (OrderedFloat(dist), j)
                            })
                            .collect();
                        exact.sort_unstable();

                        exact
                            .into_iter()
                            .map(|(OrderedFloat(dist), j)| (j, dist))
                            .unzip()
                    })
                    .collect()
            })
            .collect();

        Ok(blocks.into_iter().flatten().collect())
    }
}

///////////
// Tests //
///////////

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for ExhaustiveIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "exhaustive";
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::Mat;

    fn create_simple_matrix() -> Mat<f32> {
        // 5 points in 3D space
        let data = [
            1.0, 0.0, 0.0, // Point 0: [1, 0, 0]
            0.0, 1.0, 0.0, // Point 1: [0, 1, 0]
            0.0, 0.0, 1.0, // Point 2: [0, 0, 1]
            1.0, 1.0, 0.0, // Point 3: [1, 1, 0]
            1.0, 0.0, 1.0, // Point 4: [1, 0, 1]
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
    }

    #[test]
    fn test_exhaustive_index_creation_euclidean() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.vectors_flat.len(), 15);
        assert!(index.norms.is_empty()); // No norms for Euclidean
    }

    #[test]
    fn test_exhaustive_index_creation_cosine() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Cosine);

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.vectors_flat.len(), 15);
        assert_eq!(index.norms.len(), 5); // Norms computed for Cosine
    }

    #[test]
    fn test_exhaustive_query_finds_self_euclidean() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);

        // Query with point 0, should find itself first
        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1).unwrap();

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_query_finds_self_cosine() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Cosine);

        // Query with point 0, should find itself first
        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1).unwrap();

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_query_euclidean_multiple() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3).unwrap();

        // Should find point 0 first (exact match)
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Results should be sorted by distance
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_exhaustive_query_cosine_orthogonal() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Cosine);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 5).unwrap(); // Get all 5

        // Should find point 0 first (identical direction)
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Points 3 and 4 are at 45° (closer than orthogonal)
        assert_relative_eq!(distances[1], 1.0 - 1.0 / 2.0_f32.sqrt(), epsilon = 1e-5);
        assert_relative_eq!(distances[2], 1.0 - 1.0 / 2.0_f32.sqrt(), epsilon = 1e-5);

        // Points 1 and 2 are orthogonal (furthest away)
        assert_relative_eq!(distances[3], 1.0, epsilon = 1e-5);
        assert_relative_eq!(distances[4], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_query_k_larger_than_dataset() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);

        let query = vec![1.0, 0.0, 0.0];
        // Ask for 10 neighbours but only 5 points exist
        let (indices, _) = index.query(&query, 10).unwrap();

        // Should return exactly 5 results
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_exhaustive_query_row() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);

        // Query using a row from the matrix
        let (indices, distances) = index.query_row(mat.row(0), 1).unwrap();

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_euclidean_distances() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 5).unwrap();

        // Distance from [1,0,0] to [1,0,0] = 0
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Distance from [1,0,0] to [1,0,1] = 1
        // Distance from [1,0,0] to [1,1,0] = 1
        // Both should appear next (order might vary)
        assert!(distances[1] <= 1.01);
        assert!(distances[2] <= 1.01);

        // Distance from [1,0,0] to [0,1,0] = sqrt(2) ≈ 1.414
        // Distance from [1,0,0] to [0,0,1] = sqrt(2) ≈ 1.414
        assert_relative_eq!(distances[3], 2.0, epsilon = 0.1);
        assert_relative_eq!(distances[4], 2.0, epsilon = 0.1);
    }

    #[test]
    fn test_exhaustive_all_points_found() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);

        let query = vec![0.5, 0.5, 0.5];
        let (indices, _) = index.query(&query, 5).unwrap();

        // All 5 points should be found
        assert_eq!(indices.len(), 5);

        // All unique indices
        let mut sorted_indices = indices.clone();
        sorted_indices.sort_unstable();
        assert_eq!(sorted_indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_exhaustive_larger_dataset() {
        // Create a larger dataset
        let n = 50;
        let dim = 10;
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 10.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);

        // Query for point 0
        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5).unwrap();

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0); // Should find exact match first
    }

    #[test]
    fn test_exhaustive_cosine_parallel_vectors() {
        let data = [
            1.0, 2.0, 3.0, // Vector 0
            2.0, 4.0, 6.0, // Vector 1 (parallel to 0, scaled by 2)
            -2.0, 1.0, 0.0, // Vector 2 (actually orthogonal to 0)
        ];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Cosine);

        let query = vec![1.0, 2.0, 3.0];
        let (indices, distances) = index.query(&query, 3).unwrap();

        // Should find itself first
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        // Parallel vector should be second with distance ≈ 0
        assert_eq!(indices[1], 1);
        assert_relative_eq!(distances[1], 0.0, epsilon = 1e-5);

        // Orthogonal vector should be last with distance = 1
        assert_eq!(indices[2], 2);
        assert_relative_eq!(distances[2], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_implements_vector_distance() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);

        // Test that we can call VectorDistance methods
        let dist = index.euclidean_distance(0, 1);
        assert!(dist > 0.0); // [1,0,0] vs [0,1,0] should have distance > 0

        let dist_self = index.euclidean_distance(0, 0);
        assert_relative_eq!(dist_self, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_exhaustive_cosine_implements_vector_distance() {
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Cosine);

        // Test that we can call VectorDistance methods
        let dist = index.cosine_distance(0, 1);
        assert_relative_eq!(dist, 1.0, epsilon = 1e-5); // Orthogonal vectors

        let dist_self = index.cosine_distance(0, 0);
        assert_relative_eq!(dist_self, 0.0, epsilon = 1e-5);
    }

    /// Clustered points, distinct and non-degenerate.
    ///
    /// The jitter period (97) is coprime with the cluster count over the sizes
    /// used here, so every row is unique and "each point is its own nearest
    /// neighbour" holds strictly. The offset keeps every norm non-zero so the
    /// cosine metric stays defined. Spread is comparable to the cluster
    /// separation, so distances stay resolvable at `f32`.
    fn clustered_matrix(n: usize, dim: usize) -> Mat<f32> {
        Mat::from_fn(n, dim, |i, j| {
            let cluster = (i % 5) as f32 * 40.0;
            let jitter = ((i * 7 + j * 13) % 97) as f32 * 0.5;
            1.0 + cluster + jitter
        })
    }

    /// Clusters packed far tighter than `f32` can resolve through the
    /// expansion, for pinning down what survives catastrophic cancellation.
    fn cancellation_matrix(n: usize, dim: usize) -> Mat<f32> {
        Mat::from_fn(n, dim, |i, j| {
            1000.0 + ((i * 7 + j * 13) % 23) as f32 * 1e-3 + i as f32 * 1e-4
        })
    }

    /// Compare the two paths on the property that actually holds.
    ///
    /// Selection on the GEMM path happens against the expanded distances, so a
    /// candidate sitting on the boundary of the retained set can swap with the
    /// one just outside it. What must hold is that the distances handed back
    /// are the exact ones, sorted, and no worse than the scan's.
    fn assert_paths_agree(index: &ExhaustiveIndex<f32>, queries: &[f32], nq: usize, k: usize) {
        let scan = index
            .query_batch(queries, nq, k, Some(false), false)
            .unwrap();
        let gemm = index
            .query_batch(queries, nq, k, Some(true), false)
            .unwrap();

        let mut matched = 0usize;
        let mut total = 0usize;

        for ((s_ids, s_dist), (g_ids, g_dist)) in scan.iter().zip(gemm.iter()) {
            assert!(g_dist.windows(2).all(|w| w[0] <= w[1]), "not sorted");

            for (a, b) in s_dist.iter().zip(g_dist.iter()) {
                assert_relative_eq!(a, b, epsilon = 1e-4);
            }

            let retained: std::collections::HashSet<_> = g_ids.iter().collect();
            matched += s_ids.iter().filter(|i| retained.contains(i)).count();
            total += s_ids.len();
        }

        // Boundary swaps are permitted, wholesale disagreement is not.
        let agreement = matched as f64 / total as f64;
        assert!(agreement > 0.99, "agreement {agreement} too low");
    }

    #[test]
    fn test_gemm_and_scan_agree_euclidean() {
        let mat = clustered_matrix(400, 24);
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);
        let (queries, nq, _) = mat.as_ref().into_row_major();

        assert_paths_agree(&index, &queries, nq, 10);
    }

    #[test]
    fn test_gemm_and_scan_agree_cosine() {
        let mat = clustered_matrix(400, 24);
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Cosine);
        let (queries, nq, _) = mat.as_ref().into_row_major();

        assert_paths_agree(&index, &queries, nq, 10);
    }

    #[test]
    fn test_gemm_returns_exact_zero_for_self_match() {
        // The expansion cancels hardest on coincident vectors; the re-rank is
        // what keeps the reported distance exact rather than merely clamped.
        let mat = clustered_matrix(300, 16);
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);
        let (queries, nq, _) = mat.as_ref().into_row_major();

        let gemm = index
            .query_batch(&queries, nq, 3, Some(true), false)
            .unwrap();
        for (i, (ids, dists)) in gemm.iter().enumerate() {
            assert_eq!(ids[0], i);
            assert_eq!(dists[0], 0.0);
        }
    }

    #[test]
    fn test_gemm_distances_stay_exact_under_cancellation() {
        // Norms near 1e6 against separations near 1e-3: the expansion cannot
        // resolve these, so which candidates get selected genuinely wobbles.
        // The re-rank is what guarantees that whatever comes back carries its
        // exact distance, in the right order. That is the contract.
        let mat = cancellation_matrix(300, 16);
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);
        let (queries, nq, dim) = mat.as_ref().into_row_major();

        let gemm = index
            .query_batch(&queries, nq, 8, Some(true), false)
            .unwrap();

        for (i, (ids, dists)) in gemm.iter().enumerate() {
            let q = &queries[i * dim..(i + 1) * dim];
            for (&id, &d) in ids.iter().zip(dists.iter()) {
                assert_eq!(d, index.euclidean_distance_to_query(id, q));
            }
            assert!(dists.windows(2).all(|w| w[0] <= w[1]));
        }
    }

    #[test]
    fn test_gemm_handles_coincident_vectors() {
        // Three exact copies of each point. The expansion cancels to a
        // negative number here, so this is what the clamp and the re-rank are
        // for: distance exactly zero, and the tie broken on the lower index.
        let mat = Mat::from_fn(300, 12, |i, j| {
            ((i / 3) as f32 + 1.0) * (j as f32 + 1.0) * 0.5
        });
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);
        let (queries, nq, _) = mat.as_ref().into_row_major();

        let scan = index
            .query_batch(&queries, nq, 3, Some(false), false)
            .unwrap();
        let gemm = index
            .query_batch(&queries, nq, 3, Some(true), false)
            .unwrap();

        for (i, (ids, dists)) in gemm.iter().enumerate() {
            let group = i / 3;
            assert_eq!(ids, &vec![group * 3, group * 3 + 1, group * 3 + 2]);
            assert_eq!(dists, &vec![0.0, 0.0, 0.0]);
        }
        assert_eq!(scan, gemm);
    }

    #[test]
    fn test_gemm_path_refused_for_manhattan() {
        let mat = clustered_matrix(300, 16);
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::Manhattan);

        // Even when explicitly asked for, since the expansion has no analogue.
        assert!(!index.gemm_applies(1_000, Some(true)));
        assert!(!index.gemm_applies(1_000, None));
    }

    /// Batch size at which the default heuristic flips to the GEMM path.
    fn gemm_floor() -> usize {
        rayon::current_num_threads().max(1) * GEMM_MIN_QUERIES_PER_THREAD
    }

    #[test]
    fn test_gemm_dispatch_follows_batch_size() {
        let mat = clustered_matrix(300, 16);
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);

        assert!(!index.gemm_applies(gemm_floor() - 1, None));
        assert!(index.gemm_applies(gemm_floor(), None));
        assert!(index.gemm_applies(1, Some(true)));
        assert!(!index.gemm_applies(100_000, Some(false)));
    }

    #[test]
    fn test_gemm_handles_k_larger_than_dataset() {
        let mat = clustered_matrix(200, 8);
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);
        let (queries, nq, _) = mat.as_ref().into_row_major();

        let gemm = index
            .query_batch(&queries, nq, 500, Some(true), false)
            .unwrap();
        assert!(gemm.iter().all(|(ids, _)| ids.len() == 200));
    }

    #[test]
    fn test_gemm_handles_ragged_final_block() {
        // A batch that is not a multiple of the query tile still returns one
        // result per query, in order.
        let mat = clustered_matrix(257, 12);
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);
        let (queries, nq, _) = mat.as_ref().into_row_major();

        let gemm = index
            .query_batch(&queries, nq, 5, Some(true), false)
            .unwrap();
        assert_eq!(gemm.len(), 257);
        for (i, (ids, _)) in gemm.iter().enumerate() {
            assert_eq!(ids[0], i);
        }
    }

    #[test]
    fn test_exhaustive_query_consistency() {
        // Test that query and query_row give same results
        let mat = create_simple_matrix();
        let index = ExhaustiveIndex::new(mat.as_ref(), Dist::SquaredEuclidean);

        let query_vec = vec![1.0, 0.0, 0.0];
        let (indices1, distances1) = index.query(&query_vec, 3).unwrap();
        let (indices2, distances2) = index.query_row(mat.row(0), 3).unwrap();

        assert_eq!(indices1, indices2);
        for i in 0..distances1.len() {
            assert_relative_eq!(distances1[i], distances2[i], epsilon = 1e-5);
        }
    }
}
