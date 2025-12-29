use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::iter::Sum;
use thousands::*;

use crate::binary::binariser::*;
use crate::binary::dist_binary::*;

///////////////////////////
// ExhaustiveIndexBinary //
///////////////////////////

///////////////////////////
// ExhaustiveIndexBinary //
///////////////////////////

/// Exhaustive (brute-force) binary nearest neighbour index
///
/// Stores vectors as binary codes and uses Hamming distance for queries.
///
/// ### Fields
///
/// * `vectors_flat_binarised` - Binary codes, flattened (n * n_bytes)
/// * `n_bytes` - Bytes per vector (n_bits / 8)
/// * `n` - Number of samples
/// * `binariser` - Binariser for encoding query vectors
pub struct ExhaustiveIndexBinary<T> {
    // shared ones
    pub vectors_flat_binarised: Vec<u8>,
    pub n_bytes: usize,
    pub n: usize,
    binariser: Binariser<T>,
}

//////////////////////////
// VectorDistanceBinary //
//////////////////////////

impl<T> VectorDistanceBinary for ExhaustiveIndexBinary<T> {
    fn vectors_flat_binarised(&self) -> &[u8] {
        &self.vectors_flat_binarised
    }

    fn n_bytes(&self) -> usize {
        self.n_bytes
    }
}

impl<T> ExhaustiveIndexBinary<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    /// Generate a new exhaustive binary index
    ///
    /// Binarises all vectors using the specified hash function and stores them
    /// as compact binary codes. This works solely for Cosine distance!
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features)
    /// * `n_bits` - Number of bits per binary code (must be multiple of 8)
    /// * `seed` - Random seed for binariser
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive binary index
    pub fn new(data: MatRef<T>, n_bits: usize, seed: usize) -> Self {
        assert!(n_bits % 8 == 0, "n_bits must be multiple of 8");

        let n_bytes = n_bits / 8;
        let n = data.nrows();
        let dim = data.ncols();

        let binariser = Binariser::new(dim, n_bits, seed);

        let mut vectors_flat_binarised: Vec<u8> = Vec::with_capacity(n * n_bytes);

        for i in 0..n {
            let original: Vec<T> = data.row(i).iter().cloned().collect();
            vectors_flat_binarised.extend(binariser.encode(&original));
        }

        Self {
            vectors_flat_binarised,
            n_bytes,
            n,
            binariser,
        }
    }

    ///////////
    // Query //
    ///////////

    /// Query function
    ///
    /// Exhaustive search over all binary codes using Hamming distance.
    /// Binary codes are generated via the hash function specified during construction
    /// (SimHash for cosine similarity, MaxPool for Euclidean distance).
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (will be binarised internally)
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` where distances are Hamming distances
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<u32>) {
        let query_binary = self.binariser.encode(query_vec);
        let k = k.min(self.n);

        let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(k + 1);

        for idx in 0..self.n {
            let dist = self.hamming_distance_query(&query_binary, idx);

            if heap.len() < k {
                heap.push((dist, idx));
            } else if dist < heap.peek().unwrap().0 {
                heap.pop();
                heap.push((dist, idx));
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);

        let (distances, indices): (Vec<_>, Vec<_>) = results.into_iter().unzip();

        (indices, distances)
    }

    /// Query function for row references
    ///
    /// Exhaustive search using Hamming distance on binarised query.
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    #[inline]
    pub fn query_row(&self, query_row: RowRef<T>, k: usize) -> (Vec<usize>, Vec<u32>) {
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k)
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector against all others to build complete kNN graph.
    /// Uses original float vectors which are binarised via the hash function,
    /// then Hamming distances computed between binary codes.
    ///
    /// ### Params
    ///
    /// * `data` - Original float data (needed to query each vector)
    /// * `k` - Number of neighbours per vector
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`
    pub fn generate_knn(
        &self,
        data: MatRef<T>,
        k: usize,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<u32>>>) {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        assert_eq!(data.nrows(), self.n, "Data row count mismatch");

        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<u32>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let vec: Vec<T> = data.row(i).iter().cloned().collect();

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

                self.query(&vec, k)
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
            + self.vectors_flat_binarised.capacity()
            + self.binariser.memory_usage_bytes()
    }
}

///////////////////////////
// Exhaustive4BitIndex   //
///////////////////////////

/// Exhaustive (brute-force) 4-bit quantised nearest neighbour index
///
/// Stores vectors as 4-bit codes and uses asymmetric distance for queries.
/// Symmetric distance is used for kNN graph construction.
///
/// ### Fields
///
/// * `vectors_flat_encoded` - 4-bit codes, flattened (n * n_bytes)
/// * `n_bytes` - Bytes per vector (ceil(n_projections / 2))
/// * `n` - Number of samples
/// * `quantiser` - Quantiser for encoding vectors and building LUTs
pub struct Exhaustive4BitIndex<T> {
    pub vectors_flat_encoded: Vec<u8>,
    pub n_bytes: usize,
    pub n: usize,
    quantiser: Quantiser4Bit<T>,
}

//////////////////////////
// VectorDistance4Bit   //
//////////////////////////

impl<T> VectorDistance4Bit<T> for Exhaustive4BitIndex<T>
where
    T: Float,
{
    fn vectors_flat_encoded(&self) -> &[u8] {
        &self.vectors_flat_encoded
    }

    fn n_bytes(&self) -> usize {
        self.n_bytes
    }

    fn n_projections(&self) -> usize {
        self.quantiser.n_projections
    }

    fn bucket_centres(&self) -> &[T] {
        &self.quantiser.bucket_centres
    }
}

/// Helper struct for max-heap that orders by distance (we want min-heap behaviour)
struct DistanceEntry<T> {
    dist: T,
    idx: usize,
}

impl<T: Float> PartialEq for DistanceEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl<T: Float> Eq for DistanceEntry<T> {}

impl<T: Float> PartialOrd for DistanceEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for max-heap to act as min-heap
        other.dist.partial_cmp(&self.dist)
    }
}

impl<T: Float> Ord for DistanceEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl<T> Exhaustive4BitIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
{
    /// Create a new exhaustive 4-bit index
    ///
    /// Fits the quantiser on data and encodes all vectors.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features)
    /// * `n_projections` - Number of random projections
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `seed` - Random seed for quantiser
    ///
    /// ### Returns
    ///
    /// Initialised exhaustive 4-bit index
    pub fn new(data: MatRef<T>, n_projections: usize, metric: Dist4Bit, seed: u64) -> Self {
        let n = data.nrows();
        let dim = data.ncols();

        // Create and fit quantiser
        let mut quantiser = Quantiser4Bit::new(dim, n_projections, metric, seed);

        // Collect data as slices for fitting
        let rows: Vec<Vec<T>> = (0..n)
            .map(|i| data.row(i).iter().cloned().collect())
            .collect();

        quantiser.fit(rows.iter().map(|r| r.as_slice()), n);

        let n_bytes = quantiser.code_size();

        // Encode all vectors
        let mut vectors_flat_encoded: Vec<u8> = Vec::with_capacity(n * n_bytes);

        for row in &rows {
            vectors_flat_encoded.extend(quantiser.encode(row));
        }

        Self {
            vectors_flat_encoded,
            n_bytes,
            n,
            quantiser,
        }
    }

    ///////////
    // Query //
    ///////////

    /// Query for k nearest neighbours
    ///
    /// Uses asymmetric distance: query at full precision, documents quantised.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` where distances are approximate squared distances
    #[inline]
    pub fn query(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        let query_lut = self.quantiser.build_query_lut(query_vec);
        let k = k.min(self.n);

        // Max-heap storing (distance, index), but with reversed ordering
        let mut heap: BinaryHeap<DistanceEntry<T>> = BinaryHeap::with_capacity(k + 1);

        for idx in 0..self.n {
            let dist = self.asymmetric_distance(&query_lut, idx);

            if heap.len() < k {
                heap.push(DistanceEntry { dist, idx });
            } else if let Some(worst) = heap.peek() {
                if dist < worst.dist {
                    heap.pop();
                    heap.push(DistanceEntry { dist, idx });
                }
            }
        }

        // Extract and sort by distance
        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

        let indices: Vec<usize> = results.iter().map(|e| e.idx).collect();
        let distances: Vec<T> = results.iter().map(|e| e.dist).collect();

        (indices, distances)
    }

    /// Query using a row reference
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of nearest neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    #[inline]
    pub fn query_row(&self, query_row: RowRef<T>, k: usize) -> (Vec<usize>, Vec<T>) {
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k)
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Uses symmetric distance (document-to-document) since we don't have
    /// a separate query. This is slightly less accurate than asymmetric
    /// but captures the broad structure well.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`
    pub fn generate_knn(
        &self,
        k: usize,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        use std::sync::{
            atomic::{AtomicUsize, Ordering as AtomicOrdering},
            Arc,
        };

        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                if verbose {
                    let count = counter.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                    if count % 100_000 == 0 {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            self.n.separate_with_underscores()
                        );
                    }
                }

                self.query_symmetric(i, k)
            })
            .collect();

        if return_dist {
            let (indices, distances): (Vec<_>, Vec<_>) = results.into_iter().unzip();
            (indices, Some(distances))
        } else {
            let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _)| idx).collect();
            (indices, None)
        }
    }

    /// Query using symmetric distance (for kNN graph construction)
    ///
    /// Finds k nearest neighbours of vector at index `query_idx` using
    /// document-to-document symmetric distance.
    ///
    /// ### Params
    ///
    /// * `query_idx` - Index of query vector in the index
    /// * `k` - Number of nearest neighbours
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    fn query_symmetric(&self, query_idx: usize, k: usize) -> (Vec<usize>, Vec<T>) {
        let k = k.min(self.n - 1); // Exclude self

        let mut heap: BinaryHeap<DistanceEntry<T>> = BinaryHeap::with_capacity(k + 1);

        for idx in 0..self.n {
            if idx == query_idx {
                continue;
            }

            let dist = self.symmetric_distance(query_idx, idx);

            if heap.len() < k {
                heap.push(DistanceEntry { dist, idx });
            } else if let Some(worst) = heap.peek() {
                if dist < worst.dist {
                    heap.pop();
                    heap.push(DistanceEntry { dist, idx });
                }
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

        let indices: Vec<usize> = results.iter().map(|e| e.idx).collect();
        let distances: Vec<T> = results.iter().map(|e| e.dist).collect();

        (indices, distances)
    }

    /// Returns memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat_encoded.capacity()
            + self.quantiser.memory_usage_bytes()
    }

    /// Returns the distance metric
    pub fn metric(&self) -> Dist4Bit {
        self.quantiser.metric
    }

    /// Returns the number of projections
    pub fn n_projections(&self) -> usize {
        self.quantiser.n_projections
    }

    /// Returns input dimensionality
    pub fn dim(&self) -> usize {
        self.quantiser.dim
    }
}
