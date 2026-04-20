//! NNDescent implementation in ann-search-rs. Uses concepts of the original
//! implementation, PyNNDescent and EFANNA. Leverages Annoy over Kd forest for
//! graph initialisation.

use faer::{MatRef, RowRef};
use fixedbitset::FixedBitSet;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use rdst::RadixKey;
use rdst::RadixSort;
use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::BinaryHeap,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};
use thousands::*;

use crate::cpu::annoy::*;
use crate::prelude::*;
use crate::utils::*;

///////////////////
// Thread locals //
///////////////////

thread_local! {
    static HEAP_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize, bool)>> =
        const { RefCell::new(BinaryHeap::new()) };
    static HEAP_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize, bool)>> =
        const { RefCell::new(BinaryHeap::new()) };
    static PID_SET: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };

    static SORT_BUF_F32: RefCell<Vec<(f32, usize, bool)>> =
        const { RefCell::new(Vec::new()) };
    static SORT_BUF_F64: RefCell<Vec<(f64, usize, bool)>> =
        const { RefCell::new(Vec::new()) };

    static QUERY_VISITED: RefCell<FixedBitSet> = const { RefCell::new(FixedBitSet::new()) };
    static QUERY_CANDIDATES_F32: QueryCandF32 =
        const { RefCell::new(BinaryHeap::new()) };
    static QUERY_CANDIDATES_F64: QueryCandF64 =
        const { RefCell::new(BinaryHeap::new()) };
    static QUERY_RESULTS_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize)>> =
        const { RefCell::new(BinaryHeap::new()) };
    static QUERY_RESULTS_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize)>> =
        const { RefCell::new(BinaryHeap::new()) };
}

///////////
// Types //
///////////

/// Type alias for the query candidates for f32
pub type QueryCandF32 = RefCell<BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>>>;

/// Type alias for the query candidates for f64
pub type QueryCandF64 = RefCell<BinaryHeap<Reverse<(OrderedFloat<f64>, usize)>>>;

/////////////
// Helpers //
/////////////

////////////////
// Neighbours //
////////////////

/// Neighbour entry in the k-NN graph (build phase only)
///
/// Flat structure in C representation for cache locality. The high bit
/// of `pid_and_flag` stores the is-new flag, leaving 31 bits for the
/// point id (sufficient for ~2 billion points).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Neighbour<T> {
    /// Point index + new/old flag in the high bit
    pid_and_flag: u32,
    /// Distance to the neighbour
    pub dist: T,
}

/// Sentinel PID used to mark empty slots in the flat graph.
pub const SENTINEL_PID: usize = u32::MAX as usize >> 1;

impl<T: Copy> Neighbour<T> {
    const IS_NEW_MASK: u32 = 1 << 31;
    const PID_MASK: u32 = !Self::IS_NEW_MASK;

    /// Create a new neighbour entry
    ///
    /// The point ID must fit in 31 bits; the 32nd bit is reserved for the
    /// is-new flag. PIDs up to ~2 billion are supported.
    ///
    /// ### Params
    ///
    /// * `pid` - Point id (must fit in 31 bits)
    /// * `dist` - Distance to the neighbour
    /// * `is_new` - Whether this neighbour has been explored yet
    ///
    /// ### Returns
    ///
    /// Packed neighbour entry with encoded flag
    #[inline(always)]
    pub fn new(pid: usize, dist: T, is_new: bool) -> Self {
        debug_assert!(pid <= Self::PID_MASK as usize, "PID exceeds 31-bit limit");
        let flag = if is_new { Self::IS_NEW_MASK } else { 0 };
        Self {
            pid_and_flag: (pid as u32) | flag,
            dist,
        }
    }

    /// Whether this neighbour has not yet been explored
    ///
    /// New neighbours participate in local joins during the next iteration;
    /// old ones only contribute to old-new pair generation.
    ///
    /// ### Returns
    ///
    /// `true` if the high bit is set, `false` otherwise
    #[inline(always)]
    pub fn is_new(&self) -> bool {
        (self.pid_and_flag & Self::IS_NEW_MASK) != 0
    }

    /// Return the point id with the flag bit masked out
    ///
    /// ### Returns
    ///
    /// Point index in the range `[0, 2^31)`
    #[inline(always)]
    pub fn pid(&self) -> usize {
        (self.pid_and_flag & Self::PID_MASK) as usize
    }

    /// Whether this slot is empty (holds the sentinel PID)
    ///
    /// ### Returns
    ///
    /// `true` if this slot does not hold a valid neighbour
    #[inline(always)]
    pub fn is_sentinel(&self) -> bool {
        self.pid() == SENTINEL_PID
    }

    /// Mark this neighbour as old (already explored)
    ///
    /// Clears the high bit whilst preserving the point id and distance.
    #[inline(always)]
    pub fn mark_old(&mut self) {
        self.pid_and_flag &= Self::PID_MASK;
    }
}

///////////////
// Graph ptr //
///////////////

/// Unsafe pointer wrapper for lock-free parallel writes to the flat graph.
///
/// Safety is guaranteed by the update pattern: segments are grouped by
/// target node, so no two threads ever write to the same `target * k`
/// block simultaneously.
#[derive(Copy, Clone)]
struct UnsafeGraphPtr<T>(*mut Neighbour<T>);

unsafe impl<T> Send for UnsafeGraphPtr<T> {}
unsafe impl<T> Sync for UnsafeGraphPtr<T> {}

/////////////
// Updates //
/////////////

/// Helper structure for updates with Radix sort
#[derive(Clone, Copy)]
pub struct Update<T> {
    /// Target node id
    target: usize,
    /// Source node id
    source: usize,
    /// Distance between the two
    dist: T,
}

impl<T> Update<T> {
    /// Create a new update triple
    ///
    /// Represents a candidate edge from `source` to `target` with the pre-
    /// computed distance. Both directions are typically emitted per candidate
    /// pair so that radix sorting by target groups them for lock-free
    /// application.
    ///
    /// ### Params
    ///
    /// * `target` - Node receiving the edge
    /// * `source` - Node on the other end of the edge
    /// * `dist` - Distance between the two nodes
    ///
    /// ### Returns
    ///
    /// Update triple ready for radix sorting
    pub fn new(target: usize, source: usize, dist: T) -> Self {
        Self {
            target,
            source,
            dist,
        }
    }
}

/// Implement the RadixKey for target
impl<T> RadixKey for Update<T> {
    const LEVELS: usize = 4;

    #[inline]
    fn get_level(&self, level: usize) -> u8 {
        (self.target >> (level * 8)) as u8
    }
}

/// Apply sorted neighbour updates to the flat graph.
///
/// Separated by concrete type (f32/f64) because of thread-local heap
/// storage. The sorted layout allows lock-free processing since updates
/// targeting the same node are contiguous.
pub trait ApplySortedUpdates<T> {
    /// Apply sorted updates to the flat 1D graph.
    ///
    /// ## Algorithm
    ///
    /// 1. Find boundaries between different target nodes in the sorted updates.
    /// 2. Extract each target's update batch as a contiguous slice.
    /// 3. Process batches in parallel.
    /// 4. Merge new candidates with existing neighbours via thread-local heaps.
    /// 5. Write results back to the flat graph via disjoint pointer writes.
    ///
    /// ### Params
    ///
    /// * `updates` - Must be sorted by target (first element)
    /// * `graph` - Flat graph of size `n * k`
    /// * `k` - Neighbours per node
    /// * `updates_count` - Atomic counter for edge updates
    fn apply_sorted_updates(
        &self,
        updates: &[Update<T>],
        graph: &mut [Neighbour<T>],
        k: usize,
        updates_count: &AtomicUsize,
    );
}

////////////////////
// NNDescentQuery //
////////////////////

/// Query interface for the NN-Descent index.
pub trait NNDescentQuery<T> {
    /// Internal query dispatch (delegates to metric-specific implementation).
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `query_norm` - Pre-computed L2 norm (Cosine only; ignored for Euclidean)
    /// * `k` - Number of neighbours to return
    /// * `ef` - Beam width for search
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted by distance ascending
    fn query_internal(
        &self,
        query_vec: &[T],
        query_norm: T,
        k: usize,
        ef: usize,
    ) -> (Vec<usize>, Vec<T>);

    /// Beam search using Euclidean distance.
    fn query_euclidean(
        &self,
        query_vec: &[T],
        k: usize,
        ef: usize,
        visited: &mut FixedBitSet,
        candidates: &mut BinaryHeap<Reverse<(OrderedFloat<T>, usize)>>,
        results: &mut BinaryHeap<(OrderedFloat<T>, usize)>,
    ) -> (Vec<usize>, Vec<T>);

    /// Beam search using Cosine distance.
    #[allow(clippy::too_many_arguments)]
    fn query_cosine(
        &self,
        query_vec: &[T],
        query_norm: T,
        k: usize,
        ef: usize,
        visited: &mut FixedBitSet,
        candidates: &mut BinaryHeap<Reverse<(OrderedFloat<T>, usize)>>,
        results: &mut BinaryHeap<(OrderedFloat<T>, usize)>,
    ) -> (Vec<usize>, Vec<T>);
}

//////////
// Main //
//////////

/// NN-Descent index for approximate nearest neighbour search.
///
/// Builds a k-NN graph via the NN-Descent algorithm, using an Annoy
/// forest for initialisation and beam search for querying.
///
/// ### Flat graph layout
///
/// Both the build-phase graph (`Vec<Neighbour<T>>`) and the final query
/// graph (`Vec<(usize, T)>`) are stored as contiguous 1D arrays of size
/// `n * k`. Node `i`'s neighbours occupy indices `[i*k .. (i+1)*k]`,
/// sorted by distance ascending. Empty trailing slots are filled with
/// sentinel values (`SENTINEL_PID`, `T::MAX`).
///
/// This layout gives better cache locality during graph updates and
/// queries compared to a `Vec<Vec<...>>` and eliminates per-node heap
/// allocations entirely.
///
/// ### Memory-efficient update strategy
///
/// During construction, candidate updates are processed in chunks
/// (~50k source nodes) to bound peak memory to
/// `O(chunk_size * max_candidates)` rather than `O(n * max_candidates)`.
/// Each chunk emits both edge directions, sorts by target, and applies
/// updates lock-free via disjoint pointer writes.
pub struct NNDescent<T> {
    /// Original vectors, flattened row-major
    pub vectors_flat: Vec<T>,
    /// Dimensionality of the vectors
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Neighbours per node in the generated kNN graph
    pub k: usize,
    /// Pre-computed L2 norms (Cosine only; empty for Euclidean)
    pub norms: Vec<T>,
    /// Distance metric of the index
    metric: Dist,
    /// Annoy index initialisation and finding the entry points
    forest: AnnoyIndex<T>,
    /// Flat k-NN graph of size `n * k`
    graph: Vec<(usize, T)>,
    /// Whether construction converged
    converged: bool,
    /// Original indices - for trait purposes
    original_ids: Vec<usize>,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for NNDescent<T>
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

impl<T> NNDescent<T>
where
    T: AnnSearchFloat,
    Self: ApplySortedUpdates<T>,
    Self: NNDescentQuery<T>,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Build a new NN-Descent index.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (samples x features)
    /// * `metric` - Distance metric
    /// * `k` - Neighbours per node (default 30)
    /// * `max_candidates` - Max candidates per node per iteration
    /// * `max_iter` - Maximum iterations
    /// * `n_trees` - Annoy forest size
    /// * `delta` - Convergence threshold (fraction of edges updated)
    /// * `diversify_prob` - Probability of pruning redundant edges (0 to disable)
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data: MatRef<T>,
        metric: Dist,
        k: Option<usize>,
        max_candidates: Option<usize>,
        max_iter: Option<usize>,
        n_trees: Option<usize>,
        delta: T,
        diversify_prob: T,
        seed: usize,
        verbose: bool,
    ) -> Self {
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

        let n_trees = n_trees.unwrap_or_else(|| {
            let calculated = 5 + ((n as f64).powf(0.25)).round() as usize;
            calculated.min(32)
        });

        let max_iter = max_iter.unwrap_or_else(|| {
            let calculated = ((n as f64).log2().round()) as usize;
            calculated.max(5)
        });

        let k = k.unwrap_or(30);
        let max_candidates = max_candidates.unwrap_or(k.min(60));

        let start = Instant::now();
        let forest = AnnoyIndex::new(data, n_trees, metric, seed);
        if verbose {
            println!("Built Annoy: {:.2?}", start.elapsed());
        }

        let builder = NNDescent {
            vectors_flat,
            dim,
            n,
            k,
            metric,
            norms,
            graph: Vec::new(),
            converged: false,
            forest,
            original_ids: (0..n).collect(),
        };

        let (build_graph, converged) =
            builder.generate_index(k, max_iter, delta, max_candidates, seed, verbose);

        let graph = if diversify_prob > T::zero() {
            builder.diversify_graph(&build_graph, k, diversify_prob, seed)
        } else {
            build_graph
        };

        NNDescent {
            vectors_flat: builder.vectors_flat,
            dim: builder.dim,
            n: builder.n,
            k,
            metric: builder.metric,
            norms: builder.norms,
            graph,
            converged,
            forest: builder.forest,
            original_ids: (0..n).collect(),
        }
    }

    /// Whether the algorithm converged during construction.
    pub fn index_converged(&self) -> bool {
        self.converged
    }

    /// Compute chunk size for memory-bounded update processing
    ///
    /// Targets roughly 200 MB of update storage per chunk based on the size
    /// of an `Update<T>` and the expected number of updates per source node.
    /// Clamped to at least 10k nodes (or the full dataset if smaller) and at
    /// most the total number of nodes.
    ///
    /// ### Params
    ///
    /// * `max_candidates` - Maximum candidates sampled per node per iteration
    ///
    /// ### Returns
    ///
    /// Number of source nodes to process per chunk
    fn compute_chunk_size(&self, max_candidates: usize) -> usize {
        const TARGET_BYTES: usize = 200 * 1024 * 1024;
        const BYTES_PER_UPDATE: usize = 24;

        let updates_per_source = max_candidates * 2;
        let bytes_per_source = updates_per_source * BYTES_PER_UPDATE;

        let chunk_size = TARGET_BYTES / bytes_per_source.max(1);
        let min_chunk = 10_000.min(self.n);
        chunk_size.clamp(min_chunk, self.n)
    }

    /// Initialise the flat k-NN graph using the Annoy forest
    ///
    /// Each node queries Annoy for `k+1` candidates, skips the self-match,
    /// and takes the next `k`. Results are marked new so they participate
    /// in the first iteration's local joins. Unused trailing slots are
    /// padded with sentinels.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per node
    ///
    /// ### Returns
    ///
    /// Flat graph of size `n * k` with Annoy-seeded initial neighbours
    fn init_with_forest(&self, k: usize) -> Vec<Neighbour<T>> {
        let sentinel = Neighbour::new(SENTINEL_PID, T::max_value(), false);
        let mut graph = vec![sentinel; self.n * k];

        graph.par_chunks_mut(k).enumerate().for_each(|(i, slot)| {
            let query = &self.vectors_flat[i * self.dim..(i + 1) * self.dim];
            let search_k = k * self.forest.n_trees;
            let (indices, distances) = self.forest.query(query, k + 1, Some(search_k));

            for (j, (idx, dist)) in indices
                .into_iter()
                .zip(distances)
                .skip(1)
                .take(k)
                .enumerate()
            {
                slot[j] = Neighbour::new(idx, dist, true);
            }
        });

        graph
    }

    /// Run the main NN-Descent algorithm with chunked updates
    ///
    /// Alternates between building candidate lists (new and old, forward and
    /// reverse) and applying pairwise distance updates back into the graph.
    /// Processes candidates in chunks to bound peak memory. Terminates early
    /// when the fraction of edge updates drops below `delta`.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per node
    /// * `max_iter` - Maximum iterations before giving up
    /// * `delta` - Convergence threshold (fraction of edges updated)
    /// * `max_candidates` - Max candidates sampled per node per iteration
    /// * `seed` - Random seed for per-iteration sampling
    /// * `verbose` - Print progress information
    ///
    /// ### Returns
    ///
    /// Tuple of (flat graph as `(pid, dist)` pairs, converged flag)
    fn generate_index(
        &self,
        k: usize,
        max_iter: usize,
        delta: T,
        max_candidates: usize,
        seed: usize,
        verbose: bool,
    ) -> (Vec<(usize, T)>, bool) {
        if verbose {
            println!(
                "Running NN-Descent: {} samples, max_candidates={}",
                self.n.separate_with_underscores(),
                max_candidates
            );
        }

        let mut converged = false;

        let start = Instant::now();
        let mut graph = self.init_with_forest(k);

        if verbose {
            println!("Queried Annoy index: {:.2?}", start.elapsed());
        }

        let chunk_size = self.compute_chunk_size(max_candidates);
        let n_chunks = self.n.div_ceil(chunk_size);

        if verbose {
            println!(
                " Using chunk size {} ({} chunks) for memory-efficient updates",
                chunk_size.separate_with_underscores(),
                n_chunks
            );
        }

        let mut new_cands = vec![Vec::with_capacity(max_candidates * 2); self.n];
        let mut old_cands = vec![Vec::with_capacity(max_candidates * 2); self.n];
        let mut new_cands_sym = vec![Vec::with_capacity(max_candidates); self.n];
        let mut old_cands_sym = vec![Vec::with_capacity(max_candidates); self.n];

        for iter in 0..max_iter {
            let updates_count = AtomicUsize::new(0);
            let iter_seed = (seed as u64).wrapping_add(iter as u64);

            if verbose {
                println!(" Preparing candidates for iter {}", iter + 1);
            }
            self.build_candidates(
                &graph,
                k,
                max_candidates,
                iter_seed,
                &mut new_cands,
                &mut old_cands,
                &mut new_cands_sym,
                &mut old_cands_sym,
            );

            self.mark_as_old(&mut graph, k, &new_cands);

            if verbose {
                println!(
                    " Processing updates for iter {} ({} chunks)",
                    iter + 1,
                    n_chunks
                );
            }

            for chunk_idx in 0..n_chunks {
                let chunk_start = chunk_idx * chunk_size;
                let chunk_end = (chunk_start + chunk_size).min(self.n);

                let mut chunk_updates = self.generate_updates_for_chunk(
                    &new_cands,
                    &old_cands,
                    &graph,
                    k,
                    chunk_start,
                    chunk_end,
                );

                chunk_updates.radix_sort_unstable();

                self.apply_sorted_updates(&chunk_updates, &mut graph, k, &updates_count);
            }

            let update_count = updates_count.load(Ordering::Relaxed);
            let update_rate = T::from_usize(update_count).unwrap()
                / T::from_usize(self.n * max_candidates).unwrap();

            if verbose {
                println!(
                    "  Iter {}: {} edge updates (rate={:.4})",
                    iter + 1,
                    update_count.separate_with_underscores(),
                    update_rate.to_f64().unwrap(),
                );
            }

            if update_rate < delta {
                if verbose {
                    println!("  Converged after {} iterations", iter + 1);
                }
                converged = true;
                break;
            }
        }

        if verbose {
            println!("Total time: {:.2?}", start.elapsed());
        }

        let res = graph.into_iter().map(|n| (n.pid(), n.dist)).collect();

        (res, converged)
    }

    /// Build candidate lists for the local join step.
    ///
    /// For each node, samples up to `max_candidates` new and old neighbours,
    /// then adds symmetric reverse candidates to ensure connectivity.
    ///
    /// ### Params
    ///
    /// * `graph` - Current flat k-NN graph
    /// * `k` - Neighbours per node
    /// * `max_candidates` - Maximum candidates to sample per node
    /// * `iter_seed` - Per-iteration seed for reproducible sampling
    /// * `new_cands` - Output: sampled new (unexplored) neighbours per node
    /// * `old_cands` - Output: sampled old (explored) neighbours per node
    /// * `new_cands_sym` - Output: reverse edges into `new_cands` (cleared and
    ///   repopulated)
    /// * `old_cands_sym` - Output: reverse edges into `old_cands` (cleared and
    ///   repopulated)
    #[allow(clippy::too_many_arguments)]
    fn build_candidates(
        &self,
        graph: &[Neighbour<T>],
        k: usize,
        max_candidates: usize,
        iter_seed: u64,
        new_cands: &mut [Vec<usize>],
        old_cands: &mut [Vec<usize>],
        new_cands_sym: &mut [Vec<usize>],
        old_cands_sym: &mut [Vec<usize>],
    ) {
        for v in new_cands_sym.iter_mut().chain(old_cands_sym.iter_mut()) {
            v.clear();
        }

        // Phase 1: Parallel sampling - each thread writes only to its own node
        let n = self.n;
        new_cands
            .par_iter_mut()
            .zip(old_cands.par_iter_mut())
            .enumerate()
            .for_each(|(i, (new_c, old_c))| {
                new_c.clear();
                old_c.clear();

                let mut rng = SmallRng::seed_from_u64(iter_seed.wrapping_add(i as u64));
                let base = i * k;

                let mut new_temp: Vec<(f64, usize)> = Vec::new();
                let mut old_temp: Vec<(f64, usize)> = Vec::new();

                for slot in &graph[base..base + k] {
                    if slot.is_sentinel() {
                        continue;
                    }
                    let j = slot.pid();
                    if j >= n {
                        continue;
                    }

                    let priority = rng.random::<f64>();
                    if slot.is_new() {
                        new_temp.push((priority, j));
                    } else {
                        old_temp.push((priority, j));
                    }
                }

                // O(n) partial sort instead of O(n log n) full sort
                if new_temp.len() > max_candidates {
                    new_temp.select_nth_unstable_by(max_candidates - 1, |a, b| {
                        a.0.partial_cmp(&b.0).unwrap()
                    });
                    new_temp.truncate(max_candidates);
                }
                new_c.extend(new_temp.iter().map(|&(_, idx)| idx));

                if old_temp.len() > max_candidates {
                    old_temp.select_nth_unstable_by(max_candidates - 1, |a, b| {
                        a.0.partial_cmp(&b.0).unwrap()
                    });
                    old_temp.truncate(max_candidates);
                }
                old_c.extend(old_temp.iter().map(|&(_, idx)| idx));
            });

        // Phase 2: Symmetric candidates (sequential - cross-node writes)
        for i in 0..self.n {
            for &j in &new_cands[i] {
                if j < self.n {
                    new_cands_sym[j].push(i);
                }
            }
            for &j in &old_cands[i] {
                if j < self.n {
                    old_cands_sym[j].push(i);
                }
            }
        }

        // Phase 3: Merge symmetric, sort, dedup (parallel, per-node independent)
        new_cands
            .par_iter_mut()
            .zip(old_cands.par_iter_mut())
            .zip(new_cands_sym.par_iter())
            .zip(old_cands_sym.par_iter())
            .for_each(|(((new_c, old_c), new_sym), old_sym)| {
                new_c.extend_from_slice(new_sym);
                new_c.sort_unstable();
                new_c.dedup();

                old_c.extend_from_slice(old_sym);
                old_c.sort_unstable();
                old_c.dedup();
            });
    }

    /// Mark neighbours as old if they were sampled into the new-candidate list
    ///
    /// After sampling, any neighbour that survived into `new_cands[i]` will
    /// have been "explored" during this iteration's local joins, so flip its
    /// flag so subsequent iterations treat it as old.
    ///
    /// ### Params
    ///
    /// * `graph` - Current flat k-NN graph (mutated in place)
    /// * `k` - Neighbours per node
    /// * `new_cands` - Sorted new-candidate lists per node
    fn mark_as_old(&self, graph: &mut [Neighbour<T>], k: usize, new_cands: &[Vec<usize>]) {
        for i in 0..self.n {
            if new_cands[i].is_empty() {
                continue;
            }

            let base = i * k;
            for slot in &mut graph[base..base + k] {
                if slot.is_sentinel() {
                    continue;
                }
                if slot.is_new() && new_cands[i].binary_search(&slot.pid()).is_ok() {
                    slot.mark_old();
                }
            }
        }
    }

    /// Generate distance updates from a chunk of source nodes.
    ///
    /// Emits both edge directions `(p, q, d)` and `(q, p, d)` so that
    /// the caller can sort by target and apply lock-free.
    ///
    /// ### Params
    ///
    /// * `new_cands` - New (unexplored) candidate lists per node
    /// * `old_cands` - Old (explored) candidate lists per node
    /// * `graph` - Current flat k-NN graph
    /// * `k` - Neighbours per node
    /// * `chunk_start` - First source node index (inclusive)
    /// * `chunk_end` - Last source node index (exclusive)
    ///
    /// ### Returns
    ///
    /// Unsorted list of `(target, source, distance)` update triples
    fn generate_updates_for_chunk(
        &self,
        new_cands: &[Vec<usize>],
        old_cands: &[Vec<usize>],
        graph: &[Neighbour<T>],
        k: usize,
        chunk_start: usize,
        chunk_end: usize,
    ) -> Vec<Update<T>> {
        (chunk_start..chunk_end)
            .into_par_iter()
            .fold(
                || Vec::with_capacity(2048),
                |mut updates, i| {
                    let get_threshold = |idx: usize| -> T { graph[idx * k + k - 1].dist };

                    // new-new pairs
                    for j in 0..new_cands[i].len() {
                        let p = new_cands[i][j];
                        if p >= self.n {
                            continue;
                        }
                        let p_threshold = get_threshold(p);

                        for l in (j + 1)..new_cands[i].len() {
                            let q = new_cands[i][l];
                            if q >= self.n || p == q {
                                continue;
                            }
                            let d = self.distance(p, q);
                            if d <= p_threshold || d <= get_threshold(q) {
                                updates.push(Update::new(p, q, d));
                                updates.push(Update::new(q, p, d));
                            }
                        }
                    }

                    // new-old pairs
                    for &p in &new_cands[i] {
                        if p >= self.n {
                            continue;
                        }
                        let p_threshold = get_threshold(p);

                        for &q in &old_cands[i] {
                            if q >= self.n || p == q {
                                continue;
                            }
                            let d = self.distance(p, q);
                            if d <= p_threshold || d <= get_threshold(q) {
                                updates.push(Update::new(p, q, d));
                                updates.push(Update::new(q, p, d));
                            }
                        }
                    }

                    updates
                },
            )
            .reduce(Vec::new, |mut a, mut b| {
                if a.len() >= b.len() {
                    a.extend_from_slice(&b);
                    a
                } else {
                    b.extend_from_slice(&a);
                    b
                }
            })
    }

    /// Calculate distance between two indexed points under the index metric
    ///
    /// ### Params
    ///
    /// * `i` - Index of first vector
    /// * `j` - Index of second vector
    ///
    /// ### Returns
    ///
    /// Distance value under `self.metric`
    #[inline]
    fn distance(&self, i: usize, j: usize) -> T {
        match self.metric {
            Dist::Euclidean => self.euclidean_distance(i, j),
            Dist::Cosine => self.cosine_distance(i, j),
        }
    }

    /// Diversify the graph by probabilistically pruning redundant edges
    ///
    /// Walks each node's neighbour list in distance order, keeping the closest
    /// and discarding candidates that are closer to an already-kept neighbour
    /// than to the query node (with probability `prune_prob`). Operates on the
    /// final `(pid, dist)` graph. Pruned slots are padded with sentinels.
    ///
    /// ### Params
    ///
    /// * `graph` - Input flat graph
    /// * `k` - Neighbours per node
    /// * `prune_prob` - Probability of pruning a redundant edge
    /// * `seed` - Random seed for per-node pruning decisions
    ///
    /// ### Returns
    ///
    /// New graph with redundant edges replaced by sentinels
    fn diversify_graph(
        &self,
        graph: &[(usize, T)],
        k: usize,
        prune_prob: T,
        seed: usize,
    ) -> Vec<(usize, T)> {
        let mut result = vec![(SENTINEL_PID, T::max_value()); self.n * k];

        result
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(i, out_slot)| {
                let base = i * k;
                let neighbours = &graph[base..base + k];

                // Collect non-sentinel neighbours
                let valid: Vec<(usize, T)> = neighbours
                    .iter()
                    .copied()
                    .filter(|&(pid, _)| pid != SENTINEL_PID)
                    .collect();

                if valid.is_empty() {
                    return;
                }

                let mut rng = SmallRng::seed_from_u64((seed as u64).wrapping_add(i as u64));
                let mut kept = vec![valid[0]];

                for &(cand_idx, cand_dist) in &valid[1..] {
                    let mut should_keep = true;

                    for &(kept_idx, kept_dist) in &kept {
                        let dist_to_kept = self.distance(cand_idx, kept_idx);

                        if kept_dist > T::from_f32(f32::EPSILON).unwrap()
                            && dist_to_kept < cand_dist
                            && rng.random::<f64>() < prune_prob.to_f64().unwrap()
                        {
                            should_keep = false;
                            break;
                        }
                    }

                    if should_keep {
                        kept.push((cand_idx, cand_dist));
                    }
                }

                for (j, &entry) in kept.iter().enumerate() {
                    out_slot[j] = entry;
                }
            });

        result
    }

    ///////////
    // Query //
    ///////////

    /// Return the neighbours slice for node `idx` from the query graph
    ///
    /// ### Params
    ///
    /// * `idx` - Node index
    ///
    /// ### Returns
    ///
    /// Slice of `k` `(pid, distance)` pairs, possibly containing sentinels
    #[inline]
    fn graph_neighbours(&self, idx: usize) -> &[(usize, T)] {
        &self.graph[idx * self.k..(idx + 1) * self.k]
    }

    /// Query for k nearest neighbours using beam search.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Beam width. Higher values improve recall at the
    ///   cost of latency. Defaults to `max(2*k, 50)` clamped to 200.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted by distance ascending
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        ef_search: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        let k = k.min(self.n);
        let ef = ef_search.unwrap_or_else(|| (k * 2).clamp(50, 200)).max(k);

        let query_norm = if self.metric == Dist::Cosine {
            query_vec.iter().map(|x| *x * *x).sum::<T>().sqrt()
        } else {
            T::one()
        };

        self.query_internal(query_vec, query_norm, k, ef)
    }

    /// Query using a matrix row reference.
    ///
    /// Uses a zero-copy path when stride is 1, otherwise copies to a
    /// temporary vector.
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        ef_search: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, ef_search);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, ef_search)
    }

    /// Generate a kNN graph by querying every vector in the index.
    ///
    /// ### Returns
    ///
    /// `(knn_indices, optional distances)`
    pub fn generate_knn(
        &self,
        k: usize,
        ef_search: Option<usize>,
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

                self.query(vec, k, ef_search)
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

    /// Total heap memory used by the index, in bytes.
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);

        total += self.vectors_flat.capacity() * std::mem::size_of::<T>();
        total += self.norms.capacity() * std::mem::size_of::<T>();
        total += self.forest.memory_usage_bytes();
        total += self.graph.capacity() * std::mem::size_of::<(usize, T)>();

        total
    }
}

/// Find boundaries between different target nodes in sorted updates
///
/// Given a slice sorted by `target`, returns the indices where the target
/// changes, bracketed by 0 and `updates.len()`. The resulting `windows(2)`
/// gives per-target slice ranges suitable for lock-free parallel application.
///
/// ### Params
///
/// * `updates` - Updates sorted by target
///
/// ### Returns
///
/// Vector of boundary indices of length `num_distinct_targets + 1`
fn find_target_boundaries<T>(updates: &[Update<T>]) -> Vec<usize> {
    if updates.is_empty() {
        return vec![0, 0];
    }

    let mut boundaries = vec![0];

    for i in 1..updates.len() {
        if updates[i].target != updates[i - 1].target {
            boundaries.push(i);
        }
    }

    boundaries.push(updates.len());
    boundaries
}

///////////////////////////
// Trait implementations //
///////////////////////////

////////////////////////
// ApplySortedUpdates //
////////////////////////

/// Generates the `ApplySortedUpdates` impl for a concrete float type.
///
/// The logic is identical for f32 and f64; only the thread-local storage
/// keys differ.
macro_rules! impl_apply_sorted_updates {
    ($float:ty, $heap_tls:ident, $sort_buf_tls:ident) => {
        impl ApplySortedUpdates<$float> for NNDescent<$float> {
            fn apply_sorted_updates(
                &self,
                updates: &[Update<$float>],
                graph: &mut [Neighbour<$float>],
                k: usize,
                updates_count: &AtomicUsize,
            ) {
                if updates.is_empty() {
                    return;
                }

                let boundaries = find_target_boundaries(updates);

                let segments: Vec<(usize, &[Update<$float>])> = boundaries
                    .windows(2)
                    .filter_map(|w| {
                        let start = w[0];
                        let end = w[1];
                        if start < end {
                            Some((updates[start].target, &updates[start..end]))
                        } else {
                            None
                        }
                    })
                    .collect();

                let graph_ptr = UnsafeGraphPtr(graph.as_mut_ptr());

                segments.par_iter().for_each(|&(target, segment)| {
                    #[allow(clippy::redundant_locals)]
                    let graph_ptr = graph_ptr;
                    $heap_tls.with(|heap_cell| {
                        PID_SET.with(|set_cell| {
                            $sort_buf_tls.with(|sort_cell| {
                                let mut heap = heap_cell.borrow_mut();
                                let mut pid_set = set_cell.borrow_mut();
                                let mut sort_buf = sort_cell.borrow_mut();

                                heap.clear();
                                if pid_set.len() < self.n {
                                    pid_set.resize(self.n, false);
                                }

                                let start_idx = target * k;

                                // SAFETY: Each thread processes a unique target.
                                // No two threads alias the same slice.
                                let target_slice = unsafe {
                                    std::slice::from_raw_parts_mut(graph_ptr.0.add(start_idx), k)
                                };

                                let mut edge_updates = 0usize;

                                // Load current neighbours into the heap
                                for n in target_slice.iter() {
                                    if n.is_sentinel() {
                                        continue;
                                    }
                                    let pid = n.pid();
                                    heap.push((OrderedFloat(n.dist), pid, n.is_new()));
                                    pid_set[pid] = true;
                                }

                                // Merge incoming updates
                                for update in segment {
                                    if pid_set[update.source] {
                                        continue;
                                    }

                                    if heap.len() < k {
                                        heap.push((OrderedFloat(update.dist), update.source, true));
                                        pid_set[update.source] = true;
                                        edge_updates += 1;
                                    } else if let Some(&(OrderedFloat(worst), _, _)) = heap.peek() {
                                        if update.dist < worst {
                                            if let Some((_, old_pid, _)) = heap.pop() {
                                                pid_set[old_pid] = false;
                                            }
                                            heap.push((
                                                OrderedFloat(update.dist),
                                                update.source,
                                                true,
                                            ));
                                            pid_set[update.source] = true;
                                            edge_updates += 1;
                                        }
                                    }
                                }

                                if edge_updates > 0 {
                                    updates_count.fetch_add(edge_updates, Ordering::Relaxed);

                                    sort_buf.clear();
                                    sort_buf.extend(heap.drain().map(
                                        |(OrderedFloat(d), p, is_new)| {
                                            pid_set[p] = false;
                                            (d, p, is_new)
                                        },
                                    ));

                                    sort_buf
                                        .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                                    sort_buf.truncate(k);

                                    for (i, &(d, p, is_new)) in sort_buf.iter().enumerate() {
                                        target_slice[i] = Neighbour::new(p, d, is_new);
                                    }

                                    for i in sort_buf.len()..k {
                                        target_slice[i] =
                                            Neighbour::new(SENTINEL_PID, <$float>::MAX, false);
                                    }
                                } else {
                                    for (_, pid, _) in heap.iter() {
                                        pid_set[*pid] = false;
                                    }
                                }
                            })
                        })
                    })
                });
            }
        }
    };
}

impl_apply_sorted_updates!(f32, HEAP_F32, SORT_BUF_F32);
impl_apply_sorted_updates!(f64, HEAP_F64, SORT_BUF_F64);

////////////////////
// NNDescentQuery //
////////////////////

/// Generates the `NNDescentQuery` impl for a concrete float type.
macro_rules! impl_nndescent_query {
    ($float:ty, $cand_tls:ident, $res_tls:ident) => {
        impl NNDescentQuery<$float> for NNDescent<$float> {
            fn query_internal(
                &self,
                query_vec: &[$float],
                query_norm: $float,
                k: usize,
                ef: usize,
            ) -> (Vec<usize>, Vec<$float>) {
                QUERY_VISITED.with(|visited_cell| {
                    $cand_tls.with(|cand_cell| {
                        $res_tls.with(|res_cell| {
                            let mut visited = visited_cell.borrow_mut();
                            let mut candidates = cand_cell.borrow_mut();
                            let mut results = res_cell.borrow_mut();

                            visited.clear();
                            visited.grow(self.n);
                            candidates.clear();
                            results.clear();

                            match self.metric {
                                Dist::Euclidean => self.query_euclidean(
                                    query_vec,
                                    k,
                                    ef,
                                    &mut visited,
                                    &mut candidates,
                                    &mut results,
                                ),
                                Dist::Cosine => self.query_cosine(
                                    query_vec,
                                    query_norm,
                                    k,
                                    ef,
                                    &mut visited,
                                    &mut candidates,
                                    &mut results,
                                ),
                            }
                        })
                    })
                })
            }

            #[inline(always)]
            fn query_euclidean(
                &self,
                query_vec: &[$float],
                k: usize,
                ef: usize,
                visited: &mut FixedBitSet,
                candidates: &mut BinaryHeap<Reverse<(OrderedFloat<$float>, usize)>>,
                results: &mut BinaryHeap<(OrderedFloat<$float>, usize)>,
            ) -> (Vec<usize>, Vec<$float>) {
                let init_candidates = (ef / 2).max(2 * k).min(self.n);
                let search_k = init_candidates * 3;
                let (init_indices, _) =
                    self.forest
                        .query(query_vec, init_candidates, Some(search_k));

                for &entry_idx in &init_indices {
                    if entry_idx >= self.n || visited.contains(entry_idx) {
                        continue;
                    }
                    visited.insert(entry_idx);
                    let dist = self.euclidean_distance_to_query(entry_idx, query_vec);
                    candidates.push(Reverse((OrderedFloat(dist), entry_idx)));
                    results.push((OrderedFloat(dist), entry_idx));
                }

                while results.len() > ef {
                    results.pop();
                }

                let mut lower_bound = if results.len() >= ef {
                    results.peek().unwrap().0 .0
                } else {
                    <$float>::MAX
                };

                while let Some(Reverse((OrderedFloat(curr_dist), curr_idx))) = candidates.pop() {
                    if curr_dist > lower_bound {
                        break;
                    }

                    for &(nbr_idx, _) in self.graph_neighbours(curr_idx) {
                        if nbr_idx == SENTINEL_PID || visited.contains(nbr_idx) {
                            continue;
                        }
                        visited.insert(nbr_idx);

                        let dist = self.euclidean_distance_to_query(nbr_idx, query_vec);

                        if dist < lower_bound || results.len() < ef {
                            candidates.push(Reverse((OrderedFloat(dist), nbr_idx)));

                            if results.len() < ef {
                                results.push((OrderedFloat(dist), nbr_idx));
                                if results.len() == ef {
                                    lower_bound = results.peek().unwrap().0 .0;
                                }
                            } else if dist < lower_bound {
                                results.pop();
                                results.push((OrderedFloat(dist), nbr_idx));
                                lower_bound = results.peek().unwrap().0 .0;
                            }
                        }
                    }
                }

                let mut final_results: Vec<_> = results.drain().collect();
                final_results.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                final_results.truncate(k);

                final_results
                    .into_iter()
                    .map(|(OrderedFloat(d), i)| (i, d))
                    .unzip()
            }

            #[inline(always)]
            fn query_cosine(
                &self,
                query_vec: &[$float],
                query_norm: $float,
                k: usize,
                ef: usize,
                visited: &mut FixedBitSet,
                candidates: &mut BinaryHeap<Reverse<(OrderedFloat<$float>, usize)>>,
                results: &mut BinaryHeap<(OrderedFloat<$float>, usize)>,
            ) -> (Vec<usize>, Vec<$float>) {
                let init_candidates = (ef / 2).max(k).min(self.n);
                let search_k = init_candidates * 3;
                let (init_indices, _) =
                    self.forest
                        .query(query_vec, init_candidates, Some(search_k));

                for &entry_idx in &init_indices {
                    if entry_idx >= self.n || visited.contains(entry_idx) {
                        continue;
                    }
                    visited.insert(entry_idx);
                    let dist = self.cosine_distance_to_query(entry_idx, query_vec, query_norm);
                    candidates.push(Reverse((OrderedFloat(dist), entry_idx)));
                    results.push((OrderedFloat(dist), entry_idx));
                }

                while results.len() > ef {
                    results.pop();
                }

                let mut lower_bound = if results.len() >= ef {
                    results.peek().unwrap().0 .0
                } else {
                    <$float>::MAX
                };

                while let Some(Reverse((OrderedFloat(curr_dist), curr_idx))) = candidates.pop() {
                    if curr_dist > lower_bound {
                        break;
                    }

                    for &(nbr_idx, _) in self.graph_neighbours(curr_idx) {
                        if nbr_idx == SENTINEL_PID || visited.contains(nbr_idx) {
                            continue;
                        }
                        visited.insert(nbr_idx);

                        let dist = self.cosine_distance_to_query(nbr_idx, query_vec, query_norm);

                        if dist < lower_bound || results.len() < ef {
                            candidates.push(Reverse((OrderedFloat(dist), nbr_idx)));

                            if results.len() < ef {
                                results.push((OrderedFloat(dist), nbr_idx));
                                if results.len() == ef {
                                    lower_bound = results.peek().unwrap().0 .0;
                                }
                            } else if dist < lower_bound {
                                results.pop();
                                results.push((OrderedFloat(dist), nbr_idx));
                                lower_bound = results.peek().unwrap().0 .0;
                            }
                        }
                    }
                }

                let mut final_results: Vec<_> = results.drain().collect();
                final_results.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                final_results.truncate(k);

                final_results
                    .into_iter()
                    .map(|(OrderedFloat(d), i)| (i, d))
                    .unzip()
            }
        }
    };
}

impl_nndescent_query!(f32, QUERY_CANDIDATES_F32, QUERY_RESULTS_F32);
impl_nndescent_query!(f64, QUERY_CANDIDATES_F64, QUERY_RESULTS_F64);

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for NNDescent<T>
where
    T: AnnSearchFloat,
    Self: ApplySortedUpdates<T>,
    Self: NNDescentQuery<T>,
{
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        // Default budget
        self.query(query_vec, k, None)
    }

    fn n(&self) -> usize {
        self.n
    }

    fn metric(&self) -> Dist {
        self.metric
    }

    fn original_ids(&self) -> &[usize] {
        &self.original_ids
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn create_simple_matrix() -> Mat<f32> {
        let data = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
    }

    /// Return the non-sentinel neighbours for node `i`.
    fn neighbours(index: &NNDescent<f32>, i: usize) -> Vec<(usize, f32)> {
        index.graph[i * index.k..(i + 1) * index.k]
            .iter()
            .copied()
            .filter(|&(pid, _)| pid != SENTINEL_PID)
            .collect()
    }

    fn neighbours_f64(index: &NNDescent<f64>, i: usize) -> Vec<(usize, f64)> {
        index.graph[i * index.k..(i + 1) * index.k]
            .iter()
            .copied()
            .filter(|&(pid, _)| pid != SENTINEL_PID)
            .collect()
    }

    #[test]
    fn test_nndescent_build_euclidean() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Euclidean,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        );

        assert_eq!(index.graph.len(), 5 * 3);
        for i in 0..5 {
            assert!(neighbours(&index, i).len() <= 3);
        }
    }

    #[test]
    fn test_nndescent_build_cosine() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Cosine,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        );

        assert_eq!(index.graph.len(), 5 * 3);
        assert!(!index.norms.is_empty());
    }

    #[test]
    fn test_nndescent_query() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Euclidean,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        );

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, Some(50));

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
        assert!(indices.contains(&0));
    }

    #[test]
    fn test_nndescent_convergence() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Euclidean,
            Some(3),
            None,
            Some(100),
            None,
            0.5,
            0.0,
            42,
            false,
        );

        assert_eq!(index.graph.len(), 5 * 3);
    }

    #[test]
    fn test_nndescent_reproducibility() {
        let mat = create_simple_matrix();

        let g1 = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Euclidean,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        );
        let g2 = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Euclidean,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        );

        assert_eq!(g1.graph.len(), g2.graph.len());
        for i in 0..g1.n {
            assert_eq!(neighbours(&g1, i).len(), neighbours(&g2, i).len());
        }
    }

    #[test]
    fn test_nndescent_k_parameter() {
        let mat = create_simple_matrix();

        let gk2 = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Euclidean,
            Some(2),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        );
        let gk4 = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Euclidean,
            Some(4),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        );

        for i in 0..5 {
            assert!(neighbours(&gk2, i).len() <= 2);
        }
        for i in 0..5 {
            assert!(neighbours(&gk4, i).len() <= 4);
        }
    }

    #[test]
    fn test_nndescent_larger_dataset() {
        let n = 50;
        let dim = 10;
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                data.push((i * j) as f32 / 10.0);
            }
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Euclidean,
            Some(10),
            None,
            Some(15),
            None,
            0.001,
            0.0,
            42,
            false,
        );

        assert_eq!(index.graph.len(), n * 10);
        for i in 0..n {
            let nbrs = neighbours(&index, i);
            assert!(nbrs.len() <= 10);
            assert!(!nbrs.is_empty());
        }
    }

    #[test]
    fn test_nndescent_distance_ordering() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Euclidean,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        );

        for i in 0..5 {
            let nbrs = neighbours(&index, i);
            for w in nbrs.windows(2) {
                assert!(w[1].1 >= w[0].1);
            }
        }
    }

    #[test]
    fn test_nndescent_with_f64() {
        let data = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);

        let index = NNDescent::<f64>::new(
            mat.as_ref(),
            Dist::Euclidean,
            Some(2),
            None,
            Some(10),
            None,
            0.001,
            0.0,
            42,
            false,
        );

        assert_eq!(index.graph.len(), 3 * 2);
        for i in 0..3 {
            assert!(!neighbours_f64(&index, i).is_empty());
        }
    }

    #[test]
    fn test_nndescent_quality() {
        let n = 20;
        let dim = 3;
        let mut data = Vec::with_capacity(n * dim);

        for i in 0..10 {
            let offset = i as f32 * 0.1;
            data.extend_from_slice(&[offset, 0.0, 0.0]);
        }
        for i in 0..10 {
            let offset = 10.0 + i as f32 * 0.1;
            data.extend_from_slice(&[offset, 0.0, 0.0]);
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Euclidean,
            Some(5),
            None,
            Some(20),
            None,
            0.001,
            0.0,
            42,
            false,
        );

        let nbrs_0 = neighbours(&index, 0);
        let in_cluster = nbrs_0.iter().filter(|(idx, _)| *idx < 10).count();
        assert!(in_cluster >= 3);

        let nbrs_10 = neighbours(&index, 10);
        let in_cluster_2 = nbrs_10.iter().filter(|(idx, _)| *idx >= 10).count();
        assert!(in_cluster_2 >= 3);
    }

    #[test]
    fn test_nndescent_diversify() {
        let mat = create_simple_matrix();
        let index = NNDescent::<f32>::new(
            mat.as_ref(),
            Dist::Euclidean,
            Some(3),
            None,
            Some(10),
            None,
            0.001,
            0.5,
            42,
            false,
        );

        assert_eq!(index.graph.len(), 5 * 3);
        for i in 0..5 {
            assert!(!neighbours(&index, i).is_empty());
        }
    }
}
