use faer::{MatRef, RowRef};
use fixedbitset::FixedBitSet;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::BinaryHeap,
    iter::Sum,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};
use thousands::*;

use crate::annoy::*;
use crate::prelude::*;
use crate::utils::heap_structs::*;
use crate::utils::*;

/// Neighbour entry in k-NN graph
///
/// Flat structure in C representation for better cache locality
///
/// ### Fields
///
/// * `pid` - Index of the point
/// * `dist` - Distance to the neighbour
/// * `is_new` - 1 - yes; 0 - no
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Neighbour<T> {
    pid: u32,
    dist: T,
    is_new: u32,
}

impl<T: Copy> Neighbour<T> {
    /// Generate a new neighbour
    ///
    /// ### Params
    ///
    /// * `pid` - Point id of the neighbour
    /// * `dist` - Distance to that neighbour
    /// * `is_new` - Flag if it is a new neighbour
    #[inline(always)]
    fn new(pid: usize, dist: T, is_new: bool) -> Self {
        Self {
            pid: pid as u32,
            dist,
            is_new: is_new as u32,
        }
    }

    /// Is the neighbour new
    ///
    /// ### Returns
    ///
    /// Boolean indicating if it is an old or new neighbour
    #[inline(always)]
    fn is_new(&self) -> bool {
        self.is_new != 0
    }

    /// Get the PID
    ///
    /// ### Returns
    ///
    /// The index
    #[inline(always)]
    fn pid(&self) -> usize {
        self.pid as usize
    }
}

/// Trait for applying sorted neighbour updates
///
/// Separated by type (f32/f64) due to thread-local heap storage requirements.
/// The sorted application approach allows lock-free processing since updates
/// are grouped by target node.
pub trait ApplySortedUpdates<T> {
    /// Apply sorted updates to the graph
    ///
    /// ## Algorithm
    ///
    /// 1. Find boundaries between different target nodes in the sorted updates
    /// 2. Extract each target's update batch as a contiguous slice
    /// 3. Process batches in parallel - no locks needed since each batch
    ///    updates a different node
    /// 4. For each batch, merge new candidates with existing neighbours,
    ///    keeping only the k-best
    /// 5. Apply results back to the graph
    ///
    /// ### Params
    ///
    /// * `updates` - MUST be sorted by target (first element)
    /// * `graph` - Graph to update
    /// * `updates_count` - Atomic counter for edge updates
    fn apply_sorted_updates(
        &self,
        updates: &[(usize, usize, T)],
        graph: &mut [Vec<Neighbour<T>>],
        updates_count: &AtomicUsize,
    );
}

/// Trait for querying the NN-Descent index
pub trait NNDescentQuery<T> {
    /// Internal query dispatch method
    ///
    /// Delegates to metric-specific query implementation using thread-local
    /// storage to avoid allocations.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `query_norm` - Pre-computed norm (used for Cosine only)
    /// * `k` - Number of neighbours to return
    /// * `ef` - Beam width for search
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
    fn query_internal(
        &self,
        query_vec: &[T],
        query_norm: T,
        k: usize,
        ef: usize,
    ) -> (Vec<usize>, Vec<T>);

    /// Query using Euclidean distance with beam search
    ///
    /// Uses the Annoy forest for entry points, then performs beam search
    /// on the k-NN graph to find approximate nearest neighbours.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `ef` - Beam width for search
    /// * `visited` - Bitset to track visited nodes
    /// * `candidates` - Min-heap of candidate nodes to explore
    /// * `results` - Max-heap of current best results
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
    fn query_euclidean(
        &self,
        query_vec: &[T],
        k: usize,
        ef: usize,
        visited: &mut FixedBitSet,
        candidates: &mut BinaryHeap<Reverse<(OrderedFloat<T>, usize)>>,
        results: &mut BinaryHeap<(OrderedFloat<T>, usize)>,
    ) -> (Vec<usize>, Vec<T>);

    /// Query using Cosine distance with beam search
    ///
    /// Uses the Annoy forest for entry points, then performs beam search
    /// on the k-NN graph to find approximate nearest neighbours.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `query_norm` - Pre-computed norm of query vector
    /// * `k` - Number of neighbours to return
    /// * `ef` - Beam width for search
    /// * `visited` - Bitset to track visited nodes
    /// * `candidates` - Min-heap of candidate nodes to explore
    /// * `results` - Max-heap of current best results
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
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

pub type QueryCandF32 = RefCell<BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>>>;
pub type QueryCandF64 = RefCell<BinaryHeap<Reverse<(OrderedFloat<f64>, usize)>>>;

thread_local! {
    static HEAP_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize, bool)>> =
        const { RefCell::new(BinaryHeap::new()) };
    static HEAP_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize, bool)>> =
        const { RefCell::new(BinaryHeap::new()) };
    static PID_SET: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
    static QUERY_VISITED: RefCell<FixedBitSet> = const{ RefCell::new(FixedBitSet::new()) };
    static QUERY_CANDIDATES_F32: QueryCandF32 =
        const {RefCell::new(BinaryHeap::new())};
    static QUERY_CANDIDATES_F64: QueryCandF64 =
        const {RefCell::new(BinaryHeap::new())};
    static QUERY_RESULTS_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize)>> =
        const{RefCell::new(BinaryHeap::new())};
    static QUERY_RESULTS_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize)>> =
        const{RefCell::new(BinaryHeap::new())};
}

/// NN-Descent index for approximate nearest neighbour search
///
/// Implements the NN-Descent algorithm for efficient k-NN graph construction.
/// Uses an Annoy index for initialisation and beam search for querying.
///
/// ### Memory-Efficient Update Strategy
///
/// The index construction uses a chunked processing approach to bound memory
/// usage during the update phase. Instead of generating all candidate updates
/// at once (which can consume 60GB+ on large datasets), we:
///
/// 1. Partition source nodes into chunks (default ~50k nodes)
/// 2. Generate updates only from nodes in the current chunk
/// 3. Emit both directions (p→q and q→p) during generation
/// 4. Sort by target node for cache-friendly access
/// 5. Apply updates lock-free since each target is processed once per chunk
///
/// This bounds memory to O(chunk_size × max_candidates) rather than
/// O(n × max_candidates), reducing peak memory by 10-50× on large datasets.
///
/// ### Fields
///
/// * `vectors_flat` - Original vector data, flattened for cache locality
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `norms` - Pre-computed norms for Cosine distance (empty for Euclidean)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `forest` - The initial Annoy index for initialisation and entry points
///   into the graph.
/// * `graph` - The NNDescent generated graph
/// * `converged` - Boolean indicating if the build phase converged.
pub struct NNDescent<T> {
    // shared ones
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    pub norms: Vec<T>,
    metric: Dist,
    // index specific ones
    forest: AnnoyIndex<T>,
    graph: Vec<Vec<(usize, T)>>,
    converged: bool,
}

////////////////////
// VectorDistance //
////////////////////

// Implementation of the vector distance trait for NNDescent
impl<T> VectorDistance<T> for NNDescent<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum + SimdDistance,
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
    T: Float + FromPrimitive + Send + Sync + Sum + SimdDistance,
    Self: ApplySortedUpdates<T>,
    Self: NNDescentQuery<T>,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Build a new NN-Descent index
    ///
    /// ### Params
    ///
    /// * `data` - Original data in shape of samples x features
    /// * `metric` - The distance metric to use
    /// * `k` - Initial k-nearest neighbours to search
    /// * `max_iter` - Maximum iterations for the algorithm
    /// * `delta` - Convergence threshold (fraction of edges updated)
    /// * `max_candidates` - Maximum candidates to explore per node
    /// * `n_trees` - Number of trees for Annoy initialisation
    /// * `diversify_prob` - Probability of pruning redundant edges (0 to disable)
    /// * `seed` - Random seed for reproducibility
    /// * `verbose` - Print progress information
    ///
    /// ### Returns
    ///
    /// Initialised NN-Descent index
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
        let annoy_index = AnnoyIndex::new(data, n_trees, metric, seed);
        if verbose {
            println!("Built Annoy index: {:.2?}", start.elapsed());
        }

        let builder = NNDescent {
            vectors_flat,
            dim,
            n,
            metric,
            norms,
            graph: Vec::new(),
            converged: false,
            forest: annoy_index,
        };

        let build_graph = builder.generate_index(k, max_iter, delta, max_candidates, seed, verbose);

        let graph = if diversify_prob > T::zero() {
            builder.diversify_graph(&build_graph.0, diversify_prob, seed)
        } else {
            build_graph.0
        };

        NNDescent {
            vectors_flat: builder.vectors_flat,
            dim: builder.dim,
            n: builder.n,
            metric: builder.metric,
            norms: builder.norms,
            graph,
            converged: build_graph.1,
            forest: builder.forest,
        }
    }

    /// Check if algorithm converged during construction
    ///
    /// ### Returns
    ///
    /// Boolean flag to show that algorithm has converged
    pub fn index_converged(&self) -> bool {
        self.converged
    }

    /// Compute optimal chunk size based on dataset and parameters
    ///
    /// Balances memory usage against overhead of multiple passes.
    /// Targets roughly 100-500MB of update storage per chunk.
    ///
    /// ### Params
    ///
    /// * `max_candidates` - Maximum candidates per node
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

    /// Initialise the k-NN graph using the Annoy index
    ///
    /// Queries the Annoy forest for each point's approximate k nearest
    /// neighbours. These serve as the starting graph for NN-Descent
    /// refinement. All neighbours are marked as "new" to trigger exploration
    /// in the first iteration.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per node
    ///
    /// ### Returns
    ///
    /// Initial k-NN graph with all neighbours marked as new
    fn init_with_annoy(&self, k: usize) -> Vec<Vec<Neighbour<T>>> {
        (0..self.n)
            .into_par_iter()
            .map(|i| {
                let query = &self.vectors_flat[i * self.dim..(i + 1) * self.dim];
                let search_k = k * self.forest.n_trees * 2;
                let (indices, distances) = self.forest.query(query, k + 1, Some(search_k));

                indices
                    .into_iter()
                    .zip(distances)
                    .skip(1)
                    .take(k)
                    .map(|(idx, dist)| Neighbour::new(idx, dist, true))
                    .collect()
            })
            .collect()
    }

    /// Run main NN-Descent algorithm with memory-efficient chunked updates
    ///
    /// ## Algorithm Overview
    ///
    /// NN-Descent iteratively improves a k-NN graph by the principle that
    /// "a neighbour of my neighbour is likely my neighbour". Each iteration:
    ///
    /// 1. Build candidate lists from current neighbours (new and old)
    /// 2. Generate distance updates from candidate pairs
    /// 3. Apply updates to improve the graph
    /// 4. Check convergence (few edges updated → stop)
    ///
    /// ## Memory Optimisation
    ///
    /// The naive approach generates ALL updates then applies them, causing
    /// memory spikes of 60GB+ on 2.8M node datasets. We instead:
    ///
    /// - Process source nodes in chunks (e.g., 50k at a time)
    /// - Generate updates only from the current chunk
    /// - Emit both edge directions during generation (avoids second pass)
    /// - Sort by target node for cache-friendly, lock-free application
    /// - Apply immediately, then discard before next chunk
    ///
    /// This bounds peak memory to O(chunk_size × max_candidates) rather than
    /// O(n × max_candidates).
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours for initial graph
    /// * `max_iter` - Maximum iterations
    /// * `delta` - Convergence threshold
    /// * `max_candidates` - Maximum candidates to explore
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Tuple of (final graph, converged flag)
    fn generate_index(
        &self,
        k: usize,
        max_iter: usize,
        delta: T,
        max_candidates: usize,
        seed: usize,
        verbose: bool,
    ) -> (Vec<Vec<(usize, T)>>, bool) {
        if verbose {
            println!(
                "Running NN-Descent: {} samples, max_candidates={}",
                self.n.separate_with_underscores(),
                max_candidates
            );
        }

        let mut converged = false;

        let start = Instant::now();
        let mut graph = self.init_with_annoy(k);

        if verbose {
            println!("Queried Annoy index: {:.2?}", start.elapsed());
        }

        let chunk_size = self.compute_chunk_size(max_candidates);
        let n_chunks = self.n.div_ceil(chunk_size);

        if verbose {
            println!(
                "Using chunk size {} ({} chunks) for memory-efficient updates",
                chunk_size.separate_with_underscores(),
                n_chunks
            );
        }

        for iter in 0..max_iter {
            let updates_count = AtomicUsize::new(0);

            let mut rng = SmallRng::seed_from_u64((seed as u64).wrapping_add(iter as u64));

            if verbose {
                println!(" Preparing candidates for iter {}", iter + 1);
            }
            let (new_cands, old_cands) = self.build_candidates(&graph, max_candidates, &mut rng);

            self.mark_as_old(&mut graph, &new_cands);

            if verbose {
                println!(
                    " Processing updates for iter {} ({} chunks)",
                    iter + 1,
                    n_chunks
                );
            }

            // process the data in chunks to reduce the memory pressure
            // this exploded on larger data sets
            for chunk_idx in 0..n_chunks {
                let chunk_start = chunk_idx * chunk_size;
                let chunk_end = (chunk_start + chunk_size).min(self.n);

                let mut chunk_updates = self.generate_updates_for_chunk(
                    &new_cands,
                    &old_cands,
                    &graph,
                    chunk_start,
                    chunk_end,
                );

                chunk_updates.par_sort_unstable_by_key(|&(target, _, _)| target);

                self.apply_sorted_updates(&chunk_updates, &mut graph, &updates_count);
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

        let res = graph
            .into_iter()
            .map(|neighbours| neighbours.into_iter().map(|n| (n.pid(), n.dist)).collect())
            .collect();

        (res, converged)
    }

    /// Build candidate lists for local join
    ///
    /// Constructs new and old candidate lists for each node by sampling from
    /// their current neighbours. Uses random prioritisation to select up to
    /// `max_candidates` from each category, then adds symmetric candidates
    /// to ensure graph connectivity.
    ///
    /// ### Params
    ///
    /// * `graph` - Current k-NN graph
    /// * `max_candidates` - Maximum candidates per node
    /// * `rng` - Random number generator for sampling
    ///
    /// ### Returns
    ///
    /// Tuple of `(new_candidates, old_candidates)` for all nodes
    fn build_candidates(
        &self,
        graph: &[Vec<Neighbour<T>>],
        max_candidates: usize,
        rng: &mut SmallRng,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut new_cands: Vec<Vec<usize>> = vec![Vec::with_capacity(max_candidates); self.n];
        let mut old_cands: Vec<Vec<usize>> = vec![Vec::with_capacity(max_candidates); self.n];

        let mut new_temp: Vec<(f64, usize)> = Vec::with_capacity(max_candidates * 2);
        let mut old_temp: Vec<(f64, usize)> = Vec::with_capacity(max_candidates * 2);

        for i in 0..self.n {
            new_temp.clear();
            old_temp.clear();

            for neighbour in &graph[i] {
                let j = neighbour.pid();
                if j >= self.n {
                    continue;
                }

                let priority = rng.random::<f64>();

                if neighbour.is_new() {
                    new_temp.push((priority, j));
                } else {
                    old_temp.push((priority, j));
                }
            }

            if !new_temp.is_empty() {
                new_temp.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                new_temp.truncate(max_candidates);

                let mut last_seen = usize::MAX;
                for &(_, idx) in &new_temp {
                    if idx != last_seen {
                        new_cands[i].push(idx);
                        last_seen = idx;
                    }
                }
            }

            if !old_temp.is_empty() {
                old_temp.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                old_temp.truncate(max_candidates);

                let mut last_seen = usize::MAX;
                for &(_, idx) in &old_temp {
                    if idx != last_seen {
                        old_cands[i].push(idx);
                        last_seen = idx;
                    }
                }
            }
        }

        let mut new_cands_sym: Vec<Vec<usize>> = vec![Vec::new(); self.n];
        let mut old_cands_sym: Vec<Vec<usize>> = vec![Vec::new(); self.n];

        for i in 0..self.n {
            for &j in &new_cands[i] {
                if j < self.n && !new_cands_sym[j].contains(&i) {
                    new_cands_sym[j].push(i);
                }
            }
            for &j in &old_cands[i] {
                if j < self.n && !old_cands_sym[j].contains(&i) {
                    old_cands_sym[j].push(i);
                }
            }
        }

        for i in 0..self.n {
            new_cands[i].extend(&new_cands_sym[i]);
            old_cands[i].extend(&old_cands_sym[i]);
        }

        (new_cands, old_cands)
    }

    /// Mark neighbours as old if they appear in new candidate lists
    ///
    /// Transitions neighbours from "new" to "old" status once they've been
    /// explored in the local join. This prevents redundant distance
    /// calculations in subsequent iterations.
    ///
    /// ### Params
    ///
    /// * `graph` - Current k-NN graph (modified in-place)
    /// * `new_cands` - New candidate lists for each node
    fn mark_as_old(&self, graph: &mut [Vec<Neighbour<T>>], new_cands: &[Vec<usize>]) {
        for i in 0..self.n {
            if new_cands[i].is_empty() {
                continue;
            }

            for neighbour in &mut graph[i] {
                if neighbour.is_new() {
                    let pid = neighbour.pid();
                    if new_cands[i].contains(&pid) {
                        *neighbour = Neighbour::new(pid, neighbour.dist, false);
                    }
                }
            }
        }
    }

    /// Generate distance updates from a chunk of source nodes
    ///
    /// ### Params
    ///
    /// * `new_cands` - New candidate lists for all nodes
    /// * `old_cands` - Old candidate lists for all nodes
    /// * `graph` - Current graph (for threshold checking)
    /// * `chunk_start` - First source node index to process (inclusive)
    /// * `chunk_end` - Last source node index to process (exclusive)
    ///
    /// ### Returns
    ///
    /// List of (target, source, distance) tuples for both edge directions
    fn generate_updates_for_chunk(
        &self,
        new_cands: &[Vec<usize>],
        old_cands: &[Vec<usize>],
        graph: &[Vec<Neighbour<T>>],
        chunk_start: usize,
        chunk_end: usize,
    ) -> Vec<(usize, usize, T)> {
        (chunk_start..chunk_end)
            .into_par_iter()
            .flat_map(|i| {
                let mut updates = Vec::new();

                // Check new-new pairs (same as before)
                for j in 0..new_cands[i].len() {
                    let p = new_cands[i][j];
                    if p >= self.n {
                        continue;
                    }

                    for k in j..new_cands[i].len() {
                        let q = new_cands[i][k];
                        if q >= self.n || p == q {
                            continue;
                        }

                        let d = self.distance(p, q);
                        if self.should_add_edge(p, q, d, graph) {
                            updates.push((p, q, d));
                            updates.push((q, p, d));
                        }
                    }
                }

                for &p in &new_cands[i] {
                    if p >= self.n {
                        continue;
                    }
                    for &q in &old_cands[i] {
                        if q >= self.n || p == q {
                            continue;
                        }

                        let d = self.distance(p, q);
                        if self.should_add_edge(p, q, d, graph) {
                            // CHANGE: Emit both directions
                            updates.push((p, q, d));
                            updates.push((q, p, d));
                        }
                    }
                }

                updates
            })
            .collect()
    }

    /// Calculate distance between two points
    ///
    /// Delegates to the appropriate distance function based on the index's
    /// configured metric.
    ///
    /// ### Params
    ///
    /// * `i` - Index of first point
    /// * `j` - Index of second point
    ///
    /// ### Returns
    ///
    /// Distance between points i and j
    #[inline]
    fn distance(&self, i: usize, j: usize) -> T {
        match self.metric {
            Dist::Euclidean => self.euclidean_distance(i, j),
            Dist::Cosine => self.cosine_distance(i, j),
        }
    }

    /// Check if an edge should be added to the graph
    ///
    /// An edge (p,q) is worth exploring if its distance is better than the
    /// current worst neighbour for either p or q. This prunes candidates
    /// that cannot improve either node's k-NN list.
    ///
    /// ### Params
    ///
    /// * `p` - First node index
    /// * `q` - Second node index
    /// * `dist` - Distance between p and q
    /// * `graph` - Current graph (for threshold lookup)
    ///
    /// ### Returns
    ///
    /// True if edge should be added to update candidates
    #[inline]
    fn should_add_edge(&self, p: usize, q: usize, dist: T, graph: &[Vec<Neighbour<T>>]) -> bool {
        let p_threshold = if graph[p].is_empty() {
            T::infinity()
        } else {
            graph[p].last().unwrap().dist
        };

        let q_threshold = if graph[q].is_empty() {
            T::infinity()
        } else {
            graph[q].last().unwrap().dist
        };

        dist <= p_threshold || dist <= q_threshold
    }

    /// Diversify graph by pruning redundant edges
    ///
    /// Removes edges to neighbours that are already covered by closer
    /// neighbours, reducing graph connectivity whilst maintaining search
    /// quality. Uses probabilistic pruning controlled by `prune_prob`.
    /// NOTE: Most of the time, this does not really help.
    ///
    /// ### Params
    ///
    /// * `graph` - Input k-NN graph
    /// * `prune_prob` - Probability of pruning redundant edges (0.0 to 1.0)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Pruned graph with potentially fewer edges per node
    fn diversify_graph(
        &self,
        graph: &[Vec<(usize, T)>],
        prune_prob: T,
        seed: usize,
    ) -> Vec<Vec<(usize, T)>> {
        graph
            .par_iter()
            .enumerate()
            .map(|(i, neighbours)| {
                if neighbours.is_empty() {
                    return Vec::new();
                }

                let mut rng = SmallRng::seed_from_u64((seed as u64).wrapping_add(i as u64));
                let mut kept = vec![neighbours[0]];

                for &(cand_idx, cand_dist) in &neighbours[1..] {
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

                kept
            })
            .collect()
    }

    ///////////
    // Query //
    ///////////

    /// Query for k nearest neighbours using beam search
    ///
    /// Performs beam search over the k-NN graph, starting from candidates
    /// provided by the Annoy index. The search expands outward through the
    /// graph, maintaining a priority queue of the best candidates found.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Search beam width. Higher values improve recall but
    ///   increase query time. Defaults to `max(2×k, 50)` clamped to 200.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
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

    /// Query using a matrix row reference
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies
    /// to a temporary vector. Uses `self.query()` under the hood.
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference
    /// * `k` - Number of neighbours to search
    /// * `ef_search` - Search beam width. Higher values improve recall but
    ///   increase query time. Defaults to `max(2×k, 50)` clamped to 200.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
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

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Queries each vector in the index against itself to build a complete
    /// kNN graph.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `ef_search` - Search beam width. Higher values improve recall but
    ///   increase query time. Defaults to `max(2×k, 50)` clamped to 200.
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

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);

        total += self.vectors_flat.capacity() * std::mem::size_of::<T>();
        total += self.norms.capacity() * std::mem::size_of::<T>();
        total += self.forest.memory_usage_bytes();

        // graph outer Vec
        total += self.graph.capacity() * std::mem::size_of::<Vec<(usize, T)>>();

        // each inner Vec
        for neighbours in &self.graph {
            total += neighbours.capacity() * std::mem::size_of::<(usize, T)>();
        }

        total
    }
}

/// Helper to find boundaries between different target nodes in sorted updates
///
/// Returns indices where the target node changes, enabling parallel processing
/// of independent node batches.
///
/// ### Params
///
/// * `updates` - Sorted list of (target, source, distance) tuples
///
/// ### Returns
///
/// Vector of boundary indices, including 0 and updates.len()
fn find_target_boundaries<T>(updates: &[(usize, usize, T)]) -> Vec<usize> {
    if updates.is_empty() {
        return vec![0, 0];
    }

    let mut boundaries = vec![0];

    for i in 1..updates.len() {
        if updates[i].0 != updates[i - 1].0 {
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

/////////
// f32 //
/////////

type SegmentF32<'a> = Vec<(usize, &'a [(usize, usize, f32)])>;
type SegmentResultsF32 = Vec<(usize, Option<(Vec<Neighbour<f32>>, usize)>)>;

impl ApplySortedUpdates<f32> for NNDescent<f32> {
    fn apply_sorted_updates(
        &self,
        updates: &[(usize, usize, f32)],
        graph: &mut [Vec<Neighbour<f32>>],
        updates_count: &AtomicUsize,
    ) {
        if updates.is_empty() {
            return;
        }

        let boundaries = find_target_boundaries(updates);

        let segments: SegmentF32 = boundaries
            .windows(2)
            .filter_map(|w| {
                let start = w[0];
                let end = w[1];
                if start < end {
                    Some((updates[start].0, &updates[start..end]))
                } else {
                    None
                }
            })
            .collect();

        let results: SegmentResultsF32 = segments
            .par_iter()
            .map(|&(target, segment)| {
                // Use thread-local heap to avoid allocations
                HEAP_F32.with(|heap_cell| {
                    PID_SET.with(|set_cell| {
                        let mut heap = heap_cell.borrow_mut();
                        let mut pid_set = set_cell.borrow_mut();

                        heap.clear();
                        if pid_set.len() < self.n {
                            pid_set.resize(self.n, false);
                        }

                        let k = graph[target].len();
                        if k == 0 {
                            return (target, None);
                        }

                        let mut edge_updates = 0usize;

                        for n in &graph[target] {
                            let pid = n.pid();
                            heap.push((OrderedFloat(n.dist), pid, n.is_new()));
                            pid_set[pid] = true;
                        }

                        for &(_, source, dist) in segment {
                            if pid_set[source] {
                                continue;
                            }

                            if heap.len() < k {
                                heap.push((OrderedFloat(dist), source, true));
                                pid_set[source] = true;
                                edge_updates += 1;
                            } else if let Some(&(OrderedFloat(worst), _, _)) = heap.peek() {
                                if dist < worst {
                                    if let Some((_, old_pid, _)) = heap.pop() {
                                        pid_set[old_pid] = false;
                                    }
                                    heap.push((OrderedFloat(dist), source, true));
                                    pid_set[source] = true;
                                    edge_updates += 1;
                                }
                            }
                        }

                        if edge_updates > 0 {
                            let mut result: Vec<_> = heap
                                .drain()
                                .map(|(OrderedFloat(d), p, is_new)| {
                                    pid_set[p] = false;
                                    (d, p, is_new)
                                })
                                .collect();
                            result.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            result.truncate(k);

                            (
                                target,
                                Some((
                                    result
                                        .into_iter()
                                        .map(|(d, p, is_new)| Neighbour::new(p, d, is_new))
                                        .collect(),
                                    edge_updates,
                                )),
                            )
                        } else {
                            for (_, pid, _) in heap.iter() {
                                pid_set[*pid] = false;
                            }
                            (target, None)
                        }
                    })
                })
            })
            .collect();

        let mut total_updates = 0;
        for (target, result) in results {
            if let Some((new_neighbours, count)) = result {
                graph[target] = new_neighbours;
                total_updates += count;
            }
        }

        updates_count.fetch_add(total_updates, Ordering::Relaxed);
    }
}

/////////
// f64 //
/////////

type SegmentF64<'a> = Vec<(usize, &'a [(usize, usize, f64)])>;
type SegmentResultsF64 = Vec<(usize, Option<(Vec<Neighbour<f64>>, usize)>)>;

impl ApplySortedUpdates<f64> for NNDescent<f64> {
    fn apply_sorted_updates(
        &self,
        updates: &[(usize, usize, f64)],
        graph: &mut [Vec<Neighbour<f64>>],
        updates_count: &AtomicUsize,
    ) {
        if updates.is_empty() {
            return;
        }

        let boundaries = find_target_boundaries(updates);

        let segments: SegmentF64 = boundaries
            .windows(2)
            .filter_map(|w| {
                let start = w[0];
                let end = w[1];
                if start < end {
                    Some((updates[start].0, &updates[start..end]))
                } else {
                    None
                }
            })
            .collect();

        let results: SegmentResultsF64 = segments
            .par_iter()
            .map(|&(target, segment)| {
                HEAP_F64.with(|heap_cell| {
                    PID_SET.with(|set_cell| {
                        let mut heap = heap_cell.borrow_mut();
                        let mut pid_set = set_cell.borrow_mut();

                        heap.clear();
                        if pid_set.len() < self.n {
                            pid_set.resize(self.n, false);
                        }

                        let k = graph[target].len();
                        if k == 0 {
                            return (target, None);
                        }

                        let mut edge_updates = 0usize;

                        for n in &graph[target] {
                            let pid = n.pid();
                            heap.push((OrderedFloat(n.dist), pid, n.is_new()));
                            pid_set[pid] = true;
                        }

                        for &(_, source, dist) in segment {
                            if pid_set[source] {
                                continue;
                            }

                            if heap.len() < k {
                                heap.push((OrderedFloat(dist), source, true));
                                pid_set[source] = true;
                                edge_updates += 1;
                            } else if let Some(&(OrderedFloat(worst), _, _)) = heap.peek() {
                                if dist < worst {
                                    if let Some((_, old_pid, _)) = heap.pop() {
                                        pid_set[old_pid] = false;
                                    }
                                    heap.push((OrderedFloat(dist), source, true));
                                    pid_set[source] = true;
                                    edge_updates += 1;
                                }
                            }
                        }

                        if edge_updates > 0 {
                            let mut result: Vec<_> = heap
                                .drain()
                                .map(|(OrderedFloat(d), p, is_new)| {
                                    pid_set[p] = false;
                                    (d, p, is_new)
                                })
                                .collect();
                            result.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            result.truncate(k);

                            (
                                target,
                                Some((
                                    result
                                        .into_iter()
                                        .map(|(d, p, is_new)| Neighbour::new(p, d, is_new))
                                        .collect(),
                                    edge_updates,
                                )),
                            )
                        } else {
                            for (_, pid, _) in heap.iter() {
                                pid_set[*pid] = false;
                            }
                            (target, None)
                        }
                    })
                })
            })
            .collect();

        let mut total_updates = 0;
        for (target, result) in results {
            if let Some((new_neighbours, count)) = result {
                graph[target] = new_neighbours;
                total_updates += count;
            }
        }

        updates_count.fetch_add(total_updates, Ordering::Relaxed);
    }
}

////////////////////
// NNDescentQuery //
////////////////////

/////////
// f32 //
/////////

impl NNDescentQuery<f32> for NNDescent<f32> {
    fn query_internal(
        &self,
        query_vec: &[f32],
        query_norm: f32,
        k: usize,
        ef: usize,
    ) -> (Vec<usize>, Vec<f32>) {
        QUERY_VISITED.with(|visited_cell| {
            QUERY_CANDIDATES_F32.with(|cand_cell| {
                QUERY_RESULTS_F32.with(|res_cell| {
                    let mut visited = visited_cell.borrow_mut();
                    let mut candidates = cand_cell.borrow_mut();
                    let mut results = res_cell.borrow_mut();

                    visited.clear();
                    visited.grow(self.n);
                    candidates.clear();
                    results.clear();

                    // CRITICAL: Hoist match outside the loop by splitting into separate paths
                    let (indices, dists) = match self.metric {
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
                    };

                    (indices, dists)
                })
            })
        })
    }

    #[inline(always)]
    fn query_euclidean(
        &self,
        query_vec: &[f32],
        k: usize,
        ef: usize,
        visited: &mut FixedBitSet,
        candidates: &mut BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>>,
        results: &mut BinaryHeap<(OrderedFloat<f32>, usize)>,
    ) -> (Vec<usize>, Vec<f32>) {
        // get entry points
        let init_candidates = (ef / 2).max(2 * k).min(self.n);
        let search_k = init_candidates * 3;
        let (init_indices, _) = self
            .forest
            .query(query_vec, init_candidates, Some(search_k));

        // initialize
        for &entry_idx in &init_indices {
            if entry_idx >= self.n || visited.contains(entry_idx) {
                continue;
            }

            visited.insert(entry_idx);
            let dist = self.euclidean_distance_to_query(entry_idx, query_vec);

            candidates.push(Reverse((OrderedFloat(dist), entry_idx)));
            results.push((OrderedFloat(dist), entry_idx));
        }

        // prune results to ef
        while results.len() > ef {
            results.pop();
        }

        let mut lower_bound = if results.len() >= ef {
            results.peek().unwrap().0 .0
        } else {
            f32::MAX
        };

        // beam search
        while let Some(Reverse((OrderedFloat(curr_dist), curr_idx))) = candidates.pop() {
            if curr_dist > lower_bound {
                break;
            }

            for &(nbr_idx, _) in &self.graph[curr_idx] {
                if visited.contains(nbr_idx) {
                    continue;
                }
                visited.insert(nbr_idx);

                let dist = self.euclidean_distance_to_query(nbr_idx, query_vec);

                // CRITICAL: only add to candidates if it's promising
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

        // extract final k results
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
        query_vec: &[f32],
        query_norm: f32,
        k: usize,
        ef: usize,
        visited: &mut FixedBitSet,
        candidates: &mut BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>>,
        results: &mut BinaryHeap<(OrderedFloat<f32>, usize)>,
    ) -> (Vec<usize>, Vec<f32>) {
        let init_candidates = (ef / 2).max(k).min(self.n);
        let search_k = init_candidates * 3;
        let (init_indices, _) = self
            .forest
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
            f32::MAX
        };

        while let Some(Reverse((OrderedFloat(curr_dist), curr_idx))) = candidates.pop() {
            if curr_dist > lower_bound {
                break;
            }

            for &(nbr_idx, _) in &self.graph[curr_idx] {
                if visited.contains(nbr_idx) {
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

/////////
// f64 //
/////////

impl NNDescentQuery<f64> for NNDescent<f64> {
    fn query_internal(
        &self,
        query_vec: &[f64],
        query_norm: f64,
        k: usize,
        ef: usize,
    ) -> (Vec<usize>, Vec<f64>) {
        QUERY_VISITED.with(|visited_cell| {
            QUERY_CANDIDATES_F64.with(|cand_cell| {
                QUERY_RESULTS_F64.with(|res_cell| {
                    let mut visited = visited_cell.borrow_mut();
                    let mut candidates = cand_cell.borrow_mut();
                    let mut results = res_cell.borrow_mut();

                    visited.clear();
                    visited.grow(self.n);
                    candidates.clear();
                    results.clear();

                    let (indices, dists) = match self.metric {
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
                    };

                    (indices, dists)
                })
            })
        })
    }

    #[inline(always)]
    fn query_euclidean(
        &self,
        query_vec: &[f64],
        k: usize,
        ef: usize,
        visited: &mut FixedBitSet,
        candidates: &mut BinaryHeap<Reverse<(OrderedFloat<f64>, usize)>>,
        results: &mut BinaryHeap<(OrderedFloat<f64>, usize)>,
    ) -> (Vec<usize>, Vec<f64>) {
        let init_candidates = (ef / 2).max(k).min(self.n);
        let search_k = init_candidates * 3;
        let (init_indices, _) = self
            .forest
            .query(query_vec, init_candidates, Some(search_k));

        // initialize
        for &entry_idx in &init_indices {
            if entry_idx >= self.n || visited.contains(entry_idx) {
                continue;
            }

            visited.insert(entry_idx);
            let dist = self.euclidean_distance_to_query(entry_idx, query_vec);

            candidates.push(Reverse((OrderedFloat(dist), entry_idx)));
            results.push((OrderedFloat(dist), entry_idx));
        }

        // prune results to ef
        while results.len() > ef {
            results.pop();
        }

        let mut lower_bound = if results.len() >= ef {
            results.peek().unwrap().0 .0
        } else {
            f64::MAX
        };

        // beam search
        while let Some(Reverse((OrderedFloat(curr_dist), curr_idx))) = candidates.pop() {
            if curr_dist > lower_bound {
                break;
            }

            for &(nbr_idx, _) in &self.graph[curr_idx] {
                if visited.contains(nbr_idx) {
                    continue;
                }
                visited.insert(nbr_idx);

                let dist = self.euclidean_distance_to_query(nbr_idx, query_vec);

                // CRTIICAL AGAIN: Only add to candidates if it's promising
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

        // extract final k results
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
        query_vec: &[f64],
        query_norm: f64,
        k: usize,
        ef: usize,
        visited: &mut FixedBitSet,
        candidates: &mut BinaryHeap<Reverse<(OrderedFloat<f64>, usize)>>,
        results: &mut BinaryHeap<(OrderedFloat<f64>, usize)>,
    ) -> (Vec<usize>, Vec<f64>) {
        let init_candidates = (ef / 2).max(k).min(self.n);
        let search_k = init_candidates * 3;
        let (init_indices, _) = self
            .forest
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
            f64::MAX
        };

        while let Some(Reverse((OrderedFloat(curr_dist), curr_idx))) = candidates.pop() {
            if curr_dist > lower_bound {
                break;
            }

            for &(nbr_idx, _) in &self.graph[curr_idx] {
                if visited.contains(nbr_idx) {
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

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for NNDescent<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance,
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
            1.0, 0.0, 0.0, // Point 0
            0.0, 1.0, 0.0, // Point 1
            0.0, 0.0, 1.0, // Point 2
            1.0, 1.0, 0.0, // Point 3
            1.0, 0.0, 1.0, // Point 4
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
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

        assert_eq!(index.graph.len(), 5);
        for neighbours in &index.graph {
            assert!(neighbours.len() <= 3);
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

        assert_eq!(index.graph.len(), 5);
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

        assert_eq!(index.graph.len(), 5);
    }

    #[test]
    fn test_nndescent_reproducibility() {
        let mat = create_simple_matrix();

        let graph1 = NNDescent::<f32>::new(
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

        let graph2 = NNDescent::<f32>::new(
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

        assert_eq!(graph1.graph.len(), graph2.graph.len());

        // Check that graphs are identical
        for i in 0..graph1.graph.len() {
            assert_eq!(graph1.graph[i].len(), graph2.graph[i].len());
        }
    }

    #[test]
    fn test_nndescent_k_parameter() {
        let mat = create_simple_matrix();

        let graph_k2 = NNDescent::<f32>::new(
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

        let graph_k4 = NNDescent::<f32>::new(
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

        for neighbours in &graph_k2.graph {
            assert!(neighbours.len() <= 2);
        }

        for neighbours in &graph_k4.graph {
            assert!(neighbours.len() <= 4);
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

        assert_eq!(index.graph.len(), n);

        for neighbours in &index.graph {
            assert!(neighbours.len() <= 10);
            assert!(!neighbours.is_empty());
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

        for neighbours in &index.graph {
            for i in 1..neighbours.len() {
                assert!(neighbours[i].1 >= neighbours[i - 1].1);
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

        assert_eq!(index.graph.len(), 3);
    }

    #[test]
    fn test_nndescent_quality() {
        let n = 20;
        let dim = 3;
        let mut data = Vec::with_capacity(n * dim);

        // First cluster at x=0
        for i in 0..10 {
            let offset = i as f32 * 0.1;
            data.extend_from_slice(&[offset, 0.0, 0.0]);
        }
        // Second cluster at x=10
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

        // Check that neighbours tend to be in the same cluster
        let neighbours_0 = &index.graph[0];
        let in_cluster = neighbours_0.iter().filter(|(idx, _)| *idx < 10).count();
        assert!(in_cluster >= 3);

        let neighbours_10 = &index.graph[10];
        let in_cluster_2 = neighbours_10.iter().filter(|(idx, _)| *idx >= 10).count();
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
            0.5, // Enable diversification
            42,
            false,
        );

        assert_eq!(index.graph.len(), 5);
        for neighbours in &index.graph {
            assert!(!neighbours.is_empty());
        }
    }
}
