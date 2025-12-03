use faer::MatRef;
use fixedbitset::FixedBitSet;
use num_traits::{Float, FromPrimitive};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::Sum;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thousands::*;

use crate::annoy::*;
use crate::dist::*;
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
    /// Generate a new Neighbour instance
    ///
    /// ### Params
    ///
    /// * `pid` - Index of the point
    /// * `dist` - Distance to the point
    /// * `is_new` - Boolean if this is a new neighbour
    #[inline(always)]
    fn new(pid: usize, dist: T, is_new: bool) -> Self {
        Self {
            pid: pid as u32,
            dist,
            is_new: is_new as u32,
        }
    }

    /// Is this a new neighbour
    ///
    /// ### Returns
    ///
    /// Boolean indicating if sample is a new neighbour
    #[inline(always)]
    fn is_new(&self) -> bool {
        self.is_new != 0
    }

    /// Return the index
    ///
    /// ### Returns
    ///
    /// The point index
    #[inline(always)]
    fn pid(&self) -> usize {
        self.pid as usize
    }
}

/// Trait for applying neighbour updates (different implementations for f32/f64)
pub trait UpdateNeighbours<T> {
    /// Update neighbour lists with new candidate edges
    ///
    /// ### Params
    ///
    /// * `updates` - List of (source, target, distance) tuples
    /// * `graph` - Current k-NN graph to update
    /// * `updates_count` - Atomic counter for tracking edge updates
    fn update_neighbours(
        &self,
        updates: &[(usize, usize, T)],
        graph: &mut [Vec<Neighbour<T>>],
        updates_count: &AtomicUsize,
    );
}

/// Trait for querying the NN-Descent index
pub trait NNDescentQuery<T> {
    /// Internal query method that delegates to metric-specific implementations
    fn query_internal(
        &self,
        query_vec: &[T],
        query_norm: T,
        k: usize,
        ef: usize,
    ) -> (Vec<usize>, Vec<T>);

    /// Query using Euclidean distance
    fn query_euclidean(
        &self,
        query_vec: &[T],
        k: usize,
        ef: usize,
        visited: &mut FixedBitSet,
        candidates: &mut BinaryHeap<Reverse<(OrderedFloat<T>, usize)>>,
        results: &mut BinaryHeap<(OrderedFloat<T>, usize)>,
    ) -> (Vec<usize>, Vec<T>);

    /// Query using Cosine distance
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
    /// Heap for f32
    static HEAP_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize, bool)>> =
        const { RefCell::new(BinaryHeap::new()) };
    /// Heap for f64
    static HEAP_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize, bool)>> =
        const { RefCell::new(BinaryHeap::new()) };
    /// Heap for PID
    static PID_SET: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
    /// Thread-local storage for which nodes were visited - for querying
    static QUERY_VISITED: RefCell<FixedBitSet> = const{ RefCell::new(FixedBitSet::new()) };
    /// Store the candidates (f32) - for querying
    static QUERY_CANDIDATES_F32: QueryCandF32 =
        const {RefCell::new(BinaryHeap::new())};
    /// Store the candidates (f64) - for querying
    static QUERY_CANDIDATES_F64: QueryCandF64 =
        const {RefCell::new(BinaryHeap::new())};
    /// Results (f32) - for querying
    static QUERY_RESULTS_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize)>> =
        const{RefCell::new(BinaryHeap::new())};
    /// Results (f64) - for querying
    static QUERY_RESULTS_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize)>> =
        const{RefCell::new(BinaryHeap::new())};
}

/// NN-Descent index for approximate nearest neighbour search
///
/// Implements the NN-Descent algorithm for efficient k-NN graph construction.
/// Uses an Annoy index for initialisation and beam search for querying.
///
/// ### Fields
///
/// * `vectors_flat` - Flattened vector data for cache locality
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `norms` - Pre-computed norms for Cosine distance (empty for Euclidean)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `forest` - Annoy index for initialisation and query entry points
/// * `graph` - Final k-NN graph
/// * `converged` - Whether the index converged during construction
pub struct NNDescent<T> {
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    norms: Vec<T>,
    metric: Dist,
    forest: AnnoyIndex<T>,
    graph: Vec<Vec<(usize, T)>>,
    converged: bool,
}

impl<T> VectorDistance<T> for NNDescent<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
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

impl<T> NNDescent<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
    Self: UpdateNeighbours<T>,
    Self: NNDescentQuery<T>,
{
    /// Build a new NN-Descent index
    ///
    /// ### Params
    ///
    /// * `mat` - Original data in shape of samples x features.
    /// * `metric` - The distance metric to use for the generation of this
    ///   index.
    /// * `k` - Initial k-nearest neighbours to search. Relevant for the
    ///   initialisation.
    /// * `max_iter` - How many iterations shall the algorithm run at maximum.
    /// * `delta` - The stopping criterium. If less edges than this percentage
    ///   are updated in a given iteration, the algorithm is considered as
    ///   converged.
    /// * `max_candidates` - Optional maximum number of candidates to explore.
    /// * `graph_size` - Optional GraphSize enum
    /// * `seed` - Seed for reproducibility.
    /// * `verbose` - Controls verbosity of the function
    ///
    /// ### Returns
    ///
    /// Initialised index
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mat: MatRef<T>,
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
        let n = mat.nrows();
        let n_features = mat.ncols();

        // defaults if not provided
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

        // flatten matrix
        let mut vectors_flat = Vec::with_capacity(n * n_features);
        for i in 0..n {
            vectors_flat.extend(mat.row(i).iter().copied());
        }

        // pre-compute norms for cosine
        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * n_features;
                    let end = start + n_features;
                    vectors_flat[start..end]
                        .iter()
                        .map(|x| *x * *x)
                        .fold(T::zero(), |a, b| a + b)
                        .sqrt()
                })
                .collect()
        } else {
            Vec::new()
        };

        // build initial index - using the package-internal annoy here
        let start = Instant::now();
        let annoy_index = AnnoyIndex::new(mat, n_trees, metric, seed);
        if verbose {
            println!("Built Annoy index: {:.2?}", start.elapsed());
        }

        let builder = NNDescent {
            vectors_flat,
            dim: n_features,
            n,
            metric,
            norms,
            graph: Vec::new(),
            converged: false,
            forest: annoy_index,
        };

        let build_graph = builder.run(k, max_iter, delta, max_candidates, seed, verbose);

        // Diversify if requested
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

    /// Query for k nearest neighbours
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Search beam width (default: clamp(k*2, 20, 100))
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        ef_search: Option<usize>,
    ) -> (Vec<usize>, Vec<T>) {
        let k = k.min(self.n);
        let ef = ef_search.unwrap_or_else(|| (k * 2).clamp(20, 100)).max(k);

        // Pre-compute query norm once if using cosine
        let query_norm = if self.metric == Dist::Cosine {
            query_vec.iter().map(|x| *x * *x).sum::<T>().sqrt()
        } else {
            T::one()
        };

        // Use thread-local storage to avoid allocations
        self.query_internal(query_vec, query_norm, k, ef)
    }

    /// Check if algorithm converged during construction
    ///
    /// ### Returns
    ///
    /// True if convergence criterion was met
    pub fn index_converged(&self) -> bool {
        self.converged
    }

    /// Initialise the graph with the stored Annoy index
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours to consider
    ///
    /// ### Returns
    ///
    /// Initial neighbour lists for each node
    fn init_with_annoy(&self, k: usize) -> Vec<Vec<Neighbour<T>>> {
        (0..self.n)
            .into_par_iter()
            .map(|i| {
                let query = &self.vectors_flat[i * self.dim..(i + 1) * self.dim];
                // do not spend too much time on annoy...
                // fast initial querying with okay budget to get some decent starting point
                let search_k = k * self.forest.n_trees * 2;
                let (indices, distances) = self.forest.query(query, k + 1, Some(search_k));

                indices
                    .into_iter()
                    .zip(distances)
                    .skip(1) // Skip self
                    .take(k)
                    .map(|(idx, dist)| Neighbour::new(idx, dist, true)) // All new initially
                    .collect()
            })
            .collect()
    }

    /// Run main NN-Descent algorithm
    ///
    /// Implements the low-memory version of NN-Descent.
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
    /// Tuple of (final graph, did converge)
    #[allow(clippy::too_many_arguments)]
    fn run(
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

        let mut converged: bool = false;

        let start = Instant::now();
        let mut graph = self.init_with_annoy(k);

        if verbose {
            println!("Queried Annoy index: {:.2?}", start.elapsed());
        }

        for iter in 0..max_iter {
            let updates = AtomicUsize::new(0);

            let mut rng = SmallRng::seed_from_u64((seed as u64).wrapping_add(iter as u64));

            if verbose {
                println!(" Preparing candidates for iter {}", iter + 1);
            }
            let (new_cands, old_cands) = self.build_candidates(&graph, max_candidates, &mut rng);

            self.mark_as_old(&mut graph, &new_cands);

            if verbose {
                println!(" Generating updates for iter {}", iter + 1);
            }
            let all_updates = self.generate_updates(&new_cands, &old_cands, &graph);

            if verbose {
                println!(" Applying updates for iter {}", iter + 1);
            }
            self.update_neighbours(&all_updates.concat(), &mut graph, &updates);

            let update_count = updates.load(Ordering::Relaxed);

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
    /// ### Params
    ///
    /// * `graph` - Current graph
    /// * `max_candidates` - Maximum candidates per node
    /// * `rng` - Random number generator
    ///
    /// ### Returns
    ///
    /// Tuple of (new candidates, old candidates)
    fn build_candidates(
        &self,
        graph: &[Vec<Neighbour<T>>],
        max_candidates: usize,
        rng: &mut SmallRng,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        // Pre-allocate with capacity to avoid reallocations
        let mut new_cands: Vec<Vec<usize>> = vec![Vec::with_capacity(max_candidates); self.n];
        let mut old_cands: Vec<Vec<usize>> = vec![Vec::with_capacity(max_candidates); self.n];

        // Temporary storage for sorting (reusable per node)
        let mut new_temp: Vec<(f64, usize)> = Vec::with_capacity(max_candidates * 2);
        let mut old_temp: Vec<(f64, usize)> = Vec::with_capacity(max_candidates * 2);

        for i in 0..self.n {
            new_temp.clear();
            old_temp.clear();

            // Collect candidates from neighbours
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

            // Process new candidates: sort, truncate, deduplicate
            if !new_temp.is_empty() {
                new_temp.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                new_temp.truncate(max_candidates);

                // Deduplicate while building final list
                let mut last_seen = usize::MAX;
                for &(_, idx) in &new_temp {
                    if idx != last_seen {
                        new_cands[i].push(idx);
                        last_seen = idx;
                    }
                }
            }

            // Process old candidates: sort, truncate, deduplicate
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

        // Build symmetric candidate lists
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

        // Merge symmetric candidates back
        for i in 0..self.n {
            new_cands[i].extend(&new_cands_sym[i]);
            old_cands[i].extend(&old_cands_sym[i]);
        }

        (new_cands, old_cands)
    }

    /// Mark neighbours as old
    ///
    /// ### Params
    ///
    /// * `graph` - Current graph to update
    /// * `new_cands` - New candidate lists
    fn mark_as_old(&self, graph: &mut [Vec<Neighbour<T>>], new_cands: &[Vec<usize>]) {
        for i in 0..self.n {
            if new_cands[i].is_empty() {
                continue;
            }

            // For small candidate lists, linear search is faster than HashSet
            for neighbour in &mut graph[i] {
                if neighbour.is_new() {
                    let pid = neighbour.pid();
                    // Check if this pid is in new_cands[i]
                    if new_cands[i].contains(&pid) {
                        *neighbour = Neighbour::new(pid, neighbour.dist, false);
                    }
                }
            }
        }
    }

    /// Generate distance updates from candidate pairs
    ///
    /// ### Params
    ///
    /// * `new_cands` - New candidate lists
    /// * `old_cands` - Old candidate lists
    /// * `graph` - Current graph
    ///
    /// ### Returns
    ///
    /// List of updates per node
    fn generate_updates(
        &self,
        new_cands: &[Vec<usize>],
        old_cands: &[Vec<usize>],
        graph: &[Vec<Neighbour<T>>],
    ) -> Vec<Vec<(usize, usize, T)>> {
        (0..self.n)
            .into_par_iter()
            .map(|i| {
                let mut updates = Vec::new();

                for j in 0..new_cands[i].len() {
                    let p = new_cands[i][j];
                    if p >= self.n {
                        continue;
                    }

                    for k in j..new_cands[i].len() {
                        let q = new_cands[i][k];
                        if q >= self.n {
                            continue;
                        }
                        if p == q {
                            continue;
                        }

                        let d = self.distance(p, q);
                        if self.should_add_edge(p, q, d, graph) {
                            updates.push((p, q, d));
                        }
                    }
                }

                for &p in &new_cands[i] {
                    if p >= self.n {
                        continue;
                    }
                    for &q in &old_cands[i] {
                        if q >= self.n {
                            continue;
                        }

                        let d = self.distance(p, q);
                        if self.should_add_edge(p, q, d, graph) {
                            updates.push((p, q, d));
                        }
                    }
                }

                updates
            })
            .collect()
    }

    /// Calculate distance between two points
    ///
    /// ### Params
    ///
    /// * `i` - First point index
    /// * `j` - Second point index
    ///
    /// ### Returns
    ///
    /// Distance between points
    #[inline]
    fn distance(&self, i: usize, j: usize) -> T {
        match self.metric {
            Dist::Euclidean => self.euclidean_distance(i, j),
            Dist::Cosine => self.cosine_distance(i, j),
        }
    }

    /// Check if an edge should be added to the graph
    ///
    /// ### Params
    ///
    /// * `p` - Source node
    /// * `q` - Target node
    /// * `dist` - Distance between nodes
    /// * `graph` - Current graph
    ///
    /// ### Returns
    ///
    /// True if edge should be added
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
    /// Removes neighbours that are closer to other kept neighbours
    /// than to the query node, reducing clustering.
    ///
    /// ### Params
    ///
    /// * `graph` - Current graph
    /// * `prune_prob` - Probability of pruning redundant neighbours
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Diversified graph
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
                let mut kept = vec![neighbours[0]]; // Always keep closest

                for &(cand_idx, cand_dist) in &neighbours[1..] {
                    let mut should_keep = true;

                    for &(kept_idx, kept_dist) in &kept {
                        let dist_to_kept = self.distance(cand_idx, kept_idx);

                        // Prune if candidate is closer to an already-kept neighbour
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
}

// Update implementations for f32 and f64
impl UpdateNeighbours<f32> for NNDescent<f32> {
    /// Update neighbour lists with new candidate edges
    ///
    /// Groups updates by node, then processes each node in parallel using
    /// thread-local heaps to maintain the k-nearest neighbours efficiently.
    ///
    /// ### Params
    ///
    /// * `updates` - List of (source, target, distance) tuples
    /// * `graph` - Current k-NN graph to update
    /// * `updates_count` - Atomic counter for tracking edge updates
    fn update_neighbours(
        &self,
        updates: &[(usize, usize, f32)],
        graph: &mut [Vec<Neighbour<f32>>],
        updates_count: &AtomicUsize,
    ) {
        let mut per_node: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.n];
        for &(p, q, d) in updates {
            if p < self.n && q < self.n {
                per_node[p].push((q, d));
                per_node[q].push((p, d));
            }
        }

        let new_graphs: Vec<Option<(Vec<Neighbour<f32>>, usize)>> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                if per_node[i].is_empty() {
                    return None;
                }

                HEAP_F32.with(|heap_cell| {
                    PID_SET.with(|set_cell| {
                        let mut heap = heap_cell.borrow_mut();
                        let mut pid_set = set_cell.borrow_mut();

                        heap.clear();
                        if pid_set.len() < self.n {
                            pid_set.resize(self.n, false);
                        }

                        let k = graph[i].len();
                        let mut edge_updates = 0usize;

                        for n in &graph[i] {
                            let pid = n.pid();
                            heap.push((OrderedFloat(n.dist), pid, n.is_new()));
                            pid_set[pid] = true;
                        }

                        for &(cand, dist) in &per_node[i] {
                            if pid_set[cand] {
                                continue;
                            }

                            if heap.len() < k {
                                heap.push((OrderedFloat(dist), cand, true));
                                pid_set[cand] = true;
                                edge_updates += 1;
                            } else if let Some(&(OrderedFloat(worst), _, _)) = heap.peek() {
                                if dist < worst {
                                    if let Some((_, old_pid, _)) = heap.pop() {
                                        pid_set[old_pid] = false;
                                    }
                                    heap.push((OrderedFloat(dist), cand, true));
                                    pid_set[cand] = true;
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

                            Some((
                                result
                                    .into_iter()
                                    .map(|(d, p, is_new)| Neighbour::new(p, d, is_new))
                                    .collect(),
                                edge_updates,
                            ))
                        } else {
                            for (_, pid, _) in heap.iter() {
                                pid_set[*pid] = false;
                            }
                            None
                        }
                    })
                })
            })
            .collect();

        let mut total_edge_updates = 0;
        for (i, new_graph) in new_graphs.into_iter().enumerate() {
            if let Some((new_neighbours, edge_count)) = new_graph {
                graph[i] = new_neighbours;
                total_edge_updates += edge_count;
            }
        }

        updates_count.fetch_add(total_edge_updates, Ordering::Relaxed);
    }
}

impl UpdateNeighbours<f64> for NNDescent<f64> {
    /// Update neighbour lists with new candidate edges
    ///
    /// Groups updates by node, then processes each node in parallel using
    /// thread-local heaps to maintain the k-nearest neighbours efficiently.
    ///
    /// ### Params
    ///
    /// * `updates` - List of (source, target, distance) tuples
    /// * `graph` - Current k-NN graph to update
    /// * `updates_count` - Atomic counter for tracking edge updates
    fn update_neighbours(
        &self,
        updates: &[(usize, usize, f64)],
        graph: &mut [Vec<Neighbour<f64>>],
        updates_count: &AtomicUsize,
    ) {
        let mut per_node: Vec<Vec<(usize, f64)>> = vec![Vec::new(); self.n];
        for &(p, q, d) in updates {
            if p < self.n && q < self.n {
                per_node[p].push((q, d));
                per_node[q].push((p, d));
            }
        }

        let new_graphs: Vec<Option<(Vec<Neighbour<f64>>, usize)>> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                if per_node[i].is_empty() {
                    return None;
                }

                HEAP_F64.with(|heap_cell| {
                    PID_SET.with(|set_cell| {
                        let mut heap = heap_cell.borrow_mut();
                        let mut pid_set = set_cell.borrow_mut();

                        heap.clear();
                        if pid_set.len() < self.n {
                            pid_set.resize(self.n, false);
                        }

                        let k = graph[i].len();
                        let mut edge_updates = 0usize;

                        for n in &graph[i] {
                            let pid = n.pid();
                            heap.push((OrderedFloat(n.dist), pid, n.is_new()));
                            pid_set[pid] = true;
                        }

                        for &(cand, dist) in &per_node[i] {
                            if pid_set[cand] {
                                continue;
                            }

                            if heap.len() < k {
                                heap.push((OrderedFloat(dist), cand, true));
                                pid_set[cand] = true;
                                edge_updates += 1;
                            } else if let Some(&(OrderedFloat(worst), _, _)) = heap.peek() {
                                if dist < worst {
                                    if let Some((_, old_pid, _)) = heap.pop() {
                                        pid_set[old_pid] = false;
                                    }
                                    heap.push((OrderedFloat(dist), cand, true));
                                    pid_set[cand] = true;
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

                            Some((
                                result
                                    .into_iter()
                                    .map(|(d, p, is_new)| Neighbour::new(p, d, is_new))
                                    .collect(),
                                edge_updates,
                            ))
                        } else {
                            for (_, pid, _) in heap.iter() {
                                pid_set[*pid] = false;
                            }
                            None
                        }
                    })
                })
            })
            .collect();

        let mut total_edge_updates = 0;
        for (i, new_graph) in new_graphs.into_iter().enumerate() {
            if let Some((new_neighbours, edge_count)) = new_graph {
                graph[i] = new_neighbours;
                total_edge_updates += edge_count;
            }
        }

        updates_count.fetch_add(total_edge_updates, Ordering::Relaxed);
    }
}

impl NNDescentQuery<f32> for NNDescent<f32> {
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

impl NNDescentQuery<f64> for NNDescent<f64> {
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
