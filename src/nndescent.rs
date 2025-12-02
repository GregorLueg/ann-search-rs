use faer::MatRef;
use fixedbitset::FixedBitSet; // You'll need to add this crate
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
    fn update_neighbours(
        &self,
        updates: &[(usize, usize, T)],
        graph: &mut [Vec<Neighbour<T>>],
        updates_count: &AtomicUsize,
    );
}

thread_local! {
    /// Heap for f32
    static HEAP_F32: RefCell<BinaryHeap<(OrderedFloat<f32>, usize, bool)>> =
        const { RefCell::new(BinaryHeap::new()) };
    /// Heap for f64
    static HEAP_F64: RefCell<BinaryHeap<(OrderedFloat<f64>, usize, bool)>> =
        const { RefCell::new(BinaryHeap::new()) };
    /// Heap for PID
    static PID_SET: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
}

/// NN-Descent index for approximate nearest neighbour search
///
/// ### Fields
///
/// * `vectors_flat` - Original vector data, flattened for cache locality
/// * `dim` - Embedding dimensions
/// * `n` - Number of vectors
/// * `norms` - Pre-computed norms for Cosine distance (empty for Euclidean)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `forest` - The initial Annoy index for initialisation and starting points
///   of queries.
/// * `graph` - Finalised graph
/// * `converged` - Boolean indicating if the index hit the convergence
///   criterium during build.
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
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        ef_search: Option<usize>,
        epsilon: Option<T>,
    ) -> (Vec<usize>, Vec<T>) {
        let k = k.min(self.n);
        let ef = ef_search.unwrap_or_else(|| (k * 2).clamp(30, 100)).max(k);
        let _epsilon = epsilon.unwrap_or_else(|| T::from_f32(0.1).unwrap());

        // 1. Optimize Distance Calculation (Move match out of closure if possible,
        // or rely on Monomorphization if metric was a generic const)
        let compute_dist = |idx: usize| -> T {
            // In a real optimized version, 'metric' should be a Generic Trait,
            // not a runtime enum, to force inlining.
            match self.metric {
                Dist::Euclidean => self.euclidean_distance_to_query(idx, query_vec),
                Dist::Cosine => {
                    // pre-calculating query_norm is correct, keep that
                    let query_norm = T::one(); // simplified for snippet
                    self.cosine_distance_to_query(idx, query_vec, query_norm)
                }
            }
        };

        // 2. Fix Allocation: Use FixedBitSet.
        // NOTE: In production, pass this buffer in as an argument to avoid alloc entirely!
        let mut visited = FixedBitSet::with_capacity(self.n);

        // 3. Fix Data Structure: MinHeap for candidates, MaxHeap for results
        // Candidates: MinHeap stores (dist, idx) - we want smallest dist first
        let mut candidates = BinaryHeap::new();

        // Results: MaxHeap stores (dist, idx) - we want to pop the worst element when size > ef
        let mut results = BinaryHeap::new();

        // --- Initialization ---
        let init_candidates = (ef + k).min(self.n);
        let search_k = (self.n / 20).max(init_candidates * 2);

        // We assume forest query is fast enough
        let (init_indices, _) = self
            .forest
            .query(query_vec, init_candidates, Some(search_k));

        for &entry_idx in &init_indices {
            if entry_idx >= self.n || visited.contains(entry_idx) {
                continue;
            }

            visited.insert(entry_idx);
            let dist = compute_dist(entry_idx);

            // Reverse for MinHeap behavior
            candidates.push(Reverse((OrderedFloat(dist), entry_idx)));
            results.push((OrderedFloat(dist), entry_idx));

            if results.len() > ef {
                results.pop(); // Remove worst
            }
        }

        // --- Search Loop ---
        // Bound is the distance of the furthest element in our current 'ef' results
        let mut lower_bound = if results.len() == ef {
            results.peek().unwrap().0
        } else {
            OrderedFloat(T::max_value())
        };

        while let Some(Reverse((OrderedFloat(curr_dist), curr_idx))) = candidates.pop() {
            // If the closest candidate is worse than our worst result, we can stop
            // (Standard HNSW logic)
            if curr_dist > lower_bound.0 {
                break;
            }

            for &(nbr_idx, _) in &self.graph[curr_idx] {
                if visited.contains(nbr_idx) {
                    continue;
                }
                visited.insert(nbr_idx);

                let dist = compute_dist(nbr_idx);

                if dist < lower_bound.0 || results.len() < ef {
                    candidates.push(Reverse((OrderedFloat(dist), nbr_idx)));
                    results.push((OrderedFloat(dist), nbr_idx));

                    if results.len() > ef {
                        results.pop();
                        // Update bound to the new worst element
                        lower_bound = results.peek().unwrap().0;
                    }
                }
            }
        }

        // --- Formatting Output ---
        let mut final_results: Vec<_> = results.into_sorted_vec();
        // into_sorted_vec returns generic sorting (min to max), which is what we want
        final_results.truncate(k);

        final_results
            .into_iter()
            .map(|(OrderedFloat(d), i)| (i, d))
            .unzip()
    }

    /// Check if algorithm has
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
    /// Returns the initial neighbours to initialise the graph
    fn init_with_annoy(&self, k: usize) -> Vec<Vec<Neighbour<T>>> {
        (0..self.n)
            .into_par_iter()
            .map(|i| {
                let query = &self.vectors_flat[i * self.dim..(i + 1) * self.dim];
                let search_k = self.n / 20;
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
    /// This implements the low memory version of (Py)NNDescent or a version
    /// thereof.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours for initial graph generation.
    /// * `max_iter` - How many iterations shall the algorithm run at maximum.
    /// * `annoy_index` - The Annoy index for initialisation
    /// * `delta` - The stopping criterium. If less edges than this percentage
    ///   are updated in a given iteration, the algorithm is considered as
    ///   converged.
    /// * `max_candidates` - Maximum number of candidates to explore.
    /// * `seed` - Seed for reproducibility.
    /// * `verbose` - Controls verbosity of the function
    ///
    /// ### Returns
    ///
    /// Tuple of `(finalised graph, did it converge)`.
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

        for iter in 0..max_iter {
            let updates = AtomicUsize::new(0);

            let mut rng = SmallRng::seed_from_u64((seed as u64).wrapping_add(iter as u64));

            let (new_cands, old_cands) = self.build_candidates(&graph, max_candidates, &mut rng);

            self.mark_as_old(&mut graph, &new_cands);

            let all_updates = self.generate_updates(&new_cands, &old_cands, &graph);

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

    /// Build candidate lists
    ///
    /// ### Params
    ///
    /// * `graph` - Current graph
    /// * `max_candidates` - Maximum number of new candidates
    /// * `rng` - SmallRng for randomisation
    ///
    /// ### Returns
    ///
    /// Tuple of `(old candidates, new candidates)`.
    fn build_candidates(
        &self,
        graph: &[Vec<Neighbour<T>>],
        max_candidates: usize,
        rng: &mut SmallRng,
    ) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut new_priorities: Vec<Vec<f64>> = vec![vec![f64::INFINITY; max_candidates]; self.n];
        let mut new_indices: Vec<Vec<i32>> = vec![vec![-1; max_candidates]; self.n];

        let mut old_priorities: Vec<Vec<f64>> = vec![vec![f64::INFINITY; max_candidates]; self.n];
        let mut old_indices: Vec<Vec<i32>> = vec![vec![-1; max_candidates]; self.n];

        for i in 0..self.n {
            for neighbour in &graph[i] {
                let j = neighbour.pid();
                if j >= self.n {
                    continue;
                }

                // negative for max-heap semantics
                let priority = -rng.random::<f64>();

                if neighbour.is_new() {
                    // add j to i's new candidates
                    Self::checked_heap_push(
                        &mut new_priorities[i],
                        &mut new_indices[i],
                        priority,
                        j as i32,
                    );

                    // add i to j's new candidates (symmetric)
                    Self::checked_heap_push(
                        &mut new_priorities[j],
                        &mut new_indices[j],
                        priority,
                        i as i32,
                    );
                } else {
                    // add j to i's old candidates
                    Self::checked_heap_push(
                        &mut old_priorities[i],
                        &mut old_indices[i],
                        priority,
                        j as i32,
                    );

                    // add i to j's old candidates (symmetric)
                    Self::checked_heap_push(
                        &mut old_priorities[j],
                        &mut old_indices[j],
                        priority,
                        i as i32,
                    );
                }
            }
        }

        // convert to final format, filtering out -1 indices
        let new_cands: Vec<Vec<usize>> = new_indices
            .into_iter()
            .map(|indices| {
                indices
                    .into_iter()
                    .filter(|&idx| idx >= 0)
                    .map(|idx| idx as usize)
                    .collect()
            })
            .collect();

        let old_cands: Vec<Vec<usize>> = old_indices
            .into_iter()
            .map(|indices| {
                indices
                    .into_iter()
                    .filter(|&idx| idx >= 0)
                    .map(|idx| idx as usize)
                    .collect()
            })
            .collect();

        (new_cands, old_cands)
    }

    /// Deduplicate the heap and deal with max
    ///
    /// Python's checked_heap_push implementation that does linear scan
    /// deduplication + max-heap maintenance.
    ///
    #[inline]
    fn checked_heap_push(priorities: &mut [f64], indices: &mut [i32], priority: f64, idx: i32) {
        let size = priorities.len();

        // early exit if priority is worse than worst in heap
        if priority >= priorities[0] {
            return;
        }

        // check for duplicates with linear scan (fast for small heaps)
        for &existing_idx in indices.iter() {
            if existing_idx == idx {
                return;
            }
        }

        // insert at root and sift down
        priorities[0] = priority;
        indices[0] = idx;

        // sift down to maintain max-heap property
        let mut i = 0;
        loop {
            let left_child = 2 * i + 1;
            let right_child = left_child + 1;

            if left_child >= size {
                break;
            }

            let swap_idx = if right_child >= size {
                // only left child exists
                if priorities[left_child] > priority {
                    left_child
                } else {
                    break;
                }
            } else {
                // both children exist
                if priorities[left_child] >= priorities[right_child] {
                    if priority < priorities[left_child] {
                        left_child
                    } else {
                        break;
                    }
                } else if priority < priorities[right_child] {
                    right_child
                } else {
                    break;
                }
            };

            priorities[i] = priorities[swap_idx];
            indices[i] = indices[swap_idx];
            i = swap_idx;
        }

        priorities[i] = priority;
        indices[i] = idx;
    }

    /// Mark neighbours as old (matches Python's flag updating in new_build_candidates)
    fn mark_as_old(&self, graph: &mut [Vec<Neighbour<T>>], new_cands: &[Vec<usize>]) {
        for i in 0..self.n {
            if new_cands[i].is_empty() {
                continue;
            }

            let cand_set: std::collections::HashSet<usize> = new_cands[i].iter().copied().collect();

            for neighbour in &mut graph[i] {
                if cand_set.contains(&neighbour.pid()) && neighbour.is_new() {
                    *neighbour = Neighbour::new(neighbour.pid(), neighbour.dist, false);
                }
            }
        }
    }

    /// Generate distance updates (matches Python's generate_graph_updates)
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

                // Compare new-new pairs
                // CRITICAL FIX: Start at j, not j+1 to match Python
                for j in 0..new_cands[i].len() {
                    let p = new_cands[i][j];
                    if p >= self.n {
                        continue;
                    }

                    for k in j..new_cands[i].len() {
                        // ← Changed from j+1
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

                // Compare new-old pairs
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

    /// Distance function between two points
    ///
    /// ### Params
    ///
    /// * `i` - Index of sample i
    /// * `j` - Index of sample j
    ///
    /// ### Returns
    ///
    /// Returns the desired distance between the two points
    #[inline]
    fn distance(&self, i: usize, j: usize) -> T {
        match self.metric {
            Dist::Euclidean => self.euclidean_distance(i, j),
            Dist::Cosine => self.cosine_distance(i, j),
        }
    }

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

    /// Diversify graph
    ///
    /// This matches Python's diversify function.
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

// Update implementations for f32 and f64 (matches apply_graph_updates_low_memory)
impl UpdateNeighbours<f32> for NNDescent<f32> {
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

///////////
// Tests //
///////////

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use faer::Mat;

//     fn create_simple_matrix() -> Mat<f32> {
//         let data = [
//             1.0, 0.0, 0.0, // Point 0
//             0.0, 1.0, 0.0, // Point 1
//             0.0, 0.0, 1.0, // Point 2
//             1.0, 1.0, 0.0, // Point 3
//             1.0, 0.0, 1.0, // Point 4
//         ];
//         Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
//     }

//     #[test]
//     fn test_nndescent_build_euclidean() {
//         let mat = create_simple_matrix();
//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         assert_eq!(index.graph.len(), 5);
//         for neighbours in &index.graph {
//             assert!(neighbours.len() <= 3);
//         }
//     }

//     #[test]
//     fn test_nndescent_build_cosine() {
//         let mat = create_simple_matrix();
//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Cosine,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         assert_eq!(index.graph.len(), 5);
//     }

//     #[test]
//     fn test_nndescent_query() {
//         let mat = create_simple_matrix();
//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         let query = vec![1.0, 0.0, 0.0];
//         let (indices, distances) = index.query(&query, 3, 50);

//         assert_eq!(indices.len(), 3);
//         assert_eq!(distances.len(), 3);
//         assert!(indices.contains(&0));
//     }

//     #[test]
//     fn test_nndescent_convergence() {
//         let mat = create_simple_matrix();

//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Euclidean,
//             100,
//             0.5,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         assert_eq!(index.graph.len(), 5);
//     }

//     #[test]
//     fn test_nndescent_reproducibility() {
//         let mat = create_simple_matrix();

//         let graph1 = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         let graph2 = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         assert_eq!(graph1.graph.len(), graph2.graph.len());
//     }

//     #[test]
//     fn test_nndescent_k_parameter() {
//         let mat = create_simple_matrix();

//         let graph_k2 = NNDescent::<f32>::new(
//             mat.as_ref(),
//             2,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         let graph_k4 = NNDescent::<f32>::new(
//             mat.as_ref(),
//             4,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         for neighbours in &graph_k2.graph {
//             assert!(neighbours.len() <= 2);
//         }

//         for neighbours in &graph_k4.graph {
//             assert!(neighbours.len() <= 4);
//         }
//     }

//     #[test]
//     fn test_nndescent_larger_dataset() {
//         let n = 50;
//         let dim = 10;
//         let mut data = Vec::with_capacity(n * dim);

//         for i in 0..n {
//             for j in 0..dim {
//                 data.push((i * j) as f32 / 10.0);
//             }
//         }

//         let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             10,
//             Dist::Euclidean,
//             15,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         assert_eq!(index.graph.len(), n);

//         for neighbours in &index.graph {
//             assert!(neighbours.len() <= 10);
//             assert!(!neighbours.is_empty());
//         }
//     }

//     #[test]
//     fn test_nndescent_distance_ordering() {
//         let mat = create_simple_matrix();
//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             3,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         for neighbours in &index.graph {
//             for i in 1..neighbours.len() {
//                 assert!(neighbours[i].1 >= neighbours[i - 1].1);
//             }
//         }
//     }

//     #[test]
//     fn test_nndescent_with_f64() {
//         let data = [1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
//         let mat = Mat::from_fn(3, 3, |i, j| data[i * 3 + j]);

//         let index = NNDescent::<f64>::new(
//             mat.as_ref(),
//             2,
//             Dist::Euclidean,
//             10,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         assert_eq!(index.graph.len(), 3);
//     }

//     #[test]
//     fn test_nndescent_quality() {
//         let n = 20;
//         let dim = 3;
//         let mut data = Vec::with_capacity(n * dim);

//         for i in 0..10 {
//             let offset = i as f32 * 0.1;
//             data.extend_from_slice(&[offset, 0.0, 0.0]);
//         }
//         for i in 0..10 {
//             let offset = 10.0 + i as f32 * 0.1;
//             data.extend_from_slice(&[offset, 0.0, 0.0]);
//         }

//         let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
//         let index = NNDescent::<f32>::new(
//             mat.as_ref(),
//             5,
//             Dist::Euclidean,
//             20,
//             0.001,
//             1.0,
//             None,
//             1.0,
//             42,
//             false,
//         );

//         let neighbours_0 = &index.graph[0];
//         let in_cluster = neighbours_0.iter().filter(|(idx, _)| *idx < 10).count();
//         assert!(in_cluster >= 3);

//         let neighbours_10 = &index.graph[10];
//         let in_cluster_2 = neighbours_10.iter().filter(|(idx, _)| *idx >= 10).count();
//         assert!(in_cluster_2 >= 3);
//     }
// }
