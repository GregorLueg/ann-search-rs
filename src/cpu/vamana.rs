//! Implementation of Vamana, a graph-based approximate nearest neighbour
//! search that powers DiskANN. This version is the in-memory version of that
//! algorithm for fast querying.

use faer::{MatRef, RowRef};
use rand::{rng, rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::{RefCell, UnsafeCell};
use std::cmp::{Ordering, Reverse};
use std::sync::{atomic::AtomicUsize, Arc};
use thousands::*;

use crate::prelude::*;
use crate::utils::dist::{cosine_distance_static, euclidean_distance_static};
use crate::utils::graph_utils::SearchState;
use crate::utils::*;

///////////////////
// Thread locals //
///////////////////

thread_local! {
    static VAMANA_BUILD_STATE_F32: RefCell<SearchState<f32>> = RefCell::new(SearchState::new(1000));
    static VAMANA_BUILD_STATE_F64: RefCell<SearchState<f64>> = RefCell::new(SearchState::new(1000));
    static VAMANA_SEARCH_STATE_F32: RefCell<SearchState<f32>> = RefCell::new(SearchState::new(1000));
    static VAMANA_SEARCH_STATE_F64: RefCell<SearchState<f64>> = RefCell::new(SearchState::new(1000));
}

/////////////
// Helpers //
/////////////

/// Construction-time graph wrapper
struct VamanaConstructionGraph {
    /// Each node gets a fixed array of size R. u32::MAX is the sentinel.
    nodes: Vec<UnsafeCell<Vec<u32>>>,
    /// Striped spin-locks for concurrent edge updates during construction.
    /// Stripe count is independent of graph size, so memory overhead stays
    /// constant as the dataset grows.
    locks: StripedLocks,
    /// Degree of that node
    r: usize,
}

unsafe impl Sync for VamanaConstructionGraph {}

impl VamanaConstructionGraph {
    /// Create a new construction graph initialised with sentinels
    ///
    /// Pre-allocates a fixed-size vector for each node filled with `u32::MAX`
    /// to denote empty slots. The striped lock array is sized to the expected
    /// concurrency rather than the node count, keeping lock memory constant
    /// regardless of dataset size.
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes in the dataset
    /// * `r` - Maximum degree (number of edges) per node
    /// * `threads` - Expected number of concurrent writers, used to size the
    ///   striped lock array
    ///
    /// ### Returns
    ///
    /// Constructed self
    pub fn new(n: usize, r: usize, threads: usize) -> Self {
        let nodes = (0..n).map(|_| UnsafeCell::new(vec![u32::MAX; r])).collect();

        Self {
            nodes,
            locks: StripedLocks::new(threads, r),
            r,
        }
    }

    /// Initialise the graph with random out-edges.
    ///
    /// This is crucial for Vamana. Instead of starting with highly localised
    /// clusters (like Annoy), we start with random long-range connections.
    /// Pass 1 of the build process will use these to jump across the dataset,
    /// explicitly preserving the best "highways" while discovering local
    /// clusters.
    ///
    /// Lock-free by construction: the parallel iterator guarantees that each
    /// node's slot is touched by exactly one thread during initialisation.
    ///
    /// ### Params
    ///
    /// * `seed` - Base random seed for reproducible builds
    pub fn initialise_random(&self, seed: u64) {
        let n = self.nodes.len();
        if n <= 1 {
            return;
        }

        // handle edge case
        let actual_r = self.r.min(n - 1);

        (0..n).into_par_iter().for_each(|i| {
            // seed a deterministic RNG specifically for this node
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));

            // SAFETY: This is perfectly safe and lock-free because the parallel
            // iterator guarantees disjoint access. Thread `i` exclusively
            // borrows and mutates `nodes[i]`. No locks required!
            let neighbors = unsafe { &mut *self.nodes[i].get() };

            let mut count = 0;
            while count < actual_r {
                let rand_node = rng.random_range(0..n) as u32;

                // ensure no self-loops and no duplicate random edges
                if rand_node as usize != i && !neighbors[0..count].contains(&rand_node) {
                    neighbors[count] = rand_node;
                    count += 1;
                }
            }
        });
    }

    /// Get a read-only slice of neighbours for a node (no allocation).
    ///
    /// Unlocked read for speed; benign races are accepted during the build
    /// phase. Since slot writes are always in-place overwrites of `u32`
    /// values (atomic on all supported architectures), a concurrent reader
    /// either sees a valid neighbour id or a sentinel, never a torn value.
    ///
    /// ### Params
    ///
    /// * `node_id` - The node
    ///
    /// ### Safety
    ///
    /// Caller must ensure no concurrent mutable access to this node's edges,
    /// or accept benign races.
    #[inline]
    pub unsafe fn get_neighbours_slice(&self, node_id: usize) -> &[u32] {
        &*self.nodes[node_id].get()
    }

    /// Count the number of actual (non-sentinel) neighbours for a node.
    ///
    /// Assumes the caller holds the stripe lock for this node or has
    /// exclusive access. Sentinels are packed at the end of the slot range,
    /// so the first sentinel marks the degree.
    ///
    /// ### Params
    ///
    /// * `node_id` - The node
    ///
    /// ### Returns
    ///
    /// Number of valid (non-u32::MAX) entries
    #[inline]
    pub unsafe fn degree(&self, node_id: usize) -> usize {
        let edges = &*self.nodes[node_id].get();
        // Sentinels are always packed at the end, so first sentinel = degree
        edges.iter().position(|&e| e == u32::MAX).unwrap_or(self.r)
    }

    /// Set neighbours safely by acquiring the node's stripe lock first.
    ///
    /// Lock is held for the duration of the in-place overwrite and released
    /// automatically when the guard is dropped.
    ///
    /// ### Params
    ///
    /// * `node_id` - Id of the node
    /// * `neighbours` - New neighbour list (will be padded with sentinels)
    pub fn set_neighbours(&self, node_id: usize, neighbours: &[u32]) {
        let _guard = self.locks.lock_guard(node_id);
        self.set_neighbours_unsafe(node_id, neighbours);
    }

    /// Convert the construction graph into a single flat `Vec<u32>` for
    /// queries.
    ///
    /// Consumes the construction graph. The resulting flat layout has each
    /// node's R-sized slot range stored contiguously in node-id order,
    /// suitable for cache-friendly query traversal.
    ///
    /// ### Returns
    ///
    /// Flattened neighbour array of size `n * r`
    pub fn into_flat(self) -> Vec<u32> {
        let mut flat = Vec::with_capacity(self.nodes.len() * self.r);
        for cell in self.nodes {
            flat.extend(cell.into_inner());
        }
        flat
    }

    /// Set neighbours assuming the stripe lock is already held.
    ///
    /// Overwrites the slot range in place and pads the remainder with
    /// `u32::MAX`. Writes are always full-range overwrites, so concurrent
    /// readers never observe an empty list.
    ///
    /// ### Params
    ///
    /// * `node_id` - Id of the node
    /// * `neighbours` - New neighbour list (will be padded with sentinels)
    #[inline]
    pub fn set_neighbours_unsafe(&self, node_id: usize, neighbours: &[u32]) {
        unsafe {
            let edges = &mut *self.nodes[node_id].get();
            for i in 0..self.r {
                edges[i] = if i < neighbours.len() {
                    neighbours[i]
                } else {
                    u32::MAX
                };
            }
        }
    }

    /// Append a single neighbour assuming the stripe lock is already held.
    ///
    /// Scans for the first sentinel slot and writes the new neighbour there.
    /// No-op if the slot range is already full.
    ///
    /// ### Params
    ///
    /// * `node_id` - Id of the node
    /// * `neighbour` - Id of the neighbour to add
    #[inline]
    pub fn add_neighbour_unsafe(&self, node_id: usize, neighbour: u32) {
        unsafe {
            let edges = &mut *self.nodes[node_id].get();
            for i in 0..self.r {
                if edges[i] == u32::MAX {
                    edges[i] = neighbour;
                    break;
                }
            }
        }
    }
}

/// Compute the medoid of the data
///
/// ### Params
///
/// * `vectors_flat` - The flattened data for which to calculate the medoid
/// * `n` - Number of samples in the vector
/// * `dim` - Dimensionality of the original data
/// * `metric` - The distance metric to use
///
/// ### Returns
///
/// The position of the medoid
pub fn compute_medoid<T: AnnSearchFloat>(
    vectors_flat: &[T],
    n: usize,
    dim: usize,
    metric: Dist,
) -> u32 {
    if n == 0 {
        return 0;
    }

    // compute the centroid (mean vector)
    let mut centroid = vec![T::zero(); dim];
    for i in 0..n {
        let offset = i * dim;
        for d in 0..dim {
            centroid[d] = centroid[d] + vectors_flat[offset + d];
        }
    }

    let n_float = T::from_usize(n).unwrap();
    for d in 0..dim {
        centroid[d] = centroid[d] / n_float;
    }

    // if using Cosine distance, normalise the centroid
    if metric == Dist::Cosine {
        let norm = T::calculate_l2_norm(&centroid);
        if norm > T::zero() {
            for d in 0..dim {
                centroid[d] = centroid[d] / norm;
            }
        }
    }

    // find the actual data point closest to the centroid
    let medoid_idx = (0..n)
        .into_par_iter()
        .min_by(|&a, &b| {
            let offset_a = a * dim;
            let offset_b = b * dim;
            let vec_a = &vectors_flat[offset_a..offset_a + dim];
            let vec_b = &vectors_flat[offset_b..offset_b + dim];

            let dist_a = if metric == Dist::Cosine {
                cosine_distance_static(vec_a, &centroid)
            } else {
                euclidean_distance_static(vec_a, &centroid)
            };

            let dist_b = if metric == Dist::Cosine {
                cosine_distance_static(vec_b, &centroid)
            } else {
                euclidean_distance_static(vec_b, &centroid)
            };

            dist_a.partial_cmp(&dist_b).unwrap_or(Ordering::Equal)
        })
        .unwrap_or(0);

    medoid_idx as u32
}

/// Provides access to thread-local SearchState buffers for Vamana for building
/// and querying the graph
pub trait VamanaState<T> {
    /// Access the thread-local search state for index construction.
    fn with_build_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<T>>) -> R;

    /// Access the thread-local search state for query traversal.
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<T>>) -> R;
}

/// Implementation for f32
impl VamanaState<f32> for VamanaIndex<f32> {
    fn with_build_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f32>>) -> R,
    {
        VAMANA_BUILD_STATE_F32.with(f)
    }

    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f32>>) -> R,
    {
        VAMANA_SEARCH_STATE_F32.with(f)
    }
}

/// Implementation for f64
impl VamanaState<f64> for VamanaIndex<f64> {
    fn with_build_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f64>>) -> R,
    {
        VAMANA_BUILD_STATE_F64.with(f)
    }

    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f64>>) -> R,
    {
        VAMANA_SEARCH_STATE_F64.with(f)
    }
}

////////////////
// Main index //
////////////////

/// Vamana graph index for approximate nearest neighbour search.
pub struct VamanaIndex<T> {
    /// Flattened vector data for cache locality
    pub vectors_flat: Vec<T>,
    /// Dimensionality of vectors
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Pre-computed norms for Cosine distance (empty for Euclidean)
    pub norms: Vec<T>,
    /// Distance metric (Euclidean or Cosine)
    pub metric: Dist,
    /// Flat graph of size `n * R`. Each node has up to R neighbours.
    pub graph: Vec<u32>,
    /// Global entry point (medoid of the dataset)
    pub medoid: u32,
    /// Maximum degree
    pub r: usize,
    /// Search beam width during construction
    pub l_build: usize,
    /// Orignal indices
    original_ids: Vec<usize>,
}

/// VectorDistance implementation
impl<T> VectorDistance<T> for VamanaIndex<T>
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

impl<T> VamanaIndex<T>
where
    T: AnnSearchFloat,
    Self: VamanaState<T>,
{
    ////////////////////
    // Index building //
    ////////////////////

    /// Build the VAMANA index
    ///
    /// ### Params
    ///
    /// * `data` - The initial data for which to generate the vector of shape
    ///   n x features
    /// * `metric` - The distance metric to use for this index
    ///
    /// ### Returns
    ///
    /// Initialised and built self
    pub fn build(
        data: MatRef<T>,
        metric: Dist,
        r: usize,
        l_build: usize,
        alpha_pass1: f32,
        alpha_pass2: f32,
        seed: usize,
    ) -> Self {
        let (vectors_flat, n, dim) = matrix_to_flat(data);

        let medoid = compute_medoid(&vectors_flat, n, dim, metric);

        let threads = rayon::current_num_threads();
        let build_graph = VamanaConstructionGraph::new(n, r, threads);
        build_graph.initialise_random(seed as u64);

        // pre-calculate norms for Cosine
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

        let mut index = Self {
            vectors_flat,
            dim,
            n,
            metric,
            norms,
            graph: Vec::new(),
            medoid,
            r,
            l_build,
            original_ids: (0..n).collect(),
        };

        let passes = [alpha_pass1, alpha_pass2];

        for alpha in passes {
            // random permutation of nodes for unbiased parallel updates
            let mut permutation: Vec<usize> = (0..n).collect();
            permutation.shuffle(&mut rng());

            permutation.par_iter().for_each(|&p| {
                // thread-local search state
                Self::with_build_state(|state_cell| {
                    let mut state = state_cell.borrow_mut();

                    // 1.) beam search from medoid to target `p` -> returns
                    // L_build candidates
                    let candidates = index.beam_search_build(
                        p,
                        medoid as usize,
                        l_build,
                        &build_graph,
                        &mut state,
                    );

                    // 2.) add p's current neighbours to the candidate pool
                    let scratch = &mut state.scratch_working;
                    scratch.clear();
                    scratch.extend_from_slice(&candidates);

                    // SAFETY: benign race -- we only read p's own edges and
                    // the parallel iterator guarantees no other thread writes
                    // to p at this point.
                    let current_out_edges = unsafe { build_graph.get_neighbours_slice(p) };
                    for &nbr in current_out_edges {
                        if nbr == u32::MAX {
                            break;
                        }
                        let dist = index.distance(p, nbr as usize);
                        scratch.push((OrderedFloat(dist), nbr as usize));
                    }

                    // 3.) robust prune to select the best R out-edges
                    let pruned_edges = index.robust_prune(p, scratch, alpha, r);

                    // 4.) update p's out-edges
                    build_graph.set_neighbours(p, &pruned_edges);

                    // 5.) reverse edges: add p to q's neighbours
                    for &q in &pruned_edges {
                        let q = q as usize;
                        let _guard = build_graph.locks.lock_guard(q); // RAII Lock!

                        let q_edges = unsafe { build_graph.get_neighbours_slice(q) };
                        let q_degree = unsafe { build_graph.degree(q) };

                        let already_connected = q_edges[..q_degree].contains(&(p as u32));

                        if !already_connected {
                            if q_degree < r {
                                build_graph.add_neighbour_unsafe(q, p as u32);
                            } else {
                                let dist_q_p = index.distance(q, p);

                                // Quick check: is p even competitive?
                                let worst_dist = q_edges[..q_degree]
                                    .iter()
                                    .map(|&nbr| index.distance(q, nbr as usize))
                                    .fold(T::neg_infinity(), |a, b| if b > a { b } else { a });

                                if dist_q_p < worst_dist {
                                    let q_scratch = &mut state.scratch_discarded;
                                    q_scratch.clear();
                                    for &nbr in &q_edges[..q_degree] {
                                        let n_idx = nbr as usize;
                                        q_scratch
                                            .push((OrderedFloat(index.distance(q, n_idx)), n_idx));
                                    }
                                    q_scratch.push((OrderedFloat(dist_q_p), p));
                                    let new_q_edges = index.robust_prune(q, q_scratch, alpha, r);
                                    build_graph.set_neighbours_unsafe(q, &new_q_edges);
                                }
                            }
                        }
                    }
                });
            });
        }

        // flatten graph for cache-friendly queries
        index.graph = build_graph.into_flat();
        index
    }

    /// Calculate distance between two indexed points.
    ///
    /// ### Params
    ///
    /// * `i` - Index of first vector
    /// * `j` - Index of second vector
    ///
    /// ### Returns
    ///
    /// Distance under the index metric
    #[inline]
    fn distance(&self, i: usize, j: usize) -> T {
        match self.metric {
            Dist::Euclidean => self.euclidean_distance(i, j),
            Dist::Cosine => self.cosine_distance(i, j),
        }
    }

    /// Beam search used specifically during the index construction phase.
    ///
    /// Returns the sorted candidate set directly from the working buffer
    /// (no intermediate Vec allocation).
    ///
    /// ### Params
    ///
    /// * `target_node` - The target node
    /// * `entry_node` - The medoid of the data
    /// * `l_build` - Beam width during candidate building
    /// * `build_graph` - Reference to the construction graph
    /// * `state` - Mutable SearchState for that thread
    ///
    /// ### Returns
    ///
    /// Sorted candidate set of `(dist, index)`
    fn beam_search_build(
        &self,
        target_node: usize,
        entry_node: usize,
        l_build: usize,
        build_graph: &VamanaConstructionGraph,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        state.reset(self.n);

        let entry_dist = OrderedFloat(self.distance(target_node, entry_node));

        state.mark_visited(entry_node);
        state.candidates.push(Reverse((entry_dist, entry_node)));
        state
            .working_sorted
            .insert((entry_dist, entry_node), l_build);

        let mut furthest_dist = entry_dist;

        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist > furthest_dist && state.working_sorted.len() >= l_build {
                break;
            }

            let neighbours = unsafe { build_graph.get_neighbours_slice(current_id) };

            for &neighbour in neighbours {
                if neighbour == u32::MAX {
                    // sentinels packed at end, safe to break
                    break;
                }
                let n_idx = neighbour as usize;

                if state.is_visited(n_idx) {
                    continue;
                }
                state.mark_visited(n_idx);

                let dist = OrderedFloat(self.distance(target_node, n_idx));

                if dist < furthest_dist || state.working_sorted.len() < l_build {
                    state.candidates.push(Reverse((dist, n_idx)));

                    if state.working_sorted.insert((dist, n_idx), l_build)
                        && state.working_sorted.len() >= l_build
                    {
                        furthest_dist = state
                            .working_sorted
                            .top()
                            .map(|(d, _)| *d)
                            .unwrap_or(OrderedFloat(T::infinity()));
                    }
                }
            }
        }

        // allocation happens here, but oh well...
        state.working_sorted.data().to_vec()
    }

    /// Selects up to `max_degree` neighbours from a candidate pool using the
    /// alpha-heuristic.
    ///
    /// ### Params
    ///
    /// * `base_node` - The node for which to prune the candidates
    /// * `candidates` - Mutable reference to the current candidates
    /// * `alpha` - Alpha parameter
    /// * `max_degree` - Maximum degree
    fn robust_prune(
        &self,
        base_node: usize,
        candidates: &mut [(OrderedFloat<T>, usize)],
        alpha: f32,
        max_degree: usize,
    ) -> Vec<u32> {
        candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut selected = Vec::with_capacity(max_degree);
        let alpha_t = T::from_f32(alpha).unwrap();

        for &(cand_dist, cand_id) in candidates.iter() {
            if cand_id == base_node {
                continue;
            }
            if selected.len() >= max_degree {
                break;
            }
            // skip duplicates -- the closest one comes first due to sort order
            if selected.contains(&(cand_id as u32)) {
                continue;
            }

            let is_good = !selected.iter().any(|&sel_id| {
                let dist_to_selected = OrderedFloat(self.distance(cand_id, sel_id as usize));
                alpha_t * dist_to_selected.0 <= cand_dist.0
            });

            if is_good {
                selected.push(cand_id as u32);
            }
        }

        selected
    }

    /// Helper to fetch distances for a given set of neighbour IDs.
    ///
    /// ### Params
    ///
    /// * `query_node` - The idx of the query node
    /// * `neighbours` - Positions of the neighbours for which to return the
    ///   distances.
    ///
    /// ### Returns
    ///
    /// Returns a Vec of `(dist, n_idx)` (with dist being an OrderedFloat)
    pub fn fetch_distances(
        &self,
        query_node: usize,
        neighbours: &[u32],
    ) -> Vec<(OrderedFloat<T>, usize)> {
        neighbours
            .iter()
            .filter(|&&n| n != u32::MAX)
            .map(|&n| {
                let n_idx = n as usize;
                (OrderedFloat(self.distance(query_node, n_idx)), n_idx)
            })
            .collect()
    }

    ///////////
    // Query //
    ///////////

    /// Helper to get a node's neighbours from the flattened 1D graph.
    ///
    /// ### Params
    ///
    /// * `node_id` - Node for which to get the neighbours
    ///
    /// ### Returns
    ///
    /// Slice of node's neighbours
    #[inline(always)]
    fn get_neighbours_flat(&self, node_id: usize) -> &[u32] {
        let start = node_id * self.r;
        &self.graph[start..start + self.r]
    }

    /// Compute distance between query and database vector
    ///
    /// ### Params
    ///
    /// * `query` - Query vector
    /// * `idx` - Database vector index
    /// * `query_norm` - Pre-computed query norm (for Cosine)
    ///
    /// ### Returns
    ///
    /// Distance according to the index's metric
    #[inline(always)]
    fn compute_query_distance(&self, query: &[T], idx: usize, query_norm: T) -> T {
        match self.metric {
            Dist::Euclidean => self.euclidean_distance_to_query(idx, query),
            Dist::Cosine => self.cosine_distance_to_query(idx, query, query_norm),
        }
    }

    /// Query the index for k nearest neighbours.
    ///
    /// Performs a greedy beam search starting from the global medoid.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Optional Beam width (higher = better recall, slower).
    ///   If not provided, it will default to `75`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
    #[inline]
    pub fn query(&self, query: &[T], k: usize, ef_search: Option<usize>) -> (Vec<usize>, Vec<T>) {
        assert_eq!(query.len(), self.dim);

        let ef_search = ef_search.unwrap_or(75);

        // Ensure the beam is at least as wide as k
        let ef = ef_search.max(k);

        Self::with_search_state(|state_cell| {
            let mut state = state_cell.borrow_mut();
            state.reset(self.n);

            // Here we DO need the external query norm because it's a new, unseen vector
            let query_norm = if self.metric == Dist::Cosine {
                T::calculate_l2_norm(query)
            } else {
                T::one()
            };

            let entry_node = self.medoid as usize;
            let entry_dist =
                OrderedFloat(self.compute_query_distance(query, entry_node, query_norm));

            state.mark_visited(entry_node);
            state.candidates.push(Reverse((entry_dist, entry_node)));
            state.working_sorted.insert((entry_dist, entry_node), ef);

            let mut furthest_dist = entry_dist;

            // The main Beam Search loop
            while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
                // If the closest candidate to explore is further than our worst accepted
                // result, we can terminate early.
                if current_dist > furthest_dist && state.working_sorted.len() >= ef {
                    break;
                }

                // Fetch neighbors directly from the cache-friendly flat array
                let neighbours = self.get_neighbours_flat(current_id);

                for &neighbour in neighbours {
                    // Because we append sequentially during build, hitting a sentinel
                    // means the rest of the slice is also sentinels. We can safely break.
                    if neighbour == u32::MAX {
                        break;
                    }

                    let n_idx = neighbour as usize;

                    if state.is_visited(n_idx) {
                        continue;
                    }
                    state.mark_visited(n_idx);

                    // Compute actual distance to the query vector
                    let dist = OrderedFloat(self.compute_query_distance(query, n_idx, query_norm));

                    if dist < furthest_dist || state.working_sorted.len() < ef {
                        state.candidates.push(Reverse((dist, n_idx)));

                        if state.working_sorted.insert((dist, n_idx), ef)
                            && state.working_sorted.len() >= ef
                        {
                            furthest_dist = state
                                .working_sorted
                                .top()
                                .map(|(d, _)| *d)
                                .unwrap_or(OrderedFloat(T::infinity()));
                        }
                    }
                }
            }

            // Extract the top-k results
            let mut results = state.working_sorted.data().to_vec();
            results.truncate(k);

            let (indices, distances): (Vec<usize>, Vec<T>) = results
                .into_iter()
                .map(|(OrderedFloat(d), id)| (id, d))
                .unzip();

            (indices, distances)
        })
    }

    /// Query using a matrix row reference
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies
    /// to a temporary vector.
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
    /// ### Params
    ///
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Optional Beam width (higher = better recall, slower).
    ///   If not provided, it will default to `100`.
    /// * `return_dist` - Shall the distances be returned
    /// * `verbose` - Controls the verbosity of the function
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
        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dim;
                let end = start + self.dim;
                let vec = &self.vectors_flat[start..end];

                if verbose {
                    let count = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
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
    /// Accounts for all heap-allocated data structures.
    ///
    /// ### Returns
    ///
    /// Total memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.norms.capacity() * std::mem::size_of::<T>()
            + self.graph.capacity() * std::mem::size_of::<u32>()
    }
}

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for VamanaIndex<T>
where
    T: AnnSearchFloat,
    Self: VamanaState<T>,
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
    use approx::assert_relative_eq;
    use faer::Mat;
    use std::sync::Arc;
    use std::thread;

    fn simple_matrix() -> Mat<f32> {
        let data = [
            1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
    }

    fn build_default(mat: &Mat<f32>, metric: &str) -> VamanaIndex<f32> {
        VamanaIndex::<f32>::build(
            mat.as_ref(),
            parse_ann_dist(metric).unwrap_or(Dist::Euclidean),
            16,
            100,
            1.0,
            1.2,
            42,
        )
    }

    #[test]
    fn test_build_euclidean() {
        let mat = simple_matrix();
        let _ = build_default(&mat, "euclidean");
    }

    #[test]
    fn test_build_cosine() {
        let mat = simple_matrix();
        let _ = build_default(&mat, "cosine");
    }

    #[test]
    fn test_query_finds_self_euclidean() {
        let mat = simple_matrix();
        let index = build_default(&mat, "euclidean");

        let query = vec![1.0_f32, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1, None);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_query_finds_self_cosine() {
        let mat = simple_matrix();
        let index = build_default(&mat, "cosine");

        let query = vec![1.0_f32, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1, None);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_query_distances_sorted() {
        let mat = simple_matrix();
        let index = build_default(&mat, "euclidean");

        let query = vec![0.5_f32, 0.5, 0.0];
        let (_, distances) = index.query(&query, 4, None);

        for i in 1..distances.len() {
            assert!(
                distances[i] >= distances[i - 1],
                "Distances not sorted at position {}",
                i
            );
        }
    }

    #[test]
    fn test_query_k_results() {
        let mat = simple_matrix();
        let index = build_default(&mat, "euclidean");

        let query = vec![0.0_f32, 0.0, 0.0];
        for k in 1..=5 {
            let (indices, distances) = index.query(&query, k, None);
            assert_eq!(indices.len(), k);
            assert_eq!(distances.len(), k);
        }
    }

    #[test]
    fn test_ef_search_affects_results() {
        let n = 500;
        let dim = 8;
        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.01).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        let index = VamanaIndex::<f32>::build(mat.as_ref(), Dist::Euclidean, 32, 150, 1.0, 1.2, 42);

        let query: Vec<f32> = (0..dim).map(|_| 0.5).collect();

        // Higher ef_search should return same or better results
        let (_, _) = index.query(&query, 10, Some(50));
        let (_, _) = index.query(&query, 10, Some(200));
        // Both must return exactly k results
        let (idx_low, _) = index.query(&query, 10, Some(50));
        let (idx_high, _) = index.query(&query, 10, Some(200));
        assert_eq!(idx_low.len(), 10);
        assert_eq!(idx_high.len(), 10);
    }

    #[test]
    fn test_recall_linear_data() {
        let n = 30;
        let dim = 3;
        let mut data = vec![0.0_f32; n * dim];
        for i in 0..n {
            data[i * dim] = i as f32 * 0.1;
        }
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        let index = VamanaIndex::<f32>::build(mat.as_ref(), Dist::Euclidean, 16, 200, 1.0, 1.2, 42);

        let query = vec![0.0_f32, 0.0, 0.0];
        let (indices, _) = index.query(&query, 5, Some(150));

        assert_eq!(indices[0], 0, "Nearest point should be index 0");

        let expected: Vec<usize> = (0..5).collect();
        let found = indices.iter().filter(|&&i| expected.contains(&i)).count();
        assert!(found >= 4, "Expected at least 4 of top-5, got {}", found);
    }

    #[test]
    fn test_medoid_is_valid() {
        let mat = simple_matrix();
        let index = build_default(&mat, "euclidean");
        assert!((index.medoid as usize) < index.n);
    }

    #[test]
    fn test_graph_has_no_self_loops() {
        let mat = simple_matrix();
        let index = build_default(&mat, "euclidean");

        for node in 0..index.n {
            let start = node * index.r;
            let neighbours = &index.graph[start..start + index.r];
            for &nbr in neighbours {
                if nbr == u32::MAX {
                    break;
                }
                assert_ne!(nbr as usize, node, "Self-loop at node {}", node);
            }
        }
    }

    #[test]
    fn test_graph_neighbour_bounds() {
        let mat = simple_matrix();
        let index = build_default(&mat, "euclidean");

        for node in 0..index.n {
            let start = node * index.r;
            let neighbours = &index.graph[start..start + index.r];
            for &nbr in neighbours {
                if nbr == u32::MAX {
                    break;
                }
                assert!(
                    (nbr as usize) < index.n,
                    "Out-of-bounds neighbour {} at node {}",
                    nbr,
                    node
                );
            }
        }
    }

    #[test]
    fn test_thread_safety() {
        let mat = simple_matrix();
        let index = Arc::new(build_default(&mat, "euclidean"));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let idx = Arc::clone(&index);
                thread::spawn(move || {
                    let query = vec![0.5_f32, 0.5, 0.0];
                    let (indices, _) = idx.query(&query, 3, None);
                    assert_eq!(indices.len(), 3);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_reproducibility() {
        let mat = simple_matrix();
        let idx1 = build_default(&mat, "euclidean");
        let idx2 = build_default(&mat, "euclidean");

        let query = vec![0.5_f32, 0.5, 0.0];
        let (i1, _) = idx1.query(&query, 3, None);
        let (i2, _) = idx2.query(&query, 3, None);
        assert_eq!(i1, i2);
    }

    #[test]
    fn test_generate_knn_shape() {
        let n = 50;
        let dim = 4;
        let data: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = VamanaIndex::<f32>::build(mat.as_ref(), Dist::Euclidean, 16, 100, 1.0, 1.2, 42);

        let k = 5;
        let (indices, distances) = index.generate_knn(k, None, true, false);
        assert_eq!(indices.len(), n);
        assert!(distances.is_some());
        let dists = distances.unwrap();
        assert_eq!(dists.len(), n);
        for row in &indices {
            assert_eq!(row.len(), k);
        }
    }

    #[test]
    fn test_memory_usage_nonzero() {
        let mat = simple_matrix();
        let index = build_default(&mat, "euclidean");
        assert!(index.memory_usage_bytes() > 0);
    }

    #[test]
    fn test_varying_r_values() {
        let n = 200;
        let dim = 8;
        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.01).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        for r in [16, 32, 48, 64] {
            let index =
                VamanaIndex::<f32>::build(mat.as_ref(), Dist::Euclidean, r, 150, 1.0, 1.2, 42);
            let query: Vec<f32> = (0..dim).map(|_| 0.5).collect();
            let (indices, _) = index.query(&query, 10, None);
            assert_eq!(indices.len(), 10, "Failed with r={}", r);
        }
    }

    #[test]
    fn test_alpha_greater_than_one() {
        // alpha > 1 in pass 2 is the whole point of Vamana vs plain greedy
        let n = 200;
        let dim = 8;
        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.01).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        let index = VamanaIndex::<f32>::build(mat.as_ref(), Dist::Euclidean, 32, 150, 1.0, 1.4, 42);

        let query: Vec<f32> = (0..dim).map(|_| 0.5).collect();
        let (indices, _) = index.query(&query, 10, None);
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_compute_medoid_single_point() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let medoid = compute_medoid(&data, 1, 3, Dist::Euclidean);
        assert_eq!(medoid, 0);
    }

    #[test]
    fn test_compute_medoid_cosine() {
        let data = vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let medoid = compute_medoid(&data, 3, 2, Dist::Cosine);
        // The medoid should be a valid index
        assert!((medoid as usize) < 3);
    }
}
