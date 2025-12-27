use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::{
    cell::UnsafeCell,
    cmp::Reverse,
    collections::BinaryHeap,
    iter::Sum,
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};
use thousands::*;

use crate::utils::dist::*;
use crate::utils::heap_structs::*;
use crate::utils::*;

/////////////
// Helpers //
/////////////

pub type NeighbourUpdates<T> = Vec<(usize, Vec<(OrderedFloat<T>, usize)>)>;

/// Lock-free bitset for node-level locking during construction
///
/// Uses atomic bit operations instead of heavyweight RwLock. Each bit
/// represents the lock state of one node.
///
/// ### Performance
///
/// - Lock acquisition: ~5ns vs ~50ns for RwLock
/// - No kernel involvement (user-space only)
/// - Better cache locality (64 locks per cache line)
///
/// This is based on what Usearch does in their C++ code and with help of
/// Claude
#[derive(Debug)]
pub struct AtomicNodeLocks {
    bits: Vec<AtomicU64>,
}

impl AtomicNodeLocks {
    /// Create a new lock set for the given capacity
    ///
    /// ### Params
    ///
    /// * `capacity` - Maximum number of nodes that can be locked
    pub fn new(capacity: usize) -> Self {
        let num_slots = capacity.div_ceil(64);
        Self {
            bits: (0..num_slots).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Attempt to acquire lock without blocking
    ///
    /// ### Params
    ///
    /// * `idx` - Node index to lock
    ///
    /// ### Returns
    ///
    /// `true` if lock was already held (failed to acquire)
    /// `false` if lock was acquired successfully
    #[inline(always)]
    pub fn try_lock(&self, idx: usize) -> bool {
        let slot = idx / 64;
        let bit = 1u64 << (idx % 64);
        let prev = self.bits[slot].fetch_or(bit, Ordering::Acquire);
        (prev & bit) != 0
    }

    /// Spin-lock until acquired
    ///
    /// ### Params
    ///
    /// * `idx` - Node index to lock
    #[inline(always)]
    pub fn lock(&self, idx: usize) {
        while self.try_lock(idx) {
            std::hint::spin_loop();
        }
    }

    /// Release lock
    ///
    /// ### Params
    ///
    /// * `idx` - Node index to unlock
    #[inline(always)]
    pub fn unlock(&self, idx: usize) {
        let slot = idx / 64;
        let bit = !(1u64 << (idx % 64));
        self.bits[slot].fetch_and(bit, Ordering::Release);
    }

    /// Acquire lock with RAII guard
    ///
    /// ### Params
    ///
    /// * `idx` - Node index to lock
    ///
    /// ### Returns
    ///
    /// Guard that automatically releases lock on drop
    #[inline(always)]
    pub fn lock_guard(&self, idx: usize) -> NodeLockGuard<'_> {
        self.lock(idx);
        NodeLockGuard { locks: self, idx }
    }
}

/// RAII guard for atomic lock
///
/// Automatically releases lock when dropped.
pub struct NodeLockGuard<'a> {
    locks: &'a AtomicNodeLocks,
    idx: usize,
}

impl<'a> Drop for NodeLockGuard<'a> {
    /// Drop the Lock
    fn drop(&mut self) {
        self.locks.unlock(self.idx);
    }
}

/// Search state for HNSW queries and construction
///
/// Optimised for low-allocation via epoch-based visitation checks and
/// pre-allocated scratch buffers.
///
/// ### Key Optimizations
///
/// 1. **Epoch-based visits**: O(1) reset instead of clearing HashSet
/// 2. **Sorted working set**: Maintains sorted order incrementally for top-K
///    results
/// 3. **Pre-allocated buffers**: Reused across calls to avoid allocations
///
/// ### Fields
///
/// * `visited` - Stores the "visit epoch" ID for each node
/// * `visit_id` - The current epoch ID
/// * `candidates` - Min-heap for graph traversal (always heap, most efficient)
/// * `working_sorted` - Sorted buffer for top-K results (small ef optimization)
/// * `scratch_working` - Reuse buffer for heuristic selection
/// * `scratch_discarded` - Reuse buffer for heuristic selection
pub struct SearchState<T> {
    visited: Vec<usize>,
    visit_id: usize,
    candidates: BinaryHeap<Reverse<(OrderedFloat<T>, usize)>>,
    working_sorted: SortedBuffer<(OrderedFloat<T>, usize)>,
    scratch_working: Vec<(OrderedFloat<T>, usize)>,
    scratch_discarded: Vec<(OrderedFloat<T>, usize)>,
}

impl<T> SearchState<T>
where
    T: Float + Sum,
{
    /// Generate a new search state
    ///
    /// ### Params
    ///
    /// * `capacity` - The pre-allocated capacity of the search state
    fn new(capacity: usize) -> Self {
        Self {
            visited: vec![0; capacity],
            visit_id: 1,
            candidates: BinaryHeap::with_capacity(capacity),
            working_sorted: SortedBuffer::with_capacity(capacity),
            scratch_working: Vec::with_capacity(capacity),
            scratch_discarded: Vec::with_capacity(capacity),
        }
    }

    /// Reset the search state for a new query
    ///
    /// ### Params
    ///
    /// * `n` - New length for resizing
    fn reset(&mut self, n: usize) {
        if self.visited.len() < n {
            self.visited.resize(n, 0);
        }

        self.visit_id = self.visit_id.wrapping_add(1);
        if self.visit_id == 0 {
            self.visited.fill(0);
            self.visit_id = 1;
        }

        self.candidates.clear();
        self.working_sorted.clear();
        self.scratch_working.clear();
        self.scratch_discarded.clear();
    }

    /// Has node been visited
    ///
    /// ### Returns
    ///
    /// Boolean
    #[inline(always)]
    fn is_visited(&self, node: usize) -> bool {
        self.visited[node] == self.visit_id
    }

    /// Marks node as visisted
    #[inline(always)]
    fn mark_visited(&mut self, node: usize) {
        self.visited[node] = self.visit_id;
    }
}

impl<T: Ord> Default for SortedBuffer<T> {
    /// Default implementation
    fn default() -> Self {
        Self::new()
    }
}

/// Construction-time neighbour storage with concurrent write support
///
/// During construction, each node's neighbours are stored in a separate
/// Vec wrapped in UnsafeCell, with atomic bitset locks enabling parallel
/// insertions without batching or synchronisation barriers.
///
/// ### Safety
///
/// Uses UnsafeCell for interior mutability. Safe because:
/// - Atomic locks ensure exclusive access during mutations
/// - No aliases exist while holding lock
/// - Lock guard ensures proper unlock on panic
///
/// This was ported from the Usearch C++ code and written with Claude.
///
/// Fields
///
/// * `nodes` - Each node's neighbours wrapped in UnsafeCell for interior
///   mutability
/// * `locks` - Lock-free atomic bitset for node locking
/// * `max_neighbours` - Maximum neighbours per node (M or M*2 depending on
///   layer)
/// * `_phantom` - Phantom data to tie the lifetime to T
struct ConstructionGraph<T> {
    nodes: Vec<UnsafeCell<Vec<u32>>>,
    locks: AtomicNodeLocks,
    max_neighbours: Vec<usize>,
    _phantom: PhantomData<T>,
}

// SAFETY: We ensure thread-safety via atomic locks
unsafe impl<T> Sync for ConstructionGraph<T> {}

impl<T> ConstructionGraph<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
{
    /// Create a new construction graph
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes
    /// * `layer_assignments` - Layer for each node
    /// * `m` - M parameter
    fn new(n: usize, layer_assignments: &[u8], m: usize) -> Self {
        let nodes = (0..n).map(|_| UnsafeCell::new(Vec::new())).collect();
        let max_neighbours = layer_assignments
            .iter()
            .map(|&layer| if layer == 0 { m * 2 } else { m })
            .collect();

        Self {
            nodes,
            locks: AtomicNodeLocks::new(n),
            max_neighbours,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get current neighbours for a node
    ///
    /// ### Safety
    ///
    /// Safe to read without lock as we only clone the data.
    /// This is called during search when we need to traverse neighbours.
    ///
    /// ### Params
    ///
    /// * `node_id` - The node idx
    fn get_neighbours(&self, node_id: usize) -> Vec<u32> {
        // SAFETY: We're only reading and cloning, no mutation
        unsafe { (*self.nodes[node_id].get()).clone() }
    }

    /// Write new neighbours for a node
    ///
    /// ### Safety
    ///
    /// Lock guard ensures exclusive access. Safe because:
    /// - Lock is held for entire duration
    /// - No other thread can access this node's data
    /// - Lock guard ensures unlock on panic
    ///
    /// ### Params
    ///
    /// * `node_id` - The node idx
    /// * `neighbours` - Slice of tuples of OrderedFloat and indices of
    ///   neighbours
    fn set_neighbours(&self, node_id: usize, neighbours: &[(OrderedFloat<T>, usize)]) {
        let _guard = self.locks.lock_guard(node_id);

        // SAFETY: Lock guard ensures exclusive access
        let node = unsafe { &mut *self.nodes[node_id].get() };
        node.clear();

        for &(_, neighbour_id) in neighbours.iter().take(self.max_neighbours[node_id]) {
            if neighbour_id != node_id {
                node.push(neighbour_id as u32);
            }
        }

        // Pad with INVALID (u32::MAX)
        while node.len() < self.max_neighbours[node_id] {
            node.push(u32::MAX);
        }
    }

    /// Finalise construction and convert to flat layout
    ///
    /// Consumes the construction graph and produces the flat representation
    /// optimised for subsequent query performance.
    ///
    /// ### Returns
    ///
    /// Tuple of (neighbours_flat, neighbour_offsets)
    fn into_flat(self) -> (Vec<u32>, Vec<usize>) {
        let mut neighbours_flat = Vec::new();
        let mut neighbour_offsets = Vec::new();

        for node in self.nodes {
            neighbour_offsets.push(neighbours_flat.len());
            // SAFETY: We own the graph, no concurrent access possible
            let neighbours = node.into_inner();
            neighbours_flat.extend(neighbours);
        }

        (neighbours_flat, neighbour_offsets)
    }
}

//////////////////////////
// Thread-local buffers //
//////////////////////////

thread_local! {
    static SEARCH_STATE_F32: std::cell::RefCell<SearchState<f32>> = std::cell::RefCell::new(SearchState::new(1000));
    static BUILD_STATE_F32: std::cell::RefCell<SearchState<f32>> = std::cell::RefCell::new(SearchState::new(1000));
    static SEARCH_STATE_F64: std::cell::RefCell<SearchState<f64>> = std::cell::RefCell::new(SearchState::new(1000));
    static BUILD_STATE_F64: std::cell::RefCell<SearchState<f64>> = std::cell::RefCell::new(SearchState::new(1000));
}

/// Trait for accessing type-specific thread-local search state
pub trait HnswState<T> {
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<T>>) -> R;

    fn with_build_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<T>>) -> R;
}

impl HnswState<f32> for HnswIndex<f32> {
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f32>>) -> R,
    {
        SEARCH_STATE_F32.with(f)
    }

    fn with_build_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f32>>) -> R,
    {
        BUILD_STATE_F32.with(f)
    }
}

impl HnswState<f64> for HnswIndex<f64> {
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f64>>) -> R,
    {
        SEARCH_STATE_F64.with(f)
    }

    fn with_build_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<f64>>) -> R,
    {
        BUILD_STATE_F64.with(f)
    }
}

////////////////////
// Main structure //
////////////////////

/// HNSW (Hierarchical Navigable Small World) graph index
///
/// Key optimisations borrowed from the Usearch implementation of HNSW.
///
/// 1. **Atomic Bitset Locks**: Lock-free node locking during construction
/// 2. **Hybrid Priority Queues**: Sorted buffer for small queries, heap for
///    large ones.
/// 3. **Optimised Search**: No repeated sorting, incremental maintenance
///
/// ### Fields
///
/// * `vectors_flat` - Flat structure of vectors for cache locality
/// * `dim` - Feature dimensions
/// * `n` - Number of samples
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `norms` - Precomputed norms for Cosine distance
/// * `layer_assignments` - Which layer each node belongs to (0 = base layer)
/// * `neighbours_flat` - All neighbour PIDs in flat array
/// * `neighbour_offsets` - Starting index of each node's neighbours
/// * `entry_point` - Node ID with highest layer
/// * `max_layer` - Highest layer in the graph
/// * `m` - Number of connections per layer (M parameter)
/// * `ef_construction` - Size of dynamic candidate list during construction
/// * `extend_candidates` - Whether to extend candidate set in heuristic
///   (Algorithm 4)
/// * `keep_pruned` - Whether to keep pruned connections if space available
pub struct HnswIndex<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
{
    // shared ones
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    pub norms: Vec<T>,
    metric: Dist,
    // index-specific ones
    layer_assignments: Vec<u8>,
    neighbours_flat: Vec<u32>,
    neighbour_offsets: Vec<usize>,
    entry_point: u32,
    max_layer: u8,
    m: usize,
    ef_construction: usize,
    extend_candidates: bool,
    keep_pruned: bool,
}

impl<T> VectorDistance<T> for HnswIndex<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
{
    /// Get the flat vectors
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    /// Get the dimensions
    fn dim(&self) -> usize {
        self.dim
    }

    /// The the norms for Cosine distance calculations
    fn norms(&self) -> &[T] {
        &self.norms
    }
}

impl<T> HnswIndex<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
    Self: HnswState<T>,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Build HNSW index
    ///
    /// ### Params
    ///
    /// * `data` - Embedding matrix (rows = samples, cols = features)
    /// * `m` - Number of bidirectional connections per layer (typical: 16-32)
    /// * `ef_construction` - Size of candidate list during construction
    ///   (typical: 100-200)
    /// * `dist_metric` - "euclidean" or "cosine"
    /// * `seed` - Random seed
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// Built HnswIndex ready for querying
    pub fn build(
        data: MatRef<T>,
        m: usize,
        ef_construction: usize,
        dist_metric: &str,
        seed: usize,
        verbose: bool,
    ) -> Self {
        let metric = parse_ann_dist(dist_metric).unwrap_or(Dist::Cosine);

        let (vectors_flat, n, dim) = matrix_to_flat(data);

        if verbose {
            println!(
                "Building HNSW index with {} nodes, M = {}",
                n.separate_with_underscores(),
                m
            );
        }

        let start_total = Instant::now();

        // Compute norms for cosine distance
        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    let end = start + dim;
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

        // Assign layers using exponential distribution
        let ml = T::one() / T::from_usize(m).unwrap().ln();
        let mut rng = SmallRng::seed_from_u64(seed as u64);

        let layer_assignments: Vec<u8> = (0..n)
            .map(|_| {
                let uniform: f64 = rng.random();
                let uniform_t = T::from_f64(uniform).unwrap();
                ((-uniform_t.ln() * ml).floor().to_u8().unwrap()).min(15)
            })
            .collect();

        let max_layer = *layer_assignments.iter().max().unwrap_or(&0);
        let entry_point = layer_assignments
            .iter()
            .position(|&l| l == max_layer)
            .unwrap_or(0) as u32;

        if verbose {
            println!("Max layer: {}, Entry point: {}", max_layer, entry_point);
        }

        // Create construction graph with atomic locks
        let construction_graph = ConstructionGraph::new(n, &layer_assignments, m);

        let mut index = HnswIndex {
            vectors_flat,
            dim,
            n,
            metric,
            norms,
            layer_assignments,
            neighbours_flat: Vec::new(),
            neighbour_offsets: Vec::new(),
            entry_point,
            max_layer,
            m,
            ef_construction,
            extend_candidates: false,
            keep_pruned: true,
        };

        index.build_graph(&construction_graph, verbose);

        // Convert construction graph to flat layout
        let (neighbours_flat, neighbour_offsets) = construction_graph.into_flat();
        index.neighbours_flat = neighbours_flat;
        index.neighbour_offsets = neighbour_offsets;

        let end_total = start_total.elapsed();
        if verbose {
            println!("Total HNSW build time: {:.2?}", end_total);
        }

        index
    }

    /// Build the graph structure layer by layer
    ///
    /// This function uses parallelism under the hood via the
    /// `ConstructionGraph` structure that has atomic locks to avoid race
    /// conditions.
    ///
    /// ### Params
    ///
    /// * `graph` - Construction graph with atomic lock support
    /// * `verbose` - Print progress information
    fn build_graph(&self, graph: &ConstructionGraph<T>, verbose: bool) {
        let nodes_by_layer: Vec<Vec<usize>> = (0..=self.max_layer)
            .map(|layer| {
                (0..self.n)
                    .filter(|&i| self.layer_assignments[i] >= layer)
                    .collect()
            })
            .collect();

        for layer in (0..=self.max_layer).rev() {
            let start = Instant::now();
            let layer_nodes = &nodes_by_layer[layer as usize];

            if verbose {
                println!(
                    "Building layer {} with {} nodes",
                    layer,
                    layer_nodes.len().separate_with_underscores()
                );
            }

            self.build_layer_parallel(layer, layer_nodes, graph);

            if verbose {
                println!("  Layer {} built in {:.2?}", layer, start.elapsed());
            }
        }
    }

    /// Build a layer with true parallel construction (no batching)
    ///
    /// All nodes are processed in parallel with concurrent writes to the
    /// construction graph. No synchronisation barriers needed.
    ///
    /// ### Params
    ///
    /// * `layer` - Layer to build
    /// * `nodes` - Nodes to insert at this layer
    /// * `graph` - Construction graph with concurrent write support
    fn build_layer_parallel(&self, layer: u8, nodes: &[usize], graph: &ConstructionGraph<T>) {
        nodes.par_iter().for_each(|&node| {
            // compute outgoing connections (thread-local state)
            let connections = Self::with_build_state(|state_cell| {
                let mut state = state_cell.borrow_mut();
                self.compute_node_connections(node, layer, graph, &mut state)
            });

            // write node -> neighbours (exclusive access to this node)
            graph.set_neighbours(node, &connections);

            // update neighbours -> node (concurrent, each thread may write to different neighbours)
            for &(_, neighbour_id) in &connections {
                self.prune_and_connect_concurrent(neighbour_id, node, layer, graph);
            }
        });
    }

    /// Compute connections for a node (reads from construction graph)
    ///
    /// ### Params
    ///
    /// * `node` - Node being inserted
    /// * `insert_layer` - Layer to insert at
    /// * `graph` - Construction graph
    /// * `state` - Thread-local search state
    ///
    /// ### Returns
    ///
    /// Selected connections for this node
    fn compute_node_connections(
        &self,
        node: usize,
        insert_layer: u8,
        graph: &ConstructionGraph<T>,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        state.reset(self.n);

        let mut entry_points = vec![(OrderedFloat(T::zero()), self.entry_point as usize)];

        // Descend through upper layers
        for layer in (insert_layer + 1..=self.max_layer).rev() {
            state.reset(self.n);
            entry_points = self.search_layer(node, layer, &entry_points, 1, graph, state);
        }

        state.reset(self.n);
        // The /2 is needed to avoid aggressive pruning of the base layer
        let ef = if insert_layer == 0 {
            self.ef_construction / 2
        } else {
            self.ef_construction
        };

        let candidates = self.search_layer(node, insert_layer, &entry_points, ef, graph, state);

        self.select_heuristic(node, &candidates, insert_layer, state)
    }

    /// Prune and connect with concurrent access (race-condition free)
    ///
    /// Atomically updates a neighbour's connection list by adding a new node,
    /// then pruning using the heuristic selection algorithm.
    ///
    /// ### Params
    ///
    /// * `neighbour_id` - Existing neighbour to update
    /// * `new_node` - New node to add as potential neighbour
    /// * `layer` - Current layer
    /// * `graph` - Construction graph with atomic locks
    fn prune_and_connect_concurrent(
        &self,
        neighbour_id: usize,
        new_node: usize,
        layer: u8,
        graph: &ConstructionGraph<T>,
    ) {
        // acquire atomic lock for exclusive access
        let _guard = graph.locks.lock_guard(neighbour_id);

        // SAFETY: Lock guard ensures exclusive access to this node
        // unsafe does feel unsafe...
        let node = unsafe { &mut *graph.nodes[neighbour_id].get() };
        let current = node.clone();

        let mut candidates: Vec<(OrderedFloat<T>, usize)> = Vec::with_capacity(current.len() + 1);

        // add all valid existing neighbours with their distances
        for &neighbour in current.iter() {
            if neighbour == u32::MAX {
                break;
            }

            let neighbour_node_id = neighbour as usize;
            let dist = OrderedFloat(match self.metric {
                Dist::Euclidean => self.euclidean_distance(neighbour_id, neighbour_node_id),
                Dist::Cosine => self.cosine_distance(neighbour_id, neighbour_node_id),
            });

            candidates.push((dist, neighbour_node_id));
        }

        // add the new node as candidate
        if new_node != neighbour_id {
            let dist = OrderedFloat(match self.metric {
                Dist::Euclidean => self.euclidean_distance(neighbour_id, new_node),
                Dist::Cosine => self.cosine_distance(neighbour_id, new_node),
            });

            candidates.push((dist, new_node));
        }

        // sort candidates by distance
        candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        // apply heuristic selection
        let selected = Self::with_build_state(|state_cell| {
            let mut state = state_cell.borrow_mut();
            self.select_heuristic(neighbour_id, &candidates, layer, &mut state)
        });

        // write back final selection
        // SAFETY: Still holding lock guard
        let max_neighbours = graph.max_neighbours[neighbour_id];
        node.clear();

        for &(_, node_id) in selected.iter().take(max_neighbours) {
            if node_id != neighbour_id {
                node.push(node_id as u32);
            }
        }

        // Pad with INVALID markers
        while node.len() < max_neighbours {
            node.push(u32::MAX);
        }

        // Lock automatically released when _guard drops at this point
    }

    /// Search layer using construction graph
    ///
    /// ### Params
    ///
    /// * `query_node` - Node we're searching from
    /// * `layer` - Current layer
    /// * `entry_points` - Starting points
    /// * `ef` - Candidate list size
    /// * `graph` - Construction graph
    /// * `state` - Search state
    ///
    /// ### Returns
    ///
    /// Sorted list of nearest neighbours
    fn search_layer(
        &self,
        query_node: usize,
        layer: u8,
        entry_points: &[(OrderedFloat<T>, usize)],
        ef: usize,
        graph: &ConstructionGraph<T>,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        state.working_sorted.clear();
        state.candidates.clear();

        // initialise with entry points
        for &(dist, pid) in entry_points {
            if !state.is_visited(pid) {
                state.mark_visited(pid);
                state.candidates.push(Reverse((dist, pid)));
                state.working_sorted.insert((dist, pid), ef);
            }
        }

        // track furthest distance in working set
        let mut furthest_dist = if state.working_sorted.len() >= ef {
            state
                .working_sorted
                .top()
                .map(|(d, _)| *d)
                .unwrap_or(OrderedFloat(T::infinity()))
        } else {
            OrderedFloat(T::infinity())
        };

        // Main traversal loop
        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            // Early termination
            if current_dist > furthest_dist {
                break;
            }

            let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };
            let neighbours = graph.get_neighbours(current_id);

            for i in 0..max_neighbours.min(neighbours.len()) {
                let neighbour = neighbours[i];
                if neighbour == u32::MAX {
                    break;
                }

                let neighbour_id = neighbour as usize;

                if self.layer_assignments[neighbour_id] < layer {
                    continue;
                }

                if state.is_visited(neighbour_id) {
                    continue;
                }

                state.mark_visited(neighbour_id);

                let dist = OrderedFloat(match self.metric {
                    Dist::Euclidean => self.euclidean_distance(query_node, neighbour_id),
                    Dist::Cosine => self.cosine_distance(query_node, neighbour_id),
                });

                if dist < furthest_dist || state.working_sorted.len() < ef {
                    state.candidates.push(Reverse((dist, neighbour_id)));

                    if state.working_sorted.insert((dist, neighbour_id), ef)
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

        state.working_sorted.data().to_vec()
    }

    /// Select neighbours using heuristic (Algorithm 4 from paper)
    ///
    /// The heuristic filters candidates to avoid clustered connections. A
    /// candidate is selected if it's closer to the query node than to any
    /// already-selected neighbour.
    ///
    /// ### Params
    ///
    /// * `node` - Node being inserted
    /// * `candidates` - Initial candidate neighbours (sorted by distance)
    /// * `layer` - Current layer
    /// * `state` - SearchState providing scratch buffers
    ///
    /// ### Returns
    ///
    /// Filtered list of neighbours (at most M or M*2)
    fn select_heuristic(
        &self,
        node: usize,
        candidates: &[(OrderedFloat<T>, usize)],
        layer: u8,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        state.scratch_working.clear();
        state.scratch_discarded.clear();

        // Increment visit generation for local operation
        state.visit_id = state.visit_id.wrapping_add(1);
        if state.visit_id == 0 {
            state.visited.fill(0);
            state.visit_id = 1;
        }

        // Filter candidates
        for &candidate in candidates {
            if candidate.1 != node && !state.is_visited(candidate.1) {
                state.scratch_working.push(candidate);
                state.mark_visited(candidate.1);
            }
        }

        // Extend candidates if enabled
        if self.extend_candidates {
            let max_check = if layer == 0 { self.m * 2 } else { self.m };

            for &(_, cand_id) in candidates {
                if cand_id == node {
                    continue;
                }

                let offset = self.neighbour_offsets[cand_id];

                for i in 0..max_check {
                    let neighbour = self.neighbours_flat[offset + i];
                    if neighbour == u32::MAX {
                        break;
                    }

                    let neighbour_id = neighbour as usize;
                    if neighbour_id == node || state.is_visited(neighbour_id) {
                        continue;
                    }

                    state.mark_visited(neighbour_id);

                    let dist = OrderedFloat(match self.metric {
                        Dist::Euclidean => self.euclidean_distance(node, neighbour_id),
                        Dist::Cosine => self.cosine_distance(node, neighbour_id),
                    });

                    state.scratch_working.push((dist, neighbour_id));
                }
            }

            state.scratch_working.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        }

        // Heuristic selection
        let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };
        let mut result = Vec::with_capacity(max_neighbours);

        for candidate in state.scratch_working.iter() {
            if result.len() >= max_neighbours {
                break;
            }

            let (cand_dist, cand_id) = *candidate;

            if cand_id == node {
                continue;
            }

            let mut closer_to_query = true;
            for &(_, result_id) in &result {
                let dist_to_result = OrderedFloat(match self.metric {
                    Dist::Euclidean => self.euclidean_distance(cand_id, result_id),
                    Dist::Cosine => self.cosine_distance(cand_id, result_id),
                });

                if dist_to_result < cand_dist {
                    closer_to_query = false;
                    break;
                }
            }

            if closer_to_query {
                result.push((cand_dist, cand_id));
            } else if self.keep_pruned {
                state.scratch_discarded.push((cand_dist, cand_id));
            }
        }

        // Add back closest pruned connections if space available
        if self.keep_pruned && !state.scratch_discarded.is_empty() {
            state
                .scratch_discarded
                .sort_unstable_by(|a, b| a.0.cmp(&b.0));

            for &candidate in state.scratch_discarded.iter() {
                if result.len() >= max_neighbours {
                    break;
                }
                result.push(candidate);
            }
        }

        result
    }

    ///////////
    // Query //
    ///////////

    /// Query the index for k nearest neighbours
    ///
    /// ### Params
    ///
    /// * `query` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Size of candidate list (higher = better recall, slower)
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances)
    #[inline]
    pub fn query(&self, query: &[T], k: usize, ef_search: usize) -> (Vec<usize>, Vec<T>) {
        assert_eq!(query.len(), self.dim);

        Self::with_search_state(|state_cell| {
            let mut state = state_cell.borrow_mut();
            state.reset(self.n);

            let query_norm = if self.metric == Dist::Cosine {
                query
                    .iter()
                    .map(|x| *x * *x)
                    .fold(T::zero(), |a, b| a + b)
                    .sqrt()
            } else {
                T::one()
            };

            let entry_dist =
                self.compute_query_distance(query, self.entry_point as usize, query_norm);
            let mut entry_points = vec![(OrderedFloat(entry_dist), self.entry_point as usize)];

            for layer in (1..=self.max_layer).rev() {
                state.reset(self.n);
                entry_points =
                    self.search_layer_query(query, query_norm, layer, &entry_points, 1, &mut state);
            }

            state.reset(self.n);
            let mut candidates = self.search_layer_query(
                query,
                query_norm,
                0,
                &entry_points,
                ef_search.max(k),
                &mut state,
            );

            candidates.truncate(k);

            let (indices, distances): (Vec<usize>, Vec<T>) = candidates
                .into_iter()
                .map(|(OrderedFloat(d), id)| (id, d))
                .unzip();

            (indices, distances)
        })
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
    /// * `ef_search` - Size of candidate list (higher = better recall, slower)
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        ef_search: usize,
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
    /// * `ef_search` - Size of candidate list (higher = better recall, slower)
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
        ef_search: usize,
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

    /// Search layer for query vector
    ///
    /// ### Params
    ///
    /// * `query` - Query vector slice
    /// * `query_norm` - Precomputed norm of query (for cosine)
    /// * `layer` - Current layer
    /// * `entry_points` - Starting points for search
    /// * `ef` - Size of candidate list
    /// * `state` - Mutable search state
    ///
    /// ### Returns
    ///
    /// List of (distance, node_id) pairs, sorted by distance
    #[inline]
    fn search_layer_query(
        &self,
        query: &[T],
        query_norm: T,
        layer: u8,
        entry_points: &[(OrderedFloat<T>, usize)],
        ef: usize,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        state.working_sorted.clear();
        state.candidates.clear();

        for &(dist, pid) in entry_points {
            if !state.is_visited(pid) {
                state.mark_visited(pid);
                state.candidates.push(Reverse((dist, pid)));
                state.working_sorted.insert((dist, pid), ef);
            }
        }

        let mut furthest_dist = if state.working_sorted.len() >= ef {
            state
                .working_sorted
                .top()
                .map(|(d, _)| *d)
                .unwrap_or(OrderedFloat(T::infinity()))
        } else {
            OrderedFloat(T::infinity())
        };

        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist > furthest_dist {
                break;
            }

            let node_layer = self.layer_assignments[current_id];
            let max_neighbours = if node_layer == 0 { self.m * 2 } else { self.m };
            let offset = self.neighbour_offsets[current_id];

            for i in 0..max_neighbours {
                let neighbour = self.neighbours_flat[offset + i];
                if neighbour == u32::MAX {
                    break;
                }

                let neighbour_id = neighbour as usize;

                if self.layer_assignments[neighbour_id] < layer {
                    continue;
                }

                if state.is_visited(neighbour_id) {
                    continue;
                }

                state.mark_visited(neighbour_id);

                let dist =
                    OrderedFloat(self.compute_query_distance(query, neighbour_id, query_norm));

                if dist < furthest_dist || state.working_sorted.len() < ef {
                    state.candidates.push(Reverse((dist, neighbour_id)));

                    if state.working_sorted.insert((dist, neighbour_id), ef)
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

        state.working_sorted.data().to_vec()
    }

    /// Compute distance between query and database vector
    ///
    /// ### Params
    ///
    /// * `query` - Query vector slice
    /// * `idx` - Database vector index
    /// * `query_norm` - Precomputed norm of query (for cosine)
    ///
    /// ### Returns
    ///
    /// Distance value (Euclidean squared or Cosine distance)
    #[inline(always)]
    fn compute_query_distance(&self, query: &[T], idx: usize, query_norm: T) -> T {
        match self.metric {
            Dist::Euclidean => self.euclidean_distance_to_query(idx, query),
            Dist::Cosine => self.cosine_distance_to_query(idx, query, query_norm),
        }
    }
}

//////////////////////
// Validation trait //
//////////////////////

impl<T> KnnValidation<T> for HnswIndex<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum,
    Self: HnswState<T>,
{
    /// Internal querying function
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        self.query(query_vec, k, 200)
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
    fn test_hnsw_build_euclidean() {
        let mat = create_simple_matrix();
        let _ = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);
    }

    #[test]
    fn test_hnsw_build_cosine() {
        let mat = create_simple_matrix();
        let _ = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "cosine", 42, false);
    }

    #[test]
    fn test_hnsw_query_finds_self() {
        let mat = create_simple_matrix();
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1, 50);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hnsw_query_euclidean() {
        let mat = create_simple_matrix();
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, 50);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_hnsw_query_cosine() {
        let mat = create_simple_matrix();
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "cosine", 42, false);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3, 50);

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hnsw_parallel_build() {
        let n = 2000;
        let dim = 10;
        let data: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 200, "euclidean", 42, false);

        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5, 50);

        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_hnsw_thread_safety() {
        let mat = create_simple_matrix();
        let index = Arc::new(HnswIndex::<f32>::build(
            mat.as_ref(),
            16,
            100,
            "euclidean",
            42,
            false,
        ));

        let mut handles = vec![];
        for _ in 0..4 {
            let index_clone = Arc::clone(&index);
            let handle = thread::spawn(move || {
                let query = vec![0.5, 0.5, 0.0];
                let (indices, _) = index_clone.query(&query, 3, 50);
                assert_eq!(indices.len(), 3);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_hnsw_reproducibility() {
        let mat = create_simple_matrix();

        let index1 = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);
        let index2 = HnswIndex::<f32>::build(mat.as_ref(), 16, 100, "euclidean", 42, false);

        let query = vec![0.5, 0.5, 0.0];
        let (indices1, _) = index1.query(&query, 3, 50);
        let (indices2, _) = index2.query(&query, 3, 50);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_hnsw_recall() {
        let n = 20;
        let dim = 3;
        let mut data = Vec::with_capacity(n * dim);

        data.extend_from_slice(&[0.0, 0.0, 0.0]);

        for i in 1..n {
            let dist = i as f32 * 0.1;
            data.extend_from_slice(&[dist, 0.0, 0.0]);
        }

        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);
        let index = HnswIndex::<f32>::build(mat.as_ref(), 16, 200, "euclidean", 42, false);

        let query = vec![0.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 5, 100);

        assert_eq!(indices[0], 0);

        let expected_neighbours: Vec<usize> = (0..5).collect();
        let found_correct = indices
            .iter()
            .filter(|&&idx| expected_neighbours.contains(&idx))
            .count();

        assert!(found_correct >= 4);
    }

    #[test]
    fn test_hnsw_small_m_layer_bug() {
        // This test catches a bug where nodes assigned to higher layers have
        // fewer neighbours stored (m) but layer 0 search incorrectly tries to
        // read 2*m neighbours, causing an index out of bounds panic.
        //
        // The bug occurs when:
        // 1. Node is assigned to layer > 0, so stores m neighbours
        // 2. During layer 0 search, code calculates max_neighbours = m * 2
        // 3. Attempts to read beyond actual storage

        let n = 929; // Exact size that triggered the bug
        let dim = 15;
        let m = 16; // Small m that makes the bug more likely
        let ef_construction = 200;

        // Create dataset similar to PCA embeddings
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                data.push(((i * dim + j) as f32).sin() * 0.1);
            }
        }
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        // Build with parameters that triggered the original bug
        let index = HnswIndex::<f32>::build(mat.as_ref(), m, ef_construction, "cosine", 42, false);

        // Query with ef_search = 100 (also from original failing case)
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let (indices, distances) = index.query(&query, 15, 100);

        assert_eq!(indices.len(), 15);
        assert_eq!(distances.len(), 15);

        // Verify results are valid
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1], "Distances not sorted");
        }

        for &idx in &indices {
            assert!(idx < n, "Invalid index returned");
        }
    }

    #[test]
    fn test_hnsw_varying_m_values() {
        // Test multiple small m values to ensure robustness
        let n = 500;
        let dim = 10;

        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.01).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        for m in [8, 12, 16, 20] {
            let index = HnswIndex::<f32>::build(mat.as_ref(), m, 200, "euclidean", 42, false);

            let query: Vec<f32> = (0..dim).map(|_| 0.5).collect();
            let (indices, _) = index.query(&query, 10, 50);

            assert_eq!(indices.len(), 10, "Failed with m = {}", m);
        }
    }

    // This was needed to reproduce the nasty bug
    // #[test]
    // #[should_panic(expected = "index out of bounds")]
    // fn test_hnsw_query_layer_bug() {
    //     // Exact reproduction of R failure case
    //     let n = 929;
    //     let dim = 15;
    //     let m = 16;
    //     let ef_construction = 200;

    //     // Create data matching PCA output
    //     let mut data = Vec::with_capacity(n * dim);
    //     for i in 0..n {
    //         for j in 0..dim {
    //             data.push(((i * 37 + j * 13) as f32).sin() * 0.5);
    //         }
    //     }
    //     let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

    //     // Build with EXACT parameters from failing R call
    //     let index = HnswIndex::<f32>::build(
    //         mat.as_ref(),
    //         m,
    //         ef_construction,
    //         "cosine", // Default from R
    //         42,
    //         false,
    //     );

    //     // Query ALL nodes with k=16 (k=15 + 1 for self-removal)
    //     // This ensures we traverse all parts of the graph
    //     for i in 0..n {
    //         let query: Vec<f32> = mat.row(i).iter().copied().collect();
    //         let _ = index.query(&query, 16, 100);
    //     }
    // }
}
