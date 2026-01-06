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
/// Uses atomic u64 chunks to allow concurrent lock operations without blocking.
/// Each bit represents one node's lock state.
///
/// ### Fields
///
/// * `bits` - Vector of atomic u64 chunks, each storing 64 lock bits
#[derive(Debug)]
pub struct AtomicNodeLocks {
    bits: Vec<AtomicU64>,
}

impl AtomicNodeLocks {
    /// Create a new lock bitset with capacity for n nodes
    ///
    /// Allocates enough u64 chunks to store one bit per node, rounding up
    /// to the nearest 64-node boundary.
    ///
    /// ### Params
    ///
    /// * `capacity` - Number of nodes to support
    ///
    /// ### Returns
    ///
    /// Initialised lock bitset with all locks released
    pub fn new(capacity: usize) -> Self {
        let num_slots = capacity.div_ceil(64);
        Self {
            bits: (0..num_slots).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Attempt to acquire a lock on a node
    ///
    /// Returns immediately without blocking. Uses compare-and-swap to
    /// atomically check and set the lock bit.
    ///
    /// ### Params
    ///
    /// * `idx` - Node index to lock
    ///
    /// ### Returns
    ///
    /// `true` if lock was already held, `false` if successfully acquired
    #[inline(always)]
    pub fn try_lock(&self, idx: usize) -> bool {
        let slot = idx / 64;
        let bit = 1u64 << (idx % 64);
        let prev = self.bits[slot].fetch_or(bit, Ordering::Acquire);
        (prev & bit) != 0
    }

    /// Acquire a lock on a node, spinning until successful
    ///
    /// Repeatedly calls try_lock until the lock is acquired. Uses spin_loop
    /// hint to reduce CPU contention.
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

    /// Release a lock on a node
    ///
    /// Atomically clears the lock bit, making it available for other threads.
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

    /// Acquire a lock and return an RAII guard
    ///
    /// The lock is automatically released when the guard is dropped.
    ///
    /// ### Params
    ///
    /// * `idx` - Node index to lock
    ///
    /// ### Returns
    ///
    /// Guard that releases the lock on drop
    #[inline(always)]
    pub fn lock_guard(&self, idx: usize) -> NodeLockGuard<'_> {
        self.lock(idx);
        NodeLockGuard { locks: self, idx }
    }
}

/// RAII guard for automatic lock release
///
/// Holds a reference to the lock bitset and the locked node index.
/// Automatically releases the lock when dropped.
///
/// ### Fields
///
/// * `locks` - Reference to the parent lock bitset
/// * `idx` - Index of the locked node
pub struct NodeLockGuard<'a> {
    locks: &'a AtomicNodeLocks,
    idx: usize,
}

impl<'a> Drop for NodeLockGuard<'a> {
    /// Drop method
    fn drop(&mut self) {
        self.locks.unlock(self.idx);
    }
}

/// Search state for HNSW queries and construction
///
/// Maintains visited tracking and candidate management for graph traversal.
/// Reused across queries to amortise allocation costs.
///
/// ### Fields
///
/// * `visited` - Per-node visit tracking using incrementing IDs
/// * `visit_id` - Current visit epoch (wraps around, triggers reset)
/// * `candidates` - Min-heap of nodes to explore, ordered by distance
/// * `working_sorted` - Sorted buffer of current best candidates
/// * `scratch_working` - Temporary storage for heuristic selection
/// * `scratch_discarded` - Temporary storage for pruned candidates
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
    /// Create a new search state with initial capacity
    ///
    /// Allocates buffers sized for the given capacity to avoid reallocations
    /// during typical queries.
    ///
    /// ### Params
    ///
    /// * `capacity` - Initial capacity for internal buffers
    ///
    /// ### Returns
    ///
    /// Initialised search state ready for use
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

    /// Reset state for a new query
    ///
    /// Clears all buffers and advances the visit epoch. If the epoch wraps
    /// around to zero, performs a full reset of the visited array.
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes in the graph (for capacity adjustment)
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

    /// Check if a node has been visited in the current query
    ///
    /// ### Params
    ///
    /// * `node` - Node index to check
    ///
    /// ### Returns
    ///
    /// `true` if node was already visited, `false` otherwise
    #[inline(always)]
    fn is_visited(&self, node: usize) -> bool {
        self.visited[node] == self.visit_id
    }

    /// Mark a node as visited in the current query
    ///
    /// ### Params
    ///
    /// * `node` - Node index to mark
    #[inline(always)]
    fn mark_visited(&mut self, node: usize) {
        self.visited[node] = self.visit_id;
    }
}

impl<T: Ord> Default for SortedBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Construction-time neighbour storage
///
/// ### Fields
///
/// * `nodes` - For each node, vector of neighbour lists (one per layer)
/// * `locks` - Lock bitset for thread-safe updates
/// * `node_levels` - Maximum layer each node appears in
/// * `m` - Base connectivity parameter
/// * `_phantom` - Phantom data for type parameter
struct ConstructionGraph<T> {
    nodes: Vec<UnsafeCell<Vec<Vec<u32>>>>,
    locks: AtomicNodeLocks,
    node_levels: Vec<u8>,
    m: usize,
    _phantom: PhantomData<T>,
}

unsafe impl<T> Sync for ConstructionGraph<T> {}

impl<T> ConstructionGraph<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
{
    /// Create a new construction graph with proper per-layer storage
    ///
    /// Pre-allocates neighbour vectors for each layer each node appears in,
    /// with layer 0 having 2*M capacity and upper layers having M capacity.
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes
    /// * `layer_assignments` - Maximum layer for each node
    /// * `m` - Base connectivity parameter
    ///
    /// ### Returns
    ///
    /// Initialised construction graph with empty neighbour lists
    fn new(n: usize, layer_assignments: &[u8], m: usize) -> Self {
        let nodes = (0..n)
            .map(|i| {
                let level = layer_assignments[i] as usize;
                // pre-allocate vectors for each layer this node appears in
                // layer 0 gets M*2, layers 1+ get M
                let mut layers = Vec::with_capacity(level + 1);
                for l in 0..=level {
                    let capacity = if l == 0 { m * 2 } else { m };
                    layers.push(Vec::with_capacity(capacity));
                }
                UnsafeCell::new(layers)
            })
            .collect();

        Self {
            nodes,
            locks: AtomicNodeLocks::new(n),
            node_levels: layer_assignments.to_vec(),
            m,
            _phantom: PhantomData,
        }
    }

    /// Get the maximum layer a node appears in
    ///
    /// ### Params
    ///
    /// * `node_id` - Node index
    ///
    /// ### Returns
    ///
    /// Highest layer this node exists in (0 = base layer only)
    #[inline]
    fn node_level(&self, node_id: usize) -> u8 {
        self.node_levels[node_id]
    }

    /// Get neighbours for a node at a specific layer
    ///
    /// Returns empty vector if the node doesn't exist at the requested layer.
    ///
    /// ### Params
    ///
    /// * `node_id` - Node index
    /// * `layer` - Layer to query
    ///
    /// ### Returns
    ///
    /// Vector of neighbour indices at this layer
    fn get_neighbours(&self, node_id: usize, layer: u8) -> Vec<u32> {
        let node_level = self.node_levels[node_id];
        if layer > node_level {
            return Vec::new();
        }
        // SAFETY: We're only reading and cloning
        unsafe {
            let layers = &*self.nodes[node_id].get();
            layers[layer as usize].clone()
        }
    }

    /// Set neighbours for a node at a specific layer
    ///
    /// Replaces the entire neighbour list for this layer, respecting the
    /// maximum neighbour count (2*M for layer 0, M for upper layers).
    ///
    /// ### Params
    ///
    /// * `node_id` - Node to update
    /// * `layer` - Layer to update
    /// * `neighbours` - New neighbour list (distance, id) pairs
    fn set_neighbours(&self, node_id: usize, layer: u8, neighbours: &[(OrderedFloat<T>, usize)]) {
        let node_level = self.node_levels[node_id];
        if layer > node_level {
            return;
        }

        let _guard = self.locks.lock_guard(node_id);

        let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };

        // SAFETY: Lock guard ensures exclusive access
        let layers = unsafe { &mut *self.nodes[node_id].get() };
        let layer_neighbours = &mut layers[layer as usize];
        layer_neighbours.clear();

        for &(_, neighbour_id) in neighbours.iter().take(max_neighbours) {
            if neighbour_id != node_id {
                layer_neighbours.push(neighbour_id as u32);
            }
        }
    }

    /// Add a single neighbour with pruning if needed
    ///
    /// If the neighbour list is full, applies the heuristic pruning strategy
    /// to maintain diversity whilst respecting the maximum neighbour count.
    ///
    /// ### Params
    ///
    /// * `node_id` - Node to update
    /// * `layer` - Layer to update
    /// * `new_neighbour` - Neighbour to add
    /// * `distance_fn` - Function to compute distances between nodes
    fn add_neighbour_with_pruning<F>(
        &self,
        node_id: usize,
        layer: u8,
        new_neighbour: usize,
        distance_fn: F,
    ) where
        F: Fn(usize, usize) -> T,
    {
        let node_level = self.node_levels[node_id];
        if layer > node_level {
            return;
        }

        let _guard = self.locks.lock_guard(node_id);

        let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };

        // SAFETY: Lock guard ensures exclusive access
        let layers = unsafe { &mut *self.nodes[node_id].get() };
        let layer_neighbours = &mut layers[layer as usize];

        // Check if already present
        if layer_neighbours
            .iter()
            .any(|&n| n as usize == new_neighbour)
        {
            return;
        }

        // If there's room, just add
        if layer_neighbours.len() < max_neighbours {
            layer_neighbours.push(new_neighbour as u32);
            return;
        }

        // Need to prune - collect all candidates with distances
        let mut candidates: Vec<(OrderedFloat<T>, usize)> = layer_neighbours
            .iter()
            .map(|&n| {
                let n = n as usize;
                (OrderedFloat(distance_fn(node_id, n)), n)
            })
            .collect();

        candidates.push((
            OrderedFloat(distance_fn(node_id, new_neighbour)),
            new_neighbour,
        ));

        // Sort by distance
        candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        // Apply simple heuristic: keep closest, but prefer diversity
        let mut selected = Vec::with_capacity(max_neighbours);

        for (dist, cand_id) in &candidates {
            if selected.len() >= max_neighbours {
                break;
            }

            // Check if this candidate is closer to query than to any selected
            let dominated = selected.iter().any(|&sel_id| {
                let dist_to_selected = OrderedFloat(distance_fn(*cand_id, sel_id));
                dist_to_selected < *dist
            });

            if !dominated || selected.len() < max_neighbours / 2 {
                selected.push(*cand_id);
            }
        }

        // Ensure we have at least some neighbours
        if selected.len() < max_neighbours && candidates.len() > selected.len() {
            for (_, cand_id) in &candidates {
                if selected.len() >= max_neighbours {
                    break;
                }
                if !selected.contains(cand_id) {
                    selected.push(*cand_id);
                }
            }
        }

        layer_neighbours.clear();
        for id in selected {
            layer_neighbours.push(id as u32);
        }
    }

    /// Convert to flat layout for queries
    ///
    /// Transforms the per-layer storage into a flattened array for better
    /// cache performance during queries. Each node's layers are stored
    /// contiguously with padding. Layout:
    /// [node0_layer0(M*2), node0_layer1(M), ..., node1_layer0(M*2), ...]
    ///
    /// ### Params
    ///
    /// * `m` - Connectivity parameter for size calculation
    ///
    /// ### Returns
    ///
    /// Tuple of (flat neighbours, offsets, level assignments)
    fn into_flat(self, m: usize) -> (Vec<u32>, Vec<usize>, Vec<u8>) {
        let n = self.nodes.len();
        let mut neighbours_flat = Vec::new();
        let mut neighbour_offsets = Vec::with_capacity(n);
        let node_levels = self.node_levels.clone();

        for (node_id, node_cell) in self.nodes.into_iter().enumerate() {
            neighbour_offsets.push(neighbours_flat.len());

            let layers = node_cell.into_inner();
            let node_level = node_levels[node_id];

            // Write each layer's neighbours with padding
            for layer in 0..=node_level {
                let max_neighbours = if layer == 0 { m * 2 } else { m };
                let layer_neighbours = if (layer as usize) < layers.len() {
                    &layers[layer as usize]
                } else {
                    &Vec::new()
                };

                // Write actual neighbours
                for &n in layer_neighbours {
                    neighbours_flat.push(n);
                }

                // Pad with INVALID
                for _ in layer_neighbours.len()..max_neighbours {
                    neighbours_flat.push(u32::MAX);
                }
            }
        }

        (neighbours_flat, neighbour_offsets, node_levels)
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
/// Implements the HNSW algorithm with multi-layer neighbour storage. Each node
/// maintains separate neighbour lists for each layer it appears in, enabling
/// efficient hierarchical search.
///
/// ### Fields
///
/// * `vectors_flat` - Flattened vector data for cache locality
/// * `dim` - Dimensionality of vectors
/// * `n` - Number of vectors
/// * `norms` - Pre-computed norms for Cosine distance (empty for Euclidean)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `layer_assignments` - Maximum layer each node appears in
/// * `neighbours_flat` - Flattened neighbour storage across all layers
/// * `neighbour_offsets` - Starting offset for each node's neighbours
/// * `entry_point` - Starting node for queries (highest-layer node)
/// * `max_layer` - Highest layer in the graph
/// * `m` - Base connectivity parameter
/// * `ef_construction` - Size of dynamic candidate list during construction
/// * `extend_candidates` - Whether to extend candidate pool (unused)
/// * `keep_pruned` - Whether to keep pruned candidates
pub struct HnswIndex<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
{
    // Vector data
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    pub norms: Vec<T>,
    metric: Dist,
    // HNSW
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

impl<T> HnswIndex<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
    Self: HnswState<T>,
{
    /// Compute the offset in neighbours_flat for a specific node and layer
    ///
    /// Calculates where in the flat array this node's layer begins, accounting
    /// for layer 0 having 2*M slots and upper layers having M slots each.
    ///
    /// ### Params
    ///
    /// * `node_id` - Node index
    /// * `layer` - Layer number
    ///
    /// ### Returns
    ///
    /// Offset into neighbours_flat
    #[inline]
    fn neighbours_offset(&self, node_id: usize, layer: u8) -> usize {
        let base = self.neighbour_offsets[node_id];
        if layer == 0 {
            base
        } else {
            // Skip layer 0 (M*2) plus previous upper layers (M each)
            base + self.m * 2 + self.m * (layer as usize - 1)
        }
    }

    /// Get the maximum number of neighbours for a layer
    ///
    /// ### Params
    ///
    /// * `layer` - Layer number
    ///
    /// ### Returns
    ///
    /// 2*M for layer 0, M for upper layers
    #[inline]
    fn max_neighbours_for_layer(&self, layer: u8) -> usize {
        if layer == 0 {
            self.m * 2
        } else {
            self.m
        }
    }

    /// Get neighbours for a node at a specific layer (for queries)
    ///
    /// Returns empty slice if the node doesn't exist at this layer.
    ///
    /// ### Params
    ///
    /// * `node_id` - Node index
    /// * `layer` - Layer to query
    ///
    /// ### Returns
    ///
    /// Slice of neighbour indices (may contain u32::MAX padding)
    #[inline]
    fn get_neighbours_at_layer(&self, node_id: usize, layer: u8) -> &[u32] {
        let node_level = self.layer_assignments[node_id];
        if layer > node_level {
            return &[];
        }

        let offset = self.neighbours_offset(node_id, layer);
        let count = self.max_neighbours_for_layer(layer);

        // Safety: bounds are guaranteed by construction
        &self.neighbours_flat[offset..offset + count]
    }

    //////////////////////
    // Index generation //
    //////////////////////

    /// Construct a new HNSW index
    ///
    /// Builds the hierarchical graph structure layer by layer from top to
    /// bottom. Assigns layers using exponential distribution, then constructs
    /// connections using greedy search and heuristic pruning.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (rows = samples, columns = dimensions)
    /// * `m` - Base connectivity parameter (neighbours per layer)
    /// * `ef_construction` - Size of dynamic candidate list during construction
    /// * `dist_metric` - Distance metric ("euclidean" or "cosine")
    /// * `seed` - Random seed for layer assignment
    /// * `verbose` - Whether to print progress information
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
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
            for l in 0..=max_layer {
                let count = layer_assignments.iter().filter(|&&x| x >= l).count();
                println!("  Layer {}: {} nodes", l, count.separate_with_underscores());
            }
        }

        // Create construction graph with proper per-layer storage
        let construction_graph = ConstructionGraph::new(n, &layer_assignments, m);

        let mut index = HnswIndex {
            vectors_flat,
            dim,
            n,
            metric,
            norms,
            layer_assignments: layer_assignments.clone(),
            neighbours_flat: Vec::new(),
            neighbour_offsets: Vec::new(),
            entry_point,
            max_layer,
            m,
            ef_construction,
            extend_candidates: false,
            keep_pruned: true,
        };

        // Build the graph layer by layer, from TOP to BOTTOM
        index.build_graph(&construction_graph, verbose);

        // Convert construction graph to flat layout
        let (neighbours_flat, neighbour_offsets, _) = construction_graph.into_flat(m);
        index.neighbours_flat = neighbours_flat;
        index.neighbour_offsets = neighbour_offsets;

        if verbose {
            println!("Total HNSW build time: {:.2?}", start_total.elapsed());
        }

        index
    }

    /// Build the graph structure layer by layer (top to bottom)
    ///
    /// Constructs upper layers first to create "highways" that lower layers
    /// can use during their construction, improving connectivity.
    ///
    /// ### Params
    ///
    /// * `graph` - Construction graph to populate
    /// * `verbose` - Whether to print progress
    fn build_graph(&self, graph: &ConstructionGraph<T>, verbose: bool) {
        for layer in (0..=self.max_layer).rev() {
            let start = Instant::now();

            let layer_nodes: Vec<usize> = (0..self.n)
                .filter(|&i| self.layer_assignments[i] >= layer)
                .collect();

            if verbose {
                println!(
                    "Building layer {} with {} nodes",
                    layer,
                    layer_nodes.len().separate_with_underscores()
                );
            }

            self.build_layer_parallel(layer, &layer_nodes, graph);

            if verbose {
                println!("  Layer {} built in {:.2?}", layer, start.elapsed());
            }
        }
    }

    /// Build a single layer with parallel processing
    ///
    /// Processes all nodes at a layer in parallel, computing their connections
    /// and forming bidirectional links with pruning.
    ///
    /// ### Params
    ///
    /// * `layer` - Layer to build
    /// * `nodes` - Nodes present at this layer
    /// * `graph` - Construction graph to update
    fn build_layer_parallel(&self, layer: u8, nodes: &[usize], graph: &ConstructionGraph<T>) {
        nodes.par_iter().for_each(|&node| {
            let connections = Self::with_build_state(|state_cell| {
                let mut state = state_cell.borrow_mut();
                self.compute_node_connections_for_layer(node, layer, graph, &mut state)
            });

            graph.set_neighbours(node, layer, &connections);

            let distance_fn = |a: usize, b: usize| -> T {
                match self.metric {
                    Dist::Euclidean => self.euclidean_distance(a, b),
                    Dist::Cosine => self.cosine_distance(a, b),
                }
            };

            for &(_, neighbour_id) in &connections {
                if neighbour_id != node && graph.node_level(neighbour_id) >= layer {
                    graph.add_neighbour_with_pruning(neighbour_id, layer, node, &distance_fn);
                }
            }
        });
    }

    /// Compute connections for a node at a specific layer
    ///
    /// Performs greedy descent through upper layers, then ef_construction
    /// search at the target layer, followed by heuristic pruning.
    ///
    /// ### Params
    ///
    /// * `node` - Node to connect
    /// * `target_layer` - Layer to compute connections for
    /// * `graph` - Construction graph
    /// * `state` - Search state for traversal
    ///
    /// ### Returns
    ///
    /// Selected neighbours as (distance, id) pairs
    fn compute_node_connections_for_layer(
        &self,
        node: usize,
        target_layer: u8,
        graph: &ConstructionGraph<T>,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        state.reset(self.n);

        let mut current_node = self.entry_point as usize;
        let mut current_dist = OrderedFloat(match self.metric {
            Dist::Euclidean => self.euclidean_distance(node, current_node),
            Dist::Cosine => self.cosine_distance(node, current_node),
        });

        for layer in (target_layer + 1..=self.max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbours = graph.get_neighbours(current_node, layer);
                for &neighbour in &neighbours {
                    if neighbour == u32::MAX {
                        continue;
                    }
                    let neighbour = neighbour as usize;

                    if graph.node_level(neighbour) < layer {
                        continue;
                    }

                    let dist = OrderedFloat(match self.metric {
                        Dist::Euclidean => self.euclidean_distance(node, neighbour),
                        Dist::Cosine => self.cosine_distance(node, neighbour),
                    });

                    if dist < current_dist {
                        current_node = neighbour;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
        }

        // At target layer, do full ef_construction search
        // Use reduced ef for base layer to avoid aggressive pruning
        let ef = if target_layer == 0 {
            self.ef_construction / 2
        } else {
            self.ef_construction
        };

        state.reset(self.n);
        let candidates =
            self.search_layer_construction(node, target_layer, current_node, ef, graph, state);

        self.select_neighbours_heuristic(node, &candidates, target_layer, state)
    }

    /// Search layer during construction
    ///
    /// Returns ef closest nodes at the target layer. During construction,
    /// traverses all available connections but only considers nodes that
    /// exist at the target layer or higher.
    ///
    /// ### Params
    ///
    /// * `query_node` - Node to find neighbours for
    /// * `target_layer` - Layer to search
    /// * `entry_node` - Starting point for search
    /// * `ef` - Size of candidate list
    /// * `graph` - Construction graph
    /// * `state` - Search state
    ///
    /// ### Returns
    ///
    /// Vector of (distance, id) pairs for closest candidates
    fn search_layer_construction(
        &self,
        query_node: usize,
        target_layer: u8,
        entry_node: usize,
        ef: usize,
        graph: &ConstructionGraph<T>,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        state.working_sorted.clear();
        state.candidates.clear();

        let entry_dist = OrderedFloat(match self.metric {
            Dist::Euclidean => self.euclidean_distance(query_node, entry_node),
            Dist::Cosine => self.cosine_distance(query_node, entry_node),
        });

        state.mark_visited(entry_node);
        state.candidates.push(Reverse((entry_dist, entry_node)));
        state.working_sorted.insert((entry_dist, entry_node), ef);

        let mut furthest_dist = entry_dist;

        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist > furthest_dist && state.working_sorted.len() >= ef {
                break;
            }

            let current_level = graph.node_level(current_id);

            for layer in 0..=current_level {
                let neighbours = graph.get_neighbours(current_id, layer);

                for &neighbour in &neighbours {
                    if neighbour == u32::MAX {
                        continue;
                    }
                    let neighbour_id = neighbour as usize;

                    if graph.node_level(neighbour_id) < target_layer {
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
        }

        state.working_sorted.data().to_vec()
    }

    /// Select neighbours using the heuristic from the HNSW paper (Algorithm 4)
    ///
    /// Applies the "keep closest and diverse" strategy: iteratively adds
    /// candidates that are closer to the query than to already-selected
    /// neighbours, promoting diversity whilst maintaining proximity.
    ///
    /// ### Params
    ///
    /// * `node` - Query node
    /// * `candidates` - Candidate neighbours with distances
    /// * `layer` - Layer being constructed
    /// * `state` - Search state with scratch buffers
    ///
    /// ### Returns
    ///
    /// Pruned neighbour list respecting max_neighbours constraint
    fn select_neighbours_heuristic(
        &self,
        node: usize,
        candidates: &[(OrderedFloat<T>, usize)],
        layer: u8,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        let max_neighbours = if layer == 0 { self.m * 2 } else { self.m };

        state.scratch_working.clear();
        state.scratch_discarded.clear();

        for &(dist, id) in candidates {
            if id != node {
                state.scratch_working.push((dist, id));
            }
        }

        state.scratch_working.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut result = Vec::with_capacity(max_neighbours);

        for &(cand_dist, cand_id) in &state.scratch_working {
            if result.len() >= max_neighbours {
                break;
            }

            let mut is_good = true;
            for &(_, selected_id) in &result {
                let dist_to_selected = OrderedFloat(match self.metric {
                    Dist::Euclidean => self.euclidean_distance(cand_id, selected_id),
                    Dist::Cosine => self.cosine_distance(cand_id, selected_id),
                });

                if dist_to_selected < cand_dist {
                    is_good = false;
                    break;
                }
            }

            if is_good {
                result.push((cand_dist, cand_id));
            } else if self.keep_pruned {
                state.scratch_discarded.push((cand_dist, cand_id));
            }
        }

        if self.keep_pruned {
            state
                .scratch_discarded
                .sort_unstable_by(|a, b| a.0.cmp(&b.0));

            for &candidate in &state.scratch_discarded {
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
    /// Performs hierarchical search starting from the entry point, greedily
    /// descending through upper layers, then conducting full search at base
    /// layer.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Size of candidate list (higher = better recall, slower)
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted by distance (nearest first)
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

            // start from entry point, descend through upper layers
            let mut current_node = self.entry_point as usize;

            // greedy search through upper layers (max_layer down to 1)
            for layer in (1..=self.max_layer).rev() {
                current_node =
                    self.greedy_search_layer_query(query, query_norm, current_node, layer);
            }

            // full search at base layer
            state.reset(self.n);
            let mut candidates = self.search_layer_query(
                query,
                query_norm,
                0,
                current_node,
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

    /// Greedy search for query vector at a specific layer
    ///
    /// Performs hill-climbing to find the local minimum (closest node) at a
    /// given layer, used for descending through upper layers.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector
    /// * `query_norm` - Pre-computed query norm (for Cosine)
    /// * `start_node` - Starting point
    /// * `layer` - Layer to search
    ///
    /// ### Returns
    ///
    /// Index of closest node found at this layer
    fn greedy_search_layer_query(
        &self,
        query: &[T],
        query_norm: T,
        start_node: usize,
        layer: u8,
    ) -> usize {
        let mut current = start_node;
        let mut current_dist = self.compute_query_distance(query, current, query_norm);

        loop {
            let mut changed = false;

            let neighbours = self.get_neighbours_at_layer(current, layer);

            for &neighbour in neighbours {
                if neighbour == u32::MAX {
                    break;
                }
                let neighbour = neighbour as usize;

                // Check if this neighbour exists at this layer
                if self.layer_assignments[neighbour] < layer {
                    continue;
                }

                let dist = self.compute_query_distance(query, neighbour, query_norm);

                if dist < current_dist {
                    current = neighbour;
                    current_dist = dist;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Full search at a layer for query
    ///
    /// Performs best-first search maintaining ef candidates, used for the
    /// final search at layer 0.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector
    /// * `query_norm` - Pre-computed query norm (for Cosine)
    /// * `layer` - Layer to search
    /// * `entry_node` - Starting point
    /// * `ef` - Size of candidate list
    /// * `state` - Search state
    ///
    /// ### Returns
    ///
    /// Vector of (distance, id) pairs for closest candidates
    fn search_layer_query(
        &self,
        query: &[T],
        query_norm: T,
        layer: u8,
        entry_node: usize,
        ef: usize,
        state: &mut SearchState<T>,
    ) -> Vec<(OrderedFloat<T>, usize)> {
        state.working_sorted.clear();
        state.candidates.clear();

        let entry_dist = OrderedFloat(self.compute_query_distance(query, entry_node, query_norm));

        state.mark_visited(entry_node);
        state.candidates.push(Reverse((entry_dist, entry_node)));
        state.working_sorted.insert((entry_dist, entry_node), ef);

        let mut furthest_dist = entry_dist;

        while let Some(Reverse((current_dist, current_id))) = state.candidates.pop() {
            if current_dist > furthest_dist && state.working_sorted.len() >= ef {
                break;
            }

            let neighbours = self.get_neighbours_at_layer(current_id, layer);

            for &neighbour in neighbours {
                if neighbour == u32::MAX {
                    break;
                }
                let neighbour_id = neighbour as usize;

                if layer > 0 && self.layer_assignments[neighbour_id] < layer {
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

    /// Query using a matrix row reference
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies
    /// to a temporary vector.
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference from a matrix
    /// * `k` - Number of neighbours to return
    /// * `ef_search` - Size of candidate list
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
    /// kNN graph in parallel.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `ef_search` - Search budget
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Whether to print progress
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

    /// Was extend candidates set to `true`
    ///
    /// ### Returns
    ///
    /// Boolean
    pub fn extend_candidates(&self) -> bool {
        self.extend_candidates
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
            + self.layer_assignments.capacity() * std::mem::size_of::<u8>()
            + self.neighbours_flat.capacity() * std::mem::size_of::<u32>()
            + self.neighbour_offsets.capacity() * std::mem::size_of::<usize>()
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
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        self.query(query_vec, k, 200)
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
    fn test_hnsw_multi_layer_storage() {
        // Verify that multi-layer storage is working correctly
        let n = 500;
        let dim = 10;

        let data: Vec<f32> = (0..n * dim).map(|i| (i as f32) * 0.01).collect();
        let mat = Mat::from_fn(n, dim, |i, j| data[i * dim + j]);

        let index = HnswIndex::<f32>::build(mat.as_ref(), 8, 100, "euclidean", 42, true);

        // Count nodes at each layer
        let mut layer_counts = vec![0usize; (index.max_layer + 1) as usize];
        for &level in &index.layer_assignments {
            for l in 0..=level {
                layer_counts[l as usize] += 1;
            }
        }

        // Layer 0 should have all nodes
        assert_eq!(layer_counts[0], n);

        // Higher layers should have fewer nodes (exponential decay)
        for l in 1..layer_counts.len() {
            assert!(
                layer_counts[l] < layer_counts[l - 1],
                "Layer {} should have fewer nodes than layer {}",
                l,
                l - 1
            );
        }

        // Verify queries work
        let query: Vec<f32> = (0..dim).map(|_| 0.5).collect();
        let (indices, _) = index.query(&query, 10, 50);
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_hnsw_varying_m_values() {
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
}
