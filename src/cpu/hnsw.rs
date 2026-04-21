//! HNSW implementation in ann-search-rs. Uses parallel updates during
//! construction of the index which comes at the cost of determinism.

use faer::{MatRef, RowRef};
use num_traits::{Float, FromPrimitive};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::{cell::UnsafeCell, cmp::Reverse, iter::Sum, marker::PhantomData, time::Instant};
use thousands::*;

use crate::prelude::*;
use crate::utils::graph_utils::*;
use crate::utils::*;

/////////////
// Helpers //
/////////////

/// Type alias for the Neighbour updates
pub type NeighbourUpdates<T> = Vec<(usize, Vec<(OrderedFloat<T>, usize)>)>;

impl<T: Ord> Default for SortedBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Construction-time neighbour storage
struct ConstructionGraph<T> {
    /// Flat storage: each layer is a fixed-size block padded with u32::MAX.
    /// Layout per node: [layer0 (M*2 slots), layer1 (M slots), ...].
    /// Sentinels (u32::MAX) mark unused slots; valid IDs are packed at the
    /// front of each layer block.
    nodes: Vec<UnsafeCell<Vec<u32>>>,
    /// Striped spin-locks for thread-safe writes. Stripe count is independent
    /// of graph size, so memory overhead stays constant as the index grows.
    locks: StripedLocks,
    /// Maximum layer each node appears in
    node_levels: Vec<u8>,
    /// Base connectivity parameter
    m: usize,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

unsafe impl<T> Sync for ConstructionGraph<T> {}

impl<T> ConstructionGraph<T>
where
    T: Float + FromPrimitive + Send + Sync + Sum,
{
    /// Create a new construction graph with fixed-size sentinel-padded storage
    ///
    /// Pre-allocates a contiguous slot array for each node, with layer 0
    /// having 2*M slots and upper layers having M slots each. All slots are
    /// initialised to `u32::MAX` (sentinel). This fixed layout ensures that
    /// concurrent lock-free readers never observe an empty or half-allocated
    /// neighbour list.
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes
    /// * `layer_assignments` - Maximum layer for each node
    /// * `m` - Base connectivity parameter
    /// * `threads` - Expected number of concurrent writers, used to size the
    ///   striped lock array
    ///
    /// ### Returns
    ///
    /// Initialised construction graph with sentinel-filled neighbour slots
    fn new(n: usize, layer_assignments: &[u8], m: usize, threads: usize) -> Self {
        let nodes = (0..n)
            .map(|i| {
                let level = layer_assignments[i] as usize;
                let total_slots = m * 2 + level * m;
                UnsafeCell::new(vec![u32::MAX; total_slots])
            })
            .collect();

        Self {
            nodes,
            locks: StripedLocks::new(threads, m),
            node_levels: layer_assignments.to_vec(),
            m,
            _phantom: PhantomData,
        }
    }

    /// Compute the offset within a node's flat slot array for a given layer
    ///
    /// Layer 0 starts at offset 0 and occupies 2*M slots. Each subsequent
    /// layer occupies M slots.
    ///
    /// ### Params
    ///
    /// * `layer` - Layer number
    ///
    /// ### Returns
    ///
    /// Starting index within the node's flat slot array
    #[inline]
    fn layer_offset(&self, layer: u8) -> usize {
        if layer == 0 {
            0
        } else {
            self.m * 2 + (layer as usize - 1) * self.m
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
    fn max_neighbours(&self, layer: u8) -> usize {
        if layer == 0 {
            self.m * 2
        } else {
            self.m
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

    /// Get a read-only slice of neighbours for a node at a specific layer
    ///
    /// Returns the fixed-size slot range for the requested layer. The slice
    /// may contain `u32::MAX` sentinels marking unused positions, packed at
    /// the end. No lock is acquired; benign torn reads are accepted during
    /// construction search because individual `u32` writes are atomic on all
    /// relevant architectures, so each slot is always either a valid node ID
    /// or a sentinel.
    ///
    /// Returns empty slice if the node does not exist at the requested layer.
    ///
    /// ### Params
    ///
    /// * `node_id` - Node index
    /// * `layer` - Layer to query
    ///
    /// ### Returns
    ///
    /// Slice of neighbour slots (may contain `u32::MAX` padding)
    ///
    /// ### Safety
    ///
    /// Caller must ensure no concurrent reallocation of this node's backing
    /// storage. Safe with fixed-size sentinel-padded layout since writes are
    /// always in-place overwrites.
    #[inline]
    pub unsafe fn get_neighbours_slice(&self, node_id: usize, layer: u8) -> &[u32] {
        let node_level = self.node_levels[node_id];
        if layer > node_level {
            return &[];
        }
        let flat = &*self.nodes[node_id].get();
        let offset = self.layer_offset(layer);
        let count = self.max_neighbours(layer);
        &flat[offset..offset + count]
    }

    /// Set neighbours for a node at a specific layer
    ///
    /// Acquires the node lock, then overwrites the layer's slot range
    /// in-place. Valid IDs are written first, followed by sentinel padding.
    /// Self-loops are filtered out. At no point does the slot range appear
    /// empty to concurrent readers.
    ///
    /// No-op if the node does not exist at the requested layer.
    ///
    /// ### Params
    ///
    /// * `node_id` - Node to update
    /// * `layer` - Layer to update
    /// * `neighbours` - New neighbour list as (distance, id) pairs
    fn set_neighbours(&self, node_id: usize, layer: u8, neighbours: &[(OrderedFloat<T>, usize)]) {
        let node_level = self.node_levels[node_id];
        if layer > node_level {
            return;
        }

        let _guard = self.locks.lock_guard(node_id);
        let max_n = self.max_neighbours(layer);
        let flat = unsafe { &mut *self.nodes[node_id].get() };
        let offset = self.layer_offset(layer);
        let slot = &mut flat[offset..offset + max_n];

        let mut i = 0;
        for &(_, neighbour_id) in neighbours.iter().take(max_n) {
            if neighbour_id != node_id {
                slot[i] = neighbour_id as u32;
                i += 1;
            }
        }
        for j in i..max_n {
            slot[j] = u32::MAX;
        }
    }

    /// Add a single neighbour with pruning if the layer is full
    ///
    /// Uses a short-critical-section pattern: snapshot the current neighbour
    /// list under lock, release the lock whilst computing distances and
    /// applying heuristic pruning in thread-local scratch, then reacquire the
    /// lock only to write the result.
    ///
    /// If another thread modified the neighbour list between snapshot and
    /// write (detected by degree comparison), the full path is retried once
    /// under a held lock to guarantee progress.
    ///
    /// Writes are always in-place overwrites of the fixed-size slot range,
    /// so concurrent readers never see an empty list.
    ///
    /// No-op if the node does not exist at the requested layer, or if the
    /// neighbour is already present.
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

        let max_n = self.max_neighbours(layer);
        let offset = self.layer_offset(layer);

        // Fast path: snapshot under lock, compute outside, write under lock
        let snapshot: Vec<u32> = {
            let _guard = self.locks.lock_guard(node_id);
            let flat = unsafe { &*self.nodes[node_id].get() };
            flat[offset..offset + max_n].to_vec()
        };

        let degree = snapshot
            .iter()
            .position(|&e| e == u32::MAX)
            .unwrap_or(max_n);

        if snapshot[..degree]
            .iter()
            .any(|&n| n as usize == new_neighbour)
        {
            return;
        }

        // Room available: try to append directly
        if degree < max_n {
            let _guard = self.locks.lock_guard(node_id);
            let flat = unsafe { &mut *self.nodes[node_id].get() };
            let slot = &mut flat[offset..offset + max_n];
            let current_degree = slot.iter().position(|&e| e == u32::MAX).unwrap_or(max_n);
            // Re-check presence in case another thread added it meanwhile
            if slot[..current_degree]
                .iter()
                .any(|&n| n as usize == new_neighbour)
            {
                return;
            }
            if current_degree < max_n {
                slot[current_degree] = new_neighbour as u32;
                return;
            }
            // List filled up between snapshot and write; fall through to
            // the pruning path below, still under lock.
            self.prune_and_write(slot, max_n, new_neighbour, node_id, &distance_fn);
            return;
        }

        // Full list: compute pruning outside the lock
        let selected =
            self.compute_pruned(&snapshot[..degree], new_neighbour, node_id, &distance_fn);

        // Reacquire to write, validate snapshot is still current
        let _guard = self.locks.lock_guard(node_id);
        let flat = unsafe { &mut *self.nodes[node_id].get() };
        let slot = &mut flat[offset..offset + max_n];
        let current_degree = slot.iter().position(|&e| e == u32::MAX).unwrap_or(max_n);

        if current_degree == degree && slot[..degree] == snapshot[..degree] {
            // Snapshot still valid: commit the pre-computed result
            for i in 0..max_n {
                slot[i] = if i < selected.len() {
                    selected[i] as u32
                } else {
                    u32::MAX
                };
            }
        } else {
            // Snapshot stale: redo pruning under the held lock
            if slot[..current_degree]
                .iter()
                .any(|&n| n as usize == new_neighbour)
            {
                return;
            }
            self.prune_and_write(slot, max_n, new_neighbour, node_id, &distance_fn);
        }
    }

    /// Apply heuristic pruning and overwrite a neighbour slot in place
    ///
    /// Used both by the slow path of `add_neighbour_with_pruning` and by the
    /// fall-back path when a snapshot is invalidated by a concurrent writer.
    /// Must be called with the caller holding the node lock.
    ///
    /// ### Params
    ///
    /// * `slot` - Mutable neighbour slot range for the target layer
    /// * `max_n` - Capacity of the slot range
    /// * `new_neighbour` - Neighbour being considered for inclusion
    /// * `node_id` - Node whose neighbours are being pruned
    /// * `distance_fn` - Function to compute distances between nodes
    fn prune_and_write<F>(
        &self,
        slot: &mut [u32],
        max_n: usize,
        new_neighbour: usize,
        node_id: usize,
        distance_fn: &F,
    ) where
        F: Fn(usize, usize) -> T,
    {
        let degree = slot.iter().position(|&e| e == u32::MAX).unwrap_or(max_n);
        let selected = self.compute_pruned(&slot[..degree], new_neighbour, node_id, distance_fn);
        for i in 0..max_n {
            slot[i] = if i < selected.len() {
                selected[i] as u32
            } else {
                u32::MAX
            };
        }
    }

    /// Compute the heuristically pruned neighbour set outside of any lock
    ///
    /// Collects the current neighbours plus the new candidate, sorts by
    /// distance to `node_id`, then applies the HNSW diversity heuristic: a
    /// candidate is included only if no already-selected neighbour is closer
    /// to it than the query node is. Caller is responsible for persisting the
    /// result to the neighbour slot.
    ///
    /// ### Params
    ///
    /// * `existing` - Current neighbour IDs (excluding sentinels)
    /// * `new_neighbour` - Candidate neighbour to consider
    /// * `node_id` - Node whose neighbourhood is being pruned
    /// * `distance_fn` - Function to compute distances between nodes
    ///
    /// ### Returns
    ///
    /// Pruned neighbour list of length at most `max_neighbours(layer)`
    fn compute_pruned<F>(
        &self,
        existing: &[u32],
        new_neighbour: usize,
        node_id: usize,
        distance_fn: &F,
    ) -> Vec<usize>
    where
        F: Fn(usize, usize) -> T,
    {
        let max_n = existing.len() + 1;
        let mut candidates: Vec<(OrderedFloat<T>, usize)> = existing
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
        candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut selected = Vec::with_capacity(max_n);
        for &(dist, cand_id) in &candidates {
            if selected.len() >= existing.len() {
                break;
            }
            let dominated = selected.iter().any(|&sel_id| {
                let dist_to_selected = OrderedFloat(distance_fn(cand_id, sel_id));
                dist_to_selected < dist
            });
            if !dominated {
                selected.push(cand_id);
            }
        }
        selected
    }

    /// Convert to flat layout for queries
    ///
    /// Consumes the construction graph and produces a flattened neighbour
    /// array suitable for cache-friendly query traversal. Each node's layers
    /// are stored contiguously, preserving the fixed-size sentinel-padded
    /// layout. The offset array records where each node's data begins.
    ///
    /// ### Returns
    ///
    /// Tuple of (flat neighbours, per-node offsets, level assignments)
    fn into_flat(self) -> (Vec<u32>, Vec<usize>, Vec<u8>) {
        let n = self.nodes.len();
        let mut neighbours_flat = Vec::new();
        let mut neighbour_offsets = Vec::with_capacity(n);
        let node_levels = self.node_levels.clone();

        for node_cell in self.nodes {
            neighbour_offsets.push(neighbours_flat.len());
            neighbours_flat.extend(node_cell.into_inner());
        }

        (neighbours_flat, neighbour_offsets, node_levels)
    }
}

//////////////////////////
// Thread-local buffers //
//////////////////////////

thread_local! {
    static SEARCH_STATE_F32: RefCell<SearchState<f32>> = RefCell::new(SearchState::new(1000));
    static BUILD_STATE_F32: RefCell<SearchState<f32>> = RefCell::new(SearchState::new(1000));
    static SEARCH_STATE_F64: RefCell<SearchState<f64>> = RefCell::new(SearchState::new(1000));
    static BUILD_STATE_F64: RefCell<SearchState<f64>> = RefCell::new(SearchState::new(1000));
}

/// Provides access to thread-local [`SearchState`] buffers for a given float type.
///
/// Separates query and construction state to allow both to run concurrently on
/// the same thread without clobbering each other's traversal bookkeeping.
pub trait HnswState<T> {
    /// Access the thread-local search state for query traversal.
    ///
    /// Calls `f` with a reference to the [`RefCell`]-wrapped state and returns
    /// its result. The borrow must not outlive the closure.
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&std::cell::RefCell<SearchState<T>>) -> R;

    /// Access the thread-local search state for index construction.
    ///
    /// Calls `f` with a reference to the [`RefCell`]-wrapped state and returns
    /// its result. The borrow must not outlive the closure.
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
pub struct HnswIndex<T>
where
    T: AnnSearchFloat,
{
    /// Flattened vector data for cache locality
    pub vectors_flat: Vec<T>,
    /// Dimensionality of vectors
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Pre-computed norms for Cosine distance (empty for Euclidean)
    pub norms: Vec<T>,
    /// Distance metric (Euclidean or Cosine)
    metric: Dist,
    /// Maximum layer each node appears in
    layer_assignments: Vec<u8>,
    /// Flattened neighbour storage across all layers
    neighbours_flat: Vec<u32>,
    /// Starting offset for each node's neighbours
    neighbour_offsets: Vec<usize>,
    /// Starting node for queries (highest-layer node)
    entry_point: u32,
    /// Highest layer in the graph
    max_layer: u8,
    /// Base connectivity parameter
    m: usize,
    /// Size of dynamic candidate list during construction
    ef_construction: usize,
    ///  Whether to extend candidate pool (unused)
    extend_candidates: bool,
    /// Original indices - for trait purposes
    original_ids: Vec<usize>,
}

impl<T> VectorDistance<T> for HnswIndex<T>
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

impl<T> HnswIndex<T>
where
    T: AnnSearchFloat,
    Self: HnswState<T>,
{
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
                    T::calculate_l2_norm(&vectors_flat[start..end])
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

        let max_layer = layer_assignments.iter().copied().fold(0u8, u8::max);
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
        let threads = rayon::current_num_threads();
        let construction_graph = ConstructionGraph::new(n, &layer_assignments, m, threads);

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
            original_ids: (0..n).collect(),
        };

        // Build the graph layer by layer, from TOP to BOTTOM
        index.build_graph(&construction_graph, verbose);

        // Convert construction graph to flat layout
        let (neighbours_flat, neighbour_offsets, _) = construction_graph.into_flat();
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
        // Sort nodes: highest layer first, then by node id for determinism
        // within a layer.
        let mut insertion_order: Vec<usize> = (0..self.n).collect();
        insertion_order.sort_unstable_by(|&a, &b| {
            self.layer_assignments[b]
                .cmp(&self.layer_assignments[a])
                .then(a.cmp(&b))
        });

        if verbose {
            println!(
                "Inserting {} nodes incrementally",
                self.n.separate_with_underscores()
            );
        }

        let start = Instant::now();

        // Insert the entry point first (it's the highest-layer node, so it
        // should be first in our sorted order, but let's be explicit)
        let ep = self.entry_point as usize;

        // First phase: Insert upper-layer nodes sequentially
        // These are few in number and form the critical highway structure.
        // Sequential insertion ensures they see each other's connections.
        let (upper_nodes, base_only_nodes): (Vec<usize>, Vec<usize>) = insertion_order
            .into_iter()
            .filter(|&id| id != ep)
            .partition(|&id| self.layer_assignments[id] > 0);

        if verbose {
            println!(
                "  Phase 1: {} upper-layer nodes (sequential)",
                upper_nodes.len().separate_with_underscores()
            );
        }

        for &node in &upper_nodes {
            Self::with_build_state(|state_cell| {
                let mut state = state_cell.borrow_mut();
                self.insert_node(node, graph, &mut state);
            });
        }

        if verbose {
            println!("  Phase 1 done in {:.2?}", start.elapsed());
            println!(
                "  Phase 2: {} base-layer nodes (parallel)",
                base_only_nodes.len().separate_with_underscores()
            );
        }

        let phase2_start = Instant::now();

        // Phase 2: Insert base-only nodes in parallel
        // The upper layers are now fully built, so these nodes can find good
        // entry points via greedy descent. Concurrent insertions at layer 0
        // are protected by per-node locks.
        base_only_nodes.par_iter().for_each(|&node| {
            Self::with_build_state(|state_cell| {
                let mut state = state_cell.borrow_mut();
                self.insert_node(node, graph, &mut state);
            });
        });

        if verbose {
            println!("  Phase 2 done in {:.2?}", phase2_start.elapsed());
            println!("  Total build in {:.2?}", start.elapsed());
        }
    }

    /// Insert a single node into the construction graph across all its layers
    ///
    /// Performs greedy descent from the entry point through upper layers,
    /// then does ef_construction search and connects the node at each
    /// layer it belongs to, from its highest layer down to layer 0.
    fn insert_node(&self, node: usize, graph: &ConstructionGraph<T>, state: &mut SearchState<T>) {
        let node_level = self.layer_assignments[node];
        let mut current_node = self.entry_point as usize;

        // greedy descent through layers above this node's highest layer
        let mut current_dist = OrderedFloat(match self.metric {
            Dist::Euclidean => self.euclidean_distance(node, current_node),
            Dist::Cosine => self.cosine_distance(node, current_node),
        });

        for layer in (node_level + 1..=self.max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;

                // SAFETY: lock-free read of fixed-size sentinel-padded slots.
                // Benign torn reads are acceptable during greedy descent; a
                // stale or partial neighbour list may slow convergence but
                // cannot produce incorrect final results.
                let neighbours = unsafe { graph.get_neighbours_slice(current_node, layer) };

                for &neighbour in neighbours {
                    if neighbour == u32::MAX {
                        break;
                    }
                    let neighbour = neighbour as usize;

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

        // for each layer this node belongs to (top-down), search and connect
        let distance_fn = |a: usize, b: usize| -> T {
            match self.metric {
                Dist::Euclidean => self.euclidean_distance(a, b),
                Dist::Cosine => self.cosine_distance(a, b),
            }
        };

        for layer in (0..=node_level).rev() {
            state.reset(self.n);

            self.search_layer_construction(
                node,
                layer,
                current_node,
                self.ef_construction,
                graph,
                state,
            );

            let candidates: Vec<(OrderedFloat<T>, usize)> = state.working_sorted.data().to_vec();
            let selected = self.select_neighbours_heuristic(node, &candidates, layer, state);

            // set this node's outgoing neighbours
            graph.set_neighbours(node, layer, &selected);

            // update reverse links: add this node to each neighbour's list
            for &(_, neighbour_id) in &selected {
                if neighbour_id != node && graph.node_level(neighbour_id) >= layer {
                    graph.add_neighbour_with_pruning(neighbour_id, layer, node, &distance_fn);
                }
            }

            // use the closest result as entry point for the next layer down
            if let Some(&(_, closest)) = selected.first() {
                current_node = closest;
            }
        }
    }

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

        // safety: bounds are guaranteed by construction
        &self.neighbours_flat[offset..offset + count]
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
    /// * `state` - Mutable reference to search state
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
    ) {
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

            // SAFETY: benign race — stale/torn reads are acceptable during
            // construction search, same rationale as Vamana.
            let neighbours = unsafe { graph.get_neighbours_slice(current_id, target_layer) };

            for &neighbour in neighbours {
                if neighbour == u32::MAX {
                    continue;
                }
                let neighbour_id = neighbour as usize;

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

            let is_good = !result.iter().any(|&(_, selected_id)| {
                let dist_to_selected = OrderedFloat(match self.metric {
                    Dist::Euclidean => self.euclidean_distance(cand_id, selected_id),
                    Dist::Cosine => self.cosine_distance(cand_id, selected_id),
                });
                dist_to_selected < cand_dist
            });

            if is_good {
                result.push((cand_dist, cand_id));
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
    T: AnnSearchFloat,
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
