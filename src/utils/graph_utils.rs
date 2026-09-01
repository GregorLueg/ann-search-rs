//! Utility functions shared across the graph-based indices that contain
//! the search state.

use fixedbitset::FixedBitSet;
use num_traits::{Float, FromPrimitive};
use std::cell::{RefCell, UnsafeCell};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::Sum;
use std::marker::PhantomData;

use crate::prelude::*;

/// Search state for HNSW/Vamana/NSG queries and construction.
///
/// Maintains visited tracking and candidate management for graph traversal.
/// Reused across queries to amortise allocation costs.
///
/// `visited` is a [`FixedBitSet`] (1 bit per node) rather than an epoch-based
/// `Vec<usize>`. At n=150k the bitset is ~19 KB and fits in L1, versus 1.2 MB
/// for the `Vec<usize>` alternative. The per-reset `clear()` cost
/// (~O(n/64) u64 writes) is negligible compared to the cache-thrashing the
/// wider array causes during dense per-node build phases (e.g. NSG's MRNG
/// prune, which resets once per node).
pub struct SearchState<T> {
    /// Per-node visit tracking as a bit-per-node bitset.
    pub visited: FixedBitSet,
    /// Min-heap of nodes to explore, ordered by distance
    pub candidates: BinaryHeap<Reverse<(OrderedFloat<T>, usize)>>,
    /// Sorted buffer of current best candidates
    pub working_sorted: SortedBuffer<(OrderedFloat<T>, usize)>,
    /// Bounded max-heap of current best candidates.
    ///
    /// Serves the same purpose as `working_sorted` but at `O(log ef)` per
    /// accepted candidate instead of the `O(ef)` memmove `Vec::insert` costs.
    /// The two coexist because the buffer's tail is only worth the swap where
    /// `ef` is large: HNSW runs at `ef_construction` / `ef_search` in the
    /// hundreds for both build and query, whereas NSG and Vamana prune against
    /// lists an order of magnitude shorter.
    pub results: BoundedMaxHeap<T>,
    /// Temporary storage for heuristic selection
    pub scratch_working: Vec<(OrderedFloat<T>, usize)>,
    /// Temporary storage for pruned candidates
    pub scratch_discarded: Vec<(OrderedFloat<T>, usize)>,
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
    pub fn new(capacity: usize) -> Self {
        Self {
            visited: FixedBitSet::with_capacity(capacity),
            candidates: BinaryHeap::with_capacity(capacity),
            working_sorted: SortedBuffer::with_capacity(capacity),
            results: BoundedMaxHeap::new(capacity),
            scratch_working: Vec::with_capacity(capacity),
            scratch_discarded: Vec::with_capacity(capacity),
        }
    }

    /// Reset state for a new query
    ///
    /// Grows the visited bitset if needed, clears all bits, and empties
    /// candidate / scratch buffers.
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes in the graph (for capacity adjustment)
    pub fn reset(&mut self, n: usize) {
        if self.visited.len() < n {
            self.visited.grow(n);
        }
        self.visited.clear();

        self.candidates.clear();
        self.working_sorted.clear();
        self.results.clear();
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
    pub fn is_visited(&self, node: usize) -> bool {
        self.visited.contains(node)
    }

    /// Mark a node as visited in the current query
    ///
    /// ### Params
    ///
    /// * `node` - Node index to mark
    #[inline(always)]
    pub fn mark_visited(&mut self, node: usize) {
        self.visited.insert(node);
    }
}

//////////////////////
// ConstructionGraph //
//////////////////////

// Shared by the graph builders. It lives here rather than beside any one index
// because it owns the striped-lock concurrency and the sentinel slot layout,
// and a second copy of that would drift.

/// Construction-time neighbour storage
pub(crate) struct ConstructionGraph<T> {
    /// One contiguous slot array for the whole graph, node blocks concatenated
    /// in node order. Layout within a node's block: [layer0 (M*2 slots),
    /// layer1 (M slots), ...]. Sentinels (u32::MAX) mark unused slots; valid
    /// IDs are packed at the front of each layer block.
    nodes: UnsafeCell<Vec<u32>>,

    /// Start of each node's block within `nodes`.
    ///
    /// Identical to the `neighbour_offsets` the finished index carries, which
    /// is what lets [`ConstructionGraph::into_flat`] hand the buffer over by
    /// move instead of copying it out node by node.
    offsets: Vec<usize>,

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
    pub(crate) fn new(n: usize, layer_assignments: &[u8], m: usize, threads: usize) -> Self {
        let mut offsets = Vec::with_capacity(n);
        let mut total = 0usize;
        for &level in layer_assignments.iter().take(n) {
            offsets.push(total);
            total += m * 2 + level as usize * m;
        }

        Self {
            nodes: UnsafeCell::new(vec![u32::MAX; total]),
            offsets,
            locks: StripedLocks::new(threads, m),
            node_levels: layer_assignments.to_vec(),
            m,
            _phantom: PhantomData,
        }
    }

    /// Base pointer of the contiguous slot array
    ///
    /// ### Returns
    ///
    /// Pointer to slot zero, valid for the lifetime of the graph
    #[inline]
    fn data_ptr(&self) -> *mut u32 {
        unsafe { (*self.nodes.get()).as_mut_ptr() }
    }

    /// Locate a node's slot range for one layer within the flat array
    ///
    /// ### Params
    ///
    /// * `node_id` - Node index
    /// * `layer` - Layer number
    ///
    /// ### Returns
    ///
    /// `(start, len)` into the flat slot array
    #[inline]
    fn slot_range(&self, node_id: usize, layer: u8) -> (usize, usize) {
        (
            self.offsets[node_id] + self.layer_offset(layer),
            self.max_neighbours(layer),
        )
    }

    /// Mutable view of a node's slot range for one layer
    ///
    /// ### Params
    ///
    /// * `node_id` - Node index
    /// * `layer` - Layer number
    ///
    /// ### Returns
    ///
    /// The layer's slot range as a mutable slice
    ///
    /// ### Safety
    ///
    /// Caller must hold this node's stripe lock. Blocks belonging to different
    /// nodes are disjoint, so concurrent writers under their own locks never
    /// alias.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    unsafe fn slots_mut(&self, node_id: usize, layer: u8) -> &mut [u32] {
        let (start, len) = self.slot_range(node_id, layer);
        std::slice::from_raw_parts_mut(self.data_ptr().add(start), len)
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
    /// 2 * M for layer 0, M for upper layers
    #[inline]
    pub(crate) fn max_neighbours(&self, layer: u8) -> usize {
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
    pub(crate) fn node_level(&self, node_id: usize) -> u8 {
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
        let (start, len) = self.slot_range(node_id, layer);
        std::slice::from_raw_parts(self.data_ptr().add(start), len)
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
    pub(crate) fn set_neighbours(
        &self,
        node_id: usize,
        layer: u8,
        neighbours: &[(OrderedFloat<T>, usize)],
    ) {
        let node_level = self.node_levels[node_id];
        if layer > node_level {
            return;
        }

        let _guard = self.locks.lock_guard(node_id);
        let max_n = self.max_neighbours(layer);
        let slot = unsafe { self.slots_mut(node_id, layer) };

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
    pub(crate) fn add_neighbour_with_pruning<F>(
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

        // Fast path: snapshot under lock, compute outside, write under lock
        let snapshot: Vec<u32> = {
            let _guard = self.locks.lock_guard(node_id);
            unsafe { self.slots_mut(node_id, layer) }.to_vec()
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
            let slot = unsafe { self.slots_mut(node_id, layer) };
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
            self.prune_and_write(slot, max_n, new_neighbour, node_id, &distance_fn);
            return;
        }

        // Full list: compute pruning outside the lock
        let selected = self.compute_pruned(
            &snapshot[..degree],
            new_neighbour,
            node_id,
            max_n,
            &distance_fn,
        );

        // Reacquire to write, validate snapshot is still current
        let _guard = self.locks.lock_guard(node_id);
        let slot = unsafe { self.slots_mut(node_id, layer) };
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
        let selected =
            self.compute_pruned(&slot[..degree], new_neighbour, node_id, max_n, distance_fn);
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
    /// to it than the query node is. Rejects are dropped rather than used to
    /// fill the remaining slots, matching the forward pass in
    /// [`HnswIndex::select_neighbours_heuristic`]. Caller is responsible for
    /// persisting the result to the neighbour slot.
    ///
    /// ### Params
    ///
    /// * `existing` - Current neighbour IDs (excluding sentinels)
    /// * `new_neighbour` - Candidate neighbour to consider
    /// * `node_id` - Node whose neighbourhood is being pruned
    /// * `max_n` - Slot capacity for the layer being pruned
    /// * `distance_fn` - Function to compute distances between nodes
    ///
    /// ### Returns
    ///
    /// Pruned neighbour list of length at most `max_n`
    fn compute_pruned<F>(
        &self,
        existing: &[u32],
        new_neighbour: usize,
        node_id: usize,
        max_n: usize,
        distance_fn: &F,
    ) -> Vec<usize>
    where
        F: Fn(usize, usize) -> T,
    {
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
        candidates.sort_unstable_by_key(|a| a.0);

        let mut selected = Vec::with_capacity(max_n);
        for &(dist, cand_id) in &candidates {
            if selected.len() >= max_n {
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

    /// Hand the flat layout over to the finished index
    ///
    /// The construction layout is already the query layout, so this is a move
    /// rather than a copy: node blocks sit contiguously in node order with the
    /// fixed-size sentinel padding intact, and the offset array records where
    /// each node's block begins.
    ///
    /// ### Returns
    ///
    /// Tuple of (flat neighbours, per-node offsets, level assignments)
    pub(crate) fn into_flat(self) -> (Vec<u32>, Vec<usize>, Vec<u8>) {
        (self.nodes.into_inner(), self.offsets, self.node_levels)
    }
}

////////////////////////////
// ThreadLocalSearchState //
////////////////////////////

thread_local! {
    static QUERY_STATE_F32: RefCell<SearchState<f32>> = RefCell::new(SearchState::new(1000));
    static QUERY_STATE_F64: RefCell<SearchState<f64>> = RefCell::new(SearchState::new(1000));
}

/// Access to a reusable per-thread [`SearchState`], keyed on the float type.
///
/// Query paths fan out over rayon and would otherwise allocate a visited
/// bitset per query; at a million nodes that is 125 KiB of allocate-and-clear
/// against a query that should cost tens of microseconds. Keyed on `T` rather
/// than on an index type, so every graph index can share the same buffers.
///
/// ### Note
///
/// Construction does *not* use this. Builders take their state through
/// `for_each_init`, which keeps build and query state from colliding when both
/// run on the same thread.
pub trait ThreadLocalSearchState: Sized {
    /// Run `f` against this thread's search state.
    ///
    /// ### Params
    ///
    /// * `f` - Closure receiving the mutable state
    ///
    /// ### Returns
    ///
    /// Whatever `f` returns
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&mut SearchState<Self>) -> R;
}

impl ThreadLocalSearchState for f32 {
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&mut SearchState<f32>) -> R,
    {
        QUERY_STATE_F32.with(|cell| f(&mut cell.borrow_mut()))
    }
}

impl ThreadLocalSearchState for f64 {
    fn with_search_state<F, R>(f: F) -> R
    where
        F: FnOnce(&mut SearchState<f64>) -> R,
    {
        QUERY_STATE_F64.with(|cell| f(&mut cell.borrow_mut()))
    }
}
