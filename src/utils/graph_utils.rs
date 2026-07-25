//! Utility functions shared across the graph-based indices that contain
//! the search state.

use fixedbitset::FixedBitSet;
use num_traits::Float;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::Sum;

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
