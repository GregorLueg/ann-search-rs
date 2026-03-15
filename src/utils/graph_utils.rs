//! Utility functions shared across the graph-based indices that contain
//! the search state.

use num_traits::Float;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::Sum;

use crate::prelude::*;

/// Search state for HNSW/Vamana queries and construction
///
/// Maintains visited tracking and candidate management for graph traversal.
/// Reused across queries to amortise allocation costs.
pub struct SearchState<T> {
    /// Per-node visit tracking using incrementing IDs
    pub visited: Vec<usize>,
    /// Current visit epoch (wraps around, triggers reset)
    pub visit_id: usize,
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
    pub fn reset(&mut self, n: usize) {
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
    pub fn is_visited(&self, node: usize) -> bool {
        self.visited[node] == self.visit_id
    }

    /// Mark a node as visited in the current query
    ///
    /// ### Params
    ///
    /// * `node` - Node index to mark
    #[inline(always)]
    pub fn mark_visited(&mut self, node: usize) {
        self.visited[node] = self.visit_id;
    }
}
