use std::cmp::Ordering;

/// Min members per leaf
pub const LEAF_MIN_MEMBERS: usize = 64;

/////////////
// Helpers //
/////////////

/// Priority queue entry for backtracking during search
///
/// Stores nodes to revisit, ordered by distance to query (further = lower
/// priority)
///
/// ### Fields
///
/// * `margin` - Negative absolute margin to hyperplane (for max-heap ->
///   min-distance)
/// * `node_idx` - Index of node to explore
pub struct BacktrackEntry {
    pub margin: f64,
    pub node_idx: u32,
}

impl PartialEq for BacktrackEntry {
    fn eq(&self, other: &Self) -> bool {
        self.margin == other.margin
    }
}

impl Eq for BacktrackEntry {}

impl PartialOrd for BacktrackEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BacktrackEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.margin
            .partial_cmp(&other.margin)
            .unwrap_or(Ordering::Equal)
    }
}

/// Bitset for tracking visited items during search
///
/// Uses one bit per item, packed into u64 chunks for cache efficiency
pub struct VisitedSet {
    data: Vec<u64>,
}

impl VisitedSet {
    /// Create a new bitset with capacity for n items
    ///
    /// Allocates enough u64 chunks to store one bit per item, rounding up
    /// to the nearest 64-item boundary.
    ///
    /// ### Params
    ///
    /// * `capacity` - Number of items to track
    ///
    /// ### Returns
    ///
    /// Initialised bitset with all bits set to 0 (unvisited)
    pub fn new(capacity: usize) -> Self {
        // Round up to nearest u64
        let size = capacity.div_ceil(64);
        Self {
            data: vec![0; size],
        }
    }

    /// Mark an item as visited
    ///
    /// Sets the bit corresponding to `index` to 1. Uses unsafe unchecked
    /// access for performance - caller must ensure index < capacity.
    ///
    /// ### Params
    ///
    /// * `index` - Item index to mark
    ///
    /// ### Returns
    ///
    /// `true` if item was already visited, `false` if newly marked
    #[inline]
    pub fn mark(&mut self, index: usize) -> bool {
        let chunk = index / 64;
        let bit = 1 << (index % 64);
        // Use get_unchecked for speed, safe if capacity is correct
        unsafe {
            let slot = self.data.get_unchecked_mut(chunk);
            if (*slot & bit) != 0 {
                return true;
            }
            *slot |= bit;
            false
        }
    }
}
