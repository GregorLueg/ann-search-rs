//! Shared primitives for NN-Descent-style flat kNN graphs.
//!
//! Both the CPU NN-Descent build (`src/cpu/nndescent.rs`) and Relative
//! NN-Descent (`src/cpu/rnn_descent.rs`) operate on the same layout: a flat
//! `Vec<Neighbour<T>>` of size `n * k`, sorted by distance ascending per
//! node, with unused trailing slots marked by [`SENTINEL_PID`]. Updates flow
//! through radix-sorted [`Update<T>`] batches applied lock-free via
//! [`ApplySortedUpdates`].
//!
//! The GPU NN-Descent module also reads [`SENTINEL_PID`] to detect empty
//! neighbour slots downloaded from device.

use rdst::RadixKey;

///////////////
// Sentinels //
///////////////

/// Sentinel point id used to mark empty slots in the flat kNN graph.
///
/// The high bit of [`Neighbour::pid_and_flag`] carries the is-new flag, so
/// only 31 bits are available for the point id. `u32::MAX >> 1` therefore
/// sits at the top of that range and is guaranteed distinct from any valid
/// point id up to ~2 billion.
pub const SENTINEL_PID: usize = u32::MAX as usize >> 1;

////////////////
// Neighbours //
////////////////

/// Neighbour entry in the flat kNN graph (build phase only).
///
/// Flat structure in C representation for cache locality. The high bit of
/// `pid_and_flag` stores the is-new flag, leaving 31 bits for the point id
/// (sufficient for ~2 billion points).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Neighbour<T> {
    /// Point index + new/old flag in the high bit
    pid_and_flag: u32,
    /// Distance to the neighbour
    pub dist: T,
}

impl<T: Copy> Neighbour<T> {
    const IS_NEW_MASK: u32 = 1 << 31;
    const PID_MASK: u32 = !Self::IS_NEW_MASK;

    /// Create a new neighbour entry.
    ///
    /// The point id must fit in 31 bits; the 32nd bit is reserved for the
    /// is-new flag. Point ids up to ~2 billion are supported.
    ///
    /// ### Params
    ///
    /// * `pid` - Point id (must fit in 31 bits)
    /// * `dist` - Distance to the neighbour
    /// * `is_new` - Whether this neighbour has been explored yet
    ///
    /// ### Returns
    ///
    /// Packed neighbour entry with the encoded flag.
    #[inline(always)]
    pub fn new(pid: usize, dist: T, is_new: bool) -> Self {
        debug_assert!(pid <= Self::PID_MASK as usize, "PID exceeds 31-bit limit");
        let flag = if is_new { Self::IS_NEW_MASK } else { 0 };
        Self {
            pid_and_flag: (pid as u32) | flag,
            dist,
        }
    }

    /// Whether this neighbour has not yet been explored.
    ///
    /// New neighbours participate in local joins during the next iteration;
    /// old ones only contribute to old-new pair generation.
    ///
    /// ### Returns
    ///
    /// `true` if the high bit is set, `false` otherwise.
    #[inline(always)]
    pub fn is_new(&self) -> bool {
        (self.pid_and_flag & Self::IS_NEW_MASK) != 0
    }

    /// Point id with the flag bit masked out.
    ///
    /// ### Returns
    ///
    /// Point index in the range `[0, 2^31)`.
    #[inline(always)]
    pub fn pid(&self) -> usize {
        (self.pid_and_flag & Self::PID_MASK) as usize
    }

    /// Whether this slot is empty (holds the sentinel point id).
    ///
    /// ### Returns
    ///
    /// `true` if this slot does not hold a valid neighbour.
    #[inline(always)]
    pub fn is_sentinel(&self) -> bool {
        self.pid() == SENTINEL_PID
    }

    /// Mark this neighbour as old (already explored).
    ///
    /// Clears the high bit whilst preserving the point id and distance.
    #[inline(always)]
    pub fn mark_old(&mut self) {
        self.pid_and_flag &= Self::PID_MASK;
    }
}

///////////////
// Graph ptr //
///////////////

/// Unsafe pointer wrapper for lock-free parallel writes to the flat graph.
///
/// Safety is guaranteed by the update pattern: segments are grouped by target
/// node, so no two threads ever write to the same `target * k` block
/// simultaneously.
///
/// ### Fields
#[derive(Copy, Clone)]
pub struct UnsafeGraphPtr<T>(
    /// Raw mutable pointer to the flat neighbour buffer
    pub *mut Neighbour<T>,
);

unsafe impl<T> Send for UnsafeGraphPtr<T> {}
unsafe impl<T> Sync for UnsafeGraphPtr<T> {}

/////////////
// Updates //
/////////////

/// Candidate edge update for radix-sorted batched application.
///
/// Represents one directed edge `source -> target` with the pre-computed
/// distance. Batches are radix-sorted by `target` so that all updates for a
/// given destination node are contiguous, enabling lock-free application via
/// [`ApplySortedUpdates`].
///
/// Node ids are stored as `u32` to keep the struct small (12 bytes for
/// `Update<f32>` before alignment). The 31-bit cap on point ids in
/// [`Neighbour`] already restricts the id range, so `u32` is sufficient.
#[derive(Clone, Copy)]
pub struct Update<T> {
    /// Target node id (the node whose adjacency list receives the edge)
    pub target: u32,
    /// Source node id (the node on the other end of the edge)
    pub source: u32,
    /// Distance between the two nodes
    pub dist: T,
}

impl<T> Update<T> {
    /// Create a new update triple.
    ///
    /// ### Params
    ///
    /// * `target` - Node receiving the edge
    /// * `source` - Node on the other end of the edge
    /// * `dist` - Distance between the two nodes
    ///
    /// ### Returns
    ///
    /// Update triple ready for radix sorting.
    #[inline(always)]
    pub fn new(target: u32, source: u32, dist: T) -> Self {
        Self {
            target,
            source,
            dist,
        }
    }
}

/// Radix key on `target` for `rdst::RadixSort`.
///
/// Sorting by target puts all updates for the same destination contiguously,
/// which is exactly the shape [`ApplySortedUpdates::apply_sorted_updates`]
/// consumes.
impl<T> RadixKey for Update<T> {
    const LEVELS: usize = 4;

    /// Claude TODO: add documentation.
    #[inline]
    fn get_level(&self, level: usize) -> u8 {
        (self.target >> (level * 8)) as u8
    }
}

/// Find contiguous target boundaries in a sorted update batch.
///
/// Returns index offsets so `updates[boundaries[i]..boundaries[i+1]]` is the
/// slice of updates targeting a single node.
///
/// ### Params
///
/// * `updates` - Updates sorted ascending by `target`
///
/// ### Returns
///
/// Boundary indices of length `num_distinct_targets + 1`.
pub fn find_target_boundaries<T>(updates: &[Update<T>]) -> Vec<usize> {
    if updates.is_empty() {
        return vec![0, 0];
    }

    let mut boundaries = vec![0];

    for i in 1..updates.len() {
        if updates[i].target != updates[i - 1].target {
            boundaries.push(i);
        }
    }

    boundaries.push(updates.len());
    boundaries
}

////////////
// Trait  //
////////////

/// Apply sorted neighbour updates to the flat kNN graph.
///
/// Implementations are split per concrete float type (`f32`, `f64`) because
/// the merge step relies on thread-local heaps keyed by numeric type. The
/// sorted layout enables lock-free processing since updates targeting the
/// same node form a contiguous slice.
///
/// ### Algorithm
///
/// 1. Find target boundaries in the sorted updates.
/// 2. Extract each target's update batch as a contiguous slice.
/// 3. Process batches in parallel.
/// 4. Merge new candidates with existing neighbours via thread-local heaps.
/// 5. Write results back to the flat graph via disjoint pointer writes.
pub trait ApplySortedUpdates<T> {
    /// Apply sorted updates to the flat `n * k` graph in place.
    ///
    /// ### Params
    ///
    /// * `updates` - Must be sorted by `target` (first field)
    /// * `graph` - Flat graph of size `n * k`
    /// * `k` - Neighbours per node
    /// * `updates_count` - Atomic counter for the number of edge changes
    fn apply_sorted_updates(
        &self,
        updates: &[Update<T>],
        graph: &mut [Neighbour<T>],
        k: usize,
        updates_count: &std::sync::atomic::AtomicUsize,
    );
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbour_roundtrip_pid_and_flag() {
        let n = Neighbour::<f32>::new(42, 1.5, true);
        assert_eq!(n.pid(), 42);
        assert!(n.is_new());
        assert_eq!(n.dist, 1.5);

        let mut m = n;
        m.mark_old();
        assert_eq!(m.pid(), 42);
        assert!(!m.is_new());
    }

    #[test]
    fn neighbour_sentinel_detected() {
        let n = Neighbour::<f32>::new(SENTINEL_PID, f32::MAX, false);
        assert!(n.is_sentinel());
    }

    #[test]
    fn neighbour_max_valid_pid() {
        // Largest PID that still fits in 31 bits without hitting the sentinel.
        let pid = SENTINEL_PID - 1;
        let n = Neighbour::<f32>::new(pid, 0.0, true);
        assert_eq!(n.pid(), pid);
        assert!(!n.is_sentinel());
    }

    #[test]
    fn find_boundaries_empty() {
        let updates: Vec<Update<f32>> = Vec::new();
        assert_eq!(find_target_boundaries(&updates), vec![0, 0]);
    }

    #[test]
    fn find_boundaries_single_target() {
        let updates: Vec<Update<f32>> = (0..5u32).map(|i| Update::new(7, i, i as f32)).collect();
        assert_eq!(find_target_boundaries(&updates), vec![0, 5]);
    }

    #[test]
    fn find_boundaries_multiple_targets() {
        let updates = vec![
            Update::<f32>::new(1, 0, 0.1),
            Update::<f32>::new(1, 2, 0.2),
            Update::<f32>::new(3, 0, 0.3),
            Update::<f32>::new(5, 1, 0.4),
            Update::<f32>::new(5, 2, 0.5),
            Update::<f32>::new(5, 3, 0.6),
        ];
        assert_eq!(find_target_boundaries(&updates), vec![0, 2, 3, 6]);
    }

    #[test]
    fn update_radix_key_orders_by_target() {
        use rdst::RadixSort;
        let mut updates = vec![
            Update::<f32>::new(300, 0, 0.0),
            Update::<f32>::new(2, 0, 0.0),
            Update::<f32>::new(70_000, 0, 0.0),
            Update::<f32>::new(1, 0, 0.0),
        ];
        updates.radix_sort_unstable();
        let targets: Vec<_> = updates.iter().map(|u| u.target).collect();
        assert_eq!(targets, vec![1, 2, 300, 70_000]);
    }
}
