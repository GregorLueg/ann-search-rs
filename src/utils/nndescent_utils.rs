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

use rayon::prelude::*;
use rdst::RadixKey;
use std::sync::atomic::{AtomicU32, Ordering};

///////////////
// Sentinels //
///////////////

/// Sentinel point id used to mark empty slots in the flat kNN graph.
///
/// The high bit of `Neighbour` carries the is-new flag, so only 31 bits are
/// available for the point id. `u32::MAX >> 1` therefore sits at the top of
/// that range and is guaranteed distinct from any valid point id up to
/// ~2 billion.
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

    /// Mark this neighbour as new (eligible for re-exploration).
    ///
    /// Sets the high bit whilst preserving the point id and distance. Relative
    /// NN-Descent re-arms every surviving edge this way between outer rounds,
    /// otherwise the new/old guard in its prune loop skips every pair and the
    /// remaining rounds do no work.
    #[inline(always)]
    pub fn mark_new(&mut self) {
        self.pid_and_flag |= Self::IS_NEW_MASK;
    }
}

////////////////////
// Graph unpacking //
////////////////////

/// Unpack a flat sentinel-padded kNN graph into per-node rows.
///
/// The flat `n * k` layout is what every NN-Descent-derived index stores and
/// what a downstream index like NSG consumes directly. This is the adapter for
/// callers who want plain kNN output instead: it hands back the graph exactly
/// as it was built, with no beam-search re-query.
///
/// Rows are already sorted by distance ascending, so `k_out` truncates from
/// the front. Sentinel slots are dropped, which means a row can come back
/// **shorter than `k`** where the descent never filled it (small `n`,
/// disconnected components). That is the one behavioural difference from the
/// `query_*_self` functions, which always return exactly `k`.
///
/// A kNN graph stores no `i -> i` edge, but every `query_*_self` in the crate
/// returns a point as its own nearest neighbour at distance zero, and so does
/// an exhaustive ground truth. `include_self` closes that gap: set it and row
/// `i` starts with `(i, 0)`, so the row is directly comparable to a self-query
/// at the same `k`. Leave it unset for a graph of true neighbours only. Getting
/// this wrong costs a silent, flat `1/k` against anything scored the other way.
///
/// ### Params
///
/// * `graph` - Flat graph of `n * k` `(pid, distance)` pairs, row `i` at
///   `[i*k .. (i+1)*k]`, sentinel-padded.
/// * `n` - Number of nodes.
/// * `k` - Neighbours per node in `graph`.
/// * `k_out` - Truncate each row to this **total** length, self-edge included
///   when `include_self` is set. `None` keeps the whole row and still prepends
///   the self-edge, giving `k + 1` entries.
/// * `include_self` - Prepend `(i, 0)` to row `i`.
/// * `return_dist` - Whether to materialise the distances.
///
/// ### Returns
///
/// `(indices, distances)` with one row per node. Distances are `None` when
/// `return_dist` is false.
pub fn unpack_knn_graph<T: Copy + Send + Sync + num_traits::Zero + PartialOrd>(
    graph: &[(usize, T)],
    n: usize,
    k: usize,
    k_out: Option<usize>,
    include_self: bool,
    return_dist: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
    let self_slot = usize::from(include_self);
    let total = k_out.unwrap_or(k + self_slot).max(self_slot);
    let take = (total - self_slot).min(k);

    // One pass over each row builds both outputs; splitting it walks the graph
    // twice for no reason.
    let rows: Vec<(Vec<usize>, Vec<T>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut ids = Vec::with_capacity(take + self_slot);
            let mut dists = if return_dist {
                Vec::with_capacity(take + self_slot)
            } else {
                Vec::new()
            };
            if include_self {
                ids.push(i);
                if return_dist {
                    dists.push(T::zero());
                }
            }
            for &(pid, dist) in &graph[i * k..(i + 1) * k] {
                if pid == SENTINEL_PID {
                    continue;
                }
                if ids.len() == take + self_slot {
                    break;
                }
                ids.push(pid);
                if return_dist {
                    dists.push(dist);
                }
            }
            (ids, dists)
        })
        .collect();

    crate::utils::pack_knn_results(rows, return_dist)
}

///////////////
// Graph ptr //
///////////////

/// Unsafe pointer wrapper for lock-free parallel writes to a flat buffer.
///
/// Safety rests on the caller partitioning the buffer: the graph writes are
/// grouped by target node so no two threads touch the same `target * k` block,
/// and the counting-sort scatter in [`UpdateGrouper::group`] hands out each
/// slot exactly once.
///
/// ### Fields
#[derive(Copy, Clone)]
pub struct UnsafeMutPtr<E>(
    /// Raw mutable pointer to the buffer
    pub *mut E,
);

unsafe impl<E> Send for UnsafeMutPtr<E> {}
unsafe impl<E> Sync for UnsafeMutPtr<E> {}

/// Flat neighbour buffer pointer, the shape the graph writers use.
pub type UnsafeGraphPtr<T> = UnsafeMutPtr<Neighbour<T>>;

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

    /// Extract byte `level` of the `target` field, least-significant first.
    #[inline]
    fn get_level(&self, level: usize) -> u8 {
        (self.target >> (level * 8)) as u8
    }
}

///////////////////
// Update grouper //
///////////////////

/// Reusable scratch for grouping an update batch by target node.
///
/// A build runs one grouping per pass, so the cursor array and the output
/// buffer are held here and reused rather than allocated each time.
///
/// ### Fields
pub struct UpdateGrouper<T> {
    /// Per-target counts, converted in place into write cursors
    cursors: Vec<AtomicU32>,
    /// Grouped updates, target-contiguous
    data: Vec<Update<T>>,
}

impl<T: Copy + Send + Sync> Default for UpdateGrouper<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Send + Sync> UpdateGrouper<T> {
    /// Create an empty grouper.
    ///
    /// ### Returns
    ///
    /// Grouper holding no scratch; the first [`UpdateGrouper::group`] sizes it.
    pub fn new() -> Self {
        Self {
            cursors: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Zero the per-target counters, growing the array if needed.
    ///
    /// Call before the pass that emits updates, so it can count them through
    /// [`UpdateGrouper::counts`].
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes
    pub fn reset_counts(&mut self, n: usize) {
        if self.cursors.len() < n {
            self.cursors = (0..n).map(|_| AtomicU32::new(0)).collect();
        } else {
            for c in self.cursors[..n].iter() {
                c.store(0, Ordering::Relaxed);
            }
        }
    }

    /// Per-target counters, to be incremented once per emitted update.
    ///
    /// Counting where the updates are produced folds the histogram into a pass
    /// that already holds the target in a register, which is worth more than it
    /// looks: as a separate pass the counting is a stream of relaxed increments
    /// at random offsets, sixteen counters to a cache line, so the line bounces
    /// between cores and the pass costs the same on eight threads as on one.
    /// Inside the producing loop the same increments overlap the work already in
    /// flight.
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes
    ///
    /// ### Returns
    ///
    /// The counter slice, one entry per node.
    #[inline]
    pub fn counts(&self, n: usize) -> &[AtomicU32] {
        &self.cursors[..n]
    }

    /// Group per-task update batches whose targets are already counted.
    ///
    /// Turns the counts from [`UpdateGrouper::counts`] into write cursors with
    /// one serial `O(n)` pass, then scatters every update into its target's
    /// segment. That is one read-and-write of the batch, against the several
    /// full passes a radix sort makes over 12-byte elements, its same-size
    /// scratch allocation, and the flattening copy needed to hand it one
    /// contiguous slice in the first place.
    ///
    /// The result is grouped, **not** sorted within a group: a target's updates
    /// arrive in whatever order the parallel scatter produced. That is enough
    /// for [`ApplySortedUpdates`], whose merge is a bounded insertion sort under
    /// a total order and so reaches the same row whatever the arrival order, and
    /// it is what lets the sort go away.
    ///
    /// ### Params
    ///
    /// * `batches` - Per-task update batches, read but not consumed
    /// * `n` - Number of nodes, bounding the target ids
    ///
    /// ### Returns
    ///
    /// The grouped batch, with every target's updates contiguous.
    pub fn group_counted(&mut self, batches: &[Vec<Update<T>>], n: usize) -> &[Update<T>] {
        let total: usize = batches.iter().map(|b| b.len()).sum();
        self.data.clear();
        if total == 0 {
            return &self.data;
        }
        debug_assert!(
            total <= u32::MAX as usize,
            "update batch exceeds u32 cursors"
        );

        let cursors = &self.cursors[..n];

        // Counts become write cursors in one pass: each slot takes the running
        // total of everything before it, which is its segment start.
        let mut acc: u32 = 0;
        for c in cursors.iter() {
            let count = c.load(Ordering::Relaxed);
            c.store(acc, Ordering::Relaxed);
            acc += count;
        }
        debug_assert_eq!(acc as usize, total);

        self.data.reserve(total);
        let out = UnsafeMutPtr(self.data.as_mut_ptr());
        batches.par_iter().for_each(|batch| {
            #[allow(clippy::redundant_locals)]
            let out = out;
            for u in batch.iter() {
                let pos = cursors[u.target as usize].fetch_add(1, Ordering::Relaxed) as usize;
                // SAFETY: the cursor for a target starts at its segment offset
                // and is bumped once per write, so every thread gets a distinct
                // slot inside a segment no other target touches. The segments
                // partition `0..total`, so every slot below is written exactly
                // once before `set_len` publishes it.
                unsafe { *out.0.add(pos) = *u };
            }
        });
        // SAFETY: the scatter above wrote every slot in `0..total`.
        unsafe { self.data.set_len(total) };

        &self.data
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

///////////
// Trait //
///////////

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

    /// Two-node graph at `k = 3` with the last slot of each row empty.
    fn padded_graph() -> Vec<(usize, f32)> {
        vec![
            (1, 0.5),
            (2, 1.5),
            (SENTINEL_PID, f32::MAX),
            (0, 0.5),
            (2, 2.5),
            (SENTINEL_PID, f32::MAX),
        ]
    }

    #[test]
    fn test_unpack_drops_sentinels() {
        let (ids, dists) = unpack_knn_graph(&padded_graph(), 2, 3, None, false, true);
        assert_eq!(ids, vec![vec![1, 2], vec![0, 2]]);
        assert_eq!(dists.unwrap(), vec![vec![0.5, 1.5], vec![0.5, 2.5]]);
    }

    #[test]
    fn test_unpack_honours_k_out() {
        let (ids, dists) = unpack_knn_graph(&padded_graph(), 2, 3, Some(1), false, true);
        assert_eq!(ids, vec![vec![1], vec![0]]);
        assert_eq!(dists.unwrap(), vec![vec![0.5], vec![0.5]]);
    }

    #[test]
    fn test_unpack_k_out_above_k_is_clamped() {
        let (ids, _) = unpack_knn_graph(&padded_graph(), 2, 3, Some(99), false, false);
        assert_eq!(ids, vec![vec![1, 2], vec![0, 2]]);
    }

    #[test]
    fn test_unpack_skips_distances_when_not_asked() {
        let (ids, dists) = unpack_knn_graph(&padded_graph(), 2, 3, None, false, false);
        assert_eq!(ids.len(), 2);
        assert!(dists.is_none());
    }

    #[test]
    fn test_unpack_all_sentinel_row_is_empty() {
        let graph = vec![(SENTINEL_PID, f32::MAX); 4];
        let (ids, dists) = unpack_knn_graph(&graph, 2, 2, None, false, true);
        assert_eq!(ids, vec![Vec::<usize>::new(), Vec::new()]);
        assert_eq!(dists.unwrap(), vec![Vec::<f32>::new(), Vec::new()]);
    }

    #[test]
    fn test_unpack_include_self_prepends_zero_edge() {
        let (ids, dists) = unpack_knn_graph(&padded_graph(), 2, 3, None, true, true);
        assert_eq!(ids, vec![vec![0, 1, 2], vec![1, 0, 2]]);
        assert_eq!(
            dists.unwrap(),
            vec![vec![0.0, 0.5, 1.5], vec![0.0, 0.5, 2.5]]
        );
    }

    #[test]
    fn test_unpack_include_self_counts_towards_k_out() {
        // `k_out` is the total row length, so asking for 2 with the self edge
        // gives self plus one true neighbour, matching a `query_*_self` at k=2.
        let (ids, dists) = unpack_knn_graph(&padded_graph(), 2, 3, Some(2), true, true);
        assert_eq!(ids, vec![vec![0, 1], vec![1, 0]]);
        assert_eq!(dists.unwrap(), vec![vec![0.0, 0.5], vec![0.0, 0.5]]);
    }

    #[test]
    fn test_unpack_include_self_on_an_empty_row_returns_only_self() {
        let graph = vec![(SENTINEL_PID, f32::MAX); 4];
        let (ids, dists) = unpack_knn_graph(&graph, 2, 2, None, true, true);
        assert_eq!(ids, vec![vec![0], vec![1]]);
        assert_eq!(dists.unwrap(), vec![vec![0.0], vec![0.0]]);
    }
}
