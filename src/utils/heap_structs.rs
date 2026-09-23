//! Structures that can be kept on the heap or sorted buffers for situations
//! where data is small enough.

use num_traits::Float;

///////////////////
// Float on heap //
///////////////////

/// Faster than the sorts on full large vectors and allows to keep data on heap
#[derive(Clone, Copy, Debug)]
pub struct OrderedFloat<T>(pub T);

/// Partial equality trait
impl<T: Float> PartialEq for OrderedFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

/// Equality trait
impl<T: Float> Eq for OrderedFloat<T> {}

/// Partial ordering trait
impl<T: Float> PartialOrd for OrderedFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Comparing one to the other
impl<T: Float> Ord for OrderedFloat<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

//////////////////
// SortedBuffer //
//////////////////

/// Sorted buffer optimised for small result sets
///
/// Maintains elements in ascending order. For smaller data sets, this can
/// be faster than using heap.
///
/// - Better cache locality (sequential access)
/// - Fewer comparisons (binary search vs heap operations)
/// - No heap maintenance overhead
///
/// ### Type Parameters
///
/// * `T` - Element type, must implement `Ord`
pub struct SortedBuffer<T> {
    data: Vec<T>,
}

impl<T: Ord> SortedBuffer<T> {
    /// Create empty sorted buffer
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Create sorted buffer with pre-allocated capacity
    ///
    /// ### Params
    ///
    /// * `capacity` - Initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Number of elements
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reserve additional capacity
    ///
    /// ### Params
    ///
    /// * `additional` - Additional capacity to reserve
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Insert element maintaining sorted order
    ///
    /// If buffer is at capacity, only inserts if element is smaller
    /// than the largest element (and removes largest).
    ///
    /// ### Params
    ///
    /// * `item` - Element to insert
    /// * `limit` - Maximum capacity
    ///
    /// ### Returns
    ///
    /// `true` if inserted, `false` if rejected
    #[inline]
    pub fn insert(&mut self, item: T, limit: usize) -> bool {
        if self.data.len() < limit {
            let pos = self.data.binary_search(&item).unwrap_or_else(|e| e);
            self.data.insert(pos, item);
            true
        } else if let Some(last) = self.data.last() {
            if &item < last {
                let pos = self.data.binary_search(&item).unwrap_or_else(|e| e);
                self.data.pop();
                self.data.insert(pos, item);
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Get largest element (last in sorted order)
    #[inline]
    pub fn top(&self) -> Option<&T> {
        self.data.last()
    }

    /// Get all elements as slice
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Ensure ascending sort order
    ///
    /// No-op since buffer is always sorted.
    #[inline]
    pub fn sort_ascending(&mut self) {
        // Already sorted, no work needed
    }

    /// Number of elements
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/////////////////////
// BoundedMaxHeap //
/////////////////////

/// Bounded max-heap retaining the `k` smallest `(distance, index)` pairs.
///
/// Distances and indices live in parallel arrays with the largest retained
/// distance at position 0. A candidate that fails the threshold test costs one
/// float comparison and never touches the heap, which is the overwhelmingly
/// common case during a scan. An accepted candidate overwrites the root and
/// sifts down once, rather than the pop-then-push pair a [`std::collections::BinaryHeap`]
/// needs, halving the work on the accept path.
///
/// Ordering is by `(distance, index)`, so ties resolve towards the smaller
/// index and the emitted order is reproducible across runs and thread counts.
/// Sorting on the distance alone leaves tied entries in an arbitrary order,
/// which makes results impossible to diff against a reference implementation.
///
/// ### Type Parameters
///
/// * `T` - Float type of the distances
#[derive(Clone, Debug)]
pub struct BoundedMaxHeap<T> {
    /// Retained distances, heap-ordered with the largest at index 0
    dists: Vec<T>,
    /// Sample indices, permuted in lockstep with `dists`
    ids: Vec<usize>,
    /// Maximum number of entries retained
    k: usize,
    /// Distance at the root once full; `T::infinity()` while still filling
    threshold: T,
}

impl<T: Float> BoundedMaxHeap<T> {
    /// Create a heap retaining at most `k` entries
    ///
    /// ### Params
    ///
    /// * `k` - Maximum number of entries to retain
    ///
    /// ### Returns
    ///
    /// An empty heap with capacity for `k` entries
    pub fn new(k: usize) -> Self {
        Self {
            dists: Vec::with_capacity(k),
            ids: Vec::with_capacity(k),
            k,
            threshold: T::infinity(),
        }
    }

    /// Whether `(da, ia)` sorts after `(db, ib)`
    ///
    /// ### Params
    ///
    /// * `da` - Distance of the first entry
    /// * `ia` - Index of the first entry
    /// * `db` - Distance of the second entry
    /// * `ib` - Index of the second entry
    ///
    /// ### Returns
    ///
    /// `true` if the first entry is the larger of the two
    #[inline(always)]
    fn greater(da: T, ia: usize, db: T, ib: usize) -> bool {
        da > db || (da == db && ia > ib)
    }

    /// Offer a candidate to the heap
    ///
    /// While the heap is filling, every candidate is retained. Once full, the
    /// candidate is compared against the cached threshold and discarded unless
    /// it beats the current root.
    ///
    /// ### Params
    ///
    /// * `dist` - Distance of the candidate
    /// * `id` - Sample index of the candidate
    ///
    /// ### Returns
    ///
    /// `true` if the candidate was retained
    #[inline(always)]
    pub fn push(&mut self, dist: T, id: usize) -> bool {
        if self.k == 0 {
            return false;
        }

        if self.dists.len() < self.k {
            self.dists.push(dist);
            self.ids.push(id);
            self.sift_up(self.dists.len() - 1);
            if self.dists.len() == self.k {
                self.threshold = self.dists[0];
            }
            return true;
        }

        // Short-circuits on the float comparison alone unless the distances
        // tie exactly, so the reject path never loads `ids[0]`.
        if dist < self.threshold || (dist == self.threshold && id < self.ids[0]) {
            self.dists[0] = dist;
            self.ids[0] = id;
            self.sift_down(self.dists.len());
            self.threshold = self.dists[0];
            return true;
        }

        false
    }

    /// Distance of the current root, or infinity while the heap is filling
    ///
    /// ### Returns
    ///
    /// The distance a candidate must beat to be retained
    #[inline(always)]
    pub fn threshold(&self) -> T {
        self.threshold
    }

    /// Empty the heap, retaining its allocation and its `k`
    pub fn clear(&mut self) {
        self.dists.clear();
        self.ids.clear();
        self.threshold = T::infinity();
    }

    /// Empty the heap and set the number of entries it retains to `k`
    ///
    /// Keeps the existing allocation when it is already large enough, so a
    /// heap held in a thread-local scratch buffer allocates once per thread
    /// rather than once per search.
    ///
    /// ### Params
    ///
    /// * `k` - Maximum number of entries to retain from now on
    pub fn reset(&mut self, k: usize) {
        self.clear();
        self.dists.reserve(k.saturating_sub(self.dists.capacity()));
        self.ids.reserve(k.saturating_sub(self.ids.capacity()));
        self.k = k;
    }

    /// Retained distances, ascending only after [`BoundedMaxHeap::sort`]
    ///
    /// ### Returns
    ///
    /// The distance array in whatever order the heap currently holds it
    #[inline]
    pub fn dists(&self) -> &[T] {
        &self.dists
    }

    /// Retained sample indices, permuted in lockstep with [`BoundedMaxHeap::dists`]
    ///
    /// ### Returns
    ///
    /// The index array in whatever order the heap currently holds it
    #[inline]
    pub fn ids(&self) -> &[usize] {
        &self.ids
    }

    /// Sort the retained entries ascending by `(distance, index)` in place
    ///
    /// Heapsorts: repeatedly swaps the root to the back of the live region and
    /// shrinks it, which leaves both arrays ascending. Leaves the heap
    /// invariant broken, so the only valid operations afterwards are reading
    /// the arrays, [`BoundedMaxHeap::clear`] or [`BoundedMaxHeap::reset`].
    pub fn sort(&mut self) {
        for end in (1..self.dists.len()).rev() {
            self.dists.swap(0, end);
            self.ids.swap(0, end);
            self.sift_down(end);
        }
    }

    /// Number of retained entries
    ///
    /// ### Returns
    ///
    /// Current heap size
    pub fn len(&self) -> usize {
        self.dists.len()
    }

    /// Whether the heap holds no entries
    ///
    /// ### Returns
    ///
    /// `true` if empty
    pub fn is_empty(&self) -> bool {
        self.dists.is_empty()
    }

    /// Restore the heap invariant upwards from `start`
    ///
    /// ### Params
    ///
    /// * `start` - Position of the entry to move up
    #[inline]
    fn sift_up(&mut self, start: usize) {
        let mut i = start;
        let dist = self.dists[i];
        let id = self.ids[i];

        while i > 0 {
            let parent = (i - 1) / 2;
            if !Self::greater(dist, id, self.dists[parent], self.ids[parent]) {
                break;
            }
            self.dists[i] = self.dists[parent];
            self.ids[i] = self.ids[parent];
            i = parent;
        }

        self.dists[i] = dist;
        self.ids[i] = id;
    }

    /// Restore the heap invariant downwards from the root over `len` entries
    ///
    /// ### Params
    ///
    /// * `len` - Number of entries participating in the heap
    #[inline]
    fn sift_down(&mut self, len: usize) {
        let mut i = 0;
        let dist = self.dists[0];
        let id = self.ids[0];

        loop {
            let left = 2 * i + 1;
            if left >= len {
                break;
            }
            let right = left + 1;

            // Descend towards the larger child.
            let child = if right < len
                && Self::greater(
                    self.dists[right],
                    self.ids[right],
                    self.dists[left],
                    self.ids[left],
                ) {
                right
            } else {
                left
            };

            if !Self::greater(self.dists[child], self.ids[child], dist, id) {
                break;
            }

            self.dists[i] = self.dists[child];
            self.ids[i] = self.ids[child];
            i = child;
        }

        self.dists[i] = dist;
        self.ids[i] = id;
    }

    /// Consume the heap, returning entries sorted ascending by `(distance, index)`
    ///
    /// Heapsorts in place: repeatedly swaps the root to the back of the live
    /// region and shrinks it, which leaves the arrays ascending.
    ///
    /// ### Returns
    ///
    /// A tuple of `(indices, distances)`
    pub fn into_sorted(mut self) -> (Vec<usize>, Vec<T>) {
        self.sort();
        (self.ids, self.dists)
    }
}

/////////////////////
// NeighbourQueue  //
/////////////////////

/// Flag bit marking an entry as already expanded, stored in the id.
const EXPANDED_BIT: u32 = 1 << 31;

/// Mask recovering the row index from a flagged id.
const ID_MASK: u32 = !EXPANDED_BIT;

/// Bounded sorted candidate list for a greedy graph walk.
///
/// Holds at most `capacity` entries ascending by distance, each flagged
/// expanded or not, and hands back the closest unexpanded one. This is the
/// structure DiskANN's greedy search uses, and it does the job of the usual
/// frontier-heap-plus-result-heap pair on its own. That pair wastes most of
/// its work: the frontier accepts every candidate that beats the beam
/// threshold *at the time*, so it grows several times larger than the beam and
/// the majority of what it holds is never popped. A candidate outside the beam
/// can never be worth expanding, so it should never be stored.
///
/// Entries are `(distance, id)` with the expanded flag in the top bit of the
/// id, which keeps the row 8 bytes wide for `f32` and the insert memmove
/// correspondingly cheap. Ids must therefore stay below `1 << 31`; every graph
/// index here already guarantees that, since `u32::MAX` is its neighbour-slot
/// sentinel.
///
/// ### Type Parameters
///
/// * `T` - Float type of the distances
#[derive(Clone, Debug)]
pub struct NeighbourQueue<T> {
    /// Entries ascending by `(distance, id)`, at most `capacity` of them
    data: Vec<(T, u32)>,
    /// Maximum number of entries retained
    capacity: usize,
    /// Lowest position that may still hold an unexpanded entry
    cursor: usize,
}

impl<T: Float> NeighbourQueue<T> {
    /// Create a queue retaining at most `capacity` entries
    ///
    /// ### Params
    ///
    /// * `capacity` - Beam width
    ///
    /// ### Returns
    ///
    /// An empty queue with room for `capacity` entries
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
            cursor: 0,
        }
    }

    /// Empty the queue, keeping its allocation and its beam width
    pub fn clear(&mut self) {
        self.data.clear();
        self.cursor = 0;
    }

    /// Empty the queue and set the beam width, keeping the allocation
    ///
    /// ### Params
    ///
    /// * `capacity` - Beam width from now on
    pub fn reset(&mut self, capacity: usize) {
        self.data.clear();
        self.data
            .reserve(capacity.saturating_sub(self.data.capacity()));
        self.capacity = capacity;
        self.cursor = 0;
    }

    /// Distance a candidate must beat to be retained
    ///
    /// Infinity while the queue is still filling, so callers need no separate
    /// "not yet full" arm.
    ///
    /// ### Returns
    ///
    /// The current rejection threshold
    #[inline(always)]
    pub fn threshold(&self) -> T {
        if self.data.len() < self.capacity {
            T::infinity()
        } else {
            self.data[self.data.len() - 1].0
        }
    }

    /// Offer a candidate to the beam
    ///
    /// Rejects in `O(1)` once the queue is full and the candidate is no better
    /// than the worst entry, which is the common case. Otherwise binary
    /// searches for the insertion point and shifts the tail.
    ///
    /// ### Params
    ///
    /// * `dist` - Distance of the candidate
    /// * `id` - Row index of the candidate, below `1 << 31`
    ///
    /// ### Returns
    ///
    /// `true` if the candidate was retained
    #[inline]
    pub fn insert(&mut self, dist: T, id: u32) -> bool {
        let len = self.data.len();
        if len == self.capacity && (self.capacity == 0 || dist >= self.data[len - 1].0) {
            return false;
        }

        // Ordered by (distance, id) so ties resolve towards the smaller index
        // and the walk is reproducible across runs and thread counts. The
        // Deliberately branchy: the conditional-move form measured slower on
        // Apple Silicon, since the array is L1-resident, the predictor handles
        // this shape, and `csel` serialises the dependent loads.
        let mut lo = 0;
        let mut hi = len;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (d, flagged) = self.data[mid];
            if d < dist || (d == dist && (flagged & ID_MASK) < id) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let pos = lo;

        if len == self.capacity {
            self.data.pop();
        }
        self.data.insert(pos, (dist, id));
        if pos < self.cursor {
            self.cursor = pos;
        }
        true
    }

    /// Take the closest entry that has not been expanded yet
    ///
    /// Marks the returned entry expanded, so the walk terminates once every
    /// retained entry has been handed out once.
    ///
    /// ### Returns
    ///
    /// `(distance, id)` of the closest unexpanded entry, or `None`
    #[inline]
    pub fn next_unexpanded(&mut self) -> Option<(T, u32)> {
        while self.cursor < self.data.len() {
            let (dist, flagged) = self.data[self.cursor];
            self.cursor += 1;
            if flagged & EXPANDED_BIT == 0 {
                self.data[self.cursor - 1].1 = flagged | EXPANDED_BIT;
                return Some((dist, flagged));
            }
        }
        None
    }

    /// Retained entries, ascending by distance
    ///
    /// ### Returns
    ///
    /// Iterator over `(distance, id)` with the expanded flag stripped
    pub fn iter(&self) -> impl Iterator<Item = (T, u32)> + '_ {
        self.data.iter().map(|&(d, f)| (d, f & ID_MASK))
    }

    /// Number of retained entries
    ///
    /// ### Returns
    ///
    /// Current length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the queue holds nothing
    ///
    /// ### Returns
    ///
    /// `true` when empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference top-k: sort every candidate by `(distance, index)`.
    fn reference(items: &[(f32, usize)], k: usize) -> (Vec<usize>, Vec<f32>) {
        let mut v = items.to_vec();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        v.truncate(k);
        (
            v.iter().map(|x| x.1).collect(),
            v.iter().map(|x| x.0).collect(),
        )
    }

    #[test]
    fn test_bounded_heap_matches_full_sort() {
        let items: Vec<(f32, usize)> = (0..200)
            .map(|i| (((i * 37) % 101) as f32 * 0.5, i))
            .collect();

        for k in [1usize, 5, 15, 64] {
            let mut heap = BoundedMaxHeap::new(k);
            for &(d, i) in &items {
                heap.push(d, i);
            }
            assert_eq!(heap.len(), k);
            assert_eq!(heap.into_sorted(), reference(&items, k));
        }
    }

    #[test]
    fn test_bounded_heap_breaks_ties_on_index() {
        // Every distance ties, so the retained set is decided purely by index.
        let mut heap = BoundedMaxHeap::new(3);
        for i in (0..10).rev() {
            heap.push(1.0f32, i);
        }
        let (ids, dists) = heap.into_sorted();
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(dists, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_bounded_heap_insertion_order_does_not_matter() {
        let items: Vec<(f32, usize)> = (0..64).map(|i| (((i * 13) % 17) as f32, i)).collect();

        let mut forward = BoundedMaxHeap::new(7);
        for &(d, i) in &items {
            forward.push(d, i);
        }
        let mut reverse = BoundedMaxHeap::new(7);
        for &(d, i) in items.iter().rev() {
            reverse.push(d, i);
        }

        assert_eq!(forward.into_sorted(), reverse.into_sorted());
    }

    #[test]
    fn test_bounded_heap_fewer_items_than_k() {
        let mut heap = BoundedMaxHeap::new(10);
        heap.push(2.0f32, 1);
        heap.push(1.0f32, 0);
        assert_eq!(heap.len(), 2);
        assert_eq!(heap.into_sorted(), (vec![0, 1], vec![1.0, 2.0]));
    }

    #[test]
    fn test_bounded_heap_zero_k_retains_nothing() {
        let mut heap = BoundedMaxHeap::new(0);
        assert!(!heap.push(1.0f32, 0));
        assert!(heap.is_empty());
        assert_eq!(heap.into_sorted(), (Vec::new(), Vec::new()));
    }

    #[test]
    fn test_bounded_heap_threshold_tracks_root() {
        let mut heap = BoundedMaxHeap::new(3);
        assert_eq!(heap.threshold(), f32::INFINITY);
        for (d, i) in [(5.0f32, 0), (3.0, 1), (9.0, 2)] {
            heap.push(d, i);
        }
        assert_eq!(heap.threshold(), 9.0);
        assert!(heap.push(1.0, 3));
        assert_eq!(heap.threshold(), 5.0);
        assert!(!heap.push(7.0, 4));
        assert_eq!(heap.threshold(), 5.0);
    }

    #[test]
    fn test_neighbour_queue_keeps_best_and_orders() {
        let mut q = NeighbourQueue::new(3);
        for (d, i) in [(5.0f32, 0), (1.0, 1), (9.0, 2), (3.0, 3), (7.0, 4)] {
            q.insert(d, i);
        }
        let got: Vec<(f32, u32)> = q.iter().collect();
        assert_eq!(got, vec![(1.0, 1), (3.0, 3), (5.0, 0)]);
        assert_eq!(q.len(), 3);
    }

    #[test]
    fn test_neighbour_queue_threshold_and_rejection() {
        let mut q = NeighbourQueue::new(2);
        assert_eq!(q.threshold(), f32::INFINITY);
        assert!(q.insert(2.0f32, 0));
        assert_eq!(q.threshold(), f32::INFINITY);
        assert!(q.insert(4.0, 1));
        assert_eq!(q.threshold(), 4.0);
        assert!(!q.insert(4.0, 2));
        assert!(q.insert(1.0, 3));
        assert_eq!(q.threshold(), 2.0);
    }

    #[test]
    fn test_neighbour_queue_hands_out_each_entry_once() {
        let mut q = NeighbourQueue::new(4);
        for (d, i) in [(3.0f32, 30), (1.0, 10), (2.0, 20)] {
            q.insert(d, i);
        }
        let mut seen = Vec::new();
        while let Some((d, id)) = q.next_unexpanded() {
            seen.push((d, id));
        }
        assert_eq!(seen, vec![(1.0, 10), (2.0, 20), (3.0, 30)]);
        assert!(q.next_unexpanded().is_none());
        // Flags survive the walk, so the retained set is still readable.
        assert_eq!(
            q.iter().map(|(_, id)| id).collect::<Vec<_>>(),
            vec![10, 20, 30]
        );
    }

    #[test]
    fn test_neighbour_queue_reopens_on_closer_insert() {
        let mut q = NeighbourQueue::new(4);
        q.insert(5.0f32, 50);
        assert_eq!(q.next_unexpanded(), Some((5.0, 50)));
        // A closer candidate arriving after the walk moved on must be handed
        // out before the walk ends, not skipped.
        q.insert(1.0, 10);
        assert_eq!(q.next_unexpanded(), Some((1.0, 10)));
        assert!(q.next_unexpanded().is_none());
    }

    #[test]
    fn test_neighbour_queue_reset_clears() {
        let mut q = NeighbourQueue::new(2);
        q.insert(1.0f32, 0);
        q.next_unexpanded();
        q.reset(3);
        assert!(q.is_empty());
        assert_eq!(q.threshold(), f32::INFINITY);
        assert!(q.next_unexpanded().is_none());
    }

    #[test]
    fn test_neighbour_queue_zero_capacity() {
        let mut q = NeighbourQueue::new(0);
        assert!(!q.insert(1.0f32, 0));
        assert!(q.is_empty());
        assert!(q.next_unexpanded().is_none());
    }
}
