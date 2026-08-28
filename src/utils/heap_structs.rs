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
        for end in (1..self.dists.len()).rev() {
            self.dists.swap(0, end);
            self.ids.swap(0, end);
            self.sift_down(end);
        }
        (self.ids, self.dists)
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
}
