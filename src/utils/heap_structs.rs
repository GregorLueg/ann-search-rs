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
