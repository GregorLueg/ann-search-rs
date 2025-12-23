use num_traits::Float;

///////////////////
// Float on heap //
///////////////////

/// Wrapper for f32 that implements Ord for use in BinaryHeap
///
/// Faster than the sorts on full vectors and allows to keep data on heap
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
