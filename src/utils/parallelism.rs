//! Contains helpers for parallelism and threading

use std::sync::atomic::{AtomicU64, Ordering};

/// Lock-free bitset for node-level locking during construction
///
/// Uses atomic u64 chunks to allow concurrent lock operations without blocking.
/// Each bit represents one node's lock state.
#[derive(Debug)]
pub struct AtomicNodeLocks {
    /// Vector of atomic u64 chunks, each storing 64 lock bits
    bits: Vec<AtomicU64>,
}

impl AtomicNodeLocks {
    /// Create a new lock bitset with capacity for n nodes
    ///
    /// Allocates enough u64 chunks to store one bit per node, rounding up
    /// to the nearest 64-node boundary.
    ///
    /// ### Params
    ///
    /// * `capacity` - Number of nodes to support
    ///
    /// ### Returns
    ///
    /// Initialised lock bitset with all locks released
    pub fn new(capacity: usize) -> Self {
        let num_slots = capacity.div_ceil(64);
        Self {
            bits: (0..num_slots).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Attempt to acquire a lock on a node
    ///
    /// Returns immediately without blocking. Uses compare-and-swap to
    /// atomically check and set the lock bit.
    ///
    /// ### Params
    ///
    /// * `idx` - Node index to lock
    ///
    /// ### Returns
    ///
    /// `true` if lock was already held, `false` if successfully acquired
    #[inline(always)]
    pub fn try_lock(&self, idx: usize) -> bool {
        let slot = idx / 64;
        let bit = 1u64 << (idx % 64);
        let prev = self.bits[slot].fetch_or(bit, Ordering::Acquire);
        (prev & bit) != 0
    }

    /// Acquire a lock on a node, spinning until successful
    ///
    /// Repeatedly calls try_lock until the lock is acquired. Uses spin_loop
    /// hint to reduce CPU contention.
    ///
    /// ### Params
    ///
    /// * `idx` - Node index to lock
    #[inline(always)]
    pub fn lock(&self, idx: usize) {
        while self.try_lock(idx) {
            std::hint::spin_loop();
        }
    }

    /// Release a lock on a node
    ///
    /// Atomically clears the lock bit, making it available for other threads.
    ///
    /// ### Params
    ///
    /// * `idx` - Node index to unlock
    #[inline(always)]
    pub fn unlock(&self, idx: usize) {
        let slot = idx / 64;
        let bit = !(1u64 << (idx % 64));
        self.bits[slot].fetch_and(bit, Ordering::Release);
    }

    /// Acquire a lock and return an RAII guard
    ///
    /// The lock is automatically released when the guard is dropped.
    ///
    /// ### Params
    ///
    /// * `idx` - Node index to lock
    ///
    /// ### Returns
    ///
    /// Guard that releases the lock on drop
    #[inline(always)]
    pub fn lock_guard(&self, idx: usize) -> NodeLockGuard<'_> {
        self.lock(idx);
        NodeLockGuard { locks: self, idx }
    }
}

/// RAII guard for automatic lock release
///
/// Holds a reference to the lock bitset and the locked node index.
/// Automatically releases the lock when dropped.
///
/// ### Fields
///
/// * `locks` - Reference to the parent lock bitset
/// * `idx` - Index of the locked node
pub struct NodeLockGuard<'a> {
    locks: &'a AtomicNodeLocks,
    idx: usize,
}

impl<'a> Drop for NodeLockGuard<'a> {
    /// Drop method
    fn drop(&mut self) {
        self.locks.unlock(self.idx);
    }
}
