//! Contains helpers for parallelism and threading

use std::sync::atomic::{AtomicU8, Ordering};

/// Cache-line-padded striped spin-lock array for concurrent graph mutations
///
/// Maps node slots to lock stripes via Fibonacci hashing. The number of
/// stripes scales with `threads * connectivity` rather than with graph size,
/// keeping the lock array comfortably within L2/L3 cache regardless of how
/// many nodes the index holds. Each stripe occupies its own cache line so
/// that concurrent threads acquiring unrelated stripes do not suffer false
/// sharing.
///
/// Collisions (two unrelated nodes mapping to the same stripe) are possible
/// but rare with the default sizing and cause only transient serialisation
/// between otherwise-unrelated threads.
pub struct StripedLocks {
    stripes: Vec<PaddedLock>,
    shift: u32,
}

/// Single striped lock slot, padded to one cache line
///
/// The explicit `#[repr(align(64))]` prevents the atomic flag from sharing a
/// cache line with any neighbouring stripe, eliminating false sharing between
/// threads that acquire different stripes concurrently.
#[repr(align(64))]
struct PaddedLock {
    flag: AtomicU8,
    _pad: [u8; 63],
}

impl StripedLocks {
    /// Fibonacci hashing constant (golden ratio scaled to 64 bits)
    const FIB: u64 = 0x9E3779B97F4A7C15;

    /// Create a new striped lock array sized for the expected concurrency
    ///
    /// The stripe count is `threads * connectivity * 4`, rounded up to the
    /// next power of two and clamped to a minimum of 256. This gives enough
    /// headroom to keep collisions rare whilst staying small enough to fit
    /// in cache.
    ///
    /// ### Params
    ///
    /// * `threads` - Expected number of concurrent writers
    /// * `connectivity` - Base connectivity parameter (M)
    ///
    /// ### Returns
    ///
    /// Initialised lock array with all stripes unlocked
    pub fn new(threads: usize, connectivity: usize) -> Self {
        let desired = (threads * connectivity * 4).max(256);
        let count = desired.next_power_of_two();
        let shift = 64 - count.trailing_zeros();
        let stripes = (0..count)
            .map(|_| PaddedLock {
                flag: AtomicU8::new(0),
                _pad: [0; 63],
            })
            .collect();
        Self { stripes, shift }
    }

    /// Compute the stripe index for a node slot
    ///
    /// Uses Fibonacci hashing to distribute slots across stripes uniformly,
    /// even for sequential or clustered slot IDs.
    ///
    /// ### Params
    ///
    /// * `slot` - Node index
    ///
    /// ### Returns
    ///
    /// Index into the stripe array
    #[inline]
    fn stripe_for(&self, slot: usize) -> usize {
        ((slot as u64).wrapping_mul(Self::FIB) >> self.shift) as usize
    }

    /// Acquire the stripe lock for a given node
    ///
    /// Spins with a CPU hint until the flag transitions from unlocked to
    /// locked. Under oversubscription (more threads than cores) this will
    /// burn cycles; callers should size build thread pools to the core count.
    ///
    /// ### Params
    ///
    /// * `slot` - Node index whose stripe should be locked
    #[inline]
    pub fn lock(&self, slot: usize) {
        let s = &self.stripes[self.stripe_for(slot)];
        while s.flag.swap(1, Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }

    /// Release the stripe lock for a given node
    ///
    /// ### Params
    ///
    /// * `slot` - Node index whose stripe should be unlocked
    #[inline]
    pub fn unlock(&self, slot: usize) {
        self.stripes[self.stripe_for(slot)]
            .flag
            .store(0, std::sync::atomic::Ordering::Release);
    }

    /// Acquire a scoped lock guard for a node
    ///
    /// Returns an RAII guard that releases the stripe lock when dropped.
    /// Preferred over raw `lock`/`unlock` to avoid leaked locks on panic.
    ///
    /// ### Params
    ///
    /// * `slot` - Node index to lock
    ///
    /// ### Returns
    ///
    /// Guard that unlocks the stripe on drop
    #[inline]
    pub fn lock_guard(&self, slot: usize) -> StripedLockGuard<'_> {
        self.lock(slot);
        StripedLockGuard { locks: self, slot }
    }
}

/// RAII guard for a striped lock
///
/// Releases the underlying stripe lock when the guard is dropped, including
/// on panic unwind.
pub struct StripedLockGuard<'a> {
    locks: &'a StripedLocks,
    slot: usize,
}

impl<'a> Drop for StripedLockGuard<'a> {
    #[inline]
    fn drop(&mut self) {
        self.locks.unlock(self.slot);
    }
}
