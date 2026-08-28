//! Thread control.
//!
//! Deliberately not `rayon::ThreadPoolBuilder::build_global()`: that succeeds at
//! most once per process and errors on every later call, which is hostile in a
//! notebook where someone will call it twice. A stored pool plus
//! `ThreadPool::install` can be rebuilt as often as the caller likes, and the
//! crate's internal `into_par_iter` picks the installed pool up.

use std::sync::{Arc, RwLock};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

//////////////////
// Shared state //
//////////////////

/// The pool `set_num_threads` installed, or `None` for rayon's global one.
static POOL: RwLock<Option<Arc<rayon::ThreadPool>>> = RwLock::new(None);

/// Message used when a lock is poisoned.
///
/// A poisoned lock means a panic inside `set_num_threads` while holding the
/// write guard, which cannot happen: the pool is built before the guard is
/// taken and the only work under the guard is one assignment.
const POISONED: &str = "thread pool lock poisoned";

/////////////////
// Entry point //
/////////////////

/// Run `f` on the configured pool, or on rayon's global pool if none is set.
///
/// Call this *inside* `Python::detach`, never around it: `Python<'py>` is not
/// `Send`, so a closure capturing it cannot satisfy the `Ungil` bound, and the
/// GIL should be dropped before the fan-out starts regardless.
///
/// ### Params
///
/// * `f` - The work to run. Must be `Send`; it is the whole build or query.
///
/// ### Returns
///
/// Whatever `f` returns. The uncontended read plus `Arc` clone is on the order
/// of tens of nanoseconds, against queries measured in microseconds.
pub(crate) fn run<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    let pool = POOL.read().expect(POISONED).clone();
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}

/////////////////
// Python API  //
/////////////////

/// Set the number of threads used for index builds and queries.
///
/// Rayon worker threads do not survive `fork`, so a `multiprocessing` child
/// using the default fork start method on Linux will hang if it touches the
/// pool. Use the `spawn` start method there.
///
/// ### Params
///
/// * `n` - Thread count. `0` clears the override and returns to rayon's global
///   pool, which honours `RAYON_NUM_THREADS`.
///
/// ### Returns
///
/// Nothing, or a `RuntimeError` if the pool could not be built.
#[pyfunction]
pub fn set_num_threads(n: usize) -> PyResult<()> {
    let new = if n == 0 {
        None
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("could not build thread pool: {e}")))?;
        Some(Arc::new(pool))
    };
    *POOL.write().expect(POISONED) = new;
    Ok(())
}

/// Number of threads currently available for index builds and queries.
///
/// ### Returns
///
/// The configured pool's size, or rayon's global pool size when no override is
/// in place.
#[pyfunction]
pub fn num_threads() -> usize {
    let pool = POOL.read().expect(POISONED).clone();
    match pool {
        Some(p) => p.current_num_threads(),
        None => rayon::current_num_threads(),
    }
}
