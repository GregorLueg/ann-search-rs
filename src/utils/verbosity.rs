//! Verbosity levels and the shared build/query progress lines.
//!
//! Every index prints the same grammar:
//!
//! ```text
//! Building <Name> index ...                     (Normal)
//!  <stage> in 1.23s                             (Detailed)
//!   Iteration 3: ...                            (Detailed)
//! ... built <Name> index in 4.56s               (Normal)
//! Querying <Name> index ...                     (Normal)
//!  Queried 100_000 / 1_000_000 samples          (Normal)
//! ... queried <Name> index in 1.03s             (Normal)
//! ```
//!
//! A sub-component (k-means inside IVF, NN-Descent inside NSG) is called with
//! [`Verbosity::Quiet`] and the parent prints one timed stage line for it, so
//! the indentation never depends on the nesting depth.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thousands::*;

/// Number of queried samples between two progress lines.
const PROGRESS_INTERVAL: usize = 100_000;

///////////////
// Verbosity //
///////////////

/// Controls how much a build or query prints
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Verbosity {
    /// Print nothing (warnings still print)
    #[default]
    Quiet,
    /// Start and end of builds and queries, plus query progress
    Normal,
    /// Normal, plus timed build stages and per-iteration lines
    Detailed,
}

impl Verbosity {
    /// Returns true if normal or detailed verbosity is set
    pub fn normal_verbosity(&self) -> bool {
        matches!(self, Verbosity::Normal | Verbosity::Detailed)
    }

    /// Returns true if detailed verbosity is set
    pub fn detailed_verbosity(&self) -> bool {
        matches!(self, Verbosity::Detailed)
    }
}

/// Parse a numeric verbosity level
///
/// For callers that carry verbosity as an integer (R and Python bindings,
/// downstream crates with their own enum).
///
/// ### Params
///
/// * `level` - `0` quiet, `1` normal, `2` detailed. Anything else is quiet.
///
/// ### Returns
///
/// The matching [`Verbosity`].
pub fn parse_annsearch_verbosity(level: usize) -> Verbosity {
    match level {
        1 => Verbosity::Normal,
        2 => Verbosity::Detailed,
        _ => Verbosity::Quiet,
    }
}

//////////////
// BuildLog //
//////////////

/// Prints the start and end line of an index build
pub(crate) struct BuildLog {
    /// What is built, as printed, e.g. `"IVF-PQ index"`
    name: &'static str,
    /// Verbosity of the build
    verbosity: Verbosity,
    /// Start of the build
    start: Instant,
}

impl BuildLog {
    /// Print the start line and start the clock
    ///
    /// ### Params
    ///
    /// * `name` - What is built or queried, as printed (`"HNSW index"`)
    /// * `verbosity` - Verbosity of the build
    ///
    /// ### Returns
    ///
    /// The running log; call [`BuildLog::finish`] at the end of the build.
    pub(crate) fn new(name: &'static str, verbosity: Verbosity) -> Self {
        if verbosity.normal_verbosity() {
            println!("Building {name} ...");
        }
        Self {
            name,
            verbosity,
            start: Instant::now(),
        }
    }

    /// Print the end line with the elapsed time
    pub(crate) fn finish(&self) {
        if self.verbosity.normal_verbosity() {
            println!("... built {} in {:.2?}", self.name, self.start.elapsed());
        }
    }
}

///////////////////
// QueryProgress //
///////////////////

/// Prints the start, progress and end lines of a batch query
///
/// Safe to [`tick`](QueryProgress::tick) from rayon workers.
pub(crate) struct QueryProgress {
    /// What is queried, as printed, e.g. `"HNSW index"`
    name: &'static str,
    /// `" (self)"` for self-queries, empty otherwise
    suffix: &'static str,
    /// Number of samples being queried
    total: usize,
    /// Verbosity of the query
    verbosity: Verbosity,
    /// Samples queried so far
    counter: AtomicUsize,
    /// Start of the query
    start: Instant,
}

impl QueryProgress {
    /// Print the start line and start the clock
    ///
    /// ### Params
    ///
    /// * `name` - What is built or queried, as printed (`"HNSW index"`)
    /// * `total` - Number of samples being queried
    /// * `self_query` - Whether the index is queried against itself
    /// * `verbosity` - Verbosity of the query
    ///
    /// ### Returns
    ///
    /// The running progress; call [`QueryProgress::finish`] at the end.
    pub(crate) fn new(
        name: &'static str,
        total: usize,
        self_query: bool,
        verbosity: Verbosity,
    ) -> Self {
        let suffix = if self_query { " (self)" } else { "" };
        if verbosity.normal_verbosity() {
            println!("Querying {name}{suffix} ...");
        }
        Self {
            name,
            suffix,
            total,
            verbosity,
            counter: AtomicUsize::new(0),
            start: Instant::now(),
        }
    }

    /// Progress lines only, no start or end line
    ///
    /// For pub helpers that cannot take a [`QueryProgress`] from their caller
    /// but still own the loop that knows how far the queries got.
    ///
    /// ### Params
    ///
    /// * `total` - Number of samples being queried
    /// * `verbosity` - Verbosity of the query
    ///
    /// ### Returns
    ///
    /// A progress that only ticks; do not call `finish` on it.
    #[cfg(feature = "gpu")]
    pub(crate) fn ticker(total: usize, verbosity: Verbosity) -> Self {
        Self {
            name: "",
            suffix: "",
            total,
            verbosity,
            counter: AtomicUsize::new(0),
            start: Instant::now(),
        }
    }

    /// Count `n` queried samples, printing a line at every
    /// `PROGRESS_INTERVAL` crossed
    ///
    /// ### Params
    ///
    /// * `n` - Number of samples just finished
    #[inline]
    pub(crate) fn tick_n(&self, n: usize) {
        if !self.verbosity.normal_verbosity() {
            return;
        }
        let before = self.counter.fetch_add(n, Ordering::Relaxed);
        let after = before + n;
        if after / PROGRESS_INTERVAL > before / PROGRESS_INTERVAL {
            println!(
                " Queried {} / {} samples",
                (after / PROGRESS_INTERVAL * PROGRESS_INTERVAL).separate_with_underscores(),
                self.total.separate_with_underscores()
            );
        }
    }

    /// Count one queried sample
    #[inline]
    pub(crate) fn tick(&self) {
        self.tick_n(1);
    }

    /// Print the end line with the elapsed time
    pub(crate) fn finish(&self) {
        if self.verbosity.normal_verbosity() {
            println!(
                "... queried {}{} in {:.2?}",
                self.name,
                self.suffix,
                self.start.elapsed()
            );
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_annsearch_verbosity_levels() {
        assert_eq!(parse_annsearch_verbosity(0), Verbosity::Quiet);
        assert_eq!(parse_annsearch_verbosity(1), Verbosity::Normal);
        assert_eq!(parse_annsearch_verbosity(2), Verbosity::Detailed);
        assert_eq!(parse_annsearch_verbosity(7), Verbosity::Quiet);
    }

    #[test]
    fn test_verbosity_levels_nest() {
        assert!(!Verbosity::Quiet.normal_verbosity());
        assert!(Verbosity::Normal.normal_verbosity());
        assert!(!Verbosity::Normal.detailed_verbosity());
        assert!(Verbosity::Detailed.normal_verbosity());
        assert!(Verbosity::Detailed.detailed_verbosity());
    }

    #[test]
    fn test_query_progress_counts_batches() {
        let p = QueryProgress::new("Test index", 250_000, false, Verbosity::Quiet);
        p.tick_n(150_000);
        // Quiet never counts
        assert_eq!(p.counter.load(Ordering::Relaxed), 0);

        let p = QueryProgress::new("Test index", 250_000, false, Verbosity::Normal);
        p.tick_n(150_000);
        p.tick();
        assert_eq!(p.counter.load(Ordering::Relaxed), 150_001);
    }
}
