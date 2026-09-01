//! k-means knobs shared by the IVF-family builders.
//!
//! The CPU and GPU builders take different params structs, so there is one
//! assembler each. Both follow the same rule: `None` means "let the crate pick
//! its own heuristic", which is not the same as passing the crate's defaults
//! explicitly, so a struct is only built when the caller actually asked for
//! something.

use ann_search_rs::prelude::KMeansTrainingParams;

#[cfg(feature = "gpu")]
use ann_search_rs::gpu::k_means_gpu::KMeansGpuParams;

/// Assemble [`KMeansTrainingParams`] from the two knobs worth exposing.
///
/// `None` means "let the crate pick its own heuristic", which is not the same
/// as passing the crate's defaults explicitly, so a struct is only built when
/// the caller actually asked for something. `init` and `LloydPath` are left on
/// the crate's defaults for now; they can be surfaced as strings later if
/// anyone needs to pin them.
///
/// ### Params
///
/// * `iters` - Lloyd iterations, or `None` for the crate default.
/// * `balanced` - Reseed starved centroids each iteration. Off by default
///   because it changes the partition, and therefore every index built on it.
///
/// ### Returns
///
/// `None` when neither knob was touched, so the builder takes its own path.
pub(crate) fn kmeans_params(iters: Option<usize>, balanced: bool) -> Option<KMeansTrainingParams> {
    if iters.is_none() && !balanced {
        return None;
    }
    let base = match iters {
        Some(i) => KMeansTrainingParams::new(i, None, None),
        None => KMeansTrainingParams::default(),
    };
    Some(base.with_balancing(balanced))
}

/// Assemble [`KMeansGpuParams`] from the knobs worth exposing.
///
/// The GPU struct is a different type from the CPU one, not a superset: both
/// halves of the partitioning run on device, so the Lloyd path and the fixed
/// iteration count mean something different here.
///
/// ### Params
///
/// * `iters` - Lloyd iterations, or `None` for the crate default.
/// * `balanced` - Reseed starved centroids each iteration, RAFT-style.
/// * `quantise_to_f16` - Hold the data buffer on the device at fp16. Halves its
///   memory and lifts effective bandwidth on the assignment kernels, at the
///   cost of needing `shader-f16` on the adapter.
///
/// ### Returns
///
/// `None` when no knob was touched, so the builder takes its own path.
///
/// ### Note
///
/// `fixed` is left off, so the loop still checks convergence. Pinning it would
/// make `iters` mean "always run this many", which is a benchmarking knob
/// rather than a user-facing one.
#[cfg(feature = "gpu")]
pub(crate) fn kmeans_gpu_params(
    iters: Option<usize>,
    balanced: bool,
    quantise_to_f16: bool,
) -> Option<KMeansGpuParams> {
    if iters.is_none() && !balanced && !quantise_to_f16 {
        return None;
    }
    let base = match iters {
        Some(i) => KMeansGpuParams::new(i, None, false, quantise_to_f16),
        None => KMeansGpuParams {
            quantise_to_f16,
            ..KMeansGpuParams::default()
        },
    };
    Some(base.with_balancing(balanced))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_untouched_knobs_defer_to_the_crate() {
        assert!(kmeans_params(None, false).is_none());
    }

    #[test]
    fn test_balancing_alone_keeps_the_default_iterations() {
        let params = kmeans_params(None, true).expect("balancing forces a params struct");
        assert_eq!(params.iters, KMeansTrainingParams::default().iters);
        assert!(params.balanced);
    }

    #[test]
    fn test_iterations_are_carried_through() {
        let params = kmeans_params(Some(7), false).expect("iters forces a params struct");
        assert_eq!(params.iters, 7);
        assert!(!params.balanced);
    }
}
