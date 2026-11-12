//! k-means knobs shared by the IVF-family builders.
//!
//! The CPU and GPU builders take different params structs, so there is one
//! assembler each. Both follow the same rule: `None` means "let the crate pick
//! its own default", which is e-means at its default epoch budget, so a struct
//! is only built when the caller actually asked for something.
//!
//! `kmeans_iters` and `kmeans_balanced` only mean something to Lloyd's, and
//! `kmeans_epochs` only to e-means. The crate silently ignores the knobs that do
//! not apply, so mixing them up is rejected here instead.

use ann_search_rs::prelude::{EMeansParams, KMeansAlgorithm, KMeansTrainingParams};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(feature = "gpu")]
use ann_search_rs::gpu::k_means_gpu::KMeansGpuParams;

/// Lloyd's iteration cap when the caller picks Lloyd's without `kmeans_iters`.
/// Matches the iteration count the crate's own defaults carry.
const LLOYD_DEFAULT_ITERS: usize = 30;

/// The algorithm a caller asked for, with only the knobs that apply to it.
enum Choice {
    /// e-means, with the epoch budget or `None` for the crate default
    EMeans(Option<f64>),
    /// Lloyd's, with the iteration cap and the balancing flag
    Lloyd(usize, bool),
}

/// Parse the algorithm string and reject knobs that do not apply to it.
///
/// ### Params
///
/// * `algorithm` - `"emeans"` or `"lloyd"`.
/// * `epochs` - e-means epochs, or `None`. Lloyd's only.
/// * `iters` - Lloyd iterations, or `None`. e-means only.
/// * `balanced` - Reseed starved centroids each iteration. Lloyd's only.
///
/// ### Returns
///
/// The resolved [`Choice`], or a `ValueError` naming the offending knob.
fn resolve(
    algorithm: &str,
    epochs: Option<f64>,
    iters: Option<usize>,
    balanced: bool,
) -> PyResult<Choice> {
    match algorithm {
        "emeans" => {
            if iters.is_some() || balanced {
                return Err(PyValueError::new_err(
                    "kmeans_iters and kmeans_balanced only apply to \
                     kmeans_algorithm='lloyd'; e-means takes kmeans_epochs",
                ));
            }
            Ok(Choice::EMeans(epochs))
        }
        "lloyd" => {
            if epochs.is_some() {
                return Err(PyValueError::new_err(
                    "kmeans_epochs only applies to kmeans_algorithm='emeans'",
                ));
            }
            Ok(Choice::Lloyd(
                iters.unwrap_or(LLOYD_DEFAULT_ITERS),
                balanced,
            ))
        }
        other => Err(PyValueError::new_err(format!(
            "kmeans_algorithm must be 'emeans' or 'lloyd', got '{other}'"
        ))),
    }
}

/// Assemble [`KMeansTrainingParams`] from the knobs worth exposing.
///
/// `init` and `LloydPath` are left on the crate's defaults; they can be
/// surfaced as strings later if anyone needs to pin them.
///
/// ### Params
///
/// * `algorithm` - `"emeans"` or `"lloyd"`.
/// * `epochs` - e-means epochs, or `None` for the crate default.
/// * `iters` - Lloyd iterations, or `None` for 30.
/// * `balanced` - Reseed starved centroids each iteration. Off by default
///   because it changes the partition, and therefore every index built on it.
///
/// ### Returns
///
/// `None` for e-means at its defaults, so the builder takes its own path, or a
/// `ValueError` when a knob does not apply to the chosen algorithm.
pub(crate) fn kmeans_params(
    algorithm: &str,
    epochs: Option<f64>,
    iters: Option<usize>,
    balanced: bool,
) -> PyResult<Option<KMeansTrainingParams>> {
    Ok(match resolve(algorithm, epochs, iters, balanced)? {
        Choice::EMeans(None) => None,
        Choice::EMeans(Some(e)) => Some(
            KMeansTrainingParams::default()
                .with_algorithm(KMeansAlgorithm::EMeans(EMeansParams::new(e))),
        ),
        Choice::Lloyd(i, b) => Some(KMeansTrainingParams::new(i, None, None).with_balancing(b)),
    })
}

/// Assemble [`KMeansGpuParams`] from the knobs worth exposing.
///
/// The GPU struct is a different type from the CPU one, not a superset: both
/// halves of the partitioning run on device, so the Lloyd path and the fixed
/// iteration count mean something different here.
///
/// ### Params
///
/// * `algorithm` - `"emeans"` or `"lloyd"`.
/// * `epochs` - e-means epochs, or `None` for the crate default.
/// * `iters` - Lloyd iterations, or `None` for 30.
/// * `balanced` - Reseed starved centroids each iteration, RAFT-style.
/// * `quantise_to_f16` - Hold the data buffer on the device at fp16. Halves its
///   memory and lifts effective bandwidth on the assignment kernels, at the
///   cost of needing `shader-f16` on the adapter. Applies to both algorithms.
///
/// ### Returns
///
/// `None` for e-means at its defaults without fp16, so the builder takes its
/// own path, or a `ValueError` when a knob does not apply.
///
/// ### Note
///
/// Lloyd's `fixed` is left off, so the loop still checks convergence. Pinning
/// it would make `iters` mean "always run this many", which is a benchmarking
/// knob rather than a user-facing one.
#[cfg(feature = "gpu")]
pub(crate) fn kmeans_gpu_params(
    algorithm: &str,
    epochs: Option<f64>,
    iters: Option<usize>,
    balanced: bool,
    quantise_to_f16: bool,
) -> PyResult<Option<KMeansGpuParams>> {
    Ok(match resolve(algorithm, epochs, iters, balanced)? {
        Choice::EMeans(None) if !quantise_to_f16 => None,
        Choice::EMeans(e) => {
            let em = e.map(EMeansParams::new).unwrap_or_default();
            Some(
                KMeansGpuParams {
                    quantise_to_f16,
                    ..KMeansGpuParams::default()
                }
                .with_algorithm(KMeansAlgorithm::EMeans(em)),
            )
        }
        Choice::Lloyd(i, b) => {
            Some(KMeansGpuParams::new(i, None, false, quantise_to_f16).with_balancing(b))
        }
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_untouched_knobs_defer_to_the_crate() {
        assert!(kmeans_params("emeans", None, None, false)
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_epochs_are_carried_through() {
        let params = kmeans_params("emeans", Some(2.5), None, false)
            .unwrap()
            .expect("epochs force a params struct");
        assert_eq!(
            params.algorithm,
            KMeansAlgorithm::EMeans(EMeansParams::new(2.5))
        );
    }

    #[test]
    fn test_lloyd_carries_iterations_and_balancing() {
        let params = kmeans_params("lloyd", None, Some(7), true)
            .unwrap()
            .expect("lloyd forces a params struct");
        assert_eq!(params.algorithm, KMeansAlgorithm::Lloyd);
        assert_eq!(params.iters, 7);
        assert!(params.balanced);
    }

    #[test]
    fn test_lloyd_without_iterations_uses_the_default_cap() {
        let params = kmeans_params("lloyd", None, None, false)
            .unwrap()
            .expect("lloyd forces a params struct");
        assert_eq!(params.iters, LLOYD_DEFAULT_ITERS);
    }

    #[test]
    fn test_mismatched_knobs_are_rejected() {
        assert!(kmeans_params("emeans", None, Some(10), false).is_err());
        assert!(kmeans_params("emeans", None, None, true).is_err());
        assert!(kmeans_params("lloyd", Some(5.0), None, false).is_err());
        assert!(kmeans_params("hamerly", None, None, false).is_err());
    }
}
