//! k-means knobs shared by the IVF-family builders.

use ann_search_rs::prelude::KMeansTrainingParams;

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
