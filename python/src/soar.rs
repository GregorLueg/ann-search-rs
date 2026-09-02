//! SOAR handle.
//!
//! IVF with spilling: every point also lands in a second cell, chosen by a rule
//! that accounts for the residual it already has in its primary cell. Manhattan
//! is not supported.

use ann_search_rs::cpu::soar::SoarIndex;
use ann_search_rs::prelude::SoarRule;
use ann_search_rs::utils::k_means_utils::{DEFAULT_ORTHOGONAL_LAMBDA, DEFAULT_SHIFT_MU};
use ann_search_rs::{build_soar_index, query_soar_index, query_soar_self};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;
use crate::kmeans::kmeans_params;

/// Translate the spilling rule from its Python spelling.
///
/// [`SoarRule`] carries a payload, so it cannot cross the boundary as a plain
/// tag. `None` is passed straight through rather than resolved here: the
/// library picks per metric, `Orthogonal` for cosine and `Shifted` otherwise,
/// and duplicating that choice would let the two drift.
///
/// ### Params
///
/// * `rule` - `"nearest"`, `"shifted"` or `"orthogonal"`, or `None`.
/// * `param` - `mu` or `lambda` for the two parameterised rules, or `None` for
///   the library's own value. Ignored by `"nearest"`.
///
/// ### Returns
///
/// The rule to build with, or a message on an unknown name. The Python layer
/// validates first, so this is the backstop.
///
/// ### Note
///
/// The error is a `String` rather than a `PyErr` so the tests below run under a
/// plain `cargo test`. The test binary is not linked against libpython, so
/// touching `PyValueError` from one aborts on a flat-namespace symbol lookup at
/// load time.
pub(crate) fn soar_rule(rule: Option<&str>, param: Option<f64>) -> Result<Option<SoarRule>, String> {
    match rule {
        None => Ok(None),
        Some("nearest") => Ok(Some(SoarRule::Nearest)),
        Some("shifted") => Ok(Some(SoarRule::Shifted {
            mu: param.unwrap_or(DEFAULT_SHIFT_MU),
        })),
        Some("orthogonal") => Ok(Some(SoarRule::Orthogonal {
            lambda: param.unwrap_or(DEFAULT_ORTHOGONAL_LAMBDA),
        })),
        Some(other) => Err(format!(
            "unknown SOAR rule {other:?}; expected 'nearest', 'orthogonal' or 'shifted'"
        )),
    }
}

ann_handle!(PySoar, SoarInner, SoarIndex, "Soar", field, {
    /// Cluster the data and spill every point into a second cell.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer.
    /// * `nlist` - Voronoi cells, or `None` for the library's `sqrt(n)`
    ///   heuristic.
    /// * `rule` - Spilling rule, or `None` to let the library pick per metric.
    /// * `rule_param` - `mu` for `"shifted"`, `lambda` for `"orthogonal"`.
    /// * `kmeans_iters` - Lloyd iterations, or `None` for the crate default.
    /// * `kmeans_balanced` - Reseed starved centroids each iteration.
    /// * `seed` - Fixes the k-means initialisation.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// The built handle, or a clustering error.
    #[staticmethod]
    #[pyo3(signature = (
        x, *, metric, nlist = None, rule = None, rule_param = None, kmeans_iters = None,
        kmeans_balanced = false, seed = 42, verbose = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        metric: String,
        nlist: Option<usize>,
        rule: Option<&str>,
        rule_param: Option<f64>,
        kmeans_iters: Option<usize>,
        kmeans_balanced: bool,
        seed: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        let rule = soar_rule(rule, rule_param).map_err(PyValueError::new_err)?;
        let kmp = kmeans_params(kmeans_iters, kmeans_balanced);
        build_dispatch!(py, x, SoarInner, |data, n, dim| build_soar_index(
            (data, n, dim),
            nlist,
            rule,
            kmp,
            &metric,
            seed,
            verbose
        ))
    }

    /// Probe the nearest cells for external queries.
    ///
    /// ### Params
    ///
    /// * `q` - Queries by features, matching the index's float type.
    /// * `k` - Neighbours per query.
    /// * `nprobe` - Cells to visit, or `None` for the library's heuristic. The
    ///   main recall knob.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)`, dense and padded. See [`QueryOut`].
    #[pyo3(signature = (q, k, *, nprobe = None, return_distance = true, verbose = false))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
        k: usize,
        nprobe: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            SoarInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_soar_index((data, n, dim), idx, k, nprobe, return_distance, verbose)
            }),
            SoarInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_soar_index((data, n, dim), idx, k, nprobe, return_distance, verbose)
            }),
        }
    }

    /// Full kNN graph over the indexed data.
    ///
    /// Takes the same Voronoi-cell fast path as the plain IVF version.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per point.
    /// * `nprobe` - Cells to visit, or `None` for the heuristic.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` for every indexed point. See [`QueryOut`].
    #[pyo3(signature = (k, *, nprobe = None, return_distance = true, verbose = false))]
    fn query_self<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        nprobe: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            SoarInner::F32(idx) => {
                self_arm!(py, k, || query_soar_self(
                    idx,
                    k,
                    nprobe,
                    return_distance,
                    verbose
                ))
            }
            SoarInner::F64(idx) => {
                self_arm!(py, k, || query_soar_self(
                    idx,
                    k,
                    nprobe,
                    return_distance,
                    verbose
                ))
            }
        }
    }
});

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_rule_defers_to_the_library() {
        assert!(soar_rule(None, Some(0.7))
            .expect("None is always valid")
            .is_none());
    }

    #[test]
    fn test_missing_param_falls_back_to_the_crate_default() {
        let rule = soar_rule(Some("shifted"), None).expect("known rule");
        assert_eq!(
            rule,
            Some(SoarRule::Shifted {
                mu: DEFAULT_SHIFT_MU
            })
        );
    }

    #[test]
    fn test_param_is_carried_through() {
        let rule = soar_rule(Some("orthogonal"), Some(2.5)).expect("known rule");
        assert_eq!(rule, Some(SoarRule::Orthogonal { lambda: 2.5 }));
    }

    #[test]
    fn test_unknown_rule_is_rejected() {
        assert!(soar_rule(Some("nonsense"), None).is_err());
    }
}
