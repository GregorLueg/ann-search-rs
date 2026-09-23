//! Quantised graph handle.
//!
//! A Vamana graph where every vertex carries its neighbours' one-bit RaBitQ
//! codes in the fast-scan layout. The float vectors stay resident for the exact
//! distances the walk anchors on, so this is a speed structure, not a small one.

use ann_search_rs::binary::qg::QgIndex;
use ann_search_rs::{build_qg_index, query_qg_index, query_qg_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(PyQg, QgInner, QgIndex, "Qg", method, {
    /// Build the Vamana graph and encode every vertex's neighbour block.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer. Manhattan is not
    ///   supported and the library rejects it.
    /// * `degree` - Neighbour slots per vertex, a non-zero multiple of 32. The
    ///   library rejects anything else.
    /// * `l_build` - Beam width for the second Vamana pass. The first runs at
    ///   the crate's narrower default.
    /// * `alpha_pass1` - Prune slack, first pass.
    /// * `alpha_pass2` - Prune slack, second pass.
    /// * `seed` - Fixes the graph and the rotation.
    ///
    /// ### Returns
    ///
    /// The built handle, or an unsupported-metric or invalid-degree error.
    #[staticmethod]
    #[pyo3(signature = (
        x, *, metric, degree, l_build, alpha_pass1, alpha_pass2, seed = 42
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        metric: String,
        degree: usize,
        l_build: usize,
        alpha_pass1: f32,
        alpha_pass2: f32,
        seed: usize,
    ) -> PyResult<Self> {
        build_dispatch!(py, x, QgInner, |data, n, dim| build_qg_index(
            (data, n, dim),
            degree,
            l_build,
            None,
            alpha_pass1,
            alpha_pass2,
            &metric,
            seed
        ))
    }

    /// Walk the graph for external queries.
    ///
    /// ### Params
    ///
    /// * `q` - Queries by features, matching the index's float type.
    /// * `k` - Neighbours per query.
    /// * `ef_search` - Beam width. Raise for recall.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)`. Distances are exact. See [`QueryOut`].
    #[pyo3(signature = (q, k, *, ef_search, return_distance = true, verbose = false))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
        k: usize,
        ef_search: usize,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            QgInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_qg_index((data, n, dim), idx, k, ef_search, return_distance, verbose)
            }),
            QgInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_qg_index((data, n, dim), idx, k, ef_search, return_distance, verbose)
            }),
        }
    }

    /// Full kNN graph over the indexed data.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per point.
    /// * `ef_search` - Beam width. Raise for recall.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` for every indexed point. See [`QueryOut`].
    #[pyo3(signature = (k, *, ef_search, return_distance = true, verbose = false))]
    fn query_self<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        ef_search: usize,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            QgInner::F32(idx) => {
                self_arm!(py, k, || query_qg_self(
                    idx,
                    k,
                    ef_search,
                    return_distance,
                    verbose
                ))
            }
            QgInner::F64(idx) => {
                self_arm!(py, k, || query_qg_self(
                    idx,
                    k,
                    ef_search,
                    return_distance,
                    verbose
                ))
            }
        }
    }
});
