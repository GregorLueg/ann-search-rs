//! Annoy handle.
//!
//! Random projection forest. More trees means better recall and a larger
//! index. Manhattan is not supported.

use ann_search_rs::cpu::annoy::AnnoyIndex;
use ann_search_rs::{build_annoy_index, query_annoy_index, query_annoy_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(PyAnnoy, AnnoyInner, AnnoyIndex, "Annoy", {
    /// Grow the forest.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer.
    /// * `n_trees` - Trees in the forest. Trades index size for recall.
    /// * `seed` - Fixes the random split hyperplanes.
    ///
    /// ### Returns
    ///
    /// The built handle, or a build error.
    #[staticmethod]
    #[pyo3(signature = (x, *, metric, n_trees, seed = 42))]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        metric: String,
        n_trees: usize,
        seed: usize,
    ) -> PyResult<Self> {
        build_dispatch!(py, x, AnnoyInner, |data, n, dim| build_annoy_index(
            (data, n, dim),
            &metric,
            n_trees,
            seed
        ))
    }

    /// Descend the forest for external queries.
    ///
    /// ### Params
    ///
    /// * `q` - Queries by features, matching the index's float type.
    /// * `k` - Neighbours per query.
    /// * `search_budget` - Candidates to inspect, or `None` for the library's
    ///   own heuristic. Raise for recall.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)`, dense and padded. See [`QueryOut`].
    #[pyo3(signature = (q, k, *, search_budget = None, return_distance = true, verbose = false))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
        k: usize,
        search_budget: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            AnnoyInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_annoy_index(
                    (data, n, dim),
                    idx,
                    k,
                    search_budget,
                    return_distance,
                    verbose,
                )
            }),
            AnnoyInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_annoy_index(
                    (data, n, dim),
                    idx,
                    k,
                    search_budget,
                    return_distance,
                    verbose,
                )
            }),
        }
    }

    /// Full kNN graph over the indexed data.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per point.
    /// * `search_budget` - Candidates to inspect, or `None` for the heuristic.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` for every indexed point. See [`QueryOut`].
    #[pyo3(signature = (k, *, search_budget = None, return_distance = true, verbose = false))]
    fn query_self<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        search_budget: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            AnnoyInner::F32(idx) => self_arm!(py, k, || query_annoy_self(
                idx,
                k,
                search_budget,
                return_distance,
                verbose
            )),
            AnnoyInner::F64(idx) => self_arm!(py, k, || query_annoy_self(
                idx,
                k,
                search_budget,
                return_distance,
                verbose
            )),
        }
    }
});
