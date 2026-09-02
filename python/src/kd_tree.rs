//! Kd-tree forest handle.
//!
//! A forest of randomised spill trees. More trees means better recall and a
//! larger index, as with Annoy, but the splits are axis-aligned.

use ann_search_rs::cpu::kd_forest::KdTreeIndex;
use ann_search_rs::prelude::AnnSearchErrors;
use ann_search_rs::{build_kd_tree_index, query_kd_tree_index, query_kd_tree_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(PyKdTree, KdTreeInner, KdTreeIndex, "KdTree", field, {
    /// Grow the forest.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer.
    /// * `n_trees` - Trees in the forest. Trades index size for recall.
    /// * `seed` - Fixes the split dimensions and the spill overlap.
    ///
    /// ### Returns
    ///
    /// The built handle. This builder cannot fail.
    #[staticmethod]
    #[pyo3(signature = (x, *, metric, n_trees, seed = 42))]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        metric: String,
        n_trees: usize,
        seed: usize,
    ) -> PyResult<Self> {
        build_dispatch!(py, x, KdTreeInner, |data, n, dim| Ok::<_, AnnSearchErrors>(
            build_kd_tree_index((data, n, dim), &metric, n_trees, seed)
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
            KdTreeInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_kd_tree_index(
                    (data, n, dim),
                    idx,
                    k,
                    search_budget,
                    return_distance,
                    verbose,
                )
            }),
            KdTreeInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_kd_tree_index(
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
            KdTreeInner::F32(idx) => self_arm!(py, k, || query_kd_tree_self(
                idx,
                k,
                search_budget,
                return_distance,
                verbose
            )),
            KdTreeInner::F64(idx) => self_arm!(py, k, || query_kd_tree_self(
                idx,
                k,
                search_budget,
                return_distance,
                verbose
            )),
        }
    }
});
