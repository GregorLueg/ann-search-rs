//! Kd-tree forest handle.
//!
//! A forest of randomised spill trees. More trees means better recall and a
//! larger index, as with Annoy, but the splits are axis-aligned.

use ann_search_rs::cpu::kd_forest::KdTreeIndex;
use ann_search_rs::parse_annsearch_verbosity;
use ann_search_rs::{build_kd_tree_index, query_kd_tree_index, query_kd_tree_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(PyKdTree, KdTreeInner, KdTreeIndex, "KdTree", {
    /// Grow the forest.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer.
    /// * `n_trees` - Trees in the forest. Trades index size for recall.
    /// * `seed` - Fixes the split dimensions and the spill overlap.
    /// * `verbose` - `0` quiet, `1` normal, `2` detailed. Prints to the process
    ///   stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// The built handle. This builder cannot fail.
    #[staticmethod]
    #[pyo3(signature = (x, *, metric, n_trees, seed = 42, verbose = 0))]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        metric: String,
        n_trees: usize,
        seed: usize,
        verbose: usize,
    ) -> PyResult<Self> {
        build_dispatch!(py, x, KdTreeInner, |data, n, dim| {
            build_kd_tree_index(
                (data, n, dim),
                &metric,
                n_trees,
                seed,
                parse_annsearch_verbosity(verbose),
            )
        })
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
    /// * `verbose` - `0` quiet, `1` normal, `2` detailed. Prints to the process
    ///   stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)`, dense and padded. See [`QueryOut`].
    #[pyo3(signature = (q, k, *, search_budget = None, return_distance = true, verbose = 0))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
        k: usize,
        search_budget: Option<usize>,
        return_distance: bool,
        verbose: usize,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            KdTreeInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_kd_tree_index(
                    (data, n, dim),
                    idx,
                    k,
                    search_budget,
                    return_distance,
                    parse_annsearch_verbosity(verbose),
                )
            }),
            KdTreeInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_kd_tree_index(
                    (data, n, dim),
                    idx,
                    k,
                    search_budget,
                    return_distance,
                    parse_annsearch_verbosity(verbose),
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
    /// * `verbose` - `0` quiet, `1` normal, `2` detailed. Prints to the process
    ///   stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` for every indexed point. See [`QueryOut`].
    #[pyo3(signature = (k, *, search_budget = None, return_distance = true, verbose = 0))]
    fn query_self<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        search_budget: Option<usize>,
        return_distance: bool,
        verbose: usize,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            KdTreeInner::F32(idx) => self_arm!(py, k, || query_kd_tree_self(
                idx,
                k,
                search_budget,
                return_distance,
                parse_annsearch_verbosity(verbose)
            )),
            KdTreeInner::F64(idx) => self_arm!(py, k, || query_kd_tree_self(
                idx,
                k,
                search_budget,
                return_distance,
                parse_annsearch_verbosity(verbose)
            )),
        }
    }
});
