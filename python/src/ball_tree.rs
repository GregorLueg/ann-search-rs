//! BallTree handle.
//!
//! Metric tree of nested hyperspheres, pruned by the triangle inequality.
//! Manhattan is not supported.

use ann_search_rs::cpu::ball_tree::BallTreeIndex;
use ann_search_rs::{build_balltree_index, query_balltree_index, query_balltree_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(PyBallTree, BallTreeInner, BallTreeIndex, "BallTree", {
    /// Build the tree.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer.
    /// * `seed` - Fixes the pivot choice at each split.
    ///
    /// ### Returns
    ///
    /// The built handle, or a build error.
    #[staticmethod]
    #[pyo3(signature = (x, *, metric, seed = 42))]
    fn build(py: Python<'_>, x: &Bound<'_, PyAny>, metric: String, seed: usize) -> PyResult<Self> {
        build_dispatch!(py, x, BallTreeInner, |data, n, dim| build_balltree_index(
            (data, n, dim),
            &metric,
            seed
        ))
    }

    /// Descend the tree for external queries.
    ///
    /// ### Params
    ///
    /// * `q` - Queries by features, matching the index's float type.
    /// * `k` - Neighbours per query.
    /// * `search_budget` - Nodes to visit per query, or `None` for the
    ///   library's own heuristic of 5% of the indexed points. Raise for recall.
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
            BallTreeInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_balltree_index(
                    (data, n, dim),
                    idx,
                    k,
                    search_budget,
                    return_distance,
                    verbose,
                )
            }),
            BallTreeInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_balltree_index(
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
    /// * `search_budget` - Nodes to visit per point, or `None` for the
    ///   heuristic.
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
            BallTreeInner::F32(idx) => self_arm!(py, k, || query_balltree_self(
                idx,
                k,
                search_budget,
                return_distance,
                verbose
            )),
            BallTreeInner::F64(idx) => self_arm!(py, k, || query_balltree_self(
                idx,
                k,
                search_budget,
                return_distance,
                verbose
            )),
        }
    }
});
