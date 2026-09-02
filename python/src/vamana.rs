//! Vamana handle.
//!
//! The DiskANN graph: one flat graph of out-degree `r`, pruned in two passes
//! with the relaxed-neighbour rule. The alphas are `f32` in the library
//! regardless of the index element type, so no cast is needed here.

use ann_search_rs::cpu::vamana::VamanaIndex;
use ann_search_rs::prelude::AnnSearchErrors;
use ann_search_rs::{build_vamana_index, query_vamana_index, query_vamana_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(PyVamana, VamanaInner, VamanaIndex, "Vamana", field, {
    /// Build and prune the graph.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer.
    /// * `r` - Maximum out-degree.
    /// * `l_build` - Candidate list size during construction.
    /// * `alpha_pass1` - Relaxation factor for the first pruning pass.
    /// * `alpha_pass2` - Relaxation factor for the second. Above 1.0 keeps
    ///   longer edges, which shortens search paths.
    /// * `seed` - Fixes the entry point and traversal order.
    ///
    /// ### Returns
    ///
    /// The built handle. Infallible in the library, hence the `Ok` wrapper.
    #[staticmethod]
    #[pyo3(signature = (x, *, metric, r, l_build, alpha_pass1, alpha_pass2, seed = 42))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        metric: String,
        r: usize,
        l_build: usize,
        alpha_pass1: f32,
        alpha_pass2: f32,
        seed: usize,
    ) -> PyResult<Self> {
        build_dispatch!(py, x, VamanaInner, |data, n, dim| Ok::<_, AnnSearchErrors>(
            build_vamana_index(
                (data, n, dim),
                r,
                l_build,
                alpha_pass1,
                alpha_pass2,
                &metric,
                seed
            )
        ))
    }

    /// Greedy search from the entry point for external queries.
    ///
    /// ### Params
    ///
    /// * `q` - Queries by features, matching the index's float type.
    /// * `k` - Neighbours per query.
    /// * `ef_search` - Candidate list size, or `None` for the library's
    ///   heuristic. Raise for recall.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)`, dense and padded. See [`QueryOut`].
    #[pyo3(signature = (q, k, *, ef_search = None, return_distance = true, verbose = false))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
        k: usize,
        ef_search: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            VamanaInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_vamana_index((data, n, dim), idx, k, ef_search, return_distance, verbose)
            }),
            VamanaInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_vamana_index((data, n, dim), idx, k, ef_search, return_distance, verbose)
            }),
        }
    }

    /// Full kNN graph over the indexed data.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per point.
    /// * `ef_search` - Candidate list size, or `None` for the heuristic.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` for every indexed point. See [`QueryOut`].
    #[pyo3(signature = (k, *, ef_search = None, return_distance = true, verbose = false))]
    fn query_self<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        ef_search: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            VamanaInner::F32(idx) => self_arm!(py, k, || query_vamana_self(
                idx,
                k,
                ef_search,
                return_distance,
                verbose
            )),
            VamanaInner::F64(idx) => self_arm!(py, k, || query_vamana_self(
                idx,
                k,
                ef_search,
                return_distance,
                verbose
            )),
        }
    }
});
