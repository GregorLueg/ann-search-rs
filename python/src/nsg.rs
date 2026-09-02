//! NSG handle.
//!
//! `build_nsg_index` runs its own NN-Descent internally to get the kNN graph it
//! then refines, which is why `knn_k` is a build cost rather than a query knob.
//! The library also exposes `build_nsg_from_knn_index` for reusing an existing
//! NN-Descent index; that path is not bound yet.

use ann_search_rs::cpu::nsg::NsgIndex;
use ann_search_rs::{build_nsg_index, query_nsg_index, query_nsg_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(PyNsg, NsgInner, NsgIndex, "Nsg", {
    /// Build the kNN graph, then refine it into the navigating spreading-out
    /// graph.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer.
    /// * `r` - Maximum out-degree of the refined graph.
    /// * `l_build` - Candidate list size during refinement.
    /// * `c` - Candidate pool cap per node.
    /// * `knn_k` - Neighbours in the intermediate NN-Descent graph.
    /// * `seed` - Fixes the NN-Descent initialisation and the navigating node.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// The built handle, or a build error.
    #[staticmethod]
    #[pyo3(signature = (x, *, metric, r, l_build, c, knn_k, seed = 42, verbose = false))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        metric: String,
        r: usize,
        l_build: usize,
        c: usize,
        knn_k: usize,
        seed: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        build_dispatch!(py, x, NsgInner, |data, n, dim| build_nsg_index(
            (data, n, dim),
            r,
            l_build,
            c,
            knn_k,
            &metric,
            seed,
            verbose
        ))
    }

    /// Search from the navigating node for external queries.
    ///
    /// ### Params
    ///
    /// * `q` - Queries by features, matching the index's float type.
    /// * `k` - Neighbours per query.
    /// * `ef_search` - Candidate list size, or `None` for the library's
    ///   heuristic. Raise for recall.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
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
            NsgInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_nsg_index((data, n, dim), idx, k, ef_search, return_distance, verbose)
            }),
            NsgInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_nsg_index((data, n, dim), idx, k, ef_search, return_distance, verbose)
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
            NsgInner::F32(idx) => {
                self_arm!(py, k, || query_nsg_self(
                    idx,
                    k,
                    ef_search,
                    return_distance,
                    verbose
                ))
            }
            NsgInner::F64(idx) => {
                self_arm!(py, k, || query_nsg_self(
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
