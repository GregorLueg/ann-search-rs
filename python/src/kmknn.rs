//! kMkNN handle.
//!
//! Exact: the k-means partition prunes clusters by the triangle inequality
//! rather than approximating. No search-time knobs, and Manhattan is not
//! supported.

use ann_search_rs::cpu::kmknn::KmknnIndex;
use ann_search_rs::{build_kmknn_index, query_kmknn_index, query_kmknn_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;
use crate::kmeans::kmeans_params;

ann_handle!(PyKmknn, KmknnInner, KmknnIndex, "Kmknn", field, {
    /// Partition the data and build the pruning bounds.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer.
    /// * `nlist` - Clusters, or `None` for the library's `sqrt(n)` heuristic.
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
        x, *, metric, nlist = None, kmeans_iters = None, kmeans_balanced = false,
        seed = 42, verbose = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        metric: String,
        nlist: Option<usize>,
        kmeans_iters: Option<usize>,
        kmeans_balanced: bool,
        seed: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        let kmp = kmeans_params(kmeans_iters, kmeans_balanced);
        build_dispatch!(py, x, KmknnInner, |data, n, dim| build_kmknn_index(
            (data, n, dim),
            &metric,
            nlist,
            kmp,
            seed,
            verbose
        ))
    }

    /// Search for external queries.
    ///
    /// ### Params
    ///
    /// * `q` - Queries by features, matching the index's float type.
    /// * `k` - Neighbours per query.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)`, exact. See [`QueryOut`].
    #[pyo3(signature = (q, k, *, return_distance = true, verbose = false))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
        k: usize,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            KmknnInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_kmknn_index((data, n, dim), idx, k, return_distance, verbose)
            }),
            KmknnInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_kmknn_index((data, n, dim), idx, k, return_distance, verbose)
            }),
        }
    }

    /// Full kNN graph over the indexed data.
    ///
    /// The index reorders rows internally but maps back through its original
    /// ids, so the indices returned are input-row indices.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per point.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` for every indexed point. See [`QueryOut`].
    #[pyo3(signature = (k, *, return_distance = true, verbose = false))]
    fn query_self<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            KmknnInner::F32(idx) => {
                self_arm!(py, k, || query_kmknn_self(idx, k, return_distance, verbose))
            }
            KmknnInner::F64(idx) => {
                self_arm!(py, k, || query_kmknn_self(idx, k, return_distance, verbose))
            }
        }
    }
});
