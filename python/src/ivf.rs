//! IVF handle.
//!
//! Inverted file over k-means Voronoi cells. `nlist` sets how finely the space
//! is cut at build time, `nprobe` how many cells a query visits.

use ann_search_rs::cpu::ivf::IvfIndex;
use ann_search_rs::{build_ivf_index, query_ivf_index, query_ivf_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;
use crate::kmeans::kmeans_params;

ann_handle!(PyIvf, IvfInner, IvfIndex, "Ivf", {
    /// Cluster the data into the inverted lists.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer.
    /// * `nlist` - Voronoi cells, or `None` for the library's `sqrt(n)`
    ///   heuristic.
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
        build_dispatch!(py, x, IvfInner, |data, n, dim| build_ivf_index(
            (data, n, dim),
            nlist,
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
    /// * `verbose` - Progress to the process stdout.
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
            IvfInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_ivf_index((data, n, dim), idx, k, nprobe, return_distance, verbose)
            }),
            IvfInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_ivf_index((data, n, dim), idx, k, nprobe, return_distance, verbose)
            }),
        }
    }

    /// Full kNN graph over the indexed data.
    ///
    /// Exploits the Voronoi structure directly, visiting nearby cells per point
    /// rather than re-querying from outside. The index reorders rows internally
    /// but maps back through its original ids, so these are input-row indices.
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
            IvfInner::F32(idx) => {
                self_arm!(py, k, || query_ivf_self(
                    idx,
                    k,
                    nprobe,
                    return_distance,
                    verbose
                ))
            }
            IvfInner::F64(idx) => {
                self_arm!(py, k, || query_ivf_self(
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
