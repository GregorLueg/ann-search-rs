//! IVF BF16 handle.
//!
//! The same inverted file as [`crate::ivf`], with the posting lists holding
//! `bf16` rather than `f32`. Roughly half the vector memory for a codec error
//! that lands well inside IVF's own approximation.

use ann_search_rs::prelude::DimensionValidation;
use ann_search_rs::quantised::ivf_bf16::IvfIndexBf16;
use ann_search_rs::{build_ivf_bf16_index, query_ivf_bf16_index, query_ivf_bf16_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;
use crate::kmeans::kmeans_params;

ann_handle!(PyIvfBf16, IvfBf16Inner, IvfIndexBf16, "IvfBf16", method, {
    /// Cluster the data and store the lists at `bf16`.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer. Manhattan is not
    ///   supported and the library rejects it.
    /// * `nlist` - Voronoi cells, or `None` for the library's `sqrt(n)`
    ///   heuristic.
    /// * `kmeans_iters` - Lloyd iterations, or `None` for the crate default.
    /// * `kmeans_balanced` - Reseed starved centroids each iteration.
    /// * `seed` - Fixes the k-means initialisation.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// The built handle, or a clustering or unsupported-metric error.
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
        build_dispatch!(py, x, IvfBf16Inner, |data, n, dim| build_ivf_bf16_index(
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
    /// * `nprobe` - Cells to visit, or `None` for the library's heuristic.
    ///   The main recall knob.
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
            IvfBf16Inner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_ivf_bf16_index((data, n, dim), idx, k, nprobe, return_distance, verbose)
            }),
            IvfBf16Inner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_ivf_bf16_index((data, n, dim), idx, k, nprobe, return_distance, verbose)
            }),
        }
    }

    /// Full kNN graph over the indexed data.
    ///
    /// Takes the Voronoi-cell fast path rather than re-entering from
    /// outside.
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
            IvfBf16Inner::F32(idx) => {
                self_arm!(py, k, || query_ivf_bf16_self(
                    idx,
                    k,
                    nprobe,
                    return_distance,
                    verbose
                ))
            }
            IvfBf16Inner::F64(idx) => {
                self_arm!(py, k, || query_ivf_bf16_self(
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
