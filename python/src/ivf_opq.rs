//! IVF-OPQ handle.
//!
//! [`crate::ivf_pq`] with the learned rotation in front of the sub-codebooks.
//! Worth it when the axes carry wildly different variance, which is the usual
//! case for a raw embedding space and rarely the case after a PCA.

use ann_search_rs::prelude::DimensionValidation;
use ann_search_rs::quantised::ivf_opq::IvfOpqIndex;
use ann_search_rs::{build_ivf_opq_index, query_ivf_opq_index, query_ivf_opq_index_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;
use crate::kmeans::kmeans_params;

ann_handle!(PyIvfOpq, IvfOpqInner, IvfOpqIndex, "IvfOpq", method, {
    /// Cluster, learn the rotation, then encode each rotated residual.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `m` - Subspaces. `dim` must divide by it.
    /// * `metric` - Already validated by the Python layer. Manhattan is not
    ///   supported and the library rejects it.
    /// * `nlist` - Voronoi cells, or `None` for the library's `sqrt(n)`
    ///   heuristic.
    /// * `kmeans_iters` - Lloyd iterations for the cells, or `None` for the
    ///   crate default.
    /// * `kmeans_balanced` - Reseed starved centroids each iteration.
    /// * `n_pq_centroids` - Centroids per subspace, or `None` for the
    ///   crate default of 256.
    /// * `opq_iters` - Alternating rotation/codebook iterations, or `None`
    ///   for the crate default. This is the build-cost knob.
    /// * `seed` - Fixes both the cell and the codebook initialisation.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// The built handle, or a clustering, codebook or unsupported-metric
    /// error.
    #[staticmethod]
    #[pyo3(signature = (
            x, *, m, metric, nlist = None, kmeans_iters = None, kmeans_balanced = false,
            n_pq_centroids = None, opq_iters = None, seed = 42, verbose = false
        ))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        m: usize,
        metric: String,
        nlist: Option<usize>,
        kmeans_iters: Option<usize>,
        kmeans_balanced: bool,
        n_pq_centroids: Option<usize>,
        opq_iters: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        let kmp = kmeans_params(kmeans_iters, kmeans_balanced);
        build_dispatch!(py, x, IvfOpqInner, |data, n, dim| build_ivf_opq_index(
            (data, n, dim),
            nlist,
            m,
            kmp,
            n_pq_centroids,
            opq_iters,
            &metric,
            seed,
            verbose
        ))
    }

    /// Probe the nearest cells and score their codes.
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
    /// `(indices, distances)`. Distances are the codec's estimate. See
    /// [`QueryOut`].
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
            IvfOpqInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_ivf_opq_index((data, n, dim), idx, k, nprobe, return_distance, verbose)
            }),
            IvfOpqInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_ivf_opq_index((data, n, dim), idx, k, nprobe, return_distance, verbose)
            }),
        }
    }

    /// Full kNN graph over the indexed data.
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
            IvfOpqInner::F32(idx) => {
                self_arm!(py, k, || query_ivf_opq_index_self(
                    idx,
                    k,
                    nprobe,
                    return_distance,
                    verbose
                ))
            }
            IvfOpqInner::F64(idx) => {
                self_arm!(py, k, || query_ivf_opq_index_self(
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
