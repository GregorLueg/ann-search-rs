//! IVF GPU handle.
//!
//! Inverted file with both halves on the device: k-means trains there, the
//! reordered vectors stay resident, and queries batch against the cells without
//! a readback. That residency is the reason this index is bounded by device
//! memory in a way the CPU IVF is not.

use ann_search_rs::gpu::ivf_gpu::IvfIndexGpu;
use ann_search_rs::{build_ivf_index_gpu, query_ivf_index_gpu, query_ivf_index_gpu_self};
use pyo3::prelude::*;

use crate::dispatch::QueryOut;
use crate::gpu_handle::{default_device, f32_array, gpu_handle, Rt};
use crate::kmeans::kmeans_gpu_params;

gpu_handle!(PyIvfGpu, IvfIndexGpu, "IvfGpu", {
    /// Train the cells on the device and leave the vectors there.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32.
    /// * `metric` - Already validated by the Python layer.
    /// * `nlist` - Voronoi cells, or `None` for the library's `sqrt(n)`
    ///   heuristic.
    /// * `kmeans_iters` - Lloyd iterations, or `None` for the crate default.
    /// * `kmeans_balanced` - Reseed starved centroids each iteration.
    /// * `quantise_to_f16` - Hold the training buffer at fp16 on the device.
    ///   Needs `shader-f16` on the adapter.
    /// * `seed` - Fixes the k-means initialisation.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// The built handle, or a clustering error.
    #[staticmethod]
    #[pyo3(signature = (
        x, *, metric, nlist = None, kmeans_iters = None, kmeans_balanced = false,
        quantise_to_f16 = false, seed = 42, verbose = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        metric: String,
        nlist: Option<usize>,
        kmeans_iters: Option<usize>,
        kmeans_balanced: bool,
        quantise_to_f16: bool,
        seed: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        let a = f32_array!(x);
        let (data, n, dim) = crate::convert::flat(&a)?;
        let kmp = kmeans_gpu_params(kmeans_iters, kmeans_balanced, quantise_to_f16);
        let inner = py
            .detach(|| {
                crate::pool::run(|| {
                    build_ivf_index_gpu::<f32, Rt>(
                        (data, n, dim),
                        nlist,
                        kmp,
                        &metric,
                        seed,
                        verbose,
                        default_device(),
                    )
                })
            })
            .map_err(crate::error::AnnErr)?;
        Ok(Self { inner, n, dim })
    }

    /// Probe the nearest cells for external queries.
    ///
    /// ### Params
    ///
    /// * `q` - Queries by features, C-contiguous float32.
    /// * `k` - Neighbours per query.
    /// * `nprobe` - Cells to visit, or `None` for the library's heuristic. The
    ///   main recall knob.
    /// * `nquery` - Queries to stage on the device per batch, or `None` for the
    ///   library's own chunking. Lower it if the upload does not fit.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)`, dense and padded. See [`QueryOut`].
    #[pyo3(signature = (
        q, k, *, nprobe = None, nquery = None, return_distance = true, verbose = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn query<'py>(
        &self,
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
        k: usize,
        nprobe: Option<usize>,
        nquery: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        let a = f32_array!(q);
        let (data, n, dim) = crate::convert::flat(&a)?;
        let idx = &self.inner;
        let (ids, dists) = py
            .detach(|| {
                crate::pool::run(|| {
                    query_ivf_index_gpu(
                        (data, n, dim),
                        idx,
                        k,
                        nprobe,
                        nquery,
                        return_distance,
                        verbose,
                    )
                })
            })
            .map_err(crate::error::AnnErr)?;
        crate::dispatch::pack(py, ids, dists, k)
    }

    /// Full kNN graph over the indexed data.
    ///
    /// Queries the resident copy of the vectors, so there is no upload.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per point.
    /// * `nprobe` - Cells to visit, or `None` for the heuristic.
    /// * `nquery` - Points to stage per batch, or `None` for the library's own
    ///   chunking.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` for every indexed point. See [`QueryOut`].
    #[pyo3(signature = (
        k, *, nprobe = None, nquery = None, return_distance = true, verbose = false
    ))]
    fn query_self<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        nprobe: Option<usize>,
        nquery: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        let idx = &self.inner;
        let (ids, dists) = py
            .detach(|| {
                crate::pool::run(|| {
                    query_ivf_index_gpu_self(idx, k, nprobe, nquery, return_distance, verbose)
                })
            })
            .map_err(crate::error::AnnErr)?;
        crate::dispatch::pack(py, ids, dists, k)
    }
});
