//! Exhaustive GPU handle.
//!
//! Brute force on the device. Exact by construction, and the cheapest way to
//! get ground truth for a dataset too large to score on the CPU.

use ann_search_rs::gpu::exhaustive_gpu::ExhaustiveIndexGpu;
use ann_search_rs::{
    build_exhaustive_index_gpu, query_exhaustive_index_gpu, query_exhaustive_index_gpu_self,
};
use pyo3::prelude::*;

use crate::dispatch::QueryOut;
use crate::gpu_handle::{default_device, f32_array, gpu_handle, Rt};

gpu_handle!(PyExhaustiveGpu, ExhaustiveIndexGpu, "ExhaustiveGpu", {
    /// Upload the data and record its norms.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32.
    /// * `metric` - Already validated by the Python layer.
    ///
    /// ### Returns
    ///
    /// The built handle, or a build error.
    #[staticmethod]
    #[pyo3(signature = (x, *, metric))]
    fn build(py: Python<'_>, x: &Bound<'_, PyAny>, metric: String) -> PyResult<Self> {
        let a = f32_array!(x);
        let (data, n, dim) = crate::convert::flat(&a)?;
        let inner = py
            .detach(|| {
                crate::pool::run(|| {
                    build_exhaustive_index_gpu::<f32, Rt>((data, n, dim), &metric, default_device())
                })
            })
            .map_err(crate::error::AnnErr)?;
        Ok(Self { inner, n, dim })
    }

    /// Score every query against every indexed point on the device.
    ///
    /// ### Params
    ///
    /// * `q` - Queries by features, C-contiguous float32.
    /// * `k` - Neighbours per query.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)`, dense and padded. See [`QueryOut`].
    #[pyo3(signature = (q, k, *, return_distance = true, verbose = false))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
        k: usize,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        let a = f32_array!(q);
        let (data, n, dim) = crate::convert::flat(&a)?;
        let idx = &self.inner;
        let (ids, dists) = py
            .detach(|| {
                crate::pool::run(|| {
                    query_exhaustive_index_gpu((data, n, dim), idx, k, return_distance, verbose)
                })
            })
            .map_err(crate::error::AnnErr)?;
        crate::dispatch::pack(py, ids, dists, k)
    }

    /// Full kNN graph over the indexed data.
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
        let idx = &self.inner;
        let (ids, dists) = py
            .detach(|| {
                crate::pool::run(|| {
                    query_exhaustive_index_gpu_self(idx, k, return_distance, verbose)
                })
            })
            .map_err(crate::error::AnnErr)?;
        crate::dispatch::pack(py, ids, dists, k)
    }
});
