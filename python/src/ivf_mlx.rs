//! IVF MLX handle.
//!
//! k-means trains on the device, the cluster-ordered vectors stay resident,
//! and a Metal kernel scans the probed cells.

use ann_search_rs::mlx::ivf_mlx::IvfIndexMlx;
use ann_search_rs::{build_ivf_index_mlx, query_ivf_index_mlx, query_ivf_index_mlx_self};
use pyo3::prelude::*;

use crate::dispatch::QueryOut;
use crate::kmeans::kmeans_params;
use crate::mlx_handle::{detach, f32_array, mlx_handle};

mlx_handle!(PyIvfMlx, IvfIndexMlx, "IvfMlx", {
    /// Train the cells on the device and leave the vectors there.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32.
    /// * `metric` - Already validated by the Python layer.
    /// * `nlist` - Voronoi cells, or `None` for `sqrt(n)`.
    /// * `kmeans_iters` - Lloyd iterations, or `None` for the crate's 30. MLX
    ///   runs every one; there is no early stop.
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
        let a = f32_array(x)?;
        let (data, n, dim) = crate::convert::flat(&a)?;
        let kmp = kmeans_params(kmeans_iters, kmeans_balanced);
        let inner = detach(py, || {
            build_ivf_index_mlx((data, n, dim), nlist, kmp, &metric, seed, verbose)
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
    /// * `nprobe` - Cells to visit, or `None` for `sqrt(nlist)`.
    /// * `nquery` - Queries per device tile, or `None` for the library's own
    ///   tiling.
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
        let a = f32_array(q)?;
        let (data, n, dim) = crate::convert::flat(&a)?;
        let idx = &self.inner;
        let (ids, dists) = detach(py, || {
            query_ivf_index_mlx(
                (data, n, dim),
                idx,
                k,
                nprobe,
                nquery,
                return_distance,
                verbose,
            )
        })
        .map_err(crate::error::AnnErr)?;
        crate::dispatch::pack(py, ids, dists, k)
    }

    /// Full kNN graph over the indexed data.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per point.
    /// * `nprobe` - Cells to visit, or `None` for `sqrt(nlist)`.
    /// * `nquery` - Points per device tile, or `None` for the library's own
    ///   tiling.
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
        let (ids, dists) = detach(py, || {
            query_ivf_index_mlx_self(idx, k, nprobe, nquery, return_distance, verbose)
        })
        .map_err(crate::error::AnnErr)?;
        crate::dispatch::pack(py, ids, dists, k)
    }
});
