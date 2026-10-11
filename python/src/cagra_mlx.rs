//! CAGRA MLX handle.
//!
//! NN-Descent and the CAGRA prune on the device, then a SIMD-group Metal beam
//! search with forest-routed entry points. Unlike the wgpu handle there is no
//! lazy upload: the MLX index is resident from the build on, so queries take
//! `&self`.

use ann_search_rs::mlx::cagra_mlx::CagraMlxSearchParams;
use ann_search_rs::mlx::nndescent_index_mlx::NNDescentIndexMlx;
use ann_search_rs::{
    build_nndescent_index_mlx, query_nndescent_index_mlx, query_nndescent_index_mlx_self,
};
use pyo3::prelude::*;

use crate::dispatch::QueryOut;
use crate::mlx_handle::{detach, f32_array, mlx_handle};

/// Assemble [`CagraMlxSearchParams`] from the four beam knobs.
///
/// ### Params
///
/// * `beam_width` - Width of the search beam. The main recall knob.
/// * `max_beam_iters` - Iteration cap.
/// * `n_entry_points` - Entry points into the graph.
/// * `expand_per_iter` - Beam entries expanded per iteration.
///
/// ### Returns
///
/// `None` when no knob was touched, so the library sizes the beam from `k`.
fn cagra_params(
    beam_width: Option<usize>,
    max_beam_iters: Option<usize>,
    n_entry_points: Option<usize>,
    expand_per_iter: Option<usize>,
) -> Option<CagraMlxSearchParams> {
    if beam_width.is_none()
        && max_beam_iters.is_none()
        && n_entry_points.is_none()
        && expand_per_iter.is_none()
    {
        return None;
    }
    Some(CagraMlxSearchParams::new(
        beam_width,
        max_beam_iters,
        n_entry_points,
        expand_per_iter,
    ))
}

mlx_handle!(PyCagraMlx, NNDescentIndexMlx, "CagraMlx", {
    /// Descend on the device, then prune into a navigational graph.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32.
    /// * `metric` - Already validated by the Python layer.
    /// * `k` - Final neighbours per node, or `None` for 30.
    /// * `build_k` - Working degree before pruning, or `None` for `1.5 * k`.
    /// * `max_iters` - NN-Descent iteration cap, or `None` for 15.
    /// * `n_trees` - Forest size for the initial graph, or `None` for auto.
    /// * `delta` - Convergence threshold, or `None` for 0.001.
    /// * `rho` - Local-join sampling rate, or `None` for 1.0.
    /// * `refine_knn` - Two-hop refinement sweeps after the main loop.
    /// * `seed` - Fixes the seed graph.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// The built handle, or a build error.
    #[staticmethod]
    #[pyo3(signature = (
        x, *, metric, k = None, build_k = None, max_iters = None, n_trees = None, delta = None,
        rho = None, refine_knn = None, seed = 42, verbose = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        metric: String,
        k: Option<usize>,
        build_k: Option<usize>,
        max_iters: Option<usize>,
        n_trees: Option<usize>,
        delta: Option<f32>,
        rho: Option<f32>,
        refine_knn: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        let a = f32_array(x)?;
        let (data, n, dim) = crate::convert::flat(&a)?;
        let inner = detach(py, || {
            build_nndescent_index_mlx(
                (data, n, dim),
                &metric,
                k,
                build_k,
                max_iters,
                n_trees,
                delta,
                rho,
                refine_knn,
                seed,
                verbose,
            )
        })
        .map_err(crate::error::AnnErr)?;
        Ok(Self { inner, n, dim })
    }

    /// Beam-search the navigational graph for external queries.
    ///
    /// ### Params
    ///
    /// * `q` - Queries by features, C-contiguous float32.
    /// * `k` - Neighbours per query.
    /// * `beam_width` - Beam width, or `None` to size it from `k`.
    /// * `max_beam_iters` - Iteration cap, or `None`.
    /// * `n_entry_points` - Graph entry points, or `None`.
    /// * `expand_per_iter` - Beam entries expanded per iteration, or `None`.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)`, dense and padded. See [`QueryOut`].
    #[pyo3(signature = (
        q, k, *, beam_width = None, max_beam_iters = None, n_entry_points = None,
        expand_per_iter = None, return_distance = true, verbose = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn query<'py>(
        &self,
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
        k: usize,
        beam_width: Option<usize>,
        max_beam_iters: Option<usize>,
        n_entry_points: Option<usize>,
        expand_per_iter: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        let a = f32_array(q)?;
        let (data, n, dim) = crate::convert::flat(&a)?;
        let params = cagra_params(beam_width, max_beam_iters, n_entry_points, expand_per_iter);
        let idx = &self.inner;
        let (ids, dists) = detach(py, || {
            query_nndescent_index_mlx((data, n, dim), idx, k, params, return_distance, verbose)
        })
        .map_err(crate::error::AnnErr)?;
        crate::dispatch::pack(py, ids, dists, k)
    }

    /// Full kNN graph over the indexed data via the beam search.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per point.
    /// * `beam_width` - Beam width, or `None` to size it from `k`.
    /// * `max_beam_iters` - Iteration cap, or `None`.
    /// * `n_entry_points` - Graph entry points, or `None`.
    /// * `expand_per_iter` - Beam entries expanded per iteration, or `None`.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Accepted and ignored; the library's self-query takes no
    ///   verbosity flag. Present so the estimator can pass it uniformly.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` for every indexed point. See [`QueryOut`].
    #[pyo3(signature = (
        k, *, beam_width = None, max_beam_iters = None, n_entry_points = None,
        expand_per_iter = None, return_distance = true, verbose = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn query_self<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        beam_width: Option<usize>,
        max_beam_iters: Option<usize>,
        n_entry_points: Option<usize>,
        expand_per_iter: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        let _ = verbose;
        let params = cagra_params(beam_width, max_beam_iters, n_entry_points, expand_per_iter);
        let idx = &self.inner;
        let (ids, dists) = detach(py, || {
            query_nndescent_index_mlx_self(idx, k, params, return_distance)
        })
        .map_err(crate::error::AnnErr)?;
        crate::dispatch::pack(py, ids, dists, k)
    }

    /// Read the descent graph back without searching it.
    ///
    /// The kNN graph NN-Descent produced, taken before the CAGRA prune. Host
    /// memory only; nothing runs on the device.
    ///
    /// ### Params
    ///
    /// * `k` - Total row length, self-edge included when `include_self` is
    ///   set. `None` keeps the build-time degree, which is the ceiling.
    /// * `include_self` - Prepend `(i, 0)` to row `i`.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` for every indexed point, rows the descent never
    /// filled padded. See [`QueryOut`].
    #[pyo3(signature = (k = None, *, include_self = true, return_distance = true))]
    fn extract_knn<'py>(
        &self,
        py: Python<'py>,
        k: Option<usize>,
        include_self: bool,
        return_distance: bool,
    ) -> PyResult<QueryOut<'py>> {
        let idx = &self.inner;
        let kk = k.unwrap_or(idx.k);
        let (ids, dists) = detach(py, || idx.extract_knn(k, include_self, return_distance));
        crate::dispatch::pack(py, ids, dists, kk)
    }
});
