//! CAGRA GPU handle.
//!
//! NN-Descent built on the device, then pruned into a CAGRA navigational graph
//! and searched with a beam. The fastest route to a kNN graph here when a GPU
//! is present.
//!
//! This is the one handle that is not `frozen`. `query_batch_gpu` takes
//! `&mut self` to memoise its upload of the navigational graph, so queries go
//! through `borrow_mut` and two Python threads cannot query one index at once.
//! Building with `retain_gpu = True` does that upload up front, which is why it
//! is the default.

use ann_search_rs::gpu::cagra_gpu_search::CagraGpuSearchParams;
use ann_search_rs::gpu::nndescent_gpu::NNDescentGpu;
use ann_search_rs::{
    build_nndescent_index_gpu, extract_nndescent_knn_gpu, query_nndescent_index_gpu,
    query_nndescent_index_gpu_self,
};
use pyo3::prelude::*;

use crate::dispatch::QueryOut;
use crate::gpu_handle::{default_device, f32_array, gpu_handle, Rt};

/// Assemble [`CagraGpuSearchParams`] from the four beam knobs.
///
/// ### Params
///
/// * `beam_width` - Width of the search beam. The main recall knob.
/// * `max_beam_iters` - Iteration cap. Rule of thumb is 2-3x the beam width.
/// * `n_entry_points` - Entry points into the graph.
/// * `expand_per_iter` - Extra neighbours explored per iteration, usually 1-4.
///
/// ### Returns
///
/// `None` when no knob was touched, so the library sizes the beam from `k`
/// instead of taking a half-specified struct.
fn cagra_params(
    beam_width: Option<usize>,
    max_beam_iters: Option<usize>,
    n_entry_points: Option<usize>,
    expand_per_iter: Option<usize>,
) -> Option<CagraGpuSearchParams> {
    if beam_width.is_none()
        && max_beam_iters.is_none()
        && n_entry_points.is_none()
        && expand_per_iter.is_none()
    {
        return None;
    }
    Some(CagraGpuSearchParams::new(
        beam_width,
        max_beam_iters,
        n_entry_points,
        expand_per_iter,
    ))
}

gpu_handle!(PyCagraGpu, NNDescentGpu, "CagraGpu", {
    /// Descend on the device, then prune into a navigational graph.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32.
    /// * `metric` - Already validated by the Python layer.
    /// * `k` - Final neighbours per node, or `None` for the library's 30.
    /// * `build_k` - Working degree before pruning, or `None` for `1.5 * k`.
    /// * `max_iters` - NN-Descent iteration cap, or `None` for 15.
    /// * `n_trees` - Forest size for the initial graph, or `None` for auto.
    /// * `delta` - Convergence threshold, or `None` for 0.001.
    /// * `rho` - Local-join sampling rate, or `None` for 1.0.
    /// * `refine_knn` - Two-hop refinement sweeps after the main loop.
    /// * `retain_gpu` - Upload the navigational graph at build time. On by
    ///   default: the first query would otherwise pay for it.
    /// * `seed` - Fixes the seed graph.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// The built handle, or a build error.
    #[staticmethod]
    #[pyo3(signature = (
        x, *, metric, k = None, build_k = None, max_iters = None, n_trees = None, delta = None,
        rho = None, refine_knn = None, retain_gpu = true, seed = 42, verbose = false
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
        retain_gpu: bool,
        seed: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        let a = f32_array!(x);
        let (data, n, dim) = crate::convert::flat(&a)?;
        let inner = py
            .detach(|| {
                crate::pool::run(|| {
                    build_nndescent_index_gpu::<f32, Rt>(
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
                        retain_gpu,
                        default_device(),
                    )
                })
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
    /// * `expand_per_iter` - Neighbours explored per iteration, or `None`.
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
        slf: &Bound<'py, Self>,
        q: &Bound<'py, PyAny>,
        k: usize,
        beam_width: Option<usize>,
        max_beam_iters: Option<usize>,
        n_entry_points: Option<usize>,
        expand_per_iter: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        let py = slf.py();
        let a = f32_array!(q);
        let (data, n, dim) = crate::convert::flat(&a)?;
        let params = cagra_params(beam_width, max_beam_iters, n_entry_points, expand_per_iter);
        let mut guard = slf.borrow_mut();
        let idx = &mut guard.inner;
        let (ids, dists) = py
            .detach(|| {
                crate::pool::run(|| {
                    query_nndescent_index_gpu(
                        (data, n, dim),
                        idx,
                        k,
                        params,
                        return_distance,
                        verbose,
                    )
                })
            })
            .map_err(crate::error::AnnErr)?;
        drop(guard);
        crate::dispatch::pack(py, ids, dists, k)
    }

    /// Full kNN graph over the indexed data.
    ///
    /// Beam-searches for every indexed point, so the result is the refined
    /// graph rather than the raw NN-Descent output.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per point.
    /// * `beam_width` - Beam width, or `None` to size it from `k`.
    /// * `max_beam_iters` - Iteration cap, or `None`.
    /// * `n_entry_points` - Graph entry points, or `None`.
    /// * `expand_per_iter` - Neighbours explored per iteration, or `None`.
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
        slf: &Bound<'py, Self>,
        k: usize,
        beam_width: Option<usize>,
        max_beam_iters: Option<usize>,
        n_entry_points: Option<usize>,
        expand_per_iter: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        let _ = verbose;
        let py = slf.py();
        let params = cagra_params(beam_width, max_beam_iters, n_entry_points, expand_per_iter);
        let mut guard = slf.borrow_mut();
        let idx = &mut guard.inner;
        let (ids, dists) = py
            .detach(|| {
                crate::pool::run(|| query_nndescent_index_gpu_self(idx, k, params, return_distance))
            })
            .map_err(crate::error::AnnErr)?;
        drop(guard);
        crate::dispatch::pack(py, ids, dists, k)
    }

    /// Read the descent graph back without searching it.
    ///
    /// Returns the kNN graph NN-Descent produced, held alongside the
    /// navigational one and taken before the CAGRA prune. Nothing is
    /// dispatched to the device.
    ///
    /// ### Params
    ///
    /// * `k` - Total row length, self-edge included when `include_self` is
    ///   set. `None` keeps the build-time degree, which is the ceiling.
    /// * `include_self` - Prepend `(i, 0)` to row `i`. A kNN graph stores no
    ///   such edge, but every `query_self` and any exhaustive ground truth
    ///   counts a point as its own nearest neighbour.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` for every indexed point. Rows the descent never
    /// filled come back padded, which the search paths never produce. See
    /// [`QueryOut`].
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
        let (ids, dists) = py
            .detach(|| {
                crate::pool::run(|| {
                    extract_nndescent_knn_gpu(idx, k, include_self, return_distance)
                })
            })
            .map_err(crate::error::AnnErr)?;
        crate::dispatch::pack(py, ids, dists, kk)
    }
});

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_untouched_knobs_defer_to_the_library() {
        assert!(cagra_params(None, None, None, None).is_none());
    }

    #[test]
    fn test_one_knob_forces_a_params_struct() {
        let params = cagra_params(Some(64), None, None, None).expect("beam width forces a struct");
        assert_eq!(params.beam_width, Some(64));
        assert_eq!(params.max_beam_iters, None);
    }
}
