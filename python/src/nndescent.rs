//! NN-Descent handle.
//!
//! `delta` and `diversify_prob` are the index's own float type in the library,
//! so they arrive as `f64` and are cast inside the dispatch closure where `T`
//! is known. The Python layer rejects non-finite values first, which is what
//! makes the cast infallible.

use ann_search_rs::cpu::nndescent::NNDescent;
use ann_search_rs::{
    build_nndescent_index, extract_nndescent_knn, query_nndescent_index, query_nndescent_self,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(PyNnDescent, NnDescentInner, NNDescent, "NnDescent", field, {
    /// Descend to a kNN graph by iterative local join.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer.
    /// * `delta` - Early-termination threshold on the improvement rate.
    /// * `diversify_prob` - Bernoulli probability of pruning a redundant edge
    ///   after descent. `0.0` disables pruning, `1.0` always prunes when the
    ///   rule fires.
    /// * `k` - Neighbours in the graph. Also the query-time ceiling, so
    ///   changing it means rebuilding.
    /// * `max_iter` - Descent iterations, or `None` for the library's default.
    /// * `max_candidates` - Candidate pool per node, or `None`.
    /// * `n_trees` - Random projection trees used to seed the graph, or `None`.
    /// * `seed` - Fixes the initialisation and sampling.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// The built handle, a `ValueError` if `delta` or `diversify_prob` is not
    /// finite, or a build error.
    #[staticmethod]
    #[pyo3(signature = (
        x, *, metric, delta, diversify_prob, k = None, max_iter = None,
        max_candidates = None, n_trees = None, seed = 42, verbose = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        metric: String,
        delta: f64,
        diversify_prob: f64,
        k: Option<usize>,
        max_iter: Option<usize>,
        max_candidates: Option<usize>,
        n_trees: Option<usize>,
        seed: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        if !delta.is_finite() || !diversify_prob.is_finite() {
            return Err(PyValueError::new_err(
                "delta and diversify_prob must be finite",
            ));
        }
        build_dispatch!(py, x, NnDescentInner, |data, n, dim| {
            // Checked finite above, so the narrowing cast cannot fail.
            let delta = num_traits::cast(delta).expect("finite f64 casts into the index float");
            let div =
                num_traits::cast(diversify_prob).expect("finite f64 casts into the index float");
            build_nndescent_index(
                (data, n, dim),
                &metric,
                delta,
                div,
                k,
                max_iter,
                max_candidates,
                n_trees,
                seed,
                verbose,
            )
        })
    }

    /// Search the descended graph for external queries.
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
            NnDescentInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_nndescent_index((data, n, dim), idx, k, ef_search, return_distance, verbose)
            }),
            NnDescentInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_nndescent_index((data, n, dim), idx, k, ef_search, return_distance, verbose)
            }),
        }
    }

    /// Full kNN graph over the indexed data.
    ///
    /// This is what NN-Descent produces natively, which makes it the cheapest
    /// route to a self-kNN graph in the crate.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per point, capped by the build-time `k`.
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
            NnDescentInner::F32(idx) => self_arm!(py, k, || query_nndescent_self(
                idx,
                k,
                ef_search,
                return_distance,
                verbose
            )),
            NnDescentInner::F64(idx) => self_arm!(py, k, || query_nndescent_self(
                idx,
                k,
                ef_search,
                return_distance,
                verbose
            )),
        }
    }

    /// Read the descent graph back without searching it.
    ///
    /// `query_self` beam-searches every point and so refines the graph; this
    /// reshapes what the descent already built. An index with
    /// `diversify_prob > 0` has lost the pruned edges here, so it wants
    /// `query_self` instead.
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
        match &self.inner {
            NnDescentInner::F32(idx) => {
                let kk = k.unwrap_or(idx.k);
                self_arm!(py, kk, || extract_nndescent_knn(
                    idx,
                    k,
                    include_self,
                    return_distance
                ))
            }
            NnDescentInner::F64(idx) => {
                let kk = k.unwrap_or(idx.k);
                self_arm!(py, kk, || extract_nndescent_knn(
                    idx,
                    k,
                    include_self,
                    return_distance
                ))
            }
        }
    }
});
