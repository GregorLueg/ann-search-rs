//! HNSW handle.
//!
//! `ef_search` is required rather than optional here, matching the library:
//! HNSW has no internal default to fall back on.

use ann_search_rs::cpu::hnsw::HnswIndex;
use ann_search_rs::prelude::AnnSearchErrors;
use ann_search_rs::{build_hnsw_index, query_hnsw_index, query_hnsw_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(PyHnsw, HnswInner, HnswIndex, "Hnsw", field, {
    /// Build the graph.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `m` - Edges per node on the upper layers; the base layer gets `2 * m`.
    /// * `ef_construction` - Candidate list size during insertion. Larger means
    ///   a better graph and a slower build.
    /// * `metric` - Already validated by the Python layer.
    /// * `seed` - Level assignment is randomised; this fixes it.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
    ///
    /// ### Returns
    ///
    /// The built handle. This builder is infallible in the library, hence the
    /// `Ok` wrapper below.
    #[staticmethod]
    #[pyo3(signature = (x, *, m, ef_construction, metric, seed = 42, verbose = false))]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        m: usize,
        ef_construction: usize,
        metric: String,
        seed: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        build_dispatch!(py, x, HnswInner, |data, n, dim| Ok::<_, AnnSearchErrors>(
            build_hnsw_index((data, n, dim), m, ef_construction, &metric, seed, verbose)
        ))
    }

    /// Search the graph for external queries.
    ///
    /// ### Params
    ///
    /// * `q` - Queries by features, matching the index's float type.
    /// * `k` - Neighbours per query.
    /// * `ef_search` - Candidate list size. Raise for recall.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)`, dense and padded. See [`QueryOut`].
    #[pyo3(signature = (q, k, *, ef_search, return_distance = true, verbose = false))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
        k: usize,
        ef_search: usize,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            HnswInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_hnsw_index((data, n, dim), idx, k, ef_search, return_distance, verbose)
            }),
            HnswInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_hnsw_index((data, n, dim), idx, k, ef_search, return_distance, verbose)
            }),
        }
    }

    /// Full kNN graph over the indexed data.
    ///
    /// Walks the graph directly rather than re-entering it per point.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per point.
    /// * `ef_search` - Candidate list size. Raise for recall.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` for every indexed point. See [`QueryOut`].
    #[pyo3(signature = (k, *, ef_search, return_distance = true, verbose = false))]
    fn query_self<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        ef_search: usize,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            HnswInner::F32(idx) => {
                self_arm!(py, k, || query_hnsw_self(
                    idx,
                    k,
                    ef_search,
                    return_distance,
                    verbose
                ))
            }
            HnswInner::F64(idx) => {
                self_arm!(py, k, || query_hnsw_self(
                    idx,
                    k,
                    ef_search,
                    return_distance,
                    verbose
                ))
            }
        }
    }
});
