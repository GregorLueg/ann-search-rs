//! Quantised HNSW handle.
//!
//! An HNSW built *and* searched entirely on 8-bit codes, inspired by pyglass.
//! One scale shared across all dimensions is what makes the integer code
//! distance preserve the ordering of the float one, so a single kernel serves
//! construction and query.

use ann_search_rs::quantised::hnsw_quantised::index::HnswSq8uIndex;
use ann_search_rs::{build_hnsw_sq8u_index, query_hnsw_sq8u_index, query_hnsw_sq8u_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;
use crate::quant::quant_params;

ann_handle!(
    PyHnswSq8u,
    HnswSq8uInner,
    HnswSq8uIndex,
    "HnswSq8u",
    method,
    {
        /// Calibrate, encode, and build the graph on the codes.
        ///
        /// ### Params
        ///
        /// * `x` - Samples by features, C-contiguous float32 or float64.
        /// * `m` - Edges per node on the upper layers; the base layer gets
        ///   `2 * m`.
        /// * `ef_construction` - Candidate list size during insertion.
        /// * `metric` - Already validated by the Python layer. Manhattan is not
        ///   supported and the library rejects it.
        /// * `quant_drop_ratio` - Fraction trimmed from each tail of every
        ///   dimension before the range is fixed, or `None` for the crate
        ///   default.
        /// * `quant_sample_rows` - Rows sampled for calibration, or `None` to
        ///   auto-pick.
        /// * `seed` - Fixes the level assignment and the calibration sample.
        /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
        ///
        /// ### Returns
        ///
        /// The built handle, or a calibration or unsupported-metric error.
        #[staticmethod]
        #[pyo3(signature = (
            x, *, m, ef_construction, metric, quant_drop_ratio = None,
            quant_sample_rows = None, seed = 42, verbose = false
        ))]
        #[allow(clippy::too_many_arguments)]
        fn build(
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            m: usize,
            ef_construction: usize,
            metric: String,
            quant_drop_ratio: Option<f64>,
            quant_sample_rows: Option<usize>,
            seed: usize,
            verbose: bool,
        ) -> PyResult<Self> {
            let qp = quant_params(quant_drop_ratio, quant_sample_rows, seed);
            build_dispatch!(py, x, HnswSq8uInner, |data, n, dim| build_hnsw_sq8u_index(
                (data, n, dim),
                m,
                ef_construction,
                &metric,
                seed,
                qp,
                verbose
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
        /// `(indices, distances)`. Distances are the codec's estimate. See
        /// [`QueryOut`].
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
                HnswSq8uInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                    query_hnsw_sq8u_index(
                        (data, n, dim),
                        idx,
                        k,
                        ef_search,
                        return_distance,
                        verbose,
                    )
                }),
                HnswSq8uInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                    query_hnsw_sq8u_index(
                        (data, n, dim),
                        idx,
                        k,
                        ef_search,
                        return_distance,
                        verbose,
                    )
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
                HnswSq8uInner::F32(idx) => {
                    self_arm!(py, k, || query_hnsw_sq8u_self(
                        idx,
                        k,
                        ef_search,
                        return_distance,
                        verbose
                    ))
                }
                HnswSq8uInner::F64(idx) => {
                    self_arm!(py, k, || query_hnsw_sq8u_self(
                        idx,
                        k,
                        ef_search,
                        return_distance,
                        verbose
                    ))
                }
            }
        }
    }
);
