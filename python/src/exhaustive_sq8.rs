//! Exhaustive SQ8 handle.
//!
//! One byte per dimension, with per-dimension offsets and a single shared
//! scale. The shared scale is what lets the integer code distance preserve the
//! ordering of the float one, so the scan runs entirely on `u8` kernels.

use ann_search_rs::prelude::DimensionValidation;
use ann_search_rs::quantised::exhaustive_sq8::ExhaustiveSq8Index;
use ann_search_rs::{
    build_exhaustive_sq8_index, query_exhaustive_sq8_index, query_exhaustive_sq8_self,
};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;
use crate::quant::quant_params;

ann_handle!(
    PyExhaustiveSq8,
    ExhaustiveSq8Inner,
    ExhaustiveSq8Index,
    "ExhaustiveSq8",
    method,
    {
        /// Calibrate the quantiser and encode the vectors.
        ///
        /// ### Params
        ///
        /// * `x` - Samples by features, C-contiguous float32 or float64.
        /// * `metric` - Already validated by the Python layer. Manhattan is not
        ///   supported and the library rejects it.
        /// * `quant_drop_ratio` - Fraction trimmed from each tail of every
        ///   dimension before the range is fixed, or `None` for the crate
        ///   default. Values outside the trimmed range clamp to the end codes.
        /// * `quant_sample_rows` - Rows sampled for calibration, or `None` to
        ///   auto-pick.
        /// * `seed` - Fixes the calibration row sample.
        /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
        ///
        /// ### Returns
        ///
        /// The built handle, or a calibration or unsupported-metric error.
        #[staticmethod]
        #[pyo3(signature = (
            x, *, metric, quant_drop_ratio = None, quant_sample_rows = None, seed = 42,
            verbose = false
        ))]
        #[allow(clippy::too_many_arguments)]
        fn build(
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            metric: String,
            quant_drop_ratio: Option<f64>,
            quant_sample_rows: Option<usize>,
            seed: usize,
            verbose: bool,
        ) -> PyResult<Self> {
            let qp = quant_params(quant_drop_ratio, quant_sample_rows, seed);
            build_dispatch!(py, x, ExhaustiveSq8Inner, |data, n, dim| {
                build_exhaustive_sq8_index((data, n, dim), &metric, qp, verbose)
            })
        }

        /// Score every query against every indexed point, on codes.
        ///
        /// ### Params
        ///
        /// * `q` - Queries by features, matching the index's float type.
        /// * `k` - Neighbours per query.
        /// * `return_distance` - Skips the copy into numpy, not the computation.
        /// * `verbose` - Progress to the process stdout.
        ///
        /// ### Returns
        ///
        /// `(indices, distances)`. Distances are the codec's estimate. See
        /// [`QueryOut`].
        #[pyo3(signature = (q, k, *, return_distance = true, verbose = false))]
        fn query<'py>(
            &self,
            py: Python<'py>,
            q: &Bound<'py, PyAny>,
            k: usize,
            return_distance: bool,
            verbose: bool,
        ) -> PyResult<QueryOut<'py>> {
            match &self.inner {
                ExhaustiveSq8Inner::F32(idx) => {
                    query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                        query_exhaustive_sq8_index((data, n, dim), idx, k, return_distance, verbose)
                    })
                }
                ExhaustiveSq8Inner::F64(idx) => {
                    query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                        query_exhaustive_sq8_index((data, n, dim), idx, k, return_distance, verbose)
                    })
                }
            }
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
            match &self.inner {
                ExhaustiveSq8Inner::F32(idx) => {
                    self_arm!(py, k, || query_exhaustive_sq8_self(
                        idx,
                        k,
                        return_distance,
                        verbose
                    ))
                }
                ExhaustiveSq8Inner::F64(idx) => {
                    self_arm!(py, k, || query_exhaustive_sq8_self(
                        idx,
                        k,
                        return_distance,
                        verbose
                    ))
                }
            }
        }
    }
);
