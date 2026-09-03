//! Exhaustive BF16 handle.
//!
//! Storage drops to `bf16`, which keeps `f32`'s exponent range and throws away
//! mantissa bits from roughly the third digit on. The scan is still exhaustive,
//! so the only loss is the codec's, but distances are computed in `f32` so the
//! values are widened back on every comparison and the query runs slower than
//! the `f32` one, not faster.

use ann_search_rs::prelude::DimensionValidation;
use ann_search_rs::quantised::exhaustive_bf16::ExhaustiveIndexBf16;
use ann_search_rs::{
    build_exhaustive_bf16_index, query_exhaustive_bf16_index, query_exhaustive_bf16_self,
};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(
    PyExhaustiveBf16,
    ExhaustiveBf16Inner,
    ExhaustiveIndexBf16,
    "ExhaustiveBf16",
    method,
    {
        /// Quantise the vectors to `bf16` and store them.
        ///
        /// ### Params
        ///
        /// * `x` - Samples by features, C-contiguous float32 or float64.
        /// * `metric` - Already validated by the Python layer. Manhattan is not
        ///   supported and the library rejects it.
        /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
        ///
        /// ### Returns
        ///
        /// The built handle, or an unsupported-metric error.
        #[staticmethod]
        #[pyo3(signature = (x, *, metric, verbose = false))]
        fn build(
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            metric: String,
            verbose: bool,
        ) -> PyResult<Self> {
            build_dispatch!(py, x, ExhaustiveBf16Inner, |data, n, dim| {
                build_exhaustive_bf16_index((data, n, dim), &metric, verbose)
            })
        }

        /// Score every query against every indexed point, at `bf16`.
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
        /// `(indices, distances)`. Distances carry the codec's rounding. See
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
                ExhaustiveBf16Inner::F32(idx) => {
                    query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                        query_exhaustive_bf16_index(
                            (data, n, dim),
                            idx,
                            k,
                            return_distance,
                            verbose,
                        )
                    })
                }
                ExhaustiveBf16Inner::F64(idx) => {
                    query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                        query_exhaustive_bf16_index(
                            (data, n, dim),
                            idx,
                            k,
                            return_distance,
                            verbose,
                        )
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
                ExhaustiveBf16Inner::F32(idx) => {
                    self_arm!(py, k, || query_exhaustive_bf16_self(
                        idx,
                        k,
                        return_distance,
                        verbose
                    ))
                }
                ExhaustiveBf16Inner::F64(idx) => {
                    self_arm!(py, k, || query_exhaustive_bf16_self(
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
