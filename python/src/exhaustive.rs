//! Brute-force handle.
//!
//! Exact by construction, and the ground truth the approximate indices are
//! measured against. No search-time knobs.

use ann_search_rs::cpu::exhaustive::ExhaustiveIndex;
use ann_search_rs::prelude::AnnSearchErrors;
use ann_search_rs::{build_exhaustive_index, query_exhaustive_index, query_exhaustive_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(
    PyExhaustive,
    ExhaustiveInner,
    ExhaustiveIndex,
    "Exhaustive",
    field,
    {
        /// Store the vectors.
        ///
        /// ### Params
        ///
        /// * `x` - Samples by features, C-contiguous float32 or float64.
        /// * `metric` - Already validated by the Python layer.
        ///
        /// ### Returns
        ///
        /// The built handle. Infallible in the library, hence the `Ok` wrapper.
        #[staticmethod]
        #[pyo3(signature = (x, *, metric))]
        fn build(py: Python<'_>, x: &Bound<'_, PyAny>, metric: String) -> PyResult<Self> {
            build_dispatch!(py, x, ExhaustiveInner, |data, n, dim| Ok::<
                _,
                AnnSearchErrors,
            >(
                build_exhaustive_index((data, n, dim), &metric)
            ))
        }

        /// Score every query against every indexed point.
        ///
        /// ### Params
        ///
        /// * `q` - Queries by features, matching the index's float type.
        /// * `k` - Neighbours per query.
        /// * `return_distance` - Skips the copy into numpy, not the computation.
        /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
        ///
        /// ### Returns
        ///
        /// `(indices, distances)`, exact. See [`QueryOut`].
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
                ExhaustiveInner::F32(idx) => {
                    query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                        query_exhaustive_index((data, n, dim), idx, k, return_distance, verbose)
                    })
                }
                ExhaustiveInner::F64(idx) => {
                    query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                        query_exhaustive_index((data, n, dim), idx, k, return_distance, verbose)
                    })
                }
            }
        }

        /// Full kNN graph over the indexed data.
        ///
        /// Every point matches itself first, at distance zero.
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
                ExhaustiveInner::F32(idx) => {
                    self_arm!(py, k, || query_exhaustive_self(
                        idx,
                        k,
                        return_distance,
                        verbose
                    ))
                }
                ExhaustiveInner::F64(idx) => {
                    self_arm!(py, k, || query_exhaustive_self(
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
