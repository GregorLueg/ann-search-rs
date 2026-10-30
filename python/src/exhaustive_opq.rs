//! Exhaustive OPQ handle.
//!
//! PQ with a learned orthogonal rotation applied first, so the subspaces carry
//! comparable variance instead of whatever the original axis order happened to
//! give them. Costs more at build time and nothing at query time, since the
//! rotation folds into the query once.

use ann_search_rs::parse_annsearch_verbosity;
use ann_search_rs::prelude::DimensionValidation;
use ann_search_rs::quantised::exhaustive_opq::ExhaustiveOpqIndex;
use ann_search_rs::{
    build_exhaustive_opq_index, query_exhaustive_opq_index, query_exhaustive_opq_index_self,
};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(
    PyExhaustiveOpq,
    ExhaustiveOpqInner,
    ExhaustiveOpqIndex,
    "ExhaustiveOpq",
    method,
    {
        /// Learn the rotation and the sub-codebooks, then encode.
        ///
        /// ### Params
        ///
        /// * `x` - Samples by features, C-contiguous float32 or float64.
        /// * `m` - Subspaces. `dim` must divide by it.
        /// * `metric` - Already validated by the Python layer. Manhattan is not
        ///   supported and the library rejects it.
        /// * `max_iters` - Alternating rotation/codebook iterations, or `None`
        ///   for the crate default. This is the build-cost knob.
        /// * `n_pq_centroids` - Centroids per subspace, or `None` for the
        ///   crate default of 256. Cannot exceed the sample count.
        /// * `seed` - Fixes the codebook initialisation.
        /// * `verbose` - `0` quiet, `1` normal, `2` detailed. Prints to the process
        ///   stdout, not `sys.stdout`.
        ///
        /// ### Returns
        ///
        /// The built handle, or a codebook or unsupported-metric error.
        #[staticmethod]
        #[pyo3(signature = (
            x, *, m, metric, max_iters = None, n_pq_centroids = None, seed = 42, verbose = 0
        ))]
        #[allow(clippy::too_many_arguments)]
        fn build(
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            m: usize,
            metric: String,
            max_iters: Option<usize>,
            n_pq_centroids: Option<usize>,
            seed: usize,
            verbose: usize,
        ) -> PyResult<Self> {
            build_dispatch!(py, x, ExhaustiveOpqInner, |data, n, dim| {
                build_exhaustive_opq_index(
                    (data, n, dim),
                    m,
                    max_iters,
                    n_pq_centroids,
                    &metric,
                    seed,
                    parse_annsearch_verbosity(verbose),
                )
            })
        }

        /// Score every query against every code, by table lookup.
        ///
        /// ### Params
        ///
        /// * `q` - Queries by features, matching the index's float type.
        /// * `k` - Neighbours per query.
        /// * `return_distance` - Skips the copy into numpy, not the computation.
        /// * `verbose` - `0` quiet, `1` normal, `2` detailed. Prints to the process
        ///   stdout, not `sys.stdout`.
        ///
        /// ### Returns
        ///
        /// `(indices, distances)`. Distances are the codec's estimate. See
        /// [`QueryOut`].
        #[pyo3(signature = (q, k, *, return_distance = true, verbose = 0))]
        fn query<'py>(
            &self,
            py: Python<'py>,
            q: &Bound<'py, PyAny>,
            k: usize,
            return_distance: bool,
            verbose: usize,
        ) -> PyResult<QueryOut<'py>> {
            match &self.inner {
                ExhaustiveOpqInner::F32(idx) => {
                    query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                        query_exhaustive_opq_index(
                            (data, n, dim),
                            idx,
                            k,
                            return_distance,
                            parse_annsearch_verbosity(verbose),
                        )
                    })
                }
                ExhaustiveOpqInner::F64(idx) => {
                    query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                        query_exhaustive_opq_index(
                            (data, n, dim),
                            idx,
                            k,
                            return_distance,
                            parse_annsearch_verbosity(verbose),
                        )
                    })
                }
            }
        }

        /// Full kNN graph over the indexed data.
        ///
        /// Each point is decoded from its codes and queried back against the
        /// index, so the reconstruction error enters twice here and once in
        /// [`query`](Self::query).
        ///
        /// ### Params
        ///
        /// * `k` - Neighbours per point.
        /// * `return_distance` - Skips the copy into numpy, not the computation.
        /// * `verbose` - `0` quiet, `1` normal, `2` detailed. Prints to the process
        ///   stdout, not `sys.stdout`.
        ///
        /// ### Returns
        ///
        /// `(indices, distances)` for every indexed point. See [`QueryOut`].
        #[pyo3(signature = (k, *, return_distance = true, verbose = 0))]
        fn query_self<'py>(
            &self,
            py: Python<'py>,
            k: usize,
            return_distance: bool,
            verbose: usize,
        ) -> PyResult<QueryOut<'py>> {
            match &self.inner {
                ExhaustiveOpqInner::F32(idx) => {
                    self_arm!(py, k, || query_exhaustive_opq_index_self(
                        idx,
                        k,
                        return_distance,
                        parse_annsearch_verbosity(verbose)
                    ))
                }
                ExhaustiveOpqInner::F64(idx) => {
                    self_arm!(py, k, || query_exhaustive_opq_index_self(
                        idx,
                        k,
                        return_distance,
                        parse_annsearch_verbosity(verbose)
                    ))
                }
            }
        }
    }
);
