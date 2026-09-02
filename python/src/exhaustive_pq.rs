//! Exhaustive PQ handle.
//!
//! Each vector is split into `m` subvectors, and each subvector is replaced by
//! the id of its nearest sub-codebook centroid. A query builds one lookup table
//! per subspace and then scores every point by summing `m` table reads, which
//! is where the compression pays: `m` bytes per vector instead of `dim` floats.

use ann_search_rs::prelude::DimensionValidation;
use ann_search_rs::quantised::exhaustive_pq::ExhaustivePqIndex;
use ann_search_rs::{
    build_exhaustive_pq_index, query_exhaustive_pq_index, query_exhaustive_pq_index_self,
};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(
    PyExhaustivePq,
    ExhaustivePqInner,
    ExhaustivePqIndex,
    "ExhaustivePq",
    method,
    {
        /// Train the sub-codebooks and encode the vectors.
        ///
        /// ### Params
        ///
        /// * `x` - Samples by features, C-contiguous float32 or float64.
        /// * `m` - Subspaces. `dim` must divide by it.
        /// * `metric` - Already validated by the Python layer. Manhattan is not
        ///   supported and the library rejects it.
        /// * `max_iters` - Lloyd iterations for the sub-codebooks, or `None`
        ///   for the crate default.
        /// * `n_pq_centroids` - Centroids per subspace, or `None` for the
        ///   crate default of 256. Cannot exceed the sample count.
        /// * `seed` - Fixes the codebook initialisation.
        /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
        ///
        /// ### Returns
        ///
        /// The built handle, or a codebook or unsupported-metric error.
        #[staticmethod]
        #[pyo3(signature = (
            x, *, m, metric, max_iters = None, n_pq_centroids = None, seed = 42, verbose = false
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
            verbose: bool,
        ) -> PyResult<Self> {
            build_dispatch!(py, x, ExhaustivePqInner, |data, n, dim| {
                build_exhaustive_pq_index(
                    (data, n, dim),
                    m,
                    max_iters,
                    n_pq_centroids,
                    &metric,
                    seed,
                    verbose,
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
                ExhaustivePqInner::F32(idx) => {
                    query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                        query_exhaustive_pq_index((data, n, dim), idx, k, return_distance, verbose)
                    })
                }
                ExhaustivePqInner::F64(idx) => {
                    query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                        query_exhaustive_pq_index((data, n, dim), idx, k, return_distance, verbose)
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
                ExhaustivePqInner::F32(idx) => {
                    self_arm!(py, k, || query_exhaustive_pq_index_self(
                        idx,
                        k,
                        return_distance,
                        verbose
                    ))
                }
                ExhaustivePqInner::F64(idx) => {
                    self_arm!(py, k, || query_exhaustive_pq_index_self(
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
