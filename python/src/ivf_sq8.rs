//! IVF SQ8 handle.
//!
//! Inverted file with the posting lists held as 8-bit codes. A quarter of
//! IVF's vector memory, and the integer kernels usually make the scan faster
//! rather than slower.

use ann_search_rs::prelude::DimensionValidation;
use ann_search_rs::quantised::ivf_sq8::IvfSq8Index;
use ann_search_rs::{build_ivf_sq8_index, query_ivf_sq8_index, query_ivf_sq8_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;
use crate::kmeans::kmeans_params;
use crate::quant::quant_params;

ann_handle!(
    PyIvfSq8,
    IvfSq8Inner,
    IvfSq8Index,
    "IvfSq8",
    method,
    {
        /// Cluster the data and encode the lists to 8-bit codes.
        ///
        /// ### Params
        ///
        /// * `x` - Samples by features, C-contiguous float32 or float64.
        /// * `metric` - Already validated by the Python layer. Manhattan is not
        ///   supported and the library rejects it.
        /// * `nlist` - Voronoi cells, or `None` for the library's `sqrt(n)`
        ///   heuristic.
        /// * `kmeans_iters` - Lloyd iterations, or `None` for the crate default.
        /// * `kmeans_balanced` - Reseed starved centroids each iteration.
        /// * `quant_drop_ratio` - Fraction trimmed from each tail of every
        ///   dimension before the range is fixed, or `None` for the crate
        ///   default.
        /// * `quant_sample_rows` - Rows sampled for calibration, or `None` to
        ///   auto-pick.
        /// * `seed` - Fixes the k-means initialisation and the calibration
        ///   sample.
        /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
        ///
        /// ### Returns
        ///
        /// The built handle, or a clustering, calibration or
        /// unsupported-metric error.
        #[staticmethod]
        #[pyo3(signature = (
            x, *, metric, nlist = None, kmeans_iters = None, kmeans_balanced = false,
            quant_drop_ratio = None, quant_sample_rows = None, seed = 42, verbose = false
        ))]
        #[allow(clippy::too_many_arguments)]
        fn build(
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            metric: String,
            nlist: Option<usize>,
            kmeans_iters: Option<usize>,
            kmeans_balanced: bool,
            quant_drop_ratio: Option<f64>,
            quant_sample_rows: Option<usize>,
            seed: usize,
            verbose: bool,
        ) -> PyResult<Self> {
            let kmp = kmeans_params(kmeans_iters, kmeans_balanced);
            let qp = quant_params(quant_drop_ratio, quant_sample_rows, seed);
            build_dispatch!(py, x, IvfSq8Inner, |data, n, dim| build_ivf_sq8_index(
                (data, n, dim),
                nlist,
                kmp,
                &metric,
                seed,
                qp,
                verbose
            ))
        }

        /// Probe the nearest cells for external queries.
        ///
        /// ### Params
        ///
        /// * `q` - Queries by features, matching the index's float type.
        /// * `k` - Neighbours per query.
        /// * `nprobe` - Cells to visit, or `None` for the library's heuristic.
        ///   The main recall knob.
        /// * `return_distance` - Skips the copy into numpy, not the computation.
        /// * `verbose` - Progress to the process stdout.
        ///
        /// ### Returns
        ///
        /// `(indices, distances)`, dense and padded. See [`QueryOut`].
        #[pyo3(signature = (q, k, *, nprobe = None, return_distance = true, verbose = false))]
        fn query<'py>(
            &self,
            py: Python<'py>,
            q: &Bound<'py, PyAny>,
            k: usize,
            nprobe: Option<usize>,
            return_distance: bool,
            verbose: bool,
        ) -> PyResult<QueryOut<'py>> {
            match &self.inner {
                IvfSq8Inner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                    query_ivf_sq8_index((data, n, dim), idx, k, nprobe, return_distance, verbose)
                }),
                IvfSq8Inner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                    query_ivf_sq8_index((data, n, dim), idx, k, nprobe, return_distance, verbose)
                }),
            }
        }

        /// Full kNN graph over the indexed data.
        ///
        /// Takes the Voronoi-cell fast path rather than re-entering from
        /// outside.
        ///
        /// ### Params
        ///
        /// * `k` - Neighbours per point.
        /// * `nprobe` - Cells to visit, or `None` for the heuristic.
        /// * `return_distance` - Skips the copy into numpy, not the computation.
        /// * `verbose` - Progress to the process stdout.
        ///
        /// ### Returns
        ///
        /// `(indices, distances)` for every indexed point. See [`QueryOut`].
        #[pyo3(signature = (k, *, nprobe = None, return_distance = true, verbose = false))]
        fn query_self<'py>(
            &self,
            py: Python<'py>,
            k: usize,
            nprobe: Option<usize>,
            return_distance: bool,
            verbose: bool,
        ) -> PyResult<QueryOut<'py>> {
            match &self.inner {
                IvfSq8Inner::F32(idx) => {
                    self_arm!(py, k, || query_ivf_sq8_self(
                        idx,
                        k,
                        nprobe,
                        return_distance,
                        verbose
                    ))
                }
                IvfSq8Inner::F64(idx) => {
                    self_arm!(py, k, || query_ivf_sq8_self(
                        idx,
                        k,
                        nprobe,
                        return_distance,
                        verbose
                    ))
                }
            }
        }
    }
);
