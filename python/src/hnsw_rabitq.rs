//! HNSW-RaBitQ handle.
//!
//! An HNSW linked on exact distances whose float vectors are then dropped, so
//! the index is its topology plus one multi-bit RaBitQ code per vertex. Queries
//! answer from the codes alone.

use ann_search_rs::binary::rabitq::codec::HnswRaBitQIndex;
use ann_search_rs::{build_hnsw_rabitq_index, query_hnsw_rabitq_index, query_hnsw_rabitq_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(
    PyHnswRaBitQ,
    HnswRaBitQInner,
    HnswRaBitQIndex,
    "HnswRaBitQ",
    method,
    {
        /// Link the graph on exact distances, then encode and drop the vectors.
        ///
        /// ### Params
        ///
        /// * `x` - Samples by features, C-contiguous float32 or float64.
        /// * `m` - Edges per node on the upper layers; the base layer gets
        ///   `2 * m`.
        /// * `ef_construction` - Candidate list size during insertion.
        /// * `metric` - Already validated by the Python layer. Manhattan is not
        ///   supported and the library rejects it.
        /// * `ex_bits` - Magnitude bits per coordinate on top of the sign bit,
        ///   `0..=8`. The library rejects anything wider.
        /// * `nlist` - Centroids the codes are taken against, or `None` for
        ///   `sqrt(n)`.
        /// * `seed` - Fixes the level assignment, the rotation and the k-means.
        /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
        ///
        /// ### Returns
        ///
        /// The built handle, or an unsupported-metric or code-width error.
        #[staticmethod]
        #[pyo3(signature = (
            x, *, m, ef_construction, metric, ex_bits, nlist = None, seed = 42,
            verbose = false
        ))]
        #[allow(clippy::too_many_arguments)]
        fn build(
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            m: usize,
            ef_construction: usize,
            metric: String,
            ex_bits: usize,
            nlist: Option<usize>,
            seed: usize,
            verbose: bool,
        ) -> PyResult<Self> {
            build_dispatch!(py, x, HnswRaBitQInner, |data, n, dim| {
                build_hnsw_rabitq_index(
                    (data, n, dim),
                    m,
                    ef_construction,
                    &metric,
                    ex_bits,
                    nlist,
                    None,
                    seed,
                    verbose,
                )
            })
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
                HnswRaBitQInner::F32(idx) => {
                    query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                        query_hnsw_rabitq_index(
                            (data, n, dim),
                            idx,
                            k,
                            ef_search,
                            return_distance,
                            verbose,
                        )
                    })
                }
                HnswRaBitQInner::F64(idx) => {
                    query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                        query_hnsw_rabitq_index(
                            (data, n, dim),
                            idx,
                            k,
                            ef_search,
                            return_distance,
                            verbose,
                        )
                    })
                }
            }
        }

        /// Full kNN graph over the indexed data.
        ///
        /// Stored vertices query through their own codes, so nothing is
        /// re-encoded.
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
                HnswRaBitQInner::F32(idx) => {
                    self_arm!(py, k, || query_hnsw_rabitq_self(
                        idx,
                        k,
                        ef_search,
                        return_distance,
                        verbose
                    ))
                }
                HnswRaBitQInner::F64(idx) => {
                    self_arm!(py, k, || query_hnsw_rabitq_self(
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
