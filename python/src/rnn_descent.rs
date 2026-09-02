//! RNN-Descent handle.
//!
//! Relative NN-Descent: builds and prunes the graph in one pass, so it reaches
//! a sparse navigable graph without the separate NSG-style refinement step.

use ann_search_rs::cpu::rnn_descent::RnnDescentIndex;
use ann_search_rs::{build_rnn_descent_index, query_rnn_descent_index, query_rnn_descent_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(
    PyRnnDescent,
    RnnDescentInner,
    RnnDescentIndex,
    "RnnDescent",
    field,
    {
        /// Build and prune the graph.
        ///
        /// ### Params
        ///
        /// * `x` - Samples by features, C-contiguous float32 or float64.
        /// * `metric` - Already validated by the Python layer.
        /// * `s` - Out-degree of the random seed graph.
        /// * `r` - Maximum adjacency per node after reverse edges are added.
        /// * `t1` - Outer rounds.
        /// * `t2` - Neighbour-update passes per outer round.
        /// * `n_trees` - Forest size for the initial graph, or `None` for the
        ///   library's heuristic.
        /// * `seed` - Fixes the seed graph.
        /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
        ///
        /// ### Returns
        ///
        /// The built handle, or a build error.
        #[staticmethod]
        #[pyo3(signature = (
        x, *, metric, s, r, t1, t2, n_trees = None, seed = 42, verbose = false
    ))]
        #[allow(clippy::too_many_arguments)]
        fn build(
            py: Python<'_>,
            x: &Bound<'_, PyAny>,
            metric: String,
            s: usize,
            r: usize,
            t1: usize,
            t2: usize,
            n_trees: Option<usize>,
            seed: usize,
            verbose: bool,
        ) -> PyResult<Self> {
            build_dispatch!(py, x, RnnDescentInner, |data, n, dim| {
                build_rnn_descent_index(
                    (data, n, dim),
                    s,
                    r,
                    t1,
                    t2,
                    &metric,
                    n_trees,
                    seed,
                    verbose,
                )
            })
        }

        /// Walk the graph for external queries.
        ///
        /// ### Params
        ///
        /// * `q` - Queries by features, matching the index's float type.
        /// * `k` - Neighbours per query.
        /// * `ef_search` - Beam width, or `None` for the library's default of 100.
        ///   The main recall knob.
        /// * `k_search` - Neighbours expanded per visited node, or `None` for
        ///   `min(32, r)`.
        /// * `return_distance` - Skips the copy into numpy, not the computation.
        /// * `verbose` - Progress to the process stdout, not `sys.stdout`.
        ///
        /// ### Returns
        ///
        /// `(indices, distances)`, dense and padded. See [`QueryOut`].
        #[pyo3(signature = (
        q, k, *, ef_search = None, k_search = None, return_distance = true, verbose = false
    ))]
        #[allow(clippy::too_many_arguments)]
        fn query<'py>(
            &self,
            py: Python<'py>,
            q: &Bound<'py, PyAny>,
            k: usize,
            ef_search: Option<usize>,
            k_search: Option<usize>,
            return_distance: bool,
            verbose: bool,
        ) -> PyResult<QueryOut<'py>> {
            match &self.inner {
                RnnDescentInner::F32(idx) => {
                    query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                        query_rnn_descent_index(
                            (data, n, dim),
                            idx,
                            k,
                            ef_search,
                            k_search,
                            return_distance,
                            verbose,
                        )
                    })
                }
                RnnDescentInner::F64(idx) => {
                    query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                        query_rnn_descent_index(
                            (data, n, dim),
                            idx,
                            k,
                            ef_search,
                            k_search,
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
        /// * `ef_search` - Beam width, or `None` for the default.
        /// * `k_search` - Neighbours expanded per visited node, or `None`.
        /// * `return_distance` - Skips the copy into numpy, not the computation.
        /// * `verbose` - Progress to the process stdout.
        ///
        /// ### Returns
        ///
        /// `(indices, distances)` for every indexed point. See [`QueryOut`].
        #[pyo3(signature = (
        k, *, ef_search = None, k_search = None, return_distance = true, verbose = false
    ))]
        fn query_self<'py>(
            &self,
            py: Python<'py>,
            k: usize,
            ef_search: Option<usize>,
            k_search: Option<usize>,
            return_distance: bool,
            verbose: bool,
        ) -> PyResult<QueryOut<'py>> {
            match &self.inner {
                RnnDescentInner::F32(idx) => self_arm!(py, k, || query_rnn_descent_self(
                    idx,
                    k,
                    ef_search,
                    k_search,
                    return_distance,
                    verbose
                )),
                RnnDescentInner::F64(idx) => self_arm!(py, k, || query_rnn_descent_self(
                    idx,
                    k,
                    ef_search,
                    k_search,
                    return_distance,
                    verbose
                )),
            }
        }
    }
);
