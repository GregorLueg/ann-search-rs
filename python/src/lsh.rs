//! LSH handle.
//!
//! Multi-probe locality-sensitive hashing over random projections. Manhattan is
//! not supported.
//!
//! The library's two query entry points disagree on `n_probe`: the cross-set
//! one takes it bare, the self one takes an `Option` and fills it with
//! `num_projections()`. Both are `Option` here, and the cross-set arm applies
//! the same default, so the two sides of the Python surface behave alike.

use ann_search_rs::cpu::lsh::LSHIndex;
use ann_search_rs::{build_lsh_index, query_lsh_index, query_lsh_self};
use pyo3::prelude::*;

use crate::dispatch::{build_dispatch, query_arm, self_arm, QueryOut};
use crate::handle::ann_handle;

ann_handle!(PyLsh, LshInner, LSHIndex, "Lsh", {
    /// Number of random projections per table.
    ///
    /// The natural unit for `n_probe`, and its default: each table offers
    /// `2 * num_projections` single-slot perturbations, so that is the ceiling
    /// past which raising `n_probe` buys nothing.
    ///
    /// ### Returns
    ///
    /// The projection count the index was built with.
    #[getter]
    fn num_projections(&self) -> usize {
        match &self.inner {
            LshInner::F32(i) => i.num_projections(),
            LshInner::F64(i) => i.num_projections(),
        }
    }

    /// Bits each quantised projection contributes.
    ///
    /// ### Returns
    ///
    /// What the index resolved `slot_bits` to: 1 for cosine, 2 otherwise,
    /// unless it was pinned at build time.
    #[getter]
    fn slot_bits(&self) -> usize {
        match &self.inner {
            LshInner::F32(i) => i.slot_bits(),
            LshInner::F64(i) => i.slot_bits(),
        }
    }

    /// Hash the data into the tables.
    ///
    /// ### Params
    ///
    /// * `x` - Samples by features, C-contiguous float32 or float64.
    /// * `metric` - Already validated by the Python layer.
    /// * `num_tables` - Independent hash tables. More means better recall and a
    ///   larger index.
    /// * `bits_per_hash` - Bits per table key. Lower widens the buckets, so
    ///   recall rises and queries slow down.
    /// * `slot_bits` - Bits per projection, or `None` to let the index pick: 1
    ///   for cosine, 2 for squared Euclidean, which needs more than a sign to
    ///   see vector magnitude.
    /// * `seed` - Fixes the random projections.
    ///
    /// ### Returns
    ///
    /// The built handle, or a build error when `bits_per_hash` is out of range
    /// or the dataset is too large for the hash width.
    #[staticmethod]
    #[pyo3(signature = (x, *, metric, num_tables, bits_per_hash, slot_bits = None, seed = 42))]
    fn build(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        metric: String,
        num_tables: usize,
        bits_per_hash: usize,
        slot_bits: Option<usize>,
        seed: usize,
    ) -> PyResult<Self> {
        build_dispatch!(py, x, LshInner, |data, n, dim| build_lsh_index(
            (data, n, dim),
            &metric,
            num_tables,
            bits_per_hash,
            slot_bits,
            seed
        ))
    }

    /// Probe the neighbouring buckets for external queries.
    ///
    /// ### Params
    ///
    /// * `q` - Queries by features, matching the index's float type.
    /// * `k` - Neighbours per query.
    /// * `n_probe` - Buckets to probe per table, or `None` for one per
    ///   projection. The main recall knob.
    /// * `max_candidates` - Cap on candidates scored per query, or `None` for
    ///   no cap.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout, not `sys.stdout`. Also
    ///   reports the miss rate, since a query can hash into empty buckets.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)`, dense and padded. See [`QueryOut`].
    #[pyo3(signature = (
        q, k, *, n_probe = None, max_candidates = None, return_distance = true, verbose = false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn query<'py>(
        &self,
        py: Python<'py>,
        q: &Bound<'py, PyAny>,
        k: usize,
        n_probe: Option<usize>,
        max_candidates: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        let probes = n_probe.unwrap_or_else(|| self.num_projections());
        match &self.inner {
            LshInner::F32(idx) => query_arm!(py, q, k, f32, "float32", |data, n, dim| {
                query_lsh_index(
                    (data, n, dim),
                    idx,
                    k,
                    probes,
                    max_candidates,
                    return_distance,
                    verbose,
                )
            }),
            LshInner::F64(idx) => query_arm!(py, q, k, f64, "float64", |data, n, dim| {
                query_lsh_index(
                    (data, n, dim),
                    idx,
                    k,
                    probes,
                    max_candidates,
                    return_distance,
                    verbose,
                )
            }),
        }
    }

    /// Full kNN graph over the indexed data.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per point.
    /// * `n_probe` - Buckets to probe per table, or `None` for one per
    ///   projection.
    /// * `max_candidates` - Cap on candidates scored per point, or `None`.
    /// * `return_distance` - Skips the copy into numpy, not the computation.
    /// * `verbose` - Progress to the process stdout.
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` for every indexed point. See [`QueryOut`].
    #[pyo3(signature = (
        k, *, n_probe = None, max_candidates = None, return_distance = true, verbose = false
    ))]
    fn query_self<'py>(
        &self,
        py: Python<'py>,
        k: usize,
        n_probe: Option<usize>,
        max_candidates: Option<usize>,
        return_distance: bool,
        verbose: bool,
    ) -> PyResult<QueryOut<'py>> {
        match &self.inner {
            LshInner::F32(idx) => self_arm!(py, k, || query_lsh_self(
                idx,
                k,
                n_probe,
                max_candidates,
                return_distance,
                verbose
            )),
            LshInner::F64(idx) => self_arm!(py, k, || query_lsh_self(
                idx,
                k,
                n_probe,
                max_candidates,
                return_distance,
                verbose
            )),
        }
    }
});
