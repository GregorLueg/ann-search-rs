//! Float-type dispatch for the per-algorithm methods.
//!
//! The two dtype arms of every `build`, `query` and `query_self` are
//! token-identical and differ only in the type inferred for `T`, so each is
//! written once here and expanded twice. `$mk` is an expression fragment, which
//! means each expansion is an independent closure with its own inference; a
//! generic helper function could not do this without restating every
//! algorithm's trait bounds, and those bounds differ per algorithm.
//!
//! Note the nesting in every macro: `py.detach(|| pool::run(|| ...))`, never
//! the other way round. `Python<'py>` is not `Send`, so a `pool::run` closure
//! capturing it would not satisfy `Ungil`; and the GIL should be dropped before
//! the rayon fan-out starts regardless.

use pyo3::prelude::*;

///////////
// Types //
///////////

/// What every `query` and `query_self` hands back.
///
/// Dense `(n, k)` neighbour indices, and distances unless the caller asked to
/// skip them. The distance array is erased to `PyAny` because its element type
/// follows the index, and the two dtype arms have to agree on one return type.
pub(crate) type QueryOut<'py> = (Bound<'py, numpy::PyArray2<i64>>, Option<Bound<'py, PyAny>>);

////////////
// Macros //
////////////

/// Build an index from a numpy array, picking the arm from the array's dtype.
///
/// Expands `$mk` once per float type. `$mk` must be a closure
/// `|data, n, dim| -> Result<Index<T>, AnnSearchErrors>`; the four infallible
/// builders in the library wrap their result in `Ok` at the call site so every
/// arm has the same shape.
macro_rules! build_dispatch {
    ($py:ident, $x:ident, $inner:ident, $mk:expr) => {{
        if let Ok(a) = $x.extract::<::numpy::PyReadonlyArray2<'_, f32>>() {
            let (data, n, dim) = crate::convert::flat(&a)?;
            let f = $mk;
            let idx = $py
                .detach(|| crate::pool::run(|| f(data, n, dim)))
                .map_err(crate::error::AnnErr)?;
            Ok(Self {
                inner: $inner::F32(idx),
            })
        } else if let Ok(a) = $x.extract::<::numpy::PyReadonlyArray2<'_, f64>>() {
            let (data, n, dim) = crate::convert::flat(&a)?;
            let f = $mk;
            let idx = $py
                .detach(|| crate::pool::run(|| f(data, n, dim)))
                .map_err(crate::error::AnnErr)?;
            Ok(Self {
                inner: $inner::F64(idx),
            })
        } else {
            Err(::pyo3::exceptions::PyTypeError::new_err(
                "X must be a 2-D numpy array of dtype float32 or float64",
            ))
        }
    }};
}

/// One arm of a cross-set query.
///
/// The query array has to match the index's own float type. Casting it here
/// instead would hide a silent doubling of the caller's memory, so the Python
/// layer does the conversion knowingly and this is the backstop.
macro_rules! query_arm {
    ($py:ident, $q:ident, $k:ident, $t:ty, $label:literal, $mk:expr) => {{
        let a = $q
            .extract::<::numpy::PyReadonlyArray2<'_, $t>>()
            .map_err(|_| {
                ::pyo3::exceptions::PyTypeError::new_err(::std::concat!(
                    "index is ",
                    $label,
                    "; the query array must be ",
                    $label
                ))
            })?;
        let (data, n, dim) = crate::convert::flat(&a)?;
        let f = $mk;
        let (ids, dists) = $py
            .detach(|| crate::pool::run(|| f(data, n, dim)))
            .map_err(crate::error::AnnErr)?;
        crate::dispatch::pack($py, ids, dists, $k)
    }};
}

/// One arm of a self-query, producing the full kNN graph over the indexed data.
///
/// Every index has its own fast path for this (IVF walks its Voronoi cells,
/// HNSW stays inside the graph), which is why it is a separate entry point
/// rather than a query against the fitted data.
macro_rules! self_arm {
    ($py:ident, $k:ident, $mk:expr) => {{
        let f = $mk;
        let (ids, dists) = $py
            .detach(|| crate::pool::run(f))
            .map_err(crate::error::AnnErr)?;
        crate::dispatch::pack($py, ids, dists, $k)
    }};
}

pub(crate) use {build_dispatch, query_arm, self_arm};

//////////////
// Packing  //
//////////////

/// Shared tail of every query arm: densify and hand back to Python.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `ids` - Ragged neighbour indices, one row per query.
/// * `dists` - Ragged distances, or `None` when the caller skipped them.
/// * `k` - Neighbours requested, and the row stride of both outputs.
///
/// ### Returns
///
/// `(indices, distances)` as numpy arrays, with short rows padded. See
/// [`crate::convert::pack_idx`] and [`crate::convert::pack_dist`] for the
/// padding values and why they were chosen.
pub(crate) fn pack<'py, T>(
    py: Python<'py>,
    ids: Vec<Vec<usize>>,
    dists: Option<Vec<Vec<T>>>,
    k: usize,
) -> PyResult<QueryOut<'py>>
where
    T: numpy::Element + num_traits::Float,
{
    let ids = crate::convert::pack_idx(py, &ids, k)?;
    let dists = match dists {
        Some(d) => Some(crate::convert::pack_dist(py, &d, k)?.into_any()),
        None => None,
    };
    Ok((ids, dists))
}
