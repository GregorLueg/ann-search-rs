//! numpy in, numpy out.
//!
//! Input goes through `PyReadonlyArray2::as_slice`, which feeds the crate's
//! `impl AnnMatrix<T> for (&[T], usize, usize)` directly. Output has to be
//! densified: `query_parallel` does not pad, so an approximate index can hand
//! back a row shorter than `k`.

use numpy::{
    Element, IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/////////////////
// Fill values //
/////////////////

/// Neighbour slot that was never populated.
///
/// `-1` is the pynndescent and umap convention. It survives fancy indexing as
/// "the last row" rather than raising, so callers must mask on `>= 0` before
/// slicing with it; `kneighbors_graph` does.
const NO_NEIGHBOUR: i64 = -1;

////////////
// Inputs //
////////////

/// Borrow a C-contiguous 2-D array as `(data, n_rows, n_cols)`.
///
/// The returned slice borrows from `a`, so `a` must outlive any
/// `Python::detach` the slice is passed into. Keep this call outside the
/// closure.
///
/// ### Params
///
/// * `a` - Read-only view of a 2-D numpy array.
///
/// ### Returns
///
/// `(data, n_rows, n_cols)`, row-major, in the shape the crate's
/// `AnnMatrix` tuple impl wants. Errors if the array is not C-contiguous;
/// the Python layer runs `np.ascontiguousarray` first, so this is a backstop.
pub(crate) fn flat<'a, T: Element>(
    a: &'a PyReadonlyArray2<'_, T>,
) -> PyResult<(&'a [T], usize, usize)> {
    let shape = a.shape();
    let (n, dim) = (shape[0], shape[1]);
    let data = a.as_slice().map_err(|_| {
        PyValueError::new_err("array must be C-contiguous; use np.ascontiguousarray")
    })?;
    Ok((data, n, dim))
}

/////////////
// Outputs //
/////////////

/// Flatten ragged rows into a dense `n * k` buffer, padding short rows.
///
/// One allocation, no reallocation: the buffer is filled with `fill` up front
/// and each row's prefix is written over it. Rows longer than `k` are
/// truncated, which cannot happen today but keeps a future over-returning
/// index from panicking here.
///
/// ### Params
///
/// * `rows` - One row per query, each holding between 0 and `k` entries.
/// * `k` - Neighbours requested, and the row stride of the output.
/// * `fill` - Value written into slots no row entry reached.
/// * `convert` - Applied to each entry on the way in.
///
/// ### Returns
///
/// A row-major buffer of `rows.len() * k` elements.
fn densify<S, D>(rows: &[Vec<S>], k: usize, fill: D, convert: impl Fn(S) -> D) -> Vec<D>
where
    S: Copy,
    D: Copy,
{
    let mut out = vec![fill; rows.len() * k];
    for (i, row) in rows.iter().enumerate() {
        let slots = &mut out[i * k..(i + 1) * k];
        for (slot, &value) in slots.iter_mut().zip(row.iter().take(k)) {
            *slot = convert(value);
        }
    }
    out
}

/// Pack ragged neighbour indices into a dense `(n, k)` `int64` array.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `rows` - Neighbour indices, one row per query.
/// * `k` - Neighbours requested.
///
/// ### Returns
///
/// An `(n, k)` array. `into_pyarray` moves the buffer into the numpy object
/// and `reshape` returns a view, so neither copies. Short rows are padded with
/// [`NO_NEIGHBOUR`].
pub(crate) fn pack_idx<'py>(
    py: Python<'py>,
    rows: &[Vec<usize>],
    k: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let out = densify(rows, k, NO_NEIGHBOUR, |v| v as i64);
    out.into_pyarray(py).reshape([rows.len(), k])
}

/// Pack ragged distances into a dense `(n, k)` array of the index's float type.
///
/// Padding is `+inf` rather than NaN: it keeps each row totally ordered, so
/// `argsort`, `min` and the sortedness callers rely on all still hold, and
/// `sqrt(inf) == inf` leaves the Python layer's Euclidean transform a no-op on
/// padding. NaN would poison all four.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `rows` - Distances, one row per query, aligned with the index rows.
/// * `k` - Neighbours requested.
///
/// ### Returns
///
/// An `(n, k)` array of `T`, matching the element type of the index rather
/// than widening to `f64`.
pub(crate) fn pack_dist<'py, T>(
    py: Python<'py>,
    rows: &[Vec<T>],
    k: usize,
) -> PyResult<Bound<'py, PyArray2<T>>>
where
    T: Element + num_traits::Float,
{
    let out = densify(rows, k, T::infinity(), |v| v);
    out.into_pyarray(py).reshape([rows.len(), k])
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_densify_pads_short_rows() {
        let rows = vec![vec![1usize, 2], vec![3]];
        let out = densify(&rows, 3, NO_NEIGHBOUR, |v| v as i64);
        assert_eq!(out, vec![1, 2, -1, 3, -1, -1]);
    }

    #[test]
    fn test_densify_truncates_long_rows() {
        let rows = vec![vec![1usize, 2, 3, 4]];
        let out = densify(&rows, 2, NO_NEIGHBOUR, |v| v as i64);
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn test_densify_empty_row_is_all_padding() {
        let rows: Vec<Vec<usize>> = vec![vec![]];
        let out = densify(&rows, 3, NO_NEIGHBOUR, |v| v as i64);
        assert_eq!(out, vec![-1, -1, -1]);
    }

    #[test]
    fn test_densify_no_rows_gives_no_buffer() {
        let rows: Vec<Vec<usize>> = vec![];
        assert!(densify(&rows, 5, NO_NEIGHBOUR, |v| v as i64).is_empty());
    }

    #[test]
    fn test_densify_distance_padding_stays_ordered() {
        let rows = vec![vec![0.5f32, 1.5]];
        let out = densify(&rows, 4, f32::INFINITY, |v| v);
        assert_eq!(out[..2], [0.5, 1.5]);
        assert!(out[2..]
            .iter()
            .all(|v| v.is_infinite() && v.is_sign_positive()));
        assert!(out.windows(2).all(|w| w[0] <= w[1]));
    }
}
