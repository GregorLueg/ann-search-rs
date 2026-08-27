//! Flexible matrix inputs.
//!
//! Every index in this crate flattens its input to a row-major `Vec<T>` on
//! construction and works on slices from there. [`AnnMatrix`] is that single
//! conversion point, which is what lets the public API take a faer matrix, an
//! ndarray 2-D array or a plain row-major buffer without any of the algorithms
//! caring which.

use faer::{Mat, MatRef};
use num_traits::Float;

use crate::utils::{matrix_to_flat, FlattenData};

#[cfg(feature = "ndarray")]
use ndarray::{Array2, ArrayView2};

///////////////
// AnnMatrix //
///////////////

/// Anything that can be handed to a builder or a batch query as a
/// samples-by-features matrix.
///
/// The contract is orientation, not layout: rows are samples, columns are
/// features, and [`AnnMatrix::into_row_major`] must hand back a row-major buffer.
/// Implementations are zero-copy where the source already agrees
/// ([`FlattenData`] and a standard-layout owned `Array2`) and copy where it
/// does not (faer stores column-major, so `MatRef` transposes on the way out).
///
/// Taking `self` by value is what lets [`FlattenData`] implement this as the
/// identity. Every `build_*` and `query_*` in the crate root flattens before
/// delegating, so the index constructors monomorphise once over
/// `FlattenData<T>`. Calling a constructor directly with some other input type
/// instantiates its body a second time; pass [`matrix_to_flat`] output instead
/// if that matters to you.
///
/// ### Note
///
/// `MatRef<'_, T>` is `Copy`, so consuming `self` costs an existing
/// `mat.as_ref()` call site nothing.
pub trait AnnMatrix<T> {
    /// Consume the input and produce a row-major flat buffer with its shape.
    ///
    /// ### Returns
    ///
    /// `(data, n_samples, n_features)`, where
    /// `data.len() == n_samples * n_features` and sample `i` occupies
    /// `data[i * n_features..(i + 1) * n_features]`.
    fn into_row_major(self) -> FlattenData<T>;
}

//////////////////
// Flat sources //
//////////////////

/// Identity conversion. Zero copy, and the shape the whole crate runs on
/// internally.
impl<T> AnnMatrix<T> for FlattenData<T> {
    fn into_row_major(self) -> FlattenData<T> {
        self
    }
}

/// A borrowed row-major buffer with its shape, for callers arriving over FFI
/// who already hold a contiguous array.
///
/// ### Panics
///
/// If `data.len()` is not `n_samples * n_features`.
impl<T> AnnMatrix<T> for (&[T], usize, usize)
where
    T: Copy,
{
    fn into_row_major(self) -> FlattenData<T> {
        let (data, n, dim) = self;
        assert_eq!(
            data.len(),
            n * dim,
            "flat input length {} does not match shape {n} x {dim}",
            data.len()
        );
        (data.to_vec(), n, dim)
    }
}

///////////
// faer  //
///////////

impl<T> AnnMatrix<T> for MatRef<'_, T>
where
    T: Float,
{
    fn into_row_major(self) -> FlattenData<T> {
        matrix_to_flat(self)
    }
}

impl<T> AnnMatrix<T> for &Mat<T>
where
    T: Float,
{
    fn into_row_major(self) -> FlattenData<T> {
        matrix_to_flat(self.as_ref())
    }
}

/// faer stores column-major, so an owned `Mat` still has to be transposed out.
/// Owning it buys nothing.
impl<T> AnnMatrix<T> for Mat<T>
where
    T: Float,
{
    fn into_row_major(self) -> FlattenData<T> {
        matrix_to_flat(self.as_ref())
    }
}

/////////////
// ndarray //
/////////////

#[cfg(feature = "ndarray")]
impl<T> AnnMatrix<T> for ArrayView2<'_, T>
where
    T: Copy,
{
    fn into_row_major(self) -> FlattenData<T> {
        let (n, dim) = (self.nrows(), self.ncols());
        // `as_slice` returns Some only for standard (row-major) layout. Do not
        // reach for `as_slice_memory_order`: on an F-order array it hands back
        // the column-major buffer, which would be written out as row-major.
        match self.as_slice() {
            Some(slice) => (slice.to_vec(), n, dim),
            // `iter` visits in logical order, rightmost index fastest, which is
            // row-major whatever the strides say.
            None => (self.iter().copied().collect(), n, dim),
        }
    }
}

#[cfg(feature = "ndarray")]
impl<T> AnnMatrix<T> for &Array2<T>
where
    T: Copy,
{
    fn into_row_major(self) -> FlattenData<T> {
        self.view().into_row_major()
    }
}

#[cfg(feature = "ndarray")]
impl<T> AnnMatrix<T> for Array2<T>
where
    T: Copy,
{
    fn into_row_major(self) -> FlattenData<T> {
        let (n, dim) = (self.nrows(), self.ncols());

        if !self.is_standard_layout() {
            return (self.iter().copied().collect(), n, dim);
        }

        // Standard layout: hand over the allocation. The check has to come
        // first, `into_raw_vec_and_offset` consumes self and there is no way
        // back. What it returns is the whole allocation, and a sliced array can
        // sit at a non-zero offset with trailing slack.
        let (mut data, offset) = self.into_raw_vec_and_offset();
        let offset = offset.unwrap_or(0);
        if offset != 0 {
            data.drain(..offset);
        }
        data.truncate(n * dim);

        (data, n, dim)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Row-major reference data: element (i, j) is i * 10 + j.
    fn reference(n: usize, dim: usize) -> Vec<f32> {
        (0..n)
            .flat_map(|i| (0..dim).map(move |j| (i * 10 + j) as f32))
            .collect()
    }

    fn faer_mat(n: usize, dim: usize) -> Mat<f32> {
        Mat::from_fn(n, dim, |i, j| (i * 10 + j) as f32)
    }

    #[test]
    fn test_flatten_data_is_identity() {
        let expected = reference(4, 3);
        let (flat, n, dim) = (expected.clone(), 4, 3).into_row_major();
        assert_eq!(flat, expected);
        assert_eq!((n, dim), (4, 3));
    }

    #[test]
    fn test_flat_slice_matches_reference() {
        let expected = reference(4, 3);
        let (flat, n, dim) = (expected.as_slice(), 4, 3).into_row_major();
        assert_eq!(flat, expected);
        assert_eq!((n, dim), (4, 3));
    }

    #[test]
    #[should_panic(expected = "does not match shape")]
    fn test_flat_slice_rejects_wrong_shape() {
        let data = reference(4, 3);
        let _ = (data.as_slice(), 3, 3).into_row_major();
    }

    #[test]
    fn test_faer_matref_transposes_to_row_major() {
        let mat = faer_mat(4, 3);
        let (flat, n, dim) = mat.as_ref().into_row_major();
        assert_eq!(flat, reference(4, 3));
        assert_eq!((n, dim), (4, 3));
    }

    #[test]
    fn test_faer_owned_and_borrowed_agree() {
        let mat = faer_mat(4, 3);
        let borrowed = (&mat).into_row_major();
        let owned = mat.into_row_major();
        assert_eq!(borrowed, owned);
        assert_eq!(borrowed.0, reference(4, 3));
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ndarray_standard_layout_matches_faer() {
        let arr = Array2::from_shape_fn((4, 3), |(i, j)| (i * 10 + j) as f32);
        let (flat, n, dim) = arr.view().into_row_major();
        assert_eq!(flat, reference(4, 3));
        assert_eq!((n, dim), (4, 3));
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ndarray_owned_hands_over_buffer() {
        let arr = Array2::from_shape_fn((4, 3), |(i, j)| (i * 10 + j) as f32);
        assert_eq!(arr.clone().into_row_major().0, reference(4, 3));
        assert_eq!((&arr).into_row_major().0, reference(4, 3));
    }

    /// A transposed view is not standard layout, so this exercises the
    /// `iter` fallback. Getting it wrong yields the column-major buffer.
    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ndarray_transposed_view_stays_row_major() {
        let arr = Array2::from_shape_fn((3, 4), |(i, j)| (j * 10 + i) as f32);
        let view = arr.t();
        assert!(view.as_slice().is_none());

        let (flat, n, dim) = view.into_row_major();
        assert_eq!((n, dim), (4, 3));
        assert_eq!(flat, reference(4, 3));
    }

    /// An owned array that is C-contiguous but sitting at a non-zero offset
    /// with trailing slack, which is what the drain/truncate is for.
    #[cfg(feature = "ndarray")]
    #[test]
    fn test_ndarray_owned_sliced_handles_offset() {
        use ndarray::s;

        let mut arr = Array2::from_shape_fn((6, 3), |(i, j)| (i * 10 + j) as f32);
        arr.slice_collapse(s![1..5, ..]);
        assert!(arr.is_standard_layout());

        let (flat, n, dim) = arr.into_row_major();
        assert_eq!((n, dim), (4, 3));
        let expected: Vec<f32> = (1..5)
            .flat_map(|i| (0..3).map(move |j| (i * 10 + j) as f32))
            .collect();
        assert_eq!(flat, expected);
    }

    /// Every supported input describing the same matrix must produce the same
    /// buffer. This is the property the whole public API leans on.
    #[test]
    fn test_all_inputs_agree() {
        let expected = reference(5, 4);
        let mat = faer_mat(5, 4);

        let mut results = vec![
            mat.as_ref().into_row_major(),
            (&mat).into_row_major(),
            (expected.as_slice(), 5, 4).into_row_major(),
            (expected.clone(), 5, 4).into_row_major(),
        ];

        #[cfg(feature = "ndarray")]
        {
            let arr = Array2::from_shape_fn((5, 4), |(i, j)| (i * 10 + j) as f32);
            results.push(arr.view().into_row_major());
            results.push((&arr).into_row_major());
            results.push(arr.into_row_major());
        }

        for got in &results {
            assert_eq!(got.0, expected);
            assert_eq!((got.1, got.2), (5, 4));
        }
    }

    /// The property the public API actually leans on: the same data through
    /// any accepted input type builds the same index and returns the same
    /// neighbours.
    #[test]
    fn test_public_api_agrees_across_input_types() {
        use crate::{build_hnsw_index, query_hnsw_index};

        let (n, dim) = (200usize, 16usize);
        let flat: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
            .collect();
        let mat = Mat::from_fn(n, dim, |i, j| flat[i * dim + j]);

        let faer_index = build_hnsw_index(mat.as_ref(), 16, 100, "euclidean", 42, false);
        let (faer_idx, faer_dist) =
            query_hnsw_index(mat.as_ref(), &faer_index, 5, 50, true, false).unwrap();
        let faer_dist = faer_dist.unwrap();

        // Same data as a flat row-major buffer: same index, same neighbours.
        let flat_index =
            build_hnsw_index((flat.as_slice(), n, dim), 16, 100, "euclidean", 42, false);
        let (flat_idx, flat_dist) =
            query_hnsw_index((flat.as_slice(), n, dim), &flat_index, 5, 50, true, false).unwrap();

        assert_eq!(faer_idx, flat_idx);
        assert_eq!(faer_dist, flat_dist.unwrap());

        #[cfg(feature = "ndarray")]
        {
            let arr = Array2::from_shape_fn((n, dim), |(i, j)| flat[i * dim + j]);

            let nd_index = build_hnsw_index(arr.view(), 16, 100, "euclidean", 42, false);
            let (nd_idx, nd_dist) =
                query_hnsw_index(arr.view(), &nd_index, 5, 50, true, false).unwrap();
            assert_eq!(faer_idx, nd_idx);
            assert_eq!(faer_dist, nd_dist.unwrap());

            // A transposed view describes the same matrix but reaches the
            // strided fallback, so it must land in the same place.
            let arr_t = Array2::from_shape_fn((dim, n), |(j, i)| flat[i * dim + j]);
            let (t_idx, _) =
                query_hnsw_index(arr_t.t(), &nd_index, 5, 50, true, false).unwrap();
            assert_eq!(faer_idx, t_idx);
        }
    }
}
