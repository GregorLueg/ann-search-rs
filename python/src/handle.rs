//! The `ann_handle!` macro: one opaque `#[pyclass]` per algorithm.
//!
//! Every index is generic over the float type, so each handle carries an
//! `enum Inner { F32(..), F64(..) }` and dispatches on it. One pyclass per
//! (algorithm, dtype) pair would not remove the dtype check, only move it out
//! of Rust and into a Python invariant that will drift, and it would leak
//! `_HnswF32` into every repr, error message and pickle.
//!
//! The macro expands to **concrete `f32` / `f64` code, never generics**. That
//! is what discharges the per-algorithm trait bounds: `HnswState`,
//! `VamanaState`, `ApplySortedUpdates` and `NNDescentQuery` are implemented
//! only for the concrete float types, so a concrete expansion needs no `where`
//! clause and does not even need the traits in scope.
//!
//! Handles are `#[pyclass(frozen)]`. Every index is a plain `Vec`-of-POD with
//! no interior mutability, so this is sound, it drops pyo3's runtime borrow
//! flag, and it lets two Python threads query one index at once.
//!
//! One wrinkle: `n` and `dim` are public fields on the CPU indices and private
//! ones behind accessors on the quantised indices. Rather than churn the
//! library's structs, the macro takes a `field` / `method` token and
//! [`read_shape!`] expands to whichever spelling applies.

/// Read `n` or `dim` off an index, whichever way that index exposes it.
///
/// ### Params
///
/// * `field` / `method` - How this index exposes its shape.
/// * `$i` - The index.
/// * `$f` - `n` or `dim`.
///
/// ### Returns
///
/// The value, as `usize`.
macro_rules! read_shape {
    (field, $i:expr, $f:ident) => {
        $i.$f
    };
    (method, $i:expr, $f:ident) => {
        $i.$f()
    };
}

/// Generate the opaque handle for one index type.
///
/// ### Params
///
/// * `$cls` - Name of the generated `#[pyclass]` struct.
/// * `$inner` - Name of the generated float-dispatch enum.
/// * `$index` - The library's index type, generic over its float.
/// * `$name` - Name Python sees, also used in reprs and error messages.
/// * `$shape` - `field` when the index exposes `n` and `dim` as public fields,
///   `method` when it exposes them as accessors. See [`read_shape!`].
/// * `$extra` - The per-algorithm `build`, `query` and `query_self`, spliced
///   into the same `#[pymethods]` block because only one such block per class
///   is allowed without pyo3's `multiple-pymethods` feature.
macro_rules! ann_handle {
    ($cls:ident, $inner:ident, $index:ident, $name:literal, $shape:ident, { $($extra:tt)* }) => {
        /// Float-type dispatch for the handle below.
        pub(crate) enum $inner {
            /// Index built over `f32` samples.
            F32($index<f32>),
            /// Index built over `f64` samples.
            F64($index<f64>),
        }

        #[::pyo3::pyclass(name = $name, module = "ann_search._ann_search", frozen)]
        pub struct $cls {
            /// The built index, tagged by its float type.
            pub(crate) inner: $inner,
        }

        #[::pyo3::pymethods]
        impl $cls {
            /// Element type the index was built with.
            ///
            /// ### Returns
            ///
            /// `"float32"` or `"float64"`.
            #[getter]
            fn dtype(&self) -> &'static str {
                match &self.inner {
                    $inner::F32(_) => "float32",
                    $inner::F64(_) => "float64",
                }
            }

            /// Number of indexed samples.
            ///
            /// ### Returns
            ///
            /// The row count the index was built from.
            #[getter]
            fn n_samples(&self) -> usize {
                match &self.inner {
                    $inner::F32(i) => crate::handle::read_shape!($shape, i, n),
                    $inner::F64(i) => crate::handle::read_shape!($shape, i, n),
                }
            }

            /// Number of features per sample.
            ///
            /// ### Returns
            ///
            /// The column count the index was built from.
            #[getter]
            fn dim(&self) -> usize {
                match &self.inner {
                    $inner::F32(i) => crate::handle::read_shape!($shape, i, dim),
                    $inner::F64(i) => crate::handle::read_shape!($shape, i, dim),
                }
            }

            /// Write the index bundle to disk.
            ///
            /// ### Params
            ///
            /// * `path` - Target *directory*, not a file. Created if missing.
            ///
            /// ### Returns
            ///
            /// Nothing, or the first IO or encoding error.
            fn save(
                &self,
                py: ::pyo3::Python<'_>,
                path: ::std::path::PathBuf,
            ) -> crate::error::AnnResult<()> {
                py.detach(|| match &self.inner {
                    $inner::F32(i) => ::ann_search_rs::save_index(i, &path),
                    $inner::F64(i) => ::ann_search_rs::save_index(i, &path),
                })?;
                Ok(())
            }

            /// Read an index bundle written by [`save`](Self::save).
            ///
            /// The float width is sniffed rather than passed in: the bundle
            /// header records it and is checked before anything is decoded, so
            /// the `f32` attempt costs one open and a header read when the file
            /// turns out to hold `f64`.
            ///
            /// ### Params
            ///
            /// * `path` - Directory holding the bundle.
            ///
            /// ### Returns
            ///
            /// The reconstructed handle, or the first header mismatch, IO or
            /// decoding error. A bundle written by a different algorithm fails
            /// the kind check.
            #[staticmethod]
            fn load(
                py: ::pyo3::Python<'_>,
                path: ::std::path::PathBuf,
            ) -> crate::error::AnnResult<Self> {
                let inner = py.detach(
                    || -> ::std::result::Result<$inner, ::ann_search_rs::prelude::AnnSearchErrors> {
                        match ::ann_search_rs::load_index::<$index<f32>>(&path) {
                            Ok(i) => Ok($inner::F32(i)),
                            Err(::ann_search_rs::prelude::AnnSearchErrors::FloatWidthMismatch {
                                ..
                            }) => Ok($inner::F64(::ann_search_rs::load_index::<$index<f64>>(
                                &path,
                            )?)),
                            Err(e) => Err(e),
                        }
                    },
                )?;
                Ok(Self { inner })
            }

            /// Serialise the handle for pickling.
            ///
            /// ### Returns
            ///
            /// One dtype tag byte followed by a verbatim `index.bin`. See
            /// [`crate::state`] for why it round-trips a tempdir rather than
            /// calling bincode directly.
            fn __getstate__(
                &self,
                py: ::pyo3::Python<'_>,
            ) -> crate::error::AnnResult<::pyo3::Py<::pyo3::types::PyBytes>> {
                let buf = py.detach(|| match &self.inner {
                    $inner::F32(i) => crate::state::to_state(i, crate::state::F32_TAG),
                    $inner::F64(i) => crate::state::to_state(i, crate::state::F64_TAG),
                })?;
                Ok(::pyo3::types::PyBytes::new(py, &buf).unbind())
            }

            /// Rebuild a handle from a [`__getstate__`](Self::__getstate__)
            /// payload.
            ///
            /// ### Params
            ///
            /// * `state` - Tag byte followed by an `index.bin`.
            ///
            /// ### Returns
            ///
            /// The reconstructed handle. No algorithm tag is needed: a payload
            /// from a different index fails the bundle's kind check before
            /// anything is decoded.
            #[staticmethod]
            fn from_bytes(
                py: ::pyo3::Python<'_>,
                state: &[u8],
            ) -> crate::error::AnnResult<Self> {
                let inner = py.detach(
                    || -> ::std::result::Result<$inner, ::ann_search_rs::prelude::AnnSearchErrors> {
                        match state.first().copied() {
                            Some(crate::state::F32_TAG) => {
                                Ok($inner::F32(crate::state::from_state(state)?))
                            }
                            Some(crate::state::F64_TAG) => {
                                Ok($inner::F64(crate::state::from_state(state)?))
                            }
                            _ => Err(::ann_search_rs::prelude::AnnSearchErrors::DecodeError(
                                ::std::concat!("corrupt ", $name, " payload: bad dtype tag").into(),
                            )),
                        }
                    },
                )?;
                Ok(Self { inner })
            }

            /// Pickle hook.
            ///
            /// `__reduce__` rather than `__setstate__`: the latter needs
            /// `&mut self`, which `frozen` forbids, and would route pickle
            /// through `__new__` and therefore an empty-handle variant with a
            /// dead match arm in every method.
            ///
            /// ### Returns
            ///
            /// `(from_bytes, (state,))`.
            fn __reduce__<'py>(
                &self,
                py: ::pyo3::Python<'py>,
            ) -> ::pyo3::PyResult<(
                ::pyo3::Bound<'py, ::pyo3::PyAny>,
                (::pyo3::Py<::pyo3::types::PyBytes>,),
            )> {
                use ::pyo3::types::PyAnyMethods;
                let ctor = py.get_type::<Self>().getattr("from_bytes")?;
                Ok((ctor, (self.__getstate__(py)?,)))
            }

            /// Debug representation.
            ///
            /// ### Returns
            ///
            /// The algorithm name with its shape and float type.
            fn __repr__(&self) -> String {
                format!(
                    "<{} n_samples={} dim={} dtype={}>",
                    $name,
                    self.n_samples(),
                    self.dim(),
                    self.dtype()
                )
            }

            $($extra)*
        }
    };
}

pub(crate) use {ann_handle, read_shape};
