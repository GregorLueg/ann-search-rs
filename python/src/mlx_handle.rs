//! Shared plumbing for the MLX handles.
//!
//! MLX streams are thread affine, so the core crate keeps every MLX index
//! `!Send` and `!Sync`. That changes two things relative to the GPU handles:
//!
//! - The classes are `unsendable`: pyo3 raises if one is touched from a thread
//!   other than the one that built it.
//! - Work runs on the calling thread, not through [`crate::pool::run`], whose
//!   `install` would hop onto a rayon worker. Host-side parallelism inside the
//!   MLX paths uses rayon's global pool, so `set_num_threads` does not govern
//!   it.

use pyo3::prelude::*;

/// Asserts `Send` for a value that never actually leaves the current thread.
struct StayOnThread<T>(T);

// SAFETY: only used by `detach` below, where `Python::detach` runs the closure
// synchronously on the calling thread and hands the result straight back. The
// closures capture MLX handles and slices into numpy buffers, never a Python
// object, so releasing the GIL around them is sound.
unsafe impl<T> Send for StayOnThread<T> {}

impl<T> StayOnThread<T> {
    /// Unwrap. A method rather than `.0`, so a `move` closure captures the
    /// whole wrapper and not just its (non-`Send`) field.
    fn into_inner(self) -> T {
        self.0
    }
}

/// Run thread-affine MLX work with the GIL released.
///
/// ### Params
///
/// * `py` - GIL token.
/// * `f` - The build or query. Must not touch Python objects.
///
/// ### Returns
///
/// Whatever `f` returns.
pub(crate) fn detach<R>(py: Python<'_>, f: impl FnOnce() -> R) -> R {
    let f = StayOnThread(f);
    py.detach(move || StayOnThread(f.into_inner()()))
        .into_inner()
}

/// Borrow a numpy array as `f32`, or explain why it cannot be.
///
/// ### Params
///
/// * `x` - The array. The Python layer casts first, so this only fires when
///   the compiled handle is driven directly.
///
/// ### Returns
///
/// The read-only borrow, or a `TypeError` naming the constraint.
pub(crate) fn f32_array<'py>(x: &Bound<'py, PyAny>) -> PyResult<numpy::PyReadonlyArray2<'py, f32>> {
    x.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "MLX indices are float32 only; pass a C-contiguous float32 array",
        )
    })
}

/// Generate the opaque handle for one MLX index.
///
/// ### Params
///
/// * `$cls` - Name of the generated `#[pyclass]` struct.
/// * `$index` - The library's index type. MLX is f32 only, so not generic.
/// * `$name` - Name Python sees, also used in reprs and error messages.
/// * `$extra` - The per-algorithm `build`, `query` and `query_self`, spliced
///   into the one `#[pymethods]` block pyo3 allows per class.
macro_rules! mlx_handle {
    ($cls:ident, $index:ident, $name:literal, { $($extra:tt)* }) => {
        #[::pyo3::pyclass(name = $name, module = "ann_search._ann_search", unsendable)]
        pub struct $cls {
            /// The built index, resident on the device.
            pub(crate) inner: $index,
            /// Rows the index was built from; not every MLX index exposes it.
            pub(crate) n: usize,
            /// Columns the index was built from, for the same reason.
            pub(crate) dim: usize,
        }

        #[::pyo3::pymethods]
        impl $cls {
            /// Element type the index was built with.
            ///
            /// ### Returns
            ///
            /// Always `"float32"`; MLX has no f64 on the GPU.
            #[getter]
            fn dtype(&self) -> &'static str {
                "float32"
            }

            /// Number of indexed samples.
            ///
            /// ### Returns
            ///
            /// The row count the index was built from.
            #[getter]
            fn n_samples(&self) -> usize {
                self.n
            }

            /// Number of features per sample.
            ///
            /// ### Returns
            ///
            /// The column count the index was built from.
            #[getter]
            fn dim(&self) -> usize {
                self.dim
            }

            /// Debug representation.
            ///
            /// ### Returns
            ///
            /// The algorithm name with its shape.
            fn __repr__(&self) -> String {
                format!("<{} n_samples={} dim={} dtype=float32>", $name, self.n, self.dim)
            }

            $($extra)*
        }
    };
}

pub(crate) use mlx_handle;
