//! Shared plumbing for the GPU handles.
//!
//! The GPU indices differ from the CPU ones in three ways that the CPU
//! `ann_handle!` macro cannot absorb, which is why they get their own:
//!
//! - **f32 only.** WGSL has no `f64`, so there is no `F64` arm to dispatch to
//!   and the dtype enum collapses to nothing. The Python layer casts on the way
//!   in rather than letting a silent `f64` build fail deep in a kernel.
//! - **No serialisation.** GPU indices sit outside the crate's `serialise`
//!   feature, so there is no `save`, `load` or pickle. The Python estimator
//!   raises for those rather than exposing a half-working handle.
//! - **`n` and `dim` are not uniformly public.** `IvfIndexGpu` keeps them
//!   private, so every handle carries its own copy, recorded at build time.
//!
//! The runtime is pinned to `WgpuRuntime`. `R` is a type parameter in the
//! library so the kernels can be tested against the CPU backend, but a generic
//! cannot cross into Python: the device type is `R::Device`, and there is
//! nothing on the Python side to choose it with.

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

/// The only runtime the bindings expose.
pub(crate) type Rt = WgpuRuntime;

/// Fetch the default wgpu device.
///
/// ### Returns
///
/// The adapter wgpu picks for this machine: Metal on macOS, Vulkan or DX12
/// elsewhere, falling back to whatever `WgpuDevice::default()` resolves to.
pub(crate) fn default_device() -> WgpuDevice {
    WgpuDevice::default()
}

/// Borrow a numpy array as `f32`, or explain why it cannot be.
///
/// The Python layer casts before it gets here, so this only fires when the
/// compiled handle is driven directly. pyo3's own conversion error names the
/// Rust type rather than the constraint, which is not much use from Python.
macro_rules! f32_array {
    ($x:ident) => {
        $x.extract::<::numpy::PyReadonlyArray2<'_, f32>>()
            .map_err(|_| {
                ::pyo3::exceptions::PyTypeError::new_err(
                    "GPU indices are float32 only, since WGSL has no f64; pass a \
                     C-contiguous float32 array",
                )
            })?
    };
}

/// Generate the opaque handle for one GPU index.
///
/// ### Params
///
/// * `$cls` - Name of the generated `#[pyclass]` struct.
/// * `$index` - The library's index type, generic over float and runtime.
/// * `$name` - Name Python sees, also used in reprs and error messages.
/// * `$extra` - The per-algorithm `build`, `query` and `query_self`, spliced
///   into the same `#[pymethods]` block because only one such block per class
///   is allowed without pyo3's `multiple-pymethods` feature.
macro_rules! gpu_handle {
    ($cls:ident, $index:ident, $name:literal, { $($extra:tt)* }) => {
        #[::pyo3::pyclass(name = $name, module = "ann_search._ann_search")]
        pub struct $cls {
            /// The built index. Always `f32`, always on `WgpuRuntime`.
            pub(crate) inner: $index<f32, crate::gpu_handle::Rt>,
            /// Rows the index was built from, recorded here because not every
            /// GPU index exposes its own.
            pub(crate) n: usize,
            /// Columns the index was built from, recorded for the same reason.
            pub(crate) dim: usize,
        }

        #[::pyo3::pymethods]
        impl $cls {
            /// Element type the index was built with.
            ///
            /// ### Returns
            ///
            /// Always `"float32"`. Present so the GPU handles answer the same
            /// question as the CPU ones.
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

pub(crate) use {f32_array, gpu_handle};
