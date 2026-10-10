//! Experimental MLX backend (Apple Silicon only).
//!
//! Not a cubecl runtime: cubecl has no MLX backend, so each algorithm here is
//! re-expressed in MLX ops through mlx-c. Exists to compare MLX's tuned ops
//! against the cubecl/wgpu kernels in [`crate::gpu`]. f32 only, since MLX has
//! no f64 on the GPU.

pub mod cagra_mlx;
pub mod exhaustive_mlx;
mod ffi;
