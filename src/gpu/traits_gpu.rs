//! General trait implementations for the cubeCL backend to simplify this

use cubecl::frontend::{CubePrimitive, Float};
use cubecl::CubeElement;

/// Trait for GPU-accelerated operations in ann-search-rs
pub trait AnnSearchGpuFloat: Float + CubePrimitive + CubeElement {}

impl<T> AnnSearchGpuFloat for T where T: Float + CubePrimitive + CubeElement {}
