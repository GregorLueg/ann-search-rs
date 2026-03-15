//! Shared traits and trait boundaries that are used across the crate.

use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::iter::Sum;

use crate::utils::SimdDistance;

/// Trait for floating-point types used in the ann-search-rs crate. Contains
/// SIMD optimised operations, needed trait boundaries for faer matrix
/// operations and general numerical trait implementations
pub trait AnnSearchFloat:
    Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + ComplexField
{
}

impl<T> AnnSearchFloat for T where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + ComplexField
{
}
