use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::iter::Sum;

use crate::utils::SimdDistance;

/// Trait for floating-point types used in Bixverse. Has all of the common
/// floating-point operations and traits.
pub trait AnnSearchFloat:
    Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance
{
}

impl<T> AnnSearchFloat for T where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance
{
}
