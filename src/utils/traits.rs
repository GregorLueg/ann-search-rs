//! Shared traits and trait boundaries that are used across the crate.

use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::iter::Sum;

use crate::utils::SimdDistance;

#[cfg(feature = "serialise")]
use serde::{de::DeserializeOwned, Serialize};

/// Trait for floating-point types used in the ann-search-rs crate. Contains
/// SIMD optimised operations, needed trait boundaries for faer matrix
/// operations and general numerical trait implementations
///
/// ### Note
///
/// This is the `serialise`-off arm. The `serialise` arm below must carry the
/// same bounds plus `Serialize + DeserializeOwned`; nothing enforces that, and
/// a bound added to one arm only shows up as "builds with the feature, fails
/// without". Change both.
#[cfg(not(feature = "serialise"))]
pub trait AnnSearchFloat:
    Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + ComplexField
{
}

#[cfg(not(feature = "serialise"))]
impl<T> AnnSearchFloat for T where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + SimdDistance + ComplexField
{
}

/// Trait for floating-point types used in the ann-search-rs crate. Contains
/// SIMD optimised operations, needed trait boundaries for faer matrix
/// operations and general numerical trait implementations
///
/// ### Note
///
/// With the `serialise` feature on, `Serialize` and `DeserializeOwned` join the
/// bound. Every index derives serde conditionally, and an index over `T` is only
/// serialisable if `T` is; folding that in here keeps the ~26 `IndexIo` impls
/// down to their `KIND` tag instead of repeating the bounds. `f32` and `f64`
/// both satisfy it either way.
///
/// This does make `serialise` non-additive. Cargo unifies features across the
/// dependency graph, so one crate anywhere turning it on adds the serde pair
/// for every other consumer. `SimdDistance` is public and unsealed, so a
/// downstream custom float type would stop compiling for a reason its own
/// manifest does not explain. Keep the bounds identical to the arm above
/// otherwise.
#[cfg(feature = "serialise")]
pub trait AnnSearchFloat:
    Float
    + FromPrimitive
    + ToPrimitive
    + Send
    + Sync
    + Sum
    + SimdDistance
    + ComplexField
    + Serialize
    + DeserializeOwned
{
}

#[cfg(feature = "serialise")]
impl<T> AnnSearchFloat for T where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + Sum
        + SimdDistance
        + ComplexField
        + Serialize
        + DeserializeOwned
{
}
