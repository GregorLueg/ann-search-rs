//! Mapping [`AnnSearchErrors`] onto Python exceptions.
//!
//! No new error enum: the library already has one, and this crate adds no
//! failure modes of its own beyond what pyo3 raises directly. `AnnSearchErrors`
//! and `PyErr` are both foreign here, so the `From` impl needs a local newtype.

use ann_search_rs::prelude::AnnSearchErrors;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;

////////////////
// Exceptions //
////////////////

create_exception!(
    _ann_search,
    AnnSearchError,
    PyException,
    "Base class for every error raised by the ann-search Rust core."
);

create_exception!(
    _ann_search,
    IndexIoError,
    AnnSearchError,
    "An index bundle is missing, truncated, or of the wrong kind, float width or format version."
);

/////////////
// Newtype //
/////////////

/// Carries an [`AnnSearchErrors`] to the FFI boundary.
///
/// A tuple struct, which means it doubles as the conversion function:
/// `.map_err(AnnErr)?` in any method returning [`PyResult`], and a plain `?` in
/// any returning [`AnnResult`].
pub(crate) struct AnnErr(
    /// The error the library raised.
    pub AnnSearchErrors,
);

impl From<AnnSearchErrors> for AnnErr {
    fn from(e: AnnSearchErrors) -> Self {
        Self(e)
    }
}

impl From<AnnErr> for PyErr {
    fn from(e: AnnErr) -> PyErr {
        let msg = e.0.to_string();
        match e.0 {
            // The caller handed us bad numbers.
            AnnSearchErrors::DimensionMismatch { .. }
            | AnnSearchErrors::DistanceNotSupported(_)
            | AnnSearchErrors::TooFewSamplesForCentroids { .. }
            | AnnSearchErrors::InvalidLshBits { .. }
            | AnnSearchErrors::LshTooManySamples { .. } => PyValueError::new_err(msg),

            // Filesystem.
            AnnSearchErrors::IoError(ref io) if io.kind() == std::io::ErrorKind::NotFound => {
                PyFileNotFoundError::new_err(msg)
            }
            AnnSearchErrors::IoError(_) => PyOSError::new_err(msg),

            // A bundle that is not what we asked for.
            AnnSearchErrors::NotAnIndexFile { .. }
            | AnnSearchErrors::TruncatedIndexFile { .. }
            | AnnSearchErrors::TrailingBytes { .. }
            | AnnSearchErrors::UnsupportedFormatVersion { .. }
            | AnnSearchErrors::IndexKindMismatch { .. }
            | AnnSearchErrors::FloatWidthMismatch { .. }
            | AnnSearchErrors::EncodeError(_)
            | AnnSearchErrors::DecodeError(_) => IndexIoError::new_err(msg),

            // `AnnSearchErrors` is `#[non_exhaustive]`, and most of its
            // remaining variants sit behind features this crate does not
            // enable. The catch-all is required, not laziness.
            _ => AnnSearchError::new_err(msg),
        }
    }
}

/// Result alias for methods whose only failure mode is an [`AnnSearchErrors`].
pub(crate) type AnnResult<T> = Result<T, AnnErr>;
