//! Pickle payloads.
//!
//! A payload is one tag byte holding `size_of::<Elem>()` followed by a verbatim
//! `index.bin`. Going through a tempdir rather than calling bincode directly
//! reuses the crate's magic, format version, kind tag and float-width check
//! with no duplication, and makes a pickle byte-identical to the `index.bin` a
//! `save()` writes. `src/serialise/mod.rs` says outright that bincode is not
//! self-describing and the version bump is manual; a second framing here would
//! be a second thing to bump in lockstep, and it would not be.
//!
//! The cost is two extra full-size disk I/Os per pickle. If that ever matters
//! (multi-GB indices, or a read-only `$TMPDIR`), the escape hatch is
//! `bincode::serde::encode_to_vec` against the index struct plus a hand-rolled
//! header.

use std::io::Read;

use ann_search_rs::prelude::{AnnSearchErrors, IndexIo};
use ann_search_rs::{load_index, save_index};

//////////////
// Tag byte //
//////////////

/// Leading byte of an `f32` payload: `size_of::<f32>()`.
///
/// An explicit tag rather than a read of the bundle header's float-width field,
/// which would couple this crate to a layout `src/serialise/mod.rs` is free to
/// change.
pub(crate) const F32_TAG: u8 = 4;

/// Leading byte of an `f64` payload: `size_of::<f64>()`.
pub(crate) const F64_TAG: u8 = 8;

/// Name the crate gives the payload inside a saved bundle directory.
const INDEX_FILE: &str = "index.bin";

///////////////////
// Serialisation //
///////////////////

/// Serialise an index to a self-describing byte payload.
///
/// ### Params
///
/// * `idx` - The built index.
/// * `tag` - [`F32_TAG`] or [`F64_TAG`], matching `I::Elem`.
///
/// ### Returns
///
/// The tag byte followed by the bundle's `index.bin`, or the first IO or
/// encoding error.
pub(crate) fn to_state<I: IndexIo>(idx: &I, tag: u8) -> Result<Vec<u8>, AnnSearchErrors> {
    let dir = tempfile::tempdir()?;
    save_index(idx, dir.path())?;

    let mut file = std::fs::File::open(dir.path().join(INDEX_FILE))?;
    let capacity = file.metadata().map(|m| m.len() as usize + 1).unwrap_or(0);
    let mut buf = Vec::with_capacity(capacity);
    buf.push(tag);
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Rebuild an index from a payload written by [`to_state`].
///
/// The caller has already dispatched on the tag byte, so `I` is fixed here. A
/// payload written by a different algorithm is rejected by the bundle's kind
/// check before anything is decoded.
///
/// ### Params
///
/// * `state` - Tag byte followed by an `index.bin`.
///
/// ### Returns
///
/// The reconstructed index, or the first header mismatch, IO or decoding error.
pub(crate) fn from_state<I: IndexIo>(state: &[u8]) -> Result<I, AnnSearchErrors> {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join(INDEX_FILE), &state[1..])?;
    load_index(dir.path())
}
