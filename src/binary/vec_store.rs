//! On-disk store of the original vectors, memory-mapped for exact re-ranking.
//!
//! One self-describing file per store. Layout (all little endian):
//!
//! ```text
//! [0..8)    magic        b"ANNVEC\0\0"
//! [8..12)   version      u32
//! [12]      encoding     u8    0 f32, 1 f64
//! [13..16)  reserved
//! [16..24)  n            u64
//! [24..32)  dim          u64
//! [32..40)  fingerprint  u64   FNV-1a over the header fields and the norms
//! [40..64)  reserved
//! [64..)    norms        n elements, padded to a multiple of 64 bytes
//! [..)      rows         n * dim elements, row-major, original data order
//! ```
//!
//! Both sections start on a 64-byte boundary, so they cast straight to `&[T]`
//! from the page-aligned mapping. The index records the fingerprint and checks
//! it on load, which catches a store swapped in from a different dataset.

use bytemuck::Pod;
use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use crate::prelude::*;
use crate::utils::staging::StagedFiles;

////////////
// Consts //
////////////

/// Filename the store is always written under, inside whichever directory
/// the caller nominates.
pub const VECTORS_FILE: &str = "vectors.bin";

/// File magic.
const MAGIC: &[u8; 8] = b"ANNVEC\0\0";

/// Store format version. Bump on any layout change.
const STORE_VERSION: u32 = 1;

/// Length of the fixed header, see the module docs.
const HEADER_LEN: usize = 64;

/// Alignment of the norms and rows sections. A cache line, which also covers
/// the alignment of every element type.
const SECTION_ALIGN: usize = 64;

/// Encoding byte for `f32` rows.
const ENC_F32: u8 = 0;

/// Encoding byte for `f64` rows.
const ENC_F64: u8 = 1;

/////////////
// Helpers //
/////////////

/// Encoding byte for the element type `T`.
///
/// ### Returns
///
/// The encoding byte, or `None` for an element width the format has no
/// encoding for.
fn encoding_of<T>() -> Option<u8> {
    match std::mem::size_of::<T>() {
        4 => Some(ENC_F32),
        8 => Some(ENC_F64),
        _ => None,
    }
}

/// Byte offset of the rows section.
///
/// ### Params
///
/// * `n` - Number of vectors
/// * `elem` - Bytes per element
///
/// ### Returns
///
/// Offset of the first row, a multiple of [`SECTION_ALIGN`].
fn rows_offset(n: usize, elem: usize) -> usize {
    HEADER_LEN + (n * elem).next_multiple_of(SECTION_ALIGN)
}

/// FNV-1a over the shape, encoding and norm bytes.
///
/// The norms are a cheap summary of the data: two different datasets of the
/// same shape practically never share them, and hashing them costs one pass
/// over `n` elements rather than `n * dim`.
///
/// ### Params
///
/// * `n` - Number of vectors
/// * `dim` - Dimensionality
/// * `encoding` - Encoding byte
/// * `norm_bytes` - Raw bytes of the norms section, without padding
///
/// ### Returns
///
/// The 64-bit fingerprint.
fn fingerprint(n: usize, dim: usize, encoding: u8, norm_bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut h = OFFSET;
    let shape = (n as u64)
        .to_le_bytes()
        .into_iter()
        .chain((dim as u64).to_le_bytes())
        .chain([encoding]);
    for b in shape.chain(norm_bytes.iter().copied()) {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Resolve a path to a comparable absolute form.
///
/// `canonicalize` needs the whole path to exist, and the destination of a copy
/// generally does not, so only the parent directory is resolved and the
/// filename is re-joined afterwards. `Path::parent` hands back `Some("")` for a
/// bare filename rather than `None`, so an empty parent falls back to the
/// working directory too.
///
/// ### Params
///
/// * `path` - Path to resolve; its parent directory must exist
///
/// ### Returns
///
/// The resolved path, comparable against another resolved path.
fn resolve(path: &Path) -> Result<PathBuf, AnnSearchErrors> {
    let dir = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let name = path.file_name().unwrap_or_default();

    Ok(std::fs::canonicalize(dir)?.join(name))
}

/// Name the store file in an IO error.
///
/// ### Params
///
/// * `path` - File that could not be read
/// * `source` - The underlying IO error
///
/// ### Returns
///
/// The error carrying the path.
fn unavailable(path: &Path, source: std::io::Error) -> AnnSearchErrors {
    AnnSearchErrors::StoreFileUnavailable {
        path: path.display().to_string(),
        source,
    }
}

/// A store file that exists but cannot be used.
///
/// ### Params
///
/// * `path` - The store file
/// * `reason` - What is wrong with it
///
/// ### Returns
///
/// The error carrying path and reason.
fn invalid(path: &Path, reason: impl Into<String>) -> AnnSearchErrors {
    AnnSearchErrors::StoreInvalid {
        path: path.display().to_string(),
        reason: reason.into(),
    }
}

/// Read a little-endian `u64` from the header.
///
/// ### Params
///
/// * `header` - The header bytes, at least `at + 8` long
/// * `at` - Byte offset
///
/// ### Returns
///
/// The value.
fn read_u64(header: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(header[at..at + 8].try_into().expect("8-byte slice"))
}

/////////////////
// VectorStore //
/////////////////

/// Memory-mapped store of the original vectors and their L2 norms.
///
/// Rows stay in the original data order. Indices that permute their vectors
/// internally map back through their `original_ids` before indexing it.
pub struct VectorStore<T> {
    /// The whole store file, mapped
    mmap: Mmap,
    /// Canonical path of the file, needed to copy the store elsewhere
    path: PathBuf,
    /// Number of vectors
    n: usize,
    /// Vector dimensionality
    dim: usize,
    /// Fingerprint from the header, see [`fingerprint`]
    fingerprint: u64,
    /// Byte offset of the rows section
    rows_offset: usize,
    /// Marker for the element type the sections are cast to
    _phantom: PhantomData<T>,
}

impl<T> VectorStore<T>
where
    T: AnnSearchFloat + Pod,
{
    /// Write a store into `dir` and open it.
    ///
    /// The file is written under a temporary name and renamed into place.
    /// Writing the destination directly would truncate it, and building an
    /// index into a directory another live index has mapped is a normal thing
    /// to do, where truncation is a `SIGBUS` rather than an `Err`.
    ///
    /// ### Params
    ///
    /// * `dir` - Directory to write into, created if missing
    /// * `vectors_flat` - Original vectors, row-major, `n * dim`
    /// * `norms` - L2 norm of each vector, length `n`
    /// * `n` - Number of vectors
    /// * `dim` - Dimensionality
    ///
    /// ### Returns
    ///
    /// The opened store, or a shape / IO error.
    pub fn create(
        dir: impl AsRef<Path>,
        vectors_flat: &[T],
        norms: &[T],
        n: usize,
        dim: usize,
    ) -> Result<Self, AnnSearchErrors> {
        if vectors_flat.len() != n * dim {
            return Err(AnnSearchErrors::BufferLengthMismatch {
                expected: n * dim,
                actual: vectors_flat.len(),
            });
        }
        if norms.len() != n {
            return Err(AnnSearchErrors::BufferLengthMismatch {
                expected: n,
                actual: norms.len(),
            });
        }

        let dir = dir.as_ref();
        let path = dir.join(VECTORS_FILE);
        if cfg!(target_endian = "big") {
            return Err(invalid(&path, "big-endian hosts are not supported"));
        }
        let encoding = encoding_of::<T>()
            .ok_or_else(|| invalid(&path, "element type has no store encoding"))?;

        std::fs::create_dir_all(dir)?;

        let norm_bytes: &[u8] = bytemuck::cast_slice(norms);
        let mut header = [0u8; HEADER_LEN];
        header[0..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&STORE_VERSION.to_le_bytes());
        header[12] = encoding;
        header[16..24].copy_from_slice(&(n as u64).to_le_bytes());
        header[24..32].copy_from_slice(&(dim as u64).to_le_bytes());
        header[32..40].copy_from_slice(&fingerprint(n, dim, encoding, norm_bytes).to_le_bytes());

        let pad = rows_offset(n, std::mem::size_of::<T>()) - HEADER_LEN - norm_bytes.len();

        let mut staged = StagedFiles::new();
        let tmp = staged.stage(&path);
        let mut writer = BufWriter::new(File::create(&tmp)?);
        writer.write_all(&header)?;
        writer.write_all(norm_bytes)?;
        writer.write_all(&[0u8; SECTION_ALIGN][..pad])?;
        writer.write_all(bytemuck::cast_slice(vectors_flat))?;
        // BufWriter flushes on Drop but swallows the error; surface it instead
        writer.flush()?;
        drop(writer);
        staged.commit()?;

        Self::open(dir)
    }

    /// Open the store that lives in `dir`.
    ///
    /// Validates magic, version, encoding, file length and the fingerprint
    /// against the norms section. Rows are not hashed, so a corrupted row is
    /// not caught here.
    ///
    /// ### Params
    ///
    /// * `dir` - Directory holding the store file
    ///
    /// ### Returns
    ///
    /// The opened store, or the reason it cannot be used.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, AnnSearchErrors> {
        let path = dir.as_ref().join(VECTORS_FILE);
        // Canonicalised up front: the path is kept for later copies, and a
        // relative one plus a `chdir` in between would resolve elsewhere
        let path = std::fs::canonicalize(&path).map_err(|e| unavailable(&path, e))?;
        let file = File::open(&path).map_err(|e| unavailable(&path, e))?;
        let mmap = unsafe { Mmap::map(&file)? };

        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Random)?;

        if cfg!(target_endian = "big") {
            return Err(invalid(&path, "big-endian hosts are not supported"));
        }
        if mmap.len() < HEADER_LEN || &mmap[0..8] != MAGIC {
            return Err(invalid(&path, "not a vector store file"));
        }
        let version = u32::from_le_bytes(mmap[8..12].try_into().expect("4-byte slice"));
        if version != STORE_VERSION {
            return Err(invalid(
                &path,
                format!("format version {version}, expected {STORE_VERSION}"),
            ));
        }
        let encoding = mmap[12];
        if Some(encoding) != encoding_of::<T>() {
            return Err(invalid(
                &path,
                format!("encoding {encoding} does not match the index float type"),
            ));
        }

        let n = read_u64(&mmap, 16) as usize;
        let dim = read_u64(&mmap, 24) as usize;
        let recorded = read_u64(&mmap, 32);

        let elem = std::mem::size_of::<T>();
        let rows_offset = rows_offset(n, elem);
        let expected = rows_offset + n * dim * elem;
        if mmap.len() != expected {
            return Err(invalid(
                &path,
                format!("{} bytes, expected {expected}", mmap.len()),
            ));
        }

        let norm_bytes = &mmap[HEADER_LEN..HEADER_LEN + n * elem];
        if fingerprint(n, dim, encoding, norm_bytes) != recorded {
            return Err(invalid(&path, "fingerprint does not match its contents"));
        }

        Ok(Self {
            mmap,
            path,
            n,
            dim,
            fingerprint: recorded,
            rows_offset,
            _phantom: PhantomData,
        })
    }

    /// Re-open the store an index was saved with and check it belongs to it.
    ///
    /// ### Params
    ///
    /// * `dir` - Directory holding the store file
    /// * `fingerprint` - Fingerprint the index recorded; `None` means the
    ///   index never had a store, and no file is touched
    /// * `n` - Number of samples the index holds
    /// * `dim` - Dimensionality the index holds
    ///
    /// ### Returns
    ///
    /// The store, `None` when `fingerprint` is `None`, or the mismatch.
    pub fn open_for(
        dir: impl AsRef<Path>,
        fingerprint: Option<u64>,
        n: usize,
        dim: usize,
    ) -> Result<Option<Self>, AnnSearchErrors> {
        let Some(fingerprint) = fingerprint else {
            return Ok(None);
        };

        let store = Self::open(dir)?;
        if store.n != n || store.dim != dim {
            return Err(AnnSearchErrors::StoreShapeMismatch {
                index_n: n,
                index_dim: dim,
                store_n: store.n,
                store_dim: store.dim,
            });
        }
        if store.fingerprint != fingerprint {
            return Err(invalid(&store.path, "store belongs to a different index"));
        }

        Ok(Some(store))
    }

    /// Stage a copy of the store into `dir`, leaving the original in place.
    ///
    /// The copy lands under a temporary name and only becomes visible when the
    /// caller commits `staged`, alongside the rest of the same save. Nothing
    /// is staged when the store already lives in `dir`; that saves duplicating
    /// the dataset, the tmp + rename would be harmless either way.
    ///
    /// ### Params
    ///
    /// * `dir` - Directory to copy into. Must already exist.
    /// * `staged` - Staging set the copy is added to
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the underlying IO error.
    pub fn stage_copy_into(
        &self,
        dir: impl AsRef<Path>,
        staged: &mut StagedFiles,
    ) -> Result<(), AnnSearchErrors> {
        let dst = dir.as_ref().join(VECTORS_FILE);
        if self.path == resolve(&dst)? {
            return Ok(());
        }

        let tmp = staged.stage(&dst);
        std::fs::copy(&self.path, &tmp)?;

        Ok(())
    }

    /// One original vector.
    ///
    /// ### Params
    ///
    /// * `idx` - Row in original data order
    ///
    /// ### Returns
    ///
    /// The vector, length `dim`.
    #[inline]
    pub fn row(&self, idx: usize) -> &[T] {
        &self.vectors_flat()[idx * self.dim..(idx + 1) * self.dim]
    }

    /// Fingerprint the owning index records to recognise this store on load.
    ///
    /// ### Returns
    ///
    /// The fingerprint.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for VectorStore<T>
where
    T: AnnSearchFloat + Pod,
{
    fn vectors_flat(&self) -> &[T] {
        let len = self.n * self.dim * std::mem::size_of::<T>();
        bytemuck::cast_slice(&self.mmap[self.rows_offset..self.rows_offset + len])
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn norms(&self) -> &[T] {
        let len = self.n * std::mem::size_of::<T>();
        bytemuck::cast_slice(&self.mmap[HEADER_LEN..HEADER_LEN + len])
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Three 2-d vectors and their norms.
    fn fixture() -> (Vec<f32>, Vec<f32>) {
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3.74, 8.77, 13.93])
    }

    #[test]
    fn test_round_trip_rows_and_norms() {
        let dir = TempDir::new().unwrap();
        let (vectors, norms) = fixture();

        let store = VectorStore::create(dir.path(), &vectors, &norms, 3, 2).unwrap();

        assert_eq!(store.row(0), &[1.0, 2.0]);
        assert_eq!(store.row(2), &[5.0, 6.0]);
        assert_eq!(store.vectors_flat(), vectors.as_slice());
        assert_eq!(store.norms(), norms.as_slice());

        let reopened = VectorStore::<f32>::open(dir.path()).unwrap();
        assert_eq!(reopened.fingerprint(), store.fingerprint());
        assert_eq!(reopened.row(1), &[3.0, 4.0]);
    }

    #[test]
    fn test_f64_round_trip() {
        let dir = TempDir::new().unwrap();
        let store =
            VectorStore::create(dir.path(), &[1.0f64, 2.0, 3.0, 4.0], &[2.24, 5.0], 2, 2).unwrap();

        assert_eq!(store.row(1), &[3.0, 4.0]);
        assert_eq!(store.norms(), &[2.24, 5.0]);
    }

    #[test]
    fn test_sections_are_aligned() {
        let dir = TempDir::new().unwrap();
        // Odd n so the norms section needs padding
        let store = VectorStore::create(dir.path(), &[1.0f32; 3 * 5], &[1.0f32; 3], 3, 5).unwrap();

        assert_eq!(store.rows_offset % SECTION_ALIGN, 0);
        assert_eq!(store.row(2), &[1.0; 5]);
    }

    #[test]
    fn test_empty_store() {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::<f32>::create(dir.path(), &[], &[], 0, 4).unwrap();

        assert!(store.vectors_flat().is_empty());
        assert!(store.norms().is_empty());
    }

    #[test]
    fn test_create_rejects_wrong_lengths() {
        let dir = TempDir::new().unwrap();

        let rows = VectorStore::create(dir.path(), &[1.0f32, 2.0, 3.0], &[1.0, 1.0], 2, 2);
        assert!(matches!(
            rows,
            Err(AnnSearchErrors::BufferLengthMismatch { .. })
        ));

        let norms = VectorStore::create(dir.path(), &[1.0f32, 2.0, 3.0, 4.0], &[1.0], 2, 2);
        assert!(matches!(
            norms,
            Err(AnnSearchErrors::BufferLengthMismatch { .. })
        ));

        // A rejected create leaves nothing behind
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn test_open_rejects_wrong_float_type() {
        let dir = TempDir::new().unwrap();
        let (vectors, norms) = fixture();
        VectorStore::create(dir.path(), &vectors, &norms, 3, 2).unwrap();

        assert!(matches!(
            VectorStore::<f64>::open(dir.path()),
            Err(AnnSearchErrors::StoreInvalid { .. })
        ));
    }

    #[test]
    fn test_open_rejects_truncated_file() {
        let dir = TempDir::new().unwrap();
        let (vectors, norms) = fixture();
        VectorStore::create(dir.path(), &vectors, &norms, 3, 2).unwrap();

        let path = dir.path().join(VECTORS_FILE);
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();

        assert!(matches!(
            VectorStore::<f32>::open(dir.path()),
            Err(AnnSearchErrors::StoreInvalid { .. })
        ));
    }

    #[test]
    fn test_open_rejects_foreign_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(VECTORS_FILE), [0u8; 128]).unwrap();

        assert!(matches!(
            VectorStore::<f32>::open(dir.path()),
            Err(AnnSearchErrors::StoreInvalid { .. })
        ));
    }

    #[test]
    fn test_open_rejects_tampered_norms() {
        let dir = TempDir::new().unwrap();
        let (vectors, norms) = fixture();
        VectorStore::create(dir.path(), &vectors, &norms, 3, 2).unwrap();

        let path = dir.path().join(VECTORS_FILE);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[HEADER_LEN] ^= 1;
        std::fs::write(&path, &bytes).unwrap();

        assert!(matches!(
            VectorStore::<f32>::open(dir.path()),
            Err(AnnSearchErrors::StoreInvalid { .. })
        ));
    }

    #[test]
    fn test_open_for_checks_shape_and_fingerprint() {
        let dir = TempDir::new().unwrap();
        let (vectors, norms) = fixture();
        let fp = VectorStore::create(dir.path(), &vectors, &norms, 3, 2)
            .unwrap()
            .fingerprint();

        assert!(VectorStore::<f32>::open_for(dir.path(), None, 3, 2)
            .unwrap()
            .is_none());
        assert!(VectorStore::<f32>::open_for(dir.path(), Some(fp), 3, 2)
            .unwrap()
            .is_some());
        assert!(matches!(
            VectorStore::<f32>::open_for(dir.path(), Some(fp), 4, 2),
            Err(AnnSearchErrors::StoreShapeMismatch { .. })
        ));
        assert!(matches!(
            VectorStore::<f32>::open_for(dir.path(), Some(fp ^ 1), 3, 2),
            Err(AnnSearchErrors::StoreInvalid { .. })
        ));
    }

    /// Re-creating a store must not disturb a live mapping of the old file:
    /// the rename leaves the old inode alive.
    #[test]
    fn test_create_does_not_disturb_a_live_mapping() {
        let dir = TempDir::new().unwrap();
        let live =
            VectorStore::create(dir.path(), &[1.0f32, 2.0, 3.0, 4.0], &[2.24, 5.0], 2, 2).unwrap();

        VectorStore::create(dir.path(), &[9.0f32; 4], &[9.0; 2], 2, 2).unwrap();

        assert_eq!(live.row(0), &[1.0, 2.0]);
        assert_eq!(live.norms(), &[2.24, 5.0]);

        let fresh = VectorStore::<f32>::open(dir.path()).unwrap();
        assert_eq!(fresh.row(0), &[9.0, 9.0]);
    }

    /// The store path is canonicalised on open, so an unnormalised directory
    /// still copies. `Path::parent` hands back `Some("")` rather than `None`
    /// for a bare filename, which once broke `canonicalize`.
    #[test]
    fn test_unnormalised_store_path_can_be_copied() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        VectorStore::create(dir.path(), &[1.0f32, 2.0, 3.0, 4.0], &[2.24, 5.0], 2, 2).unwrap();

        let store = VectorStore::<f32>::open(nested.join("..")).unwrap();

        let target = TempDir::new().unwrap();
        let mut staged = StagedFiles::new();
        store.stage_copy_into(target.path(), &mut staged).unwrap();
        staged.commit().unwrap();

        let copied = VectorStore::<f32>::open(target.path()).unwrap();
        assert_eq!(copied.row(1), &[3.0, 4.0]);
        assert_eq!(copied.fingerprint(), store.fingerprint());
    }

    #[test]
    fn test_copy_into_own_directory_stages_nothing() {
        let dir = TempDir::new().unwrap();
        let store =
            VectorStore::create(dir.path(), &[1.0f32, 2.0, 3.0, 4.0], &[2.24, 5.0], 2, 2).unwrap();

        let mut staged = StagedFiles::new();
        store.stage_copy_into(dir.path(), &mut staged).unwrap();

        assert!(staged.is_empty());
    }

    #[test]
    fn test_missing_store_file_is_named() {
        let dir = TempDir::new().unwrap();

        match VectorStore::<f32>::open(dir.path()) {
            Err(AnnSearchErrors::StoreFileUnavailable { path, .. }) => {
                assert!(path.ends_with(VECTORS_FILE), "wrong file named: {path}")
            }
            Err(other) => panic!("unexpected error: {other}"),
            Ok(_) => panic!("opening a missing store should have failed"),
        }
    }
}
