//! Contains the (flat) vector store that has the original indices on-disk.

use bytemuck::Pod;
use memmap2::Mmap;
use num_traits::Float;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::iter::Sum;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use crate::prelude::*;
use crate::utils::staging::StagedFiles;

/// Filename the flat vectors are always written under, inside whichever
/// directory the caller nominates.
pub const VECTORS_FILE: &str = "vectors_flat.bin";

/// Filename the norms are always written under.
pub const NORMS_FILE: &str = "norms.bin";

/// Trait for vector storage backends
pub trait VectorStore<T>
where
    T: Float + Sum,
{
    /// Load in a given vector based on idx position
    ///
    /// ### Params
    ///
    /// * `idx` - Index of the vector to load
    fn load_vector(&self, idx: usize) -> &[T];

    /// Returns the dimensionality
    ///
    /// ### Returns
    ///
    /// Dimensions
    fn dim(&self) -> usize;

    /// Returns the number of vectors
    ///
    /// ### Returns
    ///
    /// N vectors
    fn n(&self) -> usize;
}

/////////////////
// VectorStore //
/////////////////

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
/// A bare "No such file or directory" leaves the user guessing whether the
/// store or `index.bin` went missing.
///
/// ### Params
///
/// * `path` - File that could not be read
/// * `source` - The underlying IO error
///
/// ### Returns
///
/// The error carrying the path.
fn store_file_error(path: &Path, source: std::io::Error) -> AnnSearchErrors {
    AnnSearchErrors::StoreFileUnavailable {
        path: path.display().to_string(),
        source,
    }
}

/// Shape of a persisted vector store.
///
/// The store itself is two live memory maps and cannot survive a round trip
/// through `serde`, so an index records this instead and re-opens the maps on
/// load. `Option<StoreMeta>` doubles as the "was there a store at all?" flag,
/// which is otherwise lost.
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct StoreMeta {
    /// Vector dimensionality
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
}

impl StoreMeta {
    /// Cross-check the recorded shape against the index that owns the store
    ///
    /// The store files carry no header of their own, so a same-shape store from
    /// a different dataset still loads. This at least catches the store that
    /// belongs to a differently sized index, which the byte-length check inside
    /// [`MmapVectorStore::new`] cannot see: that one compares the file against
    /// this metadata, not against the index.
    ///
    /// ### Params
    ///
    /// * `n` - Number of samples the owning index holds
    /// * `dim` - Dimensionality the owning index holds
    ///
    /// ### Returns
    ///
    /// `Ok(())` when the shapes agree.
    pub fn check(&self, n: usize, dim: usize) -> Result<(), AnnSearchErrors> {
        if self.n != n || self.dim != dim {
            return Err(AnnSearchErrors::StoreShapeMismatch {
                index_n: n,
                index_dim: dim,
                store_n: self.n,
                store_dim: self.dim,
            });
        }

        Ok(())
    }
}

/// Memory-mapped vector storage
///
/// Stores vectors and norms in binary files and memory-maps them.
pub struct MmapVectorStore<T> {
    /// Mapped flat vectors, `n * dim * size_of::<T>()` bytes
    mmap_vectors: Mmap,
    /// Mapped norms, `n * size_of::<T>()` bytes
    mmap_norms: Mmap,
    /// Where `mmap_vectors` came from, needed to copy the store elsewhere
    vectors_path: PathBuf,
    /// Where `mmap_norms` came from
    norms_path: PathBuf,
    /// Vector dimensionality
    dim: usize,
    /// Number of vectors
    n: usize,
    /// Marker for the float type the raw bytes are interpreted as
    _phantom: PhantomData<T>,
}

impl<T> MmapVectorStore<T>
where
    T: Float + Sum,
{
    /// Create from existing binary files
    ///
    /// Files must contain raw binary data in native endianness, which is what
    /// makes a store non-portable across endianness even though `index.bin`
    /// itself is.
    ///
    /// Both paths are canonicalised up front. They are kept for the lifetime of
    /// the store and used later to copy it elsewhere, so a relative path plus a
    /// `chdir` in between would otherwise resolve to the wrong directory.
    ///
    /// ### Params
    ///
    /// * `vectors_path` - Path to vectors file (n * dim * sizeof(T) bytes)
    /// * `norms_path` - Path to norms file (n * sizeof(T) bytes)
    /// * `dim` - Vector dimensionality
    /// * `n` - Number of vectors
    ///
    /// ### Returns
    ///
    /// The opened store, or the file that could not be read.
    pub fn new(
        vectors_path: impl AsRef<Path>,
        norms_path: impl AsRef<Path>,
        dim: usize,
        n: usize,
    ) -> Result<Self, AnnSearchErrors> {
        let vectors_path = std::fs::canonicalize(vectors_path.as_ref())
            .map_err(|e| store_file_error(vectors_path.as_ref(), e))?;
        let norms_path = std::fs::canonicalize(norms_path.as_ref())
            .map_err(|e| store_file_error(norms_path.as_ref(), e))?;

        let file_vectors =
            File::open(&vectors_path).map_err(|e| store_file_error(&vectors_path, e))?;
        let file_norms = File::open(&norms_path).map_err(|e| store_file_error(&norms_path, e))?;

        let mmap_vectors = unsafe { Mmap::map(&file_vectors)? };

        #[cfg(unix)]
        mmap_vectors.advise(memmap2::Advice::Random)?;

        let mmap_norms = unsafe { Mmap::map(&file_norms)? };

        let expected_vectors_size = n * dim * std::mem::size_of::<T>();
        let expected_norms_size = n * std::mem::size_of::<T>();

        if mmap_vectors.len() != expected_vectors_size {
            return Err(AnnSearchErrors::SizeMismatch {
                expected: expected_vectors_size,
                actual: mmap_vectors.len(),
            });
        }

        if mmap_norms.len() != expected_norms_size {
            return Err(AnnSearchErrors::SizeMismatch {
                expected: expected_norms_size,
                actual: mmap_norms.len(),
            });
        }

        Ok(Self {
            mmap_vectors,
            mmap_norms,
            vectors_path,
            norms_path,
            dim,
            n,
            _phantom: PhantomData,
        })
    }

    /// The two filenames a store occupies inside `dir`
    ///
    /// ### Params
    ///
    /// * `dir` - Directory holding, or about to hold, a store
    ///
    /// ### Returns
    ///
    /// Tuple of `(vectors_path, norms_path)`
    pub fn paths_in(dir: impl AsRef<Path>) -> (PathBuf, PathBuf) {
        let dir = dir.as_ref();
        (dir.join(VECTORS_FILE), dir.join(NORMS_FILE))
    }

    /// Re-open a store that lives in `dir`
    ///
    /// ### Params
    ///
    /// * `dir` - Directory holding the two store files
    /// * `meta` - Shape recorded when the owning index was saved; `None` means
    ///   the index never had a store, and no files are touched
    ///
    /// ### Returns
    ///
    /// The re-opened store, or `None` when `meta` is `None`.
    pub fn open_in_dir(
        dir: impl AsRef<Path>,
        meta: Option<StoreMeta>,
    ) -> Result<Option<Self>, AnnSearchErrors> {
        let Some(meta) = meta else {
            return Ok(None);
        };

        let (vectors_path, norms_path) = Self::paths_in(dir);

        Ok(Some(Self::new(vectors_path, norms_path, meta.dim, meta.n)?))
    }

    /// Stage a copy of the store into `dir`, leaving the original in place
    ///
    /// Used when an index is saved somewhere other than the directory its store
    /// already lives in. The two files land under temporary names and only
    /// become visible when the caller commits `staged`, alongside whatever else
    /// belongs to the same save.
    ///
    /// ### Note
    ///
    /// Nothing is staged when the store already lives in `dir`. That is an
    /// optimisation, not a safety property: a copy onto itself would be
    /// harmless, since the write goes to a temporary file and the rename leaves
    /// the mapped inode alive. It just saves duplicating the whole dataset.
    ///
    /// The files are copied byte for byte. Some indices permute their vectors
    /// internally, but the store is deliberately kept in the original data
    /// order and every consumer maps back through `original_ids` before
    /// indexing it. Rewriting it into an index's internal order breaks
    /// re-ranking silently.
    ///
    /// ### Params
    ///
    /// * `dir` - Directory to copy into. Must already exist.
    /// * `staged` - Staging set the two copies are added to
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the underlying IO error.
    pub fn stage_copy_into(
        &self,
        dir: impl AsRef<Path>,
        staged: &mut StagedFiles,
    ) -> Result<(), AnnSearchErrors> {
        let (dst_vectors, dst_norms) = Self::paths_in(dir);

        for (src, dst) in [
            (&self.vectors_path, dst_vectors),
            (&self.norms_path, dst_norms),
        ] {
            if src == &resolve(&dst)? {
                continue;
            }

            let tmp = staged.stage(&dst);
            std::fs::copy(src, &tmp)?;
        }

        Ok(())
    }

    /// Save vectors and norms to binary files
    ///
    /// Writes raw binary data in native endianness.
    ///
    /// ### Note
    ///
    /// Both files are written under temporary names and renamed into place at
    /// the end. Writing the destinations directly would truncate them, and
    /// building an index into a directory another live index has mapped is a
    /// normal thing to do, where truncation is a `SIGBUS` rather than an `Err`.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Flat representation of the original vectors
    /// * `norms` - Norms of the vectors
    /// * `dim` - Dimensionality of the original data
    /// * `n` - Number of original vectors in the data
    /// * `vectors_path` - File path to the flat vector representation
    /// * `norms_path` - File path to the norm of the vector
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or a shape mismatch / IO error.
    pub fn save(
        vectors_flat: &[T],
        norms: &[T],
        dim: usize,
        n: usize,
        vectors_path: impl AsRef<Path>,
        norms_path: impl AsRef<Path>,
    ) -> Result<(), AnnSearchErrors> {
        if vectors_flat.len() != n * dim {
            return Err(AnnSearchErrors::SizeMismatch {
                expected: n * dim,
                actual: vectors_flat.len(),
            });
        }

        if norms.len() != n {
            return Err(AnnSearchErrors::SizeMismatch {
                expected: n,
                actual: norms.len(),
            });
        }

        let mut staged = StagedFiles::new();

        // Write vectors
        let tmp = staged.stage(vectors_path.as_ref());
        let mut writer = BufWriter::new(File::create(&tmp)?);
        let vectors_bytes = unsafe {
            std::slice::from_raw_parts(
                vectors_flat.as_ptr() as *const u8,
                std::mem::size_of_val(vectors_flat),
            )
        };
        writer.write_all(vectors_bytes)?;
        // BufWriter flushes on Drop but swallows the error; surface it instead
        writer.flush()?;

        // Write norms
        let tmp = staged.stage(norms_path.as_ref());
        let mut writer = BufWriter::new(File::create(&tmp)?);
        let norms_bytes = unsafe {
            std::slice::from_raw_parts(norms.as_ptr() as *const u8, std::mem::size_of_val(norms))
        };
        writer.write_all(norms_bytes)?;
        writer.flush()?;

        staged.commit()
    }

    /// Helper function to return dimensions
    ///
    /// ### Returns
    ///
    /// The dimensionality
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/////////////////
// VectorStore //
/////////////////

impl<T> VectorStore<T> for MmapVectorStore<T>
where
    T: Float + Sum + Pod,
{
    fn load_vector(&self, idx: usize) -> &[T] {
        let start = idx * self.dim;
        let end = start + self.dim;
        let all_data: &[T] = bytemuck::cast_slice(&self.mmap_vectors);
        &all_data[start..end]
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn n(&self) -> usize {
        self.n
    }
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for MmapVectorStore<T>
where
    T: AnnSearchFloat,
{
    fn vectors_flat(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.mmap_vectors.as_ptr() as *const T, self.n * self.dim)
        }
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn norms(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.mmap_norms.as_ptr() as *const T, self.n) }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_save_and_load() {
        let vectors = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let norms = vec![3.74, 8.77, 13.93];
        let dim = 2;
        let n = 3;

        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        MmapVectorStore::save(&vectors, &norms, dim, n, vec_file.path(), norm_file.path()).unwrap();

        let store = MmapVectorStore::<f32>::new(vec_file.path(), norm_file.path(), dim, n).unwrap();

        assert_eq!(store.dim(), 2);
        assert_eq!(store.n(), 3);
    }

    #[test]
    fn test_load_vector() {
        let vectors = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let norms = vec![3.74, 8.77, 13.93];
        let dim = 2;
        let n = 3;

        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        MmapVectorStore::save(&vectors, &norms, dim, n, vec_file.path(), norm_file.path()).unwrap();

        let store = MmapVectorStore::<f32>::new(vec_file.path(), norm_file.path(), dim, n).unwrap();

        let v0 = store.load_vector(0);
        assert_eq!(v0, &[1.0, 2.0]);

        let v1 = store.load_vector(1);
        assert_eq!(v1, &[3.0, 4.0]);

        let v2 = store.load_vector(2);
        assert_eq!(v2, &[5.0, 6.0]);
    }

    #[test]
    fn test_vectors_flat() {
        let vectors = vec![1.0f32, 2.0, 3.0, 4.0];
        let norms = vec![2.24, 5.0];
        let dim = 2;
        let n = 2;

        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        MmapVectorStore::save(&vectors, &norms, dim, n, vec_file.path(), norm_file.path()).unwrap();

        let store = MmapVectorStore::<f32>::new(vec_file.path(), norm_file.path(), dim, n).unwrap();

        let flat = store.vectors_flat();
        assert_eq!(flat, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_norms() {
        let vectors = vec![1.0f32, 2.0, 3.0, 4.0];
        let norms = vec![2.24, 5.0];
        let dim = 2;
        let n = 2;

        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        MmapVectorStore::save(&vectors, &norms, dim, n, vec_file.path(), norm_file.path()).unwrap();

        let store = MmapVectorStore::<f32>::new(vec_file.path(), norm_file.path(), dim, n).unwrap();

        let loaded_norms = store.norms();
        assert_eq!(loaded_norms, &[2.24, 5.0]);
    }

    #[test]
    fn test_wrong_vector_file_size() {
        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        std::fs::write(vec_file.path(), [0u8; 100]).unwrap();
        std::fs::write(norm_file.path(), [0u8; 16]).unwrap();

        let result = MmapVectorStore::<f32>::new(vec_file.path(), norm_file.path(), 2, 4);
        assert!(matches!(result, Err(AnnSearchErrors::SizeMismatch { .. })));
    }

    #[test]
    fn test_wrong_norms_file_size() {
        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        std::fs::write(vec_file.path(), [0u8; 32]).unwrap();
        std::fs::write(norm_file.path(), [0u8; 100]).unwrap();

        let result = MmapVectorStore::<f32>::new(vec_file.path(), norm_file.path(), 2, 4);
        assert!(matches!(result, Err(AnnSearchErrors::SizeMismatch { .. })));
    }

    #[test]
    fn test_save_vectors_length_mismatch() {
        let vectors = vec![1.0f32, 2.0, 3.0];
        let norms = vec![2.24, 5.0];

        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        let result =
            MmapVectorStore::save(&vectors, &norms, 2, 2, vec_file.path(), norm_file.path());
        assert!(matches!(result, Err(AnnSearchErrors::SizeMismatch { .. })));
    }

    #[test]
    fn test_save_norms_length_mismatch() {
        let vectors = vec![1.0f32, 2.0, 3.0, 4.0];
        let norms = vec![2.24];

        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        let result =
            MmapVectorStore::save(&vectors, &norms, 2, 2, vec_file.path(), norm_file.path());
        assert!(matches!(result, Err(AnnSearchErrors::SizeMismatch { .. })));
    }

    /// `save` used to `File::create` both files, truncating them. A live
    /// mapping of the same paths saw the new bytes, or a `SIGBUS` if it read
    /// during the truncation window. Renaming leaves the old inode alive.
    #[test]
    fn test_save_does_not_disturb_a_live_mapping() {
        let dir = TempDir::new().unwrap();
        let (vectors_path, norms_path) = MmapVectorStore::<f32>::paths_in(dir.path());

        MmapVectorStore::save(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[2.24f32, 5.0],
            2,
            2,
            &vectors_path,
            &norms_path,
        )
        .unwrap();
        let live = MmapVectorStore::<f32>::new(&vectors_path, &norms_path, 2, 2).unwrap();

        MmapVectorStore::save(
            &[9.0f32, 9.0, 9.0, 9.0],
            &[9.0f32, 9.0],
            2,
            2,
            &vectors_path,
            &norms_path,
        )
        .unwrap();

        assert_eq!(live.load_vector(0), &[1.0, 2.0]);
        assert_eq!(live.norms(), &[2.24, 5.0]);

        let fresh = MmapVectorStore::<f32>::new(&vectors_path, &norms_path, 2, 2).unwrap();
        assert_eq!(fresh.load_vector(0), &[9.0, 9.0]);
    }

    /// A rejected `save` must not leave a half-written temporary behind.
    #[test]
    fn test_failed_save_leaves_no_temporaries() {
        let dir = TempDir::new().unwrap();
        let (vectors_path, norms_path) = MmapVectorStore::<f32>::paths_in(dir.path());

        let err = MmapVectorStore::save(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[2.24f32],
            2,
            2,
            &vectors_path,
            &norms_path,
        );
        assert!(matches!(err, Err(AnnSearchErrors::SizeMismatch { .. })));

        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect();
        assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
    }

    /// Store paths are canonicalised on open, so an unnormalised one still
    /// resolves later. Kept verbatim, `..` segments and a bare filename both
    /// broke the copy: `Path::parent` hands back `Some("")` rather than `None`,
    /// so `canonicalize` was handed an empty path and returned ENOENT.
    #[test]
    fn test_unnormalised_store_paths_can_be_copied() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("nested");
        std::fs::create_dir(&nested).unwrap();

        let (vectors_path, norms_path) = MmapVectorStore::<f32>::paths_in(dir.path());
        MmapVectorStore::save(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[2.24f32, 5.0],
            2,
            2,
            &vectors_path,
            &norms_path,
        )
        .unwrap();

        let store = MmapVectorStore::<f32>::new(
            nested.join("..").join(VECTORS_FILE),
            nested.join("..").join(NORMS_FILE),
            2,
            2,
        )
        .unwrap();

        let target = TempDir::new().unwrap();
        let mut staged = StagedFiles::new();
        store.stage_copy_into(target.path(), &mut staged).unwrap();
        staged.commit().unwrap();

        let copied =
            MmapVectorStore::<f32>::open_in_dir(target.path(), Some(StoreMeta { dim: 2, n: 2 }))
                .unwrap()
                .unwrap();

        assert_eq!(copied.load_vector(1), &[3.0, 4.0]);
    }

    /// Copying into the directory the store already lives in stages nothing.
    /// That is the optimisation, not the safety property: the tmp + rename
    /// would have been harmless either way.
    #[test]
    fn test_copy_into_own_directory_stages_nothing() {
        let dir = TempDir::new().unwrap();
        let (vectors_path, norms_path) = MmapVectorStore::<f32>::paths_in(dir.path());

        MmapVectorStore::save(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[2.24f32, 5.0],
            2,
            2,
            &vectors_path,
            &norms_path,
        )
        .unwrap();

        let store = MmapVectorStore::<f32>::new(&vectors_path, &norms_path, 2, 2).unwrap();

        let mut staged = StagedFiles::new();
        store.stage_copy_into(dir.path(), &mut staged).unwrap();

        assert!(staged.is_empty());
    }

    /// A missing store file names itself rather than reporting a bare ENOENT.
    #[test]
    fn test_missing_store_file_is_named() {
        let dir = TempDir::new().unwrap();
        let (vectors_path, norms_path) = MmapVectorStore::<f32>::paths_in(dir.path());

        match MmapVectorStore::<f32>::new(&vectors_path, &norms_path, 2, 2) {
            Err(AnnSearchErrors::StoreFileUnavailable { path, .. }) => {
                assert!(path.ends_with(VECTORS_FILE), "wrong file named: {path}")
            }
            Err(other) => panic!("unexpected error: {other}"),
            Ok(_) => panic!("opening a missing store should have failed"),
        }
    }

    #[test]
    fn test_f64_type() {
        let vectors = vec![1.0f64, 2.0, 3.0, 4.0];
        let norms = vec![2.24, 5.0];
        let dim = 2;
        let n = 2;

        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        MmapVectorStore::save(&vectors, &norms, dim, n, vec_file.path(), norm_file.path()).unwrap();

        let store = MmapVectorStore::<f64>::new(vec_file.path(), norm_file.path(), dim, n).unwrap();

        assert_eq!(store.load_vector(0), &[1.0, 2.0]);
        assert_eq!(store.dim(), 2);
    }
}
