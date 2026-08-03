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
/// filename is re-joined afterwards.
///
/// ### Params
///
/// * `path` - Path to resolve; its parent directory must exist
///
/// ### Returns
///
/// The resolved path, comparable against another resolved path.
fn resolve(path: &Path) -> Result<PathBuf, AnnSearchErrors> {
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path.file_name().unwrap_or_default();

    Ok(std::fs::canonicalize(dir)?.join(name))
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
    /// Files must contain raw binary data in native endianness.
    ///
    /// ### Params
    ///
    /// * `vectors_path` - Path to vectors file (n * dim * sizeof(T) bytes)
    /// * `norms_path` - Path to norms file (n * sizeof(T) bytes)
    /// * `dim` - Vector dimensionality
    /// * `n` - Number of vectors
    pub fn new(
        vectors_path: impl AsRef<Path>,
        norms_path: impl AsRef<Path>,
        dim: usize,
        n: usize,
    ) -> Result<Self, AnnSearchErrors> {
        let vectors_path = vectors_path.as_ref().to_path_buf();
        let norms_path = norms_path.as_ref().to_path_buf();

        let file_vectors = File::open(&vectors_path)?;
        let file_norms = File::open(&norms_path)?;

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

    /// Copy the store into `dir`, leaving the original in place
    ///
    /// Used when an index is saved somewhere other than the directory its store
    /// already lives in. Copying into the store's own directory is a no-op.
    ///
    /// ### Note
    ///
    /// The copy goes to a temporary file and is then renamed over the
    /// destination. Writing into the destination directly would truncate it,
    /// and the destination may be mapped by another live store, where
    /// truncation is a `SIGBUS` rather than an `Err`. Renaming leaves the old
    /// inode alive for anyone still holding it.
    ///
    /// ### Params
    ///
    /// * `dir` - Directory to copy into. Must already exist.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the underlying IO error.
    pub fn copy_to_dir(&self, dir: impl AsRef<Path>) -> Result<(), AnnSearchErrors> {
        let (dst_vectors, dst_norms) = Self::paths_in(dir);

        for (src, dst) in [
            (&self.vectors_path, dst_vectors),
            (&self.norms_path, dst_norms),
        ] {
            if resolve(src)? == resolve(&dst)? {
                continue;
            }

            let tmp = dst.with_extension("bin.tmp");
            std::fs::copy(src, &tmp)?;
            std::fs::rename(&tmp, &dst)?;
        }

        Ok(())
    }

    /// Save vectors and norms to binary files
    ///
    /// Writes raw binary data in native endianness.
    ///
    /// ### Params
    ///
    /// * `vectors_flat` - Flat representation of the original vectors
    /// * `norms` - Norms of the vectors
    /// * `dim` - Dimensionality of the original data
    /// * `n` - Number of original vectors in the data
    /// * `vectors_path` - File path to the flat vector representation
    /// * `norms_path` - File path to the norm of the vector
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

        // Write vectors
        let mut writer = BufWriter::new(File::create(vectors_path)?);
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
        let mut writer = BufWriter::new(File::create(norms_path)?);
        let norms_bytes = unsafe {
            std::slice::from_raw_parts(norms.as_ptr() as *const u8, std::mem::size_of_val(norms))
        };
        writer.write_all(norms_bytes)?;
        writer.flush()?;

        Ok(())
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
    use tempfile::NamedTempFile;

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
