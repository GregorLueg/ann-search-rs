use bytemuck::Pod;
use memmap2::Mmap;
use num_traits::Float;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::iter::Sum;
use std::marker::PhantomData;
use std::path::Path;

use crate::utils::dist::*;

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

/// Memory-mapped vector storage
///
/// Stores vectors and norms in binary files and memory-maps them.
pub struct MmapVectorStore<T> {
    mmap_vectors: Mmap,
    mmap_norms: Mmap,
    dim: usize,
    n: usize,
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
    ) -> std::io::Result<Self> {
        let file_vectors = File::open(vectors_path)?;
        let file_norms = File::open(norms_path)?;

        let mmap_vectors = unsafe { Mmap::map(&file_vectors)? };

        #[cfg(unix)]
        mmap_vectors.advise(memmap2::Advice::Random)?;

        let mmap_norms = unsafe { Mmap::map(&file_norms)? };

        let expected_vectors_size = n * dim * std::mem::size_of::<T>();
        let expected_norms_size = n * std::mem::size_of::<T>();

        assert_eq!(
            mmap_vectors.len(),
            expected_vectors_size,
            "Vectors file size mismatch"
        );
        assert_eq!(
            mmap_norms.len(),
            expected_norms_size,
            "Norms file size mismatch"
        );

        Ok(Self {
            mmap_vectors,
            mmap_norms,
            dim,
            n,
            _phantom: PhantomData,
        })
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
    ) -> std::io::Result<()> {
        assert_eq!(vectors_flat.len(), n * dim);
        assert_eq!(norms.len(), n);

        // Write vectors
        let mut writer = BufWriter::new(File::create(vectors_path)?);
        let vectors_bytes = unsafe {
            std::slice::from_raw_parts(
                vectors_flat.as_ptr() as *const u8,
                std::mem::size_of_val(vectors_flat),
            )
        };
        writer.write_all(vectors_bytes)?;

        // Write norms
        let mut writer = BufWriter::new(File::create(norms_path)?);
        let norms_bytes = unsafe {
            std::slice::from_raw_parts(norms.as_ptr() as *const u8, std::mem::size_of_val(norms))
        };
        writer.write_all(norms_bytes)?;

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
    T: Float + Sum + SimdDistance,
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
    #[should_panic]
    fn test_wrong_vector_file_size() {
        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        std::fs::write(vec_file.path(), [0u8; 100]).unwrap();
        std::fs::write(norm_file.path(), [0u8; 16]).unwrap();

        let _ = MmapVectorStore::<f32>::new(vec_file.path(), norm_file.path(), 2, 4).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_wrong_norms_file_size() {
        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        std::fs::write(vec_file.path(), [0u8; 32]).unwrap();
        std::fs::write(norm_file.path(), [0u8; 100]).unwrap();

        let _ = MmapVectorStore::<f32>::new(vec_file.path(), norm_file.path(), 2, 4).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_save_vectors_length_mismatch() {
        let vectors = vec![1.0f32, 2.0, 3.0];
        let norms = vec![2.24, 5.0];

        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        MmapVectorStore::save(&vectors, &norms, 2, 2, vec_file.path(), norm_file.path()).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_save_norms_length_mismatch() {
        let vectors = vec![1.0f32, 2.0, 3.0, 4.0];
        let norms = vec![2.24];

        let vec_file = NamedTempFile::new().unwrap();
        let norm_file = NamedTempFile::new().unwrap();

        MmapVectorStore::save(&vectors, &norms, 2, 2, vec_file.path(), norm_file.path()).unwrap();
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
