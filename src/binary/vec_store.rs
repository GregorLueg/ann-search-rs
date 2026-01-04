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
    fn load_vector(&self, idx: usize) -> &[T];
    fn dim(&self) -> usize;
    fn n(&self) -> usize;
}

/////////////////
// VectorStore //
/////////////////

/// Memory-mapped vector storage
///
/// Stores vectors and norms in binary files and memory-maps them.
/// The OS handles paging - only accessed data gets loaded into RAM.
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
}

impl<T> VectorStore<T> for MmapVectorStore<T>
where
    T: Float + Sum,
{
    fn load_vector(&self, idx: usize) -> &[T] {
        let start = idx * self.dim;
        unsafe {
            std::slice::from_raw_parts(
                self.mmap_vectors
                    .as_ptr()
                    .add(start * std::mem::size_of::<T>()) as *const T,
                self.dim,
            )
        }
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
    T: Float + Sum,
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
