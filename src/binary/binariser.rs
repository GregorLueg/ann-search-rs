use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

///////////////
// Binariser //
///////////////

/// Binariser using random hyperplane projections
///
/// Converts float vectors to binary codes using locality-sensitive hashing.
/// Supports SimHash (for Cosine similarity) and E2LSH (for Euclidean distance).
///
/// ### Fields
///
/// * `random_projections` - Random vectors from N(0,1), flattened (n_bits * dim)
/// * `random_offsets` - Random offsets for E2LSH (None for SimHash)
/// * `bucket_width` - Bucket width for E2LSH (None for SimHash)
/// * `n_bits` - Number of bits in binary code (e.g., 256, 512)
/// * `dim` - Input vector dimensionality
pub struct Binariser<T> {
    pub random_projections: Vec<T>,
    pub n_bits: usize,
    pub dim: usize,
}

impl<T> Binariser<T>
where
    T: Float + FromPrimitive + ToPrimitive,
{
    /// Create a new binariser
    ///
    /// Generates random projections and initialises hash function parameters.
    /// SimHash orthogonalises projections for better quality.
    ///
    /// ### Params
    ///
    /// * `dim` - Input vector dimensionality
    /// * `n_bits` - Number of bits in output (must be multiple of 8)
    /// * `bucket_width` - Bucket width for E2LSH (ignored for SimHash, defaults to 4.0 if None)
    /// * `hash_func` - Hash function type (SimHash or E2LSH)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised binariser
    pub fn new(dim: usize, n_bits: usize, seed: usize) -> Self {
        assert!(n_bits % 8 == 0, "n_bits must be multiple of 8");

        let mut binariser = Binariser {
            random_projections: Vec::new(),
            n_bits,
            dim,
        };

        binariser.prepare_simhash(seed);

        binariser
    }

    /// Encode a vector to binary
    ///
    /// Computes hash code by projecting onto random vectors and quantising.
    /// Uses different schemes based on initialised hash function.
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector (length must equal dim)
    ///
    /// ### Returns
    ///
    /// Binary code as Vec<u8> (length = n_bits / 8)
    pub fn encode(&self, vec: &[T]) -> Vec<u8> {
        assert_eq!(vec.len(), self.dim, "Vector dimension mismatch");

        let n_bytes = self.n_bits / 8;
        let mut binary = vec![0u8; n_bytes];

        // SimHash: sign of dot product
        for bit_idx in 0..self.n_bits {
            let proj_base = bit_idx * self.dim;

            let mut dot = T::zero();
            for d in 0..self.dim {
                dot = dot + vec[d] * self.random_projections[proj_base + d];
            }

            if dot >= T::zero() {
                let byte_idx = bit_idx / 8;
                let bit_pos = bit_idx % 8;
                binary[byte_idx] |= 1u8 << bit_pos;
            }
        }

        binary
    }

    /// Returns memory usage in bytes
    ///
    /// ### Returns
    ///
    /// Total bytes used by the binariser
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);
        total += self.random_projections.capacity() * std::mem::size_of::<T>();
        total
    }

    /// Generate random projections and orthogonalise them for SimHash
    ///
    /// Creates orthonormal random hyperplanes for better hash quality.
    /// Orthogonalisation via Gram-Schmidt ensures projections are independent.
    ///
    /// ### Params
    ///
    /// * `seed` - Random seed for reproducible projection generation
    fn prepare_simhash(&mut self, seed: usize) {
        let n_orthogonal = self.n_bits.min(self.dim);

        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut random_projections: Vec<T> = (0..self.n_bits * self.dim)
            .map(|_| {
                let val: f64 = rng.sample(StandardNormal);
                T::from_f64(val).unwrap()
            })
            .collect();

        // Orthogonalise the projections via Gram-Schmidt
        for i in 0..n_orthogonal {
            let i_base = i * self.dim;

            // Subtract projection onto all previous vectors
            for j in 0..i {
                let j_base = j * self.dim;
                let mut dot = T::zero();
                for d in 0..self.dim {
                    dot = dot + random_projections[i_base + d] * random_projections[j_base + d];
                }
                for d in 0..self.dim {
                    random_projections[i_base + d] =
                        random_projections[i_base + d] - dot * random_projections[j_base + d];
                }
            }

            // Normalise to unit length
            let mut norm_sq = T::zero();
            for d in 0..self.dim {
                norm_sq = norm_sq + random_projections[i_base + d] * random_projections[i_base + d];
            }
            let norm = norm_sq.sqrt();
            if norm > T::epsilon() {
                for d in 0..self.dim {
                    random_projections[i_base + d] = random_projections[i_base + d] / norm;
                }
            }
        }

        for i in n_orthogonal..self.n_bits {
            let i_base = i * self.dim;
            let mut norm_sq = T::zero();
            for d in 0..self.dim {
                norm_sq = norm_sq + random_projections[i_base + d] * random_projections[i_base + d];
            }
            let norm = norm_sq.sqrt();
            if norm > T::epsilon() {
                for d in 0..self.dim {
                    random_projections[i_base + d] = random_projections[i_base + d] / norm;
                }
            }
        }

        self.random_projections = random_projections;
    }

    /// Return raw projections without binarising
    pub fn project(&self, vec: &[T]) -> Vec<T> {
        (0..self.n_bits)
            .map(|bit_idx| {
                let proj_base = bit_idx * self.dim;
                let mut dot = T::zero();
                for d in 0..self.dim {
                    dot = dot + vec[d] * self.random_projections[proj_base + d];
                }
                dot
            })
            .collect()
    }
}
