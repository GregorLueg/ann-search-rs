use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

/////////////
// Helpers //
/////////////

/// Hash function to use for binarisation
#[derive(Clone, Debug, Copy, PartialEq, Default)]
pub enum HashFunc {
    /// Imitates Euclidean distance via E2LSH
    #[default]
    E2LSH,
    /// Imitates Cosine distance via SimHash
    SimHash,
}

/// Parse the hash function based on distance metric
///
/// ### Params
///
/// * `s` - String that defines the distance metric
///
/// ### Returns
///
/// `Option<HashFunc>` defining the Hash functions best imitating the distance
/// function.
pub fn parse_hash_func(s: &str) -> Option<HashFunc> {
    match s.to_lowercase().as_str() {
        "euclidean" => Some(HashFunc::E2LSH),
        "cosine" => Some(HashFunc::SimHash),
        _ => None,
    }
}

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
    pub random_offsets: Option<Vec<T>>,
    pub bucket_width: Option<T>,
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
    pub fn new(
        dim: usize,
        n_bits: usize,
        bucket_width: Option<T>,
        hash_func: &HashFunc,
        seed: usize,
    ) -> Self {
        assert!(n_bits % 8 == 0, "n_bits must be multiple of 8");

        let mut binariser = Binariser {
            random_projections: Vec::new(),
            random_offsets: None,
            bucket_width: None,
            n_bits,
            dim,
        };

        match hash_func {
            HashFunc::SimHash => {
                binariser.prepare_simhash(seed);
            }
            HashFunc::E2LSH => {
                let bucket_width = bucket_width.unwrap_or(T::from_f32(4.0).unwrap());
                binariser.bucket_width = Some(bucket_width);
                binariser.prepare_e2lsh(bucket_width, seed);
            }
        };

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

        // Determine which hash function to use based on presence of offsets
        if self.random_offsets.is_none() {
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
        } else {
            // E2LSH: LSB of quantised projection
            let offsets = self.random_offsets.as_ref().unwrap();
            let width = self.bucket_width.unwrap();

            for bit_idx in 0..self.n_bits {
                let proj_base = bit_idx * self.dim;

                let mut dot = T::zero();
                for d in 0..self.dim {
                    dot = dot + vec[d] * self.random_projections[proj_base + d];
                }

                // Quantise and take LSB
                let quantized = (dot + offsets[bit_idx]) / width;
                let bucket = quantized.floor().to_i64().unwrap_or(0);

                if bucket & 1 == 1 {
                    let byte_idx = bit_idx / 8;
                    let bit_pos = bit_idx % 8;
                    binary[byte_idx] |= 1u8 << bit_pos;
                }
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
        if let Some(ref offsets) = self.random_offsets {
            total += offsets.capacity() * std::mem::size_of::<T>();
        }
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
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut random_projections: Vec<T> = (0..self.n_bits * self.dim)
            .map(|_| {
                let val: f64 = rng.sample(StandardNormal);
                T::from_f64(val).unwrap()
            })
            .collect();

        // Orthogonalise the projections via Gram-Schmidt
        for i in 0..self.n_bits {
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

        self.random_projections = random_projections;
    }

    /// Generate random projections and offsets for E2LSH
    ///
    /// Creates random projections from N(0,1) and random offsets from Uniform(0, w).
    /// No orthogonalisation for E2LSH as independence is not required.
    ///
    /// ### Params
    ///
    /// * `bucket_width` - Controls quantisation granularity (smaller = finer bins)
    /// * `seed` - Random seed for reproducible generation
    fn prepare_e2lsh(&mut self, bucket_width: T, seed: usize) {
        let mut rng = StdRng::seed_from_u64(seed as u64);

        // Random projections from N(0,1)
        let random_projections: Vec<T> = (0..self.n_bits * self.dim)
            .map(|_| {
                let val: f64 = rng.sample(StandardNormal);
                T::from_f64(val).unwrap()
            })
            .collect();

        // Random offsets from Uniform(0, bucket_width)
        let random_offsets: Vec<T> = (0..self.n_bits)
            .map(|_| {
                let val: f64 = rng.random::<f64>();
                T::from_f64(val).unwrap() * bucket_width
            })
            .collect();

        self.random_projections = random_projections;
        self.random_offsets = Some(random_offsets);
    }
}
