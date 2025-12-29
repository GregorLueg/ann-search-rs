use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

///////////////
// Binariser //
///////////////

const MAX_SAMPLES_PCA: usize = 100000;

/// Initialisation of the binariser
#[derive(Default)]
pub enum BinarisationInit {
    #[default]
    ITQ,
    RandomProjections,
}

/// Helper function to parse the Binarisation initialisation
///
/// ### Params
///
/// * `s` - The string to parse
///
/// ### Returns
///
/// `Option<BinarisationInit>`
pub fn parse_binarisation_init(s: &str) -> Option<BinarisationInit> {
    match s.to_lowercase().as_str() {
        "itq" => Some(BinarisationInit::ITQ),
        "random" | "random_projections" => Some(BinarisationInit::RandomProjections),
        _ => None,
    }
}

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
    pub projections: Vec<T>,
    pub n_bits: usize,
    pub mean: Vec<T>,
    pub dim: usize,
}

impl<T> Binariser<T>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField,
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
            projections: Vec::new(),
            n_bits,
            dim,
            mean: vec![T::zero(); dim],
        };

        binariser.prepare_simhash(seed);

        binariser
    }

    pub fn initialise_with_pca(data: MatRef<T>, dim: usize, n_bits: usize, seed: usize) -> Self {
        assert!(n_bits % 8 == 0, "n_bits must be multiple of 8");

        let mut binariser = Binariser {
            projections: Vec::new(),
            n_bits,
            dim,
            mean: vec![T::zero(); dim],
        };

        binariser.prepare_pca_with_itq(data, seed, 10);

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

        for bit_idx in 0..self.n_bits {
            let proj_base = bit_idx * self.dim;
            let mut dot = T::zero();
            for d in 0..self.dim {
                let centered = vec[d] - self.mean[d]; // CENTER THE DATA
                dot = dot + centered * self.projections[proj_base + d];
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
        total += self.projections.capacity() * std::mem::size_of::<T>();
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

        self.projections = random_projections;
    }

    fn prepare_pca_with_itq(&mut self, data: MatRef<T>, seed: usize, itq_iterations: usize) {
        let n = data.nrows();
        let dim = data.ncols();
        let mut rng = StdRng::seed_from_u64(seed as u64);

        let sample_indices: Vec<usize> = if n > MAX_SAMPLES_PCA {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.shuffle(&mut rng);
            idx.into_iter().take(MAX_SAMPLES_PCA).collect()
        } else {
            (0..n).collect()
        };
        let n_samples = sample_indices.len();

        let mut sampled_data = Mat::<T>::zeros(n_samples, dim);
        let mut mean = vec![T::zero(); dim];

        for &old_idx in &sample_indices {
            for d in 0..dim {
                mean[d] = mean[d] + data[(old_idx, d)];
            }
        }
        let n_samples_t = T::from_usize(n_samples).unwrap();
        for d in 0..dim {
            mean[d] = mean[d] / n_samples_t;
        }

        for (i, &old_idx) in sample_indices.iter().enumerate() {
            for d in 0..dim {
                sampled_data[(i, d)] = data[(old_idx, d)] - mean[d];
            }
        }
        self.mean = mean;

        let svd = sampled_data.as_ref().svd().unwrap();
        let full_v = svd.V();

        let n_pca_bits = self.n_bits.min(dim);
        let mut v_pc = Mat::<T>::zeros(dim, n_pca_bits);
        for j in 0..n_pca_bits {
            for i in 0..dim {
                v_pc[(i, j)] = full_v[(i, j)];
            }
        }

        let projected_data = &sampled_data * &v_pc;
        let mut r_mat = Mat::<T>::identity(n_pca_bits, n_pca_bits);

        for _ in 0..itq_iterations {
            let rotated = &projected_data * &r_mat;
            let mut b_mat = Mat::<T>::zeros(n_samples, n_pca_bits);
            for i in 0..n_samples {
                for j in 0..n_pca_bits {
                    b_mat[(i, j)] = if rotated[(i, j)] >= T::zero() {
                        T::one()
                    } else {
                        -T::one()
                    };
                }
            }

            let c_mat = projected_data.transpose() * &b_mat;
            let svd_itq = c_mat.as_ref().svd().unwrap();

            r_mat = svd_itq.V() * svd_itq.U().transpose();
        }

        let final_projections_mat = v_pc * r_mat;
        let mut projections = Vec::with_capacity(self.n_bits * dim);

        for j in 0..n_pca_bits {
            for i in 0..dim {
                projections.push(final_projections_mat[(i, j)]);
            }
        }

        if self.n_bits > dim {
            for bit_idx in n_pca_bits..self.n_bits {
                let base = bit_idx * dim;

                for _ in 0..dim {
                    let val: f64 = rng.sample(StandardNormal);
                    projections.push(T::from_f64(val).unwrap());
                }

                if bit_idx < dim {
                    for prev_idx in 0..bit_idx {
                        let prev_base = prev_idx * dim;
                        let mut dot = T::zero();
                        for d in 0..dim {
                            dot = dot + projections[base + d] * projections[prev_base + d];
                        }
                        for d in 0..dim {
                            projections[base + d] =
                                projections[base + d] - dot * projections[prev_base + d];
                        }
                    }
                }

                let mut norm_sq = T::zero();
                for d in 0..dim {
                    norm_sq = norm_sq + projections[base + d] * projections[base + d];
                }
                let norm = norm_sq.sqrt();
                if norm > T::epsilon() {
                    for d in 0..dim {
                        projections[base + d] = projections[base + d] / norm;
                    }
                }
            }
        }

        self.projections = projections;
    }
}
