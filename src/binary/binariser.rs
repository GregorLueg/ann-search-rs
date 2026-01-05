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

const MAX_SAMPLES_PCA: usize = 100_000;
const ITQ_ITERATIONS: usize = 10;

/// Initialisation of the binariser
#[derive(Default)]
pub enum BinarisationInit {
    /// Random projection with orthogonalisation
    #[default]
    RandomProjections,
    /// Iterative Quantisation
    ITQ,
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
///
/// ### Fields
///
/// * `projections` - Random or learned projection vectors, flattened (n_bits * dim)
/// * `n_bits` - Number of bits in binary code (e.g., 256, 512)
/// * `mean` - Mean vector for centering (used when initialised with PCA)
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
    /// Create a new binariser using random projections
    ///
    /// Generates random orthogonalised projections for binary encoding.
    ///
    /// ### Params
    ///
    /// * `dim` - Input vector dimensionality
    /// * `n_bits` - Number of bits in output (must be multiple of 8)
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

    /// Initialise binariser using PCA followed by ITQ rotation
    ///
    /// Uses Principal Component Analysis to find the directions of maximum
    /// variance, then applies Iterative Quantisation (ITQ) to rotate these
    /// components for optimal binary quantisation. This typically produces
    /// better quality codes than random projections.
    ///
    /// ### Params
    ///
    /// * `data` - Training data matrix (n_samples × dim)
    /// * `dim` - Input vector dimensionality
    /// * `n_bits` - Number of bits in output (must be multiple of 8)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised binariser with PCA+ITQ projections
    pub fn initialise_with_pca(data: MatRef<T>, dim: usize, n_bits: usize, seed: usize) -> Self {
        assert!(n_bits % 8 == 0, "n_bits must be multiple of 8");

        let mut binariser = Binariser {
            projections: Vec::new(),
            n_bits,
            dim,
            mean: vec![T::zero(); dim],
        };

        binariser.prepare_pca_with_itq(data, seed, ITQ_ITERATIONS);

        binariser
    }

    /// Encode a vector to binary
    ///
    /// Projects the input vector onto learned or random hyperplanes and
    /// quantises the result to a binary code.
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
                let centered = vec[d] - self.mean[d];
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

    /// Generate random projections and orthogonalise them
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

    /// Initialise binariser using PCA followed by optimised Tiled-ITQ
    ///
    /// This variant is significantly more robust than standard ITQ for high-bit
    /// counts. It extracts the full PCA basis and applies different orthogonal
    /// rotations to chunks of bits to maximise information density.
    ///
    /// ### Training Steps:
    ///
    /// 1. Sample and center training data.
    /// 2. Compute full-basis PCA (dim x dim).
    /// 3. For the first 'dim' bits: Run ITQ optimisation iterations.
    /// 4. For remaining bits: Apply unique random orthogonal rotations to PCA
    ///    tiles.
    ///
    /// ### Params
    ///
    /// * `data` - Trainings data of samples x dim. Data will be downsampled
    ///   if n ≥ 100_000
    /// * `seed` - Random seed for reproducibility
    /// * `itq_iterations` - Number of ITQ iterations. Defaults to 10.
    fn prepare_pca_with_itq(&mut self, data: MatRef<T>, seed: usize, itq_iterations: usize) {
        let n = data.nrows();
        let dim = data.ncols();
        let mut rng = StdRng::seed_from_u64(seed as u64);

        // 1. Data Sampling (Same as your original)
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

        // 2. Full PCA to get the maximum possible orthogonal components (up to dim)
        let svd = sampled_data.as_ref().svd().unwrap();
        let full_v = svd.V(); // This is dim x dim

        // 3. Repeat and Rotate Logic
        let mut final_projections = Vec::with_capacity(self.n_bits * dim);
        let mut bits_remaining = self.n_bits;

        // We will work in chunks of size 'dim' (the max number of orthogonal components)
        while bits_remaining > 0 {
            let current_chunk_size = bits_remaining.min(dim);

            // Extract the top components for this chunk
            let mut v_chunk = Mat::<T>::zeros(dim, current_chunk_size);
            for j in 0..current_chunk_size {
                for i in 0..dim {
                    v_chunk[(i, j)] = full_v[(i, j)];
                }
            }

            // Generate a random rotation matrix for this specific tile
            let mut r_mat = Mat::<T>::zeros(current_chunk_size, current_chunk_size);
            for i in 0..current_chunk_size {
                for j in 0..current_chunk_size {
                    let val: f64 = rng.sample(StandardNormal);
                    r_mat[(i, j)] = T::from_f64(val).unwrap();
                }
            }

            // Orthogonalize rotation via QR
            let qr = r_mat.as_ref().qr();
            let mut r_ortho = qr.compute_Q();

            // Only run the iterative ITQ optimization on the FIRST chunk
            // to save training time. Subsequent chunks are randomized
            // rotations of the PCA space.
            if bits_remaining == self.n_bits {
                let projected_data = &sampled_data * &v_chunk;
                for _ in 0..itq_iterations {
                    let rotated = &projected_data * &r_ortho;
                    let mut b_mat = Mat::<T>::zeros(n_samples, current_chunk_size);
                    for i in 0..n_samples {
                        for j in 0..current_chunk_size {
                            b_mat[(i, j)] = if rotated[(i, j)] >= T::zero() {
                                T::one()
                            } else {
                                -T::one()
                            };
                        }
                    }
                    let c_mat = projected_data.transpose() * &b_mat;
                    let svd_itq = c_mat.as_ref().thin_svd().unwrap();
                    r_ortho = svd_itq.U() * svd_itq.V().transpose();
                }
            }

            // Apply the (optionally optimized) rotation to the PCA components
            let rotated_projections = v_chunk * r_ortho;

            // Flatten into our projections vector
            for j in 0..current_chunk_size {
                for i in 0..dim {
                    final_projections.push(rotated_projections[(i, j)]);
                }
            }

            bits_remaining -= current_chunk_size;
        }

        self.projections = final_projections;
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary::dist_binary::hamming_distance;
    use faer::Mat;

    // Binariser tests
    #[test]
    fn test_simhash_basic() {
        let dim = 128;
        let n_bits = 256;
        let binariser = Binariser::<f64>::new(dim, n_bits, 42);

        let vec1: Vec<f64> = (0..dim).map(|i| (i as f64) / (dim as f64)).collect();
        let binary = binariser.encode(&vec1);

        assert_eq!(binary.len(), n_bits / 8);
    }

    #[test]
    fn test_simhash_preserves_similarity() {
        let dim = 64;
        let n_bits = 128;
        let binariser = Binariser::<f64>::new(dim, n_bits, 42);

        let vec1: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let vec2: Vec<f64> = (0..dim).map(|i| i as f64 + 0.1).collect();
        let vec3: Vec<f64> = (0..dim).map(|i| -(i as f64)).collect();

        let bin1 = binariser.encode(&vec1);
        let bin2 = binariser.encode(&vec2);
        let bin3 = binariser.encode(&vec3);

        let dist_12 = hamming_distance(&bin1, &bin2);
        let dist_13 = hamming_distance(&bin1, &bin3);

        assert!(
            dist_12 < dist_13,
            "Similar vectors should have smaller Hamming distance"
        );
    }

    #[test]
    fn test_pca_itq_basic() {
        let n_samples = 1000;
        let dim = 64;
        let n_bits = 128;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = ((i + j) as f64).sin();
            }
        }

        let binariser = Binariser::<f64>::initialise_with_pca(data.as_ref(), dim, n_bits, 42);

        let vec1: Vec<f64> = (0..dim).map(|i| (i as f64).sin()).collect();
        let binary = binariser.encode(&vec1);

        assert_eq!(binary.len(), n_bits / 8);
    }

    #[test]
    fn test_pca_itq_orthogonality() {
        let n_samples = 500;
        let dim = 32;
        let n_bits = 128;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = ((i * j) as f64).sin();
            }
        }

        let binariser = Binariser::<f64>::initialise_with_pca(data.as_ref(), dim, n_bits, 42);

        for i in 0..n_bits.min(dim) {
            let i_base = i * dim;
            let mut norm_sq = 0.0;
            for d in 0..dim {
                norm_sq += binariser.projections[i_base + d] * binariser.projections[i_base + d];
            }
            let norm = norm_sq.sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "Projection {} not normalised: {}",
                i,
                norm
            );

            for j in (i + 1)..n_bits.min(dim) {
                let j_base = j * dim;
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += binariser.projections[i_base + d] * binariser.projections[j_base + d];
                }
                assert!(
                    dot.abs() < 1e-6,
                    "Projections {} and {} not orthogonal: {}",
                    i,
                    j,
                    dot
                );
            }
        }
    }

    #[test]
    fn test_centering() {
        let n_samples = 100;
        let dim = 16;
        let n_bits = 32;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = (i as f64) + 10.0;
            }
        }

        let binariser = Binariser::<f64>::initialise_with_pca(data.as_ref(), dim, n_bits, 42);

        for d in 0..dim {
            let expected_mean = (n_samples as f64 - 1.0) / 2.0 + 10.0;
            assert!((binariser.mean[d] - expected_mean).abs() < 1e-6);
        }
    }

    #[test]
    fn test_deterministic() {
        let dim = 32;
        let n_bits = 64;

        let binariser1 = Binariser::<f64>::new(dim, n_bits, 42);
        let binariser2 = Binariser::<f64>::new(dim, n_bits, 42);

        let vec: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let bin1 = binariser1.encode(&vec);
        let bin2 = binariser2.encode(&vec);

        assert_eq!(bin1, bin2);
    }

    #[test]
    fn test_parse_binarisation_init() {
        assert!(matches!(
            parse_binarisation_init("itq"),
            Some(BinarisationInit::ITQ)
        ));
        assert!(matches!(
            parse_binarisation_init("ITQ"),
            Some(BinarisationInit::ITQ)
        ));
        assert!(matches!(
            parse_binarisation_init("random"),
            Some(BinarisationInit::RandomProjections)
        ));
        assert!(matches!(
            parse_binarisation_init("random_projections"),
            Some(BinarisationInit::RandomProjections)
        ));
        assert!(parse_binarisation_init("invalid").is_none());
    }

    #[test]
    #[should_panic(expected = "n_bits must be multiple of 8")]
    fn test_invalid_n_bits() {
        let _binariser = Binariser::<f64>::new(64, 123, 42);
    }

    #[test]
    #[should_panic(expected = "Vector dimension mismatch")]
    fn test_dimension_mismatch() {
        let binariser = Binariser::<f64>::new(64, 128, 42);
        let wrong_vec: Vec<f64> = vec![0.0; 32];
        let _binary = binariser.encode(&wrong_vec);
    }
}
