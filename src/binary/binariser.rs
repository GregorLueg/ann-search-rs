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
    Itq,
    /// Sign-based binarisation
    SignBased,
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
        "itq" => Some(BinarisationInit::Itq),
        "random" | "random_projections" => Some(BinarisationInit::RandomProjections),
        "sign" | "sign_based" => Some(BinarisationInit::SignBased),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Enum representing different binarisation methods
pub enum BinarisationMethod<T> {
    /// SimHash with random orthogonalised projections
    SimHash { projections: Vec<T> },
    /// ITQ with PCA-derived projections and mean centering
    Itq { projections: Vec<T>, mean: Vec<T> },
    /// Sign-based binarisation (no projections needed)
    SignBased,
}

// Generate random projections and orthogonalise them
///
/// Creates orthonormal random hyperplanes for better hash quality.
/// Orthogonalisation via Gram-Schmidt ensures projections are independent.
///
/// ### Params
///
/// * `dim` - Input vector dimensionality
/// * `n_bits` - Number of bits in output
/// * `seed` - Random seed for reproducible projection generation
///
/// ### Returns
///
/// Flattened projection matrix (n_bits × dim)
fn prepare_simhash_projections<T>(dim: usize, n_bits: usize, seed: usize) -> Vec<T>
where
    T: Float + FromPrimitive + Copy,
{
    let n_orthogonal = n_bits.min(dim);

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut random_projections: Vec<T> = (0..n_bits * dim)
        .map(|_| {
            let val: f64 = rng.sample(StandardNormal);
            T::from_f64(val).unwrap()
        })
        .collect();

    // Orthogonalise the projections via Gram-Schmidt
    for i in 0..n_orthogonal {
        let i_base = i * dim;

        // Subtract projection onto all previous vectors
        for j in 0..i {
            let j_base = j * dim;
            let mut dot = T::zero();
            for d in 0..dim {
                dot = dot + random_projections[i_base + d] * random_projections[j_base + d];
            }
            for d in 0..dim {
                random_projections[i_base + d] =
                    random_projections[i_base + d] - dot * random_projections[j_base + d];
            }
        }

        // Normalise to unit length
        let mut norm_sq = T::zero();
        for d in 0..dim {
            norm_sq = norm_sq + random_projections[i_base + d] * random_projections[i_base + d];
        }
        let norm = norm_sq.sqrt();
        if norm > T::epsilon() {
            for d in 0..dim {
                random_projections[i_base + d] = random_projections[i_base + d] / norm;
            }
        }
    }

    for i in n_orthogonal..n_bits {
        let i_base = i * dim;
        let mut norm_sq = T::zero();
        for d in 0..dim {
            norm_sq = norm_sq + random_projections[i_base + d] * random_projections[i_base + d];
        }
        let norm = norm_sq.sqrt();
        if norm > T::epsilon() {
            for d in 0..dim {
                random_projections[i_base + d] = random_projections[i_base + d] / norm;
            }
        }
    }

    random_projections
}

/// Initialise binariser using PCA followed by optimised Tiled-ITQ
///
/// This variant is significantly more robust than standard ITQ for high-bit
/// counts. It extracts the full PCA basis and applies different orthogonal
/// rotations to chunks of bits to maximise information density.
///
/// ### Training Steps:
///
/// 1. Sample and centre training data.
/// 2. Compute full-basis PCA (dim × dim).
/// 3. For the first 'dim' bits: Run ITQ optimisation iterations.
/// 4. For remaining bits: Apply unique random orthogonal rotations to PCA
///    tiles.
///
/// ### Params
///
/// * `data` - Training data of samples × dim. Data will be downsampled if
///   n ≥ 100,000
/// * `n_bits` - Number of bits in output
/// * `seed` - Random seed for reproducibility
/// * `itq_iterations` - Number of ITQ iterations
///
/// ### Returns
///
/// Tuple of (projections, mean) where projections is flattened (n_bits × dim)
fn prepare_itq_projections<T>(
    data: MatRef<T>,
    n_bits: usize,
    seed: usize,
    itq_iterations: usize,
) -> (Vec<T>, Vec<T>)
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField,
{
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

    // 2. Full PCA to get the maximum possible orthogonal components (up to dim)
    let svd = sampled_data.as_ref().svd().unwrap();
    let full_v = svd.V();

    // 3. Repeat and Rotate Logic
    let mut final_projections = Vec::with_capacity(n_bits * dim);
    let mut bits_remaining = n_bits;

    while bits_remaining > 0 {
        let current_chunk_size = bits_remaining.min(dim);

        let mut v_chunk = Mat::<T>::zeros(dim, current_chunk_size);
        for j in 0..current_chunk_size {
            for i in 0..dim {
                v_chunk[(i, j)] = full_v[(i, j)];
            }
        }

        let mut r_mat = Mat::<T>::zeros(current_chunk_size, current_chunk_size);
        for i in 0..current_chunk_size {
            for j in 0..current_chunk_size {
                let val: f64 = rng.sample(StandardNormal);
                r_mat[(i, j)] = T::from_f64(val).unwrap();
            }
        }

        let qr = r_mat.as_ref().qr();
        let mut r_ortho = qr.compute_Q();

        if bits_remaining == n_bits {
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

        let rotated_projections = v_chunk * r_ortho;

        for j in 0..current_chunk_size {
            for i in 0..dim {
                final_projections.push(rotated_projections[(i, j)]);
            }
        }

        bits_remaining -= current_chunk_size;
    }

    (final_projections, mean)
}

/// Encode a vector to binary using projection-based methods
///
/// Projects the input vector onto learned or random hyperplanes and
/// quantises the result to a binary code.
///
/// ### Params
///
/// * `vec` - Input vector (length must equal dim)
/// * `projections` - Hyperplane projections (length must equal n_bits × dim)
/// * `mean` - Mean vector for centring (empty slice for SimHash, populated for
///   ITQ)
/// * `n_bits` - Number of bits to encode
/// * `dim` - Dimension of the input vector
///
/// ### Returns
///
/// Binary code as Vec<u8> (length = n_bits / 8)
fn encode_with_projections<T>(
    vec: &[T],
    projections: &[T],
    mean: &[T],
    n_bits: usize,
    dim: usize,
) -> Vec<u8>
where
    T: Float,
{
    assert_eq!(vec.len(), dim, "Vector dimension mismatch");

    let n_bytes = n_bits / 8;
    let mut binary = vec![0u8; n_bytes];

    for bit_idx in 0..n_bits {
        let proj_base = bit_idx * dim;
        let mut dot = T::zero();
        for d in 0..dim {
            let centered = if mean.is_empty() {
                vec[d]
            } else {
                vec[d] - mean[d]
            };
            dot = dot + centered * projections[proj_base + d];
        }

        if dot >= T::zero() {
            let byte_idx = bit_idx / 8;
            let bit_pos = bit_idx % 8;
            binary[byte_idx] |= 1u8 << bit_pos;
        }
    }

    binary
}

/// Encode a vector to binary using sign-based binarisation
///
/// Simply takes the sign of each component. Positive values (including zero)
/// map to 1, negative values map to 0.
///
/// ### Params
///
/// * `vec` - Input vector (length must equal dim)
/// * `dim` - Dimension of the input vector
///
/// ### Returns
///
/// Binary code as Vec<u8> (length = (dim + 7) / 8)
fn encode_sign_based<T: Float>(vec: &[T], dim: usize) -> Vec<u8> {
    let n_bytes = dim.div_ceil(8);
    let mut binary = vec![0u8; n_bytes];

    for (bit_idx, &val) in vec.iter().enumerate() {
        if val >= T::zero() {
            let byte_idx = bit_idx / 8;
            let bit_pos = bit_idx % 8;
            binary[byte_idx] |= 1u8 << bit_pos;
        }
    }

    binary
}

/// Binariser for converting float vectors to binary codes
///
/// Supports three binarisation methods:
///
/// - **SimHash**: Random orthogonalised projections
/// - **ITQ**: PCA followed by Iterative Quantisation
/// - **SignBased**: Simple sign binarisation (no training required)
///
/// ### Fields
///
/// * `method` - The binarisation method and its parameters
/// * `n_bits` - Number of bits in output codes
/// * `dim` - Input vector dimensionality
pub struct Binariser<T> {
    pub method: BinarisationMethod<T>,
    pub n_bits: usize,
    pub dim: usize,
}

impl<T> Binariser<T>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField,
{
    /// Create a new binariser using random projections (SimHash)
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
    pub fn new_simhash(dim: usize, n_bits: usize, seed: usize) -> Self {
        assert!(n_bits.is_multiple_of(8), "n_bits must be multiple of 8");

        let projections = prepare_simhash_projections(dim, n_bits, seed);
        Self {
            method: BinarisationMethod::SimHash { projections },
            n_bits,
            dim,
        }
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
    pub fn new_itq(data: MatRef<T>, dim: usize, n_bits: usize, seed: usize) -> Self {
        assert!(n_bits.is_multiple_of(8), "n_bits must be multiple of 8");

        let (projections, mean) = prepare_itq_projections(data, n_bits, seed, ITQ_ITERATIONS);

        Self {
            method: BinarisationMethod::Itq { projections, mean },
            n_bits,
            dim,
        }
    }

    /// Create a new binariser using sign-based binarisation
    ///
    /// No training required. Simply binarises based on the sign of each
    /// component. Output has exactly `dim` bits (one per dimension).
    ///
    /// ### Params
    ///
    /// * `dim` - Input vector dimensionality
    ///
    /// ### Returns
    ///
    /// Initialised binariser
    pub fn new_sign_based(dim: usize) -> Self {
        Self {
            method: BinarisationMethod::SignBased,
            n_bits: dim, // sign-based always produces dim bits
            dim,
        }
    }

    /// Encode a vector to binary
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector (length must equal dim)
    ///
    /// ### Returns
    ///
    /// Binary code as Vec<u8>
    pub fn encode(&self, vec: &[T]) -> Vec<u8> {
        assert_eq!(vec.len(), self.dim);

        match &self.method {
            BinarisationMethod::SimHash { projections } => {
                encode_with_projections(vec, projections, &[], self.n_bits, self.dim)
            }
            BinarisationMethod::Itq { projections, mean } => {
                encode_with_projections(vec, projections, mean, self.n_bits, self.dim)
            }
            BinarisationMethod::SignBased => encode_sign_based(vec, self.dim),
        }
    }

    /// Returns memory usage in bytes
    ///
    /// ### Returns
    ///
    /// Total bytes used by the binariser
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);
        match &self.method {
            BinarisationMethod::SimHash { projections } => {
                total += projections.capacity() * std::mem::size_of::<T>();
            }
            BinarisationMethod::Itq { projections, mean } => {
                total += projections.capacity() * std::mem::size_of::<T>();
                total += mean.capacity() * std::mem::size_of::<T>();
            }
            BinarisationMethod::SignBased => {}
        }
        total
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

    #[test]
    fn test_simhash_basic() {
        let dim = 128;
        let n_bits = 256;
        let binariser = Binariser::<f64>::new_simhash(dim, n_bits, 42);

        let vec1: Vec<f64> = (0..dim).map(|i| (i as f64) / (dim as f64)).collect();
        let binary = binariser.encode(&vec1);

        assert_eq!(binary.len(), n_bits / 8);
    }

    #[test]
    fn test_simhash_preserves_similarity() {
        let dim = 64;
        let n_bits = 128;
        let binariser = Binariser::<f64>::new_simhash(dim, n_bits, 42);

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
    fn test_itq_basic() {
        let n_samples = 1000;
        let dim = 64;
        let n_bits = 128;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = ((i + j) as f64).sin();
            }
        }

        let binariser = Binariser::<f64>::new_itq(data.as_ref(), dim, n_bits, 42);

        let vec1: Vec<f64> = (0..dim).map(|i| (i as f64).sin()).collect();
        let binary = binariser.encode(&vec1);

        assert_eq!(binary.len(), n_bits / 8);
    }

    #[test]
    fn test_itq_orthogonality() {
        let n_samples = 500;
        let dim = 32;
        let n_bits = 128;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = ((i * j) as f64).sin();
            }
        }

        let binariser = Binariser::<f64>::new_itq(data.as_ref(), dim, n_bits, 42);

        if let BinarisationMethod::Itq { projections, .. } = &binariser.method {
            for i in 0..n_bits.min(dim) {
                let i_base = i * dim;
                let mut norm_sq = 0.0;
                for d in 0..dim {
                    norm_sq += projections[i_base + d] * projections[i_base + d];
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
                        dot += projections[i_base + d] * projections[j_base + d];
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
        } else {
            panic!("Expected ITQ method");
        }
    }

    #[test]
    fn test_itq_centering() {
        let n_samples = 100;
        let dim = 16;
        let n_bits = 32;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = (i as f64) + 10.0;
            }
        }

        let binariser = Binariser::<f64>::new_itq(data.as_ref(), dim, n_bits, 42);

        if let BinarisationMethod::Itq { mean, .. } = &binariser.method {
            for d in 0..dim {
                let expected_mean = (n_samples as f64 - 1.0) / 2.0 + 10.0;
                assert!((mean[d] - expected_mean).abs() < 1e-6);
            }
        } else {
            panic!("Expected ITQ method");
        }
    }

    #[test]
    fn test_sign_based_basic() {
        let dim = 128;
        let binariser = Binariser::<f64>::new_sign_based(dim);

        assert_eq!(binariser.n_bits, dim);

        let vec: Vec<f64> = (0..dim)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let binary = binariser.encode(&vec);

        assert_eq!(binary.len(), dim.div_ceil(8));

        // Check first few bits match expected pattern
        for i in 0..8 {
            let byte_idx = i / 8;
            let bit_pos = i % 8;
            let bit_set = (binary[byte_idx] & (1u8 << bit_pos)) != 0;
            assert_eq!(bit_set, i % 2 == 0, "Bit {} should be {}", i, i % 2 == 0);
        }
    }

    #[test]
    fn test_sign_based_preserves_similarity() {
        let dim = 64;
        let binariser = Binariser::<f64>::new_sign_based(dim);

        let vec1: Vec<f64> = (0..dim).map(|i| (i + 1) as f64).collect(); // Start from 1
        let vec2: Vec<f64> = (0..dim).map(|i| (i + 1) as f64 + 0.1).collect();
        let vec3: Vec<f64> = (0..dim).map(|i| -((i + 1) as f64)).collect(); // All negative

        let bin1 = binariser.encode(&vec1);
        let bin2 = binariser.encode(&vec2);
        let bin3 = binariser.encode(&vec3);

        let dist_12 = hamming_distance(&bin1, &bin2);
        let dist_13 = hamming_distance(&bin1, &bin3);

        assert_eq!(
            dist_12, 0,
            "Vectors with same signs should have zero Hamming distance"
        );
        assert_eq!(
            dist_13, dim as u32,
            "Vectors with opposite signs should have maximum Hamming distance"
        );
    }

    #[test]
    fn test_deterministic() {
        let dim = 32;
        let n_bits = 64;

        let binariser1 = Binariser::<f64>::new_simhash(dim, n_bits, 42);
        let binariser2 = Binariser::<f64>::new_simhash(dim, n_bits, 42);

        let vec: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let bin1 = binariser1.encode(&vec);
        let bin2 = binariser2.encode(&vec);

        assert_eq!(bin1, bin2);
    }

    #[test]
    fn test_parse_binarisation_init() {
        assert!(matches!(
            parse_binarisation_init("itq"),
            Some(BinarisationInit::Itq)
        ));
        assert!(matches!(
            parse_binarisation_init("ITQ"),
            Some(BinarisationInit::Itq)
        ));
        assert!(matches!(
            parse_binarisation_init("random"),
            Some(BinarisationInit::RandomProjections)
        ));
        assert!(matches!(
            parse_binarisation_init("random_projections"),
            Some(BinarisationInit::RandomProjections)
        ));
        assert!(matches!(
            parse_binarisation_init("sign"),
            Some(BinarisationInit::SignBased)
        ));
        assert!(matches!(
            parse_binarisation_init("sign_based"),
            Some(BinarisationInit::SignBased)
        ));
        assert!(parse_binarisation_init("invalid").is_none());
    }

    #[test]
    #[should_panic]
    fn test_invalid_n_bits_simhash() {
        let _binariser = Binariser::<f64>::new_simhash(64, 123, 42);
    }

    #[test]
    #[should_panic]
    fn test_invalid_n_bits_itq() {
        let data = Mat::<f64>::zeros(100, 64);
        let _binariser = Binariser::<f64>::new_itq(data.as_ref(), 64, 123, 42);
    }

    #[test]
    #[should_panic]
    fn test_dimension_mismatch() {
        let binariser = Binariser::<f64>::new_simhash(64, 128, 42);
        let wrong_vec: Vec<f64> = vec![0.0; 32];
        let _binary = binariser.encode(&wrong_vec);
    }

    #[test]
    fn test_memory_usage() {
        let dim = 32;
        let n_bits = 64;

        let simhash = Binariser::<f64>::new_simhash(dim, n_bits, 42);
        let simhash_mem = simhash.memory_usage_bytes();
        assert!(simhash_mem > 0);

        let sign_based = Binariser::<f64>::new_sign_based(dim);
        let sign_mem = sign_based.memory_usage_bytes();
        assert!(sign_mem > 0);
        assert!(simhash_mem > sign_mem, "SimHash should use more memory");
    }
}
