//! Contains the binarisers

use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

use crate::prelude::*;

///////////////
// Binariser //
///////////////

const MAX_SAMPLES_PCA: usize = 100_000;

/// Initialisation of the binariser
#[derive(Default, Eq, PartialEq)]
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub enum BinarisationInit {
    /// Random projection with orthogonalisation
    #[default]
    RandomProjections,
    /// PCA-based hashing
    PcaHashing,
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
        "pca" | "pca_hashing" => Some(BinarisationInit::PcaHashing),
        "random" | "random_projections" => Some(BinarisationInit::RandomProjections),
        "sign" | "sign_based" => Some(BinarisationInit::SignBased),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Enum representing different binarisation methods
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub enum BinarisationMethod<T> {
    /// SimHash with random orthogonalised projections
    SimHash {
        /// The random, orthogonal projection
        projections: Vec<T>,
        /// The mean value across each feature. Random hyperplanes pass through
        /// the origin, so without centring every bit is biased on data whose
        /// mean sits far from it and the codes collapse.
        mean: Vec<T>,
    },
    /// PCA hashing with learned projections and mean centring
    PcaHashing {
        /// Projections based on PCA loadings
        projections: Vec<T>,
        /// The mean value across each feature
        mean: Vec<T>,
    },
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

/// Row indices to train on, capped at [`MAX_SAMPLES_PCA`]
///
/// Shared by the two trained binarisers so they agree on the subsample for a
/// given seed.
///
/// ### Params
///
/// * `n` - Number of rows in the training matrix
/// * `seed` - Random seed for reproducible subsampling
///
/// ### Returns
///
/// All row indices when `n <= MAX_SAMPLES_PCA`, otherwise a random subset of
/// that size.
fn training_sample_indices(n: usize, seed: usize) -> Vec<usize> {
    if n <= MAX_SAMPLES_PCA {
        return (0..n).collect();
    }

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut rng);
    idx.truncate(MAX_SAMPLES_PCA);

    idx
}

/// Per-feature mean over the sampled rows
///
/// ### Params
///
/// * `data` - Training data matrix (n_samples x dim)
/// * `sample_indices` - Rows to average over, must be non-empty
///
/// ### Returns
///
/// The mean of each feature, length `dim`.
fn feature_mean<T>(data: MatRef<T>, sample_indices: &[usize]) -> Vec<T>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField,
{
    let dim = data.ncols();
    let mut mean = vec![T::zero(); dim];

    for &idx in sample_indices {
        for d in 0..dim {
            mean[d] = mean[d] + data[(idx, d)];
        }
    }

    let n_t = T::from_usize(sample_indices.len().max(1)).unwrap();
    for d in 0..dim {
        mean[d] = mean[d] / n_t;
    }

    mean
}

/// Initialise binariser using PCA hashing
///
/// Learns a projection from the top principal components of the training
/// data. Each bit corresponds to the sign of a data point's score on one
/// principal component. The top PCs capture the directions of greatest
/// variance, so the resulting bits preserve the most informative structure
/// in the data.
///
/// ### Algorithm
///
/// 1. Sample and centre training data
/// 2. Compute thin SVD to obtain the top-k right singular vectors (loadings)
/// 3. Use the loadings directly as projection directions
///
/// ### Limitations
///
/// PCA hashing can only produce `min(n_bits, dim)` meaningful bits. If
/// `n_bits > dim`, the excess bits are filled with random orthogonal
/// projections, as there are no additional variance directions to capture.
///
/// The top principal components tend to carry disproportionately more
/// variance than lower ones, so not all bits are equally informative.
/// This is a known trade-off compared to methods like ITQ which attempt
/// to balance variance across bits via rotation.
///
/// ### Params
///
/// * `data` - Training data matrix (n_samples x dim). Automatically
///   downsampled if n_samples exceeds MAX_SAMPLES_PCA
/// * `n_bits` - Number of bits in output
/// * `seed` - Random seed for reproducibility (used for subsampling and
///   fallback random projections)
///
/// ### Returns
///
/// Tuple of (projections, mean) where projections is flattened
/// (n_bits x dim) and mean is the per-feature mean (length dim)
fn prepare_pca_projections<T>(data: MatRef<T>, n_bits: usize, seed: usize) -> (Vec<T>, Vec<T>)
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField,
{
    let n = data.nrows();
    let dim = data.ncols();
    let effective_bits = n_bits.min(dim);

    let sample_indices = training_sample_indices(n, seed);
    let n_samples = sample_indices.len();
    let mean = feature_mean(data, &sample_indices);

    // centre data
    let mut centered = Mat::<T>::zeros(n_samples, dim);
    for (i, &idx) in sample_indices.iter().enumerate() {
        for d in 0..dim {
            centered[(i, d)] = data[(idx, d)] - mean[d];
        }
    }

    // thin SVD to obtain loadings
    let svd = centered.as_ref().thin_svd().unwrap();
    let v_full = svd.V(); // dim x min(n_samples, dim)

    // extract top-k loadings as projection directions
    let k = effective_bits.min(v_full.ncols());
    let mut projections = Vec::with_capacity(n_bits * dim);
    for j in 0..k {
        for i in 0..dim {
            projections.push(v_full[(i, j)]);
        }
    }

    // pad with random projections if n_bits > effective rank
    if n_bits > k {
        let extra = prepare_simhash_projections::<T>(dim, n_bits - k, seed + 1);
        projections.extend(extra);
    }

    (projections, mean)
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
) -> Result<Vec<u8>, AnnSearchErrors>
where
    T: Float,
{
    if vec.len() != dim {
        return Err(AnnSearchErrors::DimensionMismatch {
            index_dim: dim,
            query_dim: vec.len(),
        });
    }

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

    Ok(binary)
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

/// Sign-encode the residual of `vec` against `centroid`, in place
///
/// Bit `d` is set when `scale.0 * vec[d] - scale.1 * centroid[d] >= 0`.
///
/// The scale pair exists so Cosine can compare unit-length vectors without
/// dividing. Since `‖vec‖ * ‖centroid‖ > 0`,
///
/// ```text
/// [vec_d/‖vec‖ - c_d/‖c‖ >= 0]  ==  [‖c‖ * vec_d - ‖vec‖ * c_d >= 0]
/// ```
///
/// so passing `(‖centroid‖, ‖vec‖)` scales the comparison by a positive
/// constant and leaves every sign untouched. Squared Euclidean passes `(1, 1)`
/// and gets the plain residual.
///
/// ### Params
///
/// * `vec` - Input vector, length `dim`
/// * `centroid` - Centroid of the vector's assigned cell, length `dim`
/// * `scale` - `(vector scale, centroid scale)`, both strictly positive
/// * `dim` - Dimensionality
/// * `out` - Destination code, length `dim.div_ceil(8)`. Zeroed on entry.
pub(crate) fn encode_sign_residual_into<T: Float>(
    vec: &[T],
    centroid: &[T],
    scale: (T, T),
    dim: usize,
    out: &mut [u8],
) {
    // Load-bearing: the bit loop only ORs, so a reused buffer keeps stale bits
    out.fill(0);

    for bit_idx in 0..dim {
        if scale.0 * vec[bit_idx] - scale.1 * centroid[bit_idx] >= T::zero() {
            out[bit_idx / 8] |= 1u8 << (bit_idx % 8);
        }
    }
}

/// Binariser for converting float vectors to binary codes
///
/// Supports three binarisation methods:
///
/// - **SimHash**: Random orthogonalised projections
/// - **PcaHashing**: Signs of the top principal components, padded with random
///   orthogonal directions when `n_bits > dim`. No rotation learning, so this
///   is plain PCA hashing rather than ITQ.
/// - **SignBased**: Simple sign binarisation (no training required)
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct Binariser<T> {
    /// The binarisation method and its parameters
    pub method: BinarisationMethod<T>,
    /// Number of bits in output codes
    pub n_bits: usize,
    /// Input vector dimensionality
    pub dim: usize,
}

impl<T> Binariser<T>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField,
{
    /// Create a new binariser using random projections (SimHash)
    ///
    /// Generates random orthogonalised projections for binary encoding. The
    /// hyperplanes pass through the origin, so the data is centred on its
    /// per-feature mean first: without that, data sitting far from the origin
    /// puts almost every point on the same side of almost every hyperplane and
    /// the codes carry next to no information.
    ///
    /// ### Params
    ///
    /// * `data` - Training data matrix (n_samples x dim), used only to fit the
    ///   centring mean. Automatically downsampled if n_samples exceeds
    ///   MAX_SAMPLES_PCA
    /// * `dim` - Input vector dimensionality
    /// * `n_bits` - Number of bits in output (must be multiple of 8)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised binariser
    pub fn new_simhash(
        data: MatRef<T>,
        dim: usize,
        n_bits: usize,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        if !n_bits.is_multiple_of(8) {
            return Err(AnnSearchErrors::NBitsMustBe8Multiple { n_bits });
        }

        let projections = prepare_simhash_projections(dim, n_bits, seed);
        let mean = feature_mean(data, &training_sample_indices(data.nrows(), seed));

        Ok(Self {
            method: BinarisationMethod::SimHash { projections, mean },
            n_bits,
            dim,
        })
    }

    /// Initialise binariser using PCA hashing
    ///
    /// Uses Principal Component Analysis to find the directions of maximum
    /// variance in the training data, then binarises by taking the sign of
    /// each data point's projection onto these directions.
    ///
    /// ### Params
    ///
    /// * `data` - Training data matrix (n_samples x dim)
    /// * `dim` - Input vector dimensionality
    /// * `n_bits` - Number of bits in output (must be multiple of 8)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised binariser with PCA projections
    pub fn new_pca_hashing(
        data: MatRef<T>,
        dim: usize,
        n_bits: usize,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        if !n_bits.is_multiple_of(8) {
            return Err(AnnSearchErrors::NBitsMustBe8Multiple { n_bits });
        }

        let (projections, mean) = prepare_pca_projections(data, n_bits, seed);

        Ok(Self {
            method: BinarisationMethod::PcaHashing { projections, mean },
            n_bits,
            dim,
        })
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

    /// Length of the codes this binariser emits
    ///
    /// This is the only correct stride for a flattened code array. It is *not*
    /// derivable from the `n_bits` a caller passed to a `build_*` function:
    /// sign-based binarisation ignores that argument and always produces `dim`
    /// bits, so taking the stride from the argument corrupts the layout
    /// whenever `n_bits != dim`.
    ///
    /// ### Returns
    ///
    /// Bytes per encoded vector.
    pub fn n_bytes(&self) -> usize {
        self.n_bits.div_ceil(8)
    }

    /// Encode a vector to binary
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector (length must equal dim)
    ///
    /// ### Returns
    ///
    /// Binary code as `Vec<u8>`
    pub fn encode(&self, vec: &[T]) -> Result<Vec<u8>, AnnSearchErrors> {
        if vec.len() != self.dim {
            return Err(AnnSearchErrors::DimensionMismatch {
                index_dim: self.dim,
                query_dim: vec.len(),
            });
        }

        match &self.method {
            BinarisationMethod::SimHash { projections, mean } => {
                encode_with_projections(vec, projections, mean, self.n_bits, self.dim)
            }
            BinarisationMethod::PcaHashing { projections, mean } => {
                encode_with_projections(vec, projections, mean, self.n_bits, self.dim)
            }
            BinarisationMethod::SignBased => Ok(encode_sign_based(vec, self.dim)),
        }
    }

    /// Encode a vector as the sign of its residual against a centroid
    ///
    /// Sign bits taken in the global frame encode which cluster a point sits
    /// in, not where it sits inside that cluster, because a cluster far from
    /// the origin puts every one of its members on the same side of every
    /// coordinate plane. Taking the residual against the cluster's own centroid
    /// moves the frame to the cluster, so the bits carry the within-cluster
    /// structure the search funnel actually needs.
    ///
    /// Codes produced this way are only comparable against other codes taken
    /// against the *same* centroid.
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector (length must equal `dim`)
    /// * `centroid` - Centroid to take the residual against (length must equal
    ///   `dim`)
    /// * `scale` - `(vector scale, centroid scale)`, see
    ///   [`encode_sign_residual_into`]
    ///
    /// ### Returns
    ///
    /// Binary code as `Vec<u8>`, or
    /// [`AnnSearchErrors::ResidualEncodingUnsupported`] for the projection-based
    /// methods, which carry a frame of their own.
    pub fn encode_residual(
        &self,
        vec: &[T],
        centroid: &[T],
        scale: (T, T),
    ) -> Result<Vec<u8>, AnnSearchErrors> {
        if vec.len() != self.dim {
            return Err(AnnSearchErrors::DimensionMismatch {
                index_dim: self.dim,
                query_dim: vec.len(),
            });
        }
        if !matches!(self.method, BinarisationMethod::SignBased) {
            return Err(AnnSearchErrors::ResidualEncodingUnsupported);
        }

        let mut out = vec![0u8; self.n_bytes()];
        encode_sign_residual_into(vec, centroid, scale, self.dim, &mut out);

        Ok(out)
    }

    /// Returns memory usage in bytes
    ///
    /// ### Returns
    ///
    /// Total bytes used by the binariser
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);
        match &self.method {
            BinarisationMethod::SimHash { projections, mean } => {
                total += projections.capacity() * std::mem::size_of::<T>();
                total += mean.capacity() * std::mem::size_of::<T>();
            }
            BinarisationMethod::PcaHashing { projections, mean } => {
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
        let zero_mean = Mat::<f64>::zeros(8, dim);
        let binariser =
            Binariser::<f64>::new_simhash(zero_mean.as_ref(), dim, n_bits, 42).unwrap();

        let vec1: Vec<f64> = (0..dim).map(|i| (i as f64) / (dim as f64)).collect();
        let binary = binariser.encode(&vec1).unwrap();

        assert_eq!(binary.len(), n_bits / 8);
    }

    #[test]
    fn test_simhash_preserves_similarity() {
        let dim = 64;
        let n_bits = 128;
        let zero_mean = Mat::<f64>::zeros(8, dim);
        let binariser =
            Binariser::<f64>::new_simhash(zero_mean.as_ref(), dim, n_bits, 42).unwrap();

        let vec1: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let vec2: Vec<f64> = (0..dim).map(|i| i as f64 + 0.1).collect();
        let vec3: Vec<f64> = (0..dim).map(|i| -(i as f64)).collect();

        let bin1 = binariser.encode(&vec1).unwrap();
        let bin2 = binariser.encode(&vec2).unwrap();
        let bin3 = binariser.encode(&vec3).unwrap();

        let dist_12 = hamming_distance(&bin1, &bin2);
        let dist_13 = hamming_distance(&bin1, &bin3);

        assert!(
            dist_12 < dist_13,
            "Similar vectors should have smaller Hamming distance"
        );
    }

    #[test]
    fn test_pca_hashing_basic() {
        let n_samples = 1000;
        let dim = 64;
        let n_bits = 128;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = ((i + j) as f64).sin();
            }
        }

        let binariser = Binariser::<f64>::new_pca_hashing(data.as_ref(), dim, n_bits, 42).unwrap();

        let vec1: Vec<f64> = (0..dim).map(|i| (i as f64).sin()).collect();
        let binary = binariser.encode(&vec1).unwrap();

        assert_eq!(binary.len(), n_bits / 8);
    }

    #[test]
    fn test_pca_hashing_orthogonality() {
        let n_samples = 500;
        let dim = 32;
        let n_bits = 128;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = ((i * j) as f64).sin();
            }
        }

        let binariser = Binariser::<f64>::new_pca_hashing(data.as_ref(), dim, n_bits, 42).unwrap();

        if let BinarisationMethod::PcaHashing { projections, .. } = &binariser.method {
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
            panic!("Expected PcaHashing method");
        }
    }

    #[test]
    fn test_pca_hashing_centring() {
        let n_samples = 100;
        let dim = 16;
        let n_bits = 32;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = (i as f64) + 10.0;
            }
        }

        let binariser = Binariser::<f64>::new_pca_hashing(data.as_ref(), dim, n_bits, 42).unwrap();

        if let BinarisationMethod::PcaHashing { mean, .. } = &binariser.method {
            for d in 0..dim {
                let expected_mean = (n_samples as f64 - 1.0) / 2.0 + 10.0;
                assert!((mean[d] - expected_mean).abs() < 1e-6);
            }
        } else {
            panic!("Expected PcaHashing method");
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
        let binary = binariser.encode(&vec).unwrap();

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

        let vec1: Vec<f64> = (0..dim).map(|i| (i + 1) as f64).collect();
        let vec2: Vec<f64> = (0..dim).map(|i| (i + 1) as f64 + 0.1).collect();
        let vec3: Vec<f64> = (0..dim).map(|i| -((i + 1) as f64)).collect();

        let bin1 = binariser.encode(&vec1).unwrap();
        let bin2 = binariser.encode(&vec2).unwrap();
        let bin3 = binariser.encode(&vec3).unwrap();

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

        let zero_mean = Mat::<f64>::zeros(8, dim);
        let binariser1 =
            Binariser::<f64>::new_simhash(zero_mean.as_ref(), dim, n_bits, 42).unwrap();
        let binariser2 =
            Binariser::<f64>::new_simhash(zero_mean.as_ref(), dim, n_bits, 42).unwrap();

        let vec: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let bin1 = binariser1.encode(&vec).unwrap();
        let bin2 = binariser2.encode(&vec).unwrap();

        assert_eq!(bin1, bin2);
    }

    #[test]
    fn test_parse_binarisation_init() {
        assert!(matches!(
            parse_binarisation_init("pca"),
            Some(BinarisationInit::PcaHashing)
        ));
        assert!(matches!(
            parse_binarisation_init("pca_hashing"),
            Some(BinarisationInit::PcaHashing)
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
    fn test_invalid_n_bits_simhash() {
        let zero_mean = Mat::<f64>::zeros(8, 64);
        let result = Binariser::<f64>::new_simhash(zero_mean.as_ref(), 64, 123, 42);
        assert!(matches!(
            result,
            Err(AnnSearchErrors::NBitsMustBe8Multiple { n_bits: 123 })
        ));
    }

    #[test]
    fn test_invalid_n_bits_pca_hashing() {
        let data = Mat::<f64>::zeros(100, 64);
        let result = Binariser::<f64>::new_pca_hashing(data.as_ref(), 64, 123, 42);
        assert!(matches!(
            result,
            Err(AnnSearchErrors::NBitsMustBe8Multiple { n_bits: 123 })
        ));
    }

    #[test]
    fn test_dimension_mismatch() {
        let zero_mean = Mat::<f64>::zeros(8, 64);
        let binariser =
            Binariser::<f64>::new_simhash(zero_mean.as_ref(), 64, 128, 42).unwrap();
        let result = binariser.encode(&vec![0.0; 32]);
        assert!(matches!(
            result,
            Err(AnnSearchErrors::DimensionMismatch {
                index_dim: 64,
                query_dim: 32
            })
        ));
    }

    /// SimHash hyperplanes pass through the origin, so an uncentred fit on data
    /// sitting far from it puts nearly every point on the same side of nearly
    /// every plane. Codes then collapse towards a single value and Hamming
    /// distance stops discriminating. Centring on the training mean restores a
    /// roughly balanced bit per plane.
    #[test]
    fn test_simhash_centres_off_origin_data() {
        let (n, dim, n_bits) = (256, 32, 64);

        // Tight cloud a long way from the origin
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(19);
        let data = Mat::<f64>::from_fn(n, dim, |_, _| 50.0 + rng.random::<f64>() * 2.0 - 1.0);

        let binariser =
            Binariser::<f64>::new_simhash(data.as_ref(), dim, n_bits, 42).unwrap();

        let codes: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                let row: Vec<f64> = data.row(i).iter().cloned().collect();
                binariser.encode(&row).unwrap()
            })
            .collect();

        // Every bit position should be set by a decent share of the points
        for bit in 0..n_bits {
            let set = codes
                .iter()
                .filter(|c| (c[bit / 8] >> (bit % 8)) & 1 == 1)
                .count();
            let balance = set as f64 / n as f64;

            assert!(
                (0.05..=0.95).contains(&balance),
                "bit {bit} is near-constant across the data (balance {balance:.3}), \
                 which is what an uncentred fit produces"
            );
        }

        // and distinct points must not collapse onto one code
        let mut unique = codes.clone();
        unique.sort_unstable();
        unique.dedup();

        assert!(
            unique.len() > n / 2,
            "only {} distinct codes for {n} points",
            unique.len()
        );
    }

    /// The `(1, 1)` scale pair is the plain residual.
    #[test]
    fn test_encode_residual_squared_euclidean_matches_plain_sign() {
        let dim = 24;
        let binariser = Binariser::<f64>::new_sign_based(dim);

        let vec: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.7).sin() * 3.0 + 1.0).collect();
        let centroid: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.3).cos()).collect();

        let residual: Vec<f64> = vec.iter().zip(&centroid).map(|(v, c)| v - c).collect();

        assert_eq!(
            binariser.encode_residual(&vec, &centroid, (1.0, 1.0)).unwrap(),
            encode_sign_based(&residual, dim)
        );
    }

    /// The `(‖c‖, ‖v‖)` pair is the residual between unit-length vectors, which
    /// is what Cosine wants, expressed without a division.
    #[test]
    fn test_encode_residual_cosine_matches_normalised_sign() {
        let dim = 24;
        let binariser = Binariser::<f64>::new_sign_based(dim);

        let vec: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.7).sin() * 3.0 + 1.0).collect();
        let centroid: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.3).cos()).collect();

        let vn = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        let cn = centroid.iter().map(|x| x * x).sum::<f64>().sqrt();

        let residual: Vec<f64> = vec
            .iter()
            .zip(&centroid)
            .map(|(v, c)| v / vn - c / cn)
            .collect();

        assert_eq!(
            binariser.encode_residual(&vec, &centroid, (cn, vn)).unwrap(),
            encode_sign_based(&residual, dim)
        );
    }

    /// The projection methods carry a learned or random frame of their own, so
    /// a per-cell threshold shift is a different method, not a variant.
    #[test]
    fn test_encode_residual_rejects_projection_methods() {
        let dim = 32;
        let zero_mean = Mat::<f64>::zeros(8, dim);

        let simhash =
            Binariser::<f64>::new_simhash(zero_mean.as_ref(), dim, 64, 42).unwrap();
        let pca = Binariser::<f64>::new_pca_hashing(zero_mean.as_ref(), dim, 64, 42).unwrap();

        let vec = vec![1.0; dim];
        let centroid = vec![0.5; dim];

        for binariser in [simhash, pca] {
            assert!(matches!(
                binariser.encode_residual(&vec, &centroid, (1.0, 1.0)),
                Err(AnnSearchErrors::ResidualEncodingUnsupported)
            ));
        }
    }

    /// A `dim` that is not a multiple of 8 leaves padding bits, which must stay
    /// zero so they XOR away in the Hamming kernels.
    #[test]
    fn test_encode_residual_zeroes_padding_bits() {
        for dim in [30, 31, 32] {
            let binariser = Binariser::<f64>::new_sign_based(dim);

            // All residuals positive, so every real bit is set
            let vec = vec![1.0; dim];
            let centroid = vec![0.0; dim];
            let code = binariser.encode_residual(&vec, &centroid, (1.0, 1.0)).unwrap();

            assert_eq!(code.len(), dim.div_ceil(8));

            let set: u32 = code.iter().map(|b| b.count_ones()).sum();
            assert_eq!(set, dim as u32, "padding bits leaked at dim = {dim}");
        }
    }

    #[test]
    fn test_memory_usage() {
        let dim = 32;
        let n_bits = 64;

        let zero_mean = Mat::<f64>::zeros(8, dim);
        let simhash =
            Binariser::<f64>::new_simhash(zero_mean.as_ref(), dim, n_bits, 42).unwrap();
        let simhash_mem = simhash.memory_usage_bytes();
        assert!(simhash_mem > 0);

        let sign_based = Binariser::<f64>::new_sign_based(dim);
        let sign_mem = sign_based.memory_usage_bytes();
        assert!(sign_mem > 0);
        assert!(simhash_mem > sign_mem, "SimHash should use more memory");
    }
}
